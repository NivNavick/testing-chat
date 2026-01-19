"""
ChromaDB-based schema field embeddings service.

Stores embeddings of target schema fields for column-level matching.
Supports vertical context expansion for better semantic matching.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import chromadb
from chromadb.config import Settings

from csv_analyzer.core.schema_registry import SchemaRegistry, get_schema_registry
from csv_analyzer.multilingual_embeddings_client import MultilingualEmbeddingsClient

if TYPE_CHECKING:
    from csv_analyzer.contexts.registry import VerticalContext

logger = logging.getLogger(__name__)


class SchemaEmbeddingsService:
    """
    Service for embedding and querying target schema fields using ChromaDB.
    
    This enables column-level matching: given an unknown column, find the
    most similar target schema fields.
    """
    
    COLLECTION_NAME = "schema_fields"
    
    def __init__(
        self,
        embeddings_client: MultilingualEmbeddingsClient,
        persist_directory: Optional[str] = None,
        schema_registry: Optional[SchemaRegistry] = None,
    ):
        """
        Initialize the schema embeddings service.
        
        Args:
            embeddings_client: Client for generating embeddings
            persist_directory: Directory to persist ChromaDB data
            schema_registry: Schema registry (uses global if not provided)
        """
        self.embeddings_client = embeddings_client
        self.schema_registry = schema_registry or get_schema_registry()
        
        # Initialize ChromaDB
        if persist_directory is None:
            persist_directory = str(
                Path(__file__).parent.parent / "storage" / "chroma"
            )
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"ChromaDB initialized at {persist_directory}")
        logger.info(f"Collection '{self.COLLECTION_NAME}' has {self.collection.count()} documents")
    
    def index_all_schemas(
        self,
        force_reindex: bool = False,
        vertical_context: Optional["VerticalContext"] = None,
    ):
        """
        Index all schema fields into ChromaDB.
        
        Args:
            force_reindex: If True, delete existing and reindex all
            vertical_context: Optional vertical context to expand embeddings with
                             semantic descriptions for better matching
        """
        if force_reindex:
            # Delete and recreate collection
            self.chroma_client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.chroma_client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Deleted existing collection for reindexing")
        
        # Get all fields from all schemas
        all_fields = self.schema_registry.get_all_fields()
        
        if not all_fields:
            logger.warning("No schema fields found to index")
            return
        
        # Check what's already indexed
        existing_ids = set()
        if self.collection.count() > 0:
            existing = self.collection.get()
            existing_ids = set(existing["ids"])
        
        # Prepare documents to add
        ids = []
        documents = []
        metadatas = []
        
        for vertical, doc_type, field in all_fields:
            field_id = f"{vertical}_{doc_type}_{field.name}"
            
            if field_id in existing_ids:
                continue  # Skip already indexed
            
            # Get embedding text
            embedding_text = field.get_embedding_text()
            
            # Expand with vertical context if available
            if vertical_context and vertical_context.name == vertical:
                embedding_text = vertical_context.expand_embedding_text(
                    field.name, embedding_text
                )
                logger.debug(
                    f"Expanded '{field.name}' embedding with context: "
                    f"{len(embedding_text)} chars"
                )
            
            ids.append(field_id)
            documents.append(embedding_text)
            metadatas.append({
                "vertical": vertical,
                "document_type": doc_type,
                "field_name": field.name,
                "field_type": field.type,
                "required": field.required,
                "description": field.description,
                "aliases": ",".join(field.aliases),  # All languages in one list
            })
        
        if not ids:
            logger.info("All schema fields already indexed")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(ids)} schema fields...")
        
        embeddings = []
        for doc in documents:
            # Use passage prefix for documents being stored
            prefixed = f"passage: {doc}"
            emb = self.embeddings_client._model.encode(
                prefixed,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.append(emb.tolist())
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        
        logger.info(f"✅ Indexed {len(ids)} schema fields into ChromaDB")
        logger.info(f"Total documents in collection: {self.collection.count()}")
    
    def find_matching_fields(
        self,
        column_text: str,
        vertical: Optional[str] = None,
        document_type: Optional[str] = None,
        n_results: int = 5,
        query_embedding: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find schema fields that match a column description.
        
        Args:
            column_text: Text description of the column (name + type + samples)
            vertical: Optional filter by vertical
            document_type: Optional filter by document type
            n_results: Number of results to return
            query_embedding: Optional pre-computed embedding (avoids regeneration)
            
        Returns:
            List of matching fields with similarity scores
        """
        if self.collection.count() == 0:
            logger.warning("No schema fields indexed. Run index_all_schemas() first.")
            return []
        
        # Use pre-computed embedding or generate new one
        if query_embedding is None:
            prefixed = f"query: {column_text}"
            query_embedding = self.embeddings_client._model.encode(
                prefixed,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Build where filter
        where_filter = None
        if vertical and document_type:
            where_filter = {
                "$and": [
                    {"vertical": vertical},
                    {"document_type": document_type}
                ]
            }
        elif vertical:
            where_filter = {"vertical": vertical}
        elif document_type:
            where_filter = {"document_type": document_type}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        matches = []
        if results["ids"] and results["ids"][0]:
            for i, id_ in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                metadata = results["metadatas"][0][i]
                matches.append({
                    "field_id": id_,
                    "field_name": metadata["field_name"],
                    "document_type": metadata["document_type"],
                    "vertical": metadata["vertical"],
                    "field_type": metadata["field_type"],
                    "required": metadata["required"],
                    "description": metadata["description"],
                    "similarity": round(similarity, 4),
                })
        
        return matches
    
    def _build_alias_index(self, vertical: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build an index of aliases to schema fields for exact matching.
        
        Returns:
            Dict mapping lowercase alias -> list of {field_name, document_type, vertical, field_type, ...}
        """
        alias_index = {}
        schemas = self.schema_registry.get_schemas_by_vertical(vertical) if vertical else []
        
        # If no vertical specified, get all schemas
        if not schemas:
            for v in ["medical"]:  # Add more verticals as needed
                schemas.extend(self.schema_registry.get_schemas_by_vertical(v))
        
        for schema in schemas:
            for field in schema.fields:
                # Index by field name
                field_name_lower = field.name.lower()
                if field_name_lower not in alias_index:
                    alias_index[field_name_lower] = []
                alias_index[field_name_lower].append({
                    "field_name": field.name,
                    "document_type": schema.name,
                    "vertical": schema.vertical,
                    "field_type": field.type,
                    "required": field.required,
                    "description": field.description or "",
                })
                
                # Index by aliases
                for alias in (field.aliases or []):
                    alias_lower = alias.lower().strip()
                    if alias_lower not in alias_index:
                        alias_index[alias_lower] = []
                    alias_index[alias_lower].append({
                        "field_name": field.name,
                        "document_type": schema.name,
                        "vertical": schema.vertical,
                        "field_type": field.type,
                        "required": field.required,
                        "description": field.description or "",
                    })
        
        return alias_index
    
    def score_columns_against_schemas(
        self,
        columns: List[Dict[str, Any]],
        vertical: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score a list of columns against all schemas.
        
        OPTIMIZED: Uses batch embedding generation for ~5x speedup.
        
        IMPORTANT: First checks for EXACT alias matches (similarity=1.0),
        then falls back to embeddings for non-matched columns.
        
        Args:
            columns: List of column profiles (from column_profiler)
            vertical: Optional vertical filter
            
        Returns:
            {
                "column_matches": {
                    "col_name": [{"field": "...", "doc_type": "...", "similarity": 0.9, "source_type": "...", ...}, ...]
                },
                "document_type_scores": {
                    "employee_shifts": {"score": 0.85, "matched_fields": 4, "total_required": 4},
                    "medical_actions": {"score": 0.65, "matched_fields": 3, "total_required": 6},
                },
                "column_embeddings": {
                    "col_name": <embedding array>  # Cached for reuse
                }
            }
        """
        from csv_analyzer.core.text_representation import column_to_embedding_text
        from csv_analyzer.core.type_compatibility import get_type_compatibility
        
        # Build alias index for exact matching
        alias_index = self._build_alias_index(vertical)
        
        column_matches = {}
        column_embeddings = {}  # Cache embeddings for potential reuse
        doc_type_votes = {}  # doc_type -> list of adjusted similarities
        
        # ============================================================
        # PHASE 1: Collect alias matches and prepare columns for batch embedding
        # ============================================================
        alias_matches_by_col = {}  # col_name -> list of alias matches
        cols_needing_embedding = []  # (col_name, col_type, col_text)
        
        for col in columns:
            col_name = col["column_name"]
            col_type = col.get("detected_type", "unknown")
            col_text = column_to_embedding_text(col)
            
            # Check for exact alias match
            col_name_lower = col_name.lower().strip()
            if col_name_lower in alias_index:
                alias_matches = alias_index[col_name_lower]
                logger.debug(f"Exact alias match for '{col_name}': {[m['field_name'] for m in alias_matches]}")
                
                enriched = []
                for match in alias_matches:
                    target_type = match.get("field_type", "string")
                    type_compat = get_type_compatibility(col_type, target_type)
                    
                    enriched.append({
                        "field_id": f"{match['vertical']}_{match['document_type']}_{match['field_name']}",
                        "field_name": match["field_name"],
                        "document_type": match["document_type"],
                        "vertical": match["vertical"],
                        "field_type": target_type,
                        "required": match["required"],
                        "description": match["description"],
                        "source_type": col_type,
                        "raw_similarity": 1.0,  # Exact match!
                        "type_compatibility": round(type_compat, 3),
                        "similarity": round(1.0 * type_compat, 4),
                        "match_source": "alias",
                    })
                alias_matches_by_col[col_name] = enriched
            
            # All columns get embedding (for additional matches beyond alias)
            cols_needing_embedding.append((col_name, col_type, col_text))
        
        # ============================================================
        # PHASE 2: Batch generate embeddings (THE KEY OPTIMIZATION!)
        # ============================================================
        if cols_needing_embedding:
            # Prepare all texts with prefix
            texts = [f"query: {col_text}" for _, _, col_text in cols_needing_embedding]
            
            # BATCH ENCODE - much faster than one-by-one!
            logger.debug(f"Batch encoding {len(texts)} column texts...")
            all_embeddings = self.embeddings_client._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,  # Disable progress bar for cleaner logs
            )
            
            # Store embeddings by column name
            for i, (col_name, _, _) in enumerate(cols_needing_embedding):
                column_embeddings[col_name] = all_embeddings[i]
        
        # ============================================================
        # PHASE 3: Match embeddings to schema fields
        # ============================================================
        for col_name, col_type, col_text in cols_needing_embedding:
            enriched_matches = list(alias_matches_by_col.get(col_name, []))
            existing_field_ids = {m["field_id"] for m in enriched_matches}
            
            # Find matching fields using pre-computed embedding
            query_embedding = column_embeddings[col_name]
            embedding_matches = self.find_matching_fields(
                column_text=col_text,
                vertical=vertical,
                n_results=5,
                query_embedding=query_embedding,
            )
            
            # Add embedding matches (but don't duplicate alias matches)
            for match in embedding_matches:
                if match["field_id"] in existing_field_ids:
                    continue  # Skip - already have an alias match
                    
                target_type = match.get("field_type", "string")
                type_compat = get_type_compatibility(col_type, target_type)
                raw_similarity = match["similarity"]
                adjusted_similarity = raw_similarity * type_compat
                
                enriched_matches.append({
                    **match,
                    "source_type": col_type,
                    "raw_similarity": raw_similarity,
                    "type_compatibility": round(type_compat, 3),
                    "similarity": round(adjusted_similarity, 4),
                    "match_source": "embedding",
                })
            
            # Sort by similarity (alias matches will be at top with 1.0)
            enriched_matches.sort(key=lambda x: x["similarity"], reverse=True)
            column_matches[col_name] = enriched_matches
            
            # Vote for document types based on adjusted similarities
            for match in enriched_matches:
                doc_type = match["document_type"]
                similarity = match["similarity"]
                match_source = match.get("match_source", "embedding")
                
                if doc_type not in doc_type_votes:
                    doc_type_votes[doc_type] = {"alias_cols": set(), "embedding_sims": []}
                
                if match_source == "alias":
                    doc_type_votes[doc_type]["alias_cols"].add(col_name)
                else:
                    doc_type_votes[doc_type]["embedding_sims"].append(similarity)
        
        # Calculate document type scores
        # Formula: alias_ratio * 0.7 + embedding_avg * 0.3
        # Prioritizes document types with more alias-matched columns
        document_type_scores = {}
        total_cols = len(columns)
        
        for doc_type, votes in doc_type_votes.items():
            # Get schema to check required fields
            schema = self.schema_registry.get_schema(vertical or "medical", doc_type)
            required_count = len(schema.get_required_fields()) if schema else 0
            
            alias_col_count = len(votes["alias_cols"])
            embedding_sims = votes["embedding_sims"]
            
            # Alias match ratio (how many columns matched by alias / total columns)
            alias_ratio = alias_col_count / total_cols if total_cols > 0 else 0
            
            # Average embedding similarity for non-alias matches
            embedding_avg = sum(embedding_sims) / len(embedding_sims) if embedding_sims else 0.5
            
            # Combined score: heavily weight alias matches
            # If many columns match aliases, that's a strong signal
            score = (alias_ratio * 0.7) + (embedding_avg * 0.3)
            
            document_type_scores[doc_type] = {
                "score": round(score, 4),
                "alias_matched_columns": alias_col_count,
                "total_columns": total_cols,
                "required_fields": required_count,
            }
        
        return {
            "column_matches": column_matches,
            "document_type_scores": document_type_scores,
            "column_embeddings": column_embeddings,
        }
    
    def get_column_matches_for_document_type(
        self,
        columns: List[Dict[str, Any]],
        column_embeddings: Dict[str, Any],
        document_type: str,
        vertical: Optional[str] = None,
        n_results: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get column matches filtered to a specific document type.
        
        Uses cached embeddings to avoid regenerating them.
        IMPORTANT: Also checks alias index for exact matches (priority over embeddings).
        
        Args:
            columns: List of column profiles
            column_embeddings: Cached embeddings from score_columns_against_schemas
            document_type: Document type to filter by
            vertical: Optional vertical filter
            n_results: Number of results per column
            
        Returns:
            Dict of column_name -> list of matches (all from the specified document_type)
        """
        from csv_analyzer.core.text_representation import column_to_embedding_text
        from csv_analyzer.core.type_compatibility import get_type_compatibility
        
        # Build alias index for exact matching (IMPORTANT!)
        alias_index = self._build_alias_index(vertical)
        
        column_matches = {}
        
        for col in columns:
            col_name = col["column_name"]
            col_type = col.get("detected_type", "unknown")
            col_text = column_to_embedding_text(col)
            
            enriched_matches = []
            existing_field_ids = set()
            
            # FIRST: Check for exact alias matches (priority!)
            col_name_lower = col_name.lower().strip()
            if col_name_lower in alias_index:
                alias_matches = alias_index[col_name_lower]
                # Filter to the target document type
                for match in alias_matches:
                    if match["document_type"] == document_type:
                        target_type = match.get("field_type", "string")
                        type_compat = get_type_compatibility(col_type, target_type)
                        field_id = f"{match['vertical']}_{match['document_type']}_{match['field_name']}"
                        
                        if field_id not in existing_field_ids:
                            enriched_matches.append({
                                "field_id": field_id,
                                "field_name": match["field_name"],
                                "document_type": match["document_type"],
                                "vertical": match["vertical"],
                                "field_type": target_type,
                                "required": match["required"],
                                "description": match["description"],
                                "source_type": col_type,
                                "raw_similarity": 1.0,  # Exact alias match!
                                "type_compatibility": round(type_compat, 3),
                                "similarity": round(1.0 * type_compat, 4),
                                "match_source": "alias",
                            })
                            existing_field_ids.add(field_id)
            
            # SECOND: Use cached embedding for additional matches
            query_embedding = column_embeddings.get(col_name)
            if query_embedding is None:
                logger.warning(f"No cached embedding for column '{col_name}', generating new one")
                prefixed = f"query: {col_text}"
                query_embedding = self.embeddings_client._model.encode(
                    prefixed,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            
            # Find matching fields filtered by document type
            matches = self.find_matching_fields(
                column_text=col_text,
                vertical=vertical,
                document_type=document_type,
                n_results=n_results,
                query_embedding=query_embedding,
            )
            
            # Add embedding matches (but don't duplicate alias matches)
            for match in matches:
                if match["field_id"] in existing_field_ids:
                    continue  # Skip - already have alias match
                    
                target_type = match.get("field_type", "string")
                type_compat = get_type_compatibility(col_type, target_type)
                raw_similarity = match["similarity"]
                adjusted_similarity = raw_similarity * type_compat
                
                enriched_matches.append({
                    **match,
                    "source_type": col_type,
                    "raw_similarity": raw_similarity,
                    "type_compatibility": round(type_compat, 3),
                    "similarity": round(adjusted_similarity, 4),
                    "match_source": "embedding",
                })
            
            # Sort by adjusted similarity (alias matches with 1.0 will be at top)
            enriched_matches.sort(key=lambda x: x["similarity"], reverse=True)
            column_matches[col_name] = enriched_matches
        
        return column_matches


# Global service instance
_service: Optional[SchemaEmbeddingsService] = None


def get_schema_embeddings_service(
    embeddings_client: Optional[MultilingualEmbeddingsClient] = None,
) -> SchemaEmbeddingsService:
    """Get the global schema embeddings service instance."""
    global _service
    if _service is None:
        if embeddings_client is None:
            from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client
            embeddings_client = get_multilingual_embeddings_client()
        _service = SchemaEmbeddingsService(embeddings_client)
    return _service
