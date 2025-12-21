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
    ) -> List[Dict[str, Any]]:
        """
        Find schema fields that match a column description.
        
        Args:
            column_text: Text description of the column (name + type + samples)
            vertical: Optional filter by vertical
            document_type: Optional filter by document type
            n_results: Number of results to return
            
        Returns:
            List of matching fields with similarity scores
        """
        if self.collection.count() == 0:
            logger.warning("No schema fields indexed. Run index_all_schemas() first.")
            return []
        
        # Generate query embedding
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
    
    def score_columns_against_schemas(
        self,
        columns: List[Dict[str, Any]],
        vertical: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score a list of columns against all schemas.
        
        Args:
            columns: List of column profiles (from column_profiler)
            vertical: Optional vertical filter
            
        Returns:
            {
                "column_matches": {
                    "col_name": [{"field": "...", "doc_type": "...", "similarity": 0.9}, ...]
                },
                "document_type_scores": {
                    "employee_shifts": {"score": 0.85, "matched_fields": 4, "total_required": 4},
                    "medical_actions": {"score": 0.65, "matched_fields": 3, "total_required": 6},
                }
            }
        """
        from csv_analyzer.core.text_representation import column_to_embedding_text
        
        column_matches = {}
        doc_type_votes = {}  # doc_type -> list of similarities
        
        for col in columns:
            col_name = col["column_name"]
            col_text = column_to_embedding_text(col)
            
            # Find matching fields
            matches = self.find_matching_fields(
                column_text=col_text,
                vertical=vertical,
                n_results=5,
            )
            
            column_matches[col_name] = matches
            
            # Vote for document types based on matches
            for match in matches:
                doc_type = match["document_type"]
                similarity = match["similarity"]
                
                if doc_type not in doc_type_votes:
                    doc_type_votes[doc_type] = []
                doc_type_votes[doc_type].append(similarity)
        
        # Calculate document type scores
        document_type_scores = {}
        for doc_type, similarities in doc_type_votes.items():
            # Get schema to check required fields
            schema = self.schema_registry.get_schema(vertical or "medical", doc_type)
            required_count = len(schema.get_required_fields()) if schema else 0
            
            # Score = average of top similarities (one per unique match)
            # We take unique matches to avoid counting same doc_type multiple times
            unique_sims = similarities[:len(columns)]  # Limit to column count
            avg_score = sum(unique_sims) / len(unique_sims) if unique_sims else 0
            
            document_type_scores[doc_type] = {
                "score": round(avg_score, 4),
                "matched_columns": len(unique_sims),
                "total_columns": len(columns),
                "required_fields": required_count,
            }
        
        return {
            "column_matches": column_matches,
            "document_type_scores": document_type_scores,
        }


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
