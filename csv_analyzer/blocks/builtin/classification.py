"""
Classification Block - Classify and canonize CSVs.

Detects document types and maps column names to canonical schema fields.
Uses cached embeddings service for production performance.
"""

import logging
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)


# ============================================================================
# Cached Services (Singleton Pattern)
# ============================================================================

class _CachedServices:
    """
    Singleton cache for expensive ML services.
    
    Ensures embeddings client, schema registry, and schema service
    are only initialized once per process.
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._embeddings_client = None
        self._schema_registry = None
        self._schema_service = None
        self._canonizers: Dict[str, "SchemaCanonizer"] = {}
        self._initialized = True
    
    @property
    def embeddings_client(self):
        """Lazy-load embeddings client."""
        if self._embeddings_client is None:
            logger.info("Initializing multilingual embeddings client (one-time)...")
            from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client
            self._embeddings_client = get_multilingual_embeddings_client()
        return self._embeddings_client
    
    @property
    def schema_registry(self):
        """Lazy-load schema registry."""
        if self._schema_registry is None:
            logger.info("Loading schema registry (one-time)...")
            from csv_analyzer.core.schema_registry import get_schema_registry
            self._schema_registry = get_schema_registry()
        return self._schema_registry
    
    @property
    def schema_service(self):
        """Lazy-load schema embeddings service."""
        if self._schema_service is None:
            logger.info("Initializing schema embeddings service (one-time)...")
            from csv_analyzer.core.schema_embeddings import SchemaEmbeddingsService
            self._schema_service = SchemaEmbeddingsService(
                embeddings_client=self.embeddings_client,
                schema_registry=self.schema_registry,
            )
            # Index schemas only if not already indexed
            if self._schema_service.collection.count() == 0:
                logger.info("Indexing schema embeddings (one-time)...")
                self._schema_service.index_all_schemas(force_reindex=False)
            else:
                logger.info(f"Schema embeddings already indexed: {self._schema_service.collection.count()} documents")
        return self._schema_service
    
    def get_canonizer(self, vertical: str, openai_client=None) -> "SchemaCanonizer":
        """Get or create a cached canonizer for a vertical."""
        if vertical not in self._canonizers:
            logger.info(f"Creating canonizer for vertical: {vertical}")
            self._canonizers[vertical] = SchemaCanonizer(
                vertical=vertical,
                openai_client=openai_client,
                schema_service=self.schema_service,
                schema_registry=self.schema_registry,
            )
        return self._canonizers[vertical]


def get_cached_services() -> _CachedServices:
    """Get the singleton cached services instance."""
    return _CachedServices()


# ============================================================================
# Schema Canonizer
# ============================================================================

class SchemaCanonizer:
    """
    Canonizes CSV columns to schema field names using embeddings.
    
    This ensures that regardless of customer's column naming (Hebrew, English, Spanish...),
    columns are mapped to canonical English schema field names.
    
    Uses cached services for production performance.
    """
    
    def __init__(
        self,
        vertical: str = "medical",
        openai_client=None,
        schema_service=None,
        schema_registry=None,
    ):
        self.vertical = vertical
        self.openai_client = openai_client
        
        # Use provided services or get from cache
        if schema_service is None or schema_registry is None:
            cache = get_cached_services()
            self.schema_service = cache.schema_service
            self.schema_registry = cache.schema_registry
        else:
            self.schema_service = schema_service
            self.schema_registry = schema_registry
    
    def classify_and_canonize(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Tuple[Optional[str], pd.DataFrame, Dict[str, str]]:
        """
        Classify document type and canonize column names to schema fields.
        
        Args:
            df: DataFrame to classify and canonize
            threshold: Minimum similarity threshold for column mapping
            
        Returns:
            Tuple of (document_type, canonized_df, column_mappings)
        """
        from csv_analyzer.columns_analyzer import profile_dataframe
        
        # Profile columns
        column_profiles = profile_dataframe(df)
        logger.debug(f"Profiled {len(column_profiles)} columns")
        
        # Score columns against all schemas
        result = self.schema_service.score_columns_against_schemas(
            columns=column_profiles,
            vertical=self.vertical,
        )
        
        doc_type_scores = result["document_type_scores"]
        column_matches = result["column_matches"]
        
        if not doc_type_scores:
            logger.warning("No document type scores returned")
            return None, df, {}
        
        # Get top candidates
        sorted_scores = sorted(
            doc_type_scores.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True
        )
        
        # Log top scores
        for dt, scores in sorted_scores[:3]:
            logger.debug(f"  {dt}: {scores.get('score', 0):.3f}")
        
        # Get all candidates for verification
        all_candidates = [(dt, info.get("score", 0)) for dt, info in sorted_scores]
        
        # Use OpenAI to verify if available, otherwise use top embedding match
        if self.openai_client:
            doc_type = self._verify_document_type(df, all_candidates, column_profiles)
        else:
            doc_type = all_candidates[0][0] if all_candidates else None
        
        logger.info(f"Detected document type: {doc_type}")
        
        # Build column mappings for the detected document type
        mappings = {}
        for col_name, matches in column_matches.items():
            if matches:
                doc_matches = [m for m in matches if m.get("document_type") == doc_type]
                if doc_matches:
                    best_match = doc_matches[0]
                    if best_match["similarity"] >= threshold:
                        canonical_name = best_match["field_name"]
                        mappings[col_name] = canonical_name
        
        # Canonize: rename columns to schema field names
        df_canonized = df.copy()
        for orig_col, canonical_col in mappings.items():
            if orig_col in df_canonized.columns and orig_col != canonical_col:
                if canonical_col not in df_canonized.columns:
                    df_canonized = df_canonized.rename(columns={orig_col: canonical_col})
                    logger.debug(f"  Canonized: {orig_col} → {canonical_col}")
        
        return doc_type, df_canonized, mappings
    
    def _determine_doc_type_from_matches(
        self,
        column_matches: Dict[str, List[Dict]],
        column_profiles: List[Dict],
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[str]:
        """
        Determine document type from pre-computed column matches.
        
        Args:
            column_matches: Pre-computed matches from batch scoring
            column_profiles: Column profiles for this file
            df: Optional DataFrame for OpenAI verification
            
        Returns:
            Best matching document type
        """
        # Aggregate votes by document type
        doc_type_votes: Dict[str, Dict] = {}
        
        for col_name, matches in column_matches.items():
            for match in matches:
                doc_type = match["document_type"]
                similarity = match["similarity"]
                match_source = match.get("match_source", "embedding")
                
                if doc_type not in doc_type_votes:
                    doc_type_votes[doc_type] = {"alias_cols": set(), "embedding_sims": []}
                
                if match_source == "alias":
                    doc_type_votes[doc_type]["alias_cols"].add(col_name)
                else:
                    doc_type_votes[doc_type]["embedding_sims"].append(similarity)
        
        if not doc_type_votes:
            return None
        
        # Calculate scores
        total_cols = len(column_profiles)
        candidates = []
        
        for doc_type, votes in doc_type_votes.items():
            alias_count = len(votes["alias_cols"])
            embedding_sims = votes["embedding_sims"]
            
            alias_ratio = alias_count / total_cols if total_cols > 0 else 0
            embedding_avg = sum(embedding_sims) / len(embedding_sims) if embedding_sims else 0.5
            
            # Combined score: heavily weight alias matches
            score = alias_ratio * 0.7 + embedding_avg * 0.3
            candidates.append((doc_type, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Log top candidates
        for dt, score in candidates[:3]:
            logger.debug(f"  {dt}: {score:.3f}")
        
        # Use OpenAI to verify if available
        if self.openai_client and df is not None:
            return self._verify_document_type(df, candidates, column_profiles)
        
        return candidates[0][0] if candidates else None
    
    def _verify_document_type(
        self,
        df: pd.DataFrame,
        candidates: List[Tuple[str, float]],
        column_profiles: List[Dict],
    ) -> str:
        """Use OpenAI to verify document type based on column names and values."""
        if not self.openai_client or not candidates:
            return candidates[0][0] if candidates else None
        
        # Build column summary
        col_summaries = []
        for profile in column_profiles[:15]:
            col_name = profile.get("column_name", "unknown")
            col_type = profile.get("detected_type", "unknown")
            samples = profile.get("sample_values", [])[:3]
            col_summaries.append(f"- {col_name} ({col_type}): {samples}")
        
        columns_text = "\n".join(col_summaries)
        candidates_text = ", ".join([f"{dt} ({score:.2f})" for dt, score in candidates])
        
        # Get document type descriptions
        doc_descriptions = []
        all_valid_types = []
        for schema in self.schema_registry.get_schemas_by_vertical(self.vertical):
            all_valid_types.append(schema.name)
            desc = schema.description or f"Document type: {schema.name}"
            doc_descriptions.append(f"- {schema.name}: {desc}")
        
        descriptions_text = "\n".join(doc_descriptions)
        
        prompt = f"""Given a CSV file with these columns and sample values:

{columns_text}

Which document type best describes this file?

Embedding scores: {candidates_text}

Available document types:
{descriptions_text}

Respond with ONLY the document type name."""

        try:
            response = self.openai_client.client.chat.completions.create(
                model=self.openai_client.model,
                messages=[
                    {"role": "system", "content": "You are a document classification expert. Respond with only the document type name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50,
            )
            result = response.choices[0].message.content.strip().lower()
            
            if result in all_valid_types:
                return result
            
            for dt in all_valid_types:
                if dt in result or result in dt:
                    return dt
            
            logger.warning(f"OpenAI returned '{result}', using top embedding candidate")
            return candidates[0][0]
            
        except Exception as e:
            logger.warning(f"OpenAI verification failed: {e}")
            return candidates[0][0]


# ============================================================================
# Classification Block
# ============================================================================

class ClassificationBlock(BaseBlock):
    """
    Classify and canonize CSV files.
    
    For each input file:
    1. Detect document type (employee_shifts, medical_actions, etc.)
    2. Map columns to canonical schema field names
    3. Merge files of the same document type
    
    OPTIMIZED: Batches embeddings across ALL files for maximum throughput.
    Uses cached ML services for fast repeated execution.
    """
    
    def run(self) -> Dict[str, str]:
        """
        Classify input files.
        
        Returns:
            Dict with 'classified_data' key containing S3 URI of manifest
        """
        # Load input files
        files_manifest = self.load_input("files")
        
        # Handle input format
        if isinstance(files_manifest, list):
            file_uris = files_manifest
        elif isinstance(files_manifest, pd.DataFrame):
            file_uris = files_manifest.iloc[:, 0].tolist()
        else:
            file_uris = list(files_manifest.values()) if isinstance(files_manifest, dict) else [files_manifest]
        
        # Get parameters
        vertical = self.get_param("vertical", "medical")
        threshold = self.get_param("threshold", 0.5)
        
        # Get or create cached canonizer
        try:
            from csv_analyzer.openai_client import get_openai_client
            openai_client = get_openai_client()
        except Exception as e:
            self.logger.warning(f"OpenAI client not available: {e}")
            openai_client = None
        
        cache = get_cached_services()
        canonizer = cache.get_canonizer(vertical, openai_client)
        
        # ============================================================
        # PHASE 1: Load all files and profile columns
        # ============================================================
        self.logger.info(f"Loading {len(file_uris)} files...")
        file_data = []  # List of (file_uri, df, column_profiles)
        
        from csv_analyzer.columns_analyzer import profile_dataframe
        
        for file_uri in file_uris:
            # Load the file
            if isinstance(file_uri, str) and file_uri.startswith("s3://"):
                df = self.load_from_s3(file_uri)
            elif Path(file_uri).exists():
                if file_uri.endswith(".json"):
                    df = self.load_from_s3(file_uri)  # JSON loader works for local files too
                else:
                    df = pd.read_csv(file_uri)
            else:
                self.logger.warning(f"File not found: {file_uri}")
                continue
            
            # Profile columns (fast, no ML)
            profiles = profile_dataframe(df)
            file_data.append((file_uri, df, profiles))
            self.logger.debug(f"  Loaded {Path(file_uri).name}: {len(df)} rows, {len(profiles)} columns")
        
        # ============================================================
        # PHASE 2: Batch classify all files at once
        # ============================================================
        self.logger.info(f"Classifying {len(file_data)} files (batched embeddings)...")
        
        # Collect all unique column profiles across all files for batch embedding
        all_profiles = []
        profile_to_file_idx = []  # Track which file each profile came from
        
        for file_idx, (_, _, profiles) in enumerate(file_data):
            for profile in profiles:
                all_profiles.append(profile)
                profile_to_file_idx.append(file_idx)
        
        # Single batch call for ALL columns across ALL files
        result = canonizer.schema_service.score_columns_against_schemas(
            columns=all_profiles,
            vertical=vertical,
        )
        
        # ============================================================
        # PHASE 3: Assign document types and canonize per file
        # ============================================================
        classified: Dict[str, pd.DataFrame] = {}
        
        for file_idx, (file_uri, df, profiles) in enumerate(file_data):
            filename = Path(file_uri).name
            
            # Get column matches for this file's columns
            file_col_names = {p["column_name"] for p in profiles}
            file_column_matches = {
                col: matches 
                for col, matches in result["column_matches"].items()
                if col in file_col_names
            }
            
            # Determine document type for this file
            doc_type = canonizer._determine_doc_type_from_matches(
                file_column_matches, 
                profiles,
                df if openai_client else None,
            )
            
            if doc_type is None:
                self.logger.warning(f"Could not classify: {filename}")
                continue
            
            # Build column mappings for canonization
            mappings = {}
            for col_name, matches in file_column_matches.items():
                if matches:
                    doc_matches = [m for m in matches if m.get("document_type") == doc_type]
                    if doc_matches:
                        best_match = doc_matches[0]
                        if best_match["similarity"] >= threshold:
                            mappings[col_name] = best_match["field_name"]
            
            # Canonize columns
            df_canonized = df.copy()
            for orig_col, canonical_col in mappings.items():
                if orig_col in df_canonized.columns and orig_col != canonical_col:
                    if canonical_col not in df_canonized.columns:
                        df_canonized = df_canonized.rename(columns={orig_col: canonical_col})
            
            self.logger.info(f"  {filename} → {doc_type} ({len(mappings)} columns)")
            
            # Merge with existing data of same type
            if doc_type in classified:
                classified[doc_type] = pd.concat(
                    [classified[doc_type], df_canonized],
                    ignore_index=True,
                    sort=False,
                )
            else:
                classified[doc_type] = df_canonized
        
        # Log summary
        for doc_type, df in classified.items():
            self.logger.info(f"  {doc_type}: {len(df)} rows total")
        
        # Save classified data to S3
        manifest_uri = self.save_classified_data(classified)
        
        return {"classified_data": manifest_uri}


# Register the block
@BlockRegistry.register(
    name="classify_csvs",
    inputs=[{"name": "files", "ontology": DataType.S3_REFERENCES}],
    outputs=[{"name": "classified_data", "ontology": DataType.CLASSIFIED_DATA}],
    parameters=[
        {"name": "vertical", "type": "string", "default": "medical", "description": "Vertical/domain for schema matching"},
        {"name": "threshold", "type": "float", "default": 0.5, "description": "Minimum similarity threshold"},
    ],
    block_class=ClassificationBlock,
    description="Classify CSVs and canonize columns to schema field names (cached ML)",
)
def classify_csvs(ctx: BlockContext) -> Dict[str, str]:
    """Classify and canonize CSV files."""
    return ClassificationBlock(ctx).run()
