"""
Classification Block - Classify and canonize CSVs.

Detects document types and maps column names to canonical schema fields.
Uses cached embeddings service for production performance.

Supports two execution modes for large-scale data:
1. Legacy: Load full file into pandas for profiling
2. Optimized: Use DuckDB RESERVOIR sampling for large files (>50K rows)
   - Profile only a sample (5000 rows by default)
   - Apply column renames via SQL (zero-copy)
   - Stream output to Parquet
"""

import logging
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from csv_analyzer.workflows.block import BlockRegistry
from csv_analyzer.workflows.ontology import DataType
from csv_analyzer.workflows.base_block import BaseBlock, BlockContext

logger = logging.getLogger(__name__)

# Threshold for auto-switching to DuckDB sampling mode
LARGE_FILE_THRESHOLD = 50_000  # rows
DEFAULT_SAMPLE_SIZE = 5_000  # rows to sample for profiling


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
    2. Apply schema-specific preprocessing (e.g., split rate "73/82")
    3. Map columns to canonical schema field names
    4. Merge files of the same document type
    
    OPTIMIZED: 
    - Batches embeddings across ALL files for maximum throughput
    - Uses DuckDB RESERVOIR sampling for large files (>50K rows)
    - Applies column renames via SQL for zero-copy operations
    - Streams output to Parquet format for efficient I/O
    
    Uses cached ML services for fast repeated execution.
    """
    
    def _count_rows_fast(self, file_path: str) -> int:
        """
        Count rows in a file efficiently using DuckDB.
        
        For CSV files, this scans but doesn't load data into memory.
        For Parquet files, this reads metadata only (instant).
        """
        try:
            if file_path.endswith('.parquet'):
                return self.duckdb.execute(f"""
                    SELECT COUNT(*) FROM read_parquet('{file_path}')
                """).fetchone()[0]
            else:
                return self.duckdb.execute(f"""
                    SELECT COUNT(*) FROM read_csv_auto('{file_path}')
                """).fetchone()[0]
        except Exception as e:
            self.logger.warning(f"Could not count rows for {file_path}: {e}")
            return 0
    
    def _sample_file_for_profiling(
        self,
        file_path: str,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ) -> pd.DataFrame:
        """
        Sample rows from a file using DuckDB RESERVOIR sampling.
        
        This loads only a sample into memory, enabling profiling of
        files larger than available RAM.
        
        Args:
            file_path: Path to CSV or Parquet file
            sample_size: Number of rows to sample
            
        Returns:
            Sampled DataFrame
        """
        self.logger.info(f"Sampling {sample_size} rows from {Path(file_path).name} using DuckDB...")
        
        try:
            if file_path.endswith('.parquet'):
                return self.duckdb.execute(f"""
                    SELECT * FROM read_parquet('{file_path}')
                    USING SAMPLE RESERVOIR({sample_size} ROWS)
                """).fetchdf()
            else:
                return self.duckdb.execute(f"""
                    SELECT * FROM read_csv_auto('{file_path}')
                    USING SAMPLE RESERVOIR({sample_size} ROWS)
                """).fetchdf()
        except Exception as e:
            self.logger.warning(f"DuckDB sampling failed, falling back to pandas: {e}")
            # Fallback to pandas with nrows
            if file_path.endswith('.parquet'):
                return pd.read_parquet(file_path).head(sample_size)
            else:
                return pd.read_csv(file_path, nrows=sample_size)
    
    def _apply_column_renames_duckdb(
        self,
        file_path: str,
        mappings: Dict[str, str],
        output_path: str,
    ) -> str:
        """
        Apply column renames using DuckDB SQL (zero-copy, streaming).
        
        This processes the file without loading it entirely into memory.
        
        Args:
            file_path: Input file path
            mappings: Dict of original_column -> canonical_column
            output_path: Output Parquet file path
            
        Returns:
            Path to output file
        """
        # Build SELECT clause with renames
        # Get all columns from the file
        if file_path.endswith('.parquet'):
            cols_result = self.duckdb.execute(f"""
                SELECT column_name FROM (DESCRIBE SELECT * FROM read_parquet('{file_path}'))
            """).fetchall()
        else:
            cols_result = self.duckdb.execute(f"""
                SELECT column_name FROM (DESCRIBE SELECT * FROM read_csv_auto('{file_path}'))
            """).fetchall()
        
        all_columns = [row[0] for row in cols_result]
        
        # Build select expressions
        select_exprs = []
        for col in all_columns:
            if col in mappings and col != mappings[col]:
                canonical = mappings[col]
                # Rename: SELECT "original" AS "canonical"
                select_exprs.append(f'"{col}" AS "{canonical}"')
            else:
                select_exprs.append(f'"{col}"')
        
        select_clause = ", ".join(select_exprs)
        
        # Read and write in streaming fashion
        if file_path.endswith('.parquet'):
            read_func = f"read_parquet('{file_path}')"
        else:
            read_func = f"read_csv_auto('{file_path}')"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.duckdb.execute(f"""
            COPY (SELECT {select_clause} FROM {read_func})
            TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        
        self.logger.info(f"Wrote canonized file to {output_path}")
        return output_path
    
    def _apply_schema_preprocessing(
        self,
        df: pd.DataFrame,
        doc_type: str,
        vertical: str,
    ) -> pd.DataFrame:
        """
        Apply schema-specific preprocessing transforms.
        
        Loads the schema for the detected document type and applies
        any column_transforms defined in its preprocessing section.
        
        Args:
            df: Input DataFrame
            doc_type: Detected document type (e.g., "employee_monthly_salary")
            vertical: Vertical/domain (e.g., "medical")
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Load raw YAML schema to get preprocessing config
            import yaml
            from pathlib import Path
            
            # Find schema file
            cache = get_cached_services()
            schema_registry = cache.schema_registry
            schemas_dir = Path(schema_registry.schemas_dir) / vertical
            schema_file = schemas_dir / f"{doc_type}.yaml"
            
            if not schema_file.exists():
                self.logger.debug(f"No schema file found: {schema_file}")
                return df
            
            # Load raw YAML
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_def = yaml.safe_load(f)
            
            # Get preprocessing config from schema
            preprocessing_config = schema_def.get("preprocessing", {})
            column_transforms = preprocessing_config.get("column_transforms", [])
            
            if not column_transforms:
                return df
            
            # Filter transforms to only those with source_column (not source_column_pattern)
            # TODO: Add support for source_column_pattern matching
            valid_transforms = [t for t in column_transforms if "source_column" in t]
            
            if not valid_transforms:
                self.logger.debug(f"No applicable transforms for {doc_type} (pattern-based transforms not yet supported)")
                return df
            
            self.logger.info(f"Applying {len(valid_transforms)} schema transforms for {doc_type}")
            
            # Apply transforms using ValueTransformer
            from csv_analyzer.preprocessing.transformers import ValueTransformer, TransformConfig
            
            transformer = ValueTransformer()
            transform_configs = [TransformConfig.from_dict(t) for t in valid_transforms]
            
            result = transformer.apply_all(df, transform_configs)
            
            if result.errors:
                for error in result.errors:
                    self.logger.warning(f"Transform error: {error}")
            
            if result.columns_added:
                self.logger.info(f"  Added columns: {', '.join(result.columns_added)}")
            
            return result.df
            
        except Exception as e:
            self.logger.warning(f"Failed to apply schema preprocessing for {doc_type}: {e}")
            return df
    
    def run(self) -> Dict[str, str]:
        """
        Classify input files.
        
        Automatically uses DuckDB sampling for large files (>50K rows).
        
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
        use_duckdb = self.get_param("use_duckdb", None)  # None = auto-detect
        sample_size = self.get_param("sample_size", DEFAULT_SAMPLE_SIZE)
        output_format = self.get_param("output_format", "json")  # or "parquet"
        
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
        # PHASE 1: Load/sample files and profile columns
        # ============================================================
        self.logger.info(f"Loading {len(file_uris)} files...")
        file_data = []  # List of (file_uri, df_or_none, column_profiles, row_count, is_large)
        
        from csv_analyzer.columns_analyzer import profile_dataframe
        
        for file_uri in file_uris:
            file_path = str(file_uri)
            
            # Check if file exists
            if file_path.startswith("s3://"):
                # S3 files - load fully for now (TODO: add S3 streaming)
                df = self.load_from_s3(file_path)
                row_count = len(df)
                is_large = False
            elif Path(file_path).exists():
                if file_path.endswith(".json"):
                    df = self.load_from_s3(file_path)
                    row_count = len(df)
                    is_large = False
                else:
                    # CSV or Parquet - check size first
                    row_count = self._count_rows_fast(file_path)
                    is_large = row_count > LARGE_FILE_THRESHOLD
                    
                    # Auto-detect DuckDB mode based on file size
                    use_duckdb_for_file = use_duckdb if use_duckdb is not None else is_large
                    
                    if use_duckdb_for_file and is_large:
                        self.logger.info(
                            f"Large file detected ({row_count:,} rows > {LARGE_FILE_THRESHOLD:,}). "
                            f"Using DuckDB sampling for profiling."
                        )
                        # Sample for profiling only - don't load full file
                        df = self._sample_file_for_profiling(file_path, sample_size)
                    else:
                        # Small file - load fully
                        if file_path.endswith('.parquet'):
                            df = pd.read_parquet(file_path)
                        else:
                            df = pd.read_csv(file_path)
            else:
                self.logger.warning(f"File not found: {file_uri}")
                continue
            
            # Profile columns (on sample or full data)
            profiles = profile_dataframe(df)
            file_data.append((file_uri, df, profiles, row_count, is_large))
            self.logger.debug(
                f"  Loaded {Path(file_path).name}: {row_count:,} rows, {len(profiles)} columns"
                f"{' (sampled)' if is_large else ''}"
            )
        
        # ============================================================
        # PHASE 2: Batch classify all files at once
        # ============================================================
        self.logger.info(f"Classifying {len(file_data)} files (batched embeddings)...")
        
        # Collect all unique column profiles across all files for batch embedding
        all_profiles = []
        profile_to_file_idx = []  # Track which file each profile came from
        
        for file_idx, (_, _, profiles, _, _) in enumerate(file_data):
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
        
        for file_idx, (file_uri, df, profiles, row_count, is_large) in enumerate(file_data):
            filename = Path(str(file_uri)).name
            file_path = str(file_uri)
            
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
            
            self.logger.info(f"  {filename} → {doc_type} ({len(mappings)} columns mapped)")
            
            # For large files, use DuckDB streaming for canonization
            if is_large and not file_path.startswith("s3://") and not file_path.endswith(".json"):
                self.logger.info(f"Using DuckDB streaming for large file canonization...")
                
                # First, load the full file for preprocessing (if needed)
                # For now, we need to load for preprocessing, but could optimize later
                if file_path.endswith('.parquet'):
                    df_full = pd.read_parquet(file_path)
                else:
                    df_full = pd.read_csv(file_path)
                
                # Apply schema-specific preprocessing
                df_preprocessed = self._apply_schema_preprocessing(df_full, doc_type, vertical)
                
                # Canonize columns (in-place for memory efficiency)
                for orig_col, canonical_col in mappings.items():
                    if orig_col in df_preprocessed.columns and orig_col != canonical_col:
                        if canonical_col not in df_preprocessed.columns:
                            df_preprocessed.rename(columns={orig_col: canonical_col}, inplace=True)
                
                df_canonized = df_preprocessed
            else:
                # Small file or S3/JSON - use existing logic
                # Apply schema-specific preprocessing (e.g., split rate "73/82")
                df_preprocessed = self._apply_schema_preprocessing(df, doc_type, vertical)
                
                # Canonize columns
                df_canonized = df_preprocessed.copy()
                for orig_col, canonical_col in mappings.items():
                    if orig_col in df_canonized.columns and orig_col != canonical_col:
                        if canonical_col not in df_canonized.columns:
                            df_canonized = df_canonized.rename(columns={orig_col: canonical_col})
            
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
            self.logger.info(f"  {doc_type}: {len(df):,} rows total")
        
        # Save classified data
        if output_format == "parquet":
            manifest_uri = self._save_classified_data_parquet(classified)
        else:
            manifest_uri = self.save_classified_data(classified)
        
        return {"classified_data": manifest_uri}
    
    def _save_classified_data_parquet(
        self,
        classified: Dict[str, pd.DataFrame],
        output_name: str = "classified_data",
    ) -> str:
        """
        Save CLASSIFIED_DATA output as Parquet files (more efficient).
        
        Args:
            classified: Dict mapping doc_type to DataFrame
            output_name: Base name for the output
            
        Returns:
            URI of the manifest file
        """
        refs = {}
        for doc_type, df in classified.items():
            uri = self.save_to_parquet(f"{output_name}/{doc_type}", df)
            refs[doc_type] = uri
            self.logger.info(f"Saved {doc_type}: {len(df):,} rows → {uri}")
        
        # Save manifest as JSON (small file)
        manifest_uri = self.save_to_s3(f"{output_name}_manifest", refs)
        return manifest_uri


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
