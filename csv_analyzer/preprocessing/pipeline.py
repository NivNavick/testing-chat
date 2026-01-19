"""
Preprocessing Pipeline Orchestrator.

Combines structure detection, value transformations, and row filtering
into a unified preprocessing pipeline.

Supports two execution modes:
1. Legacy: Pandas-based transforms with DataFrame copies
2. Optimized: DuckDB SQL-based transforms for large files
   - Zero-copy column operations
   - Streaming I/O to Parquet
   - Out-of-core processing for files larger than memory
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

from csv_analyzer.preprocessing.structure_detector import (
    CSVStructure,
    StructureDetector,
)
from csv_analyzer.preprocessing.transformers import (
    TransformConfig,
    TransformResult,
    ValueTransformer,
    LocationNormalizer,
    LocationRule,
)
from csv_analyzer.preprocessing.row_filter import (
    FilterConfig,
    FilterResult,
    RowFilter,
)
from csv_analyzer.sessions.session import ExtractedMetadata

logger = logging.getLogger(__name__)

# Threshold for switching to DuckDB mode
LARGE_FILE_THRESHOLD = 50_000  # rows


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing a CSV file.
    
    Loaded from the 'preprocessing' section of a schema YAML file.
    """
    # Structure detection hints
    structure_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Column value transformations
    column_transforms: List[TransformConfig] = field(default_factory=list)
    
    # Row filters
    row_filters: List[FilterConfig] = field(default_factory=list)
    
    # General options
    auto_detect_structure: bool = True
    skip_empty_rows: bool = True
    
    # Metadata injection options
    inject_metadata_columns: bool = True  # Add extracted metadata as columns
    metadata_column_prefix: str = "_meta_"  # Prefix for metadata columns
    
    # Location normalization options
    normalize_locations: bool = True  # Apply location normalization
    location_output_column: str = "_normalized_location"
    location_include_display: bool = True
    location_source_columns: Optional[List[str]] = None  # None = use defaults
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingConfig":
        """Create from dictionary (from YAML preprocessing section)."""
        column_transforms = [
            TransformConfig.from_dict(t) 
            for t in data.get("column_transforms", [])
        ]
        
        row_filters = [
            FilterConfig.from_dict(f)
            for f in data.get("row_filters", [])
        ]
        
        return cls(
            structure_hints=data.get("structure_hints", {}),
            column_transforms=column_transforms,
            row_filters=row_filters,
            auto_detect_structure=data.get("auto_detect_structure", True),
            skip_empty_rows=data.get("skip_empty_rows", True),
            inject_metadata_columns=data.get("inject_metadata_columns", True),
            metadata_column_prefix=data.get("metadata_column_prefix", "_meta_"),
            normalize_locations=data.get("normalize_locations", True),
            location_output_column=data.get("location_output_column", "_normalized_location"),
            location_include_display=data.get("location_include_display", True),
            location_source_columns=data.get("location_source_columns"),
        )
    
    @classmethod
    def from_yaml_file(cls, yaml_path: Union[str, Path]) -> "PreprocessingConfig":
        """Load preprocessing config from a YAML file."""
        path = Path(yaml_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        preprocessing_data = data.get("preprocessing", {})
        return cls.from_dict(preprocessing_data)


@dataclass
class ProcessedCSV:
    """
    Result of preprocessing a CSV file.
    
    Contains the processed DataFrame and all metadata about
    the preprocessing steps that were applied.
    """
    df: pd.DataFrame
    original_path: Optional[str] = None
    
    # Structure detection results
    structure: Optional[CSVStructure] = None
    extracted_metadata: Optional[ExtractedMetadata] = None
    
    # Transformation results
    transform_result: Optional[TransformResult] = None
    
    # Filter results
    filter_result: Optional[FilterResult] = None
    
    # Summary stats
    original_rows: int = 0
    final_rows: int = 0
    columns_added: List[str] = field(default_factory=list)
    columns_modified: List[str] = field(default_factory=list)
    metadata_columns_added: List[str] = field(default_factory=list)
    location_columns_added: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (without DataFrame)."""
        return {
            "original_path": self.original_path,
            "structure": self.structure.to_dict() if self.structure else None,
            "extracted_metadata": self.extracted_metadata.to_dict() if self.extracted_metadata else None,
            "original_rows": self.original_rows,
            "final_rows": self.final_rows,
            "columns_added": self.columns_added,
            "columns_modified": self.columns_modified,
            "metadata_columns_added": self.metadata_columns_added,
            "location_columns_added": self.location_columns_added,
            "transform_stats": {
                "applied": self.transform_result.transformations_applied if self.transform_result else 0,
                "rows_affected": self.transform_result.rows_affected if self.transform_result else 0,
            },
            "filter_stats": {
                "applied": self.filter_result.filters_applied if self.filter_result else 0,
                "rows_removed": self.filter_result.rows_removed if self.filter_result else 0,
            },
        }


class PreprocessingPipeline:
    """
    Orchestrates CSV preprocessing.
    
    Pipeline stages:
    1. Structure detection (multi-row headers, metadata extraction)
    2. Header normalization (load CSV with correct header row)
    3. Value transformations (split ranges, normalize dates, clean prefixes)
    4. Row filtering (SQL-based filtering)
    
    Usage:
        pipeline = PreprocessingPipeline()
        
        # Option 1: With schema file
        result = pipeline.process(
            "shifts.csv",
            schema_path="schemas/medical/employee_shifts.yaml"
        )
        
        # Option 2: With explicit config
        config = PreprocessingConfig(
            column_transforms=[
                TransformConfig(
                    source_column="כניסה",
                    transform_type="clean_prefix",
                    pattern=r"^\\*\\s*",
                ),
            ],
        )
        result = pipeline.process("shifts.csv", config=config)
        
        print(result.df)
        print(result.extracted_metadata)
    """
    
    def __init__(
        self,
        openai_client=None,
        context: Optional[Dict[str, Any]] = None,
        context_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            openai_client: Optional OpenAI client for AI-based structure detection
            context: Optional context dict (date_range, etc.)
            context_path: Optional path to context YAML (for location normalization rules)
        """
        self.openai_client = openai_client
        self.context = context or {}
        self.context_path = context_path
        
        self.structure_detector = StructureDetector(openai_client)
        self.value_transformer = ValueTransformer(context)
        self.row_filter = RowFilter()
        
        # Load location normalizer from context if provided
        self.location_normalizer: Optional[LocationNormalizer] = None
        if context_path:
            self.location_normalizer = self.load_location_normalizer_from_context(context_path)
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value for transformations."""
        self.context[key] = value
        self.value_transformer.set_context(key, value)
    
    def inject_metadata_columns(
        self,
        df: pd.DataFrame,
        metadata: ExtractedMetadata,
        prefix: str = "_meta_",
    ) -> tuple:
        """
        Inject extracted metadata as columns into the DataFrame.
        
        Each metadata field becomes a column with the same value in every row.
        This allows SQL queries to reference the metadata.
        
        Args:
            df: Input DataFrame
            metadata: Extracted metadata from structure detection
            prefix: Prefix for metadata column names (default: "_meta_")
            
        Returns:
            Tuple of (modified_df, list of added column names)
        """
        columns_added = []
        
        # Map metadata fields to column names
        metadata_fields = {
            "employee_name": metadata.employee_name,
            "employee_id": metadata.employee_id,
            "location": metadata.location,
            "date_range_start": metadata.date_range_start,
            "date_range_end": metadata.date_range_end,
            "department": metadata.department,
            "company_name": metadata.company_name,
            "contract_type": metadata.contract_type,
        }
        
        # Add custom metadata fields
        for key, value in metadata.custom.items():
            metadata_fields[key] = value
        
        # Inject non-null fields as columns
        for field_name, value in metadata_fields.items():
            if value is not None:
                col_name = f"{prefix}{field_name}"
                df[col_name] = value
                columns_added.append(col_name)
                logger.debug(f"Injected metadata column: {col_name} = {value}")
        
        if columns_added:
            logger.info(f"Injected {len(columns_added)} metadata columns: {columns_added}")
        
        return df, columns_added
    
    def load_location_normalizer_from_context(
        self,
        context_path: Union[str, Path],
    ) -> Optional[LocationNormalizer]:
        """
        Load location normalization rules from a context YAML file.
        
        Args:
            context_path: Path to context YAML file (e.g., contexts/medical.yaml)
            
        Returns:
            LocationNormalizer if rules found, None otherwise
        """
        path = Path(context_path)
        
        if not path.exists():
            logger.warning(f"Context file not found: {context_path}")
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            context_data = yaml.safe_load(f)
        
        loc_config = context_data.get("location_normalization")
        if not loc_config:
            return None
        
        rules = [
            LocationRule.from_dict(r)
            for r in loc_config.get("rules", [])
        ]
        
        if not rules:
            return None
        
        normalizer = LocationNormalizer(rules)
        logger.info(f"Loaded {len(rules)} location normalization rules from {context_path}")
        
        return normalizer
    
    def apply_location_normalization(
        self,
        df: pd.DataFrame,
        context_path: Optional[Union[str, Path]] = None,
        normalizer: Optional[LocationNormalizer] = None,
        source_columns: Optional[List[str]] = None,
        output_column: str = "_normalized_location",
        include_display: bool = True,
    ) -> tuple:
        """
        Apply location normalization to a DataFrame.
        
        Normalizes various location representations to canonical identifiers,
        enabling cross-document joins.
        
        Args:
            df: Input DataFrame
            context_path: Path to context YAML with location rules
            normalizer: Pre-configured LocationNormalizer (overrides context_path)
            source_columns: Columns to check for location (in priority order)
            output_column: Name for the normalized location column
            include_display: Also add a display name column
            
        Returns:
            Tuple of (modified DataFrame, list of columns added)
        """
        # Get or create normalizer
        if normalizer is None and context_path:
            normalizer = self.load_location_normalizer_from_context(context_path)
        
        if normalizer is None:
            logger.debug("No location normalizer available, skipping")
            return df, []
        
        # Default source columns
        if source_columns is None:
            source_columns = [
                "_meta_location",
                "department",
                "מחלקה",
                "location",
                "branch",
                "סניף",
            ]
        
        # Filter to columns that exist
        existing_cols = [c for c in source_columns if c in df.columns]
        
        if not existing_cols:
            logger.debug(f"No source columns found for location normalization: {source_columns}")
            return df, []
        
        logger.info(f"Applying location normalization using columns: {existing_cols}")
        
        df, rows_affected = normalizer.apply_to_dataframe(
            df,
            existing_cols,
            output_column=output_column,
            include_display=include_display,
        )
        
        columns_added = [output_column]
        if include_display:
            columns_added.append(f"{output_column}_display")
        
        logger.info(f"Location normalization: {rows_affected}/{len(df)} rows matched")
        
        return df, columns_added
    
    def process(
        self,
        csv_file: Union[str, Path, pd.DataFrame],
        schema_path: Optional[Union[str, Path]] = None,
        config: Optional[PreprocessingConfig] = None,
        detect_structure: bool = True,
    ) -> ProcessedCSV:
        """
        Process a CSV file through the preprocessing pipeline.
        
        Args:
            csv_file: Path to CSV file or DataFrame
            schema_path: Path to schema YAML with preprocessing section
            config: Explicit preprocessing configuration (overrides schema)
            detect_structure: Whether to detect multi-row headers
            
        Returns:
            ProcessedCSV with the processed DataFrame and metadata
        """
        # Load config from schema if provided
        if config is None and schema_path is not None:
            config = PreprocessingConfig.from_yaml_file(schema_path)
        elif config is None:
            config = PreprocessingConfig()
        
        # Track original path
        original_path = str(csv_file) if not isinstance(csv_file, pd.DataFrame) else None
        
        logger.info(f"Starting preprocessing pipeline for: {original_path or 'DataFrame'}")
        
        # Stage 1: Structure detection
        structure = None
        extracted_metadata = None
        
        if detect_structure and config.auto_detect_structure and not isinstance(csv_file, pd.DataFrame):
            logger.info("Stage 1: Detecting CSV structure...")
            structure = self.structure_detector.detect(csv_file, use_ai=self.openai_client is not None)
            
            if structure.has_multi_row_header:
                logger.info(
                    f"Detected multi-row header: header_row={structure.header_row}, "
                    f"metadata_rows={len(structure.metadata_rows)}"
                )
                
                # Extract metadata and update context
                meta_dict = structure.get_extracted_metadata()
                extracted_metadata = ExtractedMetadata(
                    employee_name=meta_dict.get("employee_name"),
                    employee_id=meta_dict.get("employee_id"),
                    location=meta_dict.get("location"),
                    date_range_start=meta_dict.get("date_range_start"),
                    date_range_end=meta_dict.get("date_range_end"),
                    department=meta_dict.get("department"),
                    company_name=meta_dict.get("company_name"),
                    contract_type=meta_dict.get("contract_type"),
                )
                
                # Update context with extracted values
                if meta_dict.get("date_range_start"):
                    self.set_context("date_range_start", meta_dict["date_range_start"])
                if meta_dict.get("date_range_end"):
                    self.set_context("date_range_end", meta_dict["date_range_end"])
        
        # Stage 2: Load DataFrame with correct structure
        logger.info("Stage 2: Loading and normalizing headers...")
        if isinstance(csv_file, pd.DataFrame):
            df = csv_file.copy()
        elif structure and structure.has_multi_row_header:
            df, _ = self.structure_detector.normalize_csv(csv_file, structure)
        else:
            df = pd.read_csv(csv_file)
        
        original_rows = len(df)
        
        # Skip empty rows if configured
        if config.skip_empty_rows:
            df = df.dropna(how='all')
            if len(df) < original_rows:
                logger.info(f"Removed {original_rows - len(df)} empty rows")
        
        # Stage 2.5: Inject metadata columns
        metadata_columns_added = []
        if config.inject_metadata_columns and extracted_metadata:
            logger.info("Stage 2.5: Injecting metadata columns...")
            df, metadata_columns_added = self.inject_metadata_columns(
                df, 
                extracted_metadata,
                prefix=config.metadata_column_prefix,
            )
        
        # Stage 2.6: Location normalization
        location_columns_added = []
        if config.normalize_locations and self.location_normalizer:
            logger.info("Stage 2.6: Applying location normalization...")
            df, location_columns_added = self.apply_location_normalization(
                df,
                normalizer=self.location_normalizer,
                source_columns=config.location_source_columns,
                output_column=config.location_output_column,
                include_display=config.location_include_display,
            )
        
        # Stage 3: Value transformations
        transform_result = None
        columns_added = []
        columns_modified = []
        
        if config.column_transforms:
            logger.info(f"Stage 3: Applying {len(config.column_transforms)} transformations...")
            transform_result = self.value_transformer.apply_all(df, config.column_transforms)
            df = transform_result.df
            columns_added = transform_result.columns_added
            columns_modified = transform_result.columns_modified
            
            if transform_result.errors:
                for error in transform_result.errors:
                    logger.warning(f"Transform error: {error}")
        else:
            logger.info("Stage 3: No transformations configured, skipping...")
        
        # Stage 4: Row filtering
        filter_result = None
        
        if config.row_filters:
            logger.info(f"Stage 4: Applying {len(config.row_filters)} filters...")
            filter_result = self.row_filter.apply_all(df, config.row_filters)
            df = filter_result.df
            
            if filter_result.errors:
                for error in filter_result.errors:
                    logger.warning(f"Filter error: {error}")
        else:
            logger.info("Stage 4: No filters configured, skipping...")
        
        final_rows = len(df)
        
        logger.info(
            f"Preprocessing complete: {original_rows} → {final_rows} rows "
            f"({original_rows - final_rows} removed)"
        )
        
        return ProcessedCSV(
            df=df,
            original_path=original_path,
            structure=structure,
            extracted_metadata=extracted_metadata,
            transform_result=transform_result,
            filter_result=filter_result,
            original_rows=original_rows,
            final_rows=final_rows,
            columns_added=columns_added,
            columns_modified=columns_modified,
            metadata_columns_added=metadata_columns_added,
            location_columns_added=location_columns_added,
        )
    
    def process_with_auto_transforms(
        self,
        csv_file: Union[str, Path],
        detect_time_ranges: bool = True,
        detect_hebrew_dates: bool = True,
        clean_time_prefixes: bool = True,
    ) -> ProcessedCSV:
        """
        Process a CSV with automatic transform detection.
        
        Analyzes columns and applies appropriate transformations automatically.
        
        Args:
            csv_file: Path to CSV file
            detect_time_ranges: Auto-detect and split time range columns
            detect_hebrew_dates: Auto-detect and normalize Hebrew date columns
            clean_time_prefixes: Auto-clean time column prefixes (e.g., "* 14:30")
            
        Returns:
            ProcessedCSV with the processed DataFrame
        """
        import re
        
        # First, detect structure and load
        structure = self.structure_detector.detect(csv_file)
        
        if structure.has_multi_row_header:
            df, _ = self.structure_detector.normalize_csv(csv_file, structure)
            
            # Update context from metadata
            meta = structure.get_extracted_metadata()
            for key, value in meta.items():
                if value:
                    self.set_context(key, value)
        else:
            df = pd.read_csv(csv_file)
        
        # Auto-detect transforms
        transforms = []
        
        for col in df.columns:
            sample_values = df[col].dropna().head(10).tolist()
            if not sample_values:
                continue
            
            sample_str = str(sample_values[0])
            
            # Detect time ranges (e.g., "19:00 - 19:15")
            if detect_time_ranges:
                time_range_pattern = r'^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}$'
                if re.match(time_range_pattern, sample_str):
                    transforms.append(TransformConfig(
                        source_column=col,
                        transform_type="split_range",
                        pattern=r'^(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})$',
                        output_columns=[
                            {"name": f"{col}_start", "group": 1},
                            {"name": f"{col}_end", "group": 2},
                        ],
                    ))
                    continue
            
            # Detect Hebrew dates (e.g., "ב - 01")
            if detect_hebrew_dates:
                hebrew_date_pattern = r'^[אבגדהוש]\s*-\s*\d{1,2}$'
                if re.match(hebrew_date_pattern, sample_str):
                    transforms.append(TransformConfig(
                        source_column=col,
                        transform_type="hebrew_date_normalize",
                    ))
                    continue
            
            # Detect time prefixes (e.g., "* 14:30")
            if clean_time_prefixes:
                time_prefix_pattern = r'^\*\s*\d{1,2}:\d{2}$'
                if re.match(time_prefix_pattern, sample_str):
                    transforms.append(TransformConfig(
                        source_column=col,
                        transform_type="clean_prefix",
                        pattern=r'^\*\s*',
                    ))
                    continue
        
        if transforms:
            logger.info(f"Auto-detected {len(transforms)} transformations")
        
        # Build extracted metadata
        extracted_metadata = None
        if structure.metadata_rows:
            meta_dict = structure.get_extracted_metadata()
            extracted_metadata = ExtractedMetadata(
                employee_name=meta_dict.get("employee_name"),
                employee_id=meta_dict.get("employee_id"),
                location=meta_dict.get("location"),
                date_range_start=meta_dict.get("date_range_start"),
                date_range_end=meta_dict.get("date_range_end"),
                department=meta_dict.get("department"),
                company_name=meta_dict.get("company_name"),
                contract_type=meta_dict.get("contract_type"),
            )
        
        # Inject metadata columns
        metadata_columns_added = []
        if extracted_metadata:
            df, metadata_columns_added = self.inject_metadata_columns(df, extracted_metadata)
        
        # Apply location normalization if normalizer available
        location_columns_added = []
        if self.location_normalizer:
            df, location_columns_added = self.apply_location_normalization(
                df,
                normalizer=self.location_normalizer,
            )
        
        # Apply transforms
        config = PreprocessingConfig(
            column_transforms=transforms,
            auto_detect_structure=False,  # Already done
            inject_metadata_columns=False,  # Already done above
            normalize_locations=False,  # Already done above
        )
        
        # Process with existing DataFrame
        result = self.process(df, config=config, detect_structure=False)
        result.original_path = str(csv_file)
        result.structure = structure
        result.extracted_metadata = extracted_metadata
        result.metadata_columns_added = metadata_columns_added
        result.location_columns_added = location_columns_added
        
        return result
    
    def close(self) -> None:
        """Release resources."""
        self.row_filter.close()
    
    # =========================================================================
    # DuckDB-based Processing (for large files)
    # =========================================================================
    
    def process_with_duckdb(
        self,
        csv_file: Union[str, Path],
        output_path: Optional[str] = None,
        transforms: Optional[List[TransformConfig]] = None,
    ) -> str:
        """
        Process a CSV using DuckDB SQL transforms for large-scale data.
        
        This method uses streaming I/O and zero-copy operations,
        enabling processing of files larger than available memory.
        
        Args:
            csv_file: Path to input CSV or Parquet file
            output_path: Path for output Parquet file (auto-generated if None)
            transforms: List of transform configurations
            
        Returns:
            Path to output Parquet file
            
        Example:
            pipeline = PreprocessingPipeline()
            output = pipeline.process_with_duckdb(
                "huge_file.csv",
                transforms=[
                    TransformConfig(
                        source_column="time_range",
                        transform_type="split_range",
                        pattern=r'^(\\d{2}:\\d{2})\\s*-\\s*(\\d{2}:\\d{2})$',
                        output_columns=[
                            {"name": "start_time", "group": 1},
                            {"name": "end_time", "group": 2},
                        ],
                    ),
                ],
            )
        """
        import duckdb
        
        try:
            from csv_analyzer.core.duckdb_manager import get_duckdb
            conn = get_duckdb().conn
        except ImportError:
            conn = duckdb.connect(":memory:")
        
        csv_path = str(csv_file)
        
        # Determine output path
        if output_path is None:
            output_path = str(Path(csv_path).with_suffix('.processed.parquet'))
        
        # Get columns from file
        if csv_path.endswith('.parquet'):
            read_func = f"read_parquet('{csv_path}')"
        else:
            read_func = f"read_csv_auto('{csv_path}')"
        
        cols_result = conn.execute(f"""
            SELECT column_name FROM (DESCRIBE SELECT * FROM {read_func})
        """).fetchall()
        all_columns = [row[0] for row in cols_result]
        
        # Build SQL select expressions for transforms
        select_exprs = self._build_sql_transforms(all_columns, transforms or [])
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Execute transform and write to Parquet (streaming)
        sql = f"""
            COPY (
                SELECT {', '.join(select_exprs)}
                FROM {read_func}
            )
            TO '{output_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        
        logger.info(f"Executing DuckDB transform pipeline...")
        conn.execute(sql)
        
        logger.info(f"Processed file saved to {output_path}")
        return output_path
    
    def _build_sql_transforms(
        self,
        columns: List[str],
        transforms: List[TransformConfig],
    ) -> List[str]:
        """
        Build SQL SELECT expressions for transforms.
        
        Converts TransformConfig objects to SQL expressions.
        
        Args:
            columns: List of column names
            transforms: List of transform configurations
            
        Returns:
            List of SQL SELECT expressions
        """
        # Build map of source column -> transform
        transform_map = {t.source_column: t for t in transforms if t.source_column}
        
        # Columns to exclude (they're being transformed into new columns)
        exclude_cols = set()
        
        select_exprs = []
        additional_exprs = []  # New columns from transforms
        
        for col in columns:
            if col in transform_map:
                transform = transform_map[col]
                
                if transform.transform_type == "split_range":
                    # Split "19:00 - 19:15" into two columns
                    pattern = transform.pattern or r'^(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})$'
                    
                    for out_col in transform.output_columns:
                        group = out_col.get("group", 1)
                        out_name = out_col.get("name", f"{col}_{group}")
                        additional_exprs.append(
                            f"REGEXP_EXTRACT(\"{col}\", '{pattern}', {group}) AS \"{out_name}\""
                        )
                    
                    # Keep original column unless explicitly removed
                    if not transform.drop_source:
                        select_exprs.append(f'"{col}"')
                    else:
                        exclude_cols.add(col)
                
                elif transform.transform_type == "clean_prefix":
                    # Remove prefix like "* " from values
                    pattern = transform.pattern or r'^\*\s*'
                    select_exprs.append(
                        f"REGEXP_REPLACE(\"{col}\", '{pattern}', '') AS \"{col}\""
                    )
                
                elif transform.transform_type == "hebrew_date_normalize":
                    # Normalize Hebrew dates - basic implementation
                    # This is a simplified version; full implementation would need more logic
                    select_exprs.append(f'"{col}"')
                
                elif transform.transform_type == "rename":
                    # Rename column
                    new_name = transform.output_column or col
                    select_exprs.append(f'"{col}" AS "{new_name}"')
                
                else:
                    # Unknown transform - keep original
                    select_exprs.append(f'"{col}"')
            else:
                # No transform for this column
                if col not in exclude_cols:
                    select_exprs.append(f'"{col}"')
        
        # Add new columns from transforms
        select_exprs.extend(additional_exprs)
        
        return select_exprs
    
    def count_rows_fast(self, file_path: str) -> int:
        """
        Count rows in a file using DuckDB (fast, memory-efficient).
        
        For Parquet files, this reads metadata only.
        For CSV files, this scans but doesn't load into memory.
        """
        import duckdb
        
        try:
            from csv_analyzer.core.duckdb_manager import get_duckdb
            conn = get_duckdb().conn
        except ImportError:
            conn = duckdb.connect(":memory:")
        
        if file_path.endswith('.parquet'):
            result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{file_path}')").fetchone()
        else:
            result = conn.execute(f"SELECT COUNT(*) FROM read_csv_auto('{file_path}')").fetchone()
        
        return result[0] if result else 0
    
    def is_large_file(self, file_path: str, threshold: int = LARGE_FILE_THRESHOLD) -> bool:
        """Check if a file is large enough to warrant DuckDB processing."""
        return self.count_rows_fast(file_path) > threshold


# Factory function for easy creation
def create_pipeline(
    openai_client=None,
    context: Optional[Dict[str, Any]] = None,
    context_path: Optional[Union[str, Path]] = None,
) -> PreprocessingPipeline:
    """
    Create a preprocessing pipeline.
    
    Args:
        openai_client: Optional OpenAI client for AI structure detection
        context: Optional context dict
        context_path: Optional path to context YAML (for location rules)
        
    Returns:
        PreprocessingPipeline instance
    """
    return PreprocessingPipeline(openai_client, context, context_path)


def process_large_file(
    input_path: str,
    output_path: Optional[str] = None,
    transforms: Optional[List[TransformConfig]] = None,
    context_path: Optional[str] = None,
) -> str:
    """
    Convenience function to process a large file using DuckDB.
    
    Args:
        input_path: Path to input CSV or Parquet file
        output_path: Path for output (auto-generated if None)
        transforms: List of transform configurations
        context_path: Optional path to context YAML
        
    Returns:
        Path to output Parquet file
    """
    pipeline = create_pipeline(context_path=context_path)
    return pipeline.process_with_duckdb(input_path, output_path, transforms)

