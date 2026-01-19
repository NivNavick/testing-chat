"""
Base Block Class.

Provides shared functionality for all blocks:
- S3 save/load operations
- DataFrame serialization (JSON and Parquet formats)
- DuckDB lazy loading for large-scale data processing
- Logging helpers
- Input validation
"""

import io
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import boto3
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BlockContext:
    """
    Runtime context passed to block execution.
    
    Contains all information needed to execute a block:
    - inputs: S3 URIs or local paths from previous blocks
    - params: Merged parameters
    - S3 client and bucket for data operations (optional for local mode)
    """
    inputs: Dict[str, str] = field(default_factory=dict)  # S3 URIs or local paths
    params: Dict[str, Any] = field(default_factory=dict)  # Merged parameters
    workflow_run_id: str = ""
    block_id: str = ""
    bucket: str = ""
    s3_client: Any = None  # boto3 S3 client
    local_storage_path: str = ""  # Local path for non-S3 storage
    
    @property
    def is_local_mode(self) -> bool:
        """Check if running in local mode (no S3)."""
        return not self.bucket and self.local_storage_path
    
    def __post_init__(self):
        """Initialize S3 client if not provided."""
        if self.s3_client is None and self.bucket:
            self.s3_client = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )


class BaseBlock(ABC):
    """
    Base class for all blocks with shared S3 and utility methods.
    
    Provides:
    - S3 save/load operations
    - DataFrame serialization (JSON format)
    - Logging helpers
    - Input validation
    
    Usage:
        class MyBlock(BaseBlock):
            def run(self) -> Dict[str, str]:
                data = self.load_input("my_input")
                result = process(data)
                return {"output": self.save_to_s3("output", result)}
    """
    
    def __init__(self, ctx: BlockContext):
        """
        Initialize the block with context.
        
        Args:
            ctx: BlockContext with inputs, params, and S3 configuration
        """
        self.ctx = ctx
        self.logger = logging.getLogger(f"block.{ctx.block_id}")
        
        # Initialize S3 client if not provided
        if ctx.s3_client is None and ctx.bucket:
            ctx.s3_client = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
    
    # ==================== S3 Operations ====================
    
    def save_to_s3(
        self,
        name: str,
        data: Any,
        format: Optional[str] = None,
    ) -> str:
        """
        Save data to S3 or local storage, return URI.
        
        Args:
            name: Output name (becomes filename)
            data: DataFrame, dict, list, or any JSON-serializable data
            format: Output format - "json" (default for dicts/lists), 
                   "parquet" (default for large DataFrames), or None (auto-detect)
            
        Returns:
            S3 URI or local file path
            
        Note:
            For DataFrames with >10K rows, Parquet is used by default for
            better performance. Use format="json" to force JSON output.
        """
        # Auto-detect format based on data type and size
        if format is None:
            if isinstance(data, pd.DataFrame):
                # Always use Parquet for DataFrames - 10x faster than JSON
                format = "parquet"
            else:
                format = "json"
        
        # Route to appropriate save method
        if format == "parquet" and isinstance(data, pd.DataFrame):
            return self.save_to_parquet(name, data)
        else:
            return self._save_to_json(name, data)
    
    def _save_to_json(self, name: str, data: Any) -> str:
        """
        Save data as JSON (legacy format for backwards compatibility).
        
        Args:
            name: Output name
            data: DataFrame, dict, list, or any JSON-serializable data
            
        Returns:
            File URI
        """
        # Convert DataFrame to list of records
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict(orient="records")
        else:
            json_data = data
        
        json_bytes = json.dumps(json_data, ensure_ascii=False, default=str).encode("utf-8")
        
        # Local mode - save to local filesystem
        if self.ctx.is_local_mode:
            local_dir = Path(self.ctx.local_storage_path) / self.ctx.workflow_run_id / self.ctx.block_id
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle nested paths (e.g., "classified_data/employee_shifts")
            file_path = local_dir / f"{name}.json"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_path.write_bytes(json_bytes)
            uri = str(file_path)
            self.logger.info(f"Saved {name} → {uri}")
            return uri
        
        # S3 mode
        if not self.ctx.bucket:
            raise ValueError("S3 bucket not configured in BlockContext")
        
        prefix = f"workflows/{self.ctx.workflow_run_id}/{self.ctx.block_id}/"
        s3_key = f"{prefix}{name}.json"
        
        # Upload to S3
        self.ctx.s3_client.put_object(
            Bucket=self.ctx.bucket,
            Key=s3_key,
            Body=json_bytes,
            ContentType="application/json",
        )
        
        s3_uri = f"s3://{self.ctx.bucket}/{s3_key}"
        self.logger.info(f"Saved {name} → {s3_uri}")
        return s3_uri
    
    def load_from_s3(self, uri: str) -> Any:
        """
        Load data from S3 URI or local file path.
        
        Automatically detects format based on file extension:
        - .parquet: Load as Parquet (fast, memory-efficient)
        - .json: Load as JSON (legacy format)
        
        Args:
            uri: S3 URI (s3://bucket/key) or local file path
            
        Returns:
            DataFrame if data is Parquet or list of dicts, otherwise raw JSON data
        """
        # Handle Parquet files
        if uri.endswith('.parquet'):
            return self._load_parquet(uri)
        
        # Check if it's a local file path (JSON)
        if not uri.startswith("s3://"):
            path = Path(uri)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(f"Local file not found: {uri}")
        else:
            # S3 mode
            bucket, key = self._parse_s3_uri(uri)
            response = self.ctx.s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))
        
        # Auto-convert list of dicts to DataFrame
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return pd.DataFrame(data)
        
        return data
    
    def _load_parquet(self, uri: str) -> pd.DataFrame:
        """
        Load a Parquet file from local path or S3.
        
        Args:
            uri: Local file path or S3 URI
            
        Returns:
            DataFrame
        """
        if not uri.startswith("s3://"):
            # Local file
            path = Path(uri)
            if path.exists():
                return pd.read_parquet(path)
            else:
                raise FileNotFoundError(f"Parquet file not found: {uri}")
        else:
            # S3 - download to temp and read
            import tempfile
            bucket, key = self._parse_s3_uri(uri)
            
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                self.ctx.s3_client.download_file(bucket, key, tmp.name)
                df = pd.read_parquet(tmp.name)
                Path(tmp.name).unlink()  # Clean up
                return df
    
    def load_input(self, name: str) -> Any:
        """
        Load an input by name from S3 or return directly if already loaded data.
        
        Args:
            name: Input port name
            
        Returns:
            Loaded data (DataFrame or raw)
        """
        if name not in self.ctx.inputs:
            raise ValueError(
                f"Input '{name}' not found. Available: {list(self.ctx.inputs.keys())}"
            )
        
        value = self.ctx.inputs[name]
        
        # If value is already data (list, dict, DataFrame), return it directly
        # This supports skip_upload mode where data is passed directly
        if isinstance(value, (list, dict, pd.DataFrame)):
            self.logger.info(f"Loading input '{name}' from direct data ({type(value).__name__})")
            return value
        
        # Otherwise, it's an S3 URI string
        self.logger.info(f"Loading input '{name}' from {value}")
        return self.load_from_s3(value)
    
    def get_correlated_data(self) -> pd.DataFrame:
        """
        Get correlated data from field-level inputs.
        
        When using field-level input syntax:
            inputs:
              - field: status
                source: early_arrival.status
              - field: name
                source: early_arrival.name
        
        The engine automatically correlates rows and provides
        them as a DataFrame via this method.
        
        Returns:
            DataFrame with correlated rows, or empty DataFrame if no field inputs
        """
        correlated = self.ctx.inputs.get("_correlated_data")
        
        if correlated is None:
            return pd.DataFrame()
        
        if isinstance(correlated, pd.DataFrame):
            return correlated
        
        if isinstance(correlated, list):
            return pd.DataFrame(correlated)
        
        return pd.DataFrame()
    
    def has_correlated_data(self) -> bool:
        """Check if correlated field data is available."""
        return "_correlated_data" in self.ctx.inputs
    
    def load_classified_data(self, input_name: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Load CLASSIFIED_DATA input (manifest + individual DataFrames).
        
        Args:
            input_name: Name of the classified_data input
            
        Returns:
            Dict mapping doc_type to DataFrame
        """
        if input_name not in self.ctx.inputs:
            raise ValueError(
                f"Input '{input_name}' not found. Available: {list(self.ctx.inputs.keys())}"
            )
        
        manifest_uri = self.ctx.inputs[input_name]
        manifest = self.load_from_s3(manifest_uri)
        
        # If manifest is already a DataFrame, it's a single document
        if isinstance(manifest, pd.DataFrame):
            return {input_name: manifest}
        
        # Load each document type from manifest
        result = {}
        for doc_type, s3_uri in manifest.items():
            self.logger.info(f"Loading {doc_type} from {s3_uri}")
            result[doc_type] = self.load_from_s3(s3_uri)
        
        return result
    
    def save_classified_data(
        self,
        classified: Dict[str, pd.DataFrame],
        output_name: str = "classified_data",
    ) -> str:
        """
        Save CLASSIFIED_DATA output (individual DataFrames + manifest).
        
        Args:
            classified: Dict mapping doc_type to DataFrame
            output_name: Base name for the output
            
        Returns:
            S3 URI of the manifest file
        """
        s3_refs = {}
        for doc_type, df in classified.items():
            s3_uri = self.save_to_s3(f"{output_name}/{doc_type}", df)
            s3_refs[doc_type] = s3_uri
            self.logger.info(f"Saved {doc_type}: {len(df)} rows → {s3_uri}")
        
        # Save manifest pointing to all files
        manifest_uri = self.save_to_s3(f"{output_name}_manifest", s3_refs)
        return manifest_uri
    
    # ==================== Local File Operations ====================
    
    def load_local_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a local CSV file.
        
        Args:
            file_path: Path to local CSV file
            
        Returns:
            DataFrame
        """
        self.logger.info(f"Loading local file: {file_path}")
        return pd.read_csv(file_path)
    
    def download_from_s3(self, s3_uri: str, local_path: str) -> str:
        """
        Download a file from S3 to local path.
        
        Args:
            s3_uri: S3 URI
            local_path: Local file path to save to
            
        Returns:
            Local file path
        """
        bucket, key = self._parse_s3_uri(s3_uri)
        self.ctx.s3_client.download_file(bucket, key, local_path)
        self.logger.info(f"Downloaded {s3_uri} → {local_path}")
        return local_path
    
    # ==================== Utilities ====================
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter with optional default."""
        return self.ctx.params.get(name, default)
    
    def require_param(self, name: str) -> Any:
        """Get a required parameter, raise if missing."""
        if name not in self.ctx.params:
            raise ValueError(f"Required parameter '{name}' not provided")
        return self.ctx.params[name]
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple:
        """Parse s3://bucket/key into (bucket, key)."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        parts = s3_uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        return parts[0], parts[1]
    
    # ==================== DuckDB Integration ====================
    
    @property
    def duckdb(self) -> duckdb.DuckDBPyConnection:
        """
        Get DuckDB connection for SQL operations on large datasets.
        
        Uses the singleton DuckDBManager with memory and threading optimizations.
        
        Returns:
            DuckDB connection
        """
        from csv_analyzer.core.duckdb_manager import get_duckdb
        return get_duckdb().conn
    
    @property
    def duckdb_manager(self):
        """
        Get the DuckDB manager for advanced operations.
        
        Returns:
            DuckDBManager instance
        """
        from csv_analyzer.core.duckdb_manager import get_duckdb
        return get_duckdb()
    
    def load_input_lazy(self, name: str) -> "duckdb.DuckDBPyRelation":
        """
        Load input as a DuckDB lazy relation.
        
        Data is NOT loaded into memory until .fetchdf() or .fetchall() is called.
        This enables processing files larger than available RAM.
        
        Args:
            name: Input port name
            
        Returns:
            DuckDB relation (lazy - no memory used yet)
            
        Example:
            # Process millions of rows without memory issues
            relation = self.load_input_lazy("data")
            filtered = self.query("SELECT * FROM data WHERE status = 'EARLY'", data=relation)
            df = filtered.fetchdf()  # Only now is data loaded
        """
        if name not in self.ctx.inputs:
            raise ValueError(
                f"Input '{name}' not found. Available: {list(self.ctx.inputs.keys())}"
            )
        
        value = self.ctx.inputs[name]
        
        # If value is already a DataFrame, convert to relation
        if isinstance(value, pd.DataFrame):
            self.logger.info(f"Loading input '{name}' as lazy relation from DataFrame")
            return self.duckdb.from_df(value)
        
        # If value is a DuckDB relation, return it directly
        if isinstance(value, duckdb.DuckDBPyRelation):
            return value
        
        # Otherwise, it's a file path/URI - load lazily based on format
        uri = str(value)
        self.logger.info(f"Loading input '{name}' as lazy relation from {uri}")
        
        if uri.endswith('.parquet'):
            return self.duckdb_manager.read_parquet_lazy(uri)
        elif uri.endswith('.csv'):
            return self.duckdb_manager.read_csv_lazy(uri)
        elif uri.endswith('.json'):
            # For JSON, we need to load via pandas first (DuckDB JSON support varies)
            df = self.load_from_s3(uri)
            return self.duckdb.from_df(df)
        else:
            # Try JSON as default (backwards compatibility)
            df = self.load_from_s3(uri)
            return self.duckdb.from_df(df)
    
    def query(
        self,
        sql: str,
        **tables: "Union[pd.DataFrame, duckdb.DuckDBPyRelation]",
    ) -> "duckdb.DuckDBPyRelation":
        """
        Execute SQL query with named tables.
        
        Tables can be DataFrames or DuckDB relations. They are registered
        as temporary tables for the query.
        
        Args:
            sql: SQL query string referencing table names
            **tables: Named tables (DataFrame or DuckDB relation)
            
        Returns:
            DuckDB relation with query results (lazy)
            
        Example:
            result = self.query('''
                SELECT 
                    employee_name,
                    COUNT(*) as early_count,
                    SUM(minutes_early) as total_minutes
                FROM arrivals
                WHERE status = 'EARLY'
                GROUP BY employee_name
            ''', arrivals=arrivals_df)
            
            df = result.fetchdf()  # Execute and get DataFrame
        """
        return self.duckdb_manager.query(sql, **tables)
    
    def save_to_parquet(
        self,
        name: str,
        data: "Union[pd.DataFrame, duckdb.DuckDBPyRelation]",
        compression: str = "zstd",
    ) -> str:
        """
        Save data to Parquet format (streaming, memory-efficient).
        
        For DuckDB relations, data is streamed directly to disk without
        loading into pandas first. This enables saving datasets larger
        than available memory.
        
        Args:
            name: Output name (becomes filename)
            data: DataFrame or DuckDB relation
            compression: Compression codec (zstd, snappy, gzip, none)
            
        Returns:
            File path to saved Parquet file
            
        Example:
            # Save large query result without memory issues
            result = self.query("SELECT * FROM huge_table WHERE ...")
            path = self.save_to_parquet("filtered_data", result)
        """
        # Determine output path
        if self.ctx.is_local_mode:
            local_dir = Path(self.ctx.local_storage_path) / self.ctx.workflow_run_id / self.ctx.block_id
            local_dir.mkdir(parents=True, exist_ok=True)
            file_path = local_dir / f"{name}.parquet"
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # For S3, write to temp then upload
            import tempfile
            file_path = Path(tempfile.mkdtemp()) / f"{name}.parquet"
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write data
        if isinstance(data, duckdb.DuckDBPyRelation):
            # Streaming write from DuckDB relation
            self.duckdb.register("_output_data", data)
            self.duckdb.execute(f"""
                COPY (SELECT * FROM _output_data) 
                TO '{file_path}' (FORMAT PARQUET, COMPRESSION {compression})
            """)
        elif isinstance(data, pd.DataFrame):
            # Direct pandas write
            data.to_parquet(str(file_path), compression=compression, index=False)
        else:
            raise TypeError(f"Unsupported data type for Parquet: {type(data)}")
        
        # Upload to S3 if not in local mode
        if not self.ctx.is_local_mode:
            prefix = f"workflows/{self.ctx.workflow_run_id}/{self.ctx.block_id}/"
            s3_key = f"{prefix}{name}.parquet"
            
            self.ctx.s3_client.upload_file(
                str(file_path),
                self.ctx.bucket,
                s3_key,
            )
            
            uri = f"s3://{self.ctx.bucket}/{s3_key}"
            # Clean up temp file
            file_path.unlink()
        else:
            uri = str(file_path)
        
        self.logger.info(f"Saved Parquet {name} → {uri}")
        return uri
    
    def load_parquet_lazy(self, path: str) -> "duckdb.DuckDBPyRelation":
        """
        Load a Parquet file as a lazy DuckDB relation.
        
        Args:
            path: Path to Parquet file
            
        Returns:
            DuckDB relation (lazy)
        """
        return self.duckdb_manager.read_parquet_lazy(path)
    
    def load_csv_lazy(self, path: str) -> "duckdb.DuckDBPyRelation":
        """
        Load a CSV file as a lazy DuckDB relation.
        
        Args:
            path: Path to CSV file
            
        Returns:
            DuckDB relation (lazy)
        """
        return self.duckdb_manager.read_csv_lazy(path)
    
    def sample_file(
        self,
        path: str,
        n: int = 5000,
        method: str = "reservoir",
    ) -> pd.DataFrame:
        """
        Sample rows from a file without loading it entirely.
        
        Useful for profiling or classification where you don't need
        the full dataset.
        
        Args:
            path: Path to CSV or Parquet file
            n: Number of rows to sample
            method: "reservoir" (uniform random) or "system" (faster, less uniform)
            
        Returns:
            Sampled DataFrame
        """
        return self.duckdb_manager.sample_file(path, n, method)
    
    def iter_input_chunks(
        self,
        name: str,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over input data in chunks.
        
        Useful for processing files that don't fit in memory.
        
        Args:
            name: Input port name
            chunk_size: Rows per chunk
            
        Yields:
            DataFrame chunks
        """
        if name not in self.ctx.inputs:
            raise ValueError(
                f"Input '{name}' not found. Available: {list(self.ctx.inputs.keys())}"
            )
        
        value = self.ctx.inputs[name]
        
        if isinstance(value, pd.DataFrame):
            # Chunk the DataFrame
            for i in range(0, len(value), chunk_size):
                yield value.iloc[i:i + chunk_size]
        elif isinstance(value, str):
            # File path - use DuckDB chunking
            yield from self.duckdb_manager.iter_chunks(value, chunk_size)
        else:
            raise TypeError(f"Cannot iterate chunks for type: {type(value)}")
    
    # ==================== Abstract Method ====================
    
    @abstractmethod
    def run(self) -> Dict[str, str]:
        """
        Execute the block logic.
        
        Returns:
            Dict mapping output_name to S3 URI
        """
        pass

