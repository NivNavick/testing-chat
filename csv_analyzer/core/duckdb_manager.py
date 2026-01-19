"""
DuckDB Connection Manager.

Singleton manager for DuckDB connections with optimizations for large-scale
data processing. Provides lazy evaluation, out-of-core processing, and
efficient I/O operations.
"""

import logging
import os
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional, Union

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBManager:
    """
    Singleton manager for DuckDB connections with large-data optimizations.
    
    Features:
    - Memory limit configuration for out-of-core processing
    - Thread pool configuration for parallel execution
    - Temp directory for spill-to-disk operations
    - Lazy evaluation via DuckDB Relations
    
    Usage:
        manager = get_duckdb()
        
        # Lazy read (no memory used until .fetchdf())
        relation = manager.read_csv_lazy("large_file.csv")
        
        # Execute SQL on lazy relations
        result = manager.execute('''
            SELECT * FROM relation WHERE col > 100
        ''')
        
        # Only now is data loaded (and only matching rows)
        df = result.fetchdf()
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
        
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._config: Dict[str, Any] = {}
        self._initialized = True
    
    def _init_connection(
        self,
        memory_limit: str = "8GB",
        threads: int = 4,
        temp_directory: Optional[str] = None,
    ) -> None:
        """
        Initialize DuckDB connection with optimizations.
        
        Args:
            memory_limit: Maximum memory DuckDB can use (e.g., "8GB", "4GB")
            threads: Number of threads for parallel execution
            temp_directory: Directory for spill-to-disk operations
        """
        if self._conn is not None:
            return
        
        self._conn = duckdb.connect(":memory:")
        
        # Configure memory limit for out-of-core processing
        self._conn.execute(f"SET memory_limit='{memory_limit}'")
        logger.info(f"DuckDB memory limit set to {memory_limit}")
        
        # Configure thread count
        self._conn.execute(f"SET threads={threads}")
        logger.info(f"DuckDB threads set to {threads}")
        
        # Configure temp directory for spilling
        if temp_directory is None:
            temp_directory = os.path.join(tempfile.gettempdir(), "duckdb_temp")
        
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
        self._conn.execute(f"SET temp_directory='{temp_directory}'")
        logger.info(f"DuckDB temp directory set to {temp_directory}")
        
        # Performance optimizations
        self._conn.execute("SET preserve_insertion_order=false")  # Faster inserts
        self._conn.execute("SET enable_progress_bar=false")  # Cleaner logs
        
        self._config = {
            "memory_limit": memory_limit,
            "threads": threads,
            "temp_directory": temp_directory,
        }
        
        logger.info("DuckDB connection initialized with large-data optimizations")
    
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """Get the DuckDB connection, initializing if needed."""
        if self._conn is None:
            self._init_connection()
        return self._conn
    
    def configure(
        self,
        memory_limit: Optional[str] = None,
        threads: Optional[int] = None,
        temp_directory: Optional[str] = None,
    ) -> None:
        """
        Reconfigure the DuckDB connection.
        
        Args:
            memory_limit: Maximum memory (e.g., "8GB")
            threads: Number of threads
            temp_directory: Temp directory path
        """
        if memory_limit:
            self.conn.execute(f"SET memory_limit='{memory_limit}'")
            self._config["memory_limit"] = memory_limit
            logger.info(f"DuckDB memory limit updated to {memory_limit}")
        
        if threads:
            self.conn.execute(f"SET threads={threads}")
            self._config["threads"] = threads
            logger.info(f"DuckDB threads updated to {threads}")
        
        if temp_directory:
            Path(temp_directory).mkdir(parents=True, exist_ok=True)
            self.conn.execute(f"SET temp_directory='{temp_directory}'")
            self._config["temp_directory"] = temp_directory
            logger.info(f"DuckDB temp directory updated to {temp_directory}")
    
    # =========================================================================
    # Lazy Reading Methods
    # =========================================================================
    
    def read_csv_lazy(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "duckdb.DuckDBPyRelation":
        """
        Read CSV as a lazy DuckDB relation.
        
        No data is loaded into memory until .fetchdf() or .fetchall() is called.
        This enables processing files larger than available RAM.
        
        Args:
            path: Path to CSV file (local or S3 with httpfs)
            columns: Optional list of columns to read (projection pushdown)
            **kwargs: Additional arguments passed to read_csv_auto
            
        Returns:
            DuckDB relation (lazy - no memory used yet)
        """
        if columns:
            col_list = ", ".join([f'"{c}"' for c in columns])
            return self.conn.execute(f"""
                SELECT {col_list} FROM read_csv_auto('{path}')
            """)
        return self.conn.read_csv(path, **kwargs)
    
    def read_parquet_lazy(
        self,
        path: str,
        columns: Optional[List[str]] = None,
    ) -> "duckdb.DuckDBPyRelation":
        """
        Read Parquet as a lazy DuckDB relation.
        
        Parquet format enables efficient columnar reads - only requested
        columns are loaded from disk.
        
        Args:
            path: Path to Parquet file
            columns: Optional list of columns to read
            
        Returns:
            DuckDB relation (lazy)
        """
        if columns:
            col_list = ", ".join([f'"{c}"' for c in columns])
            return self.conn.execute(f"""
                SELECT {col_list} FROM read_parquet('{path}')
            """)
        return self.conn.read_parquet(path)
    
    def read_json_lazy(self, path: str) -> "duckdb.DuckDBPyRelation":
        """
        Read JSON as a lazy DuckDB relation.
        
        Args:
            path: Path to JSON file
            
        Returns:
            DuckDB relation (lazy)
        """
        return self.conn.execute(f"SELECT * FROM read_json_auto('{path}')")
    
    def from_df(self, df: pd.DataFrame, name: Optional[str] = None) -> "duckdb.DuckDBPyRelation":
        """
        Create a lazy relation from a pandas DataFrame.
        
        Args:
            df: Pandas DataFrame
            name: Optional name to register the table
            
        Returns:
            DuckDB relation
        """
        if name:
            self.conn.register(name, df)
            return self.conn.table(name)
        return self.conn.from_df(df)
    
    # =========================================================================
    # Query Execution
    # =========================================================================
    
    def execute(self, sql: str) -> "duckdb.DuckDBPyRelation":
        """
        Execute SQL and return a lazy relation.
        
        Args:
            sql: SQL query string
            
        Returns:
            DuckDB relation (lazy - results computed on demand)
        """
        return self.conn.execute(sql)
    
    def query(
        self,
        sql: str,
        **tables: "Union[pd.DataFrame, duckdb.DuckDBPyRelation]",
    ) -> "duckdb.DuckDBPyRelation":
        """
        Execute SQL with named tables.
        
        Registers provided DataFrames/relations as tables, then executes SQL.
        
        Args:
            sql: SQL query referencing table names
            **tables: Named tables (DataFrame or DuckDB relation)
            
        Returns:
            DuckDB relation with query results
            
        Example:
            result = manager.query(
                "SELECT * FROM shifts WHERE date > '2024-01-01'",
                shifts=shifts_df
            )
        """
        for name, table in tables.items():
            if isinstance(table, pd.DataFrame):
                self.conn.register(name, table)
            elif isinstance(table, duckdb.DuckDBPyRelation):
                # Create a view from the relation
                self.conn.register(name, table)
            else:
                raise TypeError(f"Unsupported table type: {type(table)}")
        
        return self.conn.execute(sql)
    
    def register(
        self,
        name: str,
        data: "Union[pd.DataFrame, duckdb.DuckDBPyRelation, str]",
    ) -> None:
        """
        Register data as a named table.
        
        Args:
            name: Table name
            data: DataFrame, DuckDB relation, or file path
        """
        if isinstance(data, str):
            # Assume it's a file path
            if data.endswith(".parquet"):
                self.conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{data}')")
            elif data.endswith(".csv"):
                self.conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{data}')")
            elif data.endswith(".json"):
                self.conn.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_json_auto('{data}')")
            else:
                raise ValueError(f"Unsupported file format: {data}")
        else:
            self.conn.register(name, data)
        
        logger.debug(f"Registered table: {name}")
    
    # =========================================================================
    # Writing Methods
    # =========================================================================
    
    def write_parquet(
        self,
        relation: "duckdb.DuckDBPyRelation",
        path: str,
        compression: str = "zstd",
    ) -> str:
        """
        Write a relation to Parquet format (streaming).
        
        Data is streamed directly to disk without loading into pandas.
        
        Args:
            relation: DuckDB relation to write
            path: Output file path
            compression: Compression codec (zstd, snappy, gzip, none)
            
        Returns:
            Path to written file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use COPY for streaming write
        self.conn.execute(f"""
            COPY ({relation.sql_query() if hasattr(relation, 'sql_query') else 'SELECT * FROM relation'})
            TO '{path}' (FORMAT PARQUET, COMPRESSION {compression})
        """)
        
        logger.info(f"Written Parquet: {path}")
        return path
    
    def write_csv(
        self,
        relation: "duckdb.DuckDBPyRelation",
        path: str,
        header: bool = True,
    ) -> str:
        """
        Write a relation to CSV format (streaming).
        
        Args:
            relation: DuckDB relation to write
            path: Output file path
            header: Include header row
            
        Returns:
            Path to written file
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn.execute(f"""
            COPY ({relation.sql_query() if hasattr(relation, 'sql_query') else 'SELECT * FROM relation'})
            TO '{path}' (FORMAT CSV, HEADER {header})
        """)
        
        logger.info(f"Written CSV: {path}")
        return path
    
    # =========================================================================
    # Sampling Methods
    # =========================================================================
    
    def sample(
        self,
        relation: "duckdb.DuckDBPyRelation",
        n: int = 5000,
        method: str = "reservoir",
    ) -> pd.DataFrame:
        """
        Sample rows from a relation without loading the full dataset.
        
        Args:
            relation: DuckDB relation to sample from
            n: Number of rows to sample
            method: Sampling method ("reservoir" for uniform random, 
                    "system" for faster but less uniform)
            
        Returns:
            Sampled DataFrame
        """
        if method == "reservoir":
            sample_clause = f"USING SAMPLE RESERVOIR({n} ROWS)"
        elif method == "system":
            # System sampling is faster but less uniform
            sample_clause = f"USING SAMPLE {n} ROWS"
        else:
            sample_clause = f"USING SAMPLE RESERVOIR({n} ROWS)"
        
        # Register the relation and sample from it
        self.conn.register("_sample_source", relation)
        result = self.conn.execute(f"""
            SELECT * FROM _sample_source {sample_clause}
        """).fetchdf()
        
        return result
    
    def sample_file(
        self,
        path: str,
        n: int = 5000,
        method: str = "reservoir",
    ) -> pd.DataFrame:
        """
        Sample rows directly from a file without loading it entirely.
        
        Args:
            path: Path to CSV or Parquet file
            n: Number of rows to sample
            method: Sampling method
            
        Returns:
            Sampled DataFrame
        """
        if method == "reservoir":
            sample_clause = f"USING SAMPLE RESERVOIR({n} ROWS)"
        else:
            sample_clause = f"USING SAMPLE {n} ROWS"
        
        if path.endswith(".parquet"):
            sql = f"SELECT * FROM read_parquet('{path}') {sample_clause}"
        elif path.endswith(".csv"):
            sql = f"SELECT * FROM read_csv_auto('{path}') {sample_clause}"
        else:
            sql = f"SELECT * FROM read_csv_auto('{path}') {sample_clause}"
        
        return self.conn.execute(sql).fetchdf()
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_columns(self, path: str) -> List[str]:
        """
        Get column names from a file without loading data.
        
        Args:
            path: Path to CSV or Parquet file
            
        Returns:
            List of column names
        """
        if path.endswith(".parquet"):
            result = self.conn.execute(f"""
                SELECT column_name 
                FROM (DESCRIBE SELECT * FROM read_parquet('{path}'))
            """).fetchall()
        else:
            result = self.conn.execute(f"""
                SELECT column_name 
                FROM (DESCRIBE SELECT * FROM read_csv_auto('{path}'))
            """).fetchall()
        
        return [row[0] for row in result]
    
    def count_rows(self, path: str) -> int:
        """
        Count rows in a file efficiently.
        
        For Parquet, this reads metadata only (very fast).
        For CSV, this still requires scanning but doesn't load data.
        
        Args:
            path: Path to file
            
        Returns:
            Row count
        """
        if path.endswith(".parquet"):
            result = self.conn.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{path}')
            """).fetchone()
        else:
            result = self.conn.execute(f"""
                SELECT COUNT(*) FROM read_csv_auto('{path}')
            """).fetchone()
        
        return result[0] if result else 0
    
    def iter_chunks(
        self,
        path: str,
        chunk_size: int = 100_000,
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over a file in chunks.
        
        Useful for processing files that don't fit in memory.
        
        Args:
            path: Path to file
            chunk_size: Rows per chunk
            
        Yields:
            DataFrame chunks
        """
        total_rows = self.count_rows(path)
        offset = 0
        
        while offset < total_rows:
            if path.endswith(".parquet"):
                chunk = self.conn.execute(f"""
                    SELECT * FROM read_parquet('{path}')
                    LIMIT {chunk_size} OFFSET {offset}
                """).fetchdf()
            else:
                chunk = self.conn.execute(f"""
                    SELECT * FROM read_csv_auto('{path}')
                    LIMIT {chunk_size} OFFSET {offset}
                """).fetchdf()
            
            if chunk.empty:
                break
            
            yield chunk
            offset += chunk_size
    
    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("DuckDB connection closed")


# ============================================================================
# Module-level accessor
# ============================================================================

_manager: Optional[DuckDBManager] = None


def get_duckdb(
    memory_limit: str = "8GB",
    threads: int = 4,
    temp_directory: Optional[str] = None,
) -> DuckDBManager:
    """
    Get the global DuckDB manager instance.
    
    Args:
        memory_limit: Maximum memory for DuckDB
        threads: Number of threads
        temp_directory: Temp directory for spilling
        
    Returns:
        DuckDBManager singleton instance
    """
    global _manager
    if _manager is None:
        _manager = DuckDBManager()
        _manager._init_connection(
            memory_limit=memory_limit,
            threads=threads,
            temp_directory=temp_directory,
        )
    return _manager

