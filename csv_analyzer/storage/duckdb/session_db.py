"""
DuckDB Session Database - In-memory analytics engine per session.
"""

import io
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import duckdb
import pandas as pd

from csv_analyzer.core.config import get_settings


@dataclass
class LoadedTable:
    """Represents a table loaded into DuckDB."""
    name: str
    original_name: str
    row_count: int
    column_count: int
    document_type: Optional[str] = None


class SessionDatabase:
    """
    In-memory DuckDB database for a session.
    
    Tables are loaded from S3 and kept in memory for fast queries.
    The database is ephemeral and discarded when the session ends.
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.settings = get_settings()
        
        # Create in-memory DuckDB connection
        self._conn = duckdb.connect(":memory:")
        
        # Configure DuckDB
        self._conn.execute(f"SET memory_limit='{self.settings.duckdb_memory_limit}'")
        self._conn.execute(f"SET threads={self.settings.duckdb_threads}")
        
        # Track loaded tables
        self._tables: Dict[str, LoadedTable] = {}
    
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get the DuckDB connection."""
        return self._conn
    
    @property
    def loaded_tables(self) -> Dict[str, LoadedTable]:
        """Get loaded tables info."""
        return self._tables.copy()
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        document_type: str = None
    ) -> LoadedTable:
        """
        Load a DataFrame into DuckDB.
        
        Args:
            df: DataFrame to load
            table_name: Name to use in DuckDB
            document_type: Classified document type
            
        Returns:
            LoadedTable info
        """
        # Sanitize table name
        safe_name = self._sanitize_table_name(table_name)
        
        # Create table from DataFrame
        self._conn.register(f"_temp_{safe_name}", df)
        self._conn.execute(f"CREATE OR REPLACE TABLE {safe_name} AS SELECT * FROM _temp_{safe_name}")
        self._conn.unregister(f"_temp_{safe_name}")
        
        # Track the table
        table_info = LoadedTable(
            name=safe_name,
            original_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            document_type=document_type
        )
        self._tables[safe_name] = table_info
        
        return table_info
    
    def execute_query(
        self,
        sql: str,
        parameters: Dict[str, Any] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query string
            parameters: Optional query parameters
            
        Returns:
            Query results as DataFrame
        """
        if parameters:
            # Use prepared statement with parameters
            result = self._conn.execute(sql, parameters)
        else:
            result = self._conn.execute(sql)
        
        return result.fetchdf()
    
    def execute_query_with_stats(
        self,
        sql: str,
        parameters: Dict[str, Any] = None
    ) -> tuple:
        """
        Execute a query and return results with statistics.
        
        Args:
            sql: SQL query string
            parameters: Optional query parameters
            
        Returns:
            Tuple of (DataFrame, execution_time_ms, row_count)
        """
        import time
        
        start_time = time.perf_counter()
        df = self.execute_query(sql, parameters)
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        return df, execution_time_ms, len(df)
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        Get schema information for a table.
        
        Args:
            table_name: Table name
            
        Returns:
            List of column info dicts
        """
        safe_name = self._sanitize_table_name(table_name)
        
        result = self._conn.execute(f"DESCRIBE {safe_name}")
        rows = result.fetchall()
        
        return [
            {"name": row[0], "type": row[1], "nullable": row[2]}
            for row in rows
        ]
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        safe_name = self._sanitize_table_name(table_name)
        return safe_name in self._tables
    
    def drop_table(self, table_name: str) -> bool:
        """
        Drop a table from DuckDB.
        
        Args:
            table_name: Table name
            
        Returns:
            True if dropped, False if not found
        """
        safe_name = self._sanitize_table_name(table_name)
        
        if safe_name not in self._tables:
            return False
        
        self._conn.execute(f"DROP TABLE IF EXISTS {safe_name}")
        del self._tables[safe_name]
        return True
    
    def list_tables(self) -> List[str]:
        """List all loaded table names."""
        return list(self._tables.keys())
    
    def get_table_sample(
        self,
        table_name: str,
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Get a sample of rows from a table.
        
        Args:
            table_name: Table name
            limit: Number of rows
            
        Returns:
            Sample DataFrame
        """
        safe_name = self._sanitize_table_name(table_name)
        return self.execute_query(f"SELECT * FROM {safe_name} LIMIT {limit}")
    
    def compute_column_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Compute statistics for all columns in a table.
        
        Args:
            table_name: Table name
            
        Returns:
            Dict of column statistics
        """
        safe_name = self._sanitize_table_name(table_name)
        schema = self.get_table_schema(safe_name)
        
        stats = {}
        for col_info in schema:
            col_name = col_info["name"]
            col_type = col_info["type"]
            
            col_stats = {"type": col_type}
            
            try:
                # Count non-null values
                result = self._conn.execute(
                    f"SELECT COUNT(*) as total, COUNT({col_name}) as non_null FROM {safe_name}"
                ).fetchone()
                col_stats["total"] = result[0]
                col_stats["non_null"] = result[1]
                col_stats["null_count"] = result[0] - result[1]
                
                # For numeric columns, compute min/max/avg
                if "INT" in col_type.upper() or "FLOAT" in col_type.upper() or "DOUBLE" in col_type.upper():
                    result = self._conn.execute(
                        f"SELECT MIN({col_name}), MAX({col_name}), AVG({col_name}) FROM {safe_name}"
                    ).fetchone()
                    col_stats["min"] = result[0]
                    col_stats["max"] = result[1]
                    col_stats["avg"] = result[2]
                else:
                    # For other columns, count distinct
                    result = self._conn.execute(
                        f"SELECT COUNT(DISTINCT {col_name}) FROM {safe_name}"
                    ).fetchone()
                    col_stats["distinct_count"] = result[0]
                    
            except Exception:
                pass  # Skip columns that cause errors
            
            stats[col_name] = col_stats
        
        return stats
    
    def close(self):
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._tables.clear()
    
    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize a table name for DuckDB."""
        # Replace spaces and special chars with underscores
        import re
        safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure doesn't start with number
        if safe and safe[0].isdigit():
            safe = f"t_{safe}"
        
        return safe.lower()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Session database cache
_session_databases: Dict[str, SessionDatabase] = {}


def get_session_database(session_id: str) -> SessionDatabase:
    """
    Get or create a session database.
    
    Args:
        session_id: Session UUID
        
    Returns:
        SessionDatabase for the session
    """
    if session_id not in _session_databases:
        _session_databases[session_id] = SessionDatabase(session_id)
    return _session_databases[session_id]


def close_session_database(session_id: str):
    """
    Close and remove a session database.
    
    Args:
        session_id: Session UUID
    """
    if session_id in _session_databases:
        _session_databases[session_id].close()
        del _session_databases[session_id]

