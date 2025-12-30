"""
DuckDB-based data store for the Insights Engine.

Handles:
- Loading classified CSVs into DuckDB tables
- Normalizing column names to schema field names
- Managing table metadata
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd

from csv_analyzer.insights.models import DataStoreStatus, LoadedTable

logger = logging.getLogger(__name__)


class DataStore:
    """
    DuckDB-based data store for analytical queries.
    
    Stores CSV data with normalized column names (mapped to schema fields)
    for consistent cross-table joins.
    
    Usage:
        store = DataStore()
        store.load_dataframe(
            df=df,
            document_type="employee_shifts",
            column_mappings={"emp_id": "employee_id", "dt": "shift_date"},
            source_file="shifts.csv"
        )
        
        result = store.execute("SELECT * FROM employee_shifts")
    """
    
    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize the data store.
        
        Args:
            database_path: Path to DuckDB file. If None, uses in-memory database.
        """
        self.database_path = database_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._loaded_tables: Dict[str, LoadedTable] = {}
        
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create the DuckDB connection."""
        if self._conn is None:
            if self.database_path:
                self._conn = duckdb.connect(self.database_path)
                logger.info(f"Connected to DuckDB at {self.database_path}")
            else:
                self._conn = duckdb.connect(":memory:")
                logger.info("Connected to in-memory DuckDB")
        return self._conn
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        document_type: str,
        column_mappings: Dict[str, str],
        source_file: str = "unknown",
        classification_confidence: float = 0.0,
        replace: bool = True,
    ) -> LoadedTable:
        """
        Load a DataFrame into DuckDB with normalized column names.
        
        Args:
            df: The DataFrame to load
            document_type: The classified document type (becomes table name)
            column_mappings: Mapping from original column names to schema field names
                            e.g., {"emp_id": "employee_id", "dt": "shift_date"}
            source_file: Path to the original CSV file
            classification_confidence: Confidence score from classification
            replace: If True, replace existing table. If False, append.
            
        Returns:
            LoadedTable with metadata about the loaded data
        """
        # Normalize column names using mappings
        normalized_df = self._normalize_columns(df, column_mappings)
        
        # Sanitize table name (replace hyphens, spaces)
        table_name = self._sanitize_table_name(document_type)
        
        # Register or append to table
        if replace or table_name not in self._loaded_tables:
            self.connection.register(f"_temp_{table_name}", normalized_df)
            self.connection.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS 
                SELECT * FROM _temp_{table_name}
            """)
            self.connection.unregister(f"_temp_{table_name}")
            logger.info(f"Created table '{table_name}' with {len(normalized_df)} rows")
        else:
            # Append mode
            self.connection.register(f"_temp_{table_name}", normalized_df)
            self.connection.execute(f"""
                INSERT INTO {table_name}
                SELECT * FROM _temp_{table_name}
            """)
            self.connection.unregister(f"_temp_{table_name}")
            logger.info(f"Appended {len(normalized_df)} rows to '{table_name}'")
        
        # Get row count
        row_count = self.connection.execute(
            f"SELECT COUNT(*) FROM {table_name}"
        ).fetchone()[0]
        
        # Store metadata
        loaded = LoadedTable(
            document_type=document_type,
            source_file=source_file,
            row_count=row_count,
            columns=list(normalized_df.columns),
            loaded_at=datetime.now(),
            classification_confidence=classification_confidence,
            column_mappings=column_mappings,
        )
        self._loaded_tables[document_type] = loaded
        
        return loaded
    
    def _normalize_columns(
        self, 
        df: pd.DataFrame, 
        column_mappings: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Rename columns to normalized schema field names.
        
        Columns not in the mapping are kept with their original names.
        """
        # Build rename dict (only for columns that exist)
        rename_dict = {}
        for original, target in column_mappings.items():
            if original in df.columns and target:
                rename_dict[original] = target
        
        if rename_dict:
            logger.debug(f"Normalizing columns: {rename_dict}")
            return df.rename(columns=rename_dict)
        return df
    
    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize a string for use as a DuckDB table name."""
        # Replace problematic characters
        sanitized = name.replace("-", "_").replace(" ", "_").lower()
        # Remove any remaining non-alphanumeric chars except underscore
        sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
        return sanitized
    
    def execute(self, sql: str, parameters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query to execute
            parameters: Optional parameters for query (used with $name syntax)
            
        Returns:
            DataFrame with query results
        """
        if parameters:
            # DuckDB uses $name for parameters
            result = self.connection.execute(sql, parameters)
        else:
            result = self.connection.execute(sql)
        
        return result.df()
    
    def execute_raw(self, sql: str) -> Any:
        """Execute SQL without converting to DataFrame."""
        return self.connection.execute(sql)
    
    def get_status(self) -> DataStoreStatus:
        """Get current status of the data store."""
        return DataStoreStatus(
            tables=list(self._loaded_tables.values()),
            total_rows=sum(t.row_count for t in self._loaded_tables.values()),
            database_path=self.database_path,
        )
    
    def has_table(self, document_type: str) -> bool:
        """Check if a table is loaded."""
        return document_type in self._loaded_tables
    
    def get_table_info(self, document_type: str) -> Optional[LoadedTable]:
        """Get info about a loaded table."""
        return self._loaded_tables.get(document_type)
    
    def list_tables(self) -> List[str]:
        """List all loaded table names."""
        return list(self._loaded_tables.keys())
    
    def describe_table(self, document_type: str) -> pd.DataFrame:
        """Get schema information for a table."""
        table_name = self._sanitize_table_name(document_type)
        return self.connection.execute(f"DESCRIBE {table_name}").df()
    
    def sample_table(self, document_type: str, limit: int = 5) -> pd.DataFrame:
        """Get sample rows from a table."""
        table_name = self._sanitize_table_name(document_type)
        return self.connection.execute(
            f"SELECT * FROM {table_name} LIMIT {limit}"
        ).df()
    
    def drop_table(self, document_type: str) -> None:
        """Drop a table from the data store."""
        if document_type in self._loaded_tables:
            table_name = self._sanitize_table_name(document_type)
            self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            del self._loaded_tables[document_type]
            logger.info(f"Dropped table '{table_name}'")
    
    def clear(self) -> None:
        """Clear all tables from the data store."""
        for doc_type in list(self._loaded_tables.keys()):
            self.drop_table(doc_type)
        logger.info("Cleared all tables from data store")
    
    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("Closed DuckDB connection")
    
    def __enter__(self) -> "DataStore":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

