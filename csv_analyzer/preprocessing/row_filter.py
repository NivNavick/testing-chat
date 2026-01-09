"""
SQL-based Row Filter for CSV preprocessing.

Filters rows from DataFrames using SQL conditions via DuckDB.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for a row filter."""
    name: str
    condition: str  # SQL WHERE condition
    description: Optional[str] = None
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterConfig":
        """Create from dictionary (from YAML)."""
        return cls(
            name=data.get("name", "unnamed_filter"),
            condition=data["condition"],
            description=data.get("description"),
            enabled=data.get("enabled", True),
            options=data.get("options", {}),
        )


@dataclass
class FilterResult:
    """Result of applying filters to a DataFrame."""
    df: pd.DataFrame
    filters_applied: int
    rows_before: int
    rows_after: int
    rows_removed: int
    filter_details: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RowFilter:
    """
    Filters DataFrame rows using SQL conditions.
    
    Uses DuckDB as an in-memory SQL engine for powerful filtering.
    
    Usage:
        filter = RowFilter()
        
        configs = [
            FilterConfig(
                name="exclude_settlements",
                condition="שם_טיפול NOT LIKE '%התחשבנות%'",
                description="Remove settlement/accounting rows",
            ),
            FilterConfig(
                name="only_gastro",
                condition="קטגוריה = 'גסטרו'",
            ),
        ]
        
        result = filter.apply_all(df, configs)
        print(f"Removed {result.rows_removed} rows")
    """
    
    def __init__(self):
        """Initialize the row filter."""
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
    
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create DuckDB connection."""
        if self._conn is None:
            self._conn = duckdb.connect(":memory:")
        return self._conn
    
    def apply_all(
        self,
        df: pd.DataFrame,
        configs: List[Union[FilterConfig, Dict[str, Any]]],
        table_name: str = "data",
    ) -> FilterResult:
        """
        Apply all filters to the DataFrame.
        
        Filters are applied in order, and each filter operates on the
        result of the previous filter.
        
        Args:
            df: Input DataFrame
            configs: List of filter configurations
            table_name: Name for the temporary table in SQL
            
        Returns:
            FilterResult with the filtered DataFrame
        """
        result_df = df.copy()
        rows_before = len(result_df)
        filters_applied = 0
        filter_details = []
        errors = []
        
        for config in configs:
            if isinstance(config, dict):
                config = FilterConfig.from_dict(config)
            
            if not config.enabled:
                continue
            
            try:
                result_df, rows_removed = self.apply_filter(
                    result_df, config, table_name
                )
                
                filters_applied += 1
                filter_details.append({
                    "name": config.name,
                    "condition": config.condition,
                    "rows_removed": rows_removed,
                    "rows_remaining": len(result_df),
                })
                
                logger.info(
                    f"Applied filter '{config.name}': "
                    f"removed {rows_removed} rows, {len(result_df)} remaining"
                )
                
            except Exception as e:
                error_msg = f"Error applying filter '{config.name}': {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
        
        rows_after = len(result_df)
        
        return FilterResult(
            df=result_df,
            filters_applied=filters_applied,
            rows_before=rows_before,
            rows_after=rows_after,
            rows_removed=rows_before - rows_after,
            filter_details=filter_details,
            errors=errors,
        )
    
    def apply_filter(
        self,
        df: pd.DataFrame,
        config: FilterConfig,
        table_name: str = "data",
    ) -> tuple:
        """
        Apply a single filter to the DataFrame.
        
        Args:
            df: Input DataFrame
            config: Filter configuration
            table_name: Name for the temporary table
            
        Returns:
            Tuple of (filtered_df, rows_removed)
        """
        rows_before = len(df)
        
        # Sanitize table name
        safe_table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
        # Register DataFrame with DuckDB
        self.connection.register(safe_table_name, df)
        
        try:
            # Build and execute query
            # The condition should be a WHERE clause without the WHERE keyword
            condition = config.condition.strip()
            
            # Remove leading WHERE if present
            if condition.upper().startswith("WHERE "):
                condition = condition[6:]
            
            query = f"SELECT * FROM {safe_table_name} WHERE {condition}"
            
            result_df = self.connection.execute(query).df()
            rows_removed = rows_before - len(result_df)
            
            return result_df, rows_removed
            
        finally:
            # Unregister the table
            self.connection.unregister(safe_table_name)
    
    def apply_sql(
        self,
        df: pd.DataFrame,
        sql: str,
        table_name: str = "data",
    ) -> pd.DataFrame:
        """
        Apply arbitrary SQL transformation to the DataFrame.
        
        The DataFrame is available as {table_name} in the SQL query.
        
        Args:
            df: Input DataFrame
            sql: SQL query to execute
            table_name: Name for the temporary table
            
        Returns:
            Transformed DataFrame
        """
        # Sanitize table name
        safe_table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
        # Replace table name placeholder in SQL
        actual_sql = sql.replace(f"{{{table_name}}}", safe_table_name)
        actual_sql = actual_sql.replace(table_name, safe_table_name)
        
        # Register DataFrame with DuckDB
        self.connection.register(safe_table_name, df)
        
        try:
            result_df = self.connection.execute(actual_sql).df()
            return result_df
        finally:
            self.connection.unregister(safe_table_name)
    
    def preview_filter(
        self,
        df: pd.DataFrame,
        config: FilterConfig,
        limit: int = 10,
    ) -> pd.DataFrame:
        """
        Preview rows that would be REMOVED by a filter.
        
        Useful for debugging filter conditions.
        
        Args:
            df: Input DataFrame
            config: Filter configuration
            limit: Maximum rows to return
            
        Returns:
            DataFrame of rows that would be removed
        """
        # Invert the condition to get rows that would be removed
        condition = config.condition.strip()
        if condition.upper().startswith("WHERE "):
            condition = condition[6:]
        
        inverted = f"NOT ({condition})"
        
        self.connection.register("data", df)
        
        try:
            query = f"SELECT * FROM data WHERE {inverted} LIMIT {limit}"
            return self.connection.execute(query).df()
        finally:
            self.connection.unregister("data")
    
    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# Convenience functions for common filters
def create_exclude_filter(
    column: str,
    patterns: List[str],
    name: Optional[str] = None,
) -> FilterConfig:
    """
    Create a filter that excludes rows where column matches any pattern.
    
    Args:
        column: Column name to filter on
        patterns: List of LIKE patterns to exclude
        name: Optional filter name
        
    Returns:
        FilterConfig
    """
    conditions = [f'"{column}" NOT LIKE \'{p}\'' for p in patterns]
    condition = " AND ".join(conditions)
    
    return FilterConfig(
        name=name or f"exclude_{column}_patterns",
        condition=condition,
        description=f"Exclude rows where {column} matches: {patterns}",
    )


def create_include_filter(
    column: str,
    values: List[str],
    name: Optional[str] = None,
) -> FilterConfig:
    """
    Create a filter that includes only rows where column is in values.
    
    Args:
        column: Column name to filter on
        values: List of values to include
        name: Optional filter name
        
    Returns:
        FilterConfig
    """
    values_str = ", ".join(f"'{v}'" for v in values)
    condition = f'"{column}" IN ({values_str})'
    
    return FilterConfig(
        name=name or f"include_{column}_values",
        condition=condition,
        description=f"Include only rows where {column} is one of: {values}",
    )


def create_not_null_filter(
    columns: List[str],
    name: Optional[str] = None,
) -> FilterConfig:
    """
    Create a filter that excludes rows with NULL in specified columns.
    
    Args:
        columns: Column names that must not be NULL
        name: Optional filter name
        
    Returns:
        FilterConfig
    """
    conditions = [f'"{col}" IS NOT NULL' for col in columns]
    condition = " AND ".join(conditions)
    
    return FilterConfig(
        name=name or "not_null_filter",
        condition=condition,
        description=f"Exclude rows where any of {columns} is NULL",
    )

