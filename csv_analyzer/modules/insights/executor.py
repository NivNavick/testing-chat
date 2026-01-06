"""
Insight Executor - Executes insights using DuckDB and saves results.
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import pandas as pd

from csv_analyzer.storage.duckdb.session_db import SessionDatabase, get_session_database
from csv_analyzer.storage.s3.operations import S3Operations, get_s3_operations
from csv_analyzer.storage.postgres.repositories.insight_repo import InsightRepository


@dataclass
class InsightDefinition:
    """Definition of an insight."""
    name: str
    description: str
    sql_template: str
    requires_tables: List[str]
    parameters: Dict[str, Any] = None


@dataclass
class InsightExecutionResult:
    """Result of insight execution."""
    insight_name: str
    success: bool
    df: Optional[pd.DataFrame] = None
    row_count: int = 0
    execution_time_ms: float = 0
    s3_result_key: Optional[str] = None
    error_message: Optional[str] = None
    summary_stats: Optional[Dict[str, Any]] = None


class InsightExecutor:
    """
    Executes insights using DuckDB and saves results to PostgreSQL and S3.
    """
    
    def __init__(
        self,
        session_id: str,
        db: SessionDatabase = None,
        s3_ops: S3Operations = None
    ):
        self.session_id = session_id
        self.db = db or get_session_database(session_id)
        self.s3_ops = s3_ops or get_s3_operations()
    
    async def execute_insight(
        self,
        insight: InsightDefinition,
        parameters: Dict[str, Any] = None,
        save_to_postgres: bool = True,
        save_to_s3: bool = True,
        insight_repo: InsightRepository = None
    ) -> InsightExecutionResult:
        """
        Execute an insight and save results.
        
        Args:
            insight: Insight definition
            parameters: Query parameters
            save_to_postgres: Whether to save results to PostgreSQL
            save_to_s3: Whether to save CSV to S3
            insight_repo: InsightRepository for saving (required if save_to_postgres)
            
        Returns:
            InsightExecutionResult
        """
        # Verify required tables are loaded
        missing_tables = []
        for table_type in insight.requires_tables:
            # Check if any table with this type is loaded
            found = False
            for table in self.db.loaded_tables.values():
                if table.document_type == table_type:
                    found = True
                    break
            if not found:
                missing_tables.append(table_type)
        
        if missing_tables:
            return InsightExecutionResult(
                insight_name=insight.name,
                success=False,
                error_message=f"Missing required tables: {', '.join(missing_tables)}"
            )
        
        # Prepare SQL with parameters
        sql = insight.sql_template
        merged_params = {**(insight.parameters or {}), **(parameters or {})}
        
        try:
            # Execute query
            start_time = time.perf_counter()
            df = self.db.execute_query(sql, merged_params)
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Compute summary stats
            summary_stats = self._compute_summary_stats(df)
            
            # Save to S3 if requested
            s3_key = None
            if save_to_s3:
                s3_key = await self.s3_ops.upload_result_csv(
                    session_id=self.session_id,
                    insight_name=insight.name,
                    df=df
                )
            
            # Save to PostgreSQL if requested
            if save_to_postgres and insight_repo:
                result_record = await insight_repo.create_result(
                    session_id=self.session_id,
                    insight_name=insight.name,
                    insight_type="PREDEFINED",
                    executed_sql=sql,
                    row_count=len(df),
                    column_names=list(df.columns),
                    s3_result_key=s3_key,
                    execution_time_ms=execution_time_ms,
                    success=True,
                    parameters=merged_params,
                    summary_stats=summary_stats
                )
                
                # Save full result data
                await insight_repo.save_result_data(
                    result_id=result_record["id"],
                    df=df
                )
            
            return InsightExecutionResult(
                insight_name=insight.name,
                success=True,
                df=df,
                row_count=len(df),
                execution_time_ms=execution_time_ms,
                s3_result_key=s3_key,
                summary_stats=summary_stats
            )
            
        except Exception as e:
            # Save failed result to PostgreSQL if requested
            if save_to_postgres and insight_repo:
                await insight_repo.create_result(
                    session_id=self.session_id,
                    insight_name=insight.name,
                    insight_type="PREDEFINED",
                    executed_sql=sql,
                    row_count=0,
                    column_names=[],
                    execution_time_ms=0,
                    success=False,
                    error_message=str(e),
                    parameters=merged_params
                )
            
            return InsightExecutionResult(
                insight_name=insight.name,
                success=False,
                error_message=str(e)
            )
    
    async def execute_custom_sql(
        self,
        sql: str,
        name: str = "custom_query",
        save_to_postgres: bool = True,
        save_to_s3: bool = True,
        insight_repo: InsightRepository = None
    ) -> InsightExecutionResult:
        """
        Execute a custom SQL query.
        
        Args:
            sql: SQL query string
            name: Name for the result
            save_to_postgres: Whether to save results to PostgreSQL
            save_to_s3: Whether to save CSV to S3
            insight_repo: InsightRepository for saving
            
        Returns:
            InsightExecutionResult
        """
        insight = InsightDefinition(
            name=name,
            description="Custom query",
            sql_template=sql,
            requires_tables=[]  # No validation for custom queries
        )
        
        return await self.execute_insight(
            insight=insight,
            save_to_postgres=save_to_postgres,
            save_to_s3=save_to_s3,
            insight_repo=insight_repo
        )
    
    def _compute_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for result DataFrame."""
        stats = {}
        
        for col in df.columns:
            col_stats = {"type": str(df[col].dtype)}
            
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_stats["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else None
                    col_stats["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else None
                    col_stats["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                    col_stats["sum"] = float(df[col].sum()) if not pd.isna(df[col].sum()) else None
                else:
                    col_stats["unique_count"] = int(df[col].nunique())
                    
                col_stats["null_count"] = int(df[col].isna().sum())
                
            except Exception:
                pass
            
            stats[col] = col_stats
        
        return stats
    
    def get_available_insights(
        self,
        insight_definitions: List[InsightDefinition]
    ) -> List[Dict[str, Any]]:
        """
        Get list of available insights based on loaded tables.
        
        Args:
            insight_definitions: List of all insight definitions
            
        Returns:
            List of insight availability info
        """
        loaded_types = set(
            t.document_type for t in self.db.loaded_tables.values()
            if t.document_type
        )
        
        result = []
        for insight in insight_definitions:
            required = set(insight.requires_tables)
            missing = required - loaded_types
            
            result.append({
                "name": insight.name,
                "description": insight.description,
                "requires_tables": insight.requires_tables,
                "is_available": len(missing) == 0,
                "missing_tables": list(missing)
            })
        
        return result

