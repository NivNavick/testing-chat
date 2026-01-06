"""
Repository for insight results operations.
"""

import json
from typing import List, Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

import pandas as pd


class InsightRepository:
    """Repository for insight result CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_result(
        self,
        session_id: str,
        insight_name: str,
        insight_type: str,
        executed_sql: str,
        row_count: int,
        column_names: List[str],
        s3_result_key: str = None,
        execution_time_ms: float = 0,
        success: bool = True,
        error_message: str = None,
        parameters: dict = None,
        summary_stats: dict = None
    ) -> dict:
        """
        Create a new insight result record.
        
        Args:
            session_id: Session UUID
            insight_name: Name of the insight
            insight_type: Type (PREDEFINED, DYNAMIC, USER_DEFINED)
            executed_sql: SQL query that was executed
            row_count: Number of result rows
            column_names: List of column names
            s3_result_key: S3 key for the CSV export
            execution_time_ms: Execution time in milliseconds
            success: Whether execution was successful
            error_message: Error message if failed
            parameters: Query parameters used
            summary_stats: Summary statistics for columns
            
        Returns:
            Created result dict
        """
        result = await self.session.execute(
            text("""
                INSERT INTO insight_results 
                    (session_id, insight_name, insight_type, executed_sql,
                     row_count, column_names, s3_result_key, execution_time_ms,
                     success, error_message, parameters, summary_stats)
                VALUES 
                    (:session_id, :insight_name, :insight_type, :executed_sql,
                     :row_count, :column_names::jsonb, :s3_result_key, :execution_time_ms,
                     :success, :error_message, :parameters::jsonb, :summary_stats::jsonb)
                RETURNING id, session_id, insight_name, insight_type, executed_sql,
                          row_count, column_names, s3_result_key, execution_time_ms,
                          success, error_message, parameters, summary_stats, executed_at
            """),
            {
                "session_id": session_id,
                "insight_name": insight_name,
                "insight_type": insight_type,
                "executed_sql": executed_sql,
                "row_count": row_count,
                "column_names": json.dumps(column_names),
                "s3_result_key": s3_result_key,
                "execution_time_ms": execution_time_ms,
                "success": success,
                "error_message": error_message,
                "parameters": json.dumps(parameters) if parameters else None,
                "summary_stats": json.dumps(summary_stats) if summary_stats else None
            }
        )
        row = result.fetchone()
        return self._row_to_dict(row)
    
    async def save_result_data(
        self,
        result_id: str,
        df: pd.DataFrame,
        batch_size: int = 1000
    ) -> int:
        """
        Save insight result data rows to PostgreSQL.
        
        Args:
            result_id: Insight result UUID
            df: DataFrame with result data
            batch_size: Number of rows to insert per batch
            
        Returns:
            Number of rows saved
        """
        total_rows = 0
        
        for start_idx in range(0, len(df), batch_size):
            batch = df.iloc[start_idx:start_idx + batch_size]
            
            values = []
            params = {}
            
            for idx, (_, row) in enumerate(batch.iterrows()):
                row_num = start_idx + idx
                param_key = f"row_{row_num}"
                values.append(f"(:result_id, :{param_key}_num, :{param_key}_data::jsonb)")
                params[f"{param_key}_num"] = row_num
                params[f"{param_key}_data"] = row.to_json()
            
            params["result_id"] = result_id
            
            await self.session.execute(
                text(f"""
                    INSERT INTO insight_result_data (result_id, row_number, row_data)
                    VALUES {', '.join(values)}
                """),
                params
            )
            
            total_rows += len(batch)
        
        return total_rows
    
    async def get_result_by_id(self, result_id: str) -> Optional[dict]:
        """
        Get an insight result by ID.
        
        Args:
            result_id: Result UUID
            
        Returns:
            Result dict or None if not found
        """
        result = await self.session.execute(
            text("""
                SELECT id, session_id, insight_name, insight_type, executed_sql,
                       row_count, column_names, s3_result_key, execution_time_ms,
                       success, error_message, parameters, summary_stats, executed_at
                FROM insight_results
                WHERE id = :result_id
            """),
            {"result_id": result_id}
        )
        row = result.fetchone()
        return self._row_to_dict(row) if row else None
    
    async def get_result_data(
        self,
        result_id: str,
        limit: int = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get insight result data rows.
        
        Args:
            result_id: Result UUID
            limit: Maximum rows to return
            offset: Offset for pagination
            
        Returns:
            List of row data dicts
        """
        query = """
            SELECT row_number, row_data
            FROM insight_result_data
            WHERE result_id = :result_id
            ORDER BY row_number
        """
        
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        
        result = await self.session.execute(
            text(query),
            {"result_id": result_id}
        )
        
        rows = []
        for row in result.fetchall():
            row_data = json.loads(row.row_data) if isinstance(row.row_data, str) else row.row_data
            rows.append({"row_number": row.row_number, **row_data})
        return rows
    
    async def get_result_data_as_dataframe(
        self,
        result_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Get insight result data as a DataFrame.
        
        Args:
            result_id: Result UUID
            
        Returns:
            DataFrame with result data or None if not found
        """
        rows = await self.get_result_data(result_id)
        if not rows:
            return None
        
        # Remove row_number from the data
        data = [{k: v for k, v in row.items() if k != "row_number"} for row in rows]
        return pd.DataFrame(data)
    
    async def list_by_session(self, session_id: str) -> List[dict]:
        """
        List insight results for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            List of result dicts (without data)
        """
        result = await self.session.execute(
            text("""
                SELECT id, session_id, insight_name, insight_type, executed_sql,
                       row_count, column_names, s3_result_key, execution_time_ms,
                       success, error_message, parameters, summary_stats, executed_at
                FROM insight_results
                WHERE session_id = :session_id
                ORDER BY executed_at DESC
            """),
            {"session_id": session_id}
        )
        return [self._row_to_dict(row) for row in result.fetchall()]
    
    async def delete_result(self, result_id: str) -> bool:
        """
        Delete an insight result and its data.
        
        Args:
            result_id: Result UUID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            text("""
                DELETE FROM insight_results
                WHERE id = :result_id
                RETURNING id
            """),
            {"result_id": result_id}
        )
        return result.fetchone() is not None
    
    def _row_to_dict(self, row) -> dict:
        """Convert a database row to a dict."""
        if row is None:
            return None
        return {
            "id": str(row.id),
            "session_id": str(row.session_id),
            "insight_name": row.insight_name,
            "insight_type": row.insight_type,
            "executed_sql": row.executed_sql,
            "row_count": row.row_count,
            "column_names": row.column_names,
            "s3_result_key": row.s3_result_key,
            "execution_time_ms": row.execution_time_ms,
            "success": row.success,
            "error_message": row.error_message,
            "parameters": row.parameters,
            "summary_stats": row.summary_stats,
            "executed_at": row.executed_at
        }

