"""
PostgreSQL-based persistence for the analytics pipeline.
Uses asyncpg for async operations.
"""

import asyncpg
import json
import uuid
import math
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Connection settings - uses Docker PostgreSQL by default
DB_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": int(os.environ.get("POSTGRES_PORT", 5432)),
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    "database": os.environ.get("POSTGRES_DB", "analytics")
}

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


def clean_for_json(obj):
    """Clean an object for JSON serialization (handle NaN, Infinity, numpy types)"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'item'):  # numpy types
        return clean_for_json(obj.item())
    else:
        return obj


async def get_pool() -> asyncpg.Pool:
    """Get or create connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"] or None,
            database=DB_CONFIG["database"],
            min_size=2,
            max_size=10
        )
        logger.info(f"PostgreSQL pool created for {DB_CONFIG['database']}")
    return _pool


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ============================================================================
# Repository Classes
# ============================================================================

class SessionRepository:
    @staticmethod
    async def create(vertical: str) -> Dict[str, Any]:
        pool = await get_pool()
        session_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO sessions (id, vertical) VALUES ($1, $2)",
                session_id, vertical
            )
        return await SessionRepository.get(session_id)
    
    @staticmethod
    async def get(session_id: str) -> Optional[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sessions WHERE id = $1", session_id
            )
        if row:
            return dict(row)
        return None
    
    @staticmethod
    async def list_all() -> List[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            )
        return [dict(r) for r in rows]
    
    @staticmethod
    async def update_status(session_id: str, status: str):
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE sessions SET status = $1, updated_at = NOW() WHERE id = $2",
                status, session_id
            )
    
    @staticmethod
    async def delete(session_id: str):
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Cascading delete handled by FK constraints
            await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)


class FileRepository:
    @staticmethod
    async def create(
        session_id: str,
        filename: str,
        file_type: str,
        s3_key: str = None,
        row_count: int = None
    ) -> Dict[str, Any]:
        pool = await get_pool()
        file_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO uploaded_files 
                   (id, session_id, filename, file_type, s3_key, row_count) 
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                file_id, session_id, filename, file_type, s3_key, row_count
            )
        return await FileRepository.get(file_id)
    
    @staticmethod
    async def get(file_id: str) -> Optional[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM uploaded_files WHERE id = $1", file_id
            )
        return dict(row) if row else None
    
    @staticmethod
    async def list_by_session(session_id: str) -> List[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM uploaded_files WHERE session_id = $1 ORDER BY created_at",
                session_id
            )
        return [dict(r) for r in rows]


class TableRepository:
    @staticmethod
    async def create(
        session_id: str,
        table_name: str,
        document_type: str,
        description: str,
        confidence: float,
        row_count: int,
        columns: List[Dict],
        suggested_insights: List[str],
        file_id: str = None
    ) -> Dict[str, Any]:
        pool = await get_pool()
        table_id = str(uuid.uuid4())
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO analyzed_tables 
                   (id, session_id, file_id, table_name, document_type, description, 
                    confidence, row_count, columns_json, suggested_insights_json) 
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                table_id, session_id, file_id, table_name, document_type, description,
                confidence, row_count, json.dumps(columns), json.dumps(suggested_insights)
            )
        return await TableRepository.get(table_id)
    
    @staticmethod
    async def get(table_id: str) -> Optional[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM analyzed_tables WHERE id = $1", table_id
            )
        if row:
            result = dict(row)
            cols = result.get('columns_json') or '[]'
            insights = result.get('suggested_insights_json') or '[]'
            result['columns'] = json.loads(cols) if isinstance(cols, str) else cols
            result['suggested_insights'] = json.loads(insights) if isinstance(insights, str) else insights
            return result
        return None
    
    @staticmethod
    async def list_by_session(session_id: str) -> List[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM analyzed_tables WHERE session_id = $1 ORDER BY created_at",
                session_id
            )
        results = []
        for row in rows:
            r = dict(row)
            cols = r.get('columns_json') or '[]'
            insights = r.get('suggested_insights_json') or '[]'
            r['columns'] = json.loads(cols) if isinstance(cols, str) else cols
            r['suggested_insights'] = json.loads(insights) if isinstance(insights, str) else insights
            results.append(r)
        return results


class InsightRepository:
    @staticmethod
    async def create(
        session_id: str,
        insight_name: str,
        insight_type: str,
        executed_sql: str,
        parameters: Dict[str, Any],
        row_count: int,
        columns: List[str],
        data: List[Dict[str, Any]],
        execution_time_ms: float,
        success: bool = True,
        error_message: str = None
    ) -> Dict[str, Any]:
        pool = await get_pool()
        insight_id = str(uuid.uuid4())
        
        # Clean data for JSON serialization
        clean_data = clean_for_json(data)
        clean_params = clean_for_json(parameters)
        
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO insight_results 
                   (id, session_id, insight_name, insight_type, executed_sql, 
                    parameters_json, row_count, columns_json, data_json, 
                    execution_time_ms, success, error_message) 
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
                insight_id, session_id, insight_name, insight_type, executed_sql,
                json.dumps(clean_params), row_count, json.dumps(columns), json.dumps(clean_data),
                execution_time_ms, success, error_message
            )
        return await InsightRepository.get(insight_id)
    
    @staticmethod
    async def get(insight_id: str) -> Optional[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM insight_results WHERE id = $1", insight_id
            )
        if row:
            result = dict(row)
            # Parse JSON strings if needed
            params = result.get('parameters_json') or '{}'
            cols = result.get('columns_json') or '[]'
            data = result.get('data_json') or '[]'
            
            result['parameters'] = json.loads(params) if isinstance(params, str) else params
            result['columns'] = json.loads(cols) if isinstance(cols, str) else cols
            result['data'] = json.loads(data) if isinstance(data, str) else data
            return result
        return None
    
    @staticmethod
    async def list_by_session(session_id: str) -> List[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM insight_results WHERE session_id = $1 ORDER BY created_at DESC",
                session_id
            )
        results = []
        for row in rows:
            r = dict(row)
            # Parse JSON strings if needed
            params = r.get('parameters_json') or '{}'
            cols = r.get('columns_json') or '[]'
            data = r.get('data_json') or '[]'
            
            r['parameters'] = json.loads(params) if isinstance(params, str) else params
            r['columns'] = json.loads(cols) if isinstance(cols, str) else cols
            r['data'] = json.loads(data) if isinstance(data, str) else data
            results.append(r)
        return results
    
    @staticmethod
    async def get_insight_data(insight_id: str) -> List[Dict[str, Any]]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM insight_results WHERE id = $1", insight_id
            )
        if row and row['data_json']:
            data = row['data_json']
            return json.loads(data) if isinstance(data, str) else data
        return []

