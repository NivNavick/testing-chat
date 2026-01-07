"""
SQLite-based persistence for the analytics pipeline.
No external dependencies required - just works.
"""

import sqlite3
import json
import uuid
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


def clean_for_json(obj):
    """Clean an object for JSON serialization (handle NaN, Infinity, etc.)"""
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

# Default database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "analytics.db"


def get_connection(db_path: Path = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: Path = None):
    """Initialize the database schema."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            vertical TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Uploaded files
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            s3_key TEXT,
            row_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    # Analyzed tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyzed_tables (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            file_id TEXT,
            table_name TEXT NOT NULL,
            document_type TEXT,
            description TEXT,
            confidence REAL,
            row_count INTEGER,
            columns_json TEXT,
            suggested_insights_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id),
            FOREIGN KEY (file_id) REFERENCES uploaded_files(id)
        )
    """)
    
    # Insight results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insight_results (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            insight_name TEXT NOT NULL,
            insight_type TEXT DEFAULT 'predefined',
            executed_sql TEXT,
            parameters_json TEXT,
            row_count INTEGER,
            columns_json TEXT,
            data_json TEXT,
            execution_time_ms REAL,
            success INTEGER DEFAULT 1,
            error_message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    # Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_session ON uploaded_files(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tables_session ON analyzed_tables(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_session ON insight_results(session_id)")
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path or DB_PATH}")


# ============================================================================
# Repository Classes
# ============================================================================

class SessionRepository:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path
    
    def _conn(self):
        return get_connection(self.db_path)
    
    def create(self, vertical: str) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        conn = self._conn()
        conn.execute(
            "INSERT INTO sessions (id, vertical) VALUES (?, ?)",
            (session_id, vertical)
        )
        conn.commit()
        conn.close()
        return self.get(session_id)
    
    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    
    def list_all(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def update_status(self, session_id: str, status: str):
        conn = self._conn()
        conn.execute(
            "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
            (status, datetime.now().isoformat(), session_id)
        )
        conn.commit()
        conn.close()
    
    def delete(self, session_id: str):
        conn = self._conn()
        conn.execute("DELETE FROM insight_results WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM analyzed_tables WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM uploaded_files WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()


class FileRepository:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path
    
    def _conn(self):
        return get_connection(self.db_path)
    
    def create(
        self,
        session_id: str,
        filename: str,
        file_type: str,
        s3_key: str = None,
        row_count: int = None
    ) -> Dict[str, Any]:
        file_id = str(uuid.uuid4())
        conn = self._conn()
        conn.execute(
            """INSERT INTO uploaded_files 
               (id, session_id, filename, file_type, s3_key, row_count) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, session_id, filename, file_type, s3_key, row_count)
        )
        conn.commit()
        conn.close()
        return self.get(file_id)
    
    def get(self, file_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM uploaded_files WHERE id = ?", (file_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    
    def list_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM uploaded_files WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]


class TableRepository:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path
    
    def _conn(self):
        return get_connection(self.db_path)
    
    def create(
        self,
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
        table_id = str(uuid.uuid4())
        conn = self._conn()
        conn.execute(
            """INSERT INTO analyzed_tables 
               (id, session_id, file_id, table_name, document_type, description, 
                confidence, row_count, columns_json, suggested_insights_json) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (table_id, session_id, file_id, table_name, document_type, description,
             confidence, row_count, json.dumps(columns), json.dumps(suggested_insights))
        )
        conn.commit()
        conn.close()
        return self.get(table_id)
    
    def get(self, table_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM analyzed_tables WHERE id = ?", (table_id,)
        ).fetchone()
        conn.close()
        if row:
            result = dict(row)
            result['columns'] = json.loads(result['columns_json']) if result['columns_json'] else []
            result['suggested_insights'] = json.loads(result['suggested_insights_json']) if result['suggested_insights_json'] else []
            return result
        return None
    
    def list_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM analyzed_tables WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        ).fetchall()
        conn.close()
        results = []
        for row in rows:
            r = dict(row)
            r['columns'] = json.loads(r['columns_json']) if r['columns_json'] else []
            r['suggested_insights'] = json.loads(r['suggested_insights_json']) if r['suggested_insights_json'] else []
            results.append(r)
        return results


class InsightRepository:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path
    
    def _conn(self):
        return get_connection(self.db_path)
    
    def create(
        self,
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
        insight_id = str(uuid.uuid4())
        conn = self._conn()
        
        # Clean data for JSON serialization (handle NaN, Infinity, numpy types)
        clean_data = clean_for_json(data)
        clean_params = clean_for_json(parameters)
        
        conn.execute(
            """INSERT INTO insight_results 
               (id, session_id, insight_name, insight_type, executed_sql, 
                parameters_json, row_count, columns_json, data_json, 
                execution_time_ms, success, error_message) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (insight_id, session_id, insight_name, insight_type, executed_sql,
             json.dumps(clean_params), row_count, json.dumps(columns), json.dumps(clean_data),
             execution_time_ms, 1 if success else 0, error_message)
        )
        conn.commit()
        conn.close()
        return self.get(insight_id)
    
    def get(self, insight_id: str) -> Optional[Dict[str, Any]]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM insight_results WHERE id = ?", (insight_id,)
        ).fetchone()
        conn.close()
        if row:
            result = dict(row)
            result['parameters'] = json.loads(result['parameters_json']) if result['parameters_json'] else {}
            result['columns'] = json.loads(result['columns_json']) if result['columns_json'] else []
            result['data'] = json.loads(result['data_json']) if result['data_json'] else []
            result['success'] = bool(result['success'])
            return result
        return None
    
    def list_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM insight_results WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,)
        ).fetchall()
        conn.close()
        results = []
        for row in rows:
            r = dict(row)
            r['parameters'] = json.loads(r['parameters_json']) if r['parameters_json'] else {}
            r['columns'] = json.loads(r['columns_json']) if r['columns_json'] else []
            r['data'] = json.loads(r['data_json']) if r['data_json'] else []
            r['success'] = bool(r['success'])
            results.append(r)
        return results
    
    def get_insight_data(self, insight_id: str) -> List[Dict[str, Any]]:
        """Get just the data rows for an insight."""
        insight = self.get(insight_id)
        if insight:
            return insight.get('data', [])
        return []


# ============================================================================
# Initialize on import
# ============================================================================

# Auto-initialize database
init_database()

