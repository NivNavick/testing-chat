"""
PostgreSQL-backed session storage with S3 integration.

Stores session metadata in PostgreSQL and files in S3.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg

from csv_analyzer.api.config import get_settings

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Session not found."""
    pass


class DocumentNotFoundError(Exception):
    """Document not found."""
    pass


class AsyncSessionStore:
    """
    Async PostgreSQL-backed session storage.
    
    Stores:
    - Session metadata in PostgreSQL
    - Document metadata in PostgreSQL
    - Actual files in S3
    """
    
    def __init__(self, db_pool: asyncpg.Pool, s3_client: "S3Client"):
        """
        Initialize the session store.
        
        Args:
            db_pool: asyncpg connection pool
            s3_client: S3 client for file storage
        """
        self.db_pool = db_pool
        self.s3_client = s3_client
        self.settings = get_settings()
    
    # ========================================================================
    # Session Operations
    # ========================================================================
    
    async def create_session(
        self,
        name: str,
        vertical: str = "medical",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new session.
        
        Args:
            name: Session name
            vertical: Vertical domain
            metadata: Additional metadata
            
        Returns:
            Session dict with id, name, vertical, etc.
        """
        session_id = uuid.uuid4()
        s3_prefix = f"{self.settings.s3.prefix.sessions}{session_id}/"
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO api_sessions (id, name, vertical, metadata, s3_bucket, s3_prefix)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, name, vertical, created_at, updated_at, metadata, s3_bucket, s3_prefix
                """,
                session_id,
                name,
                vertical,
                metadata or {},
                self.settings.s3.bucket,
                s3_prefix,
            )
            
            logger.info(f"Created session: {session_id} ({name})")
            return dict(row)
    
    async def get_session(self, session_id: UUID) -> Dict[str, Any]:
        """
        Get session by ID.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session dict
            
        Raises:
            SessionNotFoundError: If session not found
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 
                    s.id, s.name, s.vertical, s.created_at, s.updated_at, 
                    s.metadata, s.s3_bucket, s3_prefix,
                    COUNT(d.id) as document_count
                FROM api_sessions s
                LEFT JOIN session_documents d ON s.id = d.session_id
                WHERE s.id = $1
                GROUP BY s.id
                """,
                session_id,
            )
            
            if row is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")
            
            return dict(row)
    
    async def list_sessions(
        self,
        vertical: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List sessions.
        
        Args:
            vertical: Optional vertical filter
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of session dicts
        """
        async with self.db_pool.acquire() as conn:
            if vertical:
                rows = await conn.fetch(
                    """
                    SELECT 
                        s.id, s.name, s.vertical, s.created_at, s.updated_at,
                        s.metadata, s.s3_bucket, s3_prefix,
                        COUNT(d.id) as document_count
                    FROM api_sessions s
                    LEFT JOIN session_documents d ON s.id = d.session_id
                    WHERE s.vertical = $1
                    GROUP BY s.id
                    ORDER BY s.created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    vertical,
                    limit,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT 
                        s.id, s.name, s.vertical, s.created_at, s.updated_at,
                        s.metadata, s.s3_bucket, s3_prefix,
                        COUNT(d.id) as document_count
                    FROM api_sessions s
                    LEFT JOIN session_documents d ON s.id = d.session_id
                    GROUP BY s.id
                    ORDER BY s.created_at DESC
                    LIMIT $1 OFFSET $2
                    """,
                    limit,
                    offset,
                )
            
            return [dict(row) for row in rows]
    
    async def update_session(
        self,
        session_id: UUID,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update session.
        
        Args:
            session_id: Session UUID
            name: New name (optional)
            metadata: New metadata (optional)
            
        Returns:
            Updated session dict
        """
        async with self.db_pool.acquire() as conn:
            # Build update query dynamically
            updates = []
            params = [session_id]
            param_idx = 2
            
            if name is not None:
                updates.append(f"name = ${param_idx}")
                params.append(name)
                param_idx += 1
            
            if metadata is not None:
                updates.append(f"metadata = ${param_idx}")
                params.append(metadata)
                param_idx += 1
            
            if not updates:
                return await self.get_session(session_id)
            
            query = f"""
                UPDATE api_sessions
                SET {', '.join(updates)}
                WHERE id = $1
                RETURNING id, name, vertical, created_at, updated_at, metadata, s3_bucket, s3_prefix
            """
            
            row = await conn.fetchrow(query, *params)
            
            if row is None:
                raise SessionNotFoundError(f"Session not found: {session_id}")
            
            logger.info(f"Updated session: {session_id}")
            return dict(row)
    
    async def delete_session(self, session_id: UUID) -> bool:
        """
        Delete session and all its files from S3.
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if deleted
        """
        # Get session to find S3 prefix
        session = await self.get_session(session_id)
        
        async with self.db_pool.acquire() as conn:
            # Delete from database (cascades to documents)
            result = await conn.execute(
                "DELETE FROM api_sessions WHERE id = $1",
                session_id,
            )
            
            if result == "DELETE 0":
                raise SessionNotFoundError(f"Session not found: {session_id}")
        
        # Delete files from S3
        if session.get("s3_prefix"):
            try:
                deleted_count = await self.s3_client.delete_prefix(session["s3_prefix"])
                logger.info(f"Deleted {deleted_count} files from S3 for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to delete S3 files for session {session_id}: {e}")
        
        logger.info(f"Deleted session: {session_id}")
        return True
    
    async def count_sessions(self, vertical: Optional[str] = None) -> int:
        """Count total sessions."""
        async with self.db_pool.acquire() as conn:
            if vertical:
                result = await conn.fetchval(
                    "SELECT COUNT(*) FROM api_sessions WHERE vertical = $1",
                    vertical,
                )
            else:
                result = await conn.fetchval("SELECT COUNT(*) FROM api_sessions")
            return result or 0
    
    # ========================================================================
    # Document Operations
    # ========================================================================
    
    async def add_document(
        self,
        session_id: UUID,
        document_id: str,
        filename: str,
        s3_uri: str,
        document_type: Optional[str] = None,
        row_count: int = 0,
        column_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        classification_result: Optional[Dict[str, Any]] = None,
        classification_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Add a document to a session.
        
        Args:
            session_id: Session UUID
            document_id: Document identifier
            filename: Original filename
            s3_uri: S3 URI where file is stored
            document_type: Detected document type
            row_count: Number of rows
            column_count: Number of columns
            metadata: Additional metadata
            classification_result: Classification results
            classification_confidence: Classification confidence
            
        Returns:
            Document dict
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO session_documents (
                    session_id, document_id, filename, s3_uri, document_type,
                    row_count, column_count, metadata, 
                    classification_result, classification_confidence
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (session_id, document_id) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    s3_uri = EXCLUDED.s3_uri,
                    document_type = EXCLUDED.document_type,
                    row_count = EXCLUDED.row_count,
                    column_count = EXCLUDED.column_count,
                    metadata = EXCLUDED.metadata,
                    classification_result = EXCLUDED.classification_result,
                    classification_confidence = EXCLUDED.classification_confidence
                RETURNING *
                """,
                session_id,
                document_id,
                filename,
                s3_uri,
                document_type,
                row_count,
                column_count,
                metadata or {},
                classification_result,
                classification_confidence,
            )
            
            logger.info(f"Added document {document_id} to session {session_id}")
            return dict(row)
    
    async def get_document(
        self,
        session_id: UUID,
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Get a document from a session.
        
        Args:
            session_id: Session UUID
            document_id: Document identifier
            
        Returns:
            Document dict
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM session_documents
                WHERE session_id = $1 AND document_id = $2
                """,
                session_id,
                document_id,
            )
            
            if row is None:
                raise DocumentNotFoundError(
                    f"Document not found: {document_id} in session {session_id}"
                )
            
            return dict(row)
    
    async def list_documents(
        self,
        session_id: UUID,
    ) -> List[Dict[str, Any]]:
        """
        List all documents in a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            List of document dicts
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM session_documents
                WHERE session_id = $1
                ORDER BY uploaded_at DESC
                """,
                session_id,
            )
            
            return [dict(row) for row in rows]
    
    async def delete_document(
        self,
        session_id: UUID,
        document_id: str,
    ) -> bool:
        """
        Delete a document from a session.
        
        Args:
            session_id: Session UUID
            document_id: Document identifier
            
        Returns:
            True if deleted
        """
        # Get document to find S3 URI
        doc = await self.get_document(session_id, document_id)
        
        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM session_documents
                WHERE session_id = $1 AND document_id = $2
                """,
                session_id,
                document_id,
            )
            
            if result == "DELETE 0":
                raise DocumentNotFoundError(
                    f"Document not found: {document_id} in session {session_id}"
                )
        
        # Delete file from S3
        if doc.get("s3_uri"):
            try:
                _, s3_key = self.s3_client.parse_uri(doc["s3_uri"])
                await self.s3_client.delete_file(s3_key)
                logger.info(f"Deleted S3 file for document {document_id}")
            except Exception as e:
                logger.error(f"Failed to delete S3 file for document {document_id}: {e}")
        
        logger.info(f"Deleted document {document_id} from session {session_id}")
        return True
    
    # ========================================================================
    # Workflow Execution Operations
    # ========================================================================
    
    async def create_workflow_execution(
        self,
        workflow_name: str,
        parameters: Dict[str, Any],
        session_id: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        Create a workflow execution record.
        
        Args:
            workflow_name: Name of the workflow
            parameters: Execution parameters
            session_id: Optional session ID
            
        Returns:
            Execution dict
        """
        execution_id = uuid.uuid4()
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO workflow_executions (id, workflow_name, session_id, parameters, status)
                VALUES ($1, $2, $3, $4, 'pending')
                RETURNING *
                """,
                execution_id,
                workflow_name,
                session_id,
                parameters,
            )
            
            logger.info(f"Created workflow execution: {execution_id} ({workflow_name})")
            return dict(row)
    
    async def update_workflow_execution(
        self,
        execution_id: UUID,
        status: Optional[str] = None,
        results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        completed_at: Optional[datetime] = None,
        execution_time_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update workflow execution status.
        
        Args:
            execution_id: Execution UUID
            status: New status
            results: Execution results
            error: Error message if failed
            completed_at: Completion timestamp
            execution_time_ms: Execution time in milliseconds
            
        Returns:
            Updated execution dict
        """
        async with self.db_pool.acquire() as conn:
            updates = []
            params = [execution_id]
            param_idx = 2
            
            if status is not None:
                updates.append(f"status = ${param_idx}")
                params.append(status)
                param_idx += 1
            
            if results is not None:
                updates.append(f"results = ${param_idx}")
                params.append(results)
                param_idx += 1
            
            if error is not None:
                updates.append(f"error = ${param_idx}")
                params.append(error)
                param_idx += 1
            
            if completed_at is not None:
                updates.append(f"completed_at = ${param_idx}")
                params.append(completed_at)
                param_idx += 1
            
            if execution_time_ms is not None:
                updates.append(f"execution_time_ms = ${param_idx}")
                params.append(execution_time_ms)
                param_idx += 1
            
            if not updates:
                return await self.get_workflow_execution(execution_id)
            
            query = f"""
                UPDATE workflow_executions
                SET {', '.join(updates)}
                WHERE id = $1
                RETURNING *
            """
            
            row = await conn.fetchrow(query, *params)
            
            if row is None:
                raise ValueError(f"Workflow execution not found: {execution_id}")
            
            return dict(row)
    
    async def get_workflow_execution(self, execution_id: UUID) -> Dict[str, Any]:
        """Get workflow execution by ID."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM workflow_executions WHERE id = $1",
                execution_id,
            )
            
            if row is None:
                raise ValueError(f"Workflow execution not found: {execution_id}")
            
            return dict(row)
    
    async def list_workflow_executions(
        self,
        session_id: Optional[UUID] = None,
        workflow_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List workflow executions with optional filters."""
        async with self.db_pool.acquire() as conn:
            conditions = []
            params = []
            param_idx = 1
            
            if session_id is not None:
                conditions.append(f"session_id = ${param_idx}")
                params.append(session_id)
                param_idx += 1
            
            if workflow_name is not None:
                conditions.append(f"workflow_name = ${param_idx}")
                params.append(workflow_name)
                param_idx += 1
            
            if status is not None:
                conditions.append(f"status = ${param_idx}")
                params.append(status)
                param_idx += 1
            
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            
            query = f"""
                SELECT * FROM workflow_executions
                {where_clause}
                ORDER BY started_at DESC
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            params.extend([limit, offset])
            
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]

