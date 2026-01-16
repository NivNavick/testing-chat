"""
Session Management API Routes.

Provides endpoints for:
- Creating and managing processing sessions
- Uploading documents to sessions
- Session metadata operations
"""

import io
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from csv_analyzer.api.config import get_settings
from csv_analyzer.api.dependencies import get_db, get_s3_client, get_db_pool
from csv_analyzer.api.models.requests import CreateSessionRequest, UpdateSessionRequest
from csv_analyzer.api.models.responses import (
    DocumentResponse,
    SessionListResponse,
    SessionResponse,
)
from csv_analyzer.sessions.db_store import (
    AsyncSessionStore,
    DocumentNotFoundError,
    SessionNotFoundError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Dependencies
# ============================================================================

async def get_session_store() -> AsyncSessionStore:
    """Get the async session store."""
    from csv_analyzer.api.dependencies import get_db_pool, get_s3_client
    
    db_pool = get_db_pool()
    s3_client = get_s3_client()
    return AsyncSessionStore(db_pool, s3_client)


# ============================================================================
# Data Classes for Insights Integration
# ============================================================================

class TableInfo(BaseModel):
    """Information about a table in a session."""
    document_type: str
    table_name: str
    row_count: int = 0
    column_count: int = 0


class Session(BaseModel):
    """Session information for insights integration."""
    id: str
    name: str
    vertical: str
    tables: Dict[str, TableInfo] = {}
    duckdb_conn: Any = None  # Will hold DuckDB connection
    
    class Config:
        arbitrary_types_allowed = True


# In-memory session cache for active sessions
_active_sessions: Dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    """
    Get an active session by ID.
    
    This is used by the insights API for running insights on session data.
    
    Args:
        session_id: Session ID
        
    Returns:
        Session with tables and DuckDB connection
        
    Raises:
        HTTPException: If session not found
    """
    if session_id not in _active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found or not active. "
                   f"Use POST /api/v1/sessions/{session_id}/activate to activate."
        )
    
    return _active_sessions[session_id]


async def activate_session(session_id: str, store: AsyncSessionStore) -> Session:
    """
    Activate a session by loading its documents into DuckDB.
    
    Args:
        session_id: Session ID
        store: Session store
        
    Returns:
        Activated Session with DuckDB connection
    """
    import duckdb
    
    # Get session from database
    try:
        session_data = await store.get_session(UUID(session_id))
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    
    # Create in-memory DuckDB connection
    conn = duckdb.connect(":memory:")
    
    # Load documents into DuckDB
    documents = await store.list_documents(UUID(session_id))
    tables = {}
    
    settings = get_settings()
    s3_client = get_s3_client()
    
    for doc in documents:
        if doc.get("document_type") and doc.get("s3_uri"):
            try:
                # Download file from S3 to temp directory
                _, s3_key = s3_client.parse_uri(doc["s3_uri"])
                
                temp_dir = Path(settings.storage.local_temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
                local_path = temp_dir / f"{session_id}_{doc['document_id']}.csv"
                
                await s3_client.download_file(s3_key, local_path)
                
                # Load into DuckDB
                table_name = f"doc_{doc['document_id'].replace('-', '_')}"
                df = pd.read_csv(local_path)
                conn.register(table_name, df)
                
                tables[doc["document_id"]] = TableInfo(
                    document_type=doc["document_type"],
                    table_name=table_name,
                    row_count=doc.get("row_count", len(df)),
                    column_count=doc.get("column_count", len(df.columns)),
                )
                
                # Clean up temp file
                local_path.unlink(missing_ok=True)
                
                logger.info(f"Loaded document {doc['document_id']} as table {table_name}")
                
            except Exception as e:
                logger.error(f"Failed to load document {doc['document_id']}: {e}")
    
    # Create Session object
    session = Session(
        id=session_id,
        name=session_data["name"],
        vertical=session_data["vertical"],
        tables=tables,
        duckdb_conn=conn,
    )
    
    # Cache it
    _active_sessions[session_id] = session
    
    logger.info(f"Activated session {session_id} with {len(tables)} tables")
    return session


def deactivate_session(session_id: str) -> bool:
    """
    Deactivate a session and close its DuckDB connection.
    
    Args:
        session_id: Session ID
        
    Returns:
        True if deactivated
    """
    if session_id in _active_sessions:
        session = _active_sessions[session_id]
        if session.duckdb_conn:
            try:
                session.duckdb_conn.close()
            except Exception:
                pass
        del _active_sessions[session_id]
        logger.info(f"Deactivated session {session_id}")
        return True
    return False


# ============================================================================
# Session Endpoints
# ============================================================================

@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Create a new processing session.
    
    Args:
        request: Session creation request
        
    Returns:
        Created session
    """
    try:
        name = request.name or f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = await store.create_session(
            name=name,
            vertical=request.vertical,
            metadata=request.metadata,
        )
        
        return SessionResponse(
            id=str(session["id"]),
            name=session["name"],
            vertical=session["vertical"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            s3_bucket=session.get("s3_bucket"),
            s3_prefix=session.get("s3_prefix"),
            document_count=0,
            documents=[],
            metadata=session.get("metadata"),
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    vertical: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    List all sessions.
    
    Args:
        vertical: Optional vertical filter
        limit: Maximum results
        offset: Pagination offset
        
    Returns:
        List of sessions
    """
    try:
        sessions = await store.list_sessions(
            vertical=vertical,
            limit=limit,
            offset=offset,
        )
        total = await store.count_sessions(vertical=vertical)
        
        return SessionListResponse(
            sessions=[
                SessionResponse(
                    id=str(s["id"]),
                    name=s["name"],
                    vertical=s["vertical"],
                    created_at=s["created_at"],
                    updated_at=s["updated_at"],
                    s3_bucket=s.get("s3_bucket"),
                    s3_prefix=s.get("s3_prefix"),
                    document_count=s.get("document_count", 0),
                    metadata=s.get("metadata"),
                )
                for s in sessions
            ],
            total=total,
        )
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_details(
    session_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Get session details.
    
    Args:
        session_id: Session ID
        
    Returns:
        Session with documents
    """
    try:
        session = await store.get_session(UUID(session_id))
        documents = await store.list_documents(UUID(session_id))
        
        return SessionResponse(
            id=str(session["id"]),
            name=session["name"],
            vertical=session["vertical"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            s3_bucket=session.get("s3_bucket"),
            s3_prefix=session.get("s3_prefix"),
            document_count=len(documents),
            documents=[
                DocumentResponse(
                    id=str(d["id"]),
                    session_id=str(d["session_id"]),
                    document_id=d["document_id"],
                    document_type=d.get("document_type"),
                    filename=d["filename"],
                    s3_uri=d.get("s3_uri"),
                    row_count=d.get("row_count", 0),
                    column_count=d.get("column_count", 0),
                    uploaded_at=d["uploaded_at"],
                    metadata=d.get("metadata"),
                )
                for d in documents
            ],
            metadata=session.get("metadata"),
        )
        
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Update session metadata.
    
    Args:
        session_id: Session ID
        request: Update request
        
    Returns:
        Updated session
    """
    try:
        session = await store.update_session(
            session_id=UUID(session_id),
            name=request.name,
            metadata=request.metadata,
        )
        
        return SessionResponse(
            id=str(session["id"]),
            name=session["name"],
            vertical=session["vertical"],
            created_at=session["created_at"],
            updated_at=session["updated_at"],
            s3_bucket=session.get("s3_bucket"),
            s3_prefix=session.get("s3_prefix"),
            metadata=session.get("metadata"),
        )
        
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    except Exception as e:
        logger.error(f"Failed to update session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Delete a session and all its documents.
    
    Args:
        session_id: Session ID
    """
    try:
        # Deactivate if active
        deactivate_session(session_id)
        
        await store.delete_session(UUID(session_id))
        
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Document Endpoints
# ============================================================================

@router.post("/{session_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...),
    document_type: Optional[str] = Form(None),
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Upload a document to a session.
    
    Args:
        session_id: Session ID
        file: CSV file to upload
        document_type: Optional document type
        
    Returns:
        Uploaded document info
    """
    from csv_analyzer.api.dependencies import get_s3_client
    
    try:
        # Validate session exists
        session = await store.get_session(UUID(session_id))
        
        # Read file content
        content = await file.read()
        
        # Parse CSV to get row/column counts
        try:
            df = pd.read_csv(io.BytesIO(content))
            row_count = len(df)
            column_count = len(df.columns)
        except Exception as e:
            logger.warning(f"Could not parse CSV: {e}")
            row_count = 0
            column_count = 0
        
        # Generate document ID
        document_id = str(uuid.uuid4())[:8]
        
        # Upload to S3
        s3_client = get_s3_client()
        settings = get_settings()
        
        s3_key = f"{session['s3_prefix']}documents/{document_id}_{file.filename}"
        s3_uri = await s3_client.upload_bytes(
            content,
            s3_key,
            content_type="text/csv",
        )
        
        # Add document to database
        doc = await store.add_document(
            session_id=UUID(session_id),
            document_id=document_id,
            filename=file.filename,
            s3_uri=s3_uri,
            document_type=document_type,
            row_count=row_count,
            column_count=column_count,
        )
        
        # Invalidate active session cache if exists
        if session_id in _active_sessions:
            deactivate_session(session_id)
        
        return DocumentResponse(
            id=str(doc["id"]),
            session_id=str(doc["session_id"]),
            document_id=doc["document_id"],
            document_type=doc.get("document_type"),
            filename=doc["filename"],
            s3_uri=doc.get("s3_uri"),
            row_count=doc.get("row_count", 0),
            column_count=doc.get("column_count", 0),
            uploaded_at=doc["uploaded_at"],
            metadata=doc.get("metadata"),
        )
        
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    except Exception as e:
        logger.error(f"Failed to upload document to session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{session_id}/documents", response_model=List[DocumentResponse])
async def list_documents(
    session_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    List all documents in a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        List of documents
    """
    try:
        # Validate session exists
        await store.get_session(UUID(session_id))
        
        documents = await store.list_documents(UUID(session_id))
        
        return [
            DocumentResponse(
                id=str(d["id"]),
                session_id=str(d["session_id"]),
                document_id=d["document_id"],
                document_type=d.get("document_type"),
                filename=d["filename"],
                s3_uri=d.get("s3_uri"),
                row_count=d.get("row_count", 0),
                column_count=d.get("column_count", 0),
                uploaded_at=d["uploaded_at"],
                metadata=d.get("metadata"),
            )
            for d in documents
        ]
        
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    except Exception as e:
        logger.error(f"Failed to list documents for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/{session_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    session_id: str,
    document_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Delete a document from a session.
    
    Args:
        session_id: Session ID
        document_id: Document ID
    """
    try:
        await store.delete_document(UUID(session_id), document_id)
        
        # Invalidate active session cache if exists
        if session_id in _active_sessions:
            deactivate_session(session_id)
        
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}"
        )
    except SessionNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}"
        )
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Session Activation Endpoints (for Insights)
# ============================================================================

@router.post("/{session_id}/activate")
async def activate_session_endpoint(
    session_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Activate a session by loading documents into DuckDB.
    
    This is required before running insights on the session.
    
    Args:
        session_id: Session ID
        
    Returns:
        Session info with loaded tables
    """
    try:
        session = await activate_session(session_id, store)
        
        return {
            "id": session.id,
            "name": session.name,
            "vertical": session.vertical,
            "status": "active",
            "tables": {
                doc_id: {
                    "document_type": table.document_type,
                    "table_name": table.table_name,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                }
                for doc_id, table in session.tables.items()
            },
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{session_id}/deactivate")
async def deactivate_session_endpoint(session_id: str):
    """
    Deactivate a session and free DuckDB resources.
    
    Args:
        session_id: Session ID
        
    Returns:
        Deactivation status
    """
    deactivated = deactivate_session(session_id)
    
    if not deactivated:
        return {
            "id": session_id,
            "status": "not_active",
            "message": "Session was not active",
        }
    
    return {
        "id": session_id,
        "status": "deactivated",
        "message": "Session deactivated successfully",
    }

