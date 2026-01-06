"""
Session management API routes.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from csv_analyzer.storage.postgres.connection import get_db_session
from csv_analyzer.storage.postgres.repositories.session_repo import SessionRepository
from csv_analyzer.storage.postgres.repositories.file_repo import FileRepository
from csv_analyzer.storage.postgres.repositories.table_repo import TableRepository
from csv_analyzer.storage.postgres.repositories.insight_repo import InsightRepository
from csv_analyzer.storage.s3.operations import S3Operations, get_s3_operations
from csv_analyzer.api.models.requests import CreateSessionRequest
from csv_analyzer.api.models.responses import (
    SessionResponse,
    SessionDetailResponse,
    FileResponse,
    TableResponse,
    InsightResultResponse
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    Create a new analytics session.
    
    A session groups related file uploads and insights together.
    """
    repo = SessionRepository(db_session)
    session = await repo.create(
        vertical=request.vertical,
        mode=request.mode
    )
    return SessionResponse(**session)


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    limit: int = 100,
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    List active sessions.
    """
    repo = SessionRepository(db_session)
    sessions = await repo.list_active(limit=limit)
    return [SessionResponse(**s) for s in sessions]


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    Get session details with all related data.
    """
    session_repo = SessionRepository(db_session)
    file_repo = FileRepository(db_session)
    table_repo = TableRepository(db_session)
    insight_repo = InsightRepository(db_session)
    
    session = await session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    files = await file_repo.list_by_session(session_id)
    tables = await table_repo.list_by_session(session_id)
    results = await insight_repo.list_by_session(session_id)
    
    # TODO: Get relationships and rules
    # TODO: Calculate available insights based on loaded tables
    
    return SessionDetailResponse(
        session=SessionResponse(**session),
        files=[FileResponse(**f) for f in files],
        tables=[TableResponse(**t) for t in tables],
        relationships=[],
        rules=[],
        insights_available=[],
        results=[InsightResultResponse(**r) for r in results]
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    s3_ops: S3Operations = Depends(get_s3_operations)
):
    """
    Close and delete a session, including all S3 files.
    """
    session_repo = SessionRepository(db_session)
    
    session = await session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete S3 files
    deleted_files = await s3_ops.delete_session_files(session_id)
    
    # Delete session (cascades to all related data)
    await session_repo.delete(session_id)
    
    return {
        "message": "Session deleted",
        "s3_files_deleted": deleted_files
    }

