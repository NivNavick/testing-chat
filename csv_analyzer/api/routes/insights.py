"""
Insights API routes.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from csv_analyzer.storage.postgres.connection import get_db_session
from csv_analyzer.storage.postgres.repositories.session_repo import SessionRepository
from csv_analyzer.storage.postgres.repositories.insight_repo import InsightRepository
from csv_analyzer.storage.s3.operations import S3Operations, get_s3_operations
from csv_analyzer.api.models.requests import RunInsightRequest, GenerateInsightRequest
from csv_analyzer.api.models.responses import (
    InsightResultResponse,
    InsightResultDataResponse,
    AvailableInsightResponse,
    DownloadUrlResponse
)

router = APIRouter(prefix="/sessions/{session_id}", tags=["insights"])


@router.get("/insights", response_model=List[AvailableInsightResponse])
async def list_available_insights(
    session_id: str,
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    List available insights for this session.
    
    Insights are available based on what tables have been loaded.
    """
    # Verify session exists
    session_repo = SessionRepository(db_session)
    session = await session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Get loaded table types from classification
    # TODO: Match against insight definitions
    # TODO: Return available insights with requirements
    
    return []  # Placeholder


@router.post("/insights/run", response_model=InsightResultResponse)
async def run_insight(
    session_id: str,
    request: RunInsightRequest,
    db_session: AsyncSession = Depends(get_db_session),
    s3_ops: S3Operations = Depends(get_s3_operations)
):
    """
    Run an insight query.
    
    The insight will be executed using DuckDB and results saved to:
    - PostgreSQL (full data for querying)
    - S3 (CSV export for download)
    """
    # Verify session exists
    session_repo = SessionRepository(db_session)
    session = await session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Load insight definition
    # TODO: Load required tables from S3 into DuckDB
    # TODO: Execute query
    # TODO: Save results to PostgreSQL and S3
    
    raise HTTPException(
        status_code=501,
        detail="Insight execution not yet implemented"
    )


@router.post("/insights/generate", response_model=InsightResultResponse)
async def generate_insight(
    session_id: str,
    request: GenerateInsightRequest,
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    Generate a custom insight using LLM.
    
    Describe what you want to analyze in natural language,
    and the system will generate and execute the appropriate SQL.
    """
    # Verify session exists
    session_repo = SessionRepository(db_session)
    session = await session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # TODO: Use LLM to generate SQL from description
    # TODO: Validate generated SQL
    # TODO: Execute and save results
    
    raise HTTPException(
        status_code=501,
        detail="Insight generation not yet implemented"
    )


@router.get("/results", response_model=List[InsightResultResponse])
async def list_results(
    session_id: str,
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    List all insight results for a session.
    """
    # Verify session exists
    session_repo = SessionRepository(db_session)
    session = await session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    insight_repo = InsightRepository(db_session)
    results = await insight_repo.list_by_session(session_id)
    return [InsightResultResponse(**r) for r in results]


@router.get("/results/{result_id}", response_model=InsightResultDataResponse)
async def get_result(
    session_id: str,
    result_id: str,
    limit: int = 1000,
    offset: int = 0,
    db_session: AsyncSession = Depends(get_db_session),
    s3_ops: S3Operations = Depends(get_s3_operations)
):
    """
    Get an insight result with data.
    """
    insight_repo = InsightRepository(db_session)
    
    result = await insight_repo.get_result_by_id(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    if result["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Result not found in this session")
    
    # Get result data from PostgreSQL
    data = await insight_repo.get_result_data(result_id, limit=limit, offset=offset)
    
    # Generate download URL if S3 key exists
    download_url = None
    if result.get("s3_result_key"):
        download_url = await s3_ops.generate_download_url(result["s3_result_key"])
    
    return InsightResultDataResponse(
        result=InsightResultResponse(**result),
        data=data,
        download_url=download_url
    )


@router.get("/results/{result_id}/download", response_model=DownloadUrlResponse)
async def get_result_download_url(
    session_id: str,
    result_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    s3_ops: S3Operations = Depends(get_s3_operations)
):
    """
    Get a presigned URL to download result CSV.
    """
    insight_repo = InsightRepository(db_session)
    
    result = await insight_repo.get_result_by_id(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    if result["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Result not found in this session")
    
    if not result.get("s3_result_key"):
        raise HTTPException(status_code=404, detail="No CSV export available")
    
    url = await s3_ops.generate_download_url(
        key=result["s3_result_key"],
        expiration=3600
    )
    
    return DownloadUrlResponse(url=url, expires_in=3600)

