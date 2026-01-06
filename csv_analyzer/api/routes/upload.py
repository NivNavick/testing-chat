"""
File upload API routes.
"""

import io
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from csv_analyzer.storage.postgres.connection import get_db_session
from csv_analyzer.storage.postgres.repositories.session_repo import SessionRepository
from csv_analyzer.storage.postgres.repositories.file_repo import FileRepository
from csv_analyzer.storage.postgres.repositories.table_repo import TableRepository
from csv_analyzer.storage.s3.operations import S3Operations, get_s3_operations
from csv_analyzer.api.models.responses import (
    FileResponse,
    TableResponse,
    UploadResponse,
    DownloadUrlResponse
)

router = APIRouter(prefix="/sessions/{session_id}", tags=["files"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    db_session: AsyncSession = Depends(get_db_session),
    s3_ops: S3Operations = Depends(get_s3_operations)
):
    """
    Upload a file (CSV or XLSX) for processing.
    
    The file will be:
    1. Uploaded to S3
    2. Tables extracted (for XLSX, multiple tables may be extracted)
    3. Tables classified
    4. Relationships detected
    """
    # Verify session exists
    session_repo = SessionRepository(db_session)
    session = await session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Validate file type
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Supported: CSV, XLSX, XLS"
        )
    
    # Read file content
    content = await file.read()
    file_obj = io.BytesIO(content)
    
    # Upload to S3
    s3_info = await s3_ops.upload_raw_file(
        session_id=session_id,
        filename=file.filename,
        file_obj=file_obj
    )
    
    # Create file record
    file_repo = FileRepository(db_session)
    file_record = await file_repo.create(
        session_id=session_id,
        filename=file.filename,
        file_type=s3_info.file_type,
        s3_key=s3_info.key,
        file_size_bytes=s3_info.size_bytes
    )
    
    # Mark as processing
    await file_repo.update_status(file_record["id"], "processing")
    
    # Extract tables (simplified for now)
    # TODO: Use the actual table extraction logic from xlsx_table_extractor
    table_repo = TableRepository(db_session)
    tables = []
    
    try:
        import pandas as pd
        
        file_obj.seek(0)
        
        if s3_info.file_type == "csv":
            # Single table for CSV
            df = pd.read_csv(file_obj)
            table = await table_repo.create(
                file_id=file_record["id"],
                session_id=session_id,
                table_name=file.filename.rsplit(".", 1)[0],
                row_count=len(df),
                column_count=len(df.columns),
                columns=[
                    {"name": col, "type": str(df[col].dtype)}
                    for col in df.columns
                ],
                structure_type="STANDARD"
            )
            tables.append(table)
        else:
            # XLSX - may have multiple sheets/tables
            xlsx = pd.ExcelFile(file_obj)
            
            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name=sheet_name)
                if len(df) > 0:
                    table = await table_repo.create(
                        file_id=file_record["id"],
                        session_id=session_id,
                        table_name=sheet_name,
                        location=sheet_name,
                        row_count=len(df),
                        column_count=len(df.columns),
                        columns=[
                            {"name": str(col), "type": str(df[col].dtype)}
                            for col in df.columns
                        ],
                        structure_type="STANDARD"
                    )
                    tables.append(table)
        
        # Mark as completed
        await file_repo.update_status(file_record["id"], "completed")
        
    except Exception as e:
        await file_repo.update_status(
            file_record["id"],
            "failed",
            error_message=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )
    
    # TODO: Run classification on extracted tables
    # TODO: Detect relationships between tables
    # TODO: Calculate available insights
    
    return UploadResponse(
        file=FileResponse(**file_record),
        tables=[TableResponse(**t) for t in tables],
        insights_available=[]  # TODO: Calculate based on loaded tables
    )


@router.get("/files", response_model=List[FileResponse])
async def list_files(
    session_id: str,
    db_session: AsyncSession = Depends(get_db_session)
):
    """
    List all files uploaded to a session.
    """
    # Verify session exists
    session_repo = SessionRepository(db_session)
    session = await session_repo.get_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    file_repo = FileRepository(db_session)
    files = await file_repo.list_by_session(session_id)
    return [FileResponse(**f) for f in files]


@router.get("/files/{file_id}/download", response_model=DownloadUrlResponse)
async def get_file_download_url(
    session_id: str,
    file_id: str,
    db_session: AsyncSession = Depends(get_db_session),
    s3_ops: S3Operations = Depends(get_s3_operations)
):
    """
    Get a presigned URL to download a file.
    """
    file_repo = FileRepository(db_session)
    file_record = await file_repo.get_by_id(file_id)
    
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    
    if file_record["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="File not found in this session")
    
    url = await s3_ops.generate_download_url(
        key=file_record["s3_key"],
        expiration=3600
    )
    
    return DownloadUrlResponse(url=url, expires_in=3600)

