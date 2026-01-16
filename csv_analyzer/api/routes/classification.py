"""
CSV Classification API Routes.

Provides endpoints for:
- Classifying CSV files
- Batch classification
- Listing verticals and document types
"""

import io
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from csv_analyzer.api.config import get_settings
from csv_analyzer.api.dependencies import get_db, get_s3_client, run_sync
from csv_analyzer.api.models.requests import ClassifyRequest
from csv_analyzer.api.models.responses import (
    ClassificationResponse,
    BatchClassificationResponse,
    ColumnMappingResponse,
    DocumentTypeListResponse,
    DocumentTypeResponse,
    VerticalListResponse,
    VerticalResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Classification Endpoints
# ============================================================================

@router.post("/classify", response_model=ClassificationResponse)
async def classify_csv(
    file: UploadFile = File(...),
    vertical: Optional[str] = Form(None),
    k: int = Form(5),
    hybrid: bool = Form(True),
    use_dspy: bool = Form(False),
    force_reindex: bool = Form(False),
):
    """
    Classify an uploaded CSV file.
    
    Analyzes the CSV structure and content to determine:
    - Document type (e.g., employee_shifts, medical_actions)
    - Suggested column mappings to target schema
    - Confidence scores
    
    Args:
        file: CSV file to classify
        vertical: Optional vertical domain filter (e.g., 'medical')
        k: Number of similar examples to retrieve (default: 5)
        hybrid: Use hybrid scoring (document + column level)
        use_dspy: Use DSPy for LLM-based column classification
        force_reindex: Force reindex of schema embeddings
        
    Returns:
        Classification results with document type and mappings
    """
    settings = get_settings()
    
    try:
        # Read file content
        content = await file.read()
        
        # Save to temp file for classification engine
        temp_dir = Path(settings.storage.local_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_path = temp_dir / f"classify_{uuid.uuid4()}.csv"
        temp_path.write_bytes(content)
        
        try:
            # Run classification in thread pool (sync engine)
            result = await run_sync(
                _classify_csv_sync,
                str(temp_path),
                vertical,
                k,
                hybrid,
                use_dspy,
                force_reindex,
            )
            
            return result
            
        finally:
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


def _classify_csv_sync(
    csv_path: str,
    vertical: Optional[str],
    k: int,
    hybrid: bool,
    use_dspy: bool,
    force_reindex: bool,
) -> ClassificationResponse:
    """
    Synchronous classification function to run in thread pool.
    """
    from csv_analyzer.db.connection import init_database, Database
    from csv_analyzer.engines.classification_engine import ClassificationEngine
    from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client
    
    # Initialize database if not already done
    if not Database.is_initialized():
        settings = get_settings()
        init_database(
            host=settings.database.host,
            port=settings.database.port,
            database=settings.database.name,
            user=settings.database.user,
            password=settings.database.password,
            run_migrations=False,
        )
    
    # Get embeddings client
    embeddings_client = get_multilingual_embeddings_client()
    
    if not embeddings_client.is_available:
        raise ValueError("Embedding model not available")
    
    # Create DSPy service if requested
    dspy_service = None
    if use_dspy:
        try:
            from csv_analyzer.services.dspy_service import create_dspy_service
            dspy_service = create_dspy_service(enabled=True)
        except Exception as e:
            logger.warning(f"Failed to initialize DSPy: {e}")
    
    # Create classification engine
    engine = ClassificationEngine(
        embeddings_client,
        dspy_service=dspy_service,
    )
    
    # Classify
    if hybrid and vertical:
        result = engine.classify_hybrid(
            csv_file=csv_path,
            vertical=vertical,
            k=k,
            force_reindex=force_reindex,
        )
        
        # Convert to response
        return ClassificationResponse(
            document_type=result.document_type,
            vertical=result.vertical,
            confidence=result.final_score,
            final_score=result.final_score,
            document_score=result.document_score,
            column_score=result.column_score,
            coverage_score=result.coverage_score,
            suggested_mappings={
                source: ColumnMappingResponse(
                    source_column=source,
                    target_field=mapping.get("target_field") or mapping.get("target"),
                    confidence=mapping.get("confidence", 0),
                    required=mapping.get("required", False),
                    source_type=mapping.get("source_type"),
                    target_type=mapping.get("field_type"),
                    sources=mapping.get("sources", ["embeddings"]),
                    reason=mapping.get("reason"),
                )
                for source, mapping in result.suggested_mappings.items()
            },
            similar_examples=result.similar_examples,
            column_profiles=result.column_profiles,
            all_scores=result.all_scores,
        )
    else:
        result = engine.classify(
            csv_file=csv_path,
            vertical=vertical,
            k=k,
        )
        
        return ClassificationResponse(
            document_type=result.document_type,
            vertical=result.vertical,
            confidence=result.confidence,
            suggested_mappings={
                source: ColumnMappingResponse(
                    source_column=source,
                    target_field=mapping.get("target"),
                    confidence=mapping.get("confidence", 0),
                    required=mapping.get("required", False),
                )
                for source, mapping in result.suggested_mappings.items()
            },
            similar_examples=result.similar_examples,
            column_profiles=result.column_profiles,
        )


@router.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(
    files: List[UploadFile] = File(...),
    vertical: Optional[str] = Form(None),
    k: int = Form(5),
    hybrid: bool = Form(True),
):
    """
    Classify multiple CSV files in batch.
    
    Args:
        files: List of CSV files to classify
        vertical: Optional vertical domain filter
        k: Number of similar examples per file
        hybrid: Use hybrid scoring
        
    Returns:
        Batch classification results
    """
    settings = get_settings()
    results = []
    errors = []
    
    for file in files:
        try:
            # Read file content
            content = await file.read()
            
            # Save to temp file
            temp_dir = Path(settings.storage.local_temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            temp_path = temp_dir / f"classify_{uuid.uuid4()}.csv"
            temp_path.write_bytes(content)
            
            try:
                # Classify
                result = await run_sync(
                    _classify_csv_sync,
                    str(temp_path),
                    vertical,
                    k,
                    hybrid,
                    False,  # use_dspy
                    False,  # force_reindex
                )
                results.append(result)
                
            finally:
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Failed to classify {file.filename}: {e}")
            errors.append({
                "filename": file.filename,
                "error": str(e),
            })
    
    return BatchClassificationResponse(
        results=results,
        total_files=len(files),
        successful=len(results),
        failed=len(errors),
        errors=errors,
    )


# ============================================================================
# Metadata Endpoints
# ============================================================================

@router.get("/verticals", response_model=VerticalListResponse)
async def list_verticals():
    """
    List available verticals.
    
    Returns:
        List of verticals with their document types
    """
    try:
        verticals = await run_sync(_list_verticals_sync)
        
        return VerticalListResponse(
            verticals=verticals,
            total=len(verticals),
        )
        
    except Exception as e:
        logger.error(f"Failed to list verticals: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _list_verticals_sync() -> List[VerticalResponse]:
    """List verticals synchronously."""
    from csv_analyzer.db.connection import Database, init_database
    from csv_analyzer.api.config import get_settings
    
    if not Database.is_initialized():
        settings = get_settings()
        init_database(
            host=settings.database.host,
            port=settings.database.port,
            database=settings.database.name,
            user=settings.database.user,
            password=settings.database.password,
            run_migrations=False,
        )
    
    with Database.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT v.name, v.description,
                       COALESCE(array_agg(dt.name) FILTER (WHERE dt.name IS NOT NULL), '{}') as document_types
                FROM verticals v
                LEFT JOIN document_types dt ON v.id = dt.vertical_id
                GROUP BY v.id, v.name, v.description
                ORDER BY v.name
            """)
            rows = cur.fetchall()
            
            return [
                VerticalResponse(
                    name=row[0],
                    display_name=row[0].replace("_", " ").title(),
                    description=row[1],
                    document_types=list(row[2]) if row[2] else [],
                )
                for row in rows
            ]


@router.get("/document-types", response_model=DocumentTypeListResponse)
async def list_document_types(
    vertical: Optional[str] = None,
):
    """
    List available document types.
    
    Args:
        vertical: Optional vertical filter
        
    Returns:
        List of document types
    """
    try:
        document_types = await run_sync(_list_document_types_sync, vertical)
        
        return DocumentTypeListResponse(
            document_types=document_types,
            total=len(document_types),
        )
        
    except Exception as e:
        logger.error(f"Failed to list document types: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _list_document_types_sync(vertical: Optional[str]) -> List[DocumentTypeResponse]:
    """List document types synchronously."""
    from csv_analyzer.db.connection import Database, init_database
    from csv_analyzer.api.config import get_settings
    
    if not Database.is_initialized():
        settings = get_settings()
        init_database(
            host=settings.database.host,
            port=settings.database.port,
            database=settings.database.name,
            user=settings.database.user,
            password=settings.database.password,
            run_migrations=False,
        )
    
    with Database.get_connection() as conn:
        with conn.cursor() as cur:
            if vertical:
                cur.execute("""
                    SELECT dt.name, v.name as vertical, dt.description,
                           (SELECT COUNT(*) FROM target_schemas ts 
                            WHERE ts.document_type_id = dt.id) as field_count
                    FROM document_types dt
                    JOIN verticals v ON dt.vertical_id = v.id
                    WHERE v.name = %s
                    ORDER BY dt.name
                """, (vertical,))
            else:
                cur.execute("""
                    SELECT dt.name, v.name as vertical, dt.description,
                           (SELECT COUNT(*) FROM target_schemas ts 
                            WHERE ts.document_type_id = dt.id) as field_count
                    FROM document_types dt
                    JOIN verticals v ON dt.vertical_id = v.id
                    ORDER BY v.name, dt.name
                """)
            
            rows = cur.fetchall()
            
            return [
                DocumentTypeResponse(
                    name=row[0],
                    vertical=row[1],
                    display_name=row[0].replace("_", " ").title(),
                    description=row[2],
                    field_count=row[3] or 0,
                )
                for row in rows
            ]

