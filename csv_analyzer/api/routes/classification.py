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


# ============================================================================
# Ground Truth / Training Endpoints
# ============================================================================

from pydantic import BaseModel
from typing import Dict


class GroundTruthIngestRequest(BaseModel):
    """Request model for ground truth ingestion."""
    vertical: str
    document_type: str
    column_mappings: Dict[str, str]
    external_id: Optional[str] = None
    source_description: Optional[str] = None
    labeler: Optional[str] = None
    notes: Optional[str] = None


class GroundTruthResponse(BaseModel):
    """Response model for ground truth records."""
    id: int
    external_id: str
    vertical: str
    document_type: str
    source_description: Optional[str]
    row_count: int
    column_count: int
    column_mappings: Dict[str, str]
    labeler: Optional[str]
    notes: Optional[str]
    created_at: Optional[str]


class GroundTruthListResponse(BaseModel):
    """List of ground truth records."""
    records: List[GroundTruthResponse]
    total: int


@router.post("/training/ingest", response_model=GroundTruthResponse)
async def ingest_ground_truth(
    file: UploadFile = File(...),
    vertical: str = Form(...),
    document_type: str = Form(...),
    column_mappings: str = Form(...),  # JSON string
    external_id: Optional[str] = Form(None),
    source_description: Optional[str] = Form(None),
    labeler: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
):
    """
    Ingest a CSV file as ground truth for training the classification engine.
    
    This endpoint allows you to add labeled examples that the classification
    engine will use to recognize similar documents in the future.
    
    Args:
        file: CSV file to ingest as ground truth
        vertical: Vertical domain (e.g., 'medical', 'banking')
        document_type: Document type (e.g., 'employee_shifts', 'medical_actions')
        column_mappings: JSON string mapping source columns to target fields
                        e.g., '{"emp_id": "employee_id", "work_date": "shift_date"}'
        external_id: Optional unique identifier (auto-generated if not provided)
        source_description: Optional description of the data source
        labeler: Optional name/email of person who labeled this
        notes: Optional notes about this ground truth
        
    Returns:
        Created ground truth record info
        
    Example:
        curl -X POST /api/v1/classification/training/ingest \\
          -F "file=@shifts.csv" \\
          -F "vertical=medical" \\
          -F "document_type=employee_shifts" \\
          -F 'column_mappings={"עובד": "employee_name", "תאריך": "shift_date"}'
    """
    import json
    
    settings = get_settings()
    
    # Parse column mappings JSON
    try:
        mappings = json.loads(column_mappings)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid column_mappings JSON: {e}"
        )
    
    if not isinstance(mappings, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="column_mappings must be a JSON object"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Save to temp file
        temp_dir = Path(settings.storage.local_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        temp_path = temp_dir / f"ingest_{uuid.uuid4()}.csv"
        temp_path.write_bytes(content)
        
        try:
            # Run ingestion in thread pool
            result = await run_sync(
                _ingest_ground_truth_sync,
                str(temp_path),
                vertical,
                document_type,
                mappings,
                external_id,
                source_description or f"Uploaded: {file.filename}",
                labeler,
                notes,
            )
            
            return result
            
        finally:
            temp_path.unlink(missing_ok=True)
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to ingest ground truth: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


def _ingest_ground_truth_sync(
    csv_path: str,
    vertical: str,
    document_type: str,
    column_mappings: Dict[str, str],
    external_id: Optional[str],
    source_description: Optional[str],
    labeler: Optional[str],
    notes: Optional[str],
) -> GroundTruthResponse:
    """Synchronous ground truth ingestion."""
    from csv_analyzer.db.connection import init_database, Database
    from csv_analyzer.engines.ingestion_engine import IngestionEngine
    from csv_analyzer.multilingual_embeddings_client import get_multilingual_embeddings_client
    from csv_analyzer.db.repositories.ground_truth_repo import GroundTruthRepository
    
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
    
    # Create ingestion engine and ingest
    engine = IngestionEngine(embeddings_client)
    
    gt_id = engine.ingest_csv(
        csv_file=csv_path,
        vertical=vertical,
        document_type=document_type,
        column_mappings=column_mappings,
        external_id=external_id,
        source_description=source_description,
        labeler=labeler,
        notes=notes,
    )
    
    # Fetch the created record
    record = GroundTruthRepository.get_by_external_id(
        external_id or f"gt_{vertical}_{document_type}_{gt_id}"
    )
    
    if not record:
        # Fallback: fetch by ID
        with Database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT gt.id, gt.external_id, v.name as vertical, dt.name as document_type,
                           gt.source_description, gt.row_count, gt.column_count,
                           gt.column_mappings, gt.labeler, gt.notes, gt.created_at
                    FROM ground_truth gt
                    JOIN document_types dt ON gt.document_type_id = dt.id
                    JOIN verticals v ON dt.vertical_id = v.id
                    WHERE gt.id = %s
                """, (gt_id,))
                row = cur.fetchone()
                
                if row:
                    import json
                    return GroundTruthResponse(
                        id=row[0],
                        external_id=row[1],
                        vertical=row[2],
                        document_type=row[3],
                        source_description=row[4],
                        row_count=row[5],
                        column_count=row[6],
                        column_mappings=json.loads(row[7]) if isinstance(row[7], str) else row[7],
                        labeler=row[8],
                        notes=row[9],
                        created_at=str(row[10]) if row[10] else None,
                    )
    
    import json
    return GroundTruthResponse(
        id=record.get("id", gt_id),
        external_id=record.get("external_id", ""),
        vertical=vertical,
        document_type=document_type,
        source_description=record.get("source_description"),
        row_count=record.get("row_count", 0),
        column_count=record.get("column_count", 0),
        column_mappings=record.get("column_mappings", {}),
        labeler=record.get("labeler"),
        notes=record.get("notes"),
        created_at=str(record.get("created_at")) if record.get("created_at") else None,
    )


@router.get("/training/ground-truth", response_model=GroundTruthListResponse)
async def list_ground_truth(
    vertical: Optional[str] = None,
    document_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """
    List ground truth records used for classification training.
    
    Args:
        vertical: Optional filter by vertical
        document_type: Optional filter by document type
        limit: Maximum records to return
        offset: Pagination offset
        
    Returns:
        List of ground truth records
    """
    try:
        records = await run_sync(
            _list_ground_truth_sync,
            vertical,
            document_type,
            limit,
            offset,
        )
        
        return GroundTruthListResponse(
            records=records,
            total=len(records),
        )
        
    except Exception as e:
        logger.error(f"Failed to list ground truth: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _list_ground_truth_sync(
    vertical: Optional[str],
    document_type: Optional[str],
    limit: int,
    offset: int,
) -> List[GroundTruthResponse]:
    """List ground truth records synchronously."""
    from csv_analyzer.db.connection import Database, init_database
    import json
    
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
            query = """
                SELECT gt.id, gt.external_id, v.name as vertical, dt.name as document_type,
                       gt.source_description, gt.row_count, gt.column_count,
                       gt.column_mappings, gt.labeler, gt.notes, gt.created_at
                FROM ground_truth gt
                JOIN document_types dt ON gt.document_type_id = dt.id
                JOIN verticals v ON dt.vertical_id = v.id
                WHERE 1=1
            """
            params = []
            
            if vertical:
                query += " AND v.name = %s"
                params.append(vertical)
            
            if document_type:
                query += " AND dt.name = %s"
                params.append(document_type)
            
            query += " ORDER BY gt.created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            return [
                GroundTruthResponse(
                    id=row[0],
                    external_id=row[1],
                    vertical=row[2],
                    document_type=row[3],
                    source_description=row[4],
                    row_count=row[5],
                    column_count=row[6],
                    column_mappings=json.loads(row[7]) if isinstance(row[7], str) else row[7],
                    labeler=row[8],
                    notes=row[9],
                    created_at=str(row[10]) if row[10] else None,
                )
                for row in rows
            ]


@router.get("/training/ground-truth/{external_id}", response_model=GroundTruthResponse)
async def get_ground_truth(external_id: str):
    """
    Get a specific ground truth record by external ID.
    
    Args:
        external_id: External ID of the ground truth record
        
    Returns:
        Ground truth record details
    """
    try:
        record = await run_sync(_get_ground_truth_sync, external_id)
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ground truth not found: {external_id}"
            )
        
        return record
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get ground truth: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _get_ground_truth_sync(external_id: str) -> Optional[GroundTruthResponse]:
    """Get ground truth record synchronously."""
    from csv_analyzer.db.connection import Database, init_database
    import json
    
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
                SELECT gt.id, gt.external_id, v.name as vertical, dt.name as document_type,
                       gt.source_description, gt.row_count, gt.column_count,
                       gt.column_mappings, gt.labeler, gt.notes, gt.created_at
                FROM ground_truth gt
                JOIN document_types dt ON gt.document_type_id = dt.id
                JOIN verticals v ON dt.vertical_id = v.id
                WHERE gt.external_id = %s
            """, (external_id,))
            row = cur.fetchone()
            
            if not row:
                return None
            
            return GroundTruthResponse(
                id=row[0],
                external_id=row[1],
                vertical=row[2],
                document_type=row[3],
                source_description=row[4],
                row_count=row[5],
                column_count=row[6],
                column_mappings=json.loads(row[7]) if isinstance(row[7], str) else row[7],
                labeler=row[8],
                notes=row[9],
                created_at=str(row[10]) if row[10] else None,
            )


@router.delete("/training/ground-truth/{external_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ground_truth(external_id: str):
    """
    Delete a ground truth record.
    
    Args:
        external_id: External ID of the ground truth record to delete
    """
    try:
        deleted = await run_sync(_delete_ground_truth_sync, external_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ground truth not found: {external_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete ground truth: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _delete_ground_truth_sync(external_id: str) -> bool:
    """Delete ground truth record synchronously."""
    from csv_analyzer.db.connection import Database, init_database
    
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
            cur.execute(
                "DELETE FROM ground_truth WHERE external_id = %s",
                (external_id,)
            )
            deleted = cur.rowcount > 0
            conn.commit()
            
            if deleted:
                logger.info(f"Deleted ground truth: {external_id}")
            
            return deleted


@router.get("/training/stats")
async def get_training_stats():
    """
    Get statistics about the training data.
    
    Returns:
        Statistics including counts by vertical and document type
    """
    try:
        stats = await run_sync(_get_training_stats_sync)
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _get_training_stats_sync() -> Dict[str, Any]:
    """Get training statistics synchronously."""
    from csv_analyzer.db.connection import Database, init_database
    
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
            # Total ground truth records
            cur.execute("SELECT COUNT(*) FROM ground_truth")
            total_records = cur.fetchone()[0]
            
            # Records by vertical
            cur.execute("""
                SELECT v.name, COUNT(gt.id)
                FROM ground_truth gt
                JOIN document_types dt ON gt.document_type_id = dt.id
                JOIN verticals v ON dt.vertical_id = v.id
                GROUP BY v.name
                ORDER BY COUNT(gt.id) DESC
            """)
            by_vertical = {row[0]: row[1] for row in cur.fetchall()}
            
            # Records by document type
            cur.execute("""
                SELECT v.name || '/' || dt.name, COUNT(gt.id)
                FROM ground_truth gt
                JOIN document_types dt ON gt.document_type_id = dt.id
                JOIN verticals v ON dt.vertical_id = v.id
                GROUP BY v.name, dt.name
                ORDER BY COUNT(gt.id) DESC
            """)
            by_document_type = {row[0]: row[1] for row in cur.fetchall()}
            
            # Column mappings in knowledge base
            cur.execute("SELECT COUNT(*) FROM column_mappings_kb")
            total_column_mappings = cur.fetchone()[0]
            
            return {
                "total_ground_truth_records": total_records,
                "by_vertical": by_vertical,
                "by_document_type": by_document_type,
                "total_column_mappings": total_column_mappings,
            }

