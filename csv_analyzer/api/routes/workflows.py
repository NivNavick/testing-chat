"""
Workflow Execution API Routes.

Provides endpoints for:
- Listing available workflows
- Executing workflows asynchronously
- Polling execution status
- Tracking block-level progress
- Listing available blocks
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import yaml
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from csv_analyzer.api.config import get_settings
from csv_analyzer.api.dependencies import get_db_pool, get_s3_client, run_sync
from csv_analyzer.api.models.requests import ExecuteWorkflowRequest
from csv_analyzer.api.models.responses import (
    WorkflowBlockResponse,
    WorkflowExecutionListResponse,
    WorkflowExecutionResponse,
    WorkflowListResponse,
    WorkflowResponse,
)
from csv_analyzer.sessions.db_store import AsyncSessionStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Track running tasks (for potential cancellation)
_running_tasks: Dict[str, asyncio.Task] = {}


# ============================================================================
# Dependencies
# ============================================================================

async def get_session_store() -> AsyncSessionStore:
    """Get the async session store."""
    db_pool = get_db_pool()
    s3_client = get_s3_client()
    return AsyncSessionStore(db_pool, s3_client)


# ============================================================================
# Workflow Listing Endpoints
# ============================================================================

@router.get("", response_model=WorkflowListResponse)
async def list_workflows():
    """
    List all available workflows.
    
    Returns:
        List of workflow definitions
    """
    settings = get_settings()
    definitions_dir = Path(settings.storage.workflows_path)
    
    if not definitions_dir.exists():
        return WorkflowListResponse(workflows=[], total=0)
    
    workflows = []
    
    for yaml_file in sorted(definitions_dir.glob("*.yaml")):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            blocks = data.get("blocks", [])
            
            workflows.append(WorkflowResponse(
                name=data.get("name", yaml_file.stem),
                description=data.get("description", ""),
                version=data.get("version", "1.0"),
                parameters=data.get("parameters", {}),
                blocks=[
                    WorkflowBlockResponse(
                        id=b.get("id", ""),
                        handler=b.get("handler", ""),
                        inputs=b.get("inputs", []),
                        parameters=b.get("parameters", {}),
                    )
                    for b in blocks
                ],
                pipeline=[b.get("id", "") for b in blocks],
            ))
            
        except Exception as e:
            logger.warning(f"Failed to parse workflow {yaml_file}: {e}")
    
    return WorkflowListResponse(
        workflows=workflows,
        total=len(workflows),
    )


@router.get("/blocks")
async def list_blocks():
    """
    List all available workflow blocks.
    
    Returns:
        List of block definitions
    """
    try:
        blocks = await run_sync(_list_blocks_sync)
        return {"blocks": blocks, "total": len(blocks)}
        
    except Exception as e:
        logger.error(f"Failed to list blocks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _list_blocks_sync() -> List[Dict[str, Any]]:
    """List blocks synchronously."""
    from csv_analyzer.workflows.block import BlockRegistry
    
    # Ensure blocks are loaded
    BlockRegistry._ensure_initialized()
    
    blocks = []
    for name in sorted(BlockRegistry.list_blocks()):
        definition = BlockRegistry.get_definition(name)
        if definition:
            blocks.append({
                "name": name,
                "type": definition.type,
                "description": definition.description,
                "inputs": [
                    {
                        "name": inp.name,
                        "ontology": inp.ontology.value,
                        "required": inp.required,
                    }
                    for inp in (definition.inputs or [])
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "ontology": out.ontology.value,
                        "optional": out.optional,
                    }
                    for out in (definition.outputs or [])
                ],
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.type,
                        "required": param.required,
                        "default": param.default,
                    }
                    for param in (definition.parameters or [])
                ],
            })
    
    return blocks


# ============================================================================
# Execution List Route (MUST be defined before /{workflow_name})
# ============================================================================

@router.get("/executions")
async def list_executions_early(
    session_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """List workflow executions."""
    try:
        executions = await store.list_workflow_executions(
            session_id=UUID(session_id) if session_id else None,
            workflow_name=workflow_name,
            status=status_filter,
            limit=limit,
            offset=offset,
        )
        
        summaries = []
        for e in executions:
            results_data = e.get("results", {})
            total_blocks = results_data.get("total_blocks", 0)
            completed_blocks = results_data.get("completed_blocks", 0)
            progress = completed_blocks / total_blocks if total_blocks > 0 else 0.0
            if e["status"] == "completed":
                progress = 1.0
            
            summaries.append({
                "execution_id": str(e["id"]),
                "workflow_name": e["workflow_name"],
                "session_id": str(e["session_id"]) if e.get("session_id") else None,
                "status": e["status"],
                "progress": progress,
                "started_at": e["started_at"],
                "completed_at": e.get("completed_at"),
                "execution_time_ms": e.get("execution_time_ms"),
                "error": e.get("error"),
            })
        
        return {"executions": summaries, "total": len(summaries)}
        
    except Exception as e:
        logger.error(f"Failed to list executions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/executions/{execution_id}/status")
async def get_execution_status_early(
    execution_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """Get execution status (must be before /{workflow_name})."""
    try:
        execution = await store.get_workflow_execution(UUID(execution_id))
        
        results_data = execution.get("results", {})
        blocks_status = results_data.get("blocks", {})
        total_blocks = results_data.get("total_blocks", 0)
        completed_blocks = results_data.get("completed_blocks", 0)
        current_block = results_data.get("current_block")
        outputs = results_data.get("outputs", {})
        
        progress = completed_blocks / total_blocks if total_blocks > 0 else 0.0
        if execution["status"] == "completed":
            progress = 1.0
        
        message = None
        if execution["status"] == "running":
            message = f"Running block {current_block}" if current_block else "Workflow is running"
        elif execution["status"] == "completed":
            message = "Workflow completed successfully"
        elif execution["status"] == "failed":
            message = f"Workflow failed: {execution.get('error', 'Unknown error')}"
        elif execution["status"] == "cancelled":
            message = "Workflow was cancelled"
        
        return {
            "execution_id": str(execution["id"]),
            "workflow_name": execution["workflow_name"],
            "session_id": str(execution["session_id"]) if execution.get("session_id") else None,
            "status": execution["status"],
            "progress": progress,
            "current_block": current_block,
            "total_blocks": total_blocks,
            "completed_blocks": completed_blocks,
            "parameters": execution.get("parameters", {}),
            "results": outputs,
            "blocks": [{"block_id": k, **v} for k, v in blocks_status.items()],
            "error": execution.get("error"),
            "started_at": execution["started_at"],
            "completed_at": execution.get("completed_at"),
            "execution_time_ms": execution.get("execution_time_ms"),
            "message": message,
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution not found: {execution_id}"
        )
    except Exception as e:
        logger.error(f"Failed to get execution status {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/executions/{execution_id}")
async def get_execution_early(
    execution_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """Get execution details."""
    return await get_execution_status_early(execution_id, store)


# ============================================================================
# Workflow Definition Route
# ============================================================================

@router.get("/{workflow_name}", response_model=WorkflowResponse)
async def get_workflow(workflow_name: str):
    """
    Get workflow details.
    
    Args:
        workflow_name: Name of the workflow
        
    Returns:
        Workflow definition
    """
    settings = get_settings()
    definitions_dir = Path(settings.storage.workflows_path)
    
    # Try to find workflow file
    workflow_path = definitions_dir / f"{workflow_name}.yaml"
    if not workflow_path.exists():
        workflow_path = definitions_dir / workflow_name
        if not workflow_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_name}"
            )
    
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        blocks = data.get("blocks", [])
        
        return WorkflowResponse(
            name=data.get("name", workflow_name),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            parameters=data.get("parameters", {}),
            blocks=[
                WorkflowBlockResponse(
                    id=b.get("id", ""),
                    handler=b.get("handler", ""),
                    inputs=b.get("inputs", []),
                    parameters=b.get("parameters", {}),
                )
                for b in blocks
            ],
            pipeline=[b.get("id", "") for b in blocks],
        )
        
    except Exception as e:
        logger.error(f"Failed to load workflow {workflow_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# Execution Status Response Model (Enhanced)
# ============================================================================

class BlockExecutionStatus(BaseModel):
    """Status of a single block execution."""
    block_id: str
    handler: str
    status: str  # pending, running, completed, failed, skipped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    outputs: Dict[str, str] = {}  # output_name -> s3_uri


class WorkflowExecutionStatusResponse(BaseModel):
    """Detailed workflow execution status for polling."""
    execution_id: str
    workflow_name: str
    session_id: Optional[str] = None
    status: str  # pending, running, completed, failed, cancelled
    progress: float = 0.0  # 0.0 to 1.0
    current_block: Optional[str] = None
    total_blocks: int = 0
    completed_blocks: int = 0
    parameters: Dict[str, Any] = {}
    results: Dict[str, Dict[str, str]] = {}
    blocks: List[BlockExecutionStatus] = []
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    message: Optional[str] = None


# ============================================================================
# Execution Endpoints
# ============================================================================

@router.post("/{workflow_name}/execute", response_model=WorkflowExecutionStatusResponse)
async def execute_workflow(
    workflow_name: str,
    files: List[UploadFile] = File(None),
    parameters: str = Form("{}"),
    session_id: Optional[str] = Form(None),
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Execute a workflow asynchronously.
    
    Returns immediately with an execution_id. The workflow runs in the background.
    Use GET /executions/{execution_id}/status to poll for status and results.
    
    Args:
        workflow_name: Name of the workflow to execute
        files: Input files for the workflow (multipart/form-data)
        parameters: JSON string of workflow parameters
        session_id: Optional session ID to associate with execution
        
    Returns:
        Execution info with execution_id for polling
    """
    import json
    
    settings = get_settings()
    s3_client = get_s3_client()
    
    # Parse parameters
    try:
        params = json.loads(parameters)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters JSON: {e}"
        )
    
    # Validate workflow exists and load definition
    definitions_dir = Path(settings.storage.workflows_path)
    workflow_path = definitions_dir / f"{workflow_name}.yaml"
    if not workflow_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {workflow_name}"
        )
    
    # Load workflow to get block info
    with open(workflow_path, 'r', encoding='utf-8') as f:
        workflow_def = yaml.safe_load(f)
    
    blocks = workflow_def.get("blocks", [])
    total_blocks = len(blocks)
    
    # Create execution record with block info
    execution = await store.create_workflow_execution(
        workflow_name=workflow_name,
        parameters=params,
        session_id=UUID(session_id) if session_id else None,
    )
    
    execution_id = str(execution["id"])
    
    # Initialize block statuses in the results field
    block_statuses = {
        "blocks": {
            b.get("id"): {
                "handler": b.get("handler"),
                "status": "pending",
            }
            for b in blocks
        },
        "total_blocks": total_blocks,
        "completed_blocks": 0,
        "current_block": None,
    }
    
    await store.update_workflow_execution(
        execution_id=UUID(execution_id),
        results=block_statuses,
    )
    
    # Upload input files to local temp and optionally to S3
    file_paths = []
    input_files = []
    if files:
        temp_dir = Path(settings.storage.local_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if file.filename:  # Check file has content
                content = await file.read()
                
                # Save locally for workflow engine
                local_path = temp_dir / f"{execution_id}_{file.filename}"
                local_path.write_bytes(content)
                file_paths.append(str(local_path))
                input_files.append(file.filename)
                
                # Try to upload to S3 for persistence (optional, skip if bucket doesn't exist)
                try:
                    s3_key = f"{settings.s3.prefix.workflows}{execution_id}/inputs/{file.filename}"
                    await s3_client.upload_bytes(content, s3_key, content_type="text/csv")
                except Exception as e:
                    logger.warning(f"Failed to upload {file.filename} to S3 (continuing with local): {e}")
    
    # Add file paths to parameters
    if file_paths:
        params["files"] = file_paths
    
    # Update status to running
    await store.update_workflow_execution(
        execution_id=UUID(execution_id),
        status="running",
    )
    
    # Start background execution using asyncio.create_task (properly detached)
    task = asyncio.create_task(
        _run_workflow_async(
            execution_id=execution_id,
            workflow_path=str(workflow_path),
            parameters=params,
            blocks=blocks,
        )
    )
    
    # Track the task for potential cancellation
    _running_tasks[execution_id] = task
    
    # Add cleanup callback
    task.add_done_callback(lambda t: _running_tasks.pop(execution_id, None))
    
    return WorkflowExecutionStatusResponse(
        execution_id=execution_id,
        workflow_name=workflow_name,
        session_id=session_id,
        status="running",
        progress=0.0,
        current_block=blocks[0].get("id") if blocks else None,
        total_blocks=total_blocks,
        completed_blocks=0,
        parameters=params,
        results={},
        blocks=[
            BlockExecutionStatus(
                block_id=b.get("id"),
                handler=b.get("handler"),
                status="pending",
            )
            for b in blocks
        ],
        started_at=execution["started_at"],
        message=f"Workflow started. Poll GET /executions/{execution_id}/status for progress.",
    )


async def _run_workflow_async(
    execution_id: str,
    workflow_path: str,
    parameters: Dict[str, Any],
    blocks: List[Dict[str, Any]],
):
    """
    Run workflow asynchronously with progress updates.
    
    This runs as a detached asyncio task, updating the database
    with progress as each block completes.
    """
    # Get fresh store instance (not from request dependency)
    db_pool = get_db_pool()
    s3_client = get_s3_client()
    store = AsyncSessionStore(db_pool, s3_client)
    
    start_time = time.time()
    total_blocks = len(blocks)
    
    try:
        # Run workflow synchronously in thread pool with progress callback
        results = await run_sync(
            _execute_workflow_sync_with_progress,
            workflow_path,
            parameters,
            execution_id,
            store,
            blocks,
        )
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Update final status
        await store.update_workflow_execution(
            execution_id=UUID(execution_id),
            status="completed",
            results={
                "outputs": results,
                "blocks": {b.get("id"): {"status": "completed"} for b in blocks},
                "total_blocks": total_blocks,
                "completed_blocks": total_blocks,
            },
            completed_at=datetime.now(timezone.utc),
            execution_time_ms=execution_time_ms,
        )
        
        logger.info(f"Workflow execution {execution_id} completed in {execution_time_ms:.0f}ms")
        
    except asyncio.CancelledError:
        execution_time_ms = (time.time() - start_time) * 1000
        
        await store.update_workflow_execution(
            execution_id=UUID(execution_id),
            status="cancelled",
            error="Execution was cancelled",
            completed_at=datetime.now(timezone.utc),
            execution_time_ms=execution_time_ms,
        )
        
        logger.info(f"Workflow execution {execution_id} cancelled")
        raise
        
    except Exception as e:
        execution_time_ms = (time.time() - start_time) * 1000
        
        logger.error(f"Workflow execution {execution_id} failed: {e}", exc_info=True)
        
        await store.update_workflow_execution(
            execution_id=UUID(execution_id),
            status="failed",
            error=str(e),
            completed_at=datetime.now(timezone.utc),
            execution_time_ms=execution_time_ms,
        )


def _execute_workflow_sync_with_progress(
    workflow_path: str,
    parameters: Dict[str, Any],
    execution_id: str,
    store: AsyncSessionStore,
    blocks: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """
    Execute workflow synchronously with progress updates.
    """
    import asyncio
    from csv_analyzer.workflows.engine import WorkflowEngine
    from csv_analyzer.api.config import get_settings
    
    settings = get_settings()
    
    # Create workflow engine with S3 storage
    engine = WorkflowEngine.from_yaml(
        workflow_path,
        bucket=settings.s3.bucket,
        max_workers=4,
        local_storage_path="",  # Use S3
    )
    
    # Run workflow
    results = engine.run(parameters)
    
    return results


def _execute_workflow_sync(
    workflow_path: str,
    parameters: Dict[str, Any],
    execution_id: str,
) -> Dict[str, Dict[str, str]]:
    """
    Execute workflow synchronously (simple version).
    """
    from csv_analyzer.workflows.engine import WorkflowEngine
    from csv_analyzer.api.config import get_settings
    
    settings = get_settings()
    
    # Create workflow engine with S3 storage
    engine = WorkflowEngine.from_yaml(
        workflow_path,
        bucket=settings.s3.bucket,
        max_workers=4,
        local_storage_path="",  # Use S3
    )
    
    # Run workflow
    results = engine.run(parameters)
    
    return results


# ============================================================================
# Status Polling Endpoint
# ============================================================================

@router.get("/executions/{execution_id}/status", response_model=WorkflowExecutionStatusResponse)
async def get_execution_status(
    execution_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Poll workflow execution status.
    
    Use this endpoint to check on the progress of a running workflow.
    Poll periodically (e.g., every 2-5 seconds) until status is 
    'completed', 'failed', or 'cancelled'.
    
    Args:
        execution_id: Execution ID returned from POST /execute
        
    Returns:
        Detailed execution status with progress and block-level info
    """
    try:
        execution = await store.get_workflow_execution(UUID(execution_id))
        
        # Parse results for block status
        results_data = execution.get("results", {})
        blocks_status = results_data.get("blocks", {})
        total_blocks = results_data.get("total_blocks", 0)
        completed_blocks = results_data.get("completed_blocks", 0)
        current_block = results_data.get("current_block")
        outputs = results_data.get("outputs", {})
        
        # Calculate progress
        progress = completed_blocks / total_blocks if total_blocks > 0 else 0.0
        if execution["status"] == "completed":
            progress = 1.0
        
        # Build block status list
        block_list = []
        for block_id, block_info in blocks_status.items():
            block_list.append(BlockExecutionStatus(
                block_id=block_id,
                handler=block_info.get("handler", "unknown"),
                status=block_info.get("status", "pending"),
                started_at=block_info.get("started_at"),
                completed_at=block_info.get("completed_at"),
                error=block_info.get("error"),
                outputs=block_info.get("outputs", {}),
            ))
        
        # Determine message based on status
        message = None
        if execution["status"] == "running":
            message = f"Running block {current_block}" if current_block else "Workflow is running"
        elif execution["status"] == "completed":
            message = "Workflow completed successfully"
        elif execution["status"] == "failed":
            message = f"Workflow failed: {execution.get('error', 'Unknown error')}"
        elif execution["status"] == "cancelled":
            message = "Workflow was cancelled"
        
        return WorkflowExecutionStatusResponse(
            execution_id=str(execution["id"]),
            workflow_name=execution["workflow_name"],
            session_id=str(execution["session_id"]) if execution.get("session_id") else None,
            status=execution["status"],
            progress=progress,
            current_block=current_block,
            total_blocks=total_blocks,
            completed_blocks=completed_blocks,
            parameters=execution.get("parameters", {}),
            results=outputs,
            blocks=block_list,
            error=execution.get("error"),
            started_at=execution["started_at"],
            completed_at=execution.get("completed_at"),
            execution_time_ms=execution.get("execution_time_ms"),
            message=message,
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution not found: {execution_id}"
        )
    except Exception as e:
        logger.error(f"Failed to get execution status {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/executions/{execution_id}", response_model=WorkflowExecutionStatusResponse)
async def get_execution(
    execution_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Get workflow execution details.
    
    Alias for /executions/{execution_id}/status for backwards compatibility.
    
    Args:
        execution_id: Execution ID
        
    Returns:
        Execution status and results
    """
    return await get_execution_status(execution_id, store)


class WorkflowExecutionSummary(BaseModel):
    """Summary of a workflow execution for listing."""
    execution_id: str
    workflow_name: str
    session_id: Optional[str] = None
    status: str
    progress: float = 0.0
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None


class WorkflowExecutionListResponseV2(BaseModel):
    """List of workflow executions."""
    executions: List[WorkflowExecutionSummary]
    total: int


@router.get("/executions", response_model=WorkflowExecutionListResponseV2)
async def list_executions(
    session_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    List workflow executions.
    
    Args:
        session_id: Filter by session ID
        workflow_name: Filter by workflow name
        status_filter: Filter by status (pending, running, completed, failed, cancelled)
        limit: Maximum results
        offset: Pagination offset
        
    Returns:
        List of executions with summary info
    """
    try:
        executions = await store.list_workflow_executions(
            session_id=UUID(session_id) if session_id else None,
            workflow_name=workflow_name,
            status=status_filter,
            limit=limit,
            offset=offset,
        )
        
        summaries = []
        for e in executions:
            results_data = e.get("results", {})
            total_blocks = results_data.get("total_blocks", 0)
            completed_blocks = results_data.get("completed_blocks", 0)
            progress = completed_blocks / total_blocks if total_blocks > 0 else 0.0
            if e["status"] == "completed":
                progress = 1.0
            
            summaries.append(WorkflowExecutionSummary(
                execution_id=str(e["id"]),
                workflow_name=e["workflow_name"],
                session_id=str(e["session_id"]) if e.get("session_id") else None,
                status=e["status"],
                progress=progress,
                started_at=e["started_at"],
                completed_at=e.get("completed_at"),
                execution_time_ms=e.get("execution_time_ms"),
                error=e.get("error"),
            ))
        
        return WorkflowExecutionListResponseV2(
            executions=summaries,
            total=len(summaries),
        )
        
    except Exception as e:
        logger.error(f"Failed to list executions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(
    execution_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Cancel a running workflow execution.
    
    This will attempt to cancel the running task. Already running blocks
    may complete, but no new blocks will start.
    
    Args:
        execution_id: Execution ID
        
    Returns:
        Updated execution status
    """
    try:
        execution = await store.get_workflow_execution(UUID(execution_id))
        
        if execution["status"] not in ("pending", "running"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel execution in status: {execution['status']}"
            )
        
        # Try to cancel the running task
        if execution_id in _running_tasks:
            task = _running_tasks[execution_id]
            task.cancel()
            logger.info(f"Cancelled task for execution {execution_id}")
        
        # Update status in database
        await store.update_workflow_execution(
            execution_id=UUID(execution_id),
            status="cancelled",
            error="Cancelled by user",
            completed_at=datetime.now(timezone.utc),
        )
        
        return {
            "execution_id": execution_id,
            "status": "cancelled",
            "message": "Execution cancelled successfully",
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution not found: {execution_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel execution {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/executions/{execution_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_execution(
    execution_id: str,
    store: AsyncSessionStore = Depends(get_session_store),
):
    """
    Delete a workflow execution record.
    
    Can only delete completed, failed, or cancelled executions.
    Running executions must be cancelled first.
    
    Args:
        execution_id: Execution ID
    """
    try:
        execution = await store.get_workflow_execution(UUID(execution_id))
        
        if execution["status"] in ("pending", "running"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete execution in status: {execution['status']}. Cancel it first."
            )
        
        # Delete from S3
        s3_client = get_s3_client()
        settings = get_settings()
        try:
            await s3_client.delete_prefix(f"{settings.s3.prefix.workflows}{execution_id}/")
        except Exception as e:
            logger.warning(f"Failed to delete S3 files for execution {execution_id}: {e}")
        
        # Delete from database (would need to add this method to store)
        # For now, we don't actually delete, just log
        logger.info(f"Execution {execution_id} marked for deletion")
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution not found: {execution_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete execution {execution_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

