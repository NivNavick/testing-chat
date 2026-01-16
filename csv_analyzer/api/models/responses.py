"""
Pydantic response models for the API.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    type: str = "error"
    status_code: int = 500


class ColumnMappingResponse(BaseModel):
    """Column mapping suggestion."""
    source_column: str
    target_field: Optional[str]
    confidence: float
    required: bool = False
    source_type: Optional[str] = None
    target_type: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    reason: Optional[str] = None


class ClassificationResponse(BaseModel):
    """Classification result response."""
    document_type: Optional[str]
    vertical: Optional[str]
    confidence: float
    final_score: Optional[float] = None
    document_score: Optional[float] = None
    column_score: Optional[float] = None
    coverage_score: Optional[float] = None
    suggested_mappings: Dict[str, ColumnMappingResponse]
    similar_examples: List[Dict[str, Any]]
    column_profiles: List[Dict[str, Any]]
    all_scores: Optional[Dict[str, Dict[str, float]]] = None


class BatchClassificationResponse(BaseModel):
    """Batch classification results."""
    results: List[ClassificationResponse]
    total_files: int
    successful: int
    failed: int
    errors: List[Dict[str, str]] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    """Document information response."""
    id: str
    session_id: str
    document_id: str
    document_type: Optional[str]
    filename: str
    s3_uri: Optional[str]
    row_count: int
    column_count: int
    uploaded_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    """Session information response."""
    id: str
    name: str
    vertical: str
    created_at: datetime
    updated_at: datetime
    s3_bucket: Optional[str] = None
    s3_prefix: Optional[str] = None
    document_count: int = 0
    documents: List[DocumentResponse] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class SessionListResponse(BaseModel):
    """List of sessions."""
    sessions: List[SessionResponse]
    total: int


class WorkflowBlockResponse(BaseModel):
    """Workflow block information."""
    id: str
    handler: str
    description: Optional[str] = None
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    """Workflow definition response."""
    name: str
    description: str
    version: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    blocks: List[WorkflowBlockResponse] = Field(default_factory=list)
    pipeline: List[str] = Field(default_factory=list)


class WorkflowListResponse(BaseModel):
    """List of workflows."""
    workflows: List[WorkflowResponse]
    total: int


class WorkflowExecutionResponse(BaseModel):
    """Workflow execution result."""
    execution_id: str
    workflow_name: str
    session_id: Optional[str] = None
    status: str  # pending, running, completed, failed
    parameters: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Dict[str, str]] = Field(default_factory=dict)  # block_id -> output_name -> s3_uri
    error: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None


class WorkflowExecutionListResponse(BaseModel):
    """List of workflow executions."""
    executions: List[WorkflowExecutionResponse]
    total: int


class VerticalResponse(BaseModel):
    """Vertical information."""
    name: str
    display_name: str
    description: Optional[str] = None
    document_types: List[str] = Field(default_factory=list)


class VerticalListResponse(BaseModel):
    """List of verticals."""
    verticals: List[VerticalResponse]
    total: int


class DocumentTypeResponse(BaseModel):
    """Document type information."""
    name: str
    vertical: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    field_count: int = 0


class DocumentTypeListResponse(BaseModel):
    """List of document types."""
    document_types: List[DocumentTypeResponse]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: str = "unknown"
    s3: str = "unknown"
    version: str

