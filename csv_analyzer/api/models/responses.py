"""
Pydantic models for API responses.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    """Response for session data."""
    id: str
    vertical: str
    mode: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


class FileResponse(BaseModel):
    """Response for uploaded file data."""
    id: str
    session_id: str
    filename: str
    file_type: str
    s3_key: str
    file_size_bytes: int
    original_structure: Optional[str] = None
    processing_status: str
    error_message: Optional[str] = None
    uploaded_at: datetime


class TableResponse(BaseModel):
    """Response for extracted table data."""
    id: str
    file_id: str
    session_id: str
    table_name: str
    location: Optional[str] = None
    row_count: int
    column_count: int
    columns: Optional[List[Dict[str, Any]]] = None
    structure_type: Optional[str] = None
    transforms_applied: Optional[List[str]] = None
    created_at: datetime


class ClassificationResponse(BaseModel):
    """Response for table classification."""
    id: str
    table_id: str
    document_type: str
    confidence: float
    method: str
    column_mappings: Optional[Dict[str, Any]] = None
    unmapped_columns: Optional[List[str]] = None
    llm_reasoning: Optional[str] = None
    created_at: datetime


class RelationshipResponse(BaseModel):
    """Response for table relationship."""
    id: str
    session_id: str
    from_table_id: str
    from_column: str
    to_table_id: str
    to_column: str
    relationship_type: str
    confidence: float
    transform_sql: Optional[str] = None
    verified: bool
    created_at: datetime


class RuleResponse(BaseModel):
    """Response for business rule."""
    id: str
    session_id: str
    rule_type: str
    description: str
    applies_to_tables: Optional[List[str]] = None
    condition_sql: Optional[str] = None
    formula_sql: Optional[str] = None
    confidence: float
    source: str
    evidence: Optional[str] = None
    active: bool
    created_at: datetime


class InsightResultResponse(BaseModel):
    """Response for insight result."""
    id: str
    session_id: str
    insight_name: str
    insight_type: str
    executed_sql: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    row_count: int
    column_names: Optional[List[str]] = None
    summary_stats: Optional[Dict[str, Any]] = None
    s3_result_key: Optional[str] = None
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None
    executed_at: datetime


class InsightResultDataResponse(BaseModel):
    """Response for insight result with data."""
    result: InsightResultResponse
    data: List[Dict[str, Any]]
    download_url: Optional[str] = None


class AvailableInsightResponse(BaseModel):
    """Response for available insight."""
    name: str
    description: str
    requires_tables: List[str]
    parameters: Optional[Dict[str, Any]] = None
    is_available: bool = False
    missing_tables: Optional[List[str]] = None


class UploadResponse(BaseModel):
    """Response for file upload."""
    file: FileResponse
    tables: List[TableResponse]
    insights_available: List[str]


class SessionDetailResponse(BaseModel):
    """Detailed session response with all related data."""
    session: SessionResponse
    files: List[FileResponse]
    tables: List[TableResponse]
    relationships: List[RelationshipResponse] = []
    rules: List[RuleResponse] = []
    insights_available: List[str] = []
    results: List[InsightResultResponse] = []


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    code: Optional[str] = None


class DownloadUrlResponse(BaseModel):
    """Response with presigned download URL."""
    url: str
    expires_in: int = Field(default=3600, description="URL expiration in seconds")

