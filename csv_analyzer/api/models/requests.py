"""
Pydantic request models for the API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request to create a new processing session."""
    name: Optional[str] = Field(None, description="Human-readable session name")
    vertical: str = Field("medical", description="Vertical domain (e.g., 'medical')")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class UploadDocumentRequest(BaseModel):
    """Request metadata for document upload."""
    document_type: Optional[str] = Field(None, description="Document type if known")
    filename: Optional[str] = Field(None, description="Original filename")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ClassifyRequest(BaseModel):
    """Request to classify a CSV file."""
    vertical: Optional[str] = Field(None, description="Vertical domain filter")
    k: int = Field(5, ge=1, le=20, description="Number of similar examples to retrieve")
    hybrid: bool = Field(True, description="Use hybrid scoring (document + column)")
    use_dspy: bool = Field(False, description="Use DSPy for LLM-based column classification")
    force_reindex: bool = Field(False, description="Force reindex of schema embeddings")


class ExecuteWorkflowRequest(BaseModel):
    """Request to execute a workflow."""
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Workflow parameters")
    session_id: Optional[str] = Field(None, description="Session ID to use for workflow")
    local_storage: bool = Field(False, description="Use local storage instead of S3")
    max_workers: int = Field(4, ge=1, le=16, description="Maximum parallel workers")


class BatchClassifyRequest(BaseModel):
    """Request to classify multiple CSV files."""
    vertical: Optional[str] = Field(None, description="Vertical domain filter")
    k: int = Field(5, ge=1, le=20, description="Number of similar examples per file")
    hybrid: bool = Field(True, description="Use hybrid scoring")


class UpdateSessionRequest(BaseModel):
    """Request to update a session."""
    name: Optional[str] = Field(None, description="New session name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")

