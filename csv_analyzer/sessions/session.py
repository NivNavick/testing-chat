"""
Processing Session for multi-document CSV processing.

Manages multiple documents within a session, tracking:
- Loaded documents and their metadata
- Applied transformations
- Classification results
- Extracted metadata (e.g., date ranges, employee info)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMetadata:
    """Metadata extracted from CSV structure (e.g., multi-row headers)."""
    employee_name: Optional[str] = None
    employee_id: Optional[str] = None
    location: Optional[str] = None
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    department: Optional[str] = None
    company_name: Optional[str] = None
    contract_type: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "employee_name": self.employee_name,
            "employee_id": self.employee_id,
            "location": self.location,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
            "department": self.department,
            "company_name": self.company_name,
            "contract_type": self.contract_type,
            "custom": self.custom,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedMetadata":
        """Create from dictionary."""
        return cls(
            employee_name=data.get("employee_name"),
            employee_id=data.get("employee_id"),
            location=data.get("location"),
            date_range_start=data.get("date_range_start"),
            date_range_end=data.get("date_range_end"),
            department=data.get("department"),
            company_name=data.get("company_name"),
            contract_type=data.get("contract_type"),
            custom=data.get("custom", {}),
        )


@dataclass
class TransformationLog:
    """Log of a transformation applied to a document."""
    transform_type: str
    source_column: Optional[str]
    target_columns: List[str]
    rows_affected: int
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transform_type": self.transform_type,
            "source_column": self.source_column,
            "target_columns": self.target_columns,
            "rows_affected": self.rows_affected,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationLog":
        """Create from dictionary."""
        return cls(
            transform_type=data["transform_type"],
            source_column=data.get("source_column"),
            target_columns=data.get("target_columns", []),
            rows_affected=data.get("rows_affected", 0),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            details=data.get("details", {}),
        )


@dataclass
class DocumentInfo:
    """Information about a document in the session."""
    document_id: str
    source_path: str
    document_type: Optional[str] = None
    classification_confidence: float = 0.0
    column_mappings: Dict[str, str] = field(default_factory=dict)
    row_count: int = 0
    column_count: int = 0
    columns: List[str] = field(default_factory=list)
    extracted_metadata: Optional[ExtractedMetadata] = None
    transformations: List[TransformationLog] = field(default_factory=list)
    loaded_at: datetime = field(default_factory=datetime.now)
    preprocessed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "document_type": self.document_type,
            "classification_confidence": self.classification_confidence,
            "column_mappings": self.column_mappings,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "extracted_metadata": self.extracted_metadata.to_dict() if self.extracted_metadata else None,
            "transformations": [t.to_dict() for t in self.transformations],
            "loaded_at": self.loaded_at.isoformat(),
            "preprocessed": self.preprocessed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentInfo":
        """Create from dictionary."""
        metadata = None
        if data.get("extracted_metadata"):
            metadata = ExtractedMetadata.from_dict(data["extracted_metadata"])
        
        transformations = [
            TransformationLog.from_dict(t) 
            for t in data.get("transformations", [])
        ]
        
        return cls(
            document_id=data["document_id"],
            source_path=data["source_path"],
            document_type=data.get("document_type"),
            classification_confidence=data.get("classification_confidence", 0.0),
            column_mappings=data.get("column_mappings", {}),
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", 0),
            columns=data.get("columns", []),
            extracted_metadata=metadata,
            transformations=transformations,
            loaded_at=datetime.fromisoformat(data["loaded_at"]) if data.get("loaded_at") else datetime.now(),
            preprocessed=data.get("preprocessed", False),
        )


class ProcessingSession:
    """
    Manages multiple documents within a processing session.
    
    A session tracks:
    - All loaded documents and their metadata
    - Applied transformations
    - Classification results
    - Shared context (e.g., date range for relative dates)
    
    Usage:
        session = ProcessingSession()
        
        # Add documents
        session.add_document("shifts.csv", df, metadata)
        session.add_document("payroll.csv", df2, metadata2)
        
        # Set shared context
        session.set_context("date_range", ("2025-12-01", "2025-12-31"))
        
        # Get all documents
        for doc in session.documents:
            print(doc.document_type, doc.row_count)
        
        # Save session
        store = FileBasedSessionStore()
        store.save(session)
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        vertical: str = "medical",
        name: Optional[str] = None,
    ):
        """
        Initialize a processing session.
        
        Args:
            session_id: Unique ID for the session. Generated if not provided.
            vertical: The vertical context (e.g., "medical")
            name: Optional human-readable name for the session
        """
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.vertical = vertical
        self.name = name or f"Session {self.session_id}"
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        self._documents: Dict[str, DocumentInfo] = {}
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._context: Dict[str, Any] = {}
        
        logger.info(f"Created session '{self.name}' (ID: {self.session_id})")
    
    @property
    def documents(self) -> List[DocumentInfo]:
        """Get all documents in the session."""
        return list(self._documents.values())
    
    @property
    def document_types(self) -> List[str]:
        """Get unique document types in the session."""
        return list(set(
            d.document_type for d in self._documents.values() 
            if d.document_type
        ))
    
    def add_document(
        self,
        source_path: str,
        df: pd.DataFrame,
        document_type: Optional[str] = None,
        classification_confidence: float = 0.0,
        column_mappings: Optional[Dict[str, str]] = None,
        extracted_metadata: Optional[ExtractedMetadata] = None,
        preprocessed: bool = False,
    ) -> DocumentInfo:
        """
        Add a document to the session.
        
        Args:
            source_path: Path to the source CSV file
            df: The DataFrame (possibly preprocessed)
            document_type: Classified document type
            classification_confidence: Confidence score
            column_mappings: Column name mappings
            extracted_metadata: Metadata extracted from CSV structure
            preprocessed: Whether the document has been preprocessed
            
        Returns:
            DocumentInfo for the added document
        """
        document_id = str(uuid.uuid4())[:8]
        
        doc_info = DocumentInfo(
            document_id=document_id,
            source_path=str(source_path),
            document_type=document_type,
            classification_confidence=classification_confidence,
            column_mappings=column_mappings or {},
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            extracted_metadata=extracted_metadata,
            preprocessed=preprocessed,
        )
        
        self._documents[document_id] = doc_info
        self._dataframes[document_id] = df
        self.updated_at = datetime.now()
        
        logger.info(
            f"Added document '{Path(source_path).name}' to session "
            f"(type: {document_type}, rows: {len(df)})"
        )
        
        return doc_info
    
    def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Get document info by ID."""
        return self._documents.get(document_id)
    
    def get_dataframe(self, document_id: str) -> Optional[pd.DataFrame]:
        """Get DataFrame by document ID."""
        return self._dataframes.get(document_id)
    
    def get_document_by_path(self, source_path: str) -> Optional[DocumentInfo]:
        """Get document info by source path."""
        path_str = str(source_path)
        for doc in self._documents.values():
            if doc.source_path == path_str or Path(doc.source_path).name == Path(path_str).name:
                return doc
        return None
    
    def get_documents_by_type(self, document_type: str) -> List[DocumentInfo]:
        """Get all documents of a specific type."""
        return [
            doc for doc in self._documents.values()
            if doc.document_type == document_type
        ]
    
    def update_document(
        self,
        document_id: str,
        df: Optional[pd.DataFrame] = None,
        document_type: Optional[str] = None,
        classification_confidence: Optional[float] = None,
        column_mappings: Optional[Dict[str, str]] = None,
    ) -> Optional[DocumentInfo]:
        """
        Update a document in the session.
        
        Args:
            document_id: ID of the document to update
            df: New DataFrame (if changed)
            document_type: New document type
            classification_confidence: New confidence score
            column_mappings: New column mappings
            
        Returns:
            Updated DocumentInfo or None if not found
        """
        doc = self._documents.get(document_id)
        if not doc:
            return None
        
        if df is not None:
            self._dataframes[document_id] = df
            doc.row_count = len(df)
            doc.column_count = len(df.columns)
            doc.columns = list(df.columns)
        
        if document_type is not None:
            doc.document_type = document_type
        
        if classification_confidence is not None:
            doc.classification_confidence = classification_confidence
        
        if column_mappings is not None:
            doc.column_mappings = column_mappings
        
        self.updated_at = datetime.now()
        return doc
    
    def add_transformation_log(
        self,
        document_id: str,
        transform_type: str,
        source_column: Optional[str] = None,
        target_columns: Optional[List[str]] = None,
        rows_affected: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a transformation applied to a document."""
        doc = self._documents.get(document_id)
        if doc:
            log = TransformationLog(
                transform_type=transform_type,
                source_column=source_column,
                target_columns=target_columns or [],
                rows_affected=rows_affected,
                details=details or {},
            )
            doc.transformations.append(log)
            self.updated_at = datetime.now()
    
    def set_context(self, key: str, value: Any) -> None:
        """
        Set a context value for the session.
        
        Context values are shared across all documents in the session.
        Examples: date_range, default_location, etc.
        """
        self._context[key] = value
        self.updated_at = datetime.now()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._context.get(key, default)
    
    @property
    def context(self) -> Dict[str, Any]:
        """Get all context values."""
        return self._context.copy()
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the session."""
        if document_id in self._documents:
            del self._documents[document_id]
            if document_id in self._dataframes:
                del self._dataframes[document_id]
            self.updated_at = datetime.now()
            return True
        return False
    
    def clear(self) -> None:
        """Clear all documents from the session."""
        self._documents.clear()
        self._dataframes.clear()
        self._context.clear()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary for serialization.
        
        Note: DataFrames are not included in serialization.
        Use FileBasedSessionStore for full persistence.
        """
        return {
            "session_id": self.session_id,
            "vertical": self.vertical,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "documents": {
                doc_id: doc.to_dict() 
                for doc_id, doc in self._documents.items()
            },
            "context": self._context,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingSession":
        """
        Create session from dictionary.
        
        Note: DataFrames must be reloaded separately.
        """
        session = cls(
            session_id=data["session_id"],
            vertical=data.get("vertical", "medical"),
            name=data.get("name"),
        )
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session._context = data.get("context", {})
        
        for doc_id, doc_data in data.get("documents", {}).items():
            session._documents[doc_id] = DocumentInfo.from_dict(doc_data)
        
        return session
    
    def __repr__(self) -> str:
        return (
            f"ProcessingSession(id={self.session_id!r}, "
            f"name={self.name!r}, documents={len(self._documents)})"
        )

