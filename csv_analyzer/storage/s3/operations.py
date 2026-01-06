"""
High-level S3 operations for the Analytics Platform.
Handles file organization and naming conventions.
"""

import io
import uuid
from datetime import datetime
from typing import BinaryIO, Optional, List
from dataclasses import dataclass

import pandas as pd

from .client import S3Client, get_s3_client


@dataclass
class S3FileInfo:
    """Information about a file stored in S3."""
    key: str
    session_id: str
    filename: str
    file_type: str
    size_bytes: int = 0
    upload_time: datetime = None


class S3Operations:
    """
    High-level S3 operations for file management.
    
    S3 Structure:
        raw/{session_id}/{file_id}_{original_filename}
        results/{session_id}/{insight_name}_{timestamp}.csv
        reports/{session_id}/summary_report_{timestamp}.pdf
    """
    
    def __init__(self, client: S3Client = None):
        self.client = client or get_s3_client()
    
    def _generate_file_id(self) -> str:
        """Generate a unique file ID."""
        return str(uuid.uuid4())[:8]
    
    def _get_raw_key(self, session_id: str, file_id: str, filename: str) -> str:
        """Generate S3 key for raw uploaded file."""
        return f"raw/{session_id}/{file_id}_{filename}"
    
    def _get_result_key(self, session_id: str, insight_name: str) -> str:
        """Generate S3 key for result file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"results/{session_id}/{insight_name}_{timestamp}.csv"
    
    def _get_report_key(self, session_id: str, report_name: str) -> str:
        """Generate S3 key for report file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"reports/{session_id}/{report_name}_{timestamp}.pdf"
    
    def _detect_content_type(self, filename: str) -> str:
        """Detect content type from filename."""
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        
        content_types = {
            "csv": "text/csv",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xls": "application/vnd.ms-excel",
            "json": "application/json",
            "pdf": "application/pdf",
        }
        
        return content_types.get(ext, "application/octet-stream")
    
    async def upload_raw_file(
        self,
        session_id: str,
        filename: str,
        file_obj: BinaryIO,
        metadata: dict = None
    ) -> S3FileInfo:
        """
        Upload a raw file (CSV/XLSX) to S3.
        
        Args:
            session_id: Session ID
            filename: Original filename
            file_obj: File object to upload
            metadata: Additional metadata
            
        Returns:
            S3FileInfo with upload details
        """
        file_id = self._generate_file_id()
        key = self._get_raw_key(session_id, file_id, filename)
        content_type = self._detect_content_type(filename)
        
        # Get file size
        file_obj.seek(0, 2)  # Seek to end
        size_bytes = file_obj.tell()
        file_obj.seek(0)  # Reset to beginning
        
        # Determine file type
        ext = filename.lower().split(".")[-1] if "." in filename else "unknown"
        
        # Upload with metadata
        upload_metadata = {
            "session_id": session_id,
            "file_id": file_id,
            "original_filename": filename,
            **(metadata or {})
        }
        
        self.client.upload_file(
            file_obj=file_obj,
            key=key,
            content_type=content_type,
            metadata=upload_metadata
        )
        
        return S3FileInfo(
            key=key,
            session_id=session_id,
            filename=filename,
            file_type=ext,
            size_bytes=size_bytes,
            upload_time=datetime.utcnow()
        )
    
    async def upload_result_csv(
        self,
        session_id: str,
        insight_name: str,
        df: pd.DataFrame
    ) -> str:
        """
        Upload a result DataFrame as CSV to S3.
        
        Args:
            session_id: Session ID
            insight_name: Name of the insight
            df: DataFrame to upload
            
        Returns:
            S3 key of uploaded file
        """
        key = self._get_result_key(session_id, insight_name)
        
        # Convert DataFrame to CSV bytes
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        csv_bytes = buffer.getvalue()
        
        self.client.upload_bytes(
            data=csv_bytes,
            key=key,
            content_type="text/csv",
            metadata={
                "session_id": session_id,
                "insight_name": insight_name,
                "row_count": str(len(df)),
                "column_count": str(len(df.columns))
            }
        )
        
        return key
    
    async def download_raw_file(self, key: str) -> bytes:
        """
        Download a raw file from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            File contents as bytes
        """
        return self.client.download_file(key)
    
    async def download_raw_file_to_dataframe(
        self,
        key: str,
        file_type: str = None
    ) -> pd.DataFrame:
        """
        Download a raw file and load it as a DataFrame.
        
        Args:
            key: S3 object key
            file_type: File type (csv, xlsx). Auto-detected if not provided.
            
        Returns:
            DataFrame with file contents
        """
        data = await self.download_raw_file(key)
        buffer = io.BytesIO(data)
        
        # Auto-detect file type from key if not provided
        if file_type is None:
            file_type = key.lower().split(".")[-1]
        
        if file_type == "csv":
            return pd.read_csv(buffer)
        elif file_type in ("xlsx", "xls"):
            return pd.read_excel(buffer)
        else:
            # Try CSV as default
            return pd.read_csv(buffer)
    
    async def generate_download_url(
        self,
        key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate a presigned download URL.
        
        Args:
            key: S3 object key
            expiration: URL expiration in seconds
            
        Returns:
            Presigned URL
        """
        return self.client.generate_presigned_url(
            key=key,
            expiration=expiration,
            operation="get_object"
        )
    
    async def delete_session_files(self, session_id: str) -> int:
        """
        Delete all files for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of files deleted
        """
        deleted = 0
        
        # Delete raw files
        for prefix in ["raw", "results", "reports"]:
            keys = self.client.list_files(f"{prefix}/{session_id}/")
            for key in keys:
                if self.client.delete_file(key):
                    deleted += 1
        
        return deleted
    
    async def list_session_files(
        self,
        session_id: str,
        file_category: str = "raw"
    ) -> List[str]:
        """
        List files for a session.
        
        Args:
            session_id: Session ID
            file_category: Category (raw, results, reports)
            
        Returns:
            List of S3 keys
        """
        return self.client.list_files(f"{file_category}/{session_id}/")


def get_s3_operations() -> S3Operations:
    """Get S3Operations instance."""
    return S3Operations()

