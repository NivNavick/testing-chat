"""
Repository for uploaded file operations.
"""

from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class FileRepository:
    """Repository for uploaded file CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        session_id: str,
        filename: str,
        file_type: str,
        s3_key: str,
        file_size_bytes: int = 0,
        original_structure: str = None
    ) -> dict:
        """
        Create a new uploaded file record.
        
        Args:
            session_id: Session UUID
            filename: Original filename
            file_type: File type (csv, xlsx, etc.)
            s3_key: S3 object key
            file_size_bytes: File size in bytes
            original_structure: Detected structure type
            
        Returns:
            Created file dict
        """
        result = await self.session.execute(
            text("""
                INSERT INTO uploaded_files 
                    (session_id, filename, file_type, s3_key, file_size_bytes, 
                     original_structure, processing_status)
                VALUES 
                    (:session_id, :filename, :file_type, :s3_key, :file_size_bytes,
                     :original_structure, 'pending')
                RETURNING id, session_id, filename, file_type, s3_key, 
                          file_size_bytes, original_structure, processing_status, 
                          uploaded_at
            """),
            {
                "session_id": session_id,
                "filename": filename,
                "file_type": file_type,
                "s3_key": s3_key,
                "file_size_bytes": file_size_bytes,
                "original_structure": original_structure
            }
        )
        row = result.fetchone()
        return self._row_to_dict(row)
    
    async def get_by_id(self, file_id: str) -> Optional[dict]:
        """
        Get a file by ID.
        
        Args:
            file_id: File UUID
            
        Returns:
            File dict or None if not found
        """
        result = await self.session.execute(
            text("""
                SELECT id, session_id, filename, file_type, s3_key, 
                       file_size_bytes, original_structure, processing_status,
                       error_message, uploaded_at
                FROM uploaded_files
                WHERE id = :file_id
            """),
            {"file_id": file_id}
        )
        row = result.fetchone()
        return self._row_to_dict(row) if row else None
    
    async def list_by_session(self, session_id: str) -> List[dict]:
        """
        List files for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            List of file dicts
        """
        result = await self.session.execute(
            text("""
                SELECT id, session_id, filename, file_type, s3_key, 
                       file_size_bytes, original_structure, processing_status,
                       error_message, uploaded_at
                FROM uploaded_files
                WHERE session_id = :session_id
                ORDER BY uploaded_at ASC
            """),
            {"session_id": session_id}
        )
        return [self._row_to_dict(row) for row in result.fetchall()]
    
    async def update_status(
        self,
        file_id: str,
        status: str,
        error_message: str = None
    ) -> bool:
        """
        Update file processing status.
        
        Args:
            file_id: File UUID
            status: New status (pending, processing, completed, failed)
            error_message: Error message if failed
            
        Returns:
            True if updated, False if not found
        """
        result = await self.session.execute(
            text("""
                UPDATE uploaded_files
                SET processing_status = :status, error_message = :error_message
                WHERE id = :file_id
                RETURNING id
            """),
            {"file_id": file_id, "status": status, "error_message": error_message}
        )
        return result.fetchone() is not None
    
    async def update_structure(
        self,
        file_id: str,
        original_structure: str
    ) -> bool:
        """
        Update detected file structure.
        
        Args:
            file_id: File UUID
            original_structure: Detected structure type
            
        Returns:
            True if updated, False if not found
        """
        result = await self.session.execute(
            text("""
                UPDATE uploaded_files
                SET original_structure = :original_structure
                WHERE id = :file_id
                RETURNING id
            """),
            {"file_id": file_id, "original_structure": original_structure}
        )
        return result.fetchone() is not None
    
    async def delete(self, file_id: str) -> bool:
        """
        Delete a file record.
        
        Args:
            file_id: File UUID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            text("""
                DELETE FROM uploaded_files
                WHERE id = :file_id
                RETURNING id
            """),
            {"file_id": file_id}
        )
        return result.fetchone() is not None
    
    def _row_to_dict(self, row) -> dict:
        """Convert a database row to a dict."""
        if row is None:
            return None
        return {
            "id": str(row.id),
            "session_id": str(row.session_id),
            "filename": row.filename,
            "file_type": row.file_type,
            "s3_key": row.s3_key,
            "file_size_bytes": row.file_size_bytes,
            "original_structure": row.original_structure,
            "processing_status": row.processing_status,
            "error_message": row.error_message,
            "uploaded_at": row.uploaded_at
        }

