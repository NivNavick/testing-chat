"""
Repository for extracted table operations.
"""

import json
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class TableRepository:
    """Repository for extracted table CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        file_id: str,
        session_id: str,
        table_name: str,
        location: str = None,
        row_count: int = 0,
        column_count: int = 0,
        columns: list = None,
        structure_type: str = None,
        transforms_applied: list = None
    ) -> dict:
        """
        Create a new extracted table record.
        
        Args:
            file_id: File UUID
            session_id: Session UUID
            table_name: Table name
            location: Location in file (e.g., "Sheet1:A1:M50")
            row_count: Number of rows
            column_count: Number of columns
            columns: Column info as list of dicts
            structure_type: Structure type (STANDARD, PIVOT, etc.)
            transforms_applied: List of transforms applied
            
        Returns:
            Created table dict
        """
        result = await self.session.execute(
            text("""
                INSERT INTO extracted_tables 
                    (file_id, session_id, table_name, location, row_count, 
                     column_count, columns, structure_type, transforms_applied)
                VALUES 
                    (:file_id, :session_id, :table_name, :location, :row_count,
                     :column_count, :columns::jsonb, :structure_type, :transforms_applied::jsonb)
                RETURNING id, file_id, session_id, table_name, location, 
                          row_count, column_count, columns, structure_type, 
                          transforms_applied, created_at
            """),
            {
                "file_id": file_id,
                "session_id": session_id,
                "table_name": table_name,
                "location": location,
                "row_count": row_count,
                "column_count": column_count,
                "columns": json.dumps(columns) if columns else None,
                "structure_type": structure_type,
                "transforms_applied": json.dumps(transforms_applied) if transforms_applied else None
            }
        )
        row = result.fetchone()
        return self._row_to_dict(row)
    
    async def get_by_id(self, table_id: str) -> Optional[dict]:
        """
        Get a table by ID.
        
        Args:
            table_id: Table UUID
            
        Returns:
            Table dict or None if not found
        """
        result = await self.session.execute(
            text("""
                SELECT id, file_id, session_id, table_name, location, 
                       row_count, column_count, columns, structure_type, 
                       transforms_applied, created_at
                FROM extracted_tables
                WHERE id = :table_id
            """),
            {"table_id": table_id}
        )
        row = result.fetchone()
        return self._row_to_dict(row) if row else None
    
    async def list_by_session(self, session_id: str) -> List[dict]:
        """
        List tables for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            List of table dicts
        """
        result = await self.session.execute(
            text("""
                SELECT id, file_id, session_id, table_name, location, 
                       row_count, column_count, columns, structure_type, 
                       transforms_applied, created_at
                FROM extracted_tables
                WHERE session_id = :session_id
                ORDER BY created_at ASC
            """),
            {"session_id": session_id}
        )
        return [self._row_to_dict(row) for row in result.fetchall()]
    
    async def list_by_file(self, file_id: str) -> List[dict]:
        """
        List tables extracted from a file.
        
        Args:
            file_id: File UUID
            
        Returns:
            List of table dicts
        """
        result = await self.session.execute(
            text("""
                SELECT id, file_id, session_id, table_name, location, 
                       row_count, column_count, columns, structure_type, 
                       transforms_applied, created_at
                FROM extracted_tables
                WHERE file_id = :file_id
                ORDER BY created_at ASC
            """),
            {"file_id": file_id}
        )
        return [self._row_to_dict(row) for row in result.fetchall()]
    
    async def update_columns(
        self,
        table_id: str,
        columns: list
    ) -> bool:
        """
        Update table columns info.
        
        Args:
            table_id: Table UUID
            columns: Column info as list of dicts
            
        Returns:
            True if updated, False if not found
        """
        result = await self.session.execute(
            text("""
                UPDATE extracted_tables
                SET columns = :columns::jsonb
                WHERE id = :table_id
                RETURNING id
            """),
            {"table_id": table_id, "columns": json.dumps(columns)}
        )
        return result.fetchone() is not None
    
    async def delete(self, table_id: str) -> bool:
        """
        Delete a table record.
        
        Args:
            table_id: Table UUID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            text("""
                DELETE FROM extracted_tables
                WHERE id = :table_id
                RETURNING id
            """),
            {"table_id": table_id}
        )
        return result.fetchone() is not None
    
    def _row_to_dict(self, row) -> dict:
        """Convert a database row to a dict."""
        if row is None:
            return None
        return {
            "id": str(row.id),
            "file_id": str(row.file_id),
            "session_id": str(row.session_id),
            "table_name": row.table_name,
            "location": row.location,
            "row_count": row.row_count,
            "column_count": row.column_count,
            "columns": row.columns,
            "structure_type": row.structure_type,
            "transforms_applied": row.transforms_applied,
            "created_at": row.created_at
        }

