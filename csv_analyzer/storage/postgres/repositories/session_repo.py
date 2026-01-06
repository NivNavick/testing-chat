"""
Repository for session operations.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class SessionRepository:
    """Repository for session CRUD operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(
        self,
        vertical: str,
        mode: str = "AUTO"
    ) -> dict:
        """
        Create a new session.
        
        Args:
            vertical: Business vertical (e.g., 'medical')
            mode: Processing mode (AUTO, GUIDED, STRICT, DISCOVERY)
            
        Returns:
            Created session dict
        """
        result = await self.session.execute(
            text("""
                INSERT INTO sessions (vertical, mode, status)
                VALUES (:vertical, :mode, 'active')
                RETURNING id, vertical, mode, status, created_at, updated_at
            """),
            {"vertical": vertical, "mode": mode}
        )
        row = result.fetchone()
        return {
            "id": str(row.id),
            "vertical": row.vertical,
            "mode": row.mode,
            "status": row.status,
            "created_at": row.created_at,
            "updated_at": row.updated_at
        }
    
    async def get_by_id(self, session_id: str) -> Optional[dict]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session dict or None if not found
        """
        result = await self.session.execute(
            text("""
                SELECT id, vertical, mode, status, created_at, updated_at, closed_at
                FROM sessions
                WHERE id = :session_id
            """),
            {"session_id": session_id}
        )
        row = result.fetchone()
        if row is None:
            return None
        
        return {
            "id": str(row.id),
            "vertical": row.vertical,
            "mode": row.mode,
            "status": row.status,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "closed_at": row.closed_at
        }
    
    async def list_active(self, limit: int = 100) -> List[dict]:
        """
        List active sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dicts
        """
        result = await self.session.execute(
            text("""
                SELECT id, vertical, mode, status, created_at, updated_at
                FROM sessions
                WHERE status = 'active'
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"limit": limit}
        )
        
        return [
            {
                "id": str(row.id),
                "vertical": row.vertical,
                "mode": row.mode,
                "status": row.status,
                "created_at": row.created_at,
                "updated_at": row.updated_at
            }
            for row in result.fetchall()
        ]
    
    async def update_status(
        self,
        session_id: str,
        status: str
    ) -> bool:
        """
        Update session status.
        
        Args:
            session_id: Session UUID
            status: New status
            
        Returns:
            True if updated, False if not found
        """
        result = await self.session.execute(
            text("""
                UPDATE sessions
                SET status = :status
                WHERE id = :session_id
                RETURNING id
            """),
            {"session_id": session_id, "status": status}
        )
        return result.fetchone() is not None
    
    async def close(self, session_id: str) -> bool:
        """
        Close a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if closed, False if not found
        """
        result = await self.session.execute(
            text("""
                UPDATE sessions
                SET status = 'closed', closed_at = NOW()
                WHERE id = :session_id
                RETURNING id
            """),
            {"session_id": session_id}
        )
        return result.fetchone() is not None
    
    async def delete(self, session_id: str) -> bool:
        """
        Delete a session and all related data.
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            text("""
                DELETE FROM sessions
                WHERE id = :session_id
                RETURNING id
            """),
            {"session_id": session_id}
        )
        return result.fetchone() is not None

