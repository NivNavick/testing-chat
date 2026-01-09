"""
File-based session storage for persisting sessions to disk.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from csv_analyzer.sessions.session import ProcessingSession

logger = logging.getLogger(__name__)


class FileBasedSessionStore:
    """
    Persists sessions to disk as JSON + CSV files.
    
    Session structure on disk:
        sessions/
            {session_id}/
                session.json      # Session metadata
                documents/
                    {doc_id}.csv  # DataFrames
                    
    Usage:
        store = FileBasedSessionStore("./sessions")
        
        # Save a session
        store.save(session)
        
        # Load a session
        session = store.load("abc123")
        
        # List all sessions
        for session_info in store.list_sessions():
            print(session_info["session_id"])
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the session store.
        
        Args:
            base_path: Base directory for session storage.
                      Defaults to ./sessions in the current directory.
        """
        if base_path is None:
            base_path = os.path.join(os.getcwd(), "sessions")
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Session store initialized at {self.base_path}")
    
    def _session_dir(self, session_id: str) -> Path:
        """Get the directory for a session."""
        return self.base_path / session_id
    
    def _documents_dir(self, session_id: str) -> Path:
        """Get the documents directory for a session."""
        return self._session_dir(session_id) / "documents"
    
    def save(self, session: ProcessingSession) -> str:
        """
        Save a session to disk.
        
        Args:
            session: The session to save
            
        Returns:
            Path to the saved session directory
        """
        session_dir = self._session_dir(session.session_id)
        documents_dir = self._documents_dir(session.session_id)
        
        # Create directories
        session_dir.mkdir(parents=True, exist_ok=True)
        documents_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session metadata
        session_data = session.to_dict()
        session_file = session_dir / "session.json"
        
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        # Save DataFrames
        for doc_id in session._documents.keys():
            df = session.get_dataframe(doc_id)
            if df is not None:
                csv_path = documents_dir / f"{doc_id}.csv"
                df.to_csv(csv_path, index=False)
        
        logger.info(
            f"Saved session '{session.name}' to {session_dir} "
            f"({len(session.documents)} documents)"
        )
        
        return str(session_dir)
    
    def load(self, session_id: str) -> Optional[ProcessingSession]:
        """
        Load a session from disk.
        
        Args:
            session_id: ID of the session to load
            
        Returns:
            The loaded session, or None if not found
        """
        session_dir = self._session_dir(session_id)
        session_file = session_dir / "session.json"
        
        if not session_file.exists():
            logger.warning(f"Session '{session_id}' not found at {session_file}")
            return None
        
        # Load session metadata
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        
        session = ProcessingSession.from_dict(session_data)
        
        # Load DataFrames
        documents_dir = self._documents_dir(session_id)
        if documents_dir.exists():
            for doc_id in session._documents.keys():
                csv_path = documents_dir / f"{doc_id}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    session._dataframes[doc_id] = df
                    logger.debug(f"Loaded DataFrame for document {doc_id}")
        
        logger.info(
            f"Loaded session '{session.name}' from {session_dir} "
            f"({len(session.documents)} documents)"
        )
        
        return session
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session from disk.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if deleted, False if not found
        """
        session_dir = self._session_dir(session_id)
        
        if not session_dir.exists():
            return False
        
        import shutil
        shutil.rmtree(session_dir)
        
        logger.info(f"Deleted session '{session_id}'")
        return True
    
    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        session_file = self._session_dir(session_id) / "session.json"
        return session_file.exists()
    
    def list_sessions(self) -> List[dict]:
        """
        List all saved sessions.
        
        Returns:
            List of session info dictionaries with keys:
            - session_id
            - name
            - vertical
            - created_at
            - updated_at
            - document_count
        """
        sessions = []
        
        for session_dir in self.base_path.iterdir():
            if not session_dir.is_dir():
                continue
            
            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue
            
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                sessions.append({
                    "session_id": data["session_id"],
                    "name": data.get("name", ""),
                    "vertical": data.get("vertical", ""),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "document_count": len(data.get("documents", {})),
                })
            except Exception as e:
                logger.warning(f"Error reading session {session_dir}: {e}")
        
        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return sessions
    
    def get_latest_session(self) -> Optional[ProcessingSession]:
        """Get the most recently updated session."""
        sessions = self.list_sessions()
        if not sessions:
            return None
        
        return self.load(sessions[0]["session_id"])
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Delete sessions older than max_age_days.
        
        Args:
            max_age_days: Maximum age in days before deletion
            
        Returns:
            Number of sessions deleted
        """
        deleted = 0
        cutoff = datetime.now()
        
        for session_info in self.list_sessions():
            try:
                updated_at = datetime.fromisoformat(session_info["updated_at"])
                age_days = (cutoff - updated_at).days
                
                if age_days > max_age_days:
                    self.delete(session_info["session_id"])
                    deleted += 1
            except Exception as e:
                logger.warning(f"Error checking session age: {e}")
        
        if deleted:
            logger.info(f"Cleaned up {deleted} old sessions")
        
        return deleted

