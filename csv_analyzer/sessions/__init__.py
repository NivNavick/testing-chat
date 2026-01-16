"""
Session management for multi-document CSV processing.

Provides:
- ProcessingSession: Manages multiple documents within a session
- FileBasedSessionStore: Persists sessions to disk
- AsyncSessionStore: Async PostgreSQL-backed session storage
"""

from csv_analyzer.sessions.session import ProcessingSession, DocumentInfo
from csv_analyzer.sessions.store import FileBasedSessionStore
from csv_analyzer.sessions.db_store import (
    AsyncSessionStore,
    SessionNotFoundError,
    DocumentNotFoundError,
)

__all__ = [
    "ProcessingSession",
    "DocumentInfo",
    "FileBasedSessionStore",
    "AsyncSessionStore",
    "SessionNotFoundError",
    "DocumentNotFoundError",
]

