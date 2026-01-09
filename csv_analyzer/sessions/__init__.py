"""
Session management for multi-document CSV processing.

Provides:
- ProcessingSession: Manages multiple documents within a session
- FileBasedSessionStore: Persists sessions to disk
"""

from csv_analyzer.sessions.session import ProcessingSession, DocumentInfo
from csv_analyzer.sessions.store import FileBasedSessionStore

__all__ = [
    "ProcessingSession",
    "DocumentInfo",
    "FileBasedSessionStore",
]

