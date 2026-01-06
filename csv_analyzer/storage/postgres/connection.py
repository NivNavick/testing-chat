"""
PostgreSQL database connection management.
Uses SQLAlchemy async for connection pooling.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)

from csv_analyzer.core.config import get_settings


class DatabaseManager:
    """Manages PostgreSQL database connections."""
    
    def __init__(self):
        self._engine = None
        self._session_factory = None
    
    def init(self, database_url: str = None):
        """
        Initialize the database engine and session factory.
        
        Args:
            database_url: PostgreSQL connection URL. Uses settings if not provided.
        """
        if database_url is None:
            settings = get_settings()
            database_url = settings.postgres_async_url
        
        self._engine = create_async_engine(
            database_url,
            echo=False,  # Set to True for SQL logging
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    @property
    def engine(self):
        """Get the database engine."""
        if self._engine is None:
            self.init()
        return self._engine
    
    @property
    def session_factory(self):
        """Get the session factory."""
        if self._session_factory is None:
            self.init()
        return self._session_factory
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session as an async context manager.
        
        Usage:
            async with db.session() as session:
                # Use session
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def close(self):
        """Close all database connections."""
        if self._engine:
            await self._engine.dispose()


# Global database manager instance
db = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting a database session.
    
    Usage in FastAPI:
        @app.get("/")
        async def handler(session: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with db.session() as session:
        yield session


async def init_database():
    """Initialize the database (create tables)."""
    import os
    from sqlalchemy import text
    
    migrations_dir = os.path.join(
        os.path.dirname(__file__),
        "migrations"
    )
    
    async with db.engine.begin() as conn:
        # Read and execute the initial schema
        schema_file = os.path.join(migrations_dir, "001_initial_schema.sql")
        if os.path.exists(schema_file):
            with open(schema_file, "r") as f:
                schema_sql = f.read()
            
            # Execute each statement separately
            for statement in schema_sql.split(";"):
                statement = statement.strip()
                if statement:
                    await conn.execute(text(statement))

