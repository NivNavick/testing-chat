"""
PostgreSQL database connection pool with pgvector support.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

# Register pgvector types
try:
    from pgvector.psycopg2 import register_vector
except ImportError:
    register_vector = None
    logging.warning("pgvector not installed. Run: pip install pgvector")

logger = logging.getLogger(__name__)


class Database:
    """
    Database connection pool manager.
    
    Usage:
        Database.init()  # Initialize once at startup
        
        with Database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM verticals")
                rows = cur.fetchall()
    """
    
    _pool: Optional[ThreadedConnectionPool] = None
    _config: dict = {}
    
    @classmethod
    def init(
        cls,
        host: str = "localhost",
        port: int = 5432,
        database: str = "csv_mapping",
        user: str = "postgres",
        password: str = "postgres",
        min_connections: int = 2,
        max_connections: int = 10,
    ):
        """
        Initialize the database connection pool.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        if cls._pool is not None:
            logger.warning("Database pool already initialized. Closing existing pool.")
            cls.close()
        
        cls._config = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password,
        }
        
        try:
            cls._pool = ThreadedConnectionPool(
                min_connections,
                max_connections,
                **cls._config
            )
            logger.info(f"✅ Database pool initialized: {user}@{host}:{port}/{database}")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @classmethod
    @contextmanager
    def get_connection(cls, skip_vector_registration: bool = False):
        """
        Get a connection from the pool.
        
        Usage:
            with Database.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(...)
        
        Args:
            skip_vector_registration: If True, skip pgvector type registration
                                     (useful for running migrations that create the extension)
        """
        if cls._pool is None:
            raise RuntimeError("Database not initialized. Call Database.init() first.")
        
        conn = cls._pool.getconn()
        
        # Register pgvector types for this connection (if extension exists)
        if register_vector and not skip_vector_registration:
            try:
                register_vector(conn)
            except Exception:
                # Extension might not be installed yet, that's ok
                pass
        
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cls._pool.putconn(conn)
    
    @classmethod
    def get_dict_cursor_connection(cls):
        """
        Get a connection that returns dicts instead of tuples.
        """
        if cls._pool is None:
            raise RuntimeError("Database not initialized. Call Database.init() first.")
        
        conn = cls._pool.getconn()
        
        if register_vector:
            register_vector(conn)
        
        return conn
    
    @classmethod
    def return_connection(cls, conn):
        """Return a connection to the pool."""
        if cls._pool:
            cls._pool.putconn(conn)
    
    @classmethod
    def close(cls):
        """Close all connections in the pool."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            logger.info("Database pool closed")
    
    @classmethod
    def run_migration(cls, migration_file: Optional[str] = None):
        """
        Run a SQL migration file.
        
        Args:
            migration_file: Path to SQL file. If None, runs the default initial schema.
        """
        if migration_file is None:
            # Default to initial schema
            migration_file = Path(__file__).parent / "migrations" / "001_initial_schema.sql"
        
        migration_path = Path(migration_file)
        if not migration_path.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_file}")
        
        sql = migration_path.read_text()
        
        # Skip vector registration during migration (we're creating the extension)
        with cls.get_connection(skip_vector_registration=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        
        logger.info(f"✅ Migration completed: {migration_path.name}")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the database pool is initialized."""
        return cls._pool is not None


def init_database(
    host: str = "localhost",
    port: int = 5432,
    database: str = "csv_mapping",
    user: str = "postgres",
    password: str = "postgres",
    run_migrations: bool = True,
):
    """
    Convenience function to initialize database and run migrations.
    
    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        run_migrations: Whether to run migrations after connecting
    """
    Database.init(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )
    
    if run_migrations:
        Database.run_migration()
