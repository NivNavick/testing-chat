"""
Shared dependencies for FastAPI routes.

Provides:
- Async PostgreSQL connection pool
- Async S3 client
- Configuration access
- Thread pool executor for sync operations
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg

from csv_analyzer.api.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Global instances
_db_pool: Optional[asyncpg.Pool] = None
_executor: Optional[ThreadPoolExecutor] = None
_s3_client: Optional["S3Client"] = None


async def _setup_json_codec(conn: asyncpg.Connection) -> None:
    """Set up JSON codec for a connection."""
    import json
    await conn.set_type_codec(
        'jsonb',
        encoder=json.dumps,
        decoder=json.loads,
        schema='pg_catalog',
        format='text',
    )
    await conn.set_type_codec(
        'json',
        encoder=json.dumps,
        decoder=json.loads,
        schema='pg_catalog',
        format='text',
    )


async def init_db_pool(settings: Settings) -> asyncpg.Pool:
    """Initialize the async PostgreSQL connection pool."""
    global _db_pool
    
    if _db_pool is not None:
        return _db_pool
    
    logger.info(
        f"Initializing PostgreSQL connection pool: "
        f"{settings.database.user}@{settings.database.host}:{settings.database.port}/{settings.database.name}"
    )
    
    _db_pool = await asyncpg.create_pool(
        host=settings.database.host,
        port=settings.database.port,
        database=settings.database.name,
        user=settings.database.user,
        password=settings.database.password,
        min_size=settings.database.pool_min,
        max_size=settings.database.pool_max,
        init=_setup_json_codec,  # Set up JSON codec for each connection
    )
    
    logger.info("PostgreSQL connection pool initialized successfully")
    return _db_pool


async def close_db_pool() -> None:
    """Close the PostgreSQL connection pool."""
    global _db_pool
    
    if _db_pool is not None:
        await _db_pool.close()
        _db_pool = None
        logger.info("PostgreSQL connection pool closed")


def get_db_pool() -> asyncpg.Pool:
    """Get the database connection pool."""
    if _db_pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db_pool first.")
    return _db_pool


async def get_db() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Dependency to get a database connection from the pool.
    
    Usage:
        @router.get("/items")
        async def get_items(db: asyncpg.Connection = Depends(get_db)):
            rows = await db.fetch("SELECT * FROM items")
            return rows
    """
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield conn


def init_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Initialize the thread pool executor for sync operations."""
    global _executor
    
    if _executor is not None:
        return _executor
    
    _executor = ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"Thread pool executor initialized with {max_workers} workers")
    return _executor


def shutdown_executor() -> None:
    """Shutdown the thread pool executor."""
    global _executor
    
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("Thread pool executor shutdown")


def get_executor() -> ThreadPoolExecutor:
    """Get the thread pool executor."""
    if _executor is None:
        raise RuntimeError("Executor not initialized. Call init_executor first.")
    return _executor


async def run_sync(func, *args, **kwargs):
    """
    Run a synchronous function in the thread pool.
    
    Usage:
        result = await run_sync(sync_function, arg1, arg2)
    """
    executor = get_executor()
    loop = asyncio.get_event_loop()
    
    if kwargs:
        return await loop.run_in_executor(
            executor,
            lambda: func(*args, **kwargs)
        )
    else:
        return await loop.run_in_executor(
            executor,
            func,
            *args
        )


async def init_s3_client(settings: Settings) -> "S3Client":
    """Initialize the async S3 client."""
    global _s3_client
    
    if _s3_client is not None:
        return _s3_client
    
    # Import here to avoid circular imports
    from csv_analyzer.api.storage.s3_client import S3Client
    
    _s3_client = S3Client(
        bucket=settings.s3.bucket,
        region=settings.s3.region,
        endpoint_url=settings.s3.endpoint_url,
        prefix_config=settings.s3.prefix,
    )
    
    logger.info(f"S3 client initialized for bucket: {settings.s3.bucket}")
    return _s3_client


async def close_s3_client() -> None:
    """Close the S3 client."""
    global _s3_client
    
    if _s3_client is not None:
        await _s3_client.close()
        _s3_client = None
        logger.info("S3 client closed")


def get_s3_client() -> "S3Client":
    """Get the S3 client."""
    if _s3_client is None:
        raise RuntimeError("S3 client not initialized. Call init_s3_client first.")
    return _s3_client


@asynccontextmanager
async def lifespan_context(settings: Optional[Settings] = None):
    """
    Context manager for application lifespan.
    
    Initializes and cleans up all resources.
    """
    if settings is None:
        settings = get_settings()
    
    # Startup
    await init_db_pool(settings)
    init_executor(max_workers=settings.api.workers)
    await init_s3_client(settings)
    
    yield
    
    # Shutdown
    await close_s3_client()
    shutdown_executor()
    await close_db_pool()

