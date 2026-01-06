"""
FastAPI dependency injection.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from csv_analyzer.storage.postgres.connection import db, get_db_session
from csv_analyzer.storage.postgres.repositories.session_repo import SessionRepository
from csv_analyzer.storage.postgres.repositories.file_repo import FileRepository
from csv_analyzer.storage.postgres.repositories.table_repo import TableRepository
from csv_analyzer.storage.postgres.repositories.insight_repo import InsightRepository
from csv_analyzer.storage.s3.client import S3Client, get_s3_client
from csv_analyzer.storage.s3.operations import S3Operations, get_s3_operations


async def get_session_repo(
    session: AsyncSession = None
) -> AsyncGenerator[SessionRepository, None]:
    """Get SessionRepository dependency."""
    if session:
        yield SessionRepository(session)
    else:
        async with db.session() as session:
            yield SessionRepository(session)


async def get_file_repo(
    session: AsyncSession = None
) -> AsyncGenerator[FileRepository, None]:
    """Get FileRepository dependency."""
    if session:
        yield FileRepository(session)
    else:
        async with db.session() as session:
            yield FileRepository(session)


async def get_table_repo(
    session: AsyncSession = None
) -> AsyncGenerator[TableRepository, None]:
    """Get TableRepository dependency."""
    if session:
        yield TableRepository(session)
    else:
        async with db.session() as session:
            yield TableRepository(session)


async def get_insight_repo(
    session: AsyncSession = None
) -> AsyncGenerator[InsightRepository, None]:
    """Get InsightRepository dependency."""
    if session:
        yield InsightRepository(session)
    else:
        async with db.session() as session:
            yield InsightRepository(session)


def get_s3() -> S3Client:
    """Get S3Client dependency."""
    return get_s3_client()


def get_s3_ops() -> S3Operations:
    """Get S3Operations dependency."""
    return get_s3_operations()

