"""
Health check and status API routes.
"""

from fastapi import APIRouter

from csv_analyzer.core.config import get_settings
from csv_analyzer.storage.postgres.connection import db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Basic health check.
    """
    return {"status": "healthy"}


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with component status.
    """
    settings = get_settings()
    
    components = {
        "api": {"status": "healthy"},
        "config": {"status": "healthy"}
    }
    
    # Check PostgreSQL
    try:
        async with db.session() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        components["postgres"] = {"status": "healthy"}
    except Exception as e:
        components["postgres"] = {"status": "unhealthy", "error": str(e)}
    
    # Check S3 (just config, not actual connection)
    if settings.s3_bucket:
        components["s3"] = {"status": "configured", "bucket": settings.s3_bucket}
    else:
        components["s3"] = {"status": "not_configured"}
    
    # Overall status
    all_healthy = all(
        c.get("status") in ["healthy", "configured"]
        for c in components.values()
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": components
    }


@router.get("/info")
async def api_info():
    """
    Get API information.
    """
    return {
        "name": "Analytics Platform API",
        "version": "1.0.0",
        "description": "Intelligent file processing, classification, and insights generation",
        "endpoints": {
            "sessions": "/api/v1/sessions",
            "upload": "/api/v1/sessions/{session_id}/upload",
            "insights": "/api/v1/sessions/{session_id}/insights",
            "results": "/api/v1/sessions/{session_id}/results"
        }
    }

