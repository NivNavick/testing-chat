"""
API Routes for CSV Analyzer.

Aggregates all route modules into a single router.
"""

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# Include session routes
try:
    from csv_analyzer.api.routes.pipeline import router as sessions_router
    router.include_router(sessions_router, prefix="/sessions", tags=["Sessions"])
except ImportError as e:
    logger.warning(f"Could not import sessions router: {e}")

# Include classification routes
try:
    from csv_analyzer.api.routes.classification import router as classification_router
    router.include_router(classification_router, prefix="/classification", tags=["Classification"])
except ImportError as e:
    logger.warning(f"Could not import classification router: {e}")

# Include workflow routes
try:
    from csv_analyzer.api.routes.workflows import router as workflows_router
    router.include_router(workflows_router, prefix="/workflows", tags=["Workflows"])
except ImportError as e:
    logger.warning(f"Could not import workflows router: {e}")

# Include insights router
# Note: insights_config already has prefix "/api/insights" so we mount it without additional prefix
try:
    from csv_analyzer.api.routes.insights_config import router as insights_router
    # Remove the /api prefix since we're already under /api/v1
    # The insights router will be at /api/v1/insights
    insights_router.prefix = "/insights"
    router.include_router(insights_router, tags=["Insights"])
except ImportError as e:
    logger.warning(f"Could not import insights router: {e}")

__all__ = ["router"]

