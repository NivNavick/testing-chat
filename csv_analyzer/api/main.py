"""
FastAPI Application for CSV Analyzer.

Main entry point for the API server.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from csv_analyzer.api.config import get_settings
from csv_analyzer.api.dependencies import lifespan_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    settings = get_settings()
    
    logger.info(f"Starting {settings.api.title} v{settings.api.version}")
    
    async with lifespan_context(settings):
        logger.info("Application startup complete")
        yield
        logger.info("Application shutdown initiated")
    
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        description="CSV Analyzer API - Classification, Workflow Execution, and Insights",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Register routers
    register_routers(app)
    
    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    from csv_analyzer.api.exceptions import register_exception_handlers as register_handlers
    from csv_analyzer.sessions.db_store import SessionNotFoundError, DocumentNotFoundError
    
    # Register custom exception handlers
    register_handlers(app)
    
    # Additional handlers for domain-specific exceptions
    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
        """Handle session not found."""
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exc),
                "type": "not_found",
                "status_code": 404,
            },
        )
    
    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
        """Handle document not found."""
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exc),
                "type": "not_found",
                "status_code": 404,
            },
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors as 400 Bad Request."""
        return JSONResponse(
            status_code=400,
            content={
                "detail": str(exc),
                "type": "validation_error",
                "status_code": 400,
            },
        )
    
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """Handle file not found as 404."""
        return JSONResponse(
            status_code=404,
            content={
                "detail": str(exc),
                "type": "not_found",
                "status_code": 404,
            },
        )


def register_routers(app: FastAPI) -> None:
    """Register API routers."""
    from csv_analyzer.api.routes import router as api_router
    
    # Mount all routes under /api/v1
    app.include_router(api_router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API info."""
        settings = get_settings()
        return {
            "name": settings.api.title,
            "version": settings.api.version,
            "docs": "/docs",
            "health": "/health",
        }


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "csv_analyzer.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
        workers=1,  # Use 1 for reload mode
    )

