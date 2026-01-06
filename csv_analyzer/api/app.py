"""
FastAPI Application for Analytics Platform.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from csv_analyzer.core.config import get_settings
from csv_analyzer.storage.postgres.connection import db, init_database
from csv_analyzer.api.routes import sessions, upload, insights, health, pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    
    # Initialize database connection
    db.init(settings.postgres_async_url)
    
    # Run migrations
    try:
        await init_database()
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")
    
    yield
    
    # Shutdown
    await db.close()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Analytics Platform API",
        description="API for intelligent file processing, classification, and insights generation",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(sessions.router, prefix="/api/v1")
    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(insights.router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "Analytics Platform API",
            "version": "1.0.0",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "csv_analyzer.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug
    )

