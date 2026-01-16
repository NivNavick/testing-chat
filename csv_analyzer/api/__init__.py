"""
CSV Analyzer API Package.

FastAPI application with async PostgreSQL and S3 storage.
"""

from csv_analyzer.api.main import app, create_app

__all__ = ["app", "create_app"]

