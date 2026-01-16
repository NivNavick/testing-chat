"""
Custom exceptions and error handling for the API.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class APIError(Exception):
    """Base API exception."""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type: str = "api_error",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)


class NotFoundError(APIError):
    """Resource not found."""
    
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} not found: {identifier}",
            status_code=status.HTTP_404_NOT_FOUND,
            error_type="not_found",
            details={"resource": resource, "identifier": identifier},
        )


class ValidationError(APIError):
    """Validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_type="validation_error",
            details=details,
        )


class ConflictError(APIError):
    """Resource conflict."""
    
    def __init__(self, message: str, resource: Optional[str] = None):
        details = {"resource": resource} if resource else {}
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_type="conflict",
            details=details,
        )


class ServiceUnavailableError(APIError):
    """External service unavailable."""
    
    def __init__(self, service: str, message: Optional[str] = None):
        super().__init__(
            message=message or f"Service unavailable: {service}",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_type="service_unavailable",
            details={"service": service},
        )


class StorageError(APIError):
    """Storage operation failed."""
    
    def __init__(self, operation: str, message: str):
        super().__init__(
            message=f"Storage error ({operation}): {message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type="storage_error",
            details={"operation": operation},
        )


class DatabaseError(APIError):
    """Database operation failed."""
    
    def __init__(self, message: str):
        super().__init__(
            message=f"Database error: {message}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_type="database_error",
        )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str
    type: str
    status_code: int
    details: Dict[str, Any] = {}


def create_error_response(exc: APIError) -> JSONResponse:
    """Create a JSON response from an API error."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "type": exc.error_type,
            "status_code": exc.status_code,
            "details": exc.details,
        },
    )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions."""
    return create_error_response(exc)


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic exceptions."""
    import logging
    import traceback
    
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": "internal_error",
            "status_code": 500,
            "details": {},
        },
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "http_error",
            "status_code": exc.status_code,
            "details": {},
        },
    )


def register_exception_handlers(app):
    """Register all exception handlers with the app."""
    from fastapi import HTTPException
    
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

