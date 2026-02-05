"""
Common Pydantic Schemas
=======================
"""

from typing import Generic, TypeVar, Optional, List

from pydantic import BaseModel


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
    redis: str
    celery: str


class DetailedHealthResponse(HealthResponse):
    """Detailed health check."""
    database_latency_ms: float
    redis_latency_ms: float
    active_jobs: int
    pending_tasks: int
    uptime_seconds: float
