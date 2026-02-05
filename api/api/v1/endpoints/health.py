"""
Health Check Endpoints
======================
"""

import time
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.database import get_db
from api.core.cache import get_redis
from api.schemas.common import HealthResponse, DetailedHealthResponse

router = APIRouter()

START_TIME = time.time()


@router.get("", response_model=HealthResponse)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        database="unknown",
        redis="unknown",
        celery="unknown",
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    db: AsyncSession = Depends(get_db),
):
    """
    Detailed health check with latency measurements.
    
    Checks:
    - Database connectivity
    - Redis connectivity
    - Celery worker status
    """
    # Database check
    db_status = "healthy"
    db_latency = 0.0
    try:
        start = time.time()
        await db.execute(text("SELECT 1"))
        db_latency = (time.time() - start) * 1000
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    # Redis check
    redis_status = "healthy"
    redis_latency = 0.0
    try:
        start = time.time()
        redis = await get_redis()
        await redis.ping()
        redis_latency = (time.time() - start) * 1000
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    # Celery check
    celery_status = "unknown"
    pending_tasks = 0
    try:
        from api.workers.celery_app import celery_app
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        if stats:
            celery_status = "healthy"
            # Count pending tasks
            reserved = inspect.reserved()
            if reserved:
                pending_tasks = sum(len(tasks) for tasks in reserved.values())
        else:
            celery_status = "no workers"
    except Exception as e:
        celery_status = f"unavailable: {str(e)}"
    
    # Active jobs count
    from api.models.job import DetectionJob
    from sqlalchemy import select, func
    active_result = await db.execute(
        select(func.count(DetectionJob.id)).where(
            DetectionJob.status == "processing"
        )
    )
    active_jobs = active_result.scalar() or 0
    
    # Uptime
    uptime = time.time() - START_TIME
    
    return DetailedHealthResponse(
        status="healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
        version=settings.APP_VERSION,
        database=db_status,
        redis=redis_status,
        celery=celery_status,
        database_latency_ms=db_latency,
        redis_latency_ms=redis_latency,
        active_jobs=active_jobs,
        pending_tasks=pending_tasks,
        uptime_seconds=uptime,
    )
