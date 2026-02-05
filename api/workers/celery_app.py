"""
Celery Application Configuration
================================

Celery app for background task processing.
"""

from celery import Celery

from api.core.config import settings


# =============================================================================
# CELERY APP
# =============================================================================

celery_app = Celery(
    "bus_vision",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "api.workers.detection_tasks",
        "api.workers.report_tasks",
        "api.workers.maintenance_tasks",
    ],
)


# =============================================================================
# CONFIGURATION
# =============================================================================

celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task routing
    task_routes={
        "api.workers.detection_tasks.*": {"queue": "detection"},
        "api.workers.report_tasks.*": {"queue": "reports"},
        "api.workers.maintenance_tasks.*": {"queue": "maintenance"},
    },
    
    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Result settings
    result_expires=86400,  # 24 hours
    
    # Worker settings
    worker_prefetch_multiplier=1,  # For GPU tasks
    worker_concurrency=4,
    
    # Rate limiting
    task_annotations={
        "api.workers.detection_tasks.process_video_task": {
            "rate_limit": "10/m",  # Max 10 video processing per minute
        },
    },
    
    # Beat schedule (periodic tasks)
    beat_schedule={
        "cleanup-old-files": {
            "task": "api.workers.maintenance_tasks.cleanup_old_files_task",
            "schedule": 86400,  # Daily
        },
        "generate-daily-report": {
            "task": "api.workers.report_tasks.generate_daily_report_task",
            "schedule": 86400,  # Daily at midnight
        },
    },
)


# =============================================================================
# SIGNALS
# =============================================================================

@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing."""
    print(f"Request: {self.request!r}")
    return {"status": "ok"}
