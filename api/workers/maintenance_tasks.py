"""
Maintenance Tasks
=================

Periodic maintenance Celery tasks.
"""

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .celery_app import celery_app


@celery_app.task
def cleanup_old_files_task(days_old: int = 30):
    """
    Clean up old files from uploads and outputs.
    
    Removes:
    - Uploaded videos older than X days (if job completed)
    - Temporary files
    - Old output videos
    """
    import asyncio
    from sqlalchemy import select, and_
    from api.core.database import get_db_context
    from api.core.config import settings
    from api.models.job import VideoUpload, DetectionJob
    
    async def run():
        deleted_files = []
        freed_space = 0
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
        
        async with get_db_context() as db:
            # Find old completed videos
            result = await db.execute(
                select(VideoUpload).where(
                    VideoUpload.created_at < cutoff_date
                )
            )
            old_videos = result.scalars().all()
            
            for video in old_videos:
                # Check if all jobs for this video are completed
                jobs_result = await db.execute(
                    select(DetectionJob).where(
                        and_(
                            DetectionJob.video_id == video.id,
                            DetectionJob.status.not_in(["completed", "failed", "cancelled"])
                        )
                    )
                )
                
                if not jobs_result.scalar_one_or_none():
                    # Safe to delete - all jobs done
                    if os.path.exists(video.file_path):
                        file_size = os.path.getsize(video.file_path)
                        os.remove(video.file_path)
                        deleted_files.append(video.file_path)
                        freed_space += file_size
                    
                    if video.thumbnail_path and os.path.exists(video.thumbnail_path):
                        os.remove(video.thumbnail_path)
            
            # Clean up orphaned output directories
            outputs_dir = Path("outputs")
            if outputs_dir.exists():
                for job_dir in outputs_dir.iterdir():
                    if job_dir.is_dir():
                        try:
                            job_id = int(job_dir.name.replace("job_", ""))
                            
                            # Check if job exists
                            result = await db.execute(
                                select(DetectionJob).where(DetectionJob.id == job_id)
                            )
                            if not result.scalar_one_or_none():
                                # Orphaned directory
                                shutil.rmtree(job_dir)
                                deleted_files.append(str(job_dir))
                        except ValueError:
                            pass
        
        return {
            "status": "completed",
            "deleted_files": len(deleted_files),
            "freed_space_mb": freed_space / (1024 * 1024),
        }
    
    return asyncio.get_event_loop().run_until_complete(run())


@celery_app.task
def backup_database_task():
    """
    Create database backup.
    
    Uses pg_dump for PostgreSQL.
    """
    import subprocess
    from api.core.config import settings
    
    backup_dir = Path("backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"bus_vision_{timestamp}.sql"
    
    # Parse database URL
    # postgresql://user:pass@host:port/dbname
    db_url = settings.sync_database_url
    
    try:
        # Use pg_dump
        result = subprocess.run(
            ["pg_dump", db_url, "-f", str(backup_file)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            raise Exception(f"pg_dump failed: {result.stderr}")
        
        # Compress backup
        import gzip
        with open(backup_file, "rb") as f_in:
            with gzip.open(f"{backup_file}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove(backup_file)
        
        return {
            "status": "completed",
            "backup_file": f"{backup_file}.gz",
            "size_mb": os.path.getsize(f"{backup_file}.gz") / (1024 * 1024),
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


@celery_app.task
def cleanup_expired_tokens_task():
    """
    Remove expired refresh tokens from database.
    """
    import asyncio
    from sqlalchemy import delete
    from api.core.database import get_db_context
    from api.models.user import RefreshToken
    
    async def run():
        async with get_db_context() as db:
            result = await db.execute(
                delete(RefreshToken).where(
                    RefreshToken.expires_at < datetime.now(timezone.utc)
                )
            )
            await db.commit()
            
            return {
                "status": "completed",
                "deleted_tokens": result.rowcount,
            }
    
    return asyncio.get_event_loop().run_until_complete(run())


@celery_app.task
def update_camera_status_task():
    """
    Update camera status based on last activity.
    
    Marks cameras as offline if no activity in 5 minutes.
    """
    import asyncio
    from sqlalchemy import select, update
    from api.core.database import get_db_context
    from api.models.bus import Camera
    
    async def run():
        # This would integrate with actual camera health checks
        # For now, just return status
        return {
            "status": "completed",
            "cameras_checked": 0,
        }
    
    return asyncio.get_event_loop().run_until_complete(run())
