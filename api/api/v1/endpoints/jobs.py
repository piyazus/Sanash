"""
Detection Jobs Endpoints
========================
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.database import get_db
from api.core.dependencies import get_current_user
from api.models.user import User
from api.models.job import DetectionJob, VideoUpload
from api.models.detection import Detection
from api.schemas.job import (
    JobCreate,
    JobUpdate,
    JobResponse,
    JobProgress,
    JobResults,
    JobListItem,
)

router = APIRouter()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}
    
    async def connect(self, job_id: int, websocket: WebSocket):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    def disconnect(self, job_id: int, websocket: WebSocket):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
    
    async def broadcast(self, job_id: int, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    job_data: JobCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new detection job.
    
    The job will be queued for processing by Celery workers.
    """
    # Verify video exists and belongs to user
    result = await db.execute(
        select(VideoUpload).where(
            VideoUpload.id == job_data.video_id,
            VideoUpload.user_id == user.id
        )
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Calculate total frames
    total_frames = None
    if video.fps and video.duration_seconds:
        total_frames = int(video.fps * video.duration_seconds)
    
    # Create job
    job = DetectionJob(
        video_id=video.id,
        user_id=user.id,
        bus_id=job_data.bus_id,
        status="queued",
        total_frames=total_frames,
        config=job_data.config.model_dump(),
    )
    
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Queue Celery task
    try:
        from api.workers.detection_tasks import process_video_task
        task = process_video_task.delay(job.id)
        job.celery_task_id = task.id
        await db.commit()
    except Exception as e:
        # If Celery not available, mark as pending manual processing
        job.status = "pending"
        await db.commit()
    
    return job


@router.get("", response_model=list[JobListItem])
async def list_jobs(
    skip: int = 0,
    limit: int = 20,
    status_filter: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's detection jobs."""
    query = select(DetectionJob).where(DetectionJob.user_id == user.id)
    
    if status_filter:
        query = query.where(DetectionJob.status == status_filter)
    
    query = query.order_by(DetectionJob.created_at.desc()).offset(skip).limit(limit)
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return jobs


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get job details."""
    result = await db.execute(
        select(DetectionJob).where(
            DetectionJob.id == job_id,
            DetectionJob.user_id == user.id
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    
    return job


@router.get("/{job_id}/progress", response_model=JobProgress)
async def get_job_progress(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get real-time job progress."""
    result = await db.execute(
        select(DetectionJob).where(
            DetectionJob.id == job_id,
            DetectionJob.user_id == user.id
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    
    # Calculate ETA
    eta_seconds = None
    if job.status == "processing" and job.progress > 0 and job.started_at:
        elapsed = (datetime.now(timezone.utc) - job.started_at).total_seconds()
        if job.progress > 0:
            total_estimated = elapsed / (job.progress / 100)
            eta_seconds = total_estimated - elapsed
    
    # Get detection count
    count_result = await db.execute(
        select(func.count(Detection.id)).where(Detection.job_id == job_id)
    )
    detections_count = count_result.scalar()
    
    return JobProgress(
        status=job.status,
        progress=job.progress,
        current_frame=job.current_frame,
        total_frames=job.total_frames,
        eta_seconds=eta_seconds,
        detections_count=detections_count,
    )


@router.get("/{job_id}/results", response_model=JobResults)
async def get_job_results(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get job results after completion."""
    result = await db.execute(
        select(DetectionJob).where(
            DetectionJob.id == job_id,
            DetectionJob.user_id == user.id
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job not completed. Status: {job.status}",
        )
    
    # Build result URLs
    base_url = f"/api/v1/jobs/{job_id}"
    
    return JobResults(
        job_id=job_id,
        status=job.status,
        output_video_url=f"{base_url}/video" if job.output_video_path else None,
        report_url=f"{base_url}/report" if job.report_path else None,
        detections_csv_url=f"{base_url}/detections.csv",
        analytics=None,  # Populated from analytics summary
    )


@router.delete("/{job_id}")
async def delete_job(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a job and its data."""
    result = await db.execute(
        select(DetectionJob).where(
            DetectionJob.id == job_id,
            DetectionJob.user_id == user.id
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    
    # Cancel if running
    if job.status == "processing" and job.celery_task_id:
        try:
            from api.workers.celery_app import celery_app
            celery_app.control.revoke(job.celery_task_id, terminate=True)
        except Exception:
            pass
    
    await db.delete(job)
    await db.commit()
    
    return {"success": True, "message": "Job deleted"}


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running job."""
    result = await db.execute(
        select(DetectionJob).where(
            DetectionJob.id == job_id,
            DetectionJob.user_id == user.id
        )
    )
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    
    if job.status not in ["queued", "processing"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}",
        )
    
    # Cancel Celery task
    if job.celery_task_id:
        try:
            from api.workers.celery_app import celery_app
            celery_app.control.revoke(job.celery_task_id, terminate=True)
        except Exception:
            pass
    
    job.status = "cancelled"
    await db.commit()
    
    return {"success": True, "message": "Job cancelled"}


# WebSocket endpoint for live updates
@router.websocket("/{job_id}/live")
async def job_live_updates(
    websocket: WebSocket,
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    WebSocket for real-time job progress updates.
    
    Messages:
    - {type: "progress", frame: 1500, people_count: 23, fps: 15.2, progress: 45.2}
    - {type: "anomaly", severity: "high", description: "Overcrowding detected"}
    - {type: "complete", total_time: 3600, results_url: "..."}
    """
    # Verify job exists
    result = await db.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    
    if not job:
        await websocket.close(code=4004)
        return
    
    await manager.connect(job_id, websocket)
    
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(job_id, websocket)
