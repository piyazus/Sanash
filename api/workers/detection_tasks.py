"""
Video Detection Tasks
=====================

Celery tasks for video processing using bus_tracker.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from celery import shared_task

from .celery_app import celery_app


@celery_app.task(bind=True, max_retries=3)
def process_video_task(self, job_id: int):
    """
    Process video with person detection.
    
    This is the main video processing task that:
    1. Loads the detection model
    2. Processes video frames
    3. Updates job progress in real-time
    4. Stores detections in database
    5. Generates analytics summary
    6. Creates output video and reports
    
    Args:
        job_id: Detection job ID
    """
    import asyncio
    from sqlalchemy import select
    from api.core.database import get_db_context
    from api.models.job import DetectionJob, VideoUpload
    from api.models.detection import Detection
    from api.models.analytics import AnalyticsSummary
    
    async def run():
        async with get_db_context() as db:
            # Load job
            result = await db.execute(
                select(DetectionJob).where(DetectionJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            
            if not job:
                raise ValueError(f"Job {job_id} not found")
            
            # Load video
            result = await db.execute(
                select(VideoUpload).where(VideoUpload.id == job.video_id)
            )
            video = result.scalar_one_or_none()
            
            if not video:
                raise ValueError(f"Video {job.video_id} not found")
            
            # Update status
            job.status = "processing"
            job.started_at = datetime.now(timezone.utc)
            await db.commit()
            
            try:
                # Import detection engine
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from bus_tracker.detector import PersonDetector
                from bus_tracker.config import DetectionConfig
                
                # Initialize detector with job config
                config = DetectionConfig()
                config.CONFIDENCE_THRESHOLD = job.config.get("confidence", 0.5)
                config.FRAME_SKIP = job.config.get("frame_skip", 2)
                
                detector = PersonDetector(
                    model_path=job.config.get("model", "yolov8m.pt"),
                    config=config,
                )
                
                # Process video
                video_path = Path(video.file_path)
                output_dir = Path("outputs") / f"job_{job_id}"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Processing callback for progress updates
                last_update = time.time()
                detections_buffer = []
                
                def progress_callback(frame_num, total_frames, detections, fps):
                    nonlocal last_update, detections_buffer
                    
                    # Buffer detections
                    for det in detections:
                        detections_buffer.append({
                            "job_id": job_id,
                            "frame_number": frame_num,
                            "timestamp": datetime.now(timezone.utc),
                            "track_id": det.get("track_id"),
                            "bbox": det.get("bbox"),
                            "confidence": det.get("confidence"),
                        })
                    
                    # Batch save every 500 detections
                    if len(detections_buffer) >= 500:
                        save_detections_sync(detections_buffer)
                        detections_buffer = []
                    
                    # Update progress every 2 seconds
                    if time.time() - last_update >= 2:
                        update_progress_sync(
                            job_id, frame_num, total_frames, fps
                        )
                        last_update = time.time()
                        
                        # Send WebSocket update
                        broadcast_progress_sync(
                            job_id,
                            {
                                "type": "progress",
                                "frame": frame_num,
                                "people_count": len(detections),
                                "fps": fps,
                                "progress": (frame_num / total_frames) * 100 if total_frames else 0,
                            }
                        )
                
                # Run detection
                results = detector.process_video(
                    str(video_path),
                    output_path=str(output_dir / "output.mp4"),
                    progress_callback=progress_callback,
                    dashboard=False,
                )
                
                # Save remaining detections
                if detections_buffer:
                    save_detections_sync(detections_buffer)
                
                # Generate analytics summary
                summary = AnalyticsSummary(
                    job_id=job_id,
                    total_unique_people=results.get("unique_tracks", 0),
                    total_detections=results.get("total_detections", 0),
                    peak_occupancy=results.get("peak_occupancy", 0),
                    avg_occupancy=results.get("avg_occupancy", 0),
                    total_entries=results.get("total_entries", 0),
                    total_exits=results.get("total_exits", 0),
                    processing_time_seconds=results.get("processing_time", 0),
                    frames_processed=results.get("frames_processed", 0),
                    avg_fps=results.get("avg_fps", 0),
                )
                db.add(summary)
                
                # Update job as completed
                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                job.progress = 100.0
                job.output_video_path = str(output_dir / "output.mp4")
                
                await db.commit()
                
                # Send completion notification
                broadcast_progress_sync(
                    job_id,
                    {
                        "type": "complete",
                        "total_time_seconds": results.get("processing_time", 0),
                        "results_url": f"/api/v1/jobs/{job_id}/results",
                    }
                )
                
                return {
                    "status": "completed",
                    "job_id": job_id,
                    "results": results,
                }
                
            except Exception as e:
                job.status = "failed"
                job.error_message = str(e)
                await db.commit()
                
                raise self.retry(exc=e)
    
    # Run async code
    return asyncio.get_event_loop().run_until_complete(run())


def save_detections_sync(detections: list):
    """Save detections to database (sync wrapper)."""
    import asyncio
    from api.core.database import get_db_context
    from api.models.detection import Detection
    
    async def save():
        async with get_db_context() as db:
            for det in detections:
                db.add(Detection(**det))
            await db.commit()
    
    asyncio.get_event_loop().run_until_complete(save())


def update_progress_sync(job_id: int, frame: int, total: int, fps: float):
    """Update job progress (sync wrapper)."""
    import asyncio
    from sqlalchemy import select
    from api.core.database import get_db_context
    from api.models.job import DetectionJob
    
    async def update():
        async with get_db_context() as db:
            result = await db.execute(
                select(DetectionJob).where(DetectionJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.current_frame = frame
                job.total_frames = total
                job.progress = (frame / total) * 100 if total else 0
                await db.commit()
    
    asyncio.get_event_loop().run_until_complete(update())


def broadcast_progress_sync(job_id: int, message: dict):
    """Broadcast WebSocket message (placeholder - needs Redis pub/sub)."""
    # In production, use Redis pub/sub for cross-process WebSocket broadcasting
    try:
        import redis
        r = redis.Redis.from_url("redis://localhost:6379/0")
        import json
        r.publish(f"job:{job_id}:progress", json.dumps(message))
    except Exception:
        pass


@celery_app.task(bind=True)
def batch_process_task(self, job_ids: list):
    """
    Process multiple videos in sequence.
    
    Useful for batch processing when submitted together.
    """
    results = []
    for job_id in job_ids:
        try:
            result = process_video_task.apply_async(args=[job_id])
            results.append({"job_id": job_id, "task_id": result.id})
        except Exception as e:
            results.append({"job_id": job_id, "error": str(e)})
    
    return results
