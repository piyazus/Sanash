"""
Video Upload Endpoints
======================
"""

import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.config import settings
from api.core.database import get_db
from api.core.dependencies import get_current_user
from api.models.user import User
from api.models.job import VideoUpload
from api.schemas.job import VideoUploadResponse, VideoListItem

router = APIRouter()


def get_video_metadata(file_path: str) -> dict:
    """Extract video metadata using OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            return {}
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "duration_seconds": duration,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "codec": "h264",  # Simplified
        }
    except Exception:
        return {}


def generate_thumbnail(video_path: str, output_path: str, time_seconds: float = 5.0):
    """Generate thumbnail at specified time."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = int(time_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Resize to thumbnail size
            thumb = cv2.resize(frame, (320, 180))
            cv2.imwrite(output_path, thumb)
        
        cap.release()
    except Exception:
        pass


@router.post("/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a video file for processing.
    
    - Supported formats: mp4, avi, mov, mkv
    - Max size: 5GB
    - Max duration: 8 hours
    """
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_VIDEO_EXTENSIONS}",
        )
    
    # Generate unique filename
    unique_id = uuid.uuid4().hex[:12]
    filename = f"{unique_id}{ext}"
    file_path = settings.UPLOAD_DIR / filename
    
    # Save file
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Check file size
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )
    
    # Extract metadata
    metadata = get_video_metadata(str(file_path))
    
    # Check duration
    if metadata.get("duration_seconds", 0) > settings.MAX_VIDEO_DURATION_HOURS * 3600:
        os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video too long. Max duration: {settings.MAX_VIDEO_DURATION_HOURS} hours",
        )
    
    # Generate thumbnail in background
    thumb_path = settings.UPLOAD_DIR / f"{unique_id}_thumb.jpg"
    background_tasks.add_task(generate_thumbnail, str(file_path), str(thumb_path))
    
    # Create database record
    video = VideoUpload(
        user_id=user.id,
        filename=filename,
        original_filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        duration_seconds=metadata.get("duration_seconds"),
        resolution=metadata.get("resolution"),
        fps=metadata.get("fps"),
        codec=metadata.get("codec"),
        thumbnail_path=str(thumb_path),
        status="ready",
    )
    
    db.add(video)
    await db.commit()
    await db.refresh(video)
    
    return video


@router.get("", response_model=list[VideoListItem])
async def list_videos(
    skip: int = 0,
    limit: int = 20,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's uploaded videos."""
    result = await db.execute(
        select(VideoUpload)
        .where(VideoUpload.user_id == user.id)
        .order_by(VideoUpload.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    videos = result.scalars().all()
    return videos


@router.get("/{video_id}", response_model=VideoUploadResponse)
async def get_video(
    video_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get video details."""
    result = await db.execute(
        select(VideoUpload).where(
            VideoUpload.id == video_id,
            VideoUpload.user_id == user.id
        )
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    return video


@router.delete("/{video_id}")
async def delete_video(
    video_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a video and its files."""
    result = await db.execute(
        select(VideoUpload).where(
            VideoUpload.id == video_id,
            VideoUpload.user_id == user.id
        )
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Delete files
    try:
        if os.path.exists(video.file_path):
            os.remove(video.file_path)
        if video.thumbnail_path and os.path.exists(video.thumbnail_path):
            os.remove(video.thumbnail_path)
    except Exception:
        pass
    
    await db.delete(video)
    await db.commit()
    
    return {"success": True, "message": "Video deleted"}
