"""
Detection Job Models
====================

Video processing jobs and their configuration.
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, Integer, Float, ForeignKey, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class VideoUpload(BaseModel):
    """
    Uploaded video file metadata.
    
    Videos are stored on disk/S3, this tracks metadata.
    """
    
    __tablename__ = "video_uploads"
    
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    
    # File info
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)  # bytes
    
    # Video metadata
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)
    resolution: Mapped[Optional[str]] = mapped_column(String(20))  # "1920x1080"
    fps: Mapped[Optional[float]] = mapped_column(Float)
    codec: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Thumbnail
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Status: uploading, processing, ready, error
    status: Mapped[str] = mapped_column(String(20), default="uploading", index=True)
    
    # Relationships
    jobs: Mapped[List["DetectionJob"]] = relationship(
        "DetectionJob", back_populates="video"
    )
    
    def __repr__(self):
        return f"<VideoUpload {self.original_filename}>"


class DetectionJob(BaseModel):
    """
    Detection processing job.
    
    Represents a single video processing task with configuration
    and progress tracking.
    """
    
    __tablename__ = "detection_jobs"
    
    # References
    video_id: Mapped[int] = mapped_column(ForeignKey("video_uploads.id"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    bus_id: Mapped[Optional[int]] = mapped_column(ForeignKey("buses.id"), index=True)
    
    # Status: queued, processing, completed, failed, cancelled
    status: Mapped[str] = mapped_column(String(50), default="queued", index=True)
    
    # Progress (0.0 - 100.0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_frame: Mapped[int] = mapped_column(Integer, default=0)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Processing timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Configuration
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    # Example config:
    # {
    #     "model": "yolov8m.pt",
    #     "confidence": 0.5,
    #     "frame_skip": 2,
    #     "zones_enabled": true,
    #     "crowded_mode": true
    # }
    
    # Results
    output_video_path: Mapped[Optional[str]] = mapped_column(String(500))
    report_path: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Error info
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Celery task ID for tracking
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    
    # Relationships
    video: Mapped["VideoUpload"] = relationship("VideoUpload", back_populates="jobs")
    user: Mapped["User"] = relationship("User", back_populates="jobs")
    bus: Mapped[Optional["Bus"]] = relationship("Bus", back_populates="jobs")
    
    detections: Mapped[List["Detection"]] = relationship(
        "Detection", back_populates="job", cascade="all, delete-orphan"
    )
    analytics: Mapped[Optional["AnalyticsSummary"]] = relationship(
        "AnalyticsSummary", back_populates="job", uselist=False, cascade="all, delete-orphan"
    )
    alerts: Mapped[List["Alert"]] = relationship(
        "Alert", back_populates="job", cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<DetectionJob {self.id} - {self.status}>"


# Import for type hints
from .user import User
from .bus import Bus
from .detection import Detection
from .analytics import AnalyticsSummary
from .alert import Alert
