"""
Pydantic Schemas for Jobs and Videos
====================================
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# VIDEO SCHEMAS
# =============================================================================

class VideoMetadata(BaseModel):
    """Video file metadata."""
    duration_seconds: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    codec: Optional[str] = None


class VideoUploadResponse(BaseModel):
    """Response after video upload."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    filename: str
    original_filename: str
    file_size: int
    status: str
    metadata: Optional[VideoMetadata] = None
    thumbnail_url: Optional[str] = None
    created_at: datetime


class VideoListItem(BaseModel):
    """Video in list view."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    original_filename: str
    duration_seconds: Optional[float]
    status: str
    created_at: datetime


# =============================================================================
# JOB SCHEMAS
# =============================================================================

class JobConfig(BaseModel):
    """Detection job configuration."""
    model: str = "yolov8m.pt"
    confidence: float = Field(default=0.5, ge=0.1, le=1.0)
    frame_skip: int = Field(default=2, ge=1, le=30)
    zones_enabled: bool = True
    crowded_mode: bool = True
    generate_report: bool = True


class JobCreate(BaseModel):
    """Create a new detection job."""
    video_id: int
    bus_id: Optional[int] = None
    config: JobConfig = Field(default_factory=JobConfig)


class JobUpdate(BaseModel):
    """Update job (limited fields)."""
    status: Optional[str] = None


class JobProgress(BaseModel):
    """Real-time job progress."""
    status: str
    progress: float
    current_frame: int
    total_frames: Optional[int]
    eta_seconds: Optional[float] = None
    detections_count: Optional[int] = None


class JobResponse(BaseModel):
    """Full job response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    video_id: int
    user_id: int
    bus_id: Optional[int]
    status: str
    progress: float
    current_frame: int
    total_frames: Optional[int]
    config: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]


class JobListItem(BaseModel):
    """Job in list view."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    video_filename: Optional[str] = None
    status: str
    progress: float
    created_at: datetime


class JobResults(BaseModel):
    """Job results after completion."""
    job_id: int
    status: str
    output_video_url: Optional[str]
    report_url: Optional[str]
    detections_csv_url: Optional[str]
    analytics: Optional[Dict[str, Any]]


# =============================================================================
# WEBSOCKET MESSAGES
# =============================================================================

class WSProgressMessage(BaseModel):
    """WebSocket progress update."""
    type: str = "progress"
    frame: int
    people_count: int
    fps: float
    progress: float


class WSAnomalyMessage(BaseModel):
    """WebSocket anomaly alert."""
    type: str = "anomaly"
    severity: str
    description: str
    timestamp: datetime


class WSCompleteMessage(BaseModel):
    """WebSocket job completion."""
    type: str = "complete"
    total_time_seconds: float
    results_url: str
