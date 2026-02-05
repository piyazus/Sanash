"""
Multi-Camera Tracking Models
============================

Models for cross-camera person tracking.
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import Integer, Float, ForeignKey, DateTime, String, BigInteger, Boolean, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class GlobalTrack(BaseModel):
    """
    A person tracked across multiple cameras.
    
    Links local track IDs from different cameras to
    one consistent identity.
    """
    
    __tablename__ = "global_tracks"
    
    # First camera seen
    first_camera_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("cameras.id"),
        index=True
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    
    # Last camera seen
    last_camera_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("cameras.id"),
        index=True
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    
    # Journey statistics
    total_cameras_visited: Mapped[int] = mapped_column(Integer, default=1)
    total_dwell_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    
    # Journey path as JSON array
    # [{"camera": "front", "entry": "2026-01-30T08:15:30", "exit": "2026-01-30T08:16:45"}]
    journey_path: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # ReID features (serialized numpy array)
    reid_features: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    exited: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Job reference (optional - for batch processing)
    job_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="SET NULL"),
        index=True
    )
    
    # Relationships
    track_mappings: Mapped[List["TrackMapping"]] = relationship(
        "TrackMapping",
        back_populates="global_track",
        cascade="all, delete-orphan"
    )
    handoffs: Mapped[List["CameraHandoff"]] = relationship(
        "CameraHandoff",
        back_populates="global_track",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<GlobalTrack id={self.id} cameras={self.total_cameras_visited}>"


class TrackMapping(BaseModel):
    """
    Maps local track IDs to global tracks.
    
    One global track can have multiple local track IDs
    (one per camera it appeared in).
    """
    
    __tablename__ = "track_mappings"
    
    global_track_id: Mapped[int] = mapped_column(
        ForeignKey("global_tracks.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id"),
        nullable=False,
        index=True
    )
    
    local_track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # ReID matching confidence when mapping was created
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    
    # When this mapping was first established
    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    
    # Frame range in this camera
    first_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    last_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Relationships
    global_track: Mapped["GlobalTrack"] = relationship(
        "GlobalTrack",
        back_populates="track_mappings"
    )
    
    def __repr__(self):
        return f"<TrackMapping global={self.global_track_id} cam={self.camera_id} local={self.local_track_id}>"


class CameraHandoff(BaseModel):
    """
    Record of a person transitioning between cameras.
    
    Used for flow analysis and validation.
    """
    
    __tablename__ = "camera_handoffs"
    
    global_track_id: Mapped[int] = mapped_column(
        ForeignKey("global_tracks.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    from_camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id"),
        nullable=False
    )
    to_camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id"),
        nullable=False
    )
    
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )
    
    # Validation
    reid_confidence: Mapped[Optional[float]] = mapped_column(Float)
    spatial_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    temporal_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Time between cameras
    time_gap_seconds: Mapped[Optional[float]] = mapped_column(Float)
    
    # Position at exit/entry
    exit_position: Mapped[Optional[dict]] = mapped_column(JSONB)  # {x, y}
    entry_position: Mapped[Optional[dict]] = mapped_column(JSONB)  # {x, y}
    
    # Job reference
    job_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="SET NULL"),
        index=True
    )
    
    # Relationships
    global_track: Mapped["GlobalTrack"] = relationship(
        "GlobalTrack",
        back_populates="handoffs"
    )
    
    @property
    def is_valid(self) -> bool:
        """Check if handoff passed all validations."""
        return self.spatial_valid and self.temporal_valid
    
    def __repr__(self):
        return f"<CameraHandoff {self.from_camera_id}->{self.to_camera_id}>"


class MultiCameraSummary(BaseModel):
    """
    Summary statistics for multi-camera job.
    """
    
    __tablename__ = "multi_camera_summaries"
    
    job_id: Mapped[int] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True
    )
    
    # Track counts
    total_global_tracks: Mapped[int] = mapped_column(Integer, default=0)
    tracks_single_camera: Mapped[int] = mapped_column(Integer, default=0)
    tracks_multi_camera: Mapped[int] = mapped_column(Integer, default=0)
    
    # Handoff statistics
    total_handoffs: Mapped[int] = mapped_column(Integer, default=0)
    valid_handoffs: Mapped[int] = mapped_column(Integer, default=0)
    invalid_handoffs: Mapped[int] = mapped_column(Integer, default=0)
    
    # Flow matrix as JSON
    # {"front_to_middle": 45, "middle_to_rear": 38, ...}
    flow_matrix: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Average transition times
    avg_transition_times: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Camera-specific stats
    camera_stats: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Processing info
    reid_model_version: Mapped[Optional[str]] = mapped_column(String(50))
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow
    )
    
    def __repr__(self):
        return f"<MultiCameraSummary job={self.job_id} tracks={self.total_global_tracks}>"
