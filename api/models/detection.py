"""
Detection Models
================

Individual detections and zone crossings.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Integer, Float, ForeignKey, DateTime, String, BigInteger
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class Detection(BaseModel):
    """
    Single person detection in a frame.
    
    High-volume table - optimized for bulk inserts.
    """
    
    __tablename__ = "detections"
    
    # Use BigInteger for high-volume data
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    job_id: Mapped[int] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Tracking
    track_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    
    # Bounding box: {x1, y1, x2, y2} or {x, y, width, height}
    bbox: Mapped[dict] = mapped_column(JSONB, nullable=False)
    
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Camera reference (optional)
    camera_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("cameras.id"),
        index=True
    )
    
    # Relationships
    job: Mapped["DetectionJob"] = relationship("DetectionJob", back_populates="detections")
    
    # Indexes defined at table level for composite indexes
    __table_args__ = (
        # Index for querying detections by job and frame
        # CREATE INDEX idx_detections_job_frame ON detections(job_id, frame_number)
    )
    
    def __repr__(self):
        return f"<Detection frame={self.frame_number} track={self.track_id}>"


class ZoneCrossing(BaseModel):
    """
    Record of a person crossing a zone boundary.
    
    Used for entry/exit counting.
    """
    
    __tablename__ = "zone_crossings"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    
    zone_id: Mapped[int] = mapped_column(
        ForeignKey("zones.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    job_id: Mapped[int] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    track_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Direction: 'in' or 'out'
    direction: Mapped[str] = mapped_column(String(10), nullable=False)
    
    # Position at crossing
    position: Mapped[Optional[dict]] = mapped_column(JSONB)  # {x, y}
    
    # Dwell time if exit (seconds since entry)
    dwell_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    zone: Mapped["Zone"] = relationship("Zone", back_populates="crossings")
    
    def __repr__(self):
        return f"<ZoneCrossing zone={self.zone_id} {self.direction}>"


class Track(BaseModel):
    """
    Complete track trajectory for a person.
    
    Aggregated from individual detections for journey analysis.
    """
    
    __tablename__ = "tracks"
    
    job_id: Mapped[int] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # First and last appearance
    first_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    last_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Journey info
    entry_zone_id: Mapped[Optional[int]] = mapped_column(ForeignKey("zones.id"))
    exit_zone_id: Mapped[Optional[int]] = mapped_column(ForeignKey("zones.id"))
    
    # Total dwell time in seconds
    dwell_time: Mapped[Optional[float]] = mapped_column(Float)
    
    # Trajectory as array of points (sampled)
    trajectory: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Average confidence
    avg_confidence: Mapped[Optional[float]] = mapped_column(Float)
    
    # Unique constraint on job + track_id
    __table_args__ = (
        # CREATE UNIQUE INDEX idx_tracks_job_track ON tracks(job_id, track_id)
    )
    
    def __repr__(self):
        return f"<Track job={self.job_id} id={self.track_id}>"


# Import for type hints
from .job import DetectionJob
from .bus import Zone
