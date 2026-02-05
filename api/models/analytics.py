"""
Analytics Models
================

Aggregated analytics and reports.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Integer, Float, ForeignKey, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class AnalyticsSummary(BaseModel):
    """
    Aggregated analytics for a detection job.
    
    Computed after job completion.
    """
    
    __tablename__ = "analytics_summaries"
    
    job_id: Mapped[int] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )
    
    # People counts
    total_unique_people: Mapped[int] = mapped_column(Integer, default=0)
    total_detections: Mapped[int] = mapped_column(Integer, default=0)
    
    # Occupancy
    peak_occupancy: Mapped[int] = mapped_column(Integer, default=0)
    peak_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    avg_occupancy: Mapped[Optional[float]] = mapped_column(Float)
    
    # Dwell time (in seconds)
    avg_dwell_time: Mapped[Optional[float]] = mapped_column(Float)
    min_dwell_time: Mapped[Optional[float]] = mapped_column(Float)
    max_dwell_time: Mapped[Optional[float]] = mapped_column(Float)
    
    # Entry/Exit
    total_entries: Mapped[int] = mapped_column(Integer, default=0)
    total_exits: Mapped[int] = mapped_column(Integer, default=0)
    
    # Processing stats
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    avg_fps: Mapped[Optional[float]] = mapped_column(Float)
    
    # Anomalies
    anomalies_detected: Mapped[int] = mapped_column(Integer, default=0)
    
    # Detailed data (time series, hourly breakdown)
    hourly_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Example:
    # {
    #     "occupancy": {"0": 5, "1": 8, ...},
    #     "entries": {"0": 2, "1": 5, ...},
    #     "exits": {"0": 1, "1": 3, ...}
    # }
    
    # Heatmap data (2D array)
    heatmap_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # When computed
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow
    )
    
    # Relationships
    job: Mapped["DetectionJob"] = relationship("DetectionJob", back_populates="analytics")
    
    def __repr__(self):
        return f"<AnalyticsSummary job={self.job_id}>"


class Report(BaseModel):
    """
    Generated report document.
    
    PDF/Excel reports created from analytics.
    """
    
    __tablename__ = "reports"
    
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    
    # Report type: daily, weekly, monthly, custom
    report_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Date range
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Configuration (which buses, metrics, etc.)
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    
    # Output files
    file_path: Mapped[Optional[str]] = mapped_column(String(500))
    file_format: Mapped[str] = mapped_column(String(10), default="pdf")  # pdf, xlsx, csv
    file_size: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Status: generating, completed, failed
    status: Mapped[str] = mapped_column(String(20), default="generating", index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Schedule info (for recurring reports)
    is_scheduled: Mapped[bool] = mapped_column(default=False)
    schedule_cron: Mapped[Optional[str]] = mapped_column(String(100))  # Cron expression
    
    def __repr__(self):
        return f"<Report {self.title}>"


# Import for type hints
from .job import DetectionJob
