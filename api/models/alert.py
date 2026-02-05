"""
Alert Models
============

System alerts and notifications.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Integer, ForeignKey, DateTime, String, Text, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class Alert(BaseModel):
    """
    System alert triggered during detection.
    
    Types: overcrowding, loitering, safety, system
    """
    
    __tablename__ = "alerts"
    
    job_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="SET NULL"),
        index=True
    )
    
    # Alert type: overcrowding, loitering, safety, unauthorized_zone, system
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Severity: low, medium, high, critical
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    # When the event occurred
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )
    
    # Frame number in video (if applicable)
    frame_number: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Additional context
    details: Mapped[Optional[dict]] = mapped_column(JSONB)
    # Example:
    # {
    #     "zone_id": 5,
    #     "occupancy": 45,
    #     "threshold": 30,
    #     "track_ids": [101, 102, 103]
    # }
    
    # Acknowledgment
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    acknowledged_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Resolution
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    job: Mapped[Optional["DetectionJob"]] = relationship("DetectionJob", back_populates="alerts")
    
    def __repr__(self):
        return f"<Alert {self.alert_type} - {self.severity}>"


class Notification(BaseModel):
    """
    Notification sent to user via email/SMS/webhook.
    
    Tracks delivery status of alert notifications.
    """
    
    __tablename__ = "notifications"
    
    alert_id: Mapped[int] = mapped_column(
        ForeignKey("alerts.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False
    )
    
    # Channel: email, sms, webhook, push
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Recipient (email address, phone number, webhook URL)
    recipient: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Status: pending, sent, delivered, failed
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    
    # Timestamps
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Error info
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    def __repr__(self):
        return f"<Notification {self.channel} to {self.recipient}>"


# Import for type hints
from .job import DetectionJob
