"""
API Models Package
==================

SQLAlchemy ORM models for all entities.
"""

from .base import BaseModel, TimestampMixin
from .user import User, ApiKey, RefreshToken
from .bus import Bus, Route, Camera, Zone
from .job import VideoUpload, DetectionJob
from .detection import Detection, ZoneCrossing, Track
from .analytics import AnalyticsSummary, Report
from .alert import Alert, Notification

__all__ = [
    # Base
    "BaseModel",
    "TimestampMixin",
    # Users
    "User",
    "ApiKey",
    "RefreshToken",
    # Fleet
    "Bus",
    "Route",
    "Camera",
    "Zone",
    # Jobs
    "VideoUpload",
    "DetectionJob",
    # Detections
    "Detection",
    "ZoneCrossing",
    "Track",
    # Analytics
    "AnalyticsSummary",
    "Report",
    # Alerts
    "Alert",
    "Notification",
]
