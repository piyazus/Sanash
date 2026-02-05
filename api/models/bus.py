"""
Bus & Camera Models
===================

Fleet management: buses, routes, and cameras.
"""

from typing import Optional, List, Any

from sqlalchemy import String, Integer, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class Bus(BaseModel):
    """
    Bus vehicle in the fleet.
    
    Each bus can have multiple cameras installed.
    """
    
    __tablename__ = "buses"
    
    number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    model: Mapped[Optional[str]] = mapped_column(String(100))
    capacity: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Status: active, maintenance, retired
    status: Mapped[str] = mapped_column(String(20), default="active", index=True)
    
    # Additional metadata
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, default=dict)
    
    # Relationships
    cameras: Mapped[List["Camera"]] = relationship(
        "Camera", back_populates="bus", cascade="all, delete-orphan"
    )
    jobs: Mapped[List["DetectionJob"]] = relationship(
        "DetectionJob", back_populates="bus"
    )
    
    def __repr__(self):
        return f"<Bus {self.number}>"


class Route(BaseModel):
    """
    Bus route definition.
    
    Routes can be used to group and compare analytics.
    """
    
    __tablename__ = "routes"
    
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    start_point: Mapped[Optional[str]] = mapped_column(String(200))
    end_point: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Estimated duration in minutes
    estimated_duration: Mapped[Optional[int]] = mapped_column(Integer)
    
    # GeoJSON of route path (optional)
    geometry: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Status: active, inactive
    status: Mapped[str] = mapped_column(String(20), default="active")
    
    def __repr__(self):
        return f"<Route {self.name}>"


class Camera(BaseModel):
    """
    Camera installed on a bus.
    
    Multiple cameras per bus for comprehensive coverage.
    """
    
    __tablename__ = "cameras"
    
    bus_id: Mapped[int] = mapped_column(ForeignKey("buses.id"), nullable=False, index=True)
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Position: front, middle, rear, door_front, door_rear
    position: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Technical specs
    resolution: Mapped[Optional[str]] = mapped_column(String(20))  # e.g., "1920x1080"
    fps: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Status: active, offline, maintenance
    status: Mapped[str] = mapped_column(String(20), default="active")
    
    # Calibration data for zone mapping
    calibration_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Relationships
    bus: Mapped["Bus"] = relationship("Bus", back_populates="cameras")
    zones: Mapped[List["Zone"]] = relationship(
        "Zone", back_populates="camera", cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Camera {self.name} on Bus {self.bus_id}>"


class Zone(BaseModel):
    """
    Detection zone defined on a camera view.
    
    Used for entry/exit counting and area monitoring.
    """
    
    __tablename__ = "zones"
    
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), nullable=False, index=True)
    
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Zone type: entry, exit, seating, restricted, counting
    zone_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Polygon points as JSON array: [{x: 0, y: 0}, {x: 100, y: 0}, ...]
    polygon: Mapped[dict] = mapped_column(JSONB, nullable=False)
    
    # Entry direction vector for directional zones
    entry_direction: Mapped[Optional[dict]] = mapped_column(JSONB)  # {x: 0, y: -1}
    
    # Display color as hex
    color: Mapped[str] = mapped_column(String(7), default="#00FF00")
    
    # Whether zone is active
    is_active: Mapped[bool] = mapped_column(default=True)
    
    # Relationships
    camera: Mapped["Camera"] = relationship("Camera", back_populates="zones")
    crossings: Mapped[List["ZoneCrossing"]] = relationship(
        "ZoneCrossing", back_populates="zone", cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Zone {self.name}>"


# Import for type hints
from .job import DetectionJob
from .detection import ZoneCrossing
