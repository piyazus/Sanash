"""
Bus Stop Models
===============

Bus stops for ETA calculations.
"""

from typing import Optional, List
from datetime import datetime

from sqlalchemy import String, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class BusStop(BaseModel):
    """
    Bus stop location.
    
    Used for ETA calculations and virtual bus stop boards.
    """
    
    __tablename__ = "bus_stops"
    
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    
    # Location
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Address info
    address: Mapped[Optional[str]] = mapped_column(String(300))
    
    # Routes that pass through this stop (stored as JSON array)
    route_ids: Mapped[Optional[list]] = mapped_column(JSONB, default=list)
    
    # Order in route (optional)
    sequence_number: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Status: active, closed, maintenance
    status: Mapped[str] = mapped_column(String(20), default="active")
    
    # Amenities: shelter, bench, etc.
    amenities: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    def __repr__(self):
        return f"<BusStop {self.name}>"


class BusPosition(BaseModel):
    """
    Real-time bus GPS position.
    
    Updated frequently by edge devices.
    """
    
    __tablename__ = "bus_positions"
    
    bus_id: Mapped[int] = mapped_column(
        ForeignKey("buses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        unique=True  # One position per bus
    )
    
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Speed in km/h
    speed: Mapped[Optional[float]] = mapped_column(Float)
    
    # Heading in degrees (0-360)
    heading: Mapped[Optional[float]] = mapped_column(Float)
    
    # Timestamp of GPS reading
    timestamp: Mapped[datetime] = mapped_column(nullable=False)
    
    # Accuracy in meters
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationship
    bus: Mapped["Bus"] = relationship("Bus", backref="position")
    
    def __repr__(self):
        return f"<BusPosition bus={self.bus_id} ({self.latitude}, {self.longitude})>"


# Import for type hints
from .bus import Bus
