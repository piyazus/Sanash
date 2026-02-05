"""
Pydantic Schemas for Buses, Cameras, and Zones
==============================================
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# BUS SCHEMAS
# =============================================================================

class BusCreate(BaseModel):
    """Create a new bus."""
    number: str
    model: Optional[str] = None
    capacity: Optional[int] = None


class BusUpdate(BaseModel):
    """Update bus."""
    number: Optional[str] = None
    model: Optional[str] = None
    capacity: Optional[int] = None
    status: Optional[str] = None


class BusResponse(BaseModel):
    """Bus response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    number: str
    model: Optional[str]
    capacity: Optional[int]
    status: str
    cameras_count: Optional[int] = None
    created_at: datetime


class BusListItem(BaseModel):
    """Bus in list view."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    number: str
    status: str
    cameras_count: Optional[int] = None


# =============================================================================
# CAMERA SCHEMAS
# =============================================================================

class CameraCreate(BaseModel):
    """Create a new camera."""
    bus_id: int
    name: str
    position: str
    resolution: Optional[str] = None
    fps: Optional[int] = None


class CameraUpdate(BaseModel):
    """Update camera."""
    name: Optional[str] = None
    position: Optional[str] = None
    status: Optional[str] = None
    calibration_data: Optional[Dict[str, Any]] = None


class CameraResponse(BaseModel):
    """Camera response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    bus_id: int
    name: str
    position: str
    resolution: Optional[str]
    fps: Optional[int]
    status: str
    zones_count: Optional[int] = None
    created_at: datetime


class CameraStatus(BaseModel):
    """Camera status check."""
    id: int
    name: str
    status: str
    last_seen: Optional[datetime] = None
    fps: Optional[float] = None


# =============================================================================
# ZONE SCHEMAS
# =============================================================================

class ZonePoint(BaseModel):
    """Point in zone polygon."""
    x: int
    y: int


class ZoneCreate(BaseModel):
    """Create a new zone."""
    camera_id: int
    name: str
    zone_type: str  # entry, exit, seating, restricted
    polygon: List[ZonePoint]
    entry_direction: Optional[Dict[str, float]] = None  # {x: 0, y: -1}
    color: str = "#00FF00"


class ZoneUpdate(BaseModel):
    """Update zone."""
    name: Optional[str] = None
    zone_type: Optional[str] = None
    polygon: Optional[List[ZonePoint]] = None
    entry_direction: Optional[Dict[str, float]] = None
    color: Optional[str] = None
    is_active: Optional[bool] = None


class ZoneResponse(BaseModel):
    """Zone response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    camera_id: int
    name: str
    zone_type: str
    polygon: List[Dict[str, int]]
    entry_direction: Optional[Dict[str, float]]
    color: str
    is_active: bool
    created_at: datetime


class ZoneCrossingTest(BaseModel):
    """Test zone crossing detection."""
    zone_id: int
    track_path: List[ZonePoint]  # Sequence of points


class ZoneCrossingResult(BaseModel):
    """Result of crossing test."""
    crossed: bool
    direction: Optional[str] = None  # "in" or "out"
    crossing_point: Optional[ZonePoint] = None


# =============================================================================
# ROUTE SCHEMAS
# =============================================================================

class RouteCreate(BaseModel):
    """Create a new route."""
    name: str
    description: Optional[str] = None
    start_point: Optional[str] = None
    end_point: Optional[str] = None
    estimated_duration: Optional[int] = None


class RouteUpdate(BaseModel):
    """Update route."""
    name: Optional[str] = None
    description: Optional[str] = None
    start_point: Optional[str] = None
    end_point: Optional[str] = None
    status: Optional[str] = None


class RouteResponse(BaseModel):
    """Route response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    description: Optional[str]
    start_point: Optional[str]
    end_point: Optional[str]
    estimated_duration: Optional[int]
    status: str
    created_at: datetime
