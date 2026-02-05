"""
Mobile API Schemas
==================

Pydantic models for mobile app endpoints.
"""

from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# BUS POSITIONS
# =============================================================================

class BusPositionResponse(BaseModel):
    """Real-time bus position with occupancy."""
    
    bus_id: int
    bus_number: str
    route_id: Optional[int] = None
    route_name: Optional[str] = None
    
    # Position
    latitude: float
    longitude: float
    speed: Optional[float] = Field(None, description="Speed in km/h")
    heading: Optional[float] = Field(None, description="Direction 0-360°")
    
    # Occupancy
    current_occupancy: int
    capacity: int
    percentage: int
    status: Literal["available", "getting_full", "crowded"]
    color: Literal["green", "yellow", "red"]
    
    last_updated: datetime


class AllBusPositionsResponse(BaseModel):
    """All bus positions for map."""
    
    total_buses: int
    buses: List[BusPositionResponse]
    timestamp: datetime


# =============================================================================
# BUS STOPS
# =============================================================================

class BusStopResponse(BaseModel):
    """Bus stop information."""
    
    stop_id: int
    name: str
    latitude: float
    longitude: float
    address: Optional[str] = None
    route_ids: List[int] = []
    status: str = "active"


class AllStopsResponse(BaseModel):
    """All bus stops for map."""
    
    total_stops: int
    stops: List[BusStopResponse]


# =============================================================================
# ARRIVALS / ETA
# =============================================================================

class BusArrivalResponse(BaseModel):
    """Incoming bus at a stop."""
    
    bus_id: int
    bus_number: str
    route_id: Optional[int] = None
    route_name: Optional[str] = None
    
    # ETA
    eta_minutes: int = Field(..., description="Estimated time of arrival in minutes")
    eta_time: datetime = Field(..., description="Estimated arrival time")
    distance_km: float = Field(..., description="Distance from stop in km")
    
    # Occupancy
    current_occupancy: int
    capacity: int
    percentage: int
    status: Literal["available", "getting_full", "crowded"]
    color: Literal["green", "yellow", "red"]


class StopArrivalsResponse(BaseModel):
    """All arriving buses at a stop."""
    
    stop_id: int
    stop_name: str
    arrivals: List[BusArrivalResponse]
    total_arrivals: int


# =============================================================================
# ROUTE PATH
# =============================================================================

class RoutePathPoint(BaseModel):
    """Single point on route path."""
    lat: float
    lon: float


class RoutePathResponse(BaseModel):
    """Route polyline for map."""
    
    route_id: int
    route_name: str
    path: List[RoutePathPoint]
    stops: List[BusStopResponse]


# =============================================================================
# SMART ROUTING
# =============================================================================

class SmartRouteRequest(BaseModel):
    """Request for smart route suggestion."""
    
    origin_lat: float
    origin_lon: float
    destination_lat: float
    destination_lon: float
    prefer_empty: bool = Field(default=True, description="Prefer less crowded buses")


class RouteStep(BaseModel):
    """Single step in a route."""
    
    type: Literal["walk", "bus"]
    instruction: str
    distance_km: float
    duration_minutes: int
    
    # For bus steps
    bus_id: Optional[int] = None
    route_name: Optional[str] = None
    from_stop: Optional[str] = None
    to_stop: Optional[str] = None
    occupancy_percentage: Optional[int] = None


class SmartRouteResponse(BaseModel):
    """Suggested route from A to B."""
    
    total_duration_minutes: int
    total_distance_km: float
    occupancy_score: int = Field(..., description="0-100, lower is less crowded")
    steps: List[RouteStep]
