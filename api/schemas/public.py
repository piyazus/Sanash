"""
Public API Schemas
==================

Pydantic models for public occupancy endpoints.
"""

from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class BusOccupancyResponse(BaseModel):
    """Response for single bus occupancy."""
    
    bus_id: int
    bus_number: str
    route_id: Optional[int] = None
    route_name: Optional[str] = None
    
    current_occupancy: int = Field(..., ge=0, description="Current number of passengers")
    capacity: int = Field(..., gt=0, description="Maximum bus capacity")
    percentage: int = Field(..., ge=0, le=100, description="Occupancy percentage")
    
    status: Literal["available", "getting_full", "crowded"] = Field(
        ..., description="Occupancy status"
    )
    color: Literal["green", "yellow", "red"] = Field(
        ..., description="Status color for UI"
    )
    
    last_updated: datetime = Field(..., description="When data was last updated")
    distance_km: Optional[float] = Field(None, description="Distance in km (for nearby endpoint)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bus_id": 45,
                "bus_number": "45",
                "route_id": 12,
                "route_name": "Route 12",
                "current_occupancy": 23,
                "capacity": 50,
                "percentage": 46,
                "status": "available",
                "color": "green",
                "last_updated": "2026-01-30T10:15:30Z"
            }
        }


class BusListResponse(BaseModel):
    """Response for list of buses with occupancy."""
    
    route_id: Optional[int] = None
    route_name: Optional[str] = None
    total_buses: int
    buses: List[BusOccupancyResponse]


class NearbyBusesResponse(BaseModel):
    """Response for nearby buses search."""
    
    latitude: float
    longitude: float
    radius_km: float
    total_buses: int
    buses: List[BusOccupancyResponse]


class BusLocation(BaseModel):
    """Bus location data."""
    
    bus_id: int
    bus_number: str
    latitude: float
    longitude: float
    last_updated: datetime


class RouteResponse(BaseModel):
    """Response for a single route."""
    
    route_id: int
    route_name: str
    total_buses: int = Field(..., ge=0, description="Number of buses on this route")
    avg_occupancy_percentage: int = Field(..., ge=0, le=100, description="Average occupancy %")
    status: str = Field(default="operational", description="Route status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "route_id": 12,
                "route_name": "Central Station - Airport",
                "total_buses": 8,
                "avg_occupancy_percentage": 65,
                "status": "operational"
            }
        }


class RouteListResponse(BaseModel):
    """Response for list of all routes."""
    
    routes: List[RouteResponse]
    total_routes: int
