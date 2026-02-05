"""
Mobile API Endpoints
====================

Endpoints optimized for React Native mobile app.

Endpoints:
- GET /buses/positions - All bus GPS positions for map
- GET /stops - All bus stops
- GET /stops/{stop_id}/arrivals - ETAs at a stop
- GET /routes/{route_id}/path - Route polyline
- WS /ws/buses - Live position updates
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.core.database import get_db
from api.core.cache import get_redis, Cache
from api.models.bus import Bus, Route
from api.models.location import BusStop, BusPosition
from api.services.occupancy_service import (
    OccupancyService,
    haversine_distance,
    calculate_percentage,
    calculate_status,
)
from api.schemas.mobile import (
    BusPositionResponse,
    AllBusPositionsResponse,
    BusStopResponse,
    AllStopsResponse,
    BusArrivalResponse,
    StopArrivalsResponse,
    RoutePathResponse,
    RoutePathPoint,
)


router = APIRouter()

CACHE_TTL = 5  # 5 seconds


# =============================================================================
# GET /buses/positions - All bus GPS positions
# =============================================================================

@router.get(
    "/buses/positions",
    response_model=AllBusPositionsResponse,
    summary="All bus positions",
    description="Get real-time GPS positions of all active buses for map display.",
)
async def get_all_bus_positions(
    db: AsyncSession = Depends(get_db),
):
    """
    Returns all active bus positions with occupancy data.
    Used for displaying bus markers on the map.
    """
    # Try cache
    redis = await get_redis()
    cache = Cache(redis)
    cache_key = "mobile:positions:all"
    
    cached = await cache.get(cache_key)
    if cached:
        return cached
    
    # Get all buses with positions
    result = await db.execute(
        select(Bus)
        .where(Bus.status == "active")
        .options(selectinload(Bus.cameras))
    )
    buses = result.scalars().all()
    
    # Get positions
    positions_result = await db.execute(select(BusPosition))
    positions_map = {p.bus_id: p for p in positions_result.scalars().all()}
    
    # Get occupancy service
    occupancy_service = OccupancyService(db)
    
    buses_data = []
    for bus in buses:
        # Get position
        pos = positions_map.get(bus.id)
        if not pos:
            # Try metadata fallback
            if bus.metadata_ and "location" in bus.metadata_:
                lat = bus.metadata_["location"].get("latitude")
                lon = bus.metadata_["location"].get("longitude")
            else:
                continue  # Skip buses without position
        else:
            lat = pos.latitude
            lon = pos.longitude
        
        # Get occupancy
        occ_data = await occupancy_service.get_bus_occupancy(bus.id)
        if not occ_data:
            continue
        
        buses_data.append({
            "bus_id": bus.id,
            "bus_number": bus.number,
            "route_id": occ_data.get("route_id"),
            "route_name": occ_data.get("route_name"),
            "latitude": lat,
            "longitude": lon,
            "speed": pos.speed if pos else None,
            "heading": pos.heading if pos else None,
            "current_occupancy": occ_data["current_occupancy"],
            "capacity": occ_data["capacity"],
            "percentage": occ_data["percentage"],
            "status": occ_data["status"],
            "color": occ_data["color"],
            "last_updated": pos.timestamp.isoformat() if pos else datetime.now(timezone.utc).isoformat()
        })
    
    response = {
        "total_buses": len(buses_data),
        "buses": buses_data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    await cache.set(cache_key, response, ttl=CACHE_TTL)
    
    return response


# =============================================================================
# GET /stops - All bus stops
# =============================================================================

@router.get(
    "/stops",
    response_model=AllStopsResponse,
    summary="All bus stops",
    description="Get all bus stops for map display.",
)
async def get_all_stops(
    db: AsyncSession = Depends(get_db),
):
    """Returns all active bus stops."""
    # Try cache
    redis = await get_redis()
    cache = Cache(redis)
    cache_key = "mobile:stops:all"
    
    cached = await cache.get(cache_key)
    if cached:
        return cached
    
    result = await db.execute(
        select(BusStop).where(BusStop.status == "active")
    )
    stops = result.scalars().all()
    
    stops_data = [
        {
            "stop_id": stop.id,
            "name": stop.name,
            "latitude": stop.latitude,
            "longitude": stop.longitude,
            "address": stop.address,
            "route_ids": stop.route_ids or [],
            "status": stop.status
        }
        for stop in stops
    ]
    
    response = {
        "total_stops": len(stops_data),
        "stops": stops_data
    }
    
    # Cache for 5 minutes (stops don't change often)
    await cache.set(cache_key, response, ttl=300)
    
    return response


# =============================================================================
# GET /stops/{stop_id}/arrivals - ETAs at a stop
# =============================================================================

@router.get(
    "/stops/{stop_id}/arrivals",
    response_model=StopArrivalsResponse,
    summary="Arrivals at stop",
    description="Get list of buses arriving at a stop with ETAs.",
)
async def get_stop_arrivals(
    stop_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns buses arriving at this stop with estimated times.
    
    ETA is calculated based on:
    - Distance from bus to stop
    - Average speed (default 20 km/h in city)
    """
    # Get stop
    result = await db.execute(
        select(BusStop).where(BusStop.id == stop_id)
    )
    stop = result.scalar_one_or_none()
    
    if not stop:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Stop not found", "stop_id": stop_id}
        )
    
    # Get buses on routes that pass this stop
    route_ids = stop.route_ids or []
    
    if not route_ids:
        return {
            "stop_id": stop.id,
            "stop_name": stop.name,
            "arrivals": [],
            "total_arrivals": 0
        }
    
    # Get all active buses
    result = await db.execute(
        select(Bus).where(Bus.status == "active")
    )
    buses = result.scalars().all()
    
    # Get positions
    positions_result = await db.execute(select(BusPosition))
    positions_map = {p.bus_id: p for p in positions_result.scalars().all()}
    
    # Get occupancy service
    occupancy_service = OccupancyService(db)
    
    arrivals = []
    AVERAGE_SPEED_KMH = 20  # Average bus speed in city
    
    for bus in buses:
        # Check if bus is on a route that passes this stop
        bus_route_id = bus.metadata_.get("route_id") if bus.metadata_ else None
        if bus_route_id not in route_ids:
            continue
        
        # Get position
        pos = positions_map.get(bus.id)
        if not pos:
            if bus.metadata_ and "location" in bus.metadata_:
                lat = bus.metadata_["location"].get("latitude")
                lon = bus.metadata_["location"].get("longitude")
            else:
                continue
        else:
            lat = pos.latitude
            lon = pos.longitude
        
        # Calculate distance
        distance = haversine_distance(stop.latitude, stop.longitude, lat, lon)
        
        # Calculate ETA
        speed = pos.speed if pos and pos.speed else AVERAGE_SPEED_KMH
        eta_hours = distance / speed
        eta_minutes = max(1, round(eta_hours * 60))
        
        # Get occupancy
        occ_data = await occupancy_service.get_bus_occupancy(bus.id)
        if not occ_data:
            continue
        
        arrivals.append({
            "bus_id": bus.id,
            "bus_number": bus.number,
            "route_id": occ_data.get("route_id"),
            "route_name": occ_data.get("route_name"),
            "eta_minutes": eta_minutes,
            "eta_time": (datetime.now(timezone.utc) + timedelta(minutes=eta_minutes)).isoformat(),
            "distance_km": round(distance, 2),
            "current_occupancy": occ_data["current_occupancy"],
            "capacity": occ_data["capacity"],
            "percentage": occ_data["percentage"],
            "status": occ_data["status"],
            "color": occ_data["color"]
        })
    
    # Sort by ETA
    arrivals.sort(key=lambda x: x["eta_minutes"])
    
    return {
        "stop_id": stop.id,
        "stop_name": stop.name,
        "arrivals": arrivals,
        "total_arrivals": len(arrivals)
    }


# =============================================================================
# GET /routes/{route_id}/path - Route polyline
# =============================================================================

@router.get(
    "/routes/{route_id}/path",
    response_model=RoutePathResponse,
    summary="Route path",
    description="Get route polyline coordinates for map display.",
)
async def get_route_path(
    route_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Returns route polyline and stops for drawing on map."""
    result = await db.execute(
        select(Route).where(Route.id == route_id)
    )
    route = result.scalar_one_or_none()
    
    if not route:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Route not found", "route_id": route_id}
        )
    
    # Get path from geometry (GeoJSON)
    path = []
    if route.geometry and "coordinates" in route.geometry:
        for coord in route.geometry["coordinates"]:
            path.append({"lat": coord[1], "lon": coord[0]})
    
    # Get stops on this route
    stops_result = await db.execute(
        select(BusStop).where(BusStop.status == "active")
    )
    all_stops = stops_result.scalars().all()
    
    route_stops = [
        {
            "stop_id": s.id,
            "name": s.name,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "address": s.address,
            "route_ids": s.route_ids or [],
            "status": s.status
        }
        for s in all_stops
        if s.route_ids and route_id in s.route_ids
    ]
    
    return {
        "route_id": route.id,
        "route_name": route.name,
        "path": path,
        "stops": route_stops
    }


# =============================================================================
# WebSocket for live updates (placeholder)
# =============================================================================

@router.websocket("/ws/buses")
async def websocket_buses(websocket: WebSocket):
    """
    WebSocket for real-time bus position updates.
    
    Sends position updates every 2 seconds.
    """
    await websocket.accept()
    
    try:
        while True:
            # In production, this would push live updates
            # For now, just keep connection alive
            import asyncio
            await asyncio.sleep(2)
            await websocket.send_json({
                "type": "ping",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    except Exception:
        pass
    finally:
        await websocket.close()
