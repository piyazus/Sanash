"""
Public API Endpoints
====================

Real-time bus occupancy data for passengers (no authentication required).

Endpoints:
- GET /routes - List all active routes
- GET /routes/{route_id}/buses - Buses on a route
- GET /buses/{bus_id}/occupancy - Single bus occupancy
- GET /buses/nearby - Find buses by location

Features:
- Redis caching (5s TTL)
- Rate limiting (100 req/min per IP)
- Standardized error responses
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.cache import get_redis, RateLimiter
from api.services.occupancy_service import OccupancyService
from api.schemas.public import (
    BusOccupancyResponse,
    BusListResponse,
    NearbyBusesResponse,
    RouteListResponse,
)


router = APIRouter()


# =============================================================================
# RATE LIMITING DEPENDENCY
# =============================================================================

RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # 60 seconds = 1 minute


async def check_rate_limit(request: Request):
    """
    Rate limiting dependency - 100 requests per minute per IP.
    
    Raises 429 Too Many Requests if limit exceeded.
    """
    try:
        redis = await get_redis()
        limiter = RateLimiter(
            redis,
            max_requests=RATE_LIMIT_REQUESTS,
            window_seconds=RATE_LIMIT_WINDOW
        )
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:public:{client_ip}"
        
        allowed = await limiter.is_allowed(key)
        
        if not allowed:
            remaining = await limiter.get_remaining(key)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {RATE_LIMIT_REQUESTS}/minute",
                    "retry_after_seconds": RATE_LIMIT_WINDOW
                }
            )
    except HTTPException:
        raise
    except Exception:
        # If Redis is down, allow request (fail open)
        pass


# =============================================================================
# GET /routes - List all active routes
# =============================================================================

@router.get(
    "/routes",
    response_model=RouteListResponse,
    summary="List all routes",
    description="Fetch all active routes with bus count and average occupancy.",
    responses={
        200: {"description": "List of routes"},
        429: {"description": "Rate limit exceeded"},
    }
)
async def get_routes(
    db: AsyncSession = Depends(get_db),
    _: None = Depends(check_rate_limit),
):
    """
    Get all active bus routes.
    
    Returns:
    - **route_id**: Unique route identifier
    - **route_name**: Display name
    - **total_buses**: Number of active buses on route
    - **avg_occupancy_percentage**: Average occupancy across all buses
    """
    service = OccupancyService(db)
    data = await service.get_all_routes()
    return data


# =============================================================================
# GET /routes/{route_id}/buses - Buses on a route
# =============================================================================

@router.get(
    "/routes/{route_id}/buses",
    response_model=BusListResponse,
    summary="Buses on a route",
    description="List all buses currently assigned to a route with occupancy data.",
    responses={
        200: {"description": "List of buses with occupancy"},
        404: {"description": "Route not found"},
        429: {"description": "Rate limit exceeded"},
    }
)
async def get_route_buses(
    route_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(check_rate_limit),
):
    """
    Get all buses on a specific route.
    
    For each bus returns:
    - **bus_id**: Unique bus identifier
    - **bus_number**: Display number (e.g., "32")
    - **current_occupancy**: Current passenger count
    - **capacity**: Maximum capacity
    - **percentage**: Occupancy percentage (0-100)
    - **status**: "available" / "getting_full" / "crowded"
    - **color**: "green" / "yellow" / "red"
    """
    service = OccupancyService(db)
    data = await service.get_route_buses(route_id)
    
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Route not found",
                "message": f"No route found with ID {route_id}"
            }
        )
    
    return data


# =============================================================================
# GET /buses/{bus_id}/occupancy - Single bus occupancy
# =============================================================================

@router.get(
    "/buses/{bus_id}/occupancy",
    response_model=BusOccupancyResponse,
    summary="Bus occupancy",
    description="Get real-time occupancy data for a specific bus.",
    responses={
        200: {"description": "Bus occupancy data"},
        404: {"description": "Bus not found"},
        429: {"description": "Rate limit exceeded"},
    }
)
async def get_bus_occupancy(
    bus_id: int,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(check_rate_limit),
):
    """
    Get real-time occupancy for a specific bus.
    
    **Calculation Logic:**
    - current_occupancy = total_entries - total_exits
    - Count never drops below 0
    
    **Status Logic:**
    - 0-60%: green / available
    - 61-80%: yellow / getting_full
    - 81-100%: red / crowded
    """
    service = OccupancyService(db)
    data = await service.get_bus_occupancy(bus_id)
    
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "Bus not found",
                "message": f"No bus found with ID {bus_id}"
            }
        )
    
    return data


# =============================================================================
# GET /buses/nearby - Find buses by location
# =============================================================================

@router.get(
    "/buses/nearby",
    response_model=NearbyBusesResponse,
    summary="Nearby buses",
    description="Find buses near a location using Haversine distance.",
    responses={
        200: {"description": "List of nearby buses"},
        429: {"description": "Rate limit exceeded"},
    }
)
async def get_nearby_buses(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    radius: float = Query(default=2.0, ge=0.1, le=50, description="Radius in km"),
    db: AsyncSession = Depends(get_db),
    _: None = Depends(check_rate_limit),
):
    """
    Find buses within a radius of a location.
    
    Uses Haversine formula to calculate great-circle distance.
    Results sorted by distance (nearest first).
    
    **Parameters:**
    - **lat**: Latitude (-90 to 90)
    - **lon**: Longitude (-180 to 180)
    - **radius**: Search radius in km (default 2km, max 50km)
    """
    service = OccupancyService(db)
    data = await service.get_nearby_buses(lat, lon, radius)
    
    return data


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get(
    "/health",
    summary="Health check",
    description="Check if public API is operational.",
)
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "api": "public",
        "version": "1.0.0"
    }
