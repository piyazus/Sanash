"""
Occupancy Service
=================

Business logic for calculating real-time bus occupancy from zone crossings.

Features:
- Current occupancy = total_entries - total_exits (never below 0)
- Status logic: 0-60% (available), 61-80% (getting_full), 81-100% (crowded)
- Redis caching with 5-second TTL
- Haversine distance for nearby buses
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple
from math import radians, sin, cos, sqrt, atan2

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.models.bus import Bus, Route
from api.models.detection import ZoneCrossing
from api.core.cache import Cache, get_redis


# =============================================================================
# OCCUPANCY THRESHOLDS
# =============================================================================

THRESHOLD_AVAILABLE = 60  # 0-60% = green/available
THRESHOLD_GETTING_FULL = 80  # 61-80% = yellow/getting_full
# 81-100% = red/crowded


def calculate_status(percentage: int) -> Tuple[str, str]:
    """
    Calculate status and color based on occupancy percentage.
    
    Returns:
        Tuple of (status, color)
    """
    if percentage <= THRESHOLD_AVAILABLE:
        return "available", "green"
    elif percentage <= THRESHOLD_GETTING_FULL:
        return "getting_full", "yellow"
    else:
        return "crowded", "red"


def calculate_percentage(current: int, capacity: int) -> int:
    """Calculate percentage, capped at 100."""
    if capacity <= 0:
        return 0
    return min(100, round((current / capacity) * 100))


# =============================================================================
# HAVERSINE DISTANCE
# =============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points in kilometers using Haversine formula.
    """
    R = 6371  # Earth's radius in km
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)
    
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c


# =============================================================================
# CACHE DECORATOR
# =============================================================================

def cache_key_bus(bus_id: int) -> str:
    return f"occupancy:bus:{bus_id}"


def cache_key_route(route_id: int) -> str:
    return f"occupancy:route:{route_id}"


def cache_key_routes_all() -> str:
    return "occupancy:routes:all"


def cache_key_nearby(lat: float, lon: float, radius: float) -> str:
    return f"occupancy:nearby:{lat:.4f}:{lon:.4f}:{radius}"


# =============================================================================
# OCCUPANCY SERVICE
# =============================================================================

class OccupancyService:
    """
    Service for calculating and caching bus occupancy data.
    
    Uses Redis caching with 5-second TTL for real-time data.
    """
    
    CACHE_TTL = 5  # 5 seconds cache for real-time data
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._cache: Optional[Cache] = None
    
    async def get_cache(self) -> Cache:
        """Get or initialize cache."""
        if self._cache is None:
            redis_client = await get_redis()
            self._cache = Cache(redis_client)
        return self._cache
    
    # =========================================================================
    # GET /buses/{bus_id}/occupancy
    # =========================================================================
    
    async def get_bus_occupancy(self, bus_id: int) -> Optional[dict]:
        """
        Get current occupancy for a specific bus.
        
        Returns cached data if available, otherwise calculates from database.
        Logic: current_occupancy = total_entries - total_exits (min 0)
        """
        cache_key = cache_key_bus(bus_id)
        
        # Try cache first
        cache = await self.get_cache()
        cached = await cache.get(cache_key)
        if cached:
            return cached
        
        # Get bus from database with eager loading
        result = await self.db.execute(
            select(Bus).where(Bus.id == bus_id)
        )
        bus = result.scalar_one_or_none()
        
        if not bus:
            return None
        
        # Calculate current occupancy
        occupancy_data = await self._calculate_bus_occupancy(bus)
        
        # Cache result
        await cache.set(cache_key, occupancy_data, ttl=self.CACHE_TTL)
        
        return occupancy_data
    
    # =========================================================================
    # GET /routes/{route_id}/buses
    # =========================================================================
    
    async def get_route_buses(self, route_id: int) -> Optional[dict]:
        """
        Get all buses on a route with their occupancy.
        Uses selectinload to avoid N+1 queries.
        """
        cache_key = cache_key_route(route_id)
        
        cache = await self.get_cache()
        cached = await cache.get(cache_key)
        if cached:
            return cached
        
        # Get route
        result = await self.db.execute(
            select(Route).where(Route.id == route_id)
        )
        route = result.scalar_one_or_none()
        
        if not route:
            return None
        
        # Get buses assigned to this route via metadata
        result = await self.db.execute(
            select(Bus)
            .where(Bus.status == "active")
            .options(selectinload(Bus.cameras))
        )
        buses = result.scalars().all()
        
        # Filter by route_id in metadata
        route_buses = []
        for bus in buses:
            if bus.metadata_ and bus.metadata_.get("route_id") == route_id:
                bus_data = await self._calculate_bus_occupancy(bus, route)
                route_buses.append(bus_data)
        
        response = {
            "route_id": route_id,
            "route_name": route.name,
            "total_buses": len(route_buses),
            "buses": route_buses
        }
        
        await cache.set(cache_key, response, ttl=self.CACHE_TTL)
        
        return response
    
    # =========================================================================
    # GET /routes
    # =========================================================================
    
    async def get_all_routes(self) -> dict:
        """
        Get all active routes with bus count and average occupancy.
        """
        cache_key = cache_key_routes_all()
        
        cache = await self.get_cache()
        cached = await cache.get(cache_key)
        if cached:
            return cached
        
        # Get all active routes
        result = await self.db.execute(
            select(Route).where(Route.status == "active")
        )
        routes = result.scalars().all()
        
        # Get all active buses once (avoid N+1)
        buses_result = await self.db.execute(
            select(Bus).where(Bus.status == "active")
        )
        all_buses = list(buses_result.scalars().all())
        
        routes_data = []
        for route in routes:
            # Filter buses for this route
            route_buses = [
                b for b in all_buses 
                if b.metadata_ and b.metadata_.get("route_id") == route.id
            ]
            
            # Calculate average occupancy
            total_percentage = 0
            for bus in route_buses:
                bus_data = await self._calculate_bus_occupancy(bus, route)
                total_percentage += bus_data["percentage"]
            
            avg_occupancy = round(total_percentage / len(route_buses)) if route_buses else 0
            
            routes_data.append({
                "route_id": route.id,
                "route_name": route.name,
                "total_buses": len(route_buses),
                "avg_occupancy_percentage": avg_occupancy,
                "status": "operational" if route_buses else "no_service"
            })
        
        response = {
            "routes": routes_data,
            "total_routes": len(routes_data)
        }
        
        await cache.set(cache_key, response, ttl=self.CACHE_TTL * 2)  # 10s TTL
        
        return response
    
    # =========================================================================
    # GET /buses/nearby
    # =========================================================================
    
    async def get_nearby_buses(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 2.0
    ) -> dict:
        """
        Find buses near a location using Haversine formula.
        """
        cache_key = cache_key_nearby(latitude, longitude, radius_km)
        
        cache = await self.get_cache()
        cached = await cache.get(cache_key)
        if cached:
            return cached
        
        # Get all active buses with location
        result = await self.db.execute(
            select(Bus).where(Bus.status == "active")
        )
        buses = result.scalars().all()
        
        nearby_buses = []
        for bus in buses:
            # Check if bus has location in metadata
            if not bus.metadata_ or "location" not in bus.metadata_:
                continue
            
            bus_lat = bus.metadata_["location"].get("latitude")
            bus_lon = bus.metadata_["location"].get("longitude")
            
            if bus_lat is None or bus_lon is None:
                continue
            
            # Calculate distance using Haversine
            distance = haversine_distance(latitude, longitude, bus_lat, bus_lon)
            
            if distance <= radius_km:
                bus_data = await self._calculate_bus_occupancy(bus)
                bus_data["distance_km"] = round(distance, 2)
                nearby_buses.append(bus_data)
        
        # Sort by distance
        nearby_buses.sort(key=lambda x: x.get("distance_km", 999))
        
        response = {
            "latitude": latitude,
            "longitude": longitude,
            "radius_km": radius_km,
            "total_buses": len(nearby_buses),
            "buses": nearby_buses
        }
        
        await cache.set(cache_key, response, ttl=self.CACHE_TTL)
        
        return response
    
    # =========================================================================
    # INTERNAL: Calculate occupancy from zone crossings
    # =========================================================================
    
    async def _calculate_bus_occupancy(
        self,
        bus: Bus,
        route: Optional[Route] = None
    ) -> dict:
        """
        Calculate current occupancy for a bus.
        
        Logic: current_occupancy = SUM(entries) - SUM(exits)
        Ensures count never drops below 0.
        
        Note: ZoneCrossing links to Zone → Camera → Bus
        """
        from api.models.bus import Camera, Zone
        
        # Get all camera IDs for this bus
        camera_result = await self.db.execute(
            select(Camera.id).where(Camera.bus_id == bus.id)
        )
        camera_ids = [c for c in camera_result.scalars().all()]
        
        if not camera_ids:
            # No cameras = no data
            current_occupancy = 0
        else:
            # Get zone IDs for these cameras
            zone_result = await self.db.execute(
                select(Zone.id).where(Zone.camera_id.in_(camera_ids))
            )
            zone_ids = [z for z in zone_result.scalars().all()]
            
            if not zone_ids:
                current_occupancy = 0
            else:
                # Count entries (direction='in')
                entries_result = await self.db.execute(
                    select(func.count(ZoneCrossing.id))
                    .where(
                        and_(
                            ZoneCrossing.zone_id.in_(zone_ids),
                            ZoneCrossing.direction == "in"
                        )
                    )
                )
                total_entries = entries_result.scalar() or 0
                
                # Count exits (direction='out')
                exits_result = await self.db.execute(
                    select(func.count(ZoneCrossing.id))
                    .where(
                        and_(
                            ZoneCrossing.zone_id.in_(zone_ids),
                            ZoneCrossing.direction == "out"
                        )
                    )
                )
                total_exits = exits_result.scalar() or 0
                
                # Calculate current occupancy (never below 0)
                current_occupancy = max(0, total_entries - total_exits)
        
        # Use bus capacity or default to 50
        capacity = bus.capacity or 50
        
        # Ensure current_occupancy doesn't exceed capacity for display
        current_occupancy = min(current_occupancy, capacity)
        
        # Calculate percentage and status
        percentage = calculate_percentage(current_occupancy, capacity)
        status, color = calculate_status(percentage)
        
        # Get route info
        route_id = None
        route_name = None
        if route:
            route_id = route.id
            route_name = route.name
        elif bus.metadata_ and "route_id" in bus.metadata_:
            route_id = bus.metadata_["route_id"]
            route_name = bus.metadata_.get("route_name")
        
        return {
            "bus_id": bus.id,
            "bus_number": bus.number,
            "route_id": route_id,
            "route_name": route_name,
            "current_occupancy": current_occupancy,
            "capacity": capacity,
            "percentage": percentage,
            "status": status,
            "color": color,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
