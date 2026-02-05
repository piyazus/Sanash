"""
Zone Endpoints
==============
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.dependencies import get_current_user, require_role
from api.models.user import User
from api.models.bus import Camera, Zone
from api.schemas.bus import (
    ZoneCreate,
    ZoneUpdate,
    ZoneResponse,
    ZoneCrossingTest,
    ZoneCrossingResult,
    ZonePoint,
)

router = APIRouter()


def point_in_polygon(point: tuple, polygon: list) -> bool:
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]["x"], polygon[i]["y"]
        xj, yj = polygon[j]["x"], polygon[j]["y"]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


@router.post("", response_model=ZoneResponse, status_code=status.HTTP_201_CREATED)
async def create_zone(
    zone_data: ZoneCreate,
    user: User = Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Create a new zone. Requires operator role."""
    # Verify camera exists
    result = await db.execute(
        select(Camera).where(Camera.id == zone_data.camera_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )
    
    # Convert points to dict format
    polygon = [{"x": p.x, "y": p.y} for p in zone_data.polygon]
    
    zone = Zone(
        camera_id=zone_data.camera_id,
        name=zone_data.name,
        zone_type=zone_data.zone_type,
        polygon=polygon,
        entry_direction=zone_data.entry_direction,
        color=zone_data.color,
    )
    
    db.add(zone)
    await db.commit()
    await db.refresh(zone)
    
    return zone


@router.get("", response_model=list[ZoneResponse])
async def list_zones(
    camera_id: int = None,
    skip: int = 0,
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List zones, optionally filtered by camera."""
    query = select(Zone).where(Zone.is_active == True)
    
    if camera_id:
        query = query.where(Zone.camera_id == camera_id)
    
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{zone_id}", response_model=ZoneResponse)
async def get_zone(
    zone_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get zone details."""
    result = await db.execute(
        select(Zone).where(Zone.id == zone_id)
    )
    zone = result.scalar_one_or_none()
    
    if not zone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Zone not found",
        )
    
    return zone


@router.patch("/{zone_id}", response_model=ZoneResponse)
async def update_zone(
    zone_id: int,
    zone_data: ZoneUpdate,
    user: User = Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Update zone. Requires operator role."""
    result = await db.execute(
        select(Zone).where(Zone.id == zone_id)
    )
    zone = result.scalar_one_or_none()
    
    if not zone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Zone not found",
        )
    
    update_data = zone_data.model_dump(exclude_unset=True)
    
    # Convert polygon if provided
    if "polygon" in update_data and update_data["polygon"]:
        update_data["polygon"] = [{"x": p.x, "y": p.y} for p in update_data["polygon"]]
    
    for field, value in update_data.items():
        setattr(zone, field, value)
    
    await db.commit()
    await db.refresh(zone)
    
    return zone


@router.delete("/{zone_id}")
async def delete_zone(
    zone_id: int,
    user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Delete zone. Requires admin role."""
    result = await db.execute(
        select(Zone).where(Zone.id == zone_id)
    )
    zone = result.scalar_one_or_none()
    
    if not zone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Zone not found",
        )
    
    await db.delete(zone)
    await db.commit()
    
    return {"success": True, "message": "Zone deleted"}


@router.post("/test-crossing", response_model=ZoneCrossingResult)
async def test_zone_crossing(
    test_data: ZoneCrossingTest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Test if a path crosses a zone.
    
    Used for debugging zone configurations.
    """
    result = await db.execute(
        select(Zone).where(Zone.id == test_data.zone_id)
    )
    zone = result.scalar_one_or_none()
    
    if not zone:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Zone not found",
        )
    
    polygon = zone.polygon
    path = [(p.x, p.y) for p in test_data.track_path]
    
    # Check each point along path
    was_inside = point_in_polygon(path[0], polygon)
    crossing_point = None
    direction = None
    
    for i, point in enumerate(path[1:], 1):
        is_inside = point_in_polygon(point, polygon)
        
        if was_inside != is_inside:
            # Crossing detected
            crossing_point = ZonePoint(x=int(point[0]), y=int(point[1]))
            direction = "in" if is_inside else "out"
            break
        
        was_inside = is_inside
    
    return ZoneCrossingResult(
        crossed=direction is not None,
        direction=direction,
        crossing_point=crossing_point,
    )
