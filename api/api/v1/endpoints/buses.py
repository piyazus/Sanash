"""
Bus Fleet Endpoints
===================
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.dependencies import get_current_user, require_role
from api.models.user import User
from api.models.bus import Bus, Camera
from api.schemas.bus import BusCreate, BusUpdate, BusResponse, BusListItem

router = APIRouter()


@router.post("", response_model=BusResponse, status_code=status.HTTP_201_CREATED)
async def create_bus(
    bus_data: BusCreate,
    user: User = Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Create a new bus. Requires operator role."""
    # Check if number exists
    result = await db.execute(
        select(Bus).where(Bus.number == bus_data.number)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bus number already exists",
        )
    
    bus = Bus(**bus_data.model_dump())
    db.add(bus)
    await db.commit()
    await db.refresh(bus)
    
    return bus


@router.get("", response_model=list[BusListItem])
async def list_buses(
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all buses."""
    query = select(Bus)
    
    if status_filter:
        query = query.where(Bus.status == status_filter)
    
    query = query.order_by(Bus.number).offset(skip).limit(limit)
    
    result = await db.execute(query)
    buses = result.scalars().all()
    
    # Add camera counts
    items = []
    for bus in buses:
        count_result = await db.execute(
            select(func.count(Camera.id)).where(Camera.bus_id == bus.id)
        )
        items.append(BusListItem(
            id=bus.id,
            number=bus.number,
            status=bus.status,
            cameras_count=count_result.scalar(),
        ))
    
    return items


@router.get("/{bus_id}", response_model=BusResponse)
async def get_bus(
    bus_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get bus details."""
    result = await db.execute(
        select(Bus).where(Bus.id == bus_id)
    )
    bus = result.scalar_one_or_none()
    
    if not bus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bus not found",
        )
    
    return bus


@router.patch("/{bus_id}", response_model=BusResponse)
async def update_bus(
    bus_id: int,
    bus_data: BusUpdate,
    user: User = Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Update bus. Requires operator role."""
    result = await db.execute(
        select(Bus).where(Bus.id == bus_id)
    )
    bus = result.scalar_one_or_none()
    
    if not bus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bus not found",
        )
    
    # Update fields
    for field, value in bus_data.model_dump(exclude_unset=True).items():
        setattr(bus, field, value)
    
    await db.commit()
    await db.refresh(bus)
    
    return bus


@router.delete("/{bus_id}")
async def delete_bus(
    bus_id: int,
    user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Delete bus. Requires admin role."""
    result = await db.execute(
        select(Bus).where(Bus.id == bus_id)
    )
    bus = result.scalar_one_or_none()
    
    if not bus:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bus not found",
        )
    
    await db.delete(bus)
    await db.commit()
    
    return {"success": True, "message": "Bus deleted"}
