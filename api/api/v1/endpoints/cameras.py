"""
Camera Endpoints
================
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.dependencies import get_current_user, require_role
from api.models.user import User
from api.models.bus import Bus, Camera
from api.schemas.bus import CameraCreate, CameraUpdate, CameraResponse, CameraStatus

router = APIRouter()


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def create_camera(
    camera_data: CameraCreate,
    user: User = Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Create a new camera. Requires operator role."""
    # Verify bus exists
    result = await db.execute(
        select(Bus).where(Bus.id == camera_data.bus_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bus not found",
        )
    
    camera = Camera(**camera_data.model_dump())
    db.add(camera)
    await db.commit()
    await db.refresh(camera)
    
    return camera


@router.get("", response_model=list[CameraResponse])
async def list_cameras(
    bus_id: int = None,
    skip: int = 0,
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List cameras, optionally filtered by bus."""
    query = select(Camera)
    
    if bus_id:
        query = query.where(Camera.bus_id == bus_id)
    
    query = query.offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(
    camera_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get camera details."""
    result = await db.execute(
        select(Camera).where(Camera.id == camera_id)
    )
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )
    
    return camera


@router.get("/{camera_id}/status", response_model=CameraStatus)
async def get_camera_status(
    camera_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get camera status (for live monitoring)."""
    result = await db.execute(
        select(Camera).where(Camera.id == camera_id)
    )
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )
    
    return CameraStatus(
        id=camera.id,
        name=camera.name,
        status=camera.status,
        last_seen=None,  # Would come from live feed monitoring
        fps=camera.fps,
    )


@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int,
    camera_data: CameraUpdate,
    user: User = Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Update camera. Requires operator role."""
    result = await db.execute(
        select(Camera).where(Camera.id == camera_id)
    )
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )
    
    for field, value in camera_data.model_dump(exclude_unset=True).items():
        setattr(camera, field, value)
    
    await db.commit()
    await db.refresh(camera)
    
    return camera


@router.delete("/{camera_id}")
async def delete_camera(
    camera_id: int,
    user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Delete camera. Requires admin role."""
    result = await db.execute(
        select(Camera).where(Camera.id == camera_id)
    )
    camera = result.scalar_one_or_none()
    
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Camera not found",
        )
    
    await db.delete(camera)
    await db.commit()
    
    return {"success": True, "message": "Camera deleted"}
