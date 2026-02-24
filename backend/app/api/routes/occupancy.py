"""
Occupancy endpoints — used by Jetson devices to push readings.
Also provides current status for dashboard polling.
"""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.bus import Bus
from app.models.occupancy_log import OccupancyLog
from app.models.operator import Operator
from app.services.auth import get_current_operator
from app.services.occupancy import process_reading

router = APIRouter(prefix="/occupancy", tags=["occupancy"])


class ReadingRequest(BaseModel):
    bus_id: str = Field(..., description="Unique bus identifier from config.yaml")
    passenger_count: int = Field(..., ge=0, le=500)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    device_timestamp: Optional[str] = None  # ISO-8601 from Jetson


class ReadingResponse(BaseModel):
    bus_id: str
    passenger_count: int
    capacity: int
    occupancy_ratio: float
    status: str  # green / yellow / red
    timestamp: str


class BusStatusOut(BaseModel):
    bus_id: str
    plate_number: str
    current_count: int
    capacity: int
    occupancy_ratio: float
    status: str
    latitude: Optional[float]
    longitude: Optional[float]
    last_seen: Optional[str]
    route_number: Optional[str]

    model_config = {"from_attributes": True}


@router.post("/reading", response_model=ReadingResponse)
async def push_reading(
    body: ReadingRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Called by Jetson devices every 30 seconds.
    No auth header required — device uses bus_id as identifier.
    (In production, add device API key validation here.)
    """
    try:
        result = await process_reading(
            db,
            bus_id=body.bus_id,
            passenger_count=body.passenger_count,
            latitude=body.latitude,
            longitude=body.longitude,
            confidence=body.confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return ReadingResponse(**result)


@router.get("/fleet", response_model=list[BusStatusOut])
async def get_fleet_status(
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    """Return live status of all buses belonging to the authenticated operator."""
    rows = await db.execute(
        select(Bus).where(Bus.operator_id == current.id)
    )
    buses = rows.scalars().all()

    result = []
    for bus in buses:
        result.append(
            BusStatusOut(
                bus_id=bus.bus_id,
                plate_number=bus.plate_number,
                current_count=bus.current_count,
                capacity=bus.capacity,
                occupancy_ratio=round(bus.occupancy_ratio, 4),
                status=bus.status,
                latitude=bus.current_lat,
                longitude=bus.current_lon,
                last_seen=bus.last_seen.isoformat() if bus.last_seen else None,
                route_number=bus.route.route_number if bus.route else None,
            )
        )
    return result


@router.get("/bus/{bus_id}/history", response_model=list[dict])
async def get_bus_history(
    bus_id: str,
    limit: int = 100,
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    """Return recent occupancy log entries for a specific bus."""
    bus_row = await db.execute(
        select(Bus).where(Bus.bus_id == bus_id, Bus.operator_id == current.id)
    )
    bus = bus_row.scalar_one_or_none()
    if bus is None:
        raise HTTPException(status_code=404, detail="Bus not found")

    logs_row = await db.execute(
        select(OccupancyLog)
        .where(OccupancyLog.bus_id == bus.id)
        .order_by(OccupancyLog.timestamp.desc())
        .limit(limit)
    )
    logs = logs_row.scalars().all()

    return [
        {
            "timestamp": log.timestamp.isoformat(),
            "passenger_count": log.passenger_count,
            "occupancy_ratio": log.occupancy_ratio,
            "status": log.status,
            "latitude": log.latitude,
            "longitude": log.longitude,
        }
        for log in logs
    ]
