"""
Bus management endpoints — register, list, update, delete buses.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.bus import Bus
from app.models.operator import Operator
from app.models.route import Route
from app.services.auth import get_current_operator

router = APIRouter(prefix="/buses", tags=["buses"])


class BusCreate(BaseModel):
    bus_id: str
    plate_number: str
    capacity: int = 60
    route_number: str | None = None


class BusOut(BaseModel):
    id: str
    bus_id: str
    plate_number: str
    capacity: int
    current_count: int
    status: str
    route_number: str | None

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[BusOut])
async def list_buses(
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.execute(select(Bus).where(Bus.operator_id == current.id))
    buses = rows.scalars().all()
    return [
        BusOut(
            id=str(b.id),
            bus_id=b.bus_id,
            plate_number=b.plate_number,
            capacity=b.capacity,
            current_count=b.current_count,
            status=b.status,
            route_number=b.route.route_number if b.route else None,
        )
        for b in buses
    ]


@router.post("/", response_model=BusOut, status_code=201)
async def create_bus(
    body: BusCreate,
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(Bus).where(Bus.bus_id == body.bus_id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="bus_id already registered")

    route_id = None
    if body.route_number:
        rrow = await db.execute(
            select(Route).where(Route.route_number == body.route_number)
        )
        route = rrow.scalar_one_or_none()
        if route:
            route_id = route.id

    bus = Bus(
        bus_id=body.bus_id,
        plate_number=body.plate_number,
        capacity=body.capacity,
        operator_id=current.id,
        route_id=route_id,
    )
    db.add(bus)
    await db.flush()
    return BusOut(
        id=str(bus.id),
        bus_id=bus.bus_id,
        plate_number=bus.plate_number,
        capacity=bus.capacity,
        current_count=bus.current_count,
        status=bus.status,
        route_number=body.route_number,
    )


@router.delete("/{bus_id}", status_code=204)
async def delete_bus(
    bus_id: str,
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    row = await db.execute(
        select(Bus).where(Bus.bus_id == bus_id, Bus.operator_id == current.id)
    )
    bus = row.scalar_one_or_none()
    if bus is None:
        raise HTTPException(status_code=404, detail="Bus not found")
    await db.delete(bus)
