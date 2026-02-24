"""
Analytics endpoints — aggregated data for the operator dashboard.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.operator import Operator
from app.services.analytics import get_fleet_summary, get_hourly_averages, get_peak_analysis
from app.services.auth import get_current_operator

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
async def fleet_summary(
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    """Real-time fleet-wide occupancy breakdown."""
    return await get_fleet_summary(db, current.id)


@router.get("/bus/{bus_id}/hourly")
async def bus_hourly(
    bus_id: str,
    hours: int = Query(24, ge=1, le=168),
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    """Hourly average occupancy for a bus over the last N hours."""
    from sqlalchemy import select
    from app.models.bus import Bus

    row = await db.execute(
        select(Bus).where(Bus.bus_id == bus_id, Bus.operator_id == current.id)
    )
    bus = row.scalar_one_or_none()
    if bus is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Bus not found")

    return await get_hourly_averages(db, bus.id, hours=hours)


@router.get("/peak")
async def peak_analysis(
    days: int = Query(7, ge=1, le=30),
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    """Hour-of-day average occupancy across the whole fleet."""
    return await get_peak_analysis(db, current.id, days=days)
