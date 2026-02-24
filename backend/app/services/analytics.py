"""
Analytics service — aggregates occupancy logs for the dashboard.
"""
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.occupancy_log import OccupancyLog
from app.models.bus import Bus
from app.models.alert import Alert


async def get_fleet_summary(db: AsyncSession, operator_id: UUID) -> dict:
    """Return real-time fleet summary for an operator."""
    result = await db.execute(
        select(Bus).where(Bus.operator_id == operator_id)
    )
    buses = result.scalars().all()

    counts = {"green": 0, "yellow": 0, "red": 0, "offline": 0}
    now = datetime.now(timezone.utc)
    offline_threshold = now - timedelta(minutes=5)

    for bus in buses:
        if bus.last_seen is None or bus.last_seen < offline_threshold:
            counts["offline"] += 1
        else:
            counts[bus.status] += 1

    return {
        "total_buses": len(buses),
        "green": counts["green"],
        "yellow": counts["yellow"],
        "red": counts["red"],
        "offline": counts["offline"],
        "timestamp": now.isoformat(),
    }


async def get_hourly_averages(
    db: AsyncSession,
    bus_id: UUID,
    hours: int = 24,
) -> list[dict]:
    """Return hourly average occupancy for a single bus over the last N hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    rows = await db.execute(
        select(
            func.date_trunc("hour", OccupancyLog.timestamp).label("hour"),
            func.avg(OccupancyLog.passenger_count).label("avg_count"),
            func.avg(OccupancyLog.occupancy_ratio).label("avg_ratio"),
            func.max(OccupancyLog.passenger_count).label("max_count"),
        )
        .where(
            and_(OccupancyLog.bus_id == bus_id, OccupancyLog.timestamp >= since)
        )
        .group_by("hour")
        .order_by("hour")
    )

    return [
        {
            "hour": row.hour.isoformat(),
            "avg_count": round(float(row.avg_count), 1),
            "avg_ratio": round(float(row.avg_ratio), 4),
            "max_count": int(row.max_count),
        }
        for row in rows
    ]


async def get_peak_analysis(
    db: AsyncSession,
    operator_id: UUID,
    days: int = 7,
) -> list[dict]:
    """Return average occupancy by hour-of-day across the fleet."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rows = await db.execute(
        select(
            func.extract("hour", OccupancyLog.timestamp).label("hour_of_day"),
            func.avg(OccupancyLog.occupancy_ratio).label("avg_ratio"),
            func.count(OccupancyLog.id).label("n"),
        )
        .join(Bus, Bus.id == OccupancyLog.bus_id)
        .where(and_(Bus.operator_id == operator_id, OccupancyLog.timestamp >= since))
        .group_by("hour_of_day")
        .order_by("hour_of_day")
    )

    return [
        {
            "hour": int(row.hour_of_day),
            "avg_ratio": round(float(row.avg_ratio), 4),
            "sample_count": int(row.n),
        }
        for row in rows
    ]
