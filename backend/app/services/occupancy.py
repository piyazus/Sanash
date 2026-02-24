"""
Occupancy service — processes Jetson inference pushes, updates bus state,
persists logs, and triggers alerts.
"""
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.bus import Bus
from app.models.occupancy_log import OccupancyLog
from app.models.alert import Alert


def _status(ratio: float) -> str:
    if ratio >= settings.THRESHOLD_RED:
        return "red"
    if ratio >= settings.THRESHOLD_YELLOW:
        return "yellow"
    return "green"


async def process_reading(
    db: AsyncSession,
    bus_id: str,
    passenger_count: int,
    latitude: float | None = None,
    longitude: float | None = None,
    confidence: float | None = None,
) -> dict:
    """
    Handle one occupancy reading from a Jetson device.

    Returns a dict with the current occupancy status for the device.
    Raises ValueError if bus_id is not registered.
    """
    result = await db.execute(select(Bus).where(Bus.bus_id == bus_id))
    bus = result.scalar_one_or_none()
    if bus is None:
        raise ValueError(f"Bus '{bus_id}' not registered")

    ratio = min(passenger_count / max(bus.capacity, 1), 1.0)
    status = _status(ratio)

    # Update live state
    bus.current_count = passenger_count
    bus.last_seen = datetime.now(timezone.utc)
    if latitude is not None:
        bus.current_lat = latitude
    if longitude is not None:
        bus.current_lon = longitude

    # Persist log entry
    log = OccupancyLog(
        bus_id=bus.id,
        passenger_count=passenger_count,
        occupancy_ratio=ratio,
        status=status,
        latitude=latitude,
        longitude=longitude,
        confidence=confidence,
    )
    db.add(log)

    # Trigger alert if above threshold and not in cooldown
    if ratio >= settings.ALERT_RED_THRESHOLD:
        await _maybe_create_alert(db, bus, ratio, passenger_count)

    return {
        "bus_id": bus_id,
        "passenger_count": passenger_count,
        "capacity": bus.capacity,
        "occupancy_ratio": round(ratio, 4),
        "status": status,
        "timestamp": log.timestamp.isoformat(),
    }


async def _maybe_create_alert(
    db: AsyncSession, bus: Bus, ratio: float, count: int
) -> None:
    """Create an alert unless one was already created within the cooldown window."""
    from datetime import timedelta

    cooldown = timedelta(minutes=settings.ALERT_COOLDOWN_MINUTES)
    cutoff = datetime.now(timezone.utc) - cooldown

    recent = await db.execute(
        select(Alert)
        .where(Alert.bus_id == bus.id, Alert.triggered_at >= cutoff, Alert.resolved == False)  # noqa: E712
        .limit(1)
    )
    if recent.scalar_one_or_none() is not None:
        return  # Still in cooldown

    alert = Alert(
        bus_id=bus.id,
        operator_id=bus.operator_id,
        occupancy_ratio=ratio,
        passenger_count=count,
        message=(
            f"Bus {bus.bus_id} is at {ratio:.0%} capacity "
            f"({count}/{bus.capacity} passengers)"
        ),
    )
    db.add(alert)
