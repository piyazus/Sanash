"""
Alerts endpoints — list and resolve overcrowding alerts.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.alert import Alert
from app.models.bus import Bus
from app.models.operator import Operator
from app.services.auth import get_current_operator

router = APIRouter(prefix="/alerts", tags=["alerts"])


class AlertOut(BaseModel):
    id: str
    bus_id: str
    triggered_at: str
    occupancy_ratio: float
    passenger_count: int | None
    resolved: bool
    resolved_at: str | None
    message: str

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[AlertOut])
async def list_alerts(
    unresolved_only: bool = True,
    limit: int = 50,
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    """Return alerts for the operator's fleet."""
    conditions = [Alert.operator_id == current.id]
    if unresolved_only:
        conditions.append(Alert.resolved == False)  # noqa: E712

    rows = await db.execute(
        select(Alert, Bus.bus_id)
        .join(Bus, Bus.id == Alert.bus_id)
        .where(and_(*conditions))
        .order_by(Alert.triggered_at.desc())
        .limit(limit)
    )

    result = []
    for alert, bus_id_str in rows:
        result.append(
            AlertOut(
                id=str(alert.id),
                bus_id=bus_id_str,
                triggered_at=alert.triggered_at.isoformat(),
                occupancy_ratio=alert.occupancy_ratio,
                passenger_count=alert.passenger_count,
                resolved=alert.resolved,
                resolved_at=alert.resolved_at.isoformat() if alert.resolved_at else None,
                message=alert.message,
            )
        )
    return result


@router.post("/{alert_id}/resolve", response_model=AlertOut)
async def resolve_alert(
    alert_id: UUID,
    current: Operator = Depends(get_current_operator),
    db: AsyncSession = Depends(get_db),
):
    """Mark an alert as resolved."""
    from datetime import datetime, timezone

    row = await db.execute(
        select(Alert, Bus.bus_id)
        .join(Bus, Bus.id == Alert.bus_id)
        .where(Alert.id == alert_id, Alert.operator_id == current.id)
    )
    pair = row.first()
    if pair is None:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert, bus_id_str = pair
    alert.resolved = True
    alert.resolved_at = datetime.now(timezone.utc)

    return AlertOut(
        id=str(alert.id),
        bus_id=bus_id_str,
        triggered_at=alert.triggered_at.isoformat(),
        occupancy_ratio=alert.occupancy_ratio,
        passenger_count=alert.passenger_count,
        resolved=alert.resolved,
        resolved_at=alert.resolved_at.isoformat(),
        message=alert.message,
    )
