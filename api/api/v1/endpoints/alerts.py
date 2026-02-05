"""
Alerts Endpoints
================
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.dependencies import get_current_user, require_role
from api.models.user import User
from api.models.alert import Alert

router = APIRouter()


from pydantic import BaseModel, ConfigDict

class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    job_id: Optional[int]
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    acknowledged: bool
    acknowledged_by: Optional[int]
    acknowledged_at: Optional[datetime]


@router.get("", response_model=list[AlertResponse])
async def list_alerts(
    skip: int = 0,
    limit: int = 50,
    acknowledged: Optional[bool] = None,
    severity: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List alerts with optional filters."""
    query = select(Alert)
    
    if acknowledged is not None:
        query = query.where(Alert.acknowledged == acknowledged)
    
    if severity:
        query = query.where(Alert.severity == severity)
    
    query = query.order_by(Alert.timestamp.desc()).offset(skip).limit(limit)
    
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get alert details."""
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )
    
    return alert


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    user: User = Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Acknowledge an alert. Requires operator role."""
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )
    
    if alert.acknowledged:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Alert already acknowledged",
        )
    
    alert.acknowledged = True
    alert.acknowledged_by = user.id
    alert.acknowledged_at = datetime.now(timezone.utc)
    
    await db.commit()
    
    return {"success": True, "message": "Alert acknowledged"}


@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int,
    notes: Optional[str] = None,
    user: User = Depends(require_role("operator")),
    db: AsyncSession = Depends(get_db),
):
    """Resolve an alert with optional notes."""
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )
    
    alert.resolved = True
    alert.resolved_at = datetime.now(timezone.utc)
    alert.resolution_notes = notes
    
    if not alert.acknowledged:
        alert.acknowledged = True
        alert.acknowledged_by = user.id
        alert.acknowledged_at = datetime.now(timezone.utc)
    
    await db.commit()
    
    return {"success": True, "message": "Alert resolved"}


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: int,
    user: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Delete an alert. Requires admin role."""
    result = await db.execute(
        select(Alert).where(Alert.id == alert_id)
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found",
        )
    
    await db.delete(alert)
    await db.commit()
    
    return {"success": True, "message": "Alert deleted"}
