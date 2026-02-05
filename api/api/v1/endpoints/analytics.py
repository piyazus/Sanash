"""
Analytics Endpoints
===================
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.dependencies import get_current_user
from api.core.cache import get_redis, Cache
from api.models.user import User
from api.models.job import DetectionJob
from api.models.detection import Detection, ZoneCrossing
from api.models.analytics import AnalyticsSummary
from api.schemas.analytics import (
    OccupancyResponse,
    OccupancyDataPoint,
    HeatmapResponse,
    FlowResponse,
    FlowDataPoint,
    AnalyticsSummaryResponse,
    DashboardStats,
)

router = APIRouter()


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get real-time dashboard statistics."""
    # Active jobs
    active_result = await db.execute(
        select(func.count(DetectionJob.id)).where(
            DetectionJob.status == "processing"
        )
    )
    active_jobs = active_result.scalar() or 0
    
    # Pending alerts (unacknowledged)
    from api.models.alert import Alert
    alerts_result = await db.execute(
        select(func.count(Alert.id)).where(Alert.acknowledged == False)
    )
    pending_alerts = alerts_result.scalar() or 0
    
    # Active buses
    from api.models.bus import Bus
    buses_result = await db.execute(
        select(func.count(Bus.id)).where(Bus.status == "active")
    )
    buses_online = buses_result.scalar() or 0
    
    return DashboardStats(
        current_occupancy=0,  # Would come from live feed
        total_people_today=0,  # Aggregate from today's jobs
        active_jobs=active_jobs,
        pending_alerts=pending_alerts,
        buses_online=buses_online,
        avg_processing_fps=15.0,  # Average from recent jobs
    )


@router.get("/occupancy", response_model=OccupancyResponse)
async def get_occupancy_data(
    bus_id: Optional[int] = None,
    job_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    granularity: str = Query(default="hour", regex="^(minute|hour|day)$"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get occupancy time series data.
    
    - Supports filtering by bus or job
    - Granularity: minute, hour, day
    - Cached for 1 hour
    """
    # Default date range: last 7 days
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=7)
    
    # Check cache
    redis = await get_redis()
    cache = Cache(redis)
    cache_key = f"occupancy:{bus_id}:{job_id}:{start_date.isoformat()}:{end_date.isoformat()}:{granularity}"
    
    cached = await cache.get(cache_key)
    if cached:
        return OccupancyResponse(**cached)
    
    # Query detections grouped by time
    # Simplified - in production, use proper time bucketing
    data = []
    
    # Get analytics summary if job specified
    if job_id:
        result = await db.execute(
            select(AnalyticsSummary).where(AnalyticsSummary.job_id == job_id)
        )
        summary = result.scalar_one_or_none()
        
        if summary and summary.hourly_data:
            hourly = summary.hourly_data.get("occupancy", {})
            for hour, count in hourly.items():
                data.append(OccupancyDataPoint(
                    timestamp=start_date.replace(hour=int(hour)),
                    count=count
                ))
    
    response = OccupancyResponse(
        bus_id=bus_id,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        data=data,
        peak_occupancy=max([d.count for d in data], default=0),
        peak_time=max(data, key=lambda x: x.count).timestamp if data else None,
        avg_occupancy=sum(d.count for d in data) / len(data) if data else 0,
    )
    
    # Cache result
    await cache.set(cache_key, response.model_dump(mode="json"), ttl=3600)
    
    return response


@router.get("/heatmap", response_model=HeatmapResponse)
async def get_heatmap_data(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get heatmap data for visualization.
    
    Returns 2D intensity array of position frequencies.
    """
    # Get analytics summary
    result = await db.execute(
        select(AnalyticsSummary).where(AnalyticsSummary.job_id == job_id)
    )
    summary = result.scalar_one_or_none()
    
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analytics not found for this job",
        )
    
    # Return heatmap data if available
    heatmap = summary.heatmap_data or {"width": 64, "height": 36, "data": []}
    
    return HeatmapResponse(
        job_id=job_id,
        width=heatmap.get("width", 64),
        height=heatmap.get("height", 36),
        data=heatmap.get("data", []),
        max_intensity=max(max(row) for row in heatmap.get("data", [[0]])) if heatmap.get("data") else 0,
    )


@router.get("/flow", response_model=FlowResponse)
async def get_flow_data(
    zone_id: Optional[int] = None,
    job_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get entry/exit flow data.
    
    Shows passenger movement over time.
    """
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=1)
    
    # Query zone crossings
    query = select(ZoneCrossing).where(
        ZoneCrossing.timestamp >= start_date,
        ZoneCrossing.timestamp <= end_date
    )
    
    if zone_id:
        query = query.where(ZoneCrossing.zone_id == zone_id)
    if job_id:
        query = query.where(ZoneCrossing.job_id == job_id)
    
    result = await db.execute(query.order_by(ZoneCrossing.timestamp))
    crossings = result.scalars().all()
    
    # Aggregate by hour
    hourly_data = {}
    for crossing in crossings:
        hour = crossing.timestamp.replace(minute=0, second=0, microsecond=0)
        if hour not in hourly_data:
            hourly_data[hour] = {"entries": 0, "exits": 0}
        
        if crossing.direction == "in":
            hourly_data[hour]["entries"] += 1
        else:
            hourly_data[hour]["exits"] += 1
    
    data = [
        FlowDataPoint(
            timestamp=hour,
            entries=vals["entries"],
            exits=vals["exits"],
            net_change=vals["entries"] - vals["exits"]
        )
        for hour, vals in sorted(hourly_data.items())
    ]
    
    total_entries = sum(d.entries for d in data)
    total_exits = sum(d.exits for d in data)
    
    return FlowResponse(
        zone_id=zone_id,
        start_date=start_date,
        end_date=end_date,
        total_entries=total_entries,
        total_exits=total_exits,
        data=data,
    )


@router.get("/summary/{job_id}", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(
    job_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get complete analytics summary for a job."""
    result = await db.execute(
        select(AnalyticsSummary).where(AnalyticsSummary.job_id == job_id)
    )
    summary = result.scalar_one_or_none()
    
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analytics not found",
        )
    
    return summary


# =============================================================================
# ADMIN DASHBOARD ENDPOINTS
# =============================================================================

@router.get("/admin/overview")
async def get_admin_overview(
    date: Optional[datetime] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get admin dashboard overview for a specific date.
    
    Returns:
    - Total passengers for the day
    - Busiest route
    - Peak hour
    - Total active buses
    """
    from api.models.bus import Bus, Route
    
    # Default to today
    if not date:
        date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    end_date = date + timedelta(days=1)
    
    # Get total passengers (entries) for the day
    entries_result = await db.execute(
        select(func.count(ZoneCrossing.id)).where(
            ZoneCrossing.timestamp >= date,
            ZoneCrossing.timestamp < end_date,
            ZoneCrossing.direction == "in"
        )
    )
    total_passengers = entries_result.scalar() or 0
    
    # Get busiest route
    from sqlalchemy.orm import joinedload
    
    route_counts = {}
    crossings_result = await db.execute(
        select(ZoneCrossing).where(
            ZoneCrossing.timestamp >= date,
            ZoneCrossing.timestamp < end_date,
            ZoneCrossing.direction == "in"
        )
    )
    
    for crossing in crossings_result.scalars():
        # Get route from job's bus
        job_result = await db.execute(
            select(DetectionJob).where(DetectionJob.id == crossing.job_id)
        )
        job = job_result.scalar_one_or_none()
        if job and job.bus_id:
            bus_result = await db.execute(
                select(Bus).where(Bus.id == job.bus_id)
            )
            bus = bus_result.scalar_one_or_none()
            if bus and bus.metadata_ and "route_id" in bus.metadata_:
                route_id = bus.metadata_["route_id"]
                route_counts[route_id] = route_counts.get(route_id, 0) + 1
    
    busiest_route = None
    busiest_route_count = 0
    if route_counts:
        busiest_route_id = max(route_counts, key=route_counts.get)
        busiest_route_count = route_counts[busiest_route_id]
        route_result = await db.execute(
            select(Route).where(Route.id == busiest_route_id)
        )
        route = route_result.scalar_one_or_none()
        if route:
            busiest_route = {
                "route_id": route.id,
                "route_name": route.name,
                "passenger_count": busiest_route_count
            }
    
    # Get peak hour
    hourly_counts = {}
    crossings_result = await db.execute(
        select(ZoneCrossing).where(
            ZoneCrossing.timestamp >= date,
            ZoneCrossing.timestamp < end_date,
            ZoneCrossing.direction == "in"
        )
    )
    
    for crossing in crossings_result.scalars():
        hour = crossing.timestamp.hour
        hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
    
    peak_hour = None
    if hourly_counts:
        peak_hour = max(hourly_counts, key=hourly_counts.get)
    
    # Active buses count
    buses_result = await db.execute(
        select(func.count(Bus.id)).where(Bus.status == "active")
    )
    active_buses = buses_result.scalar() or 0
    
    return {
        "date": date.isoformat(),
        "total_passengers": total_passengers,
        "busiest_route": busiest_route,
        "peak_hour": peak_hour,
        "peak_hour_count": hourly_counts.get(peak_hour, 0) if peak_hour else 0,
        "active_buses": active_buses,
        "hourly_breakdown": [
            {"hour": h, "count": c} 
            for h, c in sorted(hourly_counts.items())
        ]
    }


@router.get("/admin/routes/comparison")
async def compare_routes(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Compare route performance over a date range.
    
    Returns statistics for each route including:
    - Total passengers
    - Average occupancy
    - Peak times
    """
    from api.models.bus import Bus, Route
    
    # Default date range: last 7 days
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=7)
    
    # Get all routes
    routes_result = await db.execute(
        select(Route).where(Route.status == "active")
    )
    routes = routes_result.scalars().all()
    
    route_stats = []
    
    for route in routes:
        # Get buses on this route
        buses_result = await db.execute(
            select(Bus).where(Bus.status == "active")
        )
        buses = [
            b for b in buses_result.scalars()
            if b.metadata_ and b.metadata_.get("route_id") == route.id
        ]
        
        total_passengers = 0
        hourly_counts = {}
        
        for bus in buses:
            # Get jobs for this bus in date range
            jobs_result = await db.execute(
                select(DetectionJob).where(
                    DetectionJob.bus_id == bus.id,
                    DetectionJob.created_at >= start_date,
                    DetectionJob.created_at <= end_date
                )
            )
            
            for job in jobs_result.scalars():
                # Get crossings for this job
                crossings_result = await db.execute(
                    select(ZoneCrossing).where(
                        ZoneCrossing.job_id == job.id,
                        ZoneCrossing.direction == "in"
                    )
                )
                
                for crossing in crossings_result.scalars():
                    total_passengers += 1
                    hour = crossing.timestamp.hour
                    hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
        
        route_stats.append({
            "route_id": route.id,
            "route_name": route.name,
            "total_buses": len(buses),
            "total_passengers": total_passengers,
            "avg_daily_passengers": round(total_passengers / 7) if total_passengers else 0,
            "peak_hour": peak_hour,
            "peak_hour_passengers": hourly_counts.get(peak_hour, 0) if peak_hour else 0
        })
    
    # Sort by total passengers
    route_stats.sort(key=lambda x: x["total_passengers"], reverse=True)
    
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_routes": len(route_stats),
        "routes": route_stats
    }

