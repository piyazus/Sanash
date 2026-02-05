"""
Pydantic Schemas for Analytics
==============================
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# ANALYTICS SCHEMAS
# =============================================================================

class OccupancyDataPoint(BaseModel):
    """Single data point for occupancy chart."""
    timestamp: datetime
    count: int


class OccupancyResponse(BaseModel):
    """Occupancy time series data."""
    bus_id: Optional[int]
    start_date: datetime
    end_date: datetime
    granularity: str  # "minute", "hour", "day"
    data: List[OccupancyDataPoint]
    peak_occupancy: int
    peak_time: Optional[datetime]
    avg_occupancy: float


class HeatmapResponse(BaseModel):
    """Heatmap data for visualization."""
    job_id: int
    width: int
    height: int
    data: List[List[float]]  # 2D intensity array
    max_intensity: float


class FlowDataPoint(BaseModel):
    """Entry/exit flow data."""
    timestamp: datetime
    entries: int
    exits: int
    net_change: int


class FlowResponse(BaseModel):
    """Passenger flow data."""
    zone_id: Optional[int]
    start_date: datetime
    end_date: datetime
    total_entries: int
    total_exits: int
    data: List[FlowDataPoint]


class ZoneStatsResponse(BaseModel):
    """Statistics for a specific zone."""
    zone_id: int
    zone_name: str
    total_entries: int
    total_exits: int
    peak_entries_hour: Optional[int]
    peak_exits_hour: Optional[int]
    avg_dwell_time_seconds: Optional[float]


class AnalyticsSummaryResponse(BaseModel):
    """Complete analytics summary for a job."""
    model_config = ConfigDict(from_attributes=True)
    
    job_id: int
    total_unique_people: int
    total_detections: int
    peak_occupancy: int
    peak_time: Optional[datetime]
    avg_occupancy: Optional[float]
    avg_dwell_time: Optional[float]
    total_entries: int
    total_exits: int
    processing_time_seconds: Optional[float]
    frames_processed: int
    avg_fps: Optional[float]
    anomalies_detected: int
    computed_at: datetime


class DashboardStats(BaseModel):
    """Real-time dashboard statistics."""
    current_occupancy: int
    total_people_today: int
    active_jobs: int
    pending_alerts: int
    buses_online: int
    avg_processing_fps: float


# =============================================================================
# CHART DATA SCHEMAS
# =============================================================================

class ChartDataSeries(BaseModel):
    """Single series for chart."""
    name: str
    data: List[Dict[str, Any]]  # [{x: value, y: value}, ...]
    color: Optional[str] = None


class ChartData(BaseModel):
    """Chart data with multiple series."""
    title: str
    x_axis_label: str
    y_axis_label: str
    series: List[ChartDataSeries]


# =============================================================================
# COMPARISON SCHEMAS
# =============================================================================

class RouteComparison(BaseModel):
    """Compare analytics across routes."""
    route_id: int
    route_name: str
    avg_occupancy: float
    total_passengers: int
    peak_hour: Optional[int]


class BusComparison(BaseModel):
    """Compare analytics across buses."""
    bus_id: int
    bus_number: str
    jobs_count: int
    total_passengers: int
    avg_occupancy: float


class ComparisonResponse(BaseModel):
    """Comparison report."""
    start_date: datetime
    end_date: datetime
    items: List[Dict[str, Any]]
