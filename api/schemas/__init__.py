"""API Schemas Package"""

from .common import PaginatedResponse, ErrorResponse, SuccessResponse, HealthResponse
from .user import UserCreate, UserUpdate, UserResponse, UserLogin, TokenResponse, ApiKeyCreate, ApiKeyResponse
from .job import JobCreate, JobUpdate, JobResponse, JobProgress, JobResults, VideoUploadResponse
from .bus import BusCreate, BusUpdate, BusResponse, CameraCreate, CameraResponse, ZoneCreate, ZoneResponse, RouteCreate, RouteResponse
from .analytics import OccupancyResponse, HeatmapResponse, FlowResponse, AnalyticsSummaryResponse, DashboardStats

__all__ = [
    "PaginatedResponse", "ErrorResponse", "SuccessResponse", "HealthResponse",
    "UserCreate", "UserUpdate", "UserResponse", "UserLogin", "TokenResponse", "ApiKeyCreate", "ApiKeyResponse",
    "JobCreate", "JobUpdate", "JobResponse", "JobProgress", "JobResults", "VideoUploadResponse",
    "BusCreate", "BusUpdate", "BusResponse", "CameraCreate", "CameraResponse", "ZoneCreate", "ZoneResponse", "RouteCreate", "RouteResponse",
    "OccupancyResponse", "HeatmapResponse", "FlowResponse", "AnalyticsSummaryResponse", "DashboardStats",
]
