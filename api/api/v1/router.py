"""
API v1 Router
=============

Main router that includes all v1 endpoints.
"""

from fastapi import APIRouter

from .endpoints import auth, jobs, videos, analytics, buses, cameras, zones, alerts, health, multi_camera, public, mobile

api_router = APIRouter()

# Include all endpoint routers
# Public endpoints (no authentication required)
api_router.include_router(public.router, prefix="/public", tags=["Public Occupancy"])
api_router.include_router(mobile.router, prefix="/mobile", tags=["Mobile App"])

# Protected endpoints
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Detection Jobs"])
api_router.include_router(videos.router, prefix="/videos", tags=["Video Management"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
api_router.include_router(buses.router, prefix="/buses", tags=["Bus Fleet"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["Cameras"])
api_router.include_router(zones.router, prefix="/zones", tags=["Zones"])
api_router.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(multi_camera.router)  # Multi-camera tracking
