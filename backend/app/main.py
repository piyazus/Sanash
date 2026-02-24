"""
SANASH FastAPI Application
==========================

Passenger occupancy monitoring backend for Almaty transit operators.

Run locally:
    uvicorn app.main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import engine, Base
from app.api.routes import auth, occupancy, analytics, alerts, buses

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create all tables on startup (use Alembic migrations in production)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("database_ready", tables="created_or_verified")
    yield
    await engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Real-time bus occupancy monitoring API. "
        "Jetson edge devices push readings; operators view data via dashboard."
    ),
    lifespan=lifespan,
)

# CORS — allow React dashboard and mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
API_PREFIX = "/api/v1"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(occupancy.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)
app.include_router(alerts.router, prefix=API_PREFIX)
app.include_router(buses.router, prefix=API_PREFIX)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/api/v1/health", tags=["health"])
async def api_health():
    return {"status": "ok", "version": settings.APP_VERSION}
