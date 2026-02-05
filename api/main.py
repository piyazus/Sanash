"""
Bus Vision API - Main Application
==================================

FastAPI application entry point with all routes and middleware.
"""

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.config import settings
from api.core.database import init_db, close_db
from api.core.cache import init_redis, close_redis
from api.api.v1.router import api_router


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Initializes and cleans up resources.
    """
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    # Initialize database
    await init_db()
    print("Database initialized")
    
    # Initialize Redis
    await init_redis()
    print("Redis connected")
    
    # Create upload directory
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Shutdown
    print("Shutting down...")
    await close_db()
    await close_redis()


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise bus passenger detection and analytics platform",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# MIDDLEWARE
# =============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    duration = (datetime.now() - start_time).total_seconds() * 1000
    
    if settings.DEBUG:
        print(f"{request.method} {request.url.path} - {response.status_code} ({duration:.1f}ms)")
    
    return response


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    if settings.DEBUG:
        import traceback
        traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None,
        },
    )


# =============================================================================
# ROUTES
# =============================================================================

# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# Root endpoint
@app.get("/")
async def root():
    """API root - redirect to docs."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/api/docs",
    }


# Health check
@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "healthy"}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
    )
