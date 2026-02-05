"""
API Configuration using Pydantic Settings
==========================================

Centralized configuration management with environment variable support.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment variables can be set directly or via .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==========================================================================
    # APPLICATION
    # ==========================================================================
    APP_NAME: str = "Bus Vision API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # ==========================================================================
    # SERVER
    # ==========================================================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # ==========================================================================
    # DATABASE (PostgreSQL)
    # ==========================================================================
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/bus_vision"
    DATABASE_ECHO: bool = False  # Log SQL queries
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    
    # ==========================================================================
    # REDIS
    # ==========================================================================
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 3600  # 1 hour default cache TTL
    
    # ==========================================================================
    # CELERY
    # ==========================================================================
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # ==========================================================================
    # SECURITY
    # ==========================================================================
    SECRET_KEY: str = "change-this-in-production-use-openssl-rand-hex-32"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API Key settings
    API_KEY_HEADER: str = "X-API-Key"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # ==========================================================================
    # FILE STORAGE
    # ==========================================================================
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE_MB: int = 5120  # 5GB
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    
    # S3 (optional)
    USE_S3: bool = False
    S3_BUCKET: str = ""
    S3_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    
    # ==========================================================================
    # DETECTION ENGINE
    # ==========================================================================
    MODEL_PATH: str = "yolov8m.pt"
    DEFAULT_CONFIDENCE: float = 0.5
    DEFAULT_FRAME_SKIP: int = 2
    MAX_VIDEO_DURATION_HOURS: int = 8
    
    # ==========================================================================
    # NOTIFICATIONS
    # ==========================================================================
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "noreply@busvision.local"
    
    WEBHOOK_URL: Optional[str] = None
    
    @property
    def async_database_url(self) -> str:
        """Ensure we use asyncpg driver."""
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url
    
    @property
    def sync_database_url(self) -> str:
        """Sync URL for Alembic migrations."""
        url = self.DATABASE_URL
        if "+asyncpg" in url:
            url = url.replace("+asyncpg", "")
        return url


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures settings are only loaded once.
    """
    return Settings()


# Global settings instance
settings = get_settings()
