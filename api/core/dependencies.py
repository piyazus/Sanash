"""
FastAPI Dependencies
====================

Dependency injection for authentication, database, and rate limiting.
"""

from typing import Optional

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .database import get_db
from .security import decode_token, check_role, hash_api_key
from .cache import get_redis, RateLimiter


# =============================================================================
# OAUTH2 SCHEME
# =============================================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/login",
    auto_error=False,
)


# =============================================================================
# CURRENT USER
# =============================================================================

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current authenticated user from JWT token.
    
    Raises HTTPException if token is invalid or user not found.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if payload.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    
    # Import here to avoid circular imports
    from api.models.user import User
    
    result = await db.execute(
        select(User).where(User.id == int(payload.sub))
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is deactivated",
        )
    
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
):
    """Get current user if authenticated, None otherwise."""
    if not token:
        return None
    
    try:
        return await get_current_user(token, db)
    except HTTPException:
        return None


# =============================================================================
# ROLE-BASED ACCESS
# =============================================================================

def require_role(required_role: str):
    """
    Dependency factory for role-based access control.
    
    Usage:
        @router.get("/admin")
        async def admin_only(user = Depends(require_role("admin"))):
            ...
    """
    async def role_checker(user = Depends(get_current_user)):
        if not check_role(required_role, user.role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role} role or higher",
            )
        return user
    
    return role_checker


# =============================================================================
# API KEY AUTHENTICATION
# =============================================================================

async def get_api_key_user(
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate via API key header.
    
    Used for external integrations.
    """
    if not api_key:
        return None
    
    from api.models.user import ApiKey, User
    
    key_hash = hash_api_key(api_key)
    
    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash)
    )
    api_key_obj = result.scalar_one_or_none()
    
    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    # Check expiration
    from datetime import datetime, timezone
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired",
        )
    
    # Update last used
    api_key_obj.last_used = datetime.now(timezone.utc)
    await db.commit()
    
    # Get user
    result = await db.execute(
        select(User).where(User.id == api_key_obj.user_id)
    )
    return result.scalar_one_or_none()


# =============================================================================
# COMBINED AUTH
# =============================================================================

async def get_authenticated_user(
    token_user = Depends(get_current_user_optional),
    api_key_user = Depends(get_api_key_user),
):
    """
    Get authenticated user from either JWT or API key.
    
    Tries JWT first, falls back to API key.
    """
    user = token_user or api_key_user
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    
    return user


# =============================================================================
# RATE LIMITING
# =============================================================================

async def check_rate_limit(
    user = Depends(get_current_user_optional),
):
    """
    Rate limiting dependency.
    
    Uses user ID if authenticated, IP otherwise.
    """
    redis_client = await get_redis()
    limiter = RateLimiter(
        redis_client,
        max_requests=settings.RATE_LIMIT_PER_MINUTE,
        window_seconds=60,
    )
    
    # Use user ID or "anonymous" for rate limit key
    key = f"user:{user.id}" if user else "anonymous"
    
    if not await limiter.is_allowed(key):
        remaining = await limiter.get_remaining(key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )
