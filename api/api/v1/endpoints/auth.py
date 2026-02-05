"""
Authentication Endpoints
========================
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.database import get_db
from api.core.security import (
    hash_password,
    verify_password,
    create_token_pair,
    decode_token,
    generate_api_key,
)
from api.core.dependencies import get_current_user
from api.models.user import User, ApiKey, RefreshToken
from api.schemas.user import (
    UserLogin,
    TokenResponse,
    RefreshTokenRequest,
    UserCreate,
    UserResponse,
    ApiKeyCreate,
    ApiKeyResponse,
)

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new user.
    
    - Email must be unique
    - Password must be at least 8 characters
    """
    # Check if email exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Create user
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
        role="viewer",  # Default role
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
):
    """
    Login with email and password.
    
    Returns access and refresh tokens.
    """
    # Find user
    result = await db.execute(
        select(User).where(User.email == credentials.email)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )
    
    # Update last login
    user.last_login = datetime.now(timezone.utc)
    await db.commit()
    
    # Generate tokens
    access_token, refresh_token = create_token_pair(user.id, user.role)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=30 * 60,  # 30 minutes
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Get new access token using refresh token.
    """
    payload = decode_token(request.refresh_token)
    
    if not payload or payload.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )
    
    # Get user
    result = await db.execute(
        select(User).where(User.id == int(payload.sub))
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Generate new tokens
    access_token, refresh_token = create_token_pair(user.id, user.role)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=30 * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    user: User = Depends(get_current_user),
):
    """Get current authenticated user info."""
    return user


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_data: ApiKeyCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new API key.
    
    The raw key is only shown once - save it immediately.
    """
    from datetime import timedelta
    
    raw_key, hashed_key = generate_api_key()
    
    expires_at = None
    if key_data.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=key_data.expires_in_days)
    
    api_key = ApiKey(
        user_id=user.id,
        key_hash=hashed_key,
        name=key_data.name,
        expires_at=expires_at,
    )
    
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    
    return ApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,  # Only shown once
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
        last_used=None,
    )


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an API key."""
    result = await db.execute(
        select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user.id)
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    
    await db.delete(api_key)
    await db.commit()
    
    return {"success": True, "message": "API key deleted"}
