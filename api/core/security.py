"""
Security Module
===============

JWT authentication, password hashing, and API key management.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import settings


# =============================================================================
# PASSWORD HASHING
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


# =============================================================================
# JWT TOKENS
# =============================================================================

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # User ID
    exp: datetime
    type: str  # "access" or "refresh"
    role: Optional[str] = None


def create_access_token(
    user_id: int,
    role: str = "viewer",
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User ID to encode in token
        role: User role for authorization
        expires_delta: Custom expiration time
        
    Returns:
        Encoded JWT token
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "access",
        "role": role,
        "iat": datetime.now(timezone.utc),
    }
    
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """
    Create a JWT refresh token.
    
    Refresh tokens have longer expiration and can be used
    to obtain new access tokens.
    """
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "type": "refresh",
        "iat": datetime.now(timezone.utc),
    }
    
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_token_pair(user_id: int, role: str = "viewer") -> Tuple[str, str]:
    """
    Create both access and refresh tokens.
    
    Returns:
        Tuple of (access_token, refresh_token)
    """
    access_token = create_access_token(user_id, role)
    refresh_token = create_refresh_token(user_id)
    return access_token, refresh_token


def decode_token(token: str) -> Optional[TokenPayload]:
    """
    Decode and validate a JWT token.
    
    Returns:
        TokenPayload if valid, None if invalid/expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# =============================================================================
# API KEYS
# =============================================================================

import secrets
import hashlib


def generate_api_key() -> Tuple[str, str]:
    """
    Generate a new API key.
    
    Returns:
        Tuple of (raw_key, hashed_key)
        - raw_key: Show to user once, never stored
        - hashed_key: Store in database
    """
    raw_key = f"bv_{secrets.token_urlsafe(32)}"
    hashed_key = hash_api_key(raw_key)
    return raw_key, hashed_key


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(raw_key: str, hashed_key: str) -> bool:
    """Verify an API key against its hash."""
    return hash_api_key(raw_key) == hashed_key


# =============================================================================
# ROLE-BASED ACCESS CONTROL
# =============================================================================

ROLES = {
    "admin": ["read", "write", "delete", "manage_users", "manage_system"],
    "operator": ["read", "write", "delete"],
    "viewer": ["read"],
}


def has_permission(role: str, permission: str) -> bool:
    """Check if a role has a specific permission."""
    return permission in ROLES.get(role, [])


def check_role(required_role: str, user_role: str) -> bool:
    """
    Check if user role meets minimum requirement.
    
    Role hierarchy: admin > operator > viewer
    """
    hierarchy = ["viewer", "operator", "admin"]
    
    if required_role not in hierarchy or user_role not in hierarchy:
        return False
    
    return hierarchy.index(user_role) >= hierarchy.index(required_role)
