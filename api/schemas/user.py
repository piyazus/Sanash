"""
Pydantic Schemas for Users and Auth
===================================
"""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, EmailStr, Field, ConfigDict


# =============================================================================
# AUTH SCHEMAS
# =============================================================================

class UserLogin(BaseModel):
    """Login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


# =============================================================================
# USER SCHEMAS
# =============================================================================

class UserCreate(BaseModel):
    """Create new user."""
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: Optional[str] = None
    role: str = "viewer"


class UserUpdate(BaseModel):
    """Update user."""
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]


class UserListItem(BaseModel):
    """User in list view."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool


# =============================================================================
# API KEY SCHEMAS
# =============================================================================

class ApiKeyCreate(BaseModel):
    """Create API key."""
    name: str
    expires_in_days: Optional[int] = 365


class ApiKeyResponse(BaseModel):
    """API key response (includes key only on creation)."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    key: Optional[str] = None  # Only shown on creation
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]


class ApiKeyListItem(BaseModel):
    """API key in list view."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]
