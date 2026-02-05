"""
User Models
===========

User, authentication, and API key models.
"""

from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class User(BaseModel):
    """
    User account model.
    
    Supports email/password authentication and role-based access.
    """
    
    __tablename__ = "users"
    
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Role: admin, operator, viewer
    role: Mapped[str] = mapped_column(String(50), default="viewer")
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    api_keys: Mapped[List["ApiKey"]] = relationship(
        "ApiKey", back_populates="user", cascade="all, delete-orphan"
    )
    jobs: Mapped[List["DetectionJob"]] = relationship(
        "DetectionJob", back_populates="user"
    )
    
    def __repr__(self):
        return f"<User {self.email}>"


class ApiKey(BaseModel):
    """
    API key for external integrations.
    
    Keys are hashed before storage - raw key shown only once on creation.
    """
    
    __tablename__ = "api_keys"
    
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<ApiKey {self.name}>"


class RefreshToken(BaseModel):
    """
    Refresh token for JWT rotation.
    
    Stored in database to enable revocation.
    """
    
    __tablename__ = "refresh_tokens"
    
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Device info for security auditing
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))


# Import for type hints
from .job import DetectionJob
