"""
Base Model
==========

Common fields and mixins for all ORM models.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from api.core.database import Base


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=True,
    )


class BaseModel(Base, TimestampMixin):
    """
    Base model with ID and timestamps.
    
    All models should inherit from this.
    """
    
    __abstract__ = True
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
