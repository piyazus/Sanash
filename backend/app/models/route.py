"""
Bus route (e.g., Route 68 — Almaty).
"""
import uuid

from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Route(Base):
    __tablename__ = "routes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    route_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(180), nullable=False)
    num_stops: Mapped[int] = mapped_column(Integer, default=0)

    buses: Mapped[list["Bus"]] = relationship("Bus", back_populates="route")
