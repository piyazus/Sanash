"""
Bus — physical vehicle with Jetson device.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Bus(Base):
    __tablename__ = "buses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    bus_id: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    plate_number: Mapped[str] = mapped_column(String(20), nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, default=60)

    operator_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("operators.id"), nullable=False
    )
    route_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("routes.id"), nullable=True
    )

    # Live state (updated by each inference push)
    current_count: Mapped[int] = mapped_column(Integer, default=0)
    current_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    current_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    operator: Mapped["Operator"] = relationship("Operator", back_populates="buses")
    route: Mapped["Route | None"] = relationship("Route", back_populates="buses")
    occupancy_logs: Mapped[list["OccupancyLog"]] = relationship(
        "OccupancyLog", back_populates="bus"
    )
    alerts: Mapped[list["Alert"]] = relationship("Alert", back_populates="bus")

    @property
    def occupancy_ratio(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return self.current_count / self.capacity

    @property
    def status(self) -> str:
        r = self.occupancy_ratio
        if r >= 0.8:
            return "red"
        if r >= 0.5:
            return "yellow"
        return "green"
