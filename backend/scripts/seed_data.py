"""
Seed Data Script
================

Populates the SANASH database with demo data for development and testing:
- 1 operator account (demo@sanash.kz / demo1234)
- 5 bus routes (Almaty routes 12, 21, 37, 68, 92)
- 10 buses spread across routes
- 500 occupancy log entries (last 24 hours, synthetic pattern)

Usage:
    python scripts/seed_data.py
    python scripts/seed_data.py --db-url postgresql://user:pass@host/db
"""
import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.security import hash_password
from app.models import Bus, OccupancyLog, Operator, Route
from app.models.alert import Alert
from app.core.database import Base

DEMO_EMAIL = "demo@sanash.kz"
DEMO_PASSWORD = "demo1234"
DEMO_COMPANY = "Almaty Passenger Transit"

ROUTES = [
    ("12", "Airport — Central Station"),
    ("21", "Almaly — Dostyk"),
    ("37", "Sayran — Alatau"),
    ("68", "Medeu — Train Station"),
    ("92", "Bostandyk — Auezov"),
]

BUSES = [
    ("BUS_001", "777 AA 01", 60, "12"),
    ("BUS_002", "777 AB 02", 60, "12"),
    ("BUS_003", "777 AC 03", 80, "21"),
    ("BUS_004", "777 AD 04", 60, "21"),
    ("BUS_005", "777 AE 05", 60, "37"),
    ("BUS_006", "777 AF 06", 80, "37"),
    ("BUS_007", "777 AG 07", 60, "68"),
    ("BUS_008", "777 AH 08", 60, "68"),
    ("BUS_009", "777 AI 09", 80, "92"),
    ("BUS_010", "777 AJ 10", 60, "92"),
]

# Almaty city center bounding box
LAT_MIN, LAT_MAX = 43.20, 43.28
LON_MIN, LON_MAX = 76.88, 76.96


def _peak_factor(hour: int) -> float:
    """Simulate morning and evening peak occupancy."""
    if 7 <= hour <= 9:
        return 0.85
    if 17 <= hour <= 19:
        return 0.80
    if 12 <= hour <= 13:
        return 0.55
    if 22 <= hour or hour <= 5:
        return 0.15
    return 0.40


def _status(ratio: float) -> str:
    if ratio >= 0.8:
        return "red"
    if ratio >= 0.5:
        return "yellow"
    return "green"


async def seed(db_url: str) -> None:
    async_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
    engine = create_async_engine(async_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as db:
        # --- Operator ---
        existing = await db.execute(
            select(Operator).where(Operator.email == DEMO_EMAIL)
        )
        op = existing.scalar_one_or_none()
        if op is None:
            op = Operator(
                name="Demo Operator",
                email=DEMO_EMAIL,
                hashed_password=hash_password(DEMO_PASSWORD),
                company=DEMO_COMPANY,
            )
            db.add(op)
            await db.flush()
            print(f"  Created operator: {DEMO_EMAIL}")
        else:
            print(f"  Operator exists: {DEMO_EMAIL}")

        # --- Routes ---
        route_map: dict[str, Route] = {}
        for route_number, route_name in ROUTES:
            row = await db.execute(
                select(Route).where(Route.route_number == route_number)
            )
            route = row.scalar_one_or_none()
            if route is None:
                route = Route(route_number=route_number, name=route_name, num_stops=8)
                db.add(route)
                await db.flush()
                print(f"  Created route: {route_number} — {route_name}")
            route_map[route_number] = route

        # --- Buses ---
        bus_map: dict[str, Bus] = {}
        for bus_id, plate, cap, route_num in BUSES:
            row = await db.execute(select(Bus).where(Bus.bus_id == bus_id))
            bus = row.scalar_one_or_none()
            if bus is None:
                bus = Bus(
                    bus_id=bus_id,
                    plate_number=plate,
                    capacity=cap,
                    operator_id=op.id,
                    route_id=route_map[route_num].id,
                )
                db.add(bus)
                await db.flush()
                print(f"  Created bus: {bus_id} ({plate})")
            bus_map[bus_id] = bus

        # --- Occupancy Logs (last 24 hours, every 30s = 2880 readings per bus) ---
        rng = random.Random(42)
        now = datetime.now(timezone.utc)
        log_count = 0
        alert_count = 0

        for bus_id, bus in bus_map.items():
            _, _, capacity, _ = next(b for b in BUSES if b[0] == bus_id)
            ts = now - timedelta(hours=24)
            interval = timedelta(seconds=30)

            while ts <= now:
                hour = ts.hour
                base_ratio = _peak_factor(hour)
                noise = rng.gauss(0, 0.08)
                ratio = max(0.0, min(1.0, base_ratio + noise))
                count = int(ratio * capacity)
                status = _status(ratio)

                log = OccupancyLog(
                    bus_id=bus.id,
                    timestamp=ts,
                    passenger_count=count,
                    occupancy_ratio=round(ratio, 4),
                    status=status,
                    latitude=rng.uniform(LAT_MIN, LAT_MAX),
                    longitude=rng.uniform(LON_MIN, LON_MAX),
                    confidence=round(rng.uniform(0.85, 0.99), 3),
                )
                db.add(log)
                log_count += 1

                # Create a sample alert for red readings near current time
                if ratio >= 0.85 and (now - ts) < timedelta(hours=1):
                    if rng.random() < 0.05:
                        alert = Alert(
                            bus_id=bus.id,
                            operator_id=op.id,
                            triggered_at=ts,
                            occupancy_ratio=round(ratio, 4),
                            passenger_count=count,
                            message=(
                                f"Bus {bus_id} is at {ratio:.0%} capacity "
                                f"({count}/{capacity} passengers)"
                            ),
                        )
                        db.add(alert)
                        alert_count += 1

                # Update bus live state to latest
                bus.current_count = count
                bus.last_seen = ts
                bus.current_lat = log.latitude
                bus.current_lon = log.longitude

                ts += interval

                # Batch commits to avoid memory issues
                if log_count % 1000 == 0:
                    await db.commit()

        await db.commit()
        print(f"\nSeed complete:")
        print(f"  Logs:   {log_count}")
        print(f"  Alerts: {alert_count}")
        print(f"\nLogin credentials:")
        print(f"  Email:    {DEMO_EMAIL}")
        print(f"  Password: {DEMO_PASSWORD}")

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Seed SANASH demo data")
    parser.add_argument(
        "--db-url",
        default="postgresql://sanash:sanash@localhost:5432/sanash",
        help="PostgreSQL connection URL",
    )
    args = parser.parse_args()
    print(f"Seeding database: {args.db_url}")
    asyncio.run(seed(args.db_url))


if __name__ == "__main__":
    main()
