"""
=============================================================
  BUS TRACKER — VPS Сервер (FastAPI + PostgreSQL)
=============================================================
  Установка:
    pip install fastapi uvicorn asyncpg python-dotenv

  Запуск:
    uvicorn server:app --host 0.0.0.0 --port 8000

  Nginx проксирует на этот порт (см. nginx.conf)
=============================================================
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncpg
import os
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
#  КОНФИГ
# ══════════════════════════════════════════════════════════
DB_URL  = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/busdb")
API_KEY = os.getenv("API_KEY", "your-secret-api-key")

log = logging.getLogger("BusServer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Bus Tracker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # в продакшне замени на домен сайта
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════
#  БАЗА ДАННЫХ
# ══════════════════════════════════════════════════════════
db_pool: asyncpg.Pool = None

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)
    await init_db()
    log.info("БД подключена ✓")

@app.on_event("shutdown")
async def shutdown():
    await db_pool.close()

async def init_db():
    async with db_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS buses (
                bus_id        TEXT PRIMARY KEY,
                route_number  TEXT,
                capacity      INT DEFAULT 80,
                created_at    TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS telemetry (
                id            BIGSERIAL PRIMARY KEY,
                bus_id        TEXT NOT NULL,
                timestamp     TIMESTAMPTZ NOT NULL,
                total_onboard INT NOT NULL,
                entered       INT DEFAULT 0,
                exited        INT DEFAULT 0,
                occupancy_pct FLOAT,
                created_at    TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_telemetry_bus_time
                ON telemetry (bus_id, timestamp DESC);

            -- Последние данные каждого автобуса (обновляется при каждом пакете)
            CREATE TABLE IF NOT EXISTS bus_status (
                bus_id        TEXT PRIMARY KEY,
                route_number  TEXT,
                total_onboard INT DEFAULT 0,
                capacity      INT DEFAULT 80,
                occupancy_pct FLOAT DEFAULT 0,
                last_seen     TIMESTAMPTZ DEFAULT NOW(),
                latitude      FLOAT,
                longitude     FLOAT
            );
        """)

async def get_db():
    async with db_pool.acquire() as conn:
        yield conn


# ══════════════════════════════════════════════════════════
#  АУТЕНТИФИКАЦИЯ (API ключ для Jetson)
# ══════════════════════════════════════════════════════════
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Неверный API ключ")
    return x_api_key


# ══════════════════════════════════════════════════════════
#  МОДЕЛИ ДАННЫХ
# ══════════════════════════════════════════════════════════
class TelemetryPayload(BaseModel):
    bus_id:        str
    timestamp:     str
    total_onboard: int
    entered:       int = 0
    exited:        int = 0
    capacity:      int = 80
    occupancy_pct: float = 0.0

class GPSPayload(BaseModel):
    bus_id:    str
    latitude:  float
    longitude: float


# ══════════════════════════════════════════════════════════
#  ЭНДПОИНТЫ ДЛЯ JETSON
# ══════════════════════════════════════════════════════════

@app.post("/api/buses/telemetry", tags=["Jetson → Server"])
async def receive_telemetry(
    payload: TelemetryPayload,
    db=Depends(get_db),
    _=Depends(verify_api_key)
):
    """Принимает данные о пассажирах от Jetson на автобусе"""

    ts = datetime.fromisoformat(payload.timestamp.replace("Z", "+00:00"))

    # Сохраняем в историю
    await db.execute("""
        INSERT INTO telemetry (bus_id, timestamp, total_onboard, entered, exited, occupancy_pct)
        VALUES ($1, $2, $3, $4, $5, $6)
    """, payload.bus_id, ts, payload.total_onboard,
        payload.entered, payload.exited, payload.occupancy_pct)

    # Обновляем текущий статус
    await db.execute("""
        INSERT INTO bus_status (bus_id, total_onboard, capacity, occupancy_pct, last_seen)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (bus_id) DO UPDATE SET
            total_onboard = $2,
            capacity      = $3,
            occupancy_pct = $4,
            last_seen     = NOW()
    """, payload.bus_id, payload.total_onboard, payload.capacity, payload.occupancy_pct)

    log.info(f"[{payload.bus_id}] {payload.total_onboard} чел. ({payload.occupancy_pct}%)")
    return {"status": "ok"}


@app.post("/api/buses/gps", tags=["Jetson → Server"])
async def receive_gps(
    payload: GPSPayload,
    db=Depends(get_db),
    _=Depends(verify_api_key)
):
    """GPS координаты автобуса (если есть GPS модуль)"""
    await db.execute("""
        UPDATE bus_status SET latitude=$2, longitude=$3
        WHERE bus_id=$1
    """, payload.bus_id, payload.latitude, payload.longitude)
    return {"status": "ok"}


# ══════════════════════════════════════════════════════════
#  ЭНДПОИНТЫ ДЛЯ САЙТА / ПРИЛОЖЕНИЯ
# ══════════════════════════════════════════════════════════

@app.get("/api/buses", tags=["Сайт"])
async def get_all_buses(db=Depends(get_db)):
    """Все автобусы с текущим статусом (для карты)"""
    rows = await db.fetch("""
        SELECT
            bus_id,
            route_number,
            total_onboard,
            capacity,
            occupancy_pct,
            last_seen,
            latitude,
            longitude,
            CASE
                WHEN occupancy_pct >= 90 THEN 'full'
                WHEN occupancy_pct >= 60 THEN 'busy'
                WHEN occupancy_pct >= 30 THEN 'normal'
                ELSE 'empty'
            END AS status
        FROM bus_status
        ORDER BY bus_id
    """)
    return [dict(r) for r in rows]


@app.get("/api/buses/{bus_id}", tags=["Сайт"])
async def get_bus(bus_id: str, db=Depends(get_db)):
    """Детальная информация об одном автобусе"""
    row = await db.fetchrow(
        "SELECT * FROM bus_status WHERE bus_id = $1", bus_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="Автобус не найден")
    return dict(row)


@app.get("/api/buses/{bus_id}/history", tags=["Сайт"])
async def get_bus_history(bus_id: str, hours: int = 2, db=Depends(get_db)):
    """История заполненности за последние N часов"""
    rows = await db.fetch("""
        SELECT timestamp, total_onboard, occupancy_pct, entered, exited
        FROM telemetry
        WHERE bus_id = $1
          AND timestamp > NOW() - ($2 || ' hours')::interval
        ORDER BY timestamp DESC
        LIMIT 200
    """, bus_id, str(hours))
    return [dict(r) for r in rows]


@app.get("/api/stats", tags=["Сайт"])
async def get_system_stats(db=Depends(get_db)):
    """Общая статистика по всем автобусам"""
    row = await db.fetchrow("""
        SELECT
            COUNT(*)                                        AS total_buses,
            SUM(total_onboard)                              AS total_passengers,
            AVG(occupancy_pct)                              AS avg_occupancy,
            COUNT(*) FILTER (WHERE occupancy_pct >= 90)     AS full_buses,
            COUNT(*) FILTER (WHERE last_seen > NOW() - INTERVAL '1 minute') AS online_buses
        FROM bus_status
    """)
    return dict(row)


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
