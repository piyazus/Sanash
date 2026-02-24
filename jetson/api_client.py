"""
API Client
==========

HTTP client for pushing occupancy readings to the SANASH backend.
Features:
- Retry with exponential backoff
- SQLite offline buffer (sync on reconnect)
- Structured logging

Usage:
    client = APIClient(endpoint="https://api.sanash.kz", bus_id="BUS_001")
    client.push_reading(passenger_count=42, lat=43.22, lon=76.85)
"""
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib import request, error as urllib_error

log = logging.getLogger(__name__)

_ENDPOINT_PATH = "/api/v1/occupancy/reading"


class APIClient:
    """
    Pushes occupancy readings to the SANASH FastAPI backend.

    When the network is unavailable, readings are stored in a local SQLite
    database and uploaded in batch on the next successful connection.
    """

    def __init__(
        self,
        endpoint: str,
        bus_id: str,
        offline_db_path: str = "/opt/sanash/offline_buffer.db",
        timeout: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 5.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.bus_id = bus_id
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._db_path = offline_db_path
        self._init_db()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_counts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    synced INTEGER DEFAULT 0
                )
                """
            )
            conn.commit()

    def push_reading(
        self,
        passenger_count: int,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """
        Send one reading to the backend.

        Returns True if successfully uploaded (live or from buffer).
        """
        payload = {
            "bus_id": self.bus_id,
            "passenger_count": passenger_count,
            "device_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if latitude is not None:
            payload["latitude"] = round(latitude, 6)
        if longitude is not None:
            payload["longitude"] = round(longitude, 6)
        if confidence is not None:
            payload["confidence"] = round(confidence, 4)

        # Try to sync buffered readings first
        self._sync_buffer()

        success = self._post(payload)
        if not success:
            self._buffer(payload)
            log.warning(
                f"Upload failed — buffered reading (count={passenger_count})"
            )
        return success

    def _post(self, payload: dict, retries: int | None = None) -> bool:
        """POST payload to backend. Returns True on HTTP 200/201."""
        url = self.endpoint + _ENDPOINT_PATH
        body = json.dumps(payload).encode("utf-8")
        attempts = retries if retries is not None else self.retry_attempts

        for attempt in range(1, attempts + 1):
            try:
                req = request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status in (200, 201):
                        log.debug(f"Upload OK ({resp.status})")
                        return True
            except urllib_error.HTTPError as exc:
                log.warning(f"HTTP {exc.code} on attempt {attempt}/{attempts}")
            except Exception as exc:
                log.debug(f"Network error on attempt {attempt}/{attempts}: {exc}")

            if attempt < attempts:
                time.sleep(self.retry_delay * attempt)

        return False

    def _buffer(self, payload: dict) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO pending_counts (payload, created_at) VALUES (?, ?)",
                (json.dumps(payload), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def _sync_buffer(self) -> None:
        """Upload all pending buffered readings."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, payload FROM pending_counts WHERE synced = 0 ORDER BY id LIMIT 50"
            ).fetchall()

        if not rows:
            return

        log.info(f"Syncing {len(rows)} buffered readings...")
        synced_ids = []
        for row_id, payload_str in rows:
            payload = json.loads(payload_str)
            if self._post(payload, retries=1):
                synced_ids.append(row_id)
            else:
                break  # Network still down — stop trying

        if synced_ids:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    f"UPDATE pending_counts SET synced = 1 WHERE id IN "
                    f"({','.join('?' * len(synced_ids))})",
                    synced_ids,
                )
                conn.commit()
            log.info(f"Synced {len(synced_ids)} buffered readings")

    def pending_count(self) -> int:
        """Return number of unsynced buffered readings."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM pending_counts WHERE synced = 0"
            ).fetchone()
        return row[0] if row else 0
