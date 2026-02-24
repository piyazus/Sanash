"""
GPS Handler
===========

Reads NMEA sentences from a USB GPS module (e.g., u-blox) over serial.
Falls back to mock coordinates (Almaty city center) if GPS unavailable.

Usage:
    handler = GPSHandler(device="/dev/ttyUSB0", baud_rate=9600)
    lat, lon = handler.get_location()
"""
import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

# Almaty city center — used as fallback
_ALMATY_LAT = 43.2220
_ALMATY_LON = 76.8512


class GPSHandler:
    """
    Thread-safe GPS location reader.

    Reads NMEA $GPRMC / $GNRMC sentences in a background thread
    and exposes the latest valid fix via get_location().
    """

    def __init__(
        self,
        device: str = "/dev/ttyUSB0",
        baud_rate: int = 9600,
        timeout: float = 2.0,
    ):
        self._device = device
        self._baud_rate = baud_rate
        self._timeout = timeout
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if device == "mock":
            log.info("GPS: mock mode — using Almaty city center coordinates")
        else:
            self._start_reader()

    def _start_reader(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        """Background thread: continuously parse NMEA sentences."""
        try:
            import serial
        except ImportError:
            log.warning("pyserial not installed — GPS disabled")
            return

        while self._running:
            try:
                with serial.Serial(
                    self._device, self._baud_rate, timeout=self._timeout
                ) as ser:
                    log.info(f"GPS: connected to {self._device}")
                    while self._running:
                        line = ser.readline().decode("ascii", errors="replace").strip()
                        if line.startswith(("$GPRMC", "$GNRMC")):
                            lat, lon = _parse_rmc(line)
                            if lat is not None:
                                with self._lock:
                                    self._lat, self._lon = lat, lon
            except Exception as exc:
                log.debug(f"GPS read error: {exc} — retrying in 5s")
                time.sleep(5)

    def get_location(self) -> tuple[float, float]:
        """
        Return (latitude, longitude) of the latest GPS fix.
        Falls back to Almaty city center if no fix is available.
        """
        with self._lock:
            if self._lat is not None and self._lon is not None:
                return self._lat, self._lon
        return _ALMATY_LAT, _ALMATY_LON

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)


def _parse_rmc(sentence: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse $GPRMC or $GNRMC NMEA sentence.

    Returns (lat, lon) in decimal degrees, or (None, None) if invalid.

    NMEA format: $GPRMC,HHMMSS,A,DDMM.MMMM,N,DDDMM.MMMM,E,...
    """
    try:
        parts = sentence.split(",")
        if len(parts) < 7:
            return None, None
        status = parts[2]
        if status != "A":  # 'A' = active/valid fix
            return None, None

        lat_raw = parts[3]
        lat_dir = parts[4]
        lon_raw = parts[5]
        lon_dir = parts[6]

        lat = _nmea_to_decimal(lat_raw, lat_dir)
        lon = _nmea_to_decimal(lon_raw, lon_dir)
        return lat, lon
    except (IndexError, ValueError):
        return None, None


def _nmea_to_decimal(raw: str, direction: str) -> float:
    """Convert NMEA DDDMM.MMMM to decimal degrees."""
    if not raw:
        raise ValueError("Empty NMEA coordinate")
    dot = raw.index(".")
    degrees = float(raw[:dot - 2])
    minutes = float(raw[dot - 2:])
    decimal = degrees + minutes / 60.0
    if direction in ("S", "W"):
        decimal = -decimal
    return decimal
