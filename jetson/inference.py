"""
SANASH Jetson Inference Pipeline
=================================

Main loop: capture frame -> CSRNet inference -> push to API.
Runs as a systemd service on NVIDIA Jetson Nano.

Usage:
    python jetson/inference.py --config /opt/sanash/config.yaml
    python jetson/inference.py --config jetson/config.yaml --mock
"""
import argparse
import logging
import logging.handlers
import signal
import sys
import time
from pathlib import Path

import numpy as np
import yaml

log = logging.getLogger("sanash.inference")


def setup_logging(log_file: str, level: str = "INFO", max_bytes: int = 10485760, backup_count: int = 3) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def open_camera(device_id: int, width: int, height: int):
    """Open OpenCV video capture. Returns capture object or None."""
    try:
        import cv2
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            log.error(f"Camera {device_id} not available")
            return None
        log.info(f"Camera opened: /dev/video{device_id} ({width}x{height})")
        return cap
    except ImportError:
        log.warning("OpenCV not available — camera disabled")
        return None


def capture_frame(cap) -> np.ndarray | None:
    """Capture one frame from the camera. Returns RGB array or None."""
    import cv2
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def generate_mock_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generate a synthetic frame for testing without a camera."""
    rng = np.random.default_rng()
    return rng.integers(80, 200, size=(height, width, 3), dtype=np.uint8)


class InferencePipeline:
    """
    Orchestrates the full capture → infer → upload cycle.

    Attributes
    ----------
    config : dict
        Parsed config.yaml
    mock : bool
        If True, use synthetic frames and mock model
    """

    def __init__(self, config: dict, mock: bool = False):
        self.cfg = config
        self.mock = mock
        self._running = True

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._setup_components()

    def _handle_signal(self, signum, frame):
        log.info(f"Received signal {signum} — shutting down...")
        self._running = False

    def _setup_components(self) -> None:
        from jetson.model_loader import load_model
        from jetson.api_client import APIClient
        from jetson.gps_handler import GPSHandler

        inf_cfg = self.cfg["inference"]
        cam_cfg = self.cfg["camera"]
        api_cfg = self.cfg["api"]
        gps_cfg = self.cfg.get("gps", {})
        buf_cfg = self.cfg.get("offline_buffer", {})

        # Model
        self.model = load_model(
            trt_path=inf_cfg.get("model_path") if not self.mock else None,
            pth_path=inf_cfg.get("fallback_model_path") if not self.mock else None,
            width=inf_cfg["input_width"],
            height=inf_cfg["input_height"],
        )

        # Camera
        if self.mock:
            self.cap = None
        else:
            self.cap = open_camera(
                cam_cfg["device_id"], cam_cfg["width"], cam_cfg["height"]
            )

        # GPS
        gps_device = "mock" if self.mock else gps_cfg.get("device", "mock")
        self.gps = GPSHandler(device=gps_device)

        # API client
        self.client = APIClient(
            endpoint=api_cfg["endpoint"],
            bus_id=self.cfg["device"]["bus_id"],
            offline_db_path=buf_cfg.get("db_path", "/opt/sanash/offline_buffer.db"),
            timeout=api_cfg.get("timeout_seconds", 10),
            retry_attempts=api_cfg.get("retry_attempts", 3),
            retry_delay=api_cfg.get("retry_delay_seconds", 5),
        )

        self.interval = self.cfg["inference"]["interval_seconds"]
        log.info(
            f"Pipeline ready — bus={self.cfg['device']['bus_id']}, "
            f"interval={self.interval}s, mock={self.mock}"
        )

    def run(self) -> None:
        """Main inference loop."""
        log.info("Starting inference loop")
        cycle = 0

        while self._running:
            cycle_start = time.monotonic()
            cycle += 1

            try:
                # Capture frame
                if self.mock or self.cap is None:
                    frame = generate_mock_frame(
                        self.cfg["camera"]["width"], self.cfg["camera"]["height"]
                    )
                else:
                    frame = capture_frame(self.cap)
                    if frame is None:
                        log.warning("Frame capture failed — skipping cycle")
                        time.sleep(5)
                        continue

                # Inference
                t0 = time.monotonic()
                count, density = self.model.predict(frame)
                latency_ms = (time.monotonic() - t0) * 1000
                count_int = max(0, int(round(count)))

                # GPS
                lat, lon = self.gps.get_location()

                # Confidence: ratio of density map peak to mean (heuristic)
                confidence = float(
                    min(1.0, density.max() / (density.mean() + 1e-6) / 10.0)
                )

                # Upload
                ok = self.client.push_reading(
                    passenger_count=count_int,
                    latitude=lat,
                    longitude=lon,
                    confidence=round(confidence, 3),
                )

                log.info(
                    f"[cycle={cycle}] count={count_int}, lat={lat:.4f}, lon={lon:.4f}, "
                    f"latency={latency_ms:.1f}ms, upload={'OK' if ok else 'BUFFERED'}, "
                    f"buffered={self.client.pending_count()}"
                )

            except Exception as exc:
                log.exception(f"Cycle {cycle} error: {exc}")

            # Sleep until next interval
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, self.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.info("Inference loop stopped")
        self._cleanup()

    def _cleanup(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.gps.stop()


def main():
    parser = argparse.ArgumentParser(description="SANASH Jetson Inference Pipeline")
    parser.add_argument(
        "--config", default="jetson/config.yaml",
        help="Path to config.yaml (default: jetson/config.yaml)"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock camera and model (no hardware required)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_cfg = cfg.get("logging", {})
    setup_logging(
        log_file=log_cfg.get("log_file", "/tmp/sanash_inference.log"),
        level=log_cfg.get("level", "INFO"),
        max_bytes=log_cfg.get("max_bytes", 10485760),
        backup_count=log_cfg.get("backup_count", 3),
    )

    pipeline = InferencePipeline(cfg, mock=args.mock)
    pipeline.run()


if __name__ == "__main__":
    main()
