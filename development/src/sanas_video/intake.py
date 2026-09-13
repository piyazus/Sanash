"""Inspect a local video and prepare timestamped frames for independent annotation.

Requires ffmpeg/ffprobe on PATH; otherwise uses only Python's standard library.
This module never opens a network stream or estimates passenger counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".ts", ".mts", ".m4v", ".webm"}
LABEL_FIELDS = [
    "frame_id", "session_id", "vehicle_id", "channel_id", "origin", "source_sha256",
    "requested_offset_s", "actual_offset_s", "image_file", "image_sha256",
    "scene_id", "observer_count", "visible_count_r1", "visible_count_r2",
    "adjudicated_visible_count", "blind_zones", "quality_notes",
]


def run(command: list[str], timeout: float) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise ValueError(f"{Path(command[0]).name} failed: {result.stderr[-2000:].strip()}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def validate_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Session/channel ID: use 1–64 ASCII letters, digits, underscore or hyphen.")
    return value


def inspect(source: Path, ffprobe: str, timeout: float) -> dict:
    raw = json.loads(run([
        ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-show_format",
        "-show_streams", "-of", "json", str(source),
    ], timeout).stdout)
    videos = [stream for stream in raw.get("streams", []) if stream.get("codec_type") == "video"]
    if not videos:
        raise ValueError("No video stream in the input file.")
    video = videos[0]
    if video.get("disposition", {}).get("attached_pic"):
        raise ValueError("First video stream is a cover image; export the camera video separately.")
    return {"format": raw.get("format", {}), "video_stream": video,
            "video_stream_count": len(videos), "selected_stream": "0:v:0"}


def extract(source: Path, destination: Path, offset: float, ffmpeg: str, timeout: float) -> float:
    # Input timestamps are shifted to a zero-start container timeline. Select
    # the first decoded video frame at/after the requested offset and record
    # its actual PTS from showinfo; do not relabel the request as a measurement.
    result = run([
        ffmpeg, "-hide_banner", "-nostdin", "-n", "-loglevel", "info",
        "-protocol_whitelist", "file,pipe", "-copyts", "-start_at_zero", "-i", str(source),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"select=gte(t\\,{offset:.9f}),showinfo", "-frames:v", "1",
        "-fps_mode", "passthrough", str(destination),
    ], timeout)
    match = re.search(r"\[Parsed_showinfo[^\]]*\].*?\bn:\s*0\b.*?\bpts_time:([\d.eE+-]+)", result.stderr)
    if not destination.is_file() or destination.stat().st_size == 0 or not match:
        raise ValueError(f"No decoded frame at/after {offset:g}s; input may be shorter or unreadable.")
    actual = number(match.group(1))
    if actual is None or actual + 1e-6 < offset:
        raise ValueError(f"Invalid frame timestamp for requested offset {offset:g}s.")
    return actual


def write_review(output: Path, report: dict, rows: list[dict]) -> None:
    cards = []
    for row in rows:
        caption = f"{row['frame_id']} — requested {row['requested_offset_s']}s; actual {row['actual_offset_s']}s"
        cards.append(f'<figure><a href="{row["image_file"]}"><img src="{row["image_file"]}" alt="{html.escape(caption)}"></a>'
                     f'<figcaption>{html.escape(caption)}</figcaption></figure>')
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SANASH — проверка исходного видео</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f5f5f2;color:#202522}
h1{font-size:30px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}
figure{margin:0;background:white;padding:12px;border-radius:8px}img{width:100%;height:auto}figcaption{padding-top:10px;overflow-wrap:anywhere}
.status{padding:16px;background:#fff1c7;border-radius:8px}p{line-height:1.5}</style>
<h1>Проверка исходного видео</h1>'''
    origin = report["origin"]
    status = "СИНТЕТИЧЕСКИЙ ТЕСТ — пассажирские данные отсутствуют." if origin == "synthetic" else "Извлечение кадров выполнено. Подсчёт пассажиров не проводился."
    page += f'<p class="status">{status}</p><p>Источник: {html.escape(report["source_name"])}; тип: {html.escape(origin)}.</p>'
    video = report["probe"]["video_stream"]
    media_format = report["probe"]["format"]
    details = {
        "Кодек": video.get("codec_name", "неизвестно"),
        "Размер закодированного кадра": f'{video.get("width", "?")} × {video.get("height", "?")}',
        "Средняя частота кадров (дробь из метаданных)": video.get("avg_frame_rate", "неизвестно"),
        "Длительность контейнера, с": media_format.get("duration", "неизвестно"),
        "Видеопотоков в файле": report["probe"]["video_stream_count"],
        "Выбранный поток": report["probe"]["selected_stream"],
    }
    page += '<dl>' + ''.join(f'<dt>{html.escape(key)}</dt><dd>{html.escape(str(value))}</dd>'
                           for key, value in details.items()) + '</dl>'
    page += '<p><a href="annotate.html">Открыть разметку кликами</a> или заполните <a href="annotations.csv">annotations.csv</a>: сцена, независимый счёт наблюдателя, два разметчика и слепые зоны. '
    page += 'Пустое поле означает «не измерено», а не ноль. Время кадров отсчитывается от начала временной шкалы контейнера; это не UTC съёмки.</p>'
    page += '<p><a href="metadata.json">Технические сведения и хеши</a>. Кадры сохраняют исходную видимую область; файлы предназначены для локального просмотра.</p>'
    page += '<div class="grid">' + ''.join(cards) + '</div></html>'
    (output / "review.html").write_text(page, encoding="utf-8")


def prepare(source: Path, output: Path, session_id: str, channel_id: str,
            origin: str, offsets: list[float], timeout: float = 120,
            vehicle_id: str | None = None) -> dict:
    source, output = source.resolve(), output.resolve()
    validate_id(session_id)
    validate_id(channel_id)
    if origin not in {"bus", "synthetic", "other"}:
        raise ValueError("Origin must be bus, synthetic, or other.")
    if origin == "bus" and not vehicle_id:
        raise ValueError("A bus export requires --vehicle-id (an operator-confirmed ID or recorded pseudonym).")
    if vehicle_id:
        validate_id(vehicle_id)
    if not source.is_file() or source.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Provide a local video export (.mp4/.mov/.mkv/.avi/.ts/.mts/.m4v/.webm), not a URL or playlist.")
    if output.exists():
        raise ValueError("Output already exists; choose a new run directory to preserve earlier results.")
    if not offsets or len(offsets) > 30 or any(not math.isfinite(t) or t < 0 for t in offsets):
        raise ValueError("Provide 1–30 finite, nonnegative sample offsets.")
    if len(set(offsets)) != len(offsets):
        raise ValueError("Sample offsets must be unique.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be a positive finite number.")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise ValueError("ffmpeg and ffprobe must be available on PATH.")
    initial = source.stat()
    probe = inspect(source, ffprobe, timeout)
    duration = number(probe["format"].get("duration"))
    if duration is not None and any(t >= duration for t in offsets):
        raise ValueError(f"Requested sample reaches/exceeds container duration {duration:g}s; choose earlier --times.")
    source_hash = sha256(source)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    report = {"schema_version": 1, "status": "in_progress", "origin": origin,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_name": source.name, "source_bytes": initial.st_size, "source_sha256": source_hash,
              "session_id": session_id, "vehicle_id": vehicle_id,
              "channel_id": channel_id, "requested_offsets_s": offsets,
              "timestamp_basis": "container timeline shifted to zero by ffmpeg -copyts -start_at_zero; not capture UTC",
              "probe": probe, "frames": [],
              "limitations": ["No passenger inference or occupancy levels.", "No cross-camera deduplication.",
                              "Camera/bus IDs and origin are supplied by the operator, not inferred.",
                              "Selected frames are diagnostics, not independent train/test observations."]}
    report_path = output / "metadata.json"
    try:
        report["tools"] = {name: run([exe, "-version"], timeout).stdout.splitlines()[0]
                           for name, exe in [("ffmpeg", ffmpeg), ("ffprobe", ffprobe)]}
        for index, offset in enumerate(sorted(offsets), 1):
            frame_id = f"{session_id}_{channel_id}_{index:03d}"
            image_path = f"frames/{frame_id}.png"
            actual = extract(source, output / image_path, offset, ffmpeg, timeout)
            report["frames"].append({"frame_id": frame_id, "requested_offset_s": offset,
                                     "actual_offset_s": actual, "image_file": image_path,
                                     "image_sha256": sha256(output / image_path)})
        final = source.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns) or sha256(source) != source_hash:
            raise ValueError("Source changed during intake; outputs cannot be accepted. Retry with a stable export.")
        rows = []
        for frame in report["frames"]:
            row = dict.fromkeys(LABEL_FIELDS, "")
            row.update(frame)
            row.update(session_id=session_id, vehicle_id=vehicle_id or "", channel_id=channel_id,
                       origin=origin, source_sha256=source_hash)
            rows.append(row)
        with (output / "annotations.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=LABEL_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        report["status"] = "frames_ready_for_manual_review"
        write_review(output, report, rows)
        if __package__:
            from .annotations import write_annotator
        else:
            from annotations import write_annotator
        write_annotator(output, report)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--channel-id", required=True)
    parser.add_argument("--vehicle-id", help="Required for --origin bus; use an ID or documented pseudonym.")
    parser.add_argument("--origin", choices=["bus", "synthetic", "other"], required=True)
    parser.add_argument("--times", nargs="+", type=float, default=[10.0, 30.0, 50.0])
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        result = prepare(args.input, args.output, args.session_id, args.channel_id,
                         args.origin, args.times, args.timeout, args.vehicle_id)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Intake failed: {error}\n")
    print(f"{len(result['frames'])} frames ready for manual review ({result['origin']}).")
    print(args.output.resolve() / "review.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
