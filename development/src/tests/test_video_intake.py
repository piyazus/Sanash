"""Real FFmpeg integration checks using generated non-passenger video."""

import csv
import json
import math
import shutil
import subprocess

import pytest

from sanas_video import intake


pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                reason="ffmpeg/ffprobe are required")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    root = tmp_path_factory.mktemp("video_intake")
    path = root / "synthetic sample ' with spaces.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                    "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=12:duration=4",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True, timeout=30)
    return path


def prepare(clip, output, times=None):
    return intake.prepare(clip, output, "SYNTHETIC01", "CH01", "synthetic", times or [0.11, 0.6, 2.05])


def test_real_decode_records_actual_pts_and_blank_counts(clip, tmp_path):
    output = tmp_path / "run"
    report = prepare(clip, output)
    assert report["status"] == "frames_ready_for_manual_review"
    assert report["origin"] == "synthetic"
    assert report["source_sha256"] == intake.sha256(clip)
    assert report["probe"]["video_stream"]["width"] == 320
    for frame in report["frames"]:
        expected = math.ceil(frame["requested_offset_s"] * 12) / 12
        assert frame["actual_offset_s"] == pytest.approx(expected, abs=1e-5)
        image = output / frame["image_file"]
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert intake.sha256(image) == frame["image_sha256"]
    with (output / "annotations.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    for row in rows:
        assert row["observer_count"] == row["visible_count_r1"] == row["visible_count_r2"] == ""
    assert "СИНТЕТИЧЕСКИЙ ТЕСТ" in (output / "review.html").read_text(encoding="utf-8")
    assert (output / "annotate.html").is_file()


def test_container_start_offset_is_normalized(clip, tmp_path):
    shifted = tmp_path / "shifted.mp4"
    subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-n", "-i", str(clip),
                    "-c", "copy", "-output_ts_offset", "7", str(shifted)], check=True, timeout=30)
    report = prepare(shifted, tmp_path / "run", [0.11])
    assert float(report["probe"]["format"]["start_time"]) == pytest.approx(7)
    assert report["frames"][0]["actual_offset_s"] == pytest.approx(2 / 12, abs=1e-5)


def test_existing_results_are_never_overwritten(clip, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing work")
    with pytest.raises(ValueError, match="already exists"):
        prepare(clip, output)
    assert sentinel.read_text() == "existing work"


@pytest.mark.parametrize("offsets", [[4], [-1], [float("nan")], [float("inf")], [0.1, 0.1]])
def test_invalid_offsets_do_not_create_results(clip, tmp_path, offsets):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        prepare(clip, output, offsets)
    assert not output.exists()


def test_corrupt_input_is_rejected_before_output(tmp_path):
    clip = tmp_path / "broken.mp4"
    clip.write_text("This is not a video")
    with pytest.raises(ValueError, match="ffprobe.*failed"):
        prepare(clip, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_mid_run_failure_is_not_reported_as_ready(clip, tmp_path, monkeypatch):
    def broken_extract(*_args):
        raise ValueError("simulated decoder failure")
    monkeypatch.setattr(intake, "extract", broken_extract)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="decoder failure"):
        prepare(clip, output)
    report = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert not (output / "annotations.csv").exists()
    assert not (output / "review.html").exists()


def test_changed_source_invalidates_the_run(clip, tmp_path, monkeypatch):
    original_hash = intake.sha256
    calls = 0
    def changed_hash(path):
        nonlocal calls
        if path == clip.resolve():
            calls += 1
            if calls > 1:
                return "different hash"
        return original_hash(path)
    monkeypatch.setattr(intake, "sha256", changed_hash)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="Source changed"):
        prepare(clip, output, [0.11])
    assert json.loads((output / "metadata.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert not (output / "annotations.csv").exists()


def test_playlist_cannot_open_network_input(tmp_path):
    playlist = tmp_path / "remote.m3u8"
    playlist.write_text("#EXTM3U\nhttps://example.invalid/video.ts")
    with pytest.raises(ValueError, match="not a URL or playlist"):
        prepare(playlist, tmp_path / "run")


def test_channel_cannot_escape_output_directory(clip, tmp_path):
    with pytest.raises(ValueError, match="Session/channel ID"):
        intake.prepare(clip, tmp_path / "run", "SESSION", "../escape", "synthetic", [0])


def test_bus_origin_requires_explicit_vehicle_id(clip, tmp_path):
    with pytest.raises(ValueError, match="vehicle-id"):
        intake.prepare(clip, tmp_path / "run", "SESSION", "CH01", "bus", [0])
