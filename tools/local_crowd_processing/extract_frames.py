from pathlib import Path
import argparse
import math
import re
import subprocess
import sys


BLUR_THRESHOLD = 100.0
OUTPUT_ROOT = Path("frames")
BATCH_SECONDS = 125


def safe_folder_name(video_path: Path, used_names: set[str]) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._")
    name = name or "video"

    candidate = name
    suffix = 2
    while candidate.lower() in used_names:
        candidate = f"{name}_{suffix}"
        suffix += 1

    used_names.add(candidate.lower())
    return candidate


def laplacian_variance(frame) -> float:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def probe_video_seconds(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not fps or fps <= 0 or frame_count <= 0:
        cap.release()
        return 0

    total_seconds = int(math.floor((int(frame_count) - 1) / float(fps))) + 1
    cap.release()
    return total_seconds


def extract_batch(video_path: Path, output_dir: Path, start_second: int, end_second: int) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0 or frame_count <= 0:
        cap.release()
        return 0

    fps = float(fps)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    current_frame = int(round(start_second * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    try:
        for second in range(start_second, end_second):
            target_frame = int(round(second * fps))
            if target_frame >= frame_count:
                break

            while current_frame < target_frame:
                if not cap.grab():
                    return saved_count
                current_frame += 1

            ok, frame = cap.read()
            current_frame += 1
            if not ok:
                continue

            if laplacian_variance(frame) < BLUR_THRESHOLD:
                continue

            frame_name = f"{video_path.stem}_{second:06d}.jpg"
            frame_path = output_dir / frame_name
            if cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                saved_count += 1

    finally:
        cap.release()

    return saved_count


def video_total_seconds(video_path: Path) -> int:
    script_path = Path(__file__).resolve()
    command = [sys.executable, str(script_path), "--probe", str(video_path)]

    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
        print(f"{video_path}: could not read video metadata: {detail}", flush=True)
        return 0

    for line in result.stdout.splitlines():
        if line.startswith("SECONDS "):
            return int(line.split()[1])

    print(f"{video_path}: missing FPS/frame-count metadata", flush=True)
    return 0


def run_worker_batch(video_path: Path, output_dir: Path, start_second: int, end_second: int) -> int:
    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script_path),
        "--worker",
        str(video_path),
        str(output_dir),
        str(start_second),
        str(end_second),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        result = None

    if result is None:
        if end_second - start_second <= 1:
            print(f"{video_path}: skipped second {start_second} after worker timeout", flush=True)
            return 0

        midpoint = start_second + (end_second - start_second) // 2
        left_count = run_worker_batch(video_path, output_dir, start_second, midpoint)
        right_count = run_worker_batch(video_path, output_dir, midpoint, end_second)
        return left_count + right_count

    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.startswith("COUNT "):
                return int(line.split()[1])
        return 0

    if end_second - start_second <= 1:
        detail = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
        print(f"{video_path}: skipped second {start_second} after worker failure: {detail}", flush=True)
        return 0

    midpoint = start_second + (end_second - start_second) // 2
    left_count = run_worker_batch(video_path, output_dir, start_second, midpoint)
    right_count = run_worker_batch(video_path, output_dir, midpoint, end_second)
    return left_count + right_count


def extract_video_frames(video_path: Path, output_dir: Path) -> int:
    total_seconds = video_total_seconds(video_path)
    saved_count = 0

    for start_second in range(0, total_seconds, BATCH_SECONDS):
        end_second = min(start_second + BATCH_SECONDS, total_seconds)
        saved_count += run_worker_batch(video_path, output_dir, start_second, end_second)
        print(f"{video_path}: processed {end_second}/{total_seconds} seconds", flush=True)

    return saved_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract sharp JPEG frames from MP4 files.")
    parser.add_argument("--probe")
    parser.add_argument("--worker", nargs=4, metavar=("VIDEO", "OUTPUT_DIR", "START", "END"))
    args = parser.parse_args()

    if args.probe:
        print(f"SECONDS {probe_video_seconds(Path(args.probe))}", flush=True)
        return

    if args.worker:
        video_arg, output_arg, start_arg, end_arg = args.worker
        count = extract_batch(Path(video_arg), Path(output_arg), int(start_arg), int(end_arg))
        print(f"COUNT {count}", flush=True)
        return

    root = Path.cwd()
    videos = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"
    )

    if not videos:
        print("No .mp4 files found.")
        return

    OUTPUT_ROOT.mkdir(exist_ok=True)
    used_names: set[str] = set()

    for video_path in videos:
        folder_name = safe_folder_name(video_path, used_names)
        output_dir = OUTPUT_ROOT / folder_name
        print(f"{video_path}: extracting 1 frame/sec", flush=True)
        saved_count = extract_video_frames(video_path, output_dir)
        print(f"{video_path}: saved {saved_count} frames", flush=True)


if __name__ == "__main__":
    main()
