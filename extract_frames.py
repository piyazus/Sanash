from pathlib import Path
import re

import cv2


OUTPUT_DIR = Path("frames")
BLUR_THRESHOLD = 100.0
JPEG_QUALITY = 95


def safe_folder_name(video_path: Path, used_names: set[str]) -> str:
    """Create a filesystem-friendly folder name based on the video stem."""
    base_name = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._")
    if not base_name:
        base_name = "video"

    folder_name = base_name
    counter = 2
    while folder_name in used_names:
        folder_name = f"{base_name}_{counter}"
        counter += 1

    used_names.add(folder_name)
    return folder_name


def is_sharp(frame) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= BLUR_THRESHOLD


def save_frames(video_path: Path, output_folder: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"{video_path}: could not open video")
        return 0

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or frame_count <= 0:
        print(f"{video_path}: missing FPS/frame metadata")
        capture.release()
        return 0

    duration_seconds = int(frame_count / fps)
    saved_count = 0
    output_folder.mkdir(parents=True, exist_ok=True)

    for second in range(duration_seconds + 1):
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = capture.read()
        if not success:
            continue

        if not is_sharp(frame):
            continue

        output_path = output_folder / f"frame_{second:06d}s.jpg"
        cv2.imwrite(
            str(output_path),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        saved_count += 1

    capture.release()
    return saved_count


def find_videos(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.mp4") if path.is_file())


def main() -> None:
    root = Path.cwd()
    videos = find_videos(root)

    if not videos:
        print("No .mp4 files found.")
        return

    used_names: set[str] = set()
    for video_path in videos:
        folder_name = safe_folder_name(video_path, used_names)
        saved_count = save_frames(video_path, OUTPUT_DIR / folder_name)
        print(f"{video_path}: saved {saved_count} frames")


if __name__ == "__main__":
    main()
