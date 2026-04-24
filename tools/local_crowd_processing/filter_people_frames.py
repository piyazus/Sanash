from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ULTRALYTICS_CONFIG_ROOT = Path(".ultralytics").resolve()
ULTRALYTICS_CONFIG_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_ROOT))


SOURCE_ROOT = Path("frames")
SOURCE_ROOT_RESOLVED = SOURCE_ROOT.resolve()
WITH_PEOPLE_ROOT = Path("frames_with_people")
NO_PEOPLE_ROOT = Path("frames_no_people")
MODEL_NAME = "yolov8n.pt"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
BATCH_SIZE = 250


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def has_person(result) -> bool:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return False

    for class_id, confidence in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        if int(class_id) == PERSON_CLASS_ID and float(confidence) > CONFIDENCE_THRESHOLD:
            return True
    return False


def move_image(image_path: Path, destination_root: Path) -> Path:
    relative_path = image_path.resolve().relative_to(SOURCE_ROOT_RESOLVED)
    destination_path = unique_destination(destination_root / relative_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(image_path), str(destination_path))
    return destination_path


def image_files() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def count_images(root: Path) -> int:
    if not root.exists():
        return 0

    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def run_worker(manifest_path: Path) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(manifest_path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            print(exc.stdout, end="", flush=True)
        if exc.stderr:
            print(exc.stderr, end="", flush=True)
        return 124

    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.returncode == 0:
        return 0

    if result.stderr:
        print(result.stderr, end="", flush=True)
    return result.returncode


def run_worker_with_split(paths: list[Path]) -> None:
    if not paths:
        return

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", prefix="people_batch_", dir=Path.cwd(), delete=False
    ) as manifest:
        manifest_path = Path(manifest.name)
        for path in paths:
            manifest.write(f"{path.resolve()}\n")

    try:
        exit_code = run_worker(manifest_path)
    finally:
        manifest_path.unlink(missing_ok=True)

    if exit_code == 0:
        return

    if len(paths) == 1:
        print(f"Could not process {paths[0]} after worker failure.", flush=True)
        return

    midpoint = len(paths) // 2
    run_worker_with_split(paths[:midpoint])
    run_worker_with_split(paths[midpoint:])


def process_manifest(manifest_path: Path) -> None:
    from ultralytics import YOLO

    image_paths = [
        Path(line.strip())
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    model = YOLO(MODEL_NAME)
    with_people_count = 0
    no_people_count = 0
    skipped_count = 0

    for image_path in image_paths:
        if not image_path.exists():
            skipped_count += 1
            continue

        result = model.predict(
            source=str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )[0]

        if has_person(result):
            move_image(image_path, WITH_PEOPLE_ROOT)
            with_people_count += 1
        else:
            move_image(image_path, NO_PEOPLE_ROOT)
            no_people_count += 1

    print(
        f"Batch complete: with_people={with_people_count}, "
        f"without_people={no_people_count}, skipped={skipped_count}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Move extracted frames based on YOLOv8 person detection.")
    parser.add_argument("--worker", type=Path, help="Internal manifest file for a worker batch.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if args.worker:
        process_manifest(args.worker)
        return

    if not SOURCE_ROOT.exists():
        print(f"{SOURCE_ROOT} does not exist.")
        return

    image_paths = image_files()

    if not image_paths:
        print("No JPEG images found under frames/.")
        print(f"Frames with people: {count_images(WITH_PEOPLE_ROOT)}")
        print(f"Frames without people: {count_images(NO_PEOPLE_ROOT)}")
        print(f"Total processed: {count_images(WITH_PEOPLE_ROOT) + count_images(NO_PEOPLE_ROOT)}")
        return

    WITH_PEOPLE_ROOT.mkdir(exist_ok=True)
    NO_PEOPLE_ROOT.mkdir(exist_ok=True)

    batch_size = max(1, args.batch_size)
    total_images = len(image_paths)
    for start in range(0, total_images, batch_size):
        end = min(start + batch_size, total_images)
        run_worker_with_split(image_paths[start:end])
        print(f"Processed {end}/{total_images} source images", flush=True)

    with_people_count = count_images(WITH_PEOPLE_ROOT)
    no_people_count = count_images(NO_PEOPLE_ROOT)
    remaining_count = count_images(SOURCE_ROOT)

    print(f"Frames with people: {with_people_count}")
    print(f"Frames without people: {no_people_count}")
    print(f"Frames remaining in source: {remaining_count}")
    print(f"Total processed: {with_people_count + no_people_count}")


if __name__ == "__main__":
    main()
