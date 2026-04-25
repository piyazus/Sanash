from pathlib import Path
import shutil

from ultralytics import YOLO


SOURCE_DIR = Path("frames")
WITH_PEOPLE_DIR = Path("frames_with_people")
NO_PEOPLE_DIR = Path("frames_no_people")
CONFIDENCE_THRESHOLD = 0.3
PERSON_CLASS_ID = 0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BATCH_SIZE = 16


def find_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def result_has_person(result) -> bool:
    boxes = result.boxes
    if boxes is None:
        return False

    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
            return True

    return False


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_image(image_path: Path, destination_root: Path) -> Path:
    relative_path = image_path.relative_to(SOURCE_DIR)
    destination = unique_destination(destination_root / relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(image_path), str(destination))
    return destination


def chunks(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def main() -> None:
    if not SOURCE_DIR.exists():
        print(f"Source directory not found: {SOURCE_DIR}")
        return

    images = find_images(SOURCE_DIR)
    if not images:
        print("No images found in frames/.")
        print("With people: 0")
        print("Without people: 0")
        print("Total: 0")
        return

    model = YOLO("yolov8n.pt")
    with_people = 0
    without_people = 0

    for batch in chunks(images, BATCH_SIZE):
        results = model.predict(
            source=[str(path) for path in batch],
            batch=BATCH_SIZE,
            verbose=False,
        )

        for image_path, result in zip(batch, results):
            if result_has_person(result):
                move_image(image_path, WITH_PEOPLE_DIR)
                with_people += 1
            else:
                move_image(image_path, NO_PEOPLE_DIR)
                without_people += 1

    print(f"With people: {with_people}")
    print(f"Without people: {without_people}")
    print(f"Total: {with_people + without_people}")


if __name__ == "__main__":
    main()
