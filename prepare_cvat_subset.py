from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SEARCH_SUBDIRS = ("diyas", "obama")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUMERIC_GROUP_RE = re.compile(r"(\d+)")


@dataclass
class ImageRecord:
    image_id: str
    xml_name: str
    filename: str
    width: int
    height: int
    points: list[tuple[float, float]] = field(default_factory=list)
    malformed_points: int = 0
    out_of_bounds_points: int = 0

    @property
    def head_count(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class DiskEntry:
    path: Path
    filename: str
    rel_frames: str
    rel_group: str
    source_folder: str
    stem: str
    extension: str
    last_numeric_group: str
    normalized_numeric_suffix: int | None


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected true or false for --drop-empty, got {value!r}"
    )


def parse_source_map(raw_value: str | None) -> dict[str, str]:
    if raw_value is None or not raw_value.strip():
        return {}

    source_map: dict[str, str] = {}
    for raw_mapping in raw_value.split(","):
        raw_mapping = raw_mapping.strip()
        if not raw_mapping:
            continue
        if "=" not in raw_mapping:
            raise argparse.ArgumentTypeError(
                f"Expected PREFIX=source in --source-map, got {raw_mapping!r}"
            )

        prefix, source = [part.strip() for part in raw_mapping.split("=", 1)]
        source = source.replace("\\", "/").strip("/")
        if not prefix or not source:
            raise argparse.ArgumentTypeError(
                f"Expected non-empty PREFIX=source in --source-map, got {raw_mapping!r}"
            )
        source_map[prefix] = source

    return source_map


def normalize_xml_name(name: str) -> str:
    normalized = name.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def basename_from_xml_name(name: str) -> str:
    normalized = normalize_xml_name(name)
    return normalized.rsplit("/", 1)[-1]


def last_numeric_group(value: str) -> tuple[str, int | None]:
    matches = NUMERIC_GROUP_RE.findall(value)
    if not matches:
        return "", None

    raw = matches[-1]
    return raw, int(raw)


def xml_prefix_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_", 1)[0]


def key(value: str) -> str:
    return normalize_xml_name(value).casefold()


def parse_dimension(raw_value: str | None, field_name: str, image_name: str) -> int:
    if raw_value is None:
        raise ValueError(f"Image {image_name!r} is missing {field_name}")

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Image {image_name!r} has non-integer {field_name}: {raw_value!r}"
        ) from exc

    if value <= 0:
        raise ValueError(f"Image {image_name!r} has invalid {field_name}: {value}")

    return value


def parse_point_pairs(raw_points: str | None) -> tuple[list[tuple[float, float]], int]:
    if raw_points is None or not raw_points.strip():
        return [], 1

    valid_pairs: list[tuple[float, float]] = []
    malformed = 0

    for raw_pair in raw_points.split(";"):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            malformed += 1
            continue

        parts = [part.strip() for part in raw_pair.split(",")]
        if len(parts) != 2:
            malformed += 1
            continue

        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            malformed += 1
            continue

        valid_pairs.append((x, y))

    return valid_pairs, malformed


def point_is_inside_image(x: float, y: float, width: int, height: int) -> bool:
    return 0.0 <= x <= float(width) and 0.0 <= y <= float(height)


def parse_cvat_xml(xml_path: Path, label: str) -> list[ImageRecord]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records: list[ImageRecord] = []

    for image in root.iter("image"):
        image_name = image.attrib.get("name", "").strip()
        if not image_name:
            raise ValueError("Found an <image> element with no name attribute")

        width = parse_dimension(image.attrib.get("width"), "width", image_name)
        height = parse_dimension(image.attrib.get("height"), "height", image_name)
        record = ImageRecord(
            image_id=image.attrib.get("id", ""),
            xml_name=image_name,
            filename=basename_from_xml_name(image_name),
            width=width,
            height=height,
        )

        for points in image.findall("points"):
            if points.attrib.get("label") != label:
                continue

            parsed_points, malformed = parse_point_pairs(points.attrib.get("points"))
            record.malformed_points += malformed

            for x, y in parsed_points:
                if point_is_inside_image(x, y, width, height):
                    record.points.append((x, y))
                else:
                    record.out_of_bounds_points += 1

        records.append(record)

    return records


def searched_roots(frames_root: Path) -> list[Path]:
    return [frames_root / subdir for subdir in SEARCH_SUBDIRS]


def build_disk_index(
    frames_root: Path,
) -> tuple[
    dict[str, list[DiskEntry]],
    dict[str, list[DiskEntry]],
    dict[str, list[DiskEntry]],
    dict[str, list[DiskEntry]],
    dict[int, list[DiskEntry]],
]:
    rel_index: dict[str, list[DiskEntry]] = {}
    filename_index: dict[str, list[DiskEntry]] = {}
    exact_filename_index: dict[str, list[DiskEntry]] = {}
    stem_index: dict[str, list[DiskEntry]] = {}
    numeric_index: dict[int, list[DiskEntry]] = {}

    for root in searched_roots(frames_root):
        if not root.exists():
            print(f"WARNING: search folder not found: {root}")
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            rel_frames = path.relative_to(frames_root).as_posix()
            rel_group = path.relative_to(root).as_posix()
            relative_parts = path.relative_to(frames_root).parts
            source_folder = relative_parts[0] if relative_parts else ""
            raw_suffix, normalized_suffix = last_numeric_group(path.stem)
            entry = DiskEntry(
                path=path,
                filename=path.name,
                rel_frames=rel_frames,
                rel_group=rel_group,
                source_folder=source_folder,
                stem=path.stem,
                extension=path.suffix.lower(),
                last_numeric_group=raw_suffix,
                normalized_numeric_suffix=normalized_suffix,
            )

            for rel_name in {rel_frames, rel_group}:
                rel_index.setdefault(key(rel_name), []).append(entry)

            filename_index.setdefault(path.name.casefold(), []).append(entry)
            exact_filename_index.setdefault(path.name, []).append(entry)
            stem_index.setdefault(path.stem, []).append(entry)
            if normalized_suffix is not None:
                numeric_index.setdefault(normalized_suffix, []).append(entry)

    return rel_index, filename_index, exact_filename_index, stem_index, numeric_index


def choose_disk_matches(
    records: list[ImageRecord],
    rel_index: dict[str, list[DiskEntry]],
    filename_index: dict[str, list[DiskEntry]],
    exact_filename_index: dict[str, list[DiskEntry]],
    stem_index: dict[str, list[DiskEntry]],
    numeric_index: dict[int, list[DiskEntry]],
    match_strategy: str,
    source_map: dict[str, str],
) -> tuple[
    dict[int, DiskEntry],
    dict[int, str],
    dict[str, list[DiskEntry]],
    dict[int, list[DiskEntry]],
]:
    found: dict[int, DiskEntry] = {}
    match_strategies: dict[int, str] = {}
    duplicate_matches: dict[str, list[DiskEntry]] = {}
    ambiguous_matches: dict[int, list[DiskEntry]] = {}

    def exact_candidates(record: ImageRecord) -> list[DiskEntry]:
        xml_key = key(record.xml_name)
        candidates = rel_index.get(xml_key, [])

        if not candidates:
            candidates = exact_filename_index.get(record.filename, [])

        if not candidates:
            candidates = filename_index.get(record.filename.casefold(), [])

        return candidates

    def case_insensitive_candidates(record: ImageRecord) -> list[DiskEntry]:
        return filename_index.get(record.filename.casefold(), [])

    def stem_candidates(record: ImageRecord) -> list[DiskEntry]:
        return stem_index.get(Path(record.filename).stem, [])

    def numeric_suffix_candidates(record: ImageRecord) -> list[DiskEntry]:
        _, normalized_suffix = last_numeric_group(Path(record.filename).stem)
        if normalized_suffix is None:
            return []
        return numeric_index.get(normalized_suffix, [])

    def source_numeric_candidates(record: ImageRecord) -> tuple[str, list[DiskEntry]]:
        mapped_source = source_map.get(xml_prefix_from_filename(record.filename), "")
        if not mapped_source:
            return "", []

        _, normalized_suffix = last_numeric_group(Path(record.filename).stem)
        if normalized_suffix is None:
            return mapped_source, []

        candidates = [
            entry
            for entry in numeric_index.get(normalized_suffix, [])
            if entry.source_folder.casefold() == mapped_source.casefold()
        ]
        return mapped_source, candidates

    for index, record in enumerate(records):
        candidates: list[DiskEntry] = []
        selected_strategy = ""

        if match_strategy == "auto":
            strategy_order = ["exact", "case_insensitive", "stem"]
            if source_map:
                strategy_order.append("source_numeric")
            strategy_order.append("numeric_suffix")
        else:
            strategy_order = [match_strategy]

        for strategy in strategy_order:
            stop_after_strategy = False
            if strategy == "exact":
                candidates = exact_candidates(record)
            elif strategy == "case_insensitive":
                candidates = case_insensitive_candidates(record)
            elif strategy == "stem":
                candidates = stem_candidates(record)
            elif strategy == "source_numeric":
                mapped_source, source_candidates = source_numeric_candidates(record)
                if not mapped_source:
                    candidates = []
                elif len(source_candidates) > 1:
                    ambiguous_matches[index] = sorted(
                        source_candidates, key=lambda entry: str(entry.path).casefold()
                    )
                    candidates = []
                    stop_after_strategy = True
                else:
                    candidates = source_candidates
                    stop_after_strategy = True
            elif strategy == "numeric_suffix":
                numeric_candidates = numeric_suffix_candidates(record)
                if len(numeric_candidates) > 1:
                    ambiguous_matches[index] = sorted(
                        numeric_candidates, key=lambda entry: str(entry.path).casefold()
                    )
                    candidates = []
                else:
                    candidates = numeric_candidates
            else:
                raise ValueError(f"Unknown match strategy: {strategy}")

            if candidates:
                selected_strategy = strategy
                break
            if stop_after_strategy:
                break

        if candidates:
            sorted_candidates = sorted(candidates, key=lambda entry: str(entry.path).casefold())
            found[index] = sorted_candidates[0]
            match_strategies[index] = selected_strategy
            if len(sorted_candidates) > 1:
                duplicate_matches[record.filename] = sorted_candidates

    return found, match_strategies, duplicate_matches, ambiguous_matches


def safe_output_name(record: ImageRecord, needs_disambiguation: bool) -> str:
    if not needs_disambiguation:
        return record.filename

    normalized = normalize_xml_name(record.xml_name)
    safe_name = normalized.replace("/", "__").replace(":", "_")
    if safe_name == record.filename:
        safe_name = f"image_{record.image_id or 'unknown'}__{record.filename}"
    return safe_name


def build_output_name_map(records: list[tuple[int, ImageRecord]]) -> dict[int, str]:
    filename_counts: dict[str, int] = {}
    for _, record in records:
        filename_counts[record.filename.casefold()] = (
            filename_counts.get(record.filename.casefold(), 0) + 1
        )

    output_names: dict[int, str] = {}
    used_names: set[str] = set()

    for index, record in records:
        output_name = safe_output_name(
            record,
            needs_disambiguation=filename_counts[record.filename.casefold()] > 1,
        )

        candidate = output_name
        counter = 2
        while candidate.casefold() in used_names:
            path = Path(output_name)
            candidate = f"{path.stem}_{counter}{path.suffix}"
            counter += 1

        used_names.add(candidate.casefold())
        output_names[index] = candidate

    return output_names


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def write_manifests(
    manifests_dir: Path,
    records: list[ImageRecord],
    found: dict[int, DiskEntry],
    match_strategies: dict[int, str],
    duplicate_matches: dict[str, list[DiskEntry]],
    ambiguous_matches: dict[int, list[DiskEntry]],
) -> None:
    all_names = [record.xml_name for record in records]
    nonempty_names = [record.xml_name for record in records if record.head_count > 0]
    zero_names = [record.xml_name for record in records if record.head_count == 0]
    missing_names = [
        record.xml_name for index, record in enumerate(records) if index not in found
    ]

    write_lines(manifests_dir / "cvat_original_images_all.txt", all_names)
    write_lines(manifests_dir / "cvat_original_images_nonempty.txt", nonempty_names)
    write_lines(manifests_dir / "cvat_zero_annotation_images_review.txt", zero_names)
    write_lines(manifests_dir / "missing_from_disk.txt", missing_names)

    with (manifests_dir / "cvat_image_counts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_id",
                "filename",
                "width",
                "height",
                "head_count",
                "malformed_points",
                "out_of_bounds_points",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "filename": record.xml_name,
                    "width": record.width,
                    "height": record.height,
                    "head_count": record.head_count,
                    "malformed_points": record.malformed_points,
                    "out_of_bounds_points": record.out_of_bounds_points,
                }
            )

    with (manifests_dir / "found_on_disk.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "source_path", "match_strategy", "head_count"],
        )
        writer.writeheader()
        for index, record in enumerate(records):
            entry = found.get(index)
            if entry is None:
                continue
            writer.writerow(
                {
                    "filename": record.xml_name,
                    "source_path": str(entry.path),
                    "match_strategy": match_strategies.get(index, ""),
                    "head_count": record.head_count,
                }
            )

    with (manifests_dir / "final_filename_mapping.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_id",
                "xml_name",
                "xml_basename",
                "match_strategy",
                "source_path",
                "source_rel_frames",
                "source_folder",
                "source_filename",
                "source_stem",
                "source_extension",
                "source_numeric_suffix",
                "head_count",
            ],
        )
        writer.writeheader()
        for index, record in enumerate(records):
            entry = found.get(index)
            if entry is None:
                continue
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "xml_name": record.xml_name,
                    "xml_basename": record.filename,
                    "match_strategy": match_strategies.get(index, ""),
                    "source_path": str(entry.path),
                    "source_rel_frames": entry.rel_frames,
                    "source_folder": entry.source_folder,
                    "source_filename": entry.filename,
                    "source_stem": entry.stem,
                    "source_extension": entry.extension,
                    "source_numeric_suffix": entry.normalized_numeric_suffix,
                    "head_count": record.head_count,
                }
            )

    with (manifests_dir / "ambiguous_matches.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image_id",
                "xml_name",
                "xml_basename",
                "xml_numeric_suffix",
                "candidate_count",
                "source_path",
                "source_rel_frames",
                "source_folder",
                "source_filename",
                "source_numeric_suffix",
            ],
        )
        writer.writeheader()
        for index in sorted(ambiguous_matches):
            record = records[index]
            _, normalized_suffix = last_numeric_group(Path(record.filename).stem)
            candidates = ambiguous_matches[index]
            for entry in candidates:
                writer.writerow(
                    {
                        "image_id": record.image_id,
                        "xml_name": record.xml_name,
                        "xml_basename": record.filename,
                        "xml_numeric_suffix": normalized_suffix,
                        "candidate_count": len(candidates),
                        "source_path": str(entry.path),
                        "source_rel_frames": entry.rel_frames,
                        "source_folder": entry.source_folder,
                        "source_filename": entry.filename,
                        "source_numeric_suffix": entry.normalized_numeric_suffix,
                    }
                )

    with (manifests_dir / "duplicate_filenames_on_disk.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "source_path"])
        writer.writeheader()
        for filename in sorted(duplicate_matches):
            for entry in duplicate_matches[filename]:
                writer.writerow({"filename": filename, "source_path": str(entry.path)})


def count_stats(records: list[ImageRecord]) -> dict[str, float | int]:
    counts = [record.head_count for record in records]
    if not counts:
        return {
            "min_count": 0,
            "max_count": 0,
            "mean_count": 0.0,
            "median_count": 0.0,
        }

    return {
        "min_count": min(counts),
        "max_count": max(counts),
        "mean_count": float(statistics.mean(counts)),
        "median_count": float(statistics.median(counts)),
    }


def make_summary(
    records: list[ImageRecord],
    found: dict[int, DiskEntry],
    duplicate_matches: dict[str, list[DiskEntry]],
    ambiguous_matches: dict[int, list[DiskEntry]],
    match_strategy: str,
    source_map: dict[str, str],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "match_strategy": match_strategy,
        "source_map": source_map,
        "total_images_in_xml": len(records),
        "total_nonempty_images": sum(1 for record in records if record.head_count > 0),
        "total_zero_annotation_images": sum(
            1 for record in records if record.head_count == 0
        ),
        "total_head_points": sum(record.head_count for record in records),
        "total_images_found_on_disk": len(found),
        "total_images_missing_on_disk": len(records) - len(found),
        "total_duplicate_filenames_on_disk": len(duplicate_matches),
        "total_ambiguous_numeric_matches": len(ambiguous_matches),
    }
    summary.update(count_stats(records))
    return summary


def clear_output_contents(out_dir: Path) -> None:
    for path in sorted(out_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            try:
                path.unlink()
            except PermissionError:
                pass
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def prepare_output(out_dir: Path, frames_root: Path, overwrite: bool) -> None:
    resolved_out = out_dir.resolve()
    resolved_frames = frames_root.resolve()

    if resolved_out == resolved_frames:
        raise ValueError("Refusing to use frames-root itself as the output folder")

    if resolved_frames in resolved_out.parents:
        raise ValueError("Refusing to create or overwrite output inside frames-root")

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {out_dir}. Pass --overwrite to replace it."
            )
        clear_output_contents(out_dir)

    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (out_dir / "gt").mkdir(parents=True, exist_ok=True)


def save_selected_dataset(
    out_dir: Path,
    records: list[ImageRecord],
    found: dict[int, DiskEntry],
    drop_empty: bool,
) -> tuple[int, int]:
    selected_records = [
        (index, record)
        for index, record in enumerate(records)
        if not drop_empty or record.head_count > 0
    ]
    output_names = build_output_name_map(selected_records)
    images_dir = out_dir / "images" / ("nonempty" if drop_empty else "all")
    gt_dir = out_dir / "gt"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_missing = 0

    for index, record in selected_records:
        entry = found.get(index)
        if entry is None:
            skipped_missing += 1
            continue

        output_name = output_names[index]
        shutil.copy2(entry.path, images_dir / output_name)

        points = np.asarray(record.points, dtype=np.float32).reshape(-1, 2)
        np.save(gt_dir / f"{Path(output_name).stem}.npy", points)
        copied += 1

    return copied, skipped_missing


def print_final_summary(
    summary: dict[str, object], copied: int, skipped: int
) -> None:
    print("")
    print("Final summary")
    print(f"  Match strategy: {summary['match_strategy']}")
    print(f"  Images in XML: {summary['total_images_in_xml']}")
    print(f"  Nonempty images: {summary['total_nonempty_images']}")
    print(f"  Zero-annotation images: {summary['total_zero_annotation_images']}")
    print(f"  Valid head points: {summary['total_head_points']}")
    print(f"  Found on disk: {summary['total_images_found_on_disk']}")
    print(f"  Missing on disk: {summary['total_images_missing_on_disk']}")
    print(f"  Duplicate filenames on disk: {summary['total_duplicate_filenames_on_disk']}")
    print(f"  Ambiguous numeric matches: {summary['total_ambiguous_numeric_matches']}")
    print(f"  Copied images and gt files: {copied}")
    if skipped:
        print(f"  Skipped selected images missing from disk: {skipped}")
    print(
        "  Count stats: "
        f"min={summary['min_count']}, "
        f"max={summary['max_count']}, "
        f"mean={float(summary['mean_count']):.3f}, "
        f"median={float(summary['median_count']):.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a clean image subset and P2PNet point files from CVAT XML."
    )
    parser.add_argument("--xml", required=True, type=Path, help="CVAT Images 1.1 XML")
    parser.add_argument(
        "--frames-root",
        required=True,
        type=Path,
        help="Root folder containing frames_with_people/diyas and /obama",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output dataset folder to create",
    )
    parser.add_argument("--label", default="head", help="Point annotation label to keep")
    parser.add_argument(
        "--drop-empty",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Drop images with zero valid label points. Accepts true/false or a flag.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the output folder if it already exists",
    )
    parser.add_argument(
        "--match-strategy",
        choices=[
            "exact",
            "case_insensitive",
            "stem",
            "source_numeric",
            "numeric_suffix",
            "auto",
        ],
        default="exact",
        help=(
            "How XML filenames are matched to disk images. auto tries exact, "
            "case-insensitive basename, stem, source-aware numeric when "
            "--source-map is provided, then unique numeric suffix."
        ),
    )
    parser.add_argument(
        "--source-map",
        type=parse_source_map,
        default={},
        help="Comma-separated XML prefix to source folder mappings, e.g. D02=diyas,O01=obama",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path = args.xml
    frames_root = args.frames_root
    out_dir = args.out

    print("CVAT subset preparation")
    print(f"  XML: {xml_path}")
    print(f"  Frames root: {frames_root}")
    print(f"  Output: {out_dir}")
    print(f"  Label: {args.label}")
    print(f"  Drop empty: {args.drop_empty}")
    print(f"  Overwrite: {args.overwrite}")
    print(f"  Match strategy: {args.match_strategy}")
    print(f"  Source map: {args.source_map or {}}")

    if not xml_path.exists():
        print(f"ERROR: XML file not found: {xml_path}", file=sys.stderr)
        return 1
    if not frames_root.exists():
        print(f"ERROR: frames root not found: {frames_root}", file=sys.stderr)
        return 1
    if args.match_strategy == "source_numeric" and not args.source_map:
        print(
            "ERROR: --match-strategy source_numeric requires --source-map",
            file=sys.stderr,
        )
        return 1

    try:
        print("Parsing XML annotations...")
        records = parse_cvat_xml(xml_path, args.label)
        print(
            f"Parsed {len(records)} images with "
            f"{sum(record.head_count for record in records)} valid {args.label} points."
        )

        print("Indexing source images...")
        for root in searched_roots(frames_root):
            print(f"  Search: {root}")
        (
            rel_index,
            filename_index,
            exact_filename_index,
            stem_index,
            numeric_index,
        ) = build_disk_index(frames_root)
        found, match_strategies, duplicate_matches, ambiguous_matches = choose_disk_matches(
            records,
            rel_index,
            filename_index,
            exact_filename_index,
            stem_index,
            numeric_index,
            args.match_strategy,
            args.source_map,
        )
        print(f"Found {len(found)} XML images on disk.")

        summary = make_summary(
            records,
            found,
            duplicate_matches,
            ambiguous_matches,
            args.match_strategy,
            args.source_map,
        )
        missing_count = int(summary["total_images_missing_on_disk"])
        if missing_count:
            missing_ratio = missing_count / max(1, len(records))
            warning = "WARNING"
            if missing_count > 10 or missing_ratio >= 0.10:
                warning = "WARNING: MANY XML IMAGES ARE MISSING"
            print(
                f"{warning}: {missing_count}/{len(records)} XML images "
                f"({missing_ratio:.1%}) were not found under {frames_root}."
            )

        if duplicate_matches:
            print(
                "WARNING: duplicate source filenames were found for "
                f"{len(duplicate_matches)} XML filename(s). See "
                "duplicate_filenames_on_disk.csv."
            )
        if ambiguous_matches:
            print(
                "WARNING: ambiguous numeric suffix matches were found for "
                f"{len(ambiguous_matches)} XML image(s). See ambiguous_matches.csv."
            )

        print("Preparing output folders...")
        prepare_output(out_dir, frames_root, args.overwrite)
        manifests_dir = out_dir / "manifests"

        print("Writing manifests...")
        write_manifests(
            manifests_dir, records, found, match_strategies, duplicate_matches, ambiguous_matches
        )

        print("Copying selected images and writing .npy point files...")
        copied, skipped = save_selected_dataset(out_dir, records, found, args.drop_empty)

        summary_path = out_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        print(f"Wrote summary: {summary_path}")
        print_final_summary(summary, copied, skipped)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
