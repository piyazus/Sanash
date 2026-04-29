from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SEARCH_SUBDIRS = ("diyas", "obama")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUMERIC_GROUP_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class XmlImage:
    image_id: str
    xml_name: str
    basename: str
    prefix: str
    lower_basename: str
    stem: str
    extension: str
    last_numeric_group: str
    normalized_numeric_suffix: int | None


@dataclass(frozen=True)
class DiskImage:
    full_path: Path
    relative_path: str
    basename: str
    lower_basename: str
    stem: str
    extension: str
    last_numeric_group: str
    normalized_numeric_suffix: int | None
    parent_batch_folder: str
    source_folder: str


def normalize_xml_name(name: str) -> str:
    normalized = name.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def basename_from_xml_name(name: str) -> str:
    return normalize_xml_name(name).rsplit("/", 1)[-1]


def last_numeric_group(value: str) -> tuple[str, int | None]:
    matches = NUMERIC_GROUP_RE.findall(value)
    if not matches:
        return "", None

    raw = matches[-1]
    return raw, int(raw)


def xml_prefix_from_basename(basename: str) -> str:
    stem = Path(basename).stem
    return stem.split("_", 1)[0]


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


def make_xml_image(image: ET.Element) -> XmlImage:
    xml_name = image.attrib.get("name", "").strip()
    if not xml_name:
        raise ValueError("Found an <image> element with no name attribute")

    basename = basename_from_xml_name(xml_name)
    parsed = Path(basename)
    raw_suffix, normalized_suffix = last_numeric_group(parsed.stem)
    return XmlImage(
        image_id=image.attrib.get("id", ""),
        xml_name=xml_name,
        basename=basename,
        prefix=xml_prefix_from_basename(basename),
        lower_basename=basename.casefold(),
        stem=parsed.stem,
        extension=parsed.suffix.lower(),
        last_numeric_group=raw_suffix,
        normalized_numeric_suffix=normalized_suffix,
    )


def parse_xml_images(xml_path: Path) -> list[XmlImage]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return [make_xml_image(image) for image in root.iter("image")]


def searched_roots(frames_root: Path) -> list[Path]:
    return [frames_root / subdir for subdir in SEARCH_SUBDIRS]


def make_disk_image(path: Path, frames_root: Path) -> DiskImage:
    relative_path = path.relative_to(frames_root).as_posix()
    relative_parts = path.relative_to(frames_root).parts
    source_folder = relative_parts[0] if relative_parts else ""
    parent_batch_folder = path.parent.name
    raw_suffix, normalized_suffix = last_numeric_group(path.stem)

    return DiskImage(
        full_path=path,
        relative_path=relative_path,
        basename=path.name,
        lower_basename=path.name.casefold(),
        stem=path.stem,
        extension=path.suffix.lower(),
        last_numeric_group=raw_suffix,
        normalized_numeric_suffix=normalized_suffix,
        parent_batch_folder=parent_batch_folder,
        source_folder=source_folder,
    )


def inventory_disk_images(frames_root: Path) -> list[DiskImage]:
    disk_images: list[DiskImage] = []

    for root in searched_roots(frames_root):
        if not root.exists():
            print(f"WARNING: search folder not found: {root}")
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            disk_images.append(make_disk_image(path, frames_root))

    return disk_images


def index_many(values: list[DiskImage], attr: str) -> dict[object, list[DiskImage]]:
    index: dict[object, list[DiskImage]] = defaultdict(list)
    for item in values:
        index[getattr(item, attr)].append(item)
    return dict(index)


def build_source_numeric_index(
    disk_images: list[DiskImage],
) -> dict[tuple[str, int], list[DiskImage]]:
    index: dict[tuple[str, int], list[DiskImage]] = defaultdict(list)
    for image in disk_images:
        if image.normalized_numeric_suffix is None:
            continue
        index[(image.source_folder.casefold(), image.normalized_numeric_suffix)].append(
            image
        )
    return dict(index)


def ensure_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def xml_row(image: XmlImage) -> dict[str, object]:
    return {
        "image_id": image.image_id,
        "xml_name": image.xml_name,
        "basename": image.basename,
        "prefix": image.prefix,
        "stem": image.stem,
        "extension": image.extension,
        "last_numeric_group": image.last_numeric_group,
        "normalized_numeric_suffix": image.normalized_numeric_suffix,
    }


def disk_row(image: DiskImage) -> dict[str, object]:
    return {
        "full_path": str(image.full_path),
        "relative_path": image.relative_path,
        "basename": image.basename,
        "lowercase_basename": image.lower_basename,
        "stem": image.stem,
        "extension": image.extension,
        "last_numeric_group": image.last_numeric_group,
        "normalized_numeric_suffix": image.normalized_numeric_suffix,
        "parent_batch_folder": image.parent_batch_folder,
        "source_folder": image.source_folder,
    }


def choose_first(items: list[DiskImage]) -> DiskImage:
    return sorted(items, key=lambda item: item.relative_path.casefold())[0]


def make_exact_match_rows(
    xml_images: list[XmlImage], basename_index: dict[object, list[DiskImage]]
) -> tuple[list[dict[str, object]], set[str]]:
    rows: list[dict[str, object]] = []
    matched_ids: set[str] = set()

    for xml_image in xml_images:
        candidates = basename_index.get(xml_image.basename, [])
        if not candidates:
            continue
        matched_ids.add(xml_image.image_id)
        for disk_image in sorted(candidates, key=lambda item: item.relative_path.casefold()):
            rows.append(
                {
                    "image_id": xml_image.image_id,
                    "xml_name": xml_image.xml_name,
                    "xml_basename": xml_image.basename,
                    "disk_relative_path": disk_image.relative_path,
                    "disk_basename": disk_image.basename,
                }
            )

    return rows, matched_ids


def make_missing_exact_rows(xml_images: list[XmlImage], matched_ids: set[str]) -> list[dict[str, object]]:
    return [xml_row(image) for image in xml_images if image.image_id not in matched_ids]


def make_numeric_match_rows(
    xml_images: list[XmlImage],
    matched_ids: set[str],
    numeric_index: dict[object, list[DiskImage]],
) -> tuple[list[dict[str, object]], int, int]:
    rows: list[dict[str, object]] = []
    unique_count = 0
    ambiguous_count = 0

    for xml_image in xml_images:
        if xml_image.image_id in matched_ids:
            continue
        suffix = xml_image.normalized_numeric_suffix
        if suffix is None:
            continue
        candidates = numeric_index.get(suffix, [])
        if not candidates:
            continue

        if len(candidates) == 1:
            unique_count += 1
        else:
            ambiguous_count += 1

        for disk_image in sorted(candidates, key=lambda item: item.relative_path.casefold()):
            rows.append(
                {
                    "image_id": xml_image.image_id,
                    "xml_name": xml_image.xml_name,
                    "xml_basename": xml_image.basename,
                    "xml_numeric_suffix": xml_image.last_numeric_group,
                    "normalized_numeric_suffix": suffix,
                    "candidate_count": len(candidates),
                    "is_unique_disk_suffix": len(candidates) == 1,
                    "disk_relative_path": disk_image.relative_path,
                    "disk_basename": disk_image.basename,
                    "disk_source_folder": disk_image.source_folder,
                    "disk_batch_folder": disk_image.parent_batch_folder,
                }
            )

    return rows, unique_count, ambiguous_count


def source_numeric_candidates(
    xml_image: XmlImage,
    source_map: dict[str, str],
    source_numeric_index: dict[tuple[str, int], list[DiskImage]],
) -> tuple[str, list[DiskImage]]:
    mapped_source = source_map.get(xml_image.prefix, "")
    suffix = xml_image.normalized_numeric_suffix
    if not mapped_source or suffix is None:
        return mapped_source, []

    candidates = source_numeric_index.get((mapped_source.casefold(), suffix), [])
    return mapped_source, candidates


def make_source_numeric_rows(
    xml_images: list[XmlImage],
    source_map: dict[str, str],
    source_numeric_index: dict[tuple[str, int], list[DiskImage]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    match_rows: list[dict[str, object]] = []
    ambiguous_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []

    for xml_image in xml_images:
        mapped_source, candidates = source_numeric_candidates(
            xml_image, source_map, source_numeric_index
        )

        if not mapped_source:
            missing_rows.append(
                {
                    "image_id": xml_image.image_id,
                    "xml_name": xml_image.xml_name,
                    "xml_prefix": xml_image.prefix,
                    "mapped_source_folder": "",
                    "xml_numeric_suffix": xml_image.last_numeric_group,
                    "normalized_numeric_suffix": xml_image.normalized_numeric_suffix,
                    "reason": "no_source_map_for_prefix",
                }
            )
            continue

        if xml_image.normalized_numeric_suffix is None:
            missing_rows.append(
                {
                    "image_id": xml_image.image_id,
                    "xml_name": xml_image.xml_name,
                    "xml_prefix": xml_image.prefix,
                    "mapped_source_folder": mapped_source,
                    "xml_numeric_suffix": xml_image.last_numeric_group,
                    "normalized_numeric_suffix": "",
                    "reason": "no_numeric_suffix",
                }
            )
            continue

        sorted_candidates = sorted(
            candidates, key=lambda item: item.relative_path.casefold()
        )
        if len(sorted_candidates) == 1:
            disk_image = sorted_candidates[0]
            match_rows.append(
                {
                    "image_id": xml_image.image_id,
                    "xml_name": xml_image.xml_name,
                    "xml_basename": xml_image.basename,
                    "xml_prefix": xml_image.prefix,
                    "mapped_source_folder": mapped_source,
                    "xml_numeric_suffix": xml_image.last_numeric_group,
                    "normalized_numeric_suffix": xml_image.normalized_numeric_suffix,
                    "disk_relative_path": disk_image.relative_path,
                    "disk_basename": disk_image.basename,
                    "disk_source_folder": disk_image.source_folder,
                    "disk_batch_folder": disk_image.parent_batch_folder,
                }
            )
        elif len(sorted_candidates) > 1:
            for disk_image in sorted_candidates:
                ambiguous_rows.append(
                    {
                        "image_id": xml_image.image_id,
                        "xml_name": xml_image.xml_name,
                        "xml_basename": xml_image.basename,
                        "xml_prefix": xml_image.prefix,
                        "mapped_source_folder": mapped_source,
                        "xml_numeric_suffix": xml_image.last_numeric_group,
                        "normalized_numeric_suffix": xml_image.normalized_numeric_suffix,
                        "candidate_count": len(sorted_candidates),
                        "disk_relative_path": disk_image.relative_path,
                        "disk_basename": disk_image.basename,
                        "disk_source_folder": disk_image.source_folder,
                        "disk_batch_folder": disk_image.parent_batch_folder,
                    }
                )
        else:
            missing_rows.append(
                {
                    "image_id": xml_image.image_id,
                    "xml_name": xml_image.xml_name,
                    "xml_prefix": xml_image.prefix,
                    "mapped_source_folder": mapped_source,
                    "xml_numeric_suffix": xml_image.last_numeric_group,
                    "normalized_numeric_suffix": xml_image.normalized_numeric_suffix,
                    "reason": "no_candidate_in_mapped_source",
                }
            )

    return match_rows, ambiguous_rows, missing_rows


def make_similarity_rows(
    xml_images: list[XmlImage],
    matched_ids: set[str],
    disk_images: list[DiskImage],
    limit: int = 5,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for xml_image in xml_images:
        if xml_image.image_id in matched_ids:
            continue

        scored = [
            (
                difflib.SequenceMatcher(
                    None, xml_image.basename.casefold(), disk_image.basename.casefold()
                ).ratio(),
                disk_image,
            )
            for disk_image in disk_images
        ]
        scored.sort(key=lambda item: (-item[0], item[1].relative_path.casefold()))

        for rank, (score, disk_image) in enumerate(scored[:limit], start=1):
            rows.append(
                {
                    "image_id": xml_image.image_id,
                    "xml_name": xml_image.xml_name,
                    "xml_basename": xml_image.basename,
                    "rank": rank,
                    "similarity": f"{score:.6f}",
                    "disk_relative_path": disk_image.relative_path,
                    "disk_basename": disk_image.basename,
                    "disk_source_folder": disk_image.source_folder,
                    "disk_batch_folder": disk_image.parent_batch_folder,
                }
            )

    return rows


def make_duplicate_rows(
    index: dict[object, list[DiskImage]], key_name: str
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for key_value in sorted(index, key=lambda value: "" if value is None else str(value)):
        candidates = index[key_value]
        if key_value in {"", None} or len(candidates) <= 1:
            continue
        for disk_image in sorted(candidates, key=lambda item: item.relative_path.casefold()):
            rows.append(
                {
                    key_name: key_value,
                    "count": len(candidates),
                    "disk_relative_path": disk_image.relative_path,
                    "disk_basename": disk_image.basename,
                    "disk_source_folder": disk_image.source_folder,
                    "disk_batch_folder": disk_image.parent_batch_folder,
                }
            )

    return rows


def count_matches(xml_images: list[XmlImage], index: dict[object, list[DiskImage]], attr: str) -> int:
    return sum(1 for image in xml_images if index.get(getattr(image, attr), []))


def count_auto_unmatched(
    xml_images: list[XmlImage],
    exact_index: dict[object, list[DiskImage]],
    lower_index: dict[object, list[DiskImage]],
    stem_index: dict[object, list[DiskImage]],
    numeric_index: dict[object, list[DiskImage]],
    source_map: dict[str, str],
    source_numeric_index: dict[tuple[str, int], list[DiskImage]],
) -> int:
    unmatched = 0
    for image in xml_images:
        if exact_index.get(image.basename):
            continue
        if lower_index.get(image.lower_basename):
            continue
        if stem_index.get(image.stem):
            continue

        mapped_source, source_candidates = source_numeric_candidates(
            image, source_map, source_numeric_index
        )
        if mapped_source:
            if len(source_candidates) == 1:
                continue
            unmatched += 1
            continue

        suffix = image.normalized_numeric_suffix
        if suffix is not None and len(numeric_index.get(suffix, [])) == 1:
            continue
        unmatched += 1
    return unmatched


def make_likely_matches(
    xml_images: list[XmlImage],
    exact_index: dict[object, list[DiskImage]],
    lower_index: dict[object, list[DiskImage]],
    stem_index: dict[object, list[DiskImage]],
    numeric_index: dict[object, list[DiskImage]],
    source_map: dict[str, str],
    source_numeric_index: dict[tuple[str, int], list[DiskImage]],
    limit: int = 20,
) -> list[dict[str, object]]:
    likely: list[dict[str, object]] = []

    for image in xml_images:
        match_type = ""
        candidates: list[DiskImage] = []
        if exact_index.get(image.basename):
            match_type = "exact"
            candidates = exact_index[image.basename]
        elif lower_index.get(image.lower_basename):
            match_type = "case_insensitive"
            candidates = lower_index[image.lower_basename]
        elif stem_index.get(image.stem):
            match_type = "stem"
            candidates = stem_index[image.stem]
        else:
            mapped_source, source_candidates = source_numeric_candidates(
                image, source_map, source_numeric_index
            )
            if mapped_source:
                if len(source_candidates) == 1:
                    match_type = "source_numeric"
                    candidates = source_candidates
            elif image.normalized_numeric_suffix is not None:
                numeric_candidates = numeric_index.get(image.normalized_numeric_suffix, [])
                if len(numeric_candidates) == 1:
                    match_type = "numeric_suffix"
                    candidates = numeric_candidates

        if not candidates:
            continue

        disk_image = choose_first(candidates)
        likely.append(
            {
                "image_id": image.image_id,
                "xml_name": image.xml_name,
                "match_type": match_type,
                "disk_relative_path": disk_image.relative_path,
                "disk_basename": disk_image.basename,
            }
        )
        if len(likely) >= limit:
            break

    return likely


def print_list(title: str, values: list[str]) -> None:
    print(f"  {title}:")
    for value in values[:20]:
        print(f"    {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose CVAT XML filenames against local frame filenames."
    )
    parser.add_argument("--xml", required=True, type=Path, help="CVAT Images XML")
    parser.add_argument(
        "--frames-root",
        required=True,
        type=Path,
        help="Root folder containing frames_with_people/diyas and /obama",
    )
    parser.add_argument("--out", required=True, type=Path, help="Report output folder")
    parser.add_argument(
        "--source-map",
        type=parse_source_map,
        default={},
        help="Comma-separated XML prefix to source folder mappings, e.g. D02=diyas,O01=obama",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.xml.exists():
        print(f"ERROR: XML file not found: {args.xml}")
        return 1
    if not args.frames_root.exists():
        print(f"ERROR: frames root not found: {args.frames_root}")
        return 1

    print("CVAT filename mismatch diagnosis")
    print(f"  XML: {args.xml}")
    print(f"  Frames root: {args.frames_root}")
    print(f"  Output: {args.out}")
    print(f"  Source map: {args.source_map or {}}")

    xml_images = parse_xml_images(args.xml)
    disk_images = inventory_disk_images(args.frames_root)
    ensure_output_dir(args.out)

    exact_index = index_many(disk_images, "basename")
    lower_index = index_many(disk_images, "lower_basename")
    stem_index = index_many(disk_images, "stem")
    numeric_index = index_many(
        [image for image in disk_images if image.normalized_numeric_suffix is not None],
        "normalized_numeric_suffix",
    )
    source_numeric_index = build_source_numeric_index(disk_images)

    exact_rows, exact_matched_ids = make_exact_match_rows(xml_images, exact_index)
    missing_exact_rows = make_missing_exact_rows(xml_images, exact_matched_ids)
    numeric_rows, unique_numeric_count, ambiguous_numeric_count = make_numeric_match_rows(
        xml_images, exact_matched_ids, numeric_index
    )
    (
        source_numeric_match_rows,
        source_numeric_ambiguous_rows,
        source_numeric_missing_rows,
    ) = make_source_numeric_rows(xml_images, args.source_map, source_numeric_index)
    similarity_rows = make_similarity_rows(xml_images, exact_matched_ids, disk_images)
    duplicate_basename_rows = make_duplicate_rows(lower_index, "lowercase_basename")
    duplicate_numeric_rows = make_duplicate_rows(
        numeric_index, "normalized_numeric_suffix"
    )
    likely_matches = make_likely_matches(
        xml_images,
        exact_index,
        lower_index,
        stem_index,
        numeric_index,
        args.source_map,
        source_numeric_index,
    )

    case_insensitive_count = count_matches(xml_images, lower_index, "lower_basename")
    stem_count = count_matches(xml_images, stem_index, "stem")
    still_unmatched = count_auto_unmatched(
        xml_images,
        exact_index,
        lower_index,
        stem_index,
        numeric_index,
        args.source_map,
        source_numeric_index,
    )

    write_csv(
        args.out / "xml_filenames.csv",
        [
            "image_id",
            "xml_name",
            "basename",
            "prefix",
            "stem",
            "extension",
            "last_numeric_group",
            "normalized_numeric_suffix",
        ],
        [xml_row(image) for image in xml_images],
    )
    write_csv(
        args.out / "disk_image_inventory.csv",
        [
            "full_path",
            "relative_path",
            "basename",
            "lowercase_basename",
            "stem",
            "extension",
            "last_numeric_group",
            "normalized_numeric_suffix",
            "parent_batch_folder",
            "source_folder",
        ],
        [disk_row(image) for image in disk_images],
    )
    write_csv(
        args.out / "exact_matches.csv",
        ["image_id", "xml_name", "xml_basename", "disk_relative_path", "disk_basename"],
        exact_rows,
    )
    write_csv(
        args.out / "missing_exact_matches.csv",
        [
            "image_id",
            "xml_name",
            "basename",
            "prefix",
            "stem",
            "extension",
            "last_numeric_group",
            "normalized_numeric_suffix",
        ],
        missing_exact_rows,
    )
    write_csv(
        args.out / "possible_matches_by_numeric_suffix.csv",
        [
            "image_id",
            "xml_name",
            "xml_basename",
            "xml_numeric_suffix",
            "normalized_numeric_suffix",
            "candidate_count",
            "is_unique_disk_suffix",
            "disk_relative_path",
            "disk_basename",
            "disk_source_folder",
            "disk_batch_folder",
        ],
        numeric_rows,
    )
    write_csv(
        args.out / "source_numeric_matches.csv",
        [
            "image_id",
            "xml_name",
            "xml_basename",
            "xml_prefix",
            "mapped_source_folder",
            "xml_numeric_suffix",
            "normalized_numeric_suffix",
            "disk_relative_path",
            "disk_basename",
            "disk_source_folder",
            "disk_batch_folder",
        ],
        source_numeric_match_rows,
    )
    write_csv(
        args.out / "source_numeric_ambiguous.csv",
        [
            "image_id",
            "xml_name",
            "xml_basename",
            "xml_prefix",
            "mapped_source_folder",
            "xml_numeric_suffix",
            "normalized_numeric_suffix",
            "candidate_count",
            "disk_relative_path",
            "disk_basename",
            "disk_source_folder",
            "disk_batch_folder",
        ],
        source_numeric_ambiguous_rows,
    )
    write_csv(
        args.out / "source_numeric_missing.csv",
        [
            "image_id",
            "xml_name",
            "xml_prefix",
            "mapped_source_folder",
            "xml_numeric_suffix",
            "normalized_numeric_suffix",
            "reason",
        ],
        source_numeric_missing_rows,
    )
    write_csv(
        args.out / "possible_matches_by_stem_similarity.csv",
        [
            "image_id",
            "xml_name",
            "xml_basename",
            "rank",
            "similarity",
            "disk_relative_path",
            "disk_basename",
            "disk_source_folder",
            "disk_batch_folder",
        ],
        similarity_rows,
    )
    write_csv(
        args.out / "duplicate_disk_basenames.csv",
        [
            "lowercase_basename",
            "count",
            "disk_relative_path",
            "disk_basename",
            "disk_source_folder",
            "disk_batch_folder",
        ],
        duplicate_basename_rows,
    )
    write_csv(
        args.out / "duplicate_numeric_suffixes.csv",
        [
            "normalized_numeric_suffix",
            "count",
            "disk_relative_path",
            "disk_basename",
            "disk_source_folder",
            "disk_batch_folder",
        ],
        duplicate_numeric_rows,
    )

    summary = {
        "source_map": args.source_map,
        "total_xml_images": len(xml_images),
        "total_disk_images_found": len(disk_images),
        "exact_basename_matches": len(exact_matched_ids),
        "case_insensitive_basename_matches": case_insensitive_count,
        "stem_matches": stem_count,
        "source_numeric_matches": len(source_numeric_match_rows),
        "source_numeric_ambiguous": len(
            {row["image_id"] for row in source_numeric_ambiguous_rows}
        ),
        "source_numeric_missing": len(source_numeric_missing_rows),
        "unique_numeric_suffix_matches": unique_numeric_count,
        "ambiguous_numeric_suffix_matches": ambiguous_numeric_count,
        "still_unmatched_xml_images_after_auto": still_unmatched,
        "duplicate_disk_basenames": len(
            {row["lowercase_basename"] for row in duplicate_basename_rows}
        ),
        "duplicate_numeric_suffixes": len(
            {row["normalized_numeric_suffix"] for row in duplicate_numeric_rows}
        ),
        "first_20_xml_filenames": [image.xml_name for image in xml_images[:20]],
        "first_20_disk_filenames": [image.relative_path for image in disk_images[:20]],
        "first_20_likely_matches": likely_matches,
    }
    (args.out / "diagnosis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("")
    print("Diagnosis summary")
    print(f"  Total XML images: {summary['total_xml_images']}")
    print(f"  Total disk images found: {summary['total_disk_images_found']}")
    print(f"  Exact basename matches: {summary['exact_basename_matches']}")
    print(
        "  Case-insensitive matches: "
        f"{summary['case_insensitive_basename_matches']}"
    )
    print(f"  Stem matches: {summary['stem_matches']}")
    print(f"  Source-aware numeric matches: {summary['source_numeric_matches']}")
    print(f"  Source-aware ambiguous: {summary['source_numeric_ambiguous']}")
    print(f"  Source-aware missing: {summary['source_numeric_missing']}")
    print(
        "  Unique numeric suffix matches: "
        f"{summary['unique_numeric_suffix_matches']}"
    )
    print(
        "  Ambiguous numeric suffix matches: "
        f"{summary['ambiguous_numeric_suffix_matches']}"
    )
    print(
        "  Still unmatched XML images: "
        f"{summary['still_unmatched_xml_images_after_auto']}"
    )
    print_list("First 20 XML filenames", summary["first_20_xml_filenames"])
    print_list("First 20 disk filenames", summary["first_20_disk_filenames"])

    print("  First 20 likely matches:")
    for row in likely_matches:
        print(
            "    "
            f"{row['xml_name']} -> {row['disk_relative_path']} "
            f"({row['match_type']})"
        )

    print(f"Wrote reports under: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
