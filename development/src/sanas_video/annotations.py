"""Build an offline annotation page and compare two raters on identical frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def project_from_metadata(metadata: dict) -> dict:
    if metadata.get("status") != "frames_ready_for_manual_review":
        raise ValueError("Intake must have completed successfully.")
    project = {key: metadata[key] for key in
               ["source_sha256", "origin", "session_id", "vehicle_id", "channel_id", "frames"]}
    identity = [{key: frame[key] for key in ["frame_id", "image_sha256", "actual_offset_s"]}
                for frame in project["frames"]]
    raw = json.dumps([project["source_sha256"], identity], sort_keys=True).encode()
    project["project_key"] = hashlib.sha256(raw).hexdigest()
    return project


def write_annotator(output: Path, metadata: dict) -> None:
    project = project_from_metadata(metadata)
    payload = json.dumps(project, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("annotator.html").read_text(encoding="utf-8")
    (output / "annotate.html").write_text(template.replace("__SANASH_PROJECT_JSON__", payload), encoding="utf-8")


def validate_annotation(data: dict, project: dict) -> dict:
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("type") != "sanash_point_annotations":
        raise ValueError("Unsupported annotation format.")
    for key in ["project_key", "source_sha256", "origin", "session_id", "vehicle_id", "channel_id"]:
        if data.get(key) != project[key]:
            raise ValueError(f"Annotation mismatch: {key}.")
    rater = data.get("rater_id")
    if not isinstance(rater, str) or not rater.strip():
        raise ValueError("Missing rater ID.")
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != len(project["frames"]):
        raise ValueError("Annotation frame set differs from the intake.")
    seen = set()
    expected = {frame["frame_id"]: frame for frame in project["frames"]}
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Invalid frame record.")
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or frame_id not in expected or frame_id in seen:
            raise ValueError("Unknown or duplicated frame ID.")
        seen.add(frame_id)
        for key in ["image_sha256", "actual_offset_s"]:
            if frame.get(key) != expected[frame_id][key]:
                raise ValueError(f"Frame mismatch: {frame_id}/{key}.")
        if type(frame.get("reviewed")) is not bool or not isinstance(frame.get("notes"), str):
            raise ValueError("Each frame needs a reviewed flag and notes string.")
        points = frame.get("points")
        if not isinstance(points, list):
            raise ValueError("Points must be an array.")
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("Invalid point.")
            for coordinate in ["x", "y"]:
                value = point.get(coordinate)
                if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Point coordinates must be finite numbers in [0, 1].")
            if point.get("zone") not in {"front", "middle", "rear", "door", "unclear"}:
                raise ValueError("Invalid cabin zone.")
            if point.get("posture") not in {"seated", "standing", "unclear"}:
                raise ValueError("Invalid posture.")
            if point.get("visibility") not in {"full", "partial", "inferred"}:
                raise ValueError("Invalid visibility.")
    return {frame["frame_id"]: frame for frame in frames}


def compare(metadata: dict, first: dict, second: dict) -> dict:
    project = project_from_metadata(metadata)
    a, b = validate_annotation(first, project), validate_annotation(second, project)
    if first["rater_id"].strip() == second["rater_id"].strip():
        raise ValueError("Use two distinct rater IDs; duplicate exports are not independent ratings.")
    rows, excluded = [], []
    for frame in project["frames"]:
        frame_id = frame["frame_id"]
        left, right = a[frame_id], b[frame_id]
        if not left["reviewed"] or not right["reviewed"]:
            excluded.append(frame_id)
            continue
        counts = [len(left["points"]), len(right["points"])]
        rows.append({"frame_id": frame_id, "rater_1_count": counts[0], "rater_2_count": counts[1],
                     "absolute_count_difference": abs(counts[0] - counts[1]),
                     "inferred_counts": [sum(p["visibility"] == "inferred" for p in f["points"]) for f in [left, right]],
                     "zone_counts": {zone: [sum(p["zone"] == zone for p in f["points"]) for f in [left, right]]
                                     for zone in ["front", "middle", "rear", "door", "unclear"]},
                     "notes": [left["notes"], right["notes"]]})
    differences = [row["absolute_count_difference"] for row in rows]
    return {"schema_version": 1, "kind": "inter_rater_count_comparison", "origin": project["origin"],
            "project_key": project["project_key"], "raters": [first["rater_id"], second["rater_id"]],
            "paired_reviewed_frames": len(rows), "excluded_unreviewed_frames": excluded,
            "mean_absolute_count_difference": sum(differences) / len(rows) if rows else None,
            "exact_count_agreement_fraction": sum(d == 0 for d in differences) / len(rows) if rows else None,
            "frames": rows,
            "limitations": ["Count agreement is not model accuracy or point-location agreement.",
                            "Different rater IDs do not prove independent annotation.",
                            "Counts include full/partial/inferred marks; invisible people are not measured.",
                            "Nearby frames are correlated; no confidence interval or population claim is computed."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Add annotate.html to a completed intake directory.")
    build.add_argument("run", type=Path)
    check = sub.add_parser("compare", help="Compare two exported annotation JSON files.")
    check.add_argument("run", type=Path)
    check.add_argument("first", type=Path)
    check.add_argument("second", type=Path)
    check.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        metadata = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
        if args.command == "build":
            write_annotator(args.run, metadata)
            print(args.run.resolve() / "annotate.html")
        else:
            result = compare(metadata, json.loads(args.first.read_text(encoding="utf-8")),
                             json.loads(args.second.read_text(encoding="utf-8")))
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"Compared {result['paired_reviewed_frames']} reviewed frame pairs. {args.output.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Annotation error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
