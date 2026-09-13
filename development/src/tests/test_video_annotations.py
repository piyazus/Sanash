"""Annotation integrity and count comparisons; entirely synthetic fixtures."""

import copy
import json

import pytest

from sanas_video.annotations import compare, project_from_metadata, validate_annotation, write_annotator


@pytest.fixture
def metadata():
    return {"status": "frames_ready_for_manual_review", "source_sha256": "a" * 64,
            "origin": "synthetic", "session_id": "TEST", "vehicle_id": None, "channel_id": "CH01",
            "frames": [{"frame_id": f"F{i}", "image_sha256": str(i) * 64,
                        "actual_offset_s": float(i), "image_file": f"frames/F{i}.png"} for i in range(3)]}


def annotation(metadata, rater="R1"):
    data = project_from_metadata(metadata)
    data.update(schema_version=1, type="sanash_point_annotations", rater_id=rater)
    data["frames"] = [{**f, "reviewed": False, "points": [], "notes": ""} for f in data["frames"]]
    return copy.deepcopy(data)


def point(**kwargs):
    return {"x": 0.4, "y": 0.5, "zone": "middle", "posture": "seated", "visibility": "partial", **kwargs}


def test_unreviewed_frames_are_not_zero_counts(metadata):
    result = compare(metadata, annotation(metadata), annotation(metadata, "R2"))
    assert result["paired_reviewed_frames"] == 0
    assert result["mean_absolute_count_difference"] is None
    assert result["exact_count_agreement_fraction"] is None
    assert result["excluded_unreviewed_frames"] == ["F0", "F1", "F2"]


def test_reviewed_zero_and_disagreement_are_distinct(metadata):
    a, b = annotation(metadata), annotation(metadata, "R2")
    for data in [a, b]:
        for frame in data["frames"][:2]:
            frame["reviewed"] = True
    a["frames"][1]["points"] = [point(), point(visibility="inferred", zone="rear")]
    b["frames"][1]["points"] = [point()]
    result = compare(metadata, a, b)
    assert result["paired_reviewed_frames"] == 2
    assert result["mean_absolute_count_difference"] == 0.5
    assert result["exact_count_agreement_fraction"] == 0.5
    assert result["frames"][1]["inferred_counts"] == [1, 0]
    assert result["frames"][1]["zone_counts"]["rear"] == [1, 0]
    assert result["excluded_unreviewed_frames"] == ["F2"]


def test_one_unreviewed_rater_excludes_frame(metadata):
    a, b = annotation(metadata), annotation(metadata, "R2")
    a["frames"][0]["reviewed"] = True
    assert compare(metadata, a, b)["paired_reviewed_frames"] == 0


def test_duplicate_rater_is_rejected(metadata):
    with pytest.raises(ValueError, match="distinct rater"):
        compare(metadata, annotation(metadata), annotation(metadata))


@pytest.mark.parametrize("field", ["project_key", "source_sha256", "origin", "session_id", "vehicle_id", "channel_id"])
def test_mismatched_identity_is_rejected(metadata, field):
    data = annotation(metadata)
    data[field] = "other"
    with pytest.raises(ValueError, match="mismatch"):
        validate_annotation(data, project_from_metadata(metadata))


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_coordinates_are_rejected(metadata, value):
    data = annotation(metadata)
    data["frames"][0]["points"] = [point(x=value)]
    with pytest.raises(ValueError, match="coordinates"):
        validate_annotation(data, project_from_metadata(metadata))


@pytest.mark.parametrize("change", ["missing", "duplicate", "hash", "timestamp", "reviewed", "zone"])
def test_frame_integrity(metadata, change):
    data = annotation(metadata)
    if change == "missing": data["frames"].pop()
    if change == "duplicate": data["frames"][1] = copy.deepcopy(data["frames"][0])
    if change == "hash": data["frames"][0]["image_sha256"] = "bad"
    if change == "timestamp": data["frames"][0]["actual_offset_s"] = 999
    if change == "reviewed": data["frames"][0]["reviewed"] = "true"
    if change == "zone": data["frames"][0]["points"] = [point(zone="unknown-zone")]
    with pytest.raises(ValueError): validate_annotation(data, project_from_metadata(metadata))


def test_file_order_does_not_change_pairing(metadata):
    a, b = annotation(metadata), annotation(metadata, "R2")
    a["frames"][0]["reviewed"] = b["frames"][0]["reviewed"] = True
    b["frames"].reverse()
    result = compare(metadata, a, b)
    assert result["frames"][0]["frame_id"] == "F0"


def test_generated_page_escapes_embedded_json(metadata, tmp_path):
    metadata["session_id"] = '</script><script>alert("x")</script>'
    write_annotator(tmp_path, metadata)
    text = (tmp_path / "annotate.html").read_text(encoding="utf-8")
    assert metadata["session_id"] not in text
    payload = text.split('<script id="project" type="application/json">')[1].split('</script>')[0]
    assert json.loads(payload)["session_id"] == metadata["session_id"]
    assert "__SANASH_PROJECT_JSON__" not in text


def test_failed_intake_cannot_be_annotated(metadata):
    metadata["status"] = "failed"
    with pytest.raises(ValueError, match="completed successfully"):
        project_from_metadata(metadata)
