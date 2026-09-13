"""Message construction, steps U9-U12 plus the U15 and U16 rules applied by
`build` rather than only checked by `validate`.

The two properties that matter most here are cheap to state and easy to lose:
`build` never invents `measured_at`, because that stamp belongs to the frame,
and a `build` that returns has produced a message that validates.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest
from conftest import MESSAGE_FIELDS, NON_OK_STATUSES, WIRE_STATUSES

from sanas_uplink.errors import MessageError
from sanas_uplink.message import build
from sanas_uplink.schema import SCHEMA_VERSION, validate
from sanas_uplink.states import Status

MEASURED = "2026-09-04T10:00:00Z"
FROZEN = datetime(2026, 9, 4, 10, 0, 5, tzinfo=UTC)

IDS = {"vehicle_id": "bus-0417", "device_id": "sanas-dev-01"}
OK_VALUES = {"level": 3, "score": 0.62, "over_capacity": False, "confidence": 0.81}


def frozen(moment: datetime):
    """A `now` that never advances, so a test can assert an exact stamp."""
    return lambda: moment


def build_ok(**overrides) -> dict:
    kwargs = {
        **IDS,
        "measured_at": MEASURED,
        "status": "ok",
        **OK_VALUES,
        "now": frozen(FROZEN),
    }
    kwargs.update(overrides)
    return build(**kwargs)


def build_degraded(status: str = "too_dark", **overrides) -> dict:
    kwargs = {
        **IDS,
        "measured_at": MEASURED,
        "status": status,
        "reason": "lens_covered",
        "now": frozen(FROZEN),
    }
    kwargs.update(overrides)
    return build(**kwargs)


# --------------------------------------------------------------------------
# U9: what build returns.
# --------------------------------------------------------------------------


def test_build_returns_a_message_that_validates():
    assert validate(build_ok()) is None


def test_build_returns_exactly_the_thirteen_fields():
    assert set(build_ok()) == set(MESSAGE_FIELDS)


@pytest.mark.parametrize("field", MESSAGE_FIELDS)
def test_every_field_is_present_in_a_built_message(field):
    assert field in build_ok()


def test_build_carries_the_identifiers_through():
    msg = build_ok()
    assert msg["vehicle_id"] == IDS["vehicle_id"]
    assert msg["device_id"] == IDS["device_id"]


def test_optional_ids_default_to_null():
    msg = build_ok()
    assert msg["route_id"] is None
    assert msg["trip_id"] is None


def test_optional_ids_are_carried_through_when_given():
    msg = build_ok(route_id="r-12", trip_id="t-99")
    assert msg["route_id"] == "r-12"
    assert msg["trip_id"] == "t-99"
    assert validate(msg) is None


def test_build_carries_the_values_through():
    msg = build_ok()
    for field, value in OK_VALUES.items():
        assert msg[field] == value


# --------------------------------------------------------------------------
# U10: schema_version is hard-wired.
# --------------------------------------------------------------------------


def test_build_sets_the_schema_version():
    assert build_ok()["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("status", WIRE_STATUSES)
def test_schema_version_is_present_whatever_the_status(status):
    msg = build_ok() if status == "ok" else build_degraded(status)
    assert msg["schema_version"] == "1.0"


# --------------------------------------------------------------------------
# U11: measured_at comes from the caller, never from a clock.
# --------------------------------------------------------------------------


def test_measured_at_is_returned_verbatim():
    assert build_ok()["measured_at"] == MEASURED


def test_build_never_invents_measured_at():
    """A stamp from years ago survives untouched. If `build` read a clock for
    this field the value would be replaced with today."""
    old = "2020-01-01T00:00:00Z"
    msg = build_ok(measured_at=old)
    assert msg["measured_at"] == old


def test_measured_at_does_not_move_when_the_publish_clock_moves():
    first = build_ok(now=frozen(FROZEN))
    second = build_ok(now=frozen(FROZEN + timedelta(hours=3)))
    assert first["measured_at"] == second["measured_at"] == MEASURED
    assert first["published_at"] != second["published_at"]


# --------------------------------------------------------------------------
# U12: published_at is the moment of sending, and `now` freezes it.
# --------------------------------------------------------------------------


def test_published_at_comes_from_the_now_parameter():
    msg = build_ok(now=frozen(FROZEN))
    assert datetime.fromisoformat(msg["published_at"]) == FROZEN


def test_published_at_is_utc():
    msg = build_ok(now=frozen(FROZEN))
    parsed = datetime.fromisoformat(msg["published_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_published_at_is_frozen_across_two_builds_with_the_same_now():
    assert build_ok()["published_at"] == build_ok()["published_at"]


def test_published_at_differs_from_measured_at_on_the_system_clock():
    """U12's own check. `measured_at` is deliberately far in the past so the
    real clock cannot coincide with it and cannot violate rule 13 either."""
    msg = build(
        **IDS,
        measured_at="2020-01-01T00:00:00Z",
        status="ok",
        **OK_VALUES,
    )
    assert msg["published_at"] != msg["measured_at"]
    assert validate(msg) is None


def test_now_is_not_called_for_measured_at():
    """`now` is called at most once, for `published_at`. A second call would
    mean a second clock read the caller cannot see."""
    calls = []

    def counting_now():
        calls.append(1)
        return FROZEN

    build_ok(now=counting_now)
    assert len(calls) <= 1


def test_published_at_before_measured_at_is_refused():
    """A frozen clock that runs behind the frame stamp must not be smoothed
    over. Rule 13 is the caller's problem, not something `build` hides."""
    with pytest.raises(MessageError):
        build_ok(measured_at="2026-09-04T12:00:00Z", now=frozen(FROZEN))


# --------------------------------------------------------------------------
# Status handling.
# --------------------------------------------------------------------------


def test_build_accepts_the_status_enum_and_the_wire_string_alike():
    from_enum = build_ok(status=Status.OK)
    from_string = build_ok(status="ok")
    assert from_enum == from_string


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_status_accepts_enum_and_string(status):
    from_enum = build_degraded(Status(status))
    from_string = build_degraded(status)
    assert from_enum == from_string
    assert from_enum["status"] == status


@pytest.mark.parametrize("status", ["unknown", "starting", "OK", ""])
def test_unknown_status_is_refused(status):
    with pytest.raises(MessageError) as excinfo:
        build_degraded(status)
    assert "status" in str(excinfo.value)


def test_message_survives_a_json_round_trip():
    """The message exists to go on a wire. Anything that only compares equal
    in Python, an enum member for instance, has to survive the trip unchanged
    and still validate on the far side."""
    msg = build_ok(status=Status.OK)
    restored = json.loads(json.dumps(msg))
    assert restored == msg
    assert restored["status"] == "ok"
    assert validate(restored) is None


# --------------------------------------------------------------------------
# U15 by construction: a degraded build nulls the four value fields itself.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_build_forces_the_four_value_fields_to_null(status):
    msg = build_degraded(
        status, level=4, score=0.91, over_capacity=True, confidence=0.77
    )
    assert msg["level"] is None
    assert msg["score"] is None
    assert msg["over_capacity"] is None
    assert msg["confidence"] is None
    assert validate(msg) is None


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_build_does_not_fall_back_to_zero(status):
    msg = build_degraded(
        status, level=1, score=0.0, over_capacity=False, confidence=0.0
    )
    assert msg["level"] is None
    assert msg["score"] is None
    assert msg["over_capacity"] is None
    assert msg["confidence"] is None


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_build_needs_no_values_at_all(status):
    msg = build_degraded(status)
    assert msg["level"] is None
    assert validate(msg) is None


@pytest.mark.parametrize("field", ["level", "score", "over_capacity", "confidence"])
def test_ok_build_requires_every_value_field(field):
    values = dict(OK_VALUES)
    values.pop(field)
    with pytest.raises(MessageError) as excinfo:
        build(**IDS, measured_at=MEASURED, status="ok", now=frozen(FROZEN), **values)
    assert field in str(excinfo.value)


def test_ok_build_rejects_an_out_of_range_level():
    with pytest.raises(MessageError) as excinfo:
        build_ok(level=6)
    assert "level" in str(excinfo.value)


@pytest.mark.parametrize("value", [-0.1, 1.1])
@pytest.mark.parametrize("field", ["score", "confidence"])
def test_ok_build_rejects_a_score_outside_the_unit_interval(field, value):
    with pytest.raises(MessageError) as excinfo:
        build_ok(**{field: value})
    assert field in str(excinfo.value)


# --------------------------------------------------------------------------
# U16 by construction: the reason rule.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_build_requires_a_reason(status):
    with pytest.raises(MessageError) as excinfo:
        build_degraded(status, reason=None)
    assert "reason" in str(excinfo.value)


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_build_refuses_an_empty_reason(status):
    with pytest.raises(MessageError) as excinfo:
        build_degraded(status, reason="")
    assert "reason" in str(excinfo.value)


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_build_keeps_the_reason(status):
    assert build_degraded(status, reason="lens_covered")["reason"] == "lens_covered"


def test_ok_build_emits_no_reason():
    """The contract says `build` forces the value fields to null on a degraded
    status but does not say what it does with a reason handed to an `ok`
    message. Both spec-compliant outcomes are accepted: refuse it by name, or
    drop it. Silently publishing `status: ok` with a reason is not."""
    try:
        msg = build_ok(reason="recovered_from_too_dark")
    except MessageError as exc:
        assert "reason" in str(exc)
    else:
        assert msg["reason"] is None
        assert validate(msg) is None


def test_ok_build_without_a_reason_is_null():
    assert build_ok()["reason"] is None


# --------------------------------------------------------------------------
# Identifiers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["vehicle_id", "device_id"])
def test_build_refuses_an_empty_identifier(field):
    with pytest.raises(MessageError) as excinfo:
        build_ok(**{field: ""})
    assert field in str(excinfo.value)


def test_build_message_carries_no_image_data():
    """PRODUCT_SPEC 7.1: no images, no person coordinates, no personal data.
    The schema's closed field list is what enforces it; this pins the intent."""
    msg = build_ok()
    assert set(msg) == set(MESSAGE_FIELDS)
