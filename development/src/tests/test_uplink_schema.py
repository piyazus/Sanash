"""Schema and cross-field validation, steps U2-U8 and U15-U16.

Every expectation here comes from `PRODUCT_SPEC.md` 7.1 (the thirteen fields
and the null rule), `PRODUCT_SPEC.md` 6 (the six degraded states) and the rule
list in `UPLINK_API_CONTRACT.md`. Nothing comes from the implementation.

Error messages are checked only for the name of the offending field. The
contract fixes that much and nothing more, so asserting a sentence would be
asserting a decision nobody made.
"""

import pytest
from conftest import (
    MEASURED_AT,
    MESSAGE_FIELDS,
    NON_OK_STATUSES,
    WIRE_STATUSES,
    make_non_ok_message,
    make_ok_message,
)
from jsonschema import Draft202012Validator

from sanas_uplink.errors import MessageError, UplinkError
from sanas_uplink.schema import (
    MESSAGE_SCHEMA,
    SCHEMA_VERSION,
    STATUS_VALUES,
    validate,
)


def rejects(msg: dict, field: str) -> None:
    """Assert that `msg` fails validation and that the report names `field`."""
    with pytest.raises(MessageError) as excinfo:
        validate(msg)
    assert field in str(excinfo.value), (
        f"MessageError does not name the offending field {field!r}: {excinfo.value!r}"
    )


# --------------------------------------------------------------------------
# U1: the package imports at all.
# --------------------------------------------------------------------------


def test_package_imports():
    import sanas_uplink  # noqa: F401


def test_error_hierarchy():
    assert issubclass(MessageError, UplinkError)
    assert issubclass(UplinkError, Exception)


# --------------------------------------------------------------------------
# U2-U5: the schema document itself.
# --------------------------------------------------------------------------


def test_schema_is_a_valid_2020_12_schema():
    assert isinstance(MESSAGE_SCHEMA, dict)
    Draft202012Validator.check_schema(MESSAGE_SCHEMA)


def test_schema_declares_draft_2020_12():
    assert "2020-12" in str(MESSAGE_SCHEMA.get("$schema", ""))


def test_schema_version_constant():
    assert SCHEMA_VERSION == "1.0"


def test_schema_has_exactly_the_thirteen_spec_fields():
    assert tuple(MESSAGE_SCHEMA["properties"]) == MESSAGE_FIELDS


def test_schema_requires_all_thirteen_fields():
    assert set(MESSAGE_SCHEMA["required"]) == set(MESSAGE_FIELDS)


def test_schema_forbids_additional_properties():
    assert MESSAGE_SCHEMA["additionalProperties"] is False


def test_status_values_are_the_six_wire_values_in_spec_order():
    assert STATUS_VALUES == WIRE_STATUSES


# --------------------------------------------------------------------------
# The happy paths.
# --------------------------------------------------------------------------


def test_valid_ok_message_passes(ok_message):
    assert validate(ok_message) is None


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_valid_degraded_message_passes(status):
    assert validate(make_non_ok_message(status)) is None


def test_route_and_trip_may_be_null(ok_message):
    ok_message["route_id"] = None
    ok_message["trip_id"] = None
    assert validate(ok_message) is None


def test_measured_at_may_equal_published_at(ok_message):
    """ "Not earlier than" admits equality: a zero-latency publish is legal."""
    ok_message["published_at"] = ok_message["measured_at"]
    assert validate(ok_message) is None


# --------------------------------------------------------------------------
# U4: presence and absence of each of the thirteen fields.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("field", MESSAGE_FIELDS)
def test_every_field_is_present_in_a_valid_message(field, ok_message):
    assert field in ok_message
    assert validate(ok_message) is None


@pytest.mark.parametrize("field", MESSAGE_FIELDS)
def test_missing_field_is_rejected(field, ok_message):
    del ok_message[field]
    rejects(ok_message, field)


@pytest.mark.parametrize("field", MESSAGE_FIELDS)
def test_missing_field_is_rejected_on_a_degraded_message(field):
    msg = make_non_ok_message()
    del msg[field]
    rejects(msg, field)


def test_extra_field_is_rejected(ok_message):
    ok_message["passenger_boxes"] = [[10, 20, 30, 40]]
    rejects(ok_message, "passenger_boxes")


# --------------------------------------------------------------------------
# U3, U6: types and ranges, field by field.
# --------------------------------------------------------------------------

WRONG_TYPES_ON_OK = [
    ("schema_version", 1.0),
    ("schema_version", None),
    ("schema_version", ["1.0"]),
    ("vehicle_id", 42),
    ("vehicle_id", None),
    ("vehicle_id", {"id": "bus"}),
    ("device_id", 42),
    ("device_id", None),
    ("route_id", 42),
    ("route_id", []),
    ("trip_id", 42),
    ("trip_id", False),
    ("measured_at", 42),
    ("measured_at", None),
    ("published_at", 42),
    ("published_at", None),
    ("status", 42),
    ("status", None),
    ("status", ["ok"]),
    ("level", "3"),
    ("level", 1.5),
    ("level", True),
    ("score", "0.5"),
    ("score", True),
    ("over_capacity", "true"),
    ("over_capacity", 1),
    ("confidence", "0.5"),
    ("confidence", True),
]


@pytest.mark.parametrize(("field", "value"), WRONG_TYPES_ON_OK)
def test_wrong_type_is_rejected(field, value, ok_message):
    ok_message[field] = value
    rejects(ok_message, field)


@pytest.mark.parametrize("value", [42, 0.5, ["because"], {"why": "dark"}])
def test_reason_wrong_type_is_rejected(value):
    """`reason` has to be exercised on a degraded message: on `ok` it is null."""
    rejects(make_non_ok_message(reason=value), "reason")


@pytest.mark.parametrize("field", ["vehicle_id", "device_id"])
def test_identifiers_must_be_non_empty(field, ok_message):
    ok_message[field] = ""
    rejects(ok_message, field)


def test_schema_version_must_equal_one_point_zero(ok_message):
    ok_message["schema_version"] = "2.0"
    rejects(ok_message, "schema_version")


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_level_accepts_the_five_product_levels(level, ok_message):
    ok_message["level"] = level
    assert validate(ok_message) is None


@pytest.mark.parametrize("level", [0, 6, -1, 100])
def test_level_out_of_range_is_rejected(level, ok_message):
    ok_message["level"] = level
    rejects(ok_message, "level")


@pytest.mark.parametrize("field", ["score", "confidence"])
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 0, 1])
def test_unit_interval_endpoints_are_accepted(field, value, ok_message):
    ok_message[field] = value
    assert validate(ok_message) is None


@pytest.mark.parametrize("field", ["score", "confidence"])
@pytest.mark.parametrize("value", [-0.1, 1.1, -1.0, 2.0])
def test_unit_interval_is_closed(field, value, ok_message):
    ok_message[field] = value
    rejects(ok_message, field)


# --------------------------------------------------------------------------
# U5: the six wire values and the seventh.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", WIRE_STATUSES)
def test_each_wire_status_is_accepted(status):
    msg = make_ok_message() if status == "ok" else make_non_ok_message(status)
    assert validate(msg) is None


@pytest.mark.parametrize(
    "status", ["unknown", "OK", "ok ", "starting", "degraded", "too dark", ""]
)
def test_a_seventh_status_value_is_rejected(status):
    """`starting` is on this list on purpose: PIPELINE_STATES 2 has seven
    states, PRODUCT_SPEC 7.1 has six wire values, and the internal one must
    never reach the wire."""
    rejects(make_non_ok_message(status=status), "status")


# --------------------------------------------------------------------------
# Timestamps.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["yesterday", "2026-09-04", "10:00:00", "2026-13-04T10:00:00Z", "1757000000"],
)
def test_measured_at_must_be_a_timestamp(value, ok_message):
    ok_message["measured_at"] = value
    ok_message["published_at"] = value
    rejects(ok_message, "measured_at")


def test_timestamp_without_a_zone_is_rejected(ok_message):
    """Rule 5 says UTC. A naive stamp names no instant at all."""
    ok_message["measured_at"] = "2026-09-04T10:00:00"
    rejects(ok_message, "measured_at")


def test_timestamp_with_a_non_utc_offset_is_rejected(ok_message):
    """Ambiguous in the spec, flagged in the report. Rule 5 says the stamps are
    "RFC 3339 strings in UTC"; +05:00 is RFC 3339 and names the same instant,
    but it is not UTC as written, and Almaty is exactly the +05:00 that would
    slip through."""
    ok_message["measured_at"] = "2026-09-04T15:00:00+05:00"
    ok_message["published_at"] = "2026-09-04T15:00:05+05:00"
    rejects(ok_message, "measured_at")


def test_published_at_earlier_than_measured_at_is_rejected(ok_message):
    ok_message["measured_at"] = "2026-09-04T10:00:05Z"
    ok_message["published_at"] = "2026-09-04T10:00:00Z"
    with pytest.raises(MessageError) as excinfo:
        validate(ok_message)
    text = str(excinfo.value)
    assert "published_at" in text or "measured_at" in text


def test_published_at_far_earlier_than_measured_at_is_rejected(ok_message):
    ok_message["measured_at"] = "2026-09-04T10:00:00Z"
    ok_message["published_at"] = "2020-01-01T00:00:00Z"
    with pytest.raises(MessageError) as excinfo:
        validate(ok_message)
    text = str(excinfo.value)
    assert "published_at" in text or "measured_at" in text


# --------------------------------------------------------------------------
# U15, the null rule. PRODUCT_SPEC 7.1: "level и score равны null при любом
# не-ok статусе. Не 0, не последнее значение." This is the rule the product
# rests on, so it is tested from every side.
# --------------------------------------------------------------------------

VALUE_FIELDS = ("level", "score", "over_capacity", "confidence")

# For each value field: a zero-ish value, a falsey value and a plausible stale
# carry-over from the last good frame. All three must be refused equally.
NON_NULL_LEAKS = [
    ("level", 1),
    ("level", 3),
    ("level", 5),
    ("score", 0.0),
    ("score", 0),
    ("score", 0.62),
    ("over_capacity", False),
    ("over_capacity", True),
    ("confidence", 0.0),
    ("confidence", 0),
    ("confidence", 0.81),
]


@pytest.mark.parametrize("status", NON_OK_STATUSES)
@pytest.mark.parametrize(("field", "value"), NON_NULL_LEAKS)
def test_degraded_message_may_not_carry_a_value(status, field, value):
    rejects(make_non_ok_message(status, **{field: value}), field)


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_message_may_not_carry_the_whole_previous_reading(status):
    """The exact failure mode PRODUCT_SPEC 6 forbids: the last known value is
    republished under a degraded status instead of being dropped."""
    stale = make_ok_message()
    stale["status"] = status
    stale["reason"] = "camera_covered"
    with pytest.raises(MessageError):
        validate(stale)


@pytest.mark.parametrize("field", VALUE_FIELDS)
def test_ok_message_may_not_have_a_null_value_field(field, ok_message):
    ok_message[field] = None
    rejects(ok_message, field)


def test_ok_message_with_all_four_values_null_is_rejected(ok_message):
    for field in VALUE_FIELDS:
        ok_message[field] = None
    with pytest.raises(MessageError):
        validate(ok_message)


# --------------------------------------------------------------------------
# U16, the reason rule, in both directions.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_message_needs_a_reason(status):
    rejects(make_non_ok_message(status, reason=None), "reason")


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_degraded_message_reason_must_be_non_empty(status):
    rejects(make_non_ok_message(status, reason=""), "reason")


def test_ok_message_must_not_carry_a_reason(ok_message):
    ok_message["reason"] = "recovered_from_too_dark"
    rejects(ok_message, "reason")


def test_ok_message_must_not_carry_an_empty_reason(ok_message):
    ok_message["reason"] = ""
    rejects(ok_message, "reason")


# --------------------------------------------------------------------------
# Ordering: a structural failure is reported before a cross-field one.
# --------------------------------------------------------------------------


def test_structural_failure_is_reported_before_a_cross_field_failure(ok_message):
    ok_message["level"] = 9  # structural: out of the 1..5 range
    ok_message["published_at"] = "2020-01-01T00:00:00Z"  # cross-field: rule 13
    with pytest.raises(MessageError) as excinfo:
        validate(ok_message)
    assert "level" in str(excinfo.value)


def test_structural_failure_beats_the_null_rule():
    msg = make_non_ok_message(status="not_a_status", level=3)
    with pytest.raises(MessageError) as excinfo:
        validate(msg)
    assert "status" in str(excinfo.value)


# --------------------------------------------------------------------------
# validate does not mutate what it is given.
# --------------------------------------------------------------------------


def test_validate_does_not_mutate_the_message(ok_message):
    before = dict(ok_message)
    validate(ok_message)
    assert ok_message == before


def test_validate_rejects_a_non_dict():
    for value in ("{}", None, [], 42):
        with pytest.raises((MessageError, TypeError, AttributeError)):
            validate(value)


def test_measured_at_constant_is_what_the_fixtures_use(ok_message):
    assert ok_message["measured_at"] == MEASURED_AT
