"""The Avtobys uplink message schema and its validator.

The thirteen fields are copied from `PRODUCT_SPEC.md` 7.1 with no additions and
no renames. That draft is **not agreed with Innoforce** (PRODUCT_SPEC 7.2 lists
the transport, the ownership of the device to vehicle mapping and the
authorisation as open), so `SCHEMA_VERSION` being "1.0" records only that the
field exists from day one, not that anyone has signed off on the shape.

Validation is deliberately in two layers. JSON Schema covers structure: keys,
types, ranges, the closed status vocabulary. Plain Python covers the three rules
that JSON Schema expresses badly and that carry the product meaning:

- the null rule, which is the whole reason this file exists. On any non-ok
  status the four value fields are null. Not 0, not false, not the previous
  reading. PRODUCT_SPEC 6 states the reason: showing a wrong level costs more
  than showing nothing, and in the RTCI field experiment a wrong level turns
  into differential measurement error;
- the reason rule, so that "unknown" is never sent without saying why;
- `published_at` not earlier than `measured_at`, because the app uses the pair
  to show freshness and to tell a slow model from a slow network.

Structural failures are reported before cross-field ones: there is no point
arguing about the null rule in a message that has no `status` key.
"""

import re
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator

from .errors import MessageError
from .states import Status

SCHEMA_VERSION: str = "1.0"

# The six wire values in PRODUCT_SPEC 7.1 order. Derived from the enum so the
# two lists cannot drift apart.
STATUS_VALUES: tuple[str, ...] = tuple(status.value for status in Status)

# RFC 3339 restricted to UTC. `format: date-time` is not asserted by default in
# the jsonschema package and would accept any offset anyway, and PRODUCT_SPEC
# 7.1 says UTC, so the constraint is written out as a pattern. Both spellings of
# zero offset are allowed; "-00:00" is RFC 3339 for "offset unknown" and is
# accepted here rather than argued about, since nothing downstream reads it.
RFC3339_UTC_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]00:00)$"
)

# Field order as printed in PRODUCT_SPEC 7.1. Used to make "the first offending
# field" a stable, document-ordered answer instead of a dict-ordering accident.
FIELD_ORDER: tuple[str, ...] = (
    "schema_version",
    "vehicle_id",
    "device_id",
    "route_id",
    "trip_id",
    "measured_at",
    "published_at",
    "status",
    "level",
    "score",
    "over_capacity",
    "confidence",
    "reason",
)

# The four fields that the null rule governs.
VALUE_FIELDS: tuple[str, ...] = ("level", "score", "over_capacity", "confidence")

MESSAGE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://sanas.local/schemas/uplink-message-1.0.json",
    "title": "Sanas uplink message",
    "description": (
        "Draft message from a Sanas device to Avtobys, PRODUCT_SPEC.md 7.1. "
        "Not agreed with Innoforce. Carries no image, no person coordinates "
        "and no personal data."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": list(FIELD_ORDER),
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "vehicle_id": {"type": "string", "minLength": 1},
        "device_id": {"type": "string", "minLength": 1},
        "route_id": {"type": ["string", "null"]},
        "trip_id": {"type": ["string", "null"]},
        "measured_at": {"type": "string", "pattern": RFC3339_UTC_PATTERN},
        "published_at": {"type": "string", "pattern": RFC3339_UTC_PATTERN},
        "status": {"type": "string", "enum": list(STATUS_VALUES)},
        # Ranges only bind the non-null branch: JSON Schema applies `minimum`
        # to numbers, so null passes them untouched.
        "level": {"type": ["integer", "null"], "minimum": 1, "maximum": 5},
        "score": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "over_capacity": {"type": ["boolean", "null"]},
        "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": ["string", "null"]},
    },
}

_VALIDATOR = Draft202012Validator(MESSAGE_SCHEMA)

# jsonschema puts the offending key in the message text for errors that have no
# instance path, such as a missing required key or an extra property.
_QUOTED = re.compile(r"'([^']+)'")


def validate(msg: dict) -> None:
    """Return None if `msg` satisfies the schema and the cross-field rules.

    Raise `MessageError` naming the first offending field otherwise. First means
    structural before cross-field, and within each layer, PRODUCT_SPEC 7.1 field
    order.
    """
    if not isinstance(msg, dict):
        raise MessageError(f"message must be a dict, got {type(msg).__name__}")

    structural = _first_schema_error(msg)
    if structural is not None:
        raise MessageError(structural)

    _check_cross_field(msg)


def _first_schema_error(msg: dict) -> str | None:
    errors = sorted(_VALIDATOR.iter_errors(msg), key=_error_rank)
    if not errors:
        return None
    error = errors[0]
    field = _offending_field(error)
    return f"{field}: {error.message}" if field else error.message


def _error_rank(error: Any) -> tuple[int, int, str]:
    """Sort key: whole-object failures first, then PRODUCT_SPEC field order."""
    field = _offending_field(error)
    if not error.absolute_path:
        return (0, _field_index(field), error.validator or "")
    return (1, _field_index(field), error.validator or "")


def _field_index(field: str | None) -> int:
    if field in FIELD_ORDER:
        return FIELD_ORDER.index(field)
    return len(FIELD_ORDER)


def _offending_field(error: Any) -> str | None:
    if error.absolute_path:
        return str(error.absolute_path[0])
    quoted = _QUOTED.search(error.message)
    return quoted.group(1) if quoted else None


def _check_cross_field(msg: dict) -> None:
    status = msg["status"]
    status_is_ok = status == Status.OK.value

    for field in VALUE_FIELDS:
        value = msg[field]
        if status_is_ok and value is None:
            raise MessageError(f"{field} must not be null when status is 'ok'")
        if not status_is_ok and value is not None:
            raise MessageError(
                f"{field} must be null when status is {status!r}; the last "
                "known value is never reused (PRODUCT_SPEC 6)"
            )

    reason = msg["reason"]
    if status_is_ok and reason is not None:
        raise MessageError("reason must be null when status is 'ok'")
    if not status_is_ok and (reason is None or not reason.strip()):
        raise MessageError(
            f"reason must be a non-empty string when status is {status!r}"
        )

    measured_at = _parse_rfc3339(msg["measured_at"], "measured_at")
    published_at = _parse_rfc3339(msg["published_at"], "published_at")
    if published_at < measured_at:
        raise MessageError("published_at must not be earlier than measured_at")


def _parse_rfc3339(value: str, field: str) -> datetime:
    """Parse a UTC RFC 3339 string that has already passed the schema pattern."""
    text = value
    if text[-1] in "Zz":
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:  # a real calendar error, e.g. month 13
        raise MessageError(f"{field} is not a valid RFC 3339 UTC timestamp") from exc
