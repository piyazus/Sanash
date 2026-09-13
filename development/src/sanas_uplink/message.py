"""Building one uplink message.

`build` is the only place in this package that is allowed to read a clock, and
it reads it for exactly one field. `measured_at` belongs to the frame and is
supplied by the caller: taking it from the clock at send time would quietly
turn a queued or delayed measurement into a fresh one, which is the failure
PRODUCT_SPEC 7.1 separates the two timestamps to make visible.

The null rule is applied here by construction rather than trusted to the
caller. A caller that passes a level together with `status="obstructed"` has a
bug, and the honest response is to drop the level, not to publish it: the
device may only send a valid measurement or an explicit reason for its absence.
`build` then calls `validate`, so a `build` that returns is a message that
validates.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from .errors import MessageError
from .schema import SCHEMA_VERSION, validate
from .states import Status


def build(
    *,
    vehicle_id: str,
    device_id: str,
    measured_at: str,
    status: Status | str,
    route_id: str | None = None,
    trip_id: str | None = None,
    level: int | None = None,
    score: float | None = None,
    over_capacity: bool | None = None,
    confidence: float | None = None,
    reason: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict:
    """Assemble a validated message dict.

    `measured_at` is an RFC 3339 UTC string from the frame. `published_at` is
    stamped here from `now()` when given and from the system clock otherwise;
    the injection point exists so that a test can freeze time, not so that
    production has two clocks.

    On a non-ok status the four value fields are forced to None and `reason`
    must be a non-empty string. On `ok` all four must be supplied, which
    `validate` enforces.
    """
    wire_status = _wire_status(status)
    published_at = _rfc3339(now() if now is not None else datetime.now(UTC))

    if wire_status is not Status.OK:
        # Forced, not trusted. See the module docstring.
        level = None
        score = None
        over_capacity = None
        confidence = None
        if reason is None or not str(reason).strip():
            raise MessageError(
                f"reason must be a non-empty string when status is "
                f"{wire_status.value!r}"
            )

    msg = {
        "schema_version": SCHEMA_VERSION,
        "vehicle_id": vehicle_id,
        "device_id": device_id,
        "route_id": route_id,
        "trip_id": trip_id,
        "measured_at": measured_at,
        "published_at": published_at,
        "status": wire_status.value,
        "level": level,
        "score": score,
        "over_capacity": over_capacity,
        "confidence": confidence,
        "reason": reason,
    }
    validate(msg)
    return msg


def _wire_status(status: Status | str) -> Status:
    try:
        return Status(status)
    except ValueError as exc:
        raise MessageError(
            f"status is not one of the six wire values: {status!r}"
        ) from exc


def _rfc3339(moment: datetime) -> str:
    """Format a datetime as RFC 3339 UTC with a trailing Z.

    A naive datetime is read as UTC rather than rejected, because the only
    naive value that can reach here comes from a test that froze the clock
    without a tzinfo. Anything aware is converted.
    """
    if not isinstance(moment, datetime):
        raise MessageError(f"now() must return a datetime, got {type(moment).__name__}")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")
