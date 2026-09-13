"""Shared fixtures for the uplink suite.

The suite was written against `UPLINK_API_CONTRACT.md`, `PRODUCT_SPEC.md` 6 and
7.1 and `development/PIPELINE_STATES.md` 2, 2.1, 2.2 and 3, without reading the
implementation. That independence is the only reason the suite is worth
anything: a test written after looking at the code proves that the code does
what it does, not that it does what was specified.

The path shim exists because `development/src` is not an installed package and
there is no `pyproject.toml` in the repository. Nothing here writes files,
reads a clock or touches the network.
"""

import sys
from pathlib import Path

import pytest

# development/src/tests/conftest.py -> development/src
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# The thirteen fields of PRODUCT_SPEC 7.1, in the order the spec lists them.
# Order matters for STATUS_VALUES and for the "no additions, no renames" rule,
# so it is written out once and imported rather than re-typed per test.
MESSAGE_FIELDS = (
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

# The six wire values of PRODUCT_SPEC 7.1, in spec order.
WIRE_STATUSES = ("ok", "stale", "obstructed", "too_dark", "low_confidence", "offline")
NON_OK_STATUSES = tuple(s for s in WIRE_STATUSES if s != "ok")

MEASURED_AT = "2026-09-04T10:00:00Z"
PUBLISHED_AT = "2026-09-04T10:00:05Z"


def make_ok_message(**overrides) -> dict:
    """A message that satisfies every rule in the contract, for mutation."""
    msg = {
        "schema_version": "1.0",
        "vehicle_id": "bus-0417",
        "device_id": "sanas-dev-01",
        "route_id": "r-12",
        "trip_id": "t-99",
        "measured_at": MEASURED_AT,
        "published_at": PUBLISHED_AT,
        "status": "ok",
        "level": 3,
        "score": 0.62,
        "over_capacity": False,
        "confidence": 0.81,
        "reason": None,
    }
    msg.update(overrides)
    return msg


def make_non_ok_message(status: str = "offline", **overrides) -> dict:
    """A valid degraded message: four null value fields plus a reason."""
    degraded = {
        "status": status,
        "level": None,
        "score": None,
        "over_capacity": None,
        "confidence": None,
        "reason": "link_down",
    }
    degraded.update(overrides)
    return make_ok_message(**degraded)


@pytest.fixture
def ok_message() -> dict:
    return make_ok_message()


@pytest.fixture
def non_ok_message() -> dict:
    return make_non_ok_message()
