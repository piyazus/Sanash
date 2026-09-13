"""Uplink layer: the message to Avtobys and the device state machine.

Blocks 1 and 2 of `SEPTEMBER_PLAN.md` section 4, steps U1-U26. Scope is
deliberately narrow: what is sent and when the device is allowed to send it.
Transport is block 3 and is not here, because PRODUCT_SPEC 7.2 leaves the
choice between MQTT, HTTP push and polling to Innoforce, and an unanswered
question is not a design.

Nothing in this package touches a camera, a model, a network or a file. It runs
on a laptop against synthetic frames, which is what makes it testable while
`GROUND_TRUTH.md` section 6 still records the ceiling RGB system as not
implemented and no own in-cabin footage as existing.

Status of the content, in the vocabulary of `GROUND_TRUTH.md` section 1: the
message shape is a candidate draft, not agreed with Innoforce; the states and
arrows follow the draft in `development/PIPELINE_STATES.md`; every threshold is
a candidate and `StateMachineConfig.stale_after_s` is a placeholder.
"""

from .errors import MessageError, TransitionError, UplinkError
from .message import build
from .schema import (
    FIELD_ORDER,
    MESSAGE_SCHEMA,
    SCHEMA_VERSION,
    STATUS_VALUES,
    VALUE_FIELDS,
    validate,
)
from .states import (
    STARTING,
    STARTING_WIRE_REASON,
    STARTING_WIRE_STATUS,
    TRANSITION_RULES,
    TRANSITIONS,
    WIRE_REASONS,
    StateMachine,
    StateMachineConfig,
    Status,
    TransitionRule,
    can_transition,
    is_ok,
)

__all__ = [
    "FIELD_ORDER",
    "MESSAGE_SCHEMA",
    "MessageError",
    "SCHEMA_VERSION",
    "STARTING",
    "STARTING_WIRE_REASON",
    "STARTING_WIRE_STATUS",
    "STATUS_VALUES",
    "StateMachine",
    "StateMachineConfig",
    "Status",
    "TRANSITIONS",
    "TRANSITION_RULES",
    "TransitionError",
    "TransitionRule",
    "UplinkError",
    "VALUE_FIELDS",
    "WIRE_REASONS",
    "build",
    "can_transition",
    "is_ok",
    "validate",
]
