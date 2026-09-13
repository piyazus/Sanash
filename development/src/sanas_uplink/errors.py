"""Exceptions for the uplink layer.

Two failure modes are worth separating. A message that does not satisfy the
contract must never reach Avtobys, and a state change that the device is not
allowed to make must never be silently swallowed, because both would show up
downstream as a plausible looking number rather than as an error. Everything
raised here descends from `UplinkError` so that a caller can catch the whole
layer without catching unrelated `ValueError`s from the model code.
"""


class UplinkError(Exception):
    """Base class for every failure inside `sanas_uplink`."""


class MessageError(UplinkError):
    """A message failed schema validation or one of the cross-field rules.

    `args[0]` names the offending field. Callers may look for the field name in
    the text; the wording itself is not part of the contract.
    """


class TransitionError(UplinkError):
    """A state change that is not an arrow in PIPELINE_STATES.md section 2."""
