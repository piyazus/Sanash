"""States and transitions, steps U13-U14 and U17-U26.

`DIAGRAM_TRANSITIONS` below is a hand transcription of the `stateDiagram-v2`
block in `development/PIPELINE_STATES.md` section 2, arrow by arrow, sixteen of
them plus the `[*] --> starting` entry point which is not a transition between
states. Frame counts come from the table in 2.1 and from nowhere else.

Two of the tests in this file are expected to be contentious, and both are
marked in place: the 2.2 asymmetry claim, and `offline` reachable from any
state. They are written the way the specification words them, so that a
failure says the specification disagrees with itself rather than hiding it.
"""

import pytest
from conftest import NON_OK_STATUSES, WIRE_STATUSES

from sanas_uplink.errors import TransitionError, UplinkError
from sanas_uplink.message import build
from sanas_uplink.schema import STATUS_VALUES, validate
from sanas_uplink.states import (
    STARTING,
    STARTING_WIRE_REASON,
    STARTING_WIRE_STATUS,
    TRANSITIONS,
    StateMachine,
    StateMachineConfig,
    Status,
    can_transition,
    is_ok,
)

# Every arrow in PIPELINE_STATES 2.
DIAGRAM_TRANSITIONS = frozenset(
    {
        ("starting", "ok"),
        ("starting", "too_dark"),
        ("starting", "obstructed"),
        ("ok", "stale"),
        ("ok", "too_dark"),
        ("ok", "obstructed"),
        ("ok", "low_confidence"),
        ("ok", "offline"),
        ("too_dark", "ok"),
        ("obstructed", "ok"),
        ("low_confidence", "ok"),
        ("stale", "ok"),
        ("offline", "ok"),
        ("too_dark", "stale"),
        ("obstructed", "stale"),
        ("low_confidence", "stale"),
    }
)

ALL_STATES = ("starting", *WIRE_STATUSES)

ABSENT_TRANSITIONS = sorted(
    (src, dst)
    for src in ALL_STATES
    for dst in ALL_STATES
    if (src, dst) not in DIAGRAM_TRANSITIONS
)

# Frame counts, PIPELINE_STATES 2.1. Candidates, per the document, but the
# suite has to pin the candidate or it tests nothing.
DARK_FRAMES = 2
OBSTRUCTED_FRAMES = 3
LOW_CONFIDENCE_FRAMES = 2
RECOVERY_FRAMES = 2
SMOOTHING_WINDOW = 5
STALE_AFTER_S = 30.0

# The three quality flags that carry a degraded condition, with the frame count
# that confirms entry and the state it leads to.
DEGRADED_ENTRY = [
    ("brightness_ok", DARK_FRAMES, Status.TOO_DARK),
    ("variance_ok", OBSTRUCTED_FRAMES, Status.OBSTRUCTED),
    ("confidence_ok", LOW_CONFIDENCE_FRAMES, Status.LOW_CONFIDENCE),
]


class Driver:
    """Feeds frames to a machine on a synthetic monotonic clock.

    A degraded frame is fed as `valid=False` alongside the failing quality
    flag, because a frame that fails the quality check produced no measurement.
    The specification does not state the relationship between `valid` and the
    three flags; this is the reading, and it is flagged in the report.
    """

    def __init__(self, machine: StateMachine, dt: float = 1.0) -> None:
        self.machine = machine
        self.dt = dt
        self.t = 0.0

    def frames(self, count: int, **flags) -> str:
        state = self.machine.state
        for _ in range(count):
            self.t += self.dt
            state = self.machine.observe(now_s=self.t, **flags)
        return state

    def good(self, count: int = 1) -> str:
        return self.frames(count, valid=True)

    def bad(self, count: int, flag: str) -> str:
        return self.frames(count, valid=False, **{flag: False})

    def wait(self, seconds: float) -> str:
        """Let time pass without a valid measurement, one frame at the end."""
        self.t += seconds
        return self.machine.observe(now_s=self.t, valid=False)

    def reach_ok(self) -> str:
        state = self.good(SMOOTHING_WINDOW)
        assert state == Status.OK, f"expected ok after warm-up, got {state}"
        return state


def running_machine(config: StateMachineConfig | None = None) -> tuple:
    machine = StateMachine(config)
    driver = Driver(machine)
    driver.reach_ok()
    return machine, driver


# --------------------------------------------------------------------------
# U13, U14: the enum and is_ok.
# --------------------------------------------------------------------------


def test_status_has_the_six_wire_values_in_spec_order():
    assert tuple(member.value for member in Status) == WIRE_STATUSES


def test_status_members_compare_equal_to_their_wire_strings():
    for member in Status:
        assert member == member.value
        assert isinstance(member, str)


def test_starting_is_not_a_wire_value():
    assert STARTING == "starting"
    assert STARTING not in STATUS_VALUES
    assert STARTING not in {member.value for member in Status}


@pytest.mark.parametrize("status", WIRE_STATUSES)
def test_is_ok_over_every_wire_value(status):
    assert is_ok(status) is (status == "ok")


@pytest.mark.parametrize("status", list(Status))
def test_is_ok_accepts_enum_members(status):
    assert is_ok(status) is (status is Status.OK)


def test_is_ok_is_false_for_starting():
    assert is_ok(STARTING) is False


def test_transition_error_hierarchy():
    assert issubclass(TransitionError, UplinkError)


# --------------------------------------------------------------------------
# U17, U18, U26: the transition table.
# --------------------------------------------------------------------------


def test_transitions_are_exactly_the_arrows_in_the_diagram():
    assert {tuple(pair) for pair in TRANSITIONS} == set(DIAGRAM_TRANSITIONS)


def test_transition_count_matches_the_diagram():
    assert len(TRANSITIONS) == len(DIAGRAM_TRANSITIONS) == 16


def test_transitions_is_a_frozenset_of_pairs():
    assert isinstance(TRANSITIONS, frozenset)
    for pair in TRANSITIONS:
        assert isinstance(pair, tuple) and len(pair) == 2


@pytest.mark.parametrize(("src", "dst"), sorted(DIAGRAM_TRANSITIONS))
def test_every_arrow_in_the_diagram_is_allowed(src, dst):
    assert can_transition(src, dst) is True


@pytest.mark.parametrize(("src", "dst"), ABSENT_TRANSITIONS)
def test_no_arrow_the_diagram_lacks_is_allowed(src, dst):
    assert can_transition(src, dst) is False


def test_can_transition_accepts_enum_members():
    assert can_transition(Status.OK, Status.STALE) is True
    assert can_transition(Status.STALE, Status.OBSTRUCTED) is False


def test_every_state_in_the_table_is_a_known_state():
    for src, dst in TRANSITIONS:
        assert src in ALL_STATES
        assert dst in ALL_STATES


def test_starting_is_never_a_destination():
    """The device warms up once. Nothing in the diagram goes back to it."""
    assert not any(dst == STARTING for _, dst in TRANSITIONS)


# --------------------------------------------------------------------------
# Configuration. Every number is a candidate and the code has to say so.
# --------------------------------------------------------------------------


def test_config_defaults_match_the_contract():
    config = StateMachineConfig()
    assert config.smoothing_window == SMOOTHING_WINDOW
    assert config.dark_frames == DARK_FRAMES
    assert config.obstructed_frames == OBSTRUCTED_FRAMES
    assert config.low_confidence_frames == LOW_CONFIDENCE_FRAMES
    assert config.recovery_frames == RECOVERY_FRAMES
    assert config.stale_after_s == STALE_AFTER_S


def test_config_says_its_numbers_are_candidates():
    """PIPELINE_STATES 2.1 calls the confirmation delays candidates and
    `stale_after_s` is not stated anywhere in the repository. A reader of the
    code has to be told that, not left to assume the numbers were measured."""
    doc = (StateMachineConfig.__doc__ or "").lower()
    assert "candidate" in doc


# --------------------------------------------------------------------------
# U19, warm-up: starting needs a full smoothing window.
# --------------------------------------------------------------------------


def test_machine_starts_in_starting():
    assert StateMachine().state == STARTING


def test_one_valid_frame_is_not_enough():
    driver = Driver(StateMachine())
    assert driver.good(1) == STARTING


def test_starting_needs_a_full_smoothing_window():
    driver = Driver(StateMachine())
    for frame in range(1, SMOOTHING_WINDOW):
        assert driver.good(1) == STARTING, f"published after {frame} frames"
    assert driver.good(1) == Status.OK


def test_smoothing_window_size_is_configurable():
    driver = Driver(StateMachine(StateMachineConfig(smoothing_window=3)))
    assert driver.good(2) == STARTING
    assert driver.good(1) == Status.OK


def test_warm_up_survives_an_invalid_frame_and_still_needs_a_full_window():
    driver = Driver(StateMachine())
    driver.good(SMOOTHING_WINDOW - 1)
    driver.frames(1, valid=False)
    assert driver.good(SMOOTHING_WINDOW) == Status.OK


def test_observe_returns_the_state_after_the_frame():
    machine = StateMachine()
    driver = Driver(machine)
    for _ in range(SMOOTHING_WINDOW + 2):
        assert driver.good(1) == machine.state


# --------------------------------------------------------------------------
# U19-U24: entry into a degraded state, with the counts from 2.1.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("flag", "frames", "expected"), DEGRADED_ENTRY)
def test_degraded_entry_takes_exactly_the_frames_from_the_table(flag, frames, expected):
    machine, driver = running_machine()
    for frame in range(1, frames):
        assert driver.bad(1, flag) == Status.OK, f"switched after {frame} frames"
    assert driver.bad(1, flag) == expected


@pytest.mark.parametrize(("flag", "frames", "expected"), DEGRADED_ENTRY)
def test_degraded_entry_counter_resets_on_a_good_frame(flag, frames, expected):
    machine, driver = running_machine()
    driver.bad(frames - 1, flag)
    assert driver.good(1) == Status.OK
    for frame in range(1, frames):
        assert driver.bad(1, flag) == Status.OK, f"switched after {frame} frames"
    assert driver.bad(1, flag) == expected


@pytest.mark.parametrize(("flag", "frames", "expected"), DEGRADED_ENTRY)
def test_degraded_entry_thresholds_are_configurable(flag, frames, expected):
    config = StateMachineConfig(
        dark_frames=4, obstructed_frames=4, low_confidence_frames=4
    )
    machine, driver = running_machine(config)
    assert driver.bad(3, flag) == Status.OK
    assert driver.bad(1, flag) == expected


def test_ok_to_too_dark():
    machine, driver = running_machine()
    assert driver.bad(DARK_FRAMES, "brightness_ok") == Status.TOO_DARK
    assert machine.wire_status == Status.TOO_DARK


def test_ok_to_obstructed():
    machine, driver = running_machine()
    assert driver.bad(OBSTRUCTED_FRAMES, "variance_ok") == Status.OBSTRUCTED


def test_ok_to_low_confidence():
    machine, driver = running_machine()
    assert driver.bad(LOW_CONFIDENCE_FRAMES, "confidence_ok") == Status.LOW_CONFIDENCE


def test_low_confidence_boundary_frame_by_frame():
    """U24 asks for a test on the boundary: the frame before the threshold must
    still publish a level."""
    machine, driver = running_machine()
    assert driver.bad(LOW_CONFIDENCE_FRAMES - 1, "confidence_ok") == Status.OK
    assert machine.wire_status == Status.OK
    assert driver.bad(1, "confidence_ok") == Status.LOW_CONFIDENCE
    assert machine.wire_status == Status.LOW_CONFIDENCE


# --------------------------------------------------------------------------
# Leaving a degraded state: recovery_frames, and the counter resets too.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("flag", "frames", "expected"), DEGRADED_ENTRY)
def test_recovery_takes_exactly_recovery_frames(flag, frames, expected):
    machine, driver = running_machine()
    assert driver.bad(frames, flag) == expected
    for frame in range(1, RECOVERY_FRAMES):
        assert driver.good(1) == expected, f"recovered after {frame} frames"
    assert driver.good(1) == Status.OK


@pytest.mark.parametrize(("flag", "frames", "expected"), DEGRADED_ENTRY)
def test_recovery_counter_resets_on_a_relapse(flag, frames, expected):
    machine, driver = running_machine()
    driver.bad(frames, flag)
    assert driver.good(RECOVERY_FRAMES - 1) == expected
    assert driver.bad(1, flag) == expected
    for frame in range(1, RECOVERY_FRAMES):
        assert driver.good(1) == expected, f"recovered after {frame} frames"
    assert driver.good(1) == Status.OK


def test_entry_into_a_degraded_state_is_faster_than_leaving_it():
    """PIPELINE_STATES 2.2, stated as prose: "Вход в деградированное состояние
    подтверждается быстрее, чем выход из него."

    The counts in 2.1 are entry 2, 3 and 2 frames against an exit of 2. If this
    fails, 2.1 and 2.2 contradict each other and only Diyas can settle which
    one is meant. Do not fix this by editing the test.
    """
    config = StateMachineConfig()
    entry_counts = (
        config.dark_frames,
        config.obstructed_frames,
        config.low_confidence_frames,
    )
    assert max(entry_counts) < config.recovery_frames


# --------------------------------------------------------------------------
# U21: stale by timer.
# --------------------------------------------------------------------------


def test_ok_does_not_go_stale_before_the_ttl():
    machine, driver = running_machine()
    assert driver.wait(STALE_AFTER_S - 1.0) == Status.OK


def test_ok_goes_stale_after_the_ttl():
    machine, driver = running_machine()
    assert driver.wait(STALE_AFTER_S + 1.0) == Status.STALE


def test_stale_needs_no_confirmation_frames():
    """2.1 says "немедленно по таймеру". One frame past the TTL is enough."""
    machine, driver = running_machine()
    before = machine.state
    assert before == Status.OK
    assert driver.wait(STALE_AFTER_S + 1.0) == Status.STALE


def test_valid_frames_keep_the_ttl_from_expiring():
    machine, driver = running_machine(StateMachineConfig(stale_after_s=10.0))
    driver.dt = 5.0
    assert driver.good(6) == Status.OK


def test_stale_after_s_is_configurable():
    machine, driver = running_machine(StateMachineConfig(stale_after_s=5.0))
    assert driver.wait(6.0) == Status.STALE


@pytest.mark.parametrize(("flag", "frames", "expected"), DEGRADED_ENTRY)
def test_a_degraded_state_that_outlasts_the_ttl_becomes_stale(flag, frames, expected):
    machine, driver = running_machine()
    assert driver.bad(frames, flag) == expected
    driver.t += STALE_AFTER_S * 2
    assert machine.observe(now_s=driver.t, valid=False, **{flag: False}) == Status.STALE


# --------------------------------------------------------------------------
# PIPELINE_STATES 3: leaving stale refills the window.
# --------------------------------------------------------------------------


def test_leaving_stale_refills_the_smoothing_window():
    """ "При выходе из `stale` окно сглаживания пусто. Публиковать уровень по
    одному измерению нельзя." Recovery frames are not enough here; the window
    is."""
    machine, driver = running_machine()
    assert driver.wait(STALE_AFTER_S + 1.0) == Status.STALE
    for frame in range(1, SMOOTHING_WINDOW):
        assert driver.good(1) != Status.OK, f"published after {frame} frames"
    assert driver.good(1) == Status.OK


def test_leaving_stale_is_slower_than_leaving_any_other_degraded_state():
    assert SMOOTHING_WINDOW > RECOVERY_FRAMES


def test_one_valid_frame_does_not_leave_stale():
    machine, driver = running_machine()
    driver.wait(STALE_AFTER_S + 1.0)
    assert driver.good(1) != Status.OK


# --------------------------------------------------------------------------
# U22, U23 from the warm-up state: PIPELINE_STATES 2 draws only two arrows out
# of `starting` besides `ok`.
# --------------------------------------------------------------------------


def test_starting_can_go_too_dark():
    driver = Driver(StateMachine())
    assert driver.bad(DARK_FRAMES - 1, "brightness_ok") == STARTING
    assert driver.bad(1, "brightness_ok") == Status.TOO_DARK


def test_starting_can_go_obstructed():
    driver = Driver(StateMachine())
    assert driver.bad(OBSTRUCTED_FRAMES - 1, "variance_ok") == STARTING
    assert driver.bad(1, "variance_ok") == Status.OBSTRUCTED


def test_starting_never_goes_low_confidence():
    """There is no `starting --> low_confidence` arrow. A device that has never
    produced a measurement has no confidence to be low."""
    driver = Driver(StateMachine())
    assert driver.bad(LOW_CONFIDENCE_FRAMES * 3, "confidence_ok") != (
        Status.LOW_CONFIDENCE
    )


def test_starting_never_goes_stale_on_the_timer():
    """No `starting --> stale` arrow either."""
    driver = Driver(StateMachine())
    assert driver.wait(STALE_AFTER_S * 3) != Status.STALE


# --------------------------------------------------------------------------
# U25: offline.
# --------------------------------------------------------------------------


def test_publish_failure_from_ok_goes_offline():
    machine, driver = running_machine()
    assert machine.mark_publish_failed(now_s=driver.t) == Status.OFFLINE
    assert machine.state == Status.OFFLINE


def test_publish_success_recovers_from_offline():
    machine, driver = running_machine()
    machine.mark_publish_failed(now_s=driver.t)
    assert machine.mark_publish_ok(now_s=driver.t + 1.0) == Status.OK


def test_publish_success_in_ok_changes_nothing():
    machine, driver = running_machine()
    assert machine.mark_publish_ok(now_s=driver.t + 1.0) == Status.OK


def test_offline_is_reachable_from_every_state():
    """The contract says `mark_publish_failed` moves to OFFLINE from any state,
    and PIPELINE_STATES 2.1 has a "любое -> offline" row. The diagram in
    section 2 draws only `ok --> offline`, and the contract also says legal
    transitions are exactly the pairs in TRANSITIONS. Both cannot hold. This
    test asserts only that the machine and the table agree with each other, so
    a failure isolates the contradiction rather than papering over it.
    """
    for source in ALL_STATES:
        machine = StateMachine()
        driver = Driver(machine)
        reach(driver, source)
        assert machine.state == source, f"could not reach {source}"
        try:
            result = machine.mark_publish_failed(now_s=driver.t + 1.0)
        except TransitionError:
            assert not can_transition(source, Status.OFFLINE), (
                f"{source} -> offline is in TRANSITIONS but was refused"
            )
        else:
            assert result == Status.OFFLINE
            assert can_transition(source, Status.OFFLINE), (
                f"machine took {source} -> offline, which TRANSITIONS forbids"
            )


def reach(driver: Driver, state: str) -> None:
    """Drive a fresh machine into `state` using only specified transitions."""
    if state == STARTING:
        return
    if state == Status.OFFLINE:
        driver.reach_ok()
        driver.machine.mark_publish_failed(now_s=driver.t)
        return
    driver.reach_ok()
    if state == Status.OK:
        return
    if state == Status.STALE:
        driver.wait(STALE_AFTER_S + 1.0)
        return
    flag, frames, _ = next(entry for entry in DEGRADED_ENTRY if entry[2] == state)
    driver.bad(frames, flag)


# --------------------------------------------------------------------------
# What the machine puts on the wire.
# --------------------------------------------------------------------------


def test_starting_publishes_the_named_candidate_status():
    """PRODUCT_SPEC 7.1 has no wire value for `starting`. The contract puts the
    stand-in behind two named constants so the decision has one place to
    change. The test asserts the indirection, not the candidate."""
    machine = StateMachine()
    assert machine.state == STARTING
    assert machine.wire_status == STARTING_WIRE_STATUS
    assert machine.wire_reason == STARTING_WIRE_REASON


def test_starting_never_puts_starting_on_the_wire():
    machine = StateMachine()
    assert machine.wire_status in STATUS_VALUES
    assert machine.wire_status != STARTING


def test_ok_publishes_ok_without_a_reason():
    machine, driver = running_machine()
    assert machine.wire_status == Status.OK
    assert machine.wire_reason is None


@pytest.mark.parametrize(("flag", "frames", "expected"), DEGRADED_ENTRY)
def test_a_degraded_state_publishes_itself_with_a_reason(flag, frames, expected):
    machine, driver = running_machine()
    driver.bad(frames, flag)
    assert machine.wire_status == expected
    assert isinstance(machine.wire_reason, str)
    assert machine.wire_reason != ""


def test_stale_publishes_a_reason():
    machine, driver = running_machine()
    driver.wait(STALE_AFTER_S + 1.0)
    assert machine.wire_status == Status.STALE
    assert machine.wire_reason


def test_offline_publishes_a_reason():
    machine, driver = running_machine()
    machine.mark_publish_failed(now_s=driver.t)
    assert machine.wire_status == Status.OFFLINE
    assert machine.wire_reason


@pytest.mark.parametrize("state", ALL_STATES)
def test_wire_status_is_always_a_wire_value(state):
    machine = StateMachine()
    driver = Driver(machine)
    reach(driver, state)
    assert machine.wire_status in STATUS_VALUES


@pytest.mark.parametrize("state", [s for s in ALL_STATES if s != "ok"])
def test_a_non_ok_state_builds_a_message_that_validates(state):
    """The end of the chain: whatever the machine says can be handed to `build`
    unchanged and comes out a legal message with four null value fields."""
    machine = StateMachine()
    driver = Driver(machine)
    reach(driver, state)
    msg = build(
        vehicle_id="bus-0417",
        device_id="sanas-dev-01",
        measured_at="2026-09-04T10:00:00Z",
        status=machine.wire_status,
        reason=machine.wire_reason,
    )
    assert validate(msg) is None
    assert msg["level"] is None
    assert msg["score"] is None
    assert msg["over_capacity"] is None
    assert msg["confidence"] is None


@pytest.mark.parametrize("status", NON_OK_STATUSES)
def test_every_degraded_wire_status_is_a_known_status(status):
    assert status in STATUS_VALUES
    assert not is_ok(status)
