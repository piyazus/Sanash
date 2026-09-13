"""Device states, the transition table and the state machine that walks it.

Source of truth for the arrows: `development/PIPELINE_STATES.md` section 2.
`TRANSITIONS` below is that mermaid diagram enumerated arrow by arrow, nothing
added and nothing dropped. `TRANSITION_RULES` is the seven-row table of section
2.1, which carries the conditions and the confirmation delays that the diagram
does not show.

Two honesty notes that must not be smoothed over by a reader in a hurry:

1. Every threshold in `StateMachineConfig` is a **candidate**. Section 2.1 says
   the confirmation delays are to be tuned against recordings, and there are no
   recordings: `GROUND_TRUTH.md` section 7 records that no own in-cabin footage
   exists. `stale_after_s` is worse than a candidate, it is a placeholder that
   appears nowhere in the repository.
2. `starting` has no wire value. `PRODUCT_SPEC.md` 7.1 defines six statuses and
   the diagram defines seven states. What the device publishes while it warms
   up is an open product decision, isolated below in two constants.

The machine runs on frames and timestamps only. It knows nothing about the
camera, the model or the transport, which is the point: it can be written and
tested before any hardware exists, as `PIPELINE_STATES.md` section 5 argues.
"""

from dataclasses import dataclass
from enum import StrEnum

from .errors import TransitionError


class Status(StrEnum):
    """The six wire values of `PRODUCT_SPEC.md` 7.1, in the order listed there."""

    OK = "ok"
    STALE = "stale"
    OBSTRUCTED = "obstructed"
    TOO_DARK = "too_dark"
    LOW_CONFIDENCE = "low_confidence"
    OFFLINE = "offline"


# The seventh state of PIPELINE_STATES 2. Internal only: it is never sent.
STARTING: str = "starting"

# Candidate, not decided. PRODUCT_SPEC 7.1 has no wire value for `starting`.
# `stale` is used because PIPELINE_STATES 2 requires that the device publish
# either a valid measurement or an explicit reason for its absence, and during
# warm-up there is no valid measurement. Change this in one place if Diyas
# decides otherwise.
STARTING_WIRE_STATUS = Status.STALE
STARTING_WIRE_REASON = "warming_up"

# Candidate wording. PRODUCT_SPEC 6 requires that a non-ok status carry a
# reason, but it never fixes the vocabulary, and Innoforce has not seen these
# strings (business/INNOFORCE_QUESTIONS.md section 3 is unanswered). They are
# machine-readable tokens on purpose: the passenger-facing text belongs to the
# app, not to the device.
WIRE_REASONS: dict[Status, str] = {
    Status.STALE: "no_recent_valid_measurement",
    Status.OBSTRUCTED: "lens_obstructed",
    Status.TOO_DARK: "frame_too_dark",
    Status.LOW_CONFIDENCE: "model_confidence_below_threshold",
    Status.OFFLINE: "publish_failed",
}


def is_ok(status: Status | str) -> bool:
    """True only for the `ok` wire status. `starting` is not ok."""
    return _name(status) == Status.OK.value


# Every arrow of the state diagram in PIPELINE_STATES 2, in diagram order. The
# initial pseudostate `[*] --> starting` is not here: it has no source state, so
# it cannot be a (from, to) pair. It is expressed instead by StateMachine
# starting in STARTING.
TRANSITIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # starting --> ...
        (STARTING, Status.OK.value),  # first valid measurement
        (STARTING, Status.TOO_DARK.value),  # frame usable but dark
        (STARTING, Status.OBSTRUCTED.value),  # lens covered
        # ok --> ...
        (Status.OK.value, Status.STALE.value),  # no valid measurement past TTL
        (Status.OK.value, Status.TOO_DARK.value),  # brightness below threshold
        (Status.OK.value, Status.OBSTRUCTED.value),  # frame uniform
        (Status.OK.value, Status.LOW_CONFIDENCE.value),  # confidence below threshold
        (Status.OK.value, Status.OFFLINE.value),  # link lost
        # ... --> ok
        (Status.TOO_DARK.value, Status.OK.value),  # light came back
        (Status.OBSTRUCTED.value, Status.OK.value),  # lens uncovered
        (Status.LOW_CONFIDENCE.value, Status.OK.value),  # confidence above threshold
        (Status.STALE.value, Status.OK.value),  # a valid measurement arrived
        (Status.OFFLINE.value, Status.OK.value),  # link restored
        # ... --> stale
        (Status.TOO_DARK.value, Status.STALE.value),  # lasted longer than TTL
        (Status.OBSTRUCTED.value, Status.STALE.value),  # lasted longer than TTL
        (Status.LOW_CONFIDENCE.value, Status.STALE.value),  # lasted longer than TTL
    }
)


@dataclass(frozen=True)
class TransitionRule:
    """One row of the transition table in PIPELINE_STATES 2.1.

    `src` and `dst` may be the wildcards `any` and `not_ok`, exactly as the
    document writes them. `confirm` is the confirmation delay in frames where
    the document gives a frame count, and None where it gives a timer or a
    network policy instead. Every count is a candidate.
    """

    src: str
    dst: str
    condition: str
    confirm: int | None
    publishes: str


# The seven rows of PIPELINE_STATES 2.1, in document order. This is the table,
# not the diagram: it has wildcards, so it is not the set of legal pairs. The
# `confirm` values are the ones mirrored in StateMachineConfig.
TRANSITION_RULES: tuple[TransitionRule, ...] = (
    TransitionRule(
        STARTING,
        Status.OK.value,
        "N valid measurements received",
        None,  # N is the smoothing window, not a frame count of its own
        "level, score, confidence",
    ),
    TransitionRule(
        Status.OK.value,
        Status.TOO_DARK.value,
        "mean brightness below threshold",
        2,
        Status.TOO_DARK.value,
    ),
    TransitionRule(
        Status.OK.value,
        Status.OBSTRUCTED.value,
        "frame variance below threshold",
        3,
        Status.OBSTRUCTED.value,
    ),
    TransitionRule(
        Status.OK.value,
        Status.LOW_CONFIDENCE.value,
        "confidence below threshold",
        2,
        Status.LOW_CONFIDENCE.value,
    ),
    TransitionRule(
        Status.OK.value,
        Status.STALE.value,
        "no valid measurement for longer than TTL",
        None,  # immediate, on the timer
        Status.STALE.value,
    ),
    TransitionRule(
        "any",
        Status.OFFLINE.value,
        "publication does not go through",
        None,  # by network policy, which is unset
        "nothing",
    ),
    TransitionRule(
        "not_ok",
        Status.OK.value,
        "condition lifted",
        2,
        "level, score, confidence",
    ),
)


def _name(status: Status | str) -> str:
    """The wire string of a Status, a plain state name, or whatever was given.

    Unknown strings come back unchanged so that `can_transition` answers False
    instead of raising: asking about a nonsense state is a question, not an
    illegal move.
    """
    if isinstance(status, Status):
        return status.value
    return str(status)


def can_transition(src: Status | str, dst: Status | str) -> bool:
    """True when (src, dst) is an arrow of the diagram in PIPELINE_STATES 2."""
    return (_name(src), _name(dst)) in TRANSITIONS


@dataclass
class StateMachineConfig:
    """Confirmation delays and timers. **Every value here is a candidate.**

    PIPELINE_STATES 2.1 says so directly: the confirmation delays are to be
    tuned against recordings. There are no recordings, no camera and no trained
    model, so not one of these numbers has been measured. They are shaped to
    satisfy the asymmetry of section 2.2, entry into a degraded state faster
    than exit from it, and nothing more.

    Provenance of each field:

    - `smoothing_window`: 5, the median window of PRODUCT_SPEC 5, itself a
      candidate there.
    - `dark_frames`, `obstructed_frames`, `low_confidence_frames`: 2, 3, 2, the
      confirmation delays of the PIPELINE_STATES 2.1 table.
    - `recovery_frames`: 2, the two confirming frames of the non-ok to ok row.
    - `stale_after_s`: **placeholder, no source in this repository.**
      PRODUCT_SPEC 5 gives a value lifetime of 5 min, but that is the age at
      which a published value becomes useless to a passenger, not the
      device-side TTL of the diagram, and the two are not the same decision.
      30.0 is a stand-in chosen only so that the timer is testable.
    """

    smoothing_window: int = 5  # N valid measurements before starting -> ok
    dark_frames: int = 2  # ok -> too_dark
    obstructed_frames: int = 3  # ok -> obstructed
    low_confidence_frames: int = 2  # ok -> low_confidence
    recovery_frames: int = 2  # any non-ok -> ok
    stale_after_s: float = 30.0  # ok -> stale, by timer


class StateMachine:
    """The seven-state machine of PIPELINE_STATES 2, driven by frames and time.

    It holds no measurement. Deciding the level and the score is the job of the
    smoothing stage; this object only decides whether anything may be published
    at all, and if not, why not. That separation is what makes the rule "either
    a valid measurement or an explicit reason for its absence" checkable with no
    model and no camera.

    Time arrives as `now_s`, a monotonic clock in seconds, because the machine
    has to be replayable from a recording. It never reads a clock itself.
    """

    def __init__(self, config: StateMachineConfig | None = None) -> None:
        self._cfg = config if config is not None else StateMachineConfig()
        self._state: str = STARTING
        self._entered_s: float | None = None
        self._last_valid_s: float | None = None
        # Valid measurements sitting in the smoothing window. Capped at the
        # window size, and emptied on entry into `stale` per PIPELINE_STATES 3.
        self._valid_run = 0
        self._dark_run = 0
        self._obstructed_run = 0
        self._low_confidence_run = 0
        self._recovery_run = 0

    @property
    def state(self) -> str:
        """The current state: a `Status` value or `STARTING`."""
        return self._state

    @property
    def wire_status(self) -> Status:
        """The status that goes into the message for the current state."""
        if self._state == STARTING:
            return STARTING_WIRE_STATUS
        return Status(self._state)

    @property
    def wire_reason(self) -> str | None:
        """The reason field: None when ok, a machine-readable token otherwise."""
        if self._state == STARTING:
            return STARTING_WIRE_REASON
        if self._state == Status.OK.value:
            return None
        return WIRE_REASONS[Status(self._state)]

    @property
    def config(self) -> StateMachineConfig:
        return self._cfg

    def observe(
        self,
        *,
        now_s: float,
        valid: bool,
        brightness_ok: bool = True,
        variance_ok: bool = True,
        confidence_ok: bool = True,
    ) -> str:
        """Feed one frame and return the state after it.

        `valid` means the pipeline produced a measurement for this frame at all;
        the three flags are the quality-check and inference verdicts of
        PIPELINE_STATES 1. A frame fills the smoothing window only when it is
        valid and all three flags hold.
        """
        if self._entered_s is None:
            self._entered_s = now_s

        clean = bool(valid and brightness_ok and variance_ok and confidence_ok)
        if clean:
            self._last_valid_s = now_s
            self._valid_run = min(self._valid_run + 1, self._cfg.smoothing_window)

        # Confirmation counters. They reset the moment the condition is not met,
        # so "2 frames in a row" really does mean consecutive frames.
        self._dark_run = 0 if brightness_ok else self._dark_run + 1
        self._obstructed_run = 0 if variance_ok else self._obstructed_run + 1
        self._low_confidence_run = 0 if confidence_ok else self._low_confidence_run + 1
        self._recovery_run = self._recovery_run + 1 if clean else 0

        if self._state == Status.OFFLINE.value:
            # Publication is a separate pipeline stage. Frames keep arriving
            # while the link is down, but the only arrow out of `offline` is
            # "link restored", so no frame can leave it.
            return self._state
        if self._state == STARTING:
            return self._observe_starting(now_s)
        if self._state == Status.OK.value:
            return self._observe_ok(now_s)
        if self._state == Status.STALE.value:
            return self._observe_stale(now_s)
        return self._observe_degraded(now_s)

    def mark_publish_failed(self, *, now_s: float) -> str:
        """Publication failed: go `offline` from whatever state we are in.

        The diagram draws only `ok --> offline`, while the table in 2.1 has the
        row "any -> offline", and losing the link while degraded is exactly as
        real as losing it while ok. This is therefore the one move that does not
        go through `TRANSITIONS`. The mismatch between diagram and table is a
        documentation defect to raise with Diyas, not something to hide by
        pretending the device is still `too_dark` with no link.
        """
        if self._state != Status.OFFLINE.value:
            self._enter(Status.OFFLINE.value, now_s, checked=False)
        return self._state

    def mark_publish_ok(self, *, now_s: float) -> str:
        """Publication succeeded. Recovers from `offline`, no-op elsewhere.

        `offline --> ok` is the only arrow leaving `offline`, so recovery goes
        straight there rather than back to the state the device held before, and
        it does not wait for the smoothing window. Open question for Diyas: a
        device that went offline during warm-up returns as `ok` with an unfilled
        window, which the window rule of PIPELINE_STATES 3 would forbid. The
        diagram offers no other exit.
        """
        if self._state == Status.OFFLINE.value:
            self._enter(Status.OK.value, now_s)
        return self._state

    def _observe_starting(self, now_s: float) -> str:
        cfg = self._cfg
        if self._dark_run >= cfg.dark_frames:
            return self._enter(Status.TOO_DARK.value, now_s)
        if self._obstructed_run >= cfg.obstructed_frames:
            return self._enter(Status.OBSTRUCTED.value, now_s)
        # The window has to be full first: PIPELINE_STATES 3, a level may not be
        # published off a single measurement. There is no starting to
        # low_confidence arrow, so an unconfident frame simply fails to fill it.
        if self._valid_run >= cfg.smoothing_window:
            return self._enter(Status.OK.value, now_s)
        return self._state

    def _observe_ok(self, now_s: float) -> str:
        cfg = self._cfg
        # Order: quality check before inference, as in the pipeline of section 1
        # and in the row order of the 2.1 table. A frame that is both dark and
        # blank is reported as dark, because the quality stage sees it first.
        if self._dark_run >= cfg.dark_frames:
            return self._enter(Status.TOO_DARK.value, now_s)
        if self._obstructed_run >= cfg.obstructed_frames:
            return self._enter(Status.OBSTRUCTED.value, now_s)
        if self._low_confidence_run >= cfg.low_confidence_frames:
            return self._enter(Status.LOW_CONFIDENCE.value, now_s)
        if self._stale_by_last_valid(now_s):
            return self._enter(Status.STALE.value, now_s)
        return self._state

    def _observe_degraded(self, now_s: float) -> str:
        cfg = self._cfg
        # Exit needs `recovery_frames` fully clean frames, not merely the one
        # condition lifting: there is no arrow from one degraded state to
        # another, so a frame that trades darkness for a blocked lens must not
        # count as recovery. Entry costs 2 or 3 frames, exit costs 2 clean ones
        # on top of the condition being gone, which is the asymmetry of 2.2.
        if self._recovery_run >= cfg.recovery_frames:
            return self._enter(Status.OK.value, now_s)
        # "lasted longer than TTL" is measured from entry into this state.
        if self._entered_s is not None and now_s - self._entered_s > cfg.stale_after_s:
            return self._enter(Status.STALE.value, now_s)
        return self._state

    def _observe_stale(self, now_s: float) -> str:
        # PIPELINE_STATES 3: on leaving `stale` the smoothing window is empty.
        # Leaving stale therefore costs a full window, not `recovery_frames`.
        if self._valid_run >= self._cfg.smoothing_window:
            return self._enter(Status.OK.value, now_s)
        return self._state

    def _stale_by_last_valid(self, now_s: float) -> bool:
        if self._last_valid_s is None:
            return False
        return now_s - self._last_valid_s > self._cfg.stale_after_s

    def _enter(self, dst: str, now_s: float, *, checked: bool = True) -> str:
        if checked and not can_transition(self._state, dst):
            raise TransitionError(
                f"{self._state} -> {dst} is not an arrow in PIPELINE_STATES 2"
            )
        self._state = dst
        self._entered_s = now_s
        self._dark_run = 0
        self._obstructed_run = 0
        self._low_confidence_run = 0
        self._recovery_run = 0
        if dst == Status.STALE.value:
            # By definition of stale there is nothing left in the window.
            self._valid_run = 0
        return self._state

    def force(self, dst: Status | str, *, now_s: float) -> str:
        """Move to `dst`, raising `TransitionError` if that arrow does not exist.

        Exposed for replaying a recorded sequence of states. It is not a way
        around the diagram: an illegal pair raises.
        """
        return self._enter(_name(dst), now_s)
