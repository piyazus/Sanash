"""Target definition for the occupancy model.

The smoke test supervises on RAW COUNT (0-4 persons per frame), because the
substitute dataset never contains more than four occupants. The eventual
product target is a 0-1 continuous density mapped to 5 ordinal levels. Both
are expressed through TargetConfig so switching is a config change, not a
rewrite: the model head width, the loss and the decoder all derive from
``num_classes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RAW_COUNT = "raw_count"
NORMALIZED = "normalized"


@dataclass
class TargetConfig:
    """How a per-frame occupant count becomes a training target.

    mode=raw_count   classes are the counts themselves, 0..max_count.
                     decode() returns a count.
    mode=normalized  counts are bucketed into num_levels ordinal levels by
                     occupancy fraction count/capacity. decode() returns the
                     0-1 density score. Not used yet: needs real crowding data
                     and a decision on `capacity` for the target vehicle.
    """

    mode: str = RAW_COUNT
    label_column: str = "count_view"
    # raw_count
    max_count: int = 4
    # normalized (unused until real crowding data exists)
    capacity: int | None = None
    num_levels: int | None = 5
    level_edges: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])

    def __post_init__(self) -> None:
        if self.mode not in (RAW_COUNT, NORMALIZED):
            raise ValueError(f"unknown target mode {self.mode!r}")
        if self.mode == NORMALIZED:
            if not self.capacity or self.capacity <= 0:
                raise ValueError("normalized mode needs a positive capacity")
            if not self.num_levels or self.num_levels < 2:
                raise ValueError("normalized mode needs num_levels >= 2")
            if len(self.level_edges) != self.num_levels - 1:
                raise ValueError(
                    f"num_levels={self.num_levels} needs {self.num_levels - 1} "
                    f"level_edges, got {len(self.level_edges)}"
                )

    @property
    def num_classes(self) -> int:
        return self.max_count + 1 if self.mode == RAW_COUNT else int(self.num_levels)

    @property
    def num_thresholds(self) -> int:
        """Width of the CORN head: K-1 binary tasks for K ordered classes."""
        return self.num_classes - 1

    def encode(self, count: int) -> int:
        """Occupant count -> ordinal class index."""
        if self.mode == RAW_COUNT:
            return max(0, min(int(count), self.max_count))
        frac = count / self.capacity
        cls = 0
        for edge in self.level_edges:
            if frac >= edge:
                cls += 1
        return min(cls, self.num_classes - 1)

    def decode(self, cls: int) -> float:
        """Ordinal class index -> reported value.

        raw_count  -> occupant count
        normalized -> 0-1 density score (lower edge of the predicted level)
        """
        cls = max(0, min(int(cls), self.num_classes - 1))
        if self.mode == RAW_COUNT:
            return float(cls)
        edges = [0.0, *self.level_edges]
        return float(edges[cls])

    def describe(self) -> str:
        if self.mode == RAW_COUNT:
            return (
                f"raw_count on '{self.label_column}', classes 0..{self.max_count} "
                f"({self.num_classes} classes, {self.num_thresholds} CORN thresholds)"
            )
        return (
            f"normalized on '{self.label_column}', capacity={self.capacity}, "
            f"{self.num_levels} levels, edges={self.level_edges}"
        )
