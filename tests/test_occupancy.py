"""
Tests for bus occupancy threshold classification logic.

The Sanash system classifies bus occupancy into three levels:
  Green  — seats available  (<= 50% full)
  Yellow — standing room    (51–80% full)
  Red    — packed           (> 80% full)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# System under test — occupancy classification function
# ---------------------------------------------------------------------------

def get_occupancy_status(count: int, capacity: int = 60) -> str:
    """
    Classify bus occupancy into Green / Yellow / Red status.

    Parameters
    ----------
    count : int
        Current passenger count.
    capacity : int
        Total bus capacity (default 60).

    Returns
    -------
    str : 'Green', 'Yellow', or 'Red'

    Raises
    ------
    ValueError : if count is negative or exceeds capacity
    """
    if count < 0:
        raise ValueError(f"Count cannot be negative: {count}")
    if count > capacity:
        raise ValueError(f"Count {count} exceeds capacity {capacity}")

    ratio = count / capacity
    if ratio <= 0.50:
        return "Green"
    elif ratio <= 0.80:
        return "Yellow"
    else:
        return "Red"


def get_occupancy_ratio(count: int, capacity: int = 60) -> float:
    """Return occupancy as a fraction [0.0, 1.0]."""
    if capacity <= 0:
        raise ValueError("Capacity must be > 0")
    return count / capacity


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestOccupancyStatus:
    """Parametrized tests for occupancy classification."""

    # -- Green zone (0–50%) --------------------------------------------------

    def test_empty_bus_is_green(self):
        assert get_occupancy_status(0) == "Green"

    def test_one_passenger_is_green(self):
        assert get_occupancy_status(1) == "Green"

    def test_exactly_half_capacity_is_green(self):
        """Boundary: 30/60 = 50.0% → Green."""
        assert get_occupancy_status(30, 60) == "Green"

    def test_green_zone_midpoint(self):
        assert get_occupancy_status(15) == "Green"

    # -- Yellow zone (51–80%) ------------------------------------------------

    def test_just_above_half_is_yellow(self):
        """Boundary: 31/60 = 51.7% → Yellow."""
        assert get_occupancy_status(31, 60) == "Yellow"

    def test_exactly_80_percent_is_yellow(self):
        """Boundary: 48/60 = 80.0% → Yellow."""
        assert get_occupancy_status(48, 60) == "Yellow"

    def test_yellow_zone_midpoint(self):
        assert get_occupancy_status(40) == "Yellow"

    # -- Red zone (>80%) -----------------------------------------------------

    def test_just_above_80_percent_is_red(self):
        """Boundary: 49/60 = 81.7% → Red."""
        assert get_occupancy_status(49, 60) == "Red"

    def test_full_bus_is_red(self):
        assert get_occupancy_status(60, 60) == "Red"

    def test_near_full_bus_is_red(self):
        assert get_occupancy_status(55) == "Red"

    # -- Error cases ---------------------------------------------------------

    def test_negative_count_raises_value_error(self):
        with pytest.raises(ValueError, match="negative"):
            get_occupancy_status(-1)

    def test_count_exceeds_capacity_raises_value_error(self):
        with pytest.raises(ValueError, match="exceeds capacity"):
            get_occupancy_status(61, 60)

    def test_exactly_capacity_does_not_raise(self):
        """Count == capacity is valid (full bus)."""
        result = get_occupancy_status(60, 60)
        assert result == "Red"

    # -- Custom capacity (minibus) -------------------------------------------

    def test_minibus_40_seat_green(self):
        assert get_occupancy_status(20, 40) == "Green"  # 50%

    def test_minibus_40_seat_yellow(self):
        assert get_occupancy_status(25, 40) == "Yellow"  # 62.5%

    def test_minibus_40_seat_red(self):
        assert get_occupancy_status(35, 40) == "Red"  # 87.5%

    # -- Parametrized boundary sweep -----------------------------------------

    @pytest.mark.parametrize("count,expected", [
        (0, "Green"),
        (1, "Green"),
        (29, "Green"),
        (30, "Green"),   # 50% — Green boundary
        (31, "Yellow"),  # 51.7% — Yellow starts
        (40, "Yellow"),
        (48, "Yellow"),  # 80% — Yellow boundary
        (49, "Red"),     # 81.7% — Red starts
        (55, "Red"),
        (60, "Red"),
    ])
    def test_boundary_sweep(self, count, expected):
        assert get_occupancy_status(count, 60) == expected


class TestOccupancyRatio:
    """Tests for occupancy ratio computation."""

    def test_empty_is_zero(self):
        assert get_occupancy_ratio(0, 60) == pytest.approx(0.0)

    def test_full_is_one(self):
        assert get_occupancy_ratio(60, 60) == pytest.approx(1.0)

    def test_half_is_point_five(self):
        assert get_occupancy_ratio(30, 60) == pytest.approx(0.5)

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError):
            get_occupancy_ratio(10, 0)
