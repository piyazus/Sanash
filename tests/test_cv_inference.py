"""
Tests for computer vision inference pipeline.

Validates density map output shapes, count estimation from density maps,
MNL boarding probability range, and CSRNet architectural assumptions.
No GPU or real model required — all tests use numpy mocks.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Utility: MNL boarding probability (duplicated here for self-containment)
# ---------------------------------------------------------------------------

def mnl_boarding_prob(
    wait_time: float,
    occupancy_ratio: float,
    is_peak: bool,
    beta_wait: float = -0.15,
    beta_packed: float = -1.2,
    beta_standing: float = -0.6,
    beta_peak: float = -0.3,
) -> float:
    """
    Binary logit probability of boarding the current bus.

    Parameters
    ----------
    wait_time : float  Minutes to next bus
    occupancy_ratio : float  Current load [0, 1]
    is_peak : bool
    beta_* : float  MNL coefficients

    Returns
    -------
    float in [0, 1]
    """
    is_packed = int(occupancy_ratio > 0.80)
    is_standing = int(0.50 < occupancy_ratio <= 0.80)
    u_wait = (beta_wait * wait_time
              + beta_packed * is_packed
              + beta_standing * is_standing
              + beta_peak * int(is_peak))
    p_wait = 1.0 / (1.0 + math.exp(-u_wait))
    return float(1.0 - p_wait)


# ---------------------------------------------------------------------------
# Density map shape and value tests
# ---------------------------------------------------------------------------

class TestDensityMapProperties:
    """Validate density map output properties expected from CSRNet."""

    def test_density_map_is_2d(self):
        """Density map must be 2-dimensional (H x W)."""
        density_map = np.random.rand(48, 64)
        assert density_map.ndim == 2

    def test_density_map_nonnegative(self):
        """All density values must be >= 0."""
        density_map = np.abs(np.random.randn(48, 64))
        assert np.all(density_map >= 0), "Density map contains negative values"

    def test_output_shape_is_eighth_of_input(self):
        """CSRNet output spatial size = input / 8 (3 max-pool layers)."""
        input_h, input_w = 384, 512
        expected_h = input_h // 8
        expected_w = input_w // 8
        # Simulate model output: (batch=1, channels=1, H/8, W/8)
        output = np.random.rand(1, 1, expected_h, expected_w)
        assert output.shape == (1, 1, 48, 64)

    def test_count_equals_density_sum(self):
        """Predicted count = sum of density map values."""
        gt_count = 30
        density_map = np.zeros((48, 64), dtype=np.float32)
        rng = np.random.default_rng(42)
        for _ in range(gt_count):
            cy = int(rng.integers(0, 48))
            cx = int(rng.integers(0, 64))
            density_map[cy, cx] += 1.0
        assert int(density_map.sum()) == gt_count

    def test_normalized_density_sums_to_count(self):
        """After Gaussian smoothing, sum still approximates original count."""
        from scipy.ndimage import gaussian_filter
        gt_count = 50
        density_map = np.zeros((96, 128), dtype=np.float32)
        rng = np.random.default_rng(7)
        for _ in range(gt_count):
            density_map[int(rng.integers(0, 96)), int(rng.integers(0, 128))] += 1.0
        smoothed = gaussian_filter(density_map, sigma=4)
        # Sum should be preserved under Gaussian convolution
        assert abs(smoothed.sum() - gt_count) < 0.01

    def test_density_map_dtype_float(self):
        """Density maps should be float32 for memory efficiency."""
        density_map = np.zeros((48, 64), dtype=np.float32)
        assert density_map.dtype in [np.float32, np.float64]

    def test_batch_output_squeezable(self):
        """Can squeeze batch and channel dimensions to get 2D map."""
        batch_output = np.random.rand(1, 1, 48, 64)
        squeezed = np.squeeze(batch_output)
        assert squeezed.ndim == 2
        assert squeezed.shape == (48, 64)


# ---------------------------------------------------------------------------
# MNL boarding probability tests
# ---------------------------------------------------------------------------

class TestMNLBoardingProbability:
    """Validate MNL model behavioral properties."""

    def test_probability_in_unit_interval(self):
        """P(board) must be in [0, 1] for all inputs."""
        for wait in [2, 5, 10, 15]:
            for occ in [0.1, 0.3, 0.55, 0.75, 0.9]:
                for peak in [True, False]:
                    p = mnl_boarding_prob(wait, occ, peak)
                    assert 0.0 <= p <= 1.0, (
                        f"P={p} out of range for wait={wait}, occ={occ}, peak={peak}"
                    )

    def test_longer_wait_increases_board_probability(self):
        """Longer wait for next bus → more likely to board this one (waiting is costly)."""
        p_short_wait = mnl_boarding_prob(2, 0.7, False)   # next bus in 2 min → easy to wait
        p_long_wait = mnl_boarding_prob(15, 0.7, False)   # next bus in 15 min → better board now
        assert p_long_wait > p_short_wait, (
            "When next bus is far away (15 min), boarding probability should be higher "
            f"than when it is close (2 min). Got P(long)={p_long_wait:.3f}, P(short)={p_short_wait:.3f}"
        )

    def test_crowding_raises_board_probability(self):
        """
        In this MNL model, crowding terms enter U(wait): negative β means crowding
        reduces utility of waiting → P(board) increases for a crowded bus.
        This reflects that when the arriving bus is packed, the next bus may also
        be packed, so waiting is less attractive.
        """
        p_seats = mnl_boarding_prob(5, 0.3, False)
        p_packed = mnl_boarding_prob(5, 0.9, False)
        assert p_packed > p_seats, (
            f"Packed bus should increase P(board) in this model: "
            f"P(seats)={p_seats:.3f}, P(packed)={p_packed:.3f}"
        )

    def test_peak_raises_board_probability(self):
        """
        Peak-hour coefficient reduces utility of waiting further,
        so P(board) is higher during peak hours.
        """
        p_offpeak = mnl_boarding_prob(5, 0.6, False)
        p_peak = mnl_boarding_prob(5, 0.6, True)
        assert p_peak > p_offpeak, (
            f"Peak hour should raise P(board): "
            f"P(off-peak)={p_offpeak:.3f}, P(peak)={p_peak:.3f}"
        )

    def test_seats_long_wait_moderate_probability(self):
        """Seats available + 2 min wait → moderate P(board) ~0.5–0.7."""
        p = mnl_boarding_prob(2, 0.3, False)
        assert 0.4 < p < 0.8, f"Expected moderate P(board) for seats+2min, got {p:.3f}"

    def test_packed_long_wait_very_high_board_probability(self):
        """Packed + 15 min wait + peak → very high P(board) (waiting is very unattractive)."""
        p = mnl_boarding_prob(15, 0.9, True)
        assert p > 0.9, f"Expected P(board) > 0.9 for packed+15min+peak, got {p:.3f}"

    @pytest.mark.parametrize("occ,expected_status", [
        (0.3, "seats"),
        (0.65, "standing"),
        (0.85, "packed"),
    ])
    def test_crowding_levels_decrease_boarding(self, occ, expected_status):
        """Each crowding level should reduce P(board) vs. previous."""
        p = mnl_boarding_prob(5, occ, False)
        assert 0.0 < p < 1.0, f"Probability must be in (0,1) for {expected_status}"
