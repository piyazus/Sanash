"""Counting metrics.

MAE and RMSE over per-image counts are the two the crowd counting literature
reports, so they are what a later comparison against CSRNet + PFCASA needs.

`level_accuracy` is a placeholder with a mandatory `thresholds` argument. The
five level scale of the product is not defined: GROUND_TRUTH.md section 3.2
lists the meaning of the 0..1 score, the level boundaries and their relation to
a specific bus capacity as open. Hardcoding thresholds here would invent that
decision and then hide it inside a metric, so the caller has to supply them and
own them. Nothing in this package calls it with real product thresholds yet.
"""

import numpy as np


def mae(pred, true) -> float:
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    return float(np.abs(pred - true).mean())


def rmse(pred, true) -> float:
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    return float(np.sqrt(((pred - true) ** 2).mean()))


def counts_to_levels(counts, thresholds) -> np.ndarray:
    """Bin counts into ordinal levels. `thresholds` must be sorted ascending.

    With four thresholds this yields levels 0..4. The thresholds themselves are
    a product decision that has not been made.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be a strictly increasing 1-D sequence")
    return np.digitize(np.asarray(counts, dtype=np.float64), thresholds)


def level_accuracy(pred_count, true_count, thresholds) -> float:
    """Share of images whose predicted level equals the true level."""
    pred_levels = counts_to_levels(pred_count, thresholds)
    true_levels = counts_to_levels(true_count, thresholds)
    return float((pred_levels == true_levels).mean())
