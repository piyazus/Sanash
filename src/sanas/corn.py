"""CORN ordinal regression head: loss and decoding.

CORN (Conditional Ordinal Regression for Neural networks) turns K ordered
classes into K-1 binary tasks. Task k answers "is y > k?" but is trained only
on the samples that reached that point, i.e. those with y >= k. That
conditional training is what makes the resulting cumulative probabilities
rank-consistent by construction, which is the property we want: predicting
"level 3" should never be cheaper than passing through level 2.

Reference: Shi, Cao & Raschka, "Deep Neural Networks for Rank-Consistent
Ordinal Regression Based On Conditional Probabilities" (arXiv 2111.08851).
Implemented here from the formulation rather than vendored, so the repo does
not inherit a third-party licence for six lines of arithmetic. The reference
implementation (Raschka-research-group/coral-pytorch) is MIT if a
drop-in replacement is ever wanted.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def corn_loss(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Mean CORN loss.

    logits  (N, num_classes - 1) raw scores, one per ordinal threshold
    targets (N,) integer class indices in [0, num_classes - 1]
    """
    if logits.ndim != 2 or logits.shape[1] != num_classes - 1:
        raise ValueError(
            f"expected logits of shape (N, {num_classes - 1}), got {tuple(logits.shape)}"
        )
    targets = targets.long()
    total = logits.new_zeros(())
    used = 0
    for k in range(num_classes - 1):
        # Conditional subset: only samples that got at least this far.
        mask = targets >= k
        if not bool(mask.any()):
            continue
        binary = (targets[mask] > k).float()
        total = total + F.binary_cross_entropy_with_logits(
            logits[mask, k], binary, reduction="mean"
        )
        used += 1
    return total / max(used, 1)


def corn_cumulative_probs(logits: torch.Tensor) -> torch.Tensor:
    """P(y > k) for each threshold k, as the running product of conditionals."""
    return torch.cumprod(torch.sigmoid(logits), dim=1)


def corn_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Decode logits to integer class indices."""
    return (corn_cumulative_probs(logits) > threshold).sum(dim=1)


def corn_expected_class(logits: torch.Tensor) -> torch.Tensor:
    """Soft class estimate: sum of P(y > k). Useful as a continuous score."""
    return corn_cumulative_probs(logits).sum(dim=1)
