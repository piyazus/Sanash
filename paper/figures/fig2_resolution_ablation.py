"""Fig. 2 - Resolution ablation summary (two stacked panels, single column).

Spec: paper/figures.md, Fig. 2 section. Data: paper/results/apc_validation.json
only - detection rates with Wilson 95% CIs from variants.<v>, N_t MAE with
block-bootstrap 95% CIs from nt.<v>. Nothing hand-typed.
Output: fig2_resolution_ablation.pdf + .svg next to this script.

Run: python fig2_resolution_ablation.py   (deps: matplotlib, numpy only)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE.parent / "results" / "apc_validation.json"
OUT_STEM = HERE / "fig2_resolution_ablation"

VARIANTS = ("full", "mz8", "mz4")
XTICKLABELS = ("full 848x480", "8x8", "4x4")  # per spec

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
    }
)


def asym_err(vals, cis):
    lo = [v - ci[0] for v, ci in zip(vals, cis)]
    hi = [ci[1] - v for v, ci in zip(vals, cis)]
    return [lo, hi]


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    x = range(len(VARIANTS))

    det = [data["variants"][v]["detection_rate"] for v in VARIANTS]
    det_ci = [data["variants"][v]["detection_ci95"] for v in VARIANTS]
    counts = [
        (data["variants"][v]["true_positives"], data["variants"][v]["episodes"])
        for v in VARIANTS
    ]
    mae = [data["nt"][v]["mae_vs_door_truth"] for v in VARIANTS]
    mae_ci = [data["nt"][v]["mae_vs_door_truth_ci95"] for v in VARIANTS]

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.5, 3.5), sharex=True, layout="constrained"
    )

    # (a) detection rate, Wilson 95% CI
    ax_a.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)
    ax_a.errorbar(
        x,
        det,
        yerr=asym_err(det, det_ci),
        fmt="o",
        color="0.0",
        markersize=4,
        capsize=2.5,
        linewidth=0.9,
        zorder=3,
    )
    for xi, rate, (tp, ep) in zip(x, det, counts):
        ax_a.annotate(
            f"{tp}/{ep}",
            (xi, rate),
            textcoords="offset points",
            xytext=(7, -2),
            fontsize=7,
        )
    ax_a.set_ylim(0.5, 1.0)  # per spec
    ax_a.set_ylabel("episodes detected (fraction)")
    ax_a.text(0.02, 0.90, "(a)", transform=ax_a.transAxes)

    # (b) N_t MAE vs door-event truth, bootstrap 95% CI
    ax_b.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)
    ax_b.errorbar(
        x,
        mae,
        yerr=asym_err(mae, mae_ci),
        fmt="o",
        color="0.0",
        markersize=4,
        capsize=2.5,
        linewidth=0.9,
        zorder=3,
    )
    ax_b.set_ylim(0, max(ci[1] for ci in mae_ci) * 1.15)  # start at 0 per spec
    ax_b.set_ylabel("$N_t$ MAE (persons)")
    ax_b.text(0.02, 0.90, "(b)", transform=ax_b.transAxes)

    ax_b.set_xlim(-0.5, len(VARIANTS) - 0.5)
    ax_b.set_xticks(list(x))
    ax_b.set_xticklabels(XTICKLABELS)

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".svg"))
    print("wrote", OUT_STEM.with_suffix(".pdf"))
    print("wrote", OUT_STEM.with_suffix(".svg"))


if __name__ == "__main__":
    main()
