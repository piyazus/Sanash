"""Fig. 3 - Counting-band width sweep (single column).

Spec: paper/figures.md, Fig. 3 section (2026-08-10 update: reference line at
0.7 m, +/-0.35 m head-height strip, 60x60 deg FoV confirmed - see
docs/hardware/node_design.md). Data: paper/results/apc_validation.json
band_ablation only; nothing hand-typed. X axis descends 2.0 -> 0.5 m so the
narrowing band reads left to right.
Output: fig3_band_sweep.pdf + .svg next to this script.

Run: python fig3_band_sweep.py   (deps: matplotlib, numpy only)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE.parent / "results" / "apc_validation.json"
OUT_STEM = HERE / "fig3_band_sweep"

HALF_WIDTHS = ("1.0", "0.7", "0.5", "0.35", "0.25")  # JSON keys w; x = 2w
REF_X = 0.7  # m; +/-0.35 m head-height strip (node_design.md, confirmed FoV)
REF_LABEL = "±0.35 m head-height strip,\nuntilted ceiling node (60°×60° FoV)"

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


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    band = data["band_ablation"]

    x = [2 * float(w) for w in HALF_WIDTHS]
    tp = [band[w]["true_positives"] for w in HALF_WIDTHS]
    ep = [band[w]["episodes"] for w in HALF_WIDTHS]
    y = [t / e for t, e in zip(tp, ep)]
    ci = [band[w]["detection_ci95"] for w in HALF_WIDTHS]
    yerr = [[v - c[0] for v, c in zip(y, ci)], [c[1] - v for v, c in zip(y, ci)]]

    fig, ax = plt.subplots(figsize=(3.5, 2.6), layout="constrained")
    ax.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o-",
        color="0.0",
        markersize=4,
        capsize=2.5,
        linewidth=1.0,
        zorder=3,
    )
    for xi, yi, t, e in zip(x, y, tp, ep):
        ax.annotate(
            f"{t}/{e}",
            (xi, yi),
            textcoords="offset points",
            xytext=(2, 7),
            fontsize=7,
        )

    # reference line: head-height strip of the untilted ceiling node
    ax.axvline(REF_X, color="0.45", linestyle=(0, (4, 2)), linewidth=0.9, zorder=2)
    ax.annotate(
        REF_LABEL,
        xy=(REF_X, 0.47),
        xytext=(1.98, 0.415),
        fontsize=6.5,
        color="0.15",
        arrowprops={"arrowstyle": "->", "color": "0.35", "linewidth": 0.8},
    )

    ax.set_xlim(2.12, 0.38)  # descending: narrowing band reads left to right
    ax.set_xticks(x)
    ax.set_ylim(0.38, 1.0)
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("counting-band width along travel direction (m)")
    ax.set_ylabel("episodes detected (fraction of 25)")

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".svg"))
    print("wrote", OUT_STEM.with_suffix(".pdf"))
    print("wrote", OUT_STEM.with_suffix(".svg"))


if __name__ == "__main__":
    main()
