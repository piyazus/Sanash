"""Fig. 1 - Cumulative count N_t vs door-event truth (time series).

Spec: paper/figures.md, Fig. 1 section (final, 2026-08-10); semantics per the
JSON `definitions` block, which is authoritative. Data:
paper/results/apc_validation.json only; all five series plotted as-is (no
masking, offsetting, re-anchoring, or interpolation).

Five step lines: door-event truth (bold staircase, series_1hz.door_truth),
line-position truth (thin dotted, series_1hz.truth), and the three raw
anchor-0 estimates (mz8 emphasised). Episode markers on the top edge.
Right-edge labels give RAW endpoints (per JSON); final errors are NOT
annotated (caption/Table II territory).

The script asserts the definitions-block reconciliation
(estimate_end - door_truth_end == final_error_vs_door_truth) and fails
loudly if the JSON ever stops satisfying it.

Episode-marker t0 is derived from the alignment of event timestamps with the
1 Hz estimate series (established in the first build); the script fails
loudly if that alignment is ambiguous.

Output: fig1_nt_series.pdf + .svg next to this script.
Run: python fig1_nt_series.py   (deps: matplotlib, numpy only)
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

HERE = Path(__file__).resolve().parent
JSON_PATH = HERE.parent / "results" / "apc_validation.json"
OUT_STEM = HERE / "fig1_nt_series"

VARIANTS = ("full", "mz8", "mz4")
LEGEND = {
    "door": "door-event truth",
    "line": "line-position truth",
    "full": "full 848x480",
    "mz8": "8x8 sim",
    "mz4": "4x4 sim",
}

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


def derive_t0(nt: dict, grid: list) -> float:
    """Derive session-start t0 (epoch s) by aligning event timestamps with the
    1 Hz estimate series across all three variants (see first-build notes).

    An event at relative time tau in (i-1, i] first appears at sample i, so
    for a candidate t0 every per-second net event change must equal the
    estimate's step at that sample. Scan the 1 s anchor window at 5 ms
    resolution and demand a perfect match; refuse to guess otherwise."""
    est_full = nt["full"]["series_1hz"]["estimate"]
    n = len(grid)
    i1 = next(i for i in range(1, n) if est_full[i] != est_full[i - 1])
    t1 = nt["full"]["events"][0]["t_ns"] / 1e9
    base = t1 - i1  # t0 lies in [base, base + 1)

    def mismatches(t0: float) -> int:
        bad = 0
        for v in VARIANTS:
            est = nt[v]["series_1hz"]["estimate"]
            binned = {}
            for e in nt[v]["events"]:
                b = math.ceil(e["t_ns"] / 1e9 - t0)
                binned[b] = binned.get(b, 0) + (1 if e["kind"] == "board" else -1)
            for i in range(1, n):
                if est[i] - est[i - 1] != binned.get(i, 0):
                    bad += 1
        return bad

    deltas = [k / 200 for k in range(1, 200)]
    scores = {d: mismatches(base + d) for d in deltas}
    perfect = [d for d, s in scores.items() if s == 0]
    if not perfect:
        raise RuntimeError(
            "t0 alignment ambiguous (best residual %d mismatched bins) - "
            "ask Danyshpan, do not guess" % min(scores.values())
        )
    return base + (min(perfect) + max(perfect)) / 2


def endpoint_label(value: int) -> str:
    return "0" if value == 0 else f"{value:+d}"


def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    nt = data["nt"]

    grid = nt["full"]["series_1hz"]["t_rel_s"]
    door = nt["full"]["series_1hz"]["door_truth"]
    line = nt["full"]["series_1hz"]["truth"]
    for v in VARIANTS:  # identical grid + truths in all three variants
        assert nt[v]["series_1hz"]["t_rel_s"] == grid, v
        assert nt[v]["series_1hz"]["door_truth"] == door, v
        assert nt[v]["series_1hz"]["truth"] == line, v

    # definitions-block reconciliation (JSON wins; fail loudly if broken)
    for v in VARIANTS:
        est_end = nt[v]["series_1hz"]["estimate"][-1]
        err = nt[v]["final_error_vs_door_truth"]
        assert est_end - door[-1] == err, (v, est_end, door[-1], err)

    t0 = derive_t0(nt, grid)

    fig, ax = plt.subplots(figsize=(7.16, 2.7), layout="constrained")
    ax.grid(axis="y", color="0.92", linewidth=0.5, zorder=0)

    (h_door,) = ax.plot(  # bold reference staircase
        grid, door, drawstyle="steps-post", color="0.72", linewidth=2.6, zorder=2
    )
    (h_line,) = ax.plot(  # thin dotted, visually subordinate
        grid,
        line,
        drawstyle="steps-post",
        color="0.55",
        linewidth=0.7,
        linestyle=(0, (1, 1.2)),
        zorder=2.5,
    )
    (h_full,) = ax.plot(
        grid,
        nt["full"]["series_1hz"]["estimate"],
        drawstyle="steps-post",
        color="0.35",
        linewidth=0.8,
        linestyle=(0, (4, 1.5)),
        zorder=3,
    )
    (h_mz4,) = ax.plot(
        grid,
        nt["mz4"]["series_1hz"]["estimate"],
        drawstyle="steps-post",
        color="0.35",
        linewidth=0.8,
        linestyle=(0, (3, 1.5, 1, 1.5)),
        zorder=3,
    )
    (h_mz8,) = ax.plot(  # the paper's subject - emphasised
        grid,
        nt["mz8"]["series_1hz"]["estimate"],
        drawstyle="steps-post",
        color="0.0",
        linewidth=1.3,
        zorder=4,
    )

    # episode markers along the top edge: up = board, down = alight
    top = blended_transform_factory(ax.transData, ax.transAxes)
    episodes = data["variants"]["full"]["per_episode"]
    t_board = [(e["start_ns"] / 1e9 - t0) for e in episodes if e["kind"] == "board"]
    t_alight = [(e["start_ns"] / 1e9 - t0) for e in episodes if e["kind"] == "alight"]
    assert all(grid[0] <= t <= grid[-1] for t in t_board + t_alight)
    ax.scatter(
        t_board,
        [0.97] * len(t_board),
        transform=top,
        marker="^",
        s=14,
        facecolor="0.0",
        edgecolor="0.0",
        linewidth=0.5,
        zorder=5,
        clip_on=False,
    )
    ax.scatter(
        t_alight,
        [0.97] * len(t_alight),
        transform=top,
        marker="v",
        s=14,
        facecolor="#ffffff",
        edgecolor="0.0",
        linewidth=0.7,
        zorder=5,
        clip_on=False,
    )

    # right-edge RAW endpoint labels (values straight from the JSON series)
    x_lab = grid[-1] + 10
    for series, colour, weight in (
        (door, "0.45", "bold"),
        (nt["mz8"]["series_1hz"]["estimate"], "0.0", "normal"),
        (nt["full"]["series_1hz"]["estimate"], "0.35", "normal"),
        (nt["mz4"]["series_1hz"]["estimate"], "0.35", "normal"),
    ):
        ax.text(
            x_lab,
            series[-1],
            endpoint_label(series[-1]),
            fontsize=6.5,
            color=colour,
            fontweight=weight,
            ha="left",
            va="center",
            zorder=5,
        )

    all_series = [door, line] + [nt[v]["series_1hz"]["estimate"] for v in VARIANTS]
    lo = min(min(s) for s in all_series)
    hi = max(max(s) for s in all_series)
    ax.set_xlim(grid[0], grid[-1] + 50)  # small right margin for endpoint labels
    ax.set_ylim(lo - 0.4, hi + 0.9)  # headroom for the marker row
    ax.set_yticks(range(int(lo), int(hi) + 1))  # integer ticks only
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("on-board count $N_t$ (anchor 0)")

    fig.legend(
        [h_door, h_line, h_full, h_mz8, h_mz4],
        [LEGEND["door"], LEGEND["line"], LEGEND["full"], LEGEND["mz8"], LEGEND["mz4"]],
        loc="outside lower center",
        ncol=5,
        frameon=False,
        fontsize=7,
        columnspacing=1.4,
        handlelength=2.2,
    )

    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUT_STEM.with_suffix(".svg"))
    print("wrote", OUT_STEM.with_suffix(".pdf"))
    print("wrote", OUT_STEM.with_suffix(".svg"))
    print(
        "derived t0 = %.3f s epoch (episode markers: %d board, %d alight)"
        % (t0, len(t_board), len(t_alight))
    )
    print(
        "endpoints: door_truth %+d | " % door[-1]
        + " | ".join(
            "%s %+d (err %+d)"
            % (
                v,
                nt[v]["series_1hz"]["estimate"][-1],
                nt[v]["final_error_vs_door_truth"],
            )
            for v in VARIANTS
        )
    )


if __name__ == "__main__":
    main()
