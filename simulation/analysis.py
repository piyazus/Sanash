"""
Sanash Simulation Post-Processing Analysis
==========================================

Loads simulation output CSV, generates publication-ready figures,
runs ANOVA with Bonferroni post-hoc, and tornado sensitivity chart.

Usage:
    python simulation/analysis.py
    python simulation/analysis.py --input simulation/output/simulation_results.csv
"""

import argparse
import logging
import math
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}. Run: pip install -r simulation/requirements.txt"
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCENARIO_COLORS = {
    "no_info": "#90a4ae",
    "perfect_info": "#43a047",
    "imperfect_info": "#fb8c00",
}
SCENARIO_LABELS = {
    "no_info": "No Information",
    "perfect_info": "Perfect Information",
    "imperfect_info": "Imperfect Information",
}


def load_results(path: str) -> "pd.DataFrame":
    """
    Load simulation results CSV, generating synthetic data if absent.

    Parameters
    ----------
    path : str
        Path to simulation_results.csv.

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(path)
    if csv_path.exists():
        log.info(f"Loading results from {csv_path}")
        return pd.read_csv(csv_path)

    log.warning(f"Results not found at {csv_path}. Generating synthetic demo data.")
    rng = np.random.default_rng(42)
    rows = []
    scenario_params = {
        "no_info":        {"wait_mu": 6.5, "wait_sd": 1.2, "lf_cv": 0.35, "boarded": 420},
        "perfect_info":   {"wait_mu": 4.1, "wait_sd": 0.8, "lf_cv": 0.18, "boarded": 480},
        "imperfect_info": {"wait_mu": 4.8, "wait_sd": 0.9, "lf_cv": 0.23, "boarded": 465},
    }
    for scenario, p in scenario_params.items():
        for rep in range(100):
            rows.append({
                "replication": rep,
                "scenario": scenario,
                "avg_wait_time": max(0.5, rng.normal(p["wait_mu"], p["wait_sd"])),
                "median_wait_time": max(0.5, rng.normal(p["wait_mu"] * 0.9, p["wait_sd"])),
                "max_occupancy": rng.uniform(0.7, 1.0),
                "avg_load_factor": rng.uniform(0.5, 0.9),
                "load_factor_cv": max(0.05, rng.normal(p["lf_cv"], 0.04)),
                "total_boarded": int(rng.integers(p["boarded"] - 30, p["boarded"] + 30)),
                "total_refused": int(rng.integers(0, 20)),
                "total_arrived": int(rng.integers(490, 520)),
            })
    return pd.DataFrame(rows)


def plot_wait_times(df: "pd.DataFrame", output_dir: Path) -> None:
    """
    Box plot of average wait time by scenario with significance brackets.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : Path
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = ["no_info", "perfect_info", "imperfect_info"]
    data = [df[df["scenario"] == s]["avg_wait_time"].values for s in scenarios]
    labels = [SCENARIO_LABELS[s] for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("Average Wait Time (minutes)", fontsize=11)
    ax.set_title("Passenger Wait Time by Information Scenario\n(100 replications each)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_path = output_dir / "wait_times_boxplot.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_load_factor_cv(df: "pd.DataFrame", output_dir: Path) -> None:
    """
    Bar chart with error bars of load factor CV by scenario.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : Path
    """
    scenarios = ["no_info", "perfect_info", "imperfect_info"]
    means = [df[df["scenario"] == s]["load_factor_cv"].mean() for s in scenarios]
    sems = [df[df["scenario"] == s]["load_factor_cv"].sem() for s in scenarios]
    labels = [SCENARIO_LABELS[s] for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, means, yerr=sems, color=colors, alpha=0.85,
                  error_kw={"elinewidth": 2, "capsize": 6}, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylabel("Load Factor CV (std / mean)", fontsize=11)
    ax.set_title("Bus Load Distribution Equity by Scenario\n(lower CV = more even distribution)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.4)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_path = output_dir / "load_factor_cv.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_occupancy_timeseries(df: "pd.DataFrame", output_dir: Path) -> None:
    """
    Line chart of max occupancy per replication coloured by scenario.

    Parameters
    ----------
    df : pd.DataFrame
    output_dir : Path
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for scenario in ["no_info", "perfect_info", "imperfect_info"]:
        sdf = df[df["scenario"] == scenario].sort_values("replication")
        ax.plot(
            sdf["replication"], sdf["max_occupancy"],
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
            alpha=0.7, linewidth=1.5,
        )
    ax.set_xlabel("Replication", fontsize=11)
    ax.set_ylabel("Max Occupancy Ratio", fontsize=11)
    ax.set_title("Peak Bus Occupancy Across Replications", fontsize=12, fontweight="bold")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="Full capacity")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_path = output_dir / "occupancy_timeseries.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def run_anova(df: "pd.DataFrame") -> dict:
    """
    One-way ANOVA on avg_wait_time by scenario with Bonferroni post-hoc.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with f_stat, p_value, and pairwise comparison list
    """
    scenarios = ["no_info", "perfect_info", "imperfect_info"]
    groups = [df[df["scenario"] == s]["avg_wait_time"].values for s in scenarios]
    f_stat, p_val = stats.f_oneway(*groups)

    pairwise = []
    n_comparisons = 3  # C(3,2)
    for i in range(len(scenarios)):
        for j in range(i + 1, len(scenarios)):
            s1, s2 = scenarios[i], scenarios[j]
            t_stat, p = stats.ttest_ind(groups[i], groups[j])
            p_bonf = min(1.0, p * n_comparisons)
            pairwise.append({
                "comparison": f"{SCENARIO_LABELS[s1]} vs {SCENARIO_LABELS[s2]}",
                "t_stat": round(float(t_stat), 3),
                "p_raw": round(float(p), 4),
                "p_bonferroni": round(float(p_bonf), 4),
                "significant": p_bonf < 0.05,
            })

    result = {"f_stat": round(float(f_stat), 3), "p_value": round(float(p_val), 6),
               "pairwise": pairwise}
    log.info(f"ANOVA: F={f_stat:.3f}, p={p_val:.6f}")
    for pw in pairwise:
        sig = "***" if pw["significant"] else "ns"
        log.info(f"  {pw['comparison']}: t={pw['t_stat']}, p_bonf={pw['p_bonferroni']} [{sig}]")
    return result


def sensitivity_analysis(output_dir: Path) -> None:
    """
    Tornado chart showing sensitivity of avg_wait_time to beta changes (+/-20%).

    Parameters
    ----------
    output_dir : Path
    """
    try:
        from simulation import config as cfg
    except ImportError:
        try:
            import config as cfg  # type: ignore
        except ImportError:
            class cfg:  # type: ignore
                BETA_WAIT = -0.15
                BETA_CROWDING_PACKED = -1.2
                BETA_CROWDING_STANDING = -0.6
                BETA_PEAK = -0.3

    def quick_avg_wait(bw, bp, bs, bpk, n=500, seed=0):
        """Approximate avg wait via Monte Carlo (no full SimPy run)."""
        rng = np.random.default_rng(seed)
        waits = []
        for _ in range(n):
            occ = rng.uniform(0, 1)
            wait_next = rng.exponential(10)
            pk = bool(rng.random() > 0.5)
            is_packed = occ > 0.8
            is_standing = 0.5 < occ <= 0.8
            u = (bw * wait_next
                 + (bp if is_packed else 0)
                 + (bs if is_standing else 0)
                 + (bpk if pk else 0))
            p_wait = 1 / (1 + math.exp(-u))
            if rng.random() < p_wait:
                waits.append(wait_next * rng.uniform(0.3, 0.9))
            else:
                waits.append(rng.uniform(0, 3))
        return float(np.mean(waits)) if waits else 5.0

    base_args = (cfg.BETA_WAIT, cfg.BETA_CROWDING_PACKED,
                 cfg.BETA_CROWDING_STANDING, cfg.BETA_PEAK)
    base = quick_avg_wait(*base_args)

    params = [
        ("Beta Wait", cfg.BETA_WAIT, 0),
        ("Beta Packed", cfg.BETA_CROWDING_PACKED, 1),
        ("Beta Standing", cfg.BETA_CROWDING_STANDING, 2),
        ("Beta Peak", cfg.BETA_PEAK, 3),
    ]

    swings = []
    for name, base_val, idx in params:
        args_lo = list(base_args); args_lo[idx] = base_val * 0.8
        args_hi = list(base_args); args_hi[idx] = base_val * 1.2
        lo = quick_avg_wait(*args_lo)
        hi = quick_avg_wait(*args_hi)
        swings.append((name, lo - base, hi - base))

    swings.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
    labels = [s[0] for s in swings]
    lows = [s[1] for s in swings]
    highs = [s[2] for s in swings]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(labels))
    ax.barh(y, lows, color="#ef5350", alpha=0.8, label="-20% change", height=0.4)
    ax.barh(y, highs, color="#42a5f5", alpha=0.8, label="+20% change", height=0.4)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Change in Avg Wait Time (min)")
    ax.set_title("Sensitivity Tornado Chart\nBeta Coefficients +/-20%", fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    plt.tight_layout()
    save_path = output_dir / "sensitivity_tornado.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sanash Simulation Post-Processing Analysis")
    parser.add_argument("--input", "-i", default="simulation/output/simulation_results.csv",
                        help="Path to simulation results CSV")
    parser.add_argument("--output", "-o", default="simulation/output",
                        help="Output directory for figures")
    return parser.parse_args()


def main() -> None:
    """Run the full post-simulation analysis pipeline."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.input)
    log.info(f"Loaded {len(df)} rows. Scenarios: {df['scenario'].unique().tolist()}")

    plot_wait_times(df, output_dir)
    plot_load_factor_cv(df, output_dir)
    plot_occupancy_timeseries(df, output_dir)

    anova_result = run_anova(df)
    anova_path = output_dir / "anova_results.txt"
    with open(anova_path, "w") as f:
        f.write("One-Way ANOVA: avg_wait_time by scenario\n")
        f.write(f"F-statistic: {anova_result['f_stat']}\n")
        f.write(f"p-value:     {anova_result['p_value']}\n\n")
        f.write("Bonferroni pairwise comparisons:\n")
        for pw in anova_result["pairwise"]:
            sig = "SIGNIFICANT" if pw["significant"] else "not significant"
            f.write(f"  {pw['comparison']}: t={pw['t_stat']}, "
                    f"p_bonf={pw['p_bonferroni']} [{sig}]\n")
    log.info(f"Saved: {anova_path}")

    sensitivity_analysis(output_dir)
    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
