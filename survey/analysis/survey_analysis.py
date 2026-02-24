"""
Sanash Survey Analysis — Discrete Choice Experiment
=====================================================

Loads survey response CSV, fits a Multinomial Logit model, calculates
Willingness-to-Wait (WTW) values, and generates publication-ready figures.

Usage:
    python survey/analysis/survey_analysis.py
    python survey/analysis/survey_analysis.py --input survey/data/responses.csv
    python survey/analysis/survey_analysis.py --input data.csv --output results/ --verbose

Expected CSV columns:
    age_group, gender, employment, trip_freq, trip_purpose,
    q12_importance, q13_app,
    choice_1..6  (0=board, 1=wait),
    wait_time_1..6  (minutes: 2/5/10/15),
    crowding_1..6   (0=seats, 1=standing, 2=packed),
    is_peak_1..6    (0=off-peak, 1=peak)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import Logit
    from scipy import stats
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  pip install -r survey/analysis/requirements.txt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scenario design: (crowding_level, wait_time_min, is_peak)
# ---------------------------------------------------------------------------
SCENARIO_DESIGN = [
    (2, 2, 1),   # S1: packed,    2 min, peak
    (1, 5, 0),   # S2: standing,  5 min, off-peak
    (0, 10, 1),  # S3: seats,    10 min, peak
    (2, 15, 0),  # S4: packed,   15 min, off-peak
    (0, 2, 0),   # S5: seats,     2 min, off-peak
    (1, 10, 1),  # S6: standing, 10 min, peak
]

CROWDING_LABELS = {0: "Seats available", 1: "Standing room", 2: "Packed"}


# ---------------------------------------------------------------------------
# Data loading & synthetic generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = 167, seed: int = 42) -> "pd.DataFrame":
    """
    Generate synthetic survey data for demonstration / CI testing.

    Parameters
    ----------
    n : int
        Number of respondents.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic response DataFrame matching expected column schema.
    """
    rng = np.random.default_rng(seed)

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    genders = ["Male", "Female", "Prefer not to say"]
    employments = ["Student", "Full-time", "Part-time", "Unemployed", "Retired"]
    trip_freqs = ["1-2/week", "3-5/week", "6-10/week", "10+/week"]
    trip_purposes = ["Commute", "Education", "Shopping", "Leisure", "Other"]

    records = []
    for i in range(n):
        row: dict = {
            "respondent_id": i + 1,
            "age_group": rng.choice(age_groups, p=[0.30, 0.35, 0.20, 0.10, 0.05]),
            "gender": rng.choice(genders, p=[0.48, 0.49, 0.03]),
            "employment": rng.choice(employments, p=[0.35, 0.40, 0.10, 0.08, 0.07]),
            "trip_freq": rng.choice(trip_freqs, p=[0.15, 0.40, 0.30, 0.15]),
            "trip_purpose": rng.choice(trip_purposes, p=[0.45, 0.25, 0.15, 0.10, 0.05]),
            "q12_importance": rng.choice([1, 2, 3, 4, 5], p=[0.04, 0.08, 0.20, 0.38, 0.30]),
            "q13_app": rng.choice(
                ["Yes", "Maybe", "No"], p=[0.55, 0.30, 0.15]
            ),
        }
        # Simulate scenario choices based on MNL utility
        for s_idx, (crowding, wait, peak) in enumerate(SCENARIO_DESIGN, start=1):
            beta_w = -0.15
            beta_p = -1.2 if crowding == 2 else (-0.6 if crowding == 1 else 0.0)
            beta_pk = -0.3 if peak else 0.0
            utility_wait = beta_w * wait + beta_p + beta_pk
            p_wait = 1 / (1 + np.exp(-utility_wait))
            # Add individual heterogeneity
            p_wait = np.clip(p_wait + rng.normal(0, 0.08), 0.05, 0.95)
            row[f"choice_{s_idx}"] = int(rng.random() < p_wait)  # 1=wait, 0=board
            row[f"wait_time_{s_idx}"] = wait
            row[f"crowding_{s_idx}"] = crowding
            row[f"is_peak_{s_idx}"] = peak
        records.append(row)

    return pd.DataFrame(records)


def load_data(path: str) -> "pd.DataFrame":
    """
    Load survey responses from CSV, falling back to synthetic data if absent.

    Parameters
    ----------
    path : str
        Path to responses CSV file.

    Returns
    -------
    pd.DataFrame
        Survey response data.
    """
    csv_path = Path(path)
    if csv_path.exists():
        log.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        log.info(f"Loaded {len(df)} responses")
        return df
    else:
        log.warning(f"CSV not found at '{csv_path}'. Using synthetic data (n=167).")
        return generate_synthetic_data()


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def compute_descriptive_stats(df: "pd.DataFrame") -> dict:
    """
    Compute demographic frequency tables and summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Survey response DataFrame.

    Returns
    -------
    dict
        Dictionary of frequency tables keyed by demographic variable.
    """
    demo_cols = ["age_group", "gender", "employment", "trip_freq", "trip_purpose"]
    stats_dict: dict = {}
    for col in demo_cols:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True) * 100
            stats_dict[col] = freq
            log.info(f"\n{col}:\n{freq.to_string()}")
    # Q12 mean
    if "q12_importance" in df.columns:
        stats_dict["q12_mean"] = df["q12_importance"].mean()
        log.info(f"\nQ12 mean importance: {stats_dict['q12_mean']:.2f}")
    return stats_dict


# ---------------------------------------------------------------------------
# MNL model
# ---------------------------------------------------------------------------

def build_long_format(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Reshape wide survey data to long format for logit estimation.

    Each row in the long format represents one choice observation
    (respondent × scenario).

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format survey DataFrame.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        respondent_id, scenario, choice, wait_time,
        crowding_packed, crowding_standing, is_peak.
    """
    rows = []
    for _, resp in df.iterrows():
        for s_idx in range(1, 7):
            crowding_val = int(resp.get(f"crowding_{s_idx}", SCENARIO_DESIGN[s_idx - 1][0]))
            rows.append({
                "respondent_id": resp.get("respondent_id", 0),
                "scenario": s_idx,
                "choice": int(resp.get(f"choice_{s_idx}", 0)),
                "wait_time": float(resp.get(f"wait_time_{s_idx}", SCENARIO_DESIGN[s_idx - 1][1])),
                "crowding_packed": int(crowding_val == 2),
                "crowding_standing": int(crowding_val == 1),
                "is_peak": int(resp.get(f"is_peak_{s_idx}", SCENARIO_DESIGN[s_idx - 1][2])),
            })
    return pd.DataFrame(rows)


def fit_mnl_model(df: "pd.DataFrame") -> object:
    """
    Fit binary logit model: P(wait) = f(wait_time, crowding, peak).

    Model specification:
        U(wait) = β_wait·t + β_packed·I_packed + β_standing·I_standing
                + β_peak·I_peak + ε

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format survey DataFrame.

    Returns
    -------
    statsmodels RegressionResults
        Fitted logit model result object.
    """
    long_df = build_long_format(df)
    y = long_df["choice"]
    X = long_df[["wait_time", "crowding_packed", "crowding_standing", "is_peak"]]
    X = sm.add_constant(X)

    model = Logit(y, X)
    result = model.fit(disp=False, maxiter=200)
    log.info("\n" + result.summary().as_text())
    return result


# ---------------------------------------------------------------------------
# WTW calculation
# ---------------------------------------------------------------------------

def calculate_wtw(result: object) -> dict:
    """
    Calculate Willingness-to-Wait (WTW) from fitted logit coefficients.

    WTW = β_crowding / |β_wait|  (in minutes)

    Parameters
    ----------
    result : statsmodels LogitResults
        Fitted logit model result.

    Returns
    -------
    dict
        WTW values for packed and standing crowding levels.
    """
    params = result.params
    beta_wait = params.get("wait_time", params.iloc[1])
    beta_packed = params.get("crowding_packed", params.iloc[2])
    beta_standing = params.get("crowding_standing", params.iloc[3])

    wtw_packed = abs(beta_packed / beta_wait) if beta_wait != 0 else 0.0
    wtw_standing = abs(beta_standing / beta_wait) if beta_wait != 0 else 0.0

    wtw = {
        "wtw_packed_min": round(wtw_packed, 2),
        "wtw_standing_min": round(wtw_standing, 2),
        "beta_wait": round(float(beta_wait), 4),
        "beta_packed": round(float(beta_packed), 4),
        "beta_standing": round(float(beta_standing), 4),
    }
    log.info(f"\nWTW results:\n  Packed:   {wtw_packed:.1f} min\n  Standing: {wtw_standing:.1f} min")
    return wtw


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_demographics(df: "pd.DataFrame", output_dir: Path) -> None:
    """
    Generate 2×2 subplot of demographic distributions.

    Saves demographics.png to output_dir.

    Parameters
    ----------
    df : pd.DataFrame
        Survey response DataFrame.
    output_dir : Path
        Directory to save figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Survey Respondent Demographics (n=167)", fontsize=14, fontweight="bold")
    palette = sns.color_palette("Set2")

    cols = [
        ("age_group", "Age Group", "bar"),
        ("gender", "Gender", "pie"),
        ("employment", "Employment Status", "bar"),
        ("trip_freq", "Weekly Trip Frequency", "bar"),
    ]

    for ax, (col, title, chart_type) in zip(axes.flat, cols):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        counts = df[col].value_counts()
        if chart_type == "pie":
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                   colors=palette[:len(counts)], startangle=90)
        else:
            ax.bar(range(len(counts)), counts.values, color=palette[:len(counts)])
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    save_path = output_dir / "demographics.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_wtw(wtw_dict: dict, output_dir: Path) -> None:
    """
    Generate horizontal bar chart of Willingness-to-Wait values.

    Saves wtw_chart.png to output_dir.

    Parameters
    ----------
    wtw_dict : dict
        WTW values from calculate_wtw().
    output_dir : Path
        Directory to save figure.
    """
    labels = ["Standing room\n(50–80% full)", "Packed\n(>80% full)"]
    values = [wtw_dict["wtw_standing_min"], wtw_dict["wtw_packed_min"]]
    colors = ["#f0a500", "#d32f2f"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.5)
    ax.bar_label(bars, fmt="%.1f min", padding=4, fontsize=11)
    ax.set_xlabel("Willingness-to-Wait (minutes)", fontsize=11)
    ax.set_title("WTW: Extra Wait Accepted to Avoid Crowding", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.3)
    ax.axvline(x=0, color="black", linewidth=0.8)
    sns.despine(ax=ax)
    plt.tight_layout()

    save_path = output_dir / "wtw_chart.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_choice_proportions(df: "pd.DataFrame", output_dir: Path) -> None:
    """
    Generate stacked bar chart showing board vs wait proportions per scenario.

    Saves choice_proportions.png to output_dir.

    Parameters
    ----------
    df : pd.DataFrame
        Survey response DataFrame.
    output_dir : Path
        Directory to save figure.
    """
    scenario_labels = [
        "S1\nPacked, 2min\nPeak",
        "S2\nStanding, 5min\nOff-peak",
        "S3\nSeats, 10min\nPeak",
        "S4\nPacked, 15min\nOff-peak",
        "S5\nSeats, 2min\nOff-peak",
        "S6\nStanding, 10min\nPeak",
    ]

    board_pcts, wait_pcts = [], []
    for s_idx in range(1, 7):
        col = f"choice_{s_idx}"
        if col in df.columns:
            pct_wait = df[col].mean() * 100
        else:
            # Fallback from design
            crowding, wait, peak = SCENARIO_DESIGN[s_idx - 1]
            bw = -0.15 * wait + (-1.2 if crowding == 2 else -0.6 if crowding == 1 else 0) \
                 + (-0.3 if peak else 0)
            pct_wait = 1 / (1 + np.exp(-bw)) * 100
        wait_pcts.append(pct_wait)
        board_pcts.append(100 - pct_wait)

    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, board_pcts, label="Board now", color="#2196f3", alpha=0.85)
    ax.bar(x, wait_pcts, bottom=board_pcts, label="Wait for next", color="#ff9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=8)
    ax.set_ylabel("Respondents (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Boarding Choice Proportions by Scenario", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8)
    sns.despine(ax=ax)
    plt.tight_layout()

    save_path = output_dir / "choice_proportions.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Table export
# ---------------------------------------------------------------------------

def export_tables(
    df: "pd.DataFrame",
    result: object,
    wtw_dict: dict,
    output_dir: Path,
) -> None:
    """
    Export summary statistics, model coefficients, and WTW to CSV files.

    Parameters
    ----------
    df : pd.DataFrame
        Survey response DataFrame.
    result : statsmodels LogitResults
        Fitted model result.
    wtw_dict : dict
        WTW values from calculate_wtw().
    output_dir : Path
        Directory to save CSV files.
    """
    # Demographic summary
    demo_rows = []
    for col in ["age_group", "gender", "employment", "trip_freq", "trip_purpose"]:
        if col in df.columns:
            for val, cnt in df[col].value_counts().items():
                demo_rows.append({"variable": col, "category": val, "count": cnt,
                                  "percent": round(cnt / len(df) * 100, 1)})
    pd.DataFrame(demo_rows).to_csv(output_dir / "summary_stats.csv", index=False)
    log.info(f"Saved: {output_dir / 'summary_stats.csv'}")

    # Model coefficients
    coef_df = pd.DataFrame({
        "variable": result.params.index,
        "coefficient": result.params.values,
        "std_error": result.bse.values,
        "z_stat": result.tvalues.values,
        "p_value": result.pvalues.values,
    })
    coef_df.to_csv(output_dir / "model_coefficients.csv", index=False)
    log.info(f"Saved: {output_dir / 'model_coefficients.csv'}")

    # WTW results
    pd.DataFrame([wtw_dict]).to_csv(output_dir / "wtw_results.csv", index=False)
    log.info(f"Saved: {output_dir / 'wtw_results.csv'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sanash DCE Survey Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default="survey/data/responses.csv",
        help="Path to responses CSV (default: survey/data/responses.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="survey/output",
        help="Output directory for figures and tables (default: survey/output)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full survey analysis pipeline."""
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_data(args.input)
    log.info(f"Dataset: {len(df)} respondents, {df.shape[1]} columns")

    # 2. Descriptive statistics
    _ = compute_descriptive_stats(df)

    # 3. Fit MNL model
    result = fit_mnl_model(df)

    # 4. Calculate WTW
    wtw = calculate_wtw(result)
    print(f"\n{'='*50}")
    print("WILLINGNESS-TO-WAIT RESULTS")
    print(f"  Packed bus (>80%): {wtw['wtw_packed_min']:.1f} minutes")
    print(f"  Standing room (50-80%): {wtw['wtw_standing_min']:.1f} minutes")
    print(f"{'='*50}\n")

    # 5. Generate figures
    plot_demographics(df, output_dir)
    plot_wtw(wtw, output_dir)
    plot_choice_proportions(df, output_dir)

    # 6. Export tables
    export_tables(df, result, wtw, output_dir)

    log.info(f"Analysis complete. All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
