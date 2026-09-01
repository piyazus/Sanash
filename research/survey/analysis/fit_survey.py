"""Fit the boarding/waiting choice model on the rebuilt Sanas survey data.

Model: binary logit, 1 = wait for the next bus, 0 = board the arriving bus.
Standard errors are clustered by respondent because each person contributes six
choices. Willingness-to-wait intervals come from a nonparametric bootstrap that
resamples respondents, not individual choices.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
TERMS = ["const", "wait_time", "packed", "is_peak"]


def design(df, terms=TERMS):
    X = pd.DataFrame({"const": 1.0}, index=df.index)
    for t in terms[1:]:
        X[t] = df[t].astype(float)
    return X


def check_rank(df, terms=TERMS):
    """The six fielded profiles must support the number of parameters fitted."""
    profiles = df.drop_duplicates("scenario").sort_values("scenario")
    X = design(profiles, terms).to_numpy()
    rank = np.linalg.matrix_rank(X)
    print(f"profiles: {X.shape[0]}  parameters: {X.shape[1]}  rank: {rank}")
    if rank < X.shape[1]:
        raise SystemExit(
            f"design cannot identify {X.shape[1]} parameters (rank {rank}); "
            "drop a term before fitting"
        )
    return rank


def fit(df, terms=TERMS):
    X = design(df, terms)
    model = sm.Logit(df["choice"].astype(float), X)
    return model.fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": df["respondent_id"]}
    )


def wtw(params):
    """Extra minutes of waiting accepted to avoid a packed bus."""
    return params["packed"] / abs(params["wait_time"])


def bootstrap_wtw(df, reps, seed, terms=TERMS):
    rng = np.random.default_rng(seed)
    ids = df["respondent_id"].unique()
    by_id = {rid: g for rid, g in df.groupby("respondent_id")}
    out = []
    for _ in range(reps):
        draw = rng.choice(ids, size=len(ids), replace=True)
        sample = pd.concat([by_id[r] for r in draw], ignore_index=True)
        try:
            X = design(sample, terms)
            res = sm.Logit(sample["choice"].astype(float), X).fit(disp=0)
            out.append(wtw(res.params))
        except Exception:
            continue
    return np.array(out)


def report(df, label, reps, seed):
    print(f"\n===== {label} =====")
    print(f"respondents: {df['respondent_id'].nunique()}  observations: {len(df)}")
    res = fit(df)
    table = pd.DataFrame(
        {
            "coefficient": res.params,
            "std_error": res.bse,
            "z": res.tvalues,
            "p_value": res.pvalues,
            "ci_low": res.conf_int()[0],
            "ci_high": res.conf_int()[1],
        }
    )
    print(table.to_string(float_format=lambda v: f"{v:0.4f}"))
    point = wtw(res.params)
    boot = bootstrap_wtw(df, reps, seed)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(
        f"WTW(packed vs standing) = {point:0.2f} min  "
        f"95% bootstrap CI [{lo:0.2f}, {hi:0.2f}]  (reps kept: {len(boot)})"
    )
    return table, point, (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fit on 20 respondents only")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    long = pd.read_csv(DATA / "choices_long.csv")
    people = pd.read_csv(DATA / "respondents.csv")
    df = long.merge(people, on="respondent_id", how="left")

    if args.smoke:
        keep = sorted(df["respondent_id"].unique())[:20]
        df = df[df["respondent_id"].isin(keep)]
        check_rank(df)
        report(df, "SMOKE TEST (20 respondents)", reps=50, seed=args.seed)
        return

    check_rank(df)
    table, point, ci = report(df, "MAIN: all respondents", args.reps, args.seed)
    table.to_csv(OUT / "model_coefficients.csv")
    pd.DataFrame([{"wtw_packed_min": point, "ci_low": ci[0], "ci_high": ci[1]}]).to_csv(
        OUT / "wtw.csv", index=False
    )

    eligible = df[~df["trip_freq"].isin(["non-user", "<1/week", "missing"])]
    report(
        eligible, "SENSITIVITY: excludes non-users and <1/week", args.reps, args.seed
    )


if __name__ == "__main__":
    main()
