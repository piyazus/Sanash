"""Descriptive tables for the Results section, on the rebuilt data."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
long = pd.read_csv(DATA / "choices_long.csv")
people = pd.read_csv(DATA / "respondents.csv")
df = long.merge(people, on="respondent_id", how="left")

print("=== sample (n=%d) ===" % len(people))
for col in [
    "age_group",
    "occupation",
    "trip_freq",
    "crowding_exposure",
    "max_wait",
    "app_use",
]:
    counts = people[col].value_counts()
    pct = (100 * counts / len(people)).round(1)
    print(f"\n{col}")
    for k in counts.index:
        print(f"  {str(k)[:44]:46s} {counts[k]:4d}  {pct[k]:5.1f}%")

print("\n=== wait share by scenario ===")
tab = df.groupby("scenario").agg(
    n=("choice", "size"),
    wait=("choice", "sum"),
    wait_pct=("choice", lambda s: round(100 * s.mean(), 1)),
    wait_time=("wait_time", "first"),
    packed=("packed", "first"),
    is_peak=("is_peak", "first"),
)
print(tab.to_string())

print("\n=== scenario 2 vs 4 (both 5 min, both peak; differ only in crowding) ===")
pair = df[df["scenario"].isin([2, 4])]
for label, sub in [
    ("all", pair),
    ("student", pair[pair["occupation"] == "student"]),
    (
        "employed (full+part)",
        pair[pair["occupation"].isin(["employed_full", "employed_part"])],
    ),
    ("rides daily", pair[pair["trip_freq"] == "daily"]),
    (
        "rides 3-4/week or less",
        pair[pair["trip_freq"].isin(["3-4/week", "1-2/week", "<1/week"])],
    ),
]:
    s2 = sub[sub["scenario"] == 2]["choice"]
    s4 = sub[sub["scenario"] == 4]["choice"]
    if len(s2) == 0 or len(s4) == 0:
        continue
    gap = 100 * (s2.mean() - s4.mean())
    print(
        f"  {label:24s} n={len(s2):3d}/{len(s4):3d}  "
        f"packed={100 * s2.mean():5.1f}%  standing={100 * s4.mean():5.1f}%  "
        f"gap={gap:5.1f} pp"
    )
