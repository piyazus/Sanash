"""Rebuild the Sanas stated-preference dataset from the raw Google Forms export.

The previously prepared file (responses_prepared.csv) applied undocumented
recodes: it relabelled the fielded 14-24 age band as 18-24, folded non-users of
public buses into the 1-2/week category, imputed blanks, and coded missing and
free-text scenario answers as "board". This script rebuilds the choice data
directly from the raw export with every mapping stated explicitly.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "responses.csv"
OUT = ROOT / "data"

# Scenario attributes read off the fielded question wording (columns 4-9).
# crowding: 1 = standing room only, 2 = packed tight.
SCENARIOS = {
    4: {"scenario": 1, "wait_time": 2, "packed": 1, "is_peak": 1},
    5: {"scenario": 2, "wait_time": 5, "packed": 1, "is_peak": 1},
    6: {"scenario": 3, "wait_time": 10, "packed": 1, "is_peak": 0},
    7: {"scenario": 4, "wait_time": 5, "packed": 0, "is_peak": 1},
    8: {"scenario": 5, "wait_time": 3, "packed": 0, "is_peak": 0},
    9: {"scenario": 6, "wait_time": 7, "packed": 1, "is_peak": 1},
}

TRIP_FREQ = {
    "Күн сайын немесе дерлік күн сайын": "daily",
    "Аптасына 3–4 рет": "3-4/week",
    "Аптасына 1–2 рет": "1-2/week",
    "Аптасына бір реттен сирек": "<1/week",
    "Қоғамдық көлікті пайдаланбаймын": "non-user",
}
AGE = {
    "14–24 жас": "14-24",
    "25–34 жас": "25-34",
    "35–44 жас": "35-44",
    "45–54 жас": "45-54",
    "55 жастан жоғары": "55+",
    "Жауап бергім келмейді": "refused",
}
OCCUPATION = {
    "Студентпін": "student",
    "Толық толық жұмыс күні бойынша істеймін": "employed_full",
    "Толық емес жұмыс күн графигі бойынша жұмыс істеймін": "employed_part",
    "Жұмыссызбын немесе зейнеткермін": "unemployed_retired",
    "Жауап бергім келмейді": "refused",
}
MAX_WAIT = {
    "Күтпеймін, қай автобус келсе — соған мінемін": "0",
    "1–2 минут": "1-2",
    "3–5 минут": "3-5",
    "6–10 минут": "6-10",
    "10 минуттан артық": ">10",
}
APP_USE = {
    "Әрдайым, автобус күткен сайын": "always",
    "Көп жағдайда": "most",
    "Кейде, жағдайға байланысты": "sometimes",
    "Сирек": "rarely",
    "Мүлде пайдаланбас едім": "never",
}


def code(value, table):
    """Map a raw Google Forms answer to a short label, else flag it.

    Whitespace is normalised because some option strings in the export carry
    double spaces.
    """
    stem = " ".join(value.split("/")[0].split())
    if not stem:
        return "missing"
    return table.get(stem, "other")


def classify_choice(value):
    """1 = wait for the next bus, 0 = board the arriving bus, None = neither.

    Blank answers and free-text 'Other' responses are not decisions and are
    excluded from estimation rather than being coded as boarding.
    """
    if not value.strip():
        return None
    if "Келесі автобус" in value or "Wait" in value:
        return 1
    if "мінемін" in value or "Board" in value:
        return 0
    return None


def main():
    with RAW.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    data = rows[1:]

    respondents, long, dropped = [], [], {"blank": 0, "free_text": 0}
    for idx, row in enumerate(data, start=1):
        rid = f"R{idx:03d}"
        respondents.append(
            {
                "respondent_id": rid,
                "timestamp": row[0],
                "trip_freq": code(row[1], TRIP_FREQ),
                "crowding_exposure": row[3].split("/")[0].strip() or "missing",
                "max_wait": code(row[10], MAX_WAIT),
                "app_use": code(row[11], APP_USE),
                "age_group": code(row[12], AGE),
                "occupation": code(row[13], OCCUPATION),
            }
        )
        for col, attrs in SCENARIOS.items():
            choice = classify_choice(row[col])
            if choice is None:
                key = "blank" if not row[col].strip() else "free_text"
                dropped[key] += 1
                continue
            long.append({"respondent_id": rid, "choice": choice, **attrs})

    out = OUT
    with (out / "respondents.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(respondents[0]))
        w.writeheader()
        w.writerows(respondents)
    with (out / "choices_long.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(long[0]))
        w.writeheader()
        w.writerows(long)

    print(f"respondents: {len(respondents)}")
    print(f"choice observations kept: {len(long)} of {len(data) * 6}")
    print(
        f"dropped blank: {dropped['blank']}, dropped free-text: {dropped['free_text']}"
    )


if __name__ == "__main__":
    main()
