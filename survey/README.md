# Sanash Survey — Discrete Choice Experiment

## Overview

This directory contains the DCE (Discrete Choice Experiment) survey instrument and analysis
pipeline for the Sanash bus occupancy project. The survey measures how real-time crowding
information affects commuter boarding decisions in Almaty, Kazakhstan.

## Methodology

### Design: Discrete Choice Experiment (DCE)

DCE is a stated-preference technique in which respondents choose between hypothetical
alternatives that vary systematically across attributes. This approach reveals implicit
trade-offs that are difficult to observe in real travel data.

**Utility model:**
```
U(wait) = β_wait · t + β_crowding_packed · I_packed + β_crowding_standing · I_standing
        + β_peak · I_peak + ε
```

Where:
- `t` = wait time for next bus (minutes)
- `I_packed` = 1 if current bus is >80% full
- `I_standing` = 1 if current bus is 50–80% full
- `I_peak` = 1 if trip is during peak hours (7–9am or 5–7pm)
- `β` coefficients estimated via multinomial logit regression

**Willingness-to-Wait (WTW):**
```
WTW_packed   = β_crowding_packed / |β_wait|   (minutes)
WTW_standing = β_crowding_standing / |β_wait| (minutes)
```

WTW represents the additional waiting time a commuter accepts to avoid a given crowding
level — the primary behavioural metric in this study.

### Survey Design

| Parameter | Value |
|-----------|-------|
| Instrument | 13-question questionnaire |
| Demographic questions | 5 (Q1–Q5) |
| DCE scenarios | 6 (Q6–Q11) |
| Attitudinal questions | 2 (Q12–Q13) |
| Scenario attributes | Crowding level × Wait time × Time of day |
| Design type | Orthogonal fractional factorial (6 of 24 full factorial) |

**Scenario attribute levels:**

| Attribute | Levels |
|-----------|--------|
| Crowding | Seats available (<50%), Standing room (50–80%), Packed (>80%) |
| Wait time | 2, 5, 10, 15 minutes |
| Time of day | Peak (7–9am / 5–7pm), Off-peak |

**6 orthogonally balanced scenarios:**

| Scenario | Crowding | Wait | Time |
|----------|----------|------|------|
| 1 | Packed | 2 min | Peak |
| 2 | Standing room | 5 min | Off-peak |
| 3 | Seats available | 10 min | Peak |
| 4 | Packed | 15 min | Off-peak |
| 5 | Seats available | 2 min | Off-peak |
| 6 | Standing room | 10 min | Peak |

### Sample

| Parameter | Value |
|-----------|-------|
| Total respondents | n = 167 |
| Inclusion criteria | Almaty residents, age ≥ 18, use public buses ≥ once/week |
| Exclusion criteria | Non-Almaty residents; taxi/private car only users |
| Collection method | Google Forms (online), March 2025 |
| Languages | Kazakh, Russian, English |
| Sampling method | Convenience sample at major bus stops + university campuses |

## File Structure

```
survey/
├── instrument/
│   ├── survey_questions_en.md    # English version (13 questions)
│   ├── survey_questions_ru.md    # Russian version
│   └── survey_questions_kz.md   # Kazakh version
├── data/
│   └── responses.csv             # (not tracked in git — add your CSV here)
├── analysis/
│   ├── survey_analysis.py        # Main analysis script
│   └── requirements.txt          # Python dependencies
├── output/                       # Generated figures and tables (gitignored)
└── README.md                     # This file
```

## Running the Analysis

### Install dependencies
```bash
pip install -r survey/analysis/requirements.txt
```

### Run with real data
```bash
python survey/analysis/survey_analysis.py --input survey/data/responses.csv
```

### Run with synthetic demo data (no CSV needed)
```bash
python survey/analysis/survey_analysis.py
```

### Output files
After running, `survey/output/` will contain:
- `demographics.png` — Age, gender, employment, trip frequency charts
- `wtw_chart.png` — Willingness-to-Wait bar chart by crowding level
- `choice_proportions.png` — Board vs wait choice proportions per scenario
- `summary_stats.csv` — Demographic frequency tables
- `model_coefficients.csv` — MNL regression coefficients with standard errors
- `wtw_results.csv` — WTW values in minutes

## Expected Results

Based on comparable studies (Wardman & Whelan, 2011; Li et al., 2017):
- WTW_packed: 8–15 minutes
- WTW_standing: 3–7 minutes
- Peak time penalty: 2–5 minutes

## References

- Ben-Akiva, M., & Lerman, S. R. (1985). *Discrete Choice Analysis*. MIT Press.
- Hensher, D. A., Rose, J. M., & Greene, W. H. (2005). *Applied Choice Analysis*. Cambridge.
- Wardman, M., & Whelan, G. (2011). Twenty years of rail crowding valuation. *Transport Reviews*, 31(3).
