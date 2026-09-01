# Stated-preference survey: boarding versus waiting under crowding

Status of this directory: **verified data, estimated model.** It is not a field
experiment and carries no causal claim. See `GROUND_TRUTH.md` section 3.1 for how
this fits the RTCI research track.

## What was collected

| | |
|---|---|
| Respondents | 215 |
| Collection window | 2026-02-04 to 2026-02-23 |
| Instrument | Google Forms, bilingual Kazakh/English, 13 questions |
| Sampling | Convenience sample. No sampling frame, no quotas, not population-representative |
| Personal data | None. The export holds a timestamp and closed-choice answers only |
| Raw file | `data/raw/responses.csv`, md5 `7f81731ab526613883c21d021de56f00` |

Six of the 13 questions are binary choice scenarios; the rest cover trip
frequency, trip purpose, crowding exposure, maximum acceptable wait, willingness
to use a crowding app, age band and occupation.

## Design as fielded

Each scenario names the crowding of the bus that has just arrived, the wait for
the next bus, and the time of day. The next bus always has seats.

| Scenario | Crowding of arriving bus | Wait for next | Time of day |
|---|---|---|---|
| 1 | packed | 2 min | Wed 08:00, peak |
| 2 | packed | 5 min | Wed 08:00, peak |
| 3 | packed | 10 min | Wed 14:00, off-peak |
| 4 | standing room | 5 min | Wed 08:00, peak |
| 5 | standing room | 3 min | Wed 14:00, off-peak |
| 6 | packed | 7 min | Wed 18:00, peak |

The arriving bus takes **two** crowding levels, packed and standing room. Wait
levels are 2, 3, 5, 7 and 10 minutes. Scenarios 2 and 4 differ only in crowding,
which makes their contrast a single-attribute comparison.

Six profiles support four identifiable parameters (design matrix rank 4).
`analysis/fit_survey.py` checks the rank before fitting and refuses to run an
over-specified model.

## Results

Binary logit, outcome 1 = wait for the next bus, standard errors clustered by
respondent, 1247 of 1290 possible choice observations from 209 respondents.

| Term | Coefficient | Clustered SE | p |
|---|---|---|---|
| const | 0.9331 | 0.1673 | <0.001 |
| wait_time | -0.1386 | 0.0241 | <0.001 |
| packed | 1.1027 | 0.1515 | <0.001 |
| is_peak | -0.5172 | 0.1111 | <0.001 |

Willingness to wait to avoid a packed bus rather than a standing-room bus:
**7.96 minutes, 95% bootstrap CI [5.89, 11.09]** (1000 replications resampling
respondents). Excluding the 11 respondents who reported using buses less than
weekly or not at all: 7.68 minutes, CI [5.64, 10.55].

This is packed versus standing room. It is not a value relative to a seated
trip, because no scenario offered a seated arriving bus.

## Known limitations

- 77.7% of respondents are in the 14-24 age band and 73.5% are students, so the
  sample describes young frequent riders, not Almaty commuters generally.
- The age band as fielded starts at 14, so respondents may include minors.
- Gender was not asked.
- 43 of 1290 scenario answers were blank or free text and are excluded from
  estimation rather than imputed.
- Stated preference measures intention, not behaviour. The RTCI field experiment
  remains the only planned source of a causal estimate.

## Reproducing

```
python analysis/build_instrument.py   # instrument/fielded_instrument.md from the export
python analysis/rebuild_survey.py     # data/respondents.csv, data/choices_long.csv
python analysis/fit_survey.py --smoke # 20-respondent smoke test
python analysis/fit_survey.py         # outputs/model_coefficients.csv, outputs/wtw.csv
python analysis/descriptives.py       # sample and scenario tables
```

Requires pandas, numpy and statsmodels.

## Superseded material on `origin/main`

`origin/main` is an unrelated Git history that still carries `survey/README.md`,
`docs/RESEARCH_GUIDE.md` and `survey/instrument/survey_questions_{en,ru,kz}.md`.
Those files state a sample of n=167 collected in March 2025, three crowding
levels and a 15-minute wait level, and they describe a gender question. None of
that matches the export. The n=167 figure is the default of the synthetic data
generator in `survey/analysis/survey_analysis.py` on that branch. Do not cite
those files.
