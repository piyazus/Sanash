# Survey results: stated boarding response to displayed crowding

Draft v1, 2026-09-01. Section number provisional; the manuscript currently
holds `01_introduction.md` only. No citations appear in this section by
design, as it reports measurements rather than prior work. Numbers are
reproducible from `research/survey/analysis/` and are archived in
`research/survey/outputs/`.

---

## Instrument and sample

A stated-choice instrument was administered online in Kazakh and English
between 4 and 23 February 2026 and returned 215 responses. Each respondent
faced six binary scenarios. In every scenario a bus has arrived at a stated
crowding level and the next bus, which has seats available, arrives after a
stated wait; the respondent chooses to board or to wait. The arriving bus took
two crowding levels, packed and standing room. Wait levels were 2, 3, 5, 7 and
10 minutes, and scenarios were set at 08:00, 14:00 or 18:00 to represent
morning peak, off-peak and evening peak conditions.

The six profiles are: packed at 2 minutes in peak; packed at 5 minutes in
peak; packed at 10 minutes off-peak; standing room at 5 minutes in peak;
standing room at 3 minutes off-peak; and packed at 7 minutes in peak. Scenarios
two and four are identical except for crowding, which makes their contrast a
single-attribute comparison.

The sample is a convenience sample without a sampling frame or quotas. It is
dominated by young frequent riders: 167 respondents (77.7%) fall in the 14-24
age band, 158 (73.5%) are students, and 170 (79.1%) report riding daily or
almost daily. Work or study is a trip purpose for 199 respondents (92.6%).
Gender was not collected. These characteristics bound the population to which
the estimates apply.

## Estimation

The outcome is binary, coded one for waiting and zero for boarding. Of 1290
possible scenario responses, 1247 from 209 respondents were usable; 36 blank
answers and 7 free-text answers that stated neither alternative were excluded
from estimation rather than imputed to either alternative.

The six fielded profiles yield a design matrix of rank four, which limits the
specification to four identifiable parameters. A binary logit was fitted on an
intercept, wait time in minutes, an indicator for the packed level and an
indicator for peak conditions. Standard errors are clustered by respondent
because each individual contributes up to six correlated observations.

| Term | Coefficient | Clustered SE | z | p |
|---|---|---|---|---|
| Intercept | 0.9331 | 0.1673 | 5.58 | < 0.001 |
| Wait time (min) | -0.1386 | 0.0241 | -5.76 | < 0.001 |
| Packed | 1.1027 | 0.1515 | 7.28 | < 0.001 |
| Peak | -0.5172 | 0.1111 | -4.66 | < 0.001 |

Willingness to wait is the ratio of the crowding coefficient to the absolute
value of the wait-time coefficient. Respondents accept **7.96 additional
minutes** of waiting to travel on a standing-room bus rather than a packed
one, with a 95% confidence interval of [5.89, 11.09] obtained from 1000
bootstrap replications resampling respondents. Excluding the 11 respondents
who reported riding less than weekly or not at all leaves the estimate
essentially unchanged at 7.68 minutes, interval [5.64, 10.55].

Because no scenario offered a seated arriving bus, this quantity is the value
of avoiding packed conditions relative to standing room, not relative to a
seat. It is therefore not directly comparable to crowding multipliers
estimated against a seated baseline.

## Descriptive and subgroup patterns

The share choosing to wait falls monotonically with wait time within the
packed condition, from 77.5% at 2 minutes to 68.9% at 5 minutes and 64.4% at
7 minutes. Holding wait time and time of day fixed at five minutes and morning
peak, 68.9% wait when the arriving bus is packed against 42.8% when it offers
standing room, a difference of 26.1 percentage points.

This difference varies across subgroups. Students show a gap of 33.3
percentage points and daily riders 31.5 percentage points, while respondents
in employment show 6.8 percentage points and respondents riding three to four
times a week or less show 2.6 percentage points. The employment and
low-frequency cells contain 44 and 39 respondents respectively, so these
contrasts are reported as descriptive and are not the basis of any estimate.

Stated and revealed-in-scenario tolerance are internally consistent. Across
the four packed scenarios, the share choosing to wait rises from 26.4% among
respondents who said they would not wait at all, through 54.7%, 67.0% and
87.6% for those naming 1-2, 3-5 and 6-10 minutes respectively.

## Interpretation boundary

These results describe stated intentions under hypothetical conditions. They
establish that displayed crowding is associated with a large shift in stated
boarding choice within this sample, and they provide the effect magnitude
needed to size the planned field experiment. They do not identify a causal
effect of displaying occupancy in a deployed application, and no causal
language is used here. The causal estimand remains the object of the field
experiment described in `research/FIELD_EXPERIMENT_PROTOCOL.md`.
