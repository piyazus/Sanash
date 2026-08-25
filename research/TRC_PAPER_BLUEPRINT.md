# Transportation Research Part C Paper Blueprint

Статус: evidence plan, не manuscript  
Дата: 2026-08-26

## Working title

> From sensing to boarding: Causal effects of app-based real-time crowding
> information on bus passenger choices and service performance

## One-sentence thesis

An edge-generated RTCI intervention is causally evaluated at the individual
boarding-decision level and embedded in a behaviour-aware transport model to
measure consequences for waiting, load distribution and service reliability.

## Required contribution package

| Contribution | Minimum evidence |
|---|---|
| Technology | Independently validated occupancy measurement, latency, freshness, availability and failure states |
| Behaviour | Randomized or credible quasi-experimental effect on actual boarding |
| Model | Interpretable boarding/waiting response to crowding contrast and headway |
| System | Counterfactual impact on waiting, vehicle loads and reliability |
| Context | Almaty/Central Asia contribution supported by literature, not assumed |

## Paper structure

### 1. Introduction

- transport crowding problem;
- opportunity and risk of RTCI;
- limits of stated preferences/simulation-only evidence;
- candidate causal/app/bus/geographic gap;
- purpose and four contributions.

No claim of «first» until ResearchRabbit + database screening is complete.

### 2. Related work

Organize by synthesis themes:

1. crowding disutility and perception;
2. willingness to wait and boarding choice;
3. RTCI prediction/information quality;
4. field and behavioural experiments;
5. operational/load-distribution models;
6. sensing as an enabling measurement layer.

End with a comparison table showing design, real/stated behaviour, bus/rail,
information channel, causal identification and system outcome.

### 3. Sanas RTCI system

- occupancy definition and five-level semantics;
- sensor/camera geometry;
- inference and temporal calibration;
- confidence/unknown state;
- latency/freshness;
- Avtobys message/UI;
- privacy and failure handling.

Model architecture details appear only insofar as they affect information
quality and reproducibility.

### 4. Measurement validation

- collection and manual ground truth;
- trip/day/bus split;
- accuracy and calibration metrics;
- day/night/dense/occlusion performance;
- latency/availability;
- pre-specified intervention gate.

### 5. Behavioural instrument and survey

- factorial choice attributes;
- sampling;
- cognitive pilot;
- mixed-logit/latent-class model;
- how results informed intervention and power;
- hypothetical-bias limitation.

Survey is not presented as causal field evidence.

### 6. Field experiment

- population, route and eligibility;
- randomization/fallback design;
- treatment/control UI;
- exposure and outcome measurement;
- primary estimand;
- power/stopping;
- ethics/privacy;
- missing data and attrition.

### 7. Behavioural results

- CONSORT-style flow adapted to app experiment;
- balance and treatment delivery;
- primary ITT effect with uncertainty;
- waiting-time effect;
- pre-specified interactions;
- information error/trust;
- robustness/sensitivity.

### 8. Transport-system model

- route dynamics and calibration;
- field-estimated boarding response;
- counterfactual adoption/accuracy/headway scenarios;
- load distribution;
- waiting burden;
- denied boarding;
- dwell/headway reliability;
- failure or adverse-effect scenarios.

### 9. Discussion

- causal interpretation boundary;
- why results differ/agree with prior RTCI work;
- technical accuracy versus behavioural trust;
- operator and app design implications;
- external validity beyond route/Almaty;
- privacy and equity;
- limitations separated into measurement, identification, content and scale.

### 10. Conclusion

Answer RQ with effect and uncertainty, not general promises. Separate observed
pilot result from simulated rollout.

## Planned tables

1. Related-work evidence/design comparison.
2. Dataset and sample characteristics.
3. Sensor performance by operating condition.
4. Treatment/control balance and delivery.
5. Primary/secondary causal estimates.
6. Counterfactual system outcomes.

## Planned figures

1. Closed-loop system: occupancy -> information -> behaviour -> future load.
2. Experimental timeline and randomization.
3. Treatment/control Avtobys UI.
4. Sensor calibration/confusion by crowding level.
5. Treatment effect by crowding contrast/headway.
6. Observed and counterfactual load/headway distributions.

No figure is built before its verified data source exists.

## Artifacts required for reproducibility

- preregistration timestamp/version;
- survey instrument and scenario generator;
- randomization code/seed handling;
- UI screenshots/version;
- sensor calibration/model hashes;
- de-identified analysis dataset/data dictionary;
- cleaning and analysis code;
- simulation code/config;
- environment/dependency lock;
- ethics/privacy statement;
- limitations and data-access statement.

## Desk-rejection risks

- survey-only contribution;
- technology benchmark without transport implication;
- before/after causal overclaim;
- no observed boarding outcome;
- inaccurate or stale RTCI not modelled;
- one bus carrying information only about itself while choice concerns next bus;
- novelty based only on Almaty geography;
- model complexity without comparison;
- operational simulation not calibrated to field behaviour;
- no generalizable method beyond one route;
- hidden privacy or data-access limitations.

## Evidence gates before writing prose

1. Literature gap verified.
2. Survey and field RQ aligned.
3. Target/levels frozen.
4. Sensor gate passed.
5. Boarding outcome validated.
6. Field design preregistered.
7. Primary analysis frozen.
8. Field data collected and quality audited.
9. Simulation calibrated/validated.
10. Only then draft Results, Discussion and final claims.
