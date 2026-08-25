# Sanas RTCI Field Experiment Protocol

Статус: preregistration draft, не утверждён  
Версия: 0.1, 2026-08-26  
Target journal: Transportation Research Part C: Emerging Technologies  
Research charter: `research/RTCI_RESEARCH_CHARTER.md`

## 1. Study objective

Оценить причинный эффект показа real-time bus crowding information в Avtobys
на решение пассажира сесть в первый доступный автобус или ждать следующий.

Короткий RQ:

> How does real-time crowding information affect boarding decisions among
> Almaty bus commuters?

## 2. Primary design

Предпочтительный дизайн: randomized controlled field experiment на одном
высокочастотном маршруте Алматы.

| Элемент | Определение |
|---|---|
| Population | Eligible пользователи Avtobys на выбранных stops/time windows pilot route |
| Unit of randomization | Стабильный pseudonymous app user |
| Unit of analysis | User × valid boarding opportunity |
| Treatment | Обычный интерфейс + RTCI ближайшего и следующего автобуса |
| Control | Тот же интерфейс и ETA, но без RTCI |
| Primary outcome | Boarded first available bus: yes/no |
| Primary estimand | Intention-to-treat average treatment effect на probability boarding first bus |
| Study horizon | Определяется power analysis, не фиксируется как одна неделя |

Stable user assignment предпочтительнее randomization каждого открытия:
уменьшается contamination и пользователь не получает противоречивый интерфейс.

## 3. Fallback design

Если Innoforce не может делать user-level A/B, используется
cluster-randomized crossover по `stop × direction × time block` или
`route-day`.

- Последовательность control/treatment задаётся до начала, например ABBA.
- Каждый cluster получает оба режима.
- Weekday/weekend, peak/off-peak и direction балансируются.
- Standard errors cluster по единице randomization.
- Treatment нельзя включать в ответ на наблюдаемую перегрузку: это разрушит
  causal identification.

Before/after одной неделей допускается только как exploratory
quasi-experiment с matched control route и difference-in-differences.

## 4. Eligibility

### User eligibility

- пользователь открыл Avtobys в pilot geofence/time window;
- выбран pilot route или stop;
- device/app version поддерживает experiment logging;
- consent/legal basis утверждены;
- pseudonymous experiment ID создан без передачи исследователям identity.

### Valid boarding opportunity

Наблюдение включается, когда:

- пользователь находился на stop до прибытия первого автобуса;
- первый автобус реально обслуживал нужный route/direction;
- boarding был физически возможен;
- occupancy message прошёл freshness/quality gate;
- известен outcome `board first`, `wait`, `left/unknown`;
- следующий service был определён в протоколе.

Случаи denied boarding не исключаются автоматически: они отдельный outcome и
не являются voluntary willingness to wait.

## 5. Intervention

Treatment card показывает для первого и следующего service:

- ETA;
- crowding level с одинаковой шкалой;
- timestamp/freshness или понятный статус;
- `unknown`, если измерение невалидно.

Нельзя одновременно менять цветовую шкалу, ETA algorithm, notifications,
pricing или incentives: treatment должен отличаться только RTCI.

Точный UI проходит cognitive testing до field launch. Скриншоты и version hash
сохраняются как experiment artifact.

## 6. Outcomes

### Primary

`board_first = 1`, если eligible пользователь сел в первый доступный автобус;
`0`, если добровольно остался ждать следующий.

Denied boarding, уход со stop и неизвестный outcome кодируются отдельно.

### Secondary behavioural

- additional waiting seconds;
- number of skipped services;
- boarded second/later bus;
- route/departure switch;
- RTCI card opened/visible;
- reported trust/usefulness, если micro-survey утверждён.

### Secondary operational

- passenger load difference между последовательными автобусами;
- route-level load variance;
- dwell time;
- actual headway variance;
- denied boarding;
- excess waiting burden;
- RTCI availability, latency и stale rate.

## 7. Data linkage

Минимальная цепочка:

```text
experiment_assignment
  -> app_exposure
  -> occupancy_message shown
  -> vehicle_arrival / boarding opportunity
  -> boarding outcome
  -> downstream operational state
```

Общий ключ строится из pseudonymous user ID, stop/route/direction и time
window. Raw identity, phone number и raw video не входят в research dataset.

## 8. Event schemas

### Assignment event

```text
experiment_id
pseudonymous_user_id
assignment: control | treatment
assignment_time_utc
randomization_unit
randomization_version
```

### Exposure event

```text
experiment_id
pseudonymous_user_id
session_id
stop_id
route_id
direction_id
event_time_utc
screen_opened
rtci_card_rendered
assignment
ui_version
```

### Occupancy information shown

```text
message_id
vehicle_id
trip_id
prediction_for_time_utc
generated_time_utc
valid_until_utc
occupancy_score_0_1
occupancy_level_1_5
confidence
status: valid | degraded | unknown
model_version
calibration_version
```

### Boarding opportunity

```text
opportunity_id
pseudonymous_user_id
stop_id
route_id
direction_id
first_vehicle_id
next_vehicle_id
first_arrival_utc
next_eta_shown_s
first_crowding_shown
next_crowding_shown
first_crowding_measured
next_crowding_measured
service_disruption_flag
weather_join_key
```

### Outcome

```text
opportunity_id
outcome: boarded_first | waited | denied | left | unknown
boarded_vehicle_id
outcome_time_utc
outcome_source: ticket | phone_inference | self_report | aggregate
outcome_confidence
```

## 9. Randomization and masking

- Randomization seed, algorithm и allocation ratio фиксируются до запуска.
- Allocation выполняется server-side.
- Исследователи не меняют assignment вручную.
- Data analyst получает masked treatment labels до freeze cleaning rules,
  если operationally возможно.
- Sensor firmware/model/config одинаковы в treatment и control.
- RTCI visibility переключается только feature flag.

## 10. Sample size

Число участников пока **TBD**. Оно рассчитывается после survey/shadow data.

Необходимые входы:

- baseline boarding probability `p0`;
- minimum practically important effect `delta`;
- eligible opportunities per user;
- treatment exposure rate;
- outcome-observation rate;
- expected attrition;
- intracluster correlation для cluster fallback;
- доля informative states: первый crowded, следующий менее crowded.

Power calculation должен соответствовать unit of randomization и primary
analysis. Количество survey responses не заменяет количество field boarding
opportunities.

## 11. Primary analysis

Primary analysis: intention-to-treat.

Базовая модель:

```text
logit(P(board_first)) = alpha + beta * assigned_RTCI + pre-specified strata
```

`beta` является primary treatment effect. Covariates используются для
precision, не для создания идентификации:

- observed first/next crowding;
- actual headway;
- peak/off-peak;
- stop/direction strata;
- weather/service disruption;
- pre-treatment user travel frequency, если legal/available.

Repeated opportunities одного user требуют user-clustered uncertainty. При
cluster randomization uncertainty соответствует cluster assignment.

## 12. Secondary models

- mixed logit для repeated boarding choices;
- interaction `treatment × crowding contrast`;
- interaction `treatment × headway`;
- heterogeneous effects по trip urgency и ride duration, только если эти
  variables наблюдаются до outcome;
- per-protocol exposure effect как secondary, с явной оговоркой selection;
- operational time-series/model для load variance и headway effects;
- dynamic route simulation calibrated только после field estimates.

Все subgroup analyses либо pre-specified, либо labelled exploratory.

## 13. Missing data

Заранее различать:

- missing RTCI due sensor/network failure;
- `unknown` shown correctly;
- exposure not rendered;
- boarding outcome unknown;
- app closed before arrival;
- trip disruption.

Нельзя удалять failure cases, если failure является частью реальной treatment
delivery. Primary ITT сохраняет assignment. Дополнительный complete-case или
valid-information analysis маркируется sensitivity analysis.

## 14. Information accuracy

Для каждого показанного level сохраняются measured truth и error, если ground
truth доступен. Анализ отдельно оценивает:

- assignment effect;
- actual exposure effect;
- effect of correct vs incorrect RTCI;
- trust decay после ошибок;
- stale information.

Accuracy interaction не должна определяться постфактум без preregistration.

## 15. Go/no-go before treatment

Treatment не включается, пока не выполнены:

1. ethics/privacy/operator approval;
2. accepted FOV coverage;
3. frozen five-level semantics;
4. frozen sensor accuracy/availability thresholds;
5. stable UTC/time linkage;
6. boarding outcome validation;
7. successful dry-run randomization;
8. rollback and `unknown` state;
9. pre-analysis plan and stopping rule;
10. no critical safety/thermal/power issue.

## 16. Ethics and privacy

До запуска определить:

- legal controller/processor roles;
- consent or other lawful basis;
- passenger notice;
- handling of minors;
- raw-video retention;
- encryption/access audit;
- pseudonymization and deletion schedule;
- protocol for withdrawal, если применим;
- independent ethics/legal approval.

Research app получает только occupancy/status. Face recognition запрещён.

## 17. Threats to validity

- non-compliance: treatment assigned, но card не просмотрен;
- contamination между пользователями;
- inaccurate/stale occupancy;
- outcome misclassification;
- selection into Avtobys use;
- Hawthorne/novelty effect;
- route-specific generalizability;
- bus bunching creates endogenous crowding/headway;
- denied boarding mistaken for voluntary wait;
- concurrent marketing/service changes.

## 18. Preregistration checklist

Перед data unblinding заполнить TBD:

- route/stops/windows;
- treatment UI;
- assignment algorithm;
- primary outcome source;
- sample size and stopping rule;
- exclusion rules;
- technical acceptance thresholds;
- primary model;
- covariates/strata;
- missing-data strategy;
- secondary/subgroup hierarchy;
- ethics approval reference;
- repository and immutable analysis version.
