# Sanas RTCI Research Charter

Статус: рабочая основа для обсуждения, не preregistration  
Дата: 2026-08-26  
Владелец решения: Дияс Тлеукин

## 1. Научная цель

Sanas исследует не компьютерное зрение само по себе. CV-устройство является
измерительным слоем для причинного вопроса: меняет ли достоверная информация о
загруженности автобуса реальное решение пассажира о посадке.

### Основной research question

Короткая формулировка по требованию курса (12 слов):

> **How does real-time crowding information affect boarding decisions among
> Almaty bus commuters?**

Она используется как readable RQ. Для design, preregistration и анализа нужна
следующая operational causal formulation:

> **To what extent does displaying real-time onboard crowding information in
> the Avtobys app causally affect the probability that Almaty commuters board
> the first arriving bus rather than wait for a subsequent service?**

Для field study population нужно сузить:

> Among Avtobys users waiting at selected stops on a high-frequency Almaty bus
> route, what is the causal effect of displaying real-time crowding information
> on the probability of boarding the first arriving bus rather than waiting for
> the next service?

Вторая формулировка сильнее: в ней определены population, intervention,
comparison и observable outcome. Результат такого пилота нельзя автоматически
обобщать на всех жителей Алматы.

Курс рекомендует 10-15 слов, но это учебное правило ясности, не универсальное
требование TR-C. Нельзя сокращать RQ ценой удаления treatment, population,
comparison или outcome из operational protocol.

### Target journal

**Решено 2026-08-26:** целевой журнал — *Transportation Research Part C:
Emerging Technologies* (TR-C), а не *Transportation Engineering*.

TR-C рассматривает development, application и implications emerging
technologies in transportation systems, причём интерес журнала направлен не на
изолированную технологию, а на её конечный транспортный эффект:
https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies/about/insights

Следовательно, submission thesis должна быть шире исходного survey:

> An edge-generated, app-delivered RTCI intervention is causally evaluated at
> the boarding-decision level and embedded in a behaviour-aware model to
> quantify its effects on passenger waiting, vehicle load distribution and bus
> service reliability.

Working title:

> **From sensing to boarding: Causal effects of app-based real-time crowding
> information on bus passenger choices and service performance**

### Part C publication threshold

Для journal fit нужны четыре связанные части:

1. **Technology:** воспроизводимый sensing/information pipeline с измеренными
   accuracy, latency, freshness и failure states.
2. **Causal behaviour:** field randomization или сильный quasi-experiment,
   observation of actual boarding, а не только stated intention.
3. **Behavioural model:** boarding/waiting model, учитывающий crowding contrast,
   headway, urgency, adoption и information accuracy.
4. **Transport implications:** влияние на waiting burden, load variance,
   denied boarding, dwell/headway reliability и counterfactual rollout.

Survey-only, CV-benchmark-only или one-week before/after paper не соответствует
целевой планке. Survey остаётся instrument-development и prior/calibration
stage. Field estimates желательно встроить в dynamic route simulation или
другой validated counterfactual model, чтобы показать переносимый системный
результат, а не только локальный процент пассажиров.

Прямые precedents в TR-C подтверждают fit, но одновременно задают novelty bar:

- personalized predictive RTCI from automated data sources:
  https://doi.org/10.1016/j.trc.2020.102647
- predictive decision-support platform with endogenous passenger response:
  https://doi.org/10.1016/j.trc.2021.103139
- information and public-transport capacity interactions:
  https://doi.org/10.1016/j.trc.2016.05.007

Sanas не должен повторять эти работы. Его кандидатный новый вклад: causal
app-level field evidence for bus boarding, связанное с реально измеренной
occupancy и проверкой closed-loop operational consequences в Алматы.

### Candidate research gap

Существующая литература показывает stated willingness to wait, smart-card
revealed preferences, simulation и отдельные RTCI pilots. Кандидатный gap:

> Direct causal field evidence on how app-displayed bus crowding information
> changes individual boarding decisions is limited, especially in bus systems
> and Central Asian cities.

Это пока **кандидатный**, а не доказанный gap. Его нужно подтвердить
воспроизводимым literature review и citation chaining.

## 2. Что именно является вкладом

Главный вклад не должен формулироваться как «мы добавили crowding information»:
такая информация и WTW уже изучались.

Потенциальный оригинальный вклад состоит из четырёх частей:

1. **Geographical contribution:** первые локальные causal estimates для
   автобусных пассажиров Алматы/Центральной Азии, если обзор подтвердит
   отсутствие аналогов.
2. **Methodological contribution:** переход от stated-preference WTW к
   randomized или квазирандомизированному revealed boarding behaviour в
   мобильном приложении.
3. **Measurement contribution:** связка независимо валидированного onboard
   occupancy sensor с timestamped exposure в Avtobys и фактом посадки.
4. **Operational contribution:** оценка не только индивидуальной WTW, но и
   изменения load variance, bunching, denied boarding и waiting burden.

Ни один пункт нельзя заявлять как выполненный до получения данных.

## 3. Теоретическая основа

### Что уже существует

- Drabicki et al. формализуют **willingness to wait (WTW)** как выбор между
  более загруженным первым отправлением и ожиданием менее загруженного.
  Их stated-preference experiment относится к bus/tram и использует discrete
  choice models: https://doi.org/10.1016/j.rtbm.2023.100963
- Drabicki, Kucharski and Cats переносят WTW в dynamic public-transport
  simulation для bus bunching:
  https://doi.org/10.1007/s11116-022-10270-3
- Bansal, Hörcher and Graham оценивают crowding disutility и динамические
  правила выбора по smart-card и vehicle-location данным. Это важная
  revealed-preference основа, но не авторство WTW construct:
  https://doi.org/10.1111/rssa.12804
- Kim, Lee and Oh анализировали выбор первого или второго автобуса по недельным
  interview data в Seoul:
  https://www.worldtransitresearch.info/research/8/
- Revealed active boarding delay также оценивался по smart-card и timetable
  данным в metro systems:
  https://doi.org/10.1016/j.tra.2023.103747
- Stockholm pilot является важным field precedent для RTCI, но относится к
  platform/car distribution, а не напрямую к выбору первого автобуса:
  https://trid.trb.org/View/1483178

### Концептуальная модель

Для пассажира `i`, ожидающего departure `j`, решение можно представить как:

`BoardNow = f(current crowding, expected next crowding, headway, travel-time
urgency, trip purpose, weather, trust, information exposure, personal traits)`.

RTCI treatment должен менять информацию, а не фактическое предложение. Поэтому
нужно отдельно измерять:

- реальную загруженность первого и следующего автобуса;
- что именно показало приложение;
- freshness/ошибку информации;
- увидел ли пользователь карточку;
- факт посадки или ожидания;
- фактическое дополнительное ожидание.

## 4. Outcomes и hypotheses

### Primary outcome

`Y = 1`, если пассажир сел в первый доступный автобус; `Y = 0`, если сознательно
пропустил его и остался ждать.

Primary estimand:

> Intention-to-treat difference in the probability of boarding the first
> available bus between users assigned RTCI and users assigned ETA-only
> information, conditional on a valid boarding opportunity.

### Secondary outcomes

- фактические дополнительные минуты ожидания;
- число пропущенных отправлений;
- load difference между последовательными автобусами;
- variance загрузки внутри маршрута;
- dwell time и headway irregularity;
- denied boarding;
- perceived usefulness, trust и accuracy perception;
- app adoption/exposure rate;
- heterogeneity по trip urgency, expected ride duration, crowding difference,
  headway, возрасту и mobility constraints.

### Hypotheses

- **H1:** RTCI уменьшает вероятность посадки в первый автобус, когда он сильно
  загружен, а следующий ожидается менее загруженным.
- **H2:** эффект возрастает с разницей загруженности между первым и следующим
  автобусом.
- **H3:** эффект уменьшается при большем headway и высокой срочности поездки.
- **H4:** RTCI уменьшает load variance между последовательными автобусами, но
  может увеличить индивидуальное waiting time.
- **H5:** ошибка или устаревание RTCI снижает доверие и treatment compliance.

H2-H5 должны считаться secondary/exploratory, если power analysis не позволяет
полноценно тестировать их отдельно.

## 5. Трёхэтапная методология

### Method classification and course grounding

Работа является quantitative empirical multi-study research:

1. stated-preference survey для instrument development и behavioural priors;
2. measurement-validation study для occupancy technology;
3. randomized или квазирандомизированный field experiment;
4. behaviour-aware transport model/counterfactual analysis.

Она также содержит structured literature review, но не является
literature-review-only paper.

`research/coursework/RESEARCH_METHOD_GROUNDING.md` подтверждает применимые
требования курса: association не равна causation, methodology должна позволять
повторение, sampling/instrument нужно описывать явно, systematic и content
limitations разделяются. Курс подробно раскрывает survey и review, но не даёт
достаточной causal field-experiment methodology. Поэтому survey template нельзя
копировать как весь methods section будущей статьи.

Если field identification не является randomized/credible quasi-experimental,
в результатах разрешена только формулировка `associated with`, а не `caused`.
Численные thresholds, sample size и ethics process остаются открыты.

### Study A: stated-preference survey

Существующий survey нужно не просто «масштабировать», а сначала провести audit.
Его файла в текущем репозитории не найдено.

Рекомендуемый дизайн:

- factorial choice tasks: crowdedness first bus, crowdedness next bus,
  additional wait, trip urgency/purpose, expected ride duration;
- выбор `board now` / `wait`;
- 6-10 randomized scenarios на респондента, а не один общий вопрос;
- понятные пассажиру crowding icons, предварительно cognitive-tested;
- demographics, обычный маршрут, frequency of Avtobys use, mobility limits,
  safety/comfort attitude и trust in information;
- небольшой open-ended вопрос о причине решения.

Анализ: mixed logit или latent-class logit с respondent-level clustered
uncertainty. Обычный процент ответивших «да» недостаточен из-за повторных
choice tasks и heterogeneity.

Study A даёт stated WTW и параметры для power analysis, но не отвечает на
causal field question из-за hypothetical bias.

### Study B: measurement validation

До показа RTCI устройство работает в silent mode. Для каждого автобуса:

1. Sensor estimate и passenger-facing level записываются с timestamp.
2. Независимый manual ground truth собирается по заранее написанному labeling
   protocol.
3. Проверяются accuracy, calibration, missingness, latency и data freshness.
4. Устанавливается go/no-go threshold до просмотра field-treatment effect.

Пять уровней должны определяться через operational capacity/comfort semantics,
а не через произвольные интервалы model score. Для causal experiment особенно
опасна differential measurement error: treatment нельзя показывать неточно и
затем интерпретировать недоверие как отсутствие behavioural effect.

### Study C: field experiment

#### Предпочтительный вариант: app-level randomized A/B

- Один высокочастотный pilot route и заранее выбранные stops/time windows.
- Eligible Avtobys users рандомизируются в стабильные группы.
- Control видит ETA и обычную информацию.
- Treatment видит то же плюс RTCI для ближайших отправлений.
- Assignment, card impression, occupancy snapshot и boarding outcome имеют
  совместимые pseudonymous IDs/timestamps.
- Primary analysis: intention-to-treat; per-protocol exposure analysis только
  secondary.

Это позволяет сравнивать людей в одно время, в одной погоде и при одном
service state.

#### Если individual A/B технически невозможен

Использовать cluster-randomized crossover по stop x time block или route-day:

- несколько чередований control/treatment, например сбалансированный ABBA;
- одинаковое покрытие weekday/weekend и peak/off-peak;
- assignment публикуется заранее;
- standard errors cluster по единице randomization;
- treatment status нельзя выбирать после наблюдения crowding.

Простой дизайн «неделя до / неделя после» остаётся fallback quasi-experiment.
Он уязвим для weather, weekday composition, service changes, seasonality,
marketing, novelty и bus bunching. Если другого варианта нет, нужны matched
control route, несколько pre/post periods и difference-in-differences с
parallel-trends diagnostics.

## 6. Как наблюдать посадку

Это главный operational blocker. Возможные источники в порядке силы:

1. **Pseudonymous app-to-ticket linkage:** app exposure связывается с реальной
   validation/boarding transaction с privacy review.
2. **Opt-in smartphone inference:** geofenced waiting session + движение вместе
   с конкретным bus GPS после departure; требует отдельной validation и consent.
3. **In-app immediate confirmation:** «Сели в этот автобус / решили ждать»;
   проще, но self-report и non-response bias.
4. **Aggregate APC boarding counts:** хорошо для route-level load effects, но
   не доказывает individual effect среди пользователей, увидевших RTCI.

До выбора наблюдаемого outcome нельзя делать power analysis или обещать causal
field study.

## 7. Масштаб пилота

Оповещать весь город для первого исследования не требуется. Нужны:

- один маршрут с коротким headway, заметной вариацией загрузки и несколькими
  последовательными оборудованными автобусами;
- ограниченный список stops/time windows;
- app banner только для eligible audience;
- прозрачное описание pilot, privacy и возможной погрешности;
- coordination с Innoforce и оператором маршрута.

Для meaningful `wait for next bus` intervention недостаточно одного
оборудованного автобуса. Нужно знать или честно прогнозировать загрузку как
минимум первого и следующего отправления. Следовательно, потребуется несколько
последовательных оборудованных машин либо другой валидный источник occupancy.

## 8. Power и analysis plan

До расчёта sample size нужны:

- baseline probability of boarding first bus;
- ожидаемый minimum detectable effect;
- treatment exposure/adoption rate;
- число eligible decisions в день;
- intracluster correlation для cluster design;
- доля ситуаций, где первый автобус crowded, а следующий реально лучше.

Модель primary outcome: logistic regression с pre-specified treatment effect и
design-consistent standard errors. Добавление crowding/headway covariates
повышает precision, но не должно заменять рандомизацию.

Обязательно заранее зафиксировать:

- eligibility и valid boarding opportunity;
- handling missed sensor data;
- exclusion rules;
- primary estimand/model;
- multiple-testing policy;
- stopping rule;
- subgroup analyses;
- privacy/consent;
- preregistration до просмотра treatment outcomes.

## 9. Literature review plan

ResearchRabbit полезен для citation discovery и визуальной карты, но не
заменяет воспроизводимый database search.

### Review question

> What is known about the effect of real-time public-transport crowding
> information on passengers' boarding, waiting, route, departure-time and
> within-vehicle distribution decisions?

### Sources

- Scopus / Web of Science;
- TRID;
- Google Scholar for citation chaining;
- Transport Research International Documentation;
- IEEE Xplore для sensing/information systems;
- publisher/full-text repositories for verification.

### Search concept blocks

1. `real-time crowding information OR occupancy information OR passenger load
   information`
2. `bus OR public transport OR transit OR metro OR tram OR rail`
3. `boarding decision OR willingness to wait OR departure choice OR route
   choice OR passenger behaviour`

Все фактически использованные строки, базы, даты и result counts записываются.

### Inclusion

- peer-reviewed empirical, experimental, behavioural modelling or validated
  simulation studies;
- RTCI/occupancy information является intervention/input;
- измеряется passenger decision или operational consequence;
- urban public transport;
- English плюс доступные Russian/Kazakh studies при проверяемом full text.

### Evidence matrix

Для каждой статьи: location, mode, sample, design, stated/revealed preference,
information channel, crowding scale, outcome, identification strategy, model,
effect, limitation, verification status, DOI/full text.

Синтезировать по design и decision type, а не «один абзац на автора».
ResearchRabbit collection должна начинаться с verified seed papers выше, затем
backward/forward citation chaining. PRISMA flow отражает только реально
выполненный поиск.

## 10. Три разных dataset, которые нельзя путать

### A. Behavioural research dataset

Unit: passenger x boarding opportunity. Минимальные поля:

- assignment, exposure, timestamp, route/stop/vehicle;
- first/next ETA and actual headway;
- shown and measured crowding for first/next bus;
- boarded first / waited / left;
- waiting duration;
- trip urgency/purpose when consent permits;
- weather/service disruption;
- sensor freshness/error flags.

Такого Almaty dataset в интернете не будет: это основной новый эмпирический
результат исследования.

### B. Sensor-training dataset

Frames/clips from the actual camera geometry + independently labelled cabin
occupancy/capacity level. Публичные crowd datasets могут дать pretraining, но
не заменить Almaty pilot data.

### C. Operational dataset

Vehicle GPS/AVL, schedule/ETA, stop events, headways, ticket validations/APC,
vehicle capacity and disruptions. Его доступность нужно согласовать с
Innoforce/operator; он важнее очередного интернет-датасета для causal design.

## 11. Реалистичный план на ближайший месяц

> **Заменено 2026-09-04.** Действующий график работ на сентябрь находится в
> [`../SEPTEMBER_PLAN.md`](../SEPTEMBER_PLAN.md). Этот раздел сохранён как
> описание четырёх фаз подготовки полевого эксперимента; его недельные сроки
> больше не действуют. Charter остаётся авторитетом по научному дизайну.


### Week 1: freeze the study

- получить и проверить существующий survey;
- подтвердить research question и primary outcome;
- провести structured literature search;
- meeting с Innoforce по A/B, exposure logs, boarding outcome и route data;
- выбрать pilot route по заранее заданным критериям.

### Week 2: pilot instruments

- cognitive interviews по crowding icons и survey wording;
- survey pilot;
- camera/FOV bench test;
- draft labeling guide и data schema;
- privacy/consent review.

### Week 3: silent measurement

- несколько сенсоров на последовательных автобусах или честный reduced bench;
- manual ground-truth validation;
- survey launch;
- app RTCI prototype без влияния на реальных пассажиров.

### Week 4: go/no-go package

- survey cleaning и preliminary choice model;
- sensor accuracy/latency report;
- power calculation по реальному eligible-decision rate;
- preregistered field protocol;
- письменное go/no-go решение для live A/B.

Полноценный causal field result за один месяц маловероятен, если сейчас нет
app logging, boarding linkage и валидированного сенсора. Реалистичный результат
месяца: готовый, этически и технически исполнимый field experiment плюс
предварительные survey/sensor данные.

## 12. Решения, которые нужны от Дияса и Innoforce

1. Один route pilot или city-wide rollout: для исследования рекомендован route
   pilot.
2. Может ли Avtobys делать randomized feature exposure?
3. Можно ли наблюдать boarding outcome без нарушения privacy?
4. Есть ли доступ к vehicle IDs, GPS/ETA, route capacity и ticket/APC events?
5. Сколько последовательных автобусов можно оборудовать?
6. Какой minimum effect practically matters Innoforce/operator?
7. Кто даёт разрешение на съёмку, consent и обработку данных?
8. Где находится текущая версия survey и на каких языках она работает?
