# Sanas RTCI systematic literature review protocol

Статус: **исполняемый протокол, не отчёт о выполненном обзоре**
Дата составления: 2026-09-02
Владелец: Danyshpan (письмо и цитаты), решение о scope за Диясом
Авторитет по дизайну исследования: `research/RTCI_RESEARCH_CHARTER.md` §9
Авторитет по текущему состоянию: `GROUND_TRUTH.md`

Этот файл разворачивает набросок из §9 чартера в процедуру, которую можно
выполнить и потом повторить. Он ничего не утверждает о результатах поиска.
**Ни одно поле со счётчиком в этом файле не заполнено, потому что описанный
здесь поиск ещё не проводился.** Все `___` заполняет тот, кто запустит поиск,
в том же изменении, в котором он записывает лог запуска.

Единственный зафиксированный поисковый прогон в репозитории на дату
составления - пилот OpenAlex от 2026-08-26
(`research/wiki/sources/litsearch-run-2026-08-26.md`). Он был smoke-тестом с
лимитом 25 записей на запрос и **не является** обзором по этому протоколу.
Его записи можно переиспользовать как входные кандидаты, но он не закрывает
ни один шаг ниже.

---

## 1. Review question

### 1.1 Question as stated in the charter

Дословно из `research/RTCI_RESEARCH_CHARTER.md` §9:

> What is known about the effect of real-time public-transport crowding
> information on passengers' boarding, waiting, route, departure-time and
> within-vehicle distribution decisions?

This wording is the authority for eligibility decisions. If a screening
decision cannot be justified against this sentence, the criterion is wrong and
must be revised in this file before screening continues.

### 1.2 Structured decomposition

The review question is not a clinical intervention question, so a strict PICO
does not apply without adaptation. It is decomposed here as PICOS, with the
context element carried explicitly because a geographic gap is one of the
candidate contributions of the paper.

| Element | Content | Role in screening |
|---|---|---|
| **P** Population | Passengers of urban and suburban public transport who face a boarding, waiting, routing or departure-time decision. Includes real riders, survey respondents answering as riders, laboratory participants and simulated agents whose behavioural parameters are documented. | Criterion I1 |
| **I** Intervention / exposure | Provision to the passenger of information describing the current or predicted crowding, occupancy or load of a specific vehicle, car or platform. Provision may be real (display, app, push) or hypothetical (described inside a stated-preference task) or assumed (input to a simulation). | Criterion I2 |
| **C** Comparison | Absence of that information, a different information format, a different crowding level shown, or a different information accuracy or freshness. Studies with no internal comparison are eligible only into the context tier, see §5.4. | Criterion I3 |
| **O** Outcome | A passenger decision (board the first arriving vehicle, skip and wait, waiting time, route, itinerary, departure time, car or platform choice, mode) or an operational consequence attributable to those decisions (load distribution, load variance, bunching, headway irregularity, dwell time, denied boarding, ridership). Perception, trust and usability outcomes are eligible and tagged separately. | Criterion I4 |
| **S** Study design | Field experiment, natural or quasi-experiment, deployment with observation, stated-preference or discrete-choice experiment, revealed-preference observational study, laboratory experiment, simulation with documented behavioural calibration, or systematic review of the above. | Criterion I5 |
| **Context** | Urban or suburban public transport. Geography, income level and network type are recorded, never used to exclude. | Recorded in the matrix, not a filter |

### 1.3 Sub-questions

Каждый sub-question отвечает за один блок related work в
`research/TRC_PAPER_BLUEPRINT.md` §2. Обзор считается выполненным, только когда
на каждый есть либо ответ, либо явно записанное отсутствие свидетельств.

- **SQ1.** Where has crowding information actually been shown to real
  passengers, and what was observed?
- **SQ2.** What do passengers state they would do, and how large is the stated
  willingness to wait?
- **SQ3.** What do models and simulations predict, and where do their
  behavioural parameters come from?
- **SQ4.** What is observed about crowding response when no information is
  provided?
- **SQ5.** How is crowding measured, scaled and displayed, and is the display
  legible to the passenger at the decision moment?
- **SQ6.** What identification strategies have been used for field studies of
  transport information, and what effect sizes did they detect?
- **SQ7.** Which geographies, populations and time periods are covered, and
  which are not?

SQ7 exists because the candidate contribution in the charter is partly
geographic. A geographic gap claim requires a documented absence, that is a
recorded search string, a recorded date and a recorded hit count, not the
absence of a memory.

---

## 2. Databases and access

Доступ - это то, что фактически подтверждено на машине Дияса, а не то, что
существует в мире. Пока колонка «доступ» не заполнена по факту, coverage
обзора не определён, и это ограничение переносится в §11.

| Source | Purpose in this review | Access confirmed | Evidence for the access status |
|---|---|---|---|
| Scopus | Primary indexed search, controlled Boolean syntax, forward citation counts | **Not confirmed** | Пилот 2026-08-26 записал: не использован, нужен институциональный доступ |
| Web of Science | Second indexed search for coverage overlap and forward chaining | **Not confirmed** | То же |
| TRID (Transport Research International Documentation) | Transport-specific coverage, reports and conference material missed by Scopus and WoS | **Not confirmed** | Пилот отнёс TRID к недоступным. Отдельные записи TRID открывались по прямой ссылке из чартера, поэтому статус нужно перепроверить и записать до запуска |
| IEEE Xplore | Sensing, displays and information-systems literature feeding the measurement layer and SQ5 | **Not confirmed** | Нет записанной проверки |
| Google Scholar | Citation chaining, grey literature spot checks, retrieval of full texts | **Not confirmed as a systematic source** | Открывается в браузере, но не даёт воспроизводимые счётчики и не имеет полевого синтаксиса |
| OpenAlex | Confirmed working programmatic source, used for the keyword pass and both chaining directions | **Confirmed 2026-08-26** | Прогон в `research/wiki/sources/litsearch-run-2026-08-26.md` |
| Crossref | Metadata verification of a single DOI, first-author and year checks | **Confirmed** | Использован 2026-08-26 и 2026-09-02 |
| Semantic Scholar | Alternative chaining source | **Blocked without an API key** | Пилот получил HTTP 429 |
| Publisher sites and repositories | Full-text retrieval and verification of every extracted finding | Per record | Записывается в матрице в поле `full_text_location` |

**Правило.** Обзор запускается на том, что подтверждено. Если Scopus и Web of
Science остаются недоступными, обзор всё равно выполним на OpenAlex плюс TRID
плюс IEEE Xplore плюс chaining, но тогда в статье он описывается именно так, а
не как «Scopus and Web of Science search». Подмена подтверждённого доступа
предполагаемым - это фальсификация методологии.

**Открытый вопрос для Дияса.** Даёт ли его институт доступ к Scopus, Web of
Science или IEEE Xplore, и через какой прокси. До ответа поля «доступ» выше
остаются `Not confirmed`.

---

## 3. Search strings

### 3.1 Concept blocks

Чартер §9 задаёт три блока. Здесь первый блок разбит на два, потому что
«real-time crowding information» как одна фраза теряет работы, где
информационный и загруженностный компоненты разнесены по тексту abstract.
Это сознательное отклонение от чартера, записанное в §13.

**Block 1, crowding / occupancy state**

crowding, crowdedness, crowded, crowd level, occupancy, occupancy level,
occupancy rate, load, passenger load, load factor, vehicle load, loading,
in-vehicle congestion, congestion, fullness, bus fullness, seat availability,
seat occupancy, standing density, passenger density, passengers per square
metre, capacity utilisation, capacity utilization, denied boarding, overcrowding

**Block 2, real-time information provision**

real-time, real time, realtime, live, current, dynamic, predictive, predicted,
forecast, forecasting, advance, up-to-date, online, on-line
combined with:
information, information provision, information system, passenger information,
traveller information, traveler information, advanced traveller information,
display, public display, signage, countdown display, app, mobile app, mobile
application, smartphone application, in-app, push notification, feedback,
decision support, journey planner

**Block 3, mode**

bus, buses, transit, public transport, public transportation, public transit,
mass transit, urban transport, metro, subway, underground, tram, streetcar,
light rail, rail, railway, urban rail, commuter rail, suburban rail, bus rapid
transit, BRT

**Block 4, decision and outcome**

boarding, board, boarding decision, alighting, willingness to wait, wait,
waiting, waiting time, wait time, departure time choice, departure-time,
route choice, itinerary choice, path choice, mode choice, travel behaviour,
travel behavior, passenger behaviour, passenger behavior, choice behaviour,
discrete choice, stated preference, revealed preference, skip, skipping,
let pass, car choice, vehicle choice, platform choice, platform placement,
passenger distribution, load balancing, demand shift, peak spreading,
self-regulation, denied boarding

### 3.2 Set logic

Одинаковая логика во всех базах, чтобы наборы были сопоставимы:

- `S1` = Block 1
- `S2` = Block 2
- `S3` = Block 3
- `S4` = Block 4
- `S5` = `S1 AND S2 AND S3` - **high-recall set**, основной набор для скрининга
- `S6` = `S5 AND S4` - **high-precision set**, используется для приоритизации
  порядка чтения, а не для сокращения `S5`
- `S7` = `S1 AND S3 AND S4 NOT S2` - **context set**, крауд-реакция без
  информационного вмешательства, попадает во второй tier по §5.4

Скринится `S5` целиком. `S6` и `S7` существуют для отчётности и для порядка
работы. Если `S5` в базе даёт объём, который один человек физически не
прочитает по title/abstract, объём **не режется молча**: применяется
предзаписанный фильтр (например, год от 2005), фильтр записывается в лог
запуска, и в §11 появляется соответствующее ограничение.

Год по умолчанию не ограничивается. Тип документа по умолчанию не
ограничивается. Любой применённый фильтр записывается.

### 3.3 Scopus

Синтаксис: `TITLE-ABS-KEY`, `W/n` для близости, `*` для усечения, кавычки для
loose phrase. Строки готовы к вставке в Advanced Search как есть.

```
S1:
TITLE-ABS-KEY ( crowd* OR crowdedness OR occupancy OR "occupancy level" OR "occupancy rate" OR "passenger load" OR "load factor" OR "vehicle load" OR "in-vehicle congestion" OR congestion OR fullness OR "seat availability" OR "seat occupancy" OR "standing density" OR "passenger density" OR "capacity utili?ation" OR "denied boarding" OR overcrowd* )
```

```
S2:
TITLE-ABS-KEY ( ( "real-time" OR "real time" OR realtime OR live OR current OR dynamic OR predict* OR forecast* OR "up-to-date" OR online OR "on-line" ) W/3 ( information OR display* OR signage OR app OR "mobile application" OR "smartphone application" OR "push notification" OR feedback OR "decision support" OR "journey planner" ) )
```

```
S3:
TITLE-ABS-KEY ( bus OR buses OR transit OR "public transport*" OR "mass transit" OR "urban transport" OR metro OR subway OR underground OR tram OR streetcar OR "light rail" OR rail OR railway OR "commuter rail" OR "bus rapid transit" OR BRT )
```

```
S4:
TITLE-ABS-KEY ( boarding OR board OR alight* OR "willingness to wait" OR "waiting time" OR "wait time" OR "departure time choice" OR "departure-time" OR "route choice" OR "itinerary choice" OR "path choice" OR "mode choice" OR "travel behavio*r" OR "passenger behavio*r" OR "choice behavio*r" OR "discrete choice" OR "stated preference" OR "revealed preference" OR skip* OR "car choice" OR "vehicle choice" OR "platform choice" OR "platform placement" OR "passenger distribution" OR "load balancing" OR "peak spreading" OR "denied boarding" )
```

```
S5:
( S1 ) AND ( S2 ) AND ( S3 )
S6:
( S5 ) AND ( S4 )
S7:
( S1 ) AND ( S3 ) AND ( S4 ) AND NOT ( S2 )
```

Комментарий по одному оператору: `capacity utili?ation` использует `?` как
подстановку одного символа в Scopus. Если база отвергнет запись, заменить на
`( "capacity utilisation" OR "capacity utilization" )` и зафиксировать замену
в логе.

### 3.4 Web of Science

Синтаксис: `TS=` (topic: title, abstract, author keywords, Keywords Plus),
`NEAR/n`, `*` для усечения, `$` для одного символа.

```
S1:
TS=( crowd* OR crowdedness OR occupancy OR "occupancy level" OR "occupancy rate" OR "passenger load" OR "load factor" OR "vehicle load" OR "in-vehicle congestion" OR congestion OR fullness OR "seat availability" OR "seat occupancy" OR "standing density" OR "passenger density" OR "capacity utili*ation" OR "denied boarding" OR overcrowd* )
```

```
S2:
TS=( ( "real-time" OR "real time" OR realtime OR live OR current OR dynamic OR predict* OR forecast* OR "up-to-date" OR online ) NEAR/3 ( information OR display* OR signage OR app OR "mobile application" OR "smartphone application" OR "push notification" OR feedback OR "decision support" OR "journey planner" ) )
```

```
S3:
TS=( bus OR buses OR transit OR "public transport*" OR "mass transit" OR "urban transport" OR metro OR subway OR underground OR tram OR streetcar OR "light rail" OR rail OR railway OR "commuter rail" OR "bus rapid transit" OR BRT )
```

```
S4:
TS=( boarding OR board OR alight* OR "willingness to wait" OR "waiting time" OR "wait time" OR "departure time choice" OR "route choice" OR "itinerary choice" OR "path choice" OR "mode choice" OR "travel behavio*r" OR "passenger behavio*r" OR "choice behavio*r" OR "discrete choice" OR "stated preference" OR "revealed preference" OR skip* OR "car choice" OR "vehicle choice" OR "platform choice" OR "platform placement" OR "passenger distribution" OR "load balancing" OR "peak spreading" OR "denied boarding" )
```

```
S5: #1 AND #2 AND #3
S6: #5 AND #4
S7: #1 AND #3 AND #4 NOT #2
```

Номера наборов в WoS присваиваются сессией, поэтому в лог записывается и
номер набора, и строка, которую он представляет.

### 3.5 TRID

TRID не имеет эквивалента `TITLE-ABS-KEY` в простом поле поиска, и поведение
его полей нужно проверить на первом запуске, а не предполагать. Порядок:

1. Открыть Advanced Search, выбрать поле, которое покрывает title и abstract.
2. Вставить строку ниже.
3. Записать в лог, какое именно поле выбрано в интерфейсе, потому что от него
   зависит воспроизводимость.

```
TRID-1 (high recall):
(crowding OR crowdedness OR occupancy OR "passenger load" OR "load factor" OR fullness OR "seat availability" OR overcrowding OR "denied boarding") AND ("real-time information" OR "real time information" OR "passenger information" OR "traveler information" OR "traveller information" OR display OR "mobile app" OR "smartphone app" OR "predictive information") AND (bus OR transit OR "public transport" OR metro OR subway OR tram OR "light rail" OR rail)
```

```
TRID-2 (decision-focused):
TRID-1 AND (boarding OR "willingness to wait" OR "waiting time" OR "route choice" OR "departure time" OR "travel behavior" OR "passenger behavior")
```

```
TRID-3 (deployment and pilot reports, grey literature pass):
("crowding information" OR "occupancy information" OR "crowding display") AND (pilot OR demonstration OR deployment OR evaluation OR experiment)
```

TRID-3 существует потому, что развёрнутые пилоты часто публикуются отчётами, а
не статьями, и именно этот слой теряют Scopus и WoS.

### 3.6 IEEE Xplore

Синтаксис Command Search: поля в кавычках через двоеточие, `NEAR/n`, `*` с
минимум тремя символами до подстановки.

```
IEEE-1:
(("Abstract":crowd* OR "Abstract":occupancy OR "Abstract":"passenger load" OR "Abstract":"load factor" OR "Abstract":fullness OR "Abstract":"seat availability" OR "Abstract":overcrowd*) AND ("Abstract":"real-time" OR "Abstract":"real time" OR "Abstract":predictive OR "Abstract":live OR "Abstract":dynamic) AND ("Abstract":information OR "Abstract":display OR "Abstract":app OR "Abstract":signage OR "Abstract":"decision support") AND ("Abstract":bus OR "Abstract":transit OR "Abstract":"public transport" OR "Abstract":metro OR "Abstract":subway OR "Abstract":tram OR "Abstract":rail))
```

```
IEEE-2 (decision-focused):
IEEE-1 AND ("Abstract":boarding OR "Abstract":"willingness to wait" OR "Abstract":"waiting time" OR "Abstract":"route choice" OR "Abstract":"passenger behavior" OR "Abstract":"travel behavior")
```

```
IEEE-3 (measurement layer, feeds the sensing annex, not the core set):
(("Abstract":occupancy OR "Abstract":"crowd counting" OR "Abstract":"passenger counting" OR "Abstract":"crowd density") AND ("Abstract":bus OR "Abstract":transit OR "Abstract":"public transport" OR "Abstract":train OR "Abstract":tram) AND ("Abstract":camera OR "Abstract":vision OR "Abstract":sensor OR "Abstract":"edge device" OR "Abstract":"deep learning"))
```

IEEE-3 не входит в основной обзор. Он собирает измерительный слой, который в
статье описывается отдельно и не смешивается с поведенческим свидетельством.

### 3.7 Google Scholar

Google Scholar не является систематическим источником в этом обзоре. Его роль:
forward chaining через «Cited by», поиск полных текстов и точечная проверка
серой литературы. Причины: нет полевого синтаксиса, лимит длины запроса около
256 символов, и счётчики результатов являются оценками, а не точными числами.

```
GS-1:
"crowding information" OR "occupancy information" bus OR transit boarding OR "willingness to wait"
```

```
GS-2:
"real-time crowding" transit passenger boarding decision
```

```
GS-3 (geographic absence check, повторяется для каждой страны):
crowding information transit passengers Kazakhstan
crowding information transit passengers Almaty
crowding information transit passengers Uzbekistan
crowding information transit passengers Kyrgyzstan
```

Правила для Google Scholar:
- Записывать оценку числа результатов, но помечать её как оценку и **не
  использовать** как `records identified` в PRISMA-блоке идентификации из баз.
- Экранировать фиксированное число страниц выдачи, по умолчанию первые пять
  страниц (100 записей) на строку, число записывается в лог.
- Результаты GS-3 попадают в отчёт как documented absence: строка, дата, число
  просмотренных записей, число релевантных. Пустой результат - это результат, и
  он должен быть записан так же аккуратно, как непустой.

### 3.8 OpenAlex

Единственный источник с подтверждённым программным доступом. Синтаксис фильтра
`title_and_abstract.search` поддерживает `AND`, `OR`, `NOT` и фразы в кавычках,
но точное поведение нужно проверить на первом запуске по текущей документации
API и зафиксировать проверку в логе.

```
OA-1:
filter=title_and_abstract.search:("crowding information" OR "occupancy information" OR "crowding level" OR "passenger load information") AND (bus OR transit OR "public transport" OR metro OR tram OR rail)
```

```
OA-2:
filter=title_and_abstract.search:("real-time information" OR "passenger information" OR "traveller information") AND (crowding OR occupancy OR "load factor") AND (boarding OR "willingness to wait" OR "route choice" OR "waiting time")
```

Для OpenAlex дополнительно записывается `per_page`, курсор, общее
`meta.count` и число фактически извлечённых записей. Пилот 2026-08-26 извлекал
25 из тысяч, и именно поэтому он не является обзором.

---

## 4. What is logged for every search run

Лог живёт в `research/lit_review/search_log.md`. Его создаёт тот, кто запускает
поиск, в момент запуска, а не после. Одна строка на один набор в одной базе.
Ни одна цифра в этом файле не проставляется задним числом по памяти.

| Field | Type | Rule |
|---|---|---|
| `run_id` | string | `YYYY-MM-DD-<db>-<set>`, уникален |
| `database` | enum | Scopus, WoS, TRID, IEEE, GS, OpenAlex, other |
| `interface` | string | Web UI, API, конкретный прокси или его отсутствие |
| `date_run` | date | Дата запуска, ISO |
| `query_string` | verbatim | Точная строка, скопированная из поля поиска, без переписывания |
| `set_id` | string | Внутренний номер набора в базе (WoS #5, Scopus S5 и так далее) |
| `filters_applied` | verbatim | Год, тип документа, язык, предметная область. Если фильтров нет, пишется `none` |
| `results_count` | integer | Точное число, показанное базой. Для Google Scholar помечается `estimate` |
| `records_exported` | integer | Сколько записей фактически выгружено |
| `export_file` | path | Путь к файлу выгрузки |
| `run_by` | string | Имя человека. Если часть работы выполнил агент, пишется имя человека плюс `assisted by <agent>` |
| `notes` | free text | Отклонения синтаксиса, ошибки базы, отказ поля, всё нештатное |

Правило воспроизводимости: другой человек, имея только `database`,
`query_string`, `filters_applied` и `date_run`, должен получить набор, который
отличается только за счёт пополнения базы после этой даты.

---

## 5. Eligibility criteria

Каждый критерий сформулирован как проверяемый вопрос с ответом да или нет.
Если два человека не могут ответить одинаково, критерий переписывается, а не
интерпретируется на месте.

### 5.1 Inclusion

Запись включается, только если ответ «да» на **все** критерии I1-I6.

- **I1 Population.** Does the record concern human passengers of urban or
  suburban public transport, or agents that explicitly represent such
  passengers?
  *Test:* the abstract names passengers, riders, travellers or commuters, or
  simulated agents standing for them, as the unit whose behaviour is studied.
  *No if:* the unit is a vehicle, a network, an operator or a pedestrian crowd
  with no transport decision.

- **I2 Exposure.** Does the record involve information about the crowding,
  occupancy or load of a specific vehicle, car, platform or departure being
  made available to those passengers?
  *Test:* the information exists in one of four forms, all eligible, and the
  form is recorded: (a) deployed to real passengers, (b) described inside a
  stated-preference or laboratory task, (c) supplied as an input to a model or
  simulation, (d) manipulated in format or accuracy across conditions.
  *No if:* crowding appears only as a system state that no passenger is told
  about. Such records go to the context tier, §5.4, not to the core set.

- **I3 Comparison.** Does the record contain a comparison that could reveal an
  effect of the information?
  *Test:* at least one of: information present versus absent, differing
  crowding levels shown, differing formats, differing accuracy or freshness,
  before versus after deployment, or a modelled penetration rate that varies.
  *No if:* a single condition is described with nothing to compare it against.

- **I4 Outcome.** Does the record report at least one passenger decision
  outcome, one operational consequence of such decisions, or one perception or
  usability outcome directly tied to the information?
  *Test:* the outcome can be named in one phrase and entered in the
  `outcome_type` column of the evidence matrix.
  *No if:* the only reported outcome is sensing accuracy, forecasting error or
  system performance with no behavioural or operational claim.

- **I5 Design and reporting.** Does the record report a method in enough detail
  that its design and its result can both be entered in the evidence matrix?
  *Test:* a reader can name the design category and state what was found,
  including direction, from the retrieved text.
  *No if:* the design cannot be classified, or the result is asserted without a
  method.

- **I6 Availability and language.** Is the full text retrievable by the team and
  written in English, Russian or Kazakh?
  *Test:* a full text or an author manuscript is in hand, or is obtainable
  within two documented retrieval attempts.
  *No if:* neither holds. Record it as excluded for non-retrieval, never as
  excluded on content, and never summarise its findings.

### 5.2 Exclusion

Запись исключается, если верен **любой** критерий E1-E7. Каждый код
исключения записывается на полнотекстовой стадии.

- **E1 No passenger decision.** Sensing, counting, forecasting or benchmarking
  work with no behavioural or operational outcome. Such records may still be
  relevant to the measurement layer and go to the sensing annex, §5.5.
- **E2 Out-of-scope mode or context.** Air travel, intercity or long-distance
  rail, private car traffic information, freight, and pedestrian crowd
  management with no public-transport decision. Urban and suburban commuter
  rail is in scope; intercity rail is not.
- **E3 Crowding as risk only.** Studies where crowding is treated exclusively
  as an infection or safety exposure with no travel decision outcome.
- **E4 Not a study.** Editorials, news items, marketing material, standards
  without evidence, extended abstracts without a methods section, and slide
  decks.
- **E5 Superseded duplicate.** A preprint, working paper or conference version
  superseded by a later peer-reviewed version. The published version is kept,
  the superseded record is logged with a pointer, and the pair counts once.
- **E6 Not retrievable.** Full text not obtained after two documented attempts,
  including one request to the corresponding author where an address exists.
- **E7 Language.** Full text is not in English, Russian or Kazakh.

### 5.3 Deliberate non-criteria

Эти признаки **не** являются основанием для исключения и только записываются:
geography, income level of the country, journal ranking, year, sample size,
statistical significance of the result, and whether the finding agrees with the
project's hypotheses. Исключение по значимости результата - это встроенный
publication bias, и он запрещён этим протоколом.

### 5.4 Two tiers

- **Core set.** Passes I1-I6. Answers SQ1, SQ2, SQ3, SQ5, SQ6.
- **Context set.** Passes I1, I3, I4, I5, I6 but fails I2 because no
  information was provided: crowding valuation, revealed crowding response,
  denied boarding, dwell and bunching studies. Answers SQ4 and supplies the
  magnitudes the paper needs to interpret its own effect.
  Чтобы context set не разросся бесконечно, он комплектуется по правилу:
  систематические обзоры по crowding valuation плюс работы, найденные chaining
  от core set, плюс `S7`. Правило записывается в отчёте.

### 5.5 Sensing annex

Records failing I4 but describing occupancy measurement in vehicles or stations
are kept in a separate annex list. They support the technology section of the
paper and never enter the behavioural synthesis or the PRISMA counts of the
core review. Смешивать их с поведенческими работами нельзя: это разные
доказательства для разных утверждений.

---

## 6. Screening procedure

### 6.1 Stages

1. **Deduplication.** Merge by DOI first, then by normalised title plus first
   author plus year. Дубликаты считаются и записываются.
2. **Title and abstract screening.** Each record receives one code: `include`,
   `exclude:<code>`, or `unclear`. `unclear` always goes to full text. Правило
   существует, чтобы у одного исследователя не было соблазна решать пограничные
   случаи по настроению.
3. **Full-text screening.** Each record receives `include:core`,
   `include:context`, `include:annex` or `exclude:<code>` with a one-sentence
   written reason.
4. **Extraction.** Included records are entered in the evidence matrix, §8.

### 6.2 Calibration before screening starts

Before the first record is screened for real, screen a pilot of 30 records
drawn at random from the pooled search results, then revise any criterion that
proved ambiguous, and record the revision in §13. Screening that begins without
calibration is not reproducible, because the criteria change silently as the
screener learns the literature.

### 6.3 Disagreement resolution, two or more screeners

- Screen independently, then compare.
- Disagreements are resolved by discussion against the wording of §5, not by
  seniority.
- If discussion fails, a third person decides, and the case is recorded.
- Report percent agreement and Cohen's kappa for the title and abstract stage.

### 6.4 Single-screener mode

Это ожидаемый режим для этого проекта, и он должен быть описан честно, а не
замаскирован.

- **S1.** Screen everything once with the codes in §6.1.
- **S2.** After at least seven days, blindly re-screen a random 20% sample of
  the title and abstract stage. Compute percent agreement and Cohen's kappa
  against the first pass and report both. Если agreement ниже 0.80, критерии
  переписываются и **весь** набор скринится заново.
- **S3.** Every `exclude` at the full-text stage carries a written reason. Every
  `include` carries the criterion that was hardest to satisfy.
- **S4.** One external check on a 10% sample by a second person, when one is
  available. An LLM pass may be used as a **screening aid** and must be logged
  as such in the `run_by` field. Ассистирующая модель не является вторым
  ревьюером и не заменяет S2.
- **S5.** State in the paper's methods that screening was performed by one
  researcher with a delayed re-screen, and carry it into limitations, §11.

---

## 7. PRISMA flow

Шаблон. **Все числа заполняются только по факту выполненного поиска.** Пустой
шаблон в репозитории - это нормально, шаблон с придуманными числами - нет.

```
IDENTIFICATION
  Records identified from databases
    Scopus                                        n = ___
    Web of Science                                n = ___
    TRID                                          n = ___
    IEEE Xplore                                   n = ___
    OpenAlex                                      n = ___
    Total from databases                          n = ___
  Records identified from other sources
    Backward citation chaining                    n = ___
    Forward citation chaining                     n = ___
    Verified reference base seeds (see §10)       n = ___
    Google Scholar and grey literature            n = ___
    Total from other sources                      n = ___

  Records removed before screening
    Duplicates                                    n = ___
    Records marked ineligible by automation       n = ___
    Records removed for other reasons             n = ___

SCREENING
  Records screened on title and abstract          n = ___
  Records excluded at title and abstract          n = ___
    by code E1 / E2 / E3 / E4 / E5 / E6 / E7      n = ___ each

  Reports sought for retrieval                    n = ___
  Reports not retrieved (E6)                      n = ___

  Reports assessed for eligibility (full text)    n = ___
  Reports excluded at full text                   n = ___
    by code E1 / E2 / E3 / E4 / E5 / E6 / E7      n = ___ each

INCLUDED
  Studies included in the core set                n = ___
  Studies included in the context set             n = ___
  Records placed in the sensing annex             n = ___
  Studies entering the evidence matrix            n = ___
```

Правила:
- Числа в блоке идентификации берутся из `results_count` в логе §4, а не из
  памяти и не из оценок Google Scholar.
- Сумма исключений по кодам обязана сходиться с общим числом исключений на
  соответствующей стадии. Если не сходится, ошибка в логе, а не в арифметике
  отчёта.
- Sensing annex не входит в `Studies included in the core set`.
- Diagram рисует CADy по этим числам после того, как они получены. Фигура не
  строится раньше данных.

---

## 8. Evidence matrix

Схема столбцов из чартера §9 сохранена целиком и дополнена полями, без которых
запись нельзя ни найти, ни проверить, ни привязать к теме статьи. Файл заголовков:
`research/lit_review/evidence_matrix_template.csv`. Он содержит только заголовки
и ни одной строки данных.

| Column | Type | Allowed values | Charter §9 |
|---|---|---|---|
| `record_id` | string | `LR-001`, sequential | added |
| `citation_key` | string | Key from `research/refs/base/references.md`, or empty if the record is not in the base | added |
| `in_reference_base` | enum | `yes`, `no` | added |
| `first_author` | string | Family name of the **first** author, verified against the publisher record or Crossref | added |
| `year` | integer | Publication year of the version actually used | added |
| `title` | string | Verbatim | added |
| `venue` | string | Journal, conference or report series | added |
| `doi_or_url` | string | DOI preferred, resolvable URL otherwise | charter: DOI/full text |
| `source_of_record` | enum | `scopus`, `wos`, `trid`, `ieee`, `openalex`, `gscholar`, `backward_chain`, `forward_chain`, `refbase_seed`, `expert_suggestion` | added |
| `tier` | enum | `core`, `context`, `annex` | added |
| `location_city` | string | City or region, `multiple`, or `not applicable` for pure modelling | charter: location |
| `location_country` | string | Country, or `multiple` | charter: location |
| `mode` | enum | `bus`, `brt`, `tram`, `metro`, `urban rail`, `commuter rail`, `multimodal`, `not specified` | charter: mode |
| `setting` | enum | `field`, `laboratory`, `online survey`, `intercept survey`, `simulation`, `secondary data`, `review` | added |
| `sample_unit` | enum | `passenger`, `choice observation`, `trip`, `vehicle run`, `stop-hour`, `agent`, `study` | charter: sample |
| `sample_size` | integer or string | Number, with the unit named in `sample_unit`; `not reported` allowed and must be used when true | charter: sample |
| `study_period` | string | Data collection dates, `YYYY-MM` to `YYYY-MM`, or `not reported` | added |
| `design` | enum | `randomized field experiment`, `cluster randomized`, `crossover randomized`, `natural experiment`, `difference-in-differences`, `regression discontinuity`, `matched quasi-experiment`, `uncontrolled before-after`, `deployment with observation`, `stated preference DCE`, `stated preference non-DCE survey`, `revealed preference observational`, `laboratory experiment`, `agent-based simulation`, `analytical or assignment model`, `systematic review`, `qualitative or stakeholder study`, `other` | charter: design |
| `preference_type` | enum | `stated`, `revealed`, `both`, `simulated`, `not applicable` | charter: stated/revealed |
| `information_provided` | enum | `deployed to real passengers`, `described in survey task`, `model input`, `format or accuracy manipulated`, `none` | added |
| `information_channel` | enum | `platform display`, `in-vehicle display`, `mobile app`, `web`, `push or SMS`, `printed`, `staff announcement`, `hypothetical description`, `none`, `not reported` | charter: information channel |
| `crowding_construct` | enum | `passengers per square metre`, `load factor percent of capacity`, `passenger count`, `seat availability ordinal`, `ordinal crowding levels`, `subjective rating`, `denied boarding binary`, `other`, `not applicable` | charter: crowding scale |
| `crowding_scale_levels` | integer or string | Number of displayed or elicited levels, `continuous`, or `not applicable` | charter: crowding scale |
| `outcome_type` | enum | `board first vehicle`, `skip and wait`, `waiting time`, `route choice`, `itinerary choice`, `departure time choice`, `car or platform choice`, `mode choice`, `ridership`, `load distribution`, `load variance`, `bunching or headway`, `dwell time`, `denied boarding`, `perception or attitude`, `usability`, `other` | charter: outcome |
| `outcome_definition` | free text | How the outcome was actually measured, in one sentence | added |
| `identification_strategy` | enum | `individual randomization`, `cluster randomization`, `instrumental variable`, `difference-in-differences`, `synthetic control`, `regression discontinuity`, `matching`, `panel fixed effects`, `covariate adjustment only`, `none`, `not applicable` | charter: identification strategy |
| `model` | string | Estimator or model family, for example `mixed logit`, `latent class logit`, `binary logit with clustered SE`, `agent-based`, `descriptive` | charter: model |
| `effect_direction` | enum | `increases boarding`, `decreases boarding`, `shifts distribution`, `no detectable effect`, `mixed`, `not applicable` | charter: effect |
| `effect_size` | free text | The number as reported, with its unit. Verbatim, never converted silently | charter: effect |
| `uncertainty` | free text | CI, standard error, p value or the explicit statement that none was reported | added |
| `limitation` | free text | The limitation as stated by the authors, plus any additional limitation identified during extraction, marked as such | charter: limitation |
| `gap_type` | enum | `geographic`, `population`, `methodological`, `theoretical`, `time`, `contradictory findings`, `none` | added, from the course notes |
| `theme` | enum | `T1 deployed and observed`, `T2 stated preference`, `T3 simulation`, `T4 revealed without treatment`, `T5 perception and display`, `T6 field information experiments`, `T7 context and geography`, `T8 sensing layer` | added, maps to `TRC_PAPER_BLUEPRINT.md` §2 |
| `verification_status` | enum | `full text`, `publisher abstract`, `reconstructed abstract`, `metadata only`, `not retrievable` | charter: verification status |
| `full_text_location` | path or URL | Where the copy actually is | charter: DOI/full text |
| `screened_by` | string | Person, plus `assisted by <agent>` where true | added |
| `screening_date` | date | ISO | added |
| `notes` | free text | Discrepancies, retraction notices, version conflicts | added |

Правила заполнения:

1. `effect_size` копируется как напечатано у авторов. Пересчёт в другую единицу
   допустим отдельной колонкой в анализе, но не вместо оригинала.
2. `verification_status` = `metadata only` запрещает использовать любую строку
   из `effect_size`, `effect_direction` и `limitation` этой записи в прозе
   статьи. Такие строки остаются в матрице как известные пробелы.
3. `gap_type` использует шесть типов из курсовых заметок
   (`03_Learning/Research Writing/finding-research-gaps.md`) и заполняется по
   секциям Limitations и Future research исходной работы, а не по впечатлению.
4. `theme` определяет, в какой абзац related work попадёт запись. Синтез
   организуется по темам, а не по авторам.
5. Каждая ячейка со значением `not reported` означает «проверено, автор не
   сообщил», а не «не проверял». Если не проверял, ячейка остаётся пустой.

---

## 9. Citation chaining procedure

### 9.1 Seeds

Seeds - записи из §10, прошедшие тот же screening. Работа не становится seed
только потому, что она есть в reference base.

### 9.2 Backward chaining

1. Для каждого core seed прочитать список литературы полностью.
2. Отобрать записи, чьи title или контекст цитирования удовлетворяют I1-I4 на
   уровне заголовка.
3. Внести их в общий пул со `source_of_record = backward_chain` и прогнать
   через обычный screening.
4. Depth: **2 hops** от core seeds, **1 hop** от context seeds.

### 9.3 Forward chaining

1. Получить цитирующие работы для каждого seed в OpenAlex, и в Scopus и Web of
   Science, если доступ подтверждён, и в Google Scholar как дополнение.
2. Записать для каждого seed число цитирующих работ на дату запуска.
3. Если цитирующих работ у одного seed больше 200, применяется предзаписанное
   правило `min-seeds >= 2`: рассматриваются только работы, цитирующие минимум
   два seed. Порог и число отфильтрованных записей записываются. Это правило
   уже применялось в пилоте 2026-08-26 и оставлено ради сопоставимости.
4. Depth: **2 hops** от core seeds, **1 hop** от context seeds.

### 9.4 Stopping rule

Chaining останавливается, когда выполнено любое из условий, и то, какое именно,
записывается:

- **R1 Saturation.** A full hop yields fewer than 5% new records that pass
  title and abstract screening, relative to the number of core-set records
  already included.
- **R2 Closure.** Every record produced by the hop has already been screened.
- **R3 Depth cap.** The depth limits in §9.2 and §9.3 are reached.
- **R4 Resource cap.** A pre-declared budget of screening effort is exhausted.
  Использование R4 обязательно переносится в §11 как ограничение полноты, с
  указанием, на каком hop остановились.

### 9.5 Recording

Для каждого hop записывается: seed key, направление, база, дата, число
найденных, число новых после дедупликации, число прошедших скрининг,
сработавшее правило остановки.

---

## 10. Relation to the verified reference base

### 10.1 What the base is and is not

`research/refs/base/references.md` содержит 146 записей, проверенных по живым
страницам как библиографические записи. По собственному заявлению файла,
содержание работ за пределами названия, площадки и авторства не проверялось.
Следовательно:

- База даёт **проверенные точки входа**, а не готовый related work.
- Членство в базе **не** является включением в обзор. Каждый seed проходит §5 и
  §6 наравне с любой другой записью.
- База собиралась для outreach, то есть отбиралась в том числе по
  досягаемости авторов. Это селекционное смещение, и оно записано в §11.

### 10.2 Seed papers

Ниже перечислены **только** те citation keys, которые действительно являются
строками в `references.md` и относятся к review question.

**Core seeds, tier 1. Информация о загруженности показывается пассажиру.**
Раздел 1 базы, полностью.

- `pan_2025_itinerary`
- `drabicki_2023_willingness`
- `drabicki_2023_bunching`
- `kapatsila_2025_crowding`
- `zhangkennedy_2023_visualizations`
- `preston_2019_occupancy`
- `koutsopoulos_2021_predictive`
- `stoltz_2026_coaches`
- `kaparias_2015_countdown`
- `gentile_2005_routechoice`
- `jenelius_2020_personalized`

**Design seeds, tier 1. Полевые исследования информации в транспорте, дающие
шаблон идентификации для SQ6.**

- `brakewood_2014_tampa`
- `watkins_2011_wheresmybus`
- `hsu_2021_waiting`
- `nassir_2018_bayesian`
- `prabhakar_2013_insinc`
- `allcott_2014_shortrun`

**Context seeds, tier 2. Реакция на загруженность и её цена без
информационного вмешательства, SQ4.**

- `fedujwar_2024_valuation`
- `hurtubia_2017_discomfort`
- `raveau_2014_routechoice`
- `shao_2022_timevalue`
- `monchambert_2017_whocares`
- `kim_2021_metrocrowding`
- `lijesen_2025_occupancy`
- `seriani_2022_laboratory`
- `pi_2018_fullness`
- `ma_2019_deniedboarding`
- `ma_2024_realtimedenied`

**Display and legibility seeds, tier 2. Виден ли и понятен ли сигнал в момент
решения, SQ5.**

- `kay_2016_whenish`
- `hullman_2015_hops`
- `muller_2009_displayblindness`
- `willett_2017_embedded`
- `langheinrich_2012_engagement`

**Operational-consequence seeds, tier 2. Связь индивидуального решения с
показателями маршрута, SQ3 и системная часть статьи.**

- `daganzo_2009_headway`
- `gkiotsalitis_2021_atstop`
- `schmocker_2015_reliability`
- `ingvardson_2018_arrival`
- `currie_2013_streetcar`
- `fujiyama_2021_density`
- `saidi_2025_capacity`

**Geographic and context seeds, tier 2. SQ7.**

- `sgibnev_2016_marshrutkas`
- `tymbayeva_2025_almaty`
- `rekhviashvili_2020_informality`
- `muleev_2020_marshrutkas`

**Methodological seed для причинного вывода в транспорте.**

- `graham_2025_causal`

**Sensing annex entry points, §5.5. В поведенческий синтез не входят.**

- `kuchar_2023_review`
- `ghaderi_2024_onbus`
- `caballerogil_2025_tram`
- `bell_2023_socialdistance`
- `handte_2014_density`
- `fiorista_2025_cctv`
- `munizaga_2014_validating`

Итого seeds для chaining: 11 core, 6 design, 27 context, 1 methodological,
7 annex. Это число seeds, а не число включённых работ. Число включённых работ
неизвестно до выполнения §6.

### 10.3 Known problems in the base that this review must resolve

Проверено по Crossref 2026-09-02 при подготовке протокола. Первые три пункта
**уже исправлены** в тот же день, `validate.py` после правки выходит с кодом 0.
Записаны здесь, потому что это ровно тот класс ошибки, который проекту уже
дорого обошёлся, и он должен остаться видимым.

1. `kucharski_2023_willingness` вёл ведущей фамилией Kucharski, тогда как
   первый автор это **Arkadiusz Drabicki** (Drabicki, Cats, Kucharski, Fonzone,
   Szarata, *Research in Transportation Business & Management*, vol. 47, art.
   100963, 2023). Ключ отражал адресата outreach, а не авторство. Переименован
   в `drabicki_2023_willingness`, авторы выписаны полностью.
2. `agarwal_2024_valuation` вёл фамилией Agarwal, тогда как первый автор это
   **Rupam Fedujwar** (Fedujwar and Agarwal, *Public Transport*, vol. 16, no. 3,
   pp. 743-773, 2024). Переименован в `fedujwar_2024_valuation`.
3. `drabicki_nodate_bunching` не имел ни DOI, ни URL и помечался `validate.py`
   как unlinkable. Запись разрешена: Drabicki, Kucharski, Cats,
   *Transportation*, vol. 50, no. 3, pp. 1003-1030, DOI
   `10.1007/s11116-022-10270-3`, онлайн 2022-03-04, печатный выпуск 2023-06.
   Переименован в `drabicki_2023_bunching`.
4. `fujiyama_2021_density` и `luangboriboon_nodate_density` указывают на одну и
   ту же запись UCL. **Не исправлено**, обе строки были в проверенном списке
   источников. При extraction это одна работа, не две, и в PRISMA она
   считается один раз.

Ни одна из трёх исправленных работ не была названа в отправленном письме
(статус `LISTED`), поэтому исправлений в адрес получателей не требуется. Если
бы стояло `CITED`, требовалось бы.

### 10.4 Works the review question needs and the base does not contain

По состоянию на 2026-09-02 в `references.md` **нет строк** для нескольких
работ, на которые опирается чартер §3 и §9 и черновик рукописи, в том числе
для платформенного пилота RTCI в Стокгольме, на который чартер ссылается через
TRID. Эти работы здесь не цитируются и их результаты не пересказываются,
потому что правило проекта это запрещает.

Действие: тот, кто запускает поиск, находит их в базах по §3, проверяет по
живой странице и добавляет строкой в `references.md` с DOI и ключом **до**
того, как хоть одно утверждение из них попадёт в прозу. До этого момента они
существуют в матрице как записи с `in_reference_base = no`.

---

## 11. Limitations

Формулируются сейчас, а не после того, как результат окажется неудобным.
Разделены на систематические (свойство метода) и содержательные (свойство
доступного материала), как требует `research/coursework/RESEARCH_METHOD_GROUNDING.md`.

### 11.1 Systematic

1. **Single researcher.** Screening and extraction are performed by one person.
   The delayed re-screen of a 20% sample in §6.4 bounds but does not remove
   the risk of inconsistent application of criteria. Reported agreement values
   must appear in the paper, whatever they are.
2. **Database access.** Coverage is bounded by which databases were actually
   available, see §2. If Scopus and Web of Science remain unconfirmed, the
   review is a TRID, IEEE Xplore, OpenAlex and citation-chaining review, and
   must be described that way.
3. **Language.** English, Russian and Kazakh only. Chinese, Japanese, Korean,
   French, German, Spanish and Portuguese literature on transit crowding
   information is therefore under-represented, and the absence of a finding in
   this review is not evidence of its absence in the field.
4. **Grey literature.** Deployment pilots are frequently reported as operator
   or agency documents rather than as articles. TRID-3 in §3.5 is a partial
   remedy. Reports that are not indexed anywhere and not published online
   cannot be found by this protocol at all.
5. **Publication bias.** Deployments that produced no detectable effect are
   less likely to be written up than deployments that produced one. The review
   cannot correct for this: with heterogeneous constructs and outcomes there is
   no defensible funnel plot. The consequence is stated directly in the paper:
   the field-effect estimates available in the literature are plausibly biased
   upward, and the Sanas field experiment should be powered for a small effect
   rather than for the published magnitudes.
6. **Seed selection bias.** The reference base was assembled for cold-email
   outreach and therefore favours authors who were reachable and groups that
   were being contacted. Chaining from those seeds propagates that bias.
   Mitigation: the database searches in §3 are run independently of the seeds,
   and the proportion of finally included records that came from seeds rather
   than from database search is reported.
7. **No meta-analysis.** Effect sizes in this field are measured on
   incompatible constructs (percentage points of car choice, minutes of
   willingness to wait, probability of skipping a departure, percent ridership
   change). Synthesis is narrative and matrix-based by design, not by omission.

### 11.2 Content

8. **Construct heterogeneity of crowding.** Passengers per square metre, load
   factor, ordinal seat-availability scales and subjective ratings are not
   interchangeable. The matrix records the construct for every record, and the
   synthesis compares only within construct.
9. **Time.** Crowding tolerance and information behaviour plausibly shifted
   around 2020. The matrix records `study_period` for every record so that
   pre-2020 and post-2020 evidence can be separated rather than pooled.
10. **Verification depth.** Records at `metadata only` and `reconstructed
    abstract` cannot support any stated finding, per §8 rule 2. The count of
    such records is reported, because it bounds how much of the review rests
    on read text.

---

## 12. How the review output enters the manuscript

- Themes `T1` to `T7` in the matrix map one to one onto the related-work
  synthesis themes of `research/TRC_PAPER_BLUEPRINT.md` §2. Каждая тема
  становится абзацем или подразделом, организованным по идее, а не по автору.
- The comparison table required by the blueprint is generated from the matrix
  columns `design`, `preference_type`, `mode`, `information_channel`,
  `identification_strategy` and `outcome_type`.
- Gap types from `gap_type` support the Research Gap element of the four-part
  introduction. A gap claim in the introduction must be traceable to at least
  three matrix rows or to a documented absence with a recorded search string,
  date and hit count.
- References are numbered IEEE style by order of first appearance in the
  manuscript, consistent with `research/paper/manuscript/references.md`.
- The PRISMA diagram and any figure summarising the matrix are specified by
  Danyshpan and produced by CADy after the numbers exist.
- No claim of novelty or of being first is written before §6 and §9 are
  complete and the stopping rule that fired is recorded.

---

## 13. Deviations and change log

Любое отклонение от этого файла во время выполнения записывается здесь до того,
как оно будет применено, с датой и причиной. Обзор, чей протокол переписан
задним числом под полученный результат, не является систематическим.

| Date | Section | Deviation | Reason | Recorded by |
|---|---|---|---|---|
| 2026-09-02 | §3.1 | Charter concept block 1 split into Block 1 (crowding state) and Block 2 (real-time information) | Phrase-level block loses records where the two components are separated in the abstract | Danyshpan, at protocol design time |
| ___ | ___ | ___ | ___ | ___ |
