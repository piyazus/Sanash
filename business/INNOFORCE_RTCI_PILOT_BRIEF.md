# Innoforce / Avtobys RTCI Pilot Brief

Статус: discussion document, не commitment  
Дата: 2026-08-26

## Цель встречи

Понять, возможно ли технически и юридически провести ограниченный научный
эксперимент: показать части пользователей Avtobys real-time crowding
information и измерить, меняет ли это решение сесть в первый автобус или ждать.

Первый pilot предлагается на одном высокочастотном маршруте, не по всему городу.

## Что Sanas предоставляет

- prototype occupancy measurement;
- five-level crowding signal + confidence/status;
- timestamped device telemetry;
- shadow validation against manual ground truth;
- research protocol и analysis;
- технические accuracy/freshness/failure gates до public display.

## Что требуется выяснить у Innoforce

### 1. App experiment capability

- Можно ли включить feature только для pilot route/stops?
- Есть ли server-side feature flags?
- Возможен ли stable randomized A/B по pseudonymous user ID?
- Можно ли control и treatment держать одновременно?
- Можно ли логировать card impression, а не только app open?
- Можно ли заморозить UI version на период эксперимента?

### 2. Boarding outcome

- Есть ли связь Avtobys с ticket validation/payment event?
- Можно ли связать app session и boarding pseudonymously?
- Есть ли vehicle/route/trip/stop IDs и точные timestamps?
- Может ли phone movement относительно bus GPS использоваться opt-in?
- Возможна ли in-app кнопка `boarded / decided to wait`?
- Если individual outcome невозможен, какие aggregate boarding/APC data есть?

### 3. Operational data

- AVL/GPS и update frequency;
- scheduled и predicted arrival;
- actual stop arrival/departure;
- vehicle-trip assignment;
- route direction;
- headway и disruptions;
- vehicle certified seating/total capacity;
- ticket validations/APC;
- historical data retention и качество IDs.

### 4. RTCI integration

- Предпочтительный transport/API: HTTPS, MQTT или существующий internal bus;
- authentication и device provisioning;
- message TTL и stale handling;
- mapping `device_id -> vehicle_id -> trip_id`;
- expected end-to-end latency;
- server deduplication;
- rollback/kill switch;
- UI state для `unknown/degraded`;
- monitoring and incident contact.

### 5. Pilot operations

- Как выбрать high-frequency route с вариацией загрузки?
- Сколько последовательных автобусов можно оборудовать?
- Есть ли один стабильный bus type?
- Кто даёт доступ к автобусу и точкам крепления?
- Можно ли провести stationary volunteer/FOV test?
- Можно ли провести shadow week до показа RTCI?
- Кто отвечает за ежедневный доступ, питание и проверку устройства?

### 6. Legal, privacy, ethics

- Кто является data controller/processors?
- Какое разрешение требуется от перевозчика/акимата?
- Может ли Innoforce провести legal/privacy review?
- Как уведомляются пассажиры?
- Допускается ли raw video для research validation и на какой срок?
- Где разрешено хранить research data?
- Какие правила применяются к несовершеннолетним?
- Возможен ли pseudonymous research export?

## Минимальные события Avtobys

```text
assignment
app_session
rtci_card_impression
occupancy_message_rendered
vehicle_arrival
boarding_outcome_or_proxy
service_disruption
```

Каждое событие должно иметь compatible route/stop/trip/vehicle identifiers и
UTC timestamp.

## Предлагаемый pilot scope

- 1 high-frequency route;
- 1 bus type;
- несколько последовательных оборудованных автобусов;
- selected stops/directions/time windows;
- shadow technical period;
- затем randomized treatment только после go/no-go;
- заранее определённая длительность по power analysis;
- city-wide rollout не является частью первого pilot.

## Research comparison

Control:

```text
ETA + current standard Avtobys information
```

Treatment:

```text
same information + RTCI for first and next service
```

Нельзя одновременно менять pricing, incentives, ETA logic или notification
frequency: иначе эффект crowding information не идентифицируется.

## Необходимые решения встречи

1. Возможен ли user-level A/B?
2. Как наблюдается boarding outcome?
3. Какие operational data доступны?
4. Какой route/bus fleet доступен для pilot?
5. Кто утверждает privacy/ethics/operator access?
6. Кто technical owner integration с каждой стороны?
7. Какая практически значимая величина эффекта нужна Innoforce?

## Не обещать на встрече

- готовую accuracy текущего устройства;
- city-wide launch через месяц;
- causal result до эксперимента;
- production hardware readiness;
- continuous raw-video upload;
- пользу RTCI без измерения waiting burden и operational side effects.
