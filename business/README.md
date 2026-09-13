# Sanas: бизнес и партнёры

## Ближайший запрос — 12 сентября 2026

Подготовлен [запрос контакта техника и доступа к штатному видео](../deliverables/progress_2026-09-12/innoforce_request.md).
Нужны модель регистратора, способ получения отдельного канала, ответственный
за доступ и дата осмотра. Черновик не отправлен; адресат, доступ и дата пока
не установлены. Это проверка новой гипотезы использования существующих камер.

Направление связывает технический прототип и RTCI-эксперимент с
Innoforce/Avtobys, оператором транспорта и владельцами автобусов. Здесь нет
утверждённого коммерческого предложения для текущей потолочной RGB-архитектуры:
существующие файлы — рабочие черновики и вопросы для согласования.

## Документы

- `INNOFORCE_RTCI_PILOT_BRIEF.md` — рамка
  совместного пилота и требования к интеграции.
- `INNOFORCE_QUESTIONS.md` — незакрытые вопросы к
  Innoforce по данным, API, идентификаторам, задержке и ответственности сторон.
- `FILMING_PERMISSION_REQUEST.md` — черновик
  запроса разрешения на съёмку в салоне.
- `outreach/template.md` — шаблон обращения к
  исследователям и экспертам.
- `outreach/` — локальные контактные списки. Они содержат персональные адреса,
  исключены из Git и не должны публиковаться.

## Критический путь

До полевого пилота нужны как минимум разрешение на съёмку, доступ к автобусу,
связка `device_id` с `vehicle_id`/рейсом, согласованный транспорт сообщения,
TTL и политика отображения устаревших либо отсутствующих данных. Актуальные
технические варианты находятся в `../PRODUCT_SPEC.md`, а
исследовательский дизайн — в [`../research/README.md`](../research/README.md).

## Consolidated Markdown source archive

The original contents of the removed Markdown files are preserved below. Each collapsible block records its former path, encoding, size, and SHA-256.

The two outreach lists containing personal email addresses are preserved locally in `business/outreach/contacts_archive.txt`; that file remains ignored by Git.

| Former path | Bytes | Encoding | SHA-256 |
|---|---:|---|---|
| `business/FILMING_PERMISSION_REQUEST.md` | 9340 | UTF-8 | `5bb663b5190c9baf474a7530dbf75bd2952339a8009014ef048c469594491732` |
| `business/INNOFORCE_QUESTIONS.md` | 11652 | UTF-8 | `abdc30ccea753c9a191ede4e37136b49bf0a45a9dca14a522ce4cc9e1bcd7953` |
| `business/INNOFORCE_RTCI_PILOT_BRIEF.md` | 6027 | UTF-8 | `449b2290afd3493cc2c7fdefd645ba0f2c919434a99a3e826cf3c0cf28a2089f` |
| `business/outreach/template.md` | 2870 | UTF-8 | `a9051543183992c63d9e0f35f183b37571290a747d7fe1dc2abdc2de25225eb1` |

<details>
<summary><code>business/FILMING_PERMISSION_REQUEST.md</code> - original text</summary>

# Запрос разрешения на съёмку в салоне: черновик

Статус: **черновик, не отправлено, адресат не определён**
Дата: 2026-09-03
Authority: подчиняется [`../GROUND_TRUTH.md`](../GROUND_TRUTH.md)

## 0. Почему это отдельный документ

Разрешение на съёмку записано в Ground Truth 3.1 как внешний блокер. Без него
собственных данных не будет никогда, сколько бы железа ни было куплено. Это
единственный блокер, который не решается ни работой, ни деньгами, только
чужим согласием.

Адресат неизвестен. Кто именно выдаёт разрешение, Innoforce, оператор маршрута
или акимат, это первый вопрос раздела 4 в
[`INNOFORCE_QUESTIONS.md`](INNOFORCE_QUESTIONS.md). До ответа письмо не
отправляется, иначе оно уйдёт не туда и сожжёт первое впечатление.

## 1. Что именно запрашивается

Формулировать надо узко. Широкий запрос пугает и его дольше согласовывать.

| Параметр | Запрашиваемое значение |
|---|---|
| Что снимается | салон автобуса, камера под потолком, обзор пассажирского пространства |
| Что не снимается | лица крупным планом целенаправленно, кабина водителя, зона оплаты, улица |
| Сколько автобусов | минимально необходимое, уточняется |
| Сколько времени | ограниченный период сбора, уточняется |
| Куда попадают записи | локальное хранилище проекта, наружу не передаются |
| Кто имеет доступ | поимённый список |
| Срок хранения | ограниченный, уточняется |
| Цель | обучение и проверка модели оценки загруженности |

## 2. Что предложить в обмен на согласие

Разрешение дают охотнее, когда понятно, что получает дающий:

- оператор получает объективную картину загруженности своих рейсов, которой у
  него сейчас нет;
- данные о загрузке по времени суток и по рейсам полезны для планирования
  интервалов;
- пилот ограничен и не требует изменений в подвижном составе, кроме крепления
  устройства;
- результаты предоставляются оператору до публикации.

## 3. Что обязательно указать про приватность

Это та часть, из-за которой отказывают. Писать её надо конкретно, а не общими
словами:

- лица не распознаются, идентификация людей не производится, повторное
  опознание между поездками не производится;
- система считает людей, а не следит за конкретными людьми;
- сырое видео не покидает устройство в продуктовом режиме;
- на этапе сбора данных записи хранятся в согласованном периметре и
  удаляются по истечении срока;
- для любой публикации, включая презентации и статью, лица размываются;
- пассажиры информируются о съёмке способом, который согласует оператор.

## 4. Открытые вопросы до отправки

Ни один из них нельзя закрыть догадкой:

1. Кто является уполномоченным адресатом.
2. Какое правовое основание обработки применимо по законодательству
   Казахстана к видеосъёмке в общественном транспорте.
3. Нужно ли отдельное информирование пассажиров и в какой форме.
4. Требуется ли согласование с акиматом помимо оператора.
5. Есть ли у оператора уже действующее видеонаблюдение в салонах и какой
   режим согласования применялся к нему. Если да, это самый быстрый путь:
   встроиться в существующую процедуру вместо создания новой.

Пункт 5 стоит проверить первым. Если в автобусах уже стоят камеры
видеонаблюдения, вопрос перестаёт быть вопросом принципа и становится
вопросом процедуры.

## 5. Черновик письма

Не отправлять до закрытия раздела 4. Адресат, обращение и юридические
формулировки подставляются после того, как станет известно, кому пишем.

---

Тема: Запрос на установку измерительного устройства в салоне автобуса, пилот

Уважаемый [должность, фамилия],

Мы разрабатываем систему оценки заполненности салона автобуса, чтобы пассажир
мог до посадки увидеть в приложении, насколько полон подъезжающий автобус.
Проект ведётся в Алматы совместно с [уточнить статус договорённостей].

Для настройки и проверки системы нам нужен ограниченный сбор видеоданных в
салоне. Просим рассмотреть возможность установки камеры под потолком в
[количество] автобусах маршрута [номер] на период [срок].

Что важно про приватность. Система считает количество людей и не производит
распознавание лиц, идентификацию пассажиров или сопоставление между поездками.
В рабочем режиме видео не покидает устройство. На этапе сбора данных записи
хранятся в ограниченном периметре, доступ имеет поимённый список сотрудников,
записи удаляются по истечении согласованного срока. Для любых публикаций
изображения обезличиваются.

Что получает оператор. Объективные данные о фактической загруженности рейсов
по времени суток, которые сейчас недоступны, и результаты пилота до их
публикации.

Готовы согласовать порядок информирования пассажиров, срок хранения и любые
ограничения на съёмку в удобной для вас форме. Также готовы встроиться в уже
действующий у вас порядок согласования видеонаблюдения, если он есть.

Будем признательны за встречу или письменный ответ о принципиальной
возможности.

С уважением,
Дияс Тлеукин
[контакты]

---

## 6. Чего в письме быть не должно

- Обещаний работающего устройства и любых цифр точности. Их нет.
- Слов про эксперимент над пассажирами. На этом этапе запрашивается только
  сбор данных, поведенческий эксперимент это отдельный разговор и отдельное
  согласование.
- Расплывчатых формулировок про приватность. Именно они вызывают отказ.
- Просьбы о доступе ко всему парку. Запрос должен быть узким.

</details>

<details>
<summary><code>business/INNOFORCE_QUESTIONS.md</code> - original text</summary>

# Innoforce и оператор: что надо спросить

Статус: **черновик повестки, не отправлено, встреча не назначена**
Дата: 2026-09-03
Authority: подчиняется [`../GROUND_TRUTH.md`](../GROUND_TRUTH.md)
Основа: `research/RTCI_RESEARCH_CHARTER.md` раздел 12,
[`../PRODUCT_SPEC.md`](../PRODUCT_SPEC.md) раздел 9

## 0. Зачем этот файл

Innoforce находится на критическом пути и продукта, и исследования. Показ идёт
через Avtobys, значит без них устройство никому ничего не показывает. Половина
открытых пунктов PRODUCT_SPEC закрывается не работой, а одним разговором.

Вопросы упорядочены по тому, насколько сильно ответ меняет план. Первые четыре
блокируют всё остальное. Под каждым вопросом записано, что будет при каждом
ответе, чтобы разговор не свёлся к сбору мнений.

Формат ведения: после встречи ответы вписываются сюда с датой и именем
ответившего. Устный ответ, не записанный сюда, не существует.

---

## 1. Блокирующие вопросы

### 1.1 Может ли Avtobys показывать функцию части пользователей

Точнее: можно ли назначить стабильную группу пользователей, которая видит
загруженность, и группу, которая не видит, и удержать это назначение во
времени.

Почему критично: это единственное, что отличает эксперимент от наблюдения. Без
рандомизации причинного вывода не будет, и статья теряет главный вклад.

| Ответ | Что делаем |
|---|---|
| Да, по пользователю | Основной дизайн, рандомизация по стабильному псевдонимному ID |
| Нет, но можно по остановке или времени | Cluster-randomized crossover, чередование заранее |
| Нет вообще | Остаётся квази-эксперимент до и после, в результатах только «связано с», не «вызывает» |

### 1.2 Можно ли наблюдать факт посадки

Главный операционный блокер всего исследования. Без наблюдаемого исхода нельзя
считать размер выборки и нельзя обещать причинный результат.

Варианты по убыванию силы, спросить про каждый отдельно:

1. Связка показа в приложении с фактической валидацией проезда по
   псевдонимному ID.
2. Определение по телефону: геозона на остановке плюс движение вместе с GPS
   конкретного автобуса после отправления, с согласия пользователя.
3. Вопрос в приложении сразу после: «сели или ждёте». Проще, но это самоотчёт.
4. Только агрегированные счётчики посадок. Годится для эффектов на маршрут, но
   индивидуальный эффект не доказывает.

| Ответ | Что делаем |
|---|---|
| Доступен вариант 1 или 2 | Полноценный field experiment возможен |
| Только 3 | Эксперимент возможен, но исход самоотчётный, это записывается в ограничения |
| Только 4 | Индивидуальный причинный эффект не измеряется, меняется вопрос исследования |

### 1.3 Показывать последнее измерение или прогноз на момент прибытия

PRODUCT_SPEC 9.4. Устройство меряет автобус сейчас. Пассажир на остановке
впереди сядет в него через несколько минут, и за это время загрузка изменится.

Почему критично: от этого зависит, является ли продукт real-time crowding
information или recent crowding information. Прогноз требует второй модели,
посадок и высадок по маршруту, и эти данные есть у Innoforce, а не у нас.

| Ответ | Что делаем |
|---|---|
| Показываем последнее измерение | Проще, но надо честно назвать это в UI и в статье |
| Нужен прогноз на момент прибытия | Появляется вторая модель и потребность в исторических данных по посадкам |

### 1.4 Сколько последовательных автобусов можно оборудовать

Почему критично: смысл вмешательства в выборе между этим автобусом и
следующим. Если оборудован один автобус, показать загруженность следующего
нечем, и вся конструкция рассыпается.

| Ответ | Что делаем |
|---|---|
| Несколько подряд на одном маршруте | Работает как задумано |
| Один или два | Нужен другой источник загруженности следующего автобуса, иначе пилот не про выбор |

---

## 2. Данные

### 2.1 Какие операционные данные доступны

Спросить по каждому пункту отдельно, ответ «в целом да» бесполезен:

- GPS и AVL автобусов, частота обновления;
- расписание и расчётное время прибытия, включая историю прогнозов;
- события остановок, открытие и закрытие дверей;
- валидации проезда с временем;
- APC, если он где-то стоит;
- паспортная вместимость и число сидений по моделям автобусов;
- фактические интервалы движения и сбои.

Паспортные данные из последнего пункта нужны раньше всех остальных: без них
скор `0..1` не определён (PRODUCT_SPEC 2.4).

### 2.2 Логи показа функции

Записывает ли приложение, что карточка загруженности была реально показана и
видима, а не просто доступна. Разница между назначением и экспозицией это
разница между ITT и per-protocol анализом.

### 2.3 Кому принадлежат собранные данные

Видеозаписи из салона, разметка, обученные веса. Спросить до сбора, а не
после. Ответ влияет и на статью, и на возможность продать продукт дальше.

---

## 3. Интеграция

- Транспорт: MQTT, HTTP push или опрос. Что уже используется в их инфраструктуре.
- Авторизация устройств, выдача и ротация ключей.
- Кто держит соответствие устройства и борта.
- Их собственный TTL: как долго приложение считает значение свежим.
- Как в интерфейсе показывается состояние «нет данных». В контракте оно
  обязано быть явным значением (PRODUCT_SPEC 6).
- Бюджет задержки от публикации до появления в приложении.
- Есть ли у них тестовый контур, куда можно слать данные до реального пилота.

Черновик схемы сообщения лежит в PRODUCT_SPEC 7.1. Показать его на встрече и
получить правки, а не согласовывать словами.

---

## 4. Пилот и разрешения

- Кто даёт разрешение на съёмку в салоне: Innoforce, оператор маршрута или
  акимат. От ответа зависит, кому адресовать запрос.
- Как информируются пассажиры о съёмке и о том, что идёт исследование.
- Кто согласовывает приватность и обработку изображений.
- Какой маршрут они предложили бы: нужен короткий интервал, заметная разница
  загрузки между рейсами и несколько машин подряд.
- Готовы ли они распространить анкету волны 2 через приложение. Это точное
  попадание в ту же аудиторию, которую потом накроет эксперимент.
- Какой минимальный эффект им практически интересен. Нужно для расчёта размера
  выборки, и это вопрос к ним, а не к нам.

---

## 5. Чего на встрече не делать

- Не обещать работающее устройство. Его нет.
- Не называть точность. Ни одна модель текущей архитектуры не обучалась.
- Не соглашаться на запуск показа пассажирам до прохождения гейта валидации
  измерений (PRODUCT_SPEC 8.3). Неточный показ ломает и продукт, и эксперимент.
- Не уходить без письменных ответов хотя бы на 1.1, 1.2 и 1.4.

---

## 6. Ответы

Заполняется после встречи.

| № | Вопрос | Ответ | Кто ответил | Дата |
|---|---|---|---|---|
| 1.1 | Рандомизация показа | | | |
| 1.2 | Наблюдение посадки | | | |
| 1.3 | Измерение или прогноз | | | |
| 1.4 | Число оборудованных автобусов | | | |
| 2.1 | Операционные данные | | | |
| 2.2 | Логи экспозиции | | | |
| 2.3 | Права на данные | | | |
| 3 | Интеграция | | | |
| 4 | Разрешения и маршрут | | | |

</details>

<details>
<summary><code>business/INNOFORCE_RTCI_PILOT_BRIEF.md</code> - original text</summary>

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

</details>

<details>
<summary><code>business/outreach/template.md</code> - original text</summary>

# SANASH cold email template

Supersedes `email_template.md`, which asked for collaboration and used a
first-name salutation. Both are now forbidden. That file is kept for history in
`ARCHIVE.md` section 5.

Every paper named in an email must already be a row in
`research/refs/base/references.md`. If it is not there, stop, verify it against
a live page, add it with its DOI, and only then write the email. See the
citation rule in the repository `CLAUDE.md`.

---

## Template

```
Subject: [5 to 8 words, specific, naming their work]

Dear Professor [Surname],

[Sentence 1 to 3. Their paper, by title, and the specific thing it found or
measured. This is the opening. Not the student, not SANASH.]

[Sentence 4 to 6. The one point where that result meets a decision in a field
study of bus crowding information in Almaty, run with the city bus operator.]

[One sentence saying who is writing and what the study is. Short.]

[Closing sentence: one question, answerable in a paragraph.]

Best regards,
Diyas Tleukin
```

## Style rules

- Around 130 words. Never over 160.
- Open with the recipient's work. The student comes later, in one sentence.
- Subject line 5 to 8 words, specific, naming their work or its finding. Never
  the word "inquiry".
- No em dashes.
- No colons in prose. Colons in the subject line only if unavoidable.
- End by asking for an answer to one question. Never ask for collaboration,
  supervision, a position, a reference, or a call.
- No praise that is not a statement of fact about the work.
- No fabricated personal connection.

## Salutation

- Professor: `Dear Professor [Surname],`
- Doctorate but not a professor: `Dear Dr [Surname],`
- PhD student or no doctorate: `Dear [Full Name],`
- Never `Hi [first name]`. Not in a first email, not in a reply, not when the
  person signed with their first name.
- Never guess gender. Never use a pronoun for the recipient.

## What the study is allowed to be called

True today, and the ceiling of what an email may claim:

> a field study of real-time bus crowding information in Almaty, run with the
> city bus operator

Forbidden, because they are not true:

- a submission to Transportation Research Part C, or to any named journal
- a launch across 25 cities, or any city count
- a partnership with a state transportation company stated as a signed deal
- any accuracy figure, level threshold or deployment claim for the device

The device is not built or trained. Do not describe it as working.

## After sending

Flip the cited paper's row in `research/refs/base/references.md` from `LISTED`
to `CITED (Surname)` in the same commit as the email draft. A paper named in a
sent email that still reads `LISTED` is a bookkeeping failure, and the reference
base stops being trustworthy the moment that happens.

</details>
