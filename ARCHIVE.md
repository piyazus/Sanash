# Sanash: архив

Этот файл собирает документы, которые уже помечены как история и не описывают
текущее состояние. Ни один из них не является current spec. Единственный
источник истины о текущем состоянии: [`GROUND_TRUTH.md`](GROUND_TRUTH.md).

Собран 2026-09-03 из пяти отдельных файлов, чтобы они не лежали вперемешку с
живыми. Содержание перенесено целиком, без сокращений. Исходный путь каждого
раздела указан в HTML-комментарии перед ним.

## Что внутри

| Раздел | Исходный файл | Почему мёртв |
|---|---|---|
| 1 | `SANASH_Master_Document.md` | пивот 2026-08-25, дверной ToF и двухфазная схема отменены |
| 2 | `development/findings.md` | измерения door/depth архитектур до пивота |
| 3 | `development/sanas_cv_track_conclusions.md` | отменённая архитектура, RealSense и ToF |
| 4 | `research/paper/related_work_notes.md` | отменённый door/APC paper track |
| 5 | `business/outreach/email_template.md` | заменён `business/outreach/template.md` |
| 6 | `research/coursework/RESEARCH_METHOD_GROUNDING.md` | заменён `research/coursework/WRITING_RULES.md` |
| 7 | `research/PAPER_SPRINT_30D.md` | заменён `SEPTEMBER_PLAN.md` 2026-09-04 |


---

<!-- was: SANASH_Master_Document.md -->

**Раздел 1.** Мастер-документ v1.0 от 6 августа 2026.

## SANASH — Мастер-документ проекта

> **ИСТОРИЧЕСКИЙ ДОКУМЕНТ. НЕ ИСПОЛЬЗОВАТЬ КАК CURRENT SPEC.** Более ранний
> staleness banner ниже тоже устарел после пивота 2026-08-25: дверной ToF и
> двухфазная схема отменены. Текущий источник истины:
> [`GROUND_TRUTH.md`](GROUND_TRUTH.md).

**Версия 1.0 · 6 августа 2026 · Дияс Тлеукин**
Горизонт документа: **30 дней** (до 5 сентября 2026). Дальше — пересборка.

---

### 0. КАК МЫ РАБОТАЕМ

**Правила, которые я соблюдаю.**

1. **Ничего не выдумываю.** Каждый email, DOI, цена и габарит либо проверен на живой странице, либо помечен «не подтверждён». Если проверить не смог — пишу прямо, а не заполняю правдоподобным.
2. **Возражаю, когда не согласен.** Если план плохой — говорю это до того, как ты потратишь на него деньги или неделю.
3. **Отдаю готовые артефакты, а не планы планов.** Файл, письмо, STL, скрипт. Не «вот как можно было бы».
4. **Разделяю факт и оценку.** Измеренное — «измерено». Экстраполированное — «оценка, вот от чего».

**Правила, которые соблюдаешь ты.**

1. Один источник правды — этот документ. Расхождения между ним и другими доками разрешаются в его пользу или он правится.
2. Решения фиксируются здесь в разделе 11, а не в переписке.
3. Если что-то из плана не сделано — это не переносится молча, а помечается и пересматривается.

**Что я о тебе знаю и учитываю.** Ты сам себя ловил на том, что месяц полировал план с LLM и ничего вне документа не менялось. Поэтому этот документ короткий на рассуждения и длинный на «что заказать» и «кому написать». Раздел 4 — единственный, который надо перечитывать ежедневно.

---

### 1. НАУЧНЫЙ ВОПРОС

#### 1.1 Формулировка

> **Меняет ли пятиуровневая информация о заполненности автобуса, полученная пассажиром ДО посадки, его решение садиться или ждать следующий?**
>
> Подвопросы: как эффект зависит от **интервала движения** (headway) и от **цели поездки** (trip purpose)?

Английская версия для статьи:

> *Does pre-boarding five-level real-time crowding information change bus passengers' boarding and waiting behaviour, and how does that effect vary with headway and trip purpose?*

#### 1.2 Почему это не занято

Проверено по литературе (раздел 12, 40 источников):

| Что есть | Чего нет |
|---|---|
| Симуляции: 30–70% вероятность намеренно пропустить переполненный автобус (Drabicki, Cats, Kucharski 2022) | Полевой проверки этого числа на автобусах |
| Единственный полевой пилот RTCI — метро Стокгольма, 6 дней, ~25% заметили, −4.3 п.п. в самый забитый вагон (Zhang, Jenelius, Kottenhoff 2017) | Выбор там был между **вагонами одного поезда**, а не «сесть или ждать» |
| SP-опросы о готовности ждать при RTCI (Drabicki et al. 2023) | Реального поведения, а не деклараций |
| 4-уровневая классификация платформы метро, точность **63.4%** (Fiorista, Abdelhalim, Zhao et al. 2025, MIT+WMATA) | Пяти уровней, салона автобуса, и доставки пассажиру **до посадки** |
| 4 уровня в салоне автобуса (Meghana, Charniya et al. 2020, ICESC) | Это шестистраничный студенческий доклад без метрик в открытом доступе |

**Формулировка новизны в одну строку:** никто не измерил в поле, меняет ли дискретная многоуровневая шкала заполненности **автобуса**, показанная **до посадки**, реальное поведение пассажира.

#### 1.3 Планка

| Метрика | Лучшее опубликованное | Наша цель |
|---|---|---|
| Точность классификации уровня заполненности | 63.42% на 4 класса (Crowd-ViT, MIT+WMATA) | **≥70% на 5 классов** |
| Полевая точность коммерческих APC | 53–55% (Pronello, Турин) | — |
| MAE подсчёта в салоне | ~1 пассажир при ≥25 (Hsu, Perng, Гаосюн, 2 камеры) | **≤3 пассажира** |

70–80% на пяти классах — публикуемый результат. Планка низкая, потому что задача объективно тяжёлая: окклюзия в салоне убивает детекторы.

#### 1.4 Скрытый вклад, который сильнее основного

Pi, Qian, Steinfeld (TRR 2018) и Kovačević et al. (Sci Reports 2026) показали: **восприятие заполненности систематически расходится с фактическим числом людей**. Зависит от времени суток, вместимости машины, положения пассажира в салоне, возраста, дохода.

Значит границы пяти уровней нельзя ставить по процентам вместимости (20/40/60/80%). Их надо **калибровать по восприятию**. Это отдельный самостоятельный научный вклад, и он сильнее, чем «мы обучили классификатор».

---

### 2. ПРОДУКТ

**Что строим:** устройство в салоне автобуса. Камера смотрит в проход, вычислитель на борту оценивает заполненность, наружу уходит **одно число** — уровень от 1 до 5. Видео не покидает автобус никогда.

**Принцип в одну фразу:** видео остаётся в автобусе, интеллект уезжает.

**Пять уровней (рабочая шкала, границы калибруются в разделе 1.4):**

| Уровень | Название | Рабочее описание |
|---|---|---|
| 1 | Пусто | Свободных мест много, можно выбирать |
| 2 | Свободно | Сидячие места есть |
| 3 | Мест мало | Последние сидячие, стоячих единицы |
| 4 | Стоя | Сидячих нет, стоять комфортно |
| 5 | Давка | Плотно, посадка затруднена |

**Куда идёт число:** приложение пассажира показывает уровень для каждого подходящего автобуса **до того, как он подъехал**. Это и есть предмет исследования.

---

### 3. ТЕХНИЧЕСКАЯ БАЗА

*(см. примечание в начале документа — раздел ниже частично устарел, актуальная версия в `sanas_cv_track_conclusions.md`)*

#### 3.1 Выбор вычислителя — решение и обоснование

| | Jetson Orin Nano Super | RPi 5 + Hailo-8 |
|---|---|---|
| Цена | $399 | $125 + ~$110 |
| INT8 TOPS | 67 | 26 |
| Тулчейн | PyTorch → ONNX → **TensorRT**, за день | Hailo DFC → HEF, **без fallback** |
| Риск | низкий | **высокий** |

**Решение: Jetson Orin Nano Super.**

Причина — не производительность, а тулчейн. Hailo Dataflow Compiler поддерживает фиксированный список слоёв и **не имеет автоматического fallback**: если операция не поддержана, компиляция просто падает, инференс уходит на CPU без ускорения. У P2PNet нестандартная голова (генерация опорных точек + регрессия смещений). Это не «портировать за вечер», а одна-две недели с непредсказуемым исходом. У тебя месяц.

Экономия $517 на устройство не окупает риска, что модель вообще не заведётся.

#### 3.2 Ожидаемая производительность

Оценка, не измерение. Якорь — измеренный бенчмарк YOLOv8n на Orin Nano Super (JetPack 6.2.2, MAXN_SUPER): TensorRT FP16 **225 FPS** при 640×640.

Пересчёт на P2PNet с усечённым VGG16 при 1080p: ~370–390 GFLOPs на кадр, пиковая FP16 Orin Nano Super ≈ 16.7 TFLOPS dense, реалистичная утилизация 30–40% → **8–15 FPS**.

Нужен **1 FPS**. Запас 8–15×. Даже если оценка завышена вдвое — запас остаётся.

#### 3.3 Архитектура

```
Камера (RTSP/CSI)
   → аппаратное декодирование (NVDEC)
   → сэмплирование 1 кадр/сек
   → препроцессинг
   → TensorRT FP16 инференс (P2PNet)
   → фильтрация по confidence
   → сырой count
   → temporal smoothing (скользящее окно 10-15 с)
   → маппинг count → уровень 1..5 (границы из калибровки)
   → локальная очередь SQLite (outbox)
   → MQTT over mutual TLS через LTE
   → бэкенд → приложение пассажира
```

**Что наружу уходит:** `{bus_id, timestamp, level, confidence}`. Всё. Ни кадра, ни бокса, ни лица.

#### 3.4 Критерии приёмки

| Метрика | Порог |
|---|---|
| Точность уровня (5 классов) | ≥70% |
| MAE подсчёта | ≤3 пассажира |
| Интервал обновления | ≤10 с |
| Задержка инференса на edge | ≤200 мс |
| Доставка метаданных после ретраев | ≥99% |
| **Сырое видео за пределы автобуса** | **0 кадров** |
| Аптайм за пилот | ≥95% |

---

### 4. ПЛАН НА 30 ДНЕЙ

Четыре трека идут **параллельно**. Трек A блокирует всё остальное по срокам поставки, поэтому он первый по календарю, а не по важности.

#### Неделя 1 (6–13 августа) — заказать и начать обучение

| Трек | Задача | Статус |
|---|---|---|
| A. Железо | Заказать Jetson × 3 у Seeed **и** Silicon Highway параллельно (см. 5.6 — дефицит) | ☐ |
| A | Заказать TRACO DC-DC ×3 + Mean Well ×3 (запас), Teltonika RUT241 ×2 | ☐ |
| A | Купить локально в Алматы: Hikvision DS-2CD2143G2-I ×3, NVMe ×3 | ☐ |
| A | Оформить БИН/ЭЦП, если нет — нужно для регистрации IMEI роутеров | ☐ |
| C. Модель | Скачать BeIntelli (73.5 ГБ, CC BY 4.0), распаковать, разобрать структуру | ☐ |
| C | Аккаунт Vast.ai, проверить, проходит ли карта KZ (иначе крипта) | ☐ |
| D. Профессора | Отправить письма №1–5 из рейтинга | ☐ |

#### Неделя 2 (13–20 августа) — baseline

| Трек | Задача | Статус |
|---|---|---|
| B. CAD | Получить платы, **обмерить штангенциркулем**, поправить параметры в CAD | ☐ |
| B | Напечатать первый корпус ASA, примерить всё железо | ☐ |
| C | Вывести 5-уровневые метки из 3D-боксов BeIntelli + вместимость | ☐ |
| C | Обучить P2PNet на открытых данных, зафиксировать baseline MAE | ☐ |
| D | Письма №6–15 | ☐ |

#### Неделя 3 (20–27 августа) — сборка и классификатор

| Трек | Задача | Статус |
|---|---|---|
| A/B | Собрать стендовое устройство целиком, прогнать сутки на столе | ☐ |
| C | Обучить классификатор 5 уровней, получить точность на held-out | ☐ |
| C | Экспорт ONNX → TensorRT FP16, замерить **реальный** FPS на Jetson | ☐ |
| D | Письма №16–30, ответить всем, кто откликнулся | ☐ |

#### Неделя 4 (27 августа – 5 сентября) — доказательство и выход к автобусу

| Трек | Задача | Статус |
|---|---|---|
| A/B | Собрать 2 полевых устройства, вибротест (просто поездить с ним в машине) | ☐ |
| C | Видеодемо: устройство считает людей в реальном времени | ☐ |
| C | Отчёт: точность, MAE, FPS, задержка — цифры, а не обещания | ☐ |
| D | **С готовым устройством и цифрами — выходить на оператора и акимат** | ☐ |
| D | Письма №31–50 | ☐ |

**Логика последовательности.** К автобусу идём не с презентацией, а с работающей коробкой и измеренными числами. Это единственное, что переводит разговор из «дайте нам попробовать» в «вот прибор, дайте поставить».

---

### 5. ЗАКУПКА

*(см. примечание в начале документа — часть решений по железу устарела, см. `sanas_cv_track_conclusions.md`, раздел 8)*

#### 5.1 Три факта, которые меняют планирование

1. **NVIDIA подняла цены на Jetson 22 июля 2026 на 33–101%.** Orin Nano Super Dev Kit: $249 → **$399**. На NVIDIA Marketplace — **Out of Stock**. Причина — дефицит LPDDR.
2. **Raspberry Pi подорожал дважды за 3 месяца.** Pi 5 8GB $95 → $125, 16GB $145 → $205. План Б тоже подорожал.
3. **Главный риск — не цена, а наличие.** Заказывай параллельно у двух поставщиков и отменяй тот, что придёт позже.

#### 5.2 PRIMARY BUILD — одно устройство

| Позиция | Артикул | Цена | Проверено |
|---|---|---|---|
| Вычислитель | Jetson Orin Nano Super Dev Kit 8GB, `945-13766-0005-000` | **$399** | да (MSRP, SparkFun, Silicon Highway €348 ex VAT) |
| SSD | NVMe M.2 2280 M-key 500 ГБ (локально) | ~$60 | оценка |
| Камера | Hikvision DS-2CD2143G2-I (2.8mm), 4MP, 103° | **$140** (65 618 ₸) | да, voltmaster.kz Алматы |
| DC-DC | TRACO TEN 40-2412WIR, 9–36 В → 12 В/40 Вт, **EN 50155** | **$111.62** | да, TRC Electronics |
| LTE | Teltonika RUT241 Global, вход **9–30 В** | **$176** (€162.47 ex VAT) | да, Getic |
| Защита входа | предохранитель 5 А + TVS SMDJ33A + диод + CM-дроссель | ~$40 | оценка |
| Буфер/shutdown | supercap UPS-модуль + оптопара PC817 | ~$35 | оценка |
| Корпус | 3D-печать ASA + вставки M3 + виброгасители | ~$25 | оценка |
| Кабели/разъёмы | Deutsch DT, CAT5e SF/UTP, антенна SMA | ~$57 | оценка |
| SIM (мес.) | Tele2/Kcell IoT | ~$11 | да (590–4 990 ₸) |
| **ИТОГО** | | **≈ $1 055** | |

#### 5.3 Смета пилота

| Позиция | Кол-во | Сумма |
|---|---|---|
| Полевое устройство (полная сборка) | 2 | $2 110 |
| Стендовое (devkit + SSD + камера) | 1 | $599 |
| Запчасти (DC-DC, камера, крепёж) | — | $180 |
| Доставка DHL, 2 посылки | — | $120 |
| Пошлина 5% + НДС 16% на ~$1 600 импорта | — | $330 |
| Регистрация IMEI (2 роутера) | — | $18 |
| Облачное обучение (Vast.ai 4090, 40 ч) | — | $35 |
| SIM 3 месяца × 2 | — | $65 |
| **ИТОГО** | | **≈ $3 457** |
| **Остаток от $4 000** | | **≈ $543** |

Запас $543 разумно потратить на **один Pi5 + Hailo как research-трек** ($538) — если Hailo заведётся, себестоимость серии падает вдвое, а если нет, ты потерял неделю на параллельной ветке, а не на основной.

#### 5.4 Питание — что важно не упустить

Jetson Orin Nano Dev Kit по carrier board принимает **9–20 В**. Борт автобуса — 24 В с тяжёлыми переходными процессами. Между ними обязателен изолированный DC-DC.

**Честная оговорка:** ни один доступный к покупке модуль не имеет явного сертификата **ISO 7637-2**. TRACO TEN 40WIR имеет **EN 50155** (железнодорожный) — по устойчивости к переходникам сопоставим или строже. Это лучшее, что реально купить на этой неделе.

**Обвязка на входе 24 В обязательна** (~$40): предохранитель ATO 5 А → диод обратной полярности → **TVS SMDJ33A/36A** (гасит load dump) → CM-дроссель + X/Y конденсаторы.

**Ignition-sense:** линия зажигания 24 В → делитель → оптопара PC817 → GPIO Jetson. Демон: нет зажигания >20 с → `systemctl poweroff`. Суперконденсаторный модуль держит шину ~60 с, чтобы дописать на NVMe.

**Не бери LiFePO4 для пилота.** Суперконденсатор: нет деградации при −20 °C (алматинская зима), нет вопросов от пожарного инспектора автопарка, нет второго зарядного контура при 24 В борте.

#### 5.5 ⚠️ Регуляторный флаг: IMEI

В Казахстане **любое устройство с SIM-картой** должно пройти верификацию IMEI на `kz-imei.kz` или через eGov Mobile в течение **30 календарных дней с первого подключения**, иначе IMEI попадает в чёрный список и связь блокируется. Роутер RUT241 под это попадает. Для юрлица: БИН + ЭЦП + импортные документы, пошлина ~1 МРП за устройство.

**Следствие:** роутер надо ввозить «в белую», с инвойсом и декларацией. Значит разбивать посылки по <€200 под лимит физлица не получится — по крайней мере для роутера.

#### 5.6 Логистика и налоги

- **НДС в Казахстане с 1 января 2026 — 16%** (было 12%).
- **С 1 июля 2026** правила ЕАЭС для e-commerce: беспошлинно до €200 и 31 кг; свыше — 5% от стоимости + национальный НДС 16%.
- **Astana Hub:** если компания резидент — **освобождение от НДС при импорте** по перечню. Минус 16% на всё железо. Проверить, попадает ли оборудование в перечень.
- **chipdip.kz продаёт Jetson Orin Nano DevKit за 610 000 ₸ ≈ $1 300** — наценка ×3 к MSRP. **Не покупать.**
- **End-user screening.** Казахстан под усиленным вниманием по реэкспорту в РФ. Mouser/DigiKey/Arrow при первом заказе пришлют анкету о конечном использовании. Заложи +2–5 рабочих дней.

#### 5.7 Что заказать в понедельник

**Онлайн:**
1. Seeed Studio: Jetson Orin Nano Super Dev Kit (лимит **1 devkit на аккаунт** — используй 3 аккаунта, либо 1 devkit + 2× reComputer Super J3011 по $796.99, который **в наличии** и уже с корпусом и 128 ГБ SSD)
2. Silicon Highway (Ирландия): дубль-заказ 2× devkit €348 ex VAT
3. TRC Electronics: 3× TRACO TEN 40-2412WIR + 3× Mean Well DDR-30G-12 ($35 не жалко на запас)
4. Getic: 2× Teltonika RUT241 Global
5. Waveshare: 2× IMX219-160 ($14 — чтобы начать отладку до приезда Hikvision)

**Локально, вторник:**
6. voltmaster.kz: 3× Hikvision DS-2CD2143G2-I — 196 854 ₸
7. Любой компьютерный: 3× NVMe M.2 2280 500 ГБ
8. БИН/ЭЦП для kz-imei.kz

---

### 6. CAD

Готово: параметрическая модель на CadQuery, файлы `sanash_enclosure_base.stl/.step`, `sanash_enclosure_lid.stl/.step`, исходник `sanash_enclosure.py`.

**Габариты:** наружный **226 × 176 × 108 мм**, внутренний 220 × 170 × 100 мм.

**Что заложено:**

| Решение | Почему |
|---|---|
| Стенка 3 мм, дно 4 мм | дно несёт вибрационную нагрузку |
| Только **heat-set вставки M3**, никакой печатной резьбы | печатная резьба в ASA развалится от вибрации за месяц |
| Материал **ASA или PC**, не PLA | летом за панелью в салоне бывает 60–70 °C, PLA поплывёт |
| Канавка под уплотнительный шнур 3 мм по торцу | пыль |
| 4 монтажных уха M5 под силиконовые виброгасители | болтовое соединение без демпфера открутится |
| Жалюзи на боковых стенках | Jetson без обдува уходит в троттлинг |
| Ложемент RUT241 с двумя щеками + стяжка | роутер стоит на боку, он самый высокий элемент и задаёт 100 мм |
| Площадка на крышке над Jetson под термопрокладку | дополнительный путь отвода тепла в корпус |
| 5 кабельных вводов в задней стенке | 2× PG9 (питание, зажигание), 1× PG11 (Ethernet), 2× SMA (антенны) |

**Что надо сделать руками до печати.** Координаты крепёжных отверстий плат в модели — **параметры, а не факт**. В datasheet NVIDIA есть габарит платы, но не координаты отверстий. Когда придёт железо: обмерить штангенциркулем, вписать в блок параметров вверху `sanash_enclosure.py`, перегенерировать. Это одна команда.

**Раскладка внутри:**
- Jetson (100×79×21) — плашмя на дне слева спереди, на стойках 8 мм
- RUT241 (83×25×74) — вертикально справа сзади
- TRACO DC-DC (51×26) — слева сзади
- Плата защиты входа — в центре сзади

---

### 7. ОБУЧЕНИЕ МОДЕЛИ

#### 7.1 Датасет — с чистого листа

**Ядро: Multi-View In-Cabin Dataset (BeIntelli).** Единственный открытый датасет из реального городского автобуса.

- 9 136 синхронизированных сэмплов; 4 камеры RGB + depth + вращающийся LiDAR
- Псевдо-разметка: 3D-позы + ориентированные 3D bounding boxes на каждого пассажира
- Формат nuScenes; бейзлайны в статье: Lift-Splat-Shoot, BEVFusion
- **CC BY 4.0**, 73.5 ГБ
- Zenodo `10.5281/zenodo.20559664` · HuggingFace `evgenygorelik96/multiview_incabin_dataset`
- Авторы: Gorelik, Karrow, Sivrikaya, Albayrak, Baumann (TU Berlin + GT-ARC + MAN Truck & Bus), arXiv:2606.11739

**Как получить 5-уровневые метки без ручной разметки:** 3D-боксы дают точный count → вместимость автобуса известна → уровень выводится автоматически. Границы уровней потом калибруются по восприятию (раздел 1.4).

**Оговорки:** разметка псевдо, не выверена человеком. Один автобус, одна страна, одна конфигурация камер — генерализация под вопросом. Dataset viewer на HuggingFace сломан (schema mismatch). Работе два месяца, никто её не воспроизводил.

#### 7.2 Стек обучения

```
Претрейн детектора голов:  CrowdHuman (470k людей, 3 бокса на человека)
                           RPEE-Heads (109 913 голов, ж/д платформы, CC BY-SA)
                           WiderPerson
Претрейн density-head:     ShanghaiTech A/B + UCF-QNRF + NWPU-Crowd
Потолочная геометрия:      WEPDTOF / CEPDOF / HABBOF (BU, по запросу, non-commercial)
Indoor фикс. камера:       Mall (indoor, окклюзии) + FDST + CityUHK-X (известны угол и высота камеры)
Синтетика:                 GCC-CL (свой рендер в GTA V) + SVIRO (интерьер авто, готовая дискретная разметка мест)
Целевое дообучение:        BeIntelli
```

#### 7.3 Облачные GPU

| Провайдер | RTX 4090 | A100 80GB |
|---|---|---|
| **Vast.ai** | **$0.37/ч** | $0.67/ч PCIe |
| RunPod | $0.69/ч | $1.39/ч |
| Lambda | — | $2.79/ч |

Дообучение — 20–40 GPU-часов. **Бюджет $10–30.** Vast.ai дешевле всех, но инстансы прерываемые — чекпойнти каждую эпоху.

⚠️ RunPod и Lambda могут отклонять карты, выпущенные в KZ. Vast.ai принимает крипту. Проверь оплату **до** того, как упрёшься в это в три часа ночи.

---

### 8. РЕЙТИНГ ПРОФЕССОРОВ

*(устарел — актуальный список см. `SANASH_consolidated_researcher_list.md`, раздел ниже сохранён как исторический архив методики оценки)*

#### 8.1 Как считается

**Оценка = 0.30 × Универ + 0.40 × Тема + 0.30 × Доступность**

- **Универ (U, 0–10):** мировой ранг и вес имени в заявке на бакалавриат.
- **Тема (T, 0–10):** насколько его работа буквально про твою задачу. 10 = делает ровно это.
- **Доступность (H, 0–10):** должность (assistant/postdoc отвечают чаще full professor), явное приглашение к сотрудничеству на сайте, наличие подтверждённого email, языковой и часовой барьер.

Вес темы выше веса универа сознательно: письмо звезде не по теме уходит в игнор, письмо по теме менее известному человеку даёт соавторство.

**Email:** ✅ = видел на живой странице. Иначе — искать на сайте кафедры перед отправкой, **не выдумывать**.

*(Полная таблица 80 профессоров с расчётом баллов, а также раздел "кому не писать" и порядок волн рассылки — см. архив; в актуальную работу вошли уже провалидированные контакты из `SANASH_consolidated_researcher_list.md`.)*

---

### 9. ШАБЛОН ПИСЬМА

Самый высокий отклик дают письма с конкретным техническим вопросом по их собственной статье плюс одна строка о том, что ты уже сделал. Общие письма «хочу быть вашим студентом» у людей этого уровня уходят в игнор.

```
Subject: Five-level bus occupancy from onboard cameras — a question about [их конкретная работа]

Dear Professor [Фамилия],

I am a final-year secondary school student in Almaty, Kazakhstan, building an
onboard computer-vision system that classifies bus occupancy into five discrete
levels and delivers that level to passengers before they board.

I read [точное название работы, год]. [ОДНО конкретное техническое наблюдение
или вопрос — например: "your Table 3 shows the crowding multiplier saturating
above 4 pass/m²; does that imply the top two levels of a five-level scale are
behaviourally indistinguishable?"]

What I have so far: [одна строка с цифрами — baseline MAE, точность, FPS на
edge. НЕ обещания.]

The gap I am trying to close is that no field study has tested whether a discrete
multi-level occupancy scale, shown to passengers before boarding, changes whether
they board or wait. Simulation puts the skip probability at 30–70%; the only
field pilot was on the Stockholm metro, where the choice was between cars of one
train.

Would you be open to a 20-minute call? Even a two-line reply telling me the
approach is wrong would be valuable.

Diyas Tleukin
Almaty, Kazakhstan
[ссылка на репозиторий или страницу проекта]
```

**Правила.** Одно письмо — один человек — один вопрос. Никаких массовых рассылок под копирку: люди этого уровня узнают шаблон с первой строки. Ссылку на репозиторий или страницу проекта вкладывать обязательно — это единственное доказательство, что ты не просто написал письмо.

---

### 10. РИСКИ

| Риск | Вероятность | Что делаем |
|---|---|---|
| **Jetson не приедет вовремя** (out of stock у NVIDIA, backorder везде) | высокая | Параллельный заказ у Seeed и Silicon Highway. Fallback: reComputer Super J3011 $796.99 — **в наличии**, уже с корпусом и SSD |
| **Псевдо-разметка BeIntelli окажется мусорной** | средняя | Проверить глазами первые 200 кадров ДО обучения. Если плохо — размечать вручную подвыборку в CVAT |
| **Модель не генерализуется** с немецкого автобуса на алматинский | высокая | Это не баг, а **научный результат**: domain gap — публикуемая находка. Плюс domain adaptation через GCC-CL |
| **Доступ к автобусу не дадут** даже с готовым устройством | средняя | План Б: снять на видео салон обычного автобуса как пассажир (это законно) для валидации. План В: троллейбусный парк, маршрутки, частный перевозчик — там решение принимает один человек |
| **Карта KZ не проходит** на RunPod/Lambda | средняя | Vast.ai + крипта. Проверить на этой неделе, а не в момент, когда нужно обучать |
| **IMEI роутера заблокируют** через 30 дней | средняя | Оформить БИН/ЭЦП сейчас, сохранить инвойс и декларацию |
| **Месяца не хватит** | высокая | Приоритет при срыве: **сначала модель и цифры, потом железо**. Цифры на открытом датасете открывают двери; собранная коробка без цифр — нет |

---

### 11. ЖУРНАЛ РЕШЕНИЙ

| Дата | Решение | Обоснование |
|---|---|---|
| 06.08.2026 | Шкала — **5 уровней**, канон | Научная новизна; прямых аналогов почти нет |
| 06.08.2026 | Вычислитель — **Jetson Orin Nano Super** | Тулчейн TensorRT за день против 1–2 недель на Hailo без гарантии |
| 06.08.2026 | Датасет — **BeIntelli (открытый), с чистого листа** | Единственный открытый датасет из реального городского автобуса, CC BY 4.0 |
| 06.08.2026 | Порядок: **железо → CAD → модель → профессора → автобус** | К оператору идём с работающим прибором и цифрами, а не с презентацией |
| 06.08.2026 | Горизонт — **30 дней** | Дальше пересборка документа |
| 10.08.2026 | Камеры — **2× RealSense depth, не 4× Hikvision RGB** | См. `sanas_cv_track_conclusions.md`, разделы 5, 8 — depth-геометрия выбрана основным методом |
| 10.08.2026 | Метод подсчёта — **двухфазный план**: сначала дверной 3D-счётчик, потом RGB-классификация салона | Не требует разрешения на камеры в салоне на первой фазе |

---

### 12. ССЫЛКИ

#### Датасеты

| Датасет | Размер | Лицензия | Ссылка |
|---|---|---|---|
| **Multi-View In-Cabin (BeIntelli)** | 9 136 сэмплов, 4×RGB+depth+LiDAR | CC BY 4.0 | `10.5281/zenodo.20559664`; HF `evgenygorelik96/multiview_incabin_dataset` |
| SFMP (fisheye метро) | 20 390 кадров, rotated bbox | не указана | ieee-dataport.org, DOI 10.21227/edht-6v71 |
| PCDS (двери автобуса, RGB+depth) | 5 464 пары видео | CC BY-NC-SA 3.0 | github.com/shijieS/people-counting-dataset |
| Bus Violence | 1 400 клипов салона | CC BY 4.0 | zenodo.org/records/7044203 |
| PAMELA-UANDES | 348 послед., bbox голов | по запросу | velastin.dynu.com/videodatasets/PAMELA-UANDES/ |
| RPEE-Heads | 1 886 изобр., 109 913 голов | CC BY-SA 4.0 | doi 10.34735/ped.2024.2 |
| CrowdHuman | 24 370 изобр., ~470k людей | research | crowdhuman.org |
| Mall / FDST / CityUHK-X | indoor, фикс. камера | research | см. Awesome-Crowd-Counting |
| SVIRO | 25 000 сцен интерьера авто | CC BY-NC-SA 4.0 | sviro.kl.dfki.de |
| GCC-CL | 15 211 изобр., генератор GTA V | research | github.com/gjy3035/GCC-CL |

#### Ключевые статьи

**Дискретные уровни заполненности**
1. Wang, Cai, Zheng, Fang (2018). Bus-Crowdedness Estimation by Shallow CNN. *SNSP*, 105–110. doi:10.1109/SNSP.2018.00029
2. Meghana, Sarode, Tambade, Marathe, Charniya (2020). Automated Crowd Management in Bus Transport Service. *ICESC*, 104–109. doi:10.1109/ICESC48915.2020.9155692
3. Fiorista, Abdelhalim, Stewart, Pincus, Thistle, Zhao (2025). CCTV Data as an Emergent Data Source for Urban Rail Platform Crowding Estimation. arXiv:2508.03749
4. Huang, Tsuge, Chen, Okoshi, Nakazawa (2022). A Bus Crowdedness Sensing System Using Deep-Learning Based Object Detection. *IEICE Trans.* E105.D(10), 1712–1720. doi:10.1587/transinf.2022PCP0007

**Подсчёт в транспорте**
5. Hsu, Chen, Perng (2020). Estimation of the Number of Passengers in a Bus Using Deep Learning. *Sensors* 20(8), 2178. doi:10.3390/s20082178
6. Velastin, Fernández, Espinosa, Bay (2020). Detecting, Tracking and Counting People Getting On/Off a Metropolitan Train. *Sensors* 20(21), 6251. doi:10.3390/s20216251
7. Pronello, Garzón Ruiz (2023). Evaluating Video-Based Automated Passenger Counting Systems in Real-World Conditions. *Sensors* 23(18), 7719. doi:10.3390/s23187719
8. Mokin, Gayer, Ingacheva, Sheshkus, Arlazarov (2025). BPCS: Multi-view Bus Passenger Counting System. *ICPR 2024 WS*. doi:10.1007/978-3-031-87663-9_22
9. Mazzeo, Contino, Spagnolo, Distante, Stella, Nitti, Renò (2020). MH-MetroNet. *J. Imaging* 6(7), 62. doi:10.3390/jimaging6070062
10. Diaz-Santos, Caballero-Gil, Caballero-Gil (2025). Real-Time Passenger Flow Analysis in Tram Stations Using YOLO and Edge AI on Jetson Nano. *Computers* 14(11), 476. doi:10.3390/computers14110476
11. Rostamza, Del Re, Varughese, Olaverri-Monreal (2026). Optimising CSRNet with parameter-free attention for crowd counting in public transport. arXiv:2605.18349
12. Wu, Xie, Qin, Wang (2026). Multi-level Attention Network for Crowd Counting in Rail Transit. *Machine Intelligence Research* 23(1), 227–246. doi:10.1007/s11633-025-1581-1
13. Gorelik, Karrow, Sivrikaya, Albayrak, Baumann (2026). Multi-View In-Cabin Monitoring System for Public Transport Vehicles. arXiv:2606.11739

**Восприятие заполненности — ядро обоснования 5 уровней**
14. Pi, Qian, Steinfeld, Huang (2018). Understanding Human Perception of Bus Fullness. *TRR* 2672(8), 475–484. doi:10.1177/0361198118781398
15. Kovačević, Pitka, Ivetić, Dedeić, Miličić, Majstorović (2026). Passenger perception of vehicle occupancy in public transport. *Scientific Reports*. doi:10.1038/s41598-026-43541-5
16. Pan, Wu, Lamorea, Li, Sun, Mäkelä (2023). Passenger Perceptions, Information Preferences, and Usability of Crowding Visualizations. *CHI 2023*. doi:10.1145/3544548.3581241

**RTCI и поведение — вторая половина работы**
17. Zhang, Jenelius, Kottenhoff (2017). Impact of real-time crowding information: a Stockholm metro pilot study. *Public Transport* 9(3), 483–499. doi:10.1007/s12469-016-0150-y
18. Drabicki, Kucharski, Cats, Szarata (2021). Modelling the effects of real-time crowding information. *Transportmetrica A* 17(4), 675–713
19. Drabicki, Kucharski, Cats (2022). Mitigating bus bunching with real-time crowding information. *Transportation* 50(3), 1003–1030
20. Drabicki, Cats, Kucharski, Fonzone, Szarata (2023). Should I stay or should I board? *Res. Transp. Business & Mgmt* 47, 100963. doi:10.1016/j.rtbm.2023.100963
21. Jenelius (2020). Personalized predictive public transport crowding information. *TR Part C* 117, 102647
22. Wang, Chen, Zheng, Cheng, Wang, Lei (2021). Providing real-time bus crowding information for passengers. *TR Part A* 148, 316–329
23. Brakewood, Macfarlane, Watkins (2015). The impact of real-time information on bus ridership in NYC. *TR Part C* 53, 59–75
24. Brakewood, Barbeau, Watkins (2014). An experiment evaluating the impacts of real-time transit information, Tampa. *TR Part A* 69, 409–422
25. Watkins, Ferris, Borning, Rutherford, Layton (2011). Where Is My Bus? *TR Part A* 45(8), 839–848
26. Steinfeld, Zimmerman, Tomasic, Yoo, Aziz (2011). Mobile transit information from universal design and crowdsourcing. *TRR* 2217, 95–102
27. Hörcher, Graham, Anderson (2017). Crowding cost estimation with large scale smart card and vehicle location data. *TR Part B* 95, 105–125
28. Bansal, Hörcher, Graham (2022). A dynamic choice model to estimate the user cost of crowding. *JRSS Series A* 185(2), 615–639
29. Yap, Cats, van Arem (2020). Crowding valuation in urban tram and bus transportation. *Transportmetrica A* 16(1), 23–42
30. Tirachini, Sun, Erath, Chakirov (2016). Valuation of sitting and standing in metro trains. *Transport Policy* 47, 94–104
31. Fonzone, Schmöcker (2014). Effects of transit real-time information usage strategies. *TRR* 2417, 13–19

**Обзоры**
32. Radovan, Mršić, Đambić, Mihaljević (2024). A Review of Passenger Counting in Public Transport Concepts. *Eng* 5(4), 172. doi:10.3390/eng5040172
33. Kuchár, Pirník, Janota, Malobický, Kubík, Šišmišová (2023). Passenger Occupancy Estimation in Vehicles: A Review. *Sustainability* 15(2), 1332. doi:10.3390/su15021332
34. Darsena, Gelli, Iudice, Verde (2020). Sensing Technologies for Crowd Management in Public Transportation Systems: A Review. **arXiv:2009.12619** — список литературы этого обзора стоит пройти целиком

**Методическое ядро CV**
35. Zhang, Zhou, Chen, Gao, Ma (2016). Single-Image Crowd Counting via Multi-Column CNN. *CVPR 2016*
36. Ma, Wei, Hong, Gong (2019). Bayesian Loss for Crowd Count Estimation with Point Supervision. *ICCV 2019 oral*
37. Wang, Liu, Samaras, Hoai (2020). Distribution Matching for Crowd Counting. *NeurIPS 2020*
38. Song, Wang et al. (2021). Rethinking Counting and Localization in Crowds (P2PNet). *ICCV 2021 oral*
39. Liang, Chen, Xu, Zhou, Bai (2022). TransCrowd: weakly-supervised crowd counting with transformers. *Science China Inf. Sci.*
40. Liang, Xie, Zou, Ye, Xu, Bai (2023). CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model. *CVPR 2023*
41. Wang, Gao, Lin, Yuan (2019). Learning from Synthetic Data for Crowd Counting in the Wild. *CVPR 2019*
42. Sam, Surya, Babu (2017). Switching CNN for Crowd Counting. *CVPR 2017* — классификатор выбирает регрессор под плотность; идейно ровно дискретизация на уровни
43. Zhang, Chan (2022). Calibration-Free Multi-view Crowd Counting. *ECCV 2022*

---

### Оговорки о достоверности

- **Все FPS для P2PNet — инженерная экстраполяция** от измеренного бенчмарка YOLOv8n на Orin Nano Super, а не прямое измерение. Реальные цифры получишь на неделе 3.
- **Не подтверждены прямо у вендора:** цена Mini-Box DCDC-USB-200, цена supercap-модуля, габариты Mean Well DDR-30G-12 (страница продавца противоречит форм-фактору — сверяй по datasheet), локальная цена NVMe в Алматы, сроки DHL Seeed→Алматы, цены Hammond 1557.
- **Координаты крепёжных отверстий плат в CAD — параметры, а не факт.** Обмерить перед печатью.
- **Email профессоров:** ✅ = видел на живой странице в августе 2026. Остальные искать на сайте кафедры перед отправкой.
- **DOI не верифицированы:** журнальная версия Darsena et al.; статья в *Optik* (2020); Lin, Chen & Wang (только SSRN).


---

<!-- was: development/findings.md -->

**Раздел 2.** Измерения архитектур до пивота 2026-08-25.

## Sanas — findings so far

> **ИСТОРИЯ, НЕ CURRENT SPEC.** Этот файл фиксирует измерения архитектур до
> пивота 2026-08-25. Door/depth выводы не являются результатами текущей
> потолочной RGB-системы. Актуальное состояние: [`../GROUND_TRUTH.md`](GROUND_TRUTH.md).

Everything here is measured or cited. Numbers without a source are not in this
document. Where something is a guess or an untuned default it says so.

Written 2026-08-10. Companion documents: `datasets.md` (external data
provenance), `../experiments/log.md` (run log).

---

### 1. Headline

Three things changed the plan.

1. **The substitute dataset is not a crowding dataset.** It tops out at four
   occupants, and the public field has nothing better — this is representative,
   not unlucky.
2. **P2PNet cannot ship.** Its licence restricts use to academic research, and
   Sanas ships inside Avtobys. The same applies to most crowd-counting
   benchmarks.
3. **Depth-based occupancy from one in-cabin camera does not work.** Measured:
   10% floor coverage, Pearson 0.184, and the score stops rising past two
   occupants.

---

### 2. Substitute dataset (Zenodo 10.5281/zenodo.20559664)

Read by HTTP range requests against the ZIP central directory. The 73.5 GB
archive was never downloaded in full.

#### Record

| Field | Value |
|---|---|
| Title | Multi-View In-Cabin Monitoring System for Public Transport Vehicles |
| Licence | CC BY 4.0 |
| Files | one: `beintelli_v1.zip`, 73,515,214,805 bytes |
| md5 | `74924b5be59e706a5a89affba48f6b87` (matches `development/scripts/download_dataset.sh`) |
| Members | 298,694, 133,366,903,156 bytes uncompressed |
| Published | 2026-06-05 |

Paper: arXiv 2606.11739v1, five authors (GT-ARC / TU Berlin, plus MAN Truck &
Bus SE). The Zenodo record lists one creator; the paper lists five, so
"Gorelik et al." is correct.

#### Contents

| Path | Files | Size | Content |
|---|---|---|---|
| `body_3d/` | 52,636 | 66.30 GB | ~1.26 MB JSON per person per frame |
| `lidar/` | 10,035 | 42.09 GB | Ouster OS0-128, 4,194,304 B each |
| `cameras/color/` | 50,578 | 13.01 GB | 4 RGB cameras, 1280x720 JPEG |
| `cameras/depth/` | 51,640 | 11.67 GB | 4 depth cameras |
| `segmentations/` | 103,752 | 0.14 GB | per-person PNG masks + JSON boxes |
| `poses/` | 10,035 | 0.11 GB | 3D keypoints |
| `bboxes_3d/` | 10,035 | 0.011 GB | oriented 3D boxes, label `"human"` |
| `states/` | 9,977 | 0.002 GB | per-person state/action labels |

#### The occupancy problem

```
persons per frame (person_states.json, 9,975 frames): {1: 4989, 2: 4418, 3: 546, 4: 22}
unique person IDs across the whole dataset: 26
max simultaneous masks in one camera view: 6 (center_left)
frames with an empty cabin: 898
```

The paper states it plainly: "scenes with 1-2 occupants". The action vocabulary
(`mugging`, `vandalism`, `littering`, `smoke`, `punch`, `push`) shows what this
is — a staged behaviour and safety recording with actors.

Consequences:

- Levels 3, 4 and 5 of the intended five-level scale have **zero** training
  examples.
- Recording is a **single continuous 32-minute daytime session**, 2026-05-21
  09:10:44 to 09:43:04 UTC, split into 11 sub-sequences, ~6.6 fps effective per
  camera. There is no night, dusk or artificial-light data, so low-light
  robustness cannot be evaluated here at all.
- The vehicle was **stationary**. No vibration, no motion blur, no changing
  daylight.
- Annotations are **pseudo-labels**: 2D detector, manual filtering, SAM 3 Body
  3D for mesh and skeleton, ByteTrack for association, boxes derived from
  poses. Only the action and state labels are hand-made.

#### Practical gotchas

- On disk the layout is a custom ROS tree, **not nuScenes**. nuScenes is an
  export target produced by a script in the toolkit repo, and that repo carries
  **no licence file** — the data is CC BY 4.0 but its tooling is not.
- Depth files are named `*.jpg` but their magic bytes are `89504e47`. They are
  PNGs. 1,204 of them were renamed by content sniffing in our extraction run.
- **Depth intrinsics differ from colour intrinsics**: depth is 848x480 with
  fx≈422, colour is 1280x720 with fx≈920. Using the colour intrinsics for
  deprojection fails silently.
- Extrinsics **are** in the archive (`frame_ids.json` plus the URDF chain in
  `target_transforms.json`), so no unlicensed toolkit code is needed.
  `front_left_depth` resolves to `[-1.176, 4.220, 1.953]` in `base_link`.
- A **HuggingFace mirror** exists and is faster from Almaty: 2.81 MB/s versus
  325 KB/s from Zenodo. Both honour HTTP range requests.

#### Sensor rig used by the authors

Four Intel RealSense D435i, each on a Raspberry Pi, PoE switch, NUC 11 running
ROS 2, NTP-synchronised. Ouster OS0-128 LiDAR. RGB at 15 Hz, LiDAR at 10 Hz.
The paper gives the D435i's reliable depth window as approximately 0.3 to
3.0 m.

The LiDAR is **not** their label source. It is used for ICP calibration, for
qualitative validation of the reconstructed meshes, and as an input modality in
their BEVFusion benchmark.

---

### 3. Model candidates and licences

| Candidate | Verified numbers | Licence | Verdict |
|---|---|---|---|
| CSRNet | 16.26M params; ShanghaiTech A MAE 68.2 / MSE 115.0, B 10.6 / 16.0 | reference repo has **no licence file** | reference only; reimplement if needed |
| P2PNet | ShanghaiTech A MAE 52.74 / MSE 85.06, B 6.25 / 9.9 — best accuracy here | **"Use in source and binary forms shall only be for the purpose of academic research"** | **paper track only, never the product** |
| DINOv2 ViT-S/14 + CORN head | 21M params, ImageNet linear 81.1% | Apache-2.0; `coral-pytorch` head MIT | chosen for the first run |
| ConvNeXt-Tiny + same head | no benchmark exists for this task | MIT; timm Apache-2.0 | the edge-deployable variant |

Neither paper states inference speed, and **no latency number can be produced
for any candidate until the edge device is chosen**.

The P2PNet finding contradicts the stack recorded in the global project notes
("P2PNet, ShanghaiTech Part B"). Both halves are problems for a shipping
product: P2PNet is academic-only, and ShanghaiTech has **no stated licence at
all**, which in a due-diligence review is worse than a restrictive one.

---

### 4. Compute limits (Kaggle)

| Limit | Value | Source |
|---|---|---|
| Max size, single dataset | 200 GB | kaggle.com/docs/datasets |
| Private dataset storage | 200 GB | kaggle.com/docs/datasets |
| Top-level files per dataset | 50 | kaggle.com/docs/datasets |
| Notebook runtime | 12 h CPU/GPU, 9 h TPU | kaggle.com/docs/notebooks |
| `/kaggle/working` | 20 GB, output also capped at 20 GB | kaggle.com/docs/notebooks |
| GPU quota | ~30 h/week, "sometimes higher depending on demand" | kaggle.com/docs/efficient-gpu-usage |
| Sessions | P100, or 2×T4; 4 cores, 29 GB RAM | kaggle.com/docs/notebooks |

Not documented anywhere: the scratch disk size outside `/kaggle/working`, and
the maximum size of an individual output file. **GPU quota remaining is not
exposed by the API** and has to be read at kaggle.com/settings.

#### Kaggle mechanics learned the hard way

`kernel_sources` **does** mount the source kernel's output. It appears at
`/kaggle/input/notebooks/<owner>/<slug>/`, alongside five code artifacts
(`__output__.json`, `__results__.html`, `__script__.ipynb`, `__script__.py`,
`custom.css`). An earlier conclusion in this project that it mounts only code
was wrong — the directory walk that produced it capped recursion at three
levels and the data sits at four.

`kernels output` paginates and will not finish on a large output. Fetch a
single file instead:

```python
api.kernels_output('owner/slug', path='outputs', file_pattern=r'.*\.log$')
```

Note `file_pattern` is a regex, not a glob.

---

### 5. Runs executed

#### `sanas-extract-subset` — COMPLETE

Range-fetches a stride-10 subset straight from the HuggingFace mirror. Nothing
larger than the 44.9 MB central directory ever touches a local disk.

```
extracted 35,404 files, 0.686 GB in 1730.6 s
frames                : 1,289
sub-sequences         : 11
missing view label    : 0
count_view  dist      : {0: 218, 1: 632, 2: 389, 3: 50}
count_cabin dist      : {0: 120, 1: 656, 2: 453, 3: 60}
corrected 1,204 file extensions after magic-byte sniff
```

Both histograms sum to 1,289. The subset tops out at three occupants — the 22
four-occupant frames in the full dataset did not survive stride-10 sampling.

**Corrected 2026-08-10 (see `../experiments/log.md`, entry "two-phase plan
recorded"). The subtraction `218 - 120 = 98` is wrong** — it under-reports,
because 4 frames have `view > 0` while `cabin == 0` (pseudo-label
disagreement), so the zero-sets are not nested. Measured directly:

- `view == 0` and `cabin > 0`: **102 frames (7.9%)** — the camera sees nobody
  at all while someone is aboard.
- `view < cabin` (any undercount): **181 frames (14.0%)**. `view > cabin`:
  7 frames (0.5%).

14.0% is the operative figure. A weak-label scheme binding cabin-level counts
to a single view inherits that, not 7.6%. Either way it is a measured cost of
single-camera occlusion, not an estimate.

#### `sanas-depth-occupancy` — COMPLETE (third attempt)

Geometric baseline: deproject depth with the correct depth intrinsics,
transform into `base_link`, build a BEV occupancy grid, correlate against
ground truth. No training, no GPU.

```
front_left_depth   zero 15.5%   in 0.3-3.0 m 42.2%   beyond 42.3%

STEP 1: FLOOR COVERAGE (118 empty-cabin frames)
  occupied-region cells (every 3D box, dilated 0.3 m): 4,110
  front_left_depth covers 10.0% of the region; 3,699 cells never seen

STEP 2/3: OCCUPANCY vs GROUND TRUTH (n = 1,204)
  Pearson  r   : 0.184   95% CI [0.133, 0.231]
  Spearman rho : 0.139   95% CI [0.081, 0.199]
  monotonic in count: False

   count      n     mean    median      std
       0    118   0.0014   0.0013   0.0003
       1    615   0.0017   0.0015   0.0007
       2    412   0.0019   0.0015   0.0010
       3     59   0.0017   0.0014   0.0009
```

**Single-camera depth occupancy does not work.** Three independent reasons:

- It sees 10% of the region occupants actually use. Nine tenths of the cabin is
  outside the reliable window.
- Correlation of 0.184 excludes zero but explains roughly 3% of the variance.
- The score is **not monotonic**: it rises from zero to two occupants and falls
  at three. This is the saturation predicted for a surface sensor — a depth
  camera sees the shell of a group, not its volume — and it appeared at three
  people rather than thirty.

Absolute values are 0.0014 to 0.0019, meaning 0.14% to 0.19% of cells occupied.
The differences are near noise.

Failed attempts before this: two ERROR runs caused by the recursion-depth bug
described in section 4, not by anything in the data or the geometry.

#### Preliminary four-camera coverage (30-frame local sample, NOT the sanctioned run)

| Range gate | front_left alone | union of 4 |
|---|---|---|
| 3.0 m (spec) | 6.7% | 28.8% |
| 6.0 m | 16.1% | 54.9% |
| 10.0 m | 21.5% | 63.3% |

Only one empty-cabin frame in that sample, so treat as indicative. The full run
is built and awaiting approval. If it holds, four cameras still leave two
thirds of the occupant region unseen at spec range.

---

### 6. External data for the crowded end

Full provenance in `datasets.md`. Summary of the search:

**Nothing public supervises a crowded transit cabin.** Every in-cabin dataset
with real annotations hits the same ~4-occupant ceiling or has no count labels.
PMOF, published two months ago and purpose-built for this, also stops at four.
BUS-HAR (overhead views of crowded buses — exactly the target) is not public.
PHD is the only genuinely crowded carriage dataset found; its GitHub repo is an
empty stub, distribution is Baidu Netdisk only, and it has no licence.

Downloaded and verified, 3.47 GB total:

| Dataset | Verified content | Licence | Use |
|---|---|---|---|
| RPEE-HEADS | 1,886 images, 109,913 heads, mean **58.28** (published figure of 56.2 does not reproduce); 85.1% of images carry >30 heads; 666 dark images | CC BY-SA 4.0 per the page section — the wiki footer says CC BY 4.0 and is wrong | pretraining, crowded end |
| DISCO | **8,116** JPEGs, not the 1,935 advertised; 1,935 annotated, 6,181 unannotated; 170,269.9 instances; 593 dark, 141 very dark | CC BY 4.0, cleanest licence surveyed | pretraining, low light |

Neither archive ships a licence file. The licence exists only on the
publisher's page — worth recording before anyone asks in a review.

**RPEE-HEADS is two datasets in one.** 1,307 `field_natural` images versus 579
`lab_capped` images where participants wear bright red, fluorescent green or
numbered white caps. A detector trained on the lab third learns "saturated
coloured blob = head". Train on the field subset only, and at minimum keep the
lab images out of validation.

Camera geometry: RPEE field cameras sit 3-6 m up, angled 40-60° down with
fisheye distortion. That is the closest public geometry to an in-cabin camera
found anywhere. DISCO is taken for darkness and licence, not viewpoint.

---

### 7. Hardware implications

The evidence so far bears directly on the proposed rig.

**Camera count.** One camera covers 10% of the occupant region at spec range.
The preliminary four-camera union reaches 28.8%. A two-camera rig — one front,
one rear — is not supportable by any number measured here. The sanctioned
multizone run will settle this properly, and it costs nothing but CPU time.

**Compute.** A geometric pipeline is array arithmetic and is CPU-bound; it does
not need an NPU. An accelerator is justified by the CNN branch, and the
strongest argument for Jetson is TensorRT maturity rather than raw throughput.
"The only affordable board that will cope" is not accurate — the choice is
about toolchain risk.

**Depth as the primary signal.** Weak, on the evidence. It buys darkness
tolerance and privacy, but it saturates early, and saturation lands exactly on
the levels the product exists to distinguish. A hybrid — geometry for the
sparse end and in darkness, a network for the dense end — matches the
measurements better than either alone.

None of this is validated on a full cabin, because no such data exists yet.

---

### 8. What is still unknown

- **GPU quota remaining.** Not exposed by the API; blocks the training kernel.
- **Edge device.** Undecided, so no latency comparison is possible.
- **Four-camera coverage on the full subset.** Kernel built, not yet run.
- **Behaviour in a genuinely full cabin.** Unobservable with any public data.
  Requires own recording once municipal approval lands.
- **Occupancy definition.** Every dataset gives counts; none gives a level. The
  count-to-level mapping needs a cabin-capacity denominator that has to come
  from the product side, and whether a pram or a suitcase counts as occupied
  space is a product decision, not a modelling one.
- **RPEE-HEADS ShareAlike.** Whether trained weights constitute an adaptation
  is unsettled. Fine for the paper track; needs a legal read before shipping.

### 9. Untuned defaults currently in the code

These are guesses, exposed as parameters, and none has been tuned:

- `z_min` 0.6 m for "above seat height", inferred from box geometry; the
  archive documents no seat height.
- Depth values treated as uint16 millimetres — the RealSense convention,
  consistent with observed ranges, but stated nowhere in the archive.
- Pseudo-label score threshold 0.0, so every detection counts.
- BEV cell 0.1 m, minimum 3 points per cell, background quantile 0.5.
- Coverage region derived from every 3D box dilated 0.3 m, because the archive
  contains no cabin floor polygon.
- Validation fraction 0.25, held out by sub-sequence.


---

<!-- was: development/sanas_cv_track_conclusions.md -->

**Раздел 3.** Выводы по CV-треку на 09.08.2026.

## Sanas — выводы по CV-треку (на 09.08.2026)

> **ОТМЕНЁННАЯ АРХИТЕКТУРА.** Более ранний banner ниже сам устарел: и дверной
> ToF, и двухфазная схема отменены 2026-08-25. Этот документ оставлен только
> как история датасетных и CV-исследований. Текущий источник истины:
> [`../GROUND_TRUTH.md`](GROUND_TRUTH.md).

### 1. Датасет Горелика (BeIntelli) — статус

- **Не годится как финальный обучающий набор.** Максимум 4 человека в кадре,
  26 уникальных ID за весь датасет, одна сессия 32 минуты, только день,
  постановочная актёрская съёмка. Подтверждено самими авторами в статье
  ("scenes with 1-2 occupants").
- **Годится как проверка пайплайна.** Извлечение, метки, инференс — всё
  работает end-to-end на этих данных. Это подтверждает, что код рабочий,
  не что модель готова к продукту.
- Формат на диске — ROS-дерево, не nuScenes. nuScenes — только цель
  конвертации скриптом из тулкита.
- Автобус во время записи **стоял на месте** — нет тряски, смазывания,
  изменения освещения от движения.
- Железо авторов: 4× Intel RealSense D435i (по одной на Raspberry Pi),
  PoE-свитч, NUC 11 с ROS 2, LiDAR Ouster OS0-128.
- Экстринсики всех 4 камер относительно `base_link` уже извлечены из
  `frame_ids.json` + `target_transforms.json` — повторять калибровку не
  нужно.

### 2. Поиск публичного датасета с реальной толпой в салоне — отрицательный результат

Проверены все датасеты из таблицы связанных работ статьи Горелика плюс
независимый поиск (PHD, PMOF, Bus Violence, BUS-HAR, PAMELA-UANDES,
Beijing-BRT и другие). **Ни один публичный in-cabin датасет не покрывает
загруженный салон транспорта.** PMOF (июнь 2026, сделан специально под эту
задачу) тоже упирается в максимум 4 пассажира.

Вывод: датасет BeIntelli не слабый частный случай — он типичен для всего
поля. Загруженный конец шкалы закрывается **только собственной съёмкой**.

### 3. Что взято для предобучения (не для финальной модели)

| Датасет | Что даёт | Лицензия | Ограничение |
|---|---|---|---|
| **RPEE-HEADS** (field_natural подмножество, 1307 из 1886 кадров) | Плотная толпа (avg 57.7 голов/кадр), реальная ночь (666 тёмных кадров), геометрия камеры на мачте 3-6м под углом 40-60° — ближайшая к салонной публичная геометрия | CC BY-SA 4.0 (ShareAlike — юридический вопрос по весам модели открыт) | **lab_capped треть (579 кадров) исключить** — люди в ярких шапках, детектор выучит не то |
| **DISCO** | 1935 размеченных + 6181 неразмеченных кадров, честная ночь (593 тёмных, 141 совсем тёмных) | CC BY 4.0, чистая | Геометрически слабый — 8-15м высота, мелкие головы. Берётся только за темноту и лицензию |

Оба датасета скачаны и верифицированы (`docs/datasets.md`). Оба дают только
**счёт людей**, ни один не даёт вместимость салона (знаменатель для шкалы).

### 4. Модели-кандидаты для CV

| Кандидат | Статус | Причина |
|---|---|---|
| P2PNet | **Исключён из продукта** | Лицензия "academic research only" — несовместимо с коммерческим продуктом внутри Avtobys |
| CSRNet | Резерв | Референс-репозиторий без лицензии, нужна чистая реализация |
| **DINOv2 ViT-S/14 + CORN-голова** | Рекомендован, первый прогон | Apache-2.0, backbone заморожен — обучается только голова, минуты GPU |
| **ConvNeXt-Tiny + CORN-голова** | Рекомендован как edge-вариант | MIT/Apache-2.0, свёртки экспортируются в ONNX/TensorRT предсказуемее ViT |

Латентность ни для одного кандидата не оценена — модель edge-устройства
пока не выбрана.

### 5. Depth-геометрия вместо RGB-подсчёта — что выяснили

Идея: считать не людей, а занятый объём через BEV-сетку по облаку точек
глубины (RealSense D435i, уже есть в датасете), вместо дорогого LiDAR.

**Тест на одной камере (front_left_depth), 20 кадров, 8,140,800 пикселей:**
- 41.9% пикселей — в надёжном окне сенсора 0.3-3.0 м
- 24.6% на 3-6м, 10.4% на 6-12м, 9.4% дальше 12м, 13.7% без отклика
- Средний p95 глубины по кадру — 23.2м, что физически невозможно внутри
  автобуса → дальнее поле это шум сенсора, не геометрия

**Вывод: одна depth-камера покрывает только ближнюю треть салона надёжно.**

**Гипотеза (в разработке): 4 depth-камеры вместо одной.**
Каждая камера покрывает свою зону в радиусе ~3м, вместе — весь салон без
слепых пятен. Экстринсики для объединения в единую систему `base_link` уже
есть в архиве (из калибровки Горелика через ArUco+COLMAP, повторять не
нужно). Тест не запущен — план сформулирован, ждёт прогона:
1. Геометрическая проверка покрытия (есть ли слепые зоны у 4 камер вместе)
2. Объединённая BEV-сетка по всем 4 камерам
3. Сравнение occupancy score против ground truth (count_view/count_cabin)
4. Прямое сравнение "1 камера vs 4 камеры" по корреляции с ground truth

Их метод (SAM 3 Body 3D, ByteTrack, кластеризация поз) **не нужен нам** —
это тяжёлый ML-пайплайн для получения скелетов людей. Наша задача —
геометрический подсчёт занятых ячеек, без реконструкции людей.

### 6. Инфраструктура — что собрано и работает

- `src/sanas/` — библиотека: ziprange.py (range-чтение ZIP без полной
  загрузки), config.py, corn.py, models.py, data.py, labels.py,
  depth_occupancy.py
- Три Kaggle-ядра: `extract_subset` (CPU, извлекает ~0.5-0.7GB подвыборку
  без скачивания всего архива в 73.5GB), `depth_occupancy` (CPU,
  геометрическая проверка), `train_corn` (GPU, ждёт разрешения)
- Метка smoke-теста: **`count_view`** (людей видно в этой конкретной
  камере) — не `count_cabin`, так как одна камера физически не может
  видеть весь салон
- Целевая переменная — **сырой count 0-4**, не нормированная 0-1 шкала.
  Переход на нормировку — изменение конфига (`TargetConfig`), не
  переписывание кода

### 7. Метод подсчёта — двухфазный план (финальное решение, 10.08.2026)

После серии итераций (RGB-классификация салона → multi-zone depth →
дверной счётчик → снова RGB) принят **двухфазный план**, не выбор
одного метода навсегда.

#### Фаза 1 — сейчас, дверной 3D-счётчик (APC)

Индустриальный стандарт (iris GmbH IRMA MATRIX, INIT, Dilax) —
депроекция глубины над дверным проёмом, line-crossing подсчёт
входов/выходов, накопительный count с обнулением на конечной.

**Почему первой:**
- Не требует разрешения на камеры в реальном салоне — дверной сенсор
  можно тестировать в контролируемых условиях
- Не требует разметки загруженного салона — дверной проём это
  тривиальная сцена (1-2 человека), готовые методы применимы сразу
- Даёт рабочий продукт для демо Innoforce/акимату уже на раннем этапе

**Что отменяется на этой фазе:** multi-zone depth-тест по всему салону,
RGB-классификация целого кадра, DINOv2/ConvNeXt+CORN как основной метод
occupancy, fisheye/thermal/фильтр Калмана (вся Sanas-Fusion архитектура).
Safety-функции (детекция вандализма, краж, харассмента) — не строятся
на этой фазе, дверной счётчик физически не может их поддержать.

**Код, который остаётся, но не используется активно:**
`src/sanas/models.py`, `corn.py`, `config.py` (TargetConfig) —
понадобятся на Фазе 2, не удалять.

#### Фаза 2 — после разрешения на камеры в реальном автобусе

Переход на RGB-классификацию всего кадра салона: DINOv2 ViT-S/14 или
ConvNeXt-Tiny (backbone) + CORN-голова для occupancy. Архитектура
допускает добавление второй головы поверх того же backbone для будущей
детекции инцидентов (вандализм, кража, харассмент) — тот же экстрактор
признаков, разные "насадки" сверху.

**Данные для обучения на Фазе 2:**
- Реальная съёмка из своего автобуса (главный источник, недоступен до
  разрешения)
- RPEE-HEADS (field_natural подмножество) + DISCO — предобучение,
  уже скачаны и верифицированы
- Датасет Горелика — псевдо-разметка, pipeline-проверка
- Потенциально: временные метки входа/выхода из Фазы 1 (дверной счёт)
  как слабая (noisy) авторазметка для RGB-кадров того же салона —
  идея на будущее, не реализуется сейчас, но структуру данных стоит
  проектировать с учётом привязки timestamps камеры к дверным событиям

**Связь фаз:** Фаза 1 не тратится впустую — она даёт (а) рабочий
продукт для переговоров с Innoforce до появления полноценной RGB-
системы, (б) инфраструктуру (Jetson, крепления, интеграция с Avtobys),
которая используется и на Фазе 2, (в) потенциальный источник слабой
авторазметки для будущей RGB-модели.

### 8. Железо — финальное решение (10.08.2026)

**Вычислитель: NVIDIA Jetson Orin Nano Super Dev Kit, $399**
(либо Seeed reComputer Super J3011 ~$797 при дефиците devkit — корпус,
SSD, рабочий диапазон до 65°C, ближе к казахстанскому лету).

Причина: единственная плата, которая **гарантированно** запускает обе
модели-кандидата (DINOv2 ViT-S/14 и ConvNeXt-Tiny) без риска. Attention-
архитектуры (ViT) плохо или вообще не поддерживаются на дешёвых NPU-платах:
- **Hailo-8/8L** — DINOv2 не собирается, ViT-модели в Model Zoo только для
  Hailo-15H/10H, не для Hailo-8
- **RK3588 (Orange Pi, аналоги)** — attention считается частично на CPU,
  а не на NPU, скорость резко падает
- **RV1126/1109, Google Coral** — чисто свёрточные NPU, DINOv2 не пойдёт

Разница в цене с ближайшим конкурентом (Orange Pi 5 Max, $180) — $220,
что дешевле риска потерять две недели на непредсказуемое портирование
на чужом тулчейне.

**Камеры: 2× Intel RealSense D435i (не 4), ~$650**

Причина смены с исходных 4 камер на 2:
- 4× RealSense на одном Jetson упираются в пропускную способность USB —
  librealsense документирует, что 3-4 камеры одновременно работают только
  на сниженных профилях (848×480, 15 fps)
- Задаче не нужен высокий fps — это периодическая проверка занятости
  (1 замер в секунду), не видеопоток
- 2 камеры (перед + зад салона) — тот же подход, что у Hsu et al. (2020),
  уже проверенный в литературе на реальном автобусе

**Depth vs RGB — выбрана depth-схема (RealSense), не RGB (Hikvision)**

Ранее в проекте фигурировал план "Jetson + Hikvision IP-камеры" —
заменён на "Jetson + RealSense" по итогам сегодняшнего решения. Depth-
геометрия (BEV occupancy по облаку точек) выбрана как основной метод
измерения occupancy, соответствует уже проведённым тестам
(`depth_occupancy.py`) и общему решению по методу измерения
загруженности (раздел 5).

**Порядок для масштабирования на парк — решается после пилота:**
если ConvNeXt на пилоте догонит DINOv2 по точности → переход на RK3588
(~$180/узел, вдвое дешевле) при масштабировании. Если DINOv2 значимо
лучше → остаёмся на Jetson-модулях для всей серии.

**Температура (60-70°C за стеклом летом в Алматы)** — не различающий
фактор между платами, у всех одинаковая проблема. Решается размещением
(не на стекле, ближе к полу/в нише) и корпусом с принудительным обдувом.

### 9. Открытые блокеры

- Multi-zone depth тест — бриф отправлен под 4 камеры, **нужно пересчитать
  под 2 камеры** согласно решению по железу выше
- CAD-крепление — под 2 камеры, не 4 (задача для агента cady)
- Соглашение с Innoforce и муниципальное разрешение на камеру — **главный
  блокер всего продуктового трека**. План обхода: пассажирская съёмка на
  телефон (легально, без разрешения) → пилот с частным перевозчиком
  (маршрутка/троллейбус, один человек принимает решение) → LOI от
  Innoforce на основе рабочего прототипа → переговоры с муниципалитетом
  уже с готовой системой, а не с идеей на бумаге


---

<!-- was: research/paper/related_work_notes.md -->

**Раздел 4.** Related-work заметки отменённого door/ToF трека.

## Related-work notes (coordinator, 2026-08-10)

> **ИСТОРИЯ ОТМЕНЁННОГО DOOR/ToF PAPER TRACK.** Файл сохранился, хотя запись
> архитектурного пивота перечисляла его среди удалённых. Не использовать для
> описания текущего продукта или активной статьи без нового review. См.
> [`../../GROUND_TRUTH.md`](GROUND_TRUTH.md).

Source: Consensus academic search (two queries). All entries below are from
abstracts unless marked otherwise — Danyshpan must verify against full text
before citing, and must not copy accuracy numbers without checking the paper.
Consensus result URLs are kept for retrieval; final citations need the real
venue/DOI.

### What this does to our novelty claim

Low-resolution ToF people counting is NOT new in general:

- Lu et al. 2021 (Energy and Buildings): sparse array of low-res ToF sensors
  for zone-level occupancy counting in offices, reported error rate ~0.4%.
  https://consensus.app/papers/details/e499f8462d2a5f0f9af2ae01fb5f17ba/
- Jeong et al. 2025 (ETRI Journal): ToF camera above a doorway, mean-shift
  clustering, >90% accuracy in single-entry/exit scenarios, privacy framing
  identical to ours. https://consensus.app/papers/details/2ef4f2344cbc5df79b686542e367b124/
- Stec et al. 2019 (DASIP): ToF people counting explicitly for public
  transport, embedded processing. https://consensus.app/papers/details/f6c3215f3c5f539c959952b088efc4c7/
- VL53L5CX itself has 2026 characterization papers (Caroleo et al., Sensors;
  Kalvodova et al.) — the sensor is known to research, mostly robotics.
  https://consensus.app/papers/details/589ee630f31858569292ed0a0e16b856/
- Papanashi et al. 2026 (IEEE Access): low-res THERMAL array + ESP32 counting,
  the closest "cheap grid sensor on microcontroller" system.
  https://consensus.app/papers/details/8042b2f0480e5fe4b563f7d169ae0a20/

Depth-based APC at transit doors is industry standard and well published:

- Seidel et al. 2021, NAPC (IEEE OJ-ITS): LSTM on low-res 3D LiDAR depth over
  TRAIN doorways, ~13,000 annotated recordings, ~96% correct door-phase
  counts, counts 0-67. Public privacy-friendly dataset.
  https://consensus.app/papers/details/be89ef2d8df85a7e9b771fd735f54af8/
- Seo et al. 2026, CLAPC (IEEE OJ-ITS): CNN-LSTM, ToF depth above train doors,
  99.79%/99.97% boarding/alighting; explicitly states meeting the +-1%
  requirement of **VDV Guideline 457** (this is the acceptance standard our
  trade study must reference; the +-1% figure now has a citable source).
  Also does spatial down-sampling of 2D video for privacy — partial overlap
  with our resolution-ablation idea, but on grayscale video, not depth grids.
  https://consensus.app/papers/details/388aa1ec3d56565ea4265315b7fd5b56/
- Yahiaoui et al. 2010 (J. Electronic Imaging): dense stereo over bus door,
  99%/97% on two datasets. https://consensus.app/papers/details/eb5dd65da290548cb8b03d6877a7cb02/
- Sun et al. 2018 (IEEE T-ITS): PCDS — public RGB-D dataset, >4,500 videos at
  BUS entrance doors. **Candidate future validation data for our counter —
  flag to Donatello for a later run.**
  https://consensus.app/papers/details/6a8012616f5e51c98b8a76732c7bafae/
- Pronello et al. 2023 (Sensors): commercial video APC claimed 98% by vendor,
  measured 53-55% in a real field trial; their RPi+YOLOv5 got 72-75%. Key
  honesty citation for the gap between vendor claims and field performance.
  https://consensus.app/papers/details/1cd73e6562255fd79564ed3af4265e32/
- Liu et al. 2023: 3D LiDAR + RGB APC for sightseeing trams; notes ToF and
  mmWave explored, LiDAR underexplored.
  https://consensus.app/papers/details/2a80c510a10c52aa98510c969841d479/
- Nurseitov et al. 2026 (J. Imaging): YOLOv8+DeepSORT passenger counting,
  Kazakhstan group — regionally relevant citation.
  https://consensus.app/papers/details/116add0819c855e899c51e43df2133c5/

### Resulting honest novelty framing (for the paper)

NOT claimable: "first low-res ToF people counter" (buildings did it), "first
depth APC at transit doors" (industry + NAPC/CLAPC).

Claimable, pending our results:
1. Feasibility of a **single-chip 64-zone ToF (VL53L7CX-class) as a bus APC**,
   i.e. the extreme low-cost end (~$7 sensor vs commercial APC units), via a
   **resolution-ablation study** (full depth camera -> 8x8 -> 4x4) on real
   transit-cabin data. CLAPC downsampled grayscale video; nobody in the found
   set ablates depth resolution down to single-chip multizone grids for door
   counting.
2. **Vestibule-interior virtual-line placement**: measured finding that an
   oblique in-cabin rig cannot reproduce top-down aperture geometry, and the
   line must move inside the cabin (Finding C in development/experiments/log.md) — a
   deployment-geometry result the APC literature (which assumes top-down
   mounting) does not cover.
3. The measured **negative result** for cabin-wide oblique depth occupancy
   (10% coverage, r=0.184, non-monotonic — development/findings.md) as evidence FOR
   door-based counting over cabin-wide depth sensing.
4. System integration: terminus-reset cumulative count + ordinal 5-level
   crowding output for a rider-facing app (Avtobys), with Phase-2 RGB ordinal
   head as drift corrector — design contribution, clearly marked unvalidated.

### Verification status

All of the above: inferred-from-abstract via Consensus. Scite is out of monthly
quota until 2026-09-01, so full-text verification must go through the papers
themselves (arXiv/OA links) before the reference list is final.


---

<!-- was: business/outreach/email_template.md -->

**Раздел 5.** Прежний cold email шаблон, отменён 2026-09-02.

## RTCI outreach — cold email template

> **Superseded 2026-09-02.** Use `business/outreach/template.md`. This
> version asks for collaboration and uses a first-name salutation, both of
> which are now forbidden. Kept for history.

Sender: Diyas. Topic: real-time crowding info & transit boarding decisions
(Avtobys/Sanash field-experiment paper). Ask: propose working together on
the topic — no draft/paper attached, CV attached instead.

Placeholders below fill in from Diyas's CV once uploaded. Do not invent
bio, achievements, or prior lab affiliations — pull only from the CV.

### Subject line
`[Prospective Collaborator] Real-time crowding info & transit boarding decisions`

(swap "Collaborator" for "Ph.D. Student" / "RA" per contact if the ask
shifts to seeking a position rather than proposing collaboration)

### Body

```
Dear Prof./[Last Name],

My name is Diyas, [ONE LINE FROM CV — role/affiliation]. I'm working on
[Sanash/Avtobys — bus occupancy estimation + a field experiment testing
whether real-time crowding information changes rider boarding decisions].

[ONE LINE tying their specific paper/field to the project — e.g. "Your
work on [present bias / information & decision-making / X] is directly
relevant to the boarding-decision side of this."]

I'd like to propose working together on this — happy to share more detail
on a call, or by email if that's easier. CV attached.

Best regards,
Diyas
```

### Notes
- Field line must reference something specific to the recipient's actual
  research (not generic praise) — pull from their real publications.
- No fabricated personal connection ("I was touched by...") — state the
  actual overlap plainly.
- Attach CV once provided by Diyas.


---

<!-- was: research/coursework/RESEARCH_METHOD_GROUNDING.md -->

**Раздел 6.** Синтез учебного курса от 2026-08-26. Заменён `research/coursework/WRITING_RULES.md`, который собран из первичных транскриптов и колод и прошёл аудит цитат. Аудит установил, что этот файл не содержал ошибок по существу: спорное место про RQ про Алматы он сам пометил как inferred и вынес в открытые вопросы. Он вытеснен полнотой, а не неверностью.

## Research Method Grounding

Compiled 2026-08-26. Synthesizes every video and document in
`research/coursework/` for research methodology (research question, gap,
literature review, methodology, design/dataset/metric/validation choices).
Does not touch Sanas product architecture or `GROUND_TRUTH.md`.

This file adds one thing `course_notes.md` (2026-08-20, already in this
directory) does not have: coverage of `video1311882563.mp4`, a 20-slide data
analysis lecture never referenced anywhere else in the repo, and an audit of
`literature_review_methodology.docx`, a generated (not course) artifact.
Everything already in `course_notes.md` is treated as verified prior work and
summarized here, not redone. Read `course_notes.md` directly for full quotes
and the complete argument on each topic.

### Source inventory

| File | Type | Watched/read | Transcript | Limitations |
|---|---|---|---|---|
| `1st lesson- part 1.mp4` (34:56) | recording | Previously transcribed + read (per `course_notes.md`, verified against `transcripts/lesson1_part1.txt` this session) | Yes, faster-whisper small int8 | Names/technical terms occasionally mangled per README |
| `1st lesson- part 2.mp4` (19:00) | recording | Previously transcribed + read | Yes, `transcripts/lesson1_part2.txt` | Same as above |
| `video1768002125.mp4` (24:56) | recording | Previously transcribed + read | Yes, `transcripts/video1768002125.txt` | none noted |
| `video1168216779.mp4` (26:16) | recording | Previously transcribed + read | Yes, `transcripts/video1168216779.txt` | none noted |
| `video1433589796.mp4` (53:55) | recording | Previously transcribed + read | Yes, `transcripts/video1433589796.txt` | none noted |
| `video1101430871.mp4` (46:02) | recording | Previously transcribed + read | Yes, `transcripts/video1101430871.txt` | none noted |
| **`video1311882563.mp4` (32:34)** | recording | **New this session.** No transcript exists and none was generated (out of scope: task says extract from visible slides only when transcript is absent, don't invent speech). Video track sampled at 1 frame/60s (33 frames) plus a scene-detect pass; confirmed via `ffprobe` to have an AAC audio track that was not transcribed. | **No** | **Speech content is unknown.** Only the visible slide deck (`Analysis slides.pdf`, 20 pages, shown via screen share) could be read. Presenter is on camera throughout but slides are the only accessible content. Not listed in `README.md` or `course_notes.md` — this is new material to the repo's documentation, not new material to the disk. |
| `Foundations class- group 3.pdf` | slides, 20 pp | Previously read (page count verified: 20 pp match) | n/a | none |
| `Research - intro.pdf` | slides, 20 pp | Previously read (verified: 20 pp) | n/a | none |
| `2nd class-Reading.pdf` | slides, 19 pp | Previously read (verified: 19 pp) | n/a | none |
| `Lit. review.pdf` | slides, 15 pp | Previously read (verified: 15 pp) | n/a | none |
| `Methodology.pdf` | slides, 17 pp | Previously read (verified: 17 pp) | n/a | none |
| `Methodology for emp. papers.pdf` | slides, 15 pp | Previously read (verified: 15 pp) | n/a | **Truncated at component 5 of 6** (ethics/consent component missing from the file itself, not from the reading) |
| `Mohlaroy--RAS.docx` | worked student paper, 22 refs | Previously read; confirmed this session (title and final reference match `course_notes.md` exactly, 102 non-empty paragraphs) | n/a | none |
| `course_notes.md` | synthesis, ~11.8k words | Read in full this session | n/a | Treated as verified prior work, not re-derived |
| `README.md` | index | Read in full this session | n/a | none |
| **`literature_review_methodology.docx`** | **not course material** — a generated methodology draft | **New this session.** Read in full (10 non-empty paragraphs) | n/a | See note below. Not a Terra course artifact; excluded from the course synthesis sections. Docx properties: `author: python-docx`, `last_modified_by: moneyman`, `modified: 2026-08-25`. This is machine-authored output describing a scoping review already run on a Sanas-adjacent research question ("How effectively do deep learning-based computer vision methods estimate passenger density in public transport vehicles?"), citing Scite, CSRNet-lineage, and "this review's parent project." It is evidence of a prior literature-review exercise, not evidence of course content, and it is not corroborated elsewhere in this repo (no matching entry in `development/experiments/log.md` or `GROUND_TRUTH.md`). Its "Eligibility criteria" section heading has no body text — the file itself is incomplete. **Status: unverified artifact, not authoritative for product or paper claims.** |
| `~$terature_review_methodology.docx` | Word lock file | Not a document — confirmed via `ls`: 162-byte owner-lock file for the `.docx` above, created because that file was open in Word at time of inspection | n/a | Not content; excluded from all synthesis below |

### Lesson-by-lesson notes

Sections 1–6 below are drawn from the six previously-transcribed recordings
and are condensed from `course_notes.md` §§1–16, which remains the source of
full quotes and timestamps. Section 7 is new.

#### 1. Foundations (`1st lesson- part 1/2.mp4`, combined ~54 min)

What research is/is not; paraphrase vs copy; AI policy (strict, this tutor);
no personal pronouns; bias vs opinion. Full detail in `course_notes.md` §1.

#### 2. Research question, gap, sourcing (same two recordings)

RQ formula `How/Why does [factor] affect [outcome] among [group/context]?`,
10–15 words; red flags; CRAAP; database list; bibliography tag; Zotero.
`course_notes.md` §§2, 4.

#### 3. Reading academic literature (`video1768002125.mp4`, `video1168216779.mp4`)

SMART reading order (title→abstract→intro→conclusion→headings→figures→
methods/results last); abstract's four parts; finding RQ and contribution.
`course_notes.md` §5. These two recordings are also the "blueprint/outline"
session: four-part introduction, research-gap types, purpose statement.
`course_notes.md` §§10–11.

#### 4. Literature review (`video1433589796.mp4`, 53:55)

Conversation metaphor; theme structure; summary vs synthesis; four-part
paragraph; comparing/contrasting; citation rules; `Mohlaroy--RAS.docx`
dissected live from ~30:00 as the model paragraph and model paper.
`course_notes.md` §§6, 12–13.

#### 5. Methodology for literature-review papers (`video1101430871.mp4`, 46:02)

Five components (design, databases, search strategy, eligibility criteria,
analysis strategy) plus screening and word budget; Q&A on paper count and AI
use. `course_notes.md` §8.

#### 6. Methodology for empirical papers (slides only, no matching recording)

Six components (design, population, sample, instrument, data collection,
ethics); survey vs interview vs mixed methods; the empirical-method
landscape including "training a model on a public dataset counts as
computational empirical research." `course_notes.md` §9. Ethics component
text is missing from the source PDF itself (truncated at page 15/15).

#### 7. Data analysis lecture — `video1311882563.mp4` (32:34) — NEW

**No transcript.** The following is read from the 20-slide deck shown on
screen (`Analysis slides.pdf`, per the visible file path in the recording:
`C:/Users/kamil/Downloads/Analysis-%20slides.pdf`). Slide numbers below are
PDF page numbers, not video timestamps — no reliable timestamp-to-slide map
exists without a transcript, so **all points here are slide-text-only,
speaker commentary is not represented, and nothing below should be quoted as
something a tutor said.** Frame sampling was 1 frame/60s across the full
32:34 (33 samples), so slide dwell time and any brief interstitial slides
between samples cannot be ruled out as missed. Twelve "Step" slides were
found; two intervening slide numbers (14, in the run between Steps 8 and 9;
and any slides between 15–17 title cards) were not individually captured but
their content is inferable from the surrounding steps and step numbering is
otherwise continuous 1–12.

Deck content, by Step:

- **Step 1** (p.8): "Your RQ is your filter." Worked example — RQ *"To what
  extent does real-time bus occupancy information change boarding decisions
  of Almaty commuters?"* — with a 10-item example survey variable list (age,
  gender, income, commute frequency, bus usage, transportation preferences,
  app usage, perceived crowding, willingness to wait, response to occupancy
  information). Explicit instruction: don't analyze all variables, ask "does
  this help answer my RQ?" for each one.
- **Step 2** (p.9): The "so what?" test. A number alone (e.g. "61% of
  respondents are 18–25") is not a finding until tied back to the RQ; a
  comparison across groups (72% vs 49%) becomes a finding once framed as a
  23-point difference relevant to the RQ.
- **Step 3** (p.10): Data cleaning before analysis — check for missing
  responses, duplicate responses, impossible responses ("does the data make
  sense"), inconsistent responses, unusable/meaningless open-ended responses.
- **Step 4** (p.11): Descriptive statistics as a required first pass before
  looking for relationships — frequency, percentage, mean, median, range,
  each illustrated with a worked number.
- **Step 5** (p.12): Three questions to ask of any RQ once the basic
  distribution is understood — "what is happening" (distribution), "who is
  different" (comparison), "what moves together" (relationship) — each
  illustrated with worked survey numbers from the same Almaty bus RQ.
- **Step 6** (p.13): Don't just report a raw percentage difference between
  groups ("72% of frequent commuters said yes") — ask why the difference
  exists and whether it is relevant to the RQ.
- **Step 8** (p.15): "Association ≠ Causation." Explicit worked example:
  finding that transit-app users are more likely to value occupancy
  information does not mean app use *causes* that value — lists confounds
  (prior interest in tech, age, commute frequency, other variables). Gives
  safe language ("X was associated with Y") vs dangerous language ("X caused
  Y").
- **Step 9** (p.16): A null or unexpected result is not a failed study —
  "almost no difference by age" is itself a result. Distinguishes stated
  preference from revealed behavior ("people prefer less crowded buses, but
  few would wait for the next one").
- **Step 10** (p.17): Look for contradictions inside your own data as a
  source of interesting findings — high preference + low behavior, high
  awareness + low usage, strong attitude + weak action.
- **Step 11** (p.18): "Do not graph everything." A graph must answer the RQ;
  worked bad example (pie chart of favorite colors, doesn't answer the RQ)
  vs good example (bar chart comparing two groups' willingness to change
  boarding decisions).
- **Step 12** (p.19): Every figure needs a job — a graph/table should show a
  difference, a trend, a distribution, a relationship, or an important
  pattern, not just exist.
- **p.20**: "Homework to be uploaded in google classroom!" — no further
  detail visible; matches the no-stated-deadlines gap already flagged in
  `course_notes.md` §16.

Steps 1–12 numbering implies at least one earlier "Step 0" / framing slide
before p.8 not distinctly captured by the 60-second sampling; the deck's
opening pages (1–7) were only partly sampled and mostly showed a title/agenda
pattern consistent with the other decks in this directory, not additional
numbered steps. This is a coverage gap, not a claim that no content exists
there — a future pass extracting all 20 pages directly from the PDF (if it
can be located — the PDF itself is not in this repo, only its on-screen
capture in the recording) would close it completely.

**Note on the RQ example used throughout this deck:** the worked RQ ("real-time
bus occupancy information... boarding decisions of Almaty commuters") is
topically adjacent to Sanas/RTCI. This is a generic teaching example
constructed by the deck's author for illustration — there is no evidence in
this deck, in the other recordings, or elsewhere in this repo that it is
connected to the actual Sanas or RTCI research question, and it must not be
treated as such. It is plausible this specific worked example was custom-built
for this student's course submission (unclear from slide content alone,
since no instructor speech was accessible) — flagged as inferred, not
confirmed.

### Presentation and document notes

Slide-by-slide detail for the six PDFs is in `course_notes.md` (§§1–16 cite
specific pages throughout; the deck-by-deck table is in that file's header
and `README.md`'s file table). Not reproduced here to avoid duplication.
Summary of what each contributes methodologically:

- `Foundations class- group 3.pdf` (20 pp): research question formula, scope
  control, red flags, CRAAP, annotated bibliography.
- `Research - intro.pdf` (20 pp): scholarly search workflow, bibliography
  tag, credible-vs-reliable distinction, peer-review/author/content signals.
- `2nd class-Reading.pdf` (19 pp): SMART reading order, abstract structure,
  active reading.
- `Lit. review.pdf` (15 pp): conversation metaphor, theme structure,
  summary-vs-synthesis, four-part paragraph, citation rules, IEEE numbering.
- `Methodology.pdf` (17 pp): five-component lit-review methodology, review
  type taxonomy, search-strategy reporting, eligibility criteria rules,
  two-pass screening, PRISMA flow diagram guidance, word budget.
- `Methodology for emp. papers.pdf` (15 pp, truncated): empirical-method
  landscape, why surveys are the default, six-component survey methodology,
  question-type taxonomy, sampling terms and size guidance. Ethics component
  (stated as component 6 of 6) has no body text in the file.
- `Mohlaroy--RAS.docx` (worked example, 22 refs): a complete scoping review
  applying every rule above — PRISMA-ScR, named databases, full search
  string, paired inclusion/exclusion criteria, two-reviewer screening,
  narrative synthesis, absence-as-finding, five specific (non-generic)
  limitations.

### Research-question framework

From `course_notes.md` §2, unchanged:

1. Start from a broad topic, narrow using the formula `How/Why does [factor]
   affect [outcome] among [group/context]?`.
2. Target 10–15 words. Longer is an explicit, named failure mode.
3. Required characteristics: focused, clear, researchable, specific,
   significant.
4. Red flags: too broad, unmeasurable, opinion-based, emotionally loaded,
   requires inaccessible data.
5. Narrow along time, place, population, platform, or variable — pick
   whichever axis actually bounds the question.
6. Read existing literature *before* finalizing the RQ, partly to sharpen it
   and partly to avoid duplicating existing work.

**Research gap types**, from `course_notes.md` §11.4: population gap,
geographical gap, temporal gap, comparative gap, and gaps from age group,
culture, method, theoretical perspective, or technology change. A gap must
be evidenced from the literature you reviewed, not asserted. Overclaiming
("nobody has studied X") is a named failure mode.

**Common mistakes**, consolidated from §§2, 11: RQs that are three sentences
long; questions with no measurable variable; treating "more research is
needed" as sufficient gap justification instead of naming what specifically
is missing; scope so broad it cannot be executed ("effects of technology on
society").

### Literature-review workflow

From `course_notes.md` §§4, 6, unchanged:

1. **Search**: build a keyword list from 3 concepts (topic, population,
   outcome typically), expand each with synonyms, join with OR within a
   concept and AND across concepts, check controlled vocabulary (MeSH, ERIC
   descriptors). Calibrate to 100–800 records; adjust if far outside that
   range.
2. **Inclusion/exclusion criteria**: write before screening, not after,
   to avoid confirmation bias. Must be testable from an abstract alone, each
   tied explicitly to the RQ, tight restrictions justified in one sentence,
   applied consistently.
3. **Screening**: two passes — title/abstract (fast, generous), then full
   text (slow, strict, log every exclusion reason). Deduplicate before
   counting. Report the funnel numbers (total → deduped → screened →
   full-text → included) in three sentences; a PRISMA flow diagram is the
   stronger version of the same disclosure.
4. **Extraction table**: fixed dimensions decided before screening (e.g.
   author/year/country/design/sample/method/findings), consistently applied.
5. **Synthesis by themes**: organize by idea, never by author name; use the
   four-part paragraph (topic sentence → evidence → analysis → transition);
   report agreement, tension, and *why* sources diverge, not just that they
   do.
6. **Citation verification**: cite every quote, paraphrase, specific study
   reference, and statistic; skip citations for common knowledge; IEEE
   numbering is by first appearance, never re-numbered, never alphabetical.

### Methodology framework

From `course_notes.md` §§7–9, unchanged:

- **Choosing design**: the type of evidence collected (numerical, textual,
  or both) must follow from the RQ, not be a style preference. Literature
  review (analyze existing papers) vs empirical (collect original data) are
  the two course-taught routes; empirical subtypes include experimental,
  lab/wet-lab, computational/simulation (explicitly includes "training an ML
  model on a public dataset"), observational, secondary data analysis, case
  study, content analysis.
- **Variables/hypotheses**: not explicitly named as a standalone
  vocabulary block in the course material; embedded in the instrument design
  guidance (state what is measured, why, and what question types capture
  it) and in Step 1 of the analysis lecture (define which collected
  variables actually bear on the RQ before analyzing any of them).
- **Dataset/sampling**: population (who the study is about, and who the
  conclusions can honestly generalize to) vs sample (who actually
  participated) are explicitly distinct. Convenience sampling is
  course-acceptable if disclosed as such. Target ≥100 responses for a
  quantitative survey where feasible; below ~30, percentages become
  unstable.
- **Instrument design**: state what was measured, item count, question
  types, and why those specific questions were chosen; borrowing a
  validated instrument from a cited source is explicitly flagged as a
  strength. Balance closed (analyzable) and open-ended (depth) questions.
- **Baseline/comparison**: not named as a modeling term in this course (it
  teaches survey/lit-review methodology, not ML baselines) — the closest
  analogue is Step 5–6 of the analysis lecture: every comparison between
  groups must be checked for relevance to the RQ, not just reported because
  it exists.
- **Metrics**: descriptive statistics first (frequency, percentage, mean,
  median, range) before any relationship analysis; every reported number
  must pass "does this help answer the RQ" before it is worth including.
- **Validation**: the closest course concept is the reproducibility
  standard stated at the very start of the course (`course_notes.md` §1.1) —
  a stranger reading only the methodology should be able to repeat the study
  and land on comparable results/comparable article set. For lit reviews
  this is operationalized as reporting the exact search string and date; for
  surveys as reporting the exact instrument and sampling method.
- **Association vs causation**: explicit, repeated warning (course_notes.md
  general theme, reinforced heavily in the newly-read analysis lecture's
  Step 8) — an observed association between two survey variables cannot be
  reported as one causing the other without ruling out confounds; use "X was
  associated with Y," never "X caused Y," unless the design supports it.
- **Limitations**: expected even though absent from the Methodology deck's
  own slide list — the tutor singled it out as the best feature of the
  worked example (`course_notes.md` §8.9a). Two kinds: systematic (from your
  own process — e.g. English-only search) and content (from what the
  included sources actually said, or didn't). Limitations should be
  specific, not the generic "small sample size."
- **Reproducibility**: the course's stated bar throughout — see
  "Validation" above. Also underlies the newly-read analysis lecture's
  insistence on reporting the actual search string, actual sample numbers,
  and disclosing partial screening rather than presenting it as full
  coverage.

### What applies to Sanas

Methodological, not architectural or product, conclusions only. No RQ,
target semantics, or tech stack is proposed here — those remain open per
`GROUND_TRUTH.md` §3.2 and §10.

1. **The course's RQ formula and 10–15 word constraint apply directly** to
   whatever RQ eventually gets set for the RTCI field-experiment paper (a
   causal question about crowding information and boarding decisions) —
   this is squarely the empirical-paper track the course describes (survey/
   field experiment collecting original data), not the literature-review
   track.
2. **The "so what" test and the distribution/comparison/relationship
   triad (Steps 2 and 5 of the newly-read analysis lecture)** are a usable
   checklist for whatever RTCI survey or field-experiment data eventually
   gets analyzed, independent of what the final RQ turns out to be.
3. **The association-vs-causation warning is directly load-bearing for
   RTCI**, since RTCI's entire premise is a claimed causal effect of
   real-time crowding information on boarding decisions. The course's "safe
   language" rule (associated with, not caused by) should gate how any
   correlational pilot or survey result is described in the paper, prior to
   and separate from whatever causal identification strategy the field
   experiment itself uses.
4. **The five/six-component methodology templates (lit review and
   empirical)** are directly reusable skeletons for the RTCI paper's
   methodology section once its design is chosen — but the course does not
   resolve which design fits a field experiment specifically; that is
   closer to "experimental" in the empirical-method landscape (§9.1) than to
   "survey," and the course's survey-specific instrument/sampling guidance
   would need adaptation, not direct reuse.
5. **`literature_review_methodology.docx`** describes a scoping review
   already conducted on a CV/crowd-density research question adjacent to
   Sanas, following the course's five-component structure (design,
   databases, search strategy — with full Boolean string and record counts
   — eligibility criteria placeholder, screening funnel, thematic analysis).
   It is unverified (no corroborating log entry, incomplete eligibility
   criteria section, machine-authored) and must not be cited as a completed
   literature review for the Sanas or RTCI paper without independent
   verification of its claimed record counts and included studies.
6. **The limitations-as-a-required-section pattern** (course_notes.md
   §8.9a, reinforced by Mohlaroy's worked example) should be planned into
   the RTCI paper's methodology from the start, with systematic and content
   limitations kept distinct.
7. **Every one of these is contingent on decisions Diyas has not made yet**
   (RQ wording, design type, dataset, split, metric — see next section) —
   the course supplies the *method for making and reporting those choices*,
   not the choices themselves.

### Open questions for Diyas

1. What is the RTCI paper's exact research question, phrased to the
   course's formula and word budget? (Currently open — no RQ text found
   anywhere in this repo.)
2. Is the RTCI paper empirical (field experiment, per its stated causal
   premise) or does it also need a literature-review component analyzing
   existing crowding-information/passenger-behavior research first? The
   course says every paper needs a literature review regardless of route
   (§3) — has that been scoped yet for RTCI?
3. Which empirical-method subtype does the RTCI field experiment map to —
   the course's "experimental" category (§9.1), and if so, what is the
   controlled variable, what is held constant, and what is the outcome
   measure?
4. What is the sampling frame and target sample size for the RTCI field
   experiment, and is it a convenience sample (course-acceptable if
   disclosed) or something more structured?
5. Is `literature_review_methodology.docx` connected to any actual prior
   work session, or is it an unrelated test/generated artifact that should
   be moved out of `research/coursework/` entirely? Its claimed 604/509/40/
   35/19/11-record funnel is currently uncorroborated anywhere else in the
   repo.
6. Does the RTCI paper's eventual methodology section need the course's
   survey-specific six components (population/sample/instrument/etc.), or
   does a field-experiment design need a different component set the course
   doesn't cover (the course's own landscape table names "experimental" but
   only elaborates the survey path in depth)?
7. Should `video1311882563.mp4` be transcribed properly (faster-whisper, as
   the other six were) so its speaker commentary — currently completely
   inaccessible — can be checked against the slide text the way `README.md`
   already flags for the other six recordings?
8. Is the Almaty-bus-occupancy RQ example in the newly-read analysis deck
   this student's own submitted RQ for the course, or a generic instructor
   example — and if it is the student's own, does that create any
   consistency obligation with the actual Sanas/RTCI research question?
9. What acceptance criteria or validation strategy will the RTCI paper use
   to satisfy the course's reproducibility bar (§1.1) — could an independent
   reader repeat the field experiment and expect comparable results?
10. Given `Methodology for emp. papers.pdf`'s truncation at component 5
    (ethics/consent never covered by any source in this directory), has the
    RTCI field experiment's ethics/consent process been defined anywhere
    outside this course material, given it involves real passengers?

### Traceability table

| Conclusion | Source file | Page/slide/timestamp | Status |
|---|---|---|---|
| RQ formula `How/Why does [factor] affect [outcome] among [group/context]?`, 10–15 words | `Foundations class- group 3.pdf`; `1st lesson- part 1/2.mp4` | deck + `course_notes.md` §2.2–2.3 | explicit |
| Research gap = missing/under-examined, not "never studied" | `video1768002125.mp4` / `video1168216779.mp4` | `course_notes.md` §11.4 | explicit (tutor quote) |
| SMART reading order (title→abstract→intro→conclusion→headings→figures→methods/results last) | `2nd class-Reading.pdf` | `course_notes.md` §5.3 | explicit |
| Summary vs synthesis distinction, "single most important habit" | `Lit. review.pdf`; `video1433589796.mp4` | `course_notes.md` §6.4 | explicit |
| Five-component lit-review methodology (design, databases, search strategy, eligibility, analysis) | `Methodology.pdf`; `video1101430871.mp4` | `course_notes.md` §8.2 | explicit |
| Six-component empirical methodology (design, population, sample, instrument, collection, ethics) | `Methodology for emp. papers.pdf` | `course_notes.md` §9.4 | explicit; ethics component's own text is missing from source (p.15/15 cutoff) |
| Association ≠ causation, "X was associated with Y" not "X caused Y" | `video1311882563.mp4`, Step 8 (p.15) | slide text only, no transcript | explicit (slide), speaker commentary unknown |
| RQ-as-filter: only analyze survey variables that answer the RQ | `video1311882563.mp4`, Step 1 (p.8) | slide text only | explicit (slide) |
| Distribution/comparison/relationship as the three analysis questions | `video1311882563.mp4`, Step 5 (p.12) | slide text only | explicit (slide) |
| Null/contradictory results are still valid findings | `video1311882563.mp4`, Steps 9–10 (pp.16–17) | slide text only | explicit (slide) |
| "Do not graph everything," every figure needs a job | `video1311882563.mp4`, Steps 11–12 (pp.18–19) | slide text only | explicit (slide) |
| Worked Almaty-bus-occupancy RQ example may be a generic teaching device, not connected to actual Sanas RQ | `video1311882563.mp4`, throughout | slide text only | inferred — no corroboration found either way |
| Mohlaroy paper as worked model of theme paragraph, absence-as-finding, specific limitations | `Mohlaroy--RAS.docx`; `video1433589796.mp4` ~30:00 | `course_notes.md` §13 | explicit |
| `literature_review_methodology.docx` describes an already-run scoping review on a Sanas-adjacent CV question | `literature_review_methodology.docx` | full document (10 paragraphs) | explicit in the document; **uncorroborated elsewhere in repo** — treat the document's own claims as unverified |
| `literature_review_methodology.docx` is machine-generated, not course material, modified 2026-08-25 | `literature_review_methodology.docx` core properties | docx metadata | explicit (file property) |
| Course teaches survey/lit-review methodology in depth; field-experiment/RTCI-style causal design is only named, not elaborated | `Methodology for emp. papers.pdf` §9.1 list; absence of further detail | `course_notes.md` §9.1 | inferred from absence |


---

<!-- was: research/PAPER_SPRINT_30D.md -->

**Раздел 7.** Предложение 30-дневного спринта от 2026-08-31. Заменено `SEPTEMBER_PLAN.md`, который покрывает оба трека, а не только статью. Дизайн эксперимента, гипотезы H1-H3 и таблица вырезанного из scope здесь сохранены: в новом плане они не повторяются, а даются ссылкой.

## 30-day empirical paper sprint

Статус: **предложение**, не решение. Пока Дияс не подтвердил, ни
`GROUND_TRUTH.md`, ни `RTCI_RESEARCH_CHARTER.md` не меняются.
Дата составления: 2026-08-31. Целевая дата submission: 2026-09-30.

### Решение о scope

Финализируется **не** TR-C package из `TRC_PAPER_BLUEPRINT.md`. Из четырёх
вкладов блюпринта данных нет ни для одного, а ревью TR-C идёт 3-6 месяцев.

Финализируется **Study A отдельной статьёй**: stated-preference choice
experiment по выбору `board first arriving bus` против `wait for next
service` в зависимости от отображаемой загруженности.

Обоснование: это единственный компонент RTCI-программы, чьи данные
собираются силами команды за 2 недели без устройства, без разрешения на
съёмку в автобусе и без интеграции с Avtobys.

TR-C остаётся долгосрочной целью. Эта статья становится её Study A
reference и цитируется в будущем field-experiment paper.

### Что вырезано и почему

| Вырезано | Причина |
|---|---|
| CV measurement validation | `development/src/` пуст, своих кадров нет. Прогон CSRNet на ShanghaiTech не является валидацией автобусного сенсора и займёт ту же неделю, что и survey |
| Field experiment | Нет разрешения, нет интеграции, нет randomization механизма |
| Transport-system simulation | Нет калибровочных данных по маршруту |
| Hardware bench | Железо не куплено и не собрано |

Упоминание устройства в статье допустимо только как motivation в Intro и
как limitation в Discussion. Никаких заявлений о его точности.

### Целевой журнал

Приоритет: **Travel Behaviour and Society** (Elsevier).
Резерв: Case Studies on Transport Policy, Transport Policy.
Оба принимают SP-исследования из emerging-market контекста.

Не TR-C: SP без field validation не проходит их scope.

### Estimand и гипотезы

Формулируются в Study A терминах, без причинных заявлений о реальном
поведении.

- RQ: как отображаемый уровень загруженности сдвигает выбор между
  ближайшим и следующим отправлением при контроле времени ожидания
  и времени в пути.
- H1: вероятность посадки убывает с ростом отображаемой загруженности.
  Направление согласуется с внешним свидетельством Kapatsila 2026-08-26
  (`research/EXPERT_CONSULTATIONS.md`, статус свидетельства открыт).
- H2: существует willingness-to-wait в минутах за снижение загруженности
  на один уровень.
- H3: эффект неоднороден по цели поездки, дефициту времени и полу.

Hypothetical bias фиксируется как limitation в отдельном разделе. Статья
не заявляет causal field evidence.

### Дизайн эксперимента

- Альтернативы: два ближайших отправления. Consideration set ограничен
  по совету Kapatsila п.1, не полный набор маршрутов.
- Атрибуты: отображаемый уровень загруженности (5 уровней, тот же
  зелёный-красный градиент, что в продукте), время ожидания следующего
  автобуса, время в пути, доступность сидячего места.
- Уровни и D-efficient дизайн фиксируются до пилота.
- 8-10 choice tasks на респондента, 2-3 блока.
- Cognitive pilot n=20 до основного запуска.

### Выборка

- Цель: 400-600 завершённых ответов, пользователи транспорта Алматы.
- Квоты: возраст, пол, частота поездок.
- Каналы: университет, Telegram/Instagram, интерсепт на остановках.
- Минимум для mixed logit: 350 завершённых. Ниже - режем до MNL плюс
  latent class на двух классах.

### График

| Дни | Окно | Что |
|---|---|---|
| 1-4 | 31.08 - 03.09 | Freeze RQ, атрибутов, уровней. D-efficient дизайн. Инструмент на платформе. Preregistration на OSF |
| 5-6 | 04.09 - 05.09 | Cognitive pilot n=20, правки формулировок |
| 7-16 | 06.09 - 15.09 | Полевой сбор. Ежедневный мониторинг квот |
| 12-18 | 11.09 - 17.09 | Параллельно: literature review, 60-100 источников |
| 17-21 | 16.09 - 20.09 | Оценка: MNL, mixed logit, latent class, WTW, эластичности |
| 20-26 | 19.09 - 25.09 | Написание, таблицы, фигуры |
| 27-29 | 26.09 - 28.09 | Внутреннее ревью, язык, форматирование под журнал |
| 30 | 29.09 - 30.09 | Submission package |

### Жёсткие гейты

- **Гейт 1, 03.09:** инструмент готов и preregistration подан. Не готов -
  срок 30 дней сорван, переходим на review paper.
- **Гейт 2, 15.09:** собрано минимум 350 завершённых. Не собрано -
  продлеваем сбор и сдвигаем submission, а не режем анализ.
- **Гейт 3, 25.09:** полный черновик со всеми таблицами существует.

### Блокеры, требующие решения Дияса в ближайшие 72 часа

1. Требует ли его институт ethics/IRB approval на survey. Если да, срок
   рассмотрения решает, выполним ли план вообще.
2. Канал рекрутинга и бюджет на респондентов.
3. Платформа опроса (Qualtrics, Google Forms + рандомизация, собственная).
   Google Forms не даёт нормальный choice experiment без костылей.
4. Соавторы и порядок авторства.

### Текущее состояние литературы

`research/refs/rtci-supporting/` содержит 3 PDF. Для этой статьи нужно
60-100. Это самостоятельный блок работы, стартует параллельно со сбором,
не после.

### Ответственность

- Danyshpan: инструмент, литература, текст, цитаты.
- CADy: фигуры, включая mockup экрана Avtobys для choice tasks.
- Donatello: не задействован в этой статье.
