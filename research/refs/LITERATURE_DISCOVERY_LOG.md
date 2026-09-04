# Литературный discovery log

Собран 2026-09-03 из `RESEARCHRABBIT_LOG.md` и `RTCI_EVIDENCE_MAP.md`, которые
лежали в корне репозитория. Содержание перенесено целиком.

Это лог разведки и карта того, что было найдено, а не источник истины по
цитатам. Источник истины: [`base/references.md`](base/references.md).

Важная оговорка. Сессия ResearchRabbit исполнялась без доступа к
`GROUND_TRUTH.md` и к charter, что признано в самом логе (раздел «Важная
оговорка о контексте выполнения»). Суждения о релевантности в разделе 1
принимались по abstract и общему знанию области, а не по проектному контексту.
Проверять перед использованием.


---

<!-- was: RESEARCHRABBIT_LOG.md -->

**Раздел 1.** Журнал сессии ResearchRabbit, 26 августа 2026.

## ResearchRabbit — журнал работы (Sanas RTCI literature)

**Дата и время работы:** 26 августа 2026, разбито на две сессии в течение дня (UTC+X по клиенту).
**Аккаунт:** авторизован в браузере пользователя, отображается как "Sanash" в интерфейсе. Название аккаунта/e-mail не записывается.
**Точная дата последней синхронизации коллекций:** 26 августа 2026, в рамках текущей сессии (все добавления делались напрямую в веб-интерфейсе ResearchRabbit, синхронизация мгновенная).

### Важная оговорка о контексте выполнения

Эта работа выполнялась в чат-среде Claude.ai с доступом к расширению Chrome (реальный браузер пользователя, реальный авторизованный аккаунт ResearchRabbit). У исполнителя НЕ было доступа к файловой системе репозитория проекта (`GROUND_TRUTH.md`, `research/RTCI_RESEARCH_CHARTER.md`, `.claude/agents/danyshpan.md`, `business/outreach/sanash_rtci_tracker.csv`, Obsidian-файл `literature.md` и т.д.) — эти пути существуют на компьютере пользователя вне зоны доступа chat-агента. Соответственно, пункт 1 задания ("изучи контекст") выполнен НЕ был. Все решения о релевантности принимались на основе текста задания, abstract'ов в ResearchRabbit и общего знания предметной области, а не на основе GROUND_TRUTH или charter-документов проекта.

### Созданные коллекции (8 из 8)

Все следующие коллекции созданы впервые (до этой сессии в аккаунте не было ни одной коллекции):

1. Sanas RTCI — Core Verified
2. Sanas RTCI — Discovery Inbox
3. Sanas RTCI — Boarding and Willingness to Wait
4. Sanas RTCI — Crowding Valuation and Perception
5. Sanas RTCI — Field Experiments and Causal Evidence
6. Sanas RTCI — Prediction, Sensing and Information Quality
7. Sanas RTCI — Operations, Load Balancing and Bus Bunching
8. Sanas RTCI — Methods and Choice Models

### Количество papers до и после

- До начала работы: 0 papers, 0 коллекций.
- После работы: 24 записи в "All Articles" (12 уникальных seed papers, каждый сохранён в Core Verified + минимум одну тематическую коллекцию, что создаёт вторую строку на статью в некоторых представлениях интерфейса) + 1 запись в Discovery Inbox (Leprévost, 2026).
- Core Verified: 12 статей.
- Discovery Inbox: 1 статья.

### Seed papers — добавлены (12 из 13)

| # | Автор, год | Найден по | Коллекции |
|---|---|---|---|
| 1 | Drabicki et al., 2023 (Willingness to wait) | DOI 10.1016/j.rtbm.2023.100963 | Core Verified |
| 2 | Drabicki, Kucharski, Cats, 2022 (Bus bunching) | DOI 10.1007/s11116-022-10270-3 | Core Verified, Operations/Bus Bunching |
| 3 | Bansal, Hörcher, Graham, 2022 | DOI 10.1111/rssa.12804 | Core Verified, Crowding Valuation |
| 4 | Jenelius, 2020 | DOI 10.1016/j.trc.2020.102647 | Core Verified, Prediction/Sensing |
| 5 | Noursalehi, Koutsopoulos, Zhao, 2021 | DOI 10.1016/j.trc.2021.103139 | Core Verified, Prediction/Sensing |
| 6 | Bouman et al., 2016 | DOI 10.1016/j.trc.2016.05.007 | Core Verified, Boarding/WTW |
| 7 | Peftitsi, Jenelius, Cats, 2022 | DOI 10.1016/j.tra.2022.10.011 | Core Verified, Prediction/Sensing |
| 8 | Chen et al., 2023 ("I can board, but I'd rather wait") | DOI 10.1016/j.tra.2023.103747 | Core Verified, Boarding/WTW |
| 9 | Drabicki et al., 2025 (COVID + RTCI willingness to wait) | DOI 10.1016/j.tbs.2024.100895 | Core Verified, Boarding/WTW |
| 10 | Kim, Lee, Oh, 2009 (Seoul bus choice model) | Точное название | Core Verified, Boarding/WTW |
| 11 | Zhang, Jenelius, Kottenhoff, 2017 (Stockholm metro pilot) | Точное название | Core Verified, Field Experiments |
| 12 | Pan et al., 2022 ("Would You Wait?") | Точное название | Core Verified, Boarding/WTW |

### Seed papers — НЕ найдены (1 из 13)

**#13: "Sensing Technologies for Crowd Management, Adaptation, and Information Dissemination in Public Transportation Systems", arXiv:2009.12619**

Причина: не найден в базе ResearchRabbit ни по полному названию (поиск вернул нерелевантные результаты про crowd sensing/smart cities без точного совпадения), ни по arXiv ID в трёх форматах (`arXiv:2009.12619`, `2009.12619`, `10.48550/arXiv.2009.12619`) — все три запроса вернули "Nothing found". Вероятно, ResearchRabbit не индексирует этот конкретный препринт напрямую. Рекомендация: искать вручную на arxiv.org и добавить через "import" функцию ResearchRabbit, если она поддерживает прямые arXiv-ссылки, либо добавить вручную по PDF.

### Citation hops — выполнено ЧАСТИЧНО (2 из 12 seed papers)

Из-за ограничений по времени/ресурсам в рамках этой сессии, citation hops (references / cited-by / similar) были сделаны только для 2 из 12 добавленных seed papers:

1. **Drabicki et al., 2023** — просмотрен "Cited By" (34 работы, первые ~20 просмотрены). Найдена одна погранично релевантная работа: Leprévost et al., 2026, "Impact of real-time crowding information on mass-transit passengers' platform placement choices" (Transportation Research Part A) — добавлена в Discovery Inbox как пограничная (тема про размещение на платформе, не прямое решение board/wait). References (54) и Similar не просмотрены.

2. **Bansal, Hörcher, Graham, 2022** — просмотрен "Cited By" (21 работа, первые ~20 просмотрены). Явных релевантных работ по RTCI/board-wait не найдено; ближайшая по теме — Lin, 2025, "Monitoring public transport crowding exposure: Stockholm before, during, and after the COVID-19 pandemic" (Journal of Public Transportation), но это про мониторинг воздействия скученности, не про эффект информации на поведение — решено не добавлять без дополнительной проверки abstract (metadata only, no abstract). References и Similar не просмотрены.

**Не выполнены citation hops для:** Bouman 2016, Peftitsi 2022, Chen 2023, Noursalehi 2021, Jenelius 2020, Drabicki/Kucharski/Cats 2022, Drabicki 2025, Kim 2009, Zhang/Jenelius/Kottenhoff 2017, Pan 2022 — все 10 остались без исследования references/cited-by/similar.

**Это значит, что требование задания "минимум два citation hops на каждый seed paper" НЕ выполнено полностью.** Покрытие citation-graph — примерно 17% seed papers (2 из 12 найденных).

### Проблемы интерфейса или доступа

- Поле поиска в ResearchRabbit периодически не очищалось между последовательными запросами при быстром вводе через автоматизацию — приходилось повторять запрос второй раз или использовать Ctrl+A перед вводом.
- При навигации между поиском и результатами иногда терялся фокус, что приводило к промежуточным скриншотам с состоянием загрузки ("Updating search...") длительностью 3-5 секунд.
- Ни разу не потребовалась повторная авторизация — сессия оставалась стабильной на протяжении всей работы.

### Potential novelty threats (предварительно, без full-text проверки)

1. **Leprévost et al., 2026** (platform placement choices + RTCI) — на первый взгляд близко к теме проекта, но касается позиционирования на платформе, а не решения "войти сейчас vs подождать" на автобусе. Требует full-text проверки, чтобы понять, пересекается ли методология с нашим candidate gap.
2. Ни одна из просмотренных работ не заявляет о полевом эксперименте с реальными посадочными решениями пассажиров автобуса в приложении с отображением загруженности в реальном времени, тем более в Центральной Азии — прямых угроз новизне gap statement пока не обнаружено, но покрытие citation-graph слишком неполное (17%), чтобы делать окончательный вывод.

### Что требует full-text verification

Все 12 добавленных seed papers имеют статус "Verified abstract" в лучшем случае (там, где abstract присутствует в ResearchRabbit) или "Metadata only" (Drabicki 2023, Bouman 2016 неполные — abstract отсутствовал в интерфейсе на момент добавления). Ни одна статья не была открыта в оригинальном PDF/HTML для верификации полного текста в рамках этой сессии. Все записи в инвентарном CSV помечены соответствующим уровнем verification.

### Ограничения этого обзора (обязательная оговорка)

Этот обзор через ResearchRabbit — НЕ систематический поиск. Он не заменяет:
- воспроизводимый поиск по Scopus/Web of Science/TRID с PRISMA-логом (требуется отдельно для публикации в Part C);
- database search с чётко документированной строкой запроса;
- скрининг по PRISMA-протоколу с учётом критериев включения/исключения на уровне полного текста.

Использование слова "систематический" по отношению к этому обзору было бы некорректным до выполнения вышеуказанных шагов.


---

<!-- was: RTCI_EVIDENCE_MAP.md -->

**Раздел 2.** Карта evidence по темам.

## RTCI Evidence Map — Sanas project

Собрано на основе 12 seed papers, добавленных в ResearchRabbit 26 августа 2026, плюс 1 работа из частичного citation hop (Leprévost 2026). Покрытие citation-graph неполное (см. раздел 1 этого файла) — эта карта отражает только то, что было найдено, а не исчерпывающий обзор темы.

### WTW and boarding choice

- **Drabicki et al., 2023** (10.1016/j.rtbm.2023.100963) — Verified abstract отсутствует в системе (metadata only). Central seed paper по формулировке "should I stay or should I board".
- **Chen et al., 2023** (10.1016/j.tra.2023.103747) — Metadata only. Revealed-preference smart-card evidence: пассажиры метро активно ждут следующего поезда ради места, даже при наличии стоячих мест.
- **Bouman et al., 2016** (10.1016/j.trc.2016.05.007) — Metadata only. Теоретико-игровая (minority game) рамка для решений о посадке под влиянием информации о вместимости.
- **Kim, Lee, Oh, 2009** (10.3141/2112-15) — Verified abstract. Сеул, binary logit модель выбора автобуса на основе occupancy information; ближайший bus-specific прецедент.
- **Pan et al., 2022** (10.1177/03611981221076843) — Verified abstract. Incentive-based willingness to wait для автобусов (goal-framing theory), не RTCI напрямую, но методологически смежно.
- **Drabicki et al., 2025** (10.1016/j.tbs.2024.100895) — Metadata only. COVID + RTCI willingness to wait.

### Crowding valuation

- **Bansal, Hörcher, Graham, 2022** (10.1111/rssa.12804) — Verified abstract. Dynamic choice model на revealed-preference smart-card данных; valuation of crowding увеличивается на 47% при экстремальной перегрузке. Это работа ментора проекта (Prateek Bansal).

### RTCI field experiments

- **Zhang, Jenelius, Kottenhoff, 2017** (10.1007/s12469-016-0150-y) — Verified abstract. **Единственный найденный реальный полевой пилот RTCI** — Стокгольмское метро, 6-дневный тестовый период, per-car информация. Ближайший компаратор к амбициям Sanas по полевому эксперименту, но: метро (не автобус), Швеция (не Центральная Азия).
- **Drabicki, Kucharski, Cats, 2022** (10.1007/s11116-022-10270-3) — Verified abstract. Симуляция на кейс-стади Варшавы (не реальный полевой эксперимент, а simulation + stated-preference модель).

### Revealed-preference evidence

- **Chen et al., 2023** — смарт-карты метро, boarding delay choice.
- **Kim, Lee, Oh, 2009** — Сеул, автобусы, но методология survey-based (не чисто revealed).
- **Bansal, Hörcher, Graham, 2022** — крупномасштабные smart-card + AVL данные азиатского метро.

Прямых revealed-preference данных о посадке на автобус под влиянием app-displayed RTCI **не найдено** — это подтверждает (в пределах неполного покрытия обзора) обозначенный gap проекта.

### Choice models

- Bansal, Hörcher, Graham, 2022 — dynamic choice model (DCM) с переключением между compensatory/habit rules.
- Kim, Lee, Oh, 2009 — binary logit.
- Pan et al., 2022 — binary logit + mixed multinomial logit.
- Drabicki, Kucharski, Cats, 2022 — boarding choice model внутри симуляции.

### Prediction and sensing

- Jenelius, 2020 (10.1016/j.trc.2020.102647) — Metadata only, personalized predictive crowding.
- Noursalehi, Koutsopoulos, Zhao, 2021 (10.1016/j.trc.2021.103139) — Metadata only, predictive decision support platform.
- Peftitsi, Jenelius, Cats, 2022 (10.1016/j.tra.2022.10.011) — Metadata only, моделирование эффекта RTCI на распределение пассажиров по вагонам поезда.

### Information accuracy and trust

Не найдено ни одной работы в этом наборе, напрямую посвящённой accuracy/trust RTCI. Требуется отдельный целевой поиск (не выполнен в этой сессии).

### Load distribution and operations

- Drabicki, Kucharski, Cats, 2022 — bus bunching mitigation через RTCI.
- Peftitsi, Jenelius, Cats, 2022 — car-level distribution в поездах.

### Geographic evidence

- Швеция (Стокгольм) — Zhang et al. 2017.
- Польша (Варшава) — Drabicki, Kucharski, Cats 2022 (симуляция).
- Южная Корея (Сеул) — Kim, Lee, Oh 2009.
- Остальные seed papers — географию нельзя определить без полного текста (metadata only).
- **Центральная Азия / Казахстан / постсоветские транзитные системы: не найдено ни одной работы.** Это согласуется с заявленным geographic gap проекта, но обзор недостаточно полный, чтобы утверждать отсутствие таких работ окончательно.

### Confirmed research gaps

В пределах найденного (12 seed papers + 1 hop-находка):
- Нет найденной работы с рандомизированным или квази-рандомизированным полевым экспериментом RTCI на автобусах.
- Нет найденной работы с app-level A/B тестом отображения загруженности.
- Нет найденной работы из Центральной Азии или постсоветских транзитных систем.
- Нет найденной работы, объединяющей revealed-preference данные о посадке на автобус с RTCI-дисплеем конкретно (Kim et al. 2009 близко, но это survey-based, не revealed от app).

**Важная оговорка:** это не подтверждение отсутствия таких работ в мировой литературе — это отражение того, что не найдено в рамках 12 seed papers и частичного citation hop (2 из 12, ~17% покрытия graph). Для окончательного вывода о gap нужен systematic search по Scopus/WoS/TRID.

### Studies threatening our novelty

Помечены как `potentially challenges novelty` (per задание, пункт 8):

1. **Zhang, Jenelius, Kottenhoff, 2017** — реальный полевой пилот RTCI в метро. Не автобус, не Центральная Азия, но методологически ближайший конкурент по формату "реальный пилот + реальные пассажиры".
2. **Chen et al., 2023** — revealed-preference boarding-delay поведение по смарт-карт данным метро. Показывает, что подобные revealed-preference методы применялись, хотя не для автобусов и не для RTCI-дисплея конкретно.
3. **Leprévost et al., 2026** (найдена через citation hop, Discovery Inbox) — RTCI и platform placement choices; требует full-text проверки на предмет пересечения с нашей методологией.

Ни одна из найденных работ не делает практически то же самое, что заявленный gap Sanas (app-displayed RTCI → actual bus boarding decisions, Центральная Азия) — но при 17%-ном покрытии citation graph это предварительный, не окончательный вывод.
