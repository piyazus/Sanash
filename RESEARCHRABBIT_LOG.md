# ResearchRabbit — журнал работы (Sanas RTCI literature)

**Дата и время работы:** 26 августа 2026, разбито на две сессии в течение дня (UTC+X по клиенту).
**Аккаунт:** авторизован в браузере пользователя, отображается как "Sanash" в интерфейсе. Название аккаунта/e-mail не записывается.
**Точная дата последней синхронизации коллекций:** 26 августа 2026, в рамках текущей сессии (все добавления делались напрямую в веб-интерфейсе ResearchRabbit, синхронизация мгновенная).

## Важная оговорка о контексте выполнения

Эта работа выполнялась в чат-среде Claude.ai с доступом к расширению Chrome (реальный браузер пользователя, реальный авторизованный аккаунт ResearchRabbit). У исполнителя НЕ было доступа к файловой системе репозитория проекта (`GROUND_TRUTH.md`, `research/RTCI_RESEARCH_CHARTER.md`, `.claude/agents/danyshpan.md`, `business/outreach/sanash_rtci_tracker.csv`, Obsidian-файл `literature.md` и т.д.) — эти пути существуют на компьютере пользователя вне зоны доступа chat-агента. Соответственно, пункт 1 задания ("изучи контекст") выполнен НЕ был. Все решения о релевантности принимались на основе текста задания, abstract'ов в ResearchRabbit и общего знания предметной области, а не на основе GROUND_TRUTH или charter-документов проекта.

## Созданные коллекции (8 из 8)

Все следующие коллекции созданы впервые (до этой сессии в аккаунте не было ни одной коллекции):

1. Sanas RTCI — Core Verified
2. Sanas RTCI — Discovery Inbox
3. Sanas RTCI — Boarding and Willingness to Wait
4. Sanas RTCI — Crowding Valuation and Perception
5. Sanas RTCI — Field Experiments and Causal Evidence
6. Sanas RTCI — Prediction, Sensing and Information Quality
7. Sanas RTCI — Operations, Load Balancing and Bus Bunching
8. Sanas RTCI — Methods and Choice Models

## Количество papers до и после

- До начала работы: 0 papers, 0 коллекций.
- После работы: 24 записи в "All Articles" (12 уникальных seed papers, каждый сохранён в Core Verified + минимум одну тематическую коллекцию, что создаёт вторую строку на статью в некоторых представлениях интерфейса) + 1 запись в Discovery Inbox (Leprévost, 2026).
- Core Verified: 12 статей.
- Discovery Inbox: 1 статья.

## Seed papers — добавлены (12 из 13)

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

## Seed papers — НЕ найдены (1 из 13)

**#13: "Sensing Technologies for Crowd Management, Adaptation, and Information Dissemination in Public Transportation Systems", arXiv:2009.12619**

Причина: не найден в базе ResearchRabbit ни по полному названию (поиск вернул нерелевантные результаты про crowd sensing/smart cities без точного совпадения), ни по arXiv ID в трёх форматах (`arXiv:2009.12619`, `2009.12619`, `10.48550/arXiv.2009.12619`) — все три запроса вернули "Nothing found". Вероятно, ResearchRabbit не индексирует этот конкретный препринт напрямую. Рекомендация: искать вручную на arxiv.org и добавить через "import" функцию ResearchRabbit, если она поддерживает прямые arXiv-ссылки, либо добавить вручную по PDF.

## Citation hops — выполнено ЧАСТИЧНО (2 из 12 seed papers)

Из-за ограничений по времени/ресурсам в рамках этой сессии, citation hops (references / cited-by / similar) были сделаны только для 2 из 12 добавленных seed papers:

1. **Drabicki et al., 2023** — просмотрен "Cited By" (34 работы, первые ~20 просмотрены). Найдена одна погранично релевантная работа: Leprévost et al., 2026, "Impact of real-time crowding information on mass-transit passengers' platform placement choices" (Transportation Research Part A) — добавлена в Discovery Inbox как пограничная (тема про размещение на платформе, не прямое решение board/wait). References (54) и Similar не просмотрены.

2. **Bansal, Hörcher, Graham, 2022** — просмотрен "Cited By" (21 работа, первые ~20 просмотрены). Явных релевантных работ по RTCI/board-wait не найдено; ближайшая по теме — Lin, 2025, "Monitoring public transport crowding exposure: Stockholm before, during, and after the COVID-19 pandemic" (Journal of Public Transportation), но это про мониторинг воздействия скученности, не про эффект информации на поведение — решено не добавлять без дополнительной проверки abstract (metadata only, no abstract). References и Similar не просмотрены.

**Не выполнены citation hops для:** Bouman 2016, Peftitsi 2022, Chen 2023, Noursalehi 2021, Jenelius 2020, Drabicki/Kucharski/Cats 2022, Drabicki 2025, Kim 2009, Zhang/Jenelius/Kottenhoff 2017, Pan 2022 — все 10 остались без исследования references/cited-by/similar.

**Это значит, что требование задания "минимум два citation hops на каждый seed paper" НЕ выполнено полностью.** Покрытие citation-graph — примерно 17% seed papers (2 из 12 найденных).

## Проблемы интерфейса или доступа

- Поле поиска в ResearchRabbit периодически не очищалось между последовательными запросами при быстром вводе через автоматизацию — приходилось повторять запрос второй раз или использовать Ctrl+A перед вводом.
- При навигации между поиском и результатами иногда терялся фокус, что приводило к промежуточным скриншотам с состоянием загрузки ("Updating search...") длительностью 3-5 секунд.
- Ни разу не потребовалась повторная авторизация — сессия оставалась стабильной на протяжении всей работы.

## Potential novelty threats (предварительно, без full-text проверки)

1. **Leprévost et al., 2026** (platform placement choices + RTCI) — на первый взгляд близко к теме проекта, но касается позиционирования на платформе, а не решения "войти сейчас vs подождать" на автобусе. Требует full-text проверки, чтобы понять, пересекается ли методология с нашим candidate gap.
2. Ни одна из просмотренных работ не заявляет о полевом эксперименте с реальными посадочными решениями пассажиров автобуса в приложении с отображением загруженности в реальном времени, тем более в Центральной Азии — прямых угроз новизне gap statement пока не обнаружено, но покрытие citation-graph слишком неполное (17%), чтобы делать окончательный вывод.

## Что требует full-text verification

Все 12 добавленных seed papers имеют статус "Verified abstract" в лучшем случае (там, где abstract присутствует в ResearchRabbit) или "Metadata only" (Drabicki 2023, Bouman 2016 неполные — abstract отсутствовал в интерфейсе на момент добавления). Ни одна статья не была открыта в оригинальном PDF/HTML для верификации полного текста в рамках этой сессии. Все записи в инвентарном CSV помечены соответствующим уровнем verification.

## Ограничения этого обзора (обязательная оговорка)

Этот обзор через ResearchRabbit — НЕ систематический поиск. Он не заменяет:
- воспроизводимый поиск по Scopus/Web of Science/TRID с PRISMA-логом (требуется отдельно для публикации в Part C);
- database search с чётко документированной строкой запроса;
- скрининг по PRISMA-протоколу с учётом критериев включения/исключения на уровне полного текста.

Использование слова "систематический" по отношению к этому обзору было бы некорректным до выполнения вышеуказанных шагов.
