---
type: source
kind: search-run
citation: Литературный поиск OpenAlex, 2026-08-26
doi:
local: data/litsearch/ (gitignored)
verified: metadata-only
tracks: [rtci]
ingested: 2026-08-26
---

# Поисковый прогон, 2026-08-26

Первый запуск скилла `litsearch`. Одновременно является smoke-тестом
скриптов, поэтому лимиты намеренно малы. Это **не** систематический обзор
чартера §9, а его пилот.

## Что выполнено

**База:** OpenAlex. Дополнительно Crossref для проверки одного DOI.

**Keyword pass**, фильтр по году от 2010:

| Запрос | Всего совпадений | Извлечено |
|---|---:|---:|
| `real-time crowding information passenger boarding decision` | 3201 | 25 |
| `willingness to wait crowded bus next departure` | 2278 | 25 |

Лимит 25 на запрос выставлен вручную. Покрытие сознательно неполное.

**Citation chaining**, один hop, обе стороны, порог `min-seeds 2`,
не более 60 цитирующих работ на seed. Seeds, три работы из чартера §3:

- 10.1016/j.rtbm.2023.100963, Drabicki et al., WTW
- 10.1007/s11116-022-10270-3, Drabicki, Kucharski, Cats, bus bunching
- 10.1111/rssa.12804, Bansal, Hörcher, Graham, crowding disutility

Кандидатов, достижимых минимум от двух seed: 25. Одна запись выпала с
HTTP 404 при добиении метаданных.

**Скрининг:** извлечено 69 записей, дубликатов удалено 4, уникальных 65, уже
известных локально 8, новых кандидатов 57.

Файлы прогона: `data/litsearch/smoke.jsonl`, `chain.jsonl`, `screen.md`,
`smoke.manifest.jsonl`. Каталог gitignored.

## Что дал прогон

Полезное дало chaining, не keyword pass. Топ keyword-выдачи содержал отчёт
RAND по автономным автомобилям и работу про сервисного робота в аэропорту,
то есть шум от полнотекстового поиска OpenAlex. Chaining выдал ядро
литературы по crowding valuation и прямые прецеденты RTCI:
[[crowding-valuation-and-rtci-evidence]].

Три конкретных результата:

1. Найдена работа консультанта: Kapatsila, Bahamonde-Birke, van Lierop,
   Grisé, Transportation, 2025, DOI 10.1007/s11116-025-10585-x.
2. Найдены DOI для двух работ, на которые чартер ссылался внешними
   ссылками: Stockholm pilot (Zhang, Jenelius, Kottenhoff, 2016) и Kim, Lee,
   Oh (2009).
3. Найдена свежая работа Drabicki, Cats, Kucharski (2024) про влияние
   пандемии на WTW при RTCI, отсутствующая в чартере.

## Ограничения этого прогона

- 25 записей на запрос, два запроса. Concept blocks чартера §9 полностью не
  прогонялись.
- Один hop chaining. Второй hop не запускался.
- Scopus, Web of Science и TRID не использовались: нужен институциональный
  доступ.
- Semantic Scholar отдаёт HTTP 429 без ключа.
- Ни одна найденная работа не прочитана. Всё, что записано, является
  метаданными.
