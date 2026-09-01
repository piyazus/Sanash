# Research wiki index

Каталог страниц. Конвенции: [[WIKI_SCHEMA]]. Хронология: [[log]].

Wiki хранит внешнее знание и его синтез. Состояние проекта живёт в
`GROUND_TRUTH.md`, история запусков в `development/experiments/log.md`.

## Sources

| Страница | Вид | Что даёт | Проверено |
|---|---|---|---|
| [[2026-08-26-kapatsila-consultation]] | consultation | Практический взгляд на elasticity, рекалибровку, пиковость crowding и мотивацию агентства | запись, только 8:52 хвоста |
| [[cv-hardware-corpus-2026-08-26]] | batch | 44 PDF по CV, сенсорам, edge и валидации APC | заголовок, авторы, площадка сверены; методы и метрики нет |
| [[litsearch-run-2026-08-26]] | search-run | Первый прогон litsearch: 2 запроса OpenAlex плюс один hop citation chaining от трёх seed | метаданные |

## Concepts

| Страница | Что синтезирует | Источников |
|---|---|---|
| [[crowding-elasticity]] | Эластичность как альтернатива тяжёлой поведенческой модели; конфликт с чартером | 1 |
| [[willingness-to-wait]] | WTW, опорные работы, направление эффекта, следствие для survey | 2 |
| [[occupancy-sensing-methods]] | Место установки, модальность сенсора, постановка задачи; ранжированные пары как дешёвая разметка | 1 |
| [[edge-inference-constraints]] | Прецеденты Jetson Nano, NPU, embedded; чего они не снимают для Orin Nano | 1 |
| [[apc-validation-and-standards]] | VDV 457 и partitioned equivalence test как готовый инструмент под Study B | 1 |
| [[crowding-valuation-and-rtci-evidence]] | Ядро литературы по crowding valuation, прецеденты RTCI, работа консультанта | chaining, 24 записи |

## Ещё не инжестировано

- `research/refs/rtci-supporting/drabicki2023_rtbm.pdf`
- `research/refs/rtci-supporting/pi2018_perception.pdf`
- `research/refs/rtci-supporting/zhangkennedy2023_chi.pdf`
- `rtci_paper_inventory.csv`, 12 seed papers
- Отдельные страницы источников для работ из
  [[cv-hardware-corpus-2026-08-26]], которые дойдут до цитирования. Батчевая
  страница является триажом, а не citation-ready проверкой
