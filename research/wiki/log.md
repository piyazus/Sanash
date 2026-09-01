# Research wiki log

Append-only. Формат строки: `## [YYYY-MM-DD] операция | название`.

## [2026-08-26] setup | введена схема wiki

Паттерн LLM Wiki (Karpathy) инстанциирован для RTCI-трека. Созданы
`WIKI_SCHEMA.md`, `index.md`, `log.md`, каталоги `sources/` и `concepts/`.
Сырые PDF остаются в `research/refs/`, транскрипты в `data/consultations/`
(gitignored).

## [2026-08-26] ingest | Консультация Bogdan Kapatsila

Источник: локальная запись Zoom, 8 минут 52 секунды, транскрибирована
faster-whisper medium на CPU. Создана
`sources/2026-08-26-kapatsila-consultation.md`. Созданы
`concepts/crowding-elasticity.md` и `concepts/willingness-to-wait.md`.
Зафиксировано неразрешённое противоречие с чартером и блюпринтом по поводу
ценности behaviour-aware model. `GROUND_TRUTH.md` не менялся: решения нет.

## [2026-08-26] verify | Вердикты refgraph по батчу CV/hardware

Все 44 PDF из `research/refs/cv-hardware-corpus/` заново извлечены
`extract_pdf_metadata.py` и сверены с таблицей `REFGRAPH_REPORT.md`.
Заголовок, авторы и предмет совпали у всех 44. Отдельно проверены семь
утверждений с конкретикой сверх заголовка (99%/97% точность стереовидения,
датасет PCDS с дверей автобусов, WMATA и список моделей в CCTV-работе,
embedded-развёртывание MCNet, NPU-платформа, CORN, PFCASA поверх CSRNet):
все подтвердились в тексте.

Найдены три дефекта отчёта, исправления дописаны в него разделом
`Verification pass 2026-08-26`: неверная арифметика Summary (написано
keep 42, фактически keep 44, duplicate 8, reject 4), устаревший раздел про
местоположение файлов в `tmp/`, и устаревший borderline-статус `zhao2021.pdf`,
который на деле является прямо профильной работой по классификации
заполненности салона автобуса.

## [2026-08-26] ingest | Корпус CV/hardware, 44 PDF

Создана `sources/cv-hardware-corpus-2026-08-26.md` с проверенными
библиографическими строками, сгруппированными по назначению. Созданы
`concepts/occupancy-sensing-methods.md`, `concepts/edge-inference-constraints.md`,
`concepts/apc-validation-and-standards.md`. `GROUND_TRUTH.md` не менялся:
батч не закрывает ни одного открытого решения, он только показывает
доступные варианты.

## [2026-08-26] search | Первый прогон litsearch по RTCI

Написан скилл `.claude/skills/litsearch/` (SKILL.md плюс три скрипта:
`oa_search.py`, `oa_chain.py`, `screen.py`). Прогон: два keyword-запроса
OpenAlex с лимитом 25 и один hop citation chaining от трёх seed-DOI чартера.
69 записей, 65 уникальных, 57 новых кандидатов.

Найдено: работа консультанта Kapatsila et al. 2025 в Transportation,
DOI для Stockholm pilot и Kim/Lee/Oh взамен внешних ссылок чартера, и
отсутствующая в чартере работа Drabicki, Cats, Kucharski 2024 про WTW после
пандемии. Созданы `sources/litsearch-run-2026-08-26.md` и
`concepts/crowding-valuation-and-rtci-evidence.md`. Всё на уровне метаданных,
ни одна работа не прочитана. Чартер пока не менялся.

