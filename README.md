# Sanas

Sanas исследует, как real-time crowding information в Avtobys причинно меняет
решение пассажира Алматы: сесть в первый автобус или ждать следующий.
Потолочная RGB-система является измерительным слоем для этого RTCI field
experiment и должна выдавать пять уровней + непрерывный score `0..1`.

## Начать здесь

1. Прочитать [`GROUND_TRUTH.md`](GROUND_TRUTH.md). Это единственный источник
   истины о текущем продукте, фактах, гипотезах и блокерах.
2. Выполнить `git status --short --branch`. Текущий архитектурный пивот пока
   находится поверх коммита `a971279` и не зафиксирован как воспроизводимый
   baseline.
3. Для истории запусков и решений читать
   [`development/experiments/log.md`](development/experiments/log.md). Это
   хронология, а не current-state spec.

## Честный статус

Текущая потолочная RGB-архитектура ещё не реализована и не обучалась.
`development/src/` и `development/notebooks/` пусты. Оставшиеся door/depth
скрипты относятся к отменённой архитектуре и сейчас неработоспособны из-за
удалённых модулей.

На диске есть внешние crowd-датасеты и выборка прежнего APC-эксперимента.
Полный 73.5 GB Gorelik archive не скачан. Ограничения и точный инвентарь
зафиксированы в Ground Truth и
[`development/datasets.md`](development/datasets.md).

## Карта репозитория

- `GROUND_TRUTH.md`: текущие решения, проверенные факты, открытые вопросы и
  порядок действий.
- `CLAUDE.md`: правила работы агентов в этом репозитории.
- `development/experiments/log.md`: append-only история экспериментов и
  архитектурных решений.
- `development/datasets.md`: provenance, hashes, лицензии и измеренный состав
  локальных датасетов.
- `development/TECH_DATA_OPTIONS.md`: проверенные public datasets, model
  candidates, prototype architecture и месячные engineering gates.
- `research/RTCI_RESEARCH_CHARTER.md`: research question, hypotheses,
  survey-to-field methodology, outcomes и operational requirements.
- `research/FIELD_EXPERIMENT_PROTOCOL.md`: preregistration draft, event schema,
  estimands, randomization и analysis contract.
- `research/TRC_PAPER_BLUEPRINT.md`: структура и evidence gates статьи Part C.
- `business/INNOFORCE_RTCI_PILOT_BRIEF.md`: вопросы и требования к пилоту с
  Innoforce/Avtobys.
- `ARCHIVE.md`: мёртвые документы, сведённые в один файл 2026-09-03. Прежний
  мастер-документ, findings и выводы по CV-треку до пивота, related-work
  заметки door/APC и отменённый шаблон письма. Только история, не current spec.
- `research/refs/rtci-supporting/`: литература основного RTCI research track.
- `research/coursework/`: отдельные учебные материалы.
- `business/outreach/`: контакты; актуальных pitch-материалов под новую
  архитектуру пока нет.
- `data/`, `outputs/`: локальные, gitignored данные и артефакты.

## Правило конфликтов

Если документ, чат, vault или агент противоречит `GROUND_TRUTH.md`, действие
не продолжается по памяти. Сначала проверяется первичный артефакт. После нового
решения обновляются одновременно append-only log и Ground Truth.
