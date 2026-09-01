---
type: concept
tracks: [rtci]
updated: 2026-08-26
sources: 2
---

# Willingness to wait (WTW)

Выбор пассажира между посадкой в более загруженное первое отправление и
ожиданием следующего, менее загруженного.

## Опорные работы

Перенесено из `research/RTCI_RESEARCH_CHARTER.md` §3, страницы источников
ещё не созданы:

- Drabicki et al., stated-preference WTW для bus/tram, discrete choice:
  https://doi.org/10.1016/j.rtbm.2023.100963. Локально:
  `research/refs/rtci-supporting/drabicki2023_rtbm.pdf`.
- Drabicki, Kucharski, Cats: WTW в динамической симуляции, bus bunching:
  https://doi.org/10.1007/s11116-022-10270-3
- Bansal, Hörcher, Graham: crowding disutility по smart-card и AVL данным:
  https://doi.org/10.1111/rssa.12804

## Внешнее подтверждение направления

Из [[2026-08-26-kapatsila-consultation]]: в его собственной работе после
контроля travel time crowding information вносит вклад в выбор, и рост
загруженности снижает вероятность посадки. Ссылка и величина эффекта не
получены.

Это согласуется с направлением H1 чартера, но не является величиной, годной
для power analysis.

Поиск 2026-08-26 нашёл его работу: Kapatsila, Bahamonde-Birke, van Lierop,
Grisé, "The effect of crowding level information provision on the revealed
route choice of transit riders", Transportation, 2025,
DOI 10.1007/s11116-025-10585-x. Проверено по OpenAlex и Crossref, только
метаданные, abstract недоступен ни там, ни там. Постановка у него route
choice, а не boarding versus waiting. Подробнее:
[[crowding-valuation-and-rtci-evidence]].

## Уточнение дизайна survey

Тот же источник: пассажир сравнивает не десятки альтернатив, а верх списка.
Следовательно choice tasks в Study A строятся вокруг двух-трёх ближайших
отправлений. Статус: кандидат на изменение survey design, не принято.

## Чего не хватает

- Полный текст Kapatsila et al. 2025: без него effect size неизвестен.
- Страниц источников для трёх работ выше, с полями evidence matrix.
- Baseline вероятности посадки в первый автобус для Алматы. Такой величины в
  литературе не будет, её даёт только пилот.
