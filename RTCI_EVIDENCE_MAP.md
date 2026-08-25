# RTCI Evidence Map — Sanas project

Собрано на основе 12 seed papers, добавленных в ResearchRabbit 26 августа 2026, плюс 1 работа из частичного citation hop (Leprévost 2026). Покрытие citation-graph неполное (см. `RESEARCHRABBIT_LOG.md`) — эта карта отражает только то, что было найдено, а не исчерпывающий обзор темы.

## WTW and boarding choice

- **Drabicki et al., 2023** (10.1016/j.rtbm.2023.100963) — Verified abstract отсутствует в системе (metadata only). Central seed paper по формулировке "should I stay or should I board".
- **Chen et al., 2023** (10.1016/j.tra.2023.103747) — Metadata only. Revealed-preference smart-card evidence: пассажиры метро активно ждут следующего поезда ради места, даже при наличии стоячих мест.
- **Bouman et al., 2016** (10.1016/j.trc.2016.05.007) — Metadata only. Теоретико-игровая (minority game) рамка для решений о посадке под влиянием информации о вместимости.
- **Kim, Lee, Oh, 2009** (10.3141/2112-15) — Verified abstract. Сеул, binary logit модель выбора автобуса на основе occupancy information; ближайший bus-specific прецедент.
- **Pan et al., 2022** (10.1177/03611981221076843) — Verified abstract. Incentive-based willingness to wait для автобусов (goal-framing theory), не RTCI напрямую, но методологически смежно.
- **Drabicki et al., 2025** (10.1016/j.tbs.2024.100895) — Metadata only. COVID + RTCI willingness to wait.

## Crowding valuation

- **Bansal, Hörcher, Graham, 2022** (10.1111/rssa.12804) — Verified abstract. Dynamic choice model на revealed-preference smart-card данных; valuation of crowding увеличивается на 47% при экстремальной перегрузке. Это работа ментора проекта (Prateek Bansal).

## RTCI field experiments

- **Zhang, Jenelius, Kottenhoff, 2017** (10.1007/s12469-016-0150-y) — Verified abstract. **Единственный найденный реальный полевой пилот RTCI** — Стокгольмское метро, 6-дневный тестовый период, per-car информация. Ближайший компаратор к амбициям Sanas по полевому эксперименту, но: метро (не автобус), Швеция (не Центральная Азия).
- **Drabicki, Kucharski, Cats, 2022** (10.1007/s11116-022-10270-3) — Verified abstract. Симуляция на кейс-стади Варшавы (не реальный полевой эксперимент, а simulation + stated-preference модель).

## Revealed-preference evidence

- **Chen et al., 2023** — смарт-карты метро, boarding delay choice.
- **Kim, Lee, Oh, 2009** — Сеул, автобусы, но методология survey-based (не чисто revealed).
- **Bansal, Hörcher, Graham, 2022** — крупномасштабные smart-card + AVL данные азиатского метро.

Прямых revealed-preference данных о посадке на автобус под влиянием app-displayed RTCI **не найдено** — это подтверждает (в пределах неполного покрытия обзора) обозначенный gap проекта.

## Choice models

- Bansal, Hörcher, Graham, 2022 — dynamic choice model (DCM) с переключением между compensatory/habit rules.
- Kim, Lee, Oh, 2009 — binary logit.
- Pan et al., 2022 — binary logit + mixed multinomial logit.
- Drabicki, Kucharski, Cats, 2022 — boarding choice model внутри симуляции.

## Prediction and sensing

- Jenelius, 2020 (10.1016/j.trc.2020.102647) — Metadata only, personalized predictive crowding.
- Noursalehi, Koutsopoulos, Zhao, 2021 (10.1016/j.trc.2021.103139) — Metadata only, predictive decision support platform.
- Peftitsi, Jenelius, Cats, 2022 (10.1016/j.tra.2022.10.011) — Metadata only, моделирование эффекта RTCI на распределение пассажиров по вагонам поезда.

## Information accuracy and trust

Не найдено ни одной работы в этом наборе, напрямую посвящённой accuracy/trust RTCI. Требуется отдельный целевой поиск (не выполнен в этой сессии).

## Load distribution and operations

- Drabicki, Kucharski, Cats, 2022 — bus bunching mitigation через RTCI.
- Peftitsi, Jenelius, Cats, 2022 — car-level distribution в поездах.

## Geographic evidence

- Швеция (Стокгольм) — Zhang et al. 2017.
- Польша (Варшава) — Drabicki, Kucharski, Cats 2022 (симуляция).
- Южная Корея (Сеул) — Kim, Lee, Oh 2009.
- Остальные seed papers — географию нельзя определить без полного текста (metadata only).
- **Центральная Азия / Казахстан / постсоветские транзитные системы: не найдено ни одной работы.** Это согласуется с заявленным geographic gap проекта, но обзор недостаточно полный, чтобы утверждать отсутствие таких работ окончательно.

## Confirmed research gaps

В пределах найденного (12 seed papers + 1 hop-находка):
- Нет найденной работы с рандомизированным или квази-рандомизированным полевым экспериментом RTCI на автобусах.
- Нет найденной работы с app-level A/B тестом отображения загруженности.
- Нет найденной работы из Центральной Азии или постсоветских транзитных систем.
- Нет найденной работы, объединяющей revealed-preference данные о посадке на автобус с RTCI-дисплеем конкретно (Kim et al. 2009 близко, но это survey-based, не revealed от app).

**Важная оговорка:** это не подтверждение отсутствия таких работ в мировой литературе — это отражение того, что не найдено в рамках 12 seed papers и частичного citation hop (2 из 12, ~17% покрытия graph). Для окончательного вывода о gap нужен systematic search по Scopus/WoS/TRID.

## Studies threatening our novelty

Помечены как `potentially challenges novelty` (per задание, пункт 8):

1. **Zhang, Jenelius, Kottenhoff, 2017** — реальный полевой пилот RTCI в метро. Не автобус, не Центральная Азия, но методологически ближайший конкурент по формату "реальный пилот + реальные пассажиры".
2. **Chen et al., 2023** — revealed-preference boarding-delay поведение по смарт-карт данным метро. Показывает, что подобные revealed-preference методы применялись, хотя не для автобусов и не для RTCI-дисплея конкретно.
3. **Leprévost et al., 2026** (найдена через citation hop, Discovery Inbox) — RTCI и platform placement choices; требует full-text проверки на предмет пересечения с нашей методологией.

Ни одна из найденных работ не делает практически то же самое, что заявленный gap Sanas (app-displayed RTCI → actual bus boarding decisions, Центральная Азия) — но при 17%-ном покрытии citation graph это предварительный, не окончательный вывод.
