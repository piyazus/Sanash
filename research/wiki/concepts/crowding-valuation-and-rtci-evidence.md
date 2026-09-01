---
type: concept
tracks: [rtci]
updated: 2026-08-26
sources: openalex-chaining
verified: metadata-only
---

# Литература по оценке crowding и эффектам RTCI

Собрано citation chaining по трём seed-работам чартера через OpenAlex,
2026-08-26. Метод и параметры запуска: [[litsearch-run-2026-08-26]].

**Статус всей страницы: metadata-only.** Проверены заголовок, авторы,
площадка, год и DOI через OpenAlex, DOI работы Kapatsila дополнительно через
Crossref. Ни одна из статей не прочитана. Что именно в них измерено, здесь не
утверждается.

## Работа консультанта

| Год | Работа | Площадка | DOI |
|---|---|---|---|
| 2025 | Kapatsila, Bahamonde-Birke, van Lierop, Grisé. The effect of crowding level information provision on the revealed route choice of transit riders | Transportation | 10.1007/s11116-025-10585-x |

Это, судя по теме, та работа, на которую он ссылался в
[[2026-08-26-kapatsila-consultation]], утверждение 2: после контроля travel
time crowding information влияет на выбор. Совпадение темы не является
подтверждением, что цифры из разговора взяты именно отсюда. Abstract
отсутствует и в OpenAlex, и в Crossref, effect size по-прежнему неизвестен.
Нужен полный текст.

Обратить внимание на разницу постановок: у него **route choice**, у Sanas
**boarding the first arriving bus versus waiting**. Это разные outcome, и
чартер требует именно второй.

## Прямые прецеденты RTCI

| Год | Работа | Площадка | DOI |
|---|---|---|---|
| 2016 | Zhang, Jenelius, Kottenhoff. Impact of real-time crowding information: a Stockholm metro pilot study | Public Transport | 10.1007/s12469-016-0150-y |
| 2009 | Kim, Lee, Oh. Passenger choice models for analysis of impacts of real-time bus information on crowdedness | Transportation Research Record | 10.3141/2112-15 |
| 2021 | Noursalehi, Koutsopoulos, Zhao. Predictive decision support platform and its application in crowding prediction and passenger information generation | Transportation Research Part C | 10.1016/j.trc.2021.103139 |
| 2024 | Drabicki, Cats, Kucharski. Has the COVID-19 pandemic affected travellers' willingness to wait with real-time crowding information? | Travel Behaviour and Society | 10.1016/j.tbs.2024.100895 |
| 2018 | Kattan, Bai. LRT passengers' responses to advanced passenger information system (APIS) in case of information inconsistency and train crowding | Canadian Journal of Civil Engineering | 10.1139/cjce-2017-0559 |
| 2016 | Nuzzolo, Crisalli, Comi. A mesoscopic transit assignment model including real-time predictive information on crowding | Journal of Intelligent Transportation Systems | 10.1080/15472450.2016.1164047 |

Первые две закрывают дыры в чартере: Stockholm pilot там был указан ссылкой
на TRID, а работа Kim, Lee и Oh ссылкой на worldtransitresearch. Теперь у
обеих есть DOI.

Drabicki, Cats и Kucharski (2024) в чартере отсутствует, хотя две другие
работы Drabicki там есть. Работа про то, как пандемия повлияла на WTW при
наличии RTCI, то есть напрямую по теме и свежая. Кандидат на добавление в
чартер.

Kattan и Bai важны для H5: реакция пассажиров на **несогласованную**
информацию. У Sanas это риск differential measurement error.

## Оценка crowding, база

| Год | Работа | Площадка | DOI |
|---|---|---|---|
| 2013 | Tirachini, Hensher, Rose. Crowding in public transport systems: effects on users, operation and implications for the estimation of demand | TR Part A | 10.1016/j.tra.2013.06.005 |
| 2010 | Wardman, Whelan. Twenty years of rail crowding valuation studies | Transport Reviews | 10.1080/01441647.2010.519127 |
| 2016 | Hörcher, Graham, Anderson. Crowding cost estimation with large scale smart card and vehicle location data | TR Part B | 10.1016/j.trb.2016.10.015 |
| 2018 | Yap, Cats, van Arem. Crowding valuation in urban tram and bus transportation based on smart card data | Transportmetrica A | 10.1080/23249935.2018.1537319 |
| 2014 | Kroes, Kouwenhoven, Debrincat. Value of crowding on public transport in Ile-de-France | Transportation Research Record | 10.3141/2417-05 |
| 2017 | Tirachini, Hurtubia, Dekker. Estimation of crowding discomfort in public transport: results from Santiago de Chile | TR Part A | 10.1016/j.tra.2017.06.008 |
| 2016 | Tirachini, Sun, Erath. Valuation of sitting and standing in metro trains using revealed preferences | Transport Policy | 10.1016/j.tranpol.2015.12.004 |
| 2015 | Batarce, Munoz, Ortuzar. Use of mixed stated and revealed preference data for crowding valuation on public transport in Santiago | Transportation Research Record | 10.3141/2535-08 |
| 2019 | Bansal, Hurtubia, Tirachini. Flexible estimates of heterogeneity in crowding valuation in the New York City subway | Journal of Choice Modelling | 10.1016/j.jocm.2019.04.004 |

Этот блок и есть тот "готовый фреймворк", о котором говорил Kapatsila в
[[crowding-elasticity]]: два десятилетия оценок crowding valuation, включая
обзор Wardman и Whelan за двадцать лет британских исследований. Названий он не
дал, chaining их выдал.

## Операции и системный уровень

| Год | Работа | Площадка | DOI |
|---|---|---|---|
| 2016 | Cats, West, Eliasson. A dynamic stochastic model for evaluating congestion and crowding effects in transit systems | TR Part B | 10.1016/j.trb.2016.04.001 |
| 2024 | Rezazada, Nassir, Tanin. Bus bunching: a comprehensive review from demand, supply and decision-making perspectives | Transport Reviews | 10.1080/01441647.2024.2313969 |
| 2023 | Gallo, Sacco, Corman. Network-wide public transport occupancy prediction framework with multiple line interactions | IEEE OJ-ITS | 10.1109/ojits.2023.3331447 |

Relevant к четвёртой части publication package в `TRC_PAPER_BLUEPRINT.md`:
transport implications, load variance, bunching.

## Что это меняет

1. Kapatsila 2025 нужно достать полностью: это ближайший методологический
   сосед и потенциальная угроза новизне.
2. Drabicki, Cats, Kucharski 2024 добавить в чартер §3.
3. Заменить в чартере ссылки на TRID и worldtransitresearch на DOI.
4. Ни один пункт `GROUND_TRUTH.md` этой страницей не меняется.
