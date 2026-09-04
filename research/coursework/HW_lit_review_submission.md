# Literature Review Brainstorming

Research Writing, Assignment 3 (Literature review brainstorming)
2026-08-31

**Research question.** How does real-time crowding information (RTCI) affect
boarding decisions among Almaty bus commuters?

Stated operationally, for design and analysis: among app users waiting at
selected stops on a high-frequency Almaty bus route, what is the causal effect
of displaying real-time crowding information on the probability of boarding the
first arriving bus rather than waiting for the next service? The literature
below is reviewed against that outcome, which is why studies measuring which
car or which route a rider chooses are treated as adjacent evidence rather than
direct evidence.

Sixteen studies are reviewed. They are drawn from a larger reference base
assembled for this topic; the sixteen included here are those whose findings
could be established from the full text or from the publisher's abstract.
Three further papers were located but could not be opened or confirmed: Wang
et al. (2021), the only non-Western RTCI study found; Fedujwar & Agarwal
(2024), a systematic review of forty crowding-valuation studies; and Leprévost
et al. (2026), the most recent on-topic paper. They are noted where they bear
on a claim, but no finding is attributed to them.

---

## Part 1 — Literature review matrix

| # | Study | Most important finding | Theme | Gap it leaves open |
|---|---|---|---|---|
| [1] | Zhang, Jenelius & Kottenhoff (2017), Stockholm metro | Displayed per-car crowding shifted boarding: −4.3 pp on the most crowded car, +4.1 pp on the next one, with about 25% of passengers noticing the information | Deployed and observed | Which car to board, not whether to board; six days, one platform, crowding data produced by hand, pre-pandemic |
| [2] | Drabicki, Cats, Kucharski, Fonzone & Szarata (2023), Kraków | 45–75% say they would skip a severely overcrowded departure and 12–30% a moderately crowded one, accepting waits of 2–12 minutes | Stated preference | Hypothetical choices; the authors call for revealed-preference validation |
| [3] | Drabicki, Cats & Kucharski (2025), Kraków repeat | After COVID, willingness to wait is driven by expecting a seat in the next departure rather than by avoiding the crowded first one | Stated preference | Still stated; no observed post-2020 behaviour |
| [4] | Kapatsila, van Lierop, Bahamonde-Birke & Grisé (2025), Vancouver | Non-monetary incentives shift riders toward a less crowded route; riders aged 20–34 respond most, full-time workers least | Stated preference | Incentives rather than information alone; stated choice during COVID |
| [5] | Lee, Kwak & Han (2024), Seoul | 971 rail riders separate into four types with sharply different willingness to wait, which rises as headway shortens | Stated preference | Rail rather than bus; stated; full text not obtained |
| [6] | Drabicki, Kucharski & Cats (2023), Warsaw corridor | Simulated RTCI produces a 30–70% probability of intentionally skipping an overcrowded bus and cuts denial of boarding by about 40% | Simulation | The skip probability is an input taken from [2], not an observation |
| [7] | Peftitsi, Jenelius & Cats (2022), Stockholm | Predictive per-car RTCI evens out passenger distribution inside trains, with gains rising with demand up to a limit | Simulation | Simulation, and car choice again rather than boarding |
| [8] | Kapatsila, Bahamonde-Birke, van Lierop & Grisé (2025), Vancouver | In navigation-app logs crowding lowers a route's selection probability, with a time multiplier up to 2.23 for crowded rapid transit | Revealed, no treatment | Route planning inside an app, not the decision at a stop; no information-off control group |
| [9] | Yap & Cats (2021), Washington DC | Wait caused by denied boarding is valued 68% more negatively than ordinary initial wait | Revealed, no treatment | Not boarding is involuntary here, and no information was provided |
| [10] | Bansal, Hörcher & Graham (2022), Hong Kong | The average rider makes a compensatory choice on only 25.5% of occasions, while time valuation rises 47% under extreme crowding | Revealed, no treatment | Establishes inertia, but with no information intervention |
| [11] | Pi, Qian, Steinfeld & Huang (2018), Pittsburgh | Rider fullness ratings map only loosely onto measured occupancy (means 0.32, 0.52, 0.76) and shift with time of day, neighbourhood income and bus size | Perception and display | Correlational by the authors' own statement; no treatment, no boarding outcome |
| [12] | Zhang-Kennedy et al. (2023), Canada | The crowding level that deters riders fell from about 4 to about 2 passengers/m² after COVID, and the authors recommend three display levels rather than five | Perception and display | Preference and usability, not what riders then do |
| [13] | Kovačević et al. (2026), Novi Sad | Riders standing at the back of a bus report higher occupancy than those at the front; the paper states that crowding research in Southeastern Europe is limited | Perception and display | Perception only; one day, two lines, no information provided |
| [14] | Brakewood, Barbeau & Watkins (2014), Tampa | In a randomized control-group experiment on an arrival app, usual wait time fell 1.79 min in the treatment group against 0.21 min in the control (p = 0.009) | Real-time information generally | Arrival information, not crowding; the wait-time outcome is self-reported |
| [15] | Brakewood, Macfarlane & Watkins (2015), New York City | A city-wide rollout of arrival information raised ridership by a median 1.7%, concentrated on the largest routes | Real-time information generally | Aggregate ridership, not any individual boarding decision |
| [16] | Hlophe, Afolayan & Daramola (2024), Cape Town | Real-time information for Global South transit has received limited attention, and the Stockholm pilot [1] is the only behavioural precedent the authors find | Context and geography | Qualitative and stakeholder-facing; no crowding measure, no rider behaviour |

---

## Part 2 — Patterns

**Pattern 1. Effect size tracks the method used, not the city studied.**
Stated-preference work reports that 45–75% of riders would skip a severely
overcrowded departure [2], and the simulation calibrated on those statements
carries the same 30–70% skip probability through to a 40% reduction in denial
of boarding [6]. The one deployment that observed real passengers moved the
share boarding the most crowded car by 4.3 percentage points, and only about a
quarter of passengers noticed the information at all [1]. The distance between
these two families of estimate is an order of magnitude, and it does not close
when the setting changes: Kraków, Warsaw and Stockholm are comparable European
networks. Method, not context, is what separates the numbers.

**Pattern 2. The outcome that defines the question has almost never been the
outcome measured.** The literature measures which car of a train to board [1],
[7], which route to select in an app [8], and how aggregate ridership responds
[15]. The one study that measures an actual failure to board treats it as
involuntary denial of boarding, with no information provided to anyone [9].
Across all sixteen studies, the voluntary choice between boarding a crowded
vehicle now and waiting for the next one appears only where it is stated
hypothetically [2]–[5] or assumed inside a model [6], [7].

**Pattern 3. The evidence is concentrated in a small set of high-income
networks.** Western Europe and North America supply [1], [2], [3], [4], [6],
[7], [8], [9], [11], [12], [14] and [15]; East Asia supplies [5] and [10]. Two
papers state the imbalance from inside their own regions: crowding research in
Southeastern Europe is described as limited relative to Western Europe, North
America and Asia [13], and real-time information for Global South transit is
described as having received limited attention [16]. Country-filtered database
queries run for this review returned no study of crowding, crowding valuation
or crowding information for Kazakhstan, Uzbekistan or Kyrgyzstan. Since the one
non-Western RTCI study located could not be opened, the defensible claim is
that relevant work outside these regions is scarce and difficult to retrieve,
not that none exists.

**Pattern 4. "Crowding" is not one variable.** It appears as passengers per
square metre [12], as a ratio of passengers to vehicle capacity [11], as an
ordinal scale of seat availability derived from focus groups [2], as denial of
boarding [9], and as a subjective rider rating [11], [13]. These definitions do
not convert into one another. The clearest evidence is [11]: the same rider
rating corresponds to occupancy ratios spread widely around means of 0.32, 0.52
and 0.76, and the mapping shifts with time of day, neighbourhood income and
vehicle size. Comparing effect sizes across these studies therefore compares
different constructs, and a threshold taken from one system does not transfer
to another unedited.

**Pattern 5. The pre-2020 and post-2020 literatures do not describe the same
riders.** The crowding level that deters riders fell from roughly 4 to roughly
2 passengers/m² [12], and the mechanism behind willingness to wait moved from
avoiding a crowded first vehicle to expecting a seat in the second one [3]. The
only field deployment [1] ran in May 2015, and the simulation most often cited
in this field [6] is calibrated on 2019 survey data. The parameters quoted most
frequently were therefore estimated on a rider population that no longer
behaves the same way.

**Pattern 6. Where real-time information has actually been deployed, measured
effects are small.** A randomized experiment on arrival information moved
self-reported usual wait time by 1.79 minutes and changed neither trip
frequency nor transfers [14]; a city-scale rollout produced a median 1.7%
ridership increase [15]; the crowding pilot moved car choice by about four
percentage points [1]. Revealed-choice work supplies a candidate mechanism: the
average rider applies a compensatory decision rule on only a quarter of choice
occasions [10]. Together these suggest that a study of this question should be
designed for a small average effect rather than for the magnitudes reported by
stated preference and simulation.

---

## Part 3 — One disagreement

**The disagreement.** Drabicki, Kucharski & Cats [6] report that real-time
crowding information induces a 30–70% probability of intentionally skipping an
overcrowded bus, reducing denial of boarding by about 40%. Zhang, Jenelius &
Kottenhoff [1] report that deployed crowding information moved the share of
passengers boarding the most crowded car by 4.3 percentage points, that only
about 25% of passengers noticed it, and that the effect across all trains was
not statistically significant. Both concern European urban transit and both
concern how riders respond to displayed crowding, yet their magnitudes differ
by roughly an order of magnitude.

**Where the two studies differ.**

*Method.* [6] is an agent-based simulation; [1] is a six-day field deployment
with video observation of real passengers. This is the largest single
difference, and it is not incidental: the skip probability in [6] is a
calibrated input taken from the same research group's stated-preference survey
[2], so [6] cannot independently confirm or contradict [1] on this point. Its
authors say as much, writing that real-world application is what would allow
the details of RTCI use to be calibrated and validated.

*Measurement.* [6] measures whether a rider skips a vehicle. [1] measures which
car of an arriving train a rider walks toward. A rider who dislikes crowding
can act on the outcome in [1] at almost no cost in time, and on the outcome in
[6] only by waiting a full headway. These are different decisions at different
prices, so part of the smaller measured number is a property of what was
measured rather than of how riders behave.

*Population and country.* [6] models a Warsaw bus corridor using parameters
from Kraków respondents, 75% of whom were under 40, a sample that oversampled
young riders relative to the city's own travel survey [2]. [1] observed
whoever happened to be on a Stockholm metro platform during the afternoon peak.
Because willingness to wait rises sharply with age in [2], and because riders
separate into distinct types with different willingness to wait in [5], these
two behavioural populations are not interchangeable.

*Sample and exposure.* [6] can apply information to any share of a simulated
population and reports that benefits emerge from about 25% penetration. In [1],
about 25% of real passengers noticed the information at all, placing the
observed deployment at the very bottom of the range in which the simulation
expects any effect.

*Measurement quality.* [6] assumes automatically generated per-vehicle crowding
information. In [1] the crowding data was produced manually by staff at the
upstream station, which bounds how accurate and how timely the displayed
information could have been.

*Time period.* [1] ran in May 2015 and [6] is calibrated on 2019 data, both
before the shift in crowding tolerance documented in [12] and [3].

**Interpretation.** This is not yet a resolved empirical conflict, because no
study has measured the behaviour that [6] predicts. It is informative anyway:
the largest numbers in this field come from a model whose behavioural parameter
was supplied by a survey, and on the single occasion when real passengers were
observed responding to displayed crowding, the response was small and only
partly significant. A second, smaller disagreement in the same set points the
same way. [12] recommends collapsing crowding to three displayed levels,
because riders stop discriminating meaningfully above about two passengers per
square metre, while [2] elicits behaviour on a four-level scale. The field does
not agree on how many levels of crowding a rider can actually act on.

---

## Part 4 — The gap

Across this literature the direction of the effect is settled while its size in
the field is not, because the decision the question asks about has not been
observed. Stated-preference studies establish that riders say they would skip a
crowded departure, and how long they say they would wait [2]–[5]. Simulations
carry those statements forward into network-level benefits [6], [7].
Revealed-choice studies establish that crowding lowers the attractiveness of a
route or a vehicle, but in systems where no crowding information was provided,
and therefore with no treatment to which an effect could be attributed
[8]–[11]. The single deployment that did provide the information measured which
car of a train riders boarded, over six days, with data generated by hand,
before the pandemic [1]. Among the studies retrieved for this review, none
measures the voluntary choice to board the first arriving vehicle rather than
wait for the next one under a contrast between information provided and
information withheld, and the figures most often quoted for that choice are
simulated rather than observed. The gap is sharpened geographically:
country-filtered searches returned no study of crowding or crowding information
for Kazakhstan, while crowding information is now being displayed to riders in
Almaty through a live transit application.

In the terms used in this course, the gap is primarily **methodological**,
sharpened by a **geographic** absence.

---

## Part 5 — Themes

**Theme 1. Crowding information deployed and observed.** [1]. One study, and
the only one retrieved here in which real passengers were observed responding
to displayed crowding. This is why the literature has a gap rather than a
debate.

**Theme 2. What riders say they would do.** [2], [3], [4], [5].
Stated-preference studies of willingness to wait, including a post-pandemic
repeat of the same instrument [3] and an incentive variant [4].

**Theme 3. What models predict from those statements.** [6], [7]. Simulation
studies whose behavioural parameters are drawn from Theme 2, which is why they
cannot serve as independent evidence for it.

**Theme 4. Observed crowding behaviour without an information treatment.** [8],
[9], [10]. Revealed preference from application logs, fare-card and
vehicle-location data. Establishes that riders dislike crowding and that most
riders are inertial.

**Theme 5. How crowding is perceived and displayed.** [11], [12], [13].
Perception against measured occupancy, display format, number of levels, and
position inside the vehicle. This theme determines whether a treatment is
legible to the rider at all.

**Theme 6. Real-time information in general, as a design template.** [14],
[15]. A randomized experiment and a natural experiment showing how a field
study of transit information can be identified, and how small its effects are.

**Theme 7. Context and geography.** [13], [16]. Two independent statements,
from Southeastern Europe and from South Africa, that the evidence base is
concentrated elsewhere.

---

## Part 6 — Four synthesis findings

1. Studies consistently find that crowding reduces the attractiveness of a
   transit option, but the size of the reported effect depends far more on the
   method than on the setting: stated-preference and simulation work reports
   that 30–75% of riders would skip a crowded departure [2], [6], while the one
   deployed field study moved observed boarding by about four percentage
   points, with only a quarter of passengers noticing the information [1].

2. Revealed-preference evidence establishes that riders act on crowding without
   being told about it, discounting crowded routes and valuing denied-boarding
   wait 68% more negatively than ordinary wait [8], [9], although the same body
   of work shows that most riders are inertial, applying a compensatory decision
   rule on only about a quarter of occasions [10], which suggests that any
   information effect should be expected to be small.

3. Willingness to wait rises with crowding severity in every study that
   measures it, with the crowding coefficient roughly tripling between moderate
   and severe crowding [2], although the response is not uniform across riders:
   it rises sharply with age [2], separates riders into distinct latent types
   [5], and after the pandemic appears to be driven by the expectation of a seat
   in the next departure rather than by avoidance of the crowded first one [3].

4. Perceived crowding and measured occupancy are correlated but not
   interchangeable, since the same rider rating spans a wide range of occupancy
   ratios and shifts systematically with time of day, income context and vehicle
   size [11], and riders discriminate little between levels above roughly two
   passengers per square metre [12], [13], which means that studies defining
   crowding as a percentage of capacity and studies defining it as a rider
   rating are not measuring the same variable.

---

## References

[1] Y. Zhang, E. Jenelius, and K. Kottenhoff, "Impact of real-time crowding information: a Stockholm metro pilot study," *Public Transport*, vol. 9, no. 3, pp. 483–499, 2017, doi: 10.1007/s12469-016-0150-y.

[2] A. Drabicki, O. Cats, R. Kucharski, A. Fonzone, and A. Szarata, "Should I stay or should I board? Willingness to wait with real-time crowding information in urban public transport," *Research in Transportation Business & Management*, vol. 47, art. 100963, 2023, doi: 10.1016/j.rtbm.2023.100963.

[3] A. Drabicki, O. Cats, and R. Kucharski, "Has the COVID-19 pandemic affected travellers' willingness to wait with real-time crowding information?," *Travel Behaviour and Society*, vol. 38, art. 100895, 2025, doi: 10.1016/j.tbs.2024.100895.

[4] B. Kapatsila, D. van Lierop, F. J. Bahamonde-Birke, and E. Grisé, "The effect of incentives on the actions transit riders make in response to crowding," *Travel Behaviour and Society*, vol. 40, art. 101018, 2025, doi: 10.1016/j.tbs.2025.101018.

[5] H.-S. Lee, H.-C. Kwak, and E.-S. Han, "Modeling urban railway passengers' willingness to wait based on latent class analysis," *Transportation Research Record*, vol. 2678, no. 9, pp. 230–240, 2024, doi: 10.1177/03611981231225641.

[6] A. Drabicki, R. Kucharski, and O. Cats, "Mitigating bus bunching with real-time crowding information," *Transportation*, vol. 50, no. 3, pp. 1003–1030, 2023, doi: 10.1007/s11116-022-10270-3.

[7] S. Peftitsi, E. Jenelius, and O. Cats, "Modeling the effect of real-time crowding information (RTCI) on passenger distribution in trains," *Transportation Research Part A: Policy and Practice*, vol. 166, pp. 354–368, 2022, doi: 10.1016/j.tra.2022.10.011.

[8] B. Kapatsila, F. J. Bahamonde-Birke, D. van Lierop, and E. Grisé, "The effect of crowding level information provision on the revealed route choice of transit riders," *Transportation*, advance online publication, 2025, doi: 10.1007/s11116-025-10585-x.

[9] M. Yap and O. Cats, "Taking the path less travelled: Valuation of denied boarding in crowded public transport systems," *Transportation Research Part A: Policy and Practice*, vol. 147, pp. 1–13, 2021, doi: 10.1016/j.tra.2021.02.007.

[10] P. Bansal, D. Hörcher, and D. J. Graham, "A dynamic choice model to estimate the user cost of crowding with large-scale transit data," *Journal of the Royal Statistical Society Series A*, 2022, doi: 10.1111/rssa.12804.

[11] Z. Pi, X. Qian, A. Steinfeld, and D. Huang, "Understanding human perception of bus fullness," *Transportation Research Record*, vol. 2672, no. 8, pp. 475–484, 2018, doi: 10.1177/0361198118781398.

[12] L. Zhang-Kennedy et al., "Passenger perceptions, information preferences, and usability of crowding visualizations on public displays in transit stations and vehicles," in *Proc. CHI '23*, 2023, pp. 1–15, doi: 10.1145/3544548.3581241.

[13] T. Kovačević, P. Pitka, J. Ivetić, J. Dedeić, M. Miličić, and M. Majstorović, "Passenger perception of vehicle occupancy in public transport and factors that shape crowding estimations," *Scientific Reports*, vol. 16, no. 1, art. 13437, 2026, doi: 10.1038/s41598-026-43541-5.

[14] C. Brakewood, S. J. Barbeau, and K. Watkins, "An experiment evaluating the impacts of real-time transit information on bus riders in Tampa, Florida," *Transportation Research Part A: Policy and Practice*, vol. 69, pp. 409–422, 2014, doi: 10.1016/j.tra.2014.09.003.

[15] C. Brakewood, G. S. Macfarlane, and K. Watkins, "The impact of real-time information on bus ridership in New York City," *Transportation Research Part C: Emerging Technologies*, vol. 53, pp. 59–75, 2015, doi: 10.1016/j.trc.2015.01.021.

[16] A. Hlophe, A. Afolayan, and O. Daramola, "Integrated real-time information system for public commuting: Perspectives of stakeholders in South Africa," *International Journal of Transport Development and Integration*, vol. 8, no. 1, pp. 31–48, 2024, doi: 10.18280/ijtdi.080104.


---

<!-- was: research/coursework/LIT_REVIEW_BRAINSTORMING.md -->

**Приложение A. Черновик.** Рабочая версия этого же задания, написанная до сдачи. Сохранена как след рассуждения, не как сданный текст.

## Literature review brainstorming

Course: Terra research writing (Mahmud Khamraev). Assignment posted 2026-08-28,
100 points. Drafted 2026-08-31.

Research question, per `research/RTCI_RESEARCH_CHARTER.md` §1, which is the
authority for the RTCI research design: how does real-time crowding information
(RTCI) affect boarding decisions among Almaty bus commuters? Operational
formulation used for design and analysis: among app users waiting at selected
stops on a high-frequency Almaty bus route, what is the causal effect of
displaying real-time crowding information on the probability of boarding the
first arriving bus rather than waiting for the next service?

The question phrasing in the vault `literature.md` is an older working
paraphrase and is not the authority.

Source base: `01_Projects/Sanash/paper/literature.md` in the Obsidian vault
(~50 sources, nine themes, per-source verification status). The 16 studies
tabulated here are the subset whose findings were actually established from a
full text or a publisher abstract. Entries marked metadata-only in that file
are deliberately excluded from the matrix, because their findings are not
established.

Two parts of the assignment text were truncated when it was copied
(Part 3 "answer the following questions" lists no questions; Part 4 and Part 6
break off mid-sentence). Part 3 is answered against the six factors the
assignment does name: population, country/context, method, sample,
measurement, time period. Part 4 is written as a gap paragraph in the format
of HW1. Part 6 is written as four synthesis statements.

Verification key: **F** full text read, **A** publisher abstract read,
**A\*** abstract reconstructed through a search index, not read at the
publisher.

---

### Part 1 — Literature review matrix

| # | Study | Most important finding | Theme | Gap it leaves open | Ver. |
|---|---|---|---|---|---|
| [1] | Zhang, Jenelius & Kottenhoff (2017), Stockholm metro | Per-car crowding shown on a platform display moved boarding: the share boarding the most crowded car fell 4.3 pp and the second car rose 4.1 pp on trains crowded on arrival; about 25% of passengers noticed and found it useful | 1. Deployed and observed | Measures which car to board, never whether to board; six days, one platform, crowding data generated manually by staff, pre-pandemic; overall effect across all trains not significant | F |
| [2] | Drabicki, Cats, Kucharski, Fonzone & Szarata (2023), Kraków | Stated willingness to wait is 12–30% at moderate crowding and 45–75% at severe overcrowding; acceptable waits 2–4 min (moderate) to 6–12 min (severe); the crowding coefficient is 3.5 times larger for severe than for moderate crowding | 2. Stated preference | Hypothetical choices; the authors themselves warn that stated crowding valuations are prone to overestimation and call for revealed-preference validation | F |
| [3] | Drabicki, Cats & Kucharski (2025), Kraków repeat | After COVID, willingness to wait is driven by the expectation of a seat in the second departure, where before it was driven by avoiding an overcrowded first one | 2. Stated preference | Still stated; establishes that pre-2020 parameters should not be reused, without supplying observed post-2020 ones | A\* |
| [4] | Kapatsila, van Lierop, Bahamonde-Birke & Grisé (2025), Vancouver | Non-monetary incentives (raffle, game points) shift riders toward a less crowded route; ages 20–34 respond most, full-time workers least | 2. Stated preference | Incentives, not information alone; stated choice collected during COVID | F |
| [5] | Lee, Kwak & Han (2024), Seoul | Latent class analysis of 971 rail riders finds four rider types with sharply different willingness to wait, and willingness to wait rises as scheduled headway shrinks | 2. Stated preference | Rail, not bus; the comparison with [2] reportedly replicates only in part, and the full text was not obtained | A |
| [6] | Drabicki, Kucharski & Cats (2023), Warsaw corridor | Agent-based simulation calibrated on the authors' own survey: RTCI induces a 30–70% probability of intentionally skipping an overcrowded bus, cutting denial of boarding and severe overcrowding by about 40%, with benefits from roughly 25% information penetration | 3. Simulation | The skip probability is an input taken from [2], not an observation; the authors write that real-world application is what would let the behaviour be calibrated and validated | F |
| [7] | Peftitsi, Jenelius & Cats (2022), Stockholm | Simulated car-specific predictive RTCI shifts boarding and evens out the distribution inside trains, with gains rising with demand up to a point beyond which switching cars stops helping | 3. Simulation | Simulation, and again car choice rather than the boarding decision | A |
| [8] | Kapatsila, Bahamonde-Birke, van Lierop & Grisé (2025), Vancouver | Revealed route choice from navigation-app logs: crowding lowers the probability a route is chosen, with a time multiplier up to 2.23 for crowded rapid transit, and crowded regular-bus trips perceived as almost six minutes longer | 4. Revealed, no treatment | Route planning inside an app, not the decision at a stop; observational, with no information-off control group | A |
| [9] | Yap & Cats (2021), Washington DC | Additional wait caused by denied boarding is valued 68% more negatively than initial wait; one minute of initial and of denied-boarding wait are perceived as 1.62 and 2.72 minutes on board an uncrowded vehicle | 4. Revealed, no treatment | Not boarding here is involuntary, and no information was provided to anyone | F |
| [10] | Bansal, Hörcher & Graham (2022), Hong Kong | The average rider follows the compensatory decision rule on only 25.5% of route-choice occasions, while travel-time valuation rises 47% under extreme crowding | 4. Revealed, no treatment | Establishes rider inertia as a prior, but with no information intervention to respond to | A\* |
| [11] | Pi, Qian, Steinfeld & Huang (2018), Pittsburgh | 42,238 crowdsourced fullness ratings matched to automatic passenger counts: mean occupancy ratios of 0.315, 0.517 and 0.758 for "many seats", "few seats" and "full", with large and growing variance; peak-hour riders and riders in higher-income tracts tolerate more crowding for the same rating; small buses are called "full" at about 90% of capacity against 60–70% for large buses | 5. Perception and display | Correlational by the authors' own statement; no information treatment and no boarding outcome | F |
| [12] | Zhang-Kennedy et al. (2023), Canada | Survey (n=303) plus a three-station field deployment (n=44): the crowding level that deters riders fell from about 4 to about 2 passengers/m² after COVID; the simple fullness scale was easiest to understand, but 64% of field participants preferred a seat-map view for actually avoiding crowds; the authors recommend collapsing crowding to three levels, not five | 5. Perception and display | Measures preference and usability, not what riders then do | F |
| [13] | Kovačević et al. (2026), Novi Sad | On-board survey of 1,318 passengers: riders standing at the back of the vehicle perceive higher occupancy than those at the front; the paper states that research on transit crowding in Southeastern Europe is limited relative to Western Europe, North America and Asia | 5. Perception and display | Perception only, one day, two lines, no information provided | A |
| [14] | Brakewood, Barbeau & Watkins (2014), Tampa | Randomized before-after control-group experiment on a real-time arrival app: usual wait time changed by −1.79 min in the treatment group against −0.21 min in the control (t = 2.66, p = 0.009), with no significant effect on trips or transfers | 6. Real-time information generally | Arrival information, not crowding; the wait-time outcome is self-reported, and contamination forced the exclusion of 24 controls and 27 treated riders | F |
| [15] | Brakewood, Macfarlane & Watkins (2015), New York City | Borough-by-borough rollout of real-time arrival information raised ridership by about 118 rides per route per weekday, a median increase of 1.7%, concentrated in the largest quartile of routes | 6. Real-time information generally | Aggregate ridership, not any individual boarding decision; arrival information, not crowding | F |
| [16] | Hlophe, Afolayan & Daramola (2024), Cape Town | A stakeholder study of real-time information for Global South commuting states that the topic has received limited attention in the literature, and identifies the Stockholm pilot [1] as essentially the only prior behavioural evidence available to it | 7. Context and geography | Qualitative and stakeholder-facing; no crowding measure and no rider behaviour | A |

Present in the source base but deliberately not tabulated, because the finding
could not be established: Wang et al. (2021), the only non-Western RTCI study
located, whose case-study city could not be confirmed; Fedujwar & Agarwal
(2024), a systematic review of 40 crowding-valuation studies, paywalled;
Leprévost et al. (2026), the most recent on-topic paper, design unconfirmed.
The first two matter for Part 2 and are treated there as limits on my own
claims rather than as evidence.

---

### Part 2 — Patterns

**Pattern 1. Effect size tracks the method used, not the city studied.**
Stated-preference work reports that 45–75% of riders would skip a severely
overcrowded departure [2], and the simulation calibrated on those statements
carries the same 30–70% skip probability through to a 40% reduction in denial
of boarding [6]. The one deployment that observed real passengers moved the
share boarding the most crowded car by 4.3 percentage points, and only about a
quarter of passengers noticed the information at all [1]. The distance between
the two families of estimate is an order of magnitude, and it does not close
when the setting changes: Kraków, Warsaw and Stockholm are comparable European
networks. Method, not context, is what separates the numbers.

**Pattern 2. The outcome that defines the question has almost never been the
outcome measured.** The literature measures which car of a train to board [1],
[7], which route to select in an app [8], and how aggregate ridership responds
[15]. The one study that measures an actual failure to board treats it as
involuntary denial of boarding, with no information provided [9]. Across all
sixteen studies, the voluntary choice between boarding a crowded vehicle now
and waiting for the next one is measured only where it is stated
hypothetically [2]–[5] or assumed inside a model [6], [7].

**Pattern 3. The evidence is concentrated in a small set of high-income
networks.** Western Europe and North America supply [1], [2], [3], [4], [6],
[7], [8], [9], [11], [12], [14] and [15]; East Asia supplies [5] and [10].
Two papers in the set state the imbalance from inside their own regions:
Southeastern Europe is described as under-researched relative to Western
Europe, North America and Asia [13], and real-time information for Global
South transit is described as having received limited attention [16].
Filtered author-country queries run for this project returned no study of
crowding, crowding valuation or crowding information for Kazakhstan,
Uzbekistan or Kyrgyzstan. The one non-Western RTCI study located, Wang et al.
(2021), could not be opened, so the strongest defensible statement is that
relevant work outside these regions is scarce and hard to retrieve, not that
none exists.

**Pattern 4. "Crowding" is not one variable.** It appears as passengers per
square metre [12], as a ratio of passengers to vehicle capacity [11], as an
ordinal scale of seat availability derived from focus groups [2], as denial of
boarding [9], and as a subjective rider rating [11], [13]. These do not
convert into one another. The clearest evidence is [11]: the same rider rating
corresponds to occupancy ratios spread widely around means of 0.32, 0.52 and
0.76, and the mapping shifts with time of day, neighbourhood income and
vehicle size. Comparing effect sizes across these studies therefore compares
different constructs, and a threshold taken from one of them does not transfer
to another system unedited.

**Pattern 5. The pre-2020 and post-2020 literatures do not describe the same
riders.** The crowding level that deters riders fell from roughly 4 to roughly
2 passengers/m² [12], and the mechanism behind willingness to wait moved from
avoiding a crowded first vehicle to expecting a seat in the second one [3].
The only field deployment [1] ran in May 2015, and the simulation that
dominates citation of this field [6] is calibrated on 2019 survey data. The
consequence is that the parameters most often quoted in this literature were
estimated on a rider population that no longer behaves the same way.

**Pattern 6. Where real-time information has actually been deployed, measured
effects are small.** A randomized experiment on arrival information moved
self-reported usual wait time by 1.79 minutes and changed neither trip
frequency nor transfers [14]; a city-scale rollout produced a median 1.7%
ridership increase [15]; the crowding pilot moved car choice by about four
percentage points [1]. Revealed-choice work supplies a candidate mechanism:
the average rider applies a compensatory decision rule on only a quarter of
choice occasions [10]. Together these suggest that a field experiment on this
question should be designed and powered for a small average effect rather than
for the magnitudes reported by stated preference and simulation.

---

### Part 3 — One disagreement

**The disagreement.** Drabicki, Kucharski & Cats [6] report that real-time
crowding information induces a 30–70% probability of intentionally skipping an
overcrowded bus, reducing denial of boarding by about 40%. Zhang, Jenelius &
Kottenhoff [1] report that deployed crowding information moved the share of
passengers boarding the most crowded car by 4.3 percentage points, that only
about 25% of passengers noticed it, and that the effect across all trains was
not statistically significant. Both concern European urban transit and both
concern how riders respond to displayed crowding, and their magnitudes differ
by roughly an order of magnitude.

**Where the two studies differ.**

- *Method.* [6] is an agent-based simulation; [1] is a six-day field
  deployment with video observation. This is the largest single difference,
  and it is not incidental: the skip probability in [6] is a calibrated input
  taken from the same group's stated-preference survey [2], so [6] cannot
  independently confirm or contradict [1] on this point. Its authors say as
  much, writing that real-world application is what would allow the details of
  RTCI utilization to be calibrated and validated.
- *Measurement.* [6] measures whether a rider skips a vehicle. [1] measures
  which car of an arriving train a rider walks toward. A rider who dislikes
  crowding can act on [1]'s outcome at almost no time cost, and on [6]'s
  outcome only by waiting a full headway. These are different decisions at
  different prices, so the smaller measured number is partly a property of
  what was measured.
- *Population and country.* [6] models a Warsaw bus corridor with parameters
  from Kraków respondents who were 75% under 40 and oversampled young riders
  relative to the city's travel survey [2]. [1] observed whoever was on a
  Stockholm metro platform in the afternoon peak. Since willingness to wait
  rises sharply with age in [2], and riders separate into distinct types in
  [5], the two behavioural populations are not interchangeable.
- *Sample and exposure.* [6] can apply information to any share of a simulated
  population, and reports that benefits appear from about 25% penetration. In
  [1], roughly 25% of real passengers noticed the information at all, which
  places the observed deployment at the very bottom of the range where the
  simulation expects any effect.
- *Information quality.* [6] assumes generated per-vehicle crowding
  information. In [1] the crowding data was produced manually by staff at the
  upstream station rather than sensed, which bounds how accurate and timely
  the displayed information could have been.
- *Time period.* [1] ran in May 2015 and [6] is calibrated on 2019 data, both
  before the tolerance shift documented in [12] and [3].

**What I take from it.** The honest reading is that this is not yet a resolved
empirical conflict, because no study has measured the behaviour that [6]
predicts. The disagreement is informative anyway: the largest numbers in this
field come from a model whose behavioural parameter was supplied by a survey,
and the only occasion on which anyone watched real passengers respond to
displayed crowding, the response was small and partly not significant. A
second, smaller disagreement in the same set points the same way. [12]
recommends collapsing crowding to three displayed levels, because riders stop
discriminating meaningfully above about two passengers per square metre, while
[2] elicits behaviour on a four-level scale. The field does not agree on how
many levels of crowding a rider can act on.

---

### Part 4 — The gap

Across this literature the direction of the effect is settled and its size in
the field is not, because the decision the question asks about has not been
observed. Stated-preference studies establish that riders say they would skip a
crowded departure, and how long they would wait [2]–[5]. Simulations carry
those statements forward to network-level benefits [6], [7]. Revealed-choice
studies establish that crowding lowers the attractiveness of a route or a
vehicle, but in systems where no crowding information was provided, and
therefore with no treatment to attribute an effect to [8]–[11]. The single
deployment that did provide the information measured which car of a train
riders boarded, over six days, with manually generated data, before the
pandemic [1]. The result is that the voluntary choice to board the first
arriving vehicle rather than wait for the next one has never been measured
under an information-on versus information-off contrast, and the estimates most
often quoted for it are simulated rather than observed. Almaty sharpens the
same gap geographically: filtered author-country searches returned no study of
crowding or crowding information for Kazakhstan, while crowding information is
now being deployed to riders there through a live transit application.

Gap type, in the course's terms: primarily **methodological**, sharpened by a
**geographic** absence.

---

### Part 5 — Themes

**Theme 1. Crowding information deployed and observed.** [1]. One study. The
entire empirical base of the research question, and the reason this review has
a gap rather than a debate.

**Theme 2. What riders say they would do.** [2], [3], [4], [5]. Stated
preference and willingness to wait, including the post-pandemic repeat [3] and
the incentive variant [4].

**Theme 3. What models predict from those statements.** [6], [7]. Simulation
studies whose behavioural parameters come from Theme 2, which is why they
cannot serve as independent evidence for it.

**Theme 4. Observed crowding behaviour without an information treatment.**
[8], [9], [10]. Revealed preference from application logs, fare-card and
vehicle-location data. Establishes that riders dislike crowding, and that most
riders are inertial.

**Theme 5. How crowding is perceived and displayed.** [11], [12], [13].
Perception against measured occupancy, display format, level count, and where
in the vehicle the rider is standing. The theme that determines whether a
treatment is legible to the rider at all.

**Theme 6. Real-time information in general, as a design template.** [14],
[15]. The randomized experiment and the natural experiment that show how a
field study of transit information can be identified, and how small its
effects are.

**Theme 7. Context and geography.** [13], [16]. Two independent statements,
from Southeastern Europe and from South Africa, that the evidence base is
concentrated elsewhere.

---

### Part 6 — Four synthesis findings

1. Studies consistently find that crowding reduces the attractiveness of a
   transit option, but the size of the reported effect depends far more on the
   method than on the setting: stated-preference and simulation work reports
   that 30–75% of riders would skip a crowded departure [2], [6], while the one
   deployed field study moved observed boarding by about four percentage
   points, with a quarter of passengers noticing the information [1].

2. Revealed-preference evidence establishes that riders act on crowding without
   being told about it, discounting crowded routes and valuing denied-boarding
   wait 68% more negatively than ordinary wait [8], [9], although the same body
   of work shows that most riders are inertial, applying a compensatory decision
   rule on only about a quarter of occasions [10], which suggests that any
   information effect should be expected to be small.

3. Willingness to wait rises with crowding severity in every study that
   measures it, with the crowding coefficient roughly tripling between moderate
   and severe crowding [2], although it is not uniform across riders: it rises
   sharply with age [2], separates riders into distinct latent types [5], and
   after the pandemic appears to be driven by the expectation of a seat in the
   next departure rather than by avoidance of the crowded first one [3].

4. Perceived crowding and measured occupancy are correlated but not
   interchangeable, with the same rider rating spanning a wide range of
   occupancy ratios and shifting systematically with time of day, income
   context and vehicle size [11], and riders discriminating little between
   levels above roughly two passengers per square metre [12], [13], which means
   that studies defining crowding as a percentage of capacity and studies
   defining it as a rider rating are not measuring the same variable.

---

### Verification status

Read from the full text: [1], [2], [4], [6], [9], [11], [12], [14], [15]. The
figures for [14] and [15] were read from the authors' pre-publication project
report rather than from the typeset articles, and should be confirmed against
the journal versions before they enter a manuscript.

Read from the publisher abstract: [5], [7], [8], [13], [16]. The geographic
statement attributed to [13] was extracted by a tool rather than read at the
publisher, and needs to be re-read before it is quoted.

Reconstructed abstract, not read at the publisher: [3], [10]. Their findings
are reported here as summaries and should not be quoted.

Nothing in this document went through Scite, which was out of monthly quota
until 2026-09-01. Year ambiguities to resolve before the paper itself is
submitted: [1] is registered online-first in December 2016 against a 2017
issue, and [3] carries a 2024 DOI against a 2025 issue.

---

### References (IEEE)

[1] Y. Zhang, E. Jenelius, and K. Kottenhoff, "Impact of real-time crowding information: a Stockholm metro pilot study," *Public Transport*, vol. 9, no. 3, pp. 483–499, 2017, doi: 10.1007/s12469-016-0150-y.

[2] A. Drabicki, O. Cats, R. Kucharski, A. Fonzone, and A. Szarata, "Should I stay or should I board? Willingness to wait with real-time crowding information in urban public transport," *Research in Transportation Business & Management*, vol. 47, art. 100963, 2023, doi: 10.1016/j.rtbm.2023.100963.

[3] A. Drabicki, O. Cats, and R. Kucharski, "Has the COVID-19 pandemic affected travellers' willingness to wait with real-time crowding information?," *Travel Behaviour and Society*, vol. 38, art. 100895, 2025, doi: 10.1016/j.tbs.2024.100895.

[4] B. Kapatsila, D. van Lierop, F. J. Bahamonde-Birke, and E. Grisé, "The effect of incentives on the actions transit riders make in response to crowding," *Travel Behaviour and Society*, vol. 40, art. 101018, 2025, doi: 10.1016/j.tbs.2025.101018.

[5] H.-S. Lee, H.-C. Kwak, and E.-S. Han, "Modeling urban railway passengers' willingness to wait based on latent class analysis," *Transportation Research Record*, vol. 2678, no. 9, pp. 230–240, 2024, doi: 10.1177/03611981231225641.

[6] A. Drabicki, R. Kucharski, and O. Cats, "Mitigating bus bunching with real-time crowding information," *Transportation*, vol. 50, no. 3, pp. 1003–1030, 2023, doi: 10.1007/s11116-022-10270-3.

[7] S. Peftitsi, E. Jenelius, and O. Cats, "Modeling the effect of real-time crowding information (RTCI) on passenger distribution in trains," *Transportation Research Part A: Policy and Practice*, vol. 166, pp. 354–368, 2022, doi: 10.1016/j.tra.2022.10.011.

[8] B. Kapatsila, F. J. Bahamonde-Birke, D. van Lierop, and E. Grisé, "The effect of crowding level information provision on the revealed route choice of transit riders," *Transportation*, advance online publication, 2025, doi: 10.1007/s11116-025-10585-x.

[9] M. Yap and O. Cats, "Taking the path less travelled: Valuation of denied boarding in crowded public transport systems," *Transportation Research Part A: Policy and Practice*, vol. 147, pp. 1–13, 2021, doi: 10.1016/j.tra.2021.02.007.

[10] P. Bansal, D. Hörcher, and D. J. Graham, "A dynamic choice model to estimate the user cost of crowding with large-scale transit data," *Journal of the Royal Statistical Society Series A*, 2022, doi: 10.1111/rssa.12804.

[11] Z. Pi, X. Qian, A. Steinfeld, and D. Huang, "Understanding human perception of bus fullness," *Transportation Research Record*, vol. 2672, no. 8, pp. 475–484, 2018, doi: 10.1177/0361198118781398.

[12] L. Zhang-Kennedy et al., "Passenger perceptions, information preferences, and usability of crowding visualizations on public displays in transit stations and vehicles," in *Proc. CHI '23*, 2023, pp. 1–15, doi: 10.1145/3544548.3581241.

[13] T. Kovačević, P. Pitka, J. Ivetić, J. Dedeić, M. Miličić, and M. Majstorović, "Passenger perception of vehicle occupancy in public transport and factors that shape crowding estimations," *Scientific Reports*, vol. 16, no. 1, art. 13437, 2026, doi: 10.1038/s41598-026-43541-5.

[14] C. Brakewood, S. J. Barbeau, and K. Watkins, "An experiment evaluating the impacts of real-time transit information on bus riders in Tampa, Florida," *Transportation Research Part A: Policy and Practice*, vol. 69, pp. 409–422, 2014, doi: 10.1016/j.tra.2014.09.003.

[15] C. Brakewood, G. S. Macfarlane, and K. Watkins, "The impact of real-time information on bus ridership in New York City," *Transportation Research Part C: Emerging Technologies*, vol. 53, pp. 59–75, 2015, doi: 10.1016/j.trc.2015.01.021.

[16] A. Hlophe, A. Afolayan, and O. Daramola, "Integrated real-time information system for public commuting: Perspectives of stakeholders in South Africa," *International Journal of Transport Development and Integration*, vol. 8, no. 1, pp. 31–48, 2024, doi: 10.18280/ijtdi.080104.
