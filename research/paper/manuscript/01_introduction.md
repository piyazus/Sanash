# 1. Introduction

Draft v1, 2026-08-31. IEEE numbering. Reference list and per-source
verification status: `references.md` in this folder.

Structure follows the four-part formula (Background, Problem, Research Gap,
Purpose Statement) from the research-writing course notes.

---

Crowding is one of the principal sources of disutility in urban public
transport. Revealed-preference work merging automated fare collection with
vehicle location data in the Washington metro finds that the additional wait
imposed by denied boarding is valued 68% more negatively than initial wait,
and that one minute of initial and of denied-boarding wait are perceived as
1.62 and 2.72 minutes aboard an uncrowded vehicle [1]. Smartphone ticketing
and journey-planning applications now make it technically feasible to show a
rider how full an approaching vehicle is before the decision to board is
made.

Whether that display changes behaviour remains largely untested, because the
behavioural evidence base is dominated by stated preference and simulation.
An in-situ survey at eight inner-city stops in Kraków reports acceptable
waiting times of 2 to 4 minutes under moderate crowding and 6 to 12 minutes
under severe overcrowding [2]. An agent-based model of a Warsaw bus corridor,
calibrated on that survey, induces a 30 to 70% probability of intentionally
skipping an overcrowded bus and reduces denial of boarding by approximately
40% [3]; that skipping probability is a model input rather than a measured
outcome, and the authors call for real-world calibration. Against these
figures stands a single published field deployment: six afternoon-peak days
on one Stockholm metro platform, where per-car crowding information moved
4.3 percentage points of boarding away from the most crowded car at
p = 0.062, with roughly a quarter of passengers reporting that they noticed
the information at all [4].

Two gaps follow. Methodologically, no study has observed the decision that
real-time crowding information is intended to influence. The Stockholm pilot
measured which car riders entered rather than whether they boarded, and its
crowding estimates were generated manually by station staff rather than
sensed [4]; application-log evidence from Metro Vancouver measures route
selection during trip planning and contains no information-off control group
[5]. Geographically, the evidence is narrow. Filtered author-country queries
return no study of crowding, crowding valuation or crowding information for
Kazakhstan, Uzbekistan or Kyrgyzstan; a recent Serbian occupancy-perception
survey describes even Southeastern Europe as underexplored relative to
Western Europe, North America and Asia [6]; and a South African stakeholder
study finds that real-time information for Global South transit has received
limited attention in the literature [7].

<!-- DECISION REQUIRED. The purpose statement below is written for the
     stated-preference study. If the field measurement is included, use
     variant B. Do not ship both. -->

**Purpose statement, variant A (stated preference only).** This paper reports
a stated-preference choice experiment measuring how displayed crowding levels
shift the decision of Almaty bus riders between boarding the first arriving
vehicle and waiting for the next, and estimates the resulting willingness to
wait in minutes per crowding level.

**Purpose statement, variant B (stated preference plus field measurement).**
This paper reports a two-part empirical study of Almaty bus riders: a
stated-preference choice experiment estimating willingness to wait per
displayed crowding level, and a field measurement of the boarding decision
observed at the stop, providing the first paired stated and observed evidence
on the board-or-wait response to real-time crowding information outside
Western Europe, North America and East Asia.

---

## Notes for revision

1. Word count of the four parts excluding purpose statement: approximately
   340. Adding either purpose statement brings the section to roughly 385-410,
   above the 300-350 target in the course notes. Trim the Kraków or Warsaw
   sentence if the target is binding for the journal.
2. Reference [5] (Kapatsila et al. 2025, revealed route choice) is
   abstract-verified only. Volume, pages and sample size are unconfirmed.
   Confirm before submission.
3. Reference [6] is paraphrased from a full-text passage extracted by a tool
   rather than read directly. Re-read at PMC before submission.
4. The claim of geographic absence rests on filtered OpenAlex author-country
   queries recorded in the vault literature base. The Methods or an appendix
   must state the query strings, date and hit counts, otherwise the claim is
   unverifiable by a reviewer.
5. Fedujwar and Agarwal (2024), a systematic review of 40 crowding-valuation
   studies, is still unobtained. Its study-characteristics table would let the
   geographic gap be stated with a count instead of an absence argument. This
   is the single highest-value outstanding document.
