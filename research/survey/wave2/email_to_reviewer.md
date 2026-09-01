# Draft email to the collaborating professor

Draft v1, 2026-09-01. Send with `DESIGN_NOTE.md` attached. Adjust the greeting
and the closing paragraph before sending; the rest is ready.

---

**Subject:** Wave 2 stated-choice design for the Almaty crowding-information
study

Dear Professor [surname],

Thank you for agreeing to work on this. Attached is the design note for the
second wave of our stated-choice survey on bus crowding in Almaty. I would like
your review before we field anything.

**Where we are.** We ran a pilot in February 2026 and collected 215 responses.
Each respondent saw six binary scenarios: a bus has arrived at a stated crowding
level, the next bus arrives after a stated wait and has seats, board or wait.
Holding wait time and time of day fixed at five minutes and morning peak, 68.9%
chose to wait when the arriving bus was packed against 42.8% when it offered
standing room, a difference of 26.1 percentage points. A logit with standard
errors clustered by respondent, over 1247 observations from 209 respondents,
gives a willingness to wait of 7.96 minutes, 95% bootstrap CI [5.89, 11.09].

**Why we are redesigning rather than collecting more.** The pilot instrument has
three limits that sample size cannot fix.

The arriving bus took only two crowding levels, packed and standing room, so
there is no seated reference and the estimate cannot be compared with published
crowding valuations. This turns out to be structural rather than an oversight:
in a board-now-versus-wait task, a seated arriving bus makes waiting dominated,
so the seated level can never appear as a genuine alternative.

Six profiles produced a design matrix of rank four, which caps the model at four
parameters and rules out the heterogeneity specification we want.

Crowding was also partly confounded with wait time, since packed appeared at 2,
5, 7 and 10 minutes while standing room appeared only at 3 and 5.

**What wave 2 changes.** Each task now shows two buses, one at the stop and one
arriving in a stated number of minutes, both carrying a crowding level. Seated,
standing and packed then appear on both sides of the choice, which makes seated
an identifiable reference. Sixteen tasks were selected from the 24 non-dominated
candidates by modified Fedorov exchange under a null prior, constrained so each
wait level appears four times, and split into two blocks of eight that each
remain full rank on the main effects.

**Sample size.** Simulation under pilot-implied priors puts power for the
smaller standing-versus-seated contrast at 0.83 with 300 respondents and 0.61
with 200, so we are targeting 300 completed responses with a recruitment quota
of at least 120 in employment. The pilot was 73.5% students, which is the other
problem we are trying to correct.

**Where I would value your judgement most.**

1. Should crowding be shown as text labels, as photographs of Almaty bus
   interiors, or as the five-level colour scale the deployed application will
   use? The third option ties the survey to the real interface but risks
   measuring the scale rather than the crowding itself.
2. Is a wait-by-crowding interaction worth the extra tasks, or is linear-in-wait
   acceptable at this design size?
3. Would you add a fare attribute for a monetary valuation, given that Almaty
   has a flat fare?
4. Mixed logit or latent class for heterogeneity at n = 300?

The design, the instruments in Kazakh, Russian and English, and the code that
generates all of them are in the project repository, and I am happy to share
access or send any of it directly.

I would also like to ask whether you would want to be listed as a co-author on
the resulting paper. We are targeting *Transportation Research Part C*, with
this survey as one component alongside an in-vehicle occupancy sensing system
and a planned field experiment with the local transit application.

Best regards,
Diyas Tleukin
