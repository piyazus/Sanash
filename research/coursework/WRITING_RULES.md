# Writing rules for the SANASH / RTCI paper, extracted from the Terra research course

Compiled 2026-09-03 by one agent from the primary sources, audited 2026-09-04
by a second agent that re-checked every quote and attacked every conclusion.
Landed in the repository 2026-09-04.

Status: **operational rule set for the paper track, proposed not ratified.** It
is subordinate to `GROUND_TRUTH.md` and to `research/RTCI_RESEARCH_CHARTER.md`.
Where a rule here conflicts with either, they win and the conflict is fixed in
the same change.

It supersedes `research/coursework/RESEARCH_METHOD_GROUNDING.md`, which covered
the same material and is now section 6 of `ARCHIVE.md`. That supersession is
recorded in `GROUND_TRUTH.md` section 8.2 and in
`development/experiments/log.md`, per `CLAUDE.md` hard rule 7. It does not
supersede `research/coursework/course_notes.md`, which stays as the full
lesson-by-lesson conspectus and remains the only record of the material this
file does not turn into a rule. Section 11 records what each prior pass got
right, missed and got wrong.

## 0. How to read this file

**Sources actually read.** All six earlier transcripts in
`research/coursework/transcripts/`, the 2026-09-01 results-section class
transcript, and six of the seven slide decks extracted with `pdftotext -layout`:
`Lit. review.pdf`, `Methodology.pdf`, `Methodology for emp. papers.pdf`,
`Foundations class- group 3.pdf`, `2nd class-Reading.pdf`, and
`Analysis- slides.pdf` (repo root). Repo context read: `GROUND_TRUTH.md`,
`research/RTCI_RESEARCH_CHARTER.md`, `research/REPLICATION_TARGET.md`,
`research/survey/README.md`, `research/survey/wave2/DESIGN_NOTE.md`,
`research/lit_review/search_log.md`,
`research/coursework/results_section_draft.md`.

No rule in this file derives from `2nd class-Reading.pdf`. It was read and
contributed nothing the other decks do not already carry.

**Audit record, 2026-09-04.** Every blockquote in this file was re-checked
against its cited source by string match on a whitespace- and
punctuation-normalised copy of the transcript or extracted deck text, and every
transcript timestamp was re-derived from the position of the quoted text.
Failures found and fixed are listed in §14. Quotes that survived unchanged carry
no marking.

**Could not verify.** `research/coursework/Research - intro.pdf` has no usable
text layer. `pdftotext` returns two words, "Research" and "Project"; the rest of
the twenty pages are images and no OCR tool is installed in this environment.
Nothing in it can be quoted. Everything this file says about that deck is
second-hand from `course_notes.md` §§4.6-4.9 and is labelled as such.

**Quote fidelity.** Every quote below is verbatim from a machine transcript
produced by faster-whisper `small`, int8, per `README.md`. The ASR mangles names
and some technical terms, so several quotes contain visible errors ("prison's
criteria" for PRISMA, "cheese square test" for chi-square, "Alma decommuners"
for Almaty commuters, "no AI" for "no I" in R58). They are not silently
repaired. Where a quote is broken enough that the meaning depends on repair, the
rule says so rather than guessing. Slide quotes are verbatim from the extracted
text layer and are cited by slide number as printed in the deck.

**Mentor identification.** The transcripts do not label speakers. Where a mentor
is identified the evidence is given. Where identification fails, the rule cites
the recording rather than a person.

- **Mentor A** delivers `lesson1_part1`, `lesson1_part2` and `video1101430871`.
  Evidence: in `lesson1_part1` [25:29] she describes her own paper as being on
  "the Declaration of Human Rights"; in `video1101430871` [11:26] she says "my
  question was on the Declaration of Human Rights in the 21st century". Both
  recordings also name a colleague, "Matthew", but only `lesson1_part2` [00:08]
  attributes an AI-permissive position to him; the `video1101430871` mention at
  [13:30] is about search result counts. The `lesson1_part1` mention is rendered
  "Mathemodes" by the ASR and is not about AI.
- **Mentor B** delivers `video1768002125` and `video1168216779` (one session,
  split by a break). Evidence: `video1768002125` closes at [24:23] announcing a
  five-minute break, and `video1168216779` opens at [00:07] with "you're free to
  take a five minute break".
- **Mentor C** delivers `video1433589796` up to about [36:50].
- **Mentor D** interrupts `video1433589796` at [36:50] to dissect
  `Mohlaroy--RAS.docx` on screen. Evidence: at [37:04] the new speaker says "I
  was not supposed to teach today", and at [37:17] addresses the previous
  speaker by name.
- The **2026-09-01 results-section class** speaker is plausibly Mentor D (same
  demand for "real evidence", same direct address to a student, same verbal
  tics), but this is **INFERENCE** from style, not from any named
  self-identification. Rules below cite the recording, not the person.

---

## 1. Research question

### R1. The paper needs one research question, and every part of the paper is measured against it

> "Every single result you report, it must trace back I cannot trust it enough,
> it must trace back to your research question."
> (`2026-09-01_results-section-class.txt` [00:02:01])

> "Your RQ is your filter. Remember that"
> (`Analysis- slides.pdf`, slide 8, Step 1)

**Why it matters, in the mentor's logic.** The RQ is the only thing that makes a
number a finding rather than a fact. Without it there is no principle for
deciding what goes in and what goes out, so the paper becomes a data dump.

**For SANASH.** The charter (`RTCI_RESEARCH_CHARTER.md` §1) carries three
different RQ formulations: a 12-word readable one, an operational causal one,
and a narrowed field-study one. That is defensible for a research programme and
dangerous for a single paper. Pick one RQ per paper and put it in the
introduction, the methodology, the results and the discussion in the same words.
The results draft already states its RQ at the top; make that the only one in
the document. Which RQ it should be is not settled here: see C1 and the conflict
recorded there against R5.

### R2. The RQ is 10-15 words, and length failure is a named failure mode

> "Your research question should be no longer than, you know, 10 to 15 words."
> (`lesson1_part2.txt` [12:03])

> "We have had students in the past who have presented us with research
> questions which are for like three sentences long. That is just not okay."
> (`lesson1_part1.txt` [16:03])

The homework brief prints the same figure: "RQ should be 10-15 words"
(`Foundations class- group 3.pdf`, Part 2).

**Why it matters.** A question you cannot say in one breath is a question you
have not narrowed.

**For SANASH.** The charter already handles this correctly and explicitly: it
keeps the 12-word readable RQ for the course and the longer operational
formulation for the protocol, and says so. Do not let the long form leak into
the paper's title or abstract. The charter is right that 10-15 words is a course
rule, not a TR-C requirement.

### R3. The RQ must name a specific factor, a specific outcome and a specific group

> "it's specific. It has a specific platform that's mentioned to TikTok. It
> looks at a specific impact, which is the political polarization. And it looks
> at a specific group of people within a specific country."
> (`lesson1_part1.txt` [13:44])

A table of narrowing axes (time, place, population, platform, variable) is
recorded in `course_notes.md` §2.6 and attributed there to "the deck". It is
**not in the extracted text layer** of `Foundations class- group 3.pdf`, so it
either sits on an image slide or comes from a recording. Treat it as
second-hand, not as a quotable deck statement.

**For SANASH.** The RQ as used in class satisfies the spoken rule: factor is
real-time occupancy information, outcome is the boarding decision, group is
Almaty commuters. The weak link is the group. `GROUND_TRUTH.md` §3.1 records the
sample as 77.7% aged 14-24 and 73.5% students. "Almaty commuters" is therefore
the aspiration, not the population studied. See C3.

### R4. A bad RQ is one that is too broad, unmeasurable, opinion-based, emotionally loaded, or requires data you cannot get

> "a bad question typically has things that are, you know, too broad, you can't
> measure it, you know, it's purely opinion based, it's too emotional, and it
> requires inaccessible data"
> (`lesson1_part1.txt` [18:05])

**For SANASH.** The relevant red flag is the last one. The causal RQ requires
observing real boarding behaviour in Almaty, which requires Innoforce
cooperation, app randomisation and boarding linkage, none of which exist
(`RTCI_RESEARCH_CHARTER.md` §6, §12). By the mentor's own test the causal RQ is
currently not researchable with data you can access. It becomes researchable
only after Innoforce answers. The charter records this itself: §6 says no power
analysis and no promise of a causal field study is permitted until the observed
outcome is chosen. State it plainly rather than writing around it.

### R5. The course-taught RQ example in the Analysis deck is a student's own RQ, not a generic teaching example

> "Now, let's look back at, I think it's the ask question, right? To what extent
> does live boss occupancy information change boarding decisions of Alma
> decommuners, right?"
> (`2026-09-01_results-section-class.txt` [00:04:04])

> "It's yours." (same, [00:04:11])

> "I stole it from you, but I think it's okay." (same, [00:04:13])

The Analysis deck carries the same RQ in printed form: "To what extent does
real-time bus occupancy information change boarding decisions of Almaty
commuters?" (`Analysis- slides.pdf`, slide 8).

**Caveat.** The recording does not name the student being addressed. That the
addressee is Diyas is **INFERENCE** from the RQ's content and from the exchange
at [00:07:47] onward, where the same student is asked how many people were
interviewed and answers "225". No mentor says the name on tape.

**Why it matters.** It settles a question the previous pass left open, and it
means the analysis lecture and the results lecture are worked examples built on
this project's own RQ. The numbers on those slides (72%, 49%, 23 percentage
points, and the 214 respondents used in the class worked example) are
illustrative and invented, but the structure is tailored.

**For SANASH.** Two consequences. First, `RESEARCH_METHOD_GROUNDING.md`'s note
that the RQ "must not be treated as" connected to Sanas is now contradicted by
the transcript and should not be carried forward. That file flagged its own
claim as inferred and raised the alternative as its open question 8, so this is
a correction to an acknowledged uncertainty, not the exposure of an error. See
§11.2. Second, a mentor is on record having seen and accepted this RQ, which
raises the cost of changing it. C1 proposes changing it; that tension is named
there and is not resolved in this file.

---

## 2. Structure and planning

### R6. Outline before writing. If you cannot state the structure, do not start

> "So before writing your first paragraph, always ask yourself, do I know the
> structure of my paper? If the answer is no, don't start writing yet."
> (`video1768002125.txt` [04:53])

> "By the time they reach page six, they no longer know where the paper is
> going." (same, [03:48])

**For SANASH.** The repo has a blueprint (`research/TRC_PAPER_BLUEPRINT.md`) and
a sprint plan (`research/PAPER_SPRINT_30D.md`) but the drafted sections
(`research/coursework/results_section_draft.md`,
`research/paper/manuscript/01_introduction.md`) were written before the RQ was
frozen. Freeze the RQ, then write the outline, then write prose, in that order.

### R7. The empirical paper structure is fixed: introduction, literature review, methodology, results, discussion, conclusion

> "The introduction explains the problem. The literature review shows what
> previous researchers already know. The methodology explains exactly what you
> did. The results present your evidence. The discussion explains what those
> results mean." (`video1768002125.txt` [11:45])

The spoken list stops at discussion. The conclusion is supplied by the process
diagram in `Foundations class- group 3.pdf`: "Results > Analysis(discussion) >
Conclusion > Publication".

**For SANASH.** The results draft is numbered 5.1-5.4, implying results is
section 5. Keep that numbering consistent across all drafted sections so they
can be assembled without renumbering.

### R8. Every paper needs a literature review, empirical or not

> "So all the papers, no matter what you're studying, will have a literature
> review in which you have to analyze existing research"
> (`lesson1_part1.txt` [23:10])

**For SANASH.** Non-negotiable, and currently the weakest part of the project.
See §3.

### R9. Do not mix the literature-review paper and the empirical paper routes without asking

> "typically no, we recommend that you use one or the other. However, there may
> be instances in which you can kind of combine elements, but that is very rare."
> (`lesson1_part1.txt` [22:09])

**For SANASH.** The RTCI paper is unambiguously empirical. The charter §5 calls
it "quantitative empirical multi-study research", which is the right call. The
coursework literature-review homework
(`research/coursework/HW_lit_review_submission.md`) is a separate deliverable
and must not be spliced into the paper as if it were the paper's literature
review without re-doing it under the paper's own RQ.

---

## 3. Literature review

### R10. A literature review is a map of the field, not a list of summaries

> "It's an analysis of existing research connected to your topic and research
> question. It's not a book report or it's not a list of summaries of existing
> research. It's more of a map of your field."
> (`video1433589796.txt` [00:20])

> "That's simply a list of summaries." (`video1768002125.txt` [07:09])

**For SANASH.** `research/refs/base/references.md` holds 153 verified
bibliographic rows and, per `GROUND_TRUTH.md` §9, **none of the seven newest are
read** and their status is `LISTED`. The same section records that the content
of the works beyond title, venue and authorship was never checked for any row. A
bibliography is not a map. Nothing in that file can enter a literature review
until someone reads the paper.

### R11. Organise by theme, never by author. Never open a paragraph with a name

> "do not open paragraphs with names"
> (`video1433589796.txt` [08:09])

> "Do not open paragraphs with a name. Those who do not have experience - write
> 'Smith (2021) says... Johnson (2022) says... Lee (2023) says...' and that is a
> list, not a review. Lead with the idea."
> (`Lit. review.pdf`, slide 4)

> "Because readers care about ideas, not chronology."
> (`video1768002125.txt` [08:01])

Mentor D reinforced it against a real published paper:

> "if you go with authors it's going to be like such a messy thing you can never
> organize anything by authors it should be always by themes"
> (`video1433589796.txt` [38:14])

**For SANASH.** Candidate themes from what is already in the repo: stated
willingness to wait for a less crowded vehicle; revealed crowding response from
smart-card and AVL data; real-time crowding information delivered to real
passengers; simulation of information feedback on bunching and load
distribution; automated onboard occupancy sensing. Drabicki, Bansal/Hörcher/
Graham, Zhang/Jenelius/Kottenhoff and the CV literature then sit inside themes
rather than each getting a paragraph. These themes are a proposal built from
reference-base titles, not from read papers. **INFERENCE.**

### R12. Synthesise, do not summarise. Three or more sources per point

> "again don't summarize you should synthesize the information"
> (`video1433589796.txt` [12:24])

> "Summary --- What did ONE source say? ... One voice, in isolation. Useful, but
> this alone is not a review. Synthesis --- What do MULTIPLE sources tell us
> together?" (`Lit. review.pdf`, slide 8)

> "there should be at least three sources that support one point together"
> (`video1433589796.txt` [11:36])

**For SANASH.** This sets a hard floor on reading. Five themes at three sources
each is fifteen papers read properly, before any of the comparison work in R14.

### R13. Each theme paragraph is topic sentence, evidence, analysis, transition

> "1.Topic sentence. State the main idea of the paragraph in your own words
> 2.Evidence. Bring in the studies that support that idea 3.Analysis. Explain
> the patterns, agreements, and disagreements 4.Transition. Smooooothly.... into
> the next theme." (`Lit. review.pdf`, slide 9)

> "your topic or your paragraph will still be weak because analysis is an
> important part" (`video1433589796.txt` [17:48])

### R14. Map disagreements rather than picking a winner, and name the reason for the disagreement

> "Synthesis really comes alive when sources disagree. Your job is not to pick a
> winner but to map the disagreement clearly so the reader sees the whole
> debate!!" (`Lit. review.pdf`, slide 12)

The deck's worked move is to attribute the divergence to a design difference:
"The disagreement may therefore reflect age rather than a true conflict in the
evidence." (same slide)

**For SANASH.** The obvious axis is stated preference against revealed
preference. The claim that stated-preference studies find substantial
willingness to wait while revealed-preference smart-card work finds smaller
behavioural responses is a plausible framing, and if it holds it is a method
difference rather than a contradiction in the evidence, which is exactly the
move the deck teaches. It is **not supported by any read paper in this repo**
and cannot enter the text until the sources are read. **INFERENCE.**

### R15. Add a "so what" after every piece of evidence

> "the pure summary retelling each study without analyzing it is bad after you
> introduce the source you should always try to ask a question such as so what"
> (`video1433589796.txt` [25:10])

### R16. Give real evidence, not adjectives

> "so she gives real evidence okay this is what i'm craving to see in your
> guys's paper okay i want to see that i want to see real evidence whether it's
> in data whether it's in maps whether it's in uh graphs doesn't matter give me
> real data support your arguments"
> (`video1433589796.txt` [42:52])

**For SANASH.** In the model paper Mentor D was praising, the evidence sentence
read "in a scoping review of 30 31 independent studies 12 that author reported
between 60 to 100 percent of the population failed to understand how ras
functions" (`video1433589796.txt` [42:32], ASR mangled). The SANASH equivalent
is a number from a cited paper, for example a reported willingness-to-wait
value, not "crowding is known to be unpleasant".

### R17. Lean on recent work; the recency rule is 2020 or later with one or two exceptions

> "What we do recommend is that you don't pick any sources that are any late in
> 2020. We say 2020 and forwards." (`lesson1_part2.txt` [03:58])

> "My rule is that you can have one or two sources that might be out of date,
> but one or two max, try not to go over that." (same, [04:15])

**Recorded disagreement.** The Foundations homework brief says "published within
the last 10 years" (Part 3), which is a looser rule than the lecture's 2020
cutoff. Both are in the course material; they are not the same rule.

**For SANASH.** This rule conflicts with the paper's own needs. The single
closest field precedent found so far, `zhang_2016_stockholm`, is from 2016
(`REPLICATION_TARGET.md` §1), and the crowding-cost literature the argument
rests on is older still. The right move is not to hide the old sources but to
justify them under R34: an explicitly justified date window beats an unexplained
one.

### R18. Find the gap by asking what the studies have in common

> "always ask yourself, what do these studies have in common? Often the answer
> reveals the gap. If every paper studies university students, perhaps nobody
> has examined younger learners. If every study uses surveys, perhaps
> qualitative interviews are missing"
> (`video1168216779.txt` [10:16] to [10:34])

A related tip, from the literature-review lecture and not on any slide:

> "read a couple of uh papers specifically the discussion part in that
> discussion part there is like a limitation section"
> (`video1433589796.txt` [47:31])

**For SANASH.** Applied honestly, this produces the charter's candidate gap:
almost everything is rail, almost everything is stated preference or simulation,
almost nothing is Central Asia, almost nothing observes an actual boarding. The
charter labels that gap candidate rather than proven, and C2 explains why it
cannot yet be written as a sentence.
---

## 4. Introduction and the research gap

### R19. The introduction has four parts in order: background, problem, research gap, purpose statement

> "background, problem, research gap, purpose statement"
> (`video1768002125.txt` [19:01])

The five-question checklist:

> "have I introduced the topic? Have I explained why it matters? Have I clearly
> described the problem? Have I justified the research gap? Have I told readers
> exactly what my paper will do?" (`video1168216779.txt` [17:15] to [17:23])

### R20. Broad is fine, vague is not

> "such as educational as always been important or technologies changing the
> world. Although these statements are true, they don't tell readers anything
> specific about your topic." (`video1768002125.txt` [22:01])

**For SANASH.** "Public transport is important for cities" is exactly the banned
sentence. The background should open on something specific and checkable, for
example bus crowding as a routine condition in Almaty, and move directly toward
information provision.

### R21. A research gap is under-explored, not unstudied, and it must be supported with evidence

> "They think a research gap means nobody has ever studied the topic before.
> That's almost number two." (`video1168216779.txt` [07:11]; the ASR has mangled
> what was almost certainly "that's almost never true")

> "Notice that a research gap must be supported with evidence. You cannot simply
> write, there's little research blank. You need to demonstrate that by
> referring to the literature you've reviewed." (same, [08:03])

> "One mistake students make often is claiming a gap that isn't really a gap."
> (same, [09:48])

**For SANASH.** This is the sharpest rule in the course for this project. The
charter's gap sentence, "Direct causal field evidence on how app-displayed bus
crowding information changes individual boarding decisions is limited,
especially in bus systems and Central Asian cities", rests on three OpenAlex
queries and no read papers. The charter itself labels it a candidate rather than
a proven gap, and `REPLICATION_TARGET.md` §1 states that the search makes the
charter's candidate gap more plausible but does not prove it. As
written the gap claim is the failure mode the mentor names. It cannot go into
the paper until the named papers are read and the gap is stated as "X studied
rail, Y studied stated preference, Z did not observe boarding".

### R22. Do not say "more research is needed"

> "avoid simply saying that more research is needed. Explain specifically what
> remains uncertain" (`video1768002125.txt` [23:48])

### R23. The purpose statement is the clearest sentence in the paper, with topic, population and context, and no dramatic language

> "Avoid dramatic language like dis-revolutionary study completely changes our
> understanding. Academic writing values precision much more than exaggeration.
> Simple direct language is usually the strongest."
> (`video1168216779.txt` [13:02])

The test: "imagine that someone reads only your purpose statement. Would they
understand what your paper is about?" (same, [12:17])

**For SANASH.** The charter's TR-C submission thesis ("An edge-generated,
app-delivered RTCI intervention is causally evaluated at the boarding-decision
level and embedded in a behaviour-aware model...") describes the eventual
journal paper, not the one being drafted now. The charter says so itself:
survey-only work "does not meet the target bar" and remains
instrument-development. Whatever purpose statement goes into the current draft
must describe what the current paper actually does. If that paper is the
stated-preference study, the purpose statement says so.

---

## 5. Methodology

### R24. The whole standard is repeatability by a stranger

> "if anyone ever wants to repeat your study, they can. If a reader can't look at
> your study and say, Oh, I can repeat that, then your methodology isn't clear
> enough." (`video1101430871.txt` [01:32])

> "The test to keep in mind - could a stranger, reading only your methodology,
> repeat your study and expect similar results? If no, it isn't finished."
> (`Methodology for emp. papers.pdf`, slide 6)

> "A weak methodology sinks a paper faster than a weak conclusion, because if
> the method is flawed, the results mean nothing regardless of how interesting
> they sound." (same slide)

**For SANASH.** The survey is reproducible in the repo (`research/survey/` holds
the raw export, rebuilt dataset and fitted model), which is more than most
student papers have. `research/survey/README.md` already carries much of the
methodology in prose: the six scenarios and their attribute levels, the two
languages, the fielding window, the convenience sampling statement, the raw file
hash, the estimator and the exclusion of the 43 blank or free-text answers. What
is missing is narrower than a whole methodology section: the recruitment
channel, the ethics and consent treatment (R26), and the rationale for each
design choice (R25). Adapt the existing prose rather than writing it again.

### R25. Answer "why" for every methodological choice

> "I cannot stress this enough, you have to answer the why you have to be able
> to explain to someone why you have chosen decisions that you have"
> (`video1101430871.txt` [38:53])

> "The really specific thing, the main thing in your methodology is answering
> the why people want to understand why you did the things that you did."
> (same, [07:05])

**For SANASH.** Why six scenarios and not one. Why 2, 3, 5, 7 and 10 minutes as
the wait levels. Why packed against standing room rather than a five-level
scale, given the product uses five levels. Why bilingual Kazakh and English and
not Russian. That last one is currently unexplained and a reader in Almaty will
notice immediately.

### R26. The empirical methodology has six components, and ethics is one of them

> "1. Research design. What type of study was this? 2. Population. Who is the
> study about? 3. Sample. Who actually participated? 4. Instrument. What did you
> ask, and why? 5. Data collection. When, where, and how? 6. Ethical
> considerations. How did you protect participants?"
> (`Methodology for emp. papers.pdf`, slide 10)

> "Every complete survey methodology section includes all six. Miss one and we
> will notice surely!" (same slide)

**Gap in the course, worse than previously recorded.** The deck's text ends on
page 15 of 15 with a slide headed "Component 5: Types of survey questions".
Question types are not component 5 of the six-component list; component 5 is
data collection. So the deck covers components 1 to 4 in order and then a fifth
slide on a different topic. **Neither data collection nor ethics is taught**, and
no recording covers either. The course states the six-component requirement and
teaches four of them.

**For SANASH.** This is a live problem, not a formality. `GROUND_TRUTH.md` §3.1
records that the age band starts at 14, "что означает возможное участие
несовершеннолетних", and `research/survey/README.md` repeats it. A survey that
may have enrolled minors needs a written consent and data-handling statement,
and the course cannot supply the template. This must be sourced elsewhere before
submission. The charter already carries it as an open decision (§5, "ethics
process остаются открыты"; §12, item 7).

### R27. Population and sample are different things, and the population is only who your conclusions apply to

> "Be honest about scope. If you only surveyed your own school, your population
> is not 'students worldwide.' Narrowing this correctly makes your paper
> stronger, not weaker." (`Methodology for emp. papers.pdf`, slide 12)

### R28. Report the sample fully: n, age range, recruitment, location

> "Report all of these please: Number of participants, Age range, Gender
> distribution, if relevant to your question, Schools or institutions involved,
> Location, How you recruited them"
> (`Methodology for emp. papers.pdf`, slide 13)

> "a convenience sample is whoever was reachable, which is what most of you will
> use, and it's acceptable as long as you say so" (same slide)

> "aim for at least 100 responses for a quantitative survey if you can"
> (same slide)

**For SANASH.** n = 215 clears the 100 threshold. `research/survey/README.md`
already calls the sample a convenience sample with no sampling frame, no quotas
and no population representativeness, so the "say so" condition is met in the
repo and only needs carrying into the paper. The **specific recruitment channel**
is the item that is not recorded anywhere. `research/survey/wave2/DESIGN_NOTE.md`
says wave 1 was student-heavy "because of how it was distributed" without naming
the distribution route, and that has to be established from Diyas before this
component can be written. Gender was not asked, which by this slide is
acceptable if the absence is stated, and the results draft does state it.

### R29. Describe the instrument's architecture, and cite any borrowed instrument as a strength

> "Do not just say 'we made a survey.' Describe its architecture. If you adapted
> questions from a published study, say which one and cite it, because borrowing
> a validated instrument is a strength worth advertising."
> (`Methodology for emp. papers.pdf`, slide 14)

**For SANASH.** `REPLICATION_TARGET.md` §3 proposes adapting Drabicki et al.'s
willingness-to-wait design and notes that the same team reused their own design
in a changed context, which would make reuse an accepted practice in this
literature rather than borrowing. If wave 1 or wave 2 in fact follows that
design, this rule converts a possible weakness into a stated strength. The
catch: the same file says the resemblance is "предполагается, а не проверено",
and `research/survey/wave2/DESIGN_NOTE.md` does not cite Drabicki anywhere. You
cannot advertise an adaptation of a paper nobody has read.

### R30. Surveys measure what people report, not what they do, and this belongs in limitations

> "The tradeoff to be honest about surveys tells you what people report about
> themselves, not what they actually do. Self-reported data can be biased by
> memory, honesty, and how you phrased the question. Good papers acknowledge
> this in their limitations!!!" (`Methodology for emp. papers.pdf`, slide 8)

**For SANASH.** This is the most load-bearing sentence in the course for this
paper. Six hypothetical scenarios measure stated intention. The RQ asks what
information *changes*. The distance between those two is the paper's central
honesty problem and must be named in the text, not buried in a limitations list.
`GROUND_TRUTH.md` §3.1 and `research/survey/README.md` both already make the
distinction; the coursework results draft does not.

### R31. Training a model on a public dataset counts as empirical research

> "Computational and simulation studies. Common in computer science and physics.
> You build or test a model, run it on data, and report performance. Training a
> machine-learning model on a public dataset counts."
> (`Methodology for emp. papers.pdf`, slide 7)

**For SANASH.** Useful, but it does not rescue the CV track for this paper.
`GROUND_TRUTH.md` §6 records one CPU smoke test on 16/8/8 images in which the
trained model lost to a constant predictor (test MAE 140.41 against 109.82).
That is a pipeline check, not a result, and reporting it as a result would
violate R37 and `CLAUDE.md` hard rule 3.

### R32. If a paper both reviews literature and collects data, split the methodology in two

> "you might have to split into two sections. So your first section will be kind
> of like the same here in which you talk about how you found your sources and
> whatever. And then your second section of your methodology may be discussing
> the actual like data collection" (`video1101430871.txt` [44:54])

**For SANASH.** This is the correct shape: a short review-method subsection
covering the databases, search strings and dates actually run, then the survey
methodology proper. `research/lit_review/search_log.md` holds raw material for
the first subsection, but it opens by saying it is **not** a protocol review,
only exploratory runs made to test one claim. It cannot supply PRISMA numbers.
The protocol review under `research/LIT_REVIEW_PROTOCOL.md` has not been run.

### R33. Report the search string, the databases, the date the search was run, and the funnel numbers

> "Also report the time period and why, language limits, publication types, and
> the date you ran the search. Fields move, so a search has a shelf life."
> (`Methodology.pdf`, slide 8)

> "The initial search returned 340 records. After removing 45 duplicates, 295
> titles and abstracts were screened, of which 58 underwent full-text review. A
> final set of 22 studies met all eligibility criteria and was included in the
> review." (`Methodology.pdf`, slide 14, worked example)

**Recorded disagreement.** On how many records a good search should return, the
deck and Mentor A differ, and Mentor A says so out loud:

> "For example, Matthew, he said 100 to 800 is fine. I think if anything, 100 to
> 400 might be fine." (`video1101430871.txt` [13:30])

`Methodology.pdf` slide 9 carries the 100 to 800 figure. Do not report this as a
single course rule.

### R34. Write eligibility criteria before screening, make them testable, tie each to the RQ, justify the tight ones, apply them consistently

> "you have to do this before because otherwise you may find that bias kind of
> creeps into your writing" (`video1101430871.txt` [14:11])

> "make them testable. A criteria should always answer a yes or no"
> (same, [18:30])

> "If you can't explain why a rule exists, delete it" (same, [19:08])

> "Apply them consistently. If you excluded one study for using university
> students, you cannot keep another one with the same problem just because you
> liked its findings." (`Methodology.pdf`, slide 12)

**For SANASH.** `research/LIT_REVIEW_PROTOCOL.md` exists but is an uncommitted
draft (`GROUND_TRUTH.md` §9). Whatever it says has to be frozen before screening
starts, or the criteria are post-hoc by definition.

### R35. Screen in two passes and log a reason for every full-text exclusion

> "Pass 1, title and abstract. Fast. ... Being generous here is cheap; being
> generous later is expensive. Pass 2, full text. Slow."
> (`Methodology.pdf`, slide 13)

> "Remove duplicates first. The same article appearing in Google Scholar and
> ERIC is one record, not two, and forgetting this inflates your numbers
> dishonestly." (same slide)

> "track everything from day one, have a spreadsheet"
> (`video1101430871.txt` [24:03])

**For SANASH.** `research/lit_review/evidence_matrix_template.csv` is the right
artifact. It is currently a header row and nothing else.

### R36. Limitations are expected even though they are not on the methodology slides, and they must be specific

Mentor A on the model paper:

> "what was not on the slides, but I think it's very good that this person has
> included its limitations" (`video1101430871.txt` [40:16])

> "there was no pre-registration protocol that was done before conducting the
> scope, which was a limitation to their work" (same, [40:45])

She distinguished two kinds when a student asked (same, [42:32] to [43:20]):
limitations arising from your process, and limitations arising from what the
included sources actually said.

**Second-hand.** The claim that the model paper placed its limitations inside
the methodology section as §3.7 comes from `course_notes.md` §13.
`Mohlaroy--RAS.docx` was not opened for this pass and the recording does not
give a section number.

**For SANASH.** Candidate specific limitations, all of them already established
facts in this repo and most of them already written down in
`research/survey/README.md` under "Known limitations": young student-dominated
convenience sample; stated intention rather than observed boarding; two
languages excluding Russian; scenarios use two crowding levels while the product
uses five; no preregistration for the survey wave; and the crowding levels were
described in words rather than shown as the icons the app would use. The last
two are not yet in the repo list.
---

## 6. Results

### R37. Report only what you found. No causes, no interpretation, no opinion, not even a statement that the finding is interesting

> "your only job here is to report what you found, and you don't need to report
> what it means, what why it matters, and what you think caused it."
> (`2026-09-01_results-section-class.txt` [00:00:35])

> "you cannot be telling any single cause in the result section. You just report
> data. That's all." (same, [00:01:13])

> "in the result section, I want no personal opinion, no statement, nothing, not
> even telling me why it's important, not even telling me why it's interesting,
> none of that. Just raw data and description of that data in academic
> language." (same, [00:01:37])

**Within-mentor tension, recorded not flattened.** Later in the same class the
same speaker relaxes this:

> "you can say more likely, or you can even do some associations, you can talk
> about the cause, but just be careful there and do not state it with a bold
> dot." (same, [00:38:57])

The two statements are not consistent. Neither is picked here. The reading that
satisfies both, and the one this file follows for practical purposes, is: no
causal claim, and any associative statement stays in the hedged forms listed in
R41. That reading is **INFERENCE**; the mentor never reconciles the two out
loud.

**For SANASH.** The existing `results_section_draft.md` complies. It reports
percentages and gaps and never says why. Keep it that way.

### R38. Ban the words "this shows", "this suggests", "this means", "this proves" from results

> "if you catch yourself writing something like this shows, this suggests, this
> means, this proves, just please stop at this point."
> (`2026-09-01_results-section-class.txt` [00:39:14])

**For SANASH.** Run this as a literal grep over any results draft before
submission.

### R39. Results have four parts in order: who answered, the basic numbers, how groups compare, what moves together

The class structure (`2026-09-01_results-section-class.txt` [00:06:02],
[00:06:09], [00:07:00], [00:10:24], [00:20:54], [00:32:20]) matches the Analysis
deck's Step 5:

> "WHAT IS HAPPENING? ... WHO IS DIFFERENT? ... WHAT MOVES TOGETHER?"
> (`Analysis- slides.pdf`, slide 12)

> "before any findings, you tell the reader, who are you even talking about? And
> you keep this really, really short"
> (`2026-09-01_results-section-class.txt` [00:12:41])

**For SANASH.** `results_section_draft.md` already uses this exact four-part
skeleton as 5.1 to 5.4. That is the strongest thing about the draft.

### R40. Order by importance to the RQ, never by the order of survey questions

> "you do not order it the way that your survey questions appeared. It doesn't
> work like this. Please do not be doing that mechanical writing. Nobody cares
> that question four came before question seven."
> (`2026-09-01_results-section-class.txt` [00:43:06])

> "you start with whatever findings most directly and clearly answer your
> research question." (same, [00:43:39])

> "if my reader only read the first paragraph of my results, to understand the
> core answer to my question. If I ask myself this, if the answer is no, then I
> don't even know what I'm doing" (same, [00:43:53] to [00:44:04])

**Tension in the same lecture.** R39 puts demographics first; R40 says the first
paragraph should carry the core answer. Both cannot be literally true. The
resolution the lecture models on the worked transport paper is that R39 governs
the section skeleton and R40 governs the order of findings *within* each part.
Marked **INFERENCE**; the mentor never reconciles the two out loud.

**For SANASH.** The draft puts demographics first, which follows R39. It is
worth adding one sentence at the head of the results that states the headline
number, so the first paragraph does answer the RQ.

### R41. Association is not causation. Use the safe language list

> "there's safe language like x was associated with y x was more more common
> among y respondents who did x or more likely to report why you can say that
> it's okay. I just want you to keep that likely word."
> (`2026-09-01_results-section-class.txt` [00:36:40])

> "there's more dangerous language x caused y x led respondents to y x made
> people do y your survey cannot prove any causation right it truly can't"
> (same, [00:37:09])

> "Safe language you guys can use 'X was associated with Y.' Dangerous language.
> Very! 'X caused Y.'" (`Analysis- slides.pdf`, slide 15, Step 8)

**Why it matters, in the mentor's logic.** He backs it with a consequence story:
a PhD student whose paper was revoked by the journal and whose degree was
revoked (`2026-09-01_results-section-class.txt` [00:35:34] to [00:35:51]). The
point is that overclaiming is not a style error.

**For SANASH.** The charter carries the same rule independently: §5 states that
unless field identification is randomized or credibly quasi-experimental, only
"associated with" is permitted in results. See C1.

### R42. Every results sentence does two jobs: state a finding with a number, and tie it to the RQ

> "every single results sentence has exactly two jobs ... job one is to stay
> defining with the actual number attached to it"
> (`2026-09-01_results-section-class.txt` [00:40:31]; "stay defining" is the ASR
> rendering of "state a finding")

> "Job number two is to remind the reader why you're ever, why you're even
> telling them this by tying it to your research question." (same, [00:41:05])

> "mathematician, right?" (same, [00:40:59])

> "You cannot prove anything without numbers." (same, [00:41:03])

### R43. Percentage points and percent change are different calculations

> "This is simple sub subtraction between two percentages. It's not a percent
> decrease." (`2026-09-01_results-section-class.txt` [00:19:01])

> "four percentage points and a percent change. There are two different
> calculations entirely. ... And mixing them up, trust me, is the fastest way to
> make the reader understand that you don't know anything on this world about
> what you're writing" (same, [00:20:29] to [00:20:43])

**For SANASH.** The draft is already correct on this: "a difference of 26.1
percentage points", "a gap of 33.3 percentage points". Do not let a later editor
convert these into percentages.

### R44. Report the biggest and most relevant numbers, not every number

> "report the biggest numbers, report the most relevant numbers, and the ones
> that actually do give some, let's say, sort of help to researching your
> research question" (`2026-09-01_results-section-class.txt` [00:25:38])

> "Do we need to analyze all of these? NO. Of course no Ask just one question
> Does this help answer my RQ?" (`Analysis- slides.pdf`, slide 8, Step 1)

### R45. Anything that does not move you toward the RQ goes in an appendix, not the results

> "If finding does not move you closer to answering that, you should leave it out
> of results. Okay, it might be something good, it might be something
> interesting. You might put it in appendix."
> (`2026-09-01_results-section-class.txt` [00:04:15])

**For SANASH.** The trip-purpose and crowding-exposure items in the survey look
like context rather than findings and probably belong in the sample description
or an appendix. Marked **INFERENCE**: no mentor saw these items.

### R46. Cross-tabulate. Two variables together is analysis; two variables apart is not

> "Please do not look at Bus frequency and Willingness to change boarding
> separately... Put them together!!!"
> (`Analysis- slides.pdf`, slide 14, Step 7)

**Note.** `RESEARCH_METHOD_GROUNDING.md` lists Steps 1-6 and 8-12 and omits
Step 7 entirely. This is the rule that pass missed.

**For SANASH.** The draft's §5.3 does exactly this, crossing occupation and trip
frequency against the scenario choice. A cross-tabulation table would present it
better than prose.

### R47. Clean the data before analysing it, and report what you removed

> "Before analyzing you need to clean. Check these Missing responses ...
> Duplicate responses ... Impossible responses ... Inconsistent responses ...
> Unusable responses" (`Analysis- slides.pdf`, slide 10, Step 3)

**For SANASH.** The draft reports "Of 1290 possible scenario answers, 1247 were
usable; 36 were left blank and 7 were free-text replies that stated neither
option." That is the rule executed correctly and it is worth keeping verbatim.

### R48. A null or contradictory result is a result

> "Your hypothesis being wrong is not a failed study. It is not!!!"
> (`Analysis- slides.pdf`, slide 16, Step 9)

> "Interesting research very very often lives in contradictions...
> High preference + low behavior"
> (same, slide 17, Step 10)

**For SANASH.** The deck's own worked example on slide 16 is the SANASH case:
"People strongly prefer less crowded buses but only a small percentage would
actually wait for the next bus." The same slide names the exact distinction this
paper has to make: "You found a difference between: what people prefer and what
they say they would actually do."

### R49. Do not graph everything. Every figure needs a job

> "A graph is not automatically useful. Sometimes you do not even need it at all"
> (`Analysis- slides.pdf`, slide 18, Step 11)

> "Before making a graph - chill down and ask yourself What should the reader
> notice? If you don't know... well don't make the graph!" (same)

> "Every single figure needs a job. ... Your graph/table should help the reader
> see A difference ... A trend ... A distribution ... A relationship ... An
> important pattern" (same, slide 19, Step 12)

**For SANASH.** The draft has no figure. One figure has an obvious job: a
grouped bar chart of scenario 2 against scenario 4 by occupation and by trip
frequency, which is the comparison the RQ turns on. An age histogram would have
no job.

### R50. Report qualitative answers by grouping them into themes and explaining the grouping

> "how you report quality data is you report it by writing themes. Okay. You
> just group those response into themes."
> (`2026-09-01_results-section-class.txt` [00:50:43])

> "in quality analysis, you do analysis by words. in quantity analysis, you do
> analysis by numbers." (same, [00:51:20])

> "Just keep them together, keep them apart." (same, [00:51:36]; the ASR is
> broken, and the reading that quantitative and qualitative results go in
> separate subsections is **INFERENCE** from the surrounding sentences)

**For SANASH.** Seven free-text scenario replies were recorded. Too few for
themes, and the draft correctly reports them as a count rather than analysing
them.

### R51. Results should be around 600 words

> "if the word limit is at least 600 words that's something you should be having
> for your result section"
> (`2026-09-01_results-section-class.txt` [00:00:11])

He justifies it with a claim about journals ("even the lowest q1. I'm sorry q4
or tire journals, they require you to have 600 words in your results section",
same [00:00:17] to [00:00:24]) which is unverified and is not a TR-C requirement
that could be found. Treat 600 as a course floor, not a journal rule. Note
separately that `Lit. review.pdf` slide 2 sets a 600-word minimum for the
literature review; the two floors are unrelated.

**For SANASH.** The current draft is 796 words and clears it.

---

## 7. Discussion

The course spends far less time here than on results. Everything below is what
the mentors actually said, which is not much.

### R52. The discussion is where meaning, cause and implications go, and only there

> "Then you have discussion section, right? It comes later after the result
> section. You can talk about anything you want. You can discuss why they
> replied that way. If you can, of course, back up with real argument"
> (`2026-09-01_results-section-class.txt` [00:01:23])

> "The discussion answers why do those findings matter?"
> (`video1768002125.txt` [12:51])

> "you're able to identify, you know, important trends and patterns and key
> findings. And this is when you explain them, you have to say why they
> significant, why they not significant" (`lesson1_part1.txt` [09:57])

**For SANASH.** The sentences the mentor struck out of the results are the
discussion's material: why frequent riders respond more, why students differ
from employed respondents, what a 26-point gap implies for an operator. His own
suggestion, offered mid-class, is a discussion-level hypothesis:

> "I do think that people that are more busy they would not opt into waiting
> because they work somewhere in like big companies"
> (`2026-09-01_results-section-class.txt` [00:23:04])

That is an interpretation of the student/employed gap the survey already found.
Whether wave 2 can test it is constrained by
`research/survey/wave2/DESIGN_NOTE.md`, which says the employed subgroup at the
planned quota supports the packed contrast but not the standing contrast, so
heterogeneity should be tested as an interaction rather than by splitting the
sample.

### R53. Even in the discussion, back the claim

> "If you can, of course, back up with real argument, you can always state
> anything you wish because the paper is creative."
> (`2026-09-01_results-section-class.txt` [00:01:31])

The permission is real but conditional. The condition is the argument.

---

## 8. Citations and referencing

### R54. Cite every quote, every paraphrase, every specific study, every statistic. Do not cite common knowledge

> "Cite every single time you... use a direct quote paraphrase someone's idea
> refer to a specific study use statistics or data"
> (`Lit. review.pdf`, slide 10)

> "No citation needed here --- 'Water freezes at 0 C.' (common knowledge)
> Citation needed 100% 'A 2024 study found AI use increased student
> productivity.'" (same slide)

> "every um like fact like um statements should be cited"
> (`video1433589796.txt` [33:11])

Mentor C's reason for it:

> "research is about something that is well analyzed that has supporting points
> that um every single statement that is on it is something credible"
> (`video1433589796.txt` [34:11])

**For SANASH.** This aligns with `CLAUDE.md` Rule 1, which is stricter: no paper
may be cited unless it is a row in `research/refs/base/references.md`, and no
finding may be stated unless it is recorded there or the paper was read this
session. The project rule wins where they differ.

### R55. Paraphrase far more than you quote, and a paraphrase is not synonym substitution

> "One misconception is that paraphrasing means changing a few words with
> synonyms. It doesn't." (`video1168216779.txt` [19:19])

> "A true paraphrase reorganizes the sentence while preserving the author's
> idea." (same, [19:23])

The memory test: "After reading the source, can I close the article and explain
the idea from memory?" (same, [20:46])

**For SANASH.** The memory test is dangerous here in isolation. `CLAUDE.md`
records that fourteen of sixty outreach citations written from memory were
wrong. The safe version of the test is: read the paper, paraphrase from memory,
then check the paraphrase against the paper before it is used.

### R56. IEEE numbering, by order of first appearance, never renumbered, never alphabetical

> "it uses numbers in closed and square brackets. ... If you refer to this first
> source again later in the paper, it remains one. The numbering never changes."
> (`video1168216779.txt` [21:36])

> "the reference list is organized according to the order in which sources first
> appear in the paper, not alphabetically." (same, [22:01])

> "at terra we use IEE citation style" (`video1433589796.txt` [20:01])

**For SANASH.** The course requires IEEE. The repo's `references.bib` is BibTeX
with `surname_year_word` keys. The coursework submission and any journal
submission will need different bibliography styles generated from the same
source file. Plan for that rather than maintaining two lists. The claim that
`Transportation Research Part C` uses a numbered style with its own formatting
is general knowledge about the journal, not something a mentor said, and was not
checked against the journal's guide for authors. **UNVERIFIED.**

### R57. Consistency between in-text citations and the reference list

> "One of the easiest ways to lose marks on the research paper is through
> inconsistent citations. Always check that every intact citation appears in the
> reference list and that every reference listed has been cited somewhere in the
> paper." (`video1168216779.txt` [22:14])

**For SANASH.** `research/refs/base/validate.py` already enforces the
`references.md` to `references.bib` direction. It does not check the paper
against the bibliography. That check is manual and belongs in the pre-submission
list.

---

## 9. Prose and style

### R58. No personal pronouns. Third person only

> "Do not in any way use personal pronouns. So no AI, no you, no we, no us, you
> should be using things like evidence suggests. You should be speaking in the
> third person because that is no academic and formal language."
> (`lesson1_part1.txt` [06:21])

"no AI" is the ASR rendering of "no I": the list is I, you, we, us. Do not read
this sentence as an AI-policy statement. "that is no academic" is also mangled.

**Conflict with the target journal, marked INFERENCE.** No mentor said anything
about journal conventions. That `Transportation Research Part C` papers
routinely use "we" is general knowledge, not checked here. The rule as stated is
a coursework rule. The reading followed in this file is that the coursework
submission follows it and any journal submission follows the journal, and that
the two drafts must not diverge in anything but pronouns.

### R59. Argument without bias, and the two are not the same thing

> "There is a difference between a biased paper and sharing your own opinions
> and arguments. You can share your own arguments without being biased."
> (`lesson1_part1.txt` [02:37])

> "don't try to bring your own political ideas. Don't be overly negative, overly
> positive." (same, [03:01])

### R60. Precision over exaggeration

See R23. The rule applies to the whole document, not only the purpose statement.

**For SANASH.** `CLAUDE.md` Rule 3 already forbids overselling in outreach ("An
email must not promise a Transportation Research Part C submission, a launch
across 25 cities, or any working device"). The same discipline applies to the
paper. The device does not exist yet (`GROUND_TRUTH.md` §6) and the paper must
not imply otherwise.

### R61. Evidence, not assertion

> "You cannot just write for the sake of writing. You have to be able to prove
> your thoughts and your arguments with evidence"
> (`lesson1_part1.txt` [02:13])

### R62. Write the paper in passes; a bad first draft is expected

> "The professional researchers plan. They outline, they organize, they rewrite.
> Sometimes they completely rewrite in introduction three or four times before
> they are satisfied. And that's perfectly normal."
> (`video1768002125.txt` [00:09])

---

## 10. Where the mentors disagree

Recorded rather than resolved.

**10.1 AI.** Mentor A in lesson 1: "All the tutors have different opinions on
using AI. I am very anti AI. I don't really like using in any capacity."
(`lesson1_part1.txt` [05:40]). In part 2 she attributes the opposite position to
a colleague: "Matthew has a very different stance on the use of AI compared to
me. He is much more pro AI than I am. I personally, I disagree with him."
(`lesson1_part2.txt` [00:04]).

But the same Mentor A, in the methodology lecture, walks the class through using
ChatGPT on the methodology section: "Once again, all the mentors have different
opinions on AI and a bit like if you I don't mind using AI, if you use it
correctly" (`video1101430871.txt` [29:09]; the ASR is garbled here and the
sentence cannot be reconstructed with confidence, but the four steps she then
describes are unambiguous, and `Methodology.pdf` slide 16 prints them). Her
boundary: "don't copy what it gives you words" (same, [30:06]).

**All sources agree on one thing:** "Do not use AI to write your work."
(`lesson1_part2.txt` [00:27]).

**Correction to the prior pass.** `course_notes.md` §16.1 frames this as the
deck contradicting the lesson-1 tutor. The transcript shows the lesson-1 tutor
delivering the pro-AI-for-structure guidance herself. The contradiction is
within one mentor, not between a mentor and a deck.

**10.2 Search result count.** 100 to 800 records (`Methodology.pdf` slide 9, and
attributed in the lecture to Matthew) against 100 to 400 (Mentor A,
`video1101430871.txt` [13:30]). She names the disagreement explicitly.

**10.3 Source recency.** "no later than 2020" (`lesson1_part2.txt` [03:58])
against "published within the last 10 years" (`Foundations class- group 3.pdf`,
homework Part 3).

**10.4 How many papers.** 15 (`Foundations class- group 3.pdf`, Part 3) against
"20 to 30 tends to be a good amount. 20 being kind of like the minimum"
(`video1101430871.txt` [43:42]). Mentor A reconciles these herself: 15 is
acceptable "but you'll have to explain very specifically why these 15 papers
were so relevant" (same, [43:58]).

**10.5 Causal language in results.** Within the 2026-09-01 class, "you cannot be
telling any single cause in the result section" [00:01:13] against "you can even
do some associations, you can talk about the cause, but just be careful"
[00:38:57]. Same speaker, same lecture.
---

## 11. What the prior passes got right, missed, and got wrong

### 11.1 `course_notes.md` (2026-08-20, ~11.8k words)

**Right.** It is accurate and quote-faithful across the six earlier recordings
and the six decks. Its §16 correctly flags the AI contradiction, the ethics
truncation, the recency conflict and the missing deadlines. Its §13 dissection
of `Mohlaroy--RAS.docx` as the model paper was not re-derived here and is taken
on trust. Its §2 RQ framework, §6 literature-review rules, §8 five-component
methodology and §11 introduction structure are correct and this file does not
improve on them, only re-cites them.

**Missed.** It predates the 2026-09-01 results-section class by twelve days and
therefore contains **no results-section rules at all**: nothing on the four-part
results structure, the two jobs of a results sentence, the banned interpretation
verbs, percentage points against percent change, or ordering by RQ importance.
Those are §6 of this file. It also never saw `Analysis- slides.pdf`, so it has
no data-cleaning, cross-tabulation or figure rules.

**Wrong.** §16.1 attributes the pro-AI position to "the Methodology deck" and
the anti-AI position to "the lesson-1 tutor", implying two people. They are one
person. §4.2 reports the 100 to 800 calibration as the course rule and does not
record that Mentor A disagrees with it on the record. §9.10 says the deck's
treatment of ethics is missing but does not notice that the deck's slide 15,
headed "Component 5", covers question types rather than data collection, so two
components are untaught rather than one (R26).

**Attribution to check.** §2.6 gives a narrowing-axes table attributed to "the
deck". No such table is in the extracted text of `Foundations class- group 3.pdf`
(R3).

### 11.2 `RESEARCH_METHOD_GROUNDING.md` (2026-08-26, 29 KB)

**Right.** Its source-inventory table with per-source limitations is the correct
form for this kind of document and this file has copied the habit. It correctly
identifies the association-versus-causation rule as load-bearing for RTCI, and
its "Open questions for Diyas" list is still live: questions 3, 4, 6, 9 and 10
are unanswered as of today. Question 10, on ethics and consent for a survey
involving possible minors, is the same hole this file records at R26.

**Missed.** It reconstructed `Analysis- slides.pdf` from 60-second video frames
and lost **Step 7, cross-tabulation** entirely (this file's R46). It also could
not see slides 1 to 7. The deck is now readable at the repo root, so that
limitation is closed.

**Superseded, not wrong.** Its note on the Almaty bus RQ says the RQ "is a
generic teaching example constructed by the deck's author for illustration" and
"must not be treated as such". The 2026-09-01 transcript contradicts that: the
mentor says "It's yours... I stole it from you" ([00:04:11], [00:04:13]). But
the prior file did not assert this flatly. It ends the same paragraph with "It
is plausible this specific worked example was custom-built for this student's
course submission (unclear from slide content alone...) - flagged as inferred,
not confirmed", its evidence table marks the row "inferred - no corroboration
found either way", and its open question 8 asks the question directly. Reading
the first half of that paragraph without the second half misrepresents the file.
The correct description is that a flagged inference has now been resolved
against, not that a claim was wrong.

Second, it states that the course "does not give sufficient causal
field-experiment methodology". True and important. It then treats the
survey-methodology template as adaptable. The 2026-09-01 class shows the survey
template is the right one *for the survey study*, which is a more useful
conclusion than adaptation. Note that the charter says the same thing at §5:
"survey template нельзя копировать как весь methods section будущей статьи".

### 11.3 `README.md`

**Right.** The provenance note on transcription quality, and the explicit
warning to "verify against the audio before quoting anything consequential", are
both correct and have been honoured here by marking mangled quotes. Its "Known
gaps" list is accurate as far as it goes.

**Missed.** Its file table does not list the 2026-09-01 class or
`Analysis- slides.pdf`, and it says the transcripts directory holds "all six
recordings" when there are now seven. Its gap 1 repeats the count error in
`course_notes.md` §9.10 (R26).

### 11.4 `results_section_draft.md`

**Right.** It follows R39's four-part skeleton exactly. It reports numbers with
every finding (R42). It labels differences as percentage points, not percentages
(R43). It contains none of the banned verbs (R38). It reports the usable-response
funnel (R47). It states that gender was not asked rather than silently omitting
it (R28). It ends by tying the comparison back to the RQ (R42, job two).

**Missed or weak.**

1. It never states that the six scenarios were hypothetical. It says respondents
   "saw six scenarios", which does not settle it. A reader of §5.2 alone could
   take "77.5% chose to wait" as observed behaviour. R30 and R48 both require
   this to be explicit, and `research/survey/README.md` already words it
   correctly ("Stated preference measures intention, not behaviour").
2. Its stated RQ is causal ("does live bus occupancy information change the
   boarding decisions of Almaty commuters?") while its data is stated intention.
   By R1 and the mentor's FLEX anecdote ([00:02:22] to [00:03:34]), where a
   student's results did not match the scope of her RQ, this is the defect he
   warned about.
3. No figure (R49). The scenario 2 against scenario 4 comparison by subgroup has
   an obvious job.
4. §5.4 "What moves together" opens by relating stated tolerance for waiting to
   scenario choices. Both measure the same underlying construct, so that
   association is close to tautological and will not survive the "so what" test
   (R15). Its second paragraph is stronger but repeats the frequency comparison
   already given in §5.3. A relationship carrying new information, for example
   crowding exposure or trip purpose against scenario choice, would serve the
   part better. Marked **INFERENCE**: no mentor commented on this draft.
5. The number stated aloud in class was 225 respondents
   (`2026-09-01_results-section-class.txt` [00:07:52]); the verified figure in
   `GROUND_TRUTH.md`, `research/survey/README.md` and in the draft is 215.
   Separately, the 214 that appears at [00:12:58] and [00:17:42] is the mentor's
   own invented worked example, not a claim about this survey. Only the 225 is
   unexplained. It is worth being sure the mentors have the right number before
   they read the paper.

---

## 12. Conclusions for us

Ranked by how much they change what we write. Each carries its reasoning chain
so it can be checked, and each says whether it is a mentor's position or an
extrapolation.

**Scope note that applies to all ten.** There are two deliverables, not one: the
Terra coursework submission, and the eventual `Transportation Research Part C`
paper the charter targets. The course rules bind the first absolutely. They bind
the second only where they coincide with journal practice. Conclusions below say
which deliverable they are about.

### C1. No sentence written today may assert a causal effect of RTCI on boarding, and the current draft is a stated-preference paper

**Reasoning.** (a) The mentors forbid causal language on survey data: "your
survey cannot prove any causation right it truly can't"
(`2026-09-01_results-section-class.txt` [00:37:14]), reinforced by
`Analysis- slides.pdf` slide 15 and by `Methodology for emp. papers.pdf` slide 8
on self-report. (b) `GROUND_TRUTH.md` §3.1 records that the only collected data
is a stated-preference survey and explicitly labels the 7.96-minute
willingness-to-wait estimate "stated intention, а не causal effect и не
поведение в поле". (c) The field experiment is blocked on Innoforce decisions
that have not been made (`RTCI_RESEARCH_CHARTER.md` §6, §12). (d) Therefore no
sentence in the current paper can assert a causal effect of RTCI on boarding.

**What this is not.** It is not a discovery about the charter. The charter
already states that a survey-only paper "does not meet the target bar" and that
the survey "remains instrument-development and prior/calibration stage" (§1,
Part C publication threshold), and §5 already restricts results to "associated
with" unless identification is randomized or credibly quasi-experimental. The
draft version of this conclusion read as though the charter were pointing the
project at an unwritable paper. It is not. The TR-C thesis describes study 3 and
4 of a four-study programme, and writing toward it is a plan, not an error.

**What changes.** For the coursework submission: the RQ, purpose statement and
limitations must describe a stated-preference study, and the results section
must stay in R41's hedged language. For the journal paper: nothing yet.

**Conflict with R5, named not resolved.** R5 records that a mentor saw the causal
RQ, printed it on a slide, and said "It's yours". Reframing the RQ as a
stated-preference question changes an RQ the mentors have accepted. That is a
cost, not a blocker, and it should be paid openly by telling the mentor why, not
by quietly editing the header of the results draft. Which RQ wins is Diyas's
decision, not this file's.

**Status.** The language rule is the mentors' and the charter's. The
recommendation to reframe the coursework RQ is this file's, and it collides with
R5.

### C2. The research gap sentence cannot be written from the current evidence, and reading four named papers is the floor rather than the finish

**Reasoning.** (a) "a research gap must be supported with evidence. You cannot
simply write, there's little research blank" (`video1168216779.txt` [08:03]).
(b) The charter's gap sentence rests on three OpenAlex queries, and the charter
itself calls the gap candidate rather than proven. (c) `REPLICATION_TARGET.md`
§1 says the exploratory search makes the gap more plausible but does not prove
it, and §5 says the file does not describe the Stockholm pilot's design because
the paper has not been read. (d) `CLAUDE.md` Rule 1 forbids stating a finding
from an unread paper. (e) Therefore the gap paragraph is currently unwritable.

**What changes.** Reading `zhang_2016_stockholm`, `drabicki_2023_willingness`,
`drabicki_2025_covid` and `prabhakar_2024_skipping` is not background work, it
is the blocking task for the introduction. Each read fills one row of
`evidence_matrix_template.csv` and one clause of the gap sentence.

**Correction to the draft version of this conclusion.** Four reads are a
necessary condition, not a sufficient one. `research/lit_review/search_log.md`
says it is not a protocol review and gives no PRISMA numbers;
`REPLICATION_TARGET.md` §5 says exploratory discovery does not replace the
systematic review; and R12 plus C5 set a much higher reading floor for the
literature review the gap sentence has to sit inside. Four reads unblock a
first draft of the gap paragraph. They do not close the gap claim.

**Status.** Mentor's rule, applied to facts already recorded in the repo.

### C3. The population claim must shrink to what was actually sampled

**Reasoning.** (a) "Be honest about scope. If you only surveyed your own school,
your population is not 'students worldwide.' Narrowing this correctly makes your
paper stronger, not weaker." (`Methodology for emp. papers.pdf`, slide 12).
(b) The sample is 77.7% aged 14-24 and 73.5% students (`GROUND_TRUTH.md` §3.1),
and 79.1% ride daily or almost daily (`results_section_draft.md` §5.1).
(c) "Almaty bus commuters" therefore misdescribes the population by the deck's
own test.

**Already done in the repo, not yet in the paper.** `research/survey/README.md`
states under Known limitations that "the sample describes young frequent riders,
not Almaty commuters generally"; `GROUND_TRUTH.md` §3.1 records the same as an
open item; `research/survey/wave2/DESIGN_NOTE.md` builds the wave 2 recruitment
quota around it. The only place "Almaty commuters" still stands unqualified is
the RQ line at the head of `results_section_draft.md`, and the mentor's slide.
This conclusion is therefore a text fix, not a new finding.

**What changes.** The population statement in the paper becomes something like
"young, predominantly student, frequent bus users in Almaty".

**Downgraded clause.** The draft version added that this "makes the
student/employed gap in §5.3 a limitation rather than a finding, since the
employed subgroup is n = 44". That does not follow from the deck's scope rule,
which is about who conclusions generalise to, not about subgroup precision. The
better-grounded version of the same worry is in
`research/survey/wave2/DESIGN_NOTE.md`: at the planned wave 2 size the employed
subgroup supports the packed contrast but not the standing contrast, so
heterogeneity should be tested as an interaction in the pooled model rather than
by splitting the sample. Whether the wave 1 split at n = 44 should be reported
as a finding, a limitation, or an interaction is an open analysis decision.
**INFERENCE.**

**Status.** Mentor's rule applied to verified repo facts, with one clause
downgraded.

### C4. Ethics and consent must be written from outside the course, and the possible-minors issue is the reason

**Reasoning.** (a) Ethics is component 6 of 6 and "Miss one and we will notice
surely" (`Methodology for emp. papers.pdf`, slide 10). (b) The deck's text ends
at a slide headed "Component 5" that covers question types, so both data
collection and ethics go untaught, and no recording covers either. (c) The
survey's age band starts at 14, so minors may have participated
(`GROUND_TRUTH.md` §3.1, `research/survey/README.md`). (d) Therefore the paper
has a required section with no course template and a real substantive problem
inside it.

**What changes.** Source a consent and data-handling template from the target
journal's requirements or an institutional policy, not from the course. Decide
and record how minors' responses were handled. Wave 2 already screens at age 18
(`research/survey/wave2/DESIGN_NOTE.md`), which fixes the future but not the
collected data. This must also be settled before any field study touches real
passengers.

**Already open in the repo.** The charter carries it as an unresolved item at §5
("ethics process остаются открыты") and §12 item 7 ("Кто даёт разрешение на
съёмку, consent и обработку данных?"), and
`RESEARCH_METHOD_GROUNDING.md` raises it as open question 10. This conclusion
adds urgency and a source, not the problem.

**Status.** Requirement and gap documented. The urgency is this file's reading.

### C5. The literature review is blocked on reading roughly twenty papers, and the current reference base cannot substitute

**Reasoning.** (a) Synthesis needs at least three sources per point
(`video1433589796.txt` [11:36]) across three or more themes (`Lit. review.pdf`
slide 4). (b) Mentor A's own figure is "20 to 30 ... 20 being kind of like the
minimum" (`video1101430871.txt` [43:42]), against the homework brief's 15
(`Foundations class- group 3.pdf`, Part 3); see §10.4. (c) The repo has 153
verified bibliographic rows and, per `GROUND_TRUTH.md` §9, the newest seven are
all `LISTED` with none read, and no row's content beyond title, venue and
authorship was ever checked. (d) A row in `references.md` proves a paper exists;
it does not supply a finding.

**What changes.** Budget the reading explicitly. Fifteen to twenty full reads is
the real cost of the literature review, and it is the largest single unbudgeted
task in the paper.

**Status.** Mentor's numbers applied to repo facts.

### C6. The results section is close to correct and needs four specific fixes, not a rewrite

**Reasoning.** Set out in §11.4. The draft satisfies R38, R39, R42, R43 and R47.
The fixes are: say the scenarios were hypothetical; open with the headline
number; add one figure with a job; give §5.4 a relationship that carries
information the reader does not already have from §5.3.

**What changes.** Four edits, not a new draft. This is the cheapest high-value
work available.

**Status.** Fixes 1 and 2 follow directly from mentor rules. Fixes 3 and 4 are
this file's judgement, marked INFERENCE in §11.4.

### C7. "Real-time" is currently the wrong word and the paper should not use it until the product decision is made

**Reasoning.** (a) `GROUND_TRUTH.md` §3.1, open item raised 2026-09-03: showing
the last measured value is "recent crowding information, а не real-time", and
whether to show a measurement or a prediction is undecided. (b) The purpose
statement must be unambiguous and free of overstatement
(`video1168216779.txt` [12:17], [13:02]). (c) The RQ as printed on the mentor's
slide says "real-time". (d) Therefore the paper either defines the term
operationally (measurement age in seconds at the moment of display) or uses a
weaker word.

**What changes.** One definition sentence in the methodology, or a change of
term throughout. Cheap now, expensive after review.

**Note.** This applies to the field study's terminology. It does not touch the
stated-preference survey, whose scenarios described crowding in words and never
told respondents how fresh the information was.

**Status.** The product ambiguity is in `GROUND_TRUTH.md`. The conclusion that
it propagates into the paper's terminology is this file's. **INFERENCE.**

### C8. Agent-drafted prose is a structure aid, not submittable text

**Reasoning.** (a) Every source in the course agrees on one line: "Do not use AI
to write your work" (`lesson1_part2.txt` [00:27]); Mentor A's boundary in the
permissive case is "don't copy what it gives you words"
(`video1101430871.txt` [30:06]). (b) The drafts in this repo, including
`results_section_draft.md`, are agent-produced. (c) Therefore they function as
outlines and fact-checked scaffolding, and the submitted text has to be Diyas's
rewriting of them.

**What changes.** Nothing about how the drafts are produced. Everything about
the last step before submission.

**Status.** Mentors' unanimous position, applied to a fact about this repo.

### C9. The two audiences need one source and two renderings, not two drafts

**Reasoning.** (a) The course requires no personal pronouns (R58), IEEE
numbering (R56), 600-word results (R51) and a 10-15 word RQ (R2). (b) The claim
that TR-C requires none of these and permits "we" is general knowledge about the
journal, not something a mentor said and not checked against the journal's guide
for authors. (c) The charter already recognises the RQ-length case and handles
it by keeping two formulations. (d) The risk is that the coursework version and
the journal version drift apart in substance while nominally differing only in
style.

**What changes.** Keep one set of facts, numbers and citations. Let the surface
conventions differ. Never let a number differ.

**Status.** The course rules are quoted. Point (b) is **UNVERIFIED** and should
be checked against the journal's guide for authors before it is relied on.

### C10. Nothing about the CV device can be reported as a result in this paper

**Reasoning.** (a) `GROUND_TRUTH.md` §6: the ceiling RGB system is not
implemented, no own camera frames exist, and the one smoke test produced a model
that lost to a constant predictor. (b) The mentors demand real evidence
(`video1433589796.txt` [42:52]) and forbid causal or interpretive overreach.
(c) `CLAUDE.md` Hard rule 3 forbids fabricating metrics or implementation
status. (d) Therefore the measurement layer appears in this paper only as design
and motivation, never as validated performance.

**What changes.** The charter's "measurement contribution" is a future claim.
Any sentence implying a working sensor must be cut.

**Status.** Repo facts plus project rules. Not an extrapolation.

---

## 13. Things that could not be verified

1. **`Research - intro.pdf` is unreadable.** No text layer beyond the title, no
   OCR available. Its content (Zotero, bibliography tag, credible against
   reliable, the peer-review/author/content signals) reaches this file only
   through `course_notes.md` §§4.6-4.9, which claims to have read it "page by
   page". Nothing is quoted from it and no rule above depends on it alone.
2. **`video1311882563.mp4` has no transcript.** The Analysis deck's text is now
   fully readable, but the presenter's spoken commentary on those slides remains
   inaccessible. Rules R44, R46, R47, R48 and R49 rest on slide text only.
3. **Speaker identity in the 2026-09-01 class** is inferred from style. The
   speaker never names himself, and never names the student whose RQ is on the
   slide (R5).
4. **Mentor names** are unreliable throughout: the ASR produces "Mathemodes",
   "Mahmoud", "Matthew", "Max", "Adran", "Ajahn", "Ia john", "Camilla",
   "Gulshand", "Grushano". They have not been resolved and no rule depends on a
   name.
5. **The garbled AI sentence** at `video1101430871.txt` [29:09] cannot be
   reconstructed with confidence. The four steps that follow it, and
   `Methodology.pdf` slide 16, carry the substance.
6. **The 225 against 215 respondent count** discrepancy (§11.4, item 5) is
   unresolved.
7. **The mentor's claim that Q4 journals require 600-word results sections**
   ([00:00:17]) is unverified and should not be repeated.
8. **`Mohlaroy--RAS.docx`** was not opened this pass. Everything about it here
   comes from `course_notes.md` §13 and from Mentor D's live commentary in
   `video1433589796.txt`. The §3.7 section number in R36 is second-hand.
9. **The narrowing-axes table** (time, place, population, platform, variable) in
   R3 is not in the extracted text of any deck. Source unconfirmed.
10. **`Transportation Research Part C` conventions** (pronouns, citation style,
    RQ length, results length) are asserted from general knowledge in R56, R58
    and C9 and were not checked against the journal's guide for authors.
11. **The stated-preference against revealed-preference divergence** used as the
    worked example in R14 is not supported by any paper read in this repo.

---

## 14. Changelog for the 2026-09-04 audit

**Quote verification.** 118 blockquote blocks were checked segment by segment
against normalised source text, plus roughly fifteen inline quoted strings
checked by hand. Two failed on text and were corrected. Five transcript
timestamps were flagged; one was a substantive error, three were within one
transcript segment of the quoted text's true start and were corrected silently,
one was a parsing artefact.

**Corrected**

- R18: timestamp range changed from [09:03]-[09:31] to [10:16]-[10:34]. The
  quoted passage is real and verbatim; the cited range pointed at a different
  passage 73 seconds earlier, and [09:31] is not a segment boundary in the file.
- R43: the second quote silently dropped an intervening "Okay." from the
  transcript. Replaced with an ellipsis and the timestamp range extended.
- R42: "be a mathematician, right?" was cited at [00:40:54]; the actual segment
  is [00:40:59], and "You cannot prove anything without numbers" is [00:41:03],
  not part of the same segment. Split into two citations.
- R8, R15, R34: timestamps moved back one segment to where the quoted text
  actually begins ([23:10], [25:10], [19:08]).
- R16: the model paper's evidence sentence was quoted as "between 60-100% of the
  population failed to understand how RAS functions". The transcript reads
  "between 60 to 100 percent ... how ras functions". Replaced with the source
  wording and cited.
- R5: "It's yours" cited at [00:04:04]; it is at [00:04:11]. Split into three
  citations.
- R21, C2: `REPLICATION_TARGET.md` §5 was cited for the "does not prove" phrase.
  That phrase is in §1. §5 is cited only for what it does say.
- R51: word count of the results draft corrected from "roughly 700" to 796.

**Downgraded**

- §0 mentor identification: "Both recordings also refer to a more AI-permissive
  colleague named Matthew" was too strong. Only `lesson1_part2` attributes an AI
  position to him; the `video1101430871` mention is about search counts and the
  `lesson1_part1` mention is a mangled name in an unrelated sentence.
- R3: the narrowing-axes table is not in the Foundations deck's text layer.
  Restated as second-hand from `course_notes.md` §2.6.
- R7: the quoted sentence does not mention a conclusion. The Foundations process
  diagram is now cited for that element.
- R11, R14: the candidate themes and the stated-versus-revealed-preference
  divergence are built from titles, not read papers. Marked INFERENCE, since
  `CLAUDE.md` Rule 1 forbids stating a finding from an unread paper.
- R36: the "§3.7" section number for the model paper's limitations is
  second-hand from `course_notes.md` §13 and is now labelled as such.
- R37, R40, R45, R50: inferences that were being carried with the force of a
  mentor's statement are now marked INFERENCE. R37's reconciliation of the two
  contradictory instructions is one of these; §10.5 keeps the disagreement
  unresolved, as it should.
- R56, R58, C9: every claim about `Transportation Research Part C` conventions
  is now marked unverified and listed in §13.
- §11.2: the criticism of `RESEARCH_METHOD_GROUNDING.md` was based on half a
  paragraph. The file flagged its own claim as inferred, named the alternative
  as plausible, and raised it as open question 8. Recast from "wrong" to
  "superseded".
- §11.4 item 4: the claim that §5.4 is tautological applies to its first
  paragraph only. The second paragraph is redundant with §5.3 rather than
  tautological.
- C1: the framing that the charter's TR-C thesis "describes a paper that cannot
  be submitted" implied the charter was in error. The charter says the same
  thing itself, twice. Recast around the language rule, which is what the
  evidence supports, and the collision with R5 is now named.
- C2: "cannot be written until four specific papers are read" implied that four
  reads finish the job. They are the floor.
- C3: the clause turning the student/employed gap into a limitation on account
  of n = 44 does not follow from the cited slide. Replaced with the better
  grounded concern from `research/survey/wave2/DESIGN_NOTE.md` and marked
  INFERENCE.

**Corrected against the repository**

- R24: the claim that the survey methodology "exists as data and none of it
  exists as methodology text" is false. `research/survey/README.md` already
  carries the scenarios, attribute levels, languages, window, sampling statement,
  estimator and exclusion counts in prose.
- R28: the claim that the sample is not called a convenience sample anywhere is
  false; `research/survey/README.md` says so explicitly. Narrowed to the
  recruitment channel, which genuinely is unrecorded.
- R26 and C4: the course gap is larger than recorded. The deck's final slide is
  headed "Component 5" but covers question types, so data collection is untaught
  as well as ethics. Added the charter's own open items on ethics.
- R32: `research/lit_review/search_log.md` states that it is not a protocol
  review and yields no PRISMA numbers. Qualified.
- R35: `evidence_matrix_template.csv` confirmed as a single header row.
- R6: paths added for `TRC_PAPER_BLUEPRINT.md`, `PAPER_SPRINT_30D.md` and the
  introduction draft, all of which resolve under `research/`.
- Header: the claim to supersede three repository files was removed. A scratchpad
  file cannot supersede a repository file, and `CLAUDE.md` hard rule 7 requires a
  supersession to be written into `GROUND_TRUTH.md` in the same change.

**Added**

- R58 now flags "no AI" as the ASR rendering of "no I". Left unflagged, the
  quote reads as an AI-policy instruction, which is the opposite of what it is
  and is a live confusion in a document that also records an AI disagreement.
- R51 notes the separate 600-word floor for the literature review on
  `Lit. review.pdf` slide 2, so the two are not conflated.
- §11.4 item 5 now separates the mentor's invented 214 from the student's spoken
  225.
- §12 opens with a scope note distinguishing the coursework submission from the
  journal paper. Several conclusions in the draft moved between the two without
  saying so.
- §13 gained four items: the narrowing-axes source, the TR-C conventions, the
  R14 divergence claim, and the unnamed student in R5.

**Kept unchanged**

R9, R12, R13, R19, R20, R22, R25, R27, R33, R38, R39, R46, R49, R53, R57, R59,
R60, R61, R62, C8, C10 and §10 in full.

**Counts.** 62 rules in, 62 out: 19 untouched, 43 edited, none cut. 10
conclusions in, 10 out: 2 untouched, 8 edited, none cut. The mentor
disagreements in §10 and the within-mentor tensions in R37 and R40 are recorded
and not resolved.
