# Terra research course — full notes

Compiled 2026-08-20 from every lesson recording and presentation in
`research/coursework/`. Source material:

| Source | Type | Length |
|---|---|---|
| `Foundations class- group 3.pdf` | slides, 20 pp | Lesson 1 deck |
| `1st lesson- part 1.mp4` | recording | 34:56 |
| `1st lesson- part 2.mp4` | recording | 19:00 |
| `Research - intro.pdf` | slides, 20 pp | Sources, Zotero, bibliography tag |
| `2nd class-Reading.pdf` | slides, 19 pp | Reading academic literature |
| `Lit. review.pdf` | slides, 15 pp | Literature review drafting |
| `Methodology.pdf` | slides, 17 pp | Methodology, lit-review papers |
| `Methodology for emp. papers.pdf` | slides, 15 pp | Methodology, empirical papers |
| `video1168216779.mp4` | recording | 26:16 |
| `video1768002125.mp4` | recording | 24:56 |
| `video1433589796.mp4` | recording | 53:55 |
| `video1101430871.mp4` | recording | 46:02 |
| `Mohlaroy--RAS.docx` | student paper | worked example, scoping review |

Transcription: faster-whisper `small`, int8 CPU, English detected at p=0.99.
Slide text extracted with PyMuPDF; image-only decks read page by page.

Quotes below are from the recordings; slide content is marked as such. Where
the tutor said something that contradicts or extends a slide, the recording
is treated as authoritative and the difference is noted.

---

## 0. The course in one page

The deliverable is a publishable research paper. Two routes:

- **Literature review paper** — analyzes existing research, no data collection
- **Empirical paper** — collects and analyzes original data (survey, interview,
  experiment, observation)

Both require a literature review. Only the empirical route adds a data-collection
methodology, results, and analysis of your own numbers. The pipeline is the same
either way:

```
Question -> Introduction -> Literature Review -> Methodology
  -> Results -> Analysis (discussion) -> Conclusion -> Publication (optional)
```

Methodology is the section that differs most between the two routes, and the
course teaches it as two separate lessons for exactly that reason.

---

## 1. Foundations: what research is, and what it is not

### 1.1 Definition

Academic research is a systematic process of generating new knowledge, answering
questions, or investigating problems using evidence and critical thinking. It
should not be biased or shaped by your own mood and experiences.

Key characteristics: **systematic, evidence-based, objective, transparent,
reproducible.**

On reproducibility, the tutor narrowed the scope compared to the slide:

> "By reproducible, we mean specifically those who are conducting empirical
> papers. If someone [read] or studied, they should be able to find the same
> exact results that you had got by using your same methodology."

### 1.2 What research is NOT

| NOT research | Research |
|---|---|
| Copying information | Analyzing information |
| Using random websites | Using credible evidence |
| Repeating opinions | Building arguments |
| ChatGPT summaries only | Independent investigation |
| "I think…" | "Evidence suggests…" |

Points the tutor expanded on beyond the slide:

- **Paraphrase, never copy.** "If Dr. A has said we found that there is a 10%
  increase in the production of steel, you can't copy that word for word."
- **No Wikipedia, no sketchy sites.**
- **AI policy is tutor-dependent and this tutor is strict.** "All the tutors
  have different opinions on using AI. I am very anti AI… as it stands, you
  shouldn't be really using AI at any point with independent investigation.
  Everything should be in your own words." Note this conflicts with the
  Methodology deck (§8.11), which explicitly walks through using ChatGPT to
  study methodology structure. If in doubt, ask your own mentor.
- **No personal pronouns.** "No I, no you, no we, no us — you should be using
  things like *evidence suggests*. You should be speaking in the third person."

On bias, the distinction the tutor drew is worth keeping, because it is the one
students get wrong:

> "There is a difference between a biased paper and sharing your own opinions
> and arguments. You can share your own arguments without being biased… don't
> try to bring your own political ideas. Don't be overly negative, overly
> positive."

And research is not just fact-collection — each paper has a position:

> "Every paper has its own opinion. It's not just about the facts that they use.
> What you'll be doing is understanding what their arguments are, what is their
> voice, what is the main message."

---

## 2. The research question

### 2.1 Topic vs research question

A **topic** is a broad subject area: climate change, social media, artificial
intelligence, education inequality. A **research question** is the single
specific question you investigate for the whole paper.

### 2.2 The formula

```
How/Why does [factor] affect [outcome] among [group/context]?
```

Slide examples:

1. How does sleep deprivation affect academic performance among high school students?
2. Why do young voters trust social media news sources?
3. How does AI-assisted learning influence student productivity?

Worked transformation from the deck:

- Weak: *"How does social media affect people?"* — too broad, impossible to
  answer. The tutor's gloss: "Social media affects people in 10000000+ different
  ways… you cannot realistically study the impact of social media on every
  single person."
- Strong: *"How does TikTok political content influence political polarization
  among teenagers in the U.S.?"* — specific platform, specific effect, specific
  population.

### 2.3 Required characteristics

| Characteristic | Meaning |
|---|---|
| Focused | Not too broad |
| Clear | Easy to understand |
| Researchable | Evidence can be collected |
| Specific | Clearly defined variables/groups |
| Significant | Actually matters |

Length: **10–15 words.** The tutor flagged the common failure directly: "We have
had students in the past who have presented us with research questions which are
like three sentences long. That is just not okay."

### 2.4 Weak vs strong, side by side

| Weak question | Why weak |
|---|---|
| Is social media bad? | Opinion-based |
| What is climate change? | Too descriptive |
| Why is education important? | Too broad |

| Strong question | Why strong |
|---|---|
| How does Instagram usage affect self-esteem among teenagers? | Focused + measurable |
| How do urban heatwaves impact public health in Tashkent? | Specific context |
| How does AI use influence student writing habits? | Clear variables |

### 2.5 Red flags

A question is dangerous if it is:

1. too broad
2. impossible to measure
3. purely opinion-based
4. emotionally loaded
5. requires inaccessible data

The slide's example of loaded wording: *"Why is capitalism evil?"* — "It has
really biased wording and is not academically neutral."

On (5), the constraint is practical: "We don't have access to NASA labs. You have
to be able to access your evidence through the internet."

### 2.6 Scope control

The deck's framing: "Some of you are trying to research EVERYTHING. I had one
student ask me if that is a good RQ: *The effects of technology on society*.
Impossible to execute."

Narrow along these axes:

| Narrow by | Example |
|---|---|
| Time | 2020–2025 |
| Place | Uzbekistan |
| Population | Teenagers |
| Platform | TikTok |
| Variable | Political polarization |

Realism warning from the slide: "Do not try to solve climate change or interview
Putin and Trump. You are high school students so choose your RQ wisely."

### 2.7 Novelty and the research gap

A **research gap** is missing research in a field — either because it has not
been done in a specific country, or because it has not been examined enough.
Shape your paper around that gap.

Confirmed in the Q&A. A student asked whether the target is an un-researched
combination of variables:

> "Yes, perfectly spot on… you want to be identifying something that is very
> interesting, something that people will want to read, but also something that
> has a research gap in it."

The tutor also gave the defensive reason to read first — avoiding a collision
with existing work: "Before you even finalize a research question… look at all
the publications that have already been produced regarding your topic."

---

## 3. Literature review paper vs empirical paper

| Literature Review | Empirical Paper |
|---|---|
| Uses existing studies | Collects original data |
| No participants | Participants required |
| Sources are journal articles | Sources are survey/interview responses |
| Synthesizes previous findings | Produces new findings |
| Methodology is short | Methodology is detailed |
| No ethics approval usually needed | Consent and ethics matter |

Three points from the Q&A that the slides do not carry:

1. **Every paper needs a literature review**, empirical included. A student
   confirmed this and the tutor agreed: "All the papers, no matter what you're
   studying, will have a literature review in which you have to analyze existing
   research."
2. **Do not mix the two routes.** "Typically no, we recommend that you use one or
   the other… if you happen to have a very unique case where you can combine
   elements, definitely talk to the mentors first."
3. **Empirical takes longer**, but neither is more important. "With an empirical
   paper it will take you a bit longer because you are conducting your own study…
   regardless of which one you choose, they are both equally important."

The tutor's own paper was a literature review on the Universal Declaration of
Human Rights, comparing four scenarios and assessing the UDHR's successes and
limitations. Previous-cohort empirical papers: effects of reading on dementia;
effects of confidence training on young teenagers.

Two further Q&A points on choosing between them:

- **Empirical papers are not tied to your own country.** "Students find it
  easier to do that… however you shouldn't be limited to that. If you find a way
  to conduct research in the US or research in Australia, go ahead."
- **Some topics cannot be empirical.** "For humanities topics it's not always
  right to do an empirical paper. We have had people who have looked at US
  history or political science topics, and for them it didn't make sense —
  you can't really create an empirical paper on history." Professors do tend to
  prefer empirical work, but fit to topic wins.

---

## 4. Finding sources

### 4.1 Where to search

| Database | Coverage | Notes |
|---|---|---|
| Google Scholar | Everything, all subjects | Broadest and easiest; includes low-quality material, so filter hard |
| JSTOR | Humanities and social sciences | Journals and academic articles only, no newspapers or web articles |
| PubMed | Medicine, biology, chemistry | Free |
| ScienceDirect | All sciences and maths | |
| ERIC | Education research | Free |
| Scopus / Web of Science | Large multidisciplinary indexes | Usually behind institutional paywalls |
| IEEE Xplore | Engineering and computer science | |
| DOAJ | Directory of Open Access Journals | Entirely free full text |

Plain Google is a fallback, with a caveat: "In comparison to Google Scholar,
regular Google will not filter out false, uncredible papers. It will just give
you everything." A heuristic passed on from another mentor: prefer `.org`
domains, which tend to come from respectable organizations.

The methodology deck adds two things you must state alongside the database
names: **why you chose them** ("ERIC was selected because the research question
concerns classroom learning outcomes") and **any access limits you hit** (if you
could only use freely accessible full text, say so).

### 4.2 Search strategy

Operators, from the deck:

- **AND** narrows — requires both terms
- **OR** widens — catches synonyms
- **Quotation marks** lock an exact phrase
- **Asterisk** catches variants: `educat*` finds education, educational, educator

Example of a full search string, which the deck notes almost no student paper
includes:

```
("artificial intelligence" OR "machine learning") AND (education OR "student
learning") AND ("academic performance" OR outcomes)
```

Building the keyword list properly (Methodology deck):

1. Break the research question into concepts — usually three: topic, population, outcome
2. For each concept list every synonym the literature might use. AI might appear
   as "artificial intelligence", "machine learning", "generative AI", "large
   language models", or "ChatGPT"
3. Join synonyms within a concept using OR; join concepts using AND
4. Mine papers you already have — check their keyword lists and titles
5. Check for controlled vocabulary — PubMed uses MeSH terms, ERIC uses
   descriptors. These catch papers that phrase things differently from you

**Calibration signal:** a good search returns roughly **100 to 800 records**.
Twelve results means your terms are too narrow; nine thousand means too broad.

### 4.3 Date filtering

Sources should be **2020 or later**. The tutor's tolerance: "My rule is that you
can have one or two sources that might be out of date, but one or two max."

The rationale is that newer sources reflect current trends, and it matters more
for fast-moving topics like technology or AI.

### 4.4 Credible vs reliable

The tutor drew a distinction the slides do not:

> "Credible basically means it comes from a verified source — an academic
> professor, a university that's highly respectable. But reliable is something
> else. Reliable means that what they're saying is correct. A professor from
> Harvard can publish something, that can be a credible source, but they might
> not be reliable because what they're saying isn't necessarily correct."

### 4.5 The CRAAP method

| Criterion | Question to ask |
|---|---|
| **C**urrency | Is the information recent enough? |
| **R**elevance | Does it actually relate to your topic? |
| **A**uthority | Who wrote or published it? |
| **A**ccuracy | Is the evidence trustworthy and supported? |
| **P**urpose | Why was this source created? |

Detail per criterion:

- **Currency.** An AI article from 2012 is probably outdated; from 2025, more
  useful. Some topics change quickly, so newer sources matter more.
- **Relevance.** Topic: social media and political polarization. Weak source:
  general history of television. Strong source: study on TikTok algorithms and
  political opinions.
- **Authority.** Strong: university professor, researcher, academic journal,
  government institution, WHO, UN. Weak: anonymous blog writer, random social
  media account, Reddit, Wikipedia.
- **Accuracy.** Look for citations, statistics, data, references, peer review.
  Warning signs: no evidence, emotional language, unsupported claims. "Studies
  show…" is good if sources are provided; "Everyone knows…" is weak academic
  evidence.
- **Purpose.** Possible purposes: educate, inform, persuade, advertise,
  entertain. A research paper is usually informative; a company advertisement
  may be biased. Every source has a perspective or goal.

### 4.6 Peer review, author, content — the three signals

From the intro deck, as good / bad pairs:

| Signal | Good | Bad |
|---|---|---|
| Peer review | Published in a respected peer-reviewed journal | Appears on a blog or corporate website; bold claims not backed by evidence |
| Author | Employed by a respected university, or appears frequently in mainstream media | No online record |
| Content | Well-written and ordered, cites literature, conclusions emerge logically | Not written to excellent standards; claims outlandish, grandiose, or illogical |

### 4.7 The scholarly search, end to end

The intro deck models the whole loop with a worked example about climate
negotiations:

1. **Interest piqued by a media article** — found a reputable news source, and
   curiosity sparked a question: "how is it that small states and non-state
   actors come to have a powerful role in climate negotiations?"
2. **Google Scholar** — search key terms, review open-sourced academic literature
3. **Standard Google Search** — use technical terms found in those articles,
   then check out one of the academics cited to review their other research

Basic keyword search, step by step:

1. Develop a keyword list in your notes — from your assigned articles (look at
   the abstract) and from your research question
2. Type one or two keywords into Google Scholar. "Usually, a long list of
   references will pop up. Remain calm!"
3. Filter: only peer-reviewed articles, set a recent date range
4. Sift results for high-profile journals (e.g. IEEE), well-known authors or
   universities, and skim the abstract for relevance
5. Download and skim it
6. Use that article to play bibliography tag

### 4.8 Bibliography tag

The deck's method for expanding from one good paper to a reading list.

**Step 1 — pick an article from your assigned reading and skim it again.**
Read actively: highlight and take notes on

- sources cited a lot in one article or across many articles — these are the
  foundational articles in your field
- more niche sources that offer information related to your research question
- footnotes next to sentences that intrigue you

**Step 2 — go to that paper's bibliography.** Look up some of the sources listed
in your notes. Notice: have you mostly noted review articles or empirical
articles? Are your selected articles clustering around a specific topic? Then
decide what to learn about first — a niche source zooms in on your topic
quickly, a review article gives better grounding in the field first.

**Step 3 — open Google Scholar.** Paste the article title and author, find it,
skim the abstract for actual relevance. If useful: download it (or email the
course head for help), add it to your Zotero library with the connector, and
write a sentence in your Zotero notes about why it might be useful. Then decide
whether to return to your first article for another source, or play bibliography
tag again from this article's own sources.

### 4.9 Tools

- **Zotero** — reference manager to collect, organise, cite and share sources.
  Install from zotero.org; the Browser Connector adds sources while you browse
  Google Scholar. The Safari connector ships with the app and needs Safari 15 on
  macOS 11 Big Sur or later; enable it under Safari → Preferences → Extensions.
- **LaTeX** — for writing. Two shortcuts the deck recommends rather than
  learning the syntax: tablesgenerator.com for tables, mathcha.io/editor for
  flow charts and formulas.
- **Grammarly** — run the paper through before submitting.
- A good PDF annotation tool (Zotero covers this).

### 4.10 Annotated bibliography

An annotated bibliography is a list of research sources where each citation is
followed by a short explanation called an annotation.

Each annotation contains three parts:

| Part | Purpose |
|---|---|
| Summary | What is the source about? |
| Evaluation | Is the source credible and useful? |
| Relevance | How does it help your research topic? |

This is where the CRAAP method gets used in writing: "You'll be asked to produce
an annotated bibliography in which you have to use the CRAAP method to talk about
the sources that you've used."

---

## 5. Reading academic literature

### 5.1 Why papers feel hard

They use technical vocabulary, assume background knowledge, contain dense
information, and prioritize precision over simplicity. Confusion during the
first read is normal.

### 5.2 The biggest beginner mistake

Reading papers like novels. Research papers are **not** meant to be read
linearly from start to finish in detail. Good researchers skim first, look for
main ideas, and return to important sections later.

### 5.3 The SMART reading order

Read in this order, not the printed order:

1. Title
2. Abstract
3. Introduction
4. Conclusion
5. Headings / subheadings
6. Figures / tables
7. Methods and Results — **last**

This gives you the big picture before the details.

### 5.4 The abstract

A short summary of the entire paper — a 2000-word essay compressed to 200 words.
It covers what the study is about, how the research was conducted, and what the
researchers discovered.

Researchers read the abstract first to decide whether the paper is worth reading
fully. It saves enormous time.

**Four parts of an abstract:**

1. Research question — what problem is being studied?
2. Method — how was the research conducted?
3. Findings — what did the researchers discover?
4. Conclusion — why do the results matter?

Worked example from the deck, broken down:

> "This study examines whether social media usage influences political
> polarization among teenagers. Researchers surveyed 1,200 high school students
> across the United States. The findings showed that students frequently exposed
> to political content online demonstrated stronger partisan opinions. The study
> suggests social media algorithms may contribute to political polarization
> among adolescents."

| Part | Text | What it tells you |
|---|---|---|
| Research question | Does social media influence political polarization among teenagers? | The main problem being investigated |
| Method | Surveyed 1,200 high school students | Exactly how data was collected |
| Findings | Students exposed to more political content showed stronger partisan opinions | The main result |
| Conclusion | Social media algorithms may contribute to polarization | Why the findings matter |

### 5.5 Finding the research question and the contributions

The **research question** is the central problem the study tries to answer. It
is usually found in the abstract, the introduction, or at the very end of the
literature review.

**Contribution = what the paper adds to knowledge.** Look for and annotate:

- new evidence
- new theory
- new method
- new perspective
- new dataset

"It is very easy to understand the paper by jumping to those points directly."

### 5.6 Active reading

- Read the abstract of each paper first, then skim introduction and conclusion
  to confirm it is useful
- Highlight keywords and ideas that may matter to your research
- Write questions in your notes or the margins
- Email your professor with questions, or look up answers online
- Write short summaries (1–3 sentences) of each article in your notes

---

## 6. The literature review

### 6.1 What it is

A literature review is an **analysis of existing research** connected to your
topic and research question. It is not a book report, and it is not a list of
summaries. It is your **map of the field**.

A strong review does five jobs:

1. **Shows what researchers already know** — the established, agreed-upon findings
2. **Identifies disagreements** — where credible studies contradict each other
3. **Identifies trends** — the direction the field is moving over time
4. **Identifies gaps** — the questions no one has answered yet
5. **Justifies your own research** — showing exactly where your study fits in

**Minimum length: 600 words.** "Fewer than that usually means you summarized
instead of analyzed."

### 6.2 The conversation metaphor

The deck's framing, which is the single most useful idea in it. Picture five
researchers in a room debating AI in education:

| Researcher | Position |
|---|---|
| A | "AI improves learning" — better outcomes and engagement |
| B | "AI improves productivity" — students finish tasks faster |
| C | "AI can reduce critical thinking" — warns about over-reliance |
| D | "Effects depend on age" — what helps adults may not help children |
| E | "More research is needed" — the evidence is not settled |

> "Your job? Explain the whole conversation, not just one voice. Show how these
> views connect, where they clash, and what the room collectively believes. That
> weaving-together is the review."

The tutor said the same thing in the recording, and added where you sit in it:

> "With any topic there is always a conversation going on in the background
> between other researchers… your role is to look at those opinions and say, I
> agree with this aspect, I disagree with this aspect, and my opinion is X, Y, Z
> because of my research gap."

### 6.3 Structure

```
Theme 1      -> First idea in the field, backed by several studies
Theme 2      -> A related or contrasting idea
Theme 3      -> A third strand of the conversation
Research Gap -> The unanswered question your study targets
Conclusion   -> Pull the threads together and point forward
```

**Organize by idea, never by author.** The deck is emphatic:

> "Do not open paragraphs with a name. Those who do not have experience write
> 'Smith (2021) says… Johnson (2022) says… Lee (2023) says…' and that is a list,
> not a review. Lead with the idea. Then bring in the researchers as support."

### 6.4 Summary vs synthesis

Called "the single most important habit in a literature review."

**Summary** — what did ONE source say?

> "Smith (2022) found that AI improves student productivity."

One voice, in isolation. Useful, but this alone is not a review.

**Synthesis** — what do MULTIPLE sources tell us together?

> "Several studies suggest AI improves productivity and learning efficiency,
> although researchers disagree on its long-term effects on critical thinking
> (Smith et al., 2022)."

Many voices going into one insight, including where they agree and disagree.
This is the real goal.

### 6.5 The four-part paragraph

Every theme paragraph follows the same shape. Learn it once, reuse it everywhere.

1. **Topic sentence** — state the main idea of the paragraph in your own words
2. **Evidence** — bring in the studies that support that idea
3. **Analysis** — explain the patterns, agreements, and disagreements
4. **Transition** — into the next theme

Worked example from the deck:

> "Research suggests AI improves student productivity. Smith (2022) found
> AI-assisted tools cut task-completion time by 20 percent, and Johnson (2023)
> reported greater efficiency among high-school students on AI platforms.
> Together these findings show AI may lift academic productivity. Questions
> remain, though, about its long-term effect on learning outcomes."

A second worked example, on the theme *benefits of AI in education*:

> "Researchers broadly agree that AI improves accessibility and learning
> efficiency. Smith (2021) found gains in student productivity, while Johnson
> (2022) reported more personalized learning experiences. Similarly, Lee (2023)
> observed that AI tools helped students complete tasks more efficiently,
> reinforcing the earlier findings."

Why it works: idea first, the paragraph opens with the claim rather than a name;
three sources clustered supporting one point together; connective words
("while", "similarly", "reinforcing") link them; one clear theme, nothing drifts.

### 6.6 Comparing and contrasting

"Synthesis really comes alive when sources disagree. Your job is not to pick a
winner but to map the disagreement clearly so the reader sees the whole debate."

Four moves:

- **Point of agreement.** "Most studies agree that AI boosts short-term productivity…"
- **Point of tension.** "…but they diverge sharply on its effect over a full school year."
- **Grouping.** Cluster sources that share a finding, then contrast that cluster with another
- **Naming the reason.** Explain *why* they differ — different age groups, subjects, or measures

Worked example:

> "While Smith (2022) and Lee (2023) report clear productivity gains, both
> studied university students. Chen (2024), working with primary-school children,
> found no such effect. The disagreement may therefore reflect age rather than a
> true conflict in the evidence."

Signal phrases to keep on hand: *in contrast, however, similarly, building on
this, by comparison, on the other hand.*

### 6.7 Citation rules

Citing avoids plagiarism, gives credit, and lets readers trace your evidence.

Cite every time you:

- use a direct quote
- paraphrase someone's idea
- refer to a specific study
- use statistics or data

You do **not** cite common knowledge — facts a general reader already accepts and
could find in countless places.

| Example | Citation? |
|---|---|
| "Water freezes at 0°C." | No — common knowledge |
| "A 2024 study found AI use increased student productivity." | Yes — a specific finding from a specific study |

Rule of thumb: if you had to look it up in a specific source, cite it. If
everyone already knows it, you usually don't.

### 6.8 Common mistakes

Most weak reviews fail in the same few ways:

| Mistake | What it looks like | Fix |
|---|---|---|
| The "he said, she said" list | One source per sentence, no connection between them | Cluster and link |
| Pure summary | Retelling each study without analyzing it | Add a "so what" after the evidence |
| No clear themes | Sources dumped in random order | Group by idea, not by author |
| Dropped quotes | A quotation sitting alone with no lead-in or explanation | Introduce it and unpack it |
| Missing citations | Paraphrasing an idea but forgetting the source | Cite anything not common knowledge |
| Recency blind spots | Leaning only on old studies | Check whether newer work has moved the conversation |

---

## 7. Methodology — what it is

### 7.1 Definition

Quoting the San José State University Writing Center, via the deck: the
methodology section describes **how your research was conducted**, so readers can
check whether your approach is accurate and dependable. A good methodology
increases trust in your findings.

You are answering four things:

1. What evidence will be used
2. Where that evidence comes from
3. How it was collected
4. How it was analyzed

### 7.2 Why it matters

Without a methodology, findings cannot be verified, the research cannot be
replicated, and the conclusions are not trustworthy. "It becomes an essay you
wrote for your friends, not a work for academia."

**The test to keep in mind:** could a stranger, reading only your methodology,
repeat your study and expect similar results? If no, it isn't finished.

This is also the section reviewers attack first. "A weak methodology sinks a
paper faster than a weak conclusion, because if the method is flawed, the results
mean nothing regardless of how interesting they sound."

### 7.3 Where the two paths divide

Two students both study AI in education. Student A reads 30 scholarly articles
and writes about what the field already knows. Student B surveys 200 students
and reports what they said. Not the same methodology — A is doing a literature
review, B is doing empirical research.

"Your method is not a style choice. It follows from what you're asking."

A lit review methodology is short. An empirical one is detailed, because someone
has to be able to repeat what you did.

---

## 8. Methodology for literature review papers

### 8.1 The core idea

> "In a literature review, you didn't collect people. You collected papers. So
> your methodology describes how you collected papers, with exactly the same
> rigor an empirical researcher uses to describe recruiting participants."

**The test:** if a reader cannot repeat your search and land on roughly the same
set of articles, your methodology has failed. That is the whole standard.

### 8.2 The five components

1. **Research design** — what type of review was this?
2. **Databases** — where were the sources found?
3. **Search strategy** — how were sources located?
4. **Eligibility criteria** — which sources were included or excluded, and why?
5. **Analysis strategy** — how were the selected studies analyzed?

Many papers add a sixth, the **screening process**, showing how many records went
in and how many survived. "It is the easiest way to look way more professional."

### 8.3 Component 1 — name your review type

Most students write "this study employed a literature review methodology." That
is vague. Name the exact type, because each carries different expectations.

| Type | What it demands |
|---|---|
| **Narrative review** | Broad, flexible, no formal protocol. Most accessible to you |
| **Systematic review** | Strict protocol, exhaustive search, formal risk-of-bias assessment. Very demanding |
| **Scoping review** | Maps the breadth of a field — what exists and where the gaps are — without pooling results statistically. **Often the best fit for a strong student paper** |
| **Meta-analysis** | Statistically combines numerical results across studies. Requires comparable data and statistics skill |
| **Integrative review** | Combines different study types, qualitative and quantitative together |

Example sentence:

> "This study employed a scoping review methodology to map existing research on
> the impact of artificial intelligence on educational outcomes."

Why it matters defensively: "Naming the type protects you. Nobody can demand
meta-analysis rigor from a paper that clearly announced itself as a narrative
review."

### 8.4 Component 2 — databases

Name them specifically and say why you chose them, plus any access limits you
hit. See §4.1 for the list.

### 8.5 Component 3 — search strategy

"The most skipped part of the methodology, and the fastest way to separate your
paper from the pile."

Report:

- your keywords
- **the actual search string using Boolean operators** — almost no student paper
  includes this
- the time period and why
- language limits
- publication types
- **the date you ran the search** — "Fields move, so a search has a shelf life"

See §4.2 for operators and the keyword-building procedure.

### 8.6 Component 4 — eligibility criteria

Eligibility criteria are the rules deciding whether a paper enters your review.
**Write them BEFORE you start screening, never after.** "Deciding as you go is
how bias creeps in, because you will unconsciously keep papers that agree with
you."

Always two paired lists:

- **Inclusion criteria** — what a study must have. Inclusion defines the target
- **Exclusion criteria** — what disqualifies it. Exclusion removes the specific
  problems you actually ran into

They are not merely opposites.

Build across these dimensions only:

- Topic relevance, tied to your specific question and not just the general area
- Population — for example secondary students rather than university students
- Study type — empirical only, or reviews included
- Publication window
- Language
- Peer review status
- Full text availability

**A worked criteria set:**

> **Included:** peer-reviewed empirical studies published 2020 to 2025, examining
> AI tools in K-12 or secondary education, reporting learning or performance
> outcomes, published in English, with accessible full text.
>
> **Excluded:** studies unrelated to education, studies on university or adult
> learners only, duplicates across databases, opinion pieces and editorials,
> non-scholarly publications, and studies where the full text could not be
> accessed.

Presentation tip from the deck: put these in a small two-column table in your
paper — "it instantly reads like a real methods section."

**Four rules for criteria that actually work:**

1. **Make them testable.** A criterion should be answerable yes or no from the
   abstract alone. "High quality studies" is not testable. "Peer-reviewed
   empirical studies" is.
2. **Tie every criterion to your research question.** If you cannot explain why a
   rule exists, delete it. Arbitrary rules look like you were fishing.
3. **Justify the tight ones.** If you excluded everything before 2020, say why in
   one sentence. Unexplained restrictions look like hiding inconvenient evidence.
4. **Apply them consistently.** If you excluded one study for using university
   students, you cannot keep another with the same problem because you liked its
   findings.

### 8.7 Screening in two passes

Screening is two decisions, not one, and knowing this saves enormous time.

**Pass 1 — title and abstract.** Fast. You are only asking: could this possibly
meet my criteria? When unsure, keep it. "Being generous here is cheap; being
generous later is expensive."

**Pass 2 — full text.** Slow. Read properly and apply every criterion strictly.
Most exclusions happen here, and this is where you discover things the abstract
hid, like a sample that turns out to be university students.

Practical habits:

- **Log a reason for every full-text exclusion.** You will need those reasons for
  the write-up
- **Remove duplicates first.** The same article in Google Scholar and ERIC is one
  record, not two — forgetting this inflates your numbers dishonestly
- **Track everything in a spreadsheet from day one.** Rebuilding it later from
  memory is miserable and inaccurate

### 8.8 Reporting your numbers

Three sentences, "the easiest professional upgrade available to you." Report the
funnel: total records returned, records after duplicate removal, titles and
abstracts screened, full texts assessed, studies finally included.

> "The initial search returned 340 records. After removing 45 duplicates, 295
> titles and abstracts were screened, of which 58 underwent full-text review. A
> final set of 22 studies met all eligibility criteria and was included in the
> review."

Going further: draw a **PRISMA flow diagram**, a box-and-arrow chart of that
funnel. It is the standard in systematic and scoping reviews and makes a student
paper look genuinely publishable. Search "PRISMA flow diagram template" and adapt
one.

### 8.9 Component 5 — analysis strategy

How did you make sense of the papers once you had them? Name your approach:

| Approach | What it does |
|---|---|
| **Thematic synthesis** | Grouping findings into recurring themes. Most common at your level |
| **Narrative synthesis** | Describing patterns in prose without statistical pooling |
| **Chronological analysis** | Tracing how the field changed over time |
| **Comparative analysis** | Contrasting findings across populations, regions, or methods |

Also say what you extracted from each paper — typically author, year, country,
study design, sample, method, and main findings. If you kept an extraction table,
say so.

> "Following source selection, the articles were analyzed thematically to
> identify recurring findings, major debates, and research gaps within the field."

### 8.9a A limitations subsection — not on the slides, but expected

The Methodology deck lists five components and never mentions limitations. In the
lecture, though, the tutor singled out the worked example's limitations
subsection as the best thing about it:

> "What was not on the slides, but I think it's very good that this person has
> included, is limitations… It's such a great way to show that you're being
> critical of yourself and that you're being analytical. You've got this great
> process of finding your sources and analyzing them, but you can still recognize
> that there might be some limitations to your work."

Two kinds of limitation, which a student asked about directly:

1. **Systematic** — arising from your process. English-only search, so
   perspectives from non-English-speaking countries were missed; no
   pre-registered protocol; evidence overlap between two included reviews
2. **Content** — arising from what the papers actually said

The English-only limitation is common enough that the tutor said "you might feel
like you can also use the same limitation."

### 8.10 Word budget

Target **600 to 700 words**:

| Component | Words |
|---|---|
| Research design and rationale | 90–100 |
| Databases and why you chose them | 90–100 |
| Search strategy, keywords, string, filters | 150–180 |
| Eligibility criteria, inclusion and exclusion | 150–180 |
| Screening process and numbers | 90–100 |
| Analysis strategy and data extraction | 100–120 |

Search strategy and eligibility criteria together are nearly half the section.
"That is correct, because they are what make your review replicable, and they are
exactly the two components students under-write."

Going slightly over or under is fine — treat it as a guide, not a hard limit.

### 8.10a Q&A from the methodology lecture

**How many papers should I read to write a good methodology?**

> "It really depends on your paper. I'd say 20 to 30 tends to be a good amount,
> 20 being the minimum. I know we have asked you to look at 15 papers. If you can
> find five more than that, fine. If you can only find 15, that's fine as well —
> but you'll have to explain very specifically why these 15 papers were so
> relevant and why you had to exclude other papers."

**Should I use references inside the methodology?**

> "Yes. You should always use referencing when possible and when necessary… You
> won't get penalized."

**What if I am writing an empirical paper but also analyzed articles?**

Split the methodology into two sections: the first covering how you found your
sources (as above), the second covering your actual data collection.

**The recurring demand: answer "why".**

> "I cannot stress this enough, you have to answer the why. You have to be able
> to explain to someone why you have chosen the decisions that you have."

This applies to the review type, the databases, the date window, and every
eligibility criterion.

### 8.11 Using AI on this section

The Methodology deck's stated position, which **conflicts with the lesson-1
tutor's blanket anti-AI stance** (§1.2) — resolve with your own mentor:

1. Read a guide on literature review methods first, so you understand the options
2. Send your draft (introduction and lit review) to ChatGPT and ask it to analyze
   what type of review you appear to have conducted
3. Ask it to show you the standard structure of a methodology for that review
   type, with an example
4. Study how it organizes and sequences the components

Then **write yours yourself, in your own words**, following that structure.

The hard limit: "Your methodology must describe the search you actually ran. Keep
a running log while you search, with database, exact search string, date, and
number of hits. Then writing becomes description instead of invention, and you
can answer any professor who asks you about your methods."

---

## 9. Methodology for empirical papers

### 9.1 The full landscape of empirical methods

"Empirical research is bigger than surveys. You almost certainly won't use most
of these, but you should know they exist, because you'll read papers that used
them."

| Method | What it is |
|---|---|
| **Experimental** | Change one thing on purpose, keep everything else constant, measure the effect. RCTs in medicine are the gold-standard version |
| **Laboratory / wet-lab** | Cell cultures, chemical assays, animal models, tissue samples. Requires lab access, supervision, often ethics board approval |
| **Computational / simulation** | Common in CS and physics. Build or test a model, run it on data, report performance. **Training a machine-learning model on a public dataset counts** |
| **Observational** | Watch and record without intervening. Classroom observation, clinical cohort studies, field biology |
| **Secondary data analysis** | Take a large existing dataset (government statistics, public health records, an open Kaggle dataset) and analyze it in a new way. "Genuinely accessible to you, and worth remembering" |
| **Case studies** | Deep examination of one person, school, company, or event |
| **Content analysis** | Systematically code text, media, or images (e.g. analyzing 200 news headlines for framing) |

### 9.2 Why surveys are the default route

1. No lab, no equipment, no budget — Google Forms is free
2. You control the timeline — a survey can go out and close in two weeks
3. You can reach your own population easily — your school, peers, community
4. The analysis is manageable — percentages, averages, simple comparisons suffice
5. The write-up is well established — there is a standard structure

**The tradeoff to be honest about:** surveys tell you what people *report* about
themselves, not what they actually do. Self-reported data can be biased by
memory, honesty, and question phrasing. Good papers acknowledge this in their
limitations.

### 9.3 Interviews, and mixed methods

Surveys give breadth; interviews give depth. "A survey asks 200 people one
shallow question each. An interview asks 8 people twenty deep questions each."

Use interviews when you want to understand **why** people think something, not
just how many think it — the qualitative counterpart to the quantitative survey.

Practical cost: interviews take much longer per participant, need recording and
transcription, and are harder to analyze because you work with paragraphs instead
of numbers. But **6 to 10 good interviews is a legitimate study**.

**Mixed-methods design** — survey 150 people for the pattern, then interview 8 of
them to explain the pattern. "Genuinely impressive at your level and not much
harder."

### 9.4 The six components of a survey methodology

1. **Research design** — what type of study was this?
2. **Population** — who is the study about?
3. **Sample** — who actually participated?
4. **Instrument** — what did you ask, and why?
5. **Data collection** — when, where, and how?
6. **Ethical considerations** — how did you protect participants?

"Miss one and we will notice surely."

### 9.5 Component 1 — research design

The methodology always opens by naming the type of study.

> "This study employed a quantitative, survey-based research design to
> investigate how artificial intelligence influences study habits among
> high-school students."

Three things this sentence must make clear: what type of evidence was collected
(numerical, textual, or both); why that method was chosen over alternatives; how
it addresses your research question.

Vocabulary: **quantitative** means numbers and measurable patterns.
**Qualitative** means words, meaning, and experience. **Mixed-methods** combines
both. A Likert-scale survey is quantitative. An interview is qualitative. A
survey with open-ended questions attached is mixed.

### 9.6 Component 2 — population

The population is the entire group you want to understand and generalize to.
Examples: high-school students in Uzbekistan; FLEX program participants in
Central Asia; teachers at international schools in Tashkent.

Researchers almost never survey an entire population. The population simply
defines who the study is about, and therefore who your conclusions apply to.

> "The target population consisted of high-school students between the ages of
> 15 and 18."

**Be honest about scope.** If you only surveyed your own school, your population
is not "students worldwide." Narrowing this correctly makes your paper stronger,
not weaker.

### 9.7 Component 3 — sample

The sample is who actually completed your survey. Population: 10,000 students.
Sample: 120 students.

Report all of: number of participants; age range; gender distribution if relevant
to your question; schools or institutions involved; location; how you recruited
them.

> "A convenience sample of 120 students from three secondary schools participated
> in the study."

Sampling terms:

| Term | Meaning |
|---|---|
| **Convenience sample** | Whoever was reachable. What most of you will use — acceptable as long as you say so |
| **Random sample** | Everyone in the population had an equal chance of selection. Stronger but harder to achieve |
| **Stratified sampling** | Deliberately including proportions of subgroups, e.g. equal numbers from each grade |

**Size guidance:** aim for at least **100 responses** for a quantitative survey if
you can. Below about **30**, your percentages get unstable and a couple of
responses can swing your entire finding.

### 9.8 Component 4 — the instrument

"This is the longest part of your methodology, and the part students most often
rush."

You must always explain:

1. What was measured
2. How many questions the survey contained
3. What question types you used
4. Why those specific questions were chosen

> "The survey contained 18 items across three sections: demographic questions,
> AI usage frequency questions, and perception questions measured on a five-point
> Likert scale, followed by two open-ended items."

Do not just say "we made a survey" — describe its architecture. If you adapted
questions from a published study, say which one and cite it: **borrowing a
validated instrument is a strength worth advertising.**

### 9.9 Component 5 — types of survey questions

"The most interesting part, and where most papers are won or lost."

| Type | Good for | Example |
|---|---|---|
| **Multiple choice** | Facts and categories | "Which AI tool do you use most often? A) ChatGPT B) Claude C) Gemini D) I don't use AI tools" |
| **Likert scale** | Attitudes and opinions; gives numbers you can average | "AI improves my learning." 1 = Strongly disagree, 5 = Strongly agree |
| **Ranking** | Priorities | "Rank these study methods from most to least useful" |
| **Frequency** | Behavior | "How often do you use AI for homework? Daily, a few times a week, a few times a month, never" |
| **Open-ended** | Reasons and surprises | "Describe one advantage of AI for learning" |

On open-ended questions, the deck's example of a real answer you might get: *"Oh
it's a great way to cheat during homework assignments."* That is exactly the kind
of honest response open-ended questions exist to capture, and often the most
quotable material in your results.

**Balance:** mostly closed questions for analyzable data, with two or three
open-ended ones for depth. "All open-ended is a nightmare to analyze. All closed
is bloodless."

### 9.10 Component 6 — ethical considerations

The `Methodology for emp. papers.pdf` text layer ends at Component 5 (page 15 of
15), so the deck's own treatment of ethics is not in the file. What the course
does state elsewhere: consent and ethics matter for empirical papers and not
usually for literature reviews (§3). Confirm the required treatment with your
mentor before submitting.

---

## 10. Structure — the blueprint

This section comes from the recordings rather than a deck.

### 10.1 Why outline before writing

The tutor's analogy: you would not build a five-storey apartment block by "just
putting bricks together and see what happens." Architects spend weeks on a
blueprint before construction. Workers do not invent decisions while building —
they follow the blueprint.

The failure mode being prevented:

> "Many students open a blank document and immediately start typing. Perhaps they
> begin with the first article they read. Then they remember another article and
> write about that. Then they suddenly think of another idea and insert another
> paragraph somewhere in the middle. By the time they reach page six, they no
> longer know where the paper is going."

In research writing your blueprint is your **outline**. It tells you what
sections you will have, the purpose of each, what evidence belongs where, and it
keeps the paper focused.

> "Before writing your first paragraph, always ask yourself: do I know the
> structure of my paper? If the answer is no, don't start writing yet."

Also worth internalising, on expectations:

> "Professional researchers plan. They outline, they organize, they rewrite.
> Sometimes they completely rewrite an introduction three or four times before
> they are satisfied. And that's perfectly normal. Research writing is much
> closer to solving a puzzle than writing a diary."

### 10.2 A paper tells the story of an investigation

Research papers have structure the way novels do — but instead of fictional
characters, they tell the story of an investigation: begin with a problem,
explain why it matters, review previous knowledge, explain what was done, present
evidence, explain what the evidence means.

The detective framing for the empirical structure:

| Section | Detective analogy | Question it answers |
|---|---|---|
| Introduction | Introduces the mystery | Why are we studying this? |
| Literature review | What previous detectives already discovered | What do we know already? |
| Methodology | How you investigated the case | How was the study conducted? |
| Results | Reveals the evidence | What did we find? |
| Discussion | Interprets that evidence | What does it mean? |
| Conclusion | What we learned from the investigation | So what? |

### 10.3 Literature review paper structure

```
Introduction -> Theme 1 -> Theme 2 -> Theme 3 -> Research gaps -> Conclusion
```

The puzzle framing:

> "Imagine you've been given 100 puzzle pieces from 20 different boxes. Your job
> isn't to describe each individual piece. Your job is to organize them into a
> meaningful picture."

Organize by **theme, not author**, "because readers care about ideas, not
chronology." Done right, "your review feels like an argument instead of a
bibliography."

**Each major section is a miniature essay** with five components:

1. A clear heading
2. A mini-thesis explaining what that section argues
3. An introductory paragraph
4. The body paragraphs
5. A concluding paragraph that connects to the next section

And each paragraph has its own internal structure: topic sentence stating the
main idea immediately, body developing it with evidence from the literature, and
an ending that links naturally to the next idea.

### 10.4 Empirical paper structure

```
Introduction -> Literature Review -> Methodology -> Results
  -> Discussion -> Conclusion
```

### 10.5 Building the outline

If you are reviewing AI and education, sections might be: academic benefits;
ethical concerns; student perceptions; future challenges. Within each theme,
compare — which studies agree, which disagree, why findings differ, what patterns
emerge across the literature. "Those comparisons demonstrate critical thinking,
which is one of the main goals of a literature review."

A practical diagnostic:

> "An outline also helps you identify weak areas before you begin writing. If one
> section contains 10 studies while another contains only one, you may need to
> search for additional literature or reconsider your organization."

---

## 11. The introduction

### 11.1 The four-part structure

```
Broad topic -> Specific problem -> Research gap -> Purpose statement
```

Use it as a checklist before moving on to the literature review:

1. Have I introduced the topic?
2. Have I explained why it matters?
3. Have I clearly described the problem?
4. Have I justified the research gap?
5. Have I told readers exactly what my paper will do?

"If you can answer yes to all five questions, your introduction is probably
complete."

The tutor also recommends using it as a planning device: "Before writing full
paragraphs, you can simply make notes under each heading. Later, those notes
become your introduction. That's often much easier than starting at a blank page."

### 11.2 Part 1 — the background

Establishes the broad topic. "We're not discussing our own study yet. We're
simply helping the reader understand the broader context."

For a paper on AI and education, the background might explain that AI tools are
becoming increasingly common in schools and universities, changing how students
learn, write, and solve problems.

Include recent trends or important statistics where they establish significance,
but "avoid overwhelming readers with numbers. Use only the information that helps
them understand why the topic deserves attention."

**Broad but not vague.** The common failure is empty generalities: *"Education
has always been important"*, *"Technology is changing the world."* True, but they
tell readers nothing specific about your topic. If your paper is about online
learning, discuss the growth of online education. If it is about climate change,
discuss recent environmental challenges.

> "Think of yourself as guiding someone along a path. At the beginning of the
> path, they know very little about your topic. By the end of the section, they
> should understand enough to appreciate why your research matters."

Every sentence in the background should move the reader closer to your research
question.

### 11.3 Part 2 — the problem

The problem explains why the topic requires further investigation. "Research
exists because there are questions that haven't been fully answered."

Problems take several forms:

- Researchers disagree with each other
- Existing findings are inconsistent
- Technology has changed since earlier studies were conducted
- Society faces a new challenge previous research never considered

Worked example: researchers generally agree AI is becoming more common in
education, but they disagree about whether it improves critical thinking or
encourages dependence on technology. That disagreement is the problem your paper
investigates.

**Do not just say "more research is needed."** Explain specifically what remains
uncertain. "By clearly identifying the problem, you create a reason for readers
to continue. They begin to see that your study isn't just interesting, it also
addresses an important unanswered question."

### 11.4 Part 3 — the research gap

Called "one of the most important parts of the introduction," and the most
commonly misunderstood.

> "Many students think a research gap means nobody has ever studied the topic
> before. Instead, a research gap means that something important has not been
> fully explored. Researchers may have studied the general topic, but perhaps
> they haven't studied a particular population, country, method, or question."

Worked example — you are interested in AI and education, and there are already
thousands of papers. Ask instead:

- Has this been studied among high school students? *(population gap)*
- Has it been studied in Uzbekistan or Central Asia? *(geographical gap)*
- Has anyone examined long-term rather than short-term effects? *(temporal gap)*
- Has anyone compared different AI tools? *(comparative gap)*

Gaps can also come from a different age group, culture, research method,
theoretical perspective, or changes in technology over time.

**A gap must be supported with evidence.** "You cannot simply write *there is
little research on X*. You need to demonstrate that by referring to the
literature you've reviewed."

Sentence starters that signal a gap:

- "Although previous studies have examined…"
- "Limited research has investigated…"
- "Existing research merely focuses on…"
- "Comparatively little attention has been given to…"

**Do not overclaim.** "Nobody has studied artificial intelligence" is obviously
false and will not survive contact with a reader. And the stance matters: "Your
goal is not to criticize previous researchers. Their work provides the foundation
of your own. You are simply identifying the next logical question."

**How to find your gap:** "When you're searching the literature, always ask
yourself: what do these studies have in common? Often the answer reveals the gap.
If every paper studies university students, perhaps nobody has examined younger
learners. If every study uses surveys, perhaps qualitative interviews are
missing."

### 11.5 Part 4 — the purpose statement

The clearest sentence in the entire paper. There should be no ambiguity.

Examples, and note that the verb signals the paper type:

> "This paper examines how AI-assisted writing tools influence academic
> productivity among high school students." *(empirical)*

> "This literature review analyzes existing research…" *(literature review)*

Include the **topic**, the **population**, and the **context**.

Too broad: *"This paper discusses education."* What aspect? Which students? Which
country? Which issue?

Strong: *"This paper examines the relationship between AI-assisted writing tools
and academic productivity among grade 11 students in Tashkent."*

Avoid dramatic language — "this revolutionary study completely changes our
understanding." "Academic writing values precision much more than exaggeration.
Simple direct language is usually the strongest."

The test: if someone read **only** your purpose statement, would they understand
what your paper is about?

### 11.6 A worked introduction

The lecture walked through a full example on Uzbek educational institutions.
The movement to copy:

1. Opens **broadly** — education and economic growth
2. **Narrows** toward smaller educational institutions
3. Identifies the **problem** — these institutions have received relatively
   little scholarly attention despite their potential importance in Uzbekistan
4. States the **gap** — limited evidence about how investment in these
   institutions affects broader economic outcomes
5. Ends with a clear **purpose statement**

> "Notice how smoothly each paragraph builds on the previous one… When you're
> writing your own introduction, don't worry about making it sound complicated.
> Instead, focus on making it logical."

---

## 12. Using sources: quoting, paraphrasing, citing

### 12.1 Quote vs paraphrase

Academic writing is built on evidence. "Very few research papers contain only the
author's own ideas."

- **Quoting** uses the author's exact words inside quotation marks. Reserve it
  for wording that is especially memorable, precise, or important.
- **Paraphrasing** expresses the same idea in your own words and sentence
  structure while keeping the original meaning. This demonstrates that you
  understood the research rather than simply copying it.

**Paraphrase far more often than you quote.** "Your paper should sound like your
own academic voice, supported by evidence from the literature."

### 12.2 What a real paraphrase is

The misconception: paraphrasing means swapping a few words for synonyms. It does
not. "A true paraphrase reorganizes the sentence while preserving the author's
idea."

Worked example. Original:

> "Frequent social media use is associated with increased anxiety among
> adolescents."

**Weak paraphrase** — replaces *frequent* with *regular* and *associated* with
*linked*. Still too close to the original wording.

**Strong paraphrase** — changes both sentence structure and vocabulary:

> "Research suggests that adolescents who spend considerable amounts of time on
> social media tend to report higher levels of anxiety."

**The memory test:** "After reading the source, can I close the article and
explain the idea from memory? If you can, you're much more likely to produce a
genuine paraphrase rather than accidental copying."

Good paraphrasing also helps the paper flow, "because it allows you to connect
multiple papers into a single discussion instead of presenting isolated
quotations."

**Paraphrasing still requires a citation.** The idea belongs to the original
author even though the wording is yours.

### 12.3 IEEE referencing

Used in engineering, computer science, IT, and related disciplines.

- Unlike APA or MLA, IEEE does **not** put the author's name in the in-text
  citation. It uses numbers in square brackets
- The first source you cite becomes `[1]`, the second `[2]`, and so on
- If you refer to that first source again later, it stays `[1]` — **the numbering
  never changes**
- The reference list is ordered by **first appearance in the paper**, not
  alphabetically

"This system keeps the text clean and concise, especially in technical writing,
where many references may appear within a single paragraph."

Whatever style you use, the key principle is **consistency**. "One of the easiest
ways to lose marks on a research paper is through inconsistent citations. Always
check that every in-text citation appears in the reference list, and that every
reference listed has been cited somewhere in the paper."

---

## 13. Worked example: a finished student paper

`Mohlaroy--RAS.docx` is a complete scoping review that applies everything above.
Worth reading as the target standard.

This is not an incidental inclusion. A second mentor interrupted the literature
review lecture specifically to walk through this paper on screen, calling it
"the greatest and the best example of a literature paper for STEM majors." It
was written by a mentee working with a PhD from Bath University and published in
the Central Asian Medical Journal.

**Title.** *Factors Influencing Acceptance of AI-Assisted Robotic Surgery: Review
of Patient and Surgeon Perspectives*

**Structure.** Abstract → Introduction → Literature Review (2.1–2.4) →
Methodology (3.1–3.7) → Results (4.1–4.6, with two tables) → Discussion →
Conclusion → References (22 sources, IEEE numbered).

### 13.1 How it does the things the course teaches

| Course rule | How the paper does it |
|---|---|
| Name the review type (§8.3) | "A scoping review approach has been applied… carried out according to PRISMA-ScR criteria [15]" |
| Name databases specifically (§8.4) | Google Scholar, PubMed, JSTOR, ScienceDirect |
| Report the actual keywords (§8.5) | Lists all seven search phrases and states Boolean AND/OR were used to combine concept groups |
| Three concept groups (§4.2) | Technology ("robotic-assisted surgery"), population ("patient", "surgeon", "general public"), human factors ("trust", "perception", "acceptance", "adoption", "barriers") |
| Date window with justification (§8.6) | 2022–2026, "to ensure the relevancy of findings within the current framework of AI implementation in surgical procedures" — and explicitly exempts background citations from that window |
| Paired inclusion/exclusion lists (§8.6) | Four numbered inclusion criteria, four numbered exclusion criteria |
| Two-pass screening (§8.7) | Stage 1 titles/abstracts, stage 2 full text, both by two independent reviewers, discrepancies resolved by structured discussion |
| Data extraction dimensions (§8.9) | Five named dimensions fixed before screening: knowledge/awareness, trust/confidence, prior experience/satisfaction, expectations, willingness to pay |
| Name the analysis strategy (§8.9) | Narrative synthesis, with reasoning: "Due to the heterogeneity of selected studies in terms of designs, populations, and outcome measures" |
| Themed lit review, not author list (§6.3) | 2.1 promise and resistance, 2.2 patient perspectives, 2.3 surgeon perspectives, 2.4 lack of literature |
| Gap supported by evidence (§11.4) | "[8], the latest scoping review on the matter, [found] only 16 articles… however, none of those was related to the use of robotic surgical systems" |
| IEEE numbering (§12.3) | Numeric brackets throughout, reference list ordered by first appearance |

### 13.2 Two moves worth stealing

**Report absence as a finding.** Two of the five extraction dimensions were
barely covered by the included sources. Rather than quietly dropping them:

> "Two dimensions, prior experience/satisfaction and willingness to pay, were
> resolved only partially or not at all across the included sources. This is
> reported as a finding in its own right rather than omitted from the final
> review."

**Limitations that are specific, not ritual.** §3.7 names five concrete
weaknesses: only four databases and three included studies; English-only search;
no pre-registered protocol; no quality assessment of included studies; and
evidence overlap between the two included scoping reviews assessed at
reference-list level only. Compare this with the generic "this study had a small
sample size" that most student papers stop at.

### 13.3 The interesting finding, as a model of synthesis

The paper's headline result is a genuine synthesis rather than a summary — it
holds three sources against each other and reports a counter-intuitive pattern:

> As exposure to AI increased, preference for full autonomy **decreased** rather
> than rising. Surgeon interest in automated surgery dropped from a mean of 3.75
> (2021) to 3.35 (2024) on a 1–6 scale, while AI course awareness rose from 14.5%
> to 44.6%. On the patient side, comfort fell from 75.8% support for AI-assisted
> planning to 17.7% for fully autonomous intervention.

That is §6.4's "many voices going into one insight" done properly.

### 13.4 The paragraph the mentor dissected on screen

Section 2.1 of the paper was read out line by line as the model of how a theme
paragraph should work. The mentor's running commentary, condensed:

1. **Opens with an idea, not a name.** "Usually, the case for artificial
   intelligence in robotic surgery is very compelling." — "She opens the
   paragraph with an idea. She does not do it with name dropping. She gives that
   topic sentence… You never start with name dropping."
2. **Claim, with a numbered citation.** `[9]` claims AI contribution across
   pre-operative, intra-operative, and post-operative periods will yield superior
   results.
3. **Counter-argument in the same paragraph.** `[10]` argued surgical culture was
   not yet ready for AI — "in the same exact paragraph she gives a counter
   argument."
4. **A third source taking a side.** `[8]` still sees that issue as open.
5. **A fourth source shifting the lens.** `[7]` sees human acceptance, not
   technology, as the main challenge — "the fact that the culture doesn't accept
   does not mean that technology is not ready; it's solely the problem of the
   people."

The point being made: "In a single paragraph it's not a must to have all
agreements. You literally have nine, ten, eight and seven saying totally
different things… so there is a very complicated conversation going in that
paragraph. This is something I want to see in your papers."

**And the demand for real evidence.** The mentor singled out this sentence:

> "In a scoping review of 31 independent studies, [12] report that between
> 60-100% of the population failed to understand how RAS functions and tended to
> overestimate the robot's ability to operate autonomously while underestimating
> the influence of the surgeon."

> "She gives real evidence. This is what I'm craving to see in your papers.
> Whether it's in data, whether it's in maps, whether it's in graphs — doesn't
> matter. Give me real data. Support your arguments."

---

## 14. Finding a topic when you are stuck

A technique from the literature review lecture that is not on any slide, and is
the most practical tip in the course:

> "Read a couple of papers, specifically the discussion part. In that discussion
> part there is a limitation section. Some of the limitations of the paper could
> give you a rough idea on what you could work on. Usually when I'm struggling
> with finding a research topic that I'm actually interested in, I go through the
> discussion section of the articles, go through the limitations, and see what
> kind of limits this project had that I could potentially get as an idea for my
> own project."

Related, on how the research gap does work in the paper:

> "By mentioning research gaps you're telling them: you're going to be the one
> who's going to address all those gaps. That gives an importance to your paper,
> on why others should be reading it."

On working with professors, in answer to a student asking whether focusing on
Central Asia would limit their options: location should not affect it, but a
Central Asian professor "might be useful in terms of finding the right people to
connect with" and knowing where local data lives. The general advice was cold
emailing and getting several people to review a draft.

---

## 15. Assignments

Collected from the decks, in course order.

**Lesson 1 (Foundations).**

1. Choose a research topic you are genuinely interested in
2. Create a research question using the formula. Narrow enough, scope controlled,
   **10–15 words**
3. Find **15 academic/scholarly sources** via Google Scholar, JSTOR, or another
   academic database. Published within the last 10 years, related to your topic
4. Apply the **CRAAP method to every single paper** to prove credibility
5. Create a **mini annotated bibliography** with, per source: a short summary, an
   evaluation of credibility, and an explanation of usefulness to your topic

No word limit on the annotations — "total freedom… of course you want to keep it
as short and snappy as possible." And read the sources before writing the
bibliography.

**Lesson 2 (Reading academic literature).**

- Part 1, on the abstract only: identify the research question, method, findings,
  and contribution
- Part 2, active reading notes on the article itself: 5 important points, 2
  questions or confusions, 1 strength, 1 weakness
- Submit the link to the article you chose

**Blueprint lesson.** Create a simple outline, then draft your introduction using
the four-part structure.

**Literature review lesson.** Draft a literature review for your paper, **600
words minimum**, following the structures from this and previous classes.

**Methodology lesson.** Write your methodology, **600–700 words**, per the budget
in §8.10.

---

## 16. Contradictions and open questions

Flagged rather than silently resolved.

1. **AI policy.** The lesson-1 tutor is categorically against AI use ("you
   shouldn't be really using AI at any point"), notes that another mentor
   ("Matthew"/"Max") is more permissive, and the Methodology deck actively
   instructs students to run drafts through ChatGPT to study methodology
   structure (§8.11). The one point all sources agree on: **never let AI write
   the text you submit.** Confirm the boundary with your own mentor.
2. **Ethics component for empirical papers.** Listed as component 6 of 6 but the
   deck's text layer ends at component 5 (§9.10).
3. **Source recency.** Lesson 1 says nothing later than 2020 with a tolerance of
   one or two exceptions; the Foundations homework says "within the last 10
   years." The stricter rule is presumably intended.
4. **Deadlines** were not stated in any recording — "you will probably get the
   deadline today… once the homework is posted." Check Google Classroom.

---
