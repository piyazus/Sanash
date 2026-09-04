# Terra research course — source material

Lesson recordings, slide decks, and one worked student paper from the Terra
academic research course. Background material for the paper track (Danyshpan);
the course teaches paper structure, literature review, and methodology.

**Start with [`WRITING_RULES.md`](WRITING_RULES.md)** — the rules to follow
when writing, 62 of them, each with a verbatim quote and a source. Every quote
was re-checked against the transcripts and decks in a second pass.

[`course_notes.md`](course_notes.md) is the full lesson-by-lesson conspectus
behind it (~11.8k words, 16 sections). Read it when a rule needs its context.
The raw files are kept for verification and re-reading, not for routine use.

## What's here

| File | Type | Length | Covers |
|---|---|---|---|
| `Foundations class- group 3.pdf` | slides, 20 pp | — | What research is/isn't, research questions, scope, CRAAP, annotated bibliography |
| `1st lesson- part 1.mp4` | recording | 34:56 | Lesson 1, first half. Call timed out mid-slide, continues in part 2 |
| `1st lesson- part 2.mp4` | recording | 19:00 | Lesson 1, second half. Databases, search, CRAAP, homework |
| `Research - intro.pdf` | slides, 20 pp | — | Researcher mindset, Zotero, scholarly search, bibliography tag |
| `2nd class-Reading.pdf` | slides, 19 pp | — | Reading papers, SMART order, abstracts, contributions |
| `video1768002125.mp4` | recording | 24:56 | Blueprint/outline session, first half. Ends with a break |
| `video1168216779.mp4` | recording | 26:16 | Same session, second half. Research gap, purpose statement, paraphrasing, IEEE |
| `Lit. review.pdf` | slides, 15 pp | — | Literature review drafting |
| `video1433589796.mp4` | recording | 53:55 | Literature review lecture. A second mentor dissects `Mohlaroy--RAS.docx` on screen from ~30:00 |
| `Methodology.pdf` | slides, 17 pp | — | Methodology for literature review papers |
| `video1101430871.mp4` | recording | 46:02 | Methodology lecture (lit-review papers), with worked example and Q&A |
| `Methodology for emp. papers.pdf` | slides, 15 pp | — | Methodology for empirical papers. **Incomplete** — see below |
| `Mohlaroy--RAS.docx` | student paper | 22 refs | Published scoping review, held up in class as the model answer |
| `transcripts/` | text | — | Timestamped transcripts of all six recordings |

Video filenames are as delivered; the topic column is the reliable index.

## Transcripts

`transcripts/*.txt`, one per recording, `[MM:SS] text` per line. Produced with
faster-whisper `small`, int8 on CPU, VAD filtering on, English detected at
p=0.99–1.00. Filenames match the source videos.

Accuracy is good but not perfect — names and some technical terms are mangled
(e.g. "prison's criteria" for PRISMA, "i.e. citations" for IEEE). Quotes taken
into `course_notes.md` were checked against context. Verify against the audio
before quoting anything consequential.

Regenerating them needs `ffmpeg` and `faster_whisper`; both were already on this
machine. Roughly 4.5 min of compute per 35 min of audio.

## Known gaps

1. **`Methodology for emp. papers.pdf` is truncated.** It names six components
   of a survey methodology but the file ends at component 5 (page 15 of 15).
   Ethics/consent is missing, and no recording covers it.
2. **The AI policy is contradictory.** The lesson-1 tutor is categorically
   against AI use; the Methodology deck instructs students to run drafts through
   ChatGPT to study structure. All sources agree AI must not write submitted
   text. See §16 of `course_notes.md`.
3. **Source-recency rules disagree** — "nothing later than 2020" in the lecture
   vs "within the last 10 years" in the homework brief.
4. **No deadlines** are stated in any recording. They live in Google Classroom.

## Relevance to this repo

The course is written for survey and literature-review papers, but §9.1 of the
notes counts training a model on a public dataset as computational empirical
research — which is what the Sanas track already is. The RTCI field-experiment
paper is the experimental route.

The methodology, literature review, and IEEE-citation guidance applies directly
to `research/paper/`. `Mohlaroy--RAS.docx` is a useful structural reference for
a scoping review: themed literature review, five-component methodology, absence
reported as a finding, and specific rather than boilerplate limitations.

## Git status

This directory totals ~388 MB and is excluded from git by `.gitignore`
("Course/outreach material dropped into the repo, large and not project
source"), alongside `research/refs/`.

The write-ups are exempted from that exclusion, so `README.md`,
`course_notes.md`, and `transcripts/` are tracked while the recordings, decks,
and the `.docx` stay out of history. The media lives on disk only — back it up
elsewhere if it matters, since git will not.
