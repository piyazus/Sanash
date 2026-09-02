# SANASH cold email template

Supersedes `email_template.md`, which asked for collaboration and used a
first-name salutation. Both are now forbidden. That file is kept for history.

Every paper named in an email must already be a row in
`research/refs/base/references.md`. If it is not there, stop, verify it against
a live page, add it with its DOI, and only then write the email. See the
citation rule in the repository `CLAUDE.md`.

---

## Template

```
Subject: [5 to 8 words, specific, naming their work]

Dear Professor [Surname],

[Sentence 1 to 3. Their paper, by title, and the specific thing it found or
measured. This is the opening. Not the student, not SANASH.]

[Sentence 4 to 6. The one point where that result meets a decision in a field
study of bus crowding information in Almaty, run with the city bus operator.]

[One sentence saying who is writing and what the study is. Short.]

[Closing sentence: one question, answerable in a paragraph.]

Best regards,
Diyas Tleukin
```

## Style rules

- Around 130 words. Never over 160.
- Open with the recipient's work. The student comes later, in one sentence.
- Subject line 5 to 8 words, specific, naming their work or its finding. Never
  the word "inquiry".
- No em dashes.
- No colons in prose. Colons in the subject line only if unavoidable.
- End by asking for an answer to one question. Never ask for collaboration,
  supervision, a position, a reference, or a call.
- No praise that is not a statement of fact about the work.
- No fabricated personal connection.

## Salutation

- Professor: `Dear Professor [Surname],`
- Doctorate but not a professor: `Dear Dr [Surname],`
- PhD student or no doctorate: `Dear [Full Name],`
- Never `Hi [first name]`. Not in a first email, not in a reply, not when the
  person signed with their first name.
- Never guess gender. Never use a pronoun for the recipient.

## What the study is allowed to be called

True today, and the ceiling of what an email may claim:

> a field study of real-time bus crowding information in Almaty, run with the
> city bus operator

Forbidden, because they are not true:

- a submission to Transportation Research Part C, or to any named journal
- a launch across 25 cities, or any city count
- a partnership with a state transportation company stated as a signed deal
- any accuracy figure, level threshold or deployment claim for the device

The device is not built or trained. Do not describe it as working.

## After sending

Flip the cited paper's row in `research/refs/base/references.md` from `LISTED`
to `CITED (Surname)` in the same commit as the email draft. A paper named in a
sent email that still reads `LISTED` is a bookkeeping failure, and the reference
base stops being trustworthy the moment that happens.
