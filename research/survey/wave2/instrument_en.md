# Wave 2 questionnaire — English version

Project: Sanas, real-time bus occupancy information for Almaty. Version 1, 2026-09-01. Design rationale: `DESIGN_NOTE.md`.

---

## Introduction shown to the respondent

This survey asks how you choose between buses in Almaty when one bus is already at the stop and another is coming shortly. It takes about six minutes.

There are no right answers. We want your real choice.

Your answers are anonymous. We do not collect your name, phone number or email address. The results will be used for academic research on public transport and may be published in aggregate form.

You can stop at any time by closing the page.

---

## Section 1: Screening

**S1. How old are you?**

- Under 18
- 18-24
- 25-34
- 35-44
- 45-54
- 55-64
- 65 or older

_Respondents selecting "Under 18" end here._

**S2. How often do you travel by city bus in Almaty?**

- Daily or almost daily
- 3-4 times a week
- 1-2 times a week
- Less than once a week
- I do not travel by city bus

_Respondents selecting either of the last two options end here._

**S3. In the last month, how often have you encountered a bus so crowded that you hesitated to board?**

- Every trip or almost every trip
- On most trips
- On about half of my trips
- Rarely
- Never

---

## Section 2: Choice tasks

**Instructions.** In each situation you are standing at a bus stop. Both buses go where you are going and the fare is the same. One bus is at the stop now. The other arrives in a few minutes, and a mobile application tells you how full it will be. Choose the bus you would actually take.

The crowding levels are explained once, before the first task:

- **Seats available** — you will be able to sit down.
- **Standing room** — most seats are taken, you will stand, but you can move.
- **Packed** — passengers are pressed together and it is hard to move.

Each task is presented in this form:

> **Task 1 of 10.** It is a Wednesday at 14:00.
> The bus at the stop now is **packed**.
> The next bus arrives in **2 minutes** and will have **seats available**.
>
> Which do you take?
> - Board the bus at the stop now
> - Wait 2 minutes for the next bus

### Block 1 tasks

| Task | Bus at the stop | Next bus | Arrives in | Time |
|---|---|---|---|---|
| 1 | packed | seats available | 2 min | Wednesday 14:00 |
| 2 | packed | standing room | 2 min | Wednesday 08:00 |
| 3 | packed | standing room | 5 min | Wednesday 14:00 |
| 4 | packed | seats available | 5 min | Wednesday 08:00 |
| 5 | standing room | seats available | 8 min | Wednesday 14:00 |
| 6 | standing room | seats available | 8 min | Wednesday 08:00 |
| 7 | packed | seats available | 12 min | Wednesday 14:00 |
| 8 | packed | standing room | 12 min | Wednesday 08:00 |

### Block 2 tasks

| Task | Bus at the stop | Next bus | Arrives in | Time |
|---|---|---|---|---|
| 1 | standing room | seats available | 2 min | Wednesday 14:00 |
| 2 | standing room | seats available | 2 min | Wednesday 08:00 |
| 3 | packed | seats available | 5 min | Wednesday 14:00 |
| 4 | packed | standing room | 5 min | Wednesday 08:00 |
| 5 | packed | standing room | 8 min | Wednesday 14:00 |
| 6 | packed | seats available | 8 min | Wednesday 08:00 |
| 7 | standing room | seats available | 12 min | Wednesday 14:00 |
| 8 | standing room | seats available | 12 min | Wednesday 08:00 |

### Two additional tasks, excluded from estimation

| Position | Purpose | Content |
|---|---|---|
| 5 | Dominance check | The bus at the stop has seats available; the next bus arrives in 8 minutes and is packed. Choosing to wait indicates inattention. |
| 10 | Test-retest | Task 3 repeated verbatim. |

Task order is randomised within the block, except that these two hold their positions.

---

## Section 3: Attitudes toward real-time information

**A1. What is the longest you would wait for a less crowded bus, if you knew for certain that the next bus would have seats?**

- I would not wait, I take whichever bus arrives first
- 1-2 minutes
- 3-5 minutes
- 6-10 minutes
- More than 10 minutes

**A2. Do you currently use a mobile application to plan bus trips in Almaty?**

- Yes, most times I travel
- Sometimes
- No

**A3. If an application showed how full each approaching bus is, how often would you check it before boarding?**

- Every time I wait for a bus
- Most of the time
- Sometimes, depending on the situation
- Rarely
- I would never use it

**A4. How much would you trust the crowding level shown by such an application?**

- I would trust it completely
- I would trust it most of the time
- I would trust it only if it matched what I could see
- I would not trust it

---

## Section 4: Demographics

**D1. Gender**

- Male
- Female
- Prefer not to say

**D2. What is your current main occupation?**

- Student
- Employed full-time
- Employed part-time
- Self-employed or own business
- Unemployed
- Retired
- Prefer not to say

**D3. What is the main purpose of your bus trips? _(select all that apply)_**

- Travel to work
- Travel to study
- Personal errands
- Leisure or social visits
- Other

**D4. At what time of day do you most often travel by bus?**

- Morning peak, roughly 07:00-09:00
- Midday, roughly 09:00-16:00
- Evening peak, roughly 17:00-19:00
- Evening, after 19:00

**D5. How long is your typical bus trip?**

- Under 10 minutes
- 10-20 minutes
- 21-40 minutes
- More than 40 minutes

**D6. Which district of Almaty do you travel from most often?
_(list of districts, plus "outside Almaty")_**

---

## Notes for implementation

- S1 and S2 must terminate the form, not merely record an answer. Wave 1 mixed ineligible respondents into the analysed sample because screening was absent.
- The age question starts at 18. Wave 1 offered a 14-24 band, so the collected sample may include minors.
- Record the assigned block number and the task order with each response.
- Keep the Google Forms timestamp.
- Do not collect email addresses or any other identifier.
