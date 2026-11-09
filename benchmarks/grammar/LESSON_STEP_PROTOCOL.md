# The lesson's step that follows the running error: declared protocol (cadence#258)

Declared on 2026-11-09, committed to the lane branch before the first life of either stage. The
founders, arms, readings, selection rule and gates below are fixed; a failed gate is reported as
failed.

## The question

On the default composition of 0.95.2 a validation reading on the grammar chamber swings between
checkpoints, because a lesson with few wrong rows takes a step of full size. A step that falls once
to a third of the rates late in a life settled it on development lives, and a count of lessons is a
parameter a continuing life does not know the end of. Does a step whose size follows the brain's
running error, with no count, settle the exact-clause reading on fresh founders at widths 64 and 128,
with every chamber reading and every sequence-battery bar holding?

## The candidate

`learning_running_error=ρ` (opt-in, `LearnerConfig.running_error`): the learner keeps the running
share of lesson rows answered wrong, `e ← ρ e + (1 − ρ) w`, where `w` is the share of a lesson's rows
whose answer differs from the label (a lesson the brain answers right counts as zero), starting at
one. A lesson's step is scaled by `e` to the power `learning_running_error_power` (0.5 unless named).
It keeps one number and no count.

## Instruments

- Grammar chamber: `benchmarks/grammar/grammar.py` through `benchmarks/grammar/lesson_step.py`
  (`plan`, `read`). The `default` arm: `Brain.compose(35, 18, modules=(w,), seed=founder,
  backend="cpu")` and the arm's settings, nothing else named. The first grammar for 4,000 batches,
  256-sentence panels read every 500 batches and in full at 4,000.
- Sequence battery: `benchmarks/sequence/battery.py`, arm `default` (the same composition, nothing
  named but the backend), 1,500 lessons and 100 test episodes per life, the tasks of
  `lesson_step.BATTERY`.

## Readings

- Swing: the largest minus the smallest exact-clause reading at 3,000, 3,500 and 4,000 batches.
- Clauses, held-out clauses, roles (both right), two hops: the full reading at 4,000.
- Brackets, paths: the means of the readings at 2,000, 2,500, 3,000, 3,500 and 4,000.

## Stage 1, development (selection)

Founders 0, 1, 2 at widths 64 and 128; arms `control` (the default composition) and the four
candidates `e99p5`, `e999p5`, `e99p1`, `e999p1` (ρ 0.99 or 0.999, power 0.5 or 1). Thirty lives.

**Selection rule** (`lesson_step.select`): among the candidates whose mean clauses, held-out clauses,
brackets and paths are each at least the control's less .05 at both widths, the one with the lowest
mean clause swing over the two widths; ties go to the smaller power, then the larger ρ. With none
eligible, the lowest swing overall goes forward, flagged. Development readings select and decide
nothing else.

## Stage 2, confirmation (fresh founders)

Grammar chamber: founders 9625800 to 9625805, widths 64 and 128, the control and the selected
candidate (24 lives). Battery: founders 9625810 to 9625815, the control and the candidate (nine
readings from eight lives per founder and arm). No founder of this block has lived in any lane.

| Gate | Reading | Pass |
| --- | --- | --- |
| C0 | the control's swing at each width | read first and reported, no bar |
| C1 (64), C1 (128) | the candidate's clause swing | at most .05 on at least 5 of 6 founders, at each width |
| C2 (64), C2 (128) | the candidate's mean clauses, held-out, roles, brackets, paths and two hops | each at least the control's less .03 |
| C3 | every battery task: look back first and second, copying unique and repeated, period 40, before, two hops with a think moment, induction, n-back | the candidate passes the task's bar (.90; copying unique .95, period 40 .85) on at least as many founders as the control or on 5 of 6, and its mean is at least the control's less .03 |

**Verdict.** #258's narrowed acceptance is met when C1, C2 and C3 all pass; the option then ships as
an opt-in, and a default is the owner's decision. A failed C1 is a measured negative for this form of
the step. A failed C2 or C3 with C1 passing records the cost that keeps the form out.

## Compute

One AWS c7i.16xlarge, one process per life, one BLAS thread per process; the count of workers enters
no reading.

## Amendment during the confirmation

`battery.py`'s n-back task takes no test lengths, so its battery lives raised before reading; the
n-back row reads the lengths it teaches (6 to 12), as the battery reads it. The chamber's lives had
finished; no other row changes.

## Result

Read on one c7i.16xlarge. Development selected `e999p5` with the flag (no candidate held brackets
within .05 of the control at width 64). Confirmation, founders 9625800 to 9625805:

| Gate | Verdict | Reading |
| --- | --- | --- |
| C0, the control | read | swing .042 at 64 (4 of 6 within .05), .024 at 128 (5 of 6) |
| C1 (64) | fail | swing .042, 4 of 6 within |
| C1 (128) | fail | swing .039, 3 of 6 within |
| C2 (64) | fail | held-out -.101, brackets -.113, roles -.060 |
| C2 (128) | fail | held-out -.128, roles -.053 |
| C3 | pass | every battery task within .03 of the control, no bar lost |

A measured negative for a step that follows the running error: it settles no more lives than the
default composition and costs held-out clauses. `learning_running_error` is withdrawn and the library
is unchanged; the candidate's lives run on commit 968a02b.
