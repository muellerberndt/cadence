# Several records in one answer: the protocol (cadence#325)

## The declared run

Declared on 13 November 2026 on the lane branch and on the issue, before its first life. The
founders, arms, readings and gates below are fixed; a failed gate is reported as failed, and a
gate changes only by a declared amendment before the lives it reads.

### The question

Step 0 ([`README.md`](README.md#step-0-where-the-099-heads-fail)) located two terms on
development founders 0 to 2: the 0.99 heads read no record for a question's parts and no
chain of three hops, and with every record handed the association alone says eight parts at
.00. Does a brain whose heads each take a telling word of the question and learn which context
to hold to (`longterm_attention`), with a pathway from each head's region to the motor region
(`longterm_motor`), read and say several of its own records in one answer, at 100,000 records,
beside a transformer that holds the records in its context?

### The chamber

`several_world.py` as committed: 4 attribute relations and 3 links, each held by half of the
subjects, 32 places, the wordings and the two populations frozen there. A life
(`several_chamber.py`): the lesson population of 600 subjects told once; 10,000 lessons, each a
question taught at its answer's moment and then said (a chain of 2 or 3 hops with probability
.3, else parts of k subjects with k drawn in proportion to k; a part absent or a chain broken
with probability .15); then, with the records emptied, three readings on reading populations of
their own at 1,000, 10,000 and 100,000 records told once, and at each size 200 questions of each
of parts 1, 2, 4 and 8 and chains of 2 and 3 hops, 100 more of each kind of parts with each
part absent at .3, and 100 broken chains of each length. No reading teaches.

### Founders and arms

Founders 3250 to 3261 (twelve; no life of any lane used them). Every arm is
`Brain.compose(257, 264, slots=8, modules=(128,), backend="cpu", message=280,
longterm=records, longterm_turn=256, longterm_heads=8, longterm_distinct=0.1,
longterm_fade=False, ...)` at the library's defaults on the lane branch.

| Arm | Settings beside those | What it reads |
| --- | --- | --- |
| `heads` | `longterm_match=0.6, longterm_attention=1.0, longterm_motor=0.05` | The delivery under test |
| `birth` | `heads` with `longterm_attention=0.0`: the heads' maps held at zero | What the heads' lesson adds |
| `handed` | `birth`, every head handed its record at each answer | The ceiling of the answer's path |
| `rest` | `longterm_match=0.5, longterm_best=True, longterm_rest=True`: the read of 0.98 with eight heads | The heads of 0.99 |
| `none` | no message and no records | The floor |

No arm but `handed` is given a record, a cue or a place for any head: every head's read comes
from the question as heard.

**Controls** (`several_controls.py`), on founders 3250 to 3255, at the parameter count of
`heads`: a transformer (4 layers, 8 attention heads, no mask) and an LSTM. Each hears one
sequence per question: a window of 64 told facts in a shuffled order, then the question. The
window is handed: it always holds every fact the answer needs and every other fact of each
subject the question names or its chain reaches, with facts drawn at random up to 64. `online`
is one pass over the brain's 10,000 lesson questions (batches of 16); `offline` trains on fresh
draws of lesson questions until its score on 512 held lesson questions stops rising (patience
3 rounds of 10,000), 12 rounds pass or 2 hours end. The controls are read on the questions of
the 100,000 reading.

### Readings

`said`, `part`, `delivered`, `ordered`, `silent` and `refused` as `README.md` defines them, per
kind and size of question, at each size of the reading; the nursery's last tenth; each arm's
parameters; an answer's milliseconds and multiply-accumulates (every synapse once a sweep and
row of the settle, and the records' read: one accumulate per record a head's focus reaches and
per place of each key scored); the read's own milliseconds per answer; the controls'
multiply-accumulates of one forward pass over the window and the question.

### Gates

Read on `heads` at 100,000 records. A reading holds a bar on a founder where its value over the
group's questions is at or above the bar, and a gate passes where every reading it names holds
on at least 10 of the 12 founders.

- **A1, several records in one answer.** `delivered` at .95 on parts of 2, of 4 and of 8: the
  heads hold the k records the answer needs, each on a head of its own, and no other record.
- **A2, an answer of several parts.** `said` at .90 on parts of 2, of 4 and of 8: every slot
  says its answer, on subjects and facts no lesson used.
- **A3, chains through the heads.** `said` at .95 on chains of 2 hops and at .90 on chains of 3
  hops. *Amendment to the draft gate, declared here before any life:* the draft names the
  grammar chamber's paths. Those facts are heard and asked inside one episode and read from the
  recall memory's slots one hop a moment, which #302 reads. This protocol reads a chain through
  the heads of the long-term records in one answer, on this chamber. The grammar chamber's
  brain composes none of the new options, and with them unset a brain is the brain it was
  (contract tests).
- **A4, learned addresses and silence.** (a) `silent` at .95 over the parts no record answers,
  pooled over the absent groups of parts of 2, 4 and 8; (b) `refused` at .90 over the questions
  no record answers, pooled over one-part questions and broken chains; (c) the lessons' share:
  the founders' mean of `said` on chains of 3 hops of `heads` less `birth` is at least .30.
- **A5, beside the controls.** The table of `said`, parameters and multiply-accumulates of
  `heads` beside the four controls is reported whole. The gate: the mean of `said` of `heads`
  over founders 3250 to 3255 is no more than .03 under the better `online` control's mean on
  each of parts of 2, 4 and 8 and chains of 2 and 3 hops.
- **A6, cost.** The read's milliseconds per answer, the founders' mean: on parts of 8 under 8
  times the read on parts of 1, on parts of 4 under 4 times, on parts of 2 under 2 times.

### Expectations, stated before the run

A1 holds at 1.00 where no two subjects share a code, and falls by the share that do. A2 holds on
parts of 2 and 4 and is the gate most at risk on parts of 8, whose last slots are taught least.
A3 holds. A4 holds; `birth` says chains of 3 hops near .00 and parts as `heads` does, since a
newborn head reads the parts in the order heard. A5: the online controls say parts of 8 and
chains of 3 hops under the brain; the offline transformer is the control most likely to pass
it. A6 holds: a read scores the records about each head's focus.

### What a failure names

A1 failing with `ordered` at `delivered`: codes shared by two subjects, or a telling word read
as context at this size. A2 failing with A1 holding: the motor's path. A3 failing with
`delivered` low: the gates' lesson; with `delivered` high: the motor's path. A4a or A4b failing:
the level a record has to reach, at this size.

### Compute

One AWS c7i.8xlarge spot box (Name=cadence-0100-heads), one process per life and one thread per
brain process; the controls on three threads each.
