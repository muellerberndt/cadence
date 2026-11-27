# Cadance: a minimal techno stream generator

Cadance (the application's name, spelled with an a) is built on Cadence; its source is at
[cadance-minimal](https://github.com/muellerberndt/cadance-minimal). One brain hears the bars it
has played and writes the next bar, without end, and a player renders the bars as drums, a
bass, a lead and a pad. This page follows its development in the order the problems appeared:
what went wrong, how it was measured, and which part of the library answers it. Use it when
your own task has many motor slots and one continuing stream.

The readings are from one application and its material. No chamber of the library measures
them. The python blocks run on a toy stream defined on this page, with small brains built
without the recall memory. They are controls that show the mechanics of each step, not the
application's brain and not its results.

| Step | Pitfall | What to do |
| --- | --- | --- |
| [2](#2-measure-against-repeating-the-last-bar) | A falling error that is still worse than copying | Split the error into slots that stay and slots that change |
| [3](#3-give-the-brain-a-copy-path) | No short route from what was heard to what is written | A one-way sensory projection onto the motor, born as a copy |
| [4](#4-feed-the-bars-the-material-returns-to) | Changes the last bar cannot predict | The bars 2, 4, 8 and 16 back as further inputs |
| [5](#5-teach-the-slots-that-are-wrong) | Lessons that keep pulling what is already right | Surprise by slot, and labels the inputs can answer |
| [6](#6-test-what-the-hierarchy-and-the-memory-do) | Parts assumed to help | Cut them and read the error again |
| [7](#7-a-lower-error-is-not-better-play) | Selecting a brain by its lesson error | Read free play against the material, and stop on it |
| [8](#8-at-play-the-brain-hears-what-the-player-did) | Play rules that feed the brain a copy | Bound what is played without replacing it |

## 1. The task: a bar is a row of motor slots

A bar of Cadance is 213 motor slots: which kit and which voices sound, the steps of three drum
rows, the steps of three synth lines, and the sound controls. Each slot holds one of a few
values, and the brain answers all of them in one settle (`slots=` at `compose`). The
observation is the bar heard, one sensory neuron per slot value. The label is the bar the
recording plays next, one value per slot. A slot whose label is unknown or should not be taught
is [left open](../teach.md#a-slot-the-lesson-leaves-open) with `-1`.

The life is one stream per recording: bar after bar in order, each lesson on the bar that
follows what the brain has heard. Reset at the boundary between two recordings, not between
bars ([reset and save](../life.md#reset-and-save)).

The toy stream below has four slots of four values. It holds a home bar, plays a fill in the
fourth bar of a phrase, comes back, and takes a new home value every 16 bars. The brain will
hear the bars 1, 2 and 4 back.

```python
import numpy as np

SLOTS, TOKENS = 4, 4                 # a bar: four slots, each one of four values (0 is a rest)
WIDTH = SLOTS * TOKENS
LAGS = (1, 2, 4)                     # the bars the brain hears, counted back from the bar it writes


def stream(bars, seed):
    rng = np.random.default_rng(seed)
    home = rng.integers(0, TOKENS, SLOTS)
    slot, value, played = 0, 0, []
    for t in range(bars):
        if t and t % 16 == 0:                        # a new home value in one slot
            home = home.copy()
            moved = rng.integers(SLOTS)
            home[moved] = (home[moved] + rng.integers(1, TOKENS)) % TOKENS
        bar = home.copy()
        if t % 4 == 3:                               # a fill, the same one twice in eight bars
            if t % 8 == 3:
                slot = rng.integers(SLOTS)
                value = (home[slot] + rng.integers(1, TOKENS)) % TOKENS
            bar[slot] = value
        played.append(bar)
    return np.array(played)


def one_hot(bar):
    x = np.zeros(WIDTH)
    x[np.arange(SLOTS) * TOKENS + bar] = 1.0
    return x


def heard(history):
    """The observation before the next bar: one block of slot values per lag, zeros where
    the stream is younger than the lag."""
    return np.concatenate([one_hot(history[-k]) if len(history) >= k else np.zeros(WIDTH)
                           for k in LAGS])


def examples(score):
    """Observations and labels of every bar that has its lags, and the bar before each."""
    first = max(LAGS)
    x = np.array([heard(score[:t]) for t in range(first, len(score))])
    return x, score[first:].copy(), score[first - 1:-1]


train, test = stream(512, seed=1), stream(256, seed=2)
x_test, y_test, last_test = examples(test)
assert x_test.shape == (252, WIDTH * len(LAGS)) and y_test.shape == (252, SLOTS)
```

## 2. Measure against repeating the last bar

Most slots of a bar equal the bar before. A learner that repeats what it heard is right on all
of those and wrong on every slot that changes. Read that baseline before the first lesson, and
read every brain as two numbers: the error on slots that stay and the error on slots that
change.

```python
def errors(answer, labels, last):
    """Share of wrong slots among those that stay, among those that change, and over all."""
    stay, wrong = labels == last, answer != labels
    return wrong[stay].mean(), wrong[~stay].mean(), wrong.mean()


stay, change, overall = errors(last_test, y_test, last_test)       # repeating the bar heard
assert stay == 0.0 and change == 1.0
repeat = overall                                                   # the share of slots that change
assert 0.05 < repeat < 0.25
```

On Cadance's held-out recordings, repeating the bar heard is wrong in 17.1% of slots. The
first brains of the present design answered 28 to 30% wrong and stayed there for an hour of
lessons. The split showed why: they were wrong on 20 to 28% of the slots that stay. A single
error number hides this. A brain can lower it by learning some of the changes and still be worse
than a copy.

## 3. Give the brain a copy path

In a composed brain with a learned code the sensory neurons project to the code region alone.
Every route from the bar heard to the motor then runs through hidden regions, and holding 213
slots still means learning 213 identity maps through them. Those brains did not learn it in the
time given. An earlier design had copied through a route from the memory to the motor, which is
the shortcut [the deep-brain guide](../deep-brains.md#give-the-hidden-region-a-job) warns
about: the hidden regions then have no work.

From 0.104.1 a connectome of your own may project the sensory neurons one way onto a region
that does not read the code ([what the region is](../senses.md#what-the-region-is)). Cadance
adds such an arc from every heard slot value onto the motor and sets its birth: the synapse from
a heard value to the same motor value starts at a gain, the rest of the arc at zero. A brain's
efficacies start as its connectome's `sign` ([NeuralGraph](../api.md#neuralgraph-cadence)), so
the birth is written there. The arc is plastic like any synapse, so lessons move it.

```python
from cadence import Brain, Connectome, Genome, Projection, develop

SHAPE = dict(modules=(24,), learned_code=24, slots=(TOKENS,) * SLOTS)
INPUTS = WIDTH * len(LAGS)


def copy_born(gains, seed=0, **genes):
    """A composed layout plus a one-way arc from the sensory neurons onto the motor, born
    with `gains[i]` from a value heard at LAGS[i] to the same motor value and zero elsewhere."""
    layout = Genome.composed(INPUTS, WIDTH, seed=seed, **SHAPE)
    arc = Projection("sensory", "motor", reciprocal=False, scale=0.1)
    anatomy = develop(Genome(layout.regions, (*layout.projections, arc)), seed=seed)
    sensory, motor = (np.asarray(anatomy.populations[name]) for name in ("sensory", "motor"))
    on_arc = np.isin(anatomy.pre, sensory) & np.isin(anatomy.post, motor)
    column, value = anatomy.pre - sensory.min(), anatomy.post - motor.min()
    same = on_arc & (column % WIDTH == value)
    sign = np.array(anatomy.sign)
    sign[on_arc] = 0.0
    sign[same] = np.asarray(gains, dtype=float)[column[same] // WIDTH]
    born = Connectome(anatomy.n, anatomy.pre, anatomy.post, anatomy.count, sign,
                      anatomy.populations, anatomy.label)
    return Brain.compose(INPUTS, WIDTH, connectome=born, recall=False, seed=seed,
                         **{key: SHAPE[key] for key in ("learned_code", "slots")}, **genes)


def read(brain, x, labels, last):
    brain.reset()                                # the test rows are streams of their own
    answer = brain.act(x, greedy=True)
    brain.reset()
    return errors(answer, labels, last)


plain = Brain.compose(INPUTS, WIDTH, recall=False, seed=0, **SHAPE)
copier = copy_born((2.0, 0.0, 0.0))              # a copy of the bar heard; the other lags silent
x_train, y_train, _ = examples(train)
for _ in range(10):                              # ten passes of lessons for the plain brain
    for i in range(0, len(x_train), 32):
        plain.lesson(x_train[i:i + 32], y_train[i:i + 32])
plain_stay, _, plain_all = read(plain, x_test, y_test, last_test)
born_stay, born_change, born_all = read(copier, x_test, y_test, last_test)
assert plain_stay > 0.3 and plain_all > repeat   # ten passes do not give it the copy
assert born_stay < 0.02 and abs(born_all - repeat) < 0.02    # at birth it repeats the bar
```

Born on the diagonal, Cadance's brain started at the level of repeating: 2.6% of the staying
slots wrong before its first lesson, and every changing slot wrong. Every lesson from there is
about a change. The same arc born random and small did not hold (a quarter of the staying slots
wrong after its first lessons), and an arc from the learned code onto the motor stayed at 28%
overall.

Three things to watch:

- **Birth gain.** Too small and the hidden paths outvote the copy, too large and no lesson moves
  a slot. Cadance uses 2 for the bar heard; read the staying-slot error at birth to set yours.
- **Wiring populations.** Cadance placed the arc slot by slot, through one small population per
  slot. Kept in the developed connectome, they cut it into more than 256 blocks, and the
  [block layout](../api.md#blocks-cadenceblocks) fell back to one block over the whole matrix: a
  lesson took 1,979 ms in place of 48 ms in a small reproduction. From 0.104.1 the layout warns
  at the fallback and `learning_drive_budget` refuses it. Build the `Connectome` again without
  the populations that only placed synapses; the synapses are the same.
- **What stays refused.** A sensory projection onto the neurons that read the code, and any
  synapse into the sensory neurons.

## 4. Feed the bars the material returns to

A copy cannot answer a change. Before designing for the changes, count what they are. In
Cadance's material most changed slots take a value the same slot held 2, 4, 8 or 16 bars
earlier: 85% of the changes in the drum rows and 73% in the lines. A fill ends and the pattern
comes back.

```python
def returns(score, lags=(2, 4)):
    """Of the slots that change, the share whose new value the slot held at one of `lags`."""
    reach = max(lags)
    now, before = score[reach:], score[reach - 1:-1]
    back = np.zeros(now.shape, dtype=bool)
    for k in lags:
        back |= now == score[reach - k:len(score) - k]
    return back[now != before].mean()


assert returns(train) > 0.5          # most changes of the toy stream are returns
```

So the brain hears those bars. Each lag is one more block of sensory neurons with the same
layout as the bar heard, and each block has its own copy-born arc onto the motor. The motor
then starts with a vote: the value heard one bar back against the values heard further back.
Cadance's arcs are born at gain 2 for the bar heard and at gain 1 for the further lags, the
further lags on the drum rows and the lines only.

```python
echo = copy_born((2.0, 1.2, 1.2))    # the bars 2 and 4 back together outvote the bar heard
echo_stay, echo_change, echo_all = read(echo, x_test, y_test, last_test)
assert echo_change < born_change and echo_all < repeat
assert echo_stay < 0.05              # the vote costs a few staying slots
```

The toy stream is built so that this vote answers its returns at birth. Real material is not:
lessons have to find which lag to trust for which slot and when. In Cadance the brains with
these inputs were the first to answer better than repeating, 13.1 to 13.9% against 17.1% on the
held-out recordings after about an hour.

The cost is input width. With five lags Cadance's observation is 8,608 values wide, and the
learned code reads all of them (see [step 9](#9-the-cost-of-a-wide-learned-code)). At play the
lags are the bars the brain itself played, so build the observation in one function that the
trainer and the player share.

## 5. Teach the slots that are wrong

`learning_surprise=True` teaches the rows whose answer differs from the label. With many slots
a row is almost never right in all of them. In Cadance 98.8% of rows were taught, and a taught
row pulled all 213 slots, the ones the brain already answered as shown too. Under that rule the
copy eroded: one brain born at the level of repeating (17.1%) answered 22.5% wrong after 38
minutes of lessons.

`learning_surprise_slots=True` (0.104.1) pulls only the slots that differ and leaves the others
open at that row; see
[lessons only where the answer differs](../teach.md#lessons-only-where-the-answer-differs).
With `learning_surprise_share=True` the step follows the share of slots taught.

```python
BY_SLOT = dict(learning_surprise=True, learning_surprise_slots=True, learning_surprise_share=True)
by_row = copy_born((2.0, 1.2, 1.2), learning_surprise=True)
by_slot = copy_born((2.0, 1.2, 1.2), **BY_SLOT)
batch, labels = x_train[:64], y_train[:64]
row_report, slot_report = by_row.lesson(batch, labels), by_slot.lesson(batch, labels)
taught_rows = row_report["attempted_presentations"] - row_report["agreed_presentations"]
assert slot_report["taught_slots"] == labels.size - slot_report["agreed_slots"]
assert slot_report["taught_slots"] < taught_rows * SLOTS      # fewer slots pulled than by rows
```

The second half is the label. A changed slot whose new value appears at none of the lags is a
value the inputs do not hold. Teaching it asks the brain to produce a value from nothing it
heard, and in Cadance brains taught that way kept adding notes in free play. Cadance leaves such
a slot open and teaches the returns only. At full size that form was ahead after the first
batch of lessons, 13.2% against 13.8%, with fewer staying slots wrong (1.2% against 2.1%).

```python
def returns_only(score):
    x, labels, last = examples(score)
    reach = max(LAGS)
    known = np.zeros(labels.shape, dtype=bool)
    for k in LAGS:
        known |= labels == score[reach - k:len(score) - k]
    labels[~known] = -1                          # open: no pull, and the slot is not judged
    return x, labels


x_known, y_known = returns_only(train)
assert (y_known == -1).any() and (y_known == -1).mean() < 0.1
for i in range(0, len(x_known), 32):
    by_slot.lesson(x_known[i:i + 32], y_known[i:i + 32])
taught_stay, _, _ = read(by_slot, x_test, y_test, last_test)
assert taught_stay < 0.05                        # the copy is still there after the lessons
```

This is a choice about what the brain should do, not a rule. A brain taught returns only does
not learn to introduce new material. In Cadance new values arrive only through sampling.

## 6. Test what the hierarchy and the memory do

Cadance's brain has a pathway per instrument (a detail region and a plan region each) under one
coordinator, four recall heads that read at the coordinator, and no route from the memory to the
motor. The [deep-brain guide](../deep-brains.md) gives the reasons and the tests. Three readings
from the development:

- **The coordinator's cut cost nothing.** With the copy arcs and the echo inputs in place,
  muting the coordinator's outgoing influence on an evaluation copy left the held-out error
  where it was. The gain over repeating came from the arcs and the learned code. The hierarchy
  is not shown to contribute, and nothing here supports calling it a conductor. Run
  [the cut and the frozen control](../deep-brains.md#prove-that-the-hidden-region-helps) before
  you describe what a region does.
- **Read where a correction goes.** `learning_credit_window=15` adds the `credit_*` readings to
  the lesson report ([measuring credit](../settling.md#measuring-credit-and-limiting-taught-drive)).
  They say whether a hidden region moved, not whether it learned a function.
- **An option can be on and do nothing.** `learning_credit_transport` ran for twenty minutes in
  Cadance with lessons equal, bit for bit, to the run without it: on a float32 device at that
  size its reference solve never qualified, and every lesson fell back to the ordinary phases.
  From 0.104.1 each such report holds `credit_transport_refused`, and a run of them warns once
  ([transport](../settling.md#transporting-a-correction-without-amplifying-ordinary-return)).
  When you compare an option against its control, check first that the two runs differ at all.

## 7. A lower error is not better play

The lesson error is read with the recording's own bars as input. At play the brain hears what
it wrote. The two diverge, and selection by error alone would have kept the wrong brains.

- **Greedy play froze.** Every trained brain, played greedily, fell into a loop of a few bars.
  Play is sampled; Cadance uses a temperature of 0.11.
- **Read 64 bars or more, by segments.** Count notes per bar and changes per bar in bars 1 to
  8, 9 to 16, 17 to 32 and 33 to 64, and divide by the same counts in the material. A ratio that
  climbs from segment to segment is a brain that piles up notes.
- **A narrow margin plays busy.** Surprise by slot never teaches a slot again once it is
  answered as shown, so its margin can stay small. One brain answered 96% of the staying slots
  right when read greedily and wrote a kick on nearly every step when sampled at the player's
  temperature. Read the brain the way it will be played.
- **More lessons made play worse while the error held.** After the first hour, further lessons
  at the same step raised the error and made free play busier. A continuation at a quarter of
  the step lowered the error (13.9 to 13.4% in one run), and all three such runs still ended
  past the density line between 147 and 197 minutes, two of them with 9 to 10 kicks per bar
  where the material has 4.7.
- **Stop on play, not on error.** Cadance stops a run when the last segment's density passes 1.5
  times the material's, and keeps the states before it. Three of the four brains kept are from
  the first hour.

```python
def rollout(brain, opening, bars, temperature):
    """Free play: every bar written becomes part of what the brain hears next."""
    brain.reset()
    history = [bar for bar in opening]
    for _ in range(bars):
        history.append(brain.act(heard(history)[None], temperature=temperature)[0])
    brain.reset()
    return np.array(history[len(opening):])


played = rollout(by_slot, test[:4], 64, temperature=0.11)
material = (test != 0).mean()                                    # sounding slots per bar
ratio = [(played[a:b] != 0).mean() / material for a, b in ((0, 8), (8, 16), (16, 32), (32, 64))]
changes = (played[1:] != played[:-1]).sum(axis=1).mean()         # slots changed per bar
assert played.shape == (64, SLOTS) and len(ratio) == 4 and changes >= 0.0
```

Report the ratio per segment and the change rates beside every error you report. They are
counts, not a judgment of the music.

## 8. At play the brain hears what the player did

A player may bound what a brain writes. Whatever it plays in place of the brain's bar is the
brain's next input, at every lag.

Cadance first capped a busy drum row or line by replacing it with a calm copy of an earlier
bar. On 64 bars this passed. On whole tracks of 304 bars it froze them: the drums equalled the
previous bar in 68 to 84% of bars, and one bass bar was held for 122 bars. The brain heard its
own copy and copied it again.

Thinning in place keeps what the brain wrote and drops hits down to the cap (off-beat hits
first, plain hits before accents, a dropped note becomes a tie). On the same starting passage,
128 bars:

| | Calm copy | Thinned in place |
| --- | --- | --- |
| Drums equal to the previous bar | 61% | 31% |
| Different drum bars per 16 | 5.5 | 10.1 |
| Longest held bass bar / lead bar | 27 / 32 | 10 / 9 |

Two more readings from the same work:

- **Caps come from the material.** Cadance caps a drum row at the 75th percentile of its bars
  in the recordings. A cap above that is where the row settles: with a kick cap of 6 the kick sat
  at 5 to 6 per bar late in a track, with 4 it held four.
- **Long tracks show what 64 bars hide.** Over 304 bars one brain held the kick pattern of its
  starting passage for 302 bars, and the hats of another dropped out for a hundred bars. Compose
  whole tracks before you settle play settings, and count them the way you count the material.

## 9. The cost of a wide learned code

A learned code's phase runs at a step that is halved as often as its coupling demands, and the
number of halvings can change from one moment to the next. Before 0.104.1 a moment in which it
changed took 8.6 s on Cadance's brain (8,608 inputs into 1,792 code units) and now takes 2.0 s,
with the same bits; see
[how a moment settles](../senses.md#how-a-moment-settles). A stream generator has to write a bar
faster than the bar lasts, so time one moment of your brain at full width before you train it.

## 10. Checklist

1. Read the error of repeating the last observation, and the share of slots that change.
2. Report every brain as the error on slots that stay and on slots that change.
3. If staying slots are wrong, give the brain a copy path and read it at birth.
4. Count which changes are returns, and to which lags. Feed those lags.
5. With many slots, teach by slot. Leave open what the inputs cannot answer, and say that you did.
6. Cut each part you credit, and check that an option under test changes the lessons.
7. Read free play sampled, by segments, against the material. Stop a run on it.
8. Let the player bound the brain's bar, never replace it. Test play settings on whole tracks.
9. Time one moment at full width.

## What this page does not establish

- The readings are from one application, one material and a few runs each. Two seeds agree for
  the comparison of step 4; most others are single runs.
- No chamber of the library measures a sensory arc, echo inputs, surprise by slot or the
  returns-only labels. They are options and wiring the library takes, with the costs stated in
  their guides.
- The hierarchy and the recall heads are part of the brain that was measured. Their
  contribution is not shown.
- Next-bar error and a self-fed rollout are component evidence. They do not establish phrases,
  arrangement or the form of a whole track; in Cadance voices enter and leave by the player's
  settings, not by a plan of the brain.
- The counts say how close free play is to the material in density and change. Whether a track
  sounds good is judged by ear.
