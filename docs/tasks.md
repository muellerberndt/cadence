# Task recipes

How each kind of task becomes a clamp, a target, and a readout. Every recipe below is a
runnable rung with a receipt in cadence-examples or in the Kaggle set; the numbers there
say how each fares against the usual models on the same data.

| task | clamp | nudge and target | readout | where |
|---|---|---|---|---|
| classification, dense input | levels in [0, 1], one owner per feature (pixels) | cross-entropy toward the label's owner | most active output owner | digits, MNIST, Digit Recognizer |
| classification, tabular | place codes for numbers, one-hot for categories, a missing owner per column | cross-entropy | most active output owner | Titanic, Spaceship Titanic |
| regression | as above | quadratic, on a place code of the target scaled to [0, 1]; a linear calibration on out-of-fold readouts | the bump-weighted mean, mapped back | House Prices |
| multi-label (a chord) | one owner per pitch per beat | quadratic on a multi-hot target | every owner above a validation-chosen threshold | chorales |
| text, bag of words | binary presence of the top tokens | cross-entropy | most active output owner | Disaster Tweets |
| text, windowed | one-hot blocks per position | cross-entropy | most active; sample to write | text |
| imitation of a teacher | the state, as pixels | cross-entropy toward the teacher's move | most active legal move | Connect Four, sign writer |
| from reward | the state, as pixels or a place code | cross-entropy toward the action taken, weighted by its advantage | draw from the softmax; greedy to evaluate | Pong, cart-pole |
| a measured wiring | sensory owners at a declared amplitude | quadratic on the motor pattern of a fact, or none: settle and score | a declared readout set | *C. elegans* |
| a stream that drifts | place codes, the previous label as an input, one update per chunk of 32 rows | cross-entropy toward each chunk's labels at a constant rate, with a seam decay | most active output owner, scored before its label arrives | electricity, a synthetic drift stream |
| a few labels among many rows | place codes; the field on every row, or the plain classifier on the labelled tenth | quadratic masked reconstruction, then cross-entropy on the labelled rows; or cross-entropy alone with a long, slow-decay schedule | most active output owner | OpenML, a tenth of the labels |
| vocal learning (a memory and a mirror in one net) | sequence owners: fixed random conjunctions of a sound's cue and a clock, the strongest few awake; for the mirror, the peaks, pitch and loudness of the last frames | quadratic on the expected-frame group toward the frame heard at each tick while listening (one-way seams, no bias, seams decay); quadratic on the motor group toward the command just issued while babbling | the expected frame at each tick, then the command the mirror gives for it, into the syrinx | grey parrot |

## Tabular features as a place code

A number `v` in a column becomes `bins` owners with a bump `exp(-(v - c_k)^2 / 2w^2)` over
quantile-bin centres `c_k` fitted on training rows, plus one owner that is 1 when the
value is missing. A category becomes one owner per level seen often enough plus one for
the rest. The result is a clamp in [0, 1] with a few active owners per row, which is the
regime the rule wants: no owner sees a raw magnitude, every value is a pattern. `Encoder`
in the Kaggle harness is the reference implementation. Three details it learned the hard
way:

- **A value nearly every row shares is silence.** A sparse count column (a word count, an
  amount that is usually zero) has one value in nine rows out of ten. Quantile bins put
  most of their centres on that value, and every such row lights most of the column's
  owners; on the cnae-9 task (856 such columns) that meant about six thousand owners at
  full level per row, and both the patch net and an MLP sat at chance while a linear
  reader did not. Coding the shared value as no owner on (the rule the harness applies at
  a nine-in-ten share) brings the row down to a handful of active owners. Applied at a
  lower share it costs dense tables a point, because a three-level column then loses the
  owner for its most common level.
- **A column with few distinct values gets one owner per value**, with duplicate centres
  from ties collapsed, so a 0/1 flag is one owner and a 1–5 grade is five.
- **Bump width is a choice the table makes.** The width `w` is either the value range over
  `bins` (wide, smooth) or the gap to the nearest neighbouring centre (sharp). Sharp codes
  gained 1.7 points on one task and lost 1.5 on another for the patch net, and in every
  measured case a logistic regression on the same code preferred the same width, so the
  harness picks the width by a linear reader's cross-validation on the training rows.

## Regression by the quadratic nudge

The target is the value scaled to [0, 1] on training rows, the nudge `beta · (target − s)`
on the output owners. How many owners hold it matters: on the Ames housing data (the
Kaggle House Prices training file) one owner holding the value gave 0.151 log-price RMSE,
two owners holding the value and its complement 0.135, and a place code of eight bumps
over evenly spaced centres 0.134, read out as the bump-weighted mean; an ensemble of five
nets on the place code reached 0.131, against 0.130 for ridge regression and 0.135 for
gradient boosting on the same features. A value as a pattern is what the rule learns
well; a value as one owner's level is not. The raw readout is not the prediction
directly: fit `a · s + b` on out-of-fold readouts and map back through the scaling. That
keeps every choice on training rows and makes the readout as calibrated as the data allow.

## Wide sparse inputs

Thousands of mostly-silent one-hot owners (a bag of words, a wide window of characters)
are the input the rule likes least: with fan-scaled seams the drive from a few active
owners is small, and learning is slow relative to the same-shape MLP. Normalising the
clamp so that each row's total input is constant helps a little; a first layer of seams
shared across positions (an embedding) is the change the text rungs point at. `embedded`
builds that wiring and `Learner(tie_groups=...)` keeps the shared seams equal: every
position reads one embedding table, learned by the same local rule (the mean of the tied
seams' contrasts is still a function of those seams' own endpoints).

## Streams that drift

A stream has no epochs: every row is predicted before its label is used, then learned from
once. What the Kaggle stream rung found, over three seeds on OpenML's electricity data and
a synthetic stream whose concept is replaced twice: a constant learning rate of 8 with
momentum 0.9, a larger leak in the owner rule (0.3, so a silenced owner still answers), a
seam decay of 0.003 an update (`LearnerConfig.decay`, which keeps the net plastic after a
drift), and one free/nudged update per chunk of 32 rows. Three updates per chunk gained
seven points on a short probe and lost eight on electricity; the receipt uses one. Results:
electricity 0.851 for the patch net against 0.873 for logistic regression with one SGD step
per chunk, 0.854 for a same-shape MLP, 0.834 for boosting refitted on a sliding window, and
0.853 for repeating the previous label; the synthetic drift 0.796 (seeds 0.827, 0.735,
0.827) against 0.878, 0.854 and 0.859. The rule learns and relearns, slower per sample
than SGD and with a spread across seeds after a drift; the one-pass MNIST rung, where it is
ahead (0.885 against 0.844), is the same mechanism on a richer input. What not to do: scale
the initial seams up (the output owners saturate and learning stops), or use a rate of 1
with momentum (the memory saturates within a minute of a stream).

## A few labels among many rows

With a tenth of the labels on four OpenML tables the plain patch net, trained for 200
epochs at a decay of 0.99 an epoch (forty epochs on 320 rows is only a few hundred updates),
scores 0.686, 0.828, 0.913 and 0.941 against logistic regression's 0.692, 0.829, 0.911 and
0.933 on the same rows. A field net that first learns every row by masked reconstruction
and is then taught the labelled tenth helped on one table (car, 0.858 against 0.828) and
hurt on two (segment 0.867 against 0.913, kr-vs-kp 0.907 against 0.941): a model of the
table is not, at this size, a head start for a label.

## Two nudge groups in one net: a memory and a mirror

The grey parrot rung puts two learners into one settled net. The memory side is a lookup
from (sound, moment) to a cochlear frame: a sound's cue (its first 160 ms after lateral
inhibition) and a clock started at its onset wake, through fixed random conjunctions with a
winner-take-all, a few of 768 *sequence owners*, and one-way seams from those to 24
*expected-frame* owners are nudged toward the frame heard at that tick. The mirror side
reads the peaks, the dominant channel and the loudness of the last four frames through 96
hidden owners onto 16 motor owners, nudged, while the parrot babbles, toward the command it
issued a frame earlier. Nothing else changes: the same free settlement, the same centered
contrast, each update masked to its population's seams and owners (`trainable_overlaps`,
`trainable_owners`), each population with its own momentum, and a seam decay so that what
is not heard again fades. To sing, the clock runs from zero with the cue held, the memory
answers tick by tick, each answer is shown to the mirror as the latest frame, and the
mirror's command drives the syrinx.

What that rung found, in order. A memory that replays on its own expectations blurs within
a few frames; a clock and a held cue make the replay as crisp as the learning. Adding the
cue's and the clock's projections gives owners that answer to either alone; multiplying
them before the winner-take-all makes the code conjunctive, and rare sounds stop being
overwritten by frequent ones. A tied seam from the readout back into a clamped input layer
wakes the "silent" input owners a little, so every update moves every memory; one-way
seams keep them exactly silent. For a sparse code momentum only scales the step down (a
seam is touched once per hearing), and a plain step of 5 beats 0.3 with momentum; the
mirror, with dense inputs, keeps momentum 0.9 at 0.3. The readout must learn no bias, or
an unknown moment replays as the average sound. And the mirror has to read at test what it
saw in training: bump codes for pitch and loudness read the same for a replayed
expectation and for the cochlea's frame, and give silence a pattern of its own.

## What the rule does that a forward pass does not

A settled net has no fixed input side. Clamp any subset of owners and read any other: the
same trained net that predicts a label from features can be asked, with a partial clamp,
what the rest of the features would have to be. The Titanic field rung in the Kaggle set
measures it: one net whose feature owners all hang on the hidden owners by tied seams,
trained by masked reconstruction (clamp a random 70% of the columns, nudge the rest to
their true levels). Asked three questions, it answers all three from the same seams, and
each answer is a few points behind a model built for that question alone: survival 0.796
against 0.825 for the dedicated classifier, survival with 30% of the columns missing 0.770
against 0.813, age imputation 10.5 years mean error against 8.6 for a ridge regression.
That is the price of a joint model at this size; the capability itself needs no second
model and no retraining. Learning is likewise not a phase:
every prediction is a free settlement, every arriving label a nudged one, so a net can
learn from a stream one example at a time with the rule unchanged. The Digit Recognizer
stream rung measures that: one pass over 42,000 images, predict then learn, accuracy along
the way: 0.885 over the pass and 0.932 in the last window, against 0.844 and 0.895 for the
same-shape MLP taking one SGD step per chunk.
