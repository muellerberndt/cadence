# Task recipes

How each kind of task becomes a clamp, a target, and a readout. Every recipe below is a
runnable rung with a receipt in cadence-examples or in the Kaggle set; the numbers there
say how each fares against the usual models on the same data.

| task | clamp | nudge and target | readout | where |
|---|---|---|---|---|
| classification, dense input | levels in [0, 1], one owner per feature (pixels) | cross-entropy toward the label's owner | most active output owner | digits, MNIST, Digit Recognizer |
| classification, tabular | place codes for numbers, one-hot for categories, a missing owner per column | cross-entropy | most active output owner | Titanic, Spaceship Titanic |
| regression | as above | quadratic, on one output owner, target scaled to [0, 1]; a linear calibration on out-of-fold outputs | that owner's activation, mapped back | House Prices |
| multi-label (a chord) | one owner per pitch per beat | quadratic on a multi-hot target | every owner above a validation-chosen threshold | chorales |
| text, bag of words | binary presence of the top tokens | cross-entropy | most active output owner | Disaster Tweets |
| text, windowed | one-hot blocks per position | cross-entropy | most active; sample to write | text |
| imitation of a teacher | the state, as pixels | cross-entropy toward the teacher's move | most active legal move | Connect Four, sign writer |
| from reward | the state, as pixels or a place code | cross-entropy toward the action taken, weighted by its advantage | draw from the softmax; greedy to evaluate | Pong, cart-pole |
| a measured wiring | sensory owners at a declared amplitude | quadratic on the motor pattern of a fact, or none: settle and score | a declared readout set | *C. elegans* |

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

One output owner, its target the value scaled to [0, 1] on training rows, the nudge
`beta · (target − s)`. The free activation is not the prediction directly: fit `a · s + b`
on out-of-fold predictions and map back through the scaling. That keeps every choice on
training rows and makes the readout as calibrated as the data allow.

## Wide sparse inputs

Thousands of mostly-silent one-hot owners (a bag of words, a wide window of characters)
are the input the rule likes least: with fan-scaled seams the drive from a few active
owners is small, and learning is slow relative to the same-shape MLP. Normalising the
clamp so that each row's total input is constant helps a little; a first layer of seams
shared across positions (an embedding) is the change the text rungs point at. `embedded`
builds that wiring and `Learner(tie_groups=...)` keeps the shared seams equal: every
position reads one embedding table, learned by the same local rule (the mean of the tied
seams' contrasts is still a function of those seams' own endpoints).

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
