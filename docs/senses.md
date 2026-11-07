# Wide inputs: images and codes

For wide inputs (an image, a long sensor vector, thousands of features), choose how the brain
reads them:

| Setting | What the brain reads | Section |
| --- | --- | --- |
| `inputs=(height, width)` | the image through a visual cortex | [image inputs](#image-inputs) |
| `code=units` | a fixed sparse code of the observation, which the memories key on and a vector brain's cortex reads | [the observation code](#the-observation-code) |
| `learned_code=units` (opt-in) | a code that a region of the brain's own neurons learns while the brain lives | [the learned code](#the-learned-code) |
| `attention=True` with `learned_code` | Recall selects a recent event and its successor by an ordered pair of learned codes | [single-head attention](memory.md#single-head-learned-code-attention-opt-in) |

```python
import numpy as np
from cadence import Brain
```

## Image inputs

`inputs=(height, width)` or `(height, width, channels)` makes the sensory region a
`visual_cortex`: `features` maps (8) of `field` by `field` receptive fields (3) feed
the first module. Observations may be flat `(1, height * width * channels)` or keep
the image shape. The maps project one way into the first module, so no nudge reaches them and the
contrast rule teaches them nothing. The receptive fields keep their initial draw, and
learning shapes what the modules read from the maps. Phases that stop short of equilibrium,
as a deep brain's founder phases can, move the fields by what they leave unsettled.

```python
eyes = Brain.compose(inputs=(8, 8), actions=3, features=4, seed=1)
assert eyes.act(np.random.default_rng(0).random((1, 8, 8))).shape == (1,)
```

The memories key on the pixels. With `code=units` they key on a fixed sparse code
and the cortex reads the pixels ([the observation code](#the-observation-code)).

## The observation code

Set `code=units` when wide observations make the episodic separators and the recall address
map too large: at 12,824 inputs they take 1,532 MB uncoded with `memory_choice` and `recall`,
and 8.9 MB with `code=1024`.

`brain.observation_code(x)` returns the key every memory reads and writes, by default the
observation itself. `code=units` (construction; at least 32 inputs and 24 units) replaces it
with a fixed random sparse code of `units` width and unit length, one unit in ten active,
saved with the brain as `brain.sparse_code`. The code is fixed and learns nothing.

- In `compose` a vector brain's sensory region reads the code too; on an image brain and in
  `Brain(connectome, code=units)` only the memories use it.
- Keys stored under one code do not match another.
- `Genome.observation_width(inputs, code)` gives the key width before building, the size of a
  custom anatomy's `recall` and `next` regions.

For a code the brain learns, use `learned_code=units` in place of `code`
([the learned code](#the-learned-code)).

```python
from cadence import Genome

plain = Brain.compose(inputs=2048, actions=3, seed=0)
coded = Brain.compose(inputs=2048, actions=3, seed=0, code=1024)
x = np.random.default_rng(0).random((1, 2048))
assert plain.observation_code(x).shape == (1, 2048)     # the observation itself
key = coded.observation_code(x)                         # what every memory keys on
assert key.shape == (1, 1024) and (key > 0).sum() == 102 and np.isclose(np.linalg.norm(key), 1.0)
assert Genome.observation_width(2048, 1024) == 1024     # the key width before building
assert len(coded.connectome.populations["sensory"]) == 1024  # the cortex reads the code
```

## The learned code

`Brain.compose(..., learned_code=units)` adds a `code` region of that many neurons between the
sensory region and the first module. The region learns a compressed code of the observation
by local rules at accepted explicit acts and aroused `live` moments; the cortex computes on
that code and every memory keys on it, joined to the observation's own identity. The founder
`learned_code=None` builds the brain without it and keys the memories on
[the observation code](#the-observation-code); `code=units`, the fixed random code, is the
control at matched width.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=53, actions=4, seed=0, learned_code=64)
token = np.eye(53)[[3]]                              # one token, its own coordinate
brain.act(token, greedy=True)                        # the code settles with the brain
report = brain.last_settlement
assert report["qualified"] and report["code_steps"] > 0    # the code phase, then the whole graph
key = brain.observation_code(token)                  # what the memories key on: the code
assert key.shape == (1, 64 + 53)                     # joined to the token's own identity
assert (key >= 0).all() and np.isclose(np.linalg.norm(key), 1.0)
code = brain.learned_code                            # the tables the region learns
assert code.dictionary.shape == (53, 64) and code.work["updates"] == 1
```

### What the code is for

A code learned from the contexts an observation occurs in places observations that occur in
the same contexts near each other, and what is taught about some of them reaches the rest.
In the embedding chamber (`benchmarks/learned_code/embedding.py`) members of a group are
heard in the same sentence contexts, lessons come only for some members, and the held-out
members are asked: a brain with the region answers them at 1.000 on 6 of 6 development
founders and 12 of 12 fresh ones, where the raw observation and the fixed random code give
an untaught member nothing in common with its group and answer at chance. The memories keep
their bindings: the key joins the code to the observation's own identity, so a binding stays
readable while the code moves, and the binding family (`benchmarks/binding/`) reads within
.05 of a brain without the region on every task.

The region has costs. It settles its own phase at every moment, twice at a moment it learns
from, and a life takes about three to six times the processor time
([cost](#save-load-describe-cost)). The code is dense, about 28 of 64 units active per
observation in context. A cue held across a delay through the episodic memory takes about
twice the experience to master.

Present wide images at unit length: the region reads a lit pixel as one full input, so a
frame at pixel scale overwhelms its units. An observation wider than the code keys the
memories on the code alone ([what the memories key on](#what-the-memories-key-on)). The camera
chamber (`benchmarks/learned_code/camera.py`) measures a separate `CodeRegion` prototype that
imprints its dictionary on the first frames, receives 256 nursery rounds, omits the top-down
boundary and passes the body inputs beside its code into a brain without `learned_code`. With
128 units it answers the held-out pairs at .970 against .927 for the fixed random code of
1,024 units on the development founders, with a recall address map of 0.198 MB against
9.331 MB and a stored dictionary of 13.1 MB. Those scores describe the prototype; camera
behavior of `Brain.compose(learned_code=...)` has no matched measurement.

### What the region is

The region's tables are the code units' synapses and biases:

| Table | Synapses | Role |
| --- | --- | --- |
| dictionary `Phi`, `inputs` by `units`, unit-length columns | from the sensory neurons to the code units, at the read gain `1 / f(A)` (`f` the neuron's activation, `A` the model's stimulus amplitude): the region reads the sensory activation in units of one full input, `x' = s_sensory / f(A)` | the bottom-up boundary: code unit `j` is driven by `Phi_j . x'`; one input at amplitude 1 drives the unit whose column matches it by one |
| the Gram matrix of the dictionary | every ordered pair of code units, `-(Phi_j . Phi_k)` | lateral competition: units whose columns overlap inhibit each other |
| predictor `W`, `units` by the association width | from the prefrontal region, scaled by the top-down weight | the top-down boundary: the working trace's prediction of the code, `p = W s_prefrontal` |
| thresholds `t`, one per unit | the units' biases, `-t_j` | how much drive a unit needs before it publishes |
| the standing boundary | the region's quadratic anchor toward zero at the top-down weight on its units, in its own phase (`LearnedCode.standing`) | the restoring term `-top_down * s_j`; with the predictor's drive it is the pull `top_down * (p_j - s_j)` toward the prediction |

A code unit settles under the brain's neuron model to

```text
v_j = Phi_j . (x' - Phi s) + s_j + top_down * (p_j - s_j) - t_j,    s_j = f(v_j),    x' = s_sensory / f(A),    p = W s_prefrontal
```

the sparse state that reconstructs the sensory activation, as read, through the dictionary
as far as the thresholds allow, drawn toward what the context predicts. The sensory neurons
project to the code region alone, and nothing leads back into it.

### How a moment settles

A moment's settle has a code phase first: the region's sub-network (its units, its lateral
synapses, its biases, the same neuron model) settles by the brain's solver policy to the
learner's tolerance, warm from the stream's code. At a moment the region learns from it
settles twice: in the stream's context (the predictor's read of the working trace), the code
the region learns from, and on the observation alone, the context-free code the cortex reads
and every memory keys on (reported as `context_free_steps`). The code is held: in the whole
settle its units are a source, their synapses in the whole graph carry nothing, and the
moment's drive holds them at the phase's fixed point, so the rest of the brain integrates at
the model's step with its budgets, its adaptation and its learning phases as a brain without
the region has them. `last_settlement` reports the phase as `code_steps` and `code_residual`
beside the whole graph's `steps`. `code_steps` and `context_free_steps` count every attempt,
an initial phase that failed before its fallback included; the reported residual describes the
final attempt.

At the founder standing gain of 2, the code's slope-weighted lateral coupling is nonpositive,
but a synchronous sweep at the model's step can oscillate when units with overlapping
columns are active together. After each phase the region reports the step's halvings for
those negative modes (`LearnedCode.stability`), and its next phase runs at that step with its
budget scaled by the same power of two. A phase that does not qualify is settled once more at
the step the dictionary's negative modes allow over every state (`LearnedCode.rest_stability`).
If it does not qualify there either, the moment is refused with `RuntimeError` before anything
changes, as a refused free solve is. A world of one input needs this from its first moment.
Local contraction also requires the largest slope-weighted coupling eigenvalue below one.
The expert `LearnedCode` constructor's step bounds include the neuron model's actual
synaptic gain and the standing anchor separately; `Brain.compose` uses unit synaptic gain.
The code phase's chosen step is provisional until the action, wait, lesson or feedback
that uses it is accepted. Refusal restores the previous step and latest accepted phase,
while attempted-work counters and refusal diagnostics may advance.
Custom controls such as `top_down=0` with a large negative-branch leak need that separate
check: damping cannot stabilize an intrinsically growing mode.

The top-down boundary reads the working trace, the settled association activity of the moment
before, and never the association region of the moment: a prediction read from the moment's
own association activity would copy the code it reads.

### How it learns

At every accepted `act` and `step` answer the region takes one step from the moment's
settle, each row of the batch one event. In `live` it learns only while aroused: routine
answers keep all its tables still. `wait` follows the issued action's mode, keeping the
tables still while a routine `live` action awaits its outcome and learning otherwise. The
lesson's nudged phases, the outcome's eligibility phases, the imagined branches and the
readings taken on copies enter no table, as the contrast and the dopamine enter no synapse of
the region. With `a` the rectified code in context, `a_free` the context-free code, `x'` the
sensory activation as read, `z` the prefrontal activation and `p = W z`:

| Table | Rule | Constant |
| --- | --- | --- |
| dictionary | `Phi += rate * sum_rows (x' - Phi a) a^T`, each column scaled back to unit length | `RATE` 0.125 |
| lateral weights | recomputed from the dictionary | with the dictionary |
| predictor | `W += sum_rows r_row (a - p) z^T / max(1, rows * predictor)`, `r_row = predictor / max(\|z\|^2, predictor^2)`: context energy normalizes each row's gap, and the batch divisor bounds simultaneous updates | `PREDICTOR` 0.0625 |
| thresholds | `t += threshold_rate * sum_rows ((a_free > active) - target)`: a unit's share is the share of the observations that drive it on their own above the active floor; floored at 0, no ceiling; the repair: a row whose code leaves at least the share `repair` of its input's energy unexplained lowers the threshold of every unit silent in it by `threshold_rate * silence` | `THRESHOLD_RATE` 0.000625, `TARGET` 0.1, `ACTIVE` 0.05, `REPAIR` 1.0, `SILENCE` 10 |

The predictor's batch divisor prevents repeated contexts in a large batch from amplifying
the prediction error: the sum of the normalized context outer products has trace at most
`rows * predictor`, so the divided update has eigenvalues at most one. At the founder rate,
batches of at most 16 rows, including the embedding chamber's, retain the same update.

While the code fits, the residuals are small and nothing moves. The lesson's contrast and the
dopamine leave the region's synapses and biases alone (the learner's plastic masks exclude
them), and lessons, `fit`, `predict` and `imagine` take no step. Two attributes of the region
are for readings and controls: `brain.learned_code.learning = False` holds every table still
(a reading on a copy), and `brain.learned_code.top_down = 0` takes the prediction off the
region (the control without the top-down boundary); a deep copy and a checkpoint keep both.

```python
teacher = Brain.compose(inputs=53, actions=4, seed=1, learned_code=64, recall=True)
sentence = [np.eye(53)[[k]] for k in (40, 3, 44)]   # a context, a token, a context
for x in sentence:
    teacher.act(x, greedy=True)                      # three moments, three steps
assert teacher.learned_code.work["updates"] == 3
assert teacher.recall.keys.shape == (1, 4, 64 + 53) # a boundary slot, then the moments' keys
teacher.lesson(sentence[1], [2])                     # a lesson moves the plastic synapses alone
assert teacher.learned_code.work["updates"] == 3
teacher.learned_code.learning = False               # the tables stay from here on
```

### Settings

| Setting | Founder | When | What it sets |
| --- | --- | --- | --- |
| `learned_code` | `None` | construction | the region, of that many units (at least 2); `None` builds nothing |
| `learned_code_fields` | `None` | construction | the widths of the observation's fields (a sequence summing to `inputs`; `None` is one field): the units are shared among the fields in proportion to their widths (at least two each), each unit reads its field alone, and the competition among the units is block-diagonal |

Every other part of the region is a constant of it, the value the readings selected:
`LearnedCode.TOP_DOWN` 2.0 (the prediction's weight on the code units and the gain of the pull
toward it), `RATE` 0.125, `PREDICTOR` 0.0625, `THRESHOLD` 0.1 (every unit's initial
threshold), `THRESHOLD_RATE` 0.000625, `TARGET` 0.1, `ACTIVE` 0.05, `REPAIR` 1.0, `SILENCE` 10
and `IDENTITY` 0.9 (the weight of the observation's own identity in the memories' key).

Refused with `ValueError`: `learned_code` below 2, `learned_code_fields` without the region,
fields that do not sum to the inputs or leave a field fewer than two units, `code` beside
`learned_code`, an image brain, the `torch` backend, and a custom connectome without a
`code` population and the complete dictionary, lateral and predictor synapses, or whose
sensory neurons project anywhere but to the code region.

### When to declare fields

With one field (the founder, `learned_code_fields=None`) every unit reads the whole
observation, and the region learns codes of whole moments: observations that look alike
as a whole share units, so a member of a group is read as its group's members are, and a
lesson on some members reaches the rest. That is what a world of single tokens or of
whole scenes wants.

When the observation is a combination of independent channels, declare one field per
channel: the coordinates of a sentence's word slots, of who acts, what is held and where,
of a pouch beside the cell a creature stands in. Give `learned_code_fields` the channels'
widths in the order of the observation's coordinates (they sum to `inputs`). Each unit then
reads one channel, the units of a channel compete among themselves alone, and a moment's
code is the union of its channels' codes. A combination never seen before is read as
channels the brain knows, and a second input in another channel keeps its own code. With one
field the same combination arrives as an unfamiliar whole, and a second input is merged into
the code of the first, distinguished only by the code's strength.

```python
from cadence import Brain

# 8 coordinates of where, 4 of what is held: a held key keeps its own units
brain = Brain.compose(inputs=12, actions=4, seed=0, learned_code=32, learned_code_fields=(8, 4))
assert brain.learned_code.field_units == (21, 11)  # the units shared by the channels' widths
```

Declare fields for the channels the world fires. A field keeps its share of the units (by
width, at least two) for the whole life, and its units read its coordinates alone, so a
field whose coordinates never fire leaves its units silent: the repair lowers a silent
unit's threshold, which recruits nothing in a unit that receives no input. Coordinates the
world never fires belong in the field of a channel it fires.

### What the memories key on

With the region, every memory keys on the context-free code: the observation settled alone
with a zero prediction, the rectified code at unit length that `brain.observation_code(x)`
returns, joined to the observation's own identity, `unit([sqrt(1 - w) code, sqrt(w)
unit(x)])` with `w` the constant `IDENTITY` (0.9), and taken at the norm the observation
itself would have as a key. `brain.code_width` is `units + inputs`. The identity part keeps a
binding readable while the code part moves as the region learns, and a known situation is
found again along a new path. An observation wider than the code (more inputs than units, a
camera's frame) is keyed on the code alone, at the observation's norm, so that the key keeps
the code's width (`LearnedCode.joins`), and `brain.code_width` is then `units`. With
`memory_choice=True` the episodic stores separate these keys as an observation's
(1,024 cells, 24 active). A decision
keeps the key it was taken under until its outcome is recorded. `stimulus(..., memory=False)`,
`fit`, `predict` and `accuracy` settle the context-free code alone. The heard history holds
the code, in blocks of `units` per decay.

### Save, load, describe, cost

A brain with the region saves, in checkpoint format `cadence-brain/1`, the `learned_code`
metadata (units, widths, seed, fields, the step level, the two switches, work counts) and the
tables `learned_code/dictionary`, `learned_code/weights` and `learned_code/thresholds`, with
`moment/code` and `lived/code` for a decision awaiting its outcome. `Brain.load` installs the
saved tables and refuses a checkpoint whose saved synapses disagree with them. `describe()`
reports the layout's `code` with kind `"learned"`, `learned_code_fields` among the genes and
the layout's `learned_code`; `parameters()` counts the three tables; `reset()` leaves them as
it leaves the weights.

The region adds `units` neurons, `inputs * units` dictionary synapses, `units * (units - 1)`
lateral synapses and `width * units` predictor synapses, and replaces the `inputs * width`
sensory synapses of the first module by `units * width`: `compose(53, 4, learned_code=64)` has
20,224 synapses against 8,000 without the region. Each moment costs the code phase's sweeps
(about 33 per phase in the embedding chamber, twice at a moment the region learns from, and 34
to 120 per moment in the small worlds of continuous interaction) and one learning step of
`2 * inputs * units + units * width` multiply-adds. In the embedding chamber a life costs about
three times the processor time of the brain without the region; in the small worlds of
continuous interaction (the odour nursery, the key door, the cue task) a moment costs about four
to six times the time. The dictionary is stored dense, `inputs * units` values, with fields as
well.
