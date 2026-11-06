# Memory

A memory is anything the brain keeps from an earlier moment and can use in a later answer.
Everything else is computed again from the current observation. This page starts from the
situation: find yours in the table, write what it says, and read the section for how that memory
works, its settings, its readings and its limits.

<a id="start-with-the-need"></a>

## What to use

| The situation | Write | Section |
| --- | --- | --- |
| The present observation holds the answer | `recall=False, episodic=False, working_memory_amplitude=0.0` is the control to compare against; add one memory at a time | [check that a memory does the work](#check-that-a-memory-does-the-work) |
| A moment heard once has to return by its content: a name stated with a value, look back, n-back, copying | `recall=True` | [recall memory](#the-recall-memory-in-full) |
| Whole sentences or stories, read at lengths no lesson reached | `recall=True, learning_surprise=True, teaching="deep", recall_recency=0.0, recall_boundary=True, recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01, recall_relative=True` | [one composition for sequences](#one-composition-for-sequences) |
| Two or more facts in one answer | `recall_heads=2` | [several heads](#several-heads-opt-in) |
| A read that depends on what another read found | a "think" moment between the two reads | [several heads](#several-heads-opt-in) |
| "How long ago?", "was A told before B?" | `recall_when=True, recall_position_rate=0.0, recall_temperature=0.1` | [when a read item happened](#when-a-read-item-happened-opt-in) |
| The answer is the token the read found, tokens no lesson used included | `recall_tokens=..., recall_gain=1.0` | [the learned value path](#the-learned-value-path-opt-in) |
| What it did itself, where the observations are alike | `efference_amplitude=1.0` for its last command; `recall_action=True` to find commands with their moments | [efference copy](#efference-copy), [its own actions](#its-own-actions-in-the-slots-opt-in) |
| A cue carried across a short delay, with no recall memory | nothing: the working trace is on | [working trace](#working-trace) |
| One cue held over a delay and replaced by a later one | no measured memory: the trace alone does not serve it | [working trace](#working-trace) |
| Which action pays in a situation, with quick reversal | nothing: the associative store is on | [associative store](#associative-store-the-hippocampus) |
| Noisy or signed pay | `memory_rate=0.2` | [associative store](#associative-store-the-hippocampus) |
| A demonstration remembered once, or a chain paid at its end | `memory_choice=True` | [episodic memory](#episodic-memory-what-happened-in-this-situation-before) |
| A paid action stops paying and its failures pay 0 | the store alone: leave `memory_choice` off; beside the vote, `memory_extinction=0.5` | [extinction](#a-paid-action-that-stops-paying-extinction) |
| Stages that look alike, where a miss changes nothing | `memory_context_hold="unchanged"` beside `memory_choice=True` | [the context hold](#stages-that-look-alike-the-context-hold) |
| The same observation needs different answers after different earlier observations | `history_decays=(0.2, 0.8)` | [heard history](#heard-history) |
| An action paid a few moments later; a chain whose stage is observed | nothing: the actor's eligibility is on | [learned weights and eligibility](#learned-weights-and-eligibility) |
| A wide observation | `code=units` bounds every key | [the observation code](senses.md#the-observation-code) |
| Keys that overlap where meanings do | `learned_code=units, attention=True` | [single-head attention](#single-head-learned-code-attention-opt-in) |
| Types taught one after another | `learning_rehearsal=1.0` | [rehearsal](teach.md#teach-several-types-in-sequence-rehearsal) |
| A memory for a brain trained without it | `brain.grow(recall=True)` | [grow and combine brains](grow.md) |
| Situations whose keys overlap strongly, outside a brain | a standalone store behind a centered, habituated separator | [pattern separation](#pattern-separation) |

Add one mechanism at a time, and compare it with the same mechanism silenced or erased. Every
memory changes the answer through the same composed brain: the working trace, the efference copy,
the heard history and the recall memory drive neural regions the association region reads; the
associative store drives motor neurons; the episodic records add a vote to the motor readout. The
answer has to [qualify](start.md#qualification-and-refusal) whichever of them shaped it.

<a id="memory-in-the-composed-brain"></a>
<a id="complete-memory-inventory"></a>

## What each memory holds

| Memory | On by default | Switch | Holds | Keyed by | Written | Survives `reset()` | Home |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Working trace | yes | `working_memory_amplitude=0.0` silences it | the settled association activity of the moment before | none | after each act and each moment `wait` senses | no | [below](#working-trace) |
| Efference copy | no | `efference_amplitude=1.0` | a fading one-hot of the commands issued | none | after each issued act | no | [below](#efference-copy) |
| Heard history | no | `history_decays=(0.2, 0.8)` | the observations as heard, one fading block per decay | none | after each act and each moment `wait` senses | no | [below](#heard-history) |
| Associative store | yes | `episodic=False` removes it | one value per action per situation, in reward units | observation code | each outcome of the chosen action | yes | [below](#associative-store-the-hippocampus) |
| Episodic memory | no | `memory_choice=True` | place and context records that vote in the choice | observation code, and code with the temporal context | known outcomes, demonstrations, replay of the path to a reward | records yes, context no | [below](#episodic-memory-what-happened-in-this-situation-before) |
| Recall memory | no | `recall=True`; `attention=True` selects its learned-code contextual recipe | the stream's recent moments as slots, and a learned address map | observation code | a slot per moment; the map at lessons | map yes, slots no | [below](#the-recall-memory-in-full) |
| Learned code | no | `learned_code=units` | a dictionary, a predictor and thresholds: the code region's synapses and biases | none | every accepted moment, by local rules | yes | [the learned code](senses.md#the-learned-code) |
| Learned weights | yes | learning rates of zero freeze them | synapse scales and biases, actor and critic | none | lessons, and outcomes (aroused ones under `live`) | yes | [the free/nudged rule](settling.md#learning-the-freenudged-rule) |
| Actor eligibility | yes | `actor_lam=0.0` credits the last action only | each synapse's credit for recent sampled acts | none | each sampled act | no | [reward.md](reward.md#eligibility-traces-and-dopamine) |

`predict`, `accuracy` and `fit` answer independent samples and read none of these except the
learned weights; `fit` starts by resetting the streams and keeps the arousal state. Test a
memory through `act`, `step` or `live` on a continuing stream. `save` keeps every memory in
the table, per-stream traces included, and `Brain.load` continues the stream where it
stopped.

`brain.memories` holds the present memories by name, in the order the brain reads them before
a settle and writes them after it: `working_memory`, `efference`, `history`, `recall`,
`hippocampus` and, with `memory_choice=True`, `episodic` (the episodic records: their vote,
the temporal context and the paths a reward is replayed along); `brain.memories["recall"]` is
`brain.recall`. Every member keeps one contract (`cadence.memories`):

- `stimulus(drive, moment)` adds what it holds to the drive of a moment;
- `branch()` gives an imagined branch a private copy;
- `reset(batch, rows=None)` and `keep(rows)` act on streams (`Brain.reset` begins every stream
  afresh this way, and the stores keep their records);
- `to_dict()` and `arrays()` are what a checkpoint keeps of it, under `memories/<name>/`.

Stores used directly, outside a composed brain (`FastSynapses`, `PatternSeparator`,
`SynapticMemory`, the `Trace` family), are listed
[at the end](#standalone-memory-stores).

Memories have three lifetimes:

- **Per stream**, cleared by `reset()` and by an ended episode: the working trace, efference
  copy, heard history, recall slots and actor eligibility.
- **Across resets**: the associative and episodic records and the recall address map.
- **Permanent until learning changes them**: the learned weights.

The per-stream traces advance at every moment. Under
[`live`](start.md#routine-and-repair-live) a calm brain reads its records and weights and
changes neither; an aroused one learns and writes. `reset()` keeps the arousal state,
including the income and error references that decide when a `live` brain wakes;
`reset(keep=())` clears them ([reset and save](start.md#reset-and-save)).

## Check that a memory does the work

Save the acquired brain, load two copies, erase one memory in one copy and compare their
answers on the same query. If they never differ, the brain answers without that memory.
Here the actor is frozen, so only the records can change the answer:

```python
import os
import tempfile

import numpy as np
from cadence import Brain

frozen = {"actor_eta": 0.0, "actor_eta_bias": 0.0, "actor_eta_critic": 0.0}
paid = Brain.compose(inputs=3, actions=2, memory_choice=True, seed=0, **frozen)
x = np.array([[1.0, 0.0, 0.0]])
assert paid.act(x, greedy=True)[0] == 0          # the newborn's answer
action = paid.step(x)
for _ in range(6):                               # only action 1 pays
    action = paid.step(x, reward=[1.0 if action[0] == 1 else 0.0])
path = os.path.join(tempfile.mkdtemp(), "brain.npz")
paid.save(path)
remembering, forgetting = Brain.load(path), Brain.load(path)
forgetting.hippocampus.clear()                   # erase the place records
forgetting.context_memory.clear()                # and the context records
assert remembering.act(x, greedy=True)[0] == 1   # the records choose the action that paid
assert forgetting.act(x, greedy=True)[0] == 0
```

If the answers never differ, the cortex has consolidated what the records held, or they hold
nothing for that key.

| Memory | Erase it on a copy |
| --- | --- |
| Any member of `brain.memories` | `brain.memories[name].reset(1)` empties its stream: the trace, the history, the recall memory's slots, the associative store's `F`, the episodic context and path |
| Associative store | `brain.hippocampus.clear()` erases `C` too |
| Episodic memory | `brain.hippocampus.clear()` and `brain.context_memory.clear()`, as above |
| Any memory read | `brain.retune(<memory>_amplitude=0.0)` silences it and keeps what it holds |

<a id="the-recall-memory-in-full"></a>

## Recall memory

`Brain.compose(..., recall=True)` keeps each stream's recent moments as slots and gives one
back when the present asks for it. Use it when a moment heard once has to return by its
content (a name stated with a value, asked for later), or when the brain needs what followed
an earlier moment that matches the present. The founder `recall=False` is the control. Each
slot holds a key (the event as heard), a value (the event) and a next (the value of the moment
that followed). Two source regions, `recall` and `next`, carry the recalled value and next into
the association region.

Every head also reads how long ago each slot was written. This position term is on by default:
it starts at zero, so a new brain reads exactly as it would without it, and lessons teach each
head how far back to look, depending on what the moment hears
([position](#position-how-long-ago-a-moment-happened); `recall_position=0` turns it off).
`reset()` empties the slots and keeps the learned address map; `reset(keep=("slots",))` keeps
both. `attention=True` selects the memory's learned-code recipe
([single-head attention](#single-head-learned-code-attention-opt-in)).

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=4, actions=2, modules=(16,), seed=1, recall=True)
fact = np.array([[1.0, 0.0, 0.0, 1.0]])          # a name (0) stated with a value (3)
brain.act(fact, greedy=True)                     # each act writes one slot per stream
brain.act(np.array([[0.0, 1.0, 0.0, 0.0]]), greedy=True)
name = np.array([[1.0, 0.0, 0.0, 0.0]])
drive = brain.stimulus(name)                     # the name addresses the slots before its settle
recalled = drive[0, brain.connectome.populations["recall"]]
assert recalled[3] > recalled[1]                 # the moment that shares the name comes back
report = brain.lesson(name, [1])                 # a lesson also steps the address map
assert report["address_rows"] == 1.0
brain.reset()                                    # slots emptied, the learned address map kept
assert brain.recall.slots == 0 and brain.recall.head_map(0)[0].any()
```

### How a moment is addressed

Before each settle the present moment computes its address

```text
q = x + x W_x + c W_c + b
p = softmax(q . K / temperature - recency * age + a(x, c)[bucket(age)])
```

`x` is the observation as the memory hears it, `c` the settled association activity of the
moment before, `K` the stored keys, and `W_x`, `W_c`, `b` a learned map that starts at zero, so
before any lesson the address is the key and a name finds the moment it was stated in. `a` is
the [position term](#position-how-long-ago-a-moment-happened), also zero until a lesson moves
it. `p` is kept in `brain.recall.attended_heads`, one array per head. The `recall` region is
driven at `amplitude * sum p V` and the `next` region at `amplitude * sum p N`, held through
the settle as a sensory drive is: the settle answers on the mixture the address chose and
cannot revise the choice. Both regions reach the association region through one-way plastic
synapses of initial magnitude 1.2 (a constant).

### How it writes and how the address learns

After each act, and after each moment `wait` senses, every stream writes one slot and the
slot before learns the new value as its next. Beyond `recall_capacity` the oldest slot is
dropped. Imagination writes none.

The address lesson runs after each teacher lesson (`step(..., teacher=...)` or
`brain.lesson`). It reads the association region's movement between the nudged phases back
through the recall and next synapses, scores each slot's value and next against it, weighs
the scores by `softmax(score / T + log p)` at the temperature `T = 0.1`
(`Recall.address_temperature`, a constant of the memory) and steps the map on the keys so
weighted less the keys the read attended,

```text
error = sum_i (p*_i - p_i) K_i
```

at the taught rows, each at the rate `min(recall_address, 1 / |z|^2)` on `z = [x, c, 1]`. The error
is the gradient of the content logits, zero where the read is what the lesson names, as the position
maps' step is. The report gives `address_rows`, `address_error` and `address_refused` (rows whose
non-finite step was refused), and with the position term `position_rows` and `position_refused`.

At the founder `recall_address_reward=0` rewards leave the map as it is. Above zero, each
sampled act keeps the step the address lesson would take on the rows of the event heard, were
its action a lesson's label, as the map's eligibility; the rows of the cortex of the moment
before and the bias, which every moment's address shares, learn from lessons alone
([position](#position-how-long-ago-a-moment-happened) gives the reason). The actor holds the
eligibility in a trace of its own (decaying by `gamma * lam`, see [reward](reward.md)), and
each outcome steps the map by `recall_address_reward` times the dopamine times that trace;
`learn` reports it as `address_reward_step`. At 1.0 a unit of dopamine moves the event's rows by one lesson's step.
`retune` refuses a change of `recall_address` or `recall_address_reward` while a sampled
action's address contrast is pending.

### Episodes, keys and windows

Each stream has its own slots. A plain `reset()` empties them and keeps the address map. When
an episode ends, that stream's slots are emptied with its working trace and heard history,
and the cortex of the moment before, the history and the window begin afresh.
`brain.reset(keep=("slots",))` keeps the slots across a reset: each stream ends an episode
there (`Recall.end`), and the ended episode's last slot learns no successor from the next
stream's first moment.

**The episode's boundary (opt-in, `recall_boundary=True`).** A slot's next is taught by the
moment that follows it, so every event but the first has a predecessor whose next holds it, and
a read can find an event through two ports: its own slot's value, or the slot before's next.
The first event of an episode has only the first. A brain that learns to answer one cue through
the next port then has no route to the start of an episode. With the option each stream that
begins holds a boundary slot first, a silent moment of zero key and zero value whose next the
first event teaches, so both routes exist for every position. A stream begins when the memory
empties its streams, when some of them are reset (their boundary takes the latest slot, the
moment the others wrote last) and when every stream ends an episode with its slots kept. A
stream that ends its episode alone begins the next one without a boundary, and a memory of one
slot (`recall_capacity=1`) holds none. The boundary is no written moment, and its logit is its
age's alone: a cue finds it through the position term, or with an address that points away from
every stored key.

The founder key is the present event alone, so where an event recurs with different
continuations the read returns a mixture of every occurrence's successor. One construction
setting widens the key:

| Key | Holds | Address width |
| --- | --- | --- |
| the founder key | `x_t` | `code` |
| `recall_window=k` | the ordered last k events, each at its full weight | `k * code` |

`code` is the width of the [observation code](senses.md#the-observation-code), and
`brain.recall.address_width` reports the result. Keys, address, map outputs and a reward
eligibility take that width; values, nexts and the two regions keep the value width.

### Recall settings

Each gene is passed as `recall_<name>` beside `recall=True`; a `recall_` name without it is
refused. Retune genes change in a life through `brain.retune`; construction genes are fixed
when the brain is built. The options from `recall_address_reward` on are described and saved
only where they differ from their founder. A saved brain keeps its genes, slots and address map.

| Setting | Founder | When | What it sets |
| --- | --- | --- | --- |
| `recall` | `False` | construction | Adds the memory and its `recall` and `next` regions |
| `recall_amplitude` | `3.0` | retune | Drive of a slot that holds the whole weight |
| `recall_temperature` | `0.25` | retune | Softmax temperature over the slots |
| `recall_recency` | `0.25` | retune | Logit cost per moment of age |
| `recall_address` | `0.01` | retune | Address lesson rate, bounded per row by `1 / \|z\|^2`; `0.0` freezes the map |
| `recall_capacity` | `64` | construction | Slots per stream, oldest dropped first |
| `recall_next` | `True` | construction | The `next` region and each slot's next |
| `scales["recall"]` | `1.2` | construction | Initial magnitude of the two projections (a `compose` argument) |
| `recall_address_reward` | `0.0` | retune | Rate at which reward trains the address map |
| `recall_position_reward` | `0.0` | retune | Rate at which reward trains the position maps, and with the anchor its maps ([position](#position-how-long-ago-a-moment-happened)); it needs the position term and `recall_position_rate` above 0 |
| `recall_window` | `1` | construction | Window of the last k events |
| `recall_heads` | `1` | construction | Heads reading the one bank, each with its own port and address map ([several heads](#several-heads-opt-in)) |
| `recall_position` | `32` | construction | Ages below it get a bucket each, older ones one per doubling; every head learns a term per bucket ([position](#position-how-long-ago-a-moment-happened)); `0` builds none |
| `recall_position_rate` | `0.1` | retune | Rate of the position term's lesson step; `0.0` freezes it |
| `recall_when` | `False` | construction | Every head's read also delivers when its slots happened, on a region of its own ([when a read item happened](#when-a-read-item-happened-opt-in)) |
| `recall_gain` | `0.0` | construction | One learned scalar per read port on the tokens' identity into the motor region's drive, starting at zero and taught at this rate; needs `recall_tokens` ([the gain](#the-learned-value-path-opt-in)) |
| `recall_slope` | `0.0` | construction | Above 0, every head's position term also leans by age at every length, taught at that rate ([the slope](#positions-at-lengths-no-lesson-reached)); needs `recall_position` |
| `recall_relative` | `False` | construction | Every head's position term also reads each slot's age relative to the slot the previous read attended most ([the relative read](#positions-at-lengths-no-lesson-reached)); needs `recall_position` |
| `recall_anchor` | `False` | construction | Every head's position term also reads each slot's place in its episode, so "the first told" holds at every story length ([the anchor](#positions-at-lengths-no-lesson-reached)); needs `recall_position` |
| `recall_boundary` | `False` | retune | Each stream that begins holds a silent boundary slot whose next the first event teaches ([the episode's boundary](#episodes-keys-and-windows)); a retune applies from the next stream that begins |
| `recall_copy` | `0.0` | construction | Above 0, plastic synapses from every read port into the motor region at that initial magnitude ([the copy pathway](#the-copy-pathway-opt-in)) |
| `recall_tokens` | `None` | construction | Per action the observation coordinate of the same token (`-1` for none); the pathway starts as their identity, echo suppressed; needs `recall_copy` |
| `recall_action` | `False` | construction | Every slot carries the executed command in its key and value, and the maps read the command of the moment before ([its own actions](#its-own-actions-in-the-slots-opt-in)) |

The [API reference](api.md#recall-memory-genes) gives the `Recall` class signature.

### Position: how long ago a moment happened

A key says what a moment was; nothing in it says when. Two moments with the same event and
the same key differ to a read only by the recency charge, which always prefers the newer one,
so a question that depends on order ("at the beginning", "now", "the same beat one bar back")
cannot be answered from keys alone. Every head therefore reads each slot's age, the moments
since the slot was written, and adds a term to that slot's logit:

```text
logit_i = q . K_i / temperature - recency * age_i + a(x, c)[bucket(age_i)]
a(x, c) = x P_x + c P_c + p        (one map per head)
```

- **Buckets.** Ages below `recall_position` P (founder 32) have a bucket each; older ones share
  one bucket per doubling (`[P, 2P)`, `[2P, 4P)`, ...) up to the capacity.
  `Recall.position_count(P, capacity)` gives their number.
- **Read from the moment.** The term comes from the event heard `x` and the cortex of the
  moment before `c`, the same inputs the address map reads. A question word can therefore
  choose how far back its head looks: one head can look back to the oldest record of a name
  when it hears "beginning" and to the newest when it hears "now". A fixed preference per head
  could not serve both.
- **Zero at birth.** The maps start at zero. Until a lesson moves them, a brain reads exactly as
  one built with `recall_position=0`.
- **Taught by lessons.** After each lesson the address lesson's target, the slots whose stored
  events the lesson's movement names, gives each taught row the gradient of its terms toward
  that target, and each head's maps take a bounded least-mean-squares step at
  `recall_position_rate`. This happens whatever `recall_address` is: a frozen address map does
  not freeze the position term.
- **Trained by reward on request.** At `recall_position_reward` above zero, each sampled act
  keeps the step the position lesson would take, were the act's action the lesson's label, in
  the eligibility the actor traces, and dopamine steps the maps by `recall_position_reward`
  times one lesson's step at the founder's rate of 0.1, whatever `recall_position_rate` the
  lessons take (with the anchor, its maps too): the recipe's slow table (0.01) left the reward
  step a tenth as large, and the read learned order from reward on 3 of 6 development founders
  against 5 of 6 at the founder's step. The reward step moves only the rows
  of the event heard, in the position, anchored and address maps alike, each row's rate bounded
  by the event's energy; the rows of the cortex of the moment before and the bias, which every
  moment shares, learn from lessons alone. Stepped by reward, those shared rows would carry one
  question's learned look to every question asked after the same moment.

**Order from reward alone.** In the
[ordered recall world](../benchmarks/sequence/README.md#ordered-recall-orderpy) (two values
written, the first or the latest asked, reward through `live`) the founder answers at chance.
The benchmark's `reward_recipe`:

```text
recall_position_reward=4, recall_address_reward=1, recall_anchor=True, recall_next=False
```

answers it at 1.00 of the last 256 answers on 11 of 12 fresh founders. The
[anchor](#positions-at-lengths-no-lesson-reached) finds the first write at the same place in
the episode whether a gap falls between the writes or not; `recall_next=False` holds each
payload on one port, so the two cases of one question settle on the same port; the address
reward finds the writes by what they hold. Its costs: in the recall world, where the latest of
one or two values is asked, the two-write conditions fall below the founder (the latest of two
adjacent writes 0.38 against 0.58), and it fails the
[preservation panel](../benchmarks/credit/README.md#preservation): the odour nursery passes 9
lives of 12 against 11 through `live` and 4 against 7 through `step`. Use it only where order
must be learned from reward.

```python
brain = Brain.compose(inputs=8, actions=3, modules=(16,), recall=True, seed=0)
assert brain.recall.position == 32                       # on by default
assert not any(m.any() for maps in brain.recall.position_maps for m in maps)
plain = Brain.compose(inputs=8, actions=3, modules=(16,), recall=True, seed=0,
                      recall_position=0)                 # the memory without it
```

The maps are saved with the brain when the term is on.

In the [look-back benchmark](../benchmarks/attention/look_back.py) a stream of five random
symbols is followed by a cue, `first` or `last`, and the answer is the symbol at the cued place.
Taught on the cue moment alone, the term learns to look back to the start of the episode on
`first` and to the newest slot on `last`. With `recall_boundary=True` every founder answers both
cues at 1.00, against 0.29 to 0.80 per cue without the term and 0.25 for uniform random
([receipt](../benchmarks/attention/results/look-back.json)); without the boundary `first` can
fall short on some founders.

### Positions at lengths no lesson reached

**The anchor (opt-in, `recall_anchor=True`; it needs the position term).** An age counts back
from the present, so the first moment of a story has a different age in every story of another
length: a head that learned "the first told" on stories of 12 reads the wrong slot in stories of
36. With the anchor every head's term also reads each slot's place in its stream's current
episode, the moments from the episode's first to the slot:

```text
logit_i += e(x, c)[anchored(place_i)]      e(x, c) = x E_x + c E_c + e   (one map per head)
```

The anchored buckets follow the age buckets, and one more holds every slot written before the
episode began. Each stream counts the moments written in its episode (`brain.recall.told`);
`Recall.end`, a reset and `done` start it again. The maps start at zero, so a newborn reads as
without the option; lessons step them with the position maps, and rewards with
`recall_position_reward`. On stories of 24 to 60 events with the cues `first`, `second` and
`last` mixed ([sequence battery](../benchmarks/sequence/README.md)), the anchor answers `first`
and `second` at 0.99 to 1.00, against 0.31 to 0.45 without it; induction, n-back, periodic
continuation and copying read the same with and without it.

**The relative read (opt-in, `recall_relative=True`; it needs the position term).** Some reads
are placed where an earlier read pointed: copying a sequence, the token after the one just
said is the slot written after the slot the previous read found, whatever the length. With the
option every head's term also reads each slot's age relative to the slot the stream's previous
read attended most (`brain.recall.focus`, the age of that slot, which ages with every written
moment): offset 0 for that slot, -1 for the one written after it, +1 for the one before,
offsets beyond `Recall.RELATIVE` (8) sharing the outermost buckets, and one more bucket for a
stream with no read before. The maps start at zero and the lesson steps them with the position
maps. Copying sequences with repeated tokens, taught at lengths 3 to 6 and tested at 8 to 12,
the relative read answers 0.54 to 0.74 of the tokens in closed loop against 0.10 to 0.14
without it; with the recipe below it answers 0.92 to 0.94.

**The slope (opt-in, `recall_slope` above 0; it needs the position term).** The table holds one
term per age bucket, and an age no lesson reached keeps its zero: a head taught to look back to
the first of 5 to 12 symbols does not look back past 12. With the slope every head's term also
leans by age at every length:

```text
logit_i += s(x, c) * age_i / 8      s(x, c) = x S_x + c S_c + s   (one map per head)
```

One number per moment says "lean old" (positive) or "lean recent" (negative), and the table
keeps the exact offsets beside it. The maps start at zero, so a newborn reads as one without
the option, and the lesson steps them by the table's gradient read along the ages, at the rate
`recall_slope`; rewards do not train them.

**The recipe for lengths no lesson reached.** Combine the anchor, the slope at 0.1 and the table
taught slowly beside it, `recall_position_rate=0.01` (with `recall_recency=0` and the boundary;
[sequence battery](../benchmarks/sequence/README.md), 1,500 lessons):

```python
long_stories = Brain.compose(
    inputs=8, actions=3, modules=(16,), recall=True, seed=0,
    recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01,
    recall_recency=0.0, recall_boundary=True,
)
assert long_stories.recall.position == 32
```

| Recall settings beside the position term | Copying unique tokens: taught 3 to 6, read at 8 to 12, closed loop | Look back: taught 5 to 12, read at 24 to 60, "first" |
| --- | --- | --- |
| the founder table | 0.30 to 0.36 | 0.25 to 0.27 |
| the table at 0.01 | 0.86 to 0.99 | 0.31 to 0.35 |
| the slope at 0.1 | 0.47 to 0.88 | 0.62 to 0.93 |
| the slope at 0.1, the table at 0.01 | 0.99 to 1.00 | 0.70 to 0.81 |
| the anchor | | 0.72 to 0.75 |
| the anchor, the slope at 0.1, the table at 0.01 | | 1.00 (every cue 0.97 to 1.00) |

With the recipe every head reads "the first told", "the second" and "the latest" at two to five
times the longest length it was taught. On six fresh founders, look back taught at 5 to 12 and
read at 24 to 60 answers "first" and "second" at 0.99 to 1.00, and copying unique tokens at
unseen lengths 0.99 to 1.00. On induction and n-back it stays within 0.014 of the founder
table. A period longer than `recall_position` needs the table exact that far: for a period of
40 add `recall_position=64` (the recipe continues it at 0.86 to 0.94 against 0.35 to 0.39 at
32 exact ages).

**What the recipe costs.** On the
[panel of position changes](../benchmarks/sequence/README.md#the-panel-of-position-changes-232)
it keeps n-back, two hops with a think moment and ordered recall, and it costs:

- **Periodic continuation at a period of 24.** 0.83 against 0.92 after 1,000 lessons; no single
  part of the recipe causes it.
- **Content shortcuts the slow table gives up.** The slower table makes the brain take the
  position route where content would serve: a hidden route cue of the Stone Age sequence
  instrument falls to 0.25 against 0.43, and in the
  [ordered recall world](../benchmarks/sequence/README.md#ordered-recall-orderpy) taught by
  lessons the recipe answers 0.875 to 1.00 where the founder table answers every query.
- **Music.** The slope costs a composer of C64 tunes about 0.1 nats on its answer.

Use the recipe where the stories run longer than the lessons did. Where they keep the taught
lengths and few lessons must teach exact ages, the founder table serves.

### One composition for sequences

A brain that says whole sentences takes the positions recipe below at `teaching="deep"` (`Fd` in the
[grammar chamber](../benchmarks/grammar/README.md#readings-at-the-base-composition-development)):

```python
speaker = Brain.compose(
    inputs=35, actions=18, modules=(64,), seed=0, episodic=False, teaching="deep",
    recall=True, recall_recency=0.0, learning_surprise=True, recall_boundary=True,
    recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01, recall_relative=True,
)
assert speaker.learner.config.nudged_fixed
```

| One online pass, development | Clauses | Two distractors | Held-out | Roles | Brackets | Paths | Two hops | Three hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| this composition, twelve lives | .96 | .95 | .91 | .91 | .78 | .87 | .80 | .22 |
| the recipe at `teaching="normalized"`, on the distance | .57 | .24 | .52 | .31 | .33 | .27 | .03 | .00 |
| a matched transformer, one pass, three lives | 1.00 | | 1.00 | 1.00 | .18 | .36 | | .00 |

Both Cadence rows hold 24,384 parameters and take 6.3 million sweeps per life. The sequence battery
holds beside it: look back .946 to .995, n-back .974 to .995, a second hop .996 to .998, copying
repeated tokens .942 to .963, induction .987 to .998. Its costs: the
[key-door chamber](../benchmarks/keydoor/README.md) falls from 5 of 8 lives to 1 of 8 at delay 2 and
from 3 of 8 to 0 of 8 at delay 5, most of it with the recall memory alone
(2 of 8 and 1 of 8), and the read of tokens no lesson used falls at n-back
(.96 .99 .71 to .62 .99 .59).

A brain that meets look back, copying, induction, n-back, periodic continuation, two hops and "was
A told before B?" in one life takes the recipe for lengths no lesson reached with the options
that compose with it, and declares which of its answers are tokens:

```python
sequences = Brain.compose(
    inputs=26, actions=16, modules=(64,), seed=0, teaching="normalized",
    recall=True, recall_recency=0.0, learning_surprise=True, recall_boundary=True,
    recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01, recall_position=64,
    recall_relative=True, recall_when=True, recall_address_reward=1.0,
    recall_position_reward=4.0,
    recall_tokens=list(range(14)) + [-1, -1],  # actions 14 and 15 are a yes and a no
)
assert sequences.recall.copy == 0.0  # the tokens declare; no pathway is added
```

- The positions recipe, `recall_position=64` and the relative read carry look back, copying
  and a period of 40 at lengths no lesson reached; the when channel carries "before".
- The tokens keep the yes and no lessons off the read and teach those rows at every lesson
  ([answers that are no token](#the-copy-pathway-opt-in)); the token answers stay judged by
  surprise.
- The reward read finds the writes of ordered recall from reward alone; its step does not
  shrink with the recipe's slow table.
- The copy pathway is left out: beside the recipe it stops look back's read learning where to
  look.

Read on six fresh founders of the [sequence battery](../benchmarks/sequence/README.md) (1,500
lessons, 16 streams), every task at its own life: look back taught 5 to 12 and read at 24 to 60
0.92 to 1.00 for every cue, copying unique tokens taught 3 to 6 and read at 8 to 12 0.997 to
1.000 in closed loop, repeated tokens 0.925 to 0.941, induction and n-back taught short and read
at 24 to 60 0.970 to 0.998 and 0.933 to 0.994, a period of 40 0.900 to 0.939 on five founders
and 0.825 on one, two hops with a think moment 0.995 to 1.000,
"before" 0.962 to 0.978, look back's cues taught one after another with
`learning_rehearsal=1.0` 0.969 to 1.000 (taught mixed 0.972 to 1.000), and ordered recall from
reward 1.00 on 9 lives of 12 (two lives answer one action on every query, sampled or greedy;
one leaves "the latest" without a gap unanswered, where the first write's next port holds the
second write's value). Where it falls short:

- **Answer tokens never taught** are not produced (0.00 on every task), though the read names
  them (induction and periodic continuation 1.00, two hops 0.91 to 0.93). Lessons that teach
  other actions push the association's synapses onto an action never taught down: with the copy
  pathway beside this composition, on a development founder, the association drives the
  never-taught answer at -1.2 to -5.1 (copying, two hops, n-back, look back) and the winner at
  +1.1 to +7.0, against the copy's +0.3 to +1.5. The copy pathway produces them only where it
  answers from the first lesson and the association stays untaught. The shape that follows from this, a copy gain per port shared by every
  token with outputs a lesson does not name keeping their inputs, was read as a development
  emulation on two founders: induction 1.00, copying 0.995 to 0.999 and periodic continuation
  0.96 to 0.98 of the never-taught answers, two hops 0.06 to 0.18 on the second hop, look back
  0.00 to 0.15 and n-back below 0.02, at a cost to look back (`second` 0.90 on one founder) and
  copying repeated tokens (0.91 to 0.92). That form is not in the library; the
  [learned value path](#the-learned-value-path-opt-in) is the form that is.
- **N-back** costs 0.033 on the mean against the founder memory, which answers it at 0.998 to
  1.000; two founders of six read 0.933. The cost is the recipe's slow table
  (`recall_position_rate=0.01`): at the founder's rate those two founders read 0.997 and 0.972,
  and copying unique tokens at lengths no lesson reached falls to 0.17 (the content shortcut of
  the table at 0.1). The two gates ask opposite rates of the one table; the composition keeps
  the slow one.
- **Ordered recall from reward** misses on three lives of twelve, where `reward_recipe`,
  without the next port that successor reads need, misses on two. Two of the three pin the
  actor's readout on one action from their first 256 episodes: the query's outputs sit at 0.95 to
  0.99 and near zero, where the activation has no slope and neither the dopamine nor the read's
  eligibility can move the choice. A relaxation of the pinned output's bias opened both locks and
  moved another life to a miss, and the count stayed 9 of 12, so the library carries none. What
  limits the other lives is the next port: a write's value is on its own slot's recall port and on the slot before's next
  port, so with no gap between the writes "the first" and "the latest" each have a slot whose
  two ports disagree, and a life can learn either. `reward_recipe` answers both through the
  write's own slot by leaving the next port out; this composition keeps the next port for its
  successor reads.

### Several heads (opt-in)

One held read delivers one record cleanly, or a weighted mixture of several whose bindings the
cortex has to untangle. `recall_heads=k` reads the same bank through k heads in one moment:

```python
heads_brain = Brain.compose(
    inputs=12, actions=4, modules=(24,), learned_code=8, attention=True, recall_heads=2,
    seed=0,
)
assert heads_brain.recall.heads == 2
assert heads_brain.recall.ports(2) == (("recall", "next"), ("recall_1", "next_1"))
assert heads_brain.recall.address == 0.01 and heads_brain.recall.temperature == 0.25
heads_brain.act(np.eye(12)[[3]], greedy=True)
heads_brain.act(np.eye(12)[[5]], greedy=True)
assert len(heads_brain.recall.attended_heads) == 2
```

Each head is a copy of the held read: a read port of its own (the regions `recall_h` and
`next_h`, wired into the association like the first) and an address map of its own, which
starts at zero. The heads compete for the slots: a slot
can be read in full once per moment, so the heads read in turn and each takes of every slot
at most the share the heads before it left. A head keeps the shape of its own attention where
the slots allow and gives what a taken slot cannot give to the slots it prefers next, with
its whole mass while enough is left; a head whose slots the heads before took whole reads
nothing. At birth, with every address map at zero, head 0 reads the best-matching moment and
head 1 the best of what head 0 left.

The address lesson after a teaching step reads the association's movement back through each
head's own synapses, so each head's map moves toward the records whose values would have
helped through that head, among the shares it reads with. Reward trains the maps only at
`recall_address_reward` above zero, with one eligibility block per head; in a life of rewards
alone at the founder 0 the maps stay at zero. One bank is written per moment, and the heads
read it before the settle and hold their reads through it. With several heads the recipe
learns the addresses (`recall_address=0.01`) at the Recall founder's
`recall_temperature=0.25`.

In the [held-heads benchmark](../benchmarks/attention/held_heads.py) each trial witnesses four
facts `role -> value` with fresh values, and one question asks for two of them in one settle;
some value pairs are never taught.

| Answer | Two heads | One head | Uniform random |
| --- | --- | --- | --- |
| two reporting slots, taught pairs | every pair, on four founders | 21 to 24% | 3% |
| two reporting slots, untaught pairs | 77 to 98% | | |
| one combined action, taught pairs | every pair | 59 to 76% | 10% |
| one combined action, untaught pairs | 98 to 99% | | |

The combined action goes to the food, or, when the predator stands at an odd position, to
the position five steps from it. On every trial each head attends most to a different fact
([receipt](../benchmarks/attention/results/held-heads.json)). This is component evidence for
reading several facts in one settle; a decision in which one fact matters only together with
another is not covered.

**A read that needs another read.** The address is fixed before the settle, so a read that
depends on what another read found takes a second moment: a "think" moment chains the reads,
the first read landing in the cortex and the second moment's address reading from there. In
two hops (`a -> b`, `b -> c`, ask `a`) two heads with a think moment answer the second hop at
0.98 to 0.99 ([sequence battery](../benchmarks/sequence/README.md), three development seeds).

Use several heads when one answer needs several witnessed facts at once, and compare them with
one head and with the facts asked one per moment (one head reads one fact per moment through
the working trace). Long lesson curricula drive the shared readout into saturation once the
answers are right, and founders can collapse onto one action: teach with
`learning_surprise=True` from the first lesson, or use `live`, whose arousal does it
([lessons only where the answer differs](teach.md#lessons-only-where-the-answer-differs)).

### When a read item happened (opt-in)

**`recall_when=True`.** A read delivers what its slots hold and what followed them; the
position term only chooses the slots. With the option every head's read also delivers when the
read items happened, so a question such as "was A told before B?" has something to compare. Each
head gets a region of its own (`recall_when`, `recall_1_when`, ...), driven at the read's
amplitude by the attention-weighted features of its slots' ages,

```text
when = sum_i p_i [log(1 + age_i) / log 9, exp(-age_i / 8)]
```

and with the anchor the same two features of each slot's place in the episode (zero for a slot
written before it began). The regions reach the association through synapses that start at
zero, so a newborn settles as one without the option, and lessons give them weight. The address
lesson names the slots to read by what they hold and does not read the movement back through the
when regions; the region says when the slots read happened.

**Keep the read on its items.** "Was A told before B?" needs the read at A to land on A's slot
and the read at B on B's. The position lesson, and less so the address lesson, take their target
from the slots whose contents the lesson's movement points to; a yes or no answer is no slot's
content, so the lessons pull the read off the items. Freeze the position term and sharpen the
softmax:

```python
when_sharp = Brain.compose(
    inputs=8, actions=3, modules=(16,), recall=True, seed=0,
    recall_when=True, recall_position_rate=0.0, recall_temperature=0.1,
)
assert when_sharp.recall.temperature == 0.1
```

With these settings (the [sequence battery](../benchmarks/sequence/README.md)'s `when_sharp`)
the read lands on the item's slot, the working trace carries A's report to B's moment, and
"before" answers 0.94 to 0.97 on fresh founders. "How long ago was X told?" (one of four age
buckets) answers 0.87 to 0.95, against 0.51 to 0.80 without `recall_when`.

The age features at ages no lesson reached enter the association through synapses taught at
short ages: beside the recipe for unseen lengths, a look back read at 24 to 60 holds at 0.99 on
five founders of six and falls to 0.93 on one.

### The learned value path (opt-in)

A held read reaches the motor region only through the association region, which learns which
answer each read deserves. A token taught as an answer is then produced from its read; a token
never taught as an answer is read as well as any and never produced, because no lesson ever
raised its action. `recall_gain` gives the read a path of its own to the motor region: one learned
scalar per read port of every head (`recall`, and `next` where it is on), on the identity that
`recall_tokens` declares. At every moment the memory adds to each token's motor neuron

```text
drive[motor_j] += sum over the ports of  g r[token_j],    r = amplitude * sum_i p_i V_i  (a next port: N_i)
```

`r` the read as the port holds it, held through the settle as the port's own drive is. Lessons on
some tokens teach the gains, and the gains hold for every token, so a token no lesson used drives
its own action.

```python
TOKENS, HEAR, ASK, NOISE = 6, 6, 7, 8
speaker = Brain.compose(
    inputs=9, actions=TOKENS, modules=(16,), seed=3, episodic=False, recall=True,
    recall_gain=1.0, recall_tokens=range(TOKENS),   # observation i and action i: one token
)
assert (speaker.recall.gains == 0.0).all()          # a newborn acts as one without the path
speaker.recall.gains = np.array([1.0, 0.0])         # as lessons would leave a copy-back task
for token in range(TOKENS):                          # no lesson used any of these tokens
    speaker.reset()
    speaker.act(np.eye(9)[[token]] + np.eye(9)[[HEAR]], greedy=True)  # a token is heard
    speaker.act(np.eye(9)[[NOISE]], greedy=True)                       # something else
    assert speaker.act(np.eye(9)[[HEAR]] + np.eye(9)[[ASK]], greedy=True)[0] == token
```

- **Its lesson.** After each teacher lesson, beside the address lesson, each gain takes the mean
  over the taught rows of the row's rate times the sum over the tokens of the port's read of the
  token times its motor neuron's movement between the lesson's two nudged phases: the contrast
  every synapse learns by, with the source held. A row's rate is `min(recall_gain, 5 / E)`, `E` the
  energy of the row's reads over every port (`Recall.GAIN_BOUND`, a constant of the memory), so a
  row taught alone moves its own drive by at most five times the neuron's movement. Above the bound
  the rate no longer matters, and without the bound a large rate collapses the read. A step that
  is not finite is refused whole; the report gives `gain_rows` and `gain_refused`.
- **The gains learn which port holds the answer.** Nothing assigns a port. On six fresh founders the
  next port's gain ends at 1.5 to 1.9 on a period of 8 and the recall port's near zero; on two hops
  the recall port's falls to -1.6 to -1.8 while the next port's rises to 3.3 to 3.5, which removes
  the echo of the cue's own token.
- **The address lesson is untouched.** The address and position maps step on the association's
  movement read back through the port synapses; nothing is read back through the gains.
- **Under reward.** Each sampled act keeps the step the lesson would take were the act's action the
  lesson's label, in one column per port of the read's eligibility (`reward_columns`); the actor
  traces it and a unit of dopamine moves each gain by one such step. `learn` reports the whole
  read's step as `address_reward_step`.
- **A newborn is the founder.** The gains start at zero, so a newborn settles, acts and saves as one
  without the path: the motor drive gains exactly nothing. They are saved as the array `gains`,
  counted by `parameters()`, kept by `graft` and `grow`; `retune` refuses the setting. `grow` adds no
  recall memory with tokens, as for the copy pathway.
- **Limit.** The drive is a term held through the settle, which is sound for a held read. A read
  revised by the live association inside the settle would have to carry it in the reciprocal force,
  or the settle is no longer the descent of one energy; the two are not composed.

Read on six fresh founders beside [one composition for sequences](#one-composition-for-sequences), at
`recall_gain=1.0`, in the [sequence battery](../benchmarks/sequence/README.md) (1,500 lessons, 16 streams)
and the [grammar chamber](../benchmarks/grammar/README.md) (4,000 batches, the means of the founders, its
recipe in the place of the composition):

| Reading | Without the path | With the path |
| --- | --- | --- |
| Answers on tokens no lesson used: induction | 0.00 | 0.92 to 1.00 |
| a period of 8 | 0.00 | 0.93 to 0.97 |
| copying | 0.00 | 0.76 to 0.97 |
| two hops | 0.00 | 0.70 to 0.82 |
| n-back | 0.00 | 0.08 to 0.50 |
| look back | 0.00 | 0.00 |
| Look back, `first` and `second` at lengths no lesson reached | 0.98 to 1.00 | 0.98 to 1.00 |
| Copying unique tokens, closed loop | 1.00 | 0.99 to 1.00 |
| Copying repeated tokens, closed loop | 0.91 to 0.95 | 0.82 to 0.85 |
| Two hops, the second hop | 0.99 to 1.00 | 0.94 to 0.96 |
| Induction | 0.97 to 1.00 | 1.00 |
| N-back | 0.86 to 0.99 | 0.83 to 0.99 |
| Chamber: clauses | 0.70 | 0.75 |
| held-out combinations | 0.56 | 0.64 |
| agreement | 0.83 | 0.89 |
| the verb said at two distractors | 0.64 to 0.82 | 0.74 to 0.90 |
| Dyck sentences | 0.45 | 0.36 |
| paths | 0.30 | 0.21 |

- **The read stops being taught.** A surprise lesson teaches a row only where the answer is wrong, and
  the path answers early. On copying repeated tokens the read's lesson takes 406 rows against 971 on a
  development founder, and each moves the association a third as far or less. The read then falls at
  positions no lesson reached, and the answer follows it. On a path's hop the read's delivery falls the
  same way. This is the copy pathway's failure beside the recipe, smaller.
- **An action no lesson taught still loses the vote.** On look back the read names the held token
  (0.46 to 0.98) and the answer is 0.00. Lessons on other actions push the association's drive on that
  action down, and the read is spread over the slots it holds, so the path's edge for the right token
  is smaller than the association's for the taught one.

Use it for a brain that has to say tokens no lesson used where the answer is what the read holds:
continuing, copying and inducing from what was heard. Leave it out of a brain that relies on repeated
tokens at lengths no lesson reached, on Dyck sentences or on chained facts.

### The copy pathway (opt-in)

A held read reaches the motor region only through the association region, which learns which
answer each read deserves. A token taught as an answer is then produced from its read; a token
never taught as an answer is read as well as any and never produced, because no lesson ever
raised its action. `recall_copy` adds a direct pathway from every head's read ports (`recall`,
and `next` where it is on) to the motor region: plastic synapses learned by the same settle
contrast as every other synapse, inside the one settle.

```python
TOKENS, HEAR, ASK, NOISE = 6, 6, 7, 8
copier = Brain.compose(
    inputs=9, actions=TOKENS, modules=(16,), seed=3, episodic=False, recall=True,
    recall_copy=2.0, recall_tokens=range(TOKENS),   # observation i and action i: one token
)
for token in range(TOKENS):                          # no lesson: every token is untaught
    copier.reset()
    copier.act(np.eye(9)[[token]] + np.eye(9)[[HEAR]], greedy=True)  # a token is heard
    copier.act(np.eye(9)[[NOISE]], greedy=True)                       # something else
    cue = np.eye(9)[[HEAR]] + np.eye(9)[[ASK]]                        # asked for it back
    assert copier.act(cue, greedy=True)[0] == token
```

- **Without tokens** (`recall_copy=s` alone) the synapses are drawn at magnitude `s` as every
  projection is, appended last, so the brain with the pathway is the brain without it plus these
  synapses. Learning shapes them like any synapse; nothing lets a read drive an action that was
  never taught.
- **Shared identities** (`recall_tokens`). Where an observation coordinate and an action are the
  same token, declare it: `recall_tokens[j]` is the observation coordinate of action `j` (`-1`
  for none). The pathway then starts as that identity: from each recall port, the read's own
  value, at `recall_copy`; from each next port at zero; and from the present observation (the
  sensory region) at `-recall_copy`, so a token the moment holds is not copied back.
  Every other synapse of the pathway starts at zero. The tokens must be in the recall values: a
  brain without a code, or a learned code joined to an observation no wider than it; a fixed
  code is refused.
- **Every synapse learns on its own.** A lesson moves the synapses of the tokens the read and
  the observation held; the identity of a token never read stays where it started, so a token
  never taught as an answer keeps its route to its action.
- **Why the echo.** A read addressed by the present moment holds that moment's own token in its
  value: in a succession task (a cue names a token, the answer is the token that followed it)
  the recall port holds the cue's token and the next port the answer. Copying the recall port
  alone answers the cue with itself; the echo removes what the observation holds.
- **Why the next port starts at zero.** Which port holds the answer depends on the task: the
  heard token is on the recall port in a copy-back question, the successor on the next port in
  a succession. The pathway starts from the recall port, the token the read delivers, and the
  next port's gain is learned. Started equal, the two ports would drive the heard token and the
  distractor that followed it alike.
- **Answers that are no token.** The read's lessons take their target from the slots'
  contents. With tokens declared, a lesson whose answer is an action without a token (`-1`: a
  yes or a no, a count) leaves the address and position maps as they are; the cortex learns
  from it as from any lesson. "Was A told before B?" needs the read to stay on the items asked
  about, and a yes or no lesson pulled it off them. Under
  [surprise lessons](teach.md#lessons-only-where-the-answer-differs) such a row is taught at every
  lesson, answered right or not: judged only where wrong, the lessons of a judgment between two
  classes carry one class at a time and never the comparison that separates them ("before"
  stays at chance after 4,500 lessons, against 0.97 at every row). Lessons at every row collapse
  the token answers of induction, copying and two hops (0.11 to 0.68 on development founders),
  so those stay judged by surprise. The tokens declare this without the pathway as well
  (`recall_tokens` without `recall_copy`: no synapse is added), and an action list without any
  token (`recall_tokens=[-1] * actions`) declares that no answer is a slot's content: no lesson
  moves the read and every row is taught.
- **Beside the positions recipe.** The address and position lessons name the slots to read by
  the association's movement read back through the read port. With the pathway the answer
  reaches the motor region directly, the association moves little toward it, and a read that
  only position finds (look back) stops learning where to look: taught at 5 to 12 symbols and
  read at 24 to 60, the recipe with the pathway answers 0.25 to 0.33 on five fresh founders of
  six (the recipe alone 0.99 to 1.00), its position, anchored and slope maps at a third, a
  quarter and a sixth of their norms. Reading the motor's movement back through the pathway
  restores look back (0.99) and misdirects reads whose answer is a successor (two hops 0.27 to
  0.28): the recall port's identity names the slot holding the answer itself, not the slot it
  follows. Leave the pathway out of a brain that relies on the recipe.

In the [copy-tokens benchmark](../benchmarks/attention/copy_tokens.py) a token is heard, two
other tokens follow, and a cue asks for the heard token back; tokens 0 to 4 are taught as
answers and 5 to 7 never are. A second task cues a token of a sequence of five heard once and
asks for the token that followed it. On six founders of one module
([receipt](../benchmarks/attention/results/copy-tokens.json)):

| Arm | Untaught tokens produced | Taught answers | Succession | Read of the untaught |
| --- | --- | --- | --- | --- |
| founder memory | 0.00 | 0.72 to 0.98 | 0.95 to 0.99 | 0.32 to 1.00 |
| `recall_copy=2.0` with tokens | 1.00 on every founder | 1.00 | 0.94 to 0.99 | 1.00 |
| `recall_copy=1.0` with tokens | 0.17 to 1.00, mean 0.78 | 0.86 to 1.00 | 0.95 to 1.00 | 0.98 to 1.00 |
| `recall_copy=0.5` without tokens | 0.00 | 0.57 to 0.99 | 0.93 to 1.00 | 0.58 to 1.00 |
| `recall_copy=1.2` without tokens | 0.00 | 0.22 to 0.87 | 0.86 to 0.97 | 0.70 to 1.00 |

Uniform random answers 0.125. Use `recall_copy=2.0` with tokens. With it the answer follows its
read on every untaught and taught question, and a deep apical brain (`modules=(32, 24, 24)`,
`apical=0.25`) reads the same. A weaker identity loses the untaught tokens on some founders, the
association having learned the taught answers first; a pathway drawn without tokens costs taught
answers and succession. A successor never taught as an answer stays at 0.00 in every arm, since
the next port's gain grows slowly.

Use it when the answer is a token the read can deliver, and compare it with the founder memory
and with the pathway drawn without tokens on the task. The options are construction settings,
described and saved only where they are on.

### Its own actions in the slots (opt-in)

The recall slots hold what the stream sensed. What the brain did is in the working trace only
through the margin its motor region settled at, and nothing of a sampled choice.
`recall_action=True` writes the executed command into every slot, one coordinate per motor
neuron (one-hot within each motor slot; zero for a moment `wait` senses):

- **Key and value.** The slot's key gains the command after the window's events, and its value
  holds the event and the command, so the next port carries what was done at the moment after.
  Each head's read port holds the command on regions of its own (`recall_action`,
  `next_action`, `recall_1_action`, ...), whose synapses into the association start at zero.
- **The maps read the command before.** The address and position maps read `[x, a_prev]`, the
  event and the command the stream executed at the moment before, so a lesson can teach a head
  to look for the slot where the brain did what it just did. The address's fixed part is zero on
  the command coordinates.
- **A newborn reads as one without it.** Every new synapse and map entry starts at zero: a
  newborn with the option settles, attends and acts as the same founder without it until
  learning gives them weight.

```python
mover = Brain.compose(inputs=4, actions=2, modules=(16,), seed=0, recall=True,
                      recall_action=True)
x = np.eye(4)[[0]]                                     # the same observation every moment
done = [int(mover.act(x, greedy=True)[0]) for _ in range(3)]
assert (mover.recall.values[0, :, 4:] == np.eye(2)[done]).all()  # what it did, slot by slot
assert (mover.recall.acted == np.eye(2)[done[-1:]]).all()        # the command of the moment before
```

For a brain that needs only its last command, the [efference copy](#efference-copy) is the
simpler mechanism. Use the slots when a read has to find the commands with their moments, by
content and position. In the [own-action benchmark](../benchmarks/attention/own_action.py),
under one repeated observation:

| Task | Commands in the slots | Founder recall | No recall | Uniform random | Efference copy |
| --- | --- | --- | --- | --- | --- |
| alternate its own last action | 1.00 | 0.78 | 0.47 | 0.48 | 1.00 |
| press every third moment | 0.97 | 0.65 | 0.36 | 0.46 | 1.00 |

Erasing the commands before every test moment brings the slots to 0.26 and 0.33
([receipt](../benchmarks/attention/results/own-action.json); the copy at
`efference_amplitude=3.0, efference_decay=0.5`).

The [episodic memory](#episodic-memory-what-happened-in-this-situation-before) keys its context
records on the issued commands: each decision's command joins the stream's temporal context at
the share `1 - memory_context`, so the option does not change the vote.

### Single-head learned-code attention (opt-in)

`attention=True` is the shortest supported path from learned meanings to contextual recall. It
combines the learned code and the recall memory:

```text
observation -> learned meaning code -> two-moment Recall lookup
                                          |
                                          v
                         held event and successor -> neural settle -> action
```

The learned code compresses the observation using experience. Recall joins the current and
previous code into an ordered query, compares it with stored two-moment keys and returns a
weighted event and successor. That read drives its own regions while the rest of the brain
settles. Turn it on explicitly:

```python
attention_brain = Brain.compose(
    inputs=12, actions=2, modules=(24,), learned_code=8, attention=True, seed=0,
)
assert attention_brain.attention
assert attention_brain.recall.window == 2
assert attention_brain.recall.next
assert attention_brain.describe()["layout"]["composition"]["attention"]
```

The preset supplies `recall=True`, `recall_window=2`, `recall_next=True`,
`recall_address=0.0`, `recall_recency=0.0` and `recall_temperature=0.1` where they are not
given. Other `recall_*` genes can be set beside it; the window must be at least two and the
successor read must stay on. Choose the learned-code width for the task: eight is the capacity
the small benchmark tests.

The address map is frozen (`recall_address=0.0`), so selection comes from similarity between
learned-code windows. The [position term](#position-how-long-ago-a-moment-happened) is built
and learns as everywhere, so after the first lesson the read also weighs each slot's age.
`attention` neither enables nor disables the associative store or `memory_choice=True`.

In the [contextual-choice benchmark](../benchmarks/attention/contextual_choice.py) every final
observation is the same literal ASK, and two stored episodes differ only in their earlier
context and successor. Four founders answer 32 of 32 queries; present-only and memory-off
controls answer 16 of 32, as does uniform random. Erasing the relevant record breaks the
answer; erasing the other leaves it intact.

The head selects once before the neural settle, and the equilibrium interprets the read; the
interpretation cannot revise the query while that settle forms. The benchmark establishes a
bounded contextual lookup. Compare it on the task with attention off, episodic memory alone and
uniform random. Several facts in one answer take [several heads](#several-heads-opt-in).

### Recall cost

The two regions add one neuron per value coordinate each: `compose(250, 2)` grows from 380 to
880 neurons and from 20,352 to 52,352 synapses. `recall_action` adds one neuron per motor
neuron to every read port region, their synapses into the association, and as many rows to the
address and position maps and columns to the address; `recall_copy` adds the value width times
the motor neurons per port region (and with tokens one synapse per sensory neuron and action):
with two heads over a code of 128 and 323 actions the pathway adds 4 x 128 x 323 = 165,376
synapses. Each further head adds its two regions, their projections and one more address map.
The address map is `code` by `address_width` plus the association width by `address_width`;
with compose's 64 association neurons it is 0.66 MB at 256 inputs, 1,322 MB at 12,824 inputs
on the uncoded observation and 8.9 MB with `code=1024`. One stream's slots hold `capacity`
keys, values and nexts, 0.39 MB at 256 inputs, in arrays of up to twice the capacity: a write adds a
slot in the room behind the others and moves them when the room runs out. The map is learned: a
reset keeps it, `parameters()` counts it and a checkpoint saves it. The keys are the observation
code, so changing the code makes the stored keys inconsistent with new addresses.

### In a custom anatomy

`Brain(connectome, recall=True, ...)` needs a `recall` and a `next` region (or `recall` alone
with `recall_next=False`), each with one neuron per value coordinate,
`Genome.observation_width(inputs, code)`, and with `recall_heads=k` the regions `recall_h`
and `next_h` for h from 1 to k - 1 (`Recall.ports(k)` names them). With `recall_action` each of
those regions has a `<region>_action` region of one neuron per motor neuron
(`Recall.action_ports(k)`); with `recall_copy` synapses from every read port neuron into
`motor`, and with `recall_tokens` from every token's sensory neuron into its action, which the
brain starts as the declared identity. Without `recall=True` populations of those names are
ordinary ones. `tests/test_recall_memory.py` checks that the library with `recall_position=0`
reproduces the reference memory `benchmarks/binding/index_read.py` moment for moment. To check
that the memory does the work, erase a copy's slots with `brain.recall.reset(batch)` before the
query ([ablation](#check-that-a-memory-does-the-work)).

### Recall limits

- The address is fixed before the settle, so a second recall takes a second moment.
- Chained recall, where the address depends on a recalled token and the question together,
  reaches the linear map only through conjunctions held in the cortex of the moment before.
- The slots are a record within one stream. Nothing consolidates across streams except
  through the plastic synapses that read them.
- The copy pathway's identity starts on the recall port. A task whose answer is on the next
  port (a successor) has to learn that port's gain, which grows slowly; the association learns
  the taught successors first. The identity reaches only tokens in the recall values: a learned
  code narrower than the observation keys on the code alone, and there the pathway is drawn at
  random.
- With `recall_action` the commands reach the answer through synapses that start at zero; under
  reward alone the address and position maps do not learn (`recall_address_reward` 0), so only
  the values' commands are read.
- The position term is learned. Its preferences are as general as the lessons that shaped them:
  a brain taught only on stories of one length can learn "about twenty moments back" where
  "the oldest record" was meant. Vary the lengths when order matters. Ages are counted in
  moments of the stream, so a term learned at one moment rate does not transfer to another.

The benchmark-only [direct-route attention candidate](../benchmarks/attention/DIRECT_ROUTE.md)
addresses the first limit: a live route reads the learned meaning, and a locally learned map
turns it into a new query inside the same equilibrium. It is no `Brain.compose` option.

## Working trace

After each moment the trace copies the settled activity of the association region (the last
entry of `modules`) into the `prefrontal` region. The next settle holds those neurons at
`working_memory_amplitude` times the copy, and the association region reads them through
plastic synapses that start quiet, so a newborn answers mostly from the present. The copy is
fixed; learning strengthens the synapses where the carried interpretation predicts the
outcome.

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=3, actions=2, seed=0)        # the trace is on by default
brain.act(np.array([[1.0, 0.0, 0.0]]), greedy=True)       # a cue
carried = brain.working_memory.trace.copy()               # the settled association activity
drive = brain.stimulus(np.array([[0.0, 0.0, 1.0]]))       # the next moment's drive
prefrontal = brain.connectome.populations["prefrontal"]
assert np.allclose(drive[0, prefrontal], 3.0 * carried[0])  # held at the amplitude
brain.reset()                                             # a new stream starts without it
assert not brain.working_memory.trace.any()
```

| Setting | Founder | Changed by | Raise it when | Lower it when |
| --- | --- | --- | --- | --- |
| `working_memory_amplitude` | 3.0 | `retune` | the copy should weigh more in the settle and in learning | the carried context should fade (0 silences it) |
| `working_memory_decay` | 0.0, in [0, 1) | `retune` | the context changes slowly and should be averaged over moments | 0 carries the last interpretation only |
| `scales["trace"]` | 1.2 | construction (`compose` argument) | the past must matter from the first moment, before learning | the brain repeats itself and ignores what it sees |
| `modules` | `(64,)` | construction | more items or distractors: `modules=(128,)` | a narrow last module such as `(48, 24)` carries little |

`Brain(connectome, ...)` has a trace when the connectome has a `prefrontal` region.
`done=True` clears a stream's trace, so mark episodes only where the world restarts, and
teach with `learning_surprise=True` so lessons touch only the moments answered wrongly
([learning](teach.md#lessons-only-where-the-answer-differs)). The synapses that read the
trace learn only where the outcome or the lesson depends on something the brain saw earlier,
so the task has to make the past matter; distinct cues, such as one marking new information
and one marking noise, help. The trace costs 64 neurons and 4,096 synapses at the founder. Two
pasts that settled alike are one context to it, and a distractor that looks like evidence can
overwrite the context of a small brain.

What it serves, from [`tests/test_context_defaults.py`](../tests/test_context_defaults.py):

- **A cue carried across a short delay to its reward.** The founder trace does it (0.96 to 1.00
  in the delayed-response test).
- **Answers from the present alone.** The actor has to learn the trace away (a continuing
  contextual bandit with `actor_gamma=0.0, actor_lam=0.0` reads mean 0.60 against chance 0.25).
  `working_memory_amplitude=0.0` is the control.
- **One cue held and replaced by a later one over a delay.** The working trace alone does not
  serve it: a loop strong enough to hold a cue resists the next write
  ([recall benchmark](../benchmarks/recall/README.md)).

## Efference copy

A positive `efference_amplitude` at construction adds an `efference` region of one neuron per
motor neuron, holding `c <- d c + (1 - d) onehot(action)` for the command each stream issued
(`d = efference_decay`), read through a plastic projection of fixed initial scale 12.0. The
working trace carries the executed action only through the margin the motor competition
left, and nothing of a sampled choice; the copy holds the command itself.

```python
brain = Brain.compose(inputs=3, actions=2, seed=0, efference_amplitude=1.0)  # opt-in
action = brain.act(np.array([[0.0, 1.0, 0.0]]))           # a sampled act issues a command
copy = brain.efference.trace                              # fading one-hot, decay 0.2
assert np.isclose(copy[0, action[0]], 0.8)
```

`efference_amplitude` (0.0) adds the region only at construction; `retune` changes it
afterwards and refuses it on a brain built without the copy. `efference_decay` (0.2, in
[0, 1)) can be retuned. The copy costs one neuron per action and as many synapses into each
association neuron.

Use it to alternate or sequence the brain's own actions under identical input: in the
[rhythm chamber](../benchmarks/rhythm/README.md) (`modules=(32,)`, `episodic=False`,
`working_memory_decay=0.1, scales={"trace": 12.0}`, the copy at `efference_amplitude=3.0,
efference_decay=0.0`) alternation reads 0.97 and 0.93 against 0.50 and 0.40 with the trace
alone and 0.50 for uniform random. Leave it off for delayed credit: in the
[key-door chamber](../benchmarks/keydoor/README.md) a copy at amplitude 0.3 feeds less at every
delay, and at 3.0 the creature does not feed.

## Heard history

<a id="hear-the-past-at-several-timescales"></a>

The heard history keeps the observations themselves, where the trace keeps what the brain
made of them. `history_decays` adds a `history` region of one block of sensory width per
decay. After each act and each moment `wait` senses, the block of decay `d` moves
`h <- d h + (1 - d) x`, `x` the values the sensory neurons were driven with, so at a decision
it holds the past and the sensory drive the present. A plastic projection reads it. Add it
when the same observation needs different answers after different earlier observations.

```python
brain = Brain.compose(inputs=3, actions=2, seed=0, arousal=True, history_decays=(0.2, 0.8))
brain.live(np.array([[1.0, 0.0, 0.0]]))                   # a cue
brain.live(np.array([[0.0, 0.0, 1.0]]), reward=[0.0])     # the cue is gone
trace = brain.history.trace                               # two blocks of three heard values
assert trace.shape == (1, 6) and trace[0, 3] > 0          # the slow block holds the cue
```

`history_decays` (`()`, each in [0, 1)) is a construction setting and the projection's initial
magnitude 1.2 a constant; `history_amplitude` (3.0) can be retuned. On a coded vector brain the
blocks hold the code. Three decays at 256 inputs and four actions add 768 neurons and 49,152
synapses to a founder of 20,992 synapses.

## Associative store (the hippocampus)

`brain.hippocampus` is a `SynapticMemory` from the observation code to the motor neurons, one
value per action in reward units: a persistent matrix `C` shared by every stream plus a fast
residual `F` per stream. Before each settle the read `C + F` drives the motor neurons at
`memory_amplitude`. For a unit-length key `k` and the observed value `v` of the chosen action:

```text
F     *= 0.9                       # the decay, a constant
alpha  = min(1, consolidation * (1 + salience))
C     += alpha * outer(k, v - k @ C)
F     += memory_rate * outer(k, v - k @ (C + F))
```

Salience is `abs(reward)` unless `salience=` supplies a nonnegative vector; the signed value
decides what is stored. A value mask leaves the actions not taken untouched, and a batch
averages the change to `C` over its writing rows. The store writes every outcome of a
sampled act given to `step` and `learn`, zeros included (an all-zero key writes nothing);
under `live`, the outcomes of sampled acts and the routine outcomes that woke a calm brain.
`fit` and lessons write nothing to this store. Reads change nothing.

```python
brain = Brain.compose(inputs=4, actions=2, seed=7)        # the store is on by default
x = np.array([[1.0, 0.0, 0.0, 0.0]])
action = brain.step(x)                                    # a sampled act
brain.step(x, reward=[1.0])                               # its reward is recorded under x
memory = brain.hippocampus
assert np.isclose(memory.recall(x)[0, action[0]], 1.0)    # memory_rate 1.0: the last outcome
brain.reset()                                             # a new stream keeps F and C
memory.reset(1)                                           # clear the fast residual F only
assert 0.0 < memory.recall(x)[0, action[0]] < 0.2         # what consolidation kept in C
memory.clear()                                            # erase F and C
assert not memory.recall(x).any()
```

| Setting | Founder | Changed by | What it does |
| --- | --- | --- | --- |
| `episodic` | `True` | construction | `False` builds the brain without a store |
| `memory_rate` | 1.0, in [0, 1] | `retune` | how far a write moves `F` toward the outcome; 1.0 keeps the last one |
| `memory_amplitude` | 1.0 | `retune` | the strength of the read as drive |
| `consolidation` | 0.05, in [0, 1] | `retune` | the slow rate into the shared `C` |

`brain.hippocampus.writes` counts the writes. The store costs key width times actions numbers
for `C` and as many per stream for `F`. It has no separator: a write at key `k` moves the read
at `q` by `q . k` times the correction, so overlapping situations interfere. Under
`memory_choice=True` the same names set both episodic stores, and `compose` starts
`memory_rate` at 0.5.

- **Which action pays, with quick reversal.** The store learns it in one stream faster than the
  graph: in the [reversal chamber](../benchmarks/reversal/README.md), at that chamber's operating
  point (`modules=(32,)`, `working_memory_amplitude=0.3, scales={"trace": 12.0},
  working_memory_decay=0.2, consolidation=0.25`, actor rate 0.1 with bias 0.01), the store
  alone reads 1.00 after 300 exposures against 0.60 to 0.80 for the graph alone. At the composed
  founders the chamber stays at chance.
- **Beside the recall memory.** The read does not stand in for the store. With the
  [composition for sequences](#one-composition-for-sequences) on, six development founders each:
  the odour nursery passes 0 lives of 6 without the store against 5 and 6 with it, a bandit with
  reversal 0 against 4, and a contextual bandit reads .45 against 1.00.
- **Noisy or signed pay.** `memory_rate=1.0` keeps only the last outcome, so one bad outcome
  overwrites the record. Use `memory_rate=0.2`; in the
  [competing ring](../benchmarks/competing/README.md) it acquires as fast and takes about 0.10
  off the loss, most of which remains.

### Maturity and surprise-gated writes

With `maturity=True` ([reward](reward.md#maturity-opt-in)) the store keeps a noise scale `R` per
consolidated record (`brain.hippocampus.gate_noise`, zero at birth) and gates its writes by the
brain's step factor `m`. The change a write makes without the gate, to the shared `C` and to
each writing stream's `F` alike, is multiplied on each action's column by a gain, and each
writing stream moves the noise of the records its key reads toward its squared error:

```text
gain     = 1 - (1 - m) (1 - g)
g        = mean over the writing streams that observe the action of max(0, 1 - 2 sqrt(k^2 @ R) / |e|)
R[i, a] += 0.05 * k_i^2 / max(k^2) * (e_a^2 - R[i, a])   # each writing stream, its observed actions a
```

where `k` is a stream's key, `e = v - k @ (C + F)` its error on the action against its read
before the write, and `sqrt(k^2 @ R)` the noise of the records along its key. Every other
stream keeps its residual `F` around the shared `C`, as without the gate; with one writing
stream `g` is that stream's own. A young brain (`m` 1) writes as without the gate, within
rounding; a mature one (`m` near 0) moves a record only by the part of its error beyond two
noise scales, so an outcome its records expect leaves them as they are and a surprise moves
them. A record no stream writes keeps its value and its noise scale. A write whose squared
error leaves the floating-point range (an outcome near `1e154`) is refused with `ValueError`
before anything changes, as an overflowing record is.

Every write into the place records is gated: outcomes and, with `memory_choice`, replayed
credit and demonstrations. The context records (`brain.context_memory`) and the direct
payments of `memory_extinction` write as without maturity. The noise scales are saved and
loaded with the store and counted by `parameters()`; `clear()` erases them with the records.

## Learned weights and eligibility

The learned weights (synapse scales and biases, the actor's readout and the critic) are the
permanent memory. Lessons move them by the free/nudged contrast ([learning](settling.md#learning-the-freenudged-rule)),
outcomes by dopamine times each synapse's eligibility ([reward](reward.md)). The eligibility
trace is the per-stream memory of which synapses produced recent sampled acts,
`e <- actor_gamma * actor_lam * e + contrast` at each outcome (0.9 and 0.8 at the founders),
so a reward a few moments later credits the act that earned it. Greedy acts keep no
eligibility, and a calm `live` moment fades it.

```python
brain = Brain.compose(inputs=4, actions=2, seed=0, episodic=False)  # weights only
x, labels = np.eye(4)[:2], np.array([1, 0])
for _ in range(50):
    if (brain.predict(x) == labels).all():
        break
    brain.lesson(x, labels)                       # the contrast moves the synapses
brain.reset()
assert (brain.predict(x) == labels).all()         # the graph alone answers, after a reset

actor = Brain.compose(inputs=3, actions=2, seed=0)
actor.step(np.array([[1.0, 0.0, 0.0]]))                # a sampled act holds its contrast
actor.step(np.array([[0.0, 1.0, 0.0]]), reward=[0.0])  # its outcome: the contrast joins the trace
assert np.abs(actor.basal_ganglia.trace).sum() > 0
actor.reset()
assert actor.basal_ganglia.trace is None               # a new stream begins without eligibility
```

What the weights and eligibility serve:

- **An action paid a few moments later.** The eligibility credits it: in the
  [key-door chamber](../benchmarks/keydoor/README.md), at the reversal chamber's operating
  point, the creature feeds at 0.92 at delays 2 and 5, and `actor_lam=0` falls to 0.53 at
  delay 5. At delay 10 it reads 0.49 against a tabular learner's 0.83 to 0.90.
- **A multi-step chain whose stage is observed.** The weights and the store learn it together
  when each correct step pays: in the [chain chamber](../benchmarks/chain/README.md), 204
  greedy completions per thousand moments against tabular 250. With pay at the end only it
  reads 42, uniform random's rate; turn on [episodic memory](#when-to-turn-it-on) there.
- **A state the observation does not show.** No memory holds it
  ([what breaks a chain](reward.md#what-breaks-a-chain)); supply the state as an observation.

## Episodic memory: what happened in this situation before

Synapses learn slowly; an animal also learns in one go, from a place where food was found once
or a route a parent showed once. `Brain.compose(..., memory_choice=True)` adds an episodic
memory: records of what happened in a situation, which vote in every choice beside the slow
cortex ([when to turn it on](#when-to-turn-it-on)).

### How it works

```text
observation now ─────────────────────► place records ───┐
seen and done before ─┐                                  ├─► mean ─► vote ─┐
observation now ──────┴─► temporal context ─► context records ─┘           ├─► softmax ─► action
observation now ─► settle ─► motor readout ─────────────────────────────────┘
```

- **Two keys.** Each record is kept twice. `brain.hippocampus` holds the place records, keyed
  by the present [observation code](senses.md#the-observation-code). `brain.context_memory`
  holds the context records, keyed by the temporal context: the present code with a fixed
  fading trace of past codes and of the brain's own issued actions, where `memory_context` is
  the share the past keeps from one moment to the next. Two moments that look alike after
  different histories get different context keys, so a cue that is gone can decide the vote.
  The place records keep a situation together when a new path reaches it.
- **Records** hold one value per action in reward units. An uncoded key passes a fixed random
  separation first (1,024 cells, the 24 most active kept), so different situations write
  different synapses. A brain with a fixed code keys on its code
  as it is; a learned code's keys pass the separation as an observation's do.
- **What is written.** The outcome of the chosen action, once it is known: a reward or a
  punishment, the end of an attempt (`done`), a surprise that woke a calm `live` brain, or an
  outcome marked with a positive `salience=`. A zero outcome in the middle of an attempt
  writes nothing, unless `memory_extinction` extinguishes a record the action earned by its
  own payments in this place (the settings below). A demonstration (`teacher=`) is recorded
  for its situation as if the shown action had paid 1.0. `done=True` makes the last outcome
  known and starts the next attempt with a fresh temporal context and path.
- **Replay.** A reward replays the stream's recent path backwards, up to 8 decisions, each
  credited up to the reward discounted by the actor's `gamma` per step. Replay only adds
  credit. Its rules:
  - the path keeps one decision per situation, the one that left it; a decision whose own
    outcome was a reward or a punishment stays on the path even in an unchanged situation;
  - replay runs through decisions whose outcome was nothing and stops at the last known
    outcome, so nothing has to mark where an attempt began;
  - a reward with `done=True` is replayed whole into both kinds of record;
  - in a continuing life a reward is replayed in proportion to its surprise (the reward less
    what the memory expected of the paid decision): into each decision's context record, and
    into its place record only as far as the decision was an exploration rather than the
    brain's best guess. An expected reward replays nothing.
- **The vote.** At every choice, greedy or sampled, the mean of the place and the context reads
  is added to the motor readout before the softmax. The vote never enters the drive, so the
  settle is unchanged. The cortex learns from the actions taken, so what the memory chose and
  what paid consolidates into its synapses over time.
- **Two speeds.** Each store is a [`SynapticMemory`](#associative-store-the-hippocampus).
  A write corrects the stream's record halfway at once (`memory_rate` 0.5), a correction that
  fades by 0.9 per later write. The record all streams share, which `reset` keeps, moves by
  `consolidation`, faster with a supplied `salience=`. One outcome is remembered at once,
  repetition makes it lasting, and one misleading outcome cannot erase it.

### Turning it on

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=3, actions=2, arousal=True, memory_choice=True, seed=0)
x = np.array([[1.0, 0.0, 0.0]])
assert brain.act(x, greedy=True)[0] == 0   # the newborn's answer
brain.step(x, teacher=[1])                 # shown once: recorded at 1.0
assert brain.live(x)[0] == 1               # the record votes for the shown action
```

`memory_choice=True` in `Brain.compose` brings the founder values in the table below; each can
be passed beside it, also with `connectome=` for a custom anatomy. `Brain(...)` takes the
same names, but its defaults are 0 (and `memory_rate` 1.0), so name every setting there.
`memory_choice` needs `episodic=True`.

### Episodic settings

All of these are construction settings; `retune` refuses them.

| Setting | With `memory_choice=True` | Change it when |
| --- | --- | --- |
| `memory_choice` | `True` (founder `False`) | `False` keeps the [associative store](#associative-store-the-hippocampus), keyed by the senses and read into the motor drive |
| `memory_context` | `0.5` | Raise toward 1 (exclusive) when the answer depends on what came long before; lower when history does not matter; 0 keeps the place records alone and builds no `context_memory` |
| `memory_context_hold` | `None` | `"unchanged"` for stages that look alike where a miss leaves the observation as it was; `"punished"` where the world charges for misses ([the context hold](#stages-that-look-alike-the-context-hold)) |
| `memory_extinction` | `0` (off) | Set it (0.5) when a paid action can stop paying without a punishment, a reversal whose failures pay 0 ([extinction](#a-paid-action-that-stops-paying-extinction)). The rule is Sean McClane's |

The separation holds `width x 1,024` floats per store (105 MB at 12,824 inputs). A brain with
[`code=units`](senses.md#the-observation-code) has no separation of its own, so name a code for
a wide observation.

`memory_rate` (0.5 here, 1.0 without `memory_choice`), `memory_amplitude` and
`consolidation` belong to the [associative store](#associative-store-the-hippocampus);
`retune` changes them for both stores together. Constants of both stores: a stream's record
fades by 0.9 per later write, a reward replays up to 8 decisions, a demonstration is recorded
at 1.0, and the separation holds 1,024 cells, 24 active per key.

### One choice for acting, teaching and imagining

The vote joins after the settle, so the most active motor neuron is not always the act. Every
place that asks what the brain would do reads the same complete choice: readout plus vote, with
the context records read under that decision's own temporal context.

- **Lessons only where the answer differs** (`learning_surprise=True`,
  [learning](teach.md#lessons-only-where-the-answer-differs)) judge a row by the act the
  brain would issue. A demonstration corrects a brain whose memory turns a right cortical
  answer into a wrong act. An answer the memory makes right gets no lesson: no cortex update
  and no recorded demonstration. The cortex then learns those rows from the outcomes of its
  own acts.
- **Surprise in `live`** counts only the outcome of the brain's own best guess, the act it
  would issue greedily. A greedy act the vote decided is its own, and its outcome counts.
- **Imagination** (`imagine`) issues privately the act the brain would issue, to its efference
  copy, with the vote read under the branch's own temporal context. A returned state holds
  the cortex's response before the vote; act greedily
  on a [loaded copy](#check-that-a-memory-does-the-work) to read the complete act.

### When to turn it on

- A skill is taught by demonstration or by correcting executed mistakes.
- A chain of actions is paid only at its end ([sequential tasks](reward.md#what-breaks-a-chain)).
- The right answer depends on the situation's history: a cue that is gone, the next step of a
  sequence of the brain's own actions.

Beside the recall memory it keeps these uses. On the
[Stone Age chains](../benchmarks/context_key/README.md) with the
[composition for sequences](#one-composition-for-sequences) on, twelve development founders, the
fully observed chain passes greedily in 9 lives from reward and 12 from corrections with the vote,
and in none without it (sampled .74 and .99 against .06 and .21). The ambiguous chain's corrected
lives read .69 against .16. The vote stops earning where the read carries the cue: a cue the answer
does not show, taught by the terminal reward at the slow actor step, reads 1.00 at delays 0, 1 and 2
with or without the vote ([two cues](../benchmarks/credit/README.md)). The
[key-door chamber](../benchmarks/keydoor/README.md), a chain paid at its end, passes in 8 lives of 8
at delays 2 and 5 with the vote, against 5 and 3 with the store alone and 1 and 0 with the recall
memory beside the store. Its costs: two stores of 1,024 cells by the actions, so a brain holds about
twice the parameters of one without, a life takes 1.3 to 1.5 times as long, and an odour nursery
taught through `step` keeps its stable pair in 3 lives of 6 where the store alone keeps it in 6.

Leave it off where the history does not matter, and compare against the composition without it
on the task. The mechanism contracts are in
[`tests/test_episodic_memory.py`](../tests/test_episodic_memory.py) (writes, replay, the
temporal context recalling a cue that is gone) and
[`tests/test_episodic_policy.py`](../tests/test_episodic_policy.py) (the one-choice rule).

### A paid action that stops paying: extinction

The vote holds an old action after it stops paying, where its failures pay 0: an unmarked zero
outcome in the middle of an attempt writes nothing, and replay only adds credit. Without the vote
the store learns the new arm by itself: in a two-armed bandit through `live` whose paying arm moves
([extinction](../benchmarks/extinction/README.md)), twelve development founders, 12 lives of 12 reach
the new arm with the store alone, against 5 with the vote. Beside a vote that a task needs, set
`memory_extinction=0.5`.

With the setting, a zero outcome writes a learned zero for the chosen action where the
records' vote for it and its direct payment in this place both reach the setting. Only a known
outcome with a reward is a direct payment; replayed and demonstrated credit is never
extinguished. The rule reads the place: where one observation stands for several situations,
an action paid in one of them meets zeros in the others and is extinguished there as well.

In a two-armed bandit through `live` whose paying arm moves
([extinction](../benchmarks/extinction/README.md)), 11 of 12 lives learn the new arm with the
rule against 3 to 5 of 12 without it. It leaves the reward chambers of the
[preservation panel](../benchmarks/credit/README.md) at their gates. Its cost: the return to
the first arm after a second reversal is slower (median 15 trials against 6).

Other remedies and their limits:

- **A cost for misses.** Failures that cost 1 teach the bandit's new arm in 12 of 12 lives. In
  a chain learned from reward alone a miss penalty harms acquisition, since any nonzero
  intermediate reward stops replay at that decision.
- **A salience mark.** `step(..., salience=...)` and `live(..., salience=...)` mark an actual
  zero outcome as known, so the records write it while its reward stays zero for replay. The
  mark belongs to the preceding action's actual outcome, including after `wait`, and is checked
  before feedback changes anything. `live` keeps its arousal gate: a mark alone does not wake
  it or permit a calm write.

Compare with `memory_choice=False` on the same task.

### Stages that look alike: the context hold

`memory_context_hold` changes which temporal context keys a decision:

- **`"unchanged"`** keys a decision taken on the observation of the stream's previous decision
  as if that decision had not been taken, whatever its outcome (the replay path drops such a
  decision only when its outcome was nothing). Use it where a miss leaves the observation as it
  was and the stages look alike. It also takes out a decision that changes nothing visible, so
  it hides a right action that leaves the observation unchanged.
- **`"punished"`** takes a decision whose outcome was below zero, in the middle of an attempt,
  back out of the context. It needs a world that charges for misses.

A moment sensed without a decision (`wait`) ends the run of decisions the hold keys out, and an
imagined branch keys its decisions as the stream does.

On Will Glynn's Stone Age chains ([`benchmarks/context_key`](../benchmarks/context_key/README.md)):

| Chain | Arm | Clean greedy passes | Sampled success |
| --- | --- | --- | --- |
| stages look alike, a miss changes nothing | `"unchanged"` against none | 35 of 96 against 6 | .77 against .50 |
| every stage observed | `"unchanged"` against none | 40 of 48 against 43 | .92 against .94 |
| the right action changes nothing visible | `"unchanged"` against none | 0 of 48 against 9 | .32 against .44 |
| the right action changes nothing visible, misses cost .05 | `"punished"` against the cost alone | 20 of 48 against 13 | |

### Episodic limits

- Storage is fixed by the key width and the separation and does not grow with experience;
  correlated keys interfere, as in every delta-rule store.
- The keys are not learned: the memory recalls the same place and the same context, and does
  not discover which part of a history matters.
- Where history matters, the place records of aliased moments average over histories and
  halve the context records' share of the vote.
- Records are in reward units: rewards of a very different size weigh the vote more or less
  against the readout. `memory_amplitude` scales the read.
- A record that paid is not extinguished by failures that pay nothing unless
  `memory_extinction` is set ([above](#a-paid-action-that-stops-paying-extinction)).
- `predict` and `fit` answer independent samples and read no memory.

<a id="the-recall-memory-in-full"></a>

## Standalone memory stores

These stores work next to a `NeuralGraph` without `Brain.compose`: `cd.FastSynapses` with
its `cd.PatternSeparator`, `cd.SynapticMemory` used directly, and the fading traces of
`cadence.stream`. Each writes by one delta-rule correction or a fixed decay, and none
settles. The memories of a composed brain are in [Memory](memory.md); signatures are in the
[API reference](api.md#streams-cadencestream).

### One correction

For a unit key `k`, value `v` and matrix `M`, a write is `M += rate * outer(k, v - k @ M)`
(normalized LMS, close to [delta-rule fast weights](https://arxiv.org/abs/2406.06484)). At
`rate=1` the key reads back `v` at once, a repeated correct observation has zero residual,
and the read at an earlier key `q` moves by `dot(q, k)` times the error: orthogonal records
are kept, overlapping ones interfere. A read is one linear product, with no softmax and no
settling. `FastSynapses(rule="delta")` and `SynapticMemory` scale keys to unit length. A
fixed matrix cannot hold contradictory values, nearly parallel keys or more independent
values than its key rank exactly; measure retention on the streams the store will see.

```python
import numpy as np
import cadence as cd

def write(M, key, value, rate=1.0):
    error = value - key @ M
    M += rate * np.outer(key, error)
    return error

M, q, p, k = np.zeros((4, 2)), np.array([0.6, 0.8, 0, 0]), np.eye(4)[2], np.eye(4)[0]
write(M, q, np.array([1.0, 0.0]))
write(M, p, np.array([0.0, 1.0]))
old_q = q @ M
error = write(M, k, np.array([0.0, 2.0]))                 # k overlaps q, is orthogonal to p
assert np.allclose(k @ M, [0.0, 2.0])                     # read back exactly
assert np.allclose(q @ M - old_q, (q @ k) * error)        # q moved by the overlap
assert np.allclose(p @ M, [0.0, 1.0])                     # p kept its record
assert np.allclose(write(M, k, np.array([0.0, 2.0])), 0)  # a repeat writes nothing
```

### A store or a settling learner

A delta-rule store suffices when the outcome follows its key (the consequence of an action,
the reward of a choice, a fact seen at a place), the key holds what decides it, and keys
that should generalise overlap. Completing a partial reading, carrying context across a
delay and crediting late reward need a settled region ([write a cortex](anatomy.md#write-a-cortex)); the
[certificate](settling.md#the-settling-certificate) covers that settling and not the stores. A dense learner
moves every prediction with each update, which rehearsal of retained real observations
counters. Give a baseline the same replay policy, and never replay obsolete evidence.

### Reading and writing directly

`FastSynapses` ([signature](api.md#streams-cadencestream)) holds one `(pre, post)` matrix per
stream between declared neuron indices.

```python
memory = cd.FastSynapses(np.arange(4), np.arange(4, 6), rule="delta")
keys, values = np.eye(4)[:2], np.array([[1.0, 0.0], [0.0, 1.0]])   # two streams
memory.observe(keys, values)
assert np.allclose(memory.recall(keys), values)
memory.reset(2, rows=np.array([True, False]))          # only the first stream's episode ended

connectome = cd.Connectome.from_synapses(
    6, pre=[0, 1, 2, 3], post=[4, 4, 5, 5], count=[1] * 4,
    populations={"key": range(4), "value": (4, 5)},
)
graph = cd.NeuralGraph(connectome, cd.learning_neuron_model())
drive = np.zeros((2, connectome.n))
drive[:, :4] = keys
state = graph.settle_batch(memory.stimulate(drive))   # the read is added to the value neurons
memory.update(state, write=np.array([True, False]), post=np.array([[1.0, 0.0], [0.0, 0.0]]))
```

`observe(key, value, write=None)` takes `(batch, width)` ports; a boolean `write` gates each
stream, and every observe first decays all strengths by `decay` (reads never do).
`recall(key)` reads keys, `read(drive)` the key columns of a drive, and `stimulate(drive,
inplace=False, key=None)` adds the read of the drive's key columns, or of `key`, into the
post columns. `update(state, write=None,
post=None)` writes from the settled key activation with `post` as the observed value, and
only decays when `write` is `None`. Because `update` keys on activation and `stimulate` on
drive, a nonlinear activation can write under a key direction the read never uses; use
`observe`/`recall` when one external key serves both. Never write the store's own prediction back as evidence.

`rule="delta"` needs `rate` in `[0, 1]`, refuses `normalize` and `replace`, and a zero key
writes nothing. Scaling a key to unit length (`rule="delta"` or `normalize=True`) divides by
the norm of the whole key vector, the one step of the store that reads more than a synapse. The default `rule="hebb"` adds `rate * outer(k, v)`: `normalize=True` uses
unit keys and divides a read by the decayed write mass, and `replace=True` clears the rows of
a key's positive coordinates first, which suits one-hot slots and erases records under dense
keys. `amplitude` scales the read without changing the write target. An overflowing write
raises `ValueError` and changes nothing. `strength` holds the `(batch, key_width, post)`
matrices (`key_width` is the separator's `expansion`, else the pre count) and `writes`
counts rows written. `Learner.save` saves neither these stores nor the traces below; the
caller saves them.

`SynapticMemory(pre, post, decay=0.9, consolidation=0.05, separator=None, ...)` adds a
persistent `consolidated` matrix shared by all streams under the fast residual. Its rule,
salience and `value_mask` are in [the associative store](#associative-store-the-hippocampus).

### Streams and independent episodes

Rows of a batch are independent streams. Score a prediction before revealing its label.
On `FastSynapses`, `SynapticMemory`, every `Trace` and `History`, `reset(batch, rows=mask)`
clears the streams that ended and `keep(rows)` drops streams while the rest keep their
identities; `reset(batch)` without rows, or at a new batch size, clears all. A read at
another batch size returns the empty-stream baseline and leaves live records alone; a write
at another batch size starts fresh streams. `SynapticMemory.reset` clears only the fast
residuals and keeps the consolidated matrix, which is also its baseline; `clear()` erases both.

### Pattern separation

`PatternSeparator(inputs, expansion, winners, seed=0, center=0.0, fan_in=0, cover=0)` codes
keys before a store sees them: a fixed random projection onto `expansion` units, of which the
`winners` strongest positive entries stay. Codes with disjoint winners do not interfere. On
keys of cosine 0.9 a separated delta store revises at least 95 percent of its records
correctly, 50 points above the plain store ([separation tests](../tests/test_separation.py)).

```python
key_neurons, value_neurons = np.arange(32), np.arange(32, 40)
sep = cd.PatternSeparator(inputs=32, expansion=1024, winners=8, seed=0, center=0.99)
sep.habituate(np.random.default_rng(0).standard_normal((256, 32)))   # the environment's keys
separated = cd.FastSynapses(key_neurons, value_neurons, rule="delta", separator=sep)
separated.observe(np.ones((1, 32)), np.eye(8)[:1])
assert separated.key_width == 1024 and np.allclose(separated.recall(np.ones((1, 32))), np.eye(8)[:1])
```

`code(key, learn=False)` returns the `(batch, expansion)` code. With `center` above zero the
running `mean` is subtracted first, writes move it by forgetting factor `center`, and
`habituate(keys)` sets it from a sample before anything is stored; a moving mean can shift a
stored key's code. `fan_in` gives each code unit that many inputs (`index` and `weight` in
place of `projection`); `cover` (needs `fan_in`) reads each input through an ON and an OFF
channel with positive weights, each channel read by `cover` units. The projection is fixed:
it does not learn which differences matter. Storage and capacity grow to `expansion` rows
per stream. `SynapticMemory` refuses `center` other than 0, since a moving mean would move
persistent addresses; apply one fixed centering to inputs and queries instead.

### Traces

Fading traces carry the moment before into the next settle, one row per stream. Each reads
a `source` population and writes `amplitude` times the trace into the paired `target`
population through `stimulate(drive)`; decays lie in `[0, 1)`.

| Name | Defaults | Behaviour |
|---|---|---|
| `Trace(connectome, decay=0.5, amplitude=1.0, source="hidden", target="context")` | | `update(state)` moves the trace toward the source activation, `t <- decay t + (1 - decay) h`. `ringing(floor=0.1)` gives each source neuron's share of the trace, a salience for `ActorCritic.salience`. |
| `Efference(...)` | `source="motor"`, `target="efference"` | `issue(command)` decays toward the `(batch, motor)` one-hot of the executed action; `update` raises. |
| `History(connectome, decays, amplitude=3.0, source="sensory", target="history")` | | One target block per decay; `hear(observations)` moves block `d` by `h = d h + (1 - d) x`. |
| `cadence.stream.columns(index)` | | A slice for a contiguous range, else the index array; slices avoid gather copies on wide batches. |

Inside a composed brain the working trace, efference copy and heard history are owned by
[Memory](#working-trace).
