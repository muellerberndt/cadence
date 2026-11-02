# Memory

A memory is anything the brain keeps from an earlier moment and can use in a later answer.
Everything else is computed again from the current observation.

## Start with the need

Most applications need only one of these choices:

| The task needs | Start with |
| --- | --- |
| The interpretation from the previous moment | Nothing extra. The working trace is already on. |
| A remembered action and its measured outcome in the same situation | Nothing extra. The associative store is already on. |
| One-shot situations, demonstrations or reward replay | `memory_choice=True` |
| A relevant recent event and what followed it | `learned_code=units, attention=True` |
| Several witnessed facts in one answer | `learned_code=units, attention=True, recall_heads=k` |
| A recalled token as the answer, tokens never taught as answers included | `recall=True, recall_copy=2.0, recall_tokens=...` ([the copy pathway](#the-copy-pathway-opt-in)) |
| What it did itself, where the observations are alike | `recall=True, recall_action=True` ([its own actions](#its-own-actions-in-the-slots-opt-in)) |
| Both episodic voting and contextual retrieval | Use `memory_choice=True` and `attention=True` together. |
| A memory for a brain already trained without it | `brain.grow(recall=True)` ([grow and combine brains](grow.md)) |

If the present observation already contains everything needed for the action, keep the
opt-in memories off. Add one mechanism at a time and compare it with the same mechanism
silenced or erased.

The two opt-in choices are independent:

```python
from cadence import Brain

attention = Brain.compose(
    inputs=12, actions=2, learned_code=8, attention=True, seed=0,
)
episodic = Brain.compose(inputs=12, actions=2, memory_choice=True, seed=0)
both = Brain.compose(
    inputs=12, actions=2, learned_code=8,
    attention=True, memory_choice=True, seed=0,
)

assert attention.recall is not None
assert episodic.context_memory is not None
assert both.recall is not None and both.context_memory is not None
```

`attention=True` is not another name for episodic memory. It configures Recall: the learned
code supplies a compact, meaning-sensitive part of the key, Recall selects a weighted past
event and successor, and the neural equilibrium interprets that read. Episodic memory keeps
action values for places and temporal contexts and adds their vote to the action choice. Use
both when the task needs both kinds of evidence; compare each one alone to learn which memory
did the work.

Every memory ultimately changes the answer through the same composed brain. The working
trace, efference copy, heard history and Recall drive neural regions read by the association
region. The associative store drives motor neurons. Episodic memory adds a vote to the motor
readout. The resulting answer still has to [qualify](start.md#qualification-and-refusal).

Memories have three lifetimes. Per stream, cleared by `reset()` and by an ended episode: the
working trace, efference copy, heard history, recall slots and actor eligibility. Across
resets: the associative and episodic records and the recall address map. Permanent until
learning changes them: the learned weights. The per-stream traces advance at every moment;
under [`live`](start.md#routine-and-repair-live) a calm brain reads its records and
weights and changes neither, and an aroused one learns and writes. `reset()` keeps the
arousal state as it is, the income and error references that decide when a `live` brain
wakes among it, and `reset(keep=())` clears them
([reset and save](start.md#reset-and-save)).

<a id="memory-in-the-composed-brain"></a>

## Complete memory inventory

<a id="three-kinds-of-memory-in-brain"></a>

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
`brain.recall`. Every member keeps one contract
(`cadence.memories`): `stimulus(drive, moment)` adds what it holds to the drive of a moment,
`branch()` gives an imagined branch a private copy, `begin(keep=False)` starts its streams
afresh as `reset` does, `reset(batch, rows=None)` and `keep(rows)` act on streams, and
`to_dict()` and `arrays()` are what a checkpoint keeps of it, under `memories/<name>/`.

Stores used directly, outside a composed brain (`FastSynapses`, `PatternSeparator`,
`SynapticMemory`, the `Trace` family), are listed
[at the end](#stores-outside-the-composed-brain).

## Which memory for which task

The figures come from this repository's benchmarks and tests; each row names its operating
point where it differs from the founders.

| Task | Use | Evidence | Where it fails |
| --- | --- | --- | --- |
| Carry a cue across a short delay to its reward | working trace (on) | delayed response: seeds 8 to 13 solve at 0.96 to 1.00 ([test](../tests/test_context_defaults.py)) | a history that carries nothing useful, next row |
| Answer from the present when the history does not matter | the founder; try `working_memory_amplitude=0.0` as the control | continuing contextual bandit with `actor_gamma=0.0, actor_lam=0.0`, seeds 0 to 5: 0.23 to 1.00, mean 0.60, chance 0.25 ([test](../tests/test_context_defaults.py)) | the actor has to learn the trace away |
| Hold one cue and replace it with a later one over a delay | no measured memory; the working trace fails it | [recall benchmark](../benchmarks/recall/README.md): a tuned working-trace recipe without the associative store fails on fresh founders; recall memory and heard history are not measured there | a loop strong enough to hold a cue resists the next write |
| Alternate or sequence its own actions under identical input | efference copy at `efference_amplitude=3.0`, `efference_decay=0.0` | [rhythm](../benchmarks/rhythm/README.md), `modules=(32,)`, `episodic=False`, `working_memory_decay=0.1, scales={"trace": 12.0}`: alternation 0.97 and 0.93 against 0.50 and 0.40 with the trace alone; uniform random alternates at 0.50 | delayed-credit foraging ([key-door](../benchmarks/keydoor/README.md)): a copy at amplitude 0.3 feeds less at every delay, and at 3.0 the creature does not feed |
| Learn which action pays in a situation, and reverse quickly | associative store (on) | [reversal](../benchmarks/reversal/README.md), `modules=(32,)`, `working_memory_amplitude=0.3, scales={"trace": 12.0}, working_memory_decay=0.2, consolidation=0.25`, actor rate 0.1 (bias 0.01): memory alone 1.00 from 300 exposures against the graph alone 0.60 to 0.80 under rule A | at the composed founders the chamber stays at chance (`defaults` 0.48 to 0.53); noisy pay overwrites a one-shot record, next row |
| Noisy or signed pay | `memory_rate=0.2` | [competing](../benchmarks/competing/README.md): acquires as at 1.0 and halves the loss | half the loss is left |
| Situations whose keys overlap strongly | a standalone `FastSynapses` behind a centered, habituated `PatternSeparator` ([records](#standalone-memory-stores)); not available on the founder store | 64 keys of cosine 0.9, expansion 1024, 8 winners: revision accuracy at least 0.95, at least 0.5 above the plain store ([test](../tests/test_separation.py)) | the founder store has no separator; the episodic memory's fixed separator is uncentered and not measured by this test |
| Credit an action paid a few moments later | actor eligibility (on) | [key-door](../benchmarks/keydoor/README.md), `modules=(32,)` at the same operating point (scale 12.0, decay 0.2): fed 0.92 at delays 2 and 5; `actor_lam=0` falls to 0.53 at delay 5 | delay 10: 0.49 against a tabular learner's 0.83 to 0.90 |
| Learn a multi-step chain whose stage is observed | the founder (weights and store together) | [chain](../benchmarks/chain/README.md), pay at each correct step: 204 greedy completions per thousand moments against tabular 250; pay at the end only: 42, uniform random's rate | slow one-stream reward learning, see reversal |
| A state the observation does not show | no memory; supply the state as an observation | [chain](../benchmarks/chain/README.md) hidden: no arm beats the memoryless 62 | see [what breaks a chain](reward.md#what-breaks-a-chain) |
| Select the relevant two-moment context when the present question is identical | learned-code attention | [contextual choice](../benchmarks/attention/contextual_choice.py): four founders answer 128/128 intact queries; present-only, memory-off and uniform random score 50% | one held read; no within-settle query revision, multiple heads or continuing-world result |
| A moment heard once, the past at several timescales, what happened here before | recall memory, heard history, episodic memory | contract tests: [recall](../tests/test_recall_memory.py), [history](../tests/test_heard_history.py), [episodic](../tests/test_episodic_memory.py) | no general recall-horizon claim |
| Answer with the token the read found, tokens never taught as answers included | the copy pathway with declared tokens, `recall_copy=2.0` | [copy tokens](../benchmarks/attention/copy_tokens.py): six founders produce every token never taught as an answer (1.00; the founder memory 0.00 while its read names them at 0.32 to 1.00, the pathway drawn without tokens 0.00) and in-context succession stays at 0.94 to 0.99 against the founder's 0.95 to 0.99 | a successor never taught stays unanswered (0.00): the next port's gain reaches about 0.1 in 600 lessons; drawn without tokens at the recall projection's scale (1.2) the pathway costs taught answers (0.58 against 0.90) and succession (0.92 against 0.97) |
| Tell identical observations apart by what it did itself | `recall_action=True` | [own action](../benchmarks/attention/own_action.py), six founders: alternation 1.00 (founder recall memory 0.78, no recall memory 0.47, uniform random 0.48), a press every third moment 0.97 (0.65, 0.36, 0.46); with the commands erased 0.26 and 0.33 | the efference copy at `efference_amplitude=3.0` reads 1.00 on both; the Stone Age chain views are not measured here |

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
and one marking noise, help. The trace costs 64 neurons
and 4,096 synapses at the founder. Two pasts that settled alike are one context to it, and a
distractor that looks like evidence can overwrite the context of a small brain.

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

`history_decays` (`()`, each in [0, 1)) and `scales["history"]` (1.2, the projection's initial
magnitude, a `compose` argument) are construction settings; `history_amplitude` (3.0) can be
retuned. On a coded vector brain the blocks hold the code. Three decays at 256 inputs and
four actions add 768 neurons and 49,152 synapses to a founder of 21,004 synapses.

## Associative store (the hippocampus)

`brain.hippocampus` is a `SynapticMemory` from the observation code to the motor neurons, one
value per action in reward units: a persistent matrix `C` shared by every stream plus a fast
residual `F` per stream. Before each settle the read `C + F` drives the motor neurons at
`memory_amplitude`. For a unit-length key `k` and the observed value `v` of the chosen action:

```text
F     *= memory_decay
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
| `memory_decay` | 0.9, in [0, 1] | `retune` | how much of `F` each later write keeps |
| `memory_amplitude` | 1.0 | `retune` | the strength of the read as drive |
| `consolidation` | 0.05, in [0, 1] | `retune` | the slow rate into the shared `C` |

`brain.hippocampus.writes` counts the writes. The store costs key width times actions numbers
for `C` and as many per stream for `F`. It has no separator: a write at key `k` moves the read
at `q` by `q . k` times the correction, so overlapping situations interfere. Under
`memory_choice=True` the same names set both episodic stores, and `compose` starts
`memory_rate` at 0.5.

## Episodic memory

`memory_choice=True` makes the hippocampus an episodic memory. Each record is kept twice,
keyed by the observation code (place) and by the code with a fading trace of what the stream
saw and did (`brain.context_memory`), each behind a fixed separator (none with a fixed code),
and the mean of the two reads joins the motor readout in every choice. It records known
outcomes and demonstrations and replays a reward back along the recent path. Its settings,
evidence and limits are in [episodic memory](#episodic-memory-what-happened-in-this-situation-before),
below.

```python
frozen = {"actor_eta": 0.0, "actor_eta_bias": 0.0, "actor_eta_critic": 0.0}  # records alone
chooser = Brain.compose(inputs=3, actions=2, seed=0, memory_choice=True, **frozen)
place = np.array([[1.0, 0.0, 0.0]])
assert chooser.act(place, greedy=True)[0] == 0    # the newborn's answer
action = chooser.step(place)
for _ in range(6):                                # only action 1 pays
    action = chooser.step(place, reward=[1.0 if action[0] == 1 else 0.0])
assert chooser.act(place, greedy=True)[0] == 1    # the records vote for the action that paid
```

## Recall memory

<a id="recall-the-streams-own-moments"></a>

`recall=True` gives each stream a bounded list of recent moments. Every slot contains a key,
the event, and the event that followed it. Before a settle, Recall compares the current key
with the stored keys, makes one weighted read and drives `recall` and `next` regions with the
result. A learned address map can shift the query when it is enabled. The simpler
`attention=True` recipe leaves that map off and relies on the learned-code key itself.

Every head also reads how long ago each slot was written. This position term is on by default:
it starts at zero, so a new brain reads exactly as it would without it, and lessons teach each
head how far back to look, depending on what the moment hears. "Where was Bob at the beginning?"
and "where is Bob now?" can then read different slots of the same wording. See
[position](#position-how-long-ago-a-moment-happened); `recall_position=0` turns it off.

`reset()` empties the slots and keeps the learned address map;
`reset(keep=("slots",))` keeps both. Every `recall_*` setting, its cost and its limits are in
[the recall memory in full](#the-recall-memory-in-full). Settings removed after 0.84.3 and
their replacements are listed under [removed in this version](#removed-in-this-version).

```python
brain = Brain.compose(inputs=4, actions=2, modules=(16,), seed=1, recall=True)
brain.act(np.array([[1.0, 0.0, 0.0, 1.0]]), greedy=True)  # a name (0) stated with a value (3)
brain.act(np.array([[0.0, 1.0, 0.0, 0.0]]), greedy=True)  # another moment
name = np.array([[1.0, 0.0, 0.0, 0.0]])
recalled = brain.stimulus(name)[0, brain.connectome.populations["recall"]]
assert recalled[3] > recalled[1]                  # the moment that shares the name comes back
assert brain.lesson(name, [1])["address_rows"] == 1.0   # the lesson steps the address map
brain.reset()                                     # slots emptied, the address map kept
assert brain.recall.slots == 0 and brain.recall.w_x.any()
```

## Single-head learned-code attention (opt-in)

`attention=True` is the shortest supported path from learned meanings to contextual recall.
It combines two existing mechanisms rather than adding a separate attention network:

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
`recall_address=0.0`, `recall_recency=0.0` and `recall_temperature=0.1` only where they were
not explicit. A wider window and other `recall_*` genes remain available to experiments;
the window must be at least two and the successor read must stay enabled. The learned-code
width remains explicit: eight is the capacity tested by the small benchmark, not a general
default for other observations or worlds.

The zero address gain is deliberate. The **meaning code is learned**, but the extra map that
would transform the Recall query is not used by this recipe. Selection comes from similarity
between learned-code windows. This is the smallest mechanism that passed the contextual
choice controls. The [position term](#position-how-long-ago-a-moment-happened) is built here as
everywhere and learns whatever the address gain is, so after the first lesson the read also
weighs each slot's age. The contextual-choice and held-heads receipts were measured with
`recall_position=0`; at the founder position the same protocols passed every gate on the
founders run (contextual choice: 0 and 1; held heads: report 0 and 2, flee 0 and 1).

In the [contextual-choice benchmark](../benchmarks/attention/contextual_choice.py), every
final observation is the same literal ASK. Two stored episodes differ only in their earlier
context and successor. Four founders each answer 32/32 intact queries; present-only and
memory-off controls answer 16/32, as does uniform random. Erasing the relevant ASK record
breaks the answer, while erasing the other record leaves 32/32 correct, with no refusals.
The assay disables `episodic` to isolate this reader. Applications may keep the default
associative store or combine it with `memory_choice=True`; `attention` itself does not enable
or disable either one.

This is one held-read head. It selects once before the neural settle, then the shared
equilibrium interprets the read. The current interpretation cannot revise the query while
that settle forms. The benchmark therefore establishes a bounded contextual lookup, not
within-settle relevance, a Stone Age advantage or a general continuing-world benefit. Keep it
opt-in and compare it with attention off, episodic memory alone and uniform random on the task
being claimed. Several facts in one answer take [several heads](#several-heads-opt-in).

## Several heads (opt-in)

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
starts at zero. The heads compete for the slots (`recall_compete=True`, the founder): a slot
can be read in full once per moment, so the heads read in turn and each takes of every slot
at most the share the heads before it left. A head keeps the shape of its own attention where
the slots allow and gives what a taken slot cannot give to the slots it prefers next, with
its whole mass while enough is left; a head whose slots the heads before took whole reads
nothing. At birth, with every address map at zero, head 0 reads the best-matching moment and
head 1 the best of what head 0 left.

The address lesson after a teaching step reads the association's movement back through each
head's own synapses, so each head's map moves toward the records whose values would have
helped through that head, among the shares it reads with; the movement is what the lesson
still asks for once the heads have read. Reward eligibility at `recall_address_reward > 0`
keeps one block per head. One bank is written per moment, and the heads read it before the
settle and hold their reads through it, as one head does. With several heads the recipe
learns the addresses (`recall_address=0.01`) at the Recall founder's
`recall_temperature=0.25`.

Before 0.91.3 the heads did not compete. Their maps start at zero, so every head read exactly
what head 0 read, and the lesson's prior is the head's own attention: a sharp read (most of
the attention on one slot) gave it no other slot to move toward, and the heads stayed on the
same moment, at `recall_address=0` for good. On the stream of
[the heads' tests](../tests/test_recall_heads.py) two heads without competition read the
slot they share 1.83 times in one moment; competing, no slot more than once.
`recall_compete=False` keeps the earlier read, and a file of several heads saved before
0.91.3 loads with it, continuing as it was saved; `retune` changes it in a living brain.

In the [held-heads benchmark](../benchmarks/attention/held_heads.py) each trial witnesses
four facts `role -> value` with values drawn fresh, and one question asks for two of them in
one settle; both orders of some value pairs are never taught. Reported in two motor slots (six
values), two heads answer every taught pair on four founders, and on every trial each head
attends most to a different fact; one head that learns its address the same way answers 21 to
24% of the taught pairs, uniform random 3%. Combined into one action (go to the food unless the predator stands at an odd position,
then to the position five steps from it; ten positions), two heads answer every taught pair
and 98 to 99% of the food and predator pairs they were never taught; one head answers 59 to
76% of the taught pairs, uniform random 10%. The two reporting slots answer 77 to 98% of the pairs never taught;
every wrong slot on those pairs gave the other asked value, a binding the shared association
never met. These are competing heads (0.91.3); without competition (0.91.2, the same settings) the
one action answered 96 to 98% of the untaught pairs and the reporting slots 77 to 98%.
The [receipt](../benchmarks/attention/results/held-heads.json) is recomputed by the
regression suite. This is component evidence for reading several facts in one settle; a
decision in which one fact matters only together with another is not covered, and nothing here
is a continuing-world result.

The heads' addresses learn from teaching steps; reward trains them only at
`recall_address_reward` above zero. In a life of rewards alone, with the founder 0, the address
maps stay at zero: head 0 reads the best-matching moment and each later head the best of what
the heads before it left (with `recall_compete=False`, what head 0 reads).

**Chained heads (opt-in).** With `recall_chain=True` (a construction setting; it needs
`recall_heads` of 2 or more) the heads read in a chain: each head after the first adds to its
address what the head before it read, its recall and next reads through a map of its own that
starts at zero. One moment can then place a read where an earlier read pointed: in two hops
(`a -> b`, `b -> c`, ask `a`), the first head finds `a`, its next region holds `b`, and the
second head can learn to look for `b`. A newborn chain reads as the heads without it; the
address lesson steps the maps, and reward does not (`recall_address_reward` is refused with it).
A "think" moment remains the way to chain reads across moments: the second read's address then
comes from the cortex that holds the first ([#234](https://github.com/muellerberndt/cadence/issues/234)).

Use several heads when one answer needs several witnessed facts at once, and compare them with
one head and with the facts asked one per moment (one head reads one fact per moment through
the working trace). Long lesson curricula drive the shared readout into saturation once the
answers are right: teach with `learning_surprise=True` from the first lesson (or use `live`, whose
arousal does it). The benchmark switches to it at a running wrong share of .02, which serves its
1,024 lessons; over 4,096 lessons founders that never reached the threshold collapsed onto one
action ([lessons only where the answer differs](teach.md#lessons-only-where-the-answer-differs)).

## The copy pathway (opt-in)

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
  sensory region) at `-recall_copy`, so a token the moment already holds is not copied back.
  Every other synapse of the pathway starts at zero. The tokens must be in the recall values: a
  brain without a code, or a learned code joined to an observation no wider than it; a fixed
  code is refused.
- **Every synapse learns on its own.** A lesson moves the synapses of the tokens the read and
  the observation held; the identity of a token never read stays where it started, so a token
  never taught as an answer keeps its route to its action.
- **Why the echo.** A read addressed by the present moment holds that moment's own token in its
  value: in a succession task (a cue names a token, the answer is the token that followed it)
  the recall port holds the cue's token and the next port the answer. Copying the recall port
  alone answers the cue with itself; the echo removes what the observation already holds.
  On the benchmark's six founders with the echo synapses zeroed at birth (a development
  reading, outside the receipt), succession fell to 0.79 to 0.95, mean 0.87, against 0.94 to
  0.99 with the echo; untaught tokens were produced at 1.00 either way.
- **Why the next port starts at zero.** Which port holds the answer depends on the task: the
  heard token is on the recall port in a copy-back question, the successor on the next port in
  a succession. A newborn has to start from one; the recall port, the token the read delivers,
  is the start, and the next port's gain is learned. Started equal, the two ports drive the
  heard token and the distractor that followed it alike: a newborn produced the heard token on
  0.53 to 0.67 of untaught questions, and 300 lessons took that to 0.13 at most (founder 0, a
  development reading).

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

Uniform random answers 0.125. With the pathway the answer follows its read: the read and the
answer agree on every untaught and taught question, and on succession the answer trails the read
(1.00) by 0.03 on average and 0.06 at most, as the founder's does (0.03, 0.05). Two founders of a deep apical brain (`modules=(32, 24,
24)`, `apical=0.25`) read the same: untaught 1.00, succession 0.94 and 0.98 against 0.97 and
0.99. A successor never taught stays at 0.00 in every arm: the next port's gain grew to about 0.1
in 600 lessons. The weaker identity loses the untaught tokens on some founders, the association
having learned the taught answers first; the random pathway at the recall projection's own
scale, a plain direct projection, costs taught answers and succession, as the issue reported for
the composer.

Use it when the answer is a token the read can deliver, and compare it with the founder memory
and with the pathway drawn without tokens on the task. The options are construction settings,
described and saved only where they are on; a founder brain is unchanged.

## Its own actions in the slots (opt-in)

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
- **A newborn reads as today.** Every new synapse and map entry starts at zero: a newborn with
  the option settles, attends and acts as the same founder without it until learning gives them
  weight.

```python
mover = Brain.compose(inputs=4, actions=2, modules=(16,), seed=0, recall=True,
                      recall_action=True)
x = np.eye(4)[[0]]                                     # the same observation every moment
done = [int(mover.act(x, greedy=True)[0]) for _ in range(3)]
assert (mover.recall.values[0, :, 4:] == np.eye(2)[done]).all()  # what it did, slot by slot
assert (mover.recall.acted == np.eye(2)[done[-1:]]).all()        # the command of the moment before
```

In the [own-action benchmark](../benchmarks/attention/own_action.py) every observation is the
same one. Taught to alternate (the answer is the opposite of its own last action), six founders
alternate at 1.00 with the commands in the slots, against 0.78 for the founder recall memory,
0.47 without a recall memory and 0.48 for uniform random; erasing the commands before every
test moment brings them to 0.26. Taught to press every third moment (press when its own last two
actions were waits), they score 0.97, against 0.65, 0.36 and 0.46, and 0.33 with the commands
erased. The efference copy (`efference_amplitude=3.0`, `efference_decay=0.5`) reads 1.00 on both
([receipt](../benchmarks/attention/results/own-action.json)): for a brain that only needs its
last command the copy is the simpler mechanism; the slots keep the commands with the moments,
where a read can find them by content and position.

The [episodic memory](#episodic-memory-what-happened-in-this-situation-before) already keys its
context records on the issued commands: each decision's command joins the stream's temporal
context at the share `1 - memory_context`, so the option does not change the vote.

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

## Check that a memory does the work

Save the acquired brain, load two copies, erase one memory in one copy and compare their
answers on the same query. If they never differ, the brain answers without that memory.

```python
import os
import tempfile

# chooser and place from the episodic example above
path = os.path.join(tempfile.mkdtemp(), "brain.npz")
chooser.save(path)
remembering, forgetting = Brain.load(path), Brain.load(path)
forgetting.hippocampus.clear()                    # erase the place records
forgetting.context_memory.clear()                 # and the context records
assert remembering.act(place, greedy=True)[0] == 1  # the records vote for what paid
assert forgetting.act(place, greedy=True)[0] == 0   # without them the newborn answer returns
```

| Memory | Erase it on a copy |
| --- | --- |
| Any member of `brain.memories` | `brain.memories[name].reset(1)` empties its stream: the trace, the history, the recall memory's slots, the associative store's `F`, the episodic context and path |
| Associative store | `brain.hippocampus.clear()` erases `C` too |
| Episodic memory | `brain.hippocampus.clear()` and `brain.context_memory.clear()` ([episodic](#check-that-the-memory-does-the-work)) |
| Any memory read | `brain.retune(<memory>_amplitude=0.0)` silences it and keeps what it holds |

## Stores outside the composed brain

- [Standalone memory stores](#standalone-memory-stores): `FastSynapses`, `PatternSeparator`, `SynapticMemory` and the `Trace` family used directly.

## Episodic memory: what happened in this situation before

Synapses learn slowly; an animal also learns in one go, from a place where food was found once
or a route a parent showed once. `Brain.compose(..., memory_choice=True)` adds an episodic
memory: records of what happened in a situation, which vote in every choice beside the slow
cortex. It is opt-in ([when to turn it on](#when-to-turn-it-on)); [memory](#episodic-memory)
places it among the brain's other memories.

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
  separation first (`memory_expansion` cells, the `memory_winners` most active kept), so
  different situations write different synapses. A brain with a fixed code keys on its code
  as it is; a learned code's keys pass the separation as an observation's do.
- **What is written.** The outcome of the chosen action, once it is known: a reward or a
  punishment, the end of an attempt (`done`), a surprise that woke a calm `live` brain, or an
  outcome marked with a positive `salience=`. A zero outcome in the middle of an attempt
  writes nothing, unless `memory_extinction` extinguishes a record the action earned by its
  own payments in this place (the settings below). A demonstration (`teacher=`) is recorded for its situation as if the shown
  action had paid `memory_demonstrations`. `done=True` makes the last outcome known and starts
  the next attempt with a fresh temporal context and path.
- **Replay.** A reward replays the stream's recent path backwards, up to `memory_replay`
  decisions, each credited up to the reward discounted by the actor's `gamma` per step.
  Replay only adds credit. Its rules:
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
  fades by `memory_decay` per later write. The record all streams share, which `reset` keeps,
  moves by `consolidation`, faster with a supplied `salience=`. One outcome is remembered at
  once, repetition makes it lasting, and one misleading outcome cannot erase it.

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
  on a [loaded copy](#check-that-the-memory-does-the-work) to read the complete act.

### Turning it on

```python
import numpy as np
from cadence import Brain

brain = Brain.compose(inputs=3, actions=2, arousal=True, memory_choice=True, seed=0)
x = np.array([[1.0, 0.0, 0.0]])
assert brain.act(x, greedy=True)[0] == 0   # the newborn's answer
brain.step(x, teacher=[1])                 # shown once: recorded at memory_demonstrations
assert brain.live(x)[0] == 1               # the record votes for the shown action
```

`memory_choice=True` in `Brain.compose` brings the founder values in the table below; each can
be passed beside it, also with `connectome=` for a custom anatomy. `Brain(...)` takes the
same names, but its defaults are 0 (and `memory_winners` 32, `memory_rate` 1.0), so name
every setting there.
`memory_choice` needs `episodic=True`.

### Episodic settings

All of these are construction settings: pass them at construction; `retune` refuses them.

| Setting | With `memory_choice=True` | Change it when |
| --- | --- | --- |
| `memory_choice` | `True` (founder `False`) | `False` keeps the [associative store](#associative-store-the-hippocampus), keyed by the senses and read into the motor drive |
| `memory_context` | `0.5` | Raise toward 1 (exclusive) when the answer depends on what came long before; lower when history does not matter; 0 keeps the place records alone and builds no `context_memory` |
| `memory_replay` | `8` | Raise when rewards come at the end of long chains; 0 replays nothing |
| `memory_demonstrations` | `1.0` | Raise when shown actions should outweigh the brain's own trials; lower for an unreliable teacher; 0 records none |
| `memory_expansion` | `1024` (0 with a fixed code) | Raise when many situations must be told apart; 0 uses the key unseparated. With `code=units` it is set to 0 and a nonzero value is refused |
| `memory_winners` | `24` | How many separation cells stay active per key; at most `memory_expansion` |
| `memory_context_hold` | `None` | `"unchanged"` keys a decision taken on the observation of the stream's previous decision as if that decision had not been taken, the rule the replay path applies to its path; use it where a miss leaves the observation as it was and the stages look alike. It takes the wrong decision out of a decision that changes nothing visible as well, so it hides a right action that leaves the observation unchanged. `"punished"` takes a decision whose outcome was below zero, in the middle of an attempt, back out of the context; it needs a world that charges for misses. Will Glynn measured both on his Stone Age chains ([#202](https://github.com/muellerberndt/cadence/issues/202); `benchmarks/context_key`). On those chains, 48 fresh seeds per signal: `"unchanged"` passes the ambiguous chain cleanly in 35 of 96 lives against 6 (sampled success .77 against .50) and keeps the fully observed chain (40 against 43 of 48, sampled .92 against .94); on the chain whose right action changes nothing visible it costs (0 against 9 clean passes, sampled .32 against .44). `"punished"` with misses costing .05 passes the invisible-action chain in 20 of 48 lives against 13 for the cost alone |
| `memory_extinction` | `0` (off) | Set it (0.5) when a paid action can stop paying without a punishment, a reversal whose failures pay 0. A zero outcome then writes a learned zero for the chosen action where the records' vote for it and its direct payment in this place both reach the setting. Only a known outcome with a reward is a direct payment: replayed and demonstrated credit is never extinguished. The rule is Sean McClane's ([#210](https://github.com/muellerberndt/cadence/issues/210)) |
| `memory_fan_in` | `0` (dense) | Above 0 each separation cell reads that many random key values; use 32 for keys wider than 256 values. Needs `memory_expansion` and at most the key width; refused with `code=units` |

A dense separation holds `width x 1,024` floats per store (105 MB at 12,824 inputs); with
`memory_fan_in=32` it holds `1,024 x 32` indices and weights at any width. A brain with
[`code=units`](senses.md#the-observation-code) has no separation of its own.

`memory_rate` (0.5 here, 1.0 without `memory_choice`), `memory_decay`, `memory_amplitude` and
`consolidation` belong to the [associative store](#associative-store-the-hippocampus);
`retune` changes them for both stores together.

### When to turn it on

- A skill is taught by demonstration or by correcting executed mistakes.
- A chain of actions is paid only at its end ([sequential tasks](reward.md#what-breaks-a-chain)).
- The right answer depends on the situation's history: a cue that is gone, the next step of a
  sequence of the brain's own actions.

Leave it off where the history does not matter, and compare against the composition without it
on the task. The mechanism contracts are in
[`tests/test_episodic_memory.py`](../tests/test_episodic_memory.py) (writes, replay, the
temporal context recalling a cue that is gone) and
[`tests/test_episodic_policy.py`](../tests/test_episodic_policy.py) (the one-choice rule).

### Check that the memory does the work

Compare the same brain with and without its records, on loaded copies. Here the actor is frozen,
so only the records can change the answer:

```python
import os
import tempfile

frozen = {"actor_eta": 0.0, "actor_eta_bias": 0.0, "actor_eta_critic": 0.0}
paid = Brain.compose(inputs=3, actions=2, memory_choice=True, seed=0, **frozen)
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
nothing for that key. [Memory](#check-that-a-memory-does-the-work) lists the erase call
for every memory.

### Episodic limits

- Storage is fixed by the key width and the separation and does not grow with experience;
  correlated keys interfere, as in every delta-rule store.
- The keys are not learned: the memory recalls the same place and the same context, and does
  not discover which part of a history matters.
- Where history matters, the place records of aliased moments average over histories and
  halve the context records' share of the vote.
- Records are in reward units: rewards of a very different size weigh the vote more or less
  against the readout. `memory_amplitude` scales the read.
- Without `memory_extinction`, a record that paid is not extinguished by failures that pay
  nothing: while already aroused, an unmarked intermediate zero outcome writes nothing, and replay only adds
  credit. In a continuing life where
  a once-paid action stops paying and its failures pay 0, its record keeps voting for it and
  the new action is often not learned: in a two-armed bandit through `live` whose paying arm
  moved, 4 of 12 lives learned the new arm, against 12 of 12 when failures cost 1. That
  one-step result does not establish a remedy for chains learned from reward alone: any
  nonzero intermediate reward stops replay at that decision, and even tiny miss penalties
  harmed acquisition in the [reported chain comparison](https://github.com/muellerberndt/cadence/issues/210#issuecomment-6300283848).
  `step(..., salience=...)` and `live(..., salience=...)` can mark an actual zero outcome as
  known so the records write it, while leaving its reward at zero for replay. `live` keeps
  its arousal gate: a mark alone does not wake it or permit a calm write. The mark belongs
  to the preceding action's actual outcome, including after `wait`; it is checked before
  feedback changes anything. In the `step`-based
  [follow-up comparison](https://github.com/muellerberndt/cadence/issues/210#issuecomment-6320661291),
  marking misses recovered much of the penalty's harm but still lost against unmarked
  zero outcomes. Compare with `memory_choice=False` on the same task.
  `memory_extinction=0.5` extinguishes such a record: in the same bandit, 11 of 12 lives
  learned the new arm against 5 of 12 without it ([#210](https://github.com/muellerberndt/cadence/issues/210)). It reads the place: where
  one observation stands for several situations, an action paid in one of them meets zeros in
  the others, and its record is extinguished there as well.
- `predict` and `fit` answer independent samples and read no memory.

## The recall memory in full

`Brain.compose(..., recall=True)` keeps each stream's recent moments as slots and gives one
back when the present asks for it. Use it when a moment heard once has to return by its
content (a name stated with a value, asked for later), or when the brain needs what followed
an earlier moment that matches the present. The [memory overview](memory.md) compares it
with the other memories; the founder `recall=False` is the control. Each slot holds a key
(the event as heard), a value (the event) and a next (the value of the moment that
followed). Two source regions, `recall` and `next`, carry the recalled value and next into
the association region.

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
assert brain.recall.slots == 0 and brain.recall.w_x.any()
```

### How a moment is addressed

Before each settle the present moment computes its address

```text
q = heard * x + x W_x + c W_c + b
p = softmax(q . K / temperature - recency * age + a(x, c)[bucket(age)])
```

`x` is the observation as the memory hears it, `c` the settled association activity of
the moment before, `K` the stored keys, and `W_x`,
`W_c`, `b` a learned map that starts at zero, so before any lesson the address is the key
and a name finds the moment it was stated in. `a` is the
[position term](#position-how-long-ago-a-moment-happened), also zero until a lesson moves it.
`p` is kept in `brain.recall.attended`. The
`recall` region is driven at
`amplitude * sum p V` and the `next` region at `amplitude * sum p N`, held through the settle as a sensory drive is: the settle answers
on the mixture the address chose and cannot revise the choice. Both regions reach the
association region through one-way plastic synapses of initial magnitude `scales["recall"]`.

A number `recall_null` adds a slot of logit `null / temperature` that reads nothing, so a
moment that matches no slot recalls little.

### How it writes and how the address learns

After each act, and after each moment `wait` senses, every stream writes one slot and the
slot before learns the new value as its next. Beyond `recall_capacity` the oldest slot is
dropped. Imagination writes none.

The address lesson runs after each teacher lesson (`step(..., teacher=...)` or
`brain.lesson`). It reads the association region's movement between the nudged phases back
through the recall and next synapses, scores each slot's value and next against it, weighs
the scores by `softmax(score / address_temperature + log p)` and steps the map toward the
keys so weighted (`K*`) at the taught rows, each at the rate `min(recall_address, 1 / |z|^2)`
on `z = [x, c, 1]`. The report gives `address_rows`, `address_error` and `address_refused`
(rows whose non-finite step was refused), and with the position term `position_rows` and
`position_refused`. The founder step `"lms"` takes the error `K* - q`;
`recall_address_step="toward"` takes `K* - sum p K`, raising the named slots against the
attended ones whatever the keys' norms.

At the founder `recall_address_reward=0` rewards leave the map as it is. Above zero, each
sampled act keeps the step the address lesson would take, were its action a lesson's label,
as the map's eligibility. The actor holds it in a trace of its own (decaying by
`gamma * lam`, see [reward](reward.md)), and each outcome steps the map by
`recall_address_reward` times the dopamine times that trace; `learn` reports it as
`address_reward_step`. At 1.0 a unit of dopamine moves the map by one lesson's step.
`retune` refuses a change of `recall_address_step`, `recall_address` or
`recall_address_reward` while a sampled action's address contrast is pending.

### Position: how long ago a moment happened

A key says what a moment was; nothing in it says when. Two moments with the same event and
the same key differ to a read only by the recency charge, which always prefers the newer one,
so a question that depends on order ("at the beginning", "now", "the same beat one bar back")
cannot be answered from keys alone. A
transformer gives its keys a position for this reason. Here every head reads each slot's age,
the moments since the slot was written, and adds a term to that slot's logit:

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
  not freeze the position term. Rewards train it only at `recall_position_reward` above
  zero: at each sampled act the step the position lesson would take, were the act's action
  the lesson's label, joins the eligibility the actor traces, and dopamine steps the maps by
  `recall_position_reward` times one lesson's step (with the anchor, its maps too). Which slot
  a head reads by its age, the first told or the latest, is then reachable from reward alone
  ([#236](https://github.com/muellerberndt/cadence/issues/236)).

```python
brain = Brain.compose(inputs=8, actions=3, modules=(16,), recall=True, seed=0)
assert brain.recall.position == 32                       # on by default
assert not any(m.any() for maps in brain.recall.position_maps for m in maps)
plain = Brain.compose(inputs=8, actions=3, modules=(16,), recall=True, seed=0,
                      recall_position=0)                 # the memory without it
```

A checkpoint saved before 0.89.0 has no position term and loads without one, so a continuing
brain keeps reading as it did. The maps are saved with the brain when the term is on.

In the [look-back benchmark](../benchmarks/attention/look_back.py) a stream of five random
symbols is followed by a cue, `first` or `last`, and the answer is the symbol at the cued place.
Taught on the cue moment alone, the term learns to look back to the start of the episode on
`first` and to the newest slot on `last`: with the episode's boundary every founder answers both
cues at 1.00, against 0.29 to 0.80 per cue without the term and 0.25 for uniform random
([receipt](../benchmarks/attention/results/look-back.json)). Without the boundary one founder
answered `first` at 0.83.

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
without the option, and the lesson steps them with the position maps; rewards do not
([#237](https://github.com/muellerberndt/cadence/issues/237)).

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
moment the others wrote last) and when every stream ends an episode with its slots kept. A stream
that ends its episode alone begins the next one without a boundary, and a memory of one slot
(`recall_capacity=1`) holds none. The boundary is no written
moment, and its logit is its age's alone: a cue finds it through the position term, or with an
address that points away from every stored key.

The founder key is the present event alone, so where an event recurs with different
continuations the read returns a mixture of every occurrence's successor. One construction
setting widens the key:

| Key | Holds | Address width |
| --- | --- | --- |
| the founder key | `heard * x_t` | `code` |
| `recall_window=k` | the ordered last k events, the event j moments back weighted `recall_window_decay^j` | `k * code` |

`code` is the width of the [observation code](senses.md#the-observation-code), and
`brain.recall.address_width` reports the result. Keys, address, map outputs and a reward
eligibility take that width; values, nexts and the two regions keep the value width.

### Recall settings

Each gene is passed as `recall_<name>` beside `recall=True`; a `recall_` name without it is
refused. Retune genes change in a life through `brain.retune`; construction genes are fixed
when the brain is built. The options from `recall_address_step` on are described and saved
only where they differ from their founder. A saved brain keeps its genes, slots and address
map.

| Setting | Founder | When | What it sets |
| --- | --- | --- | --- |
| `recall` | `False` | construction | Adds the memory and its `recall` and `next` regions |
| `recall_amplitude` | `3.0` | retune | Drive of a slot that holds the whole weight |
| `recall_temperature` | `0.25` | retune | Softmax temperature over the slots |
| `recall_heard` | `1.0` | retune | Weight of the present event in its key and address |
| `recall_recency` | `0.25` | retune | Logit cost per moment of age |
| `recall_address` | `0.01` | retune | Address lesson rate, bounded per row by `1 / \|z\|^2`; `0.0` freezes the map |
| `recall_address_temperature` | `0.1` | retune | Temperature of the slots' scores in the address lesson |
| `recall_capacity` | `64` | construction | Slots per stream, oldest dropped first |
| `recall_next` | `True` | construction | The `next` region and each slot's next |
| `scales["recall"]` | `1.2` | construction | Initial magnitude of the two projections (a `compose` argument) |
| `recall_address_step` | `"lms"` | retune | `"toward"`: the named keys less the attended mixture |
| `recall_null` | `None` | retune | A number adds a null slot; a non-finite logit is refused |
| `recall_address_reward` | `0.0` | retune | Rate at which reward trains the address map |
| `recall_position_reward` | `0.0` | retune | Rate at which reward trains the position maps, and with the anchor its maps ([position](#position-how-long-ago-a-moment-happened)) |
| `recall_window`, `recall_window_decay` | `1`, `1.0` | construction | Window of the last k events, decay in (0, 1] |
| `recall_heads` | `1` | construction | Heads reading the one bank, each with its own port and address map ([several heads](#several-heads-opt-in)) |
| `recall_chain` | `False` | construction | Each head after the first also addresses what the head before it read ([chained heads](#several-heads-opt-in)); needs `recall_heads` of 2 or more |
| `recall_compete` | `True` | retune | Several heads read in turn, each taking of a slot at most the share the heads before it left; `False` reads every head on its own as before 0.91.3 (a file of several heads saved before 0.91.3 loads with it) |
| `recall_position` | `32` | construction | Ages below it get a bucket each, older ones one per doubling; every head learns a term per bucket ([position](#position-how-long-ago-a-moment-happened)); `0` builds none |
| `recall_position_rate` | `0.1` | retune | Rate of the position term's lesson step; `0.0` freezes it |
| `recall_anchor` | `False` | construction | Every head's position term also reads each slot's place in its episode, so "the first told" holds at every story length ([the anchor](#position-how-long-ago-a-moment-happened)); needs `recall_position` |
| `recall_boundary` | `False` | retune | Each stream that begins holds a silent boundary slot whose next the first event teaches ([the episode's boundary](#episodes-keys-and-windows)); a retune applies from the next stream that begins |
| `recall_copy` | `0.0` | construction | Above 0, plastic synapses from every read port into the motor region at that initial magnitude ([the copy pathway](#the-copy-pathway-opt-in)) |
| `recall_tokens` | `None` | construction | Per action the observation coordinate of the same token (`-1` for none); the pathway starts as their identity, echo suppressed; needs `recall_copy` |
| `recall_action` | `False` | construction | Every slot carries the executed command in its key and value, and the maps read the command of the moment before ([its own actions](#its-own-actions-in-the-slots-opt-in)) |

The [API reference](api.md#recall-memory-genes) gives the `Recall` class signature.

### All options in one example

```python
optioned = Brain.compose(
    inputs=4, actions=2, modules=(16,), seed=1, recall=True,
    recall_address_step="toward", recall_null=0.0, recall_address_reward=0.1,
)
other = np.array([[0.0, 1.0, 0.0, 0.0]])
optioned.act(fact)                               # every act writes a slot, sampled or greedy
optioned.act(other, greedy=True)
assert optioned.recall.slots == 2
optioned.stimulus(np.array([[0.0, 0.0, 1.0, 0.0]]))  # matches no slot
assert optioned.recall.attended.sum() < 1.0      # the null slot takes the rest
optioned.act(name)                               # sampled: the map's eligibility is kept
report = optioned.learn(np.array([1.0]), np.array([True]), fact)  # the outcome ends the episode
assert report["address_reward_step"] > 0.0
assert optioned.recall.slots == 0                # an ended episode empties the slots
windowed = Brain.compose(inputs=4, actions=2, modules=(16,), seed=1, recall=True, recall_window=2)
assert windowed.recall.address_width == 2 * 4
```

### Removed in this version

Settings of 0.84.3 that left in this version, each refused by name with a message naming
0.84.3 as the release that takes it, with the reading that retired it:

- `recall_write="aroused"`, a moment written only when its act was sampled: neutral on every
  chamber it was read on; every accepted moment is written.
- `recall_key="predecessor"`, the previous event and the last command in every key: neutral
  where it was read, and the window key and the efference copy carry the same information.
- `recall_episodes="keep"`, slots kept across an ended episode: it lowered success by .108 on
  a crafting chain whose observation names the stage; an ended episode empties the slots, and
  `reset(keep=("slots",))` keeps them across a reset as before.
- `recall_recency=None`, the power-law age charge `r * log(1 + age)`: it failed its
  confirmation on the camera task's founders; a number keeps the linear charge
  `recency * age`.
- `recall_value="cortex"`, the settled association activity as a slot's value: it failed all
  four binding tasks (token .06, succession .07, fresh bindings .55, camera .00); the values
  are the events.
- `recall_context` and `recall_context_decay`, a fading history of the earlier events added
  to every key: never read above zero in a chamber.

A checkpoint of 0.84.3 carries `value="event"`, `context=0.0` and `context_decay=0.5`, the one
value each had, with an empty history array; it loads, and neither `describe()` nor a
checkpoint of this version carries them.

### Recall cost

The two regions add one neuron per value coordinate each: `compose(250, 2)` grows from 380 to
880 neurons and from 20,354 to 52,354 synapses. `recall_action` adds one neuron per motor
neuron to every read port region, their synapses into the association, and as many rows to the
address and position maps and columns to the address; `recall_copy` adds the value width times
the motor neurons per port region (and with tokens one synapse per sensory neuron and action):
with two heads over a code of 128 and 323 actions the pathway adds 4 x 128 x 323 = 165,376
synapses. Each further head adds its two regions, their
projections and one more address map. The address map is `code` by `address_width`
plus the association width by `address_width`; with compose's 64 association neurons it is
0.66 MB at 256 inputs, 1,322 MB at 12,824 inputs on the uncoded observation and 8.9 MB with
`code=1024`. One stream's slots hold `capacity` keys, values and nexts, 0.39 MB at 256
inputs. The map is learned: a reset keeps it, `parameters()` counts it and a checkpoint saves it. The keys are the observation code, so
changing the code makes the stored keys inconsistent with new addresses.

### In a custom anatomy

`Brain(connectome, recall=True, ...)` needs a `recall` and a `next` region (or `recall` alone
with `recall_next=False`), each with one neuron per value coordinate,
`Genome.observation_width(inputs, code)`, and with `recall_heads=k` the regions `recall_h`
and `next_h` for h from 1 to k - 1 (`Recall.ports(k)` names them). With `recall_action` each of
those regions has a `<region>_action` region of one neuron per motor neuron
(`Recall.action_ports(k)`); with `recall_copy` synapses from every read port neuron into
`motor`, and with `recall_tokens` from every token's sensory neuron into its action, which the
brain starts as the declared identity. Without `recall=True` populations of those names
are ordinary ones.
`tests/test_recall_memory.py` checks that the library with `recall_position=0` reproduces the
reference memory `benchmarks/binding/index_read.py`, which has no position term, moment for
moment. To check that the memory does the work,
erase a copy's slots with `brain.recall.reset(batch)` before the query
([ablation](#check-that-a-memory-does-the-work)).

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
tests the first limitation without changing this supported recall API. One live route reads
from the already learned meaning, and one locally learned map turns that route into a new
query inside the same equilibrium. Its development founder passes acquisition and causal
record-use checks with 64 route coordinates, but it is not yet a `Brain.compose` option or
default: independent founders, lifecycle preservation and continuing-world comparisons are
still required.

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

`FastSynapses(pre, post, decay=1.0, rate=1.0, amplitude=1.0, normalize=False, replace=False,
rule="hebb", separator=None)` holds one `(pre, post)` matrix per stream between declared
neuron indices.

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
