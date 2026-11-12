# Parts of a brain: the catalogue

A Cadence brain is assembled from parts: senses, regions, a motor, short-term state, memories
and ways to learn. This page lists every part the library ships, in tables grouped by role, with
a small example per group and four whole brains built for a purpose. Scan for the parts your
purpose needs, then follow a part's name to the guide that owns it.

A part joins a brain in one of four ways:

| Way | When | Looks like |
| --- | --- | --- |
| An option of `Brain.compose` | at birth | `Brain.compose(inputs=4, actions=2, memory_choice=True)` |
| `brain.grow(...)` | later, on a trained brain, which keeps what it learned | `brain.grow(recall=True)` |
| `brain.graft(donor, ...)` | later, as trained regions of another brain | `host.graft(donor, ["sensory", "association", "motor"])` |
| A package beside the brain | around the brain's moments, in your own program | `cd.PopulationCode`, `cd.LongTermMemory`, `cadence.speech` |

A word on names: `modules=` sets the widths of the association regions, one row of the tables
below. Packaged modules that arrive with their synapses trained are planned
([parts that arrive trained](#parts-that-arrive-trained)).

| You want | Section |
| --- | --- |
| To see what a brain holds with nothing named | [the default brain](#the-default-brain) |
| A part by its role | [senses](#senses), [association regions](#association-regions), [the motor](#the-motor), [short-term state](#short-term-state), [memories](#memories), [learning](#learning), [many lives](#many-lives-and-where-they-run), [speech](#speech-pieces), [growing and grafting](#growing-and-grafting), [an anatomy of your own](#parts-for-an-anatomy-of-your-own) |
| A whole brain for a task | [brains for a purpose](#brains-for-a-purpose) |
| A part that comes trained | [parts that arrive trained](#parts-that-arrive-trained) |
| To know which parts refuse each other | [what does not go together](#what-does-not-go-together) |

Every example on this page runs as written, after these imports:

```python
import copy

import numpy as np
import cadence as cd
from cadence import Brain
```

## The default brain

`Brain.compose(inputs, actions)` with nothing else named builds a whole brain. Every other part
on this page is added to it or takes the place of one of its parts.

```python
brain = Brain.compose(inputs=4, actions=2, seed=0)
assert list(brain.memories) == ["working_memory", "history", "recall", "hippocampus"]
regions = sorted(name for name in brain.connectome.populations if "/" not in name)
assert regions == ["association", "history", "motor", "next", "prefrontal", "recall", "sensory"]
genes = brain.describe()["genes"]
assert genes["teaching"] == "deep" and genes["learning_surprise"] is True
```

| The default brain holds | As | Row below |
| --- | --- | --- |
| Senses | one `sensory` neuron per input | [senses](#senses) |
| One association region | 64 neurons, `modules=(64,)` | [association regions](#association-regions) |
| A motor | one `motor` neuron per action, one choice per moment | [the motor](#the-motor) |
| A working trace | the `prefrontal` region | [short-term state](#short-term-state) |
| A heard history of the moment before | the `history` region | [short-term state](#short-term-state) |
| A recall memory | 128 slots a stream, the `recall` and `next` regions | [memories](#memories) |
| An associative store | `brain.hippocampus` | [memories](#memories) |
| An actor and a critic | learning from rewards through `step`, `learn` and `live` | [learning](#learning) |
| Lessons | at the `"deep"` teaching point, taught where the answer is wrong | [learning](#learning) |
| A place to run | the fastest backend the machine has for its size | [many lives](#many-lives-and-where-they-run) |

## How to read the tables

The tables of parts share these columns.

| Column | What it says |
| --- | --- |
| Part | The part, linked to the section of the guide that owns it |
| For | Its purpose, in one sentence |
| Joins by | The exact option or call |
| Holds between moments | What a stream carries from one call to the next. `reset()` clears a stream's state and keeps what was learned and recorded |
| Learns | Whether lessons, rewards or a rule of its own change it |
| Arrives | `blank`: drawn from the seed or at zero, and learning shapes it. `fixed`: nothing in it learns. `empty`: a store without records. `trained`: shipped with trained synapses |
| Grow and graft | First whether `grow` adds the part to a trained brain (`grow` adds it), only `compose` builds it (at birth), or it is a setting that `retune` changes in a living brain. Then what a graft does with it, below |
| Reading | One measured figure with the benchmark or issue that holds it. A founder is a brain born from one seed, and fresh founders are seeds no setting was chosen on |

A blank cell means that the column does not apply to the part or, under Reading, that no reading
exists.

No part of 0.98.0 arrives trained. The column is the place where a built-in module is marked
`trained` when the library ships one ([parts that arrive trained](#parts-that-arrive-trained)).

What a graft does with a part is one of four things:

- **by region**: the part is a region of neurons, and `graft` carries its trained synapses when
  you name it together with a region it shares synapses with;
- **host's own**: a graft carries weights only, so records, settings, the actor, the critic and
  the arousal of the new brain are those of the brain whose method made it;
- **refused**: `grow` and `graft` refuse a brain that holds the part
  ([what does not go together](#what-does-not-go-together));
- **beside the brain**: the part is an object of your program, and neither call touches it.

Settings that tune a part (rates, amplitudes, temperatures) are in the part's guide, and
[tune and read a living brain](tune.md#defaults-and-expert-overrides) lists every one.

## Senses

How an observation reaches the brain. Choose by what the world gives you: a few numbers, a few
continuous readings, an image, a wide vector.

| Part | For | Joins by | Holds between moments | Learns | Arrives | Grow and graft | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [A vector of numbers](start.md#2-observations-actions-and-streams) | Senses that are a row of numbers at each moment | `inputs=n` | nothing | its synapses into the first region, by lessons and rewards | blank | at birth; by region (`sensory`) | |
| [Population code](senses.md#continuous-readings-a-population-code) | A few continuous readings (an angle, a velocity, a distance), each spread over tuned inputs | `code = cd.PopulationCode(ranges)` before the brain, then `inputs=code.inputs` | nothing | no | fixed | beside the brain; a checkpoint does not hold it | Cart-pole from four readings: 0 starts of 32 balanced on the raw readings, 1 to 22 through the code, three founders ([coaching chamber](../benchmarks/coaching/README.md), [#309](https://github.com/muellerberndt/cadence/issues/309)) |
| [An image through a visual cortex](senses.md#image-inputs) | An image, read through receptive fields that keep their draw | `inputs=(height, width)` or `(height, width, channels)`, with `features` and `field` | nothing | its receptive fields keep their draw; the regions that read them learn | blank | at birth; by region (`visual`, with the regions above it) | An `act` at 84 by 84 takes 163 ms on one core, against 2.6 ms through learned fields ([pixel chamber](../benchmarks/pixels/README.md)) |
| [Learned receptive fields](senses.md#learned-receptive-fields) | A rendered screen or a spectrogram, where an object reads the same wherever it appears | `learned_fields=features` with an image `inputs` | nothing | yes, by a local rule of their own at every moment the brain accepts | blank | at birth; refused | Fresh frames .974 and a band of places no lesson showed .948, where a convolutional network trained by gradient reads 1.000 and .145, six fresh founders ([pixel chamber](../benchmarks/pixels/README.md), [#284](https://github.com/muellerberndt/cadence/issues/284)) |
| [Ports beside the screen](senses.md#ports-beside-the-screen) | A body's own readings beside an image: health, a joint, the last action | `ports=k` with `learned_fields` | nothing | no: the values are read as given | fixed | at birth; refused, with the fields | |
| [A fixed sparse code](senses.md#the-observation-code) | Wide observations: the code bounds the size of every memory's key | `code=units` | nothing | no | fixed | at birth; by region (`sensory`, which is the code) | At 12,824 inputs the recall memory's address map takes 1,322 MB uncoded and 8.9 MB at `code=1024` ([recall cost](memory.md#recall-cost)) |
| [A learned code](senses.md#the-learned-code) | Observations met in the same contexts are read alike, so a lesson on some reaches the rest | `learned_code=units`, with `learned_code_fields` for independent channels | the stream's code, which the next settle starts from | yes, by local rules at accepted acts and aroused `live` moments | blank | at birth; `graft` takes other regions of the brain and never the `code` region | Held-out members of a group answered at 1.000 on 12 of 12 fresh founders, where the raw observation answers at chance ([the learned code](senses.md#what-the-code-is-for)) |

Audio reaches a brain as an image through the speech front end
([speech pieces](#speech-pieces)).

```python
angles = cd.PopulationCode([(-1.0, 1.0), (0.0, 5.0)], per=16)    # two continuous readings
reader = Brain.compose(inputs=angles.inputs, actions=3, seed=0)
assert reader.act(angles([0.2, 3.5]), greedy=True).shape == (1,)

eyes = Brain.compose(inputs=(84, 84), actions=3, seed=0, learned_fields=16, ports=2,
                     recall=False)
frame = np.zeros((1, 84, 84))
frame[0, 30:42, 20:32] = 1.0                                     # a square on a dark screen
row = np.concatenate([frame.reshape(1, -1), [[0.5, 1.0]]], axis=1)   # the image, then two ports
assert eyes.act(row, greedy=True).shape == (1,)
assert eyes.observation_code(row).shape == (1, 16 * 9 + 2)       # the pools, then the ports

wide = Brain.compose(inputs=2048, actions=3, seed=0, code=1024)
assert wide.observation_code(np.ones((1, 2048))).shape == (1, 1024)  # every memory keys on this
```

## Association regions

Where the brain forms its interpretation of what it senses. Every region settles together with
the others, and the motor reads the last one.

| Part | For | Joins by | Holds between moments | Learns | Arrives | Grow and graft | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [The association region](tune.md#brain) | The brain's interpretation of the moment, which the motor and the memories read | `modules=(64,)`: the last width is the association | its settled activity, which the next settle starts from | yes, by lessons and rewards | blank | at birth: its width is fixed; by region (`association`) | Spoken words: an association of 64 holds the word at .431 where its fields hold .626, and one of 256 holds .575 ([spoken phrases](../benchmarks/spoken/README.md), [#303](https://github.com/muellerberndt/cadence/issues/303)) |
| [A chain of regions](tune.md#deep-brains) | Features of features | `modules=(128, 64, 64, 64)` with `apical=0.25` and `teaching="deep"` | each region's settled activity | yes | blank | at birth: `grow` refuses a new region; by region (`module_0`, `module_1`, ...) | Held-out rows after 4,000 lessons: one region .58, two .66, three .81, four .86, against .89 for a network of two hidden layers trained by backpropagation ([depth](../benchmarks/depth/README.md)) |
| [The apical compartment](anatomy.md#the-apical-compartment) | Keeps the input from above below a share of each region's own input, so stacked regions learn | `apical=fraction` | nothing | no: it is a rule of the settle | fixed | at birth; a graft needs the same compartment in both brains | The chain's reading above is taken with it |
| [A wide association](tune.md#wide-brains) | More items, distractors or classes held at once | `modules=(256,)`; a new brain steps at its width by itself | its settled activity | yes | blank | at birth; by region | Grammar chamber at 256 neurons: paths .96 with the steps at its width against .94 at whole steps, the least life .81 against .67 ([wide brains](tune.md#wide-brains), [grammar chamber](../benchmarks/grammar/README.md)) |
| [Observers](start.md#add-optional-observers) | Regions that read every region and the motor and return their influence inside the same settle (System 2) | `observers=(8,)` | their settled activity | yes | blank | at birth: `grow` refuses a new observer; by region (`observer_0`, ...) | No measured advantage: what they bring has to be learned and measured on the task |
| [Tied projections](tune.md#tied-projections-opt-in) | The same kind of thing in several places, read through one set of synapses | `tied={("sensory", "association"): (3, 1)}` | nothing | yes: the copies hold one efficacy and step by the mean of their steps | blank | at birth; a tied brain grows where no tied region gains neurons; `graft` refuses a tied brain | A band of places no lesson showed .977 against .947 untied, at 23,236 parameters against 47,812, twelve fresh founders ([pixel chamber](../benchmarks/pixels/README.md#the-third-run-the-pool-rows-tied)) |

```python
deep = Brain.compose(inputs=12, actions=3, seed=0,
                     modules=(32, 16, 16), apical=0.25, teaching="deep")   # three regions
watched = Brain.compose(inputs=12, actions=3, seed=0, modules=(16, 8), observers=(8,))
shared = Brain.compose(inputs=12, actions=3, seed=0, recall=False,
                       tied={("sensory", "association"): (3, 1)})          # three places, one read
plain = Brain.compose(inputs=12, actions=3, seed=0, recall=False)
assert plain.parameters() - shared.parameters() == 512       # two of three copies of 4 by 64
assert "observer_0" in watched.connectome.populations
assert deep.act(np.eye(12)[[0]], greedy=True).shape == (1,)
```

## The motor

What the brain can do. The action is read from the motor neurons of the settled state.

| Part | For | Joins by | Holds between moments | Learns | Arrives | Grow and graft | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Actions](start.md#1-a-brain-has-senses-and-actions) | One choice at each moment from a fixed set | `actions=n`: one motor neuron per action, one softmax | a sampled action that awaits its outcome | yes: the synapses between the association and the motor | blank | at birth; by region (`motor`) | |
| [Slots](tune.md#operating-point-and-motor-competition) | An action with parts (kind, pitch, octave), or one choice per motor of a body | `slots=(3, 2)`, or `slots=k` for equal groups | as above, one choice per slot | yes | blank | `grow(slots=(4, 2))` adds slots after the brain's own; by region | |
| [A planning window](tune.md#a-planning-window) | The next k actions settled together, where two of them constrain each other | `actions=k * menu, slots=k` | as above | yes | blank | at birth | It is no lookahead: a sliding window of three chooses a route's first step at .31, against .33 word by word ([routes with a goal](../benchmarks/grammar/README.md#routes-with-a-goal-does-planning-pay-257), [#257](https://github.com/muellerberndt/cadence/issues/257)) |
| [Lateral competition](tune.md#operating-point-and-motor-competition) | A signed weight between the motor neurons of a slot | `lateral=w` (0.0 builds none) | nothing | yes: the lateral synapses are plastic | blank | at birth; by region (`motor`) | At -0.5 the undamped solve oscillates from 4 actions on ([motor competition](tune.md#operating-point-and-motor-competition)) |
| [A pathway from the fields to the motor](senses.md#spoken-words-a-spectrogram-as-an-image) | The motor reads the pooled fields directly, beside the association | `fields_motor=scale` with `learned_fields` | nothing | yes, by the same lessons | blank | at birth; refused, with the fields | Read speech: the letter said, given the letter held, rises from .42 to .78 with every row taught ([letters chamber](../benchmarks/letters/README.md), [#304](https://github.com/muellerberndt/cadence/issues/304)) |

```python
arm = Brain.compose(inputs=4, actions=5, slots=(3, 2), seed=0)   # a motor of three, a grip of two
seen = np.eye(4)[[0]]
arm.lesson(seen, np.array([[2, 1]]))                             # one label per slot
command = arm.act(seen, greedy=True)
assert command.shape == (1, 2)                                   # one choice per slot
```

## Short-term state

What a stream carries from one moment to the next. `reset()` clears what a stream carries, and so
does the end of an episode (`done`). The gains and maps a part learns are kept.

| Part | For | Joins by | Holds between moments | Learns | Arrives | Grow and graft | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Neural activity](start.md#6-the-stream-state-what-carries-from-moment-to-moment) | The settled state the next settle starts from | every brain | the state itself, one row per stream | no | | every brain; the streams start afresh after `grow` and `graft` | |
| [Working trace](memory.md#working-trace) | What the brain was just thinking: a cue across a short delay | every composed brain; `working_memory_amplitude=0.0` silences it | the association's settled activity of the moment before | the copy is fixed; the synapses that read it learn | blank | every composed brain; by region (`prefrontal`) | A cue carried across a short delay to its reward: 0.96 to 1.00 ([context tests](../tests/test_context_defaults.py)) |
| [Heard history](memory.md#heard-history) | What the brain just heard: the same observation needs different answers after different earlier ones | a new brain holds one block, the moment before; `history_decays=(0.2, 0.8)` for slower blocks; `history_decays=()` removes it | the observations as heard, one fading block per decay | the blocks are fixed; the synapses that read them learn | blank | `grow(history_decays=...)` adds blocks; by region (`history`) | Grammar chamber, held-out sentences: .995 with the one block against .868 without, twelve lives ([the heard history in a new brain](memory.md#the-heard-history-in-a-new-brain)) |
| [Efference copy](memory.md#efference-copy) | What the brain just did: sequencing its own actions under identical input | `efference_amplitude=1.0` | a fading one-hot of the commands it issued | the copy is fixed; the synapses that read it learn | blank | `grow(efference_amplitude=1.0)` adds it; by region (`efference`) | Alternation .97 and .93 against .50 and .40 with the trace alone ([rhythm chamber](../benchmarks/rhythm/README.md)) |
| [Held message](memory.md#held-message) | The whole episode held as one state, to answer from | `message=units`, with `message_phases`, `message_winners` and `message_boundary` | a sparse code of every moment of the episode, until the episode ends | the write is fixed; the synapses that read it learn | blank | at birth; refused | Trained clauses said from the message at .99 and held-out combinations at .95, where the heard history in its place says .07 and .00, six fresh founders ([grammar chamber](../benchmarks/grammar/README.md#the-brains-own-message)) |
| [Message blocks by role](memory.md#a-held-message) | An agent lies in the agent's blocks wherever in the episode it was heard | `message_roles=groups` with `message` | as the message, in blocks per role | as the message | blank | at birth; refused | Held-out combinations at .90 on 22 of 24 founders, .04 above blocks by position, and brackets .05 below ([protocol 6](../benchmarks/grammar/MEANING_PROTOCOL_6.md), [#301](https://github.com/muellerberndt/cadence/issues/301)) |
| [A read of one block](memory.md#held-message) | One block of the message at a time, chosen by the present moment | `message_read=True` with `message` | the read's maps, which `reset()` keeps | yes: the maps, by the recall memory's address lesson | blank | at birth; refused | Clauses, held-out combinations and role twins at .90 on 23, 18 and 20 of 24 fresh founders; longer sentences .00 ([grammar chamber](../benchmarks/grammar/README.md#held-out-combinations-said-from-the-state)) |
| [Carried gains](memory.md#how-strongly-a-carried-region-is-heard) | How loudly each carried region is heard, learned per task | a new brain holds `carried_gain=0.01`; `carried_gain=0.0` builds none | one gain per carried route, which `reset()` keeps | yes | blank: every gain starts at 1 | at birth; `grow` carries them by route; host's own | Grammar chamber: clauses 1.00 and three hops .42 against .95 and .18 at the trace's fixed amplitude, twelve lives ([carried gains](memory.md#how-strongly-a-carried-region-is-heard)) |
| [The read's route](memory.md#the-reads-drive-as-a-route) | The recall memory's read quiets where it does not pay | a new brain holds `recall_drive_gain=True` | one gain over every read port | yes | blank: the gain starts at 1 | `grow` adds it with a new memory only; host's own | The gain ends near 1.00 on the sequence tasks and at .33 on "was A told before B?" ([the read's route](memory.md#the-reads-drive-as-a-route)) |

```python
carrier = Brain.compose(inputs=6, actions=3, modules=(16,), seed=0,
                        efference_amplitude=1.0,          # what it just did
                        history_decays=(0.0, 0.8),        # what it just heard, at two speeds
                        message=12, message_winners=2)    # the whole episode
for moment in np.eye(6)[[0, 3, 5]]:
    carrier.act(moment[None], greedy=True)
assert carrier.working_memory.trace.any()                 # what it was just thinking
assert carrier.history.trace.shape == (1, 2 * 6)          # two blocks of six heard values
assert carrier.efference.trace.shape == (1, 3)            # one value per action
assert carrier.message.state.sum() == 3 * 2               # three moments, two units each
carrier.reset()                                           # a new stream carries none of it
assert not carrier.message.state.any() and not carrier.working_memory.trace.any()
```

## Memories

What the brain keeps and can use in a later answer. A new brain holds the recall memory and the
associative store. The weights themselves are the long-term skill, and
[learning](#learning) lists what changes them.

| Part | For | Joins by | Holds between moments | Learns | Arrives | Grow and graft | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Recall memory](memory.md#the-recall-memory-in-full) | A notebook of recent moments: a moment heard once returns by its content or by its place | a new brain holds it up to 4,096 observation-code coordinates; `recall=False` removes it | 128 slots a stream, which `reset()` empties; learned address and position maps, which it keeps | yes: the maps, by lessons and by reward | blank: maps at zero, no slots | `grow(recall=True)` adds it; by region for its ports (`recall`, `next`), host's own for its slots and maps | Six fresh founders: copying unique tokens 1.00, n-back .99 to 1.00, a period of 40 at 1.00 ([one composition for sequences](memory.md#one-composition-for-sequences)) |
| [More read heads](memory.md#several-heads-opt-in) | Two or more remembered facts in one answer | `recall_heads=2` | a read port and an address map per head | yes | blank | `grow(recall_heads=k)` adds heads; by region for their ports (`recall_1`, `next_1`, ...) | Two facts asked in one settle: every taught pair with two heads, 21 to 24% with one, four founders ([held heads](../benchmarks/attention/held_heads.py)) |
| [When a read item happened](memory.md#when-a-read-item-happened-opt-in) | "How long ago?", "was A told before B?" | `recall_when=True` | nothing more | the synapses that read it learn | blank | at birth, or with a grown memory | "How long ago?" .87 to .95 against .51 to .80 without it ([sequence battery](../benchmarks/sequence/README.md)) |
| [The learned value path](memory.md#the-learned-value-path-opt-in) | The answer is the token the read found, tokens no lesson used included | `recall_tokens=..., recall_gain=1.0` | one gain per read port | yes | blank: gains at zero | at birth; a brain with tokens grows no memory, heads or slots | Induction on tokens no lesson used: 0.00 without the path, 0.92 to 1.00 with it, six fresh founders ([sequence battery](../benchmarks/sequence/README.md)) |
| [Its own actions in the slots](memory.md#its-own-actions-in-the-slots-opt-in) | What the brain did, found together with the moment it did it in | `recall_action=True` | the executed command in every slot | the synapses that read it learn | blank | at birth, or with a grown memory; no new slots afterwards | Pressing every third moment: .97 against .65 for the memory without it ([own action](../benchmarks/attention/own_action.py)) |
| [Learned-code attention](memory.md#single-head-learned-code-attention-opt-in) | A notebook indexed by meaning: keys match where meanings do | `learned_code=units, attention=True` | slots keyed on two moments of learned codes | the code learns; the address map is frozen | blank | at birth | 32 of 32 contextual queries on four founders, 16 of 32 for the controls ([contextual choice](../benchmarks/attention/contextual_choice.py)) |
| [Associative store](memory.md#associative-store-the-hippocampus) | What paid where: one value per action per situation, learned in one experience | every composed brain; `episodic=False` removes it | records all streams share, which `reset()` keeps, and a fast residual per stream | yes: every outcome of a sampled act writes it | empty | at birth; host's own | A reversal learned in one stream: 1.00 after 300 exposures for the store alone, .60 to .80 for the weights alone ([odour nursery](../benchmarks/reversal/README.md)) |
| [Episodic choice](memory.md#episodic-memory-what-happened-in-this-situation-before) | One-shot records that vote in every choice: a demonstration shown once, a chain paid at its end, a cue that is gone | `memory_choice=True`, with `memory_context`, `memory_context_hold` and `memory_extinction` | place and context records, which `reset()` keeps; a temporal context and a replay path per stream | yes: outcomes, demonstrations and replay write it | empty | at birth; no new slots afterwards; host's own | A chain paid at its end: 8 lives of 8 at delays 2 and 5 with the vote, 5 and 3 with the store alone ([key-door chamber](../benchmarks/keydoor/README.md), [when to turn it on](memory.md#when-to-turn-it-on)) |
| [Long-term records](memory.md#long-term-records) | Facts told once and asked long after; a fact told again revises its record | `longterm=records` with `message=units` and a turn (`message_boundary=channel, message_boundary_holds=True`) | each stream's records, which survive `reset()` and every episode | the records are written; the synapses that read them learn | empty | at birth; refused | 10,000 records a stream: found and answered at .980 to 1.000, against .1025 to .1425 without the records at a chance of .125, six fresh founders ([memory scale](../benchmarks/memory_scale/README.md#the-third-declaration-a-brain-that-writes-and-reads-its-own-records), [#288](https://github.com/muellerberndt/cadence/issues/288)) |
| [Turns in a conversation](memory.md#turns-in-a-conversation-opt-in) | Many facts in one episode, questions that share few words with their statements, questions of two facts | `longterm_turn=channel`, with `longterm_distinct`, `longterm_best`, `longterm_rest` and `longterm_heads=2` | as above, with a count per code unit and a codebook | no | empty | at birth; refused | Wikidata facts read once, 10,000 of them: trained wordings .977 to .998, held-out wordings .957 to .996, twelve fresh founders ([encyclopedia](../benchmarks/encyclopedia/README.md), [#311](https://github.com/muellerberndt/cadence/issues/311)) |
| [`cd.LongTermMemory`](memory.md#a-long-term-memory-of-records) | A library of records beside the brain, kept by use under a fixed budget of slots | `cd.LongTermMemory(width, value_width, capacity)`; your program writes, reads and holds the read on input coordinates | its records with their depth and recent use; your program saves it (`arrays()`) | no: records are written and revised | empty | beside the brain | Among 1,000,000 records an exact key finds its record at 1.000 and a noisy one at .997 to 1.000, twelve fresh founders ([memory scale](../benchmarks/memory_scale/README.md#the-fourth-declaration-a-million-records), [#310](https://github.com/muellerberndt/cadence/issues/310)) |
| [The address of pairs](memory.md#a-million-records-opt-in) | A read whose cost grows far slower than the records | `pairs=16, expansion=16384, winners=256, candidates=4096` on the store; `longterm_pairs=16` and its settings in a brain | a table of the pairs each record is filed under | no | empty | beside the brain; at birth in a brain | A read among 1,000,000 records takes 3.4 ms on one core, 1.3 to 1.5 times its time at 100,000 ([#310](https://github.com/muellerberndt/cadence/issues/310)). A brain with it answers .960 to 1.000 of 100,000 facts heard once, twelve fresh founders ([the fifth declaration](../benchmarks/memory_scale/README.md#the-fifth-declaration-a-brain-with-its-own-records-at-100000)) |
| [Heads, parts and chains](memory.md#a-million-records-opt-in) | Several records in one read, a question of several parts, a chain of facts | `read(cue, heads=k)`, `read_each(cues)`, `chain(cue, links)` | nothing more | no | | beside the brain | Among 1,000,000 facts: four parts with a head each 1.000, eight facts about one subject .994 to 1.000 ([#310](https://github.com/muellerberndt/cadence/issues/310)) |
| [Keys held apart](memory.md#keys-held-apart-opt-in) | Two facts whose keys are one key keep two records | `apart=width`, then `write(key, value, source=row)` | a signature per record | no | | beside the brain | 1,000,000 facts, 4 in 100 sharing their key: found at 1.000 with it and .978 to .980 without ([memory scale](../benchmarks/memory_scale/README.md#facts-whose-keys-are-one-key), [#310](https://github.com/muellerberndt/cadence/issues/310)) |

The recall memory's position code (`recall_position`, `recall_anchor`, `recall_relative`,
`recall_slope`, `recall_grid`), its episode boundary (`recall_boundary`), a key over the last k
events (`recall_window`), `recall_cortex_norm` and the late release of the value path
(`recall_gain_after`, `recall_gain_next`) are settings of the one memory
([recall settings](memory.md#recall-settings)). A brain's long-term records are a
`LongTermRecords` with one `cd.LongTermMemory` per stream, and a read of either returns a
`cd.LongTermRead`.

```python
notebook = Brain.compose(inputs=4, actions=2, modules=(16,), seed=1)   # the two default memories
notebook.act(np.array([[1.0, 0.0, 0.0, 1.0]]), greedy=True)            # a name (0) with a value (3)
notebook.act(np.array([[0.0, 1.0, 0.0, 0.0]]), greedy=True)            # something else
drive = notebook.stimulus(np.array([[1.0, 0.0, 0.0, 0.0]]))            # the name alone, later
recalled = drive[0, notebook.connectome.populations["recall"]]
assert recalled[3] > recalled[1]                                       # the value comes back

voter = Brain.compose(inputs=4, actions=2, modules=(16,), seed=0, memory_choice=True)
assert "episodic" in voter.memories                                    # one-shot records that vote

library = cd.LongTermMemory(width=16, value_width=4, capacity=100)     # beside the brain
library.write(np.eye(16)[3], np.eye(4)[2])                             # a key with a value
slot, revised = library.write(np.eye(16)[3], np.eye(4)[1])             # the fact changed
assert revised and library.count == 1
assert library.value(library.read(np.eye(16)[3]))[1] == 1.0            # the record is current
```

## Learning

What changes a brain while it lives. Every brain learns from lessons and from rewards; the rest
are opt-ins that decide when it learns and what it keeps.

| Part | For | Joins by | Holds between moments | Learns | Arrives | Grow and graft | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Lessons](teach.md#supervised-lessons) | Show the right answer: every synapse compares its two neurons across three settles and takes a small step | `lesson(observations, labels)`, `step(observations, teacher=labels)` | the optimizer's running history | this is how the weights learn | | every brain; a grafted synapse starts its history afresh | |
| [Soft targets and row weights](teach.md#teach-a-distribution-weigh-the-rows) | A probability per action (a search, another brain), and rows that matter more than others | `lesson(observations, targets=p)`, `lesson(observations, labels, weights=w)` | nothing | as lessons | | every brain without `recall_tokens` or rehearsal | Inverse class weights raise the rarest answer on the two weakest founders from .74 and .83 to .92 and .93 ([scan-and-fire chamber](../benchmarks/coaching/README.md#the-scan-and-fire-chamber)) |
| [The teaching point](teach.md#which-teaching-point) | The size and form of a lesson's step | `teaching="deep"` (a new brain's), `"normalized"` or `"founder"`; `retune(teaching=...)` moves a living brain | a running scale per synapse | as lessons | | `retune` changes it | |
| [Surprise lessons](teach.md#lessons-only-where-the-answer-differs) | Teach only the rows the brain answers wrong | a new brain holds `learning_surprise=True`; `learning_surprise_share=True` for wide brains | nothing | as lessons | | `retune` changes it | Width 1,024: with the share every life ends at .95 to .99, and without it every fresh life collapses at least once ([width sweep](../benchmarks/sequence/README.md)) |
| [Rehearsal](teach.md#teach-several-types-in-sequence-rehearsal) | Types taught one after another are kept | `learning_rehearsal=1.0` | a reservoir of 512 lesson rows with their episodes | it replays lessons | empty | `retune` turns it on; a grown brain starts the reservoir afresh | Three cues taught in sequence: the first cue reads .994 to 1.000 with rehearsal and .32 to .74 without ([sequence battery](../benchmarks/sequence/README.md)) |
| [The actor and the critic](reward.md#eligibility-traces-and-dopamine) | Learn from what the world pays: each synapse keeps a trace of what it did, and one prediction error turns the traces into changes | every brain: `step(observations, reward=r)`, `live`, or `act` then `learn` | an eligibility trace per stream, and the action that awaits its outcome | yes | blank | every brain; host's own | A reward a few choices later: fed at .92 at delays 2 and 5, and at .53 at delay 5 with `actor_lam=0` ([key-door chamber](../benchmarks/keydoor/README.md), [eligibility](memory.md#learned-weights-and-eligibility)) |
| [Held credit](reward.md#a-reward-many-moments-after-the-decision) | A reward many moments after its decision, where the moments between look alike | `actor_hold=1.0`, with a tenth of the actor's step | each act's contrast until the episode ends | as the actor | | `retune` changes it | A T-maze of 50 moments: 6 of 6 fresh founders with the hold and the episodic choice, 2 of 6 without the hold ([delayed credit](../benchmarks/delayed_credit/README.md)) |
| [Entropy in the actor's step](reward.md#exploration-over-long-reward-lives-opt-in) | Long lives of small rewards that lock onto one button | `actor_entropy=0.1` | a pending act's surprise | as the actor | | `retune` changes it; `cpu` and `mlx` only | An aiming world: +0.918 against the default's -0.350, six fresh founders ([long reward lives](../benchmarks/actor_latch/README.md), [#296](https://github.com/muellerberndt/cadence/issues/296)) |
| [The value read](reward.md#decision-and-reset-contract) | The critic's value of a situation: what an episode cut by a time limit bootstraps from | `brain.value(observations)` | nothing: it reads and changes nothing | no | | every brain | |
| [Arousal](start.md#routine-and-repair-live) | One creature that answers calmly and learns while young, surprised or wanting | `arousal=True`, then `live` and `wait` | its level, its reward references and its age, which `reset()` keeps | the law is fixed; it decides when the brain learns | fixed | at birth; host's own | 40 lives through a rule that turns over twice: 40 acquire it, 39 reverse, 39 return, 40 keep an unrelated skill ([odour nursery](../benchmarks/reversal/README.md)) |
| [Maturity](reward.md#maturity-opt-in) | A brain that goes on learning keeps the skill it has | `maturity=True` | one slow variable, the use of its plasticity, and a noise scale per record | no: it scales the actor's steps and gates the store's writes | fixed | at birth; `grow` and `graft` keep the brain's own | A second skill met after the first: 51.9 points of the first lost at `memory_rate=1.0`, 41.6 at a new brain's 0.2, and 37.8 with maturity beside it, twelve fresh founders ([#169](https://github.com/muellerberndt/cadence/issues/169)) |
| [Records that fade](memory.md#records-that-fade-with-disuse-opt-in) | Unused records fade, and a skill practised for long fades far more slowly | `memory_fade=1.2e-4` | two sums of activity per key coordinate | no | fixed | at birth; `grow` and `graft` keep the brain's own | After 40,000 idle moments a skill practised for 2,000 keeps .36 and one practised for 40,000 keeps 1.00, 24 fresh founders ([#253](https://github.com/muellerberndt/cadence/issues/253)) |
| [Imagination](start.md#private-imagination) | Settle a supplied sequence of observations on a private branch | `brain.imagine([observations, ...])` | nothing: the live state is untouched | no | | every brain | |

The teaching recipes that use these parts are in [train a brain](teach.md) and
[learning from reward](reward.md): distilling another brain or a search, teaching on the brain's
own path, self-play, a critic school and a value lesson. `predict`, `accuracy` and `fit` answer
independent samples from rest and read the weights alone, so they test what the lessons left
([evaluate without fooling yourself](teach.md#evaluate-without-fooling-yourself)).

```python
cues, answers = np.eye(4), np.array([1, 0, 0, 1])
pupil = Brain.compose(inputs=4, actions=2, seed=0)
pupil.lesson(cues, answers)                                   # a lesson: the right answers
pupil.lesson(cues, targets=np.full((4, 2), 0.5))              # a distribution per row
pupil.lesson(cues, answers, weights=np.array([1.0, 1.0, 1.0, 3.0]))   # the last row counts thrice

player = Brain.compose(inputs=4, actions=2, seed=0, actor_eta=0.1, actor_eta_bias=0.01)
action = player.step(cues[[0]])                               # a sampled act awaits its outcome
action = player.step(cues[[1]], reward=[1.0])                 # the world paid the act before
assert player.value(cues[[2]]).shape == (1,)                  # the critic's value, read only

creature = Brain.compose(inputs=4, actions=2, seed=0, arousal=True,
                         maturity=True, memory_fade=1.2e-4)   # for a long life
creature.live(cues[[0]])
creature.live(cues[[1]], reward=[1.0])
assert creature.last_arousal["mode"] == "aroused"             # young brains learn from everything
```

## Many lives, and where they run

One brain can live many streams at once, and many brains can be stepped together. The same
table lists where a brain's arithmetic runs and how a moment is solved.

| Part | For | Joins by | Holds between moments | Learns | Arrives | Grow and graft | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Streams](start.md#2-observations-actions-and-streams) | One brain in many games or conversations at once: every row of a batch is one continuing life | the rows of `act`, `step` and `lesson` | each stream's own state; the weights are shared | the streams share what they learn | | every brain; `live` and `wait` take one stream | One brain's 128 streams on an NVIDIA L4: 0.119 ms a stream and moment ([many lives at once](backends.md#many-lives-at-once)) |
| [`Lives`](backends.md#many-lives-at-once) | Many brains of their own, stepped together: a population, a league, independent seeds | `cd.Lives(inputs, actions, lives, backend="torch")`, or `backend="cpu"` | each life's own weights, memories, critic and random state | each life learns alone | blank | a package around `Brain.compose`; it holds the parts its lives axis covers | On an L4, 7.4 times as fast as four CPU processes at 128 brains and 17.8 times at 1,024 ([many lives at once](backends.md#many-lives-at-once)) |
| [A life leaves and joins](api.md#lives-cadence) | Save one life, or put a lone brain in a life's place | `lives.life(i, stream=True)`, `lives.join(i, brain)` | the life's stream, with `stream=True` | | | the lone brain is an ordinary `Brain` | Lives that left continue as their lone brains: every action the same over 12 moments ([lives tests](../tests/test_lives.py)) |
| [A backend](backends.md#which-backend-should-i-use) | Where the arithmetic runs: the CPU, an NVIDIA GPU, an Apple GPU | `backend="auto"` (a new brain's), `"cpu"`, `"torch"` or `"mlx"`, with `device` and `precision` | nothing | no | | at birth; `Brain.load(path, backend=...)` moves a saved brain | The read-speech recipe trains an hour of audio in 126 seconds on an A10G with 128 streams, and in 691 on one core ([on a device](backends.md#the-read-speech-recipe-on-a-device)) |
| [The fast solve](tune.md#which-settle-and-learning-options) | The same equations in fewer neuron updates | a new brain holds `learning_fast_settle=True` | nothing | no | | `retune` changes it | About half of a grammar-chamber pass on one Xeon core ([what a moment costs](tune.md#what-a-moment-costs)) |
| [The compiled moment](tune.md#the-compiled-moment-opt-in) | The work around each solve in compiled calls on the CPU | `learning_compiled_moment=True` | nothing | no | | `retune` changes it | .54 to .56 of the default brain's seconds on one Xeon core ([the compiled moment](tune.md#the-compiled-moment-opt-in)) |
| [Qualified teaching](tune.md#which-settle-and-learning-options) | Lessons refused unless every phase satisfies the equations | `learning_qualified=True, learning_nudged_steps=1024` | nothing | no | | `retune` changes it | |

`dense_limit` and the environment variable `CADENCE_BACKEND` are settings of placement
([backends](backends.md#the-default-backend)).

```python
school = Brain.compose(inputs=4, actions=2, seed=0)
assert school.act(np.eye(4)).shape == (4,)                 # four streams of one brain

lives = cd.Lives(4, 2, 3, backend="cpu")                   # three brains of their own, seeds 0 to 2
actions = lives.step(np.eye(4)[:3])                        # one row per life
actions = lives.step(np.eye(4)[:3], reward=np.ones(3))     # each life learns from its own outcome
lone = lives.life(1)                                       # life 1 as a lone Brain, to save
assert isinstance(lone, Brain) and actions.shape == (3,)
```

## Speech pieces

`cadence.speech` is an opt-in package beside the brain: a program that does not import it gets a
library without it. Each piece is fixed arithmetic or a teacher's bookkeeping. The brain they
serve is an ordinary composed brain, and it holds all that is learned.

| Part | For | Joins by | Holds between moments | Learns | Arrives | Grow and graft | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`frontend.log_mel`](speech.md#hearing-audio-as-an-image) | Audio at 16,000 samples a second as 40 bands of log-mel energy every 10 ms | `log_mel(audio)` | nothing | no | fixed | beside the brain | |
| [`frontend.Hearing`](speech.md#hearing-audio-as-an-image) | The image a brain hears every 40 ms: the latest 24 frames, standardized per band | `Hearing.fit(frames)`, `hearing.steps(frames)`, then `inputs=hearing.shape` | per-band statistics, saved with `to_dict()` | fitted once on the training audio | fixed after the fit | beside the brain | |
| `frontend.LogMel`, `frontend.mel_bank`, `frontend.windows` | The front end at named settings for a reader of live samples, its filter bank, and the windows before they are standardized | `LogMel(hearing)` | nothing | no | fixed | beside the brain | |
| [`decode.Alphabet`](speech.md#the-composition) | The symbols a recogniser says: blank, a to z, the apostrophe and space | `Alphabet()`, then `actions=alphabet.size` | nothing | no | fixed | beside the brain | |
| [`align.Alignments`, `align.forced_path`, `align.even_path`, `align.log_beliefs`](speech.md#training-from-transcripts) | A teacher that knows each recording's text and places its letters on the steps by the brain's own beliefs | `paths.even(key, steps, text)`, `paths.realign(key, beliefs, text)`, `paths.lesson(keys, steps)` | one path per hearing of an utterance | no: it labels the brain's lessons | empty | beside the brain | From an even split the path converges on the true timing: .92 to .96 of steps right at the first pass, .999 at the last ([letters chamber](../benchmarks/letters/README.md)) |
| [`decode.collapse`, `decode.greedy`, `decode.WordBeam`](speech.md#reading-text-with-and-without-a-word-list) | Text from the brain's beliefs, plain or through a beam over a word list | `collapse(symbols)`, `WordBeam(words).decode(beliefs)` | nothing | no | fixed | beside the brain | Read speech of 2.5 hours: .33 characters after two passes, level with a CTC transformer of its size at two passes ([#304](https://github.com/muellerberndt/cadence/issues/304)) |
| [`correct.Rehearsal`](speech.md#adapting-to-a-speaker) | A correction taught beside earlier utterances, so a new speaker does not erase the others | `bank.remember(rows, labels)`, `bank.correct(brain, rows, labels)` | a bank of earlier utterances under a budget of bytes | no: it teaches the brain | empty | beside the brain | Twelve lives corrected on 13 held-out speakers: the other speakers go from .349 to .358 with it and to .352 with the correction alone ([#314](https://github.com/muellerberndt/cadence/issues/314)) |
| [`stream.StreamReader`](speech.md#streaming) | One stream of a trained brain on live audio, text out as each step is answered | `StreamReader(brain, hearing, words=...)`, `push(samples)`, `finish()` | the samples a step waits for, and the answers so far | no | | beside the brain | A 40 ms step answered in 1.05 ms at the median on one idle core ([#314](https://github.com/muellerberndt/cadence/issues/314)) |

```python
from cadence.speech.decode import Alphabet, collapse
from cadence.speech.frontend import Hearing, log_mel

rng = np.random.default_rng(0)
second = rng.standard_normal(16000) * np.abs(np.sin(np.linspace(0, 20, 16000)))  # 1 s of "audio"
frames = log_mel(second)                              # (98, 40): 10 ms frames, 40 bands
hearing = Hearing.fit([frames])                       # fit on your training audio
steps = hearing.steps(frames)                         # (24, 960): one row per 40 ms step
assert steps.shape == (24, 40 * 24) and hearing.shape == (40, 24)
alphabet = Alphabet()
assert alphabet.decode(collapse([0, 7, 7, 0, 15, 0])) == "go"   # repeats and blanks collapse
```

## Growing and grafting

Two calls put parts into a brain after its birth. Both return a new brain and leave the brains
they read unchanged. The rows under them are the other ways to combine, keep and change a brain.

| Call | For | What the new brain keeps | What starts at zero or afresh |
| --- | --- | --- | --- |
| [`brain.grow(...)`](grow.md#grow-a-trained-brain) | New, empty parts on a trained brain: `recall=True`, `recall_heads=k`, `history_decays=(...)`, `efference_amplitude=a`, `slots=(...)` | every learned weight, the optimizer's history, the memories' records and settings, the actor, the critic and the arousal | every synapse that joins a new part to the brain, and the streams |
| [`brain.graft(donor, regions)`](grow.md#graft-trained-regions-into-a-larger-brain) | Trained regions of another brain in this one, or a whole brain as one region (`into=`) | the donor's synapses, biases and gains among the grafted regions; everything else is this brain's own | the joints to the rest of this brain (`joints="zero"`), the grafted synapses' optimizer history, and the streams |
| [A second skill through lessons](grow.md#combine-the-skills-of-two-brains) | Two brains whose layouts differ | grow a slot on one, then teach it with the other's answers | the new slot |
| [`brain.save(path)`, `Brain.load(path)`](start.md#8-save-and-load) | The same life continued later, or on another backend | everything: weights, memories, streams, a pending action | nothing |
| [`brain.retune(...)`](tune.md#retune-the-same-life) | Another value of a setting in a life under way | everything | nothing; it refuses a setting that builds a part |

`grow(recall=True)` gives a brain the memory a new brain holds, and with it the heard history's
one block where the brain has none. `recall_founder=True` beside it grows the memory at its
founder settings and no history.

```python
rng = np.random.default_rng(0)
kinds = np.eye(6)[:4]                                     # four kinds of observation
donor = Brain.compose(inputs=6, actions=4, modules=(32,), seed=1, recall=False)
for _ in range(100):                                      # the donor learns the four kinds
    kind = rng.integers(0, 4, size=16)
    donor.lesson(kinds[kind] + 0.1 * rng.normal(size=(16, 6)), kind)
    if (donor.predict(kinds) == np.arange(4)).all():
        break
assert (donor.predict(kinds) == np.arange(4)).all()

bigger = donor.grow(recall=True, slots=(4, 2))            # a memory and a second slot of two
assert bigger.recall is not None and bigger.act(kinds, greedy=True).shape == (4, 2)
assert (bigger.predict(kinds)[:, 0] == np.arange(4)).all()   # what it knew is kept

host = Brain.compose(inputs=6, actions=4, modules=(32,), seed=2, recall=False)   # a newborn
grafted = host.graft(donor, ["sensory", "association", "motor"], freeze=True)
assert (grafted.predict(kinds) == donor.predict(kinds)).all()   # the skill arrives
assert not grafted.learner.plastic_synapses.all()         # the grafted synapses are frozen
```

## Parts for an anatomy of your own

The tables above are the parts `Brain.compose` assembles. The same parts are public classes, for
a brain you wire yourself: your own regions, a measured connectome, evolved wiring. `grow` takes
no custom anatomy, and `graft` takes one.

| Part | For | Guide |
| --- | --- | --- |
| `Genome`, `Region`, `Projection`, `develop` | List regions and projections, then lay them out into one `Connectome` | [your own anatomy](anatomy.md#genome-development-brain) |
| `cadence.regions`: `visual_cortex`, `cortex`, `motor_cortex`, `prefrontal_cortex` | The standard regions, as builders | [the catalogue of regions](anatomy.md#the-catalogue) |
| `Brain.compose(inputs, actions, connectome=...)`, `Brain(connectome, ...)` | Give a custom anatomy the composed founders, memories and learners | [genome, development, brain](anatomy.md#genome-development-brain) |
| `Genome.composed`, `Genome.observation_width` | The layout `compose` develops, to edit or evolve; the key width of its memories | [genome, development, brain](anatomy.md#genome-development-brain) |
| `evolve`, `genes`, `cadence.genome.mutate` | Selection over genomes and settings | [evolve a brain](anatomy.md#evolve-a-brain) |
| `Connectome` | A measured or hand-made wiring with named populations | [brains from a connectome](anatomy.md#brains-from-a-connectome) |
| `NeuralGraph`, `NeuronModel`, `Adaptation`, `Apical`, `BrainState`, `Equilibrium`, `Nudge` | The runnable graph of graded neurons and what a settle returns | [how the brain settles](settling.md), [the API](api.md#neuralgraph-cadence) |
| `cadence.blocks`: `layout`, `BlockTransport`, `block_contrast` | The cut of a graph into dense blocks that a settle sums over | [the API](api.md#blocks-cadenceblocks) |
| `Learner`, `LearnerConfig`, `GraphHolder`, `LearningPhaseError`, `layered`, `embedded`, `learning_neuron_model` | The lesson's learner over any graph, and graphs to try it on | [the free/nudged rule](settling.md#learning-the-freenudged-rule), [the API](api.md#learning-cadencelearning) |
| `ActorCritic`, `ActorCriticConfig`, `Maturity` | The reward learner over any learner | [a custom reward learner](reward.md#a-custom-reward-learner) |
| `Arousal`, `ArousalConfig` | The arousal law of one stream, usable alone | [the API](api.md#arousal-cadencearousal) |
| `Trace`, `Efference`, `History`, `Message`, `Recall` | The per-stream memories on any graph | [traces](memory.md#traces), [a held message](memory.md#a-held-message), [the API](api.md#streams-cadencestream) |
| `FastSynapses`, `SynapticMemory`, `PatternSeparator` | Delta-rule stores and the code that keeps their keys apart | [standalone memory stores](memory.md#standalone-memory-stores) |
| `LearnedCode`, `cadence.fields.LearnedFields` | The tables and rules of the learned code and of the learned receptive fields | [the API](api.md#the-learned-code-cadencecode), [what the stage is](senses.md#what-the-stage-is) |
| `LongTermMemory`, `LongTermRead`, `cadence.longterm.LongTermRecords` | The store of records, what a read of it returns, and a brain's own records | [the API](api.md#streams-cadencestream) |
| `cadence.memories`: `Memories`, `Moment`, `EpisodicRecords` | The one contract every memory of a composed brain keeps, and the episodic records | [the API](api.md#the-memories-cadencememories) |
| `calibrate_bias`, `naive_efficacy`, `preflight`, `seam_report` | Checks and starting points before the first lesson | [check before the first lesson](reward.md#check-before-the-first-lesson) |
| `certificate`, `Certificate`, `ep_structure`, `EPStructure`, `row_mass`, `lipschitz_constant`, `conformance` | What a settle guarantees, and a neuron-by-neuron reference to check a backend against | [the settling certificate](settling.md#the-settling-certificate) |
| `record_settlements`, `SettlementRecord` | Every settling step, for a display or an audit | [record every settling step](api.md#record-every-settling-step) |
| `Receipt`, `canonical_json` | A result with its sources, verifiable later | [receipts](evaluate.md#receipts) |
| `save`, `load` | A learner's checkpoint (`Brain.save` keeps a whole brain) | [checkpoints](api.md#checkpoints-cadencecheckpoint) |
| `available_backends` | The backends this machine can run | [backends](backends.md) |

```python
genome = cd.Genome.composed(6, 4, modules=(16,))           # the layout compose develops, to edit
anatomy = cd.develop(genome, seed=0)                       # one connectome, the same for one seed
custom = Brain.compose(6, 4, connectome=anatomy, seed=0, episodic=False)
names = [region.name for region in genome.regions]
assert names == ["sensory", "association", "prefrontal", "motor"]
assert custom.act(np.eye(6)[[0]], greedy=True).shape == (1,)
```

## Brains for a purpose

Four whole brains, each taken from a guide or a benchmark, so every figure has a receipt. Each
lists the parts it uses and why. The code is small enough to run in seconds; the readings come
from the full runs named beside them.

### A transcriber

The composition of the [speech recipe](speech.md): it hears recorded speech and says one letter,
a space or nothing every 40 ms.

| Part | Why |
| --- | --- |
| `frontend.Hearing` | Audio becomes a small image of 40 bands by 24 frames every 40 ms |
| `learned_fields=64` | The fields read the image the same at every moment and in every band. They double the letter read from the sound, from .18 to .35 |
| `fields_motor=1.0` with `learning_surprise=False` | A pathway from the fields straight to the motor, with every row taught: the letter said, given the letter held, rises from .42 to .78 |
| `modules=(256,)` | A wider association holds more of what the fields hold: on spoken words .575 at 256 neurons against .431 at 64, where the fields hold .626 |
| The actor's rates at zero | The brain learns from lessons alone |
| `align.Alignments`, `decode.WordBeam`, `stream.StreamReader` | A teacher that places the letters, text through a word list, and live audio |

```python
from cadence.speech.align import Alignments
from cadence.speech.decode import Alphabet, WordBeam
from cadence.speech.frontend import Hearing, log_mel
from cadence.speech.stream import StreamReader

rng = np.random.default_rng(0)
recording = rng.standard_normal(16000) * np.abs(np.sin(np.linspace(0, 20, 16000)))  # stands in
frames = log_mel(recording)
hearing = Hearing.fit([frames])                       # per-band statistics of the training audio
alphabet = Alphabet()                                 # blank, a-z, the apostrophe, space
ears = Brain.compose(hearing.shape, alphabet.size, seed=0,
                     learned_fields=64, modules=(256,), fields_motor=1.0,
                     learning_surprise=False,
                     actor_eta=0.0, actor_eta_bias=0.0, actor_eta_critic=0.0)
assert ears.parameters() == 1_285_724

rows = hearing.steps(frames)                          # one row per 40 ms step
labels = Alignments().even("first", len(rows), alphabet.encode("go on"))  # an even split to begin
ears.reset()
for row, label in zip(rows, labels, strict=True):
    ears.step(row[None], teacher=[label])             # one lesson and one act per step

ears.fields.learning = False                          # read without moving the fields
reader = StreamReader(ears, hearing, words=WordBeam(["go", "on", "no"]))
for start in range(0, 16000, 1600):
    reader.push(recording[start : start + 1600])      # 100 ms blocks, as a microphone gives them
assert len(reader.actions) == 24 and isinstance(reader.finish(), str)
```

Readings of this composition: on 2.5 hours of read speech it reads .33 characters after two
passes, level with a CTC transformer of its size at two passes, where CTC controls trained
offline to their stopping rule read .44
([letters chamber](../benchmarks/letters/README.md),
[#304](https://github.com/muellerberndt/cadence/issues/304)). At that letter level no learner
forms words. It answers a 40 ms step in 1.05 ms on one core and trains an hour of audio in 126
seconds on an NVIDIA A10G ([#314](https://github.com/muellerberndt/cadence/issues/314)). The
full recipe, with training from transcripts by forced alignment and corrections for a new
speaker, is `benchmarks/letters/recipe.py` ([the recipe script](speech.md#the-recipe-script)).

### An agent that answers from facts it read once

The brain of the memory-scale benchmark's
[third declaration](../benchmarks/memory_scale/README.md#the-third-declaration-a-brain-that-writes-and-reads-its-own-records),
at a small size. It hears a fact as four moments (who, did what, a turn, to whom), files it as a
record of its own, and answers a question about it in a later episode.

| Part | Why |
| --- | --- |
| The default brain | Its association learns, by lessons, to answer from what a memory holds |
| `message=70` | The held message is the key: what the episode is about, the same for a fact and its question |
| `message_boundary=TURN, message_boundary_holds=True` | The key is taken at the turn, before the fact says what it says, so a fact that changes keeps its key |
| `longterm=64` | Each stream's records: the moment after the turn is written under the key, and read back before the settle of a later answer |

```python
AGENTS, VERBS, PATIENTS, TURN = 6, 4, 4, 14     # agents 0-5, verbs 6-9, patients 10-13, the turn
STREAMS = 8                                     # eight streams, each with records of its own

def fact(agent, verb, patient=None):
    """Four moments: who, did what, the turn, and to whom (silent in a question)."""
    rows, moments = np.arange(STREAMS), [np.zeros((STREAMS, 15)) for _ in range(4)]
    moments[0][rows, agent] = 1.0
    moments[1][rows, AGENTS + verb] = 1.0
    moments[2][:, TURN] = 1.0
    if patient is not None:
        moments[3][rows, AGENTS + VERBS + patient] = 1.0
    return moments

def tell(brain, about, patient):
    brain.reset()                               # a new episode: the notebook's slots are emptied
    for moment in fact(about // VERBS, about % VERBS, patient):
        brain.act(moment, greedy=True)          # the moment after the turn becomes a record

def ask(brain, about, answer=None):
    brain.reset()
    *heard, silent = fact(about // VERBS, about % VERBS)
    for moment in heard:
        brain.act(moment, greedy=True)
    if answer is not None:
        brain.lesson(silent, answer)            # taught to answer from the record it reads
    return brain.act(silent, greedy=True)

knower = Brain.compose(15, PATIENTS, seed=0, message=70,
                       message_boundary=TURN, message_boundary_holds=True, longterm=64)
rng = np.random.default_rng(0)
pairs = rng.permutation(AGENTS * VERBS)
nursery, fresh = pairs[:16], pairs[16:]         # eight pairs that no lesson asks about
for _ in range(60):                             # the nursery: a fact is told, asked and taught
    about = rng.choice(nursery, size=STREAMS)
    patient = rng.integers(PATIENTS, size=STREAMS)
    tell(knower, about, patient)
    ask(knower, about, patient)

told = {}
for about in fresh:                             # eight new facts, each told once, no lesson
    told[about] = rng.integers(PATIENTS, size=STREAMS)
    tell(knower, np.full(STREAMS, about), told[about])
right = np.mean([ask(knower, np.full(STREAMS, about)) == told[about] for about in fresh])
print("facts told once, answered:", right)      # chance is .25
assert right > 0.6

records = knower.longterm.stores[0].count      # the first stream's records
changed = (told[fresh[0]] + 1) % PATIENTS       # the first new fact changes
tell(knower, np.full(STREAMS, fresh[0]), changed)
assert knower.longterm.stores[0].count == records           # revised in place: no second record
assert (ask(knower, np.full(STREAMS, fresh[0])) == changed).mean() > 0.6
```

Readings at full size: a brain with a 280-unit message answers facts it heard once among 10,000
records a stream at .980 to 1.000, and the same brain without `longterm` at .1025 to .1425, where
chance is .125 (six fresh founders,
[#288](https://github.com/muellerberndt/cadence/issues/288)). With the address of pairs it
answers .960 to 1.000 of 100,000 facts, and a revised fact is current at .995 to 1.000 and never
stale (twelve fresh founders, [#310](https://github.com/muellerberndt/cadence/issues/310)). For a
conversation, where one episode tells many facts and a question shares few words with its
statement, add `longterm_turn`, `longterm_distinct` and `longterm_best`
([turns in a conversation](memory.md#turns-in-a-conversation-opt-in),
[encyclopedia](../benchmarks/encyclopedia/README.md)).

### A body taught by a demonstrator

A hand that reaches a target and grips it, taught by a demonstrator that knows how. The recipe
is the [coaching chamber](../benchmarks/coaching/README.md)'s: watch first, then act on its own
path while the demonstrator labels every state it reaches.

| Part | Why |
| --- | --- |
| `cd.PopulationCode` | Each continuous reading is spread over tuned inputs, so "the target is close" has inputs of its own. As one raw number a reading drives the cortex in one direction only |
| `slots=(3, 2)` | One choice per motor: the arm (back, hold, forth) and the grip (open, close) |
| The default brain | Its lessons teach only the moments it answers wrong |
| `lesson`, then `act(greedy=True)` | Teach a moment before acting on it. A greedy act awaits no reward, so the next lesson is legal |

```python
STEP, NEAR, MOMENTS, BODIES = 0.1, 0.1, 16, 32
senses = cd.PopulationCode([(-0.3, 0.3), (0.0, 1.0)], per=16)   # the gap to the target, the hand

def demonstrator(hand, target):
    """One command per motor: the arm (0 back, 1 hold, 2 forth) and the grip (0 open, 1 close)."""
    gap = target - hand
    near = np.abs(gap) < NEAR
    return np.stack([np.where(near, 1, np.where(gap > 0, 2, 0)), near.astype(int)], axis=1)

def world(hand, target, command, rng):
    """The arm moves; a grip closed at the target takes it, and the next target appears."""
    hand = np.clip(hand + STEP * (command[:, 0] - 1), 0.0, 1.0)
    taken = (command[:, 1] == 1) & (np.abs(target - hand) < NEAR)
    return hand, np.where(taken, rng.random(len(hand)), target), int(taken.sum())

def episode(body, rng, own, teach=True):
    """Sixteen moments of 32 bodies; returns the targets taken. `own`: the brain's moves run."""
    body.reset()
    hand, target, takes = rng.random(BODIES), rng.random(BODIES), 0
    for _ in range(MOMENTS):
        seen = senses(np.stack([target - hand, hand], axis=1))
        shown = demonstrator(hand, target)
        if teach:
            body.lesson(seen, shown)                    # teach the moment, then act on it
        command = body.act(seen, greedy=True)
        hand, target, taken = world(hand, target, command if own else shown, rng)
        takes += taken
    return takes

def held_out(body, own=True):
    """A frozen copy on the same held-out starts: greedy acts, nothing taught."""
    return episode(copy.deepcopy(body), np.random.default_rng(100), own, teach=False)

body = Brain.compose(inputs=senses.inputs, actions=5, slots=(3, 2), seed=0)
rng = np.random.default_rng(0)
newborn, demonstrated = held_out(body), held_out(body, own=False)
for _ in range(8):
    episode(body, rng, own=False)                       # watch: the demonstrator's moves run
watched = held_out(body)
for _ in range(4):
    episode(body, rng, own=True)                        # its own path, every moment labelled
coached = held_out(body)
print("targets taken:", newborn, "newborn,", watched, "watched,", coached, "coached,",
      demonstrated, "by the demonstrator")
assert coached > 0.6 * demonstrated and coached > newborn
```

Readings of the recipe on a cart-pole (three founders, 32 held-out starts,
[coaching chamber](../benchmarks/coaching/README.md),
[#309](https://github.com/muellerberndt/cadence/issues/309)): the four raw readings balance 0
starts of 32 after watching, and through the population code 1 to 22. The first round on the
brain's own path drops every founder to 0 starts, and two of three reach 32 of 32 by the fourth
round. Following each of its own episodes with a watched one, no founder falls below 259 steps
of 500. The rounds swing, so read several founders and keep the best checkpoint
([teach on the brain's own path](teach.md#teach-on-the-brains-own-path)). A body with a camera
adds `learned_fields` and `ports` ([senses](#senses)).

### A creature that lives on what the world pays

One continuing life with no teacher: the `live` arm of the
[odour nursery](../benchmarks/reversal/README.md). It meets one of four odours at each trial and
approaches or avoids. Approaching the sugar odour pays, approaching another costs, and the sugar
moves without notice.

| Part | Why |
| --- | --- |
| `arousal=True` and `live` | The creature answers calmly from one settle and learns while young, surprised or short of its usual income |
| The actor and the critic, at `actor_eta=0.1, actor_eta_bias=0.01` | A tenth of the composed step: one stream's policy does not lock onto one action |
| The associative store, at `consolidation=0.25` | One paid outcome is a record at once, which carries the repair after the sugar moves |
| `working_memory_amplitude=0.3, working_memory_decay=0.2, scales={"trace": 12.0}` | The nursery's working trace: quiet, slow and read strongly |
| `recall=False` | Every odour shows what to do, so no notebook is needed |

```python
odours = np.eye(4)
creature = Brain.compose(4, 2, modules=(32,), seed=0, recall=False, arousal=True,
                         working_memory_amplitude=0.3, working_memory_decay=0.2,
                         scales={"trace": 12.0}, consolidation=0.25,
                         actor_eta=0.1, actor_eta_bias=0.01)
rng = np.random.default_rng(0)
odour = int(rng.integers(4))
action = creature.live(odours[[odour]])
right, modes = [], []
for trial in range(600):
    sugar = 0 if trial < 300 else 1                     # the sugar moves, unannounced
    best = int(odour == (sugar if odour < 2 else 2))    # approach (1) the sugar odour only
    right.append(int(action[0]) == best)
    reward = 0.0 if action[0] == 0 else (1.0 if best else -1.0)
    odour = int(rng.integers(4))
    action = creature.live(odours[[odour]], reward=[reward])
    modes.append(creature.last_arousal["mode"])

assert sum(right[260:300]) >= 36 and modes[260:300].count("routine") >= 36   # calm and right
assert "aroused" in modes[300:]                                              # the change wakes it
assert sum(right[-40:]) >= 36 and modes[-40:].count("routine") >= 36         # repaired, calm again
```

Readings of this composition: over 40 lives on ten fresh seeds, 40 acquire the rule, 39 reverse
when the sugar moves, 39 return when it moves back and 40 keep the unrelated pair of odours
([odour nursery](../benchmarks/reversal/README.md)). For a life of many thousands of moments in
which skills come and go, add `maturity=True` and `memory_fade`
([maturity](reward.md#maturity-opt-in),
[records that fade](memory.md#records-that-fade-with-disuse-opt-in)).

## Parts that arrive trained

### What `graft` does in 0.98.0

In 0.98.0 a trained part reaches a new brain through `graft`, with the donor brain in hand
([graft trained regions](grow.md#graft-trained-regions-into-a-larger-brain)).

- **The donor's weights are kept.** The grafted neurons take the donor's biases and gains, and
  the synapses among them take the donor's contact counts, efficacies and plasticity masks. They
  drive their neurons as they did in the donor.
- **The joints start at zero, or are kept.** A joint is a synapse of the host between a grafted
  neuron and the rest. With `joints="zero"` the graft settles at first on what it reads from its
  own regions, and learning gives the joints weight. With `joints="kept"` the joints keep the
  host's own weights, so the graft is read at once and the host changes at once.
- **`freeze=True` holds the graft.** The grafted synapses and neurons stop learning, while the
  joints and the rest of the host learn. Set `brain.learner.plastic_synapses` and
  `plastic_neurons` back to `True` to let the graft learn again.
- **What a graft carries.** Weights, biases and gains of the regions you name. Each grafted pair
  of regions holds the same number of neurons, and every synapse of the donor among them is a
  synapse of the host.
- **What a graft leaves.** The memories and their records, the settings, the actor, the critic
  and the arousal are the host's own. The streams start afresh.
- **A limit.** A donor region that settled with its donor's motor feedback, as an association
  region does, settles in the host without that feedback
  ([how a new part joins the brain](grow.md#how-a-new-part-joins-the-brain)).

The example under [growing and grafting](#growing-and-grafting) grafts a trained pathway into a
newborn and checks that the skill arrives: the grafted brain answers as its donor does, before
any lesson.

### Packaged modules (planned for v0.99.0)

A graft needs the donor brain in hand and a host whose regions match it in size and place. The
plan of [issue 321](https://github.com/muellerberndt/cadence/issues/321) packages a trained part
so that one call loads it and one call builds a host around it. **Nothing in this section is in
0.98.0.** Every name below is planned for v0.99.0 and can change before it ships.

- **A module file.** The trained regions as a brain file, with an interface record (the inputs
  the module reads, the regions it publishes, their sizes, the composition a host needs around
  it, the library version) and a card (what it was trained on, and its readings). The file is
  arrays and a JSON record. Loading one runs no code from the file, checks its interface and its
  checksums, and refuses a malformed file by name.
- **Built in and from outside.** The library ships the basic trained modules that many brains
  use, inside a stated budget of bytes in the wheel. A module file from outside the library
  loads the same way.
- **A module that stands alone.** A module is trained so that its own inputs drive it, which
  answers the limit above.

| Planned call | What it does | Status |
| --- | --- | --- |
| `cadence.modules.names()` | Lists the built-in modules | planned for v0.99.0 |
| `cadence.modules.load("place")` | Returns a built-in module | planned for v0.99.0 |
| `cadence.load_module(path)` | Returns a module from a file outside the library | planned for v0.99.0 |
| `Brain.compose(..., trained=[...])` | Builds a host with the module's regions in place and grafts it, frozen by default | planned for v0.99.0 |
| `brain.graft(module)` | Grafts a module into a host that fits | planned for v0.99.0 |

A module is worth shipping where three readings hold on fresh hosts, declared before the runs:

1. **The skill arrives.** The grafted module does in the host what it did in its donor, before
   any lesson.
2. **The host reads it.** With the module frozen, the host learns a task that needs the module's
   answer in fewer lessons than a host that learns the skill from birth.
3. **The skill stays.** After the host learns a second, unrelated task, the module's skill reads
   within its first reading.

| Module | What it holds | Arrives | Status |
| --- | --- | --- | --- |
| The place module | Where things are in a space, held while they are out of view ([issue 320](https://github.com/muellerberndt/cadence/issues/320)) | trained | planned for v0.99.0, the first built-in module |
| A visual system | Sight, from the learned receptive fields of [#284](https://github.com/muellerberndt/cadence/issues/284) | trained | in the backlog of [issue 321](https://github.com/muellerberndt/cadence/issues/321) |
| The hearing of letters | Letters heard in speech, from [#304](https://github.com/muellerberndt/cadence/issues/304) | trained | in the backlog of [issue 321](https://github.com/muellerberndt/cadence/issues/321) |

This table is where each built-in module is listed, with its card's readings, as it ships.

## What does not go together

The library refuses these combinations with a `ValueError` of one sentence that names the part.
Each row below was read from the library's own refusal on 0.98.0.

| Part | The library refuses | Do instead |
| --- | --- | --- |
| `grow` | A part inside the brain's own loop: `modules`, `observers`, `code`, `learned_code`, `lateral`, `apical`, other `inputs`. At zero such a part learns nothing, and at any other start it changes the brain | Compose the larger layout and `graft` the trained brain into it |
| `grow` | `arousal`, `maturity`, `memory_choice`, `memory_fade`, `message`, `longterm`, `learned_fields`, `tied`: choices of a brain's construction | Compose the brain with them |
| `grow` | New slots where a memory is laid out per action (`recall_action`, `recall_tokens`, `memory_choice`), where a tie cuts the motor region into blocks, or where observers sit above the motor of a deep brain | Compose the slots at birth |
| `grow` | A recall memory with `recall_tokens`, and heads for a brain with them | Compose the memory with its tokens |
| `grow` | A brain that records no construction (`compose(connectome=...)`), and rates set by hand | Graft a custom anatomy into a larger one; set the rates on the grown brain |
| `grow`, `graft` | A brain with a held message, long-term records or learned fields, as host or as donor | Compose the brain that needs them, and teach it |
| `graft` | A brain with tied projections or rates set by hand, as host or as donor | Graft untied brains |
| `graft` | A learned code's `code` region, or a joint that lands on it | Graft the regions above the code |
| `graft` | Regions that share no synapse; regions of different sizes; a donor synapse the host lacks; another neuron model or apical compartment | Graft the regions a trained pathway runs through, into a host laid out alike |
| `Lives` | `tied`, `memory_choice`, the efference copy, a held message, long-term records, a fixed or learned code, image inputs, an apical compartment, rehearsal, qualified teaching, `actor_entropy`, and a recall memory with several heads, a window, the command, the read of when or tokens | One brain's streams, or lone brains in a process pool |
| `Lives` | `backend="mlx"` and the MPS device | `backend="torch"` on CUDA or the torch CPU device, or `backend="cpu"` |
| Learned fields | A vector brain; `code` or `learned_code` beside them; an image under 8 by 8 pixels; fewer than 2 fields | An image `inputs` with `learned_fields` alone |
| `ports`, `fields_motor` | A brain without `learned_fields` | Add `learned_fields=features` |
| A learned code | An image brain; `code` beside it; the `torch` backend | A vector brain on `cpu` or `mlx` |
| `attention=True` | A brain without `learned_code`, or with `recall=False` | `learned_code=units, attention=True` |
| A held message | An image brain; a learned code | A vector brain, with or without `code=units` |
| Long-term records | A brain without `message=units`, or without a turn | `message_boundary=channel, message_boundary_holds=True`, or `longterm_turn=channel` |
| Episodic settings | `memory_choice` or `memory_fade` with `episodic=False`; `memory_context_hold` or `memory_extinction` without `memory_choice` | Keep the store, and name `memory_choice=True` |
| The recall memory | Above 4,096 observation-code coordinates `compose` asks you to choose; any `recall_` setting with `recall=False`; `recall_gain` without `recall_tokens`; `recall_tokens` on a coded brain; `recall_drive_gain` without `carried_gain` | `code=units` bounds the memory's maps |
| A tie | The sensory region of `learned_fields` with `ports`; a learned code's own projections; blocks that do not divide a region | Tie the pools without ports ([tied projections](tune.md#tied-projections-opt-in)) |
| A custom anatomy | `modules`, `observers`, `lateral`, `scales`, `features`, `field` or `fields_motor` beside `connectome=` | The anatomy brings its own layout |
| `live`, `wait` | A brain without `arousal=True`; several streams; `wait` with no action awaiting its outcome | `step` for many streams |
| `lesson` | `targets=` with `recall_tokens` or rehearsal; `weights=` with rehearsal; any lesson while a sampled action awaits its outcome | Report the outcome first, or replace the action with a greedy `act` |
| `actor_entropy` | An outcome on a torch device | `backend="cpu"` or `"mlx"` |
| `retune` | A setting that builds a part: `modules`, `tied`, `message`, `history_decays`, `carried_gain`, `memory_choice`, `memory_fade`, `maturity`, `arousal`, `recall_capacity` and the other construction settings | `grow` for the parts it adds; `compose` for the rest |
| `StreamReader` | A brain whose motor neurons do not match the alphabet | `actions=alphabet.size` |

```python
def refused(call):
    """The sentence the library refuses a call with."""
    try:
        call()
    except ValueError as error:
        return str(error)
    raise AssertionError("the library took the call")

tie = {("sensory", "association"): (3, 1)}
trained = Brain.compose(inputs=6, actions=4, modules=(16,), seed=0)
tied = Brain.compose(inputs=12, actions=3, seed=0, recall=False, tied=tie)
speaker = Brain.compose(inputs=6, actions=3, modules=(16,), seed=0, message=12, message_winners=2)

assert "changes the brain's own loop" in refused(lambda: trained.grow(modules=(16, 16)))
assert "without tied synapses" in refused(lambda: tied.graft(tied, ["sensory", "association"]))
assert "no tied projections" in refused(lambda: cd.Lives(12, 3, 2, recall=False, tied=tie))
assert "a message is not grown" in refused(lambda: speaker.grow(slots=(3, 2)))
assert "needs learned_code" in refused(lambda: Brain.compose(6, 3, attention=True))
```

A tied brain does grow where no tied region gains neurons, and a brain with a learned code, an
episodic choice, maturity or arousal grows the parts `grow` adds
([compatibility](grow.md#compatibility)).
