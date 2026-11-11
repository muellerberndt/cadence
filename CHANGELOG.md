# Changelog

## Unreleased

### A long-term memory of a million records (opt-in, #310)

`cd.LongTermMemory` reads a million records through an address whose work grows far slower than
the records. Every option is off by default; without them the store and a brain with records are
those of 0.97.0, to the bit.

- **An address of pairs** (`pairs=units`). A record is filed under every pair of its most driven
  address units, in a table of cells sized to the capacity. A cue's winners name the pairs it
  reads, and the slots found under the most pairs go to the read's competition. A key is centred
  before it drives the units, so a key of zeros and ones and a dense key are filed alike.
- **`fill(keys, values)`** writes many records told once in one pass.
- **`read_each(cues)`**: heads reading in turn, each with a cue of its own, for a question of
  several parts. **`chain(cue, links)`**: a read's value laid on the next cue.
- **Keys held apart** (`apart=width`): a write keeps a signature of the fact's `source`, so two
  facts whose keys are one key hold two records, and a read that carries its source takes its own.
- **A brain's own records** take the address through `Brain.compose(longterm=records,
  longterm_pairs=, longterm_expansion=, longterm_winners=, longterm_candidates=, longterm_fade=)`.
- With the `fast` extra (Numba) the count over the pairs runs in one compiled loop; NumPy gives
  the same slots without it.
- Instruments `benchmarks/memory_scale/million.py` and `facts.py`, declared in
  [`PROTOCOL_4.md`](benchmarks/memory_scale/PROTOCOL_4.md).

### The encyclopedia: facts read once (opt-in, #311)

A brain reads a corpus of facts once, writing its own records, and answers questions about them
from those records. Every option is off by default; without them a brain with records reads as
before.

- **Records kept as lists.** With `longterm_distinct` each stream keeps each record's entries and
  each entry's records, so the records and the codebook grow to a hundred thousand records and
  more. A checkpoint keeps the entries as rows of (stream, slot, entry).
- **The record that holds the most** (`longterm_best=True`): the read takes the record holding the
  largest share of the cue's weight, the latest among equals, so a question that names a subject
  and a relation reads the record of both, and a fact told again is read from its latest telling.
- **What the question asks beyond the first record** (`longterm_rest=True`): a second read is
  cued by the first record's value and the question's entries that record does not hold, so a
  question of two facts reads the second fact. The regions hold the chain from its end, the
  answering record on `longterm`, the first record on `longterm_1` where the brain has two heads.
- The encyclopedia chamber, `benchmarks/encyclopedia/`: facts from Wikidata (CC0), questions in
  trained and held-out wordings, revisions, questions of two facts, and matched controls (a
  closed-book LSTM and transformer, a transformer with the facts in its window, and BM25
  retrieval read by a transformer).

## 0.97.1 - 2026-11-11

Opt-ins built on the v0.98.0 lanes, and the chambers that read them. No default changes: with
every new option unset a brain acts, learns and describes to the bit as on 0.97.0 on
`backend="cpu"`, its file holds the same arrays, and `Lives` keeps every action and parameter. A
saved file's last lesson report adds `demonstration_agreed_slots`; 0.97.0 loads it.

```bash
python -m pip install cadence-net==0.97.1
```

### Opt-ins

- **`tied`** (#308, #284): `Brain.compose(..., tied={(pre, post): (blocks of pre, blocks of
  post)})` shares one set of synapses across equal blocks of a projection; a tied synapse steps
  by the mean of its copies' steps, `parameters()` counts it once, and 0.97.0 refuses the file
  ([tied projections](docs/tune.md#tied-projections-opt-in)).
- **`fields_motor=scale`** (#303, #304): with `learned_fields`, a plastic projection from the
  pools to the motor region, taught by the same lessons
  ([spoken words](docs/senses.md#spoken-words-a-spectrogram-as-an-image)).
- **`longterm_turn=channel`** (#305): every heard moment with that input channel on is a turn
  whose key is the rare units of the message heard since the last value, so an episode holds as
  many records as turns ([turns in a conversation](docs/memory.md#turns-in-a-conversation-opt-in)).
- **`longterm_distinct=share`** (#305): with `longterm_turn`, a question reads the latest record
  holding `longterm_match` of its distinct entries' weight, each entry weighted by its rarity
  among the stream's records.
- **A second head that chains records** (#305): with `longterm_distinct` and `longterm_heads=2`,
  the second head reads from what the first record is about.
- **`message_roles=groups`** (#301): `Brain.compose(message=units, message_roles=groups)` and
  `cadence.Message(..., roles=groups)` give each role of input channels its own blocks of the
  held message ([held message](docs/memory.md#held-message)).
- **`cadence.PopulationCode(ranges, per=16, floor=0.05, clamp=True)`** (#309): a fixed code of
  continuous readings, each spread over tuned inputs that stay at zero away from its value
  ([continuous readings](docs/senses.md#continuous-readings-a-population-code)).
- **`agreed_slots`** in a composed brain's lesson report (#309): the motor slots the judging
  answer gave as shown, over every row
  ([lessons only where the answer differs](docs/teach.md#lessons-only-where-the-answer-differs)).
- **`Learner.finite_lessons` and `Learner.exhausted_lessons`** (#309): counts of finite lessons
  and of those whose free phase used its whole budget; the warning of the 10th, 100th and each
  further power of ten states both
  ([finite free phases](docs/settling.md#finite-free-phases-that-run-out-of-budget)).

### Chambers

- **`benchmarks/coaching`** (#309): the cart-pole chamber (raw readings against the population
  code, teaching on the brain's own path), the scan-and-fire chamber (a majority answer) and an
  ablation of the opt-ins under coaching.
- **`benchmarks/dialogue`** (#305): a closed-domain conversation chamber with a synthetic world,
  bAbI task 1 adapters and matched controls.
- **`benchmarks/spoken`** (#303): spoken phrases of a closed vocabulary, a new speaker and a
  dialect within one life, against a transformer and an LSTM at the brain's parameter count.
- **`benchmarks/letters`** (#304): letters into words, a teacher's forced alignment, the
  read-speech adapter and the 2,000-word list of #307.
- **`benchmarks/grammar`** (#301, #302): protocols 6 (a message keyed by role) and 7 (a word said
  by its parts).
- **`benchmarks/pixels`** (#284): the fields' pool rows read through one tied set of synapses.
- **`benchmarks/settle/oracle.py`** (#290): the ceiling of a first guess for the settle.

### Fixes

- Benchmark tests pass in any order (#306): every chamber's plain-named sibling modules resolve
  to its own files under one pytest process, and tests that changed a shared table copy it.

## 0.97.0 — 2026-11-10

A new brain hears the moment before (`history_decays=(0.0,)`), its store averages the outcomes it
records (`memory_rate=0.2`), the recall memory's slope steps on each stream's own slots, and
`Lives` composes the store, as a lone brain does. These four defaults change;
`history_decays=()` with `memory_rate=1.0` composes the brain of 0.96.1, and a brain of one stream, or
of streams that hold equal slots, lives, saves and describes to the bit as on 0.96.1 on
`backend="cpu"` with those two named. Seven opt-ins join them: a message the brain
holds (`message`), long-term records keyed by it (`longterm`), the actor's own eligibility decay
(`actor_hold`), the policy's entropy in the actor's step (`actor_entropy`), receptive fields a
local rule shapes for a screen (`learned_fields`) with a body's senses beside them (`ports`), and
`Brain.value` for a truncated episode's bootstrap. `Lives` continues lives (`live`), lets a life
leave and another join, and holds the store. Every opt-in off leaves a brain as the brain that
does not name it; a checkpoint that holds one of them is refused by 0.96.1.

### Changed defaults

- **A new brain's store averages its outcomes** (`memory_rate=0.2`, #169). A new brain with the
  memory and the store composes `memory_rate=0.2`: a write moves the stream's record a fifth of
  the way toward the outcome, so a record averages over the outcomes it sees where 1.0 kept the
  last one. On 0.97's default the competing ring's short life read 46.6 of A lost at 1.0 against
  a bar of 42.7 and 39.4 at 0.2, and the twenty-fight league failed 8 of 24 runs at 1.0 against 2
  ([#169, Amendment 3](https://github.com/muellerberndt/cadence/issues/169#issuecomment-6768998711)).
  `memory_rate=1.0` writes as 0.96 did; `memory_choice` keeps its episodic records' 0.5, and a
  brain without the store or the memory takes no rate. Paired on fresh founders against 1.0 (#256):
  the grammar chamber, the sequence battery, every #256 gate and ordered recall read the same to
  the digit (lessons write nothing to the store); seconds per pass the same; the credit and
  key-door chambers move inside two standard errors but the bandit with reversal, 1 of 6 lives
  passing against 4 of 6, where the averaged record of the arm that paid holds through a block
  of 30 after the reversal (memory.md, associative store): in a world whose payoffs change often,
  name `memory_rate=1.0`. The recorded-lives fixtures hold as recorded (their brains compose
  `recall=False`).
- **The recall memory's slope steps on each stream's own slots** (#292). With the grid
  (`recall_grid`, in the default composition) the slope maps step only on a lesson's rows whose
  slots fit the position table. The rule read the width of the slot bank, which the streams of
  one brain share and which is as wide as the longest stream's slots, so a stream that began
  later took no slope step beside a longer one and did alone: its lesson depended on another
  stream's length. Each row's own slots (`held`) decide (`Recall.within_table`); a row past the
  table leans nowhere and counts in the lesson's mean, as the grid term's per-slot mask does.
  The host's path, the device's and the compiled moment's read the same rule (the compiled
  lesson takes the host's path where a row's rule and the bank's part). A brain of one stream,
  and a brain whose streams hold equal slots, keeps every bit (the founders' recorded lives on
  the CPU, and every action and parameter of one, three equal and compiled-moment brains over 70
  moments against 0.96.1). Streams of unequal length change from the first lesson that teaches a
  row within the table while the bank is past it: on two streams, one beginning again at moment
  45 of 70, the slope maps part by 0.24 and the efficacies by 0.025, every action the same.

- **A new brain hears the moment before** (`history_decays=(0.0,)`, #256). A new brain with the
  memory holds a heard history of one block of decay 0: the observation of the moment before,
  as heard, read through its own plastic projection at its own carried gain (`history_0`).
  `history_decays=()` composes the brain of 0.96.1; `recall=False`, the founder word and an
  anatomy of one's own hold a history by name alone. On the grammar chamber's words the brain
  holds 30,992 parameters where it held 28,716. `Lives` holds the history and its gains per
  block. The compiled moment (`learning_compiled_moment`) carries the history: its drive, its
  write after the act and its gains per block run in the one-call act and the taught lesson, a
  brain without a history keeps the option's bits, and on the new default the option agrees with
  the ordinary path to rounding. On fresh founders, paired against the brain
  without it: the grammar chamber reads held-out .995 against .868 and three hops .656 against
  .397, every panel at or above it; ordered recall 12 of 12 in both; every battery gate holds on
  6 of 6 founders but "was A told before B?" on the `recall_when` arm, 4 of 6 against 5 of 6;
  the n-back read less answer rises by .008 and the last cue in sequence by .005 of the mixed,
  each far inside its bar; the credit and key-door chambers move inside two standard errors
  ([the heard history in a new brain](docs/memory.md#the-heard-history-in-a-new-brain)).

  | Reading (#256, fresh founders, paired) | With the history | Without | Paired, ± two standard errors |
  | --- | --- | --- | --- |
  | Grammar chamber: held-out | .995 | .868 | +.126 ± .070 |
  | Grammar chamber: three hops | .656 | .397 | +.259 ± .140 |
  | Grammar chamber: roles | .987 | .904 | +.083 ± .069 |
  | Grammar chamber: clauses | .997 | .964 | +.034 ± .045 |
  | Ordered recall under `live`, 12 founders at .90 | 12 of 12 | 12 of 12 | .000 |
  | "Was A told before B?" (S7), `recall_when=True`, 6 founders at .90 | 4 of 6, against a bar of 5 | 5 of 6 | -.059 ± .147 |

  Held-out and three hops rise beyond two standard errors; S7 on the `recall_when` arm misses its
  bar of 5 of 6 with the history.

- **`Lives` composes the store by default** (#293). `Lives(inputs, actions, lives)` composes each
  life as `Brain.compose(inputs, actions, seed=s, **options)`, the default composition with its
  store of place records, as a lone brain composes; `episodic=False` composes without it, as
  `Lives` did in 0.96. `Lives(..., episodic=False, history_decays=())` keeps every action and
  every life's parameters of 0.96.1 on the CPU and the torch CPU device in float64 and float32.

- **A skill met by a competing one, on the new default** (#169,
  [Amendment 3](https://github.com/muellerberndt/cadence/issues/169#issuecomment-6768998711)).
  With the heard history and `memory_rate=1.0`, retention failed two gates: in the short competing
  ring the first skill's loss read 46.6 points against a bar of 42.7, and in the twenty-fight
  league 8 of 24 runs failed against a bar of 2 (runs one winner takes, 7 of them in one league),
  where the 0.96.0 default read 0 of 24. The default `memory_rate=0.2` holds both: 39.4 points and
  2 of 24. Recovery is a measured negative for every arm. The actor's rate keeps its default.

### New

#### Continuing lives on a device (#293)

- **`Lives.live`: each life's `Brain.live` with its own arousal, outcomes and endings.** Compose
  the lives with `arousal=True` and `lives.live(observations, reward=..., done=...)` runs one
  moment of every continuing life in one batched moment: the lives whose action was sampled learn
  from their outcomes while the others hold their state and parameters, the calm lives'
  eligibility fades and their ended streams start from rest, one qualified settle answers every
  life, each life's `Arousal` takes its own outcome on the host, and each life answers greedily
  or samples at the temperatures its own arousal sets, drawing from its own generator.
  `arousals` and `last_arousal` read each life's mood, and `life(i)` carries it. A life with no
  preceding action takes a zero reward and no end in `live`, `step` and `learn`, as its lone
  brain takes none. On the CPU each life is its lone brain to the bit; in float64 on a device
  every action, mood and heated slot is its lone brain's and every learned part and arousal
  reading agrees within 1e-9 over 30 moments of sparse rewards, endings and teaching. Lives that
  never call `live` keep every bit of 0.96.0.
- **The store in `Lives`.** `Lives` holds each life's place records on the
  device with a lives axis: an outcome writes the records of the lives it teaches, the outcome
  that wakes a calm life writes that life's records and settles it again under them, and with
  `memory_fade` and `maturity` each life fades its own records, scales its own actor steps by its
  own use and gates its own writes by its own surprise. `parameters(i)` and `life(i)` carry each
  life's records, their noise scales, depths and use. In float64 every action and mood is the
  lone brain's and every learned part and record agrees within 1e-9 over 30 moments; on the CPU
  each life is its lone brain to the bit. `Lives` composes the store by default (below).
- **A life leaves and joins.** `life(i, stream=True)` carries life `i`'s stream as it stands
  into the lone brain (its activity, its pending sampled action and that action's eligibility,
  its traces, working trace, recall slots, store residual and the record of its last `live`),
  on the CPU or the `backend`, `device` and `precision` named, and the lone brain continues the
  life where it left. `join(i, brain)` puts a lone brain of the lives' anatomy and composition in
  life `i`'s place with its stream afresh, so a finished episode makes room for another life
  without holding the batch. `life(i)` without `stream` starts the stream afresh, as in 0.96.0.

#### The value of a state (#298)

- **`Brain.value(observations)`: the value a truncated episode bootstraps from** (#298). The
  critic's value of each row, `(batch,)`: the drive the brain reads for the row, with the working
  trace, the efference copy, the heard history, the recall memory's slots and the place records
  as they stand, settled from the stream's activity (from rest before the brain has acted). It is
  what `learn` settles for the next state of a stream that goes on, so a row cut by a time limit
  passes `done=True` with `bootstrap=brain.value(last)`; without the store such a row takes the
  dopamine of a row that goes on, to the bit, and with the store the two part by the outcome's own
  record, which `learn` writes before it reads the next state. The read writes nothing: on
  `backend="cpu"` every array of the brain is the same after it, and a brain that reads lives on
  to the bit as one that does not. `Lives.values(observations)` reads each life's.

#### Credit for a reward many moments after the decision (#283)

- **`actor_hold`: the actor's eligibility decay apart from the critic's.** `Brain.compose(...,
  actor_hold=1.0)` (or `ActorCriticConfig(hold=1.0)`) keeps each sampled act's contrast in the
  actor's synapse, bias, recall and hearing traces until the episode ends, while the critic's
  eligibility keeps `actor_gamma * actor_lam`. A hold below 1 sets the actor's decay per moment.
  It applies in `learn` on the host, on a torch device, on MLX and in `Lives`, and in `fade`. A
  brain that names no hold keeps `gamma * lam` for every trace and lives, saves and describes to
  the bit as on 0.96.1 on `backend="cpu"`; a configuration without the hold saves without the
  field ([a reward many moments after the
  decision](docs/reward.md#a-reward-many-moments-after-the-decision)). On six fresh founders, with
  a tenth of the actor step and `memory_choice=True`, it learns a T-maze whose reward comes 50
  moments after the choice on 6 of 6 (2 of 6 without the hold, 0 of 6 at the default), at the
  level of an LSTM actor-critic; on the key-door at 50 it reads 3 of 6 against 5 of 6 without the
  hold, the readout latched (#296), and at 200 moments 3 and 0 of 6. The default is unchanged.

#### Learned receptive fields for a screen (#284, #297)

- **`learned_fields=features` (opt-in, an image brain).** `Brain.compose(inputs=(84, 84), actions,
  learned_fields=64)` reads the image through 64 receptive fields of 8 by 8 pixels in place of the
  visual cortex: a retina that centres each patch and divides it by its contrast, one set of
  synapses per field read at every second pixel, rectified above a homeostatic threshold, and the
  responses pooled over 3 by 3 regions that overlap by half. The first module reads the pools and
  every memory keys on them (576 coordinates at 64 fields), so a new brain holds the recall memory
  at 84 by 84. The fields learn from the frames of every accepted moment and lesson by the learned
  code's local rule, Hebbian on the residual of the patch they reconstruct; nothing leads back into
  them and no gradient reaches them. `brain.fields` holds them (`LearnedFields`), a checkpoint saves
  them, `describe()` and `parameters()` report them; `grow` and `graft` refuse a brain that holds
  them ([learned receptive fields](docs/senses.md#learned-receptive-fields)). A brain that names
  no `learned_fields` is unchanged to the bit.
- **Readings** (six fresh founders, 65,536 frames of lessons, grey 84 by 84): fresh frames at
  .974 (every founder at .90 or above), an untrained band of places at .948, an untrained scale at
  .975 and an untrained distractor at .947, against 1.000 and .145 on the band for a convolutional
  actor-critic of the same size; under reward alone .673 against .372 on a blank screen. Each
  founder reads all three untrained panels within .05 of its fresh frames on 4 of 6; on the other
  two the pools hold the panels and the readout trails. One stream acts in 2.6 ms per grey frame
  and 9.0 ms per colour frame on one Xeon core, against 163 ms and 220 ms through the visual
  cortex; colour frames read .767.
- **Colour through opponent channels.** A three-channel image is read as luminance, red against
  green and blue against yellow; nine fields in ten read luminance, where a shape is the same
  whatever its hue. Colour 84 by 84 frames read .953 on fresh frames (four development founders)
  against .762 with every field over the three channels jointly; the untrained band reads .907.
- **`ports=k`: a body's own senses beside the screen (opt-in, with `learned_fields`, #297).** An
  observation row is the flattened image followed by `k` values; `k` sensory neurons read them
  beside the pools, and every memory keys on both. `describe()` reports them and a checkpoint
  keeps them ([ports beside the screen](docs/senses.md#ports-beside-the-screen)).

#### A message a brain holds: `Brain.compose(message=units)` (#286)

Opt-in and off by default. Without `message` a brain holds no message region, and every founder,
the default composition and `backend="cpu"` arithmetic are as on 0.95.2.

- **`message=units`** ([held message](docs/memory.md#held-message)). A region of
  `message_phases * units` neurons the brain writes itself. After each moment it hears, the moment's
  fixed sparse code latches onto block `t mod message_phases` and stays on until the episode ends
  (`reset()` or `done`). The association reads the region through plastic synapses at
  `message_amplitude`, and with `carried_gain` one learned gain per block sets how loudly.
  `brain.message` is a `HeldMessage`, `cadence.Message` on the brain's own neurons (`state`,
  `key()`, `told`), and a member of `brain.memories`. `describe`, `retune`
  (`message_amplitude`), checkpoints and `imagine` carry it. `grow` and `graft` refuse a brain with
  a message, and a message needs a vector brain without a learned code.

- **`message_gains`, `message_boundary`, `message_boundary_holds`** (the third declared run,
  [`benchmarks/grammar/MEANING_PROTOCOL_3.md`](benchmarks/grammar/MEANING_PROTOCOL_3.md)). Under
  `carried_gain`, `message_gains` learned gains per block in place of one. `message_boundary` names
  an input channel: a heard moment with it on empties the message and begins it again, or with
  `message_boundary_holds` the message keeps what it heard before and nothing after joins.
  `cadence.Message` takes `boundary` and `boundary_holds` the same way.
  The third run (the read free): the phase count at 24 or 32 is not the cause of the quiet
  message's three-hop cost (-.09, -.12 on six founders). Held from `GO`, the message costs three
  hops nothing at the mean on six founders and costs paths -.20 and three hops -.24 on twelve; held
  from `GO` beside `history_decays=(0.0,)` it costs no panel on twelve founders and lifts clauses
  (+.03). At 256 association neurons with the read barred, the instrument's message reads roles
  .05 over the brain's own on the same founders, inside two standard errors; `message_gains=8`
  reads roles .95 on fresh founders and held-out combinations .89 (4 of 6 at .90).
  The fourth run ([`benchmarks/grammar/MEANING_PROTOCOL_4.md`](benchmarks/grammar/MEANING_PROTOCOL_4.md),
  twelve fresh founders, the read barred): with `message_read=True` and `history_decays=(0.0,)`
  beside `message=280` at 256 association neurons the brain says clauses, held-out combinations
  and role twins from its held state at .90 on 12, 11 and 11 founders (596,412 parameters), where
  the message alone reaches 12, 9 and 9; on twelve more founders 11, 7 and 9, so over 24 the
  held-out bar of 20 is not met (18). Longer clauses, deeper brackets and three hops read .00 from
  the state, and .00 to .02 for the LSTM and the transformer at the same parameter count; a block
  for every moment of the longest clause (`message_phases` 7, 8 or 16) leaves longer clauses at
  .00 to .02.
- **`message_read=True`**. A port of `units` neurons holds one block of the message, chosen by the
  present moment through learned maps (a content address over the block's units and a term per
  block). The maps start at zero and learn by the recall memory's address lesson on the message's
  own blocks: each block is scored against the association's movement between the nudged phases,
  read back through the port's synapses. `message_amplitude=0.0` beside it builds no synapses from
  the region itself.

- **The declared run**, [`benchmarks/grammar/MEANING_PROTOCOL_2.md`](benchmarks/grammar/MEANING_PROTOCOL_2.md),
  six fresh founders, the recall read barred from the meaning at `GO`. With `message=280` and 256
  association neurons the brain says trained clauses from its own message at .99, held-out
  combinations at .95 and role twins at .90 (6, 5 and 4 of 6 at .90: H1 holds on two panels and
  fails on roles), reads every clause component from the message at 1.00, says both orders of one
  meaning right at .98, paraphrases into a second grammar at .97 (.92 held out) and says clauses of
  four distractors at .90. Its previous-word control (`history_decays=(0.0,)` in the message's
  place) says .07, .00, .00. At 64 association neurons the message alone reads .79, .47, .30 and
  the read of one block beside it .89, .72, .74: under the bar. Beside a free read the quiet
  message keeps paths and costs three hops (-.22); the wide message costs paths (-.26). The LSTM at
  the brain's parameter count reads 1.00 on every trained panel offline.

#### A held message as a class: `cadence.Message` (#286)

A class no brain reads by itself: it changes no brain.

- **`cadence.Message`** ([a held message](docs/memory.md#a-held-message)). A stream's episode held
  as one state, with no neurons and no learned parameter. Each heard moment is given a fixed sparse
  code, and the `t`-th moment joins the union of block `t mod phases`. `state` keeps the order
  heard; `key()`, the union over the blocks, is the same for one set of moments in any order.

- **Where the meaning lives** ([`meaning.py where`](benchmarks/grammar/meaning.py), development
  founders). In the default composition a sentence is looked up: with the recall slots emptied at
  `GO` every panel of the grammar chamber reads .00, and with the working trace emptied nothing
  moves. The carried state holds a partial copy of the meaning that the readout does not use.

- **Said from a held state** ([the declared run](benchmarks/grammar/README.md#the-declared-run), six
  fresh founders, the message written by the instrument onto neurons the association reads, the
  read barred from the meaning in every lesson). The brain says clauses from the message alone at
  .98, as the matched LSTM does online, and paraphrases a heard sentence into the second grammar at
  .94. On combinations no lesson showed it reads under the bar at 128 association neurons
  (held-out clauses .84, role twins .81) and at the bar at 256, an arm declared as an amendment
  before its lives (clauses .99, held-out clauses .95, role twins .95; 5 or 6 founders of 6 at
  .90). Clauses of four distractors are said at .94 with 256 association neurons, where 128 read
  .66.

- **What a message needs.** Moments bound by a sparse code, order held as presence in separate
  blocks, and width. The working trace at any decay, the heard history of raw moments, sums with
  primacy or recency weights, and a message restated in motor slots each fail one of them.

- **Beside a free read.** The episode-long message lifts brackets (+.12) and the context twins and
  costs three hops (.36 to .12). The moment before alone, the library's `history_decays=(0.0,)`,
  lifts roles (.91 to .98), paths (.94 to .99) and three hops (.36 to .74) on six fresh founders,
  paired, beyond two standard errors. Neither is a default.

- **The key as a record's address.** At 280 units a cue of the same meaning in the other order finds
  its record among 256 on .98 of cues and among 1,024 on .91, and a role-swapped cue finds the twin's.
  The key holds no order of moments of one kind, and a sentence heard as words and the same meaning
  heard as concepts give unrelated keys.

#### Long-term records a brain keeps itself (opt-in, #288)

`Brain.compose(longterm=records)` gives each stream records of its own, keyed by the brain's held
message (`message=units`) from a turn (`message_boundary=channel`, `message_boundary_holds=True`).
Off by default: without it a brain is the brain without it, to the bit on `backend="cpu"`.

- **Key and write.** The key is the held message after the turn. The moment after the turn is
  written only where it says more than the record the key finds: the same value writes nothing,
  another revises that record, and a silent moment (a question) writes nothing.

- **Read.** Before every settle after a turn, `longterm_heads` heads read in turn, each holding
  its record on a region the association reads through plastic synapses, as it reads the recall
  memory's ports. The read scores each record by the cosine of its key with the message's key,
  so a key that another record's key contains reads its own record.

- **What stays.** Each stream's records are a `cd.LongTermMemory`: usage-protected under its
  budget, revised, and kept across `reset()` and episodes.

- Settings: `longterm_heads` (1), `longterm_amplitude` (3.0), `longterm_match` (.75),
  `longterm_revise` (.999).

- **Saved.** A brain with records saves every stream's records and turns in checkpoint format
  `cadence-brain/4`, which 0.96.0 and earlier refuse; a loaded brain continues its streams as the
  one saved. Every other brain keeps its format to the byte.

- **Many streams.** The streams' records are read, written and faded as one bank: one product
  over every stream's records, the read before the settle reused as the write's recognition
  (`LongTermMemory.write(..., recognised=slot)`), and one fading pass. The bank holds to the stores
  read one by one to the bit.

Declared on six fresh founders, a brain taught to answer from its records hears facts it was
never asked about. At 1,000 records a stream it finds and answers .9975 to 1.000 of them, at
10,000 records .980 to 1.000. Revised facts are current at .990 to 1.000 and never stale, and the
same brain without the option answers at chance (.09 to .16). Read by cosine on six more founders,
found at 10,000 records is 1.000 over the facts whose key is their own and .9875 to 1.000 over all
facts: two facts whose key is one key (2.43% of the facts at one founder) are one record. On one core a moment with 10,000
records a stream costs 1.3 times a moment without them over one stream and 1.7 times over 16
streams (a declared amendment, 1.66 to 1.69 against a bar of 2); on a GPU, with the stores read
one by one, 1.7 times over one stream.

#### A long-term memory of records: `cd.LongTermMemory` (opt-in, #288)

`cd.LongTermMemory(width, value_width, capacity)` is a standalone store beside the brain: records
(a key with a value) under a fixed budget of slots. No `Brain.compose` setting builds it, and a
brain without it is the same brain to the bit.

- **Read in two stages.** A sparse address (a fixed random projection onto 8,192 units, 64
  winners, as pattern separation codes a key) picks 256 candidate slots without reading a key,
  and the recall memory's read picks among them. A memory of at most `direct` slots (twice the
  address's units) scores every slot.

- **Several heads.** Each head reads its first choice among the slots the heads before it left,
  whole, so k heads return k different records.

- **Kept by use.** Every slot keeps a depth, a recent use and how far it has faded: three leaky
  sums, no count and no time stamp. A full memory replaces the most faded slot.

- **Revised.** A write that recognises a held key (cosine .9) rewrites that record.

- **Scored by cosine** with `cosine=True`: the fine read and a write's recognition score each
  candidate by the cosine of its key with the cue (the founder scores by the dot product).

Declared on six fresh founders with records of random unit codes 64 coordinates wide
([memory-scale benchmark](benchmarks/memory_scale/README.md)):

- Among 10,000 records the right one is found from its exact key at 1.000 and from a key with
  noise of its own energy at .999 to 1.000; among 100,000 at 1.000 and .986 to .997, where an
  exact nearest-neighbour store reads .991 to .999. Four heads return four different records of
  one family at 1.00.

- A brain taught to answer from the read answers on records no lesson taught at .99 to 1.00 among
  10,000 and 100,000 records, as among 128.

- Under 1,000 slots, 4,000 facts used by a Zipf law and 100,000 moments, the 10% most-used facts
  are retrievable at .963 to .980 against .782 to .823 for least-recently-used over exact keys,
  and the facts that changed are current at .971 to .981 with none stale.

- A read among 10,000 records through the address takes .25 ms on one core, 2.1 times the rest
  of the default composition's moment: the declared cost gate fails there, since the address
  costs more than scoring 10,000 keys. The founder therefore scores every slot up to 16,384
  records: .09 ms among 10,000, read after the declaration. Among 100,000 a read through the
  address takes .57 ms where the recall memory's own read takes 13.6, and on a device one product
  reads them in .4 ms.

A second declaration ([protocol](benchmarks/memory_scale/PROTOCOL_2.md)) reads two more things:

- **The cost gate for the store as it stands.** On three fresh founders a read with its value
  among 10,000 records takes .084 to .093 ms on one core against a budget of .122, on 3 of 3; the
  least a read that scores every slot does in NumPy is .071 to .079 ms. With the device's reading
  the issue's cost gate is met.

- **Keys made by `cadence.Message`** (#286) in place of the random stand-ins, on six fresh
  founders: facts of an agent and a verb that say which patient, the key taken before the patient
  is heard. An exact key finds its fact at .996 to 1.000 among 10,000 and .961 to .991 among
  100,000. A noisy key reads .852 to .908 and .567 to .706, and an exact store over the same keys
  reads the same: a key of ten active units in two groups shared with its siblings ties where it
  loses units, so the key binds before the memory does. Retention reads as on the stand-ins (.968
  to .978, changed facts current at .970 to .984). The same act and agent heard as words find
  their fact on .00 to .01 of the cues.

The same run reads the recall memory at scale ([a large memory](docs/memory.md#a-large-memory)):
at a new brain's temperature the right record holds .27 of the read's mass at 128 records and .005
at 10,000, and a taught default composition's position maps outvote the key past the distances its
lessons reached. `recall_position=0, recall_temperature=0.05` reads and answers at 1.00 to 100,000
records, with every slot scored. No setting or default changes.

#### Exploration over long reward lives (#296)

- **`actor_entropy`: the policy's entropy in the actor's step.** `Brain.compose(...,
  actor_entropy=k)` (or `ActorCriticConfig(entropy=k)`) adds to each outcome's step the sampled
  act's own contrast at the weight `k` times its surprise, `-log p(action) - H` under the policy
  that sampled it: the sampled gradient of the policy's entropy, local to the act and outside
  the trace. `last_learning["policy_entropy"]` reads each act's entropy. It learns on
  `backend="cpu"` and `"mlx"`; a torch device refuses the outcome and `Lives` refuses the
  setting by name. A brain that names no entropy keeps every bit on `backend="cpu"` and saves
  its configuration without the field ([exploration over long reward
  lives](docs/reward.md#exploration-over-long-reward-lives-opt-in)).

### Fixes

- **The learned code's step reads the same answer in a browser** (#295). `LearnedCode.stability`
  asks Cholesky factorizations whether the present halvings hold. NumPy signals a failed
  factorization through the floating-point status, which WebAssembly does not keep: in Pyodide
  `np.linalg.cholesky` of a matrix that is not positive definite returns NaN and raises nothing,
  so every test held and a one-input world kept a step its dictionary does not allow (0 halvings
  where 3 are needed). A factorization holds where it raises nothing and its factor is finite,
  the answer every platform gives alike; the four tests that read another answer in Pyodide
  0.29.3 pass there. Where `cholesky` raises, as on every 64-bit platform, nothing changes: every
  bit is as on 0.96.1.
- **Where the new options meet** (found when the lanes joined). `Lives.live` in a moment where no
  life learns faded the actor's eligibility by `gamma * lam` under `actor_hold`; it fades the
  actor's traces by the hold and the critic's by `gamma * lam`, as a lone brain's `fade`. One
  life with `actor_hold=0.97` on a torch device had left its lone brain by .025 in its
  parameters over 40 moments. `Brain.value` read the long-term records as a live read does,
  deepening the record it held, and counted the learned fields' work; it credits no record and
  counts nothing, so the checkpoint after a read is the one before it. A life that left with its
  pending action (`life(i, stream=True)`) saved a file `Brain.load` refused; it carries the drive
  its free state was settled for. `Lives` refuses a held message and long-term records by name.
  The compiled moment (`learning_compiled_moment`) left a held message's and long-term records'
  drive out of the moment (the drive 3.1 from the general path's) and raised on learned fields
  without the store; such a brain reads its drive on the general path, within 1e-12 of the brain
  without the option.

- **The torch backend on CUDA under Triton 3.1 and 3.3 (#299).** The small-group kernel's
  `tl.dot` does not compile there with a tile below 16 on a side, which a float64 tile of 8 rows
  and a batch below 16 always have, so the first settle raised `CompilationError`. Each tile
  shape takes `tl.dot` where the installed Triton compiles it, and otherwise sums the same
  products over the presynaptic axis, chosen once per shape at its first launch. Under Triton
  3.6 every shape keeps its dot and every bit; under 3.1 and 3.3 the settled states match the
  cuBLAS path (`CADENCE_TRITON=0`) within float32 rounding and within 1e-13 in float64.
- **A settled state's kept products hold the same bits on every run.** The room in which a
  compiled settle keeps its products for the next one (`fused._kept_room`) was allocated
  uninitialized; its parts no kernel writes (a range that feeds no block, a stale range's
  products, a masked state's publication) are never read, so no act, lesson or saved array
  depended on them, but they held what the allocator left, NaN at times, and two brains of one
  history differed there. The room starts at zero. A test that compared a brain's arrays before
  and after `Brain.value` saw such a NaN as a change (NaN equals nothing), on one macOS runner
  in one run: `value` writes nothing, every byte is as before. The tests compare bytes, and two
  lives of one history keep the same bits in every array, the kept products among them.
- **A lesson that teaches some rows keeps a torch state on the device.** A surprise lesson whose
  answer differs from the labels on some rows only restricted its warm state to those rows by
  copying the whole state to the host, `(rows, neurons)` per array. The rows are taken on the
  device, and every saved array and action is identical to the host path's.

### Instruments

- **A VizDoom adapter in `benchmarks/doom/`**, the Body stage of the Doom player (#118). A player
  receives the rendered RGB screen (84 by 84 or 160 by 120), the HUD's health, armour and
  ammunition as declared own-body ports and the episode's previous action as a history port,
  and nothing else. It answers with one button of the scenario's declared set per moment, held
  for declared game tics. A scorer per scenario (basic, defend the center, health gathering, on
  the Freedoom IWAD the VizDoom wheel carries) keeps completion, death, deadline and truncation
  distinct and credits each outcome once; custody records the tic, the events and the outcome
  of every moment; a checkpoint of game, body and brain resumes the same life at the same tic,
  the brain to the bit on `backend="cpu"`. VizDoom is optional, and its tests skip without it.
  The brain (`Brain.compose`), its blind and frozen controls, a random player and a small
  convolutional actor-critic run every scenario through it
  ([the Doom body](benchmarks/doom/README.md)). The library is unchanged.

- **The delayed-credit chamber** ([`benchmarks/delayed_credit`](benchmarks/delayed_credit/README.md)):
  a delayed T-maze and a delayed key-door paid only at the end of an episode, delays of 10, 50
  and 200 moments with distractors, an exact solver, yoked, shuffled, frozen and ablated controls,
  and an LSTM actor-critic with generalized advantage estimation and the online TD(lambda)
  actor-critic sized to the brain's parameter count, with tests.

- **The pixel chamber** (`benchmarks/pixels/`): a screen-dependent policy on rendered 84 by 84
  frames in grey and colour and on 160 by 120 frames (approach food, avoid a hazard, aim at a
  target), with held-out places, scales, lighting and distractors, the blank-screen, shuffled,
  frozen, uniform and fields-at-their-draw controls, and a two-convolution actor-critic at the
  brain's parameter count trained online on the same frames.

- **The long reward life chamber** ([`benchmarks/actor_latch`](benchmarks/actor_latch/README.md)):
  three worlds with a Doom player's observation (a bandit, aiming in the shape of Doom basic,
  survival in the shape of health gathering), its frozen and random controls, and the reading of
  the actor's hold by quarters of a life. `benchmarks/doom/doom_life.py` takes `--set` for the
  brain players' settings.

- **What lives on MLX would cost.** `benchmarks/device/lives_mlx_cost.py` times one sweep's
  transport with each life's own blocks on the Apple GPU against one brain's shared blocks: 1.26,
  1.45 and 3.34 times at 12, 128 and 1,024 lives on an M4. With one brain's streams on MLX as the
  bound, a lives axis there would read at most 0.34 ms per life and moment at 128 lives and 0.57
  ms at 1,024 (docs/backends.md, many lives at once). The library has no lives axis on MLX.

### Measured negatives

- **#286, sentences said from a held state: held-out combinations miss their bar.** With
  `message_read=True` and `history_decays=(0.0,)` beside `message=280` at 256 association
  neurons, the first twelve fresh founders read clauses, held-out combinations and role twins at
  .90 on 12, 11 and 11; twelve more read 11, 7 and 9. Pooled over 24 founders the brain says
  clauses on 23, held-out combinations on 18 and role twins on 20, so gate H1c (20 of 24 on
  each) fails on held-out combinations
  ([the replication](https://github.com/muellerberndt/cadence/issues/286#issuecomment-6768846248)).
  Longer clauses, deeper brackets and three hops read .00 from the state, and .00 to .02 for the
  LSTM and the transformer at the same parameter count.
- **#283, a reward many moments after the decision: the key-door at 50 moments reads 3 of 6.**
  At a tenth of the actor step, `actor_hold=1.0` learns the T-maze at 50 on 6 of 6 fresh
  founders and the key-door at 50 on 3 of 6 (5 of 6 without the hold, the readout latched); at
  200 moments 3 and 0 of 6. With `memory_choice=True` beside the hold the key-door at 50 reads 5
  of 6 ([the delayed-credit chamber](benchmarks/delayed_credit/README.md)). An ungated reading
  ([#296](https://github.com/muellerberndt/cadence/issues/296), the delayed-credit chamber,
  founders 2830 to 2835 at .90, without and with `actor_entropy=0.1`): with `actor_eta=0.1` and
  `actor_hold=1.0` the key-door at 50 reads 3 and 5 of 6, the T-maze at 200 reads 5 and 6 of 6,
  and the key-door at 200 reads 2 and 5 of 6; with the hold at the full step the T-maze at 200
  reads 0 and 5 of 6, and the key-door fails with and without the entropy.
- **#296, exploration over long reward lives: no setting meets the Doom gate.**
  `actor_entropy=0.1` ends the latch: no button exceeds .40 of a life's last quarter in any of
  its 30 lives, and aim reads +0.918 on 6 of 6 fresh founders. In the Doom body it passes 2 of
  6 founders in basic and none in defend the center or health gathering, and survival is its
  cost (1 of 6, +0.644 against a newborn's +0.810). The located term is the motor operating
  point: the entropy settles every motor output below rest, where a faint preference cannot
  lift a button; a longer critic horizon (`actor_gamma=0.99`) leaves it as it is. No arm passes
  ([the declared run](benchmarks/actor_latch/README.md#the-declared-run)).
- **#284, perception from pixels: colour invariance is not gated.** Grey 84 by 84 frames read
  .974 on fresh frames and .947 to .975 on three untrained panels on six fresh founders; each
  founder reads all three within .05 of its fresh frames on 4 of 6. Colour frames read .953 on
  four development founders, and no declared gate reads invariance to hue.
- **#288, memory at scale: two facts that share one key are one record.** At 10,000 records a
  stream, found over every unrevised fact reads .9875 to 1.000 (5 of 6 at .99; gate B6 fails at
  one founder where 2.43% of the facts share their key); over the facts whose key is their own,
  1.000 on 6 of 6 (B7 passes).
- **The bandit with reversal under `memory_rate=0.2`: 1 of 6 lives pass against 4 of 6 at 1.0**
  (#169, #256); the averaged record of the arm that paid holds through a block of 30 after the
  reversal. Recovery in the competing ring is a measured negative for every arm (#169).

### Documentation

- **The default composition and a skill met by a competing one** (#169). The memory guide records
  #169's reading of the actor's-step gates on the default composition of 0.96.0: in the competing
  ring's short life the default loses more of the first skill than a brain without the recall
  memory (51.9 against 38.6 points on twelve fresh founders), and the store's write carries the
  difference; `memory_rate=0.2` brings it to 41.6, with `maturity=True` to 37.8, and leaves the
  chambers from birth and the robot arena's gates where the default reads them. No setting or
  default changes.

## 0.96.1 — 2026-11-10

A fix for 32-bit Python, the Python a browser runs (Pyodide). A 32-bit NumPy's index type is
int32, and its `np.repeat` and `reduceat` refuse int64 counts and offsets. The learner kept its
motor slots' sizes and offsets as int64, so on 32-bit Python any brain with motor slots (equal or
unequal, `slots=(2, 3)`), `grow(slots=...)`, a lesson with `targets=` and a policy with one
temperature per slot raised `TypeError: Cannot cast array data from dtype('int64') to
dtype('int32')`. The slots' sizes and offsets, and the segment starts of the CPU path's summed
input, are kept in the platform's index type (`np.intp`). On a 64-bit platform that type is
int64, so every bit is as on 0.96.0.

## 0.96.0 — 2026-11-10

Many brains of the default composition run as one on a device, and one brain runs its moment in
compiled calls on the CPU: `cadence.Lives(inputs, actions, lives, backend="torch")` steps up to
a thousand independent lives of one anatomy in one call per moment, each life the lone brain of
its seed, and `learning_compiled_moment=True` does the work around a moment's solve in compiled
calls, a grammar-chamber pass in .54 to .56 of the default's seconds on one core. Both leave every
bit of a brain that names neither as it was. Three opt-ins join them: `memory_fade`, under which
place records nobody uses fade and deeply used ones keep, and `recall_cortex_norm`,
`recall_gain_after` and `recall_gain_next`, which let the composed attention circuit form on more
founders. The copy pathway (`recall_copy`) leaves, a `synapse_rate` of zero holds a synapse's
optimizer moments and its neuron's bias, and a brain whose store fades saves its own checkpoint
format. Every default is unchanged: a brain that names none of the new options lives, saves and
describes to the bit as on 0.95.2 on `backend="cpu"`, with `recall_copy` the one refusal.

### New

#### Many lives at once, and the lifecycle on the device (#274)

- **`Lives`: independent lives over one topology, stepped together.** `cd.Lives(inputs, actions,
  lives, backend="torch")` holds many brains of the default composition without the store
  (`episodic=False`), each with its own synapses, optimizer, critic, eligibility, recall memory
  (slots, address and position maps), carried gains, working trace and random generator, one stream
  each, and steps them in one call per moment. Every settle reads each stream through its own
  life's synapses and stops each life where its own stream meets the tolerance; a lesson judges
  each life by its own answer and updates the lives it taught; a reward moves each life on its own
  dopamine. Each life follows the lone brain of its seed on that device: in float64 every action
  the same and every learned part within 1e-9 over 30 moments of rewards, endings and teaching; on
  the CPU it is that lone brain to the bit. On an NVIDIA L4, 128 lives step at 0.118 ms a life and
  moment, the speed of one brain's streams, against 0.876 ms in a four-process pool, and 1,024
  lives at 0.049 ms. It holds one or several regions, observers and motor slots at any teaching
  point, with or without the recall memory, on CUDA and the torch CPU device, and refuses by name
  the store, arousal, a maturity, a code, an apical compartment, rehearsal, the efference copy, the
  heard history and the recall options that keep the memory on the host, the cortex at unit norm
  among them; MLX has no lives axis ([many lives at
  once](docs/backends.md#many-lives-at-once)). A rate of zero holds each life's synapses, biases
  and optimizer moments as it holds a lone brain's.
- **An outcome reads its checks once on torch.** Where the critic rests on the device, the
  dopamine is made there from the critic's value, and an outcome's checks (the dopamine's, the
  critic's, the recall memory's reward step and the update's), the critic's bias step, the
  report's steps and the dopamine of every row come to the host in one read before anything is
  installed. A torch update and a lesson read their checks and report in one read where they read
  five. On MLX the critic's and the recall memory's checks are made in the update's launch. The
  CPU path keeps its bits; a device life agrees with the host's as before.
- **A CUDA card the default backend's table has not read is calibrated** once per machine, in
  place of the margin of twice the work: the chamber's brain is timed on the card and on one core,
  and the crossing scales the table's ([the default
  backend](docs/backends.md#the-default-backend)); `CADENCE_CALIBRATE=0` keeps the margin. On an
  NVIDIA L4 it reads 50 to 60 rows of the chamber's brain against the A10G's 44.
- **The ledger of a moment.** [`benchmarks/device/ledger.py`](benchmarks/device/ledger.py) lists
  every crossing between the host and the device that a `live`, `step`, `lesson` or `act` makes,
  by the place in the library that made it, with its bytes and time, beside the seconds of the
  moment, on torch and MLX. A moment's bytes are small; its synchronizations are what it pays for.
- The options that keep a moment's memories on the host each state what they add that the device
  form does not hold, `memory_fade` and `recall_cortex_norm` among them ([memories on the
  device](docs/backends.md#memories-on-the-device)).

#### The compiled moment (opt-in, #282)

`learning_compiled_moment=True` (`Brain.compose`, `retune`, `LearnerConfig(compiled_moment=True)`)
does the work around the solve kernels of a moment in compiled calls on the CPU. A greedy act is
one compiled call: the moment's drive (the sensory part, the working trace, the recall memory's
read, with the address and every position term through one product of the maps, which stay packed
until a lesson replaces one, and the place store's read), the free solve, the choice, the working
trace's step and the memory's write; the act that begins its streams is one too. A lesson by
surprise judges in one call, and where it teaches a row the lesson is one call more, from its
phases to its written efficacies: the free phase, the two nudged phases and their checks, the
contrast, the optimizer's step with the tied and bounded efficacies, the address lesson and the
carried gains' contrast and step, and Python stores what the call returns (a learner with frozen
or tied synapses or per-synapse rates, and an update that is refused, take the passes in turn).
With the option the lesson's contrast is one compiled pass in each of its paths, its block
products through the BLAS that Numba calls. An act after a lesson that taught nothing takes the
lesson's read as its own, as it takes the lesson's solve, and writes its moment in the choice's
pass. The equations are the ordinary path's and the sums run in another order, so the bits are
other ones. Off by default; off, every bit is as before: the identity, options and slots batteries
read no difference and the founder parity reads 0 of 24 lives differing against the tree before
the change on one machine, and twelve chamber lives of the default brain read the same panels and
work. Several read heads, a window, the command or when in the slots, the gains, a learned code or
a separator take the ordinary path. Beside each other option of this release
(`recall_cortex_norm`, `memory_fade`, `maturity`, the delayed gains, a synapse rate of zero) a life
of lessons, acts, sampled moments and outcomes keeps its bits through every compiled path and
agrees with the ordinary path to rounding (`tests/test_compiled_moment.py`); with
`recall_cortex_norm` the compiled act carries the cortex at unit norm as `Recall.witness` does.

On one Xeon 8488C core with nothing else running, in one session, a grammar-chamber pass of the
default brain takes .54 to .56 of its seconds with the option (67.6 to 70.8 s against 36.7 to
39.5 s over three founders) and 2.3 to 2.4 times the matched transformer's 15.9 to 16.2 s, where
the default takes 4.2 to 4.4: under the issue's 2.5 times in that session. The kernels take
26.9 s of the compiled pass, 1.7 times the transformer: the act's and the judgment's call 20.8 s
and the taught lesson's 6.0 s. What stands around them takes 7.2 s, 4.2 s around an act's one
call (about 85 microseconds an act) and 3.0 s around a lesson's (about 130 a lesson), and the
chamber's own loop 1.7 s (`benchmarks/grammar/cost.py`, which reads both the seconds and the
parts and warms every kernel before its clock;
[the compiled moment](docs/tune.md#the-compiled-moment-opt-in)). Seconds differ by a tenth and
more between runs and machines, the transformer's more than the brain's: across sessions the
default read 4.2 to 5.3 times the transformer, and at 4.8 the option's .54 to .56 is 2.6 to 2.7
times. Within a run the option's share of the default's seconds is steady, and it is the number to
read. The kernels and the chamber's loop are .42 of the default's seconds and the Python around
the calls .11, so 2.5 times at a default of 4.8 takes fewer calls per moment (#291).

The panels, the option less the default:

- The grammar chamber over 12 paired fresh lives (founders 0 to 2, seeds 2000 to 2003): clauses
  -.012 (s.e. .012), two distractors -.005 (.040), held-out +.004 (.015), roles +.010 (.032),
  brackets +.008 (.027), paths -.014 (.010), two hops -.031 (.017), three hops +.022 (.040); over
  36 (seeds 2000 to 2011): -.011 (.007), -.012 (.017), +.002 (.016), -.003 (.020), -.010 (.019),
  +.004 (.014), +.003 (.018), -.010 (.036). Every difference is under .07 and within two standard
  errors. The one-call paths keep the option's bits, the lesson in one call among them: the
  twelve lives read the same panels and work to the bit before and after them.
- The sequence battery (look back and n-back with and without held tokens, two hops, three
  founders): -.0013 (s.e. .0042) over 15 paired lives.

#### Records that fade with disuse (opt-in, #253)

`Brain.compose(..., memory_fade=1.2e-4)` gives the store's place records the dynamics of a habit:
a record nobody uses fades, and a deeply used one fades far more slowly. Each key coordinate keeps
its depth (the unit cue's magnitude summed over the moments that cue it and the writes that use
it) and its recent use (the moments' sum at `1 - 1/2,000` a moment); every moment of life an idle
record loses `memory_fade / (1 + (depth / 2000) ** 4)` of itself. No count is kept. The default is
0.0, and a brain without the setting is the founder's to the bit on `backend="cpu"`: its arrays,
checkpoints, descriptions and parameter counts are unchanged. A fading store rests on the host.

A brain with the setting saves checkpoint format `cadence-brain/3`, with a maturity or without.
0.95.2 and earlier refuse the file ("unsupported Brain checkpoint format"), where they would
load it with the fade off. `Brain.load` reads `cadence-brain/1`, `/2` and `/3`; a brain without
the setting saves `cadence-brain/1`, or `/2` with maturity, as before.

On 24 fresh founders of the competing-skills chamber, with `maturity=True`, a skill practised
for 2,000 moments keeps .36 of itself over 40,000 blank moments and one practised for 40,000
keeps 1.00, and relearning after disuse beats first learning by +4.3 and +4.9 points (s.e. 1.1).
Without the setting both keep 1.00. The chambers from birth, P4 with a mature skill and the short
ring read as without it, and in the robot arena (48 nurseries from birth, 140 fight runs, 48
revision lives) the fights, the acquisition and the revision test hold every gate that maturity
alone holds (#253). See
[records that fade with disuse](docs/memory.md#records-that-fade-with-disuse-opt-in).

#### The composed attention circuit: the cortex at the event's weight, the gains released late (#269)

- `recall_cortex_norm=True` (opt-in): every map of the read reads the cortex of the moment before
  at unit norm per stream, so a lesson's step on the cortex columns weighs no more than its step
  on the event's rows. On the indirect-object chamber (`benchmarks/heads`) it moves the lives that
  form the composed circuit from 2 to 7 of 12 fresh founders and keeps the taught second hop on 12
  of 12. It keeps the memory off the device path. The default and `backend="cpu"` lives are
  unchanged.
- `recall_gain_after=n` and `recall_gain_next=g` (opt-in, with `recall_gain`): the learned gains
  drive nothing and learn nothing through the first `n` lessons, then every next port's gain
  starts at `g` and every recall port's at zero; `brain.recall.gain_lessons` counts the lessons
  and a saved brain keeps it. With the cortex at unit norm, `n=1000` and `g=10`, never-taught
  names are said at .97 to 1.00 on 7 of 12 fresh founders with the taught second hop at .99 to
  1.00 on 12 of 12, one life short of #269's bar. At their founders (0 and 0.0) a brain lives as
  without them.

### Changed and removed: the candidates 0.95.0 kept (#281)

- **`Learner.synapse_rate` of zero holds the whole unit.** A synapse at rate zero takes no step of
  its own and the optimizer keeps its running moments (`velocity`, `second_moment`) as they are,
  where 0.95.2 kept accumulating its contrast into them. A neuron whose every inbound synapse
  steps at zero holds its bias and the bias's moments (`velocity_bias`, `second_moment_bias`) the
  same way, where 0.95.2 stepped the bias. `Learner.held()` names the held synapses and neurons by
  their rates. A reciprocal pair shares one efficacy and steps by the mean of its two directions'
  steps, so the pair holds where both directions are at rate zero; one direction alone at zero
  takes half its partner's step (measured: 1.6e-3 on both after three lessons), while `held()`
  names it and its moments stay. The rule holds for lessons and rewards on the CPU, torch and MLX,
  and in `Lives`. What changes: a learner with a rate of exactly zero on some synapse saves those
  moments at the values they held when the rate went to zero (a newborn's zeros), and the held
  neurons' biases and bias moments at theirs; every other value of such a learner, and every
  learner whose rates are all above zero or absent, is as on 0.95.2 to the bit. No test, benchmark
  or composed brain sets a rate of zero, so no recorded bytes change.
- **Removed: the copy pathway, `recall_copy`.** `recall_gain` (0.95.0, #262) is the learned path
  from a read to its token's action, and the pathway needed the tokens' label, which cost the
  reads only position finds (look back 0.25 to 0.33 against 0.99 to 1.00) and the grammar
  chamber's paths (#259). Write `recall_gain=1.0, recall_tokens=...` in its place, or nothing
  where the task needs no token no lesson used. `recall_copy` is refused at `compose` and `retune`
  with that sentence, and a saved memory with the pathway is refused by name: cadence 0.95.2 runs
  it. A saved memory without it loads as before. `recall_tokens` stays, a declaration of which
  answers are tokens that adds no synapse. `Recall.copy_synapses` and the sequence battery's
  `heads_copy` arm left with it, and the copy-tokens instrument retired to its receipt, which
  verifies with its archived checker. `recall_copy_next` left before any release (#273).
- **Kept: `learning_surprise_share`, `teaching="normalized"`, `attention=True` and
  `recall_next=False`.** Each was read against what a new brain composes, and each stays; the
  readings are on #281. The sequence battery's `composed` arm (`--arms composed --settings ...`)
  reads a one-out of the composition.

### Fixes and the same bits in less time

- A settle of no sweeps makes no standing drive it would not read, and a lesson whose answer
  matches every label makes no targets and reports from one record per batch size. A public
  operation enters and leaves the recall memory's scope without a context manager, a lesson of a
  brain without a learned code keeps its moment without one, and the labels of one slot are
  checked in two reductions.
- **Where the composition's falls come from** (#280). The memory guide lists each fall of the
  default composition with the part that causes it, read on fresh founders by silencing one part
  at a time: the key-door, the read's drive; binding, lessons teaching only the rows that
  surprise; "how long ago?", the read's recency of 0; the competing ring, the read's drive and the
  working trace, in part; the hidden route, no single part. It also lists what removing each cause
  costs. No setting or default changes.
- The API reference lists the `Recall` signature as the class takes it, and AGENTS.md names the
  second goal epic by its title (#173).

### Instruments

The library is unchanged by these. Each reading ran under a protocol declared before its first
life, on fresh founders.

- **The lesson's step** ([`lesson_step.py`](benchmarks/grammar/lesson_step.py), #258). The default
  composition settles the grammar chamber's exact-clause reading within .05 over its last three
  checkpoints on 4 of 6 lives at width 64 and 5 of 6 at width 128.
- **Routes with a goal** ([`routes.py`](benchmarks/grammar/routes.py), #257). A chamber whose first
  hop is said before the goal it must reach, with a sliding planning window beside word by word.
- **Exposure parity** ([`exposure.py`](benchmarks/grammar/exposure.py), #270). Passes of the
  chamber's 4,000 batches beside the transformer's offline control at equal compute. Without
  rehearsal the default composition sits within .05 of the control on every reading but brackets
  (.86 against .97), which reach .99 by the fourth pass.
- The grammar chamber and the sequence battery take a `default` arm: `Brain.compose` with nothing
  named but the backend. A chamber life's work counts the rehearsal's replayed moments and rows.
- `benchmarks/heads`: the indirect-object chamber, the head-function panel, causality by silencing
  and erasure and the matched controls, with its supports and the declared readings (#269).
- `benchmarks/vocabulary`: the grammar chamber at 200, 1,000 and 5,000 words by class and by
  pieces, one hearing, the matched controls, and the read's logits split by source at every word
  whose content a heard moment holds (#266).
- `benchmarks/attention/crossing.py`: the crossing on keyed rows, the conjunction's decoder,
  supports that hold the cortex's synapses, and the matched LSTM and transformer (#267).
- `benchmarks/sequence/held.py`: R1's six tasks at tokens no lesson used, with the read's
  prediction, the vote at each miss and the answer with the association's vote silenced (#272).
- `benchmarks/device`: the ledger, a life's seconds on a device and the lives' per-life cost
  against a process pool (#274); `benchmarks/grammar/cost.py`: a pass's seconds and parts for the
  default brain, the compiled moment and the transformer on one thread (#282).

### Measured negatives

- A lesson's step scaled by the running share of lesson rows answered wrong settles 4 and 3 of 6
  lives and costs held-out clauses .10 to .13, so it stays out of the library (#258).
- A sliding planning window of three reads first hops on held-out start-goal pairs at .312 against
  word by word's .334; neither reads the goal there, and the offline controls read .75 to .83:
  planning does not pay on that chamber (#257).
- `learning_rehearsal=1.0` costs 5.7 times a pass, since it replays each drawn row's whole
  episode, and lowers paths to .82: the recall read delivers fewer hops (#270).
- Recovery after quiet with `memory_fade`: over 40,000 quiet moments after twenty arena fights a
  store sheds 12 % of its mass (23 % at most) and the carried change reads 0.134 against 0.139
  right after the fights, a gain of +0.005 (s.e. 0.003), under two standard errors; the routine
  does not return (#253).
- The compiled moment reaches 2.3 to 2.4 times the transformer's pass in one session and 2.6 to
  2.7 at a default of 4.8 times across sessions; the issue's 2.5 holds in one session and not
  across them (#282, #291).
- With the cortex at unit norm and the gains released at the thousandth lesson the composed
  circuit forms on 7 of 12 fresh founders, one life short of #269's bar; the five that miss fail at
  the read's selection, before the gains are released.

## 0.95.2 — 2026-11-09

`Brain.lesson` starts its free phase from the stream's activity, the state the act settles from,
where it started from rest. Lessons without a stream state (after `reset`, before a stream acts,
and curricula of lessons alone) teach to the bit as on 0.95.1.

### A lesson teaches the fixed point the act is taken from

The act settles warm from the stream; the lesson's free phase settled from rest. In a deep
reciprocal brain one observation can hold two stable states, one reached from the stream and one
from rest. The lesson judged and corrected the rest state, which often gave the shown answer,
while the act kept answering from the warm one, so lessons never reached the act and the warm
state drifted. A deep brain of three regions continued for 4,000 moments on a game task read 0.016
with the lesson from rest and 0.422 with the free phase warm from the stream. The free phase starts
where the act's own settle starts, with a learned code's phase in it; under `learning_surprise` the rows it teaches restrict it, and a replay, the rehearsal and
`step(..., teacher=)` teach from the state their act settles from.

What changes, measured before and after on the CPU:

- A deep brain of two regions whose cue holds two fixed points (the new test's fixture): ten
  lessons from rest leave the act on the wrong answer; ten from the stream correct it.
- Teaching inside a stream (`step(..., teacher=)`, 8 streams, 12 cues, 4 actions, 300 moments,
  three seeds): with `recall=False` the share of sampled acts that matched the teacher over the last
  100 moments rises from .68, .61, .87 to .78, .81, .94 at one region and from .61, .62, .49 to .74,
  .73, .65 at two, and the parameters move apart by up to 0.12 and 0.28. The default composition,
  with the recall memory at the deep point and lessons by surprise, reads the same acts and
  predictions, its efficacies within 2.4e-8, and at two regions one seed of three moves by 0.043
  and predicts 10 of 12 cues where it predicted 11.
- The recorded lives of the test suite that teach after an act read new checkpoint digests, their
  efficacies moved by at most 0.0021, and 2 of their 271 acts differ; they are re-recorded. The
  instrument comparisons of the recall memory pass the instrument its new `warm=True`, as the
  library teaches; its recorded runs keep the lesson from rest.

### Notes

- **The nudged phases near agreement.** At the founder and the normalized teaching points the
  nudged phases stop at the movement tolerance, which near agreement comes after one or two sweeps
  and at different counts in the two phases. In deep modules the contrast is then a distorted
  gradient: a gradient check read cosines of 0.0 to 0.47 against the exact gradient on deep
  modules, and 0.97 to 1.0 with the phases run their whole budget. `teaching="deep"` runs them
  their whole budget (`LearnerConfig(nudged_fixed=True)` for a learner); a brain with the recall
  memory teaches there by default. The defaults are unchanged.
- **`Learner.synapse_rate` of zero.** A rate of zero holds a synapse's efficacy; the biases have no
  rate and step as before, and the optimizer's running moments of a held synapse keep reading its
  contrast, so a rate raised later steps on that history.

## 0.95.1 — 2026-11-08

Two opt-in options for `Brain.lesson`, faster qualified lessons on torch, and `grow` and `graft`
without reading a brain's streams back. On `backend="cpu"` every array, answer, report and
checkpoint of a brain that names neither option is as on 0.95.0, and so is every lesson on torch
whose phases settle at the full step; a torch lesson whose phases halve their step updates on the
device, within the device's rounding of 0.95.0 (below).

### A lesson of a distribution, a lesson of weighted rows (opt-in)

- **`targets=`.** `lesson(observations, targets=p)` teaches a distribution over the actions per row
  in place of labels: `p` of shape `(rows, actions)`, nonnegative, each row summing to one (with
  several motor slots, each slot's columns side by side, each slot's summing to one). The nudged
  phases pull the motor neurons toward `p` as they pull toward a label's one-hot row, and a one-hot
  `p` teaches to the bit as its label does. Its use is distillation: a search's visit shares, or a
  teacher brain's motor activations made a distribution. Under `learning_surprise` every row of
  targets is taught, an episodic memory records no demonstration from them, and a brain with
  `recall_tokens` or `learning_rehearsal` refuses them. A learner takes the same through
  `Learner.step(drive, None, targets=p)` and `Learner.soft_targets(p)`.

  ```python
  brain.lesson(position, targets=np.array([[0.6, 0.4, 0.0]]))  # a search's visit shares
  ```

- **`weights=`.** One nonnegative number per row scales that row's pull, and with it the row's
  share of the update: weights of one teach to the bit as a lesson without weights, and a weight of
  zero teaches its row nothing. Its use is balance: rare classes or critical positions weigh more.
  Under the normalized point a weight common to every row divides out. A brain with
  `learning_rehearsal` refuses weights other than one, since the rehearsal replays its rows at
  weight one.

  ```python
  brain.lesson(x, labels, weights=np.where(labels == rare, 3.0, 1.0))
  ```

See [teach a distribution, weigh the rows](docs/teach.md#teach-a-distribution-weigh-the-rows).

### Qualified lessons on torch

A qualified solve that halves its step (a `learning_qualified=True` phase on its damping ladder,
and the rescue of an act that does not qualify) settles each halved step on a kernel the live
kernel keeps for it: made once from the parameters on the device, given the new parameters where a
lesson changed them, its states resting on the live kernel. On 0.95.0 every halving built a new
kernel and its transport plan from parameters read back to the host, and a lesson whose nudged
phase halved read its phases to the host and took its update there. A lesson whose phases halve
updates on the device as every other lesson does; after ten such lessons its parameters agree with
0.95.0's to within 3e-17 in float64 on the torch CPU device and 6e-8 in float32 on MPS. A settle's
movement (`activity_change`) stays on the device until something reads it, and the optimizer
writes its new history beside the old one without copying it first (on the torch CPU device 38 to
45 ms a call against 43 to 57 at 4.6 million synapses; on MPS within the noise). An act whose
solve takes the rescue reads its closing residual on the device as every other check of a resting
state does, within rounding of 0.95.0's check on the host. The live kernel keeps one kernel per
halved step it used, each holding the transport's weights once on the device: at width 2,048 on
MPS a lesson whose phases halve holds 353 MiB where one that does not holds 315. The MLX backend
builds its halved steps as on 0.95.0.

Seconds per lesson on an Apple M4, torch 2.14, a composed brain of 64 inputs and 8 actions, every
lesson qualified with surprise off, at a tolerance of 3e-5 and 48 sweeps a phase, where most phases
halve their step (the mean of two runs each; CUDA was not measured for this release):

| Device | Width | Rows | 0.95.0 | 0.95.1 |
| --- | --- | --- | --- | --- |
| MPS, float32 | 1,024 | 32 | 0.124 | 0.072 |
| MPS, float32 | 2,048 | 32 | 0.343 | 0.149 |
| MPS, float32 | 4,096 | 16 | 0.911 | 0.347 |
| torch CPU, float64 | 512 | 16 | 0.028 | 0.021 |
| torch CPU, float64 | 1,024 | 16 | 0.075 | 0.055 |

At 2,048 on MPS the halvings' kernels took 24 percent of a lesson's time on 0.95.0 and take 2
percent. Lessons that do not halve take the same seconds and teach the same parameters.

### `grow` and `graft` read no streams back

`grow` and `graft` write the checkpoint of the brain they start from as after `reset()` by taking
its streams aside with the replay's own pair and returning them after the save. On 0.95.0 they
saved the living brain, loaded that file on the CPU and reset the copy, so a brain of many streams
was read back whole only to drop them: at 256 streams of a wide brain the read held 12.9 GB of
eligibility, past what a 30 GB host could hold with the brain beside it. The grown and grafted
brains are as on 0.95.0 to the byte (on the CPU, the torch CPU device and MPS), and the brain they
come from continues as it was. On MLX a recall memory that left the device between the pair's two
calls comes home with its episodes' counts and its reads' focus as integers, as `leave_device`
brings them; read as floats, they made the next checkpoint one that `load` refused.

### Reward learning

For long reward training, lower `actor_eta`: at the composed 1.0 a long life of rewards can drive
the motor neurons to the limits of their activation, and the greedy answer is then the same
whatever the observation. Train long lives at a tenth of the composed rate or less, with the bias
rate a tenth of that ([actor rates](docs/reward.md#actor-rates)).

## 0.95.0 — 2026-11-07

Every new brain holds the recall memory at one composition that reads sentences and sequences with no
further setting, and runs on the fastest backend the machine has at its size. `recall=False` builds a
brain without the memory, bit-identical to what 0.94.0 composes, on `backend="cpu"`.

The composition holds every sequence gate and every grammar chamber bar on three blocks of fresh
founders, each under a protocol declared before its first life, the last on what a new brain composes
here. In one online pass of 82 seconds it reads clauses .99, held-out .93, roles .98, brackets .92,
paths .96, two hops .92 and three hops .38. Eight passes of a matched transformer at about the same
seconds read 1.00, .98, .98, .93, .69, .62 and .00, and offline at about 3.5 times the seconds it reads
1.00 on all but three hops, .00. N-back reads 1.00 at 100 items and .96 to 1.00 at 1,000, where the
transformer reads chance, and ordered recall comes from reward alone on 12 of 12 founders.

The memory and the apical compartment run on MLX and CUDA in one device form, with nothing wider than
the action crossing back per moment above a size rule. `backend="auto"` is the default of
`Brain.compose`, and a device takes float32 unless named. With nothing named, a `step` of sixteen
streams at 3,920 inputs falls from 2,459 ms to 44.6 ms on an A10G and from 10,905 ms to 148 ms on an
Apple M4, and small brains stay on the CPU, which beats a device at their size. The accepted costs are
in [what the composition gives up](docs/memory.md#what-the-composition-gives-up).

- **Default change.** `Brain.compose` gives a new brain with the memory `learning_fast_settle=True`: its free solves take
  `equilibrate(..., fast=True)` on the compiled CPU path, the same equations, order and tolerance in fewer neuron updates,
  with other bits at every width. `learning_fast_settle=False` names the ordinary solve and `learning=` keeps its whole
  configuration. `recall=False` and `recall_founder=True` keep their bits, and a saved or grown brain keeps the solve it has.
  On a third block of fresh founders, under a protocol declared before its first life, this default holds every gate and every
  chamber bar: on the grammar chamber's twelve lives clauses .99, two distractors .97, held-out .93, roles .98, brackets .92, paths
  .96, two hops .92, in 82 seconds a life where the ordinary solve took 154. No chamber of the preservation panel falls against
  `recall=False`. See [settling](docs/settling.md#settling-and-equilibrium) and
  [one composition](docs/memory.md#one-composition-for-sequences).
- **Default change.** A new brain with the memory steps at its association's width `w`, with no new setting: above 64 the
  projection from prefrontal to association and the association-motor pair take `64 / w` through `Learner.synapse_rate`,
  and `recall_address` is .01 times `64 / w`. At 64 and under nothing changes, to the bit, and `recall_founder=True`,
  `recall=False`, `attention=True` and a custom anatomy take none of it. A rate you name wins over its part:
  `recall_address`; a `teaching` other than `"deep"` or `learning=` at `compose`; `brain.learner.synapse_rate` at any time.
  `describe()` states the steps under `layout["projection_rates"]`. A checkpoint keeps them, and a brain saved under other
  rates keeps those. `grow` keeps them, since a brain's width is fixed at birth: a new slot's synapses with the association
  take the pair's step and every other new synapse steps at 1. `graft` leaves every synapse at the host's rates. Both accept
  these rates and refuse rates set by hand. On six fresh founders, twelve lives a cell, the grammar chamber's paths read .97
  (least .84) at 128 and .96 (least .81) at 256 against .94 (.72) and .94 (.67) at whole steps, no life under .80 against one
  of twelve at each width, brackets at 256 fall from .94 to .82, and a life takes 24 and 43 percent fewer
  multiply-accumulates. See [wide brains](docs/tune.md#wide-brains).
- The `"mlx"` backend costs 9 to 13 times less per act and 9 to 17 times less per taught lesson on a
  brain a launch takes, and about 1.6 times less on one it does not take, with the same equations and
  tolerance within float32 rounding; its bits change, and every CPU path keeps its own. A chunk of
  sweeps runs in one Metal launch that ends with the equations' residual, a state stays on the device
  until read, a lesson's update from 131,072 block entries and a recall address map from 1,048,576
  entries stay on the device, and a sweep no launch takes is compiled. On an Apple M4 at the speech
  size (3,920 inputs, 6.3 million synapses), at a load of 12 to 25, an act takes 8 ms against 26 on one CPU
  core, a lesson 59 against 648 and a `step` 45 against 365; at 16 rows of the chamber one core stays ahead.
  `Brain.residual` reads a launch's check of a state under its own drive, nudge and
  parameters, in float32; `on_device=False` keeps the float64 reference, which also decides a solve
  below a tolerance of 1e-4. `copy.deepcopy` of an MLX brain works. `CADENCE_MLX_LAUNCH=0` keeps one
  MLX operation at a time. See [which hardware](docs/backends.md#which-hardware).
- The `"mlx"` backend holds an apical compartment and a learned code, which it refused. A launch reads
  a neuron's basal and top-down products apart and joins them at its pool's gain per sweep and row, or at
  the gain a nudged settle holds, and a source range under another step than `dt = 1` follows its drive
  in closed form, so every neuron rule without adaptation launches. A block holding synapses of both
  compartments, or more than 1,360 settling neurons with a compartment, takes the compiled sweep. A settle
  agrees with the CPU within 2.3e-7, with equal steps and checks, and a continued life on one set of weights
  within 2e-5. At a load of 12 to 25 four regions under a compartment at 64 rows take 12 ms per act and 36 per lesson
  against 35 and 101 on one CPU core, and `modules=(256, 128)` at the speech size and 16 rows 13 and 89 against 210
  and 938. On `benchmarks/depth/and_count.py`'s four regions the mean of the ten readings from lesson 3,100
  over six seeds is .802 on the CPU and .786 on MLX, a difference of .016 with a standard error of .015.
  See [precision matters](docs/backends.md#precision-matters).
- **Default change: the fastest backend, with nothing named.** `Brain.compose` takes `backend="auto"`:
  at its first moment, when its rows are known, a brain takes the fastest backend the machine has for its
  size and rows, by the measured rule in [the default backend](docs/backends.md#the-default-backend), and a
  brain a device passes at one row takes it at compose. On an A10G the speech size's `step` reads 21 ms against
  291 on one CPU core at one row, and 35 against 997 at 16; the chamber at 16 rows stays on the core.
  `brain.placement` states what it took and why, and `describe()` and a checkpoint state it wherever the brain
  is not on the CPU by name. A device that refuses a brain falls to the next fastest and the report names it. Nothing is imported
  below a device's work. `backend="cpu"` keeps its bits; `CADENCE_BACKEND` names a backend for a whole run; tests,
  benchmarks and examples name theirs. MPS leaves the list, and a card the table has not read takes a device from
  twice the work.
- **Default change: float32 on a device.** A device named with no precision, and `backend="auto"`, take
  float32, efficacies, biases and the optimizer's history with it; `precision="float64"` stays by name, and
  `Brain.compose` takes `precision`. A float32 lesson at the speech size and 256 rows reads 81 ms against 1,025
  in float64 on an A10G.
- **A moment on CUDA.** From a moment of `streams * width * (width + 1024) >= 3,000,000` the recall
  memory, the working trace, the consolidating store, the carried routes' contrast, the critic's weights
  and eligibility and the moment's drive run on the device, by the same equations, within rounding of
  the float64 host path; of a moment only the observations and the dopamine go there, and the motor
  columns, a value per stream and the checks come back. `CADENCE_TORCH_MEMORY`
  overrides the size rule. See
  [memories on the device](docs/backends.md#memories-on-the-device).
- **A moment on MLX.** From a moment of `streams * width * (width + 1024) >= 8,000,000` the same memories, the critic's
  weights and eligibility and the moment's drive rest on MLX in float32, by the code CUDA runs, beside the actor's eligibility
  and the optimizer: a moment's reads join the launch that settles on them, a slot is written in place in unified memory,
  a step needs no array of the streams' contrast, and a wide brain's rates step there. At the speech size at 16 streams
  on an Apple M4 a `step` takes 145 ms against 2,115 on one CPU core, at a load of 12 to 25, and 256 streams peak at 7.9 GB.
  `CADENCE_MLX_MEMORY` overrides the size rule. The default brain's chamber and battery panels on MLX read after this
  release, and `backend="auto"` keeps those sizes on the CPU at 16 rows. See
  [memories on the device](docs/backends.md#memories-on-the-device).
- **A moment on a torch device crosses less.** The drive goes up once per moment, a settle from rest
  uploads nothing, a warm start stays on the device, the motor and critic read their columns and no
  state comes up whole. `copy.deepcopy` copies a brain on torch.
- **The read's reward eligibility keeps what can step.** The actor holds the event's rows of the columns
  that can step, 69 columns of 3,920 rows at the default memory and speech's width, where it held 141 MB per
  row, and a checkpoint holds the full layout. Every bit of a CPU life is unchanged. One core's `step`
  at 3,920 inputs: 475 to 262 ms at one stream, 2,408 to 992 ms at 16.
- A lesson's per-synapse passes (the optimizer's step, its projection-relative form and the contrast's gather) compile
  under NumPy's error model, so their divisions vectorize and a zero divisor gives what NumPy's form gives instead of
  Python's error. The bits of every finite step are unchanged. On one thread the optimizer's pass over 504,064 synapses
  falls from 1.6 ms to 1.0 ms. A learned code's forward weights take one compiled pass where `np.add.at` took 0.7 ms.
- A brain with a learned code acts two to three times faster, with the same bits. `LearnedCode.stability` answers
  from Cholesky factorizations at the step's bounds while the halvings hold and from the eigenvalues when they
  change, and the brain takes a moment's new thresholds on its own blocks instead of rebuilding them. At 256 code
  units and 548 inputs on one thread, an act of one stream falls from 34 ms to 12 ms, an act of sixteen streams
  from 111 ms to 47 ms, a lesson and act of sixteen from 247 ms to 114 ms.
- A lesson on a brain with a wide read is faster, with the same bits: the address lesson's readback and the carried
  gains' contrast build each route's weight matrix in one compiled pass. A heavy lesson on a brain of 548 inputs and
  504,064 synapses falls from 43 ms to 33 ms on one thread.
- A `Connectome` warns when a population is given as a sequence out of ascending order or with a neuron twice. It
  holds a population ascending, and a brain's k-th input or action takes its k-th neuron in that order, so a
  caller who listed a late region first had its inputs moved with no word. The population held is unchanged. A set
  states no order and stays silent.
- **Torch speed.** On CUDA and the torch CPU device, under a step of one without adaptation or a
  compartment, a settle makes the products of source neurons (no synapse reaches them) twice and later
  sweeps read them, and from sweep 3 the fused update covers only the neurons synapses reach; from
  4,194,304 entries times rows those products are held for the next settle or residual check of the
  same source neurons. `CADENCE_STILL_PRODUCTS=0` makes every product at every sweep. On CUDA with
  Triton the transport's launch takes blocks of presynaptic neurons, tiles by precision, batch and a
  group's rows, and groups by a rule measured on an A10G, the fused update takes tiles by size, batch
  and precision, the apical normalization is three launches, in free settles, residual checks and
  `apical_gain`, and a softmax nudge over equal groups is one. A residual check replays from a capture
  like a sweep, and a settle of no sweeps from a resident state returns it. A device sum differs from before in its last bits; the settled states match the
  float64 host path to rounding as before. See
  [the implementation notes](docs/backends.md#implementation-notes).
- A float32 torch brain checks in float32 where a solve's tolerance is at least 1e-4, and reads a
  lesson's diagnostics and held gain the same way, so it learns without a float64 copy of its kernel;
  `residual` and `apical_gain` take `own_precision`, and by default keep the float64 reference. A
  lesson's sums of squares per projection reduce one matrix in a fixed order in place of atomic adds, so
  a device life with the memory repeats its bits from run to run, and at the speech size on an A10G a
  lesson at 16 rows takes 22.6 ms against 35.6.
- `NeuralGraph.calibrate_transport`: an opt-in plan of the transport's kernels from measurements, stored
  per graph shape, device and precision, repeatable by name. `to_dict()["torch_transport"]` reports the
  plan of a torch graph. See
  [choosing kernels by measurement](docs/backends.md#choosing-kernels-by-measurement).
- A torch workspace's CUDA graphs share a memory pool of their own, so a workspace that drops every
  capture captures again, where the next capture used to fail and turn graphs off for the kernel.
- **Default change.** `Brain.compose` gives every new brain the recall memory, at the settings
  [`recall=True` gives a new brain](docs/memory.md#what-recalltrue-gives-a-new-brain).
  `recall=False` builds the brain without it; `Brain(connectome)` and a custom anatomy hold one only
  by name. Above 4,096 observation-code coordinates `compose` refuses to choose and names `code=units`,
  `recall=False` and `recall=True`, since the address map grows with the square of the width. A saved
  brain keeps what it was saved with: one saved without a memory loads without one. The composition gives up
  the key-door at delay 2 (18 to 13 of 32 lives, 13 to 10 of 24 on fresh founders), the observed chain (38 to 34
  of 48), the competing ring (.67 to .55 and .70 to .55 on its hard nursery) and the odour nursery `step` at 1,000
  (22 to 18 of 30) against `recall=False`, and against `recall_founder=True` the hidden route (.40 to .22 sampled, .25 to .08 greedy, on 30
  founders), the binding chambers' held-out answers (.97 to .89 and .99 to .92, a restated name .76 to .59), "how long ago?" (.64 to
  .56) and a chain whose stages look alike (7 to 4 of 48). See
  [what the composition gives up](docs/memory.md#what-the-composition-gives-up).
- **Default change.** `Brain.compose(..., recall=True)` gives a new brain `carried_gain=0.01` and
  `recall_drive_gain=True`: every carried route takes a learned gain from its working value, so beside the
  memory the trace falls from 3.0 where the read carries the cue, and the read itself takes one that can
  quiet it and holds it at or below its amplitude. Name `recall_drive_gain=False` or `carried_gain=0.0` to opt
  out, or `recall_founder=True` for every founder. A saved brain, and a brain that grows a memory, keeps the
  gains it has. Brains without the memory are unchanged. See
  [what `recall=True` gives a new brain](docs/memory.md#what-recalltrue-gives-a-new-brain).
- **Default change.** `Brain.compose(..., recall=True)` gives a new brain the recipe for lengths no
  lesson reached and the reward's position read: `learning_surprise=True`, `recall_recency=0.0`,
  `recall_boundary=True`, `recall_anchor=True`, `recall_slope=0.1`, `recall_position_rate=0.01`,
  `recall_relative=True`, `recall_position_reward=4.0`, `recall_grid=(7, 8, 9, 11)` and
  `recall_capacity=128`. Each has its
  own opt-out, its founder by name, and `recall_founder=True` restores them all. `recall_when=True`
  stays an opt-in for tasks that ask "before?". A saved brain keeps the values it was saved with.
  Brains without the memory are unchanged. See
  [what `recall=True` gives a new brain](docs/memory.md#what-recalltrue-gives-a-new-brain).
- **Default change.** `Brain.compose(..., recall=True)` gives a new brain `teaching="deep"` with no
  apical: its lessons take whole nudged phases. `teaching="founder"` or `recall_founder=True` opts out,
  and `learning=` keeps its whole configuration. A saved brain keeps the values it was saved with.
  Brains without the memory are unchanged. See
  [what `recall=True` gives a new brain](docs/memory.md#what-recalltrue-gives-a-new-brain).
- Add `recall_founder` to `Brain.compose`: `recall_founder=True` beside `recall=True` composes the
  memory at the founders of its settings table, whatever `recall=True` gives a new brain. It is no
  setting of the memory, so it is neither described nor saved.
- Add `recall_drive_gain` (opt-in, construction, founder off; it needs `carried_gain`): the read's drive
  is one more route of the carried gains, one learned gain over every read port, starting at
  `recall_amplitude` and reaching no higher, taught by the same contrast on lessons and the same
  eligibility on the dopamine. One parameter. `brain.carried_gains["read"]` reads it, `describe` holds the
  gene and the value, a checkpoint keeps it, `grow` carries it by route and `graft` keeps this brain's. A
  newborn settles as one without the option, and brains without it are unchanged. See
  [the read's route](docs/memory.md#the-reads-drive-as-a-route).
- Add `carried_gain` (opt-in, construction, founder off): one learned gain per carried route, the
  working trace, the efference copy and every block of the heard history, starting at the working
  value and multiplying the route's stimulus. A lesson steps it on the settle's contrast at the taught
  rows and a sampled act's eligibility on the dopamine, over the gain's own second moment, within
  `[0, 8]`. One parameter per route. `brain.carried_gains` reads them by name. Saved with the brain,
  carried by `grow` by route and by `graft`. A newborn settles as one without the option, and brains
  without it are unchanged. See [carried gain](docs/memory.md#how-strongly-a-carried-region-is-heard).
- Add `recall_grid` (opt-in, construction, founder none; needs `recall_position`): past the exact
  table every head's position term also reads each slot's age modulo each of these coprime numbers,
  one term per residue, so a few dozen terms separate the ages up to the moduli's product, where the
  table's buckets double. Once the slots pass the table the slope takes no step. A brain with the
  option reads and learns as without it until a slot passes `recall_position`; the maps are saved
  with the brain, counted by `parameters()` and carried by `grow` and `graft`. `recall_position=32,
  recall_grid=(7, 8, 9, 11), recall_capacity=128` reads a period of 40 at .999 on six fresh founders
  where a table of 64 at 64 slots reads .870, the other sequence gates to the digit. See [reads over
  long context](docs/memory.md#reads-over-long-context). Brains without the option are unchanged.
- Add `recall_gain` (opt-in, construction, founder 0.0; it needs `recall_tokens`), the learned value
  path (#262, a delivery of #260): one scalar per read port on the declared tokens' identity into the
  motor region's drive, starting at zero, so what lessons on some tokens teach holds for a token no
  lesson used, and which port holds the answer is learned. Lessons step each gain by the sum over the
  tokens of the port's read of the token times its motor neuron's movement between the nudged phases,
  at `min(rate, 5 / E)`, `E` the energy of the row's reads, refused whole when not finite (`gain_rows`,
  `gain_refused`); under reward a sampled act keeps that step in one column per port of the read's
  eligibility. The address lesson's readback is untouched. A newborn acts as one without the path.
  Saved as the array `gains`, counted by `parameters`, kept by `grow` and `graft`, refused by `retune`;
  a read revised inside the settle is not composed with it. The copy pathway (`recall_copy`) stays as
  the fixed form. On six fresh founders the answers on tokens no lesson used rise from 0.00 to
  0.92-1.00 on induction, 0.93-0.97 on a period of 8, 0.76-0.97 on copying and 0.70-0.82 on two hops;
  n-back reads 0.08-0.50 and look back 0.00. The cost is the read: a brain that answers early stops
  teaching it, copying repeated tokens falls from 0.91-0.95 to 0.82-0.85, the second hop from 0.99-1.00
  to 0.94-0.96, the grammar chamber's Dyck sentences from 0.45 to 0.36 and its paths from 0.30 to 0.21. See [memory](docs/memory.md#the-learned-value-path-opt-in).
- **The recall read's address lesson changes.** It steps on the keys it names less the keys the read
  attended, the gradient of the content logits and zero where the read is right, in place of the
  distance to the named keys, which keeps regressing the address onto the mean named key. There is
  no setting: every memory learns its addresses this way, and it adds no parameter. A brain at
  `recall=False` is bit-identical. A memory's lessons do not reproduce the bits of 0.94.0. A saved
  memory of 0.85.0 on loads with its map and continues under this lesson. See
  [how the address learns](docs/memory.md#how-it-writes-and-how-the-address-learns).
- Add `learning_fast_settle` (runtime, founder off): a composed brain's free solves take
  `equilibrate(..., fast=True)` on the compiled CPU path. A row stops at the check it is within
  1e-5, and the activation is made in arithmetic that vectorizes. The same equations, order and
  tolerance in about half of a chamber pass, with other bits. A settle's report gains `row_sweeps`. See [settling](docs/settling.md#settling-and-equilibrium). Brains without
  the option are unchanged.
- Add `dense_limit` to `Brain.compose` (construction, founder 2048): a brain whose blocks exceed it
  settles on the sparse transport in Python, about 30 times slower at a speech instrument's 3,920
  inputs; a larger limit keeps it on the blocks and the compiled kernels. With `recall=True` a brain
  that names no limit takes one as wide as its blocks wherever at least half of their entries are
  synapses, so a new memory brain over 2048 squared entries settles on the blocks and its last bits
  differ from what the sparse transport gave. A saved brain keeps its limit, and a brain without the
  memory is unchanged. See [backends](docs/backends.md).
- A brain costs less per moment, with the same bits and the same reports (#271): a chamber pass
  takes about 0.4 of its time. The settle's rows pay no atomic reference counts and run on vector loops,
  a launch reads the activations a state keeps instead of computing them again, a free solve runs in
  one compiled call, the act after a surprise lesson that taught no row reads that lesson's settle,
  a greedy choice with one motor reads the neurons where one stands clear, and the recall read's
  position buckets are table lookups and its episode terms one compiled pass. The recall bank is
  a view into a wider array and a write adds one slot in place of copying every slot, so a write's
  cost does not grow with `recall_capacity`; a caller that holds `recall.nexts` or the other slot arrays
  across a write takes `recall.snapshot()`. The kernels' first use compiles in about 10 seconds on
  an Apple M4. See [what a moment costs](docs/tune.md#what-a-moment-costs).
- With `recall_tokens`, a lesson whose answer is an action without a token (`-1`) leaves the
  recall read's address and position maps as they are: the read's lessons take their target
  from the slots' contents, which hold no yes or no. Inside the positions recipe a yes or no
  lesson pulled the read off the items asked about, and "was A told before B?" stayed at 0.51
  to 0.65 on development founders. Under surprise lessons such a row is taught at every lesson,
  answered right or not: judged only where wrong, the lessons of a judgment between two classes
  carry one class at a time ("before" stays at chance after 4,500 surprise lessons, 0.97 at every
  row), and lessons at every row collapse the token answers of induction, copying and two hops,
  which stay judged by surprise. `recall_tokens` may be given without `recall_copy`, as a
  declaration of which answers are tokens (no synapse is added), and may declare no token at
  all (`[-1] * actions`). Brains without tokens are unchanged.
- Reward steps the recall read's position maps (`recall_position_reward`) by a lesson's step at
  the founder's position rate, 0.1, whatever `recall_position_rate` the lessons take. Beside the
  positions recipe's slow table (0.01) the reward step was a tenth as large: in the ordered recall
  world under `live` the composed sequence brain learned order on 3 of 6 development founders,
  and on 5 of 6 at the founder's step (12 of 12 fresh founders with the step restored by
  setting). Brains at the founder's position rate, and brains without the reward read, are
  unchanged.

### A planning window: the next k actions settled jointly (#257)

A brain composed with `actions=k * menu, slots=k` settles its next `k` actions in one settle
under the meaning it holds, a lesson teaches the `k` slots at once, and the window advances
through the world's echo or the efference copy. No library code changes: the window is the
documented use of `slots` ([a planning window](docs/tune.md#a-planning-window)), with tests of
its contract under `tests/test_planning_window.py` and its readings in the grammar chamber
(`benchmarks/grammar/window.py`). The founder, one slot, is unchanged to the bit.

### What left (#273)

Each name below is refused at `compose` and `retune` with one sentence that says what the brain
does instead.

- `memory_expansion` and `memory_fan_in`. The episodic separation holds 1,024 dense cells, a
  constant, and a brain with a fixed code keys on the code as it is. The two names sized a store:
  no chamber or application read a value of either. Write `code=units` for a wide observation. A
  checkpoint with a sparse separation loads and continues, and `grow` refuses a brain saved with
  a separation of another size. `Brain(connectome, memory_choice=True)` separates its keys as a
  composed brain does.
- `learning_rehearsal_capacity`. The rehearsal's reservoir keeps 512 lesson rows, a constant no
  reading moved. `learning=LearnerConfig(rehearsal_capacity=...)` moves it as an expert.
- The refusal sentences of `LearnedCode.REMOVED`, seventeen names no release took. They are unknown
  settings (`TypeError`), and the retune branch that could never run went with them.
- The live checks of the attention chambers (`benchmarks/attention`): 84 scripts, tests and protocols,
  about 850 test cases. Their receipts stay and verify on archived bytes alone
  (`benchmarks/attention/test_retained_receipts.py`); release 0.94.0 holds the campaigns. Five
  instruments stay: `copy_tokens.py`, `own_action.py`, `look_back.py`, `held_heads.py` and
  `contextual_choice.py`.
- The readers of the checkpoints of 0.84.3 and earlier. `Brain.load` reads `cadence-brain/1` and
  `/2`, every file of 0.85.0 on. It refuses `cadence-generic/1` to `/10`, `cadence-arousal/1` to
  `/3` and `cadence-checkpoint/1` by name: load such a file once under cadence 0.94.0 and save it,
  and this version reads the result.
- `recall_compete`. Several heads read in turn, each taking of a slot at most the share the heads
  before it left, which is what they did at the founder. With `False` two heads read one slot on
  184 of 192 decisions, and on the held-heads benchmark under this release's address lesson the two
  settings read alike (development). Write nothing. A file saved at `False`, and a file of several heads saved by 0.88.0 to 0.91.2, whose
  heads read alike, are refused by name; cadence 0.94.0 runs them.
- `recall_null`, a slot that reads nothing. It had no reading of its own, and a null slot fails the
  fresh-bindings gate. Write nothing: a read takes its whole mass from the slots held. A file saved
  with one is refused by name; cadence 0.94.0 runs it.
- The refusal sentences of the names 0.85.0 retired: the seven `actor_` names of the centred
  dopamine and the critic's signal, `arousal_record_surprise`, `working_memory_focus`, `resting_bias`
  and the recall memory's `write`, `key`, `episodes`, `value`, `context`, `context_decay` and
  `recency=None`. They are unknown settings (`TypeError`), and no file of 0.85.0 on carries them;
  cadence 0.94.0 names each with what the brain does instead. `learning_nudge`, `learning_centered`
  and `learning_decay` keep their sentences, because `Learner.load` reads a learner's file of 0.84.3.
- The properties `Brain.memory_context`, `memory_context_hold` and `memory_extinction`. Read
  `brain.memories["episodic"].share`, `.hold` and `.extinction`. `describe()["memories"]` carries
  them too.

## 0.94.0 — 2026-11-05

One opt-in, maturity, for a brain that goes on learning after it has acquired a skill (#169,
#253). Off, every brain is 0.93.1's to the bit: the four identity batteries and the founder
parity read no difference, and a brain without maturity saves the checkpoint it saved.

On, maturity was read on fresh brains under a protocol declared before the run, in a continuing
robot arena (leagues of six brains, each raised in a 100,000-moment nursery, then fighting;
144 brain-runs per arm), against the founder at its own actor step and at step 0.003, the bar
#169 sets. Seven of #169's eight core gates held:

- Over twenty fights a mature brain loses its driving skill as little as the founder at step
  0.003 does (0.105 against 0.099, standard error 0.011), where the founder at its own step
  loses 0.170; over sixty fights the same (0.003 less than step 0.003's).
- Acquisition is not slowed: the nursery's door test rises by 0.209 against a bar of 0.106.
- The reward chambers of the preservation panel read as the founder's; no chamber life's step
  factor falls below 0.996.
- A route the arena reverses is unlearned as the founder unlearns it (a fall of 0.784 against a
  bar of 0.361).
- A skill practised for 100,000 moments loses 3.5 points while a competing skill is learned
  (the limit is 5), and the competing skill reaches what the founder's reaches; in a short life
  both read as the founder's.

The eighth failed: two of the 24 mature runs fell into the arena's absorbing calm, where every
brain settles into routine and none learns again, against one founder run, and the gate allowed
no more than the founder's. #169 stays open. Whether maturity becomes the default is read by the
defaults panel (#252); a displaced skill does not yet come back faster than it was first learned
(#254).

### Maturity: one slow variable on the actor's steps and the store's writes (#253, opt-in)

`Brain.compose(..., maturity=True)` gives the brain a maturity, for a brain that goes on learning
after it has acquired a skill (#169). Its one variable, the use, is a leaky integral of the
actor's applied synapse steps in the brain's own step units per synapse: each accepted moment of
life (`live`, `step`, or `learn` on its own) keeps `1 - 1 / maturity_recovery` of it (200,000
moments), so no count is kept and the use of a brain that lives without learning relaxes. The
step factor `m = 1 / (1 + (use / maturity_scale) ** 4)` (one half at `maturity_scale`, 10 step
units per synapse) multiplies the actor's synapse and bias steps and engages the store's
surprise-gated writes with strength `1 - m`: a mature brain's write moves a place record only by
the part of its error beyond two of the record's noise scales, so the store keeps what an
outcome confirms and moves on a surprise. Several writing streams share one gain per action,
the mean of theirs, so every stream keeps its residual around the shared records. The critic,
lessons and the recall memory's maps keep their steps.

The use and the store's noise scales are saved, loaded, kept by `grow` and `graft` and counted
by `parameters()`; `maturity_scale` and `maturity_recovery` are `retune` settings and `maturity`
a construction setting. A brain with maturity saves the new checkpoint format `cadence-brain/2`,
which 0.93.1 and earlier refuse; every other brain saves `cadence-brain/1` as before. `Maturity`
is exported from `cadence`. Off, the brain is the founder's to the bit. See
[reward](docs/reward.md#maturity-opt-in) and
[memory](docs/memory.md#maturity-and-surprise-gated-writes).

## 0.93.1 — 2026-11-04

The same answers in less time, from a simpler implementation (#249). Every array, answer,
report, count, checkpoint and refusal is as on 0.93.0: four batteries (13,076 values over the
kernels, the learner, the composed operations and checkpoints; 57,060 over lives with every
0.93 option, saved, reloaded, retuned, grown and grafted; 4,802 refusals; 4,458 values of
several-slot brains) read identical, the founder parity reads 0 of 24 lives differing, and an
independent review compared the two versions on the CPU, torch and mlx. A wide brain learns and
acts faster on the CPU; on mlx a taught life takes about a sixth of its 0.93.0 time.

### The same bits from a simpler implementation

Where 0.93.0 wrote one rule out several times, the library states it once, and every array,
answer, report, checkpoint and refusal is as it was: a battery of 13,076 recorded values over
the kernels, the learner and the composed operations, one of 57,060 over lives with every 0.93
option (saved, reloaded and continued, retuned, grown and grafted), one of 4,458 over lives of
several motor slots and one over 2,802 altered checkpoints and every setting at 27 values read
identical to 0.93.0, and the founder parity reads 0 of 24 lives differing. These changes take
192 lines out of `src/cadence`.

- The recall memory's maps of the moment (the address map, the position table and the
  anchored, relative and slope maps) are built, read, stepped, saved, loaded and checked
  through one helper per operation and one table of their kinds; a slot's place in its
  episode, its bucket and its features of when are each computed once.
- Each stream's state is named once, for the recall memory and for the episodic records,
  across reset, end, keep, a new batch, save and load; a brain checkpoint names a settled
  state's arrays once, the traces held around a moment that is not the stream's own are held
  in one place, and the rows and motor neurons of a chosen action are found in one place.
- A recall setting is validated by one branch per kind of value, the recall genes carry every
  option at its founder, the configurations' field kinds are read from their annotations, and
  the scale arguments and the position reward's pairings are refused in one place each.
- Gone: copies of head 0's port columns and synapses that nothing read, and two arguments of
  the observation code that it never read.

### Settling and learning: the same bits in less time

Every change here leaves every array, every answer and every count as it was: the recall pins,
the founder parity of 0.93.0 and a battery of 13,076 recorded values over the kernels, the
learner, the composed operations and the checkpoints read identical. Timings are medians of
calls alternated with the code before each change in one process on one thread, relative (the
brains: `Brain.compose(8, 4)` with the founder's modules, a 256-neuron or a 1,024-neuron
module, batch 2); each item's were read on a machine shared with other work, the totals below
on a quiet one.

Together, against 0.93.0 (four runs): on the 1,024-neuron module an act takes 0.70 to 0.73 of
its time, a lesson 0.72 to 0.82, a reward step 0.90; with a 256-neuron module 0.94 to 0.95, 0.90
to 0.92 and 0.91 to 0.95; a qualified learner's 0.92, 0.93 to 0.94 and 0.95 to 0.96; the
founder, a recall brain and an apical brain take the time they took (0.99 to 1.01). On the
mlx backend the 0.93.0 soak's taught life (lessons with rehearsal on four streams) takes 59 ms
per moment where 371, its living life 15 where 57, over 1,500 moments with the same final
efficacies and maps to the bit.

- A state the compiled CPU kernels settled or checked keeps the block products of its
  activations, as a private attribute about four fifths the size of the state's own arrays
  (`copy` and `pickle` leave them out; `vars(state)` shows them): a residual check of the state, or a settle
  from it, makes only the products of the ranges whose activations differ from the ones they
  were made from. A bounded solve's check makes those of the ranges its chunk's last sweep
  moved and the next chunk's first sweep makes none; a finite lesson's three diagnostic checks
  read the products their settles left, and both nudged phases start on the free phase's.
  The 1,024-module act takes 0.59 of its time, a lesson 0.91, a reward step 0.92; the
  256-module brain's 0.97, 0.98 and 0.98. The founder's four small blocks cost less to remake
  than to hand over: its act, lesson and reward step take 1.01 to 1.02, which the fourth item
  below returns.
- `Layout.flat` (the flat blocks remade after every update) returns the weights gathered in
  entry order as the array where a synapse sits at every entry, as in the composed brains'
  layouts, without zeroing the array and writing it a second time: the 1,024-module array in
  2.0 ms where 2.8. Its lesson takes 0.91 of its time, a reward step 0.98; the 256-module
  lesson 0.96.
- The optimizer step of a lesson (`fused.optimizer_step`) makes each synapse's moments and step
  in one pass where no projection-relative floor is set (the founder, `teaching="normalized"`),
  without the array of terms between two passes; the relative floor keeps its two. Without
  normalization (the founder) the step takes 0.65 to 0.80 of its time; with the absolute floor
  0.83 to 0.93 on 69,632 synapses and 0.98 to 1.08 on 1,064,960, where the pass is bound by its
  square root and divisions.
- A compiled settle reads the neuron rule once per call, where it computed the rest emission
  (an exponential) once per row and again for the kernel; `settle_batch` and `residual` find
  the graph's blocks once per call.
- The block contrast of a lesson (`block_contrast`) writes each block's float64 products into
  their place in the flat array where it made them apart and copied them (8.5 MB per lesson
  on the 1,024-neuron module); another precision's products are made apart as before. The
  1,024-module contrast in 8.0 ms where 9.5 at batch 2 and 10.3 where 14.4 at batch 1; its
  lesson takes 0.92 of its time, the 256-module lesson 0.98.
- A damping ladder (`equilibrate(damping=k)`: every qualified phase and every rescue) checks
  its smallest step's rule once per rule, where it rebuilt and checked the rule at every call;
  a rule the backend cannot represent is still refused at every call. A qualified learner's act
  takes 0.92 of its time, a lesson 0.95, a reward step 0.97.
- A bounded solve reads its stopping test with the residual's own array methods, and a damping
  ladder whose first attempt qualifies returns that attempt's report, which equals the one it
  rebuilt field for field. A qualified learner's act takes 0.98 of its time, a lesson 0.98.
- A compiled update on the mlx backend (every lesson and every reward step) gives the new brain
  a kernel of its own whose bias and weight blocks are made from the update's host weights as
  construction makes them, sharing the layout's tables, where it built a whole new brain and
  kernel; the brain before the update keeps its kernel and its blocks. A 1,024-module lesson on
  mlx takes 0.90 of its time, a reward step 0.92; a 256-module lesson 0.97.
- The mlx backend makes every block's products at every step, where it read each source
  range's equality back to the host at every step to reuse an unchanged range's products, and
  evaluates a settle's graph once at its end (at least every 64 steps of a long one), where it
  evaluated every step; a settle under a movement tolerance still evaluates every step, as its
  stop reads the step's movement. The same states to the bit, in a fraction of the host
  synchronizations: an act on mlx takes 0.32 to 0.36 of its time, a reward step 0.43 to 0.61, a
  lesson 0.69 to 0.75 (0.97 on the 1,024-neuron module).
- A settle on mlx under a movement tolerance (a finite lesson's phases, every reward step's
  eligibility phases; a qualified learner's phases solve without one) evaluates each step's largest movement with the step's state, one host
  synchronization per step where it took two. A lesson takes 0.80 to 0.94 of its time, a reward
  step 0.89 to 0.97; acts settle without the tolerance and are unchanged.

### Documentation

- `docs/tune.md` describes the current settings only. The wide-brain recipe, the constants
  and the override rules no longer narrate the releases that produced them; a retired name points
  at this changelog's tables.
- Every guide describes the current version only. Release history, retirement tables,
  issue backstories, run narration and readings from outside this repository leave the guides;
  headings and anchors are unchanged.

### Project

- The recall memory's age charge stays at its founder, `recall_recency=0.25` (#250): a gate
  for 0.0, declared before the run on fresh founders, failed on four of its eight parts. Under
  `live` with reward alone nothing trains the founder's read, so the age charge is what orders
  its slots; without it, ages no lesson reached outrank the ages lessons taught.

## 0.93.0 — 2026-11-03

Learning during a continuing life, positions at lengths no lesson reached, and wider brains
(milestone v0.93.0). Every mechanism is opt-in: a brain at its founder settings acts, learns and
saves as on 0.92.1, except that an answer whose solve does not qualify within its budget is now
rescued once before it is refused. Settings that did not meet their issue's gate ship as
documented opt-ins with the readings that bound them.

### Removed in this version

Each name below is refused at `compose`, `retune` and on load with the sentence given; a
checkpoint of 0.92.1 that carries one at another value than its founder is refused by name,
and every founder checkpoint is byte-identical. The two folded learner fields are the one
exception on load: a checkpoint that carries them at other values loads as an expert
configuration, which `describe()` names `"custom"`. Each component keeps its retirement table
(the recall memory's is `Recall.RETIRED`, and `Recall.removed(name)` gives its sentence); the
settings 0.85.0 retired keep naming 0.84.3 as the release that takes them (`recall_write was
retired in 0.85.0: every accepted moment is written; cadence-net 0.84.3 takes it`), and a
0.93.0 retirement names 0.93.0.

- `arousal_hindsight` (0.88.1): the outcome that woke a calm `live` brain credited the
  stream's last `k` calm moments; no gain over the founder was measured. "arousal_hindsight
  was retired in 0.93.0: a calm moment keeps no eligibility, so the outcome that wakes the
  brain credits no earlier action; cadence 0.92.1 takes it". A `cadence-arousal/5` state (a
  positive hindsight) is refused by name; `last_arousal["credited"]` and `ActorCritic.unfade`
  are gone.
- `learning_normalize_relative` and `learning_nudged_fixed` (0.90.0): the projection-relative
  floor and nudged phases that run their whole budget, set in every reading beside
  `teaching="normalized"`, `learning_nudged_steps=32` and `apical`. Folded into the teaching
  point `teaching="deep"`, the normalized point with the floor at 0.1 and nudged phases of a
  whole 32 sweeps; the `LearnerConfig` fields stay for `learning=LearnerConfig(...)`.
  `teaching="deep"` sets the budget `learning_nudged_steps=32` on the way in and a brain keeps
  the budget it has; `describe()` names the point by its floor and fixed phases, so a 0.92.1
  depth checkpoint loads and describes as `"deep"` at its own budget. "learning_normalize_relative was
  retired in 0.93.0: teaching='deep' sets the projection-relative floor at 0.1 with nudged
  phases of a whole 32 sweeps, the one point ever read; learning=LearnerConfig(...) sets it as
  an expert; cadence 0.92.1 takes it" (and likewise `learning_nudged_fixed`).
- `learning_temperature`: the second spelling of `temperature`, the learner's softmax
  temperature, which `describe()` emitted while the docs and chambers wrote `temperature`.
  One spelling, `temperature`, at `compose`, `retune` and in `describe()`.
  "learning_temperature was retired in 0.93.0: temperature is the one spelling of the
  learner's softmax temperature, at compose, retune and in describe; cadence 0.92.1 takes it".
- The keyword shims of 0.84.3: `reset(keep_memory=True)` and `reset(keep_arousal=True)` named
  their `keep` form, and `retune(reset_arousal=True)` named `brain.arousal.reset()`. They are
  unknown keywords now: `reset()` raises Python's `TypeError` for an unexpected keyword, and
  `retune` "unknown brain setting: 'reset_arousal'" (`TypeError`).
- `learning_initial_damping` (0.86.0): the numerical policy that started qualified teaching
  phases at a later index of the damping ladder; no chamber reading since. Retired with the
  policy: the `Equilibrium` fields `initial_damping` and `relative_damping_halvings`, the
  `equilibrate` argument, the per-phase report keys `<phase>_initial_damping` and
  `<phase>_relative_damping_halvings`, and checkpoint format `cadence-checkpoint/4` (refused
  by name on load). "learning_initial_damping was retired in 0.93.0: every qualified phase
  starts at the full step of the damping ladder; cadence 0.92.1 takes it". The attention
  chambers keep verifying retained receipts that carry the policy at its neutral zero.
- The constants of the composed brain. Each was a gene no reading ever moved from its founder
  and no recipe set; by the manifesto's amended sentence a designed number stays a gene until
  the readings show nothing changes it. The name leaves `compose`, `retune` and `describe()`
  ("<name> was retired in 0.93.0: ..., a constant; cadence 0.92.1 takes it"); the value stays
  on its dataclass or as a constant of its component; every key 0.92.1 wrote is still written
  at the constant, so founder checkpoints are byte-identical, and a saved value other than the
  constant is refused by name where the composed brain alone sets it (the episodic records'
  replay and demonstrations, the composition's scales). A `LearnerConfig`, `ActorCriticConfig`
  or `ArousalConfig` passed whole (`learning=`, `reward=`, `arousal=`) is the expert route and
  keeps its values, as the frozen recipes of the acquisition and rhythm chambers compose
  (`LearnerConfig(**recipe)` at momentum 0): `describe()` then names a moved constant by its
  old name, which `compose` and `retune` do not take, and a checkpoint carries and loads the
  expert's value. A saved memory's decay and separation are the memory's own on load, so an
  instrument's memory attached to a brain keeps them (`describe()["memories"]` shows them; they
  are no gene). `scales["history"]` and
  `scales["recall"]` are taken at 1.2 alone.
  - `learning_beta` 0.1: the nudge strength (`LearnerConfig.beta` stays for a standalone learner).
  - `learning_momentum` 0.9: the running mean of each synapse's contrast (`LearnerConfig.momentum` stays).
  - `actor_dopamine_cap` 1.0: the clip of the actor's signal (`ActorCriticConfig.dopamine_cap` stays).
  - `actor_eligibility_steps` 12: the sweeps of each reward-nudged phase (`ActorCriticConfig.eligibility_steps` stays).
  - `arousal_need_scale` 0.1: the need in reward scales (`ArousalConfig.need_scale` stays for the law alone; a 0.84.3 brain checkpoint, saved without the scaled need, is refused by name).
  - `memory_decay` 0.9: the share of a stream's record each later write keeps (both stores).
  - `memory_replay` 8 and `memory_demonstrations` 1.0: the decisions a reward replays and the value a demonstration is recorded at (`EpisodicRecords.REPLAY`, `.DEMONSTRATIONS`).
  - `memory_winners` 24: the active cells of the episodic separation per key.
  - `scales["history"]` and `scales["recall"]` 1.2 (and the 0.84.3 names `history_scale`, `recall_scale`): the initial magnitudes of the history's and the recall memory's projections; `scales` names `"sensory"` and `"trace"`.
  - `recall_heard` 1.0, `recall_address_temperature` 0.1 and `recall_window_decay` 1.0: the present event's weight in its key and its address, the temperature of the slots' scores in the address lesson, and the weight of the window's earlier events (`Recall.heard`, `.address_temperature`, `.window_decay`). A checkpoint still carries `heard` and `address_temperature` at 1.0 and 0.1, as 0.92.1 wrote them, and loads; another value, or `window_decay` other than 1.0, is refused by name ("recall_heard was retired in 0.93.0: the present event weighs 1.0 in its key and its address, the one weight ever read", and likewise the other two).
- `recall_address_step` (0.84.0, runtime): `"toward"` stepped the address by the named keys
  less the mixture it attended to; described and never read in a chamber. The step toward the
  named keys, the founder `"lms"`, is the step. The message: `recall_address_step was retired
  in 0.93.0: the address lesson steps toward the named keys, as it did at 'lms'`.
- `Recall`'s head-0 aliases `w_x`, `w_c`, `b`, `position_x`, `position_c`, `position_b` and
  `attended` are gone: `head_map(0)`, `position_map(0)` and `attended_heads[0]` are the one
  spelling, `head_maps` holds every head's address map (head 0's first), and
  `Recall.map_names(kind, head)` replaces `position_names`, `anchor_names`, `relative_names`
  and `slope_names`. An `AttributeError` names the missing alias; the checkpoint names are
  unchanged.
- `last_settlement["scope"]` (always `"free_answer"`) and `code_halvings` (a learned code's
  step, read by nothing) leave the settlement report; the other keys stay.
- `imagine` loses its `tolerance` argument: its phases settle to the learner's tolerance
  (`learning_tolerance`, 3e-3), the one tolerance an answer is held to, where they settled to
  1e-6 before. `imagine(..., tolerance=...)` raises `TypeError`.
- One name per memory for fresh streams and for a private copy: `begin(keep=False)` leaves
  the memory contract and every memory (`Brain.reset` calls `reset(0)`, or `Recall.end` for
  the slots it keeps, and leaves the stores' records and fast residuals as before), and
  `Recall.imagined()` is `branch()`, the contract's name. `AttributeError` names the missing
  method.

### Positions at lengths no lesson reached

- `recall_anchor` (opt-in, #237): every head's position term also reads each slot's place in
  its stream's current episode (the moments from the episode's first to the slot), bucketed as
  ages are, with one more bucket for slots written before the episode began. Ages count back
  from the present, so "the first told" moved with the story's length; its place in the
  episode does not. Each stream counts the moments of its episode (`Recall.told`); the maps
  start at zero and the lesson steps them with the position maps. On six confirmation founders,
  stories of 24 to 60 with the cues mixed: `first` 0.998 to 1.000 and `second` 0.990 to 1.000
  with it, 0.31 to 0.45 and 0.31 to 0.34 without.
- `recall_slope` (opt-in, #232): above 0, every head's position term also leans by age at every
  length, `s(x, c) * age / 8` with `s` read from the moment as the table is
  (`Recall.slope_maps`), so a preference for old or recent slots holds at ages no lesson
  reached. The maps start at zero and the lesson steps them at that rate by the table's
  gradient.
- `recall_relative` (opt-in, #234): every head's position term also reads each slot's age
  relative to the slot the stream's previous read attended most (`Recall.focus`), offsets
  within `Recall.RELATIVE` (8) a bucket each, so a read can be placed where an earlier read
  pointed (the token after the one just said) at any length.
- **The recipe for unseen lengths**, confirmed on six fresh founders declared before the run:
  `recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01` (with the boundary and no age
  charge) reads look back taught at 5 to 12 at 24 to 60, `first` 0.986 to 1.000 and `second`
  0.988 to 1.000, and copies unique tokens taught at 3 to 6 at 8 to 12 at 0.992 to 1.000 (the
  founder table, on the development seeds: 0.25 to 0.27, 0.25 to 0.32 and 0.30 to 0.36); with
  `recall_relative=True` as
  well it copies repeated tokens at 0.918 to 0.941 in closed loop. Zeroing the anchored maps
  drops `second` to 0.48 to 0.63. Induction and n-back stay within 0.014 of the founder. A
  period of 40 needs the table exact to 64 ages: at 32 the ages from 32 share a bucket per
  doubling and the recipe continues it at 0.35 to 0.39; with `recall_position=64` as well it
  continues it at 0.86 to 0.94 on six fresh founders declared before the run, still reading
  `first` at 0.98 to 1.00 and copying at 0.99 to 1.00 (#232's bar is .85). On the 0.90 default
  search's panel of position changes, against the founder memory, the recipe with 32 or 64 ages
  keeps n-back, two hops and ordered recall, and costs three readings: a period of 24 after
  1,000 lessons (0.83 and 0.84 against 0.92), the Stone Age hidden route after 480 guided
  episodes (0.25 against 0.43) and a composer's read and answer (0.79 and 0.80 against 0.82;
  1.61 and 1.56 against 1.47 nats). The route's cost is the slower table (at
  0.1 beside the anchor and the slope it reads 0.49), the composer's the slope (without it the
  recipe reads 0.815 and answers 1.46 nats, but look back `first` at 24 to 60 falls to 0.57 to
  0.66); the period of 24 has no single cause. The slower table costs reads that need exact
  ages: ordered recall taught by lessons answers every query at the founder rate, 0.50 to 1.00
  (mean 0.80) with the table at 0.01 alone, and 0.875 to 1.00 with the anchor and the slope
  beside it.
- `recall_when` (opt-in, #237): every head's read also delivers when what it read happened,
  `amplitude * sum_i p_i F_i` on a region of its own (`Recall.when_regions`), with `F_i` two
  features of the slot's age and, with the anchor, two of its place in the episode
  (`Recall.when_features`). The regions reach the association through synapses that start at
  zero, laid out after every other part. "How long ago was X told?" answers 0.81 to 0.91 with it
  (0.91 to 0.92 with the anchor too), 0.51 to 0.80 without. "Was A told before B?" over two
  moments reads 0.59 to 0.80 at the founder settings, where the position lesson pulls the read
  off the items asked about (the answer is no slot's content); with the position term not taught
  and a sharp read (`recall_position_rate=0, recall_temperature=0.1`) it answers 0.938 to 0.966
  on six fresh founders declared before the run, and "how long ago" 0.866 to 0.953 (mean 0.912,
  against 0.794 for the learning read with the anchor). The regions' age features at ages no
  lesson reached enter the association through synapses taught at short ages: beside the recipe
  for unseen lengths, look back read at 24 to 60 fell to 0.93 on one of six founders.

Each of these maps starts at zero, so a newborn reads as one without the option, and
checkpoints keep the maps and the per-stream counts. A chained second head (each head after the
first adding to its address what the head before it read) was built on the release branch and
withdrawn before release: it did not answer two hops asked in one moment, 0.44 to 0.46 with it
against 0.44 to 0.47 without on three development seeds, where one think moment answers 0.98
to 0.99. A recall brain saved by 0.89.0, the release
that added the position term, loads at its settings and reads as it did (a test with a file the
0.89.0 wheel wrote).

### Learning during a continuing life

- `memory_extinction` (opt-in, #210): with `memory_choice=True`, a zero outcome writes a learned
  zero for the chosen action where the records' vote for it and its direct payment in this
  place both reach the setting (0.5). A third store, `memories["episodic"].paid`, keeps which
  actions were paid directly where; replayed and demonstrated credit is never extinguished. In
  the issue's bandit (`benchmarks/extinction`), whose paying arm moves and whose failures pay 0,
  11 of 12 lives learn the new arm against 5 of 12 without it. A confirmation panel on fresh seeds found every reward
  chamber's gate where it stands without the rule; the return to the first arm after a second
  reversal is slower (median
  15 trials against 6), and a continuing chain changes on 7 of 10 seeds. The rule is Sean
  McClane's (#210).
- `memory_context_hold` (opt-in, #202): with `memory_choice=True` and a temporal context,
  `"unchanged"` keys a decision taken on the observation of the stream's previous decision as if
  that decision had not been taken, whatever its outcome, and `"punished"` takes a decision whose
  outcome was below zero, in the middle of an attempt, back out of the context.
  `benchmarks/context_key` ports Will Glynn's Stone Age chains with credit. On fresh seeds,
  declared before the run: `"unchanged"` passes the ambiguous chain cleanly in 35 of 96 lives
  against 6 (exact McNemar p < .0001) and keeps the fully observed chain (40 against 43 of 48,
  p = .51); on the chain whose right action leaves the observation unchanged it costs (0 against
  9). `"punished"`, with misses costing .05, passes that chain in 20 of 48 lives against 13
  (p = .039).
- `recall_position_reward` (opt-in, #236; runtime): reward trains the recall read's position
  maps, and with the anchor the anchored maps, through the eligibility the actor traces beside
  the address map's. The reward step moves the rows of the event heard, for the address map as
  well: stepping every row, the form tried on the release branch and withdrawn before release,
  reached #236's .90 on 5 of 11 development founders against 8 of 12, and dragged one query's
  read onto the other's through the cortex and bias rows they share. In the ordered recall
  world through `live` from reward alone, where the founder answers at chance (0.46 to 0.50 of
  its last 256 answers), `recall_position_reward=4` with `recall_address_reward=1`,
  `recall_anchor=True` and `recall_next=False` (the ordered recall benchmark's `reward_recipe`)
  was confirmed on 12 fresh founders under gates declared before the run: 1.00 of the last 256
  answers on 11 of 12 (the twelfth at chance; 11 of 12 development founders), and the recall
  world at 0.848 against the founder's 0.864 (within the declared .05), one write answered at
  0.96 to 1.00 and two writes below the founder (replacement 0.75 against 0.83, order-latest
  0.38 against 0.58). The anchor finds the first write wherever a gap puts it; without the next
  port each payload is held in one port. The preservation panel, the third gate, fails with
  the recipe and the memory it needs: the odour nursery passes 9 lives of 12 against 11
  through `live` and 4 against 7 through `step`, and the key-door nursery acquires its first
  rule in 5 of 16 against 8; the memory alone, read without reward, acquires it in 3 of 16,
  and the nursery keeps its lives with the memory alone and with the reward read in the memory
  as composed, losing them where the anchor and the missing next port meet the reward read.
  It needs the position term and a position rate above 0 (a frozen table would leave reward
  nothing to step), and a pending outcome holds its settings as it holds the address map's.
- `learning_rehearsal` (opt-in, #238; `LearnerConfig.rehearsal`, founder 0): after each lesson,
  that many times the lesson's rows (at least one) are replayed, drawn from a reservoir of the
  lessons' rows kept over the life, a uniform sample (`learning_rehearsal_capacity`, 512 rows),
  each with its episode: the moments its stream heard since the episode began (at most as many
  as the recall memory keeps slots) and the moment it was taught on. A replay runs on fresh
  streams, the earlier moments heard as greedy acts hear them and the last taught as a lesson,
  so the recall read is computed again and its maps are taught with the cortex; the streams
  then return to their state and continue as without the replay. The reservoir
  (`Brain.rehearsal_moments`) and the streams' episodes are saved with the brain; a rate retuned
  above zero keeps the streams' moments from then on. Look back's cues taught first, last,
  second, 600 lessons each, on six development founders: first 0.994 to 1.000 and last 0.968 to
  1.000 with it, 0.32 to 0.74 and 0.58 to 0.95 without, and 0.40 to 0.79 and 0.49 to 0.90 with
  the lessons and the replayed rows given as 3,600 ordinary lessons; every cue 0.998 to 1.000
  taught interleaved. Confirmed on six fresh founders declared before the run: first 0.994 to
  1.000 and last 0.984 to 1.000 with it, 0.42 to 0.85 and 0.57 to 0.89 with 3,600 ordinary
  lessons; every cue 1.000 interleaved. A life with it takes about three times as long. The
  reward chambers of `benchmarks/credit/preservation.py` teach by no lesson, so nothing is
  replayed there: every life reads as without the setting. A form that kept each row's drive
  instead of its episode, tried on the release branch and withdrawn before release, kept types
  the cortex holds but cost the cues a recall read holds (first 0.286 to 0.458 and last 0.458 to
  0.700 on the same fresh founders): a stored drive holds the read as it was when the row was
  kept, and the read's own maps were never taught again.
- A cue the answer does not show (#208, #209; `benchmarks/credit`, a two-cue chamber, and a new
  section of docs/reward.md). Taught by the terminal reward alone, `actor_eta=0.1,
  actor_eta_bias=0.01` with `memory_choice=True` learns the cue-dependent answer at delay 0 on
  11 of 12 fresh founders (the composed founder on 2 of 12); one event between the cue and the
  answer gives 4 of 6, two give none. Taught by a lesson at the answer, `recall=True,
  recall_recency=0.0` carries the cue through 4 and 8 distractors on 12 of 12 fresh founders.
  No library setting changed for these readings. Split into its parts, the composed founder's
  reward update at the answer moves the preference 40 times more along the association activity
  both cues share than along the cue, four fifths of it through the presynaptic activity; an
  actor-critic with exact gradients on a small model of the same structure learns delay 0 from
  the same reward on 6 of 6 founders. The chamber now empties several carriers together and
  reads a donor's carriers in place of the stream's own: a donor's trace and slots decide the
  answer on every development founder, and the answer falls to chance on all of them only with
  the settled activity of the moment before cleared as well. The recipe keeps a lesson-taught
  prerequisite chain at 1.00; in the reward-learning chain chamber its intermediate stages read
  on a fresh copy fall by .06 to .17 while completions hold.
- Reward at the end of an episode (#216, a new section of docs/reward.md): a check of the spread
  of the critic's values beside their mean, the critic school (demonstrated episodes replayed
  through `step` with `actor_eta`, `actor_eta_bias`, `memory_rate` and `consolidation` at zero;
  `memory_rate=0` alone leaves the consolidated record learning) and the value lesson (a second
  motor slot taught each moment's eventual outcome), measured on a light-cycle game. On lessons
  whose outcomes vary the value lesson lifts the association's best linear readout of the winner
  by 0.03 to 0.04 AUC, and after a critic school the critic ranks 0.09 to 0.14 higher at move 10;
  on a demonstrator's own games, which it mostly wins, it does nothing.
- The competing ring (#169) read again on 0.93 (benchmarks/competing/README.md): under the
  melee's signed pay the loss of an acquired skill reproduces at a smaller depth and still lies in
  the associative store's latest records; an averaged record takes about 0.10 off it, and a store
  frozen where the noisy pay begins holds the skill in that table world.

### Wider brains (#239)

- An answer whose solve does not qualify within `free_steps` continues once from where it
  stopped on the damping ladder, another `free_steps` sweeps in equal shares at the full step and
  at successively halved steps, before the brain refuses it; the settlement report counts the
  rescue (`rescue_steps`) and sums both solves' sweeps and checks. Nothing changes where the
  first solve qualifies, after a solve that is not finite, or at a budget of no sweeps. Trained
  recall brains refused acts whose settle needed more full-step sweeps than the finite learner's
  budget left them; the life that refused at lesson 343 now runs to its end. The reward
  chambers' founder lives are unchanged: samples of the context-key, two-cue and bandit lives
  and a competing-ring life repeat byte for byte on this release, no answer in them rescued.
- `learning_surprise_share` (opt-in; `LearnerConfig.surprise_share`, founder `False`): with
  surprise lessons, a lesson's step is scaled by the share of its rows it taught, so one wrong
  row of sixteen moves the net a sixteenth as far as sixteen. At width 1,024 on induction,
  surprise lessons without it swing between collapse and recovery (one seed at 0.01 after 1,000
  lessons); with it three seeds end at 0.95 to 0.99, and six fresh founders declared before the
  run at 0.97 to 0.99, two of them dipping once on the way (to 0.18 and 0.77), where every one
  of the six collapses at least once without it (to 0.01 to 0.29).
- docs/tune.md's "Wide brains" section states the three causes the width sweep located and a
  recipe of fan-in-scaled steps (`Learner.synapse_rate`) for a saturated readout, with its cost.

### Benchmarks

- `benchmarks/sequence`: the sequence battery (look back, copying, induction, n-back, periodic
  continuation, two hops, "how long ago" and "before" over stories; a reading with the anchored
  maps set to zero, and look back's cues taught one after another), ordered recall taught by
  lessons or by reward through `live` with the Stone Age instrument's recall world beside it
  (`--world recall`), and the width sweep. They are the instruments of this release's readings
  for #232, #234, #236, #237, #238 and #239. The battery's `recipe64` arm is
  the positions recipe with the table exact to 64 ages.
- `benchmarks/extinction`: the bandit of #210, failures paying 0 or -1.
- The odour nursery charges a probe copy's refused answer to the life's work (`refused_sweeps`),
  as the key-door chamber does; the settle rescue exposed a strict life's refusal at a probe.
  The acquisition protocol test gives its runs two minutes on a loaded host.
- docs/start.md, "Learning during a continuing life": the settings that reach a behaviour the
  founder's learning does not, each with the chamber that measures it.
- The readings above were taken on this release with these instruments: the declared
  confirmations of #232, #234 and #237 and the #210 bandit's panel lives on their own founders,
  each repeating its earlier reading exactly, and samples of the #202, #208 and #209
  confirmations' lives (`benchmarks/context_key`, `benchmarks/credit`), which repeat byte for
  byte. The #210 panel's other chambers and the competing ring (#169) were read before the
  review's fixes; a ring life repeats on this release with every reading identical. The value
  lesson's readings (#216) come from a game outside the library.

### Settling and learning: the same bits in less time

Every change here leaves every array, every answer and every count as it was: the recall
pins, the founder parity of 0.92.1 and a battery of 14,591 recorded values over the kernels,
the learner and the composed operations read identical. Timings are medians of interleaved
runs against 0.93.0 at 18aa94c on one thread, relative.

- The fused CPU kernels' per-call work is made once where it does not depend on the call: the
  arrays of a settle without a nudge (per batch size and width), a nudge's arrays (kept on the
  nudge for the check that follows its settle), the plain compartment's arrays and the fed
  ranges (per layout), and an apical compartment's pool tables (per compartment). A settle or
  check without a mask hands the kernel none, and a settle of no sweeps publishes the
  activations without a launch. The founder's greedy act takes 0.80 of its time, a lesson
  0.87, a reward step 0.93; a recall brain's act 0.76; a qualified learner's lesson 0.73.
- The flat block array of new weights (`Layout.flat`, remade after every update) is written
  in the order of its entries, the weights gathered by the layout's permutation first (found
  once per layout): the same array in half the time on a wide brain (the 1.06M entries of a
  1,024-neuron module in 5.9 ms where 11.8). A lesson of that brain takes 0.86 of its time, a
  reward step 0.89; with a 256-neuron module 0.89 and 0.93; the founder's 512 entries take
  microseconds either way.
- A finite lesson builds one `Nudge` per nudged phase and reads it for the phase's diagnostic
  check too, where it built the nudge twice and the kernel its arrays twice. The founder's
  lesson takes 0.92 of its time, a recall brain's 0.90 to 0.94.
- The compiled tied step of an update (`fused.tied_step`) also makes the effective weights
  and reads whether the step, the efficacies and the weights are finite, and the brain takes
  those arrays as they are (`_with_update`, the same brain and the same refusals as
  `with_parameters`): three passes over the synapses (a copy of the step, a copy of the
  efficacies, the weights' product) and three `isfinite` passes fewer per update. The
  1,024-module lesson takes 0.92 of its time, its reward step 0.89; the 256-module brain's
  0.92 and 0.92 (old and new alternated call by call in one process); the founder's 512
  synapses are microseconds either way.

### Project

- How the project is planned (AGENTS.md, CONTRIBUTING.md): the roadmap (#109), the goal epics
  (#118, a Doom player; #173, a language model that understands sequences and grammar), a
  milestone per release and the shared threads live on GitHub; how an issue is opened, folded,
  parked and closed; the work and release rules. The issue and pull request templates name the
  goal epic or milestone issue a change serves.

## 0.92.1 — 2026-11-02

- MLX discovery runs in a child interpreter (#105). `available_backends()` imported `mlx.core`
  into the running program, so an MLX whose discovery aborts the interpreter, as an unsupported
  Metal or system configuration can, took the program down with it. A child now asks MLX for its
  device once per process: when it fails, MLX is absent from the list and `backend="mlx"` raises
  `ImportError` with the child's reason (`ModuleNotFoundError`, as before, when MLX is not
  installed). The child imports from the program's import path, paths added at run time
  included, and not from its working directory or environment; an MLX already imported, or MLX
  in a frozen application, answers in the program, the former with its present default device;
  concurrent first calls ask one child, and an MLX installed later in the process is found.
  `tests/test_mlx_discovery.py` checks these with stand-in packages. Composed brains run on MLX
  against the CPU in `tests/test_mlx_backend.py` where MLX imports: they settle and learn within
  float32 rounding, live, imagine, save and continue a pending outcome, read several recall heads
  and grow, and an apical compartment is refused; with MLX 0.32 on Apple silicon (macOS 15,
  Python 3.13) the whole suite passes. docs/backends.md states the support.
- `graft` keeps this brain's signs: a grafted synapse takes the donor's contact count, efficacy
  and plasticity mask. With the donor's signs carried, a graft onto a recall copy pathway of a
  brain with `recall_tokens`, whose signs its construction lays out again on load, left the
  brain's connectome and its learner's apart after a load (their digests differed; the weights
  and the behaviour were the same). A sign is the efficacy a synapse started from, so
  `to_dict()["efficacy_changed"]` and the excitatory and inhibitory counts of a grafted brain
  read the grafted synapses against this brain's construction.
- State learned memory relevance as a goal in the manifesto, AGENTS.md, the contribution rules
  and the pull request template: the same acquired brain uses the same memories differently
  under changed context, and experience improves that selection within the coupled equilibrium.
  Retention, selection and temporal credit stay separate, with the acquisition controls, context
  interventions, generalization and recovery gates of the
  [task protocol](docs/task-design.md#test-learned-memory-relevance); the
  [named-fact selection instrument](benchmarks/relevance/README.md) is the bounded diagnostic
  for the existing System 1 graph and trace. The capability is not yet demonstrated: no runtime
  or default change, and no sentence-training claim.
- Restate the scope and evidence rules for behavior over time in CONTRIBUTING.md and the pull
  request template, and add an issue template that asks for scope, boundary and evidence.

## 0.92.0 — 2026-11-02

- Grow a trained brain: `brain.grow(**parts)` returns a new brain, the trained one with new parts.
  The parts are the recall memory (`recall=True`, with any `recall_*` setting `compose` takes but
  `recall_tokens`), more read heads of the brain's recall memory (`recall_heads`), a heard history
  or the brain's decays followed by new ones (`history_decays`), an efference copy
  (`efference_amplitude` above zero) and new output slots after the brain's own (`slots`); several
  grow in one call. The grown brain is the brain's layout composed again from its recorded
  construction and settings, with the same seed, and the parts added. It carries every learned
  parameter, the optimizer's history, the plasticity masks, the memories' records and settings,
  the actor and critic, the arousal and the random generators, keeps the brain's backend, device
  and precision, and starts its streams afresh, as after `reset()`. Every new synapse onto a
  trained neuron starts at zero with its reciprocal partner, so the grown brain settles to the
  trained brain's states and makes its greedy choices: bit for bit where the larger graph sums in
  the same order, within rounding where it does not (new slots inside the layout, another
  processor's kernel, an accelerator). New slots draw random numbers of their own, so the old
  slots' sampled choices follow other draws. The free/nudged rule moves a synapse on what its two
  neurons do, not on its own weight, so the new synapses learn. The trained brain is unchanged.
  Refused with the reason: a brain that records no construction (a custom anatomy), a part inside
  the brain's loop (`modules`, `observers`, a code: at zero its neurons learn nothing, at any
  other start the brain changes), arousal and the episodic memory, a setting `retune` changes,
  recall tokens, and new slots where a memory is laid out per action (`recall_action`,
  `recall_tokens`, `memory_choice`) or a new motor neuron would join a deep brain's apical pool.
- Graft trained regions of one brain into another: `brain.graft(donor, regions)` returns a new
  brain, this one with the donor's populations `regions` (names, or a mapping of donor names to
  this brain's, such as `{"association": "module_1"}`), and `graft(donor, into=name)` puts the
  whole donor into one population of a larger anatomy, a measured connectome built as one
  region with new regions wired to it among them. A module has no synapses inside it, so a graft
  takes the regions of a trained pathway together (a trained visual system is `visual` with the
  modules above it). The grafted neurons take the donor's biases and gains, and
  the synapses among them its efficacies, contact counts, signs and plasticity masks; their
  optimizer history starts afresh. `freeze=True` stops them learning. A joint, a synapse between
  a grafted neuron and the rest of the brain, starts at zero with its reciprocal partner
  (`joints="zero"`, the founder) or keeps this brain's weight (`joints="kept"`); wiring of this
  brain among the grafted neurons that the donor lacks starts at zero. The brain's other
  weights, its memories and their records, its settings, actor and arousal stay its own, so a
  whole donor grafted into a larger anatomy with joints at zero chooses as the donor does where
  those agree. A joint at zero learns where one of its neurons is active and the other moves
  between the phases, so a new region needs a drive of its own or joints that start with weight.
  Refused with the reason: populations of other sizes, regions that share no synapse, donor
  wiring the brain lacks among the grafted neurons or a one-way donor synapse on half of a
  reciprocal pair, another neuron model, apical compartment or apical role, a learned code's
  region or a joint landing on it, tied or rated synapses.
- [Grow and combine brains](docs/grow.md) is the guide: growing, grafting, how a new part joins
  the brain and when a joint at zero learns, teaching and living after the change, combining two
  brains' skills by teaching, compatibility and cost. Grown and grafted brains save as ordinary
  checkpoints, which 0.91.3 loads.
- `Brain.save(path, compressed=False)` writes a plain archive, as `cadence.checkpoint.save`
  does: a large brain's checkpoint in a fraction of the time a compressed one takes.
- State what learns in a composed visual cortex: its maps project one way into the first
  module, so no nudge reaches them and the contrast rule teaches them nothing. The receptive
  fields keep their initial draw, apart from what phases that stop short of equilibrium leave
  unsettled (a deep brain's founder phases can). `visual_cortex` said that learning shapes
  each receptive field, which holds where a nudge reaches the maps.

## 0.91.3 — 2026-10-31

- Several recall heads compete for the slots (`recall_compete=True`, the founder). A slot can be
  read in full once per moment: the heads read in turn, each takes of every slot at most the
  share the heads before it left, keeps the shape of its own attention where the slots allow and
  gives what a taken slot cannot give to the slots it prefers next (with its whole mass while
  enough is left; a head whose slots were taken whole reads nothing). The shares are computed
  from the logits, so the best slot left keeps its weight however far a taken slot outscored it.
  The address and position lessons and reward's address eligibility take a competing head's
  prior from the shares it read, kept from the read before any lesson steps a map; a head that
  read nothing is not taught. Through 0.91.2 every head's address map started at zero, so every
  head read what head 0 read, and the lesson, whose prior is the head's own attention, could not
  move a head off a sharp shared read: the heads stayed on one moment (at `recall_address=0`
  permanently). A deep brain of two heads (256/128/128, raw keys of 960 values, 960 slots per
  stream, `recall_address=0`) read the same slot with both heads on 184 of 192 decisions at
  weight 1.0, and at `recall_address=0.01` on 641 of 720. One head reads as before.
  `recall_compete=False` keeps the earlier read and can be retuned in a living brain; a
  checkpoint now always records the option for several heads, and a file of several heads saved
  before 0.91.3 loads with `recall_compete=False`, continuing as it was saved; 0.91.2 and
  earlier refuse a file of several heads saved by 0.91.3.

## 0.91.2 — 2026-10-30

- Bind the recall memory's reused content products to its bank for the length of one public
  operation (`act`, `lesson`, `step`, `live`, `wait`). Inside an operation no caller runs
  between reads and the memory replaces its bank rather than writing into it, so the bank
  array's identity establishes unchanged keys; the bitwise snapshot remains the check between
  operations, where an expert's in-place edit of `keys` is read as before. The binding ends with
  the operation, a copy taken during one (`deepcopy`, `imagined`) is unbound once it ends, and
  a forked child starts with none. 0.91.1 copied the whole bank after every write and compared
  it on every reuse; with many streams and slots that traffic exceeded the products it saved
  (64 streams, 960 slots and a 960-wide key are a 472 MB bank). Acts, lessons, `live`/`wait`
  with episode ends, keys and address maps are bit-identical to 0.91.1. A deep composer
  (256/128/128, two held heads, that bank) teaches and acts in 0.21 s per moment on one Mac
  core against 0.42 s on 0.91.1, with the same digest of answers, weights and keys. No task
  score or general training-speed ratio is claimed.
- `lesson`'s warning about a moment the brain just acted on names the caller's line again.

## 0.91.1 — 2026-10-29

- Reuse Torch nudge temperature inputs across changing arousal heat. Bound topology and
  tolerance caches together with their dependent capture/configuration references, so a
  continuing life retains bounded nudge storage. Tiny-temperature arithmetic keeps its own
  capture branch and reads the current temperature.
- CPU and MPS tolerance solves check before executing another sweep. `BrainState` exposes
  `discarded_steps` and `attempted_steps` for CUDA's intentional lookahead while preserving
  the returned state's `steps` index.
- Refuse `ep_structure` on apical graphs: the symmetric plain-sum EP argument does not cover
  their state-dependent normalization. Keep residual qualification and the held-gain local
  learning rule distinct from a loss-gradient certificate.
- Validate original apical checkpoint types and both topology arrays before construction;
  reject NaN/numeric masks, fractional or overflowing pool labels and coerced scalar fields.
  Valid checkpoints retain exact continuation.
- Include the neuron model's actual synaptic gain in both expert learned-code step bounds.
  Restore the previous code step and phase when an enclosing action, prediction, wait,
  lesson or feedback operation refuses. Validate feedback before preparing the next code.
- Measure held heads over the whole memory bank, with separate role-only and global maxima,
  relevant mass, distinct facts, answer counts and denominators. Check both trained arms'
  completion. The existing task gates retain their thresholds.
- Bind current attention/depth experiments to every library module, package metadata and
  protocol source. Record numerical dependencies, effective solver mode and thread settings;
  bind worker outputs before merging, and validate configurations and founder/arm census.
  Preserve original receipts and source snapshots under explicit historical verification.
- Count the actual positive and opposite teaching phases in the thread benchmark. Compare
  optimizer state and phase counts as well as parameters, and exit nonzero on disagreement.
- Restore the original constructor after each deep-memory benchmark job. Reusing a worker
  now runs the declared depth rather than multiplying the depth again on each job; source,
  configuration, cell census and nested behavioral decisions are checked explicitly.
- Reduce recall-memory host copies: an all-stream address lesson reads the canonical slot
  arrays, selected keys are gathered once, and a write builds each new bounded slot array
  with one copy while preserving snapshots. Cache immutable association indices per device.
- Reuse raw recall content products while applying the current temperature on every read.
  Check a bitwise key snapshot before reuse, so runtime retuning, in-place expert edits and
  checkpoint continuation agree. Bound the cache to one bank and four addresses per head.
- Add opt-in `CADENCE_RECALL_THREADS=N` for large host recall products and copies, with one
  reusable pool of at most 32 workers per process. Preserve the original traversal for
  noncontiguous operands; forked children start their own pool.
- Reconcile the task-design and continuing-world-model guides with the current Brain API,
  run the continuing example in installed-package documentation checks, and document the
  mathematical, serialization, refusal and executed-work boundaries above.

## 0.91.0 — 2026-10-28

- The fused CPU settle runs the rows of a batch on Numba's threads. Each sweep computes the
  block products of the whole batch (distinct blocks on distinct threads, one BLAS call per
  block, as before), then every row's step; a row reads only itself, so the rows run in
  parallel with exactly the serial arithmetic, and the batch-wide stopping rule is read after
  the sweep. Settles, residuals and whole lessons are bit-identical to 0.90.0 at every thread
  count. The thread count is Numba's (`NUMBA_NUM_THREADS`, `numba.set_num_threads`); a sweep
  of fewer than 65,536 neurons over its rows, a single row or `NUMBA_NUM_THREADS=1` never
  starts the threads (handing a sweep to them costs tens of microseconds), and a process forked
  from one whose GNU OpenMP threads ran settles serially. Workers of a process pool should set
  `NUMBA_NUM_THREADS=1`. See [backends](docs/backends.md).
- A graph with an apical compartment settles and checks its residual in the fused kernel
  (0.90.0 took the NumPy loop): the basal input summed in the NumPy loop's order, each pool's
  RMS in NumPy's pairwise order, the held gain of a nudged settle. On a Mac and on one Linux
  machine it gives the NumPy loop's bits (potentials, activations, adaptation, residuals, sweep
  counts). On the Linux machine that recorded the depth receipts it differs at rounding: one unit
  in the last place of a settle's activations, 4e-14 in the weights of the AND-count's
  two-module brain after one lesson and 7e-12 after 20, with equal sweep counts. Over thousands
  of lessons the difference moves readings: after 4,000 lessons the AND-count reads one 0.579,
  two 0.642, three 0.804 and four 0.853 there, against the NumPy loop's 0.584, 0.662, 0.809 and
  0.864. `CADENCE_FUSED=0` selects the NumPy loop. `activity_change` differs at rounding, as on
  the plain fused path.
- The per-synapse passes of a lesson run as compiled loops with NumPy's element arithmetic,
  on Numba's threads when long: the flat block scatter, the contrast's gather, the
  projection-relative floor, and the optimizer step (momentum, second moment, floor, rate, pair
  tying and bound) of a host learner whose synapses are all plastic. A lesson finds its apical
  blocks once per compartment and reads the free phase's apical gain once for both nudged
  phases. Weights, moments and reports are the same bits.
- Add `benchmarks/settle_threads.py`: seconds per lesson of a deep apical brain and a flat
  founder at several thread counts, checking that every count learns the same weights.
- Make the torch backend fast on a GPU, its answers unchanged. A torch settle keeps its state on
  the device neuron-major and sums each sweep's synaptic input by a transport plan made once per
  graph: the rows of the synapse matrix that read the same presynaptic neurons (the feature maps
  of one receptive field, a dense region's neurons) form groups, and the groups of one shape are
  one batched product, unpadded, each row written by one product in a fixed order. On CUDA with
  Triton every small group is one Triton launch and the neuron update of a sweep (potential,
  activation, adaptation, movement) another; the sweeps of a settle are captured as CUDA graphs
  and replayed; the movement test of a tolerance is read one sweep behind, so the host never
  stalls the device, and the reported state and step count stay those of the first sweep that
  met the tolerance; drives and masks reach the device through pinned buffers; the apical gain
  a nudged phase holds is read on the device, once per lesson; the learning contrast is
  computed on the device for every graph. A lesson of a four-stage visual brain with the
  apical compartment (7,956 neurons, 903,520 synapses) on a shared A10G in float64 takes 0.17 s
  at batch 64 (1.97 s in 0.90.0, 2.41 s on one CPU core), 1.64 s at batch 1,024 (7.67 s) and
  6.2 s at batch 4,096 (25.7 s); see `benchmarks/torch_gpu/`. Every plan matches the float64
  host path to rounding, with the same fixed points and step counts and activations within
  1e-10 (`tests/test_torch_settle_parity.py`), and the grouped and block plans are deterministic
  run to run (cuSPARSE, the plan of a graph whose rows share few presynaptic neurons, is not).
  `CADENCE_TRITON=0`, `CADENCE_CUDA_GRAPHS=0` and `CADENCE_TORCH_TRANSPORT=grouped|blocks|sparse`
  select the slower paths for comparison.
- A float32 torch state's residual (`NeuralGraph.residual`) and apical gain are computed on its
  device in float64 by a float64 copy of the kernel, the float64 reference they were computed by
  on the host before, so `precision="float32"` no longer pays a host transport per check: its
  lessons of the deep brain run two to three times faster than float64 on the A10G.
- A brain rebuilt on new parameters (`with_parameters`, a damped copy, a calibration candidate)
  reuses its connectome's torch transport plan instead of analysing the graph again.
- Add the copy pathway (opt-in, `recall_copy=magnitude`, founder 0.0: none;
  [#233](https://github.com/muellerberndt/cadence/issues/233)). What the recall read retrieved
  reached the answer only through the association region, so a token never taught as an answer
  was read correctly and never produced. With the option plastic synapses run from every head's
  read ports (`recall`, `next`) into the motor region, appended last and drawn at that
  magnitude, learned by the settle's contrast inside the one settle. `recall_tokens` (founder
  None) declares per action the observation coordinate of the same token (a brain without a
  code, or a learned code joined to the observation); the pathway then starts as that identity
  from each recall port at `recall_copy`, from each next port at zero, and from the present
  observation at `-recall_copy`, so the read's echo of the moment that addressed it is not
  copied back. A newborn produces every declared token its read delivers, and a token never
  read keeps its route to its action. Benchmark `benchmarks/attention/copy_tokens.py`
  (`cadence-copy-tokens/1`, six founders): tokens never taught as answers produced at 1.00
  (founder memory 0.00, whose read names them at 0.32 to 1.00), taught answers 1.00 (0.72 to
  0.98), in-context succession 0.94 to 0.99 against the founder's 0.95 to 0.99; drawn without
  tokens the pathway produces none, and at the recall projection's scale (1.2) it costs taught
  answers (0.58 against 0.90) and succession (0.92 against 0.97). A successor never taught stays
  unanswered in every arm.
- Add the executed command to the recall memory (opt-in, `recall_action=True`, founder off;
  [#235](https://github.com/muellerberndt/cadence/issues/235)). Every written slot carries the
  command the stream executed at that moment, one coordinate per motor neuron (one-hot within
  each motor slot; zero for a moment `wait` senses), in its key and its value; each read port
  reads it on a `<port>_action` region whose synapses into the association start at zero, and
  the address and position maps also read the command of the moment before
  (`Recall.acted`, saved with the brain). A newborn settles and acts as the same founder without
  the option. The episodic vote is unchanged: its context key already holds each issued command
  at the share `1 - memory_context`. Benchmark `benchmarks/attention/own_action.py`
  (`cadence-own-action/1`, six founders, identical observations): alternating its own actions
  1.00 (founder recall memory 0.78, no recall memory 0.47, uniform random 0.48; commands erased
  0.26), a press every third moment 0.97 (0.65, 0.36, 0.46; erased 0.33); the efference copy
  reads 1.00 on both.
- `Recall.witness(observations, state, command=None)`, `Recall.copy_synapses()`,
  `Recall.action_ports(heads, next)`; the three options are construction settings, described
  and saved only where they differ from their founders. Founder brains are byte-identical to
  0.90.0 (checkpoint arrays, acts and descriptions of eight compositions).
- The look-back, contextual-choice and held-heads receipts are rebuilt on this source with
  identical bodies, and the depth receipts on the NumPy loop (`CADENCE_FUSED=0`) on the machine
  that recorded them: the AND-count with identical readings, the deep memory with the same
  verdicts. 0.90.0 run there with the same thread settings gives this source's deep-memory rows
  exactly; the retained ones differed in a control arm's readings.

## 0.90.0 — 2026-10-28

- Add the apical compartment (opt-in, `Brain.compose(apical=fraction)`, founder 0: off). In a
  chain of reciprocally joined regions each region heard the region above it through the reverse
  half of its own upward pair, summed like any synapse; the loop through the pair grew with the
  weights until the settle slowed near its critical gain and a second module carried an echo of
  the answer, so depth added little. With the option the top-down half of every reciprocal pair
  between two regions lands on an apical compartment: in every sweep each region that hears a
  region above it adds `fraction * rms(b) * u / sqrt(rms(u)^2 + rms(b)^2)` to its members' basal
  input `b`, `u` their apical input, the root mean squares over the region. The direction is kept
  and the magnitude never exceeds `fraction` times the bottom-up drive. A nudged settle holds the
  gain each pool had at the free phase's state (`Brain.apical_gain`, `Nudge.apical_gain`), so a
  nudge's change of the top-down input passes undistorted where the drive sits at its bound. It
  runs inside the settle
  on the host (blocked and segmented transport) and on torch (blocks and gather-scatter), and in
  the residual; the numba kernel is skipped for such a graph and the mlx backend refuses it.
  Apical brains settle at half the founder's step. `NeuralGraph(..., apical=Apical(...))` gives any
  graph the compartment. See [the apical compartment](docs/anatomy.md#the-apical-compartment).
- Add `LearnerConfig.normalize_relative` (founder 0): above zero, in place of `normalize_floor`,
  each synapse's RMS gets that share of its projection's RMS (a bias: its range's), host and
  device, so faint credit in a deep region steps on its own scale.
- Add `LearnerConfig.nudged_fixed` (founder `False`): finite nudged phases, and the actor's
  eligibility settle, run their whole `nudged_steps` instead of stopping at the tolerance.
- Add the depth benchmarks: `benchmarks/depth/and_count.py` (`cadence-depth-and-count/1`; one to
  four modules against tanh networks of one and two hidden layers: four modules 0.864, two hidden
  layers 0.885, one module 0.584, one hidden layer 0.426) and `benchmarks/depth/memory.py`
  (`cadence-depth-memory/1`; the look-back, contextual-choice and held-heads benchmarks unedited
  on brains of two and three modules).
- System 2 observers with the compartment: an observer's return to the modules, the association
  and the motor region is a top-down half, so with `apical` set it lands apically and adds at
  most `apical` times each region's bottom-up drive. Observers are tested beside every memory,
  the learned code, held heads and the torch backend (`tests/test_observers_compat.py`).
- Add the recall memory's boundary (opt-in, `recall_boundary=True`, founder off). A slot's next
  is taught by the moment that follows it, so the first event of an episode was the one event no
  slot's next held, and a brain that answered a cue through the next port could not reach the
  start of an episode: founder 3 of the look-back benchmark answered `first` at 0.83, and brains
  of two and three modules at 0.38 to 0.44. With the option every stream that begins holds a
  silent boundary slot (zero key, zero value) whose next the first event teaches; a memory of one
  slot holds none. Every founder
  of one, two and three modules answers both look-back cues at 1.00. The look-back benchmark
  composes its memory with the boundary (`cadence-look-back/2`).
- Checkpoints: a learner whose graph has the compartment saves it (`apical_synapses`,
  `apical_groups`, fraction and floor) in the new format `cadence-checkpoint/5`, which 0.89.0 and
  earlier refuse; a learner with `normalize_relative` or `nudged_fixed` takes that format too, so
  an older loader refuses it instead of dropping the settings. Without them every checkpoint is
  written as before; the two new learner settings are left out of saved configurations,
  described genes and `to_dict` at their founder values, so every brain composed without them
  reads, saves and describes as in 0.89.0.
- The tools that read a graph as a plain synaptic sum refuse a graph with the compartment:
  `Brain.dense()`, `reference.conformance` and `certificate()`. A `SettlementRecord` carries the
  graph's compartment (`apical`, None without one). The compartment's floor must be positive.
- `Brain.retune(teaching=..., learning_...=...)` applies the teaching point and the learning
  genes together, so a change that is valid as a whole (back to `teaching="founder"` with
  `learning_normalize_relative=0`) is accepted in one call.

## 0.89.0 — 2026-10-26

- Give the recall memory a position term, on by default. A key said what a moment was and
  nothing said when, so two slots with the same key were indistinguishable to a read and order
  questions ("at the beginning", "now") could not be answered from the memory. Every head now
  adds to each slot's logit a term for the slot's age bucket, read from the moment as the
  address is (`a = x P_x + c P_c + p`, one map per head), so what a moment hears can choose how
  far back its head looks. Ages below `recall_position` (founder 32) have a bucket each, older
  ones one per doubling up to the capacity. The maps start at zero: a new brain reads exactly
  as one without them until a lesson moves them. Each lesson steps them toward the address
  lesson's target, the slots its movement names, at `recall_position_rate` (founder 0.1,
  retune), whatever `recall_address` is; rewards do not train them. `recall_position=0` builds
  none. See [position](docs/memory.md#position-how-long-ago-a-moment-happened).
- Checkpoints: a brain with the term saves its maps (`recall/position_x`, `position_c`,
  `position_b`, one set per head). A checkpoint saved before 0.89.0 names no position option and
  holds no maps; it loads without the term and reads as it did. 0.88.1 and earlier refuse every
  recall checkpoint of 0.89.0, with or without the term. Brains built with recall now differ
  from 0.88.1's after their first lesson; brains without recall are unchanged, and the Stone Age
  V5 script and P1 sequence instrument, which compose none, read identically. The regression
  records of composed recall founders now state `recall_position=0` and keep their acts; their
  descriptions and checkpoints name the option.
- Add the look-back benchmark (`benchmarks/attention/look_back.py`): five random symbols, then
  a cue, `first` or `last`, answered by the symbol at the cued place, with recency off and only
  the cue moment taught. With the term three of four founders answer both cues at 1.00 and the
  fourth at 0.83 and 1.00; without it 0.31 to 0.81 per cue; uniform random 0.25. The fourth
  founder misses the declared 0.9 gate on both cues, so the receipt's decision is not passed.
- The lesson report of a memory with the term always carries `position_rows` and
  `position_refused` (zero when nothing moved), `Brain.parameters()` counts the position maps,
  and `Recall.position_map(head)` and `Recall.position_term(head, event, cortex)` give one
  head's maps and its term at every slot (`weights(address, position)` with that term gives the
  attention the read used). The constructor's two new arguments follow `writes`, so positional
  calls of earlier versions keep their meaning.

## 0.88.1 — 2026-10-26

- Add `arousal_hindsight` (opt-in, founder 0): the outcome that wakes a calm `live` brain
  credits the stream's last `k` calm moments. The actor and critic step through the held
  transitions in order, the waking outcome last, each held greedy action's eligibility scored
  as a sampled action's (a learned recall address map is not credited); `last_arousal["credited"]` counts them.
  Routine moments still change nothing. The founder is left out of genes, descriptions and
  saves, so every founder brain and checkpoint is byte-identical to 0.88.0; a positive value
  saves as `cadence-arousal/5`, which 0.88.0 refuses. Experimental: no gain over the founder
  has been measured yet.
- Correct the teaching guides for long curricula under the normalized teaching point: teach with
  `learning_surprise=True` from the first lesson. Normalized steps do not shrink as answers
  become right, so every-row lessons drive the readout until the nudge no longer moves it and the
  answers collapse onto one action. The several-heads guide advised switching to surprise once
  the brain answers, which leaves a brain that never reaches the threshold on every-row lessons.
  On the V0.5B two-fact conjunction task (Will Glynn, EXP-72), surprise from the first lesson
  kept 5 of 5 founders at every taught pair and .85 of held-out pairs, against .53 and three
  founders collapsed under the switch. Documentation only; the defaults are unchanged.
- Warn (`RuntimeWarning`) when, in a brain with a recall memory, `lesson` follows an act on
  the same observations. The act has already written the moment to the recall memory, so the
  lesson reads the moment itself, a slot whose successor is still empty. `step(...,
  teacher=labels)` already teaches first; the greedy act that replaces a pending sampled
  action before a lesson, and the act-then-lesson DAgger recipe, are unchanged for brains
  without one.

## 0.88.0 — 2026-10-25

- Add several held memory-read heads: `recall_heads=k` reads one Recall bank through k
  copies of the held read, each with its own read port (`recall_h`, `next_h`) and its own
  learned address map, stepped by the address lesson through that head's own synapses; reward
  eligibility keeps one block per head and checkpoints keep every map. With
  `attention=True` and `recall_heads` above one the recipe learns the addresses
  (`recall_address=0.01`, `recall_temperature=0.25`). One head is the released memory, with
  the same behavior and checkpoint bytes; a checkpoint of several heads is refused by 0.87.0.
- Add the held-heads benchmark. On four founders two heads answer every taught pair when two
  of four witnessed facts are asked in one settle, reported in two slots (one head 21 to 24%,
  uniform random 3%) or combined into one action (one head 59 to 76%, uniform random 10%); the
  one-action answer reaches 96 to 98% on value pairs never taught, the reporting slots 77 to
  98%. Bounded component evidence; the receipt is recomputed by the regression suite.
- Read multi-head attention on the Stone Age sequence instrument (five founders; visible,
  alternating, consecutive and invisible chains; teacher, graded and reward arms). Episodic
  memory with two heads beat the default brain on most arms, mostly through the episodic
  memory; two heads did not beat one head, or episodic memory alone, consistently. Under
  reward alone the heads' addresses do not learn at the founder `recall_address_reward=0`.
  The option therefore stays opt-in and the defaults are unchanged.
- Make the memory guide start from common user needs before the full reference. Clarify
  the difference between episodic voting and learned-code Recall, show that both can be used
  together, update the contextual-choice evidence, and state the single held-read boundary in
  plain language. This changes documentation only, not memory behavior or defaults.

## 0.87.0 — 2026-10-25

- Add `Brain.compose(..., learned_code=units, attention=True)`, an opt-in single-head
  contextual-memory recipe built from the existing learned code and Recall path. It resolves
  to a two-moment ordered key, successor read, zero recency bias, frozen address map and 0.1
  temperature; explicit `recall_*` genes may tune it while retaining a contextual window and
  successor read. The selection is made before the neural settle and held through it. It does
  not change the default brain, enable episodic action voting, revise its query during that
  settle or add multiple heads. The construction marker is reported by `describe()` and
  survives checkpoints.
- Add the public contextual-choice regression benchmark. Across four founders its tested
  8-unit learned code selects action-relevant evidence and reaches 100% on 32 identical-ASK
  queries per founder, against 50% with present-only input, memory disabled or uniform random;
  relevant-record erasure breaks the answer and irrelevant-record erasure preserves 100%,
  with no refusals. This is bounded component evidence, not a continuing-world or default
  claim; eight code units are the benchmark's measured capacity, not a new library default.

## 0.86.0 — 2026-10-24

- Correct normalized learned-code input using the same effective weights as neural
  transport, including contact counts, model gain and presynaptic gains.
- Sum parallel code-to-cortex synapses in that correction, matching neural transport
  for custom connectomes with multiple synapses at the same endpoints.

- Add optional `initial_damping` for qualified CPU float64 teaching. It skips
  initial numerical step sizes within the existing shared budget, preserves the
  live model and original-equation qualification, and reports initial, absolute
  final and relative halving indices. The default zero keeps the existing schedule.
  Different numerical starts can select another attracting basin; check preserved
  answers and measured work on the target task. Non-default numerical policies use
  a checked learner checkpoint format that
  older loaders refuse; ordinary checkpoints keep their format and omit the neutral
  setting and redundant zero-start diagnostics to preserve their previous metadata.
  Public reports regain the derived indices on load. This introduces no memory law,
  retrieval default or general performance claim.

- Add a coupled-memory component experiment under `benchmarks/attention`: live
  query/read equations, independent derivative and local-credit checks, and retained
  exposed-fact admission failures. A separately declared acquisition repair restores
  the existing wrong-answer teaching control, using saved pre-query context. Both
  development founders and all 20 fresh confirmation founders acquire all cases at
  100% under qualified teaching. A separate supplied-content control passes both
  development founders through recall ports, including erasure and donor checks.
  Add immutable saved-row teaching, experimental multiple heads and a minimal
  two-binding protocol with matched resource controls and a diagnostic contextual
  representation probe. Original failures and the 95% gate remain intact;
  those live-query candidates and useful head specialization still need behavioral evidence.
  This adds no supported attention mode to `Brain.compose` and changes no retrieval defaults.
- Add the simpler direct-route attention component: one live 64-coordinate route and
  one born-zero locally learned map, without a separate query state or anchor. Its retained
  development founder reaches 100% allowed and held answers after 64 lessons, 100% causal
  answer/top-1 selection and 98.44% material target mass; `W=0`, relation and relevant-record
  erasures fail while unrelated erasure and payload controls pass. Regression tests retain
  the single-energy derivatives, exact gate-off control, nonzero birth gradient and result
  gates. This is positive single-head component evidence, not a supported compose option,
  recall default, multi-head result or continuing-world claim.
- Preserve the experimental memory boundary through qualified step-halving attempts
  and clear stale boundary caches on trial reset. Add a matched empty-read
  agreement control and retain its failed acquisition result; numerical work
  and the original 95% floor remain explicit.

## 0.85.1 — 2026-10-22

- `live(..., salience=[...])` accepts the explicit outcome importance already supported by
  `step`. A positive mark lets an aroused episodic brain record a known zero-valued outcome
  while leaving its actual reward at zero for replay; it also controls consolidation on a
  waking write. A mark alone does not wake a calm brain or make it learn. Validation and
  decision ownership are checked before feedback changes state. This supplies the missing
  `live` control from [#210](https://github.com/muellerberndt/cadence/issues/210), not automatic
  extinction or a general remedy for reward-only chains; the reported residual harm remains.
- A pending-action checkpoint must carry its decision-time observation, action and learned
  code where applicable. Missing records now raise on load rather than silently losing the
  action's associative-memory write. Composed checkpoints also validate that learner outputs
  retain the motor population's order and action meanings. Valid earlier formats still load.
- Recall and episodic memory accept boolean stream masks consistently for selective reset,
  end and keep operations. Invalid selections raise before any stream changes; a masked
  reset preserves the other streams' slots, context and replay paths.
- Learned-code local updates validate all input shapes and finite values before changing
  tables, and roll back tables, graph and work counts if a later proposal fails. Column
  normalization handles finite values whose squared norms overflow or underflow; exact
  cancellation preserves the previous column's direction. Ordinary normalization is unchanged.
  The fallback integration bound includes the neuron's negative activation branch, and
  invalid step halvings leave the graph and its reported step consistent. Code-phase
  diagnostics count the failed initial attempt as well as its fallback's sweeps.
- `describe()` includes the four effective expert teaching settings under `genes["learning"]`
  when `teaching` is `"custom"`. Founder and normalized teaching descriptions keep their
  existing form. This patch changes no composed defaults, checkpoint format or memory-selection
  mechanism.

## 0.85.0 — 2026-10-22

- The memory guide scopes the paid-record reversal workarounds to their measured tasks:
  intermediate penalties can stop replay and harm reward chains, and explicit salience is a
  `step` option, not a `live` option. Automatic extinction remains unimplemented (#210).
- A checkpoint saved immediately after `reset(keep="slots")` loads its retained recall
  rows even though the reset cleared the current neural state. Both current and supported
  earlier checkpoint formats preserve the next actions and reward updates at this boundary.
- A learned code follows `live`'s arousal gate: routine answers and `wait` moments while
  a routine action awaits its outcome keep its dictionary, predictor and thresholds still.
  Aroused moments resume its local learning; explicit `act` and `step` retain their
  every-moment rule. The learned-code benchmark tests now run in the default test suite.
- The learned code's predictor bounds its simultaneous update for batches above 16 rows.
  Repeated contexts in larger batches could amplify the prediction error and make its
  weights diverge. Dividing the aggregate step by `max(1, rows * predictor)` keeps its
  normalized context coupling at most one; the founder law through 16 rows is unchanged.
- `Brain.reset()` keeps the arousal state by default: its `keep` is `("arousal",)`, so a reset
  stream begins with the level, the reward references and the reward scale the brain was
  used to ([#213](https://github.com/muellerberndt/cadence/issues/213)). `reset(keep=())` is
  the reset of 0.84.3 (the arousal calm, its references and scale cleared, its age kept), and
  `keep=("slots",)` keeps the recall slots and clears the arousal as before; name both to keep
  both. In a two-armed bandit whose paying arm moved after 300 moments
  (`Brain.compose(1, 2, modules=(16,), arousal=True)`, seeds 0 to 4), adults reset with
  `keep=()` chose the newly paying arm at none of the next 120 moments, while adults reset
  keeping the arousal woke and chose it at each of the last 40, as adults living on without
  the reset did. Founders change for every brain with arousal that is reset in a life.
- The scaled need is a founder: `ArousalConfig.need_scale` (`arousal_need_scale`) is 0.1, so a
  composed brain with arousal wants at least the unmet share of a tenth of its learned reward
  spread once its youth is over and an outcome has entered the spread
  ([#198](https://github.com/muellerberndt/cadence/issues/198)). `need_scale=0` is the law of
  0.84.3 and stays an option. Founders change: every brain with arousal whose income falls
  below a tenth of its reward spread after youth now wants, explores and learns where it
  stayed calm. Saved arousal states of earlier formats load with `need_scale=0`, as written.
- `Brain.compose` resolves an unset `lateral` to 0.0 at every menu size; an explicit
  `lateral`, -0.5 included, is used as given
  ([#220](https://github.com/muellerberndt/cadence/issues/220)). 0.73.0 had resolved it to
  -0.5 for a largest motor slot of up to 8 neurons, on the measurement that the undamped free
  solve stops settling under -0.5 from 12 actions. Measured again, the oscillation is there
  from 4 actions on: 8-action composed brains at 8, 64 and 128 inputs, seeds 0 to 2, with 50
  greedy acts each on uniform random observations ran the undamped half of the free budget
  out on 94 of 450 acts, which settled only under the half-step damping fallback, at 544
  sweeps against 32 at 0.0; with 16 rows per act every act did; and `Brain.lesson` with 16
  rows per lesson, whose free phase has no damping fallback, exhausted its whole 1,024-sweep
  budget on 28 of 30 lessons at 8 actions and 7 of 30 at 4, learning from an unsettled
  state, where at 0.0 every lesson settled in about 6 sweeps. Founders change: every brain
  `compose` or `Genome.composed` makes with `lateral` unset and a largest motor slot of at
  most 8 neurons loses its motor-to-motor synapses, the benchmark chambers that leave
  `lateral` unset among them (chain, competing, replay, key-door, reversal, selective recall,
  bindings, acquisition controls, steady rhythm, reward rhythm and the CUDA chamber). Brains
  whose largest slot exceeds 8 neurons are byte-identical.
- `copy.deepcopy(brain)` works: a brain copies every array, store, state and random
  generator and shares its immutable connectomes (`Connectome` copies as itself), with the
  learned code region or without it; instruments that forked brains by hand no longer need
  to.
- `Brain.compose(..., learned_code=units)` (opt-in) adds a `code` region of the brain's own
  neurons between the sensory region and the first module: a learned compressed code of the
  observation that the cortex computes on and every memory keys on. Its tables are its
  units' synapses and biases: a dictionary from the sensory neurons (read in units of one
  full input), the dictionary's Gram matrix as lateral inhibition, a predictor from the
  prefrontal region (the working trace) as a top-down boundary with a standing pull toward
  the prediction, and a threshold per unit. A moment's settle runs the region's own phase
  first, by the brain's solver policy; the code is held as a source through the whole
  settle, so the rest of the brain keeps the model's step and its learning phases, and the
  region's phase alone steps where its competition needs it (`LearnedCode.stability`, with
  a retry at the step the dictionary allows over every state before a moment is refused).
  At a moment the region learns from it settles twice: in the stream's context, the code
  it learns from, and on the observation alone, the context-free code the cortex reads and
  every memory keys on (`context_free_steps`). The first module reads the rectified code at
  unit length. The key joins the code to the observation's own identity at weight .9,
  `unit([sqrt(1 - w) code, sqrt(w) unit(x)])` at the norm the observation itself has as a
  key, so a binding stays readable while the code moves. An observation wider than the code
  (more inputs than units, a camera's frame) is keyed on the code alone, so that the key
  keeps the code's width (`LearnedCode.joins`, `code_width` then `units`). In the camera
  chamber a separate `CodeRegion` prototype, with an imprinted dictionary, 256 nursery rounds,
  no top-down boundary and body inputs beside the code, answers the held-out pairs at .970
  against .927 for the fixed random code of 1,024 units on the development founders, with a
  recall address map of 0.198 MB against 9.331 MB; these readings do not establish the
  integrated region's camera behavior. With `memory_choice=True` the episodic
  stores separate the keys as an observation's (1,024 cells, 24 winners). The region learns
  at accepted explicit acts and aroused `live` moments by local rules (Hebbian on the reconstruction
  residual, the delta rule on the prediction's residual, homeostatic thresholds whose share
  counts units driven above the active floor by the context-free code,
  and a repair that lowers the thresholds of a silent row's units); the lesson's contrast
  and the dopamine leave it alone. The one setting is `learned_code_fields` (`None`: one
  field; a sequence of field widths shares the units among the observation's channels,
  each unit reading its channel alone). Every other part is a constant of the region
  (`LearnedCode.TOP_DOWN` 2.0, `RATE` 0.125, `PREDICTOR` 0.0625, `THRESHOLD` 0.1,
  `THRESHOLD_RATE` 0.000625, `TARGET` 0.1, `ACTIVE` 0.05, `REPAIR` 1.0, `SILENCE` 10,
  `IDENTITY` 0.9); `brain.learned_code.learning = False` holds the tables still for a
  reading on a copy, and `brain.learned_code.top_down = 0` is the control without the
  top-down boundary. In the embedding chamber held-out members of a group are answered at
  1.000 on 6 of 6 development founders and 12 of 12 fresh ones, where the raw observation
  and the fixed random code answer at chance; the binding family stays within .05 of a
  brain without the region. Settings the readings retired are refused by name with what
  the region does instead (`LearnedCode.REMOVED`): `learned_code_cortex` (the observation
  beside the code read the chamber's held-out members at .24), `learned_code_output`
  (the activations as they are read .48; a read at the observation's own loudness lost
  delayed cues), `learned_code_floor` (.61), `learned_code_scale` (a key at unit length
  read the continuing task at .770 against .935), `learned_code_place`,
  `learned_code_reads` (the code in context read 7 of 12 fresh founders at .90 against 11
  of 12), `learned_code_held` (settling the code with the whole graph lost the binding
  family) and the constants' names. An imagined branch settles the region as a real moment
  does and leaves its step, its work counts and its latest phase as it found them. The
  checkpoint carries the tables and the two switches as its `learned_code` group;
  `describe`, `parameters` and `observation_code` cover the region; `LearnedCode` is
  exported. With the founder `learned_code=None` every brain,
  checkpoint and life is the one of the library without the region
  (`tests/test_learned_code.py`).
- Settings of the recall memory that 0.84.3 took are removed, each refused by name at
  `compose` and `retune` with a message naming 0.84.3 as the release that takes it, and a
  checkpoint saved under one is refused on load with a message naming 0.84.3; each was read before
  it left, and every founder brain, checkpoint and composed life is byte-identical to the
  library before the removal (pinned), the options having been saved only where they
  differed from their founders. `recall_write="aroused"`, a moment written only when its act
  was sampled (`Recall.witness` loses `aroused`, and every write is a slot each stream
  holds), was neutral on every chamber it was read on. `recall_key="predecessor"`, the
  previous event and the one-hot of the last command in every key (`Recall.before`,
  `Recall.command`, the `recall/before` and `recall/command` checkpoint arrays, `witness`
  loses `command`), was neutral where it was read, and the window key and the efference copy
  carry the same information. `recall_episodes="keep"`, a stream's slots kept across an ended
  episode, lowered success by .108 on a crafting chain whose observation names the stage: an
  ended episode empties the slots, as the founder did, and `Brain.reset(keep=("slots",))`
  keeps them across a reset as before (`Recall.end` stays for it). `recall_recency=None`,
  the power-law age charge `r * log(1 + age)` (`Recall.charges`), failed its confirmation on
  the camera task's founders; a number keeps the linear charge `recency * age`.
  `recall_value="cortex"`, the settled association activity as a slot's value, failed all
  four binding tasks (token .06, succession .07, fresh bindings .55, camera .00): the values
  are the events, and the `recall` and `next` regions hold one neuron per code unit.
  `recall_context` and `recall_context_decay`, a fading history of the earlier events added
  to every key (`Recall.history`), were never read above zero in a chamber. A checkpoint of
  0.84.3 carries `value`, `context` and `context_decay` at their one value with an empty
  history array and loads. The binding instrument keeps its arms (`benchmarks/binding/index_read.py`) with
  their protocol.
- `MANIFESTO.md` and `AGENTS.md` use transformer capabilities as design comparisons with
  learned codes, the settling cortex, recall memory and the motor population. Recall selects
  before the neural settle and holds its read fixed; selection that revises with the present
  interpretation remains a goal. These comparisons need behavioral and work measurements.
  What was tested and did not work is removed or folded into what did, and the documentation
  is organized by situation with the interface kept as simple as possible for users.
- The memory guide's episodic part (`docs/memory.md`) states a limit of `memory_choice=True`
  in a continuing life: a record that paid is not extinguished by failures that pay nothing,
  since a zero outcome in the middle of an attempt writes nothing and replay only adds
  credit, so after a reversal whose failures pay 0 the new action is often not learned. It
  names what today's rules offer: failures that cost a little (12 of 12 lives learned the
  new arm in the guide's reading), failures marked with `salience=`, which the records
  write, or the memory left off
  ([#210](https://github.com/muellerberndt/cadence/issues/210)).
- `ArousalConfig` gains `need_scale` (`arousal_need_scale` at `compose` and `retune`; opt-in,
  founder 0): a need in units of the reward scale the brain has learned. At every outcome the
  want is at least the unmet share of `need_scale * scale`, the scale as it stood before the
  outcome entered it; the want and heat read afterwards use the scale as it stands; youth and
  a scale no outcome has entered leave it at zero, so a default `reset()` zeroes it until
  outcomes enter the new stream's scale, and `reset(keep=("arousal",))` keeps it. A save
  carries the gene under `cadence-arousal/3` only above zero; saves of earlier formats load
  with `need_scale=0`, and a positive `need_scale` under an earlier format is refused. At the
  founder every brain, checkpoint and composed life is byte-identical to the library before
  the gene (pinned); `describe()` carries the gene at zero (`arousal_need_scale` under
  `genes`), and `ArousalConfig.space()` leaves it out, so seeded
  mutations at the founder draw as before and copy the zero to the child; add it to a space
  to select on it. At `need_scale=0.1`, with the actor step at the composed founder, against
  the founder on fresh seeds the composed brain kept every preservation chamber:
  key-door rule A acquired on 10 of 16 lives against 8 of 16, with fed rule-ends calm on 20
  of 21 and unfed ones aroused (late aroused share .10 at fed rule-ends against .84 at unfed
  ones, where the founder is calm at both, .01 and .00, with 17 of 17 fed rule-ends calm and
  16 of 16 lives calm in every rule); the prerequisite chain collapsed onto one action on 3
  of 12 lives against 12 of 12 and was learned on 7 of 12 against 1 of 12; the odour nursery
  passed 8 of 12 lives in both; the two-armed bandit with reversal won 30 of 30 blocks in
  both. The cost follows from the law: in a world whose income stays below the share the
  want stays at the unmet share and the level settles above the threshold, so the brain
  stays aroused and pays learning sweeps at every moment; with pay at the end of the chain,
  5 of 6 lives stayed aroused for .60 to .67 of the life on an income of .017 to .022 per
  moment, the sixth, earning .15 per moment, for .06, and the founder's 6 lives, all
  collapsed, for .02. Through `step` the gene is inert.
- The teaching guide's section "Many lessons: the normalized point for imitation"
  (`docs/teach.md`) recommends the teaching learner's steps 0.003 and 0.0003 under a running
  RMS of forgetting 0.99 and floor 1e-4 (`teaching="normalized"`, below) with lessons of
  about 32 rows for a brain that learns a demonstrator's policy from many lessons
  ([issue 215](https://github.com/muellerberndt/cadence/issues/215)). In a copy task of 260
  inputs and three actions it agreed with the demonstrator on 0.868 / 0.866 of held-out
  moves after 240,000 samples (two seeds), against 0.874 / 0.888 for a backpropagation
  network trained with Adam on the same samples, where the composed founder reached 0.81 at
  80,000 samples. The composed founder keeps its teaching rates: with few lessons its larger
  steps acquire faster (a taught chain from a few hundred corrections, and the token-binding
  content arms at long budgets,
  [issue 218](https://github.com/muellerberndt/cadence/issues/218)). The guide also gives
  the working-trace amplitude's effect on teaching fidelity.
- The teaching guide's new section "Teach on the brain's own path" (`docs/teach.md`) gives
  the DAgger recipe for a brain that bootstraps by watching: its own greedy move is
  executed, the same demonstrator labels the observation it reached, and `lesson` teaches
  it, with a runnable snippet.
- The tuning guide (`docs/tune.md`) lists `normalize=0` among a fresh `LearnerConfig()`'s
  values.

- `Brain.reset(keep=("arousal",))` (opt-in) keeps the arousal state as it is: its level, the
  counts, the recent and long-run reward, the usual errors, the reward scale, the age and
  the work counts carry into the new stream, as when the state saved before the reset
  (`Arousal.to_dict`) is restored after it (`Arousal.from_dict`). In the two-armed bandit
  whose paying arm moved, adults reset this way woke and chose as they did living on
  without the reset, at every one of 120 moments in 5 of 5 seeds, the newly paying arm at
  each of the last 40; a default reset left them at none of the 120. Reset into a stream
  that paid as before they kept their arm calm at all 120 moments in 5 of 5 seeds. A brain
  without arousal resets as it does without it. At the default, `keep=()`, every brain
  resets, acts and saves as in 0.84.3.

- The documentation is organized by situation: 63 pages and 16,193 lines become 14 pages and
  about 5,400 lines, each page answering what you want and what to use. `docs/start.md` (a
  creature in a body: `compose`, `live`, `wait`, `reset`, `save` and `Brain.load`),
  `docs/teach.md` (lessons, `step` with a teacher, the teaching point, the brain's own
  path), `docs/reward.md` (with the sequential tasks), `docs/memory.md` (the eight memories
  of the composed brain in one table, which memory serves which task, a runnable example per
  memory, the episodic and recall memories in full, the standalone stores), `docs/senses.md`
  (images, the observation code, the learned code), `docs/tune.md` (the region map, the
  founders, `retune`, the operating point), `docs/evaluate.md` (baselines,
  independent-sample controls, receipts), `docs/anatomy.md` (genomes, cortices, connectomes,
  evolution), `docs/settling.md` (the neuron, the free/nudged rule, the certificate), and
  `docs/backends.md`, `docs/troubleshooting.md`, `docs/contracts.md` and `docs/api.md` (the
  signatures); `docs/README.md` is the single entry and the catalogue by situation, one
  guide owns each setting, and the README carries the comparison for machine-learning
  readers. The pages architecture, context, creativity, equilibrium-world-models,
  experience, howto-demo, index, interaction, missteps, pages, recursive-training, scaling,
  settling-literature, task-design and tasks, and the equilibrium folder's BOOTSTRAP, DRSN,
  ELEMENT, EXPERIMENTAL, PERFORMANCE and VARIANTS, are merged into these pages or removed.
  Every section that stays keeps its heading and text, so its anchor holds on its new page,
  and every link of the documentation, the README, the benchmark READMEs and the library's
  docstrings and warnings points to the new place.
- Docstrings and comments across `src/` describe current behaviour only, with no version
  narration, run records or issue history, and link to the new pages; docstrings that
  disagreed with the code beside them are corrected. Code and every message string are
  unchanged, verified on the syntax tree.
- `MANIFESTO.md` and `AGENTS.md` state the principle that the brain computes, and indexes its
  memories, on learned compressed representations of meaning: raw observations are what it
  receives, and the code it learns from them is what the cortex settles on and what the
  memories store and address ([issue 211](https://github.com/muellerberndt/cadence/issues/211)).
- Remove the population solver, `cadence.experimental.equilibrium` (12 modules), with its tests
  (`tests/equilibrium`, 1,053 of the 3,960 collected tests), guides (`docs/equilibrium`, 9
  pages) and examples (`examples/equilibrium`, 11 scripts with their receipts). The composed
  brain never imported it and no benchmark used it; every composed brain, checkpoint and
  composed life is byte-identical. Its projected-gradient state-and-error readback lives on in
  release 0.84.3, which also loads its checkpoints, and in the application demos that pin
  `cadence-net==0.70.0`.
- Remove the record era: `RecordPatchNet` with `RecordPatchStack`, `JointRecordPatches` and the
  structured ports, the `Records` store, `BeliefPatch`, `Steered`, `Life` with its governors,
  the orienting instruments and `cadence.population.PopulationPatch` (11 modules, 6,898 lines,
  36 top-level exports), with 15 test files, the record and belief cases of 4 more, and the
  record patch, belief, steering and population guides. The composed brain never imported
  them; every composed brain, checkpoint and composed life is byte-identical. The standalone
  stores guide keeps `FastSynapses`, `SynapticMemory`, `PatternSeparator` and the traces.
  `RecordPatchNet.dream` and `sleep`, the belief patch, the steered cortex and the governed
  life live on in release 0.84.3 and in the application demos that pin earlier releases; the
  population throughput receipt stays in `benchmarks/population` as a frozen reading.
- Remove the temporal family: `TemporalPatchNet` with `TemporalPlan` (planning by repairing
  input ports toward a goal), `TemporalMemory` (response protection) and the experimental
  `PartitionedTemporalPatchNet` (5 modules, 1,834 lines, the `cadence.experimental` package
  and 7 top-level exports), with 7 test files and their fixtures, the temporal cases of 2
  more, the memory/planning example and the temporal, planning, response-protection and
  partitioned guides. `Brain.compose` never had a transition model and used none of them;
  every composed brain, checkpoint and composed life is byte-identical. Learned transitions,
  bounded planning and response protection live on in release 0.84.3.
- Remove `PatchNet` and the hybrid Newton refinement (`patch.py` and `_refine.py`, 1,047
  lines; the exports `PatchNet`, `PatchObservation` and `RefinementReport`; the `refinement`
  argument of `Equilibrium`, whose `qualified` rows are now always its converged rows), with 4
  test files, the PatchNet cases of 3 more, and the PatchNet and recursive-settlement guides.
  `_copy_state`, which the composed brain borrowed from `patch.py`, moves unchanged to
  `brain.py`; the tests of the `Nudge` anchor stay, in `tests/test_nudge_anchor.py`. Every
  composed brain, checkpoint and composed life is byte-identical. Observer populations on
  `Brain.compose(observers=...)` remain the System 2 path; `PatchNet.recursive` and the hybrid
  solver live on in release 0.84.3.
- Move the protocol and timing instruments out of the installed package into
  `benchmarks/instruments`: `protocol.py` (held-out facts with preconditions, the shuffled
  control, gain selection under a sparsity cap) and `timing.py` (the latency of a decision and
  the machine it ran on), with their tests and their guide, which keeps running in the
  documentation test. `Protocol`, `Row`, `evaluate_predicate`, `select_gain` and `shuffled`
  leave the top-level exports; a source checkout imports them with `benchmarks/instruments` on
  `sys.path`. Remove `cadence.circuits` (437 lines: the region assembler, the bounded
  `Deliberator` search with `imagine`, `ActivityMonitor` and `reflex_arc`) with its 5 test
  files; no composed brain, benchmark or example used it, and it lives on in release 0.84.3.
  Every composed brain, checkpoint and composed life is byte-identical.
- Remove the leftovers of the stream and the actor that no composed brain used: `Echo` and
  `Afterglow` (a `Trace` with other defaults; `Trace(decay=..., source=..., target=...)` is
  what they were), `stateful` (an embedded network plus a context range for a trace), and
  `Bins` with the `population` argument of `ActorCritic` (a continuous action as one softmax
  per dimension), with their tests and API rows. The categorical motor slots keep their
  per-slot softmax, sampling and temperatures; every composed brain, checkpoint and composed
  life is byte-identical. The population code lives on in release 0.84.3.
- Remove the genes that no composed brain read away from their founders. Each name is
  refused at `compose` and `retune` with a message naming 0.84.3 as the release that takes
  it, a checkpoint of 0.84.3 that set one away from its founder is refused on load with the
  same message (the resting bias excepted, below), and one at the founders loads. At the
  founders every composed brain's parameters and every composed life are unchanged: on the
  12-brain digest ledger the arrays and the actions are identical, and the descriptions and
  checkpoints lose the removed keys. The record channel of the arousal
  (`arousal_record_surprise`, a second surprise at the error of the outcome the action's
  record forecast, with `Arousal.usual_record`, `Arousal.records` and
  `last_arousal["record_error"]`) never entered the level at its founder weight of zero; on
  the odour nursery a positive weight woke lives sooner and left more of them searching too
  briefly. `Arousal.to_dict()` writes `cadence-arousal/4` with every gene, and
  `ArousalConfig.space()` names ten genes. The centred dopamine (`actor_dopamine_center`,
  `actor_dopamine_floor`, `actor_center_scale`) and the `Valence` class leave with the
  critic's signal options (`actor_critic_signal`, `actor_critic_normalize`): the critic
  learns from the actor's dopamine, the prediction error capped at `actor_dopamine_cap`, its
  step divided by its trace's energy, the founder resolution of `"auto"`; the night replay
  chamber keeps the readings of its centred arm. The actor's momentum and RMS normalization
  (`actor_momentum`, `actor_normalize`, declined as a founder in 0.73.1) leave with the
  actor's moment arrays; the settling learner keeps its own (`LearnerConfig.momentum` and
  `.normalize`). The teaching law's one-sided contrast and its leak
  (`LearnerConfig.centered` and `.decay`, `learning_centered` and `learning_decay`) leave:
  every lesson contrasts the `+beta` and `-beta` phases. `learning_nudge` leaves `compose`
  and `retune`: a composed brain learns with the cross-entropy nudge, and the quadratic
  nudge stays on a standalone `Learner` for pattern targets. The working trace's focus
  (`working_memory_focus`, `Trace.focus`), a weighting of each neuron by its movement,
  leaves: every trace decays toward its source. The resting bias (`resting_bias`), an
  initial bias of the processing regions, leaves: every bias starts at zero, and the
  acquisition and rhythm chambers whose frozen recipes name it set the same arrays
  themselves, act for act. A checkpoint of 0.84.3 saved with a resting bias loads as it is:
  the biases it set are in the saved arrays, and the recorded value was provenance only. The
  removed genes live on in release 0.84.3.
- One checkpoint format, `cadence-brain/1`: `Brain.save` writes the learner's checkpoint, the
  brain's metadata in the `brain` member (the actor's config under `actor`, the random
  states, the free phase and a pending action, the arousal and the action `live` issued, the
  state a waiting stream sensed, the code, and under `memories` each present memory's
  settings), and one group of arrays per memory: `memories/working_memory/`,
  `memories/efference/`, `memories/history/`, `memories/recall/`, `memories/hippocampus/`
  and `memories/episodic/` (the context store, the stream's context and decision, the replay
  paths); the learned code region is the `learned_code` group. The earlier formats, one
  number per component a brain held, are no longer written; `Brain.load` reads them through
  0.85.x (`cadence-generic/1` to `/9`, which 0.84.3 reads, and `/10`, which development
  builds of the learned code wrote), and the next minor release refuses them with a message
  naming 0.84.3. The recall memory's settings that 0.84.3 always wrote
  (`value`, `context`, `context_decay` at their one value) and its empty context history
  leave the description and the checkpoint. Every array and every metadata value of a
  checkpoint is the one the earlier format held, under its new name (checked on the
  12-brain ledger and on every pinned founder, whose checkpoint and name digests are
  re-recorded); a file of 0.84.3 loads to the same brain. Chambers and verifiers that read
  saved brains read either layout.
- One memory contract, `cadence.memories`: `brain.memories` holds the present memories by
  name, in the order the brain reads them before a settle and writes them after it
  (`working_memory`, `efference`, `history`, `recall`, `hippocampus` and, with
  `memory_choice=True`, `episodic`), and every member keeps one contract (`Memory`):
  `stimulus(drive, moment)` before the settle, `branch()` for an imagined branch,
  `begin(keep=False)` at a reset, `reset(batch, rows=None)` and `keep(rows)` on streams, and
  `to_dict()` and `arrays()` for its checkpoint group `memories/<name>/`. The brain reads its
  memories into the drive, branches them for `imagine`, begins them at `reset` and saves them
  through the list; how each is written after the settle stays its own. The episodic
  memory's vote, temporal context, replay paths, outcome writes and reverse replay leave the
  brain for `EpisodicRecords`, the member `episodic`. `brain.working_memory`,
  `brain.efference`, `brain.history`, `brain.recall`, `brain.hippocampus` and
  `brain.context_memory` name the members, and `memory_choice`, `memory_context`,
  `memory_replay` and `memory_demonstrations` read the episodic member's settings. Every
  array, action, description and checkpoint on the 12-brain digest ledger is unchanged, and
  the benchmark chambers give the same outputs.
- One constructor: `Brain.compose(inputs, actions, ...)` builds every composed brain, and
  `Brain.compose(inputs, actions, connectome=...)` wraps a custom anatomy with the same
  founders, the episodic founders filled in under `memory_choice=True`. The anatomy brings
  its own layout, so `modules`, `observers`, `lateral`, the projection scales, `features`
  and `field` stay at their founders, and its sensory and motor populations hold the inputs
  and the actions. `Genome.composed(inputs, actions, ...)` returns the layout `compose`
  develops, to edit or evolve, and `Brain.observation_width` becomes
  `Genome.observation_width`. `Brain.build` and `Brain.genome`, the single-region layout (a
  sensory region, one association region and the motor region, with an optional
  prefrontal region read at scale 12), leave: `Brain.compose(inputs, actions,
  modules=(hidden,))` is that brain with a working trace read at scale 1.2, and
  `Brain(connectome, ...)` wraps any anatomy as before. Every composed brain is unchanged
  on the 12-brain digest ledger, and the benchmark chambers give the same outputs. The
  single-region layout lives on in release 0.84.3.
- One teaching knob and one map of projection scales. `teaching` at `compose` and `retune`
  sets the teaching learner's steps and their normalization: `"founder"` (steps 0.5 and 0.05
  on the raw contrast, momentum 0.9, the founder) or `"normalized"` (steps 0.003 and 0.0003
  under a running RMS of forgetting 0.99 and floor 1e-4, momentum 0.9: the point for many
  lessons, as in imitation). `learning_eta`, `learning_eta_bias`, `learning_normalize` and
  `learning_normalize_floor` leave the setting names, each refused with a message naming
  `teaching`, and `learning=LearnerConfig(...)` keeps an expert's access to any steps.
  `scales={"sensory": 1.0, "trace": 1.2, "history": 1.2, "recall": 1.2}` at `compose` names
  the initial magnitudes of the projections from the sensory region, the working trace, the
  heard history and the recall memory, a name left out at its founder; `sensory_scale`,
  `working_memory_scale`, `history_scale` and `recall_scale` are refused with a message
  naming their key, and a checkpoint's composition record keeps the scales under their
  earlier names. The founder point and the founder scales are the founders of 0.84.3: every
  composed brain is unchanged on the 12-brain digest ledger, descriptions and checkpoints
  included. The chambers that set a scale or a teaching step pass the same values the new
  way.
- One reset and one description. `reset(*, keep=())` names what a reset carries over:
  `"slots"` keeps the recall memory's slots and `"arousal"` the whole arousal state; other
  names are refused before anything changes. `retune` loses `reset_arousal` and resets
  nothing (`brain.arousal.reset()` resets the arousal), and `fit` keeps the arousal state as
  it is: a control on independent samples does not change a life. `describe()` returns four
  sections: `layout` (the regions; `code`, what the memories key on; `learned_code`;
  `composition`, the construction choices), `genes` (every setting by the name `compose`
  and `retune` take, `teaching` among them, `"custom"` for steps set outside both teaching
  points), `memories` (each member of `brain.memories` with its settings and counts) and
  `pending`; the configuration and memory sections, `observation_code`, `initialization`
  and `pending_feedback` move into them. Arrays, actions and checkpoints are unchanged on
  the 12-brain digest ledger, and every description holds what it held, in the new layout.

## 0.84.3 — 2026-10-19

- A brain saved between `lesson` and its next act loads and continues as the saved brain
  would. The lesson keeps the stream's free state and clears the drive that state was
  settled for, so that the next act settles afresh under the taught parameters; `load`
  restores the free state without a drive, as the lesson left it. Before, `load` refused the
  checkpoint with `missing saved state: actor/_drive`, and so it refused a brain saved after a
  calm `live` answer was taught before its outcome (also during `wait`), or after
  `step(..., teacher=...)` taught and its act refused. `save` writes what it wrote before, so
  such a checkpoint written by an earlier release loads too. A saved drive is still checked
  against the free state, and a sampled action awaiting its outcome still needs the drive it
  was chosen on.
- `Brain.step`, the continuous-interaction guide and the learning guide say where the reward
  of a demonstrated action belongs: the reward given to `step` is credited to the action
  `step` issued, so where a teacher's or caregiver's action ran in its place, replace the
  issued action with `act(observations, greedy=True)`, which learns nothing from it, and
  teach the shown action with `lesson(observations, labels)`; the next `step` then takes no
  reward. Reported to `step`, the shown action's reward, or the zero of an omitted one,
  would credit an action that did not run. A moment in which a teacher acts can also be
  taught with `lesson` followed by a greedy `act`, in place of `step`.
- `Brain.reset`, `retune(reset_arousal=True)`, the API reference and the composition and
  continuous-interaction guides say what these resets leave the arousal of an adult brain
  (its youth over), also after `reset(keep_memory=True)`: no income or error references and,
  at the founder `need=0`, no want, so only a surprise at its own forecasts can rouse it. In
  a two-armed bandit whose paying arm moved, none did: adults reset this way chose the newly
  paying arm at none of the next 120 moments in 5 of 5 seeds, while the same adults living
  on without the reset, reset with their arousal state restored, or given
  `retune(arousal_need=0.5)` before `reset()`, learned it. Where the world may stop paying,
  restore the arousal state after the reset, give the brain a positive `need`, or let it
  keep living without the reset.

## 0.84.2 — 2026-10-18

- A calm `live` moment that ends the episode keeps its forecast's read of the recall memory
  through the end it commits. The first act of the new episode, sampled by a brain the
  outcome woke, keeps the read's address eligibility under `recall_address_reward` and its
  next outcome steps the address map; a greedy act keeps no eligibility; for both,
  `brain.recall.attended` describes the read. Before, the read was cleared with the
  forecast's restore and the sampled act kept an eligibility of zero. A refused outcome
  leaves the read cleared. A brain without the recall memory, or whose address map does not
  learn by reward, acts, learns and saves as in 0.84.1.

## 0.84.1 — 2026-10-18

- A lesson under `learning_surprise` judges its rows by the act the brain would issue, the
  readout with the episodic vote, so a demonstration corrects a brain whose memory turns a
  right cortical answer into a wrong act, and an answer the memory makes right is no surprise
  and gets no lesson.
- `live` decides whether an issued action was the brain's own best guess from the act it would
  issue greedily, read when the action is issued, so a greedy act the episodic vote decided is
  owned and its outcome can surprise the brain.
- An imagined branch issues privately the act the brain would issue, with the episodic vote
  read under the branch's own temporal context, to its efference copy and the recall memory's
  predecessor key.
- A decision whose outcome was a reward or a punishment stays on the replay path when the next
  decision is taken in the same situation, so the next reward replays only the decisions taken
  since that outcome.

A brain without `memory_choice` acts, learns and saves as in 0.84.0.

## 0.84.0 — 2026-10-18

- Memories key on the observation code: `Brain.observation_code(x)` is what the episodic
  memory (place and context records), the hippocampus of earlier releases and the recall
  memory key on. It is the observation itself, or with `code=units` (opt-in) the brain's
  own sparse code of it: `units` code neurons with positive weights that read each input
  through two rectified channels, ON `max(x, 0)` and OFF `max(-x, 0)`, and every channel at
  least twice (each channel is dealt two readers first, and each unit's fan-in of 32
  channels, more where 32 cannot read every channel twice, is filled at random from the
  seed after), the largest sums of one unit in ten kept (102 of 1,024; too sparse a code
  leaves the cortex and the memory too little overlap between a question and the moment it
  asks for), the code scaled to unit length; a `PatternSeparator` with a fan-in and
  `cover=2`, owned by the brain (`brain.sparse_code`, `code_width`, `inputs`) and saved with
  it. A single input of either sign always reaches the code, a signed observation reaches
  the code (and the cortex, when the sensory region is the code) in both signs, and a
  nonnegative observation leaves the OFF channels silent; channels read by the same units
  are told apart by their weights. The code's seed is drawn from the brain's seed on a
  stream of its own, `SeedSequence([seed, 7103])`; the earlier `seed + 2` was the stream a
  brain of seed `seed + 2` develops and acts from, so the code of one founder and the
  connectome of another were drawn from the same stream. A coded brain's episodic stores
  take no separator of their own and refuse `memory_expansion` and `memory_fan_in`. A sense
  organ's own code (a retina, a cochlea, a V1 region) belongs to the application or the
  anatomy, not to the library. `describe()` reports `observation_code`; a brain with a code
  saves as `cadence-generic/9` with the code's settings, seed and tables, which `load`
  installs as saved; `load` checks the records' key widths against the code's and refuses a
  mismatch naming both widths, as it refuses place and context records keyed on other
  coordinates than the brain's, bounds a memory's key coordinates by the code's width (which
  can exceed the graph's neurons) and its outputs by the graph, and refuses code tables that
  do not read every channel twice with positive weights, and a code saved before the
  covering construction, whose metadata has no `cover`, naming that cause. Every brain
  without `code` is byte-identical at 4, 256 and 2,000 inputs and on an image shape, the
  checkpoint included at 4 and 2,000 inputs and on the image
  ([the observation code](docs/memory.md#the-observation-code)).
- `Brain.compose(inputs, actions, code=units)` makes the sensory region the code: every
  observation is coded before the drive, so 12,824 inputs reach the cortex as 1,024 units
  through 1,024 x 64 sensory synapses in place of 12,824 x 64, and the memories key on the
  same code; the heard history then holds blocks of the code. The founder leaves the cortex
  and the memories on the observation. A custom anatomy names a coded sensory region with
  `inputs=` beside `code=`; `Brain.observation_width(inputs, code)` sizes its recall regions.
- `Brain.compose` accepts image shapes like `Brain.genome`: `inputs=(height, width)` or
  `(height, width, channels)` makes the sensory region a `visual_cortex` (`features`,
  `field`), whose feature maps project to the first module at `sensory_scale`; everything
  else is as for a vector length, and `compose(inputs=int)` is unchanged.
- The recommended settings for wide observations, with the founder unchanged at every
  width: `memory_fan_in=32` above 256 key values (the episodic separators 0.63 MB each in
  place of 105 MB at 12,824 inputs) and `code=1024` above 1,024 inputs, where the two
  episodic separators and the recall address map go from 2 x 105 MB and 1,322 MB on the
  observation to none and 8.9 MB on the code, the code's tables 0.84 MB
  ([the observation code](docs/memory.md#the-observation-code)).
- The episodic and sequential-task guides state the founder dependence of the continuation
  after teaching: the readings in which `step` eroded a taught chain where `live` kept it
  belong to the five fresh founders of the chain instrument, and on other founders both paths
  kept it or the drift came through `live`; the remedy is to read the greedy policy stage by
  stage beside whole-chain success after the caregiver leaves, and continue through `live` or
  lower the actor step. No runtime change.
- Competitor arms for the comparison thread of
  [issue 183](https://github.com/muellerberndt/cadence/issues/183) and
  [issue 184](https://github.com/muellerberndt/cadence/issues/184):
  `benchmarks/competitors.py` holds an online recurrent actor-critic in NumPy with an Elman
  cell (`elman`, the key-door chamber's `recurrent` arm) or a gated recurrent unit (`gru`,
  its gradients written by hand and checked against finite differences), one update per
  moment, TD(lambda) eligibility traces with a gradient guard, no replay or batching, and
  counted work (parameter updates, forward passes, multiply-accumulates, guard activations,
  parameter count). The prerequisite-chain nursery runs both as arms in its observed and
  hidden modes, with their settings declared in its `protocol.json`; the odour nursery runs
  them on request (`--arms elman gru`) with settings declared in the chamber, outside its
  frozen plan, and records them in the receipt. The existing arms, receipts, gates and
  tests are unchanged.
- Lessons only where the answer differs, opt-in: `LearnerConfig(surprise=True)`
  (`Brain.compose(..., learning_surprise=True)`) teaches only the rows whose free answer
  differs from the label; agreeing rows get no nudged phase and no update, and a batch the
  net answers as shown ends after the free phase. The composed brain judges that difference
  from the settle its act reads (`step(teacher=...)` passes it as `answer`), a learner on its
  own from its free phase. The teaching report counts
  `accepted_presentations` and `agreed_presentations`, `LearnedState.taught` marks the rows,
  and an episodic memory that records demonstrations records the taught rows only. The
  composed founder keeps every-row lessons; the
  [learning guide](docs/learning.md#lessons-only-where-the-answer-differs) says when to
  select the rule.
- The heard history, opt-in ([issue 199](https://github.com/muellerberndt/cadence/issues/199)):
  `Brain.compose(..., history_decays=(0.2, 0.8, 0.95))` adds a `history` region of one block
  of sensory width per decay, each block a fading trace of the observations as heard (the
  values the sensory neurons were stimulated with, before settling), held at
  `history_amplitude` (3.0) and read by the association region through a plastic projection
  that starts like the working trace's (`history_scale`, 1.2). After each act the blocks take
  in the observation the brain acted on, so at a decision the history holds the past and the
  drive the present. `History` keeps it per stream; `reset`, the end of an episode,
  imagination, `wait`, `describe()` and `retune(history_amplitude=...)` treat it as they
  treat the working trace, and a brain with one saves as `cadence-generic/7` and continues
  exactly. The founder `history_decays=()` builds the brain without it, byte-identical; the
  [context guide](docs/memory.md#hear-the-past-at-several-timescales) says when to add it.
- The recall memory, opt-in ([issue 199](https://github.com/muellerberndt/cadence/issues/199),
  [issue 201](https://github.com/muellerberndt/cadence/issues/201)): `Brain.compose(...,
  recall=True)` adds an index of the stream's own recent moments, a `recall` and a `next`
  region read by the association region through plastic projections that start like the
  working trace's (`recall_scale`, 1.2). Each moment the brain acts on, and each moment
  `wait` senses, is written as one slot per stream (`Recall`): its key the event heard
  (`recall_heard` 1.0 times the observation code),
  its value the event (`recall_value="event"`) or the settled association activity, and the
  slot before learns it as its next. Before each settle the moment's address, the key plus a
  learned map of the event and of the association activity of the moment before, weighs the
  slots by a softmax (`recall_temperature` 0.25, `recall_recency` 0.25 per moment of age), and
  the recalled value and next drive the two regions at `recall_amplitude` (3.0), a drive held
  through the settle, which answers on it and does not revise the choice: a held-input recall,
  whose last attention `Recall.attended` keeps. After each teacher lesson the address lesson,
  a rule of its own beside the graph's contrast, scores each slot's value and next against
  the association's movement under the nudge, read back through the recall and next
  projections, weighs the scores with the address's own attention
  (`recall_address_temperature` 0.1), and takes a least-mean-squares step of the map toward
  the keys so weighted at the taught rows (`recall_address` 0.01, each row's rate bounded by
  `1 / |z|^2` on its inputs `z = [x, c, 1]`; a step that is not finite is refused and
  reported as `address_refused`); rewards leave the map as it is. `recall_capacity` (64)
  bounds the slots per stream. `Brain.lesson(observations, labels)` gives the lesson of
  `step(..., teacher=...)` without its act. `reset`, ended episodes, imagination (reads, no
  writes), `wait`, `describe()` and `retune` of the read and lesson genes treat the memory as
  they treat the heard history; the address map is a learned parameter that resets keep and
  `parameters()` counts, and a brain with the memory saves as `cadence-generic/8` and
  continues exactly. The founder `recall=False` builds the brain without it, byte-identical;
  the [memory guide](docs/recall.md) explains it, with its
  storage at camera width (14.5 MB with a code of 1,024 neurons, cortex values and an
  association region of 512 neurons, against 105 MB for one dense separator). The option
  reproduces the binding instrument
  `benchmarks/binding/index_read.py` moment for moment (`tests/test_recall_memory.py`).
- Options for the recall memory, each a gene beside `recall=True` that `retune` can change,
  described and saved only where it differs from its founder, so every founder brain and its
  checkpoint are as before ([memory guide](docs/recall.md)).
  `recall_address_step="toward"` steps the address by the named keys less the mixture it
  attended to, in place of the founder `"lms"` step toward the named keys: to first order
  the named slots' logits rise against the attended ones, whatever the keys' norms.
  `recall_recency=None` charges a slot's age `r * log(1 + age)`, `r = heard^2 / temperature /
  log(1 + capacity)` (`Recall.charges()`): an unmatched old slot's weight falls as a power of
  its age, and a full match of a unit-length key outweighs the age of every slot the capacity
  holds; a number keeps the linear charge. `recall_write="aroused"` writes a moment only when
  its act was sampled, and a moment `wait` senses while a sampled action awaits its outcome;
  in a batch a moment written for some streams is held by those streams alone, and each
  stream's latest held slot learns its own next. A number `recall_null` adds a null slot of
  logit `null / temperature` that reads nothing, so the slots' shares sum to one less its
  share; the address lesson weighs the slots alone. `recall_address_reward` above zero lets
  reward train the address map: each sampled act keeps, beside the synapses' eligibility, the
  step the address lesson would take were the act's action a lesson's label
  (`Recall.contrast_rows`), `ActorCritic` keeps it in a trace of its own (`trace_memory`,
  decaying and forgotten as the synapses' traces are) and steps the map by that rate times
  the dopamine times the trace, checked before the graph's update and installed after it
  (`address_reward_step` in the report); a checkpoint carries the trace and a pending action's
  contrast. The binding instrument's index memory takes the same genes (`address_step`,
  `recency=None`, `write` with `IndexBrain.aroused_rows`, and `null`).
- Two more options for the recall memory, described and saved only where they differ from
  their founders ([memory guide](docs/recall.md)).
  `recall_episodes="keep"` keeps a stream's slots and the address map when its episode ends
  (`done` through `step`, `learn` or `live`, and a calm `live` moment that ends it), where
  the founder `"forget"` empties them; the working trace, efference copy and heard history
  begin afresh either way, the latest slot of the ended episode learns no successor from the
  next episode's first moment (`Recall.end`), and ages go on counting the store's columns.
  `Brain.reset(keep_memory=True)` keeps the recall slots across a reset in the same way; a
  plain `reset()` empties them as before. `recall_key="predecessor"` (construction) adds the
  stream's previous moment to every key: `[heard * x_t, heard * x_(t-1), heard * c_(t-1)]`,
  the previous moment's event and the one-hot of the command the stream issued last (passed
  by the brain at each act, so no efference copy is needed; a moment `wait` senses keeps the
  last command), zeros at a stream's first moment and after an episode ends. The key, the
  address, the map's outputs and a reward eligibility hold `2 * code + actions` values in
  place of `code` (`Recall.address_width`); the values and nexts stay the events. A
  checkpoint carries the ended marks and the previous events and commands where they apply.
- The recall memory's window key, opt-in: `recall_window` k above 1 (construction, with
  `recall_window_decay` d in (0, 1], founders 1 and 1.0) makes every key an ordered window
  of the stream's last k events, `[heard * x_t, d * heard * x_(t-1), ..., d^(k-1) * heard *
  x_(t-k+1)]`, each position on coordinates of its own, so that where an event recurs with
  different continuations a moment finds the occurrences whose recent history matches its
  own. Every moment the stream hears enters the window (acts and moments `wait` senses,
  written or not); it holds zeros before a stream's first moment and after an episode ends,
  under either `recall_episodes`. `recall_context` adds its sum on the present event's block
  as before, and the predecessor key's blocks follow the window's. The keys, the address,
  the map's outputs and a reward eligibility take `k` times the code's width (`Recall.past`
  holds the previous events); values and nexts stay the events. A checkpoint carries the
  window's events and refuses a window of another length. The binding instrument's index
  memory takes the same genes, `window` and `window_decay`.
- A sparse pattern separator, opt-in: `PatternSeparator(..., fan_in=k)` lets each code
  neuron read `k` distinct inputs chosen at random from its seed, with random weights, held
  as `index` and `weight` arrays of shape `(expansion, k)` in place of the dense
  `projection`; the rectification and the winners are as before, so storage and work follow
  the expansion and the fan-in at any input width. `Brain.compose(..., memory_choice=True,
  memory_fan_in=16)` gives both episodic stores such a separator (1,024 x 16 indices and
  weights each); `memory_fan_in=0` keeps the dense one, byte-identical with its
  checkpoint, and a sparse separator saves its arrays and its fan-in. With `cover=c` as well
  each input is read through an ON and an OFF channel and every channel by `c` code neurons
  at least, the readers dealt first and the weights positive (the brain's code); the
  episodic separators keep the independent draw over the inputs.
- One holder of the current graph for several learners: `Learner.brain` reads the graph from
  the learner's `holder`, a `GraphHolder`, and an update replaces the holder's graph in one
  assignment after every check, so a refused update changes nothing. `Learner.sharing(first,
  outputs, ...)` builds a second learner on the first one's holder: each one's updates reach
  the other, where a learner built from `first.brain` keeps the graph it was given, as before.
  The composed brain, the actor-critic and checkpoints (which hold the graph, not the sharing)
  read the graph through it; a shallow `copy.copy` of a learner shares its holder, and
  `dataclasses.replace` builds a learner with a holder of its own. The
  [learning guide](docs/learning.md#two-learners-on-one-graph) shows the use.

## 0.83.0 — 2026-10-13

- An episodic memory for the composed brain, opt-in: `Brain.compose(..., memory_choice=True)`
  remembers what happened in a situation at once and lets that vote in the next choice there,
  and learns chains of actions from rewards without the world marking the end of an attempt;
  the [episodic memory guide](docs/episodic.md) explains it. Each record is kept twice, keyed
  by the observation (place) and by the temporal context, the observation with a fading trace
  of what the stream saw and did before it (`memory_context` 0.5); the mean of the two reads
  is added to the motor readout before the softmax, greedy or sampled. A record is written
  when an outcome is known (a reward or a punishment, the end of an attempt, a surprise that
  woke a calm `live` brain, or an outcome the caller marks with a positive `salience`), per
  stream; a zero outcome in the middle of an attempt leaves it. A write corrects the
  stream's record halfway at once (`memory_rate` 0.5), fading by `memory_decay` per later
  write, and consolidates into the record every stream shares at `consolidation`. A
  caregiver's demonstration is recorded at `memory_demonstrations` (1.0). A reward is replayed
  backwards along the stream's recent path, one decision per situation, back to the last
  known outcome, discounted by the actor's `gamma` (`memory_replay` 8): a reward that ends an
  attempt whole, into both kinds of record; a reward in a continuing life in proportion to
  its surprise, into the context records, and into the place records of the decisions that
  were explorations rather than the brain's best guess. Keys are separated
  (`memory_expansion` 1,024, `memory_winners` 24). `ActorCritic.act` and `probabilities`
  take a `bias` added to the readout before the softmax; eligibility still scores the
  readout's own policy, so reward consolidates what the memory chose.
  On fresh founders and seeds against the composition without it: a four-step crafting chain
  corrected by a caregiver 38.1% -> 98.7% success, and 9.0% -> 97.8% after 80 episodes alone
  through `live`; the same chain from reward alone 9.1% -> 73.2%; a second crafting chain
  2.8% -> 99.0% after the caregiver leaves, 15.0% -> 75.6% from reward alone; a four-step chain
  in one continuing life paid only at its completion, without `done`, 0 -> 108 greedy
  completions per thousand moments; reversals in a continuing life (the odour nursery) pass
  every gate in 20 of 20 lives; delayed response 6 of 6; two songs cued once 0.52 -> 0.65 of
  their notes after ten lessons; taught recall of a vanished cue 0.70 -> 0.75 over its four
  hardest conditions, unevenly. A continuing bandit whose history does not matter falls from
  0.86 to 0.75, and it is not the default. `Brain.compose` without `memory_choice` builds the
  0.82.0 composition unchanged.
  Compatibility: a brain with the episodic memory saves checkpoint format
  `cadence-generic/6`, which earlier releases refuse; every other checkpoint keeps its format.
- The sequential-task guide adds the measured continuation after teaching: a brain left alone
  through `live` keeps a taught chain where `step` erodes it; lowering the step stays the remedy
  where surprises are mostly punishments. No runtime change.

## 0.82.0 — 2026-10-12

- The composed brain's context starts quiet and is learned. `Brain.compose` gains the
  construction setting `working_memory_scale`, the initial scale of the plastic projection
  through which the prefrontal region feeds the carried interpretation back to the
  association cortex: founder 1.2, where earlier releases fixed 12. The trace keeps
  amplitude 3.0, so the prefrontal region holds the copy as clearly as before, and its
  retention default (`working_memory_decay`, also the constructor's) is now 0: the trace
  carries exactly the last interpretation, and older moments reach the present through the
  interpretations they shaped. A newborn reads its context about as strongly as amplitude 0.3 at the
  old scale did, the operating point of the chain, competing, key-door, recall and odour
  chambers, while every update of that projection scales with the ten times clearer copy, so
  learning raises the past's influence where it predicts the outcome. With the earlier
  composition an untrained brain completed a fully observed four-step chain less often than
  uniform random, and on a continuing contextual bandit whose history does not matter the
  hit rate stayed at chance; the quiet amplitude alone lost the delayed response learned
  from reward. On development seeds the composed brain now reaches a mean hit rate of 0.60
  on that bandit (seeds 0 to 5, two above 0.9; the earlier composition 0.25, none) and still
  learns the delayed response of `tests/test_generic.py` from reward in 6 of 6 seeds (8 to
  13; amplitude 0.3 at scale 12 learned 3). `Brain.build` keeps its
  `memory_scale` of 12. The new [context guide](docs/memory.md#working-trace) explains what a brain
  carries, how to wire it for retention and which settings to change.
  Compatibility: a composed checkpoint records `working_memory_scale` in its composition;
  one saved before 0.82.0 loads as 12 with its own synapses and settings, and releases
  before 0.82.0 refuse 0.82.0 composed checkpoints as invalid composition provenance.
  Frozen chamber protocols build the composition they were frozen under (scale 12, and
  retention 0.2 where the key-door and odour protocols leave it unnamed); the night-replay
  chamber's `defaults` setting follows the current composition. Benchmark READMEs that
  name the composed default describe the composition of their own release.

- Remove the readout's intrinsic plasticity of 0.81.0 (`LearnerConfig.homeostasis_rate`
  and `homeostasis_target`, `learning_homeostasis_*` in `compose` and `retune`, and
  `homeostasis_step` in learning reports). A set point per action neuron constrains how
  often each action is chosen; under reward it flattened a six-action policy, and the
  prerequisite-chain nursery drops its homeostasis arm. Checkpoints written with the
  founder rate 0 load unchanged; one written with a positive rate (`cadence-checkpoint/3`)
  is refused with a message naming 0.81 as the release that reads it.

- Clarify continuing-life state, model-specific agreement and residuals, and the
  distinction between solver updates, environment time and learning events.
  Make current architectural guidance self-contained. No runtime or default change.

- Document existing results on self-stabilization, asynchronous contraction,
  moving equilibria and sheaf diffusion, with explicit applicability limits and
  consumers in context, retention and measured-work research. Clarify that the
  sufficient contraction certificate is stronger than state qualification.
  No runtime, default or behavioral claim changes.

- Extend the settling guide with nonlinear sheaf coordination and bounded-delay
  diffusion results, their fixed-map assumptions, and source checks needed before
  reusing a scheduler or quantitative bound. No runtime change.

- Normalize machine-specific storage prefixes in benchmark publication copies,
  preserving original fingerprints, measurements and historical outcomes in an
  explicit publication manifest; verify copied bytes and receipt envelope digests.

- Make mathematical and historical-experiment guidance self-contained; distinguish
  equation qualification from behavioral and physics claims. Trace reference
  equations are stated directly alongside their checks. Correct sequential-task
  guidance to reflect the unsuccessful backward-lesson comparison and the bounded
  scope of recall and critic-rate results. No runtime change.

- The source distribution leaves out receipts no test reads (the competing-skill and chain receipts, the recall receipts except `finite-3`, and the rhythm `native-history-*` receipts); they stay in the repository and in the GitHub release assets: 14.7 MB against 54.8 MB for 0.81.0, of which 35 MB were the competing-skill and recall receipts, against PyPI's 100 MB file limit. The README's CI badge reads the `main` branch only, so a failed run on another branch no longer shows as failing. No runtime change.

## 0.81.0 — 2026-10-11

- Add `Brain.wait` for an outcome that arrives after the stream has sensed more: an
  awaited action's custody for one `live` stream
  ([#111](https://github.com/muellerberndt/cadence/issues/111),
  [#122](https://github.com/muellerberndt/cadence/issues/122)). A waiting stream settles
  each observation and advances its working trace without taking an outcome; the
  awaited action keeps its forecasts, eligibility and situation, so the outcome later
  given to `live` is credited as an immediate outcome would be, while the next state
  settles from where the stream is. Wait settles are reported by `last_settlement`,
  outside arousal's counts. Add `Brain.decision_id`, the arousal age once `live` issued
  the awaited action, and `live(..., decision_id=...)`, which refuses an outcome
  reported under another identity before any change. A stream saved while waiting uses
  checkpoint format `cadence-generic/5`; lives that do not wait keep their formats and
  contents. `ActorCritic.learn` takes the state to settle the next state from (`warm`).
  The guide adds that the caller supplies a late outcome's link to its action, so a
  delay whose credit the brain is to learn goes through `live`; that a wait follows
  event time ([#116](https://github.com/muellerberndt/cadence/issues/116)); and how to
  total a waiting stream's work.
  Defaults, learning laws and existing calls are unchanged; no behavioral gain is claimed.
- Add the readout's intrinsic plasticity, two genes of `LearnerConfig`:
  `homeostasis_rate` (founder 0) and `homeostasis_target` (0.3). At every teaching or
  reward update, on the host and on the device, each output neuron's bias moves by the rate
  times the shortfall of its free activation below the target, so a readout neither
  saturates, where a finite nudge has no slope, nor falls silent, where every contrast is
  tiny. Reachable as `learning_homeostasis_rate` and `learning_homeostasis_target` in
  `compose` and `retune`, reported by `describe()`, saved with the learner configuration;
  `apply` reports `homeostasis_step`. The founder rate leaves every composed brain's
  behavior and continuation as they were; learning reports gain `homeostasis_step` (0 at the
  founder) and saved configurations the two fields, which earlier releases ignore. A learner
  with a positive rate is saved as `cadence-checkpoint/3`, which earlier releases refuse
  instead of loading it without the gene. Padding rows that `ActorCritic.learn(...,
  observed=...)` excludes do not set the readout's operating point, and a frozen output bias
  takes and reports no step. Demonstrated limitation: the recall
  chamber's readout saturates under continued teaching in two of five development founders
  and a third drifts to chance ([#84](https://github.com/muellerberndt/cadence/issues/84));
  with the rate at 0.1 and raw local steps, all four development founders tested end 64
  repeats at clean recall of 0.95 or better (`benchmarks/recall/credit_landing.py`
  receipts). The fresh confirmation recall/4 (`benchmarks/recall/protocol-finite-4.json`,
  founders 310 to 312) failed closure: two founders recall the cue at every delay of the
  prefix and one reads the latest of two cues, while the external-history control ends at
  chance in two founders under the raw rate and the paired and nuisance floors are missed;
  on top of the normalized recipe the gene does not hold the readout against the fixed-size
  pushes in three of four development founders. With the chamber's `surprise` rule, a lesson only on the
  rows answered wrong, the control reads 1.00 in every development founder and the
  vanished-cue recall comes and goes with more lessons (development receipts). The fresh
  confirmation recall/5 (`protocol-finite-5.json`, founders 313 to 315: the surprise rule,
  the gene, raw steps at eta 0.2 on a brain composed at sensory scale 2, selected by a scale
  by rate sweep on the development founders) failed closure as well, horizons none, 0 and
  0, with replacement and order failing in every founder: the measured limit of a trace
  written after every event with one fixed weight (`benchmarks/recall/README.md`). The
  chamber's `make_brain` takes `brain.sensory_scale` (founder 1). In the odour nursery
  (development founders 0 to 5, `benchmarks/reversal` receipts of 2026-10-11) the gene at
  0.02 keeps acquisition, reversal and return from an exposure of 300 trials and impairs
  reversal after 100: one life of six does not reverse and three of six keep a wrong greedy
  choice. No default changes; #84 stays open.
- Add the competing-skill ring, `benchmarks/competing`, a bounded table-world instrument for
  [#169](https://github.com/muellerberndt/cadence/issues/169): a policy acquired by the
  arena's founder brain in a nursery, then a ring that brings a competing skill, unavoidable
  burn, both or neither, with the issue's controls applied through `retune` on one saved
  nursery brain per seed, readings from saved copies with the working trace carried and
  reset, per-projection drift and per-synapse step consistency. Development readings on six
  seeds: the acquired skill is not lost at the gene step for a certain or a weak payoff; the
  smaller steps retain no more and hold the competing skill less; for a weak payoff a frozen
  greedy test loses answers through the context its trace carries. The ring may pay as the
  arena's melee does (`--ring-pay`, a signed contingency of its own; `--brain FIELD=VALUE`
  overrides the brain point; the ledger tallies the dopamine per skill): under that pay the
  loss of #169 reproduces at every actor step while the actor's efficacies barely move, the
  dopamine on the old skill's moments is zero-mean, and what the melee overwrites is the
  episodic memory's one-shot record of the chosen action's reward (`memory_rate` 1.0);
  without the memory the nursery acquires little, and with the memory writing an average
  (`memory_rate` 0.2 or 0.05) the loss halves. A slow set point under every parameter and a
  sign-consistency stiffness, both screened on unshipped library variants, do not hold the
  skill and were removed. No gate, default or mechanism changes.
- Add `docs/sequential-tasks.md`, what the repository's measurements establish about a chain
  of actions that collapses onto one action and the pattern they support (the chain's state
  in the observation, grading over labels, backward one-step lessons, the smallest passing
  budget, a smaller actor step after acquisition, the sampled policy read beside the greedy
  one, the critic's rate for delayed pay), and `benchmarks/chain`, the prerequisite-chain
  nursery ([#111](https://github.com/muellerberndt/cadence/issues/111),
  [#113](https://github.com/muellerberndt/cadence/issues/113),
  [#84](https://github.com/muellerberndt/cadence/issues/84)): a chain of four actions with its
  state observed or hidden, pay at the end, shaped or in backward lessons, the arena founder
  brain at its gene step, a tenth of it, tapered and with the readout gene, against uniform
  random, a frozen newborn, the stationary memoryless ceiling and a tabular learner with the
  same information; readings from saved copies every two thousand moments. Development
  receipt on six seeds: with the state observed and pay at every step the founder brain
  completes the chain at 204 per thousand moments against the tabular learner's 250; with
  pay at the end its greedy policy reads uniform random's rate (42) while its sampled
  policy reads 77; with the state hidden no arm beats the stationary ceiling; the readout's
  intrinsic plasticity at its development point lowers every reading under reward, so the
  gene as declared is a teaching-time mechanism. No gate, default or mechanism changes.
- Add `benchmarks/recall/credit_landing.py`, development diagnostics for
  [#84](https://github.com/muellerberndt/cadence/issues/84) on the finite recall chamber's
  recipe: what an untrained founder's trace carries, and where each lesson's credit lands.
  Every founder tried reaches clean recall of 0.95 or better on a saved copy at some
  evaluation; in two of five the motor readout then saturates at full activation, where the
  finite nudge moves nothing and the efficacy steps fall a hundredfold, and a third drifts
  to chance: the founder dependence is consistent with a readout operating point. Chamber-
  level probes of a fixed and a homeostatic motor bias are diagnostics, not library
  mechanisms; the library gene above is the mechanism they motivated.

## 0.80.0 — 2026-10-09

- Simplify the continuing-brain interface for
  [#170](https://github.com/muellerberndt/cadence/issues/170) and
  [#176](https://github.com/muellerberndt/cadence/issues/176): `arousal=True`
  selects the founder controller, `sensory_scale` exposes the initial sensory
  projection, and `learning_*`, `actor_*`, `arousal_*` plus named memory settings
  override individual genes without replacing unrelated configuration values.
  Preserve all numerical founders and the composition when no options are passed.
- Add atomic `Brain.retune(...)` for changes within an acquired life, preserving
  parameters, optimizer history, traces, records, randomness and pending
  feedback. A reset of arousal is explicit; topology and initialization remain
  construction choices. Refuse a nudge-scale change while sampled feedback is
  pending. Add read-only `pending_feedback` for outcome-safe retries and
  JSON-safe `describe()` for layout, effective settings and saved initialization
  provenance. These are interface changes, with no new learning law.
- Normalize NumPy integer and floating configuration scalars so arithmetic stays
  consistent across save/load, and serialize scalar checkpoint metadata. Validate
  the boolean `centered` teaching setting, including when loading a checkpoint.
  Preserve accepted settings and continuation while rejecting malformed values.
- Lead guides with `compose` and `live`, centralize composed defaults and rate
  ownership, and use public overrides for expert recipes. Address
  [#171](https://github.com/muellerberndt/cadence/issues/171) through the documented
  founder/control distinction: the current efference default is zero and the
  actor rate is 1.0; historical arena settings do not define those defaults.
  Keep arena, C64, Atari and Amen model/experience choices distinct. Smaller arena
  rates remain bounded application measurements, not general default improvements.

- Add the native sensory-history rhythm instrument with matched controls, continuation
  checks and source-bound receipts. The period-four component passes; the C64 transfer
  remains a failure under [#140](https://github.com/muellerberndt/cadence/issues/140).
  Runtime laws and composed defaults are unchanged.
- Clarify that arousal gates learning without guaranteeing retention of earlier skills.
  Keep the bounded arena measurements and candidate mechanisms in
  [#169](https://github.com/muellerberndt/cadence/issues/169), with smaller actor steps as a
  measured control. Distinguish learner checkpoints from a complete composed life and
  require workload parity checks when changing machine, backend or precision.
- Add `MANIFESTO.md`, the design points in one page, with the dedicated language goal:
  whole consistent sentences formulated from the settled state, tracked in
  [#173](https://github.com/muellerberndt/cadence/issues/173).
  Include it in source distributions and the documentation link checks.
- Require every change to meet the manifesto through `CONTRIBUTING.md`, agent
  guidance and the PR template. Keep design requirements and review checks in sync,
  distinguish goals from implemented capabilities, and check contribution-guide links.

## 0.79.0 — 2026-10-09

- Apply extra arousal heat to one uniformly chosen motor slot per moment
  ([#159](https://github.com/muellerberndt/cadence/issues/159)); other slots sample
  the learned base policy. Preserve the one-slot sampling law and existing genes.
  Report actual per-slot temperatures and the heated slot, and preserve random state
  across refused answers and saved continuation.
- Credit the policy that actually sampled a reward-bearing action
  ([#160](https://github.com/muellerberndt/cadence/issues/160)). Scalar or
  per-slot behavior temperatures now reach the local actor nudges, with one common
  score scale and bounded gains. Preserve the base-temperature update, actual-action
  custody and pending saved continuation; explicit hot exploration no longer
  differentiates a different, sharper policy.
- Fix income freezing during exploration in multi-slot `Brain.live` lives
  ([#158](https://github.com/muellerberndt/cadence/issues/158)). Every actual reward
  updates recent and long-run income; surprise and its usual error remain limited
  to the brain's own greedy choices. Keep configuration defaults and action credit unchanged.
  Save the separate income count and migrate earlier checkpoints without resetting
  their acquired state.
- Retain bounded arena and nursery preservation evidence, including failed
  comparisons. Five nursery founder pairs retain their original gates, with slower
  median reacquisition. In one 80,000-moment arena pair both brains retain
  observation-dependent behavior; the corrected policy has weaker closing and
  combat scores than the earlier runtime. Continuation of six acquired fighters
  also has weaker closing and damage. These checks establish neither general
  combat improvement nor recovery of every previously acquired policy. Publish
  the receipts, source identities and limitations with the mechanisms they test.
- Preserve the completed nine-life arena confirmation: seven lives improve
  progress against eight required, seven pass the attack requirement, and six
  pass both. All nine preserve actual-outcome custody and exact saved continuation.
  The original physics and failed readiness result remain identified in the
  [confirmation archive](benchmarks/arena_confirmation/README.md).
- Retain the later [memory](benchmarks/arena_memory/README.md) and
  [reward-phase](benchmarks/arena_phase_budget/README.md) comparisons, including
  their losses of acquired attack performance. Those experimental changes are
  excluded from this release. Longer arena training remains an experiment;
  the fixes do not establish reliable fighting across founders.

## 0.78.0 — 2026-10-08

- Add the period-four loop chamber for #140, retaining the declared protocol and
  original receipts. Its passing `copy` arm uses an external sensory-history
  adapter; the native own-command and trace-only controls fail in the original
  freeze. Label selective teaching as an application policy. Require every prime,
  matched untaught founders, complete controls, no refusals and saved continuation;
  independently verify raw answers, scores and work. This bounded adapter result
  does not close native closed-loop generation. A separate trace-only development
  comparison finds learned prime-dependent playback on one founder at two existing
  trace settings; the simplest setting passes only 1/6 development founders after
  a predeclared extension, so fresh confirmation is not attempted.
- Correct the finite recall worker to honor its declared lesson count. Historical
  freezes 2 and 3 actually ran 192 episodes per founder, not the declared 384;
  preserve those authentic receipts as protocol deviations. Add strict census,
  score and work checks and retain the original `every` teaching behavior. Report
  corrected measurements separately. Full 384-episode correction audits complete
  on all six reused founders: protocol 2 horizons are none/0/0, protocol 3 has no
  passing prefix, and every founder fails nuisance acceptance. Both receipts verify;
  #84 remains open. Library behavior, interfaces and defaults are unchanged.
- Retain and replay raw recall-trace transitions, measure separation and motor
  margins, compare equal-time histories with different event counts, and enforce
  founder time limits while preserving unfinished work. Losslessly pack the evidence
  after retaining an oversized incomplete attempt; all corrected audits fit the
  unchanged 160-MiB cap. Add the missing shuffled-time
  and untaught controls to the native period-two rhythm instrument, with explicit
  actuator timing, recovery and continuation bounds declared before confirmation.
  Charge saved-copy work and remove a redundant, previously uncounted replay.
  Complete the missing paced recurrent-control comparison: all 16 cases meet
  the unchanged bounds, while the brain's two development attempts still fail
  physical timing. Verify rounded display fields at their declared precision;
  all acceptance bounds continue to use raw timestamps.
- Add key-door/3's third, returning rule, longer corridor, variable delays and
  recurrent control for [#111](https://github.com/muellerberndt/cadence/issues/111).
  The original 300 lives failed the declared acceptance gate. Return performance
  includes renewed learning, so it measures reacquisition rather than preservation
  without learning. The pouch-blind arm is a development lead, not evidence that
  the working trace retained the key. Original protocols and receipts remain intact.
- Correct the recurrent control's feedback/gradient ordering and clip its complete
  actor and critic gradients. Independently verify recorded trip statistics, probe
  probabilities and work, and require a complete census and the same lives to meet
  all acceptance criteria. Distinguish historical results from corrected audits.
  Add a separate development assay of the stable door skill before and after
  competing experience, on saved copies with learning disabled. Library behavior,
  public interfaces and composed defaults are unchanged.

## 0.77.0 — 2026-10-08

- Correct the reward-rhythm instrument's saved world boundaries, refusal retries
  and work accounting. Require the same founders to satisfy every acceptance
  criterion, retain missing controls and unfinished runs in the denominator, and
  verify receipt arithmetic against recorded events. Preserve the original
  protocols and receipts; corrected reruns have separate source-bound receipts.
  Align the runnable example's income and alternation windows. Library behavior,
  public interfaces and composed defaults remain unchanged.
- Add the reward-rhythm chamber (`benchmarks/rhythm/reward_rhythm.py`, protocols
  `protocol-reward.json`, `protocol-reward-2.json` and `protocol-reward-3.json`):
  the beat paid by the world. One
  continuing `Brain.live` life, the same drive every moment, one unit for a step on the
  other foot than the last, nothing for a repeat, paid at the next moment; no teacher. The
  walker carries the efference copy; arms without it, on the composed reward defaults, always
  learning, at the control eligibility decay, frozen, a tabular learner given the same one
  bit, and uniform random. reward/1 (key-door operating point, need 0.5) failed its gate on
  fresh seeds 301 to 305: 3/5 acquired, one of them from birth, two limping at two changed
  steps in three. reward/2 (eligibility decay 0, a gate on learned beats against the frozen
  founder) passed on fresh seeds 401 to 405: 4/5 learned, none from birth, all five calm in
  the window and continuing identically from a mid-life checkpoint; the walker without the
  copy earns occasional rewards while exploring but its greedy probe holds one foot.
  The limp, a period-three attractor
  paid above the need, is the residual failure; reward/3 (need 0.9, fresh seeds 501 to 505)
  removes it on all five founders but fails its learned-credit gate on two with the beat from
  birth. The receipts are taken on the released 0.76.0. `examples/walking_for_reward.py` shows
  one founder with and without the copy. No default changes.

## 0.76.0 — 2026-10-08

- Give the introductory guides one recommended continuing-life loop:
  `Brain.compose(..., arousal=ArousalConfig())` followed by `brain.live(...)`.
  Use the composed defaults in the starting example, explain diagnostic reports
  separately, and keep explicit teaching/batching in the `step` reference.
  Executable examples check arousal and saved continuation. Runtime defaults
  and supported operations are unchanged.
- Harden the opt-in efference copy at episode and checkpoint boundaries: routine
  terminal forecasts clear both traces before choosing the next action and restore
  them on refusal; `resting_bias` leaves command neurons silent until a command;
  loading rejects invalid command ports, read gains and missing continuation state.
- Add the efference copy, `Brain.compose(..., efference_amplitude=..., efference_decay=...)`
  and `cadence.Efference`: a trace written from the command the brain issued, one
  `efference` neuron per motor neuron, read by the association region through a plastic
  projection at the working trace's scale. The working trace copies the settled state
  before the decision, so a brain whose observations do not change could not learn what
  to do after what it did; the steady-rhythm chamber of
  [#116](https://github.com/muellerberndt/cadence/issues/116) recorded that limit at
  0.37 to 0.44 alternation against 1.00 for a control given its own previous action. The
  default `efference_amplitude=0.0` builds the released composition byte-identically; the
  copy is a gene with zero as its control. `act`, `step`, `live`, `learn`, `reset`,
  `imagine` (a private copy issued the imagined response's own best guess), `save` and
  `load` carry it; a checkpoint with the copy is `cadence-generic/4`, and checkpoints
  without one keep their formats. `predict` and `accuracy` ignore it as they ignore the
  working trace. The steady-rhythm chamber gains `protocol-2.json` (rhythm/2): the copy at
  amplitude 3.0 and decay 0.0 on the rhythm/1 selected recipe, against that recipe as its
  control, on fresh seeds 201 to 205, with the erased, shuffled, static and reset controls
  acting on the trace and the copy. Confirmation (`results/confirmation-2-2026-10-08.json.gz`,
  verified): mean intact alternation 0.97 (`every`) and 0.93 (`mismatch`) against 0.50 and
  0.40 for the control, 1.00 for the flip-flop and 0.50 for uniform random; four of five
  founders at 1.00 in the `every` arm, recovery after disturbances in 16 of 20 rows
  against 0 to 1; per-event actions identical across cadences, budgets of 64 and above,
  tolerances and host load in 20 of 20 founders; 128 refusals at a 16-sweep budget and
  more teaching sweeps with the copy. The development grid (seeds 0 to 5, every cell
  kept in `results/development-efference.json`) gives 1.00 on all six founders in the
  `every` arm against 0.78 without the copy. `examples/walking_rhythm.py` shows one
  founder with and without it. The copy is a gene with zero as its control, not a default.
- `Brain.predict` and `Brain.accuracy` now honor slotted motor readouts (`slots > 1`):
  predictions are one choice per slot, shaped `(batch, slots)` like `act` and `step`,
  instead of a single argmax over the whole motor menu, and slotted `accuracy`/`fit`
  no longer fail with a broadcast error after teaching. One-slot brains are unchanged.
- `ArousalConfig` gains `need`, a nonnegative required reward rate per moment. Its
  unmet fraction supplies an additional arousal signal even when poor rewards are
  expected. The default is zero, preserving the previous reward law. This is an
  opt-in scalar need in the existing System 1 lifecycle, not a general goal detector,
  learned goal representation or default promotion.
- Keep the need calculation finite for extreme finite rewards and need values by
  clipping before subtraction or division. New saved arousal state carries the
  `cadence-arousal/1` marker; complete legacy configurations without `need` load
  with `need=0`, while an incomplete marked configuration is rejected. This
  preserves continuation of existing zero-need lives without treating missing
  fields in new checkpoints as valid defaults.
- Add the key-door nursery (`benchmarks/keydoor`) for
  [#111](https://github.com/muellerberndt/cadence/issues/111), roadmap row 07. One
  continuing life encounters delayed door rewards and a moved key source, with
  always-learning, zero-eligibility, retimed-own-reward, frozen, no-pouch, tabular
  Q(lambda) and random controls. Two historical `key-door/1` freezes failed their
  declared gates. The second records acquisition in 20/20 gated lives, re-adaptation
  in 19/20, frugality in 19/20 and calm in 17/20 against 18 required; delay 10 is
  ungated. These are exploratory results at selected settings: shared random draws
  made cut and food schedules differ between arms. The no-pouch result does not
  identify trace memory; a scarcity variant's unchanged 90%-fed gate is unsuitable
  when successful door openings pay with probability 0.5. Moving the key does not
  test changed goals, and varying lever counts does not validate physical-time
  semantics. Broad delayed-credit and reuse acceptance remains open.
- Correct the key-door instrument under `key-door/2`: separate world and yoked
  random streams, draw food availability independently of actions, include every
  pre-door cut position, measure door openings within the last 50 completed trips,
  probe the actual phase end, record undelivered final feedback and yoked rewards,
  count probe memory reads separately, and retain completed forecast work when the
  following answer refuses. Preserve all historical receipt and
  protocol bytes; corrected runs are not frozen confirmation of that old protocol.
  Verification labels legacy limitations and reporting verifies before rendering.
  No corrected confirmation campaign or passing capability gate accompanies these
  fixes. Historical routine latency is a within-life laptop observation, not
  matched-state efficiency, reduced settlement work or electrical-energy evidence.

## 0.75.0 — 2026-10-06

- Add `Brain.live` and arousal for one continuing stream
  ([issue 88](https://github.com/muellerberndt/cadence/issues/88),
  [issue 122](https://github.com/muellerberndt/cadence/issues/122)). A brain
  constructed with `arousal=ArousalConfig()` answers a routine moment with the
  greedy choice of one qualified settle and changes no parameter, record or
  optimizer state, while the eligibility of earlier sampled actions fades by one
  step (`ActorCritic.fade`). An outcome that contradicts the forecast made before it
  (surprise) or a reward below what the stream usually pays (want) rouses it; an
  aroused brain samples at a temperature its want raises, keeps eligibility,
  learns from every outcome and writes memory, and the outcome that woke it is
  written to its memory. `Arousal` and `ArousalConfig` are exported; every
  constant of the law is a gene with hand-set founders and a declared space for
  `genes`. `Brain.act`, `ActorCritic.act` and `ActorCritic.probabilities` take
  an optional sampling `temperature`; `Brain.last_arousal` reports each moment;
  a brain with arousal saves it under the format name `cadence-generic/3`.
  `step`, the composed defaults, the settling and learning equations and the
  checkpoints of brains without arousal are unchanged.
- `ArousalConfig` gains `value_surprise` and `record_surprise`: a brain with an
  associative memory also forecasts the outcome of its chosen action from the record
  it holds, and the error of that record is a second surprise channel with its own
  usual size (`Arousal.usual_record`, `Brain.last_arousal["record_error"]`). The
  founders weigh it at zero, so the law is unchanged by default: on the nursery's
  development seeds the record channel woke the brain sooner and left more lives
  searching too briefly. Checkpoints of brains with arousal carry the record forecast
  of the awaited action.
- Add the odour nursery (`benchmarks/reversal`): one continuing life through
  acquisition, reversal and return with an unrelated stable skill, at pre-switch
  exposures from 100 to 10,000 trials, on a frozen protocol with ten fresh
  confirmation seeds, against always-learning, released-default, memory-only,
  graph-only, frozen, replay, reset, tabular and uniform-random arms, with
  source-bound receipts and a verifier. Its gates passed on fresh seeds at each of its
  three freezes; at the third, of the 40 gated lives 39 reversed, 39 returned and 40
  kept the stable pair, with median reversal lags of 15 to 30 trials, and one life
  never searched for the moved reward. It reproduces the historical finding that the
  always-learning brain stops sampling the choice that must change, counts the
  witnessed approaches until the greedy choice turns (one), and measures the
  routine share and the settling work of each mode. The associative memory
  carries the adaptation, the graph's reward learning alone does not acquire the
  task, a routine moment pays one full settle, and arousal on the released
  composition stays at chance: the operating point of one continuing stream it
  declares (working trace amplitude 0.3, consolidation 0.25, a tenth of the
  composed actor rate) is a development setting of that chamber; no default
  changes.
- Odour nursery, third freeze (`benchmarks/reversal`), for the readings
  [issue 88](https://github.com/muellerberndt/cadence/issues/88) still required:
  the probability of approaching each odour under the behaviour that acted and under
  the base policy, and the probability of each executed action, read from the living
  brain at every trial; a `replay` control that presents the brain's own witnessed
  records of the first rule to its memory again, one per trial, in place of the
  earlier counterfactual payoff; and the complete work of a life, with probes,
  checkpoint files, memory reads and writes, presentations, brains built, the sweeps
  of a refused attempt and the wall time of a moment per mode. Receipts of the earlier
  freezes verify by their own kind. The confirmation ran once on fresh seeds.
- Harden `live` continuation and its audit instruments: reject contradictory
  pending-action checkpoints and incomplete arousal genes; preserve routine
  feedback on a refused arousal update and identify already accepted sampled
  feedback. Keep arousal statistics atomic on numerical overflow, and support
  tiny positive averaging rates and sampling temperatures. Validate nursery
  receipt plans, sources and readings, and count discarded reset brains and
  frozen actions. Historical receipts retain their original sources and work
  limitations; the old-rule control is not equal-work witnessed replay. Issues
  88 and 122 retain their remaining acceptance requirements.
- Add the bounded steady-rhythm chamber (`benchmarks/rhythm`) for
  [issue 116](https://github.com/muellerberndt/cadence/issues/116): one
  continuing `Brain.compose` life taught to alternate two actions under constant
  drive, with frozen inputs, a frozen protocol, a declared physical event
  cadence, checkpoint-forked erased/shuffled/reset/static controls, a matched
  flip-flop control, a uniform-random baseline, pause/distractor disturbances,
  paced runs under solver-budget and host-load variation, and checkpoint
  continuation checks between actions and during a pause. Results are measured
  limits of the current System 1 on five fresh seeds; no core source, default,
  mechanism or gene changes.

## 0.74.0 — 2026-10-04

- `ActorCriticConfig.eta_bias` left unset derives `eta / 10` at construction, the
  rule of issue 126 applied to the actor ([issue 143](https://github.com/muellerberndt/cadence/issues/143)).
  An explicit value is used as given; construction warns when the bias rate exceeds a
  positive `eta`. The bare default (`eta=0.5`) resolves to the former 0.05, and the
  composed brain keeps its measured `eta_bias=0.05` at `eta=1.0`, so composed and
  default brains are unchanged; an actor with a lowered `eta` and no explicit bias
  rate learns with a bias step a tenth of its synapse step instead of 25 times it.
- Grouped motor slots on the composed brain ([issue 142](https://github.com/muellerberndt/cadence/issues/142)):
  `Brain.compose`, `Brain.build`, `Brain.genome` and `Brain(...)` take `slots`, a
  count of equal groups or one size per group covering the actions. Each slot is one
  softmax that settles with the others; `act` and `step` return one index per slot;
  lateral inhibition stays within a slot and the unset `lateral` follows the largest
  slot; episodic memory writes the chosen neuron of every slot; `Brain.load` rebuilds
  the actor on the saved grouping. One slot is the unchanged default.

- Report `capped`, the share of observed rows whose dopamine exceeded `dopamine_cap`
  before the clip, in every `learn` report and through `Brain.last_learning`
  ([issue 139](https://github.com/muellerberndt/cadence/issues/139)). Add the
  night-replay chamber (`benchmarks/replay/`): a day of decisions through `step`,
  a night in which a saved copy re-experiences that day, an equal-experience awake
  control, and frozen policy readings as the adoption gate. Document that a replay
  through `step` is more experience at the same rates, not consolidation, with
  the measured causes of a policy that ignores its observation (sign-only
  dopamine at the composed actor rate on one stream, the default working trace)
  and a setting under which the night helped as much as fresh experience, the
  equal-experience comparison of
  [issue 112](https://github.com/muellerberndt/cadence/issues/112). Defaults,
  equations and saved-state semantics are unchanged.

- Accumulate CUDA block transport directly into its destination to avoid a
  temporary product and a separate addition kernel per block. Add actual-device
  System 1 equation, gradient, refusal, memory and continuation checks, plus a
  source-bound CPU/CUDA runtime and memory comparison for issue 98.
  Preserve POSIX source keys in credit diagnostics and LF bytes in the frozen
  phrase fixture so the existing provenance checks also pass on Windows.

## 0.73.1 — 2026-10-04

- Distinguish signed activity below rest from silence (issue 106). Preserve the
  historical strict activity-fraction expected failure and add functional
  response, qualified acquisition/retention and full saved next-update guards
  for the existing zero and optional 0.5 processing-bias settings. Core equations,
  defaults and saved-state semantics are unchanged.

- Record the measured decline of a normalized composed default
  ([issue 131](https://github.com/muellerberndt/cadence/issues/131)): at the
  proposed `eta=0.003, normalize=0.99, momentum=0.9` for both composed
  learners, supervised acquisition contracts pass but the reward stream fails
  its re-adaptation contract (0.486 against 0.9 after a contingency change).
  Composed defaults remain unnormalized; normalized rates stay per-application
  settings behind the 0.72.1 construction warning. Documentation only; no
  default or equation changes.

- Harden the acquisition and retention instruments: admit source and input
  identities before execution, distinguish executed outcomes from accepted
  feedback, count memory writes only after commit, and charge final receipt
  writes against declared time and storage limits. Preserve refused work and
  complete case censuses, with independent verification and adversarial guards.
- Make retention preparation work in a standalone checkout or source
  distribution. Formal-source snapshots are explicitly requested inputs;
  omitting them does not claim a formal verification result.
- Add actual sampled-action association, partial-cue continuation and saved
  feedback guards, and bounded finite-horizon input/trace checks. Broaden the
  default test inventory to include the acquisition instrument guards.
- Add a portable, read-only acquisition/retention results demo and document the
  research wrap-up. The 360/360 partial-cue and 24/24 order-sensitive results are
  bounded development screens from pinned 0.73.0 workers; fresh retention
  confirmation, integrated replay and native transfer remain open in issues
  [85](https://github.com/muellerberndt/cadence/issues/85) and
  [110](https://github.com/muellerberndt/cadence/issues/110). Failed controls and
  source/custody limits remain visible.
- Require proposals to use local agreement repair into the same global
  equilibrium, test the simplest existing System 1 first, and demonstrate
  benefit while preserving acquired capabilities. Animal and human brains
  remain the reference, including their finite capacity and possible rigidity.
  Update contributor/agent instructions, review template and current guides.

The numerical runtime, public defaults, learning and memory equations, and
checkpoint contracts match 0.73.0. Experimental centering and a normalized
composed default are not promoted by this release.

## 0.73.0 — 2026-10-04

- Resolve the unset motor `lateral` of `Brain.compose`, `build` and `genome` by
  readout width: -0.5 up to 8 actions, 0.0 above
  ([issue 124](https://github.com/muellerberndt/cadence/issues/124)). Measured on
  composed brains, -0.5 settles a small action menu in the same few dozen sweeps
  as 0.0, while from 12 actions the undamped free solve stops settling and damped
  answers take about nine times the sweeps. An explicit `lateral` is used as
  given, and small-menu brains are unchanged.
- Calibrate to the competitive operating point by default
  ([issue 125](https://github.com/muellerberndt/cadence/issues/125)):
  `Learner.calibrate` left without `level` now places the mean top output per
  row and slot near 0.5 instead of the whole readout's mean, which on a 36-way
  readout had selected saturating gains (12, then 128). An explicit `level`
  keeps the mean target; reports carry `target`, `mean_output` and `top_output`
  per candidate. Single-output calibration selects as before.
- Derive an unset `LearnerConfig.eta_bias` as `eta / 10` at construction
  ([issue 126](https://github.com/muellerberndt/cadence/issues/126)); the
  standalone default stays 0.02 and the composed default becomes 0.05. An
  explicit value is kept, a resolved value rides through `dataclasses.replace`
  unless re-derived with `eta_bias=None`, and a bias rate above a positive
  synapse rate warns, since the bias step then dominates.
- Count and warn when a finite teaching free phase uses its entire `free_steps`
  budget under a movement tolerance
  ([issue 127](https://github.com/muellerberndt/cadence/issues/127)): the lesson
  was learned from a state that may not have settled. `Learner.step` reports
  `free_budget_exhausted` (`demonstration_free_budget_exhausted` through
  `Brain.step`); phases with `tolerance=None` remain declared fixed-length and
  silent. The finite update law is unchanged.
- Warn at `LearnerConfig` construction when `qualified=True` and `nudged_steps`
  is below `free_steps`, including through `dataclasses.replace` on a composed
  configuration: qualified phases are settle budgets, and the composed finite
  teaching default of 12 nudged sweeps refuses every realistic lesson
  ([issue 123](https://github.com/muellerberndt/cadence/issues/123)). A refused
  nudged or opposite phase under such a configuration names the budget mismatch
  in its `LearningPhaseError` message and new `hint` attribute. Document the
  budget semantics in the learning, composition and troubleshooting guides.
  Finite teaching, the composed defaults, the reward-eligibility contract and
  refusal transactions are unchanged; deliberately small qualified budgets
  remain allowed.

## 0.72.1 — 2026-10-03

- Warn at learner or actor-critic config construction when `normalize > 0` and
  either `eta` or `eta_bias` exceeds `0.05`; include the actor's independent bias
  rate and point the warning to the constructor caller. The diagnostic excludes
  the critic's separate rate and does not change optimizer equations or defaults.
- Document the RMS update, floor and momentum effects, independent bias rates
  and the limits of the reported Atari, Transcribe and Patch World pilots from
  [issue 131](https://github.com/muellerberndt/cadence/issues/131). The warning
  threshold and suggested development sweeps are not stability guarantees.

## 0.72.0 — 2026-10-03

- Add optional `resting_bias` to `Brain` and `Brain.compose`, with finite scalar
  validation, protected sensory/visual, working-memory and motor boundaries,
  and saved initialization metadata separate from learned biases. The default
  remains zero; responsiveness is not an acquisition or retention guarantee.
- Add the vanished-cue recall chamber (`benchmarks/recall/vanished_cue.py`, issue 84): a
  continuing brain's free recall of a cue across blank or distracting delays, against
  erased and shuffled trace controls and a separately trained history comparator.
  Freeze paired episodes, fork complete probe checkpoints, charge attempted
  teaching and action solves, and retain refusal and source records.
- Add the `lateral0-local-rms` and `lateral0-resting` candidate genes to the
  acquisition microscope, with explicit effective settings and unchanged
  first-refusal stopping rules.
- Correct acquisition evidence descriptions for effective rates, nudged budgets
  and accepted versus attempted updates. Document supervised-only continuing
  interaction separately from zero-reward transitions. Broader acquisition and
  recall acceptance remains open; no memory or learning default is changed.

- Expose immutable `Brain.last_settlement` diagnostics for successful and refused
  action/prediction solves: per-row residuals, qualification, sweeps, checks and
  damping. Keep diagnostic state outside checkpoints and preserve action and
  feedback transactions. The report explicitly excludes learning and memory work.
- Add a guided documentation entry and link previously orphaned guides. Clarify
  reciprocal composition, the equilibrium world-model hypothesis, separate
  transition predictors, concrete memory/readback mechanisms, configuration
  defaults and the scope of centered dopamine and `Life`.
- Give contributors and agents a task-to-guide map, continuing-brain construction
  recipe and explicit sleep/dream preservation guidance. Include `AGENTS.md` in
  source distributions and validate its documentation links.
- Extend the continuing example with witnessed corrective teaching, repair and
  continued use, reporting actual outcomes alongside free-answer and other
  settling work. Preserve saved pending-feedback continuation.

- Center introductory and contributor guidance on one continuing equilibrium
  brain across bootstrap, use, witnessed disruption and local correction. Add
  an executable world-model guide with explicit current integration boundaries;
  keep independent learning controls and distinct model families labeled.

## 0.71.1 — 2026-10-03

- Reject nonfinite or inconsistent states during finite bias calibration, even
  without a report. Check the final candidate before returning biases; an invalid
  solve raises `RuntimeError` without using its output to advance the search or
  changing the source graph. Finite searches still permit unsettled states.

## 0.71.0 — 2026-10-03

- Add opt-in qualified graph learning through `LearnerConfig.qualified`. The
  free, positive and required negative teaching phases must satisfy the original
  full equations before one local update. `LearningPhaseError` retains phase
  states and attempted work; refusal preserves parameters and optimizer history.
- Extend `NeuralGraph.equilibrate` with bounded numerical step halvings. Repeated
  complete-state checkpoints can move an unqualified attempt to a smaller step
  sooner, while every accepted state still meets the original residual and all
  attempts share the declared sweep budget.
- Report all teaching phases, original residuals, residual transports,
  stagnation comparisons and attempted/accepted row presentations. Preserve
  accepted and refused demonstration costs in `Brain.last_learning`.
- Make gain calibration honor qualified learning and reject invalid states.
  Its default grid spans the current gain's representable powers-of-two multiples
  from 1/256 to 256, trying the current gain first. Explicit grids retain their
  supplied order. `Learner.last_calibration` records every attempted candidate;
  an all-refused search preserves the graph and optimizer.
- Add opt-in qualification and reports to `calibrate_bias`, checking every
  midpoint and the final candidate before returning biases. Report observed means,
  target gaps and all solve work. Finite calibration remains available; a
  qualified operating point does not guarantee a requested target or acquisition.
- Expose motor competition through `Brain.compose(lateral=...)`, retaining the
  default per-pair weight of -0.5. Zero removes those lateral connections while
  preserving reciprocal processing/motor feedback.
- Separate reward eligibility duration with `ActorCriticConfig.eligibility_steps`.
  The Brain default stays at 12 finite nudged steps independently of supervised
  teaching budgets; standalone `None` retains learner-budget inheritance.
- Restore associative memories, separator state and terminal working traces when
  reward bootstrap qualification fails, preserving the actual outcome for retry.
  Accepted real feedback remains learned if a subsequent lesson or action refuses.
- Score `Brain.fit` epochs through qualified, memory-free public predictions.
  A refused score preserves lessons already accepted in that epoch.
- Add a runnable `Brain.compose` example and NumPy-only CI coverage for actual
  feedback, demonstrations, free recall and saved pending-feedback continuation.
- Add repository acquisition and retention instruments with source-frozen
  protocols, independent residual/contrast checks, charged rehearsal, preserved
  refusal/case censuses and saved continuation. Separate graph acquisition,
  working traces and consolidated associative storage.
- Clarify conditional gradient assumptions, independent learning-rate
  hyperparameters, standalone versus composed defaults, calibration limits and
  device execution. The adaptive `ActorCritic` optimizer uses host arrays;
  supported blocked PyTorch `Learner` updates remain on the device. Document
  the independent `eta_bias=0.02` and `temperature=0.2` defaults accurately.
- Add numerical and behavioral regressions for phase refusal, bounded damping,
  calibration admission, motor wiring, memory rollback and saved continuation
  across supported backends. Gradient checks retain their symmetry, smooth-branch,
  nudge-limit and loss-scaling hypotheses.
- Distinguish current application demos from archived research examples. Restore
  the recorded 0.61.0 release and 0.62.0 development history without inventing
  releases for unpublished version numbers.
- Update consumer guidance for explicit qualified protocols, calibration and
  refused-lesson retry. Correct Atari settings and pooled processing-time labels;
  retain historical finite-probe measurements and their source identity.

- Document focused local contract checks, separate foundation/population suites
  and a NumPy-only environment for shorter iteration. Keep full CI coverage,
  fixtures and assertions; optional backend skips remain explicit.

Finite supervised teaching remains the default. System 1 memory, plasticity,
private imagination and action remain available; optional System 2 continues
to join the same neural graph. Numerical qualification alone does not establish
general acquisition, lifelong retention or an efficiency advantage.

## 0.70.0 — 2026-10-02

- Make System 1 the default continuing brain, with working trace, fast and
  persistent associative memory, plasticity, action and private imagination.
- Provide `Brain.compose` for reciprocal base modules and optional
  System 2 observer regions within the same neural graph.
- Add `Brain.imagine` for private responses to supplied hypothetical
  observations. Learned environmental consequences and action planning use
  the temporal-model API.
- Qualify actions and independent predictions against the full state equations.
  Refusal preserves action state and pending feedback; real outcomes learned
  before a subsequent refusal remain learned.
- Use bounded numerical damping when a free solve needs it, then check the
  original model's residual. The total budget and finite teaching rule stay fixed.
- Provide event records and consolidation, learned temporal paths, continuous
  action planning and finite response protection through advanced APIs.
- Keep exact state-and-error feedback available in the advanced population solver.
- Simplify guides around current usage. This is experimental software; backward
  compatibility is not a design requirement. NumPy is required, with optional
  acceleration backends.

## 0.62.0 development revision — 2026-10-02

This entry records the development sources at
[`1f9daac`](https://github.com/muellerberndt/cadence/commit/1f9daac).
The recovered work became release 0.70.0; 0.62.0 was not published as a
GitHub release or on PyPI. Names below describe that development revision.

- Restore the capable pre-reset foundation from 930ee807: continuing
  `GenericBrain` interaction, `Trace`/`Afterglow`, consolidating
  `SynapticMemory`, record patches and sleep, temporal learning, private
  imagination, action planning and response protection. Preserve subsequent
  numerical, continuation and recursive-wiring hardening.
- Add `GenericBrain.compose` as a direct modular entry with working trace and
  consolidating memory, optional reciprocal observer regions, and the existing
  continuing interaction interface. Add private `GenericBrain.imagine` over
  supplied hypothetical observations; environment prediction remains the
  separate learned temporal-model contract.
- Qualify `GenericBrain.act`, `predict` and `accuracy` against the full state
  equations. Exhausted action repair preserves live state and pending feedback;
  consumed real outcomes stay learned if a following action refuses. Keep finite
  eligibility/training phases distinct from this free-answer qualification.
- Keep cortical observation optional. The foundation can already be deep and
  modular; observer feedback extends the shared graph rather than replacing
  working memory and learning with a narrower model.
- Preserve the newer state-and-error solver under
  `cadence.experimental.equilibrium`, with its own guides, examples and tests.
  Its sparse patch-connectivity checks and same-call stationary-evaluation
  optimization remain available there, without changing the restored APIs.
- Rewrite the entry guides around the biological-brain objective, working
  mechanisms and actual application source identities. The default package
  requires NumPy. Keep current GPL-3.0 licensing and historical attribution.
- Preserve original Amen, Connect Four and Atari checkpoints and browser
  engines. Library recovery, checkpoint parity and native application behavior
  require separate verification; no old receipt is silently promoted.
- Recover the capable foundation before releasing the narrower candidate as the
  default. Its separate numerical, CI and package evidence stays source-bound;
  the restored package requires its own verification.

## 0.61.0 — 2026-10-02

- Restore the principle as an enforced default: patches repair local
  disagreement to reach a coherent brain state, and further repair is driven by
  that state's mismatch with reality. `Cortex.build()` now refuses a layout in
  which a population settles with no other population, and a layout in which a
  group of populations settles apart from the rest. Every population must read
  another population's states or errors, or be read by one, and those reads
  must join all populations into one connected system; an unread sensors-only
  population or a disconnected group raises `ValueError` naming it. The
  smallest brain is two populations.
- Remove the input-only "flat" layout from the README, quickstart, layout and
  design guides, agent guides, examples and test fixtures. The layout example
  defaults to a two-population brain (`small`), with `deep` and the explicit
  `recursive` experiment. The query-cost and temporal-credit examples no longer
  build an input-only arm; their recorded receipts stay as recorded.
- Lead the README and the contributor guide with the main hypothesis and the
  simplicity premise, and contrast settlement with feed-forward backpropagation.
- The repair law, energy, qualification tolerance and admission contract are
  unchanged. `cortex.py` changed, so snapshots bind to this release's sources;
  snapshots saved by `0.60.0` load only in `0.60.0`.
- Error-reading observers remain experimental; this release still claims no
  automatic System 2, retained useful recursive correction or reproduced
  musical quality.

Earlier entries remain in the [source changelog before the guide simplification](https://github.com/muellerberndt/cadence/blob/1f9daac/CHANGELOG.md).
Version numbers 0.63.0–0.69.0 were not published; they are not missing release entries.
