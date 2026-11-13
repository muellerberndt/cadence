# Grammar chamber: whole sentences from a meaning

Readings on this page taken before 0.95 took the address lesson as the distance, which left the library in 0.95, so a memory now reads with the contrast and those tables are historical.

This bounded instrument reads the Grammar stage of the language goal
([#173](https://github.com/muellerberndt/cadence/issues/173)): agreement across distractors,
roles, nested structure (Dyck) and two-hop composition, on held-out combinations of words and
roles and on held-out lengths and depths, in sentences one continuing `Brain.compose` life says
from a meaning, from its first word to `end`. It is #173's "bounded first" language: finite,
compositional, with contexts that select different continuations of the same recent words, a
competing second grammar for persistence and recovery, a frozen copy and next-word learners at
matched information. Next-word accuracy is read beside the sentences as component evidence
only. Read [continuous interaction](../../docs/start.md#continuous-interaction), the
[recall memory](../../docs/memory.md#the-recall-memory-in-full) and the
[sequence battery](../sequence/README.md), whose recipe and life loop this chamber reuses,
before interpreting results. It establishes no default.

## The language

Eighteen words, each an input and an action of the brain: four nouns in the singular and the
plural (`dog cat fox owl dogs cats foxes owls`), two verbs in the singular and the plural
(`sees see chases chase`), `near`, four brackets `( ) [ ]` and `end`. Seventeen further inputs
carry what is not a word: the meaning's concepts (`DOG CAT FOX OWL PLURAL AGENT PATIENT MOD SEE
CHASE OPEN_ROUND OPEN_SQUARE CLOSE FROM DIALECT`), the turn (`GO`) and `SPEAK`, the flag on the
brain's own words.

An episode is one stream from a reset. The meaning arrives as concept moments, one per moment;
at `GO` the brain says the sentence one word per moment, hearing its previous word with the
`SPEAK` flag, until `end`. Three families of sentences share the words:

| Family | The meaning | The sentence | What it asks |
| --- | --- | --- | --- |
| `clause` | the verb, then the agent, its distractor nouns (`MOD`) and the patient, each a noun concept with its number and role flag; the agent and the patient in either order | `agent (near mod)* verb patient end`, e.g. `dog near foxes near owls sees cats end` | the verb agrees with the agent across distractor nouns of either number; the role flags, not the order heard, place the nouns |
| `dyck` | a prefix of `OPEN_ROUND`, `OPEN_SQUARE` and `CLOSE` concepts that never closes below zero | each bracket said (a `CLOSE` is the match of the innermost open bracket), then every bracket still open closed, innermost first, e.g. `( [ ] [ ] ) end` | a stack over the sentence |
| `path` | the world says facts `X near Y end` (heard words, no flag), one chain from a start to a sink among two distractor facts; the meaning names the start (`FROM`) | the chain, `fox near dogs near owl end` | each hop addressed by the word the brain said last: two-hop composition |

The second grammar (`g2`, the persistence part) is the clause with the patient before the verb,
`agent (near mod)* patient verb end`, over the same words; its meaning's verb moment carries
`DIALECT`.

**Held out.** Per founder (`Split.of`): two agent-number-verb triples (for example plural foxes
that chase) and one pair of nouns never agent and patient of one trained clause, in either order.
Training clauses have zero to two distractor phrases, Dyck sentences a depth up to three, paths
one or two hops. A batch is one family and one shape, sixteen streams with their own content;
the first grammar mixes clauses (one half), Dyck sentences and paths (one quarter each).

## Panels

Every panel is four batches of sixteen sentences, drawn from `SeedSequence([founder, 33])` and
the same for every arm and control of a founder.

| Panel | Sentences |
| --- | --- |
| `clause` | trained lengths (zero to two distractors), combinations that training shows |
| `clause_held` | trained lengths, each with a held-out triple or the held-out pair |
| `clause_long` | three and four distractors |
| `roles` | the held-out pair in both roles, side by side: `cat chases dog end` and `dog chases cat end` |
| `dyck` | depths two and three, nested prefixes and prefixes with closes |
| `dyck_deep` | depths four and five |
| `path` | one and two hops |
| `path_long` | three hops |
| `g2`, `g2_long` | the second grammar at trained and at three and four distractors |
| `ctx_agent` | twins: the same meaning but the agent's number; the agent and the distractors given (heard as said), the brain says the rest |
| `ctx_verb` | twins: the same words given, the meaning's verb differs |
| `ctx_dyck` | twins: the same opens but the first; the opens given, the brain closes |
| `*_removed` | each context twin with the earlier context removed: no meaning, `GO`, then the recent words |
| `*_erased_recall`, `*_erased_trace` | each context twin with the earlier context erased at the continuation's start from the recall memory (the slots of the meaning, `GO` and the first given word set to zero) or from the working trace |

An exact solver (`solve`) rebuilds every sentence of every panel, and of every training batch,
from the heard moments alone (a test checks it): every panel is answerable from what the brain
hears, except the removed contexts, which no speaker can answer.

## Scores

Every panel is read on a fork of the brain that never changes the life, greedy, closed loop.

- `exact`: the whole sentence said, from its first word through `end`, is the sentence asked.
- `agree`: the sentence says one verb and its number is the meaning's agent's; `agree_own`: the
  number of the noun the brain itself said first; `verb`: the verb's identity; `attracted`: of
  the sentences whose verb follows a noun of the other number than the agent's, the share whose
  verb agrees with that noun.
- `roles`: the first word is the agent and the word after the verb (before it in `g2`) is the
  patient, noun and number; `mods`: the distractor phrases as asked.
- `valid`: a balanced bracket sentence of the asked length, every close matching.
- `hops`: chain words said right in order; `first_hop`.
- Twin panels: `both`, the share of twins both said right; `differ`, the share whose two
  continuations differ. With the context removed, twins hear the same moments and say the same
  words by construction: the panels check that nothing else carries the context.
- `next_word`: each word said right with every previous word given (component evidence).
- Per condition: distractors (`L0` to `L4`), an attractor of the other number among them
  (`attract`, `plain`), the order the roles were heard (`canonical`, `inverted`), depth,
  nested or mixed, hops; exact and agreement per condition.
- `located` (brains with a recall memory): for every word whose content a heard moment holds
  (a noun's concept, the verb's concept, the open bracket a close matches, the fact a hop
  follows), the share where the recall read's first head attends most to that moment or to the
  moment before it, whose next it is (`delivered`: the read's value or next port holds it), the
  share said right (`said`) and said right when delivered, teacher-forced.
  A word delivered and not said is lost at the readout, one not delivered at the read.

## The life

The brain is `Brain.compose(35, 18, modules=(64,), seed=founder)` with the arm's settings. At a
moment of the meaning or of the world the brain acts greedily and is not taught. From `GO` on,
a lesson on the target word (`Brain.lesson`) comes before the greedy act on the same
observation, and the next moment hears the target's word. The lessons come from
`SeedSequence([founder, 11])`, so every arm and control of a founder hears the same sentences
in the same order.

| Arm | Settings |
| --- | --- |
| `founder` | `Brain.compose` at every founder setting |
| `recipe` | `episodic=False, teaching="normalized"` with the recall memory's recipe for lengths no lesson reached and its relative read (`recall=True, recall_recency=0.0, learning_surprise=True, recall_boundary=True, recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01, recall_relative=True`): the sequence battery's `recipe_relative`, the composition the recall guide gives for stories longer than the lessons and for copying |
| `maturity` | `recipe` with `maturity=True` |
| `frozen` | the `recipe` newborn, never taught |
| `default` | `Brain.compose` with nothing named but the backend: what a new brain gets (`recall=True` and its settings) |

**Persistence and recovery.** The first grammar for 4,000 batches, the second grammar alone for
1,500, then the first again for 500 (recovery), each phase read in full. Lessons leave
maturity's use at zero (maturity scales the actor's reward steps and the store's writes;
lessons, the recall maps and the critic keep their steps), so the chamber also reads the
second grammar acquired by paid practice: `practise` lets `Brain.step` sample every word,
hear the target's word next and be paid 1 for a right word and 0 for a wrong one, the last
outcome ending the episode through `Brain.learn`. The maturity life's trunk is the first
grammar by lessons (4,000 batches) and by practice (1,000), and its forks continue a copy of
the brain and of the training stream:

| Fork | After | Then |
| --- | --- | --- |
| `lessons` | the lessons | the second grammar by lessons (1,500), the first by lessons (500) |
| `reward_young` | the practice | `maturity_scale=1e9` (a step factor of 1): the second grammar by practice (1,500), the first by lessons (500) |
| `reward_mature` | the practice | the founder scale 10, the same |
| `reward_half` | the practice | `maturity_scale` set to the use reached, a step factor of 0.5 when the second grammar begins, the same |

The `lessons` fork lives the `recipe` arm's life to the bit; the recipe life on founder 0
checks it. After the first phase a fork is saved and loaded, both copies take the same three
lessons and say the same panel (`continuation`).

**Work.** Per phase: lessons and the rows they held and agreed on, the lessons' row sweeps, the
acts and their batch sweeps, practice steps, refusals and wall time; the step factor and use of
maturity.

## Controls

[`controls.py`](controls.py): a long short-term memory network (one layer) and a causal
transformer (two layers, two heads, sinusoidal positions, no dropout), their widths chosen so
that their parameter counts are the nearest to the recipe brain's `parameters()` (24,384: the
LSTM 60 units, 24,378 parameters; the transformer width 36, 23,490). They hear the brain's
moments, are taught each word of the sentence teacher-forced by cross-entropy and are read on
the same panels by the same scorer, greedy and closed loop. `online`: one pass in the brain's
order, one Adam step per batch (rate 0.003, gradient norm clipped at 1), through the same
persistence schedule. `offline`: the first grammar's 4,000 batches replayed in shuffled epochs
(rate 0.001) until an epoch's loss stops falling (patience 3, at most 30), an upper reference
reported apart. Work: updates, tokens, analytic multiply-accumulates (a backward pass at twice
the forward) and wall time. They are independent controls and never part of a brain's answer
path. `torch` is required for them alone.

## Run

From the repository root, with one BLAS thread per process:

```sh
PYTHONPATH=src python benchmarks/grammar/grammar.py --arms recipe founder --founders 0 1 2 \
    --phases '[["g1", 4000], ["g2", 1500], ["g1", 500]]' --checkpoints 1000 2000 3000 \
    --out DIR/brain.jsonl --save DIR/brains --workers 3
PYTHONPATH=src python benchmarks/grammar/grammar.py --plan DIR/plan.json --out DIR/brain.jsonl
PYTHONPATH=src python benchmarks/grammar/controls.py --models lstm transformer \
    --modes online offline --founders 0 1 2 --out DIR/controls.jsonl --workers 3
python -m pytest -q benchmarks/grammar
```

A plan is a JSON list of life specifications (`arm`, `founder`, `phases`, `forks`,
`checkpoints`, `continuation`, `save`); `run_life` documents them. A recipe life of the
persistence schedule takes about 20 minutes on one core, a founder life about 12, a maturity
life with its four forks about an hour; an online control about 8 minutes, an offline one up to
40.

## Attention ablations: which restriction of the transformer reproduces the brain's gap

[`ablations.py`](ablations.py) reads [#260](https://github.com/muellerberndt/cadence/issues/260):
the recall memory differs from transformer attention in five ways (the
[memory guide](../../docs/memory.md#the-recall-memory-in-full)), and each is put on the
chamber's transformer control, one at a time and in combinations, on the same sentences,
lessons and scorer, the control's width re-matched after each change so that its learned
parameter count is the nearest to the recipe brain's 24,384. The control is rebuilt with
explicit query, key, value and output maps (the same count as [`controls.py`](controls.py)'s at
the same width; a test checks it). The rows:

| Row | Restriction | The memory's property |
| --- | --- | --- |
| `none` | the control | |
| `A` | key and value maps frozen at their random initialization, credit flows through them to the layer below | keys and values are the observation code, never learned |
| `B` | attention in the first layer only, the query from the token's embedding; the second layer keeps its MLP | one held read per moment, nothing stacked on a read |
| `Bp` | `B` with the query also reading the previous position's layer output through a learned map, the positions computed in turn | the address `x W_x + c W_c`, `c` the cortex of the moment before |
| `C` | the values are the raw moments; the read enters the residual through a learned map from the moment's code and the logits through a fixed identity on the word rows; no value or output map | the value reaches the motor through the association's learned synapses, the copy pathway a fixed identity |
| `D1`, `D2` | one head; two heads with values at full width summed without an output map | heads are not learned subspaces mixed by a projection |
| `E` | no gradient into keys or values (detached: their maps never move and nothing below learns through them) | no credit reaches keys or values |
| `H1b`, `H2b` | one or two heads sharing one key and one value map, each with an address map of its own, competing for the slots (a head takes of every slot at most what the heads before left, its own attention scaled up where slots allow), summed, no output map | heads as copies of the held read; against the subspace heads `D1` and `none` |
| `AB`, `BC`, `all` | combinations; `all` is `A`, `Bp`, `C`, `D1` and `E` at once (`AE` is `E` by construction) | the memory's structure in the transformer |
| `lstmA` | the LSTM control with its input-to-hidden weights frozen random | the LSTM's analogue of `A` |
| `ctl`, `lstm` | `controls.py`'s transformer and LSTM themselves, so that they take the rows' initializations on the rows' machine | |

`none` is `controls.py`'s function at the same weights (a test copies them across) and another
initialization, so it is the baseline of the rows and not the control's life. `C` keeps a learned
map from the read into the residual: the read is linear in the values, so that map is a learned
value map of the raw moment. `Cs` has no learned path from the read at all.

`A` and `E` still read their keys from a learned embedding and, in the second layer, from what
the first attention found, and `B` bundles the query from the raw token, no stacking and keys
with no context. These rows separate them, with `code = R x + position`, `R` a fixed random map
scaled so that a one-hot moment's code has the position's norm, `e` the learned embedding and
`h` the layer's input:

| Row | First layer: query, keys and values | Second layer: query | Second layer: keys and values |
| --- | --- | --- | --- |
| `none` | `e`, `e` | `h` | `h` |
| `Ar` | `e`, `code` | `h` | `code` |
| `S1` | `e`, `e` | `h` | `code` |
| `S2` | `e`, `e` | `e` | `h` |
| `S3` | `e`, `e` | `e` | `e` |
| `S3r` | `e`, `code` | `e` | `code` |

Keys and values of `code` are frozen maps of a frozen code, so these rows also hold `E`'s
restriction wherever they hold `code`. `allr` is `all` with its keys from `code`; its values stay
the raw moment, which the identity needs. `allrn` is `allr` without the identity: the recipe brain
has no copy pathway. Neither carries the brain's next port or its working trace.

**The crossing** (`--crossing`): #228's non-additive addressing in the chamber's words. A context
(an open bracket), two facts (`A near V end`), a question (a close bracket); at `GO` the answer is
the value of the fact `c xor k`, then `end`. A separate panel, read after 3,000 online batches, on
the rows and on the recipe brain's life. A reader that ignores the crossing says 9/16 right.
Episodes are drawn from 3,072, so the lessons cover them: the panel reads whether the crossing can
be learned, not whether it generalizes.

**Initializations.** The held-out panel is bimodal: a life keeps every held-out verb or loses one
(`controls.py`'s own transformer offline reads .77 on founder 257005 and 1.00 on eight others).
`--init-seeds` draws further initializations on a founder's split, lessons and panels, and
`--spread` reads a row as the share of its lives at .95 or above.

Each row is read on the chamber's first-grammar panels after the 4,000 batches, `online` (one
pass, the brain's exposure) and `offline` (shuffled epochs to convergence), on founders 0 to 2,
with its learned (and total) parameter count, epochs and wall time.

```sh
PYTHONPATH=src python benchmarks/grammar/ablations.py --rows none A B Bp C D1 D2 E lstmA \
    --modes online offline --founders 0 1 2 --out DIR/ablations.jsonl
PYTHONPATH=src python benchmarks/grammar/ablations.py --table DIR/ablations.jsonl
```

### Readings on 0.94.0 (development founders 0 to 2)

One online pass, the mean of three lives per row, whole sentences said exactly:

| Row | Parameters | Clauses | Held-out | Brackets | Paths |
| --- | --- | --- | --- | --- | --- |
| `none` | 23,490 | 1.00 | 1.00 | .18 | .36 |
| `A` | 24,210 | 1.00 | .99 | .43 | .31 |
| `B` | 24,126 | .99 | .90 | .18 | .29 |
| `Bp` | 23,618 | 1.00 | .81 | .54 | .29 |
| `C` | 24,978 | 1.00 | 1.00 | .48 | .29 |
| `Cs` | 24,210 | .01 | .00 | .03 | .11 |
| `D1` | 23,490 | 1.00 | .98 | .23 | .32 |
| `D2` | 23,490 | 1.00 | .96 | .31 | .37 |
| `E` | 24,210 | .98 | .96 | .19 | .18 |
| `allrn` | 23,910 | 1.00 | .93 | .36 | .38 |
| `lstmA` | 23,846 | .36 | .10 | .06 | .07 |

Online brackets span .06 to .58 and paths .14 to .95 across nine founders of the one control, so
no online brackets or paths difference in this table separates from initialization. `allrn` keeps
clauses at 1.00 with the read silenced at 0.00, so the transformer's clauses rest on the read.
Offline, `none` reads 1.00, .89, 1.00, 1.00 (two lives) and `allrn` 1.00, .96, .99, .63 (six lives).

## The brain's own graph on exact gradients

[`gradients.py`](gradients.py) is the other side of the ablations and
[#183](https://github.com/muellerberndt/cadence/issues/183)'s same-graph reference: the recipe
brain's graph on the transformer's training, the same sentences and the same exposure. The
association and motor regions settle on fixed sources; the cost is the temperature times the
cross-entropy of the motor activity at the temperature of the lesson's nudge.

| Arm | The brain | The learning |
| --- | --- | --- |
| `recipe` | the chamber's recipe | the local contrast |
| `deep`, `deepna` | the recipe at `apical=0.25, teaching="deep"`, the documented point for the compartment; the teaching point alone | the local contrast |
| `observed`, `obsdeep`, `obsdeepna` | those brains | the local contrast, with the exact gradient read beside it and never applied (a test holds them to the bit) |
| `G1n`, `G1d` | the recipe; the `deep` brain | each lesson's contrast replaced by the exact implicit gradient at that moment's free fixed point, at the brain's normalized step; the held read, the trace, the surprise rule and the address map's lesson as they are |
| `G1a` | the recipe | the same gradient, with Adam at the online control's rate and the gradient norm clipped at 1 |
| `G2a`, `G2s`, `G2n` | the recipe | the gradient carried across a sentence's moments, through the working trace and the read's address, into the address map and the position maps; one step per batch; over every word (`G2a`, `G2n`) or the words said wrong (`G2s`); Adam (`G2a`, `G2s`) or the normalized step (`G2n`) |
| `Fn`, `Fd`, `FdL` | the recipe; the recipe at `teaching="deep"` with no apical | the local contrast with the address lesson's step on `sum_i (p*_i - p_i) K_i`; `Fn` and `Fd` patch it in, `FdL` reads it through the library, whose lesson it is since 0.95.0, and a test holds `Fd` and `FdL` to the bit |
| `FdT`, `FdC`, `FdR`, `FdX` | `Fd` | what crosses moments cut from birth, at every lesson and act: the working trace, the cortex the address reads, the previous read's focus, and the three |

G2 runs its forward in float64 torch on the act's own path (the settle's synchronous sweeps from
the stream's last state, since the settle has several attractors in places and the path selects
among them), differentiates each fixed point implicitly and writes its parameters back into a
`Brain`, so every panel is the library's own settle and scorer. The source neurons' biases stay
at zero, as the local contrast leaves them. `--init-seeds` draws other newborns on a founder's
split and lessons, and `--replicas` changes a newborn at its last bits, since the panels are
bimodal and last-bit arithmetic moves them.

**The checks** (`--check`, and the tests). Each gradient against central finite differences of the
fixed points: at a moment by the library's own settle, across a sentence by the unrolled forward,
each family of parameters on clauses, Dyck sentences and paths, on the newborn and after 300
lessons. The moment's gradient also against the local contrast as its nudge shrinks. A coordinate
whose perturbation moves a read's most attended slot crosses a real discontinuity of the relative
position term and is reported apart.

```sh
PYTHONPATH=src python benchmarks/grammar/gradients.py --check --founders 0 1 2 --out DIR/check.json
PYTHONPATH=src python benchmarks/grammar/gradients.py --arms recipe observed G1n G1a \
    --founders 0 1 2 --init-seeds -1 1000 1001 --out DIR/gradients.jsonl --workers 12
PYTHONPATH=src python benchmarks/grammar/ablations.py --table DIR/gradients.jsonl
PYTHONPATH=src python benchmarks/grammar/ablations.py --spread DIR/gradients.jsonl
```

### Readings at the base composition (development)

One online pass, founders 0 to 2 on their own newborns and on three further seeds each, twelve lives:

| Arm | Clauses | Two distractors | Held-out | Roles | Brackets | Paths | Two hops | Three hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `recipe` | .57 | .24 | .52 | .31 | .33 | .27 | .03 | .00 |
| `Fd` | .96 | .95 | .91 | .91 | .78 | .87 | .80 | .22 |
| `FdT` | .94 | .86 | .90 | .87 | .86 | .94 | .89 | .34 |
| `G2n`, exact gradients on the same graph (three lives) | .98 | .94 | .94 | .93 | .93 | .89 | .80 | .04 |
| Matched transformer, one pass (three lives) | 1.00 | | 1.00 | 1.00 | .18 | .36 | | .00 |
| Matched transformer, 14 epochs (two lives) | 1.00 | | .88 | .99 | 1.00 | 1.00 | | .00 |

The 24,384 parameters and the sweeps per life (6.3 million) of `Fd` equal the recipe's. A life on a
shared box takes about 495 seconds, the transformer's pass 34 and its 14 epochs 513. Paths are
bimodal under `Fd`: 9 of 12 lives at .95 or above, two at .33 and .41, and those two read .95 under
`FdT`. Lengths beyond training read .00 for every arm. A last-bit replica moves one founder by a
mean of .087 and at most .297 on these panels, so a difference under about .07 on a six-life mean
does not separate.

## A planning window

[`window.py`](window.py) reads [#257](https://github.com/muellerberndt/cadence/issues/257): the
chamber's sentences said `k` words at a time, the `k` words settled jointly. The brain is
`Brain.compose(35, k * 18, modules=(64,), slots=k)` with the arm's settings: `k` motor slots of
the eighteen words, one softmax each, in the one settle
([a planning window](../../docs/tune.md#a-planning-window)). The message arrives as in the
chamber (the concept moments, then `GO`); at each production step the brain settles the `k`
slots at once, says their words and moves the window on until a window holds `end`; the target
of a window past the sentence's end is `end` in every remaining slot. The window advances
through the world's echo (`--advance echo`: the brain hears its previous window, each word at
1.0 with `SPEAK`) or through the efference copy (`--advance efference`: the brain hears `SPEAK`
alone and its copy holds the `k` commands it issued, in training the words it said). With
`k = 1` and the echo this is the chamber's word-by-word life to the bit (a test checks the
words, the work and the located reading). Each slot has motor neurons and synapses of its own:
a position in the window is a coordinate, not a role shared across positions.

Beside the chamber's panels, per `k`: the Dyck sentences that fit in one window (`end`
included) apart from those that spill over it (`fits`, `spills`), and context twins aligned to
the window (`win_agent`, `win_verb`: clauses of two distractors whose first window is given,
the twins differing in the agent's number or the meaning's verb; for `k` above 5 the first
window holds the verb and the chamber's panels serve). `located` reads the step whose window
holds a word: whether the recall read lands on the word's meaning moment (`delivered`), the
attention it gives it (`mass`), and whether the slot says the word. Work is counted per
sentence: lessons, lesson row sweeps, acts and act sweeps, host time.

```sh
PYTHONPATH=src python benchmarks/grammar/window.py --ks 1 4 8 --founders 0 1 2 \
    --out DIR/window.jsonl --workers 2
PYTHONPATH=src python benchmarks/grammar/window.py --ks 4 --advance efference --founders 0 \
    --out DIR/window.jsonl
```

### The window on 0.94.0 (design input, development founders 0 to 2)

The chamber's sentences said through windows of 1, 3, 4 and 8 words (8 is the longest trained
clause), the same 4,000 batches of the first grammar as above, `recipe_relative`, the echo
advance; mean over the three founders (least to greatest in brackets); every entry a whole
sentence said exactly unless named.

| Window | Parameters | `clause` | `clause_held` | `clause_long` | `roles` both | Agreement | With an attractor | Without | Next word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (word by word) | 24,384 | .73 (.67-.78) | .61 (.50-.67) | .00 | .45 (.28-.56) | .82 (.78-.84) | .66 (.57-.75) | .91 (.80-.98) | .94 |
| 3 | 26,724 | .87 (.81-.94) | .68 (.64-.70) | .00 | .52 (.41-.66) | .97 (.95-1.00) | .97 (.92-1.00) | .98 (.95-1.00) | .97 (.96-.99) |
| 4 (a phrase) | 27,894 | .81 (.70-.88) | .52 (.47-.56) | .03 (.02-.03) | .51 (.41-.72) | .92 (.88-.95) | .90 (.87-.92) | .93 (.85-1.00) | .96 |
| 8 (the sentence) | 32,574 | .32 (.17-.53) | .10 (.00-.27) | .00 | .17 (.00-.47) | .86 (.78-.95) | .90 (.83-.96) | .84 (.75-.98) | .74 |

Trained clauses per length, exact: window 1 .89 / .79 / .35; window 3 .92 / .94 / .71; window 4 .92 / .79 / .60 at
zero, one and two distractors; window 8 .56 / .17 / .00.

**The verb** (`located`, teacher-forced; delivered / said / attention mass on the verb's
meaning moment), the clause at two distractors: window 1 .98 / .60 / .71; window 3
.00 / .81 / .20; window 4 .46 / .96 / .39; window 8 .04 / .69 / .15. On the held-out combinations
(the verb over zero to two distractors): window 1 .94 / .70; window 3 .22 / .83; window 4 .21 / .71; window 8
.25 / .60.

**Work per sentence** (the first grammar): window 1 0.36 lessons, 39 lesson row sweeps, 0.78 acts, 65 act batch sweeps, 13,120 synapses; window 3 0.14 lessons, 49 lesson row sweeps, 0.56 acts, 61 act batch sweeps, 17,728 synapses; window 4
0.11 lessons, 48 lesson row sweeps, 0.53 acts, 65 act batch sweeps, 20,032 synapses; window
8 0.07 lessons, 57 row sweeps, 0.49 acts, 48 act sweeps, 29,248 synapses.

- **A phrase window settles the verb with its agent.** At window 4 agreement with an
  attractor among the distractors rises from .66 to .90 and the trained clauses from .73 to
  .81, while the read delivers the verb's meaning moment less often (.46 against .98) and the
  verb is said more often (.96 against .60): the verb no longer waits on a read of its own; it
  is settled in the same settle as the agent's number. The held-out combinations do not rise
  (.52 against .61): what the window composes within a settle it does not generalize across
  the sealed pairs any better than word by word.
- **The whole sentence in one window loses the trained sentences.** At window 8 a position is
  a coordinate: the verb's slot is the second at no distractor, the fourth at one and the
  sixth at two, each learned apart, and the clauses of one and two distractors fall to .17
  and .00. Where a verb is said it agrees (.86), so the loss is of the positions, not of
  agreement.
- **Lengths beyond the window stay lost at every window** (.00 to .03): the first form
  shares no weight across positions, as #257 states.
- **The window's limit sits between a phrase and a sentence**: the phrase window gains what
  one settle can hold (the agent and its verb), and the sentence window pays for every
  position it must learn as its own coordinate.

### The window on 0.94.0, every reading of #257 (development founders 0 to 2)

The implemented reading on the same founders and lessons, at the windows the confirmation
declares (1, 3, 4 and 10; 10 is the longest sentence training shows, a Dyck sentence), by the
echo advance, and at 4 by the efference advance. The shared panels of the windows 1, 3 and 4
equal the design input's above to the score. Twins: both right / continuations differ.

| Window | `clause` | `clause_held` | held-out agreement | `clause_long` | `roles` both | `win_agent` | `win_verb` | `dyck` | `dyck_deep` that fit (n) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (word by word) | .73 (.67-.78) | .61 (.50-.67) | .72 (.59-.84) | .00 | .45 (.28-.56) | .26 (.12-.41) / .32 (.22-.44) | .46 (.25-.62) / 1.00 | .27 (.12-.39) |  (0) |
| 3 (the phrase) | .87 (.81-.94) | .68 (.64-.70) | .94 (.84-1.00) | .00 | .52 (.41-.66) | .55 (.38-.72) / .99 (.97-1.00) | .59 (.44-.72) / .81 (.62-.97) | .28 (.12-.38) |  (0) |
| 4 | .81 (.70-.88) | .52 (.47-.56) | .74 (.64-.92) | .03 (.02-.03) | .51 (.41-.72) | .57 (.50-.66) / .94 (.88-.97) | .65 (.47-.81) / .96 (.94-.97) | .10 (.00-.25) |  (0) |
| 10 (the sentence) | .32 (.25-.38) | .10 (.05-.12) | .72 (.70-.73) | .00 | .03 (.00-.06) | no twins / no twins | no twins / no twins | .11 (.05-.19) | .00 (16) |
| 4, efference | .63 (.58-.69) | .36 (.27-.42) | .67 (.64-.72) | .00 | .31 (.28-.34) | .41 (.31-.56) / .95 (.88-1.00) | .32 (.25-.38) / .93 (.91-.94) | .12 (.00-.25) |  (0) |

- **Agreement on held-out combinations** (#257's first bar): the phrase window reads .94 (.84-1.00)
  against word by word .72 (.59-.84); the held-out verb is delivered 0.22 / 0.83 at the phrase
  window and 0.94 / 0.70 word by word (delivered / said), so the phrase window says the verb
  without a read of its own.
- **Brackets within the whole-sentence window** (the second bar): no Dyck sentence of a
  held-out depth that fits the window of ten is said right on any founder, and the trained
  Dyck sentences fall to .11 (.05-.19) at that window: the whole-sentence window pays for each
  position as its own coordinate.
- **Context** (the third bar): the window-aligned twins differ in their continuations at the
  phrase window (.99 (.97-1.00) and .81 (.62-.97)) and are both right less often (.55 (.38-.72) and .59 (.44-.72)); at window 4 the
  agent is inside the echoed window, so `win_agent` there reads the echo, not the memory.
- **The efference advance** carries the window less well than the world's echo at window 4
  (clauses .63 (.58-.69) against .81 (.70-.88)): the copy holds the brain's own words, in
  training the words it said rather than the target's, and the lessons reach the readout
  through a context that drifts with its errors.
- **Work per sentence**: window 1: 0.36 lessons, 39 lesson row sweeps, 0.78 acts, 65 act batch sweeps; window 3: 0.14 lessons, 49 lesson row sweeps, 0.56 acts, 61 act batch sweeps; window 4: 0.11 lessons, 48 lesson row sweeps, 0.53 acts, 65 act batch sweeps; window 10: 0.06 lessons, 54 lesson row sweeps, 0.48 acts, 43 act batch sweeps.

### The confirmation on fresh founders (declared, 257000 to 257005)

Declared on 5 November 2026 at 20:02 before any run on the founders (the protocol and its sha256 are kept with the evidence), six fresh founders, the same lessons and panels, `recipe_relative` at windows 3 (the phrase, the gated window), 10 (the sentence) and 1 and 4 beside. Each gate at 0.90 on at least 5 of 6 founders.

| Gate | Founder: 257000 | 257001 | 257002 | 257003 | 257004 | 257005 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W1 held-out agreement, window 3 | .83 | .98 | .86 | .62 | .97 | .97 | 3 of 6, fail |
| W2 held-out-depth Dyck within the window of 10 | .00 | .00 | .00 | .00 | .00 | .00 | 0 of 6, fail |
| W3 agent twins both right, window 3 | .50 | .78 | .69 | .16 | .78 | .62 | 0 of 6, fail |
| W3 agent twins differ, window 3 | 1.00 | 1.00 | .97 | 1.00 | 1.00 | 1.00 | 6 of 6, pass |
| W3 verb twins both right, window 3 | .53 | .78 | .78 | .19 | .78 | .50 | 0 of 6, fail |
| W3 verb twins differ, window 3 | .94 | .97 | .91 | .88 | .94 | .84 | 4 of 6, fail |

W2 reads 16 held-out-depth sentences per founder that fit the window.

Beside the gates, per window (mean over the six founders, least to greatest in brackets):

| Window | `clause` | `clause_held` | held-out agreement | `clause_long` | `roles` both | `win_agent` both / differ | `win_verb` both / differ | `dyck` | `dyck_deep` that fit | verb delivered / said (held out) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | .72 (.41-.88) | .49 (.14-.70) | .71 (.58-.80) | .00 | .49 (.03-.75) | .42 (.31-.59) / .52 (.31-.84) | .54 (.28-.66) / 1.00 | .49 (.12-.81) |  | 0.85 / 0.70 |
| 3 | .82 (.59-.91) | .53 (.30-.70) | .87 (.62-.98) | .00 | .48 (.12-.59) | .59 (.16-.78) / .99 (.97-1.00) | .59 (.19-.78) / .91 (.84-.97) | .31 (.05-.45) |  | 0.31 / 0.78 |
| 4 | .75 (.70-.81) | .53 (.47-.69) | .85 (.80-.92) | .01 (.00-.03) | .49 (.31-.66) | .55 (.38-.66) / .99 (.97-1.00) | .55 (.41-.75) / .87 (.72-1.00) | .25 (.00-.52) |  | 0.21 / 0.73 |
| 10 | .39 (.23-.48) | .14 (.03-.27) | .84 (.80-.89) | .00 | .14 (.00-.31) | no twins / no twins | no twins / no twins | .30 (.20-.52) | .00 | 0.12 / 0.60 |

Work per sentence (W5): window 1: lessons 0.36, lesson row sweeps 41.69, acts 0.77, act sweeps 66.96, seconds 0.0025; window 3: lessons 0.14, lesson row sweeps 46.73, acts 0.55, act sweeps 58.91, seconds 0.0042; window 4: lessons 0.11, lesson row sweeps 46.16, acts 0.52, act sweeps 62.37, seconds 0.0026; window 10: lessons 0.06, lesson row sweeps 52.53, acts 0.47, act sweeps 44.21, seconds 0.0041.

The controls on the same founders, word by word (the window twins at 3; mean over founders; a control's held-out-depth Dyck sentences are read whole, the window split has no meaning for it):

| Control | `clause_held` agree | `win_agent` both / differ | `win_verb` both / differ | `dyck_deep` exact | multiply-accumulates | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| lstm offline | 1.00 | .98 (.91-1.00) / 1.00 | .99 (.94-1.00) / 1.00 | .09 (.00-.20) | 1.7e+12 | 211 |
| lstm online | 1.00 | .90 (.81-1.00) / 1.00 | .91 (.81-.94) / 1.00 | .05 (.00-.14) | 5.7e+10 | 6 |
| transformer offline | 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | .01 (.00-.06) | 1.1e+12 | 171 |
| transformer online | 1.00 (.98-1.00) | 1.00 / 1.00 | 1.00 / 1.00 | .00 | 5.7e+10 | 7 |

Founder parity at the opt-out (W6), the harness on the snapshot against `v0930.json`: lives differing: 0 of 24.

The full test suite from the branch (ruff, mypy and pytest as CI runs them): 4027 passed, 98 skipped, 1 xfailed, 109 warnings in 415.35s (0:06:55)


- **W1 fails at the bar and reads the mechanism**: the phrase window raises held-out
  agreement over word by word on five founders of six (word by word .58 to .80; the phrase
  window .83 to .98) and lowers it on one (.72 to .62), the founder whose phrase life is the
  weak one throughout (clauses .59, twins .16 and .19). The held-out verb is delivered by the
  read .83 to 1.00 and said .58 to .77 word by word, delivered .16 to .46 and said .59 to .89
  at the phrase window: with the window the verb is settled with its agent and no longer waits
  on a read of its own.
- **W2 fails**: no Dyck sentence of a held-out depth that fits the whole-sentence window is
  said right on any founder; at that window every position is a coordinate the brain has to
  learn apart, and the trained Dyck sentences read .31 there.
- **W3**: the continuations of the twins differ (the message is heard: the agent twins 6 of 6,
  the verb twins 4 of 6), and a window is both right at .90 on no founder.
- **Lengths beyond the window** are lost at every window on every founder (W4, .00), as the
  first form predicts without weight sharing across positions.
- **The controls** at the brain's parameter count say every panel the gates read at 0.90 or
  above (held-out agreement 1.00, the window twins both right .90 to 1.00), online in 6 to 8
  seconds at 5.7e10 multiply-accumulates, a sixteenth of the brain's work, and they fail the
  held-out depths as the brain does (.00 to .09). Per #183 no reading here is for Cadence.

A bounded negative at this size and life: the planning window's first form does what #257
states, the verb settled with its agent inside one settle, and does not reach the bars on
fresh founders. Weight sharing across positions, which lets a window generalize to lengths
beyond it, is the second form #257 names.

## Readings on 0.94.0 (development, founders 0 to 2)

Development founders 0, 1 and 2, from frozen snapshots of this folder; the maturity readings
are founder 0's. These readings are to be taken again on the release that closes the Sequences
stage of #173. Entries are whole sentences said exactly, closed loop, as the mean over the
founders with the least and greatest in brackets (`roles`: both twins of a pair right).

**The first grammar after 4,000 lesson batches.**

| Arm or control | `clause` | `clause_held` | `clause_long` | `roles` | `dyck` | `dyck_deep` | `path` | `path_long` | next word |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| the newborn | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .05 |
| `founder` | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .00 | .23 |
| `recipe` (= `maturity`) | .73 (.67-.78) | .61 (.50-.67) | .00 | .45 (.28-.56) | .27 (.12-.39) | .03 (.00-.09) | .27 (.20-.31) | .00 | .94 |
| LSTM, online | .97 | .95 (.94-.97) | .07 (.03-.11) | .88 (.81-.91) | .74 (.55-.89) | .02 (.00-.06) | .27 (.22-.30) | .00 | .99 |
| transformer, online | 1.00 | .99 (.98-1.00) | .00 | 1.00 | .35 (.25-.45) | .00 | .53 (.27-.95) | .00 | 1.00 |
| LSTM, offline (30 epochs) | 1.00 | .99 (.98-1.00) | .11 (.09-.12) | .99 (.97-1.00) | 1.00 | .11 (.05-.19) | .75 (.72-.78) | .01 (.00-.03) | 1.00 |
| transformer, offline (16 to 29 epochs) | 1.00 | 1.00 | .00 | 1.00 | 1.00 | .00 | 1.00 | .00 | 1.00 |

- **The founder** says one fixed word sequence whatever the meaning (`[ ] [ ] ) end` on
  founder 0 after the first grammar), the composer's prime-independent collapse of
  [#140](https://github.com/muellerberndt/cadence/issues/140).
- **The recipe** says clauses at trained lengths (.89, .79 and .35 at zero, one and two
  distractors), places the roles from the role flags in either order heard (.72 inverted, .75
  canonical; roles right .96) and composes held-out combinations (.61). Its next-word accuracy
  (.94) is near the controls' while its sentences are not: the component reading overstates
  the sentence.
- **Agreement**: .82 (.91 without an attractor of the other number among the distractors, .66
  with one); of the verbs that follow a noun of the other number than the agent's, .19 to .44
  agree with that noun (`attracted`). The controls agree at 1.00 at trained lengths and at .68
  to .95 at three and four distractors (the recipe .49).
- **Held-out lengths, depths and hops fail for every learner at this size**: three and four
  distractors (means at most .11), Dyck depths four and five (at most .11), three hops (at
  most .01).
  The recipe says one or two distractor phrases and then the verb (`owl near owl sees dogs end`
  for three).
- **Two hops**: the recipe says one-hop paths .52 and two-hop .01; the converged transformer
  says both 1.00, so the information suffices at this size.

**Where the sentence is lost** (`located` on the saved brains after the first grammar, the
recall read's first head, teacher-forced; delivered / said):

| Word | Delivered | Said | Said when delivered |
| --- | --- | --- | --- |
| clause nouns (agent, distractors, patient), zero to two distractors | .92 to 1.00 | .75 to 1.00 | .78 to 1.00 |
| the verb, zero / one / two distractors | .90 / 1.00 / .98 | .91 / .88 / .60 | .90 / .88 / .62 |
| the second, third distractor at three distractors | .36, .35 | .22, .17 | .20, .23 |
| the verb at three / four distractors | .01 / .19 | .23 / .07 | .00 / .13 |
| a Dyck close, depth two / three | .47 / .33 | .97 / .91 | 1.00 / 1.00 |
| a Dyck close, depth four / five | .12 / .09 | .54 / .44 | .89 / .94 |
| a path's hop, one hop / two hops / three hops | .47 / .25 / .20 to .26 | .58 / .27 to .39 / .27 to .28 | .74 / .41 to .61 / .70 to .81 |

At trained lengths the read lands on the meaning's slot of every noun the clause says and the
brain says it. The verb at two distractors is delivered .98 and said .60: the read brings the
verb's concept, the agent's number has to come from elsewhere, and agreement is lost at the
readout. At three and four distractors the read loses its place among the distractor slots and
the verb is no longer delivered: the failure of held-out lengths is at the read. A path's hop
is mostly said when it is delivered and mostly not delivered: two-hop composition fails at the
read's selection of the fact, the first failure of the 20 October facts world.

**The context.** Twins that share their recent words and differ in an earlier part (both twins
right / continuations differ):

| Reading | Agent's number | The meaning's verb | The first bracket |
| --- | --- | --- | --- |
| recipe | .29 (.16-.44) / .41 | .52 (.50-.56) / 1.00 | .55 (.28-.72) / 1.00 |
| the earlier context erased from the recall memory | .00 / .23 | .00 / .07 | .00 / .15 |
| the working trace erased | .25 / .42 | .46 / 1.00 | .31 / .92 |
| the context removed | .00 / .00 | .00 / .00 | .00 / .00 |
| LSTM online | .95 / 1.00 | .98 / 1.00 | 1.00 / 1.00 |
| transformer online | 1.00 / 1.00 | 1.00 / 1.00 | .54 / .95 |

The meaning's verb is carried across the distractors by the recall memory: erased there, the
continuations stop differing (.07) and the verb's identity is lost (founder 0: .14, 1.00
intact); erased from the trace, the continuations still differ (1.00). The Dyck closes lean on
both.

**Persistence and recovery** (lessons; the second grammar for 1,500 batches, then the first
for 500):

| Reading | `clause` | `dyck` | `path` | context, agent | `g2` |
| --- | --- | --- | --- | --- | --- |
| recipe, after the first grammar (a frozen copy reads this throughout) | .73 | .27 | .27 | .29 | .00 |
| recipe, after the second | .00 | .03 | .03 | .00 | .69 (.58-.77) |
| recipe, after 500 batches of the first | .72 (.64-.80) | .43 | .17 | .51 | .00 |
| LSTM online, after the second / recovery | .00 / .99 | .59 / .82 | .20 / .28 | .00 / .99 | .99 / .00 |
| transformer online, after the second / recovery | .00 / 1.00 | .02 / .36 | .21 / .55 | .00 / 1.00 | 1.00 / .00 |

The second grammar erases the first in the brain and in both controls although its meanings
carry `DIALECT`: after it every meaning is said in the second grammar's order (`owl owls sees
end`), and the brain loses its Dyck closes and paths with it. An eighth of the first grammar's
lessons brings the clauses back (.72 against .73), as in the controls.

**Maturity** (founder 0). Lessons never move maturity's use (lessons keep their steps by design),
so at the founder scale maturity is the recipe under lessons: the `maturity` life equals the
`recipe` life in every reading and work count. Paid practice is where it acts, and paid
practice through `Brain.step` (the actor at its founder rate 1.0, lessons at 0.003) unlearns
the lesson-taught grammar: 300 paid batches of the first grammar take the clauses from .78 to
.05, with the actor's steps off (critic only) they stay at .78, with credit to the last action
only .13, at temperature 0.1 .11. On a maturity brain of the same lessons, 300 paid batches:

| `maturity_scale` | use at the end | step factor | `clause` | `dyck` | `path` |
| --- | --- | --- | --- | --- | --- |
| 10 (the founder) | .68 | 1.00 | .22 | .50 | .13 |
| 1 | .56 | .91 | .22 | .31 | .09 |
| 0.1 | .19 | .08 | .58 | .56 | .16 |
| 0.01 | .03 | .01 | .58 | .19 | .25 |

At its founder scale maturity does not engage in this life. Engaged within the practice it
keeps more of the clauses (.58 against .22), after the first paid batches have already cost
them. This is a finding for the retention work of
[#169](https://github.com/muellerberndt/cadence/issues/169) and
[#238](https://github.com/muellerberndt/cadence/issues/238) (paid practice unlearns what lessons
taught), not a setting this chamber selects. The persistence schedule's forks that acquire the
second grammar by paid practice began after 1,000 paid batches had already taken the first
grammar to .06, so they read nothing about maturity; their receipts are kept with the
development evidence.

**Work** (the first grammar, recipe, per founder): 23,264 to 23,345 lessons (about 373,000
rows, 300,577 to 307,860 of them already answered right and not taught), 2.42 to 2.57 million
row sweeps in lessons and 3.93 to 4.27 million batch sweeps in acts, 367 to 496 seconds on one
shared core; at one multiply-accumulate per synapse (13,120) per row and sweep, about 9e11. The
online controls: 4,000 Adam steps, 5.7e10 multiply-accumulates, 14 to 28 seconds; the offline
ones 9.3e11 to 1.7e12, 6 to 11 minutes. At a sixteenth of the brain's work the online controls
say as many sentences right as the brain or more on every trained-length panel; no learner
generalizes to held-out lengths.
The second grammar: about 9,200 lessons; the recovery: about 2,900. No act or lesson was
refused; about 30 lessons of the first grammar used their whole free budget. A saved and
loaded copy continues exactly as the live one (`continuation`, every life).

## State across moments

Which state carried from one moment to the next lifts the brain, read on the base composition of
#263: the [recipe](../../docs/memory.md#one-composition-for-sequences) at `teaching="deep"`, 64
association neurons, 4,000 batches of the first grammar. Development. Twenty-four lives per arm where
marked (founders 0 to 2 by their own seed, seeds 1000 to 1002 and 2000 to 2003), twelve otherwise,
noise floor about .07. A trace at 3.0 is the founder's, at 0.3 `working_memory_amplitude=0.3`, silent at
0 with its synapses. A learned gain is `carried_gain=0.01`.

| Trace | Added | Lives | Clauses | Two distractors | Held-out | Roles | Brackets | Paths | Second hop | Three hops | Parameters | Sweeps per act |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.0 | | 24 | .95 | .91 | .90 | .87 | .72 | .88 | .82 | .18 | 24,384 | 68 |
| 0.3 | | 24 | .98 | .94 | .95 | .95 | .87 | .96 | .94 | .46 | 24,384 | 84 |
| 0.3 | learned gain | 24 | .98 | .95 | .93 | .94 | .83 | .95 | .92 | .44 | 24,385 | 80 |
| 3.0 | learned gain | 12 | 1.00 | .99 | .95 | .97 | .85 | .98 | .96 | .42 | 24,385 | 77 |
| silent | | 12 | .94 | .86 | .90 | .87 | .86 | .94 | .89 | .34 | 24,384 | 93 |

Width, three lives each, whole steps (clauses, held-out, roles, brackets, paths, three hops):

| Arm | 128 | 256 |
| --- | --- | --- |
| Trace 3.0 | .92 .88 .82 .42 .54 .06 | .62 .58 .41 .15 .25 .00 |
| Trace 3.0, learned gain | 1.00 .98 1.00 .87 .79 .32 | 1.00 .99 .99 .86 .97 .47 |
| Trace 0.3 | .98 .96 .98 .89 .97 .16 | .97 .96 .95 .95 .94 .31 |
| Trace silent | .97 .96 .93 .76 .93 .35 | .97 .96 .92 .63 .93 .60 |

Parameters at 128 and 256: 52,480 and 133,248, and one more with the learned gain. At 128 with the
[fan-in steps](../../docs/tune.md#wide-brains) the trace at 0.3 reads .99 .69 .79 .68 .89 .34, below the
whole steps. The matched controls online at about 52,000 parameters, the same exposure: the transformer
.99 .99 1.00 .21 .34 .00 in 33 seconds, the LSTM .91 .92 .83 .74 .30 .00 in 17, where a brain's life takes
450 to 860 seconds; offline the transformer reads 1.00 on the trained panels.

Passes of the same lessons, six lives (founders 0 to 2 on their own seed and on 1001; clauses, two
distractors, held-out, roles, brackets, paths, second hop, three hops):

| Arm | Rate | Pass 2 | Pass 4 |
| --- | --- | --- | --- |
| Trace 3.0 | full | .99 .97 .97 .98 .73 .82 .72 .13 | .99 1.00 .96 .97 .87 .72 .64 .03 |
| Trace 0.3 | full | .99 .97 .93 .96 .80 .92 .84 .36 | 1.00 .99 .97 .98 .92 .87 .79 .22 |
| Trace 0.3, learned gain | full | .98 .95 .97 .98 .82 .96 .94 .38 | |
| Trace 0.3 | a third | .98 .95 .94 .92 .41 .90 .81 .29 | .99 .97 .93 .96 .77 .94 .89 .54 |

What carries the answer, with the trace emptied before every act on the taught life (clauses, held-out,
roles, brackets, paths): the trace at 3.0 .38 .29 .20 .40 .72, at 0.3 .75 .69 .64 .59 .95.

These lives say which state lifts this chamber, and none of this is read on fresh founders. Two forms
read beside them stay out of the library because nothing chosen uses them: a brain built with no trace,
and the read carried to the next moment on a region of its own.

## A held meaning

[`meaning.py`](meaning.py) reads [#286](https://github.com/muellerberndt/cadence/issues/286): where the
brain keeps a sentence's meaning between hearing it and saying it, and whether the sentence can be
said from a state the brain holds, with the recall memory's slots of the meaning emptied. The
readings follow [MEANING_PROTOCOL.md](MEANING_PROTOCOL.md), declared before the run on fresh
founders. The library's part is [`cadence.Message`](../../docs/memory.md#a-held-message), a class
with no neurons. No brain changes: every arm is the default composition
(`Brain.compose(35, 18, modules=(64,), seed=founder, recall=True)`) with what the arm names.

```sh
PYTHONPATH=src python benchmarks/grammar/meaning.py where --founders 0 1 2 --out DIR/where.jsonl --workers 3
PYTHONPATH=src python benchmarks/grammar/meaning.py plan --out DIR/plans
PYTHONPATH=src python benchmarks/grammar/meaning.py lives --plan DIR/plans/brains.json --out DIR/brains.jsonl --workers 36
(cd benchmarks/grammar && PYTHONPATH=../../src python controls.py --plan DIR/plans/controls.json --out DIR/controls.jsonl --workers 16)
PYTHONPATH=src python benchmarks/grammar/meaning.py lives --plan DIR/plans/knock.json --out DIR/knock.jsonl --workers 12
PYTHONPATH=src python benchmarks/grammar/meaning.py retrieval --out DIR/retrieval.jsonl
PYTHONPATH=src python benchmarks/grammar/meaning.py read DIR/brains.jsonl --controls DIR/controls.jsonl --knock DIR/knock.jsonl
```

### Where the meaning lives in the default composition

`meaning.py where` teaches one life per founder the first grammar and reads it three ways.

- **Whole sentences under an erasure at `GO`.** `state`: every recall slot written before `GO` is
  emptied (keys, values and nexts at zero), so the sentence can come from the carried state alone.
  `silent`: the read's drive is held at zero from `GO` to `end`. `no_trace`: the working trace is
  emptied at `GO`, so the sentence comes from the read alone. `neither`: both.
- **Probes.** A linear read-out per component of the meaning, fitted offline on 70% of fresh
  sentences and scored on the rest, from the working trace entering a settle (`carried`), the
  association after it (`settled`) and what the recall read holds on its ports (`read`). They read
  arrays and change nothing.
- **The controls' states.** The chamber's LSTM and transformer at the brain's parameter count, one
  online pass, read on the same panels and probed at the same moments; the LSTM with its state
  emptied at `GO` and the transformer with the meaning's positions masked from `GO` on are their
  floors.

Development founders 0 to 5, whole sentences said exactly (`roles`: both twins), mean with least
and greatest:

| Speaker | `clause` | `clause_held` | `roles` | `dyck` | `path` |
| --- | --- | --- | --- | --- | --- |
| Brain, intact | .98 (.94-1.00) | .91 (.81-.97) | .96 (.88-1.00) | .88 (.53-1.00) | .95 (.92-1.00) |
| Brain, every slot emptied at `GO` | .00 | .00 | .00 | .00 | .00 |
| Brain, the read silent from `GO` | .00 | .00 | .00 | .00 | .00 |
| Brain, the working trace emptied at `GO` | .98 | .93 | .97 | .87 | .95 |
| Brain, a path's start slot emptied; its facts' slots emptied | | | | | .12; .01 |
| LSTM | .95 (.80-1.00) | .92 (.83-.98) | .84 (.53-1.00) | .74 (.50-1.00) | .30 (.23-.34) |
| LSTM, its state emptied at `GO` | .00 | .00 | .00 | .07 | .00 |
| Transformer | .99 | 1.00 | 1.00 | .56 (.31-.91) | .29 |
| Transformer, the meaning masked from `GO` | .00 | .00 | .00 | .04 | .00 |

Probes at `GO` (held-out accuracy, mean of six founders):

| Component | Chance | Brain, carried | Brain, the read | LSTM, carried |
| --- | --- | --- | --- | --- |
| Agent's noun | .24 | .73 | 1.00 | 1.00 |
| Agent's number | .49 | .84 | 1.00 | 1.00 |
| Verb | .51 | .85 | 1.00 | 1.00 |
| Patient's noun | .25 | .67 | .51 | .99 |
| Number of distractors | .32 | .93 | .95 | 1.00 |
| First distractor's noun | .26 | .72 | .88 | .99 |
| Second distractor's noun | .28 | .55 | .30 | .94 |
| Brackets: number open | .53 | .86 | .78 | 1.00 |
| Brackets: innermost open | .51 | .97 | .69 | .98 |
| Path: the start | .12 | .99 | 1.00 | 1.00 |

The sentence is looked up: with the slots emptied every panel reads .00, and with the working trace
emptied nothing moves. The carried state holds a partial, fading copy of the meaning that the
readout does not use: a path's start is decodable from it at .99, and with the start's slot emptied
paths fall from .95 to .12. The LSTM's state holds every component at .94 to 1.00, and it says the
panels from that state.

### A message the instrument writes

`meaning.py lives` runs lives with a message: after every act the instrument writes a code of the
episode so far onto neurons the association reads through a plastic projection (heard-history
blocks), and with `bar` it empties the recall slots of the meaning's moments at `GO`, in lessons and
in readings, so the sentence can only be said from what the brain carries. The write is fixed, as
the working trace's and the heard history's are, and what the brain makes of the message is
learned. `Message` in `meaning.py` holds the designs read in development; the declared design is
the library's `cadence.Message`.

Development, founders 0 to 2 (0 to 5 where six lives are named), the read barred in every lesson:

| What the brain carries | Clauses from the state alone | Held-out clauses | What fails |
| --- | --- | --- | --- |
| The default composition, nothing added | .02 | .00 | the working trace is not used as a message |
| The working trace at decay .8 | .02 | .01 | the same |
| The heard history at two decays | .31 | .11 | a sum of raw moments binds no noun to its role |
| Sparse codes of the moments, summed, with a primacy or a recency sum | .69 to .77 | .47 to .54 | the first of two distractors is said at .33 to .65: order is not read from magnitudes |
| A buffer of the raw moments | .59 | .25 | binding |
| Phased unions of sparse codes, 70 units a block | .88 | .73 | held-out combinations |
| The same, 128 association neurons, every row taught | .91 to .97 | .63 to .74 | held-out combinations |
| The same, 280 units a block | .98 | .89 (.81-.98) | |
| The message restated in motor slots, carried by the efference copy | .50 to .92 | .03 to .18 | slots settled together bind what was taught together |

Three properties of a message the brain can say a sentence from follow:

- **Bound moments.** A sparse code whose units read several active inputs at once holds a noun with
  its number and its role; a sum or a buffer of the raw moments does not.
- **Presence in place of magnitude.** Which of two moments of one kind came first is said where the
  two lie in different blocks, and is not said where they differ in weight.
- **Width.** A code of 70 units shares units between a noun as agent and the same noun as patient,
  and a readout taught on some pairs confuses new ones. At 280 units a block held-out clauses rise
  from .63 to .74 to .89.

### The declared run

Founders 9628600 to 9628605, one c7i.16xlarge, one online pass of the first grammar's 4,000 batches
per life, no refused lesson and no exhausted act. `state` is the message at 280 units a block with
the read barred from the meaning, 128 association neurons and every row taught, and `state_256` (the
protocol's second amendment, declared after the capacity arms were read) the same with 256
association neurons; `message` is the message at 70 units a block beside a free read; `sham` holds the code of the moment before alone on
the message's neurons; `heard` (the protocol's first amendment) is the default composition with the
library's heard history at decay 0, `history_decays=(0.0,)`. Whole sentences said exactly, mean
with least and greatest; the controls take one online pass at the arm's parameter count.

| Arm or control | Parameters | `clause` | `clause_held` | `roles` | `dyck` | `path` | Three hops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | 28,716 | .97 (.88-1.00) | .90 (.72-1.00) | .91 (.81-1.00) | .84 (.69-1.00) | .94 (.84-.98) | .36 (.02-.61) |
| `barred`, the floor | 28,716 | .01 | .00 | .00 | .11 | .19 | .01 |
| `state`, the read barred | 203,692 | .98 (.94-1.00) | .84 (.61-.98) | .81 (.62-.88) | .85 (.64-1.00) | .57 (.38-.72) | .00 |
| `state_256`, the read barred | 432,556 | .99 (.97-1.00) | .95 (.86-.98) | .95 (.84-1.00) | .89 (.64-1.00) | .56 (.36-.86) | .00 |
| `state_default`, the read barred | 101,548 | .92 (.75-.98) | .80 (.64-.91) | .75 (.53-.94) | .67 (.44-.92) | .35 (.28-.42) | .00 |
| `restated`, the read barred from clauses | 73,223 | .76 (.59-.81) | .11 (.02-.17) | .03 (.00-.09) | .55 (.00-.89) | .06 (.03-.09) | .00 |
| `message` | 46,924 | 1.00 (.98-1.00) | .93 (.78-1.00) | .95 (.88-1.00) | .96 (.89-1.00) | .84 (.23-.98) | .12 (.00-.30) |
| `message_wide` | 101,548 | .98 (.94-1.00) | .94 (.75-1.00) | .92 (.62-1.00) | .96 (.88-1.00) | .83 (.30-1.00) | .01 |
| `sham` | 46,924 | .98 (.89-1.00) | .93 (.72-1.00) | .97 (.91-1.00) | .88 (.78-1.00) | .98 (.95-1.00) | .67 (.23-.84) |
| `heard` | 30,992 | .99 (.97-1.00) | .96 (.89-1.00) | .98 (.91-1.00) | .81 (.64-.92) | .99 (.97-1.00) | .74 (.67-.81) |
| LSTM, online | 28,398 | .96 (.80-1.00) | .94 (.80-.98) | .90 (.78-.97) | .80 (.50-1.00) | .27 (.17-.34) | .00 |
| LSTM, online | 203,958 | .98 (.91-1.00) | .97 (.92-1.00) | .89 (.81-1.00) | .79 (.47-1.00) | .28 (.22-.36) | .00 |
| Transformer, online | 28,658 | 1.00 | 1.00 | 1.00 | .55 (.41-.89) | .32 (.22-.41) | .00 |
| Transformer, online | 201,978 | .99 (.97-1.00) | .99 (.94-1.00) | .98 (.94-1.00) | .32 (.09-.64) | .26 (.17-.33) | .00 |
| LSTM, online, the widest its search holds | 304,658 | .95 (.72-1.00) | .95 (.84-.98) | .89 (.66-.97) | .89 (.77-1.00) | .29 (.20-.41) | .00 |
| Transformer, online | 432,234 | .87 (.28-1.00) | .85 (.28-1.00) | .80 (.06-1.00) | .21 (.00-.50) | .13 (.06-.23) | .00 |
| LSTM, offline (30 epochs) | 28,398 | 1.00 | .98 (.89-1.00) | .95 (.81-1.00) | .99 | .73 (.53-.86) | .00 |
| Transformer, offline (13 to 14 epochs) | 28,658 | 1.00 | .99 (.92-1.00) | .99 (.94-1.00) | 1.00 | 1.00 | .00 |

**The gates** (a gate on lives holds where 5 of 6 founders reach its bar):

| Gate | Reading by founder | Verdict |
| --- | --- | --- |
| H1, said from the state: `clause` at .90 | 1.00 .98 1.00 1.00 .97 .94 | 6 of 6, holds |
| H1: `clause_held` at .90 | .80 .83 .86 .98 .61 .94 | 2 of 6, fails |
| H1: `roles`, both twins, at .90 | .88 .88 .81 .84 .62 .81 | 0 of 6, fails |
| H1 on `state_256`: `clause` | 1.00 1.00 1.00 1.00 1.00 .97 | 6 of 6, holds |
| H1 on `state_256`: `clause_held` | .98 .92 .98 .97 .86 .98 | 5 of 6, holds |
| H1 on `state_256`: `roles`, both twins | 1.00 1.00 .84 .94 .97 .97 | 5 of 6, holds |
| H2, held: the least component's probe from the message, at `GO` and at the last word, at .95 | 1.00 on every founder at both | holds |
| H2u, used: the least clause word kind said right, at .95 | .96 .96 .95 .98 .95 .94 | 3 of 6, fails |
| H2u on `state_256` | .99 .98 .99 .98 .98 .98 | 6 of 6, holds |
| H3: the key of one meaning's two orders closer than any other meaning's, at .95 | 1.00 on every founder | holds |
| H3: both orders said right, at .90 | 1.00 1.00 .98 .97 .92 .95 | 6 of 6, holds |
| H4, paraphrase from the state: trained combinations at .80 | .94 .95 1.00 1.00 .81 .94 | 6 of 6, holds |
| H4: held-out combinations at .80 | .94 .88 .75 .91 .72 .86 | 4 of 6, fails |
| R, retrieval by the code at 256 records, .90: a reordered cue; a role-swapped cue | 280 units: .97 to 1.00; 70 units: .77 to .96 | holds at 280 units (6 of 6), fails at 70 (3 and 4 of 6) |
| A1 on `default`: `clause_held`, `roles`, `ctx_agent`, `ctx_verb`, `ctx_dyck` at .90 | 4, 3, 5, 5 and 4 of 6 | fails |
| A1 on `message` | 5, 5, 6, 6 and 6 of 6 | holds |
| A1 on `heard` | 4, 6, 6, 6 and 4 of 6 | fails |
| A2: `clause_long`, `dyck_deep`, `path_long` at .80 | no life of any arm or control | fails, a measured negative |
| A3: after the second grammar, within .05 of before | `clause` .00 for every arm and control | fails |
| M1, `message` less `default`, paired | brackets +.12 (s.e. .04); three hops -.24 (.10); every other panel inside two standard errors | holds on brackets alone |
| M1, `sham` less `default` | roles +.06 (.03), three hops +.31 (.09) | holds on both |
| M1, `heard` less `default` | roles +.07 (.02), paths +.05 (.02), three hops +.38 (.07), held-out clauses +.05 (.04) | holds on roles, paths and three hops |
| K, the message on other chambers | look back -.023 (.014), copying -.011 (.004), repeated tokens +.017 (.004), induction -.017 (.020), n-back -.003, two hops +.001, a period of 40 .000, the goal route's first hop -.016 (.054) | no fall beyond .03 |
| K at 256 association neurons, the wide message (the third amendment) | the chamber: paths -.40 (.11), three hops -.27 (.10), every other panel inside two standard errors. The battery: look back -.020 (.013), copying -.014 (.002), repeated tokens +.002, induction -.029 (.012), n-back -.003, two hops +.003, a period of 40 .000, the goal route's first hop -.029 (.043) | paths and three hops fall |

**Said from the state alone.** With the read barred from the meaning the brain says clauses from the
message at .98, as the LSTM at its parameter count does online (.98), brackets at .85 against .79,
and paths at .57 against .28, the facts read from the memory and the start from the message. On new
combinations the arm at 128 association neurons reads under the LSTM: held-out clauses .84 against
.97, and both role twins .81 against .89. At 256 association neurons H1 holds: clauses .99, held-out
clauses .95 and both role twins .95, on 6, 5 and 5 founders of 6, beside an LSTM of 304,658
parameters at .95, .95 and .89 and a transformer of 432,234 at .87, .85 and .80 (one of its lives
at .28). With the message emptied at the continuation's start the context twins read .00 to .09 in
both arms: the state is what the sentence is said from.

**Held, and where the readout stops.** Every component of a clause's meaning is decodable from the
message at 1.00, at `GO` and at the last word. The carried trace holds less (the second distractor's
noun .60, the patient's .72) and the settled association at `GO` holds .87 to 1.00. Every clause
word kind is said right at .97 to 1.00 with the previous words given. At 128 association neurons the
fall is on combinations no lesson showed: on `clause_held` the verb's agreement reads .89 and the
distractors .94, on `roles` the patient .93. A sum of codes shares units between a noun as agent and
the same noun as patient, and a narrow readout that tells a taught pair apart confuses a new one.
The read does not have this term: it brings one moment at a time. At 256 association neurons the
term closes (agreement .97, the patient .98; held-out clauses +.11 and role twins +.15 over 128,
paired, s.e. .04).

**One meaning in two orders.** The key is the same for the canonical and the inverted order (cosine
1.00, against .40 for different meanings), and both orders are said right at .97. The phased state
keeps the order heard (cosine .33 between the two orders), and the settled association at `GO` reads
.90 between the two orders against .63 for different meanings.

**Paraphrase.** Hearing the first grammar's sentence as words and saying the second grammar's from
the state alone, with no concept moment and the read barred, reads .94 on combinations the lessons
showed (.99, .97 and .80 at zero, one and two distractors) and .84 on held-out ones. The default
composition taught the same lessons with its read free reads .99 and .95.

**Capacity.** Lives taught clauses of zero to four distractors alone, said exactly from the state,
by distractors:

| Message, association | Parameters | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 blocks of 70, 128 | 95,308 | .93 | .90 | .85 | .66 | .30 |
| 4 blocks of 140, 128 | 131,436 | .98 | .96 | .95 | .91 | .61 |
| 4 blocks of 280, 128 | 203,692 | .98 | 1.00 | .98 | .97 | .66 |
| 4 blocks of 280, 64 | 101,548 | .97 | .98 | .93 | .83 | .30 |
| 4 blocks of 280, 256 | 432,556 | 1.00 | 1.00 | 1.00 | 1.00 | .94 |

The code's width and the association's width each carry part: at 128 association neurons clauses
of three distractors rise from .66 to .97 with the code's width, and at 280 units a block clauses of
four distractors rise from .30 to .94 with the association's. A clause of four distractors holds
seven moments in four blocks, each block one distractor at most, so their order stays present.

**Retrieval by the code** (`meaning.py retrieval`, no brain; every clause stored beside its
role-swapped twin; the share of cues whose best record by cosine is the right one, a tie a miss):

| Cue | Units | 16 records | 64 | 256 | 1,024 |
| --- | --- | --- | --- | --- | --- |
| The meaning's concept moments in the other order | 70 | 1.00 | .98 | .89 | .65 |
| | 280 | 1.00 | 1.00 | .98 | .91 |
| The role-swapped twin's meaning (its own record, not the original's) | 70 | 1.00 | .97 | .90 | .64 |
| | 280 | 1.00 | 1.00 | .98 | .91 |
| The meaning without its patient (a record that agrees with the cue) | 280 | 1.00 | .97 | .93 | .81 |
| The sentence heard as words in the other grammar's order, against heard sentences | 280 | .48 | .53 | .49 | .20 |
| A sentence heard as words against records of concept moments | 280 | .02 | .01 | .00 | .00 |

The misses of a reordered cue at 280 units are ties with a stored clause that holds the same
distractors in the other order or one distractor twice (every miss inspected on a development
founder): the key holds no order and no repetition.
A word's code holds no role, so a heard sentence and its role-swapped twin give one key, and the
code of a word and the code of the concept it names are unrelated draws.

**Beside a free read.** With the read free no arm says a sentence from the message (clauses .01 to
.02 with the read barred at the reading), and the message is used: emptied at the continuation's start, the
`message` arm's context twins fall from .99, 1.00 and 1.00 to .05, .12 and .31. It lifts brackets
(+.12) and holds A1's five lines where `default` holds two; it costs three hops (.36 to .12) and
one life's paths (.23). The episode's length is what costs: a path's episode holds twelve to sixteen
heard words, and each block is a union of three or four of them. The moment before alone lifts what
the whole episode costs: `sham` reads three hops at .67 and `heard`, an option the library holds, at
.74, with roles at .98 and paths at .99.

**The wide message where nothing is barred** (the third amendment). At 256 association neurons the
message at 280 units a block and the state arm's drive, beside a free read, reads paths at .44
against .85 without it and three hops at .00 against .27; clauses, held-out clauses, roles and
brackets move by +.01 to +.05, inside two standard errors. On the sequence battery every task reads
within .03 of the brain without it. The message that H1 is said from is a cost on paths and hops
where the read is free: it suits the turn in which a sentence is said from it.

**Lengths and persistence.** No learner says `clause_long`, `dyck_deep` or `path_long` at .80.
`located` on `default` names the read: at three distractors the second distractor is delivered on
.25 of sentences and the verb on .09, at depths four and five a close's open bracket on .11 to .12,
and the third hop's fact on .65. After 1,500 batches of the second grammar every arm and control
says the first grammar's clauses at .00 and the second's at .97 to 1.00; 500 batches of the first
bring the brains' clauses back (.97 to .99).

**Work.** The first grammar, per life, analytic multiply-accumulates (every lesson row sweep and
every act's sweeps times its rows, over every synapse): `default` 9.5e11, `heard` 1.1e12, `message`
2.1e12, `state` 1.3e13, `state_256` 2.5e13; the controls online 6.7e10 at 28,000 parameters and
4.9e11 at 204,000, offline 1.1e12 to 2.0e12 and 9.1e12 to 1.2e13. A `heard` life takes 69 seconds
and a `state_256` life 787 on an idle core; the first block's lives ran on a shared box (a
`default` life 215 seconds, a `state` life 1,093).

### The brain's own message

[MEANING_PROTOCOL_2.md](MEANING_PROTOCOL_2.md), declared before its run on founders 9728600 to
9728605, reads the message as something the brain holds and writes itself:
[`Brain.compose(message=units)`](../../docs/memory.md#held-message), a region of the brain's own
neurons where each heard moment's fixed sparse code latches onto the block whose turn it is, read by
the association through plastic synapses with one learned gain per block, and with `message_read`
a read of one block at a time. Every arm that holds a message has its previous-word control beside
it, the same life with `history_decays=(0.0,)` in the message's place. The read is barred from the
meaning in `state_*`, `read_64`, `direct_64`, `heard_*`, `paraphrase*` (except `paraphrase_read`)
and `capacity_*`; the other arms have it free.

```sh
PYTHONPATH=src python benchmarks/grammar/meaning.py plan --own --out DIR/plans
PYTHONPATH=src python benchmarks/grammar/meaning.py lives --plan DIR/plans/brains.json --out DIR/brains.jsonl --workers 26
(cd benchmarks/grammar && PYTHONPATH=../../src python controls.py --plan DIR/plans/controls.json --out DIR/controls.jsonl --workers 6)
PYTHONPATH=src python benchmarks/grammar/meaning.py read DIR/brains.jsonl --controls DIR/controls.jsonl --own
```

Whole sentences said exactly (roles: both twins), mean and least to greatest over six founders;
the controls after one pass (online) or to the stopping rule (offline), at the parameter counts of
`default`, `read_64` and `state_256`:

| Arm | Parameters | Lives | clause | clause_held | roles | dyck | path | three hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `heard` | 30992 | 6 | 0.98 (0.89 to 1.00) | 0.89 (0.67 to 1.00) | 0.93 (0.78 to 1.00) | 0.80 (0.44 to 1.00) | 0.98 (0.95 to 1.00) | 0.71 (0.48 to 0.86) |
| `default` | 28716 | 6 | 0.97 (0.94 to 1.00) | 0.93 (0.84 to 0.98) | 0.89 (0.66 to 1.00) | 0.80 (0.72 to 0.92) | 0.91 (0.77 to 0.98) | 0.26 (0.00 to 0.47) |
| `barred` | 28716 | 6 | 0.01 (0.00 to 0.03) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.12 (0.00 to 0.25) | 0.11 (0.03 to 0.14) | 0.00 (0.00 to 0.00) |
| `read_64` | 58870 | 6 | 0.89 (0.72 to 0.95) | 0.72 (0.22 to 0.88) | 0.74 (0.34 to 0.91) | 0.70 (0.55 to 0.95) | 0.35 (0.27 to 0.39) | 0.00 (0.00 to 0.00) |
| `direct_64` | 46920 | 6 | 0.79 (0.59 to 0.94) | 0.47 (0.20 to 0.81) | 0.30 (0.09 to 0.50) | 0.83 (0.70 to 1.00) | 0.34 (0.28 to 0.42) | 0.00 (0.00 to 0.00) |
| `state_128` | 203664 | 6 | 0.97 (0.91 to 1.00) | 0.92 (0.86 to 0.97) | 0.86 (0.69 to 1.00) | 0.86 (0.72 to 1.00) | 0.46 (0.31 to 0.73) | 0.00 (0.00 to 0.00) |
| `heard_256` | 153680 | 6 | 0.07 (0.05 to 0.12) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.35 (0.25 to 0.58) | 0.67 (0.28 to 0.91) | 0.00 (0.00 to 0.00) |
| `heard_64` | 30992 | 6 | 0.04 (0.02 to 0.06) | 0.00 (0.00 to 0.02) | 0.00 (0.00 to 0.00) | 0.27 (0.17 to 0.42) | 0.58 (0.33 to 0.77) | 0.00 (0.00 to 0.00) |
| `message_quiet` | 46920 | 6 | 1.00 (1.00 to 1.00) | 0.95 (0.88 to 1.00) | 0.97 (0.88 to 1.00) | 0.95 (0.86 to 1.00) | 0.95 (0.89 to 1.00) | 0.05 (0.00 to 0.12) |
| `message_free` | 101520 | 6 | 0.98 (0.91 to 1.00) | 0.92 (0.81 to 1.00) | 0.93 (0.84 to 1.00) | 0.81 (0.69 to 0.94) | 0.65 (0.27 to 0.97) | 0.00 (0.00 to 0.00) |
| `message_fixed` | 46914 | 6 | 0.99 (0.97 to 1.00) | 0.96 (0.91 to 1.00) | 0.97 (0.88 to 1.00) | 0.88 (0.69 to 0.97) | 0.93 (0.77 to 1.00) | 0.20 (0.05 to 0.39) |
| `default_fixed` | 28714 | 6 | 0.85 (0.56 to 1.00) | 0.78 (0.42 to 0.92) | 0.77 (0.38 to 0.97) | 0.72 (0.58 to 0.91) | 0.79 (0.34 to 0.98) | 0.11 (0.00 to 0.25) |
| `paraphrase_read` | 28716 | 6 | 0.87 (0.81 to 0.92) | 0.69 (0.59 to 0.88) | 0.71 (0.50 to 0.88) | 0.78 (0.48 to 0.91) | 0.77 (0.50 to 0.91) | 0.07 (0.00 to 0.22) |
| `state_256` | 432528 | 6 | 0.99 (0.95 to 1.00) | 0.95 (0.84 to 1.00) | 0.90 (0.66 to 1.00) | 0.89 (0.72 to 1.00) | 0.52 (0.36 to 0.80) | 0.00 (0.00 to 0.00) |
| `paraphrase_heard` | 153680 | 6 | 0.21 (0.14 to 0.28) | 0.04 (0.00 to 0.06) | 0.00 (0.00 to 0.00) | 0.41 (0.25 to 0.56) | 0.39 (0.27 to 0.56) | 0.00 (0.00 to 0.00) |
| `paraphrase` | 432532 | 6 | 0.99 (0.94 to 1.00) | 0.89 (0.61 to 0.98) | 0.81 (0.50 to 1.00) | 0.95 (0.78 to 1.00) | 0.37 (0.28 to 0.47) | 0.00 (0.00 to 0.00) |
| lstm, offline | 28398 | 6 | 0.99 (0.97 to 1.00) | 0.99 (0.97 to 1.00) | 0.98 (0.91 to 1.00) | 1.00 (1.00 to 1.00) | 0.79 (0.70 to 0.86) | 0.00 (0.00 to 0.00) |
| lstm, online | 28398 | 6 | 0.98 (0.92 to 1.00) | 0.95 (0.92 to 0.98) | 0.94 (0.88 to 1.00) | 0.79 (0.38 to 0.98) | 0.24 (0.12 to 0.31) | 0.00 (0.00 to 0.00) |
| transformer, offline | 28658 | 6 | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) | 0.00 (0.00 to 0.00) |
| transformer, online | 28658 | 6 | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) | 0.47 (0.28 to 0.77) | 0.25 (0.22 to 0.33) | 0.00 (0.00 to 0.00) |
| lstm, offline | 58566 | 6 | 1.00 (0.98 to 1.00) | 1.00 (0.98 to 1.00) | 0.99 (0.97 to 1.00) | 1.00 (1.00 to 1.00) | 0.85 (0.75 to 0.97) | 0.00 (0.00 to 0.00) |
| lstm, online | 58566 | 6 | 0.98 (0.91 to 1.00) | 0.96 (0.92 to 1.00) | 0.88 (0.56 to 0.97) | 0.81 (0.52 to 1.00) | 0.24 (0.19 to 0.30) | 0.00 (0.00 to 0.00) |
| transformer, offline | 58250 | 6 | 0.96 (0.73 to 1.00) | 0.93 (0.59 to 1.00) | 0.94 (0.66 to 1.00) | 0.95 (0.81 to 1.00) | 0.85 (0.30 to 1.00) | 0.00 (0.00 to 0.00) |
| transformer, online | 58250 | 6 | 1.00 (1.00 to 1.00) | 1.00 (0.98 to 1.00) | 1.00 (1.00 to 1.00) | 0.41 (0.25 to 0.75) | 0.27 (0.19 to 0.45) | 0.00 (0.00 to 0.00) |
| lstm, offline | 430602 | 6 | 1.00 (1.00 to 1.00) | 1.00 (0.98 to 1.00) | 1.00 (1.00 to 1.00) | 1.00 (1.00 to 1.00) | 0.97 (0.94 to 0.98) | 0.00 (0.00 to 0.00) |
| lstm, online | 430602 | 6 | 1.00 (1.00 to 1.00) | 0.95 (0.91 to 1.00) | 0.91 (0.78 to 1.00) | 0.82 (0.69 to 1.00) | 0.30 (0.22 to 0.39) | 0.00 (0.00 to 0.00) |
| transformer, offline | 432234 | 6 | 1.00 (1.00 to 1.00) | 0.99 (0.97 to 1.00) | 1.00 (1.00 to 1.00) | 0.96 (0.75 to 1.00) | 0.85 (0.42 to 1.00) | 0.00 (0.00 to 0.00) |
| transformer, online | 432234 | 6 | 0.91 (0.75 to 1.00) | 0.89 (0.69 to 1.00) | 0.86 (0.66 to 1.00) | 0.28 (0.25 to 0.31) | 0.15 (0.08 to 0.23) | 0.00 (0.00 to 0.00) |

- **Said from its own message (H1).** `state_256` says clauses from its message with the meaning's
  slots emptied at `GO`: trained clauses .99 (6 of 6 at .90), held-out combinations .95 (5 of 6)
  and role twins .90 (4 of 6: .66 and .81 on two founders). Every clause component is read from the
  message at 1.00 at `GO` and at the last word (H2), the key of two orders of one meaning is the
  same and both orders are said right at .98 (H3), and the context twins read .98, .97 and .96
  with the message intact and .00 with it emptied. The previous-word control says .07, .00, .00
  (C: +.91, +.95, +.90). The LSTM at the same parameter count reads 1.00 on every trained panel
  offline and 1.00, .95, .91 after one pass.
- **At the default width (H1w).** With 64 association neurons the message alone (`direct_64`)
  reads .79, .47, .30, and the read of one block beside it (`read_64`) .89, .72, .74 (+.10, +.26,
  +.44 paired): a lift on new combinations that stays under the bar (5, 0 and 1 of 6 at .90). One
  founder collapses (.72, .22, .34) where the code at 70 units a block reads its agents apart at
  .63; on the other five, held-out clauses read .75 to .88 and roles .78 to .91, with the
  distractors and the agents the failing word kinds. At 128 association neurons the wide message
  reads .97, .92, .86 (6, 4 and 4 of 6).
- **Paraphrase (H4) and capacity (H5).** Taught paraphrases with the read barred, the brain says
  the second grammar's sentence of a heard first-grammar sentence at .97 and at .92 on held-out
  combinations (6 of 6 at .80 on both), against .99 and .95 for the default composition looking
  the heard sentence up and .10 and .01 for the control. Clauses of four distractors are said at
  .90 with 256 association neurons (1.00, .99, .99, .99, .90 at zero to four) and .74 with 128; the
  control says .15, .02, .00, .00, .00.
- **Beside a free read (F).** `message_free` (280 units a block) costs paths .26 and three hops
  .26 and says its clauses from the message (read with it silent, .01). `message_quiet` (70 units
  at amplitude .15) keeps paths (+.04), lifts clauses (+.03), brackets (+.15) and roles (+.08), and
  costs three hops .22. `message_fixed` (the same with `carried_gain=0.0`) costs nothing against
  `default_fixed` (paths +.13, three hops +.08), a reference that itself reads paths .79 and three
  hops .11 against the default composition's .91 and .26. The moment before as heard lifts paths
  +.08 and three hops +.45. The learned gains settle at 1.2 to 5.2 per block on `state_256`, 1.7 to
  3.4 on `read_64`.
- **Work.** A `state_256` life takes 1,221 s and 2.4e13 multiply-accumulates; `default` 113 s and
  9.5e11; the LSTM offline at 430,602 parameters 2.3e13.

### The quiet message's cost

[MEANING_PROTOCOL_3.md](MEANING_PROTOCOL_3.md), declared before its run on founders 9828600 to
9828605, tests where the quiet message's three-hop cost beside a free read comes from. Every arm
has the read free and the default lessons at 64 association neurons: `quiet` (70 units a block at
amplitude .15), the phase count at 24 and 32 (`message_phases`: no lesson's episode wraps at 24,
and at 32 no episode read does), a message held from `GO` (`message_boundary=33`,
`message_boundary_holds=True`: the brain's own words do not join), a message emptied at `GO`
(`message_boundary=33`: it holds the words said), and the amplitude at .10; each against `default`
and `heard` (`history_decays=(0.0,)`), paired by founder.

```sh
PYTHONPATH=src python benchmarks/grammar/meaning.py plan --third --out DIR/plans
PYTHONPATH=src python benchmarks/grammar/meaning.py lives --plan DIR/plans/free.json --out DIR/free.jsonl --workers 32
PYTHONPATH=src python benchmarks/grammar/meaning.py read DIR/free.jsonl --third free
```

| Arm | Parameters | clause | clause_held | roles | dyck | path | three hops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | 28,716 | .93 | .84 | .81 | .77 | .98 | .35 |
| `heard` | 30,992 | 1.00 | .97 | .99 | .86 | .98 | .71 |
| `quiet` | 46,920 | 1.00 | .84 | .92 | .92 | .90 | .13 |
| `quiet_24` | 137,940 | .99 | .89 | .89 | .91 | .90 | .26 |
| `quiet_32` | 174,348 | .99 | .86 | .94 | .83 | .89 | .23 |
| `quiet_hold` | 46,920 | .99 | .89 | .91 | .89 | .95 | .35 |
| `quiet_clear` | 46,920 | 1.00 | .88 | .92 | .89 | .98 | .24 |
| `quiet_10` | 46,920 | .99 | .82 | .93 | .84 | .96 | .18 |

- **The cause is the brain's own words.** With them kept out of the message (`quiet_hold`) three
  hops read the default's mean (+.00, paired) with the lifts kept (clauses +.06, held-out
  combinations +.05, roles +.10, brackets +.12) and paths -.03; with the message holding the words
  said alone (`quiet_clear`) three hops fall .11. The phase count is not the cause: at 24 and 32
  phases three hops fall .09 and .12 and paths .08 and .09. The amplitude at .10 leaves the cost
  (-.17).
- **The gate fails on the founders' spread.** F3 asks every panel to sit within .03 of the default
  on 5 of 6 founders; `quiet_hold` reaches that on clauses and held-out combinations and on three
  of six founders for paths (+.02, -.06, -.03, -.02, -.11, +.05) and three hops (+.28, +.02, -.02,
  -.03, -.09, -.14). No arm holds F3, and the protocol's second part (roles) did not run.
- **Against the previous-word signal** every message arm is under `heard` on held-out
  combinations, roles and three hops (-.36 to -.58).
- **On twelve founders** (the protocol's Amendment 1, gate on the paired mean and two standard
  errors) the held message alone costs paths .20 and three hops .24: on four founders it loses its
  place in the path, the first hop said at .41 to .81 with every previous word given. The held
  message beside the previous word (`quiet_hold_heard`, 49,196 parameters) holds the gate on every
  panel, lifts clauses (+.03) and reads three hops at the default's .42, under the previous word
  alone (.64).
- **Roles at 256 association neurons** (the protocol's second part): the instrument's message reads
  roles .05 over the brain's own on the same founders, inside two standard errors, so the gap is
  the founders' spread. One gain per 35 neurons (`message_gains=8`) reads roles .95 on fresh
  founders (6 of 6) and fails the gate on held-out combinations (4 of 6).

### Held-out combinations said from the state

[MEANING_PROTOCOL_4.md](MEANING_PROTOCOL_4.md), declared before its run on twelve fresh founders
(10028600 to 10028611), reads clauses said from the brain's own message (`message=280`) at 256
association neurons with the recall read barred from the meaning, and three additions beside it.
The gate H1c asks held-out combinations, clauses and role twins at .90 on 10 of 12 founders, and
brackets, paths and three hops no more than .03 under the message alone, paired.

```sh
PYTHONPATH=src python benchmarks/grammar/meaning.py plan --fourth --out DIR/plans
PYTHONPATH=src python benchmarks/grammar/meaning.py lives --plan DIR/plans/fourth.json --out DIR/fourth.jsonl --workers 48
PYTHONPATH=src python benchmarks/grammar/meaning.py read DIR/fourth.jsonl --fourth
```

| Arm | Parameters | Multiply-accumulates | clause | clause_held | roles | dyck | path | at .90 of 12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `own_256` | 432,528 | 2.42e13 | 1.00 | .94 | .93 | .93 | .51 | 12, 9, 9 |
| `read_256` (`message_read=True`) | 587,416 | 2.54e13 | .98 | .91 | .93 | .92 | .48 | 12, 8, 10 |
| `heard_own_256` (`history_decays=(0.0,)`) | 441,524 | 2.40e13 | 1.00 | .95 | .90 | .91 | .66 | 12, 11, 8 |
| `both_256` (both) | 596,412 | 2.54e13 | .99 | .94 | .96 | .91 | .51 | 12, 11, 11 |

- **H1c holds on `both_256` on these twelve founders and is not confirmed on twelve more**
  ([replication](#replication-and-longer-clauses)): the brain says clauses, held-out combinations
  and role twins from its held state at .90 on 12, 11 and 11 founders of twelve, with the
  meaning's recall slots emptied at `GO`. The gain is in the weakest founders and not at the mean
  (held-out +.00, roles +.02, paired against `own_256`).
- **Where held-out combinations are lost.** With the previous words given every word kind of a
  held-out clause is said at .96 to 1.00 but the verb, whose agreeing form is wrong on the weakest
  founders (.74 to .88). A probe fitted on trained combinations reads the verb and the agent's
  number of a held-out clause from the message at .99 to 1.00 and from the settled association at
  .96 to 1.00 at the last word: the meaning is held and the readout of the verb's form fails on a
  combination no lesson showed.
- **Longer clauses, deeper brackets and three hops** (the protocol's Amendment 1, six fresh
  founders) are lost from the state as by every learner beside it: `both_256` reads .00 on all
  three, the LSTM and the transformer at its parameter count .00 to .02, the default composition
  with its read free .00, .03 and .22. In a clause of three or four distractors the verb and the
  patient are held in the message (1.00 and .92 to 1.00 by the probe) and the distractors that
  share a block with an earlier moment are lost; brackets are lost past depth three, and a path's
  hops are said from the read.

### Replication and longer clauses

The fourth protocol's [Amendment 2](MEANING_PROTOCOL_4.md#result-of-amendment-2) reads
`both_256` again on twelve fresh founders: held-out combinations reach .90 on 7 of 12 and role
twins on 9 of 12, and over all 24 founders on 18 and 20 of 24, so the first twelve's pass of H1c
is not confirmed. Ten winners a moment (`message_winners=10`) lift held-out combinations to 12 of
12 (+.05, paired) and hold role twins on 9 of 12. [MEANING_PROTOCOL_5.md](MEANING_PROTOCOL_5.md)
gives the message a block for every moment of the longest clause (7, 8 and 16 blocks): longer
clauses read .00 to .02 on every arm, the patient of a long clause landing in a block where no
lesson put a patient. With 8 blocks held-out combinations and role twins read .97 and .97 on 6 of
6 founders, and paths fall .05.

### A word said by its parts

Said from the held state, held-out combinations were lost at the verb's form: the association
holds the verb's stem and the agent's number each on its own, and a readout of whole words reads
their conjunction only as well as the lessons showed it. [MEANING_PROTOCOL_7.md](MEANING_PROTOCOL_7.md)
(cadence#302) gives the motor two slots that settle together, a stem and a form
(`"parts": True` in `meaning.py`, the library's `slots=(12, 3)`). On twelve fresh founders
`parts_256` says the verb's form on held-out clauses at .98 to 1.00 with the previous words given
and reads held-out clauses at .97 to 1.00, +.06 (.02) over the motor of whole words. Its third
gate, no cost on any other panel, fails on paths: -.07 (.04) from the held state, and three hops
-.13 (.10) with the read free.

[MEANING_PROTOCOL_8.md](MEANING_PROTOCOL_8.md) reads where that cost lies. Protocol
7 taught the form `none` on every word without a number (`near`, the brackets, `end`), so the form
slot had to tell at every word whether the word has a number, which it reads from the state the
brain carries. Lives taught that way end with louder carried routes (the trace's learned gain .18
against .10, the heard history's 2.18 against 1.55, 42 development founders), and three hops,
whose third `near` stands where every lesson says `end`, fall with the louder trace. With
`"parts": "open"` a word without a number leaves the form slot open in its lesson
([a slot the lesson leaves open](../../docs/teach.md#a-slot-the-lesson-leaves-open)): the lesson
neither judges nor pulls it.

### What the held-meaning readings establish

- In the default composition a sentence's meaning lives in the recall memory's slots. The carried
  state holds a partial copy and the readout does not use it.
- A held message the brain can say a sentence from needs bound moments, presence in place of
  magnitude, and width in the code and in the association. At 128 association neurons the brain says
  trained clauses from its state alone as the matched LSTM does and reads .08 to .13 under it on
  combinations no lesson showed: H1 as first declared fails on `clause_held` and `roles`, with the
  failing term the readout of a superposed state on new combinations. At 256 association neurons,
  declared as an amendment before its lives, H1 holds: clauses .99, held-out clauses .95, role
  twins .95, with the meaning's memory emptied at `GO`.
- The message's key is the same for a meaning heard in any order and tells a clause from its
  role-swapped twin. It holds no order of moments of one kind, and words and concepts do not share
  a code.
- Beside a free read the episode-long message lifts brackets and the context twins and costs three
  hops. The moment before alone, which the library holds as `history_decays=(0.0,)`, lifts roles,
  paths and three hops on six fresh founders. None of these arms is a default.
- A message restated in motor slots fails held-out clauses (.11): slots settled together bind what
  was taught together.

- A brain that writes its own message (`Brain.compose(message=units)`) says held-out clauses from
  it at .95 and role twins at .90 (4 of 6 founders at .90) with 256 association neurons, paraphrases
  at .97 and .92, and clauses of four distractors at .90, each far over its previous-word control.
  At the default width a read of one block of the message lifts new combinations by .26 and .44
  and stays under .90. Beside a free read a quiet message keeps paths and costs three hops; the wide
  message takes the clauses over and costs paths.
- The quiet message's three-hop cost beside a free read is not the phase count. Held from `GO` it
  costs nothing at the mean on six founders and costs paths and three hops on twelve, where some
  brains lose their place in a path; beside the previous word it costs no panel on twelve
  founders and lifts clauses.

- With the read of one block and the previous word beside it (`both_256`), the brain says clauses,
  held-out combinations and role twins from its held state at .90 on 23, 18 and 20 founders of
  24. Longer and deeper sentences are lost from the state, as by the LSTM and the transformer, and
  a block for every moment does not bring longer clauses back.

These readings are on one finite language with supplied meanings. Open in #286: longer and deeper
sentences from the state, the default width, and a message beside a free read that beats the
previous word.

## Does a life settle? The lesson's step (#258)

[`lesson_step.py`](lesson_step.py) reads [#258](https://github.com/muellerberndt/cadence/issues/258)
under [its declared protocol](LESSON_STEP_PROTOCOL.md): the `default` arm for 4,000 batches at width
64 or 128, 256-sentence panels every 500 batches. A life's swing is the largest minus the smallest
exact-clause reading at 3,000, 3,500 and 4,000 batches; brackets and paths are the means of the last
five readings. `plan` writes the lives of a stage, `read` the table, the selection and the gates
(`--battery` adds the sequence battery's bars, `battery.py`'s `default` arm).

On fresh founders 9625800 to 9625805 the default composition settles 9 of 12 lives within .05 with
no new step:

| Arm | Width | Swing (within .05) | Clauses | Held-out | Roles | Brackets | Paths | Two hops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default | 64 | .042 (4 of 6) | .989 | .925 | .953 | .840 | .861 | .771 |
| Default | 128 | .024 (5 of 6) | .993 | .968 | .978 | .755 | .960 | .956 |
| Step by the running error | 64 | .042 (4 of 6) | .974 | .824 | .893 | .727 | .917 | .893 |
| Step by the running error | 128 | .039 (3 of 6) | .979 | .840 | .924 | .764 | .964 | .944 |

The step scaled by the running share of lesson rows answered wrong (forgetting .999, square root)
settles no more lives and costs held-out clauses by .10 to .13; the battery holds. It is not in the
library, and its lives run on commit 968a02b.

## Exposure parity with rehearsal (#270)

[`exposure.py`](exposure.py) reads [#270](https://github.com/muellerberndt/cadence/issues/270) under
[its declared protocol](EXPOSURE_PROTOCOL.md): the `default` arm with and without
`learning_rehearsal=1.0` taught the chamber's 4,000 batches in passes (the first in the stream's
order, the later ones in the transformer's offline epoch order), every first-grammar panel read
after each pass, beside `controls.py`'s transformer offline at the brain's parameter count. Compute
is an analytic count of the row sweeps times the synapses.

On founders 9627000 to 9627005 the default composition at the transformer's compute (one pass,
0.95e12 against 1.30e12) reads clauses .996, held-out .953, roles .964, brackets .862, paths .969
and three hops .475, against the control's 1.000, .999, .995, .966, .997 and .000. Brackets rise
with passes to .99 at the fourth; held-out clauses hold at .94 to .97. The rehearsal costs 5.7 times
a pass and lowers paths to .82 at the first pass, where the recall read delivers .78 to .89 of hops
against .98 to 1.00 without it.

## Routes with a goal: does planning pay? (#257)

[`routes.py`](routes.py) reads [#257](https://github.com/muellerberndt/cadence/issues/257) under
[its declared protocol](ROUTES_PROTOCOL.md): the world says two branches from a start as `X near Y
end` facts, the meaning names the start and the goal at the end of one branch, and the brain says
`start near a1 near ... near goal end`. The first hop is said before the goal, and only one
neighbour of the start reaches it. Eight start-goal pairs per founder are held out of training.
Speakers: word by word, a sliding window of three (plan three words, say the first, hear it, plan
again), and the matched LSTM and transformer.

Founders 9625700 to 9625705, 8,000 batches, first hop on held-out pairs: word by word .334, the
sliding window .312 (paired difference -.022, s.e. .050), the LSTM .395 online and .830 offline, the
transformer .540 online and .754 offline. Both speakers say the other neighbour as often as the goal
branch's, so neither reads the goal at the first hop, and settling three words jointly adds no
lookahead. The information suffices, as the offline controls show.

## What this does and does not establish

The chamber measures one composition and its options on one finite language of eighteen words,
with supplied meanings, supplied turn-taking and lessons on every word said. A sentence said
right on a held-out panel shows that the life composes the parts it was taught into a sentence
it was never taught, on this language; it shows nothing about open vocabularies, real
sentences or meanings the brain forms itself (#173's Meaning and Sentences stages). The
controls are trained next-word predictors at the brain's parameter count and exposure; a
control that says a panel right shows the information suffices at that size, not that the
brain must reach it by the same route. A failure names the structure, the length and, through
`located` and the erasures, the part of the brain where the sentence was lost.
