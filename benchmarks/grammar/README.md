# Grammar chamber: whole sentences from a meaning

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
