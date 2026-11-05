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
  number of the noun the brain itself said first; `verb`: the verb's identity.
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
  follows), the share where the recall read's first head attends most to that moment
  (`delivered`), the share said right (`said`) and said right when delivered, teacher-forced.
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
