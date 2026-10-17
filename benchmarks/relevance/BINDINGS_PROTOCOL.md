# Fresh bindings: selection on value pairs never shown in training

The selection readings (`SELECTION_PROTOCOL.md`) found the composed brain answering its sixteen
trained cases through the working trace and the held-out conjunctions at chance: the acquired
readout was a table over trained trace patterns. This instrument removes the footing of such a
table. Two named facts of several bits are drawn fresh for every trial, a quarter of all value
pairs never occurs in training, and the gate reads those pairs. The script is `bindings.py`, its
contract tests `tests/test_bindings.py`. It is a new script beside the selection instrument and
its controls; `selective_recall.py`, `acquisition_controls.py`, their options and their outputs
are unchanged. Nothing here adds a mechanism, a gene or a runtime default; the library is
unchanged. This note is committed before any run on the declared founders.

## The task

| item | declared |
| --- | --- |
| facts | A and B, 3 bits each (values 0 to 7), drawn for every trial |
| encoding | the instrument's: one coordinate per name, bit and value (A's bit `i` of value `v` at `2i + v`, B's at `6 + 2i + v`), neutral, ask-A, ask-B, then the exposed copy of both facts; 27 inputs |
| trial | from rest: the first fact event (A first under order 0), the second, one neutral event, the query naming A or B |
| answer | the queried fact's 3 bits, one output slot of two neurons per bit (`slots=(2, 2, 2)`), bit `i` of value `v` being `(v >> i) & 1` |
| held-out pairs | per founder, 16 of the 64 value pairs: `(p(A) + q(B)) mod 4 == 0` for two permutations `p`, `q` of the values drawn from `SeedSequence([31018, seed])` |
| training trials | rounds of 16 trials in parallel streams, each drawn fresh from `SeedSequence([31017, seed])`: a value pair uniformly from the 48 training pairs, an order and a query |
| lessons | one per round, at the query |
| budgets | 512, 2048 and 8192 rounds (8,192, 32,768 and 131,072 trials) |
| founders | 0 to 11 |

With one bit the encoding is `selective_recall.inputs` exactly. Under the held-out split every
value of A keeps 6 of the 8 values of B in training and every value of B 6 of the 8 values of A,
and every combination of a bit of A with the same bit of B occurs in training: only the 16
conjunctions are new, and a learner that reads the named fact bit by bit has met every piece of
them. A founder's life runs to 8192 rounds and is read at each budget; a shorter budget's trials
are the start of a longer one's and the readings run on forks, so each reading equals a life of
that budget.

## Learners and arms

| learner | construction | arms | lessons |
| --- | --- | --- | --- |
| composed graph | `Brain.compose(27, 6, modules=[32] or [64], episodic=False, slots=(2, 2, 2))` at the founder genes (m32, m64) | trace, exposed | `act`, `every` |
| GRU control | `competitors.GRUCell` (27 inputs, hidden 64, cell weights from `SeedSequence([31019, seed])`), a softmax over two logits per bit, readout initialized at zero | trace, exposed | supervised, every row: `through the trial` (declared control), `through the query` (diagnostic) |

- **Trace arm**: the query names a fact and carries nothing else (the memory test).
- **Exposed arm**: the query carries both facts beside the name (the acquisition control).
- **`act`**, lessons judged by the brain's own act: the founder composed with the learner's opt-in
  `learning_surprise=True`. Each round, every stream walks its prefix from rest; the complete
  pre-query stimulus is taken (the trace the prefix built); the brain acts greedily on the query;
  `learner.step(drive, target, answer=act)` teaches the rows whose act differs from the answer in
  any bit, from their rows of that drive. A round without such a row has no update.
- **`every`**, the paired control: the founder without the gene, `learner.step(drive, target)` on
  every row of the same drive, after the same act.
- **GRU, through the trial**: hidden state zero at the start of each trial; one plain gradient
  step per round on the mean cross-entropy over the round's rows and bits, the gradient taken
  through every event of the trial (backpropagation through time), its norm bounded by 5, rate
  1.0.
- **GRU, through the query**: the same, with the gradient through the query event alone and the
  state entering it held fixed: the credit path of the brain's lesson, which changes the
  interpretation at the query under a held trace and gives no credit to the operations that wrote
  the trace. It is a diagnostic for the decision below and carries no gate of its own.

## Readings

At each budget, on forks of the trained learner, without a lesson: the 128 histories (64 pairs in
either order) walk their prefix in one batch from rest, one pre-query state is saved, and each
query is answered from it under three conditions:

| condition | graph | GRU |
| --- | --- | --- |
| intact | the trace the history built | the hidden state the history built |
| erased | working trace reset before the query, the warm state kept | hidden state zero |
| transplant | the trace the donor history built, under the host's own warm state | the donor's hidden state |

The donor of a history has both facts complemented and the same order, so every bit of its
queried fact differs from the host's. Reported per founder and budget: exact-match and per-bit
accuracy on the training pairs (192 cases) and the held-out pairs (64 cases), per query, on pairs
with unequal facts, the share of unequal-fact cases answered with the fact not asked, and for the
transplant the share following the donor's queried fact and the share staying with the host's
intact answer. The training log holds, per 256 rounds, the rows taught, the act's accuracy on its
own training trials, and the lessons whose free phase from rest used its whole budget of 1024
steps (the learner's `free_budget_exhausted`; finite teaching learns from the state reached). A
refused solve leaves its cases unanswered (counted wrong) and the reading
incomplete; a refusal during training ends the life, and its later budgets are unread.

## Gate

| item | declared |
| --- | --- |
| gated reading | exact match on the held-out pairs; per-bit accuracy reported beside it |
| founder at the gate | intact held-out exact match at least 0.75 (48 of 64), the reading complete, the erased reading at chance on the held-out pairs: per-bit accuracy within 0.10 of 0.50 and exact match at most 0.25 (twice the chance of 1/8) |
| pass | at least 8 of 12 founders at the gate, at a brain and budget, on the trace arm under the `act` rule |
| read beside it with the same founder criterion | `every` on the trace arm; the exposed arm; the GRU through the trial (declared control) and through the query (diagnostic) |

The exposed arm's erased reading carries no memory test, since its query carries the facts. A
constant answer per query scores exact match 1/8 and per-bit 0.50 on the held-out pairs, where
each value of each fact occurs equally often. A founder answering at random reaches 48 of 64
held-out cases exact with probability 2.7e-30.

## Decision

| outcome | then |
| --- | --- |
| the graph passes at a brain and budget | the reading is reported with the brains and budgets at which it passes |
| the graph passes at no brain at a budget where the GRU through the trial passes | a decision note for the roadmap's learned-read candidate (a learned query exchanging content with the recalled activity inside the same equilibrium), with the numbers of both and of the query-step diagnostic; nothing of the candidate is implemented here |
| both fail | the note says what the budget and the encoding suggest |

The results and the decision note are kept with the run records outside the repository.

## Development before this note

| item | what ran | used for |
| --- | --- | --- |
| GRU rate | founders 100 and 101, trace arm, through the trial, the three budgets, rates 0.1, 0.3, 1.0 and 3.0 | rate 1.0: the smallest rate at the highest mean held-out exact match over the three budgets |
| graph timing | 40 rounds on founder 100, m32 and m64, both rules | 12 to 17 ms per round of 16 trials; nothing gated was read |

The GRU through the query uses the same rate without a selection of its own. No founder 0 to 11
was run before this note was committed.

## Amendment before the control runs

| item | declared |
| --- | --- |
| gated arm (trace, `act`) and GRU | 512, 2048 and 8192 rounds, as above |
| controls (`every` on both arms, `act` on the exposed arm) | 512 and 2048 rounds (lives to 2048) |
| why | a life of the composed brain to 8192 rounds takes 17 to more than 36 minutes on the shared machine (about 134 solver sweeps per act after training against 32 at the start); the 72 control lives at 8192 rounds would take about 12 hours on two workers |
| when | after the first two lives of the gated arm and before any control run |

The control comparisons are paired at 512 and 2048 rounds; at 8192 rounds the gated arm stands
beside the GRU alone.

## Commands

From the repository root, with Cadence on the Python path, one BLAS thread and at most two
workers; `DIR` lies outside the repository. The defaults are the declared founders, budgets, arms,
rules, GRU settings and bits.

```sh
PYTHONPATH=src python benchmarks/relevance/bindings.py --out DIR/declared --learners gru --workers 2
for brain in m64 m32; do
  PYTHONPATH=src python benchmarks/relevance/bindings.py --out DIR/declared --learners graph \
    --brains $brain --arms trace --rules act --workers 2
done
for brain in m64 m32; do  # the controls, as amended
  PYTHONPATH=src python benchmarks/relevance/bindings.py --out DIR/declared --learners graph \
    --brains $brain --arms trace --rules every --budgets 512 2048 --workers 2
  PYTHONPATH=src python benchmarks/relevance/bindings.py --out DIR/declared --learners graph \
    --brains $brain --arms exposed --budgets 512 2048 --workers 2
done
```

Each life writes one JSON row (`graph-<brain>-<arm>-<rule>-<founder>.json`,
`gru-<through>-<arm>-<founder>.json`) with its held-out pairs, resolved genes, training log,
readings, gate and work counts; `summary.json` holds the table over every row in `DIR/declared`
and the digests of the script, the instrument, the competitor cells and the library sources.
Existing rows are kept and their lives skipped on a rerun.

## What this does not claim

It reads a bounded component on reset trials with a supplied query, under lessons at the query
alone. A pass shows selection of a remembered named fact on value pairs never shown, by the
existing graph and trace, on the founders named; it establishes no continuing-life capability,
no sentence understanding and no runtime default. The GRU control has backpropagation through
time and a supervised loss on every row; it measures what these trials allow a conventional
learner, and its result tests nothing about the graph's mechanism.
