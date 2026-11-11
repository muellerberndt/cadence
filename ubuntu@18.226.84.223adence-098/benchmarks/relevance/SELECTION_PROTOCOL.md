# Selection through the working trace: three declared readings

The acquisition controls (`acquisition_controls.py`) under the surprise rule (lessons only on
the rows the brain answered wrong, from the complete pre-query stimulus) acquire the selection
task on the trace arm for some founders: the facts are absent at the query, the intact answer
is right on every case, and the answer falls to chance when the working trace is erased before
the query. All sixteen cases were in training, so that reading is finite acquisition. The three
readings below ask whether the acquired answer is selection of the named fact from the trace,
and how reliable it is across founders and budgets. Each reading is an option of the control
script or a run of its existing flags; the gates are declared here before the runs. Nothing
here adds a mechanism, a gene or a runtime default; the library is unchanged.

Common settings: `Brain.compose(11, 2, modules=..., episodic=False)` founders at their composed
genes, `--teaching surprise`, one neutral event, one permutation of the training cases per
round, the forked evaluation from one saved pre-query state, intact and with the trace erased.
The case `(A, B, order, query)` has the index `8A + 4B + 2 order + query` in `CASES`; the
evaluation history `(A, B, order)` has the index `4A + 2B + order` in `FACTS`.

## Reading 1: held-out panel (`--holdout`)

**What.** Train on 12 of the 16 cases; read the 4 held-out cases from the same saved pre-query
state as the training cases.

**The split.** One fixed split per founder, derived from the founder seed
(`SeedSequence([31016, seed])`): a permutation assigns each of the four fact pairs `(A, B)` to
one of the four `(order, query)` cells, and the case formed by the pair and its cell is held
out. Every training cell keeps three fact pairs, so within each cell both values of each fact
occur in training; every fact pair occurs in training under three cells; both queries and both
orders occur in training with both values of each fact; the held-out panel holds two
unequal-fact and two equal-fact cases. Only the four conjunctions are new. The held-out cases
and their indices are recorded in every output row under `holdout`.

**Why.** A brain that acquired a table over the sixteen conjunctions answers the held-out
cases at chance; a brain that reads the named fact out of the trace answers them.

**Reported.** Intact and erased accuracy on the 12 training cases and on the 4 held-out cases,
per founder, for mux and xor, on the trace arm (`--query trace`) and the exposed arm.

**Gate.** The reading passes for a trace-arm founder whose held-out intact accuracy is above
0.75 (4 of 4) with the erased reading at chance: erased accuracy 0.50 on the full panel and at
most 0.50 on the held-out cases. The pooled held-out accuracy over the founder set is reported
beside the single-founder gate, with chance at 0.50; with 4 cases per founder, one founder at
4 of 4 has a chance probability of 1 in 16, so the pooled count carries the weight.

## Reading 2: replacement intervention (`--replace`)

**What.** After training, on forks of one saved checkpoint of the trained brain (the
instrument's own snapshot; no teaching; the life is unchanged, checked by the checkpoint
digest before and after the reading), for each of the 16 cases: present the prefix (two fact
events, one neutral event); then, without a reset, present a second prefix in which (a) the
queried fact is flipped (relevant replacement: the correct answer changes) or (b) the other
fact is flipped (irrelevant replacement: the correct answer does not change); then the query.
The trace entering the query carries the second prefix. The intact answer of the same fork
procedure without a second prefix is the reference. Trace arm only: on the exposed arm the
query carries the facts itself and the reading would concern the query rather than the trace.

**Counts.** Relevant follow: the share of the 16 cases whose answer after the relevant
replacement equals the correct answer of the replaced facts. Irrelevant stay: the share whose
answer after the irrelevant replacement equals the intact answer of the same case. The share
of answers equal to the correct answer under the irrelevant replacement is reported beside the
stay count. For xor both replacements change the correct answer, so for xor the follow count
under each replacement is the reading.

**Why.** A brain that selects the named fact from the trace follows the relevant replacement
and ignores the irrelevant one. A brain that memorized a query-to-answer table, or that reads
something other than the fact content of the trace, fails (a).

**Gate.** On a founder that acquires mux through the trace (intact unequal-fact accuracy 1.00,
erased accuracy 0.50), relevant follow and irrelevant stay are both at least 0.9 (15 of 16).
Founders below that acquisition reading are reported and not gated.

## Reading 3: founders and budget

**What.** The existing flags: trace arm, mux and xor, one module of 64, 512 and 2048 rounds,
founders 0 to 11 (0 to 5 reused from the corrected runs, 6 to 11 fresh), surprise rule.

**Reported.** Per founder set (0 to 5, 6 to 11, 0 to 11) and budget: the count of founders at
or above accuracy 0.95, and the count meeting the trace-acquisition reading (intact
unequal-fact accuracy 1.00 with erased accuracy 0.50). No gate: the acquisition share is the
result.

## Commands

From the repository root, with Cadence on the Python path, one BLAS thread and at most three
workers; `DIR` is a directory outside the repository.

```sh
# reading 3: founders and budget
PYTHONPATH=src python benchmarks/relevance/acquisition_controls.py --out DIR/founders-512 \
  --tasks mux xor --brains m64 --rounds 512 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 \
  --teaching surprise --query trace
PYTHONPATH=src python benchmarks/relevance/acquisition_controls.py --out DIR/founders-2048 \
  --tasks mux xor --brains m64 --rounds 2048 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 \
  --teaching surprise --query trace
# reading 2: replacement intervention on the same founders
PYTHONPATH=src python benchmarks/relevance/acquisition_controls.py --out DIR/replace-512 \
  --tasks mux xor --brains m64 --rounds 512 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 \
  --teaching surprise --query trace --replace
# reading 1: held-out panel, both arms
PYTHONPATH=src python benchmarks/relevance/acquisition_controls.py --out DIR/holdout-trace \
  --tasks mux xor --brains m32 m64 --rounds 512 2048 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 \
  --teaching surprise --query trace --holdout
PYTHONPATH=src python benchmarks/relevance/acquisition_controls.py --out DIR/holdout-exposed \
  --tasks mux xor --brains m32 m64 --rounds 512 2048 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 \
  --teaching surprise --query exposed --holdout
```

The options leave every existing flag and its output unchanged; a run without `--holdout` and
`--replace` writes the same rows and summary as before, apart from timings and the digest of
the changed script. Contract tests cover the split's coverage property, the replacement
reading on a spy brain that changes its answer only with the relevant fact, and the unchanged
outputs under the existing flags.

## What these readings do not claim

They read a bounded component on reset trials with a supplied query, under lessons at the
query alone. A passing reading shows selection of a remembered named fact by the existing
graph and trace on the founders named; it establishes no continuing-life capability, no
sentence understanding and no runtime default.

## Results, 14 October 2026

The commands above, founders 0 to 11, the surprise rule, one BLAS thread and three workers; the
per-run rows are retained outside the repository. The two controls of reading 2 (`identical` and
the `_transplant` conditions, described with the `--replace` option) were added after its declared
runs; the re-run reproduces the declared conditions on every run.

| set | runs | wall (s) |
| --- | --- | --- |
| founders-512 | 24 | 71 |
| founders-2048 | 24 | 316 |
| replace-512 | 24 | 95 |
| replace-2048 | 24 | 328 |
| holdout-trace | 96 | 1038 |
| holdout-exposed | 96 | 813 |

#### Reading 3: founders and budget (trace arm, m64)

| task | rounds | founders | accuracy, mean | erased, mean | at or above 0.95 | acquire through the trace |
| --- | --- | --- | --- | --- | --- | --- |
| mux | 512 | 0 to 5 | 0.729 | 0.500 | 2/6 | 2/6 |
| mux | 512 | 6 to 11 | 0.823 | 0.500 | 2/6 | 2/6 |
| mux | 512 | 0 to 11 | 0.776 | 0.500 | 4/12 | 4/12 |
| xor | 512 | 0 to 5 | 0.719 | 0.500 | 1/6 | 2/6 |
| xor | 512 | 6 to 11 | 0.635 | 0.500 | 0/6 | 1/6 |
| xor | 512 | 0 to 11 | 0.677 | 0.500 | 1/12 | 3/12 |
| mux | 2048 | 0 to 5 | 1.000 | 0.500 | 6/6 | 6/6 |
| mux | 2048 | 6 to 11 | 1.000 | 0.500 | 6/6 | 6/6 |
| mux | 2048 | 0 to 11 | 1.000 | 0.500 | 12/12 | 12/12 |
| xor | 2048 | 0 to 5 | 0.958 | 0.500 | 5/6 | 5/6 |
| xor | 2048 | 6 to 11 | 0.823 | 0.500 | 3/6 | 3/6 |
| xor | 2048 | 0 to 11 | 0.891 | 0.500 | 8/12 | 8/12 |

#### Reading 2: replacement intervention (trace arm, m64, mux, founders that acquire through the trace; cases pooled)

| rounds | founders | relevant follow | irrelevant stay | identical stay | transplant relevant follow | transplant irrelevant stay | declared gate passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 4 | 34/64 | 42/64 | 43/64 | 64/64 | 64/64 | 0/4 |
| 2048 | 12 | 122/192 | 134/192 | 133/192 | 192/192 | 192/192 | 0/12 |

#### Reading 1: held-out panel (founders 0 to 11; 12 training and 4 held-out cases per founder)

| arm | task | brain | rounds | training intact, mean | held-out intact, mean | held-out erased, mean | held-out unequal-fact, mean | pooled held-out intact | founders at the gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trace | mux | m32 | 512 | 0.736 | 0.479 | 0.458 | 0.417 | 23/48 | 0/12 |
| trace | mux | m32 | 2048 | 0.938 | 0.542 | 0.375 | 0.458 | 26/48 | 0/12 |
| trace | mux | m64 | 512 | 0.931 | 0.458 | 0.417 | 0.250 | 22/48 | 0/12 |
| trace | mux | m64 | 2048 | 1.000 | 0.458 | 0.417 | 0.250 | 22/48 | 0/12 |
| trace | xor | m32 | 512 | 0.569 | 0.292 | 0.500 | 0.208 | 14/48 | 0/12 |
| trace | xor | m32 | 2048 | 0.743 | 0.292 | 0.500 | 0.292 | 14/48 | 0/12 |
| trace | xor | m64 | 512 | 0.715 | 0.417 | 0.500 | 0.458 | 20/48 | 1/12 |
| trace | xor | m64 | 2048 | 0.944 | 0.396 | 0.500 | 0.417 | 19/48 | 1/12 |
| exposed | mux | m32 | 512 | 1.000 | 0.688 | 0.708 | 0.500 | 33/48 | 0/12 |
| exposed | mux | m32 | 2048 | 1.000 | 0.688 | 0.708 | 0.500 | 33/48 | 0/12 |
| exposed | mux | m64 | 512 | 1.000 | 0.771 | 0.708 | 0.542 | 37/48 | 0/12 |
| exposed | mux | m64 | 2048 | 1.000 | 0.771 | 0.708 | 0.542 | 37/48 | 0/12 |
| exposed | xor | m32 | 512 | 0.951 | 0.583 | 0.521 | 0.583 | 28/48 | 0/12 |
| exposed | xor | m32 | 2048 | 1.000 | 0.583 | 0.542 | 0.583 | 28/48 | 0/12 |
| exposed | xor | m64 | 512 | 1.000 | 0.667 | 0.625 | 0.583 | 32/48 | 0/12 |
| exposed | xor | m64 | 2048 | 1.000 | 0.667 | 0.625 | 0.583 | 32/48 | 0/12 |

Chance reference: 48 held-out cases at 0.50 give 36 or more right with probability 1.1e-04; one founder at 4 of 4 has probability 0.0625.

#### How the missing pair of a cell is answered (selection task)

The symmetric function consistent with a cell's three training pairs gives the held-out unequal
pair the wrong answer, so the first column counts wrong answers on those cases; chance is half.

| arm | task | brain | rounds | held-out unequal-fact cases answered with the symmetric function of the cell's training pairs | held-out cases answered as the same facts and query in the other order (a training case) | held-out cases answered as the same facts and order under the other query (a training case) |
| --- | --- | --- | --- | --- | --- | --- |
| trace | mux | m32 | 512 | 14/24 | 21/48 | 27/48 |
| trace | mux | m32 | 2048 | 13/24 | 25/48 | 29/48 |
| trace | mux | m64 | 512 | 18/24 | 23/48 | 35/48 |
| trace | mux | m64 | 2048 | 18/24 | 22/48 | 34/48 |
| exposed | mux | m32 | 512 | 12/24 | 33/48 | 33/48 |
| exposed | mux | m32 | 2048 | 12/24 | 33/48 | 33/48 |
| exposed | mux | m64 | 512 | 11/24 | 37/48 | 35/48 |
| exposed | mux | m64 | 2048 | 11/24 | 37/48 | 35/48 |

#### Reading

1. Reading 3 passes as a reliability result on the finite panel: at the longer budget every founder
   acquires the selection through the trace, and the trace is necessary.
2. Reading 1 fails: the held-out conjunctions are read at chance on the trace arm, and the missing
   unequal pair of a cell is answered with the symmetric function of the cell's training pairs, so
   the acquired answer is a table over trained trace patterns rather than selection of the named
   fact.
3. Reading 2's declared gate fails on every acquiring founder; the identical control shows the
   failure is the second episode itself (a second prefix without a reset lies outside what the trial
   structure taught, whatever its content), and the transplant control shows the trace content alone
   decides the answer on every case.
4. What is located: the facts survive the neutral event in the trace and the readout is a function
   of the trace content, which acts as selection on the trained panel alone; the held-out panel is
   the gate the next change on the existing graph has to pass, and the second-episode reading needs
   a trial structure that contains such episodes before it can be read.
