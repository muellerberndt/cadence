# Vanished-cue recall chamber

This bounded instrument asks whether a continuing `Brain.compose` can answer from
an earlier cue when all current probe inputs are identical. It tests one declared
neural-graph/trace recipe; it does not establish general recall, choose defaults,
or complete [the recall contract](https://github.com/muellerberndt/cadence/issues/84).
Read the [world-model guide](../../docs/world-model.md) and
[numerical contracts](../../docs/contracts.md) before interpreting results.

## What runs

One brain carries its activity and working trace through cue, delay and probe
observations, then straight into the next episode. Each group of rows receives
every cue once, with independently permuted cue assignments per episode and
identical delay/distractor inputs within the group. Row identity and distractor
identity therefore do not supply the answer. The event-time unit is one accepted
observation; these runs do not model variable physical time.

Only the current probe has a teacher. The chamber calls
`Learner.step(brain.stimulus(probe), cue)` with the live trace, followed by a free
`act(..., greedy=True)`. Earlier observations use greedy acts only. No reward is
invented, no actor eligibility is created, and no action awaits reward feedback.
A qualified refused lesson is charged and skipped; the following free action has
its own qualification. A refused action stops that model's continuing life.
There is no automatic reset, repeated reward or silent cold restart.

The selected recipe uses `modules=(32,)`, `lateral=0`, no episodic memory, the
specified working-trace amplitude/decay, and the learner configuration in
`make_brain`. It is **not the complete compose-default configuration**. Qualified
teaching uses 128 nudged sweeps per phase; `--finite` selects the separate finite
12-sweep teaching contract. Labels at the current probe supply no credit through
past observations. Evaluation has no teachers or parameter updates.

## Controls and continuation

Before any model runs, the instrument writes the exact training/test observations,
labels and shuffle permutations. Those arrays are reused across decays and both
models. Test streams use a separate random generator; running or reordering a
control cannot change later inputs.

At each evaluation probe, the continuing vanished-cue brain saves its **complete**
state. Four private branches load that same checkpoint:

- **intact:** reads its retained context normally;
- **erased:** zeros the working trace/last activity and marks rows cold before
  the probe, retaining the neural warm state;
- **shuffled:** transplants all three trace arrays from a row with another cue;
- **reset:** clears both transient neural state and trace, retaining learned
  parameters.

The intact original life alone continues. A separate restored branch must give
an identical answer and byte-equal checkpoint arrays after the probe, including
optimizer and random state. This checks the saved state immediately before the
probe; it is not a general mid-cue/replacement continuation guarantee.

The **appended-history comparator** is a second, independently trained brain
with the same initial parameters, episodes and lesson budget. During both its
training and evaluation, an external history buffer supplies the original cue
in extra probe coordinates. It is a useful acquisition control only to the
extent its own measured free recall succeeds. It is **not a mathematical upper
bound**, and the extra input is never given to the vanished-cue brain. Appending
untrained coordinates to the vanished-cue model would not be a competent control.

## Run and verify

Install the intended checkout first (`python -m pip install -e .` from the repo).
Use an unused output directory; existing attempts are never overwritten. This
small command checks the protocol, not a useful retention horizon:

```sh
python benchmarks/recall/vanished_cue.py --out /tmp/recall-smoke --cues 2 --streams 4 --modules 4 --lessons 2 --test-episodes 2 --seeds 0 --decays 0.8 --delays 1
python benchmarks/recall/vanished_cue.py --verify /tmp/recall-smoke
```

`protocol.json` and compressed input arrays precede all training. Initial,
trained, pre-probe and continued checkpoints retain model custody.
`rows.jsonl` preserves completed conditions if a run is interrupted. A completed
`summary.json` is a `cadence.Receipt`; its `body` holds raw trial labels/answers,
scores, all teacher/free-solve reports, configuration and artifact hashes. Its
source manifest binds copied benchmark and imported library Python sources.
The verifier checks the receipt digest, source bytes and retained artifact hashes;
this is custody verification, not an independent reproduction of learned behavior.
A partial attempt has no completed summary and makes no recall claim.

Work includes accepted and refused teacher phases (free, positive and negative
nudges), every free answer, control fork and continuation check. Sweeps,
row-sweeps, residual checks and trace-update counts are separate. Per-call wall
time excludes checkpoint I/O; run wall time includes it. These measurements are
not physical energy or a cross-machine efficiency comparison. Scores report
attempted, refused and unrun rows; refusals count as wrong among attempted probes.
Stopping before a probe does not fabricate a scored answer.

## What the existing exploration establishes

The [earlier issue comment](https://github.com/muellerberndt/cadence/issues/84#issuecomment-5967884580)
reports an exploratory appended-cue acquisition difference under selected trace
amplitudes, decays and reset interventions. Its numerical table lacks a frozen
reproduction driver and source-bound checkpoint receipt here. Old recall
summaries also exist, but lack source binding and used an earlier control and
accounting protocol. Neither provides a general finding that the default trace
prevents learning; they motivate a controlled comparison.

This corrected instrument has not established a new recall horizon. Freeze
behavioral acceptance before a campaign and retain its failures. Replacement,
partial/noisy cues, finite-capacity competition, irregular event timing,
pre-replacement continuation and an independently checked retention/separation
bound remain outside this initial chamber. The original source-specific records
remain historical evidence; rerunning a corrected protocol produces a new result.

## Finite continuing input fixtures

[The reviewed finite-horizon protocol](FINITE_HORIZON_PROTOCOL.md) is a paused
scientific design. Its portable [pure input fixtures](finite_horizon_inputs.py)
cover delay, distractors, replacement, partial/noisy cues, finite load, ordered
histories and irregular event timing. Paired histories have opposite expected
responses with identical current queries; the explicit history control reads
only witnessed payloads. A separate literal recurrence checks trace updates,
including refusal of skipped or extra events.

These fixtures and their deterministic tests construct no brain and perform no
learning or solve. They provide protocol and preservation checks, not a measured
recall horizon. That was the fixture-only scope on 2026-10-04; the bounded worker
added below has separate results and remaining obligations. #84 remains open.
The existing `vanished_cue.py` instrument above retains its separate scope.

## The finite continuing recall chamber, 2026-10-08

[`finite_horizon.py`](finite_horizon.py) is the worker for the
[reviewed finite-horizon protocol](FINITE_HORIZON_PROTOCOL.md), with its
[frozen inputs](finite_horizon_inputs.py) and [`protocol-finite.json`](protocol-finite.json)
declaring every setting, the founders, the caps and the gates. One continuing
`Brain.compose` life per arm and founder: `vanished`, the declared recipe (working
trace decay 0.8, amplitude 1, learned at the QUERY lessons only, never a history
coordinate); `default`, the same brain with the composed working-trace defaults
(amplitude 3, decay 0.2), the simpler setting kept as the control of that gene;
`history`, the external-history comparator with byte-equal initial arrays, the same
lessons and the actual observed payloads of the last four WRITE events appended at
QUERY; `random`, the frozen uniform-random actions. Protocol 1 declares 192 training
episodes; protocols 2 and 3 declare 384. Each has 24
evaluation episodes of each of the 13 conditions, every QUERY forked into intact,
erased trace, shuffled trace (transplanted from the paired row with the opposite
value) and full reset; the first episode of every condition saved after the first
cue, before a replacement and before QUERY with a clone resuming each seam; private
imagination and a refused act at an impossible tolerance leaving the pre-query
checkpoint unchanged; clean-2 replayed from its saved cue under two timestamp
schedules, and with one versus two events at matched elapsed time. Every admitted
real act retains trace/last/cold before and after, source activation and decay in
compressed chunks for independent recurrence replay. The declared caps are 900
seconds and 160 MiB per founder. Revision 3 kills and reaps a founder's child process
at the wall deadline; it checks retained bytes between operations. A capped or
failed worker retains its completed journals and partial progress, records unknown
unfinished work explicitly, and leaves all unrun rows in the denominator. The horizon
is the largest contiguous passed prefix over 0,
1 and 2 intervening events under the prewritten gates; closure needs horizon 1 and
the nuisance, replacement, order, continuation, purity and resource gates.

**Maintainer audit.** The original worker silently used the fixture's fixed 16
training repeats even when protocols 2 and 3 declared 32. Their archived runs
therefore contain **192 training episodes, not the declared 384**. Those receipts,
their failed outcomes and their original source manifests remain unchanged; they
are protocol-deviating historical measurements, not conforming confirmations of
the selected budget. All 15 finite receipts have intact canonical digests and
source-manifest self-consistency, and their stored scores reproduce from their
raw trials. That does not repair the wrong schedule or omitted work.

Instrument revision 3 uses the declared repeat counts, retains completed-call
journals and raw recurrence arrays, charges seam/timing/imagination/refusal and
checkpoint work, and verifies the frozen inputs, trial census, selected teaching
rows, scores, raw trace replay, diagnostic arithmetic and gates. Each query reports
paired trace, neural activation and potential distances and labeled motor margins.
The history comparator separately records its 1088-byte buffer, input copying,
payload reads/writes and query-coordinate transport. Saved seam checkpoints remain
available for independent array comparisons. Its `every` rule preserves teaching
before the query act; `surprise` is an application policy that teaches only rows
answered incorrectly. It is **not `Brain.live`'s arousal law**. Full closure still
requires the original behavioral, custody and resource gates on every founder;
instrument completion does not establish that those gates pass. A rerun on
previously used seeds is an audit, not fresh confirmation.

```sh
python benchmarks/recall/finite_horizon.py --out /tmp/recall-finite
python benchmarks/recall/finite_horizon.py --verify /tmp/recall-finite
python -m pytest -q benchmarks/recall/test_finite_horizon.py
```

### recall/1 on fresh founders 301, 302 and 303: closure failed

Receipt `results/finite-1-2026-10-08.json.gz`, verified: every founder complete within
its caps (135 to 185 seconds), no refused act, two refused lessons in one history
arm, the trace audit at 1.1e-16 on 1,750 audited acts per brain arm, every seam,
imagination, refusal and timing check equal. Intact accuracy over planned rows,
with the paired-both-correct share, the erased fork, the shuffled fork against the
transplanted value, the default-trace control, the history control and the frozen
uniform-random policy:

| founder | condition | intact | paired | erased | shuffled→transplanted | default | history | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 301 | clean-0 | 0.52 | 0.04 | 0.50 | 0.52 | 0.57 | 1.00 | 0.46 |
| 301 | clean-1 | **1.00** | 1.00 | 0.50 | 1.00 | 0.51 | 1.00 | 0.43 |
| 301 | clean-2 | **1.00** | 1.00 | 0.50 | 1.00 | 0.52 | 1.00 | 0.55 |
| 301 | distractor-1 / -2 | 0.50 / 0.50 | 0.00 | 0.50 | 0.50 | 0.56 / 0.50 | 1.00 | 0.51 / 0.47 |
| 301 | noise-1 | **1.00** | 1.00 | 0.50 | 1.00 | 0.47 | 1.00 | 0.47 |
| 301 | replacement-1 | 0.87 | 0.74 | 0.50 | 0.87 | 0.54 | 1.00 | 0.52 |
| 301 | partial-1 / order-latest-1 | 0.66 / 0.50 | 0.31 / 0.00 | 0.50 | 0.66 / 0.50 | 0.50 / 0.53 | 0.86 / 1.00 | 0.48 / 0.44 |
| 302 | every condition | 0.50 | 0.00 | 0.50 | 0.50 | 0.43 to 0.55 | 1.00 | 0.43 to 0.56 |
| 303 | clean-0 | **0.99** | 0.99 | 0.50 | 0.99 | 0.50 | 1.00 | 0.43 |
| 303 | clean-1 | 0.85 | 0.71 | 0.50 | 0.85 | 0.50 | 1.00 | 0.51 |
| 303 | clean-2 | 0.62 | 0.24 | 0.50 | 0.62 | 0.50 | 1.00 | 0.54 |
| 303 | replacement-1 | 0.92 | 0.83 | 0.50 | 0.92 | 0.50 | 1.00 | 0.49 |
| 303 | distractor-1 / noise-1 / partial-1 / order-latest-1 | 0.69 / 0.66 / 0.50 / 0.50 | 0.38 / 0.31 / 0 / 0 | 0.50 | 0.69 / 0.66 / 0.50 / 0.50 | 0.50 | 1.00 / 1.00 / 0.96 / 1.00 | 0.48 to 0.56 |

Horizons: 301 none (clean-0 fails while delays 1 and 2 pass, so the contiguous-prefix
rule credits nothing), 302 none, 303 zero. Nuisance gates fail on every founder.
Capacity 2/4 and delays 4/8 are at 0.46 to 0.75 and never pass. Closure fails as
declared and #84 stays open.

What the receipt does establish. Where the declared recipe recalls, it recalls through
the trace exactly as the protocol demanded: founder 301 answers delays 1 and 2 and
the noisy cue at 1.00 while its erased and reset forks sit at 0.50, and every shuffled
fork answers the transplanted history's value at the same accuracy and the original's
at its complement, so the transplanted trace alone determines the answer. The external
history control reads 1.00 on nearly every condition, so the lessons teach the mapping
when the cue is present. The composed default trace (amplitude 3, decay 0.2) recalls
nothing on any founder or condition, the finding of the 2026-10-03 exploration now on
a frozen protocol with source-bound receipts. The declared recipe is founder-bound:
one founder learns nothing at all with the same lessons that teach its history twin,
one recalls at delays 1 and 2 but not at 0, one at 0 and partly at 1. The distractor
and order conditions fail everywhere. The subsequent development below selected trace amplitude and lesson budget before
freezes 2 and 3; the `--repeats` override marks those development runs as not frozen.

### Development founders 0 and 1: the amplitude, the decay and the lesson budget

Receipts `results/development-finite-*-2026-10-08.json.gz`, every one verified, 32 training
repeats (384 lessons) and 24 evaluation repeats unless marked. Horizon per founder, and the
intact accuracy of the declared recipe on the clean delays, the one-event distractor, the
replacement and the latest-of-two order condition (founder 0 / founder 1):

| amplitude | decay | repeats | horizon | clean-0 | clean-1 | clean-2 | distractor-1 | replacement-1 | order-latest-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.5 | 32 | none / none | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.1 | 0.8 | 32 | 0 / none | 1.00 / 0.50 | 0.76 / 0.50 | 0.74 / 0.50 | 1.00 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.3 | 0.5 | 32 | 0 / none | 1.00 / 0.50 | 0.50 / 0.77 | 0.50 / 0.62 | 0.93 / 0.71 | 0.50 / 0.53 | 0.50 / 0.50 |
| **0.3** | **0.8** | **32** | **1 / 0** | 1.00 / 0.99 | 0.98 / 0.80 | 0.94 / 0.88 | 0.92 / 0.94 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.3 | 0.8 | 64 | none / none | 0.58 / 0.50 | 0.76 / 0.50 | 0.80 / 0.50 | 0.50 / 0.50 | 0.72 / 0.50 | 0.70 / 0.50 |
| 0.3 | 0.9 | 32 | none / none | 0.52 / 0.50 | 0.67 / 0.50 | 0.55 / 0.50 | 0.70 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.5 | 0.8 | 32 | none / none | 0.50 / 0.81 | 0.50 / 0.90 | 0.50 / 1.00 | 0.50 / 0.79 | 0.50 / 0.50 | 0.50 / 0.50 |
| 1.0 (recall/1) | 0.8 | 32 | none / none | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 3.0 (composed default) | 0.8 | 32 | none / none | 0.50 / 0.49 | 0.50 / 0.51 | 0.50 / 0.49 | 0.50 / 0.50 | 0.50 / 0.66 | 0.50 / 0.52 |
| 0.3 | 0.8 | 32, surprise rule | none / 2 | 0.77 / 0.99 | 0.76 / 1.00 | 0.61 / 1.00 | 0.74 / 1.00 | 0.50 / 0.50 | 0.50 / 0.50 |
| 1.0 | 0.8 | 32, surprise rule | none / none | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 |
| 0.3 | 0.8 | 64, surprise rule | 0 / none | 1.00 / 0.50 | 0.79 / 0.50 | 0.79 / 0.50 | 0.92 / 0.50 | 0.97 / 0.50 | 0.67 / 0.50 |

Amplitude is the knob: at 1.0 and 3.0 both founders recall nothing at any delay, at 0.3 with
decay 0.8 one founder has horizon 1 and the other horizon 0 (clean-1 at 0.80 with a paired
share of 0.59). Doubling the lesson budget to 64 repeats takes the recall away again on
founder 0 (clean-0 from 1.00 to 0.58, while clean-8 rises to 0.94) and founder 1 learns
nothing: more lessons on a mapping already learned move it. Replacement and the order of two
cues are at chance at every point but one; the external history control reads 1.00 on them
throughout. The selected point, amplitude 0.3, decay 0.8, 32 repeats, was declared in
[`protocol-finite-2.json`](protocol-finite-2.json) with recall/1's gates, caps, conditions
and controls unchanged before founders 304 to 306 ran.

### recall/2 historical run on founders 304, 305 and 306: wrong budget, closure failed

Protocol SHA-256 `a91ddb6755ed476b1408ecc1c07c362fb9b8d8969ac5df3d665bad4d885237bb`; receipt
`results/finite-2-2026-10-08.json.gz`: every founder completed the instrument's
192-episode schedule, half the declared training budget, within its recorded caps
(185 seconds for the three), no refused act or lesson, the trace audit at 1.1e-16 on 1,750
audited acts per brain arm, every seam, imagination, refusal and timing check equal.

| founder | condition | intact | paired | erased | shuffled→transplanted | default | history | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 304 | every condition | 0.50 | 0.00 | 0.50 | 0.50 | 0.50 | 0.78 to 1.00 | 0.45 to 0.57 |
| 305 | every condition | 0.36 to 0.50 | 0.00 | 0.50 | 0.36 to 0.50 | 0.50 | 0.50 to 1.00 | 0.44 to 0.55 |
| 306 | clean-0 / clean-1 / clean-2 | **1.00 / 1.00 / 1.00** | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 | 0.48 to 0.53 |
| 306 | distractor-1 / distractor-2 | 0.72 / 0.50 | 0.45 / 0.00 | 0.50 | 0.72 / 0.50 | 0.50 | 1.00 | 0.45 / 0.54 |
| 306 | replacement-1 / order-latest-1 | 0.79 / 0.56 | 0.58 / 0.11 | 0.50 | 0.79 / 0.56 | 0.50 | 1.00 | 0.43 / 0.52 |
| 306 | partial-1 / noise-1 / clean-4 / clean-8 | 0.50 / 0.65 / 0.72 / 0.50 | 0.00 / 0.29 / 0.45 / 0.00 | 0.50 | same as intact | 0.50 | 1.00 | 0.46 to 0.55 |

Horizons: 304 none, 305 none, 306 zero (delays 1 and 2 clean at 1.00 through the trace, with
the erased and reset forks at 0.50, but the one-event distractor at 0.72 fails the prefix).
Closure fails and #84 stays open. These results do not test the selected
32-repeat recipe: two founders learn nothing from the same 192 lessons that teach their
history twins to 1.00, one recalls a vanished cue across two fillers and loses it to a
distractor. The composed default trace again recalls nothing on any founder.

### recall/3 historical run on founders 307, 308 and 309: wrong budget, closure failed

The loop chamber of issue #140 (`benchmarks/rhythm/loop_rhythm.py`) found that a lesson on
every row makes a learned pattern come and go from pass to pass, and the 64-repeat row above
is the same finding here. The third freeze declares the worker's application-level
`surprise` teaching rule: at every QUERY the free greedy act
first, then one lesson on the rows it answered wrong, on the drive that act read, as the
steady-rhythm chamber's `mismatch` arm teaches; a right answer teaches nothing. On the
development founders the rule gave horizons 2 and none against 1 and 0 under the every rule
at the same trace and budget, nothing at amplitude 1.0 under either rule, and the recall lost
again at 64 repeats under either rule. [`protocol-finite-3.json`](protocol-finite-3.json),
SHA-256 `957a66fbf60f077c1ffa40d4bbc91bed05e9ff270a9a8a16d108df6061e172a1`, keeps recall/2's
trace, budget, gates, caps, conditions and controls; receipt
`results/finite-3-2026-10-08.json.gz`: every founder completed the instrument's
192-episode schedule, half the declared training budget, within its recorded caps (186
seconds for the three), no refused act or lesson, the trace audit at 2.8e-17 on 1,750 audited
acts per brain arm, every seam, imagination, refusal and timing check equal. The rule gave 167
to 184 lessons on 557 to 722 rows per founder, from 192 query opportunities.
The historical every-rule runs used 192 lessons on 1,536 rows; the selected
32-repeat budget would instead offer 384 queries. The history twins needed 29 to 35
lessons. These different founder sets do not isolate the teaching-rule effect.

| founder | condition | intact | paired | erased | shuffled→transplanted | default | history | random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 307 | clean-0 / clean-1 / clean-2 / clean-4 | 0.50 / 0.50 / 0.50 / 0.56 | 0.00 to 0.12 | 0.50 | same as intact | 0.44 to 0.51 | 1.00 | 0.50 to 0.55 |
| 307 | clean-8 / capacity-2 / capacity-4 | 0.93 / 0.74 / 0.62 | 0.85 / 0.49 / 0.25 | 0.50 | same as intact | 0.46 to 0.52 | 1.00 | 0.52 to 0.56 |
| 307 | every other condition | 0.50 to 0.52 | 0.00 to 0.03 | 0.50 | same as intact | 0.45 to 0.52 | 1.00 | 0.41 to 0.55 |
| 308 | clean-0 / clean-1 / clean-2 / clean-4 / clean-8 | 0.50 / 0.52 / 0.66 / 0.85 / 0.62 | 0.00 / 0.03 / 0.31 / 0.70 / 0.25 | 0.50 | same as intact | 0.50 | 1.00 | 0.47 to 0.56 |
| 308 | partial-1 / noise-1 | 0.59 / 0.68 | 0.19 / 0.35 | 0.50 | same as intact | 0.50 | 1.00 | 0.46 / 0.54 |
| 308 | every other condition | 0.50 to 0.62 | 0.00 to 0.24 | 0.50 | same as intact | 0.50 | 1.00 | 0.47 to 0.57 |
| 309 | clean-0 / clean-1 | **0.99 / 0.89** | 0.99 / 0.78 | 0.50 | 0.99 / 0.89 | 0.50 / 0.49 | 1.00 | 0.49 / 0.47 |
| 309 | clean-2 / clean-4 / clean-8 | 0.86 / 0.85 / 0.50 | 0.73 / 0.71 / 0.00 | 0.50 | same as intact | 0.59 / 0.67 / 0.49 | 1.00 | 0.45 to 0.52 |
| 309 | distractor-1 / distractor-2 / noise-1 / partial-1 | **0.99** / 0.84 / **1.00** / 0.83 | 0.98 / 0.68 / 1.00 / 0.73 | 0.50 | same as intact | 0.46 to 0.54 | 1.00 | 0.44 to 0.55 |
| 309 | replacement-1 / order-latest-1 | 0.50 / 0.79 | 0.00 / 0.58 | 0.50 | same as intact | 0.57 / 0.53 | 1.00 | 0.53 / 0.46 |

Horizons: 307 none (its only passing condition is the eight-event delay, with no prefix
under it), 308 none, 309 one (delay 2 reads 0.86 with a paired share of 0.73, under the
gate's 0.75). Closure fails as declared and #84 stays open. Where founder 309 recalls, it
recalls through the trace: its erased and reset forks sit at 0.50 and every shuffled fork
answers the transplanted history's value; the default trace fails the declared recall gates.

These historical runs measure a founder-dependent recall limit at their actual
budgets. Founders 303, 306 and 309 have nonnegative contiguous horizons among the
nine historical founders; other founders sometimes recall particular longer
delays without passing the shorter prefix. Erasure, reset and transplantation
support a causal role for the trace in the successful conditions. The composed
trace control fails the declared recall gates. Replacement and ordered recall
remain weak, while the supplied-history comparator is often competent. These
results do not establish a necessary new memory mechanism or the outcome of the
then-unexecuted 384-episode protocol. The corrected audits below now test that budget.

### Corrected 384-episode audits: closure still fails

The separately retained [protocol 2 audit](results/audit-finite-2-maintainer-2026-10-08.json.gz)
and [protocol 3 audit](results/audit-finite-3-maintainer-2026-10-08.json.gz) were produced
by instrument revision 3 at
[`94bb365`](https://github.com/muellerberndt/cadence/commit/94bb36561c90bdd871293b2893bf3e882f4b138d),
with the recorded Cadence 0.77.0 source. These are correction audits on the six
previously used founders, **not fresh confirmation**. They use the original frozen
protocols, genes, thresholds and caps without overrides. Each brain arm completed
all **384 training and 312 evaluation episodes**, with batch size 8.

| Protocol | Founder | Vanished-cue lesson attempts / row presentations | Contiguous horizon | Qualified teacher refusals, all arms | Closure |
| --- | --- | --- | --- | --- | --- |
| 2, every | 304 | 384 / 3072 | none | 5 | failed |
| 2, every | 305 | 384 / 3072 | 0 | 3 | failed |
| 2, every | 306 | 384 / 3072 | 0 | 14 | failed |
| 3, selective | 307 | 346 / 1237 | none | 0 | failed |
| 3, selective | 308 | 364 / 1351 | none | 0 | failed |
| 3, selective | 309 | 334 / 1112 | none | 0 | failed |

Lesson/row counts include refused teaching attempts; all 22 refusals remain in the
work and acceptance records. There were no unexpected free-action refusals or unrun
evaluation rows. Every founder fails the required nuisance gate. The two protocols
use different founders, so this table is not a paired causal comparison of teaching rules.

The verifier independently replayed **78,264 admitted trace transitions**, with
maximum discrepancy at most 1.12e-16. Saved continuation, private/refused purity,
timestamp invariance, event-count diagnostics and query diagnostic arithmetic all
passed. Each history arm transported 393,216 training and 319,488 evaluation bytes
through its query-history coordinates, in addition to its separately recorded
buffer, copying and payload work. Worker time including replay was 97.4–196.3 seconds;
retained storage including the shared sources and complete receipt was 136.4–149.4 MiB
per founder, within the unchanged 900-second/160-MiB caps.

An earlier revision-2 audit exceeded the output cap because raw query vectors were
duplicated across progress and receipt JSON. That capped/interrupted attempt and
its completed journals remain local; no protocol-3 run used that encoding. Revision 3
losslessly stacks raw trace arrays and stores raw query vectors once in compressed
artifacts. Round-trip and re-signed tamper tests verify the encoding; the completed
founder-304 lives shared with the earlier attempt have identical training answers,
evaluation answers, scores and final checkpoint arrays.

These completed audits establish the failure at the declared budget, rather than
guessing from the earlier half-budget runs. They provide no promotion of a recall
horizon across all founders, and **#84 remains open under its original gates**.

## Where the lesson's credit lands, and why founders differ (development, 2026-10-10)

`credit_landing.py` is a development diagnostic on the declared recipe (trace amplitude 0.3,
decay 0.8, the `every` rule, the chamber's `LearnerConfig`). It trains founders for 64
repeats (768 lessons) and records, per lesson, the RMS efficacy step every projection
received and the motor activation at QUERY; over the run, the sign consistency of each
synapse's accepted steps; and every four repeats the clean, distractor, replacement and
order recall of a saved and reloaded copy on the frozen test episodes (the copy's own
stream, six of the thirteen conditions, so its carried context differs from the chamber's
intact fork). Receipts `results/credit-*-2026-10-10.json.gz`, like the finite and
development receipts of this date below, were produced on commit `c922553` of this chamber's
branch and are bound to its scripts and library sources: `--verify` checks them anywhere,
and `--verify --current` holds in a checkout of that commit, since later library changes
alter the bound sources. `--report <receipt>` prints the tables. These are readings on
development and spent founders with no gate; the three freezes above and their failed
closures stand.

**Clean recall appears in every founder.** Under the recipe, founders 0, 1, 304, 305 and
306 all reach clean-0 recall of 0.95 to 1.00 on a saved copy at some evaluation, 304 and
306 within the first hundred lessons. The prefrontal→association steps are among the most
sign-consistent (0.14 to 0.21 across founders, against 0.04 to 0.15 for the other
projections in four of the five); by magnitude the association↔motor efficacies move most
(final drift 2.0 to 4.2 of their RMS against 0.24 to 0.56 for prefrontal→association). The
paired association separation at QUERY grows while recall is present (0.12 to 0.27 for 306).

**Two of five readouts saturate; a third drifts to chance.** In founders 1 and 304 both
motor neurons saturate: the motor activation rises from 0.06–0.08 in the first quarter to
1.00 (304 by lesson 267, 1 by lesson 484), the pair's answers coincide, and every later
lesson moves the efficacies by 1e-6 to 1e-4, ten to a hundred times less than before. A
finite nudge cannot move a neuron at its ceiling, so the saturated readout is an absorbing
state. Founder 305 wanders between 0.5 and 0.06 and ends at chance; 306 stays between 0.05
and 0.29 and keeps recall (clean-0 0.51 to 0.99 after repeat 32, 0.66 at 64). The untrained
founders do not predict this: their WRITE-time association separation (0.22 to 0.38) and
trace separation at QUERY (0.036 to 0.061) interleave the historical learners 303, 306 and
309 with the others. The founder dependence of the three freezes is consistent with the
readout's operating point under continued teaching rather than with the memory.

**The controls, on founders 1, 304, 305 and 306.** Raw steps (`normalize=0`, `eta=0.5`,
`eta_bias` at the recipe's 0.005, one hundredth of eta) leave the motor neurons at
activation 0.003 to 0.02; 1 and 304 learn nothing in 768 lessons, 306 reads delays one and
two at 0.71 to 0.77 from lesson 576 with delay zero at chance, and 305 answers the opposite
token for most of the run (clean-0 0.00 to 0.15 from repeat 8 to 44), an inverted mapping. Without momentum, 1 and 305 read 1.00 at the end with a dip to 0.50 at repeat 44,
306 reads 0.52 to 1.00, and 304 saturates (0.89). At `eta=0.01` no founder saturates and
recall comes and goes in all four. A fixed motor bias of 2.0 puts the readout at activation
0.84 to 0.88, next to the ceiling, and little is learned. A chamber-level homeostatic probe
(after every act, each motor bias moves by 0.02 times the shortfall of its activation below
0.3) on top of the normalized steps cannot hold the readout: 306's bias winds to −14.6 and
its readout ends silent at 0.10. The same probe with raw steps reaches clean recall of 0.95
or better in all four: 305 holds it from repeat 24, 304 with dips to 0.74, 1 reads 1.00 at
repeats 48 and 56 and 0.50 at 44, 306 ends between 0.6 and 0.9; motor activation averages
0.25 to 0.32 per quarter (0.00 to 0.45 per lesson); prefrontal→association drifts below 0.008
of its RMS while association↔motor drifts 0.83 to 0.98; founder 1 has 130 of 768 lessons
refused by qualified teaching, the others none.

**The library gene.** The same rule as two `LearnerConfig` genes, `homeostasis_rate` and
`homeostasis_target` (founder rate 0), applied at every accepted lesson from the lesson's
free state, with raw steps: at rate 0.1 all four founders end at clean-0 recall of 0.95 to
1.00 (1 0.95, 304 1.00, 305 0.99, 306 0.99; clean-1 and clean-2 0.53 to 1.00), replacement
and the latest of two cues are read above chance in three (304 0.82 and 0.50, 305 0.61 and
0.71, 1 0.56 and 0.50 at the end), the motor activation averages 0.27 to 0.31 and no lesson
is refused. The accepted steps' sign consistency is 0.46 to 0.68 on every projection under
the raw steps, against 0.14 to 0.21 under the normalized recipe: normalization adds sign
noise to a small consistent signal. At rate 0.05 founder 1 reads 0.52 at the last
evaluation.

| Learning | Readout | 1 | 304 | 305 | 306 |
| --- | --- | --- | --- | --- | --- |
| recipe (normalize .99, momentum .9, eta .05) | as composed | recall, then saturates | recall, then saturates | recall, then chance | keeps recall with dips |
| recipe, momentum 0 | as composed | 1.00 at the end, a dip to 0.50 | saturates | 1.00 at the end, a dip to 0.50 | 0.52 to 1.00 |
| recipe, eta .01 | as composed | comes and goes | comes and goes | comes and goes | comes and goes |
| raw (normalize 0, eta .5) | as composed | silent, nothing | silent, nothing | silent, inverted mapping | silent; delays 1 and 2 at 0.71 to 0.77 from lesson 576 |
| recipe | motor bias 2.0 | near the ceiling, little | near the ceiling, little | near the ceiling, little | near the ceiling, little |
| recipe | homeostatic probe (0.3, 0.02 per act) | comes and goes | chance | an episode, then chance | winds up, silent |
| raw (normalize 0, eta .5) | homeostatic probe (0.3, 0.02 per act) | 1.00 at 48 and 56 | holds from 24 with dips | holds from 24 | 0.6 to 0.9 |
| raw (normalize 0, eta .5) | gene: homeostasis_rate .1, target .3 | 0.95 at the end | 1.00 from 24 | 0.99 from 8 | 0.99 at the end, dips |
| raw (normalize 0, eta .5), recall/4 fresh founders 310 / 311 / 312 at 32 repeats | gene: .1, target .3 | clean-0 0.52, 1.00, 1.00; the history control ends at chance in two founders of three | | | |
| recipe (normalize .99, momentum .9, eta .05) | gene: .1, target .3 | comes and goes, 0.98 at the end | saturates (0.99) | recall to 24, then saturates (0.83) | winds up (bias −19), chance |
| recipe (normalize .99, momentum .9, eta .05) | gene: .3, target .3 | recall at 16 to 24, then saturates (0.93) | recall to 40, then chance (0.57) | recall to 24, winds up (bias +14), silent | recall to 32, then saturates (1.00) |

**Reading.** The chamber's readout has nothing that keeps it in its responsive range. Raw
local steps on a silent readout are too small to learn; normalized steps amplify them into
fixed-size pushes whose zero-mean part random-walks the readout into saturation, where
learning stops for good. A local, counter-free intrinsic rule that keeps each motor neuron's
activation near a target removes both rails, and with it the raw local contrast learns and
keeps recall in these development runs. The library carries the rule as genes with the
founder rate zero, so every composed brain, checkpoint and continuation is unchanged by
default. Cue replacement and ordered recall are read above chance by some founders with the
responsive readout and are at chance for others; the decaying-superposition trace is the
declared limit for them.

### recall/4 on fresh founders 310, 311 and 312: closure failed

[`protocol-finite-4.json`](protocol-finite-4.json) declares the responsive readout with raw
steps (`homeostasis_rate` 0.1 toward 0.3, `normalize` 0, `eta` 0.5, `eta_bias` 0.005,
momentum 0.9) on three unused founders, with recall/2's gates, caps, conditions, controls and
32-repeat budget. Receipt `results/finite-4-2026-10-10.json.gz`, verified: every founder
completed all 384 training and 312 evaluation episodes with no refused lesson or act (129 to
193 seconds each), every seam, imagination, refusal and timing check equal. Horizons: 310
none, 311 zero, 312 none; every founder fails the nuisance gates; **closure fails and #84
stays open.**

| founder | condition | intact | paired | erased / reset | shuffled→transplanted | history | random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 310 | clean-0 / clean-1 / clean-2 | 0.52 / 0.76 / 0.73 | 0.04 / 0.51 / 0.47 | 0.50 | 0.52 / 0.76 / 0.73 | **0.50** | 0.47 to 0.51 |
| 310 | distractor-1 / noise-1 / replacement-1 / order-latest-1 | 0.62 / 0.93 / 0.78 / 0.71 | 0.25 / 0.86 / 0.55 / 0.43 | 0.50 | same as intact | **0.50** | 0.50 to 0.54 |
| 311 | clean-0 / clean-1 / clean-2 | **1.00 / 0.99 / 0.96** | 1.00 / 0.98 / 0.92 | 0.50 | 1.00 / 0.99 / 0.96 | 1.00 | 0.48 to 0.55 |
| 311 | distractor-1 / distractor-2 / partial-1 / noise-1 | 0.86 / 0.68 / 0.68 / 0.99 | 0.73 / 0.35 / 0.36 / 0.99 | 0.50 | same as intact | 1.00 | 0.46 to 0.55 |
| 311 | replacement-1 / order-latest-1 | 0.76 / **0.89** | 0.51 / 0.77 | 0.50 | same as intact | 1.00 | 0.49 |
| 312 | clean-0 / clean-1 / clean-2 | **1.00 / 0.99 / 0.86** | 1.00 / 0.99 / 0.73 | 0.50 | 1.00 / 0.99 / 0.86 | **0.50** | 0.46 to 0.49 |
| 312 | distractor-1 / distractor-2 / partial-1 / noise-1 | 0.84 / 0.88 / 0.78 / 0.95 | 0.69 / 0.77 / 0.58 / 0.90 | 0.50 | same as intact | **0.50** | 0.44 to 0.54 |
| 312 | replacement-1 / order-latest-1 | 0.58 / 0.50 | 0.17 / 0.00 | 0.50 | same as intact | **0.50** | 0.45 to 0.48 |

What the run establishes. Two of the three founders recall the vanished cue through the trace
at every delay of the prefix (311 and 312 at 0.86 to 1.00 on clean-0 to clean-2, erased and
reset forks at 0.50, every shuffled fork answering the transplanted history), and 311 reads
the latest of two cues at 0.89, the first fresh founder to pass that condition's intact floor;
312's paired share on clean-2 (0.73) and both founders' on distractor-1 (0.73 and 0.69) sit
under the 0.75 floor, partial-1 fails in all three, and replacement fails in all three.
Founder 310 learns slowly (training answers right 0.75 in the last 48 queries) and reads
clean-0 at chance while recalling at delays 1 to 4. The external-history control, which
reaches 1.00 in about thirty lessons under recall/2's normalized recipe, **ends at chance in
two of the three founders under the raw steps** (310 and 312 at 0.50 on every condition,
training answers right 0.72 and 0.50: a confident constant answer, and a mid-run fall), so their horizon gate fails on the control's
competence bound before the vanished brain's recall is counted. Raw steps at a fixed rate
learn the trace's small signal and miss the control's large one; the normalized recipe does
the reverse. The gene on top of the normalized recipe (receipts `credit-gene-0.1-recipe` and
`credit-gene-0.3-recipe`) does not hold the readout in three of the four founders at rate
0.1 and in all four at 0.3: they recall for a few repeats and then saturate or wind up (motor
activation 0.83 to 1.00, or a bias of −19 and +14 with the readout silent), the probe's
finding again at the library's rate; founder 1 at rate 0.1 keeps a recall that comes and goes
and reads 0.98 at the end with 41 lessons refused. A bias
cannot compensate synaptic drive that normalized steps push by a fixed amount at every
lesson.

A development screen of smaller raw rates on the spent founders 304 and 305 (recall/4's
protocol with eta 0.2 and 0.1, eta_bias at eta/10, the full 32-repeat training budget and 8
evaluation repeats per condition; receipts `results/development-finite-4-eta0.2-2026-10-10.json.gz`
and `...-eta0.1-...`, verified, not confirmations): at eta 0.2 founder 305 reaches
**horizon 1** with clean-0 to clean-2 at 1.00, 0.94 and 1.00 (paired 1.00, 0.88, 1.00),
distractor-1 at 1.00, partial-1 at 0.83, replacement-1 at 0.95 and order-latest-1 at 0.89, the
first founder of any recipe to read both, and fails the nuisance gate on noise-1 (0.59); 304
reads clean-1 and distractor-1 at 1.00 and clean-0 at 0.56. At eta 0.1 both history controls
learn (1.00 on every condition), 305 reads the clean delays at 0.97 to 1.00 and distractor-1
at 0.62, and 304 keeps its delay-0 failure (0.50, a constant answer) while reading clean-1 at
1.00. The history control is competent at 0.1 and marginal at 0.2 for one founder.

Two further screens on founders 304, 305, 306 and 1 (32 training repeats, 8 evaluation
repeats) rule out the step size as the lever. A per-event bound on every efficacy and bias
step, at 0.003 and at 0.01 with the raw rate 0.5, left the vanished brains' readings
byte-identical (their steps never reached the cap) and did not rescue the control (0.50 on
every condition in three of the four founders). The raw rate 0.2 with twice the lessons (64
repeats) did not either: the control read 0.50 in three founders, and 306's recall sank to
chance. A slot-wise variant of the intrinsic step, every bias of a motor slot shifted together
toward a winning activation at 0.5, was worse: the control collapsed in nearly every founder,
because a confident readout is pushed down faster. Neither the cap nor the slot-wise rule
ships, and their receipts are not shipped either: the library code they ran on was removed. The control's pattern is the same in every run: it learns within the first hundred
lessons (training answers right at 0.84 to 0.99) and loses the mapping later. A lesson on
every query keeps pushing a mapping that is right, the efficacies run toward their
cap while the homeostatic bias is unbounded, and the readout ends silent. The lever is the
lesson rule, and the screen of the chamber's `surprise` rule (a lesson only on the rows the
free answer got wrong, recall/3) with the gene and raw steps confirms it: at both rates, 0.5
and 0.2, the external-history control reads 1.00 on every condition in all four founders, the
first time the control is competent under raw steps. The vanished brains learn more slowly
under the rule (317 to 380 lessons of 384 queries, training answers right 0.56 to 0.74 in the
last 48): at 0.5, founders 304, 305 and 306 reach horizon zero (clean-0 at 0.97 to 1.00,
distractor-1 at 1.00, clean-1 at 0.72 to 0.80 under the paired floor) and founder 1 reads
chance; at 0.2, founder 1 reaches horizon one (clean-0 0.97, clean-1 0.88, distractor-1 1.00,
distractor-2 0.92, partial-1 0.83, noise-1 0.95) and the others fall back. With twice the
lessons (64 repeats, 541 to 735 lessons) the control stays at 1.00 everywhere and the vanished
brains' recall comes and goes: at 0.5, founder 305 reaches horizon one (clean 0.97, 0.97 and
1.00, distractor-1 1.00) and 306 horizon zero, while 304 and 1 end at a constant answer (0.50
with no pair right); at 0.2, 304 reads all three clean delays at 1.00 and distractor-1 at 0.80,
305 reads replacement at 0.95 and order at 0.81 with clean-1 at 0.73, and 306 and 1 end at a
constant answer. Receipts `results/development-finite-4-surprise-*-2026-10-10.json.gz` (32 and 64 repeats
at both rates) are development runs on spent founders, verified. With the readout held in range by the gene and the control competent under the
surprise rule, the instability that is left sits in the trace-reading mapping itself: it is
acquired in every founder at some evaluation and lost again under further lessons on a weak,
superposed signal, the limit this chamber declares. Founder 304's delay-0 answer is constant
at every rate under the `every` rule while delays 1 and 2 are read, and reads 1.00 under the
surprise rule.

### recall/5 on fresh founders 313, 314 and 315: closure failed, horizons none, 0 and 0

[`protocol-finite-5.json`](protocol-finite-5.json) declares the surprise rule (recall/3), the
readout's intrinsic plasticity (`homeostasis_rate` 0.1 toward 0.3) and raw steps at half the
rate (`eta` 0.2, `eta_bias` 0.002) on a brain composed with the sensory projection at twice
its scale (`sensory_scale` 2; the chamber's `make_brain` takes `brain.sensory_scale`, founder
1), with recall/2's gates, caps, conditions and controls and recall/4's 32-repeat budget, on
three unused founders. Receipt `results/finite-5-2026-10-10.json.gz`, verified: every
founder completed all 384 training and 312 evaluation episodes with no refused act (313 to
340 lessons under the rule, 313 to 490 seconds each), every seam, imagination, refusal and
timing check equal. **Closure fails and #84 stays open.** Founder 314 reads clean-0 at 1.00,
clean-1 at 0.78 (paired 0.55), clean-2 at 0.79, the distractors at 0.82 and 0.91 and noise-1
at 0.98: horizon 0 on the paired floor. Founder 315 reads clean-0 at 0.99, clean-2 at 1.00
and distractor-2 at 0.96, clean-1 at 0.73 (paired 0.47) and distractor-1 at 0.50: horizon 0,
with its external-history control at 0.76 on partial-1, under the 0.9 floor. Founder 313
reads chance on every condition but order-latest-1 (0.81) with a competent control. Partial,
replacement and order fail in all three. Erased and reset forks sit at chance wherever the
intact brain recalls, and the shuffled forks follow the transplanted trace.

**The operating-point screens behind the recipe (development founders 304, 305, 306 and 1,
32 training and 8 evaluation repeats; the selected recipe's receipt is
`results/development-finite-5-scale2-eta0.2-2026-10-10.json.gz`, the other screens' receipts,
2.7 MB each, are retained in the lane's evidence bundle outside the repository).** The
association layer of the composed brain is near silent: under the recipe its activations
have an RMS of 0.05 (0.09, 0.15 and 0.26 at sensory scale 2, 4 and 8), and the paired trace
distance at QUERY grows with the scale (0.005 to 0.028) while its ratio to the trace's RMS
does not. The novelty of a moment, the RMS change of the association state over the RMS of
the state before, does not tell the event types apart: a WRITE moves the state by about its
own size (mean 1.0 to 1.3), a neutral or a distractor event by 0.6 to 0.9, a QUERY by 0.7 to
1.2; only a neutral event after a neutral one is quiet (tenth percentile 0.05 to 0.1). A
write gated by novelty would therefore hold through repeated input only, and the chamber
has none.

| sensory scale | eta | 304 | 305 | 306 | 1 | control minimum |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5 | 0 | 0 | 0 | none | 0.98 |
| 2 | 0.5 | none (distractor-2 0.98, noise 1.00) | **2** (clean 1.00, 1.00, 1.00; distractors 1.00, 0.94; partial 0.86; noise 1.00; replacement and order 0.50) | none | none (clean-1 0.84, clean-2 0.86, distractor-1 1.00) | 0.50 (306) |
| 2 | 0.2 | none (clean-0 0.86) | **1** (clean 1.00, 1.00, 0.95; distractor-1 0.97; partial 0.78; noise 1.00; replacement 0.72; order 0.84) | none (chance) | **1** (clean 1.00, 1.00, 1.00; distractors 0.97, 0.86; noise 1.00; clean-4 0.95) | 1.00 |
| 4 | 0.5 | none (chance) | none (clean-1 0.75, distractors 1.00) | none | none | 0.50 (306) |
| 4 | 0.1 | none | 0 | none | 0 (clean 0.95, 0.92, 1.00) | 0.88 |
| 4 | 0.05 | none | none | none | 1 (clean 1.00, 0.91, 0.75; distractor-1 0.92) | 0.53 (305) |
| 8 | 0.5 | 1 (clean 1.00, 0.98, 1.00; distractor-1 0.88) | 1 (clean-0 1.00, clean-1 0.91) | none | **2** (clean 1.00, 1.00, 0.89; distractors 0.95, 0.92; noise 1.00; partial 0.72) | 0.50 (305, 306) |
| 8 | 0.1 | none | none | none | none | 0.50 |
| 8 | 0.05 | none | none | none | 0 | 0.67 (305) |

The horizon is the contiguous-prefix rule on the paired floor; the control minimum is the
external-history brain's lowest accuracy over the conditions. The effective step grows with
the scale, since a raw step is a product of activations: scale 8 with eta 0.5 reaches
horizon 2 in one founder and loses the control in two, scale 2 with eta 0.2 keeps the
control in all four and brings two founders to horizon 1. Founder 306 learns nothing at 32
repeats under every recipe of the table. Two further screens at scale 2 and eta 0.2 are
negative: a per-synapse leak (`decay` 0.005 and 0.02) leaves every founder's recall at
chance with the control at 1.00, and an intrinsic step on the association neurons as well
(a development share of the readout's rate, target 0.3) rewires the hidden layer at every
lesson, recall at chance in three founders and the control at 0.00 to 0.50 in three of four.

**Reading.** The readout's operating point (the gene) and the lesson rule make the control
competent and recall learnable in every founder at some evaluation; what the recipe does not
give is a mapping that stays, and it does not give replacement or order. These two have one
cause. The trace writes after every admitted event with one fixed weight, and the plastic
read of the trace learns a loop that re-expresses the held cue: a loop strong enough to hold
a cue through neutral and distractor events resists the next WRITE, so the brain answers the
first token where the latest is asked; a loop weak enough to let the replacement in loses
the cue to the distractor. A distractor carries its payload on the token's own coordinates
and differs from a WRITE in its marker only, so no write weight computed from the size of a
moment separates the two; the separation needs a write that is conditional on the marker
and learned, and the lesson at QUERY teaches the weights that settle the answer, not a write
two events earlier. That is the measured limit of the fixed-write trace under a teacher at
the query: horizon 0 on fresh founders with the clean and distractor conditions read at delay
0 and partly at delays 1 and 2, no revision. The candidate that follows is the write as an
action of the basal ganglia, chosen per moment and taught by a reward at the query with the
eligibility the actor keeps over its recent actions, in a recall protocol that pays
the answer instead of teaching it.
