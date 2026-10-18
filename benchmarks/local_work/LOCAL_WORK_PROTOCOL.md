# Work follows affected knowledge: protocol of the first instrument

Declared 18 October 2026, before any campaign run. Everything below is a declaration; the
numbers measured by a run stay with that run's receipts, outside this repository.

## Hypothesis

In a settled brain, a bounded disturbance (one changed input, or one changed relation)
should be repaired by local work proportional to the affected structure, with every live
coordinate still qualified at the end and the same qualified answer as a full settlement.
Routine cost should then grow much more slowly than acquired knowledge. The instrument
measures whether that holds for the released solver on an acquired graph, and whether a
dependency-aware repair scheduler, written in the instrument on top of the library's settle
and residual primitives, reaches the same qualified answer with work that follows the
disturbance rather than the brain. It changes no library code; it establishes no energy
figure and no hardware claim.

## What already exists

`benchmarks/local_equilibrium` and `benchmarks/local_equilibrium_longer` compare the public
centered local learning rule with an implicit equilibrium-gradient baseline on one small
grammar judgment. They established that the local two-phase contrast learns a nonlinear
judgment on that fixture and that its gradient matches finite differences; they measure
learning, every phase settling the whole net. Neither measures the work of settling after a
disturbance, and neither holds more than one skill. `docs/certificate.md` gives the warm-start
step bound after a stimulus change under the contraction certificate and notes that an
unchanged input needs no repair step while the check of that still costs. `Brain.act`
already answers an unchanged drive with zero sweeps and one residual transport. This
instrument starts there.

## The brain and its skills

* Skill k reads its own block of `d = 4` cue neurons (one hot) and answers in its own motor
  slot of `a = 2` actions. Its map from cues to actions is drawn per founder and per k, with
  both actions present; skill 0, the disturbed skill, has the same map at every S.
* Two anatomies carry the same skills (`local_work.build_brain`):
  * `dense`: `Brain.compose(inputs=S*d, actions=S*a, modules=(sum(widths),), slots=S,
    episodic=False)`, the released composition: every projection dense, one association
    cortex shared by every skill, one working trace of that width. `Brain.compose` offers no
    sparse development. `Projection(density=...)` would thin a projection by a random mask
    drawn over the full source-by-destination array; it gives random, not block, sparsity and
    still allocates that array, so it is not used.
  * `sparse`: S single-skill compositions `Brain.compose(inputs=d, actions=a,
    modules=(width_k,), slots=1, episodic=False)`, developed from seeds `founder` (block 0)
    and `founder * 100 + k`, joined region-major into one block-diagonal connectome: each
    skill is a reciprocal sub-graph of its own (sensory block, association block, trace
    block, motor slot) and the whole is one composed `Brain` with one learner, one working
    trace and S motor slots. The parts' populations are kept so that the library's block
    layout cuts its ranges at the block borders. At S = 1 the two anatomies are one and the
    same connectome.
* Construction settings beside the composed founder, both anatomies: `episodic=False` (the
  founder's hippocampus is outside the neural graph and takes no part in a teacher-only life;
  without it the drive depends on the senses and the trace alone); the reward learner's rates
  `actor_eta`, `actor_eta_bias`, `actor_eta_critic` at zero, so the zero-reward feedback that
  `step` gives its previous sampled act changes no parameter and every parameter change comes
  from a lesson; `learning_eta = 0.5 S` and `learning_eta_bias = 0.05 S`, because the learner
  nudges every slot at `beta / slots` and divides the contrast by the full `2 beta`, which
  makes each skill's lesson step `1 / S` of the single-skill step: the scaling teaches skill 0
  at the single-skill rate at every S. Everything else is the released founder: tolerance
  3e-3, free budget 1,024 sweeps in chunks of 32, `dt = 1`, working-trace amplitude 3 at
  decay 0, trace projection scale 1.2, lateral inhibition -0.5 within a slot.

## Acquisition

All skills together, `Brain.step(observations, teacher=labels)` on 16 parallel streams with
uniform random cues per block, at most 400 lessons. Every 20 lessons `predict` (independent
samples, no trace) answers 256 fixed probe observations with uniform cues, drawn once per
founder; the gate is accuracy at least 0.95 on every skill's slot. Acquisition stops at the
gate. A case that does not reach it within the budget is reported as not acquired and its
work comparison is still reported, flagged. The lessons, the probe accuracies, the act
sweeps and checks during acquisition and the contraction certificate of the acquired brain
are recorded.

## The life and its disturbances

After acquisition the brain is reset to one stream. Two lives continue from the same acquired
parameters, through the same observation schedule, each answering greedily
(`act(greedy=True)`; no learning, no sampling) from its own kept state:

* the **full** life: `Brain.act` as released, which qualifies the whole graph;
* the **local** life: the `LocalRepair` scheduler below, followed by the library's own
  bookkeeping of a chosen action (`_choose`: the motor readout, the working trace advancing,
  the stream keeping the state), so that its trace semantics are the library's.

The first moment of both lives is one full settlement of the all-zero cues; the local life
then reads its synaptic inputs once (one complete transport, charged as its initial check).
Schedule:

1. **Settle in.** Hold the observation until both lives are stationary (the full life: zero
   sweeps; the local life: zero updates), at most 64 moments.
2. **Input disturbances, six.** Change skill 0's cue to a new one (the sequence 1, 3, 2, 3, 1, 0,
   each different from the cue before), every other skill's cue unchanged; measure the change
   moment; hold until stationary, at most 64 moments.
3. **Relation disturbances, three.** One lesson (`Brain.lesson`) at the present observation,
   the label of skill 0 the other action than its map's, given to both lives from their own
   states; the other slots' labels unchanged; measure the next moment; hold until stationary,
   at most 64 moments. The number of synapses the lesson changed is recorded.

Every moment of both lives is measured; the change moments and the held moments after them
are reported separately, and the stationary moments on their own.

## The scheduler, exactly

`LocalRepair` keeps, from the last complete check, every neuron's potential `v`, activation
`s = act(v)`, synaptic input `u` and signed equation error `e = u + drive + bias - v`.

1. A changed drive entry changes `e` of that neuron alone by the drive difference. A changed
   weight changes `u` and `e` of the contact's receiver by the weight difference times the
   sender's activation; a changed bias changes `e` of its neuron. The worklist is the set of
   those neurons whose `|e|` exceeds the tolerance.
2. A round takes the whole worklist at once. Every member moves by the library's own neuron
   rule, `v <- v + dt e`, publishes `s = act(v)`, and its activation change times each
   outgoing weight is added to the receiver's `u`: one message per outgoing contact. The
   errors of the members and of every receiver are re-read from the cached inputs; the next
   worklist is those whose `|e|` exceeds the tolerance. Nothing else is read or written.
3. When the worklist is empty, one complete residual transport re-reads every synaptic input
   and every equation, as the library's qualification requires; the state is qualified when
   the largest error is within the tolerance. The fresh inputs replace the cache. A moment
   whose drive and relations did not change does no work and no check: the kept certificate
   stands, the equations being unchanged.
4. If the complete check finds an error beyond tolerance (an exhausted round budget, or drift
   of the cached sums), its errors seed another repair; at most three complete checks per
   moment. A moment that ends unqualified is a failure of the prototype.

The round budget is the learner's free budget (1,024); the tolerance the learner's (3e-3).
The library's `residual` of the local state is computed at every moment, uncharged, as the
independent verification of the instrument's own complete check.

## Counting

| Count | Full life (`Brain.act`) | Local life (`LocalRepair`) |
| --- | --- | --- |
| neuron updates | `n` per sweep | worklist size per round |
| contacts, nominal | `E` per sweep and per residual check | messages pushed, plus `E` per complete check |
| contacts, effective | `E_recurrent` per sweep, `E_source` for at most two sweeps (the block transport reuses the products of ranges that hear nothing once still), plus `E` per check | the same as nominal |
| residual evaluations | `n` per residual check | members and receivers per round, plus `n` per complete check |
| sweeps / rounds | the settlement report's sweeps (chunks of 32) | rounds |
| complete checks | the settlement report's residual checks | complete checks |
| host seconds | wall clock of `act` | wall clock of drive, repair and bookkeeping |

The drive comparison that finds the changed entries is one pass over `n` values; in a
continuing life the sensory adapter and the trace writer know which entries they changed, so
it is reported, not charged. The implementation scatters a round's messages to their
receivers through an `n`-length accumulator, a shortcut of the NumPy implementation that a
compiled kernel would not need; it is not counted as work. The repair also records which
neurons it updated in a moment (the cone), per skill block. The scan that finds a lesson's changed weights is one pass over
`E`; the learner knows which synapses it moved, so it is reported, not charged. An uncharged
synchronous reference runs the library's sweep over every neuron from the same state and
stops at the first sweep whose complete residual is within tolerance: it separates the chunk
quantization of `equilibrate` from the saving of a bounded worklist. At the change moment and
the two held moments after it, a tighter solution of the same equations (`equilibrate` at
1e-7, budget 512, from the full life's state) measures both lives' states; it is uncharged
and may fail to converge, which is recorded. Host seconds are single threaded (BLAS, OpenMP
and Numba thread counts pinned to one); the transport is the library's NumPy block transport
when the compiled kernel is absent, and the environment is recorded with every run. Timings
are of this implementation on this host, not an energy measurement.

## Gates

Declared before the runs; a case that fails any of the first three at any moment is reported
as a failure of the prototype on that case.

1. **Same answer.** The local life's action equals the full life's action, every slot, at
   every moment of every case (100 percent).
2. **Qualified.** The local life's state passes the library's complete residual check at the
   tolerance at every moment where it did work; the full life's state is qualified by
   construction, `act` refusing otherwise.
3. **State.** The largest activation difference between the two lives' states is at most ten
   times the tolerance (0.03) at every moment. The contraction certificate does not hold for
   acquired brains (their row mass exceeds the certified limit), so no theorem converts two
   tolerance-qualified states into a bound; the gate is a declared constant, and the measured
   distances and the distances to the tighter reference are reported with it.
4. **Acquisition.** Every skill at 0.95 within 400 lessons, else flagged, as above.

Budgets: at most 30 host minutes per case (a case over it is flagged); founders 0, 1 and 2
for development; no confirmation founders are claimed by this instrument.

## Campaigns

* **skills**: S in {1, 2, 4, 8}, every skill of width 16, both anatomies, founders 0 to 2.
  The disturbed skill's block is fixed; unrelated knowledge grows with S. The decisive
  curve is local work against S at fixed disturbance, against full-settlement work against S.
* **widths**: S = 2, widths (16, w) with w in {16, 64, 256, 1,024}, both anatomies, founders
  0 to 2. The disturbed skill is fixed; the unrelated skill widens.

For every case: the change-moment work and the whole-episode work (change plus held moments)
per disturbance kind, the stationary-moment work, the cone (neurons updated at least once
in a moment, and in the sparse anatomy how many lie outside block 0), the cost of the
complete check against the rest of the local work, the moments to stationarity of each life,
and the gates.

## Reproduce

```sh
python benchmarks/local_work/run.py --campaign skills --founders 0 1 2 --out /path/out
python benchmarks/local_work/run.py --campaign widths --founders 0 1 2 --out /path/out
python benchmarks/local_work/tables.py /path/out/summary-skills.json
python -m pytest tests/test_local_work.py -q
```

`run.py` pins the thread counts before importing NumPy and writes `environment.json`, one
JSON per case under `cases/` and `summary-<campaign>.json`. `tables.py` prints the markdown
tables. Results are not committed to this repository.

## Amendment 1 (18 October 2026, after the first campaign): the local stop margin

Declared after the first campaign's results were read, so its runs are diagnostics and
cannot count as gate results of this protocol. ``LocalRepair(margin=k)`` and
``run.py --local-margin k`` make the worklist stop when every error is within
``tolerance / k`` while the complete check stays at the tolerance, the library's
qualification. The question it asks is whether the gate failures of the first campaign (two
lives qualified at the same tolerance answering differently at near ties, and activation
distances above the declared constant) come from the local life stopping exactly at the
tolerance where the chunked full settlement overshoots, at what extra work. Its case names
carry ``-m<k>`` and its summaries ``summary-<campaign>-m<k>.json``. A margin earns nothing
by itself: the same-answer and state gates would have to be met in a fresh declared campaign
on founders not used to choose ``k``.

## Amendment 2 (18 October 2026): validation that follows the cone

Declared before its runs. The first campaign found the complete check to be the floor of
every local count (its `E` contacts are 0.8 to 1.0 of the local contacts in the block
anatomy from S = 4 or width 64 on). This amendment replaces the charged check by one whose
cost follows the cone, verifies it against the library, and reruns the widths grid.

**The cone check** (`LocalRepair(check="cone")`, `run.py --check cone`). The scheduler
marks the neurons a moment touches: those whose drive entry changed, the receivers of
changed weights and changed biases, every worklist member, and every receiver of a message.
Only their cached inputs and errors can differ from a fresh evaluation; an untouched
neuron's cached input is the value of the last fresh evaluation and nothing that enters its
equation changed (no drive entry, no incoming weight, no sender's activation, all tracked),
so a fresh recomputation can return nothing else, and its cached error is within the
worklist threshold or it would be on the worklist. When the worklist is empty the check
recomputes the synaptic inputs of the marked neurons from their incoming contacts alone,
re-evaluates their equations, and qualifies the state when their largest error is within
the tolerance. The marks are cleared by a qualified check; a moment that ends unqualified
marks every neuron, so the next check is complete whatever its mode. Counted: the incoming
contacts read (`check_contacts`) and the equations evaluated freshly (`check_equations`),
beside the messages; `contacts` is their sum. Reported, uncharged: the largest cached
error outside the check (`cached_residual`).

**The verifier**, uncharged, at every moment of every case in every mode: every neuron's
equation error from the library's dense synapse matrix (`Brain.dense`, the synaptic input of
`s` being `s @ W`) against the instrument's kept errors, and the library's own `residual`
against the instrument's largest kept error. Both must agree within 1e-9 at every moment;
a disagreement at any moment is a failed case, reported as such.

**Campaigns.** Verification: the widths grid (S = 2, widths 16, 64, 256, 1,024, both
anatomies) on founders 0 to 2 with the cone check and the declared stop (margin 1); its
counts beside the check must equal the first campaign's on the same cases. Report: the
widths grid on founders 3 to 5, both anatomies, the cone check as the charged validation,
under both stop rules (margin 1, the declared stop; margin 10, amendment 1), with the gates
of this protocol (same answer, qualified, state) and the verifier's gate. Case names carry
`-cone` (and `-m10`); summaries `summary-widths-cone.json` and `summary-widths-m10-cone.json`.

**Prediction.** In the block anatomy the local contacts (messages plus check contacts) are
flat against the width of the unrelated skill, the check reading at most skill 0's contacts;
in the dense anatomy the check follows the cone, which is the cortex, so its contacts grow
with the width. The full life's counts are unchanged.

**A reading, no gate.** The lesson footprint under `learning_momentum = 0` against the
founder 0.9 (the gene passed at construction, `run.py --momentum 0`), block anatomy, the
skills grid (S in {1, 2, 4, 8}, width 16), founders 0 to 2, complete check, declared stop:
how many synapses a lesson on skill 0 changes, how many neurons outside skill 0's block the
repair after it updates, and the lessons the skills needed. Case names carry `-mom0`,
summaries `summary-skills-mom0.json`.

## Amendment 3 (18 October 2026): a recurrent baseline, a per-slot lesson, a damped fallback

Declared before its runs. At most two worker processes; each batch starts only when the
host's one-minute load average is below 12, checked before the batch and every five minutes
while it waits.

### (a) The recurrent baseline (`recurrent.py`)

**Learners.** `SkillGRU`: the gated recurrent unit of `benchmarks/competitors.py`
(`GRUCell`, its equations and initialization) with one softmax readout per skill slot
(readout initialization normal with deviation 0.1), learning by the stream chamber's online
rule (`OnlineGRU`): at every lesson the cross-entropy of the present frame is taken back
through the last 16 frames (their cached forward passes, the current weights), its norm
clipped at 5, and one Adam step is made (rate 0.003, betas 0.9 and 0.999, epsilon 1e-8).
Two extensions, declared: a batch of streams with one hidden state each and the loss the mean
over the streams and the slots; one softmax per slot. Initialization seed
`[founder, 21, S, H]`. `Bank`: S modules, module k a `SkillGRU` of one slot reading only
block k (seed `[founder, 23, S, k, H]`): the skill's block is supplied as routing, a
privileged control, the best a modular conventional learner with an oracle router can do.

**Workload.** The brain's: the lesson stream of `acquire` (`default_rng([founder, 5, S])`,
16 observations of uniform cues per lesson, every slot labelled), its 256 probe rows
(`default_rng([founder, 7])`), the gate 0.95 per skill probed every 20 lessons. The probe
answers from a zero state (independent samples, as the brain's `predict` reads no trace).
Budget 2,000 lessons, five times the brain's 400; reaching the gate within 400 is recorded.

**Width search** on the development founders 0 to 2: for S in {1, 2, 4, 8} the hidden
widths 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 in ascending order; the selected width
`H*(S)` is the smallest at which all three founders reach the gate; the search for an S
stops there. Every run is recorded.

**Report** on founders 3 to 5, on every block-anatomy, declared-stop, cone-check case of
those founders (the skills grid run under (c) below and amendment 2's widths grid): the GRU
at `H*(S)` (at S = 2 one width for every unrelated width, the task being the same), the
size-matched GRU at `H = sum(widths)` (the brain's association width), and the bank of
width `H*(1)` per module. Each is trained on the workload, then lives the case's own
sequence of moments (`schedule_from_case`): the same cues at the same moments, and before
each relation change moment one lesson at the present observation with the case's label
(skill 0's label moved on by `1 + r`, the other slots their maps). The lesson is not a
moment: the stream's state and cached frames stay as before it. The GRU's lesson labels
every slot; the bank's teaches module 0 alone. In the life the bank's router steps module k
at a moment when block k's input changed, when its weights changed, or when its own last
step moved its state by more than 3e-3 (largest coordinate); a skipped module holds its
answer; the router's decisions are free.

**Counting.** Per decision: the multiply-accumulates of one step of one stream, input
contacts `3 H D`, recurrent `3 H H`, readout `A H` (`A = S a`; the bank: the sum over the
modules it stepped), the units recomputed (`H` per step), and the host seconds of the step.
Learning: per lesson, the batch times one forward step and one backward step per cached
frame (`backward_macs + 3 H^2`), plus six per parameter for Adam (the `OnlineGRU` count).
Reported per change moment, per held moment and per episode (the change moment and its held
moments in the case's schedule), beside the brain's full life and its cone-following local
repair on the same cases.

**Competence.** Per skill: probe accuracy; accuracy on the probe rows whose joint cue
combination no lesson showed (held-out combinations; the instrument's `acquire` now records
the same for the brain, which changes no draw and no result); and the share of the life's
moments answered with every slot's map label. A reading at matched competence, no gate; a
learner that misses the gate on a report founder is reported with its cost.

### (b) The per-slot lesson (`slot_lesson`)

The library's finite teaching law of `Brain.lesson` (the drive with every memory the act
reads, a free phase from rest, one nudged phase each way from it, one contrast update) with
the nudge mask on the taught slot's motor neurons alone, at the learner's strength
`beta / slots` and the slot's cross-entropy at the learner's temperature: the taught slot
is pulled as in the library's lesson, the other slots not at all. Two variants: `slot`, whose
nudged phases stop at the movement tolerance as the library's do, and `slot-fixed`, whose two
nudged phases run exactly `nudged_steps` sweeps each. Instrument-only.

**Life footprint.** Block anatomy, skills grid, founders 0 to 2, momentum at the founder
(0.9) and at 0, complete check, declared stop: the relation disturbances with the per-slot
lesson in both lives (`--lesson slot`, `--lesson slot-fixed`), against the library-lesson
runs of the first campaign and amendment 2's reading. Read: the synapses the lesson changed
(all, outside skill 0's block, the largest change outside it), the biases changed outside
it, the neurons outside it violated beyond tolerance right after the lesson (the repair's
seed) and the repair's cone outside it.

**Relearning** (`run_relearning`, `run.py --campaign relearn`). From the acquired brain,
skill 0's map is reversed and taught by lessons alone, 16 observations of uniform cues per
lesson (`default_rng([founder, 13, S])`; streams reset, so no lesson reads a trace), with each
lesson kind on its own clone of the same brain, until skill 0's probe accuracy on the
reversed map reaches 0.95 or 400 lessons; probes every 20 lessons. Read: lessons to relearn,
every other skill's probe accuracy against its unchanged map (lowest through the relearning
and final), and each lesson's footprint. Block anatomy, skills grid, founders 0 to 2,
momentum at the founder and at 0. A reading, no gate.

### (c) The damped fallback (`LocalRepair(fallback=..., patience=k)`)

At the start of every round the worklist's largest error is compared with the smallest seen
in the moment's current check. After `k` rounds without a new minimum, or when the round
budget is spent with a nonempty worklist, the current state goes to the library's
qualified-answer solve with its damped fallback, `Brain._equilibrate` at the learner's free
budget and tolerance (at the founder's finite law: half the budget at the step, the rest at
half the step) from that state. Its sweeps (every neuron) and checks are counted as the
full life's are (effective contacts); every cache follows the returned state by one complete
transport, counted as a complete check. The returned state's equations qualify the moment;
an unqualified return fails the moment, as the library's act would refuse it. Patience 64
rounds (two of the library's chunks).

Runs: the brain's arms of (a), the skills grid on founders 3 to 5, both anatomies, cone
check, declared stop and margin 10, all with the fallback at patience 64 (`--fallback 64`);
and the one case that spent its 1,024 rounds under margin 10 (dense, widths 16 and 16,
founder 3, cone check), rerun with it. Read: how often it triggers per case and per moment,
the work it adds, and the gates.

### Reproduce

```sh
python benchmarks/local_work/recurrent.py search --out /path/out
python benchmarks/local_work/run.py --campaign skills --founders 3 4 5 --check cone \
    --fallback 64 --out /path/out          # and again with --local-margin 10
python benchmarks/local_work/recurrent.py report --search /path/out/search.json \
    --cases /path/out/cases /path/widths/cases --founders 3 4 5 --out /path/out
python benchmarks/local_work/run.py --campaign skills --arms sparse --founders 0 1 2 \
    --lesson slot-fixed --momentum 0 --out /path/out
python benchmarks/local_work/run.py --campaign relearn --arms sparse --founders 0 1 2 \
    --momentum 0 --out /path/out
```
