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
it is reported, not charged. The scan that finds a lesson's changed weights is one pass over
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
