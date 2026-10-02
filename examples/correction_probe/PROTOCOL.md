# Actual sensor constraints and a withheld physical measurement

**Final goal.** A patch-net equilibrium brain that is more scalable, more capable and more efficient than a transformer. Every Cadence result is measured against that goal at matched information, matched task and a declared resource model. The Amen jungle composer is the first test platform.

**Principles.** The main hypothesis stays: a brain that settles in global equilibria. The building block stays as simple as possible, like in nature, and every part of the brain answers with a settled state of that same patch rule; a feed-forward readout or a copied input is a baseline, never a result. Natural evolution is preferred to design: any parameter that looks designed is a gene, picked by selection against the hand-set value, which stays as the control. Within a life only the patch rule learns; across lives the genome evolves.

This is a small development prerequisite. It tests prediction and factual
learning, not autonomous control, reward learning, attention or independently
clocked populations. Prior actuator studies already clamped actual past
predictions; this experiment does not claim that idea is new. It replaces their
inverse-action decomposition with an identifiable partial-observation task.

## Body and causal boundary

The body has two private coordinates. An exogenous, recorded two-axis command
produces `z_next = .35*z + .6*command`. Four independently defined physical
measurements are:

```text
p0 = tanh(x + .5*y)
p1 = tanh(-.5*x + y)
c  = tanh(x - y + .25*x*y)
f  = tanh(.7*x - .4*y + .35*x*y)
```

The actor receives the command, a noisy coarse position sensor, exactly one
current fine measurement, `c`, presence bits and the previous public fine/C
packet as an immutable persistence forecast. Its unavailable P coordinate stays
zero; the alternating shutter makes the previous visibility the opposite of the
current visibility. That previous record was issued before
the present transition. Body coordinates, calibration changes, missing current
fine measurement and current `f` never enter model inputs. Each available
`p_i,c` pair identifies the bounded physical state; there is no hidden-answer
ambiguity. The corresponding determinant checks are fixture tests, not an
oracle supplied to the brain. Sensor noise and occlusion draws are independent
of model/label selection. The withheld `f` is physically revealed only after
the prediction is recorded; its measured value supplies the later witness.
It measures the same post-command state as the available fine sensors. This
is current-state completion with a delayed reveal, not a pre-action forecast
of the future world; the internal population name `future` does not change that.

This sensor readout is a synthetic body, not a native game or audio device.
The tape is fixed independently of all predictions. It tests prediction, not
how a learned controller changes its own future data distribution.

## Layout and learning controls

H reads coarse/prior/command; P predicts the two fine sensors from free H.
C reads H/coarse/P states; only the observer adds current P errors. C is
clamped to the actual independent measurement during corrected queries.
A nonlinear G head reads H/P/C and **every raw actor coordinate**; F reads
G and the raw coordinates again. Thus ordinary controls are not denied facts
or a nonlinear representation. Both ordinary and observer use H4/G4; wide
uses H8/G8. Exact counts are 12 states/153 parameters, 12/155 and 20/317.
An equal seed means paired random initialization, not identical common weights.

All three use the same public patch law and 512 actual training rows replayed
four times: 128 batches × 16 rows, 2,048 presentations. Witness targets clamp
the visible P, C and measured F; missing P stays free during both teaching and
testing. All test F heads are free. Eight same-weight free-F versus factual-F
clamp comparisons expose teacher-driven feature changes without scoring a
clamped output as prediction. Priors, budget and all settings live in DESIGN.
No hidden label, hand-coded residual target, auxiliary optimizer or admission
of a corrected latent state is used.

## Fixed endpoints and branches

Record clean 64-row routine and corrected error at updates 0/32/64/128. A
routine query receives all raw facts but no output clamps; corrected query
also clamps the actual visible P/C. All three must attain MAE ≤ .04 in both
modes and halve their newborn error before a comparison is eligible.

From each acquired checkpoint run all four branches on the same 64 physical
transitions: routine/no learning, corrected/no learning, routine/actual fits,
corrected/actual fits. The coarse sensor calibration shifts by (+.22,-.18)
at transition 16. Neither that index nor the bias is a model input. At the
fixed boundaries after transitions 31 and 47, fitting branches receive one
batch containing the preceding 16 actual outcomes. This schedule is a diagnostic
control, not automatic attention. All branches still run if acquisition fails,
but their evidence cannot then establish a learned recursive advantage.

All queries are pure; batch learning preserves activity. The two no-fit final
snapshots must exactly equal the acquired snapshot, and the two fit final
snapshots must exactly equal each other. This separates changed factual
inference from learning exposure. Post-fit error uses transitions 48–63 and
separate 64-row clean/shifted queries. Report all old-skill changes and require
old clean MAE increase ≤ .005 and shifted MAE reduction ≥ 10% for a descriptive
retention success. No time/memory learning claim follows from those pure queries.

The development quality contrast uses corrected/no-fit MAE over all 48
post-change observations. Observer must improve at least 10% **and** .002
absolute over each of ordinary/wide, for every declared seed. Also require
corrected observer error at least 10% below its own unclamped routine error;
a comparator win without useful factual correction does not pass. Count all work,
including shared acquisition, checks, diagnostics and fitting. A quality
advantage with higher work is not an efficiency advantage. Record work per
endpoint so routine versus corrected comparison can be made without discarding
extra calls. This small fixed gate is not statistical confirmation.

## Execution and custody

First freeze and run only preflight seed 3109, all three layouts. Fresh screen
seeds are 3119/3121/3137, but that screen requires a separate freeze/review;
the executable never automatically expands a cohort. An eight-seed confirmation
would need an additional prospectively frozen design. A failed acquisition
is inconclusive about recursive capability, not an excuse to select a favorable
layout/seed or silently add dose. Any subsequent modification preserves this run.

All seven mathematical modules must exactly match public commit 70e7a6a before
and after execution. Private unconnected scheduling prototypes are outside the
study. Sources, all rows, schedules and native founder snapshots are frozen
before outcomes. Each arm has 60 seconds after setup and an external 65-second
process cap. The initial preflight runs serially. Dataset plus compressed full
returned-call journals/checkpoints must remain below 70 MB; this is a between-arm
and final eligibility guard, not a promise of instantaneous storage cancellation.
Local data stay below the workspace's 100-MB threshold. All planned cases,
refusals, unknown interrupted work and launch errors remain in the census.

Per arm, completion requires 128 training and four branch admissions, 1,296
pure queries, zero unknown calls, exact twins and unchanged sources. Results
contain every free prediction and physical witness; parameter arrays are hashed
in query records and full checkpoints preserve continuation. An independent
replay must reconstruct every row, target, numerical result, snapshot and gate
before treating this collector's report as verified evidence.

From the repository root, after source/tests are reviewed:

```sh
PYTHONPATH=src python examples/correction_probe/probe.py prepare /tmp/correction-preflight --phase preflight
PYTHONPATH=src python examples/correction_probe/probe.py run /tmp/correction-preflight
```

Always choose a fresh output path. These commands are the small component
experiment, not a Cadence Doom training command or a new public runtime API.
