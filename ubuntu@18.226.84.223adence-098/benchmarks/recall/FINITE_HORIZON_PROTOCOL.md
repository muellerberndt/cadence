# Finite continuing recall protocol — proposed 2026-10-04

Status: protocol reviewed; scientific work paused on 2026-10-04 while #110/#85/#106
take priority. No learning run or new recall measurement occurred. This is one candidate,
not a search. Existing source-bound failures and the older chamber stay intact.
The release includes only the reviewed pure input fixtures and independent
trace-recurrence checks. This document specifies a future assay; it is not an
implemented worker, a launch instruction or a measured recall result.

## Question and declared limit

Can one continuing `Brain.compose` remember one of two observed token values
after its cue disappears, at a fixed state size, using its working trace rather
than a supplied answer or a warm numerical initial guess? The supported horizon
will be the largest contiguous passed prefix in **0, 1, 2 intervening accepted
observations**. Closure requires at least horizon 1 and the nuisance/replacement/
order gates below; horizon 0 alone is insufficient. Delays 4/8 and concurrent
loads 2/4 measure limits and cannot silently extend the supported horizon.

One event-time unit is an admitted observation and its one trace update. Solver
sweeps, private imagined observations and refused attempts are not real events.
No physical-time forgetting claim follows. Different irregular timestamps on
identical events must produce identical continuing state; different event counts
at matched elapsed time are separately measured.

## Fixed brain and learning law

Use the existing `vanished_cue.make_brain` unchanged: `inputs=28`, `cues=2`,
`modules=(32,)`, `lateral=0`, no associative store (`episodic=False`), trace
`decay=.8`, `amplitude=1`, and the constructor's association-to-prefrontal trace
with `focus=0`. This is a selected task gene, not the compose default or a new
default proposal. Compare it with a second brain with byte-equal initial arrays
and the same lesson budget that receives the external-history control below.

Resolved teacher config: `eta=.05`, `eta_bias=.005`, `beta=.1`,
`temperature=.2`, `free_steps=1024`, `nudged_steps=128`, `tolerance=.003`,
`normalize=.99`, `normalize_floor=.0001`, `momentum=.9`, `qualified=True`,
`damping=3`; other fields retain the exact source constructor values. Graph
parameters, biases, masks, neuron model and all resolved configs are serialized
before work. The normal zero initial biases and source neuron model remain.

CPU/float64 only, fresh founder seeds **301, 302, 303**, no development founder
and no seed substitution. Freeze Python, NumPy, Cadence import path/version,
float64/longdouble epsilon, platform, BLAS configuration and thread environment;
set OMP/OpenBLAS/MKL/NumExpr threads to 1 in the worker before imports. Freeze the
complete producing Python sources and imported library source capsule. Admit
the source/config/input hashes before the first solve, and reject source drift.

Every observation uses public qualified greedy `act`; it creates no reward
eligibility or pending actual outcome. Only a training probe has a teacher, via
the existing explicit `Learner.step(brain.stimulus(probe), labels)` path. There
are no rewards, fabricated moments, associative writes or resets between
episodes. A refused lesson is charged and skipped; a refused act ends that
model's life. No repeated exposure, retry or cold restart repairs a failure.

## Stored variables and observation ports

The live historical variables are the 32 association-coordinate values in each
row's `Trace.trace`, its 32 previous association values in `Trace.last`, and its
`cold` bit. Current neural potentials/activations are measured separately.
Storage has no item table or literal cue slot: the raw trace arrays have fixed
size for eight continuing rows. Graph weights are learned long-term parameters.

Input coordinates are: neutral=0, distractor=1, START=2, WRITE=3,
QUERY positions 0..3=4..7, token payload=8..11, external history=12..27.
Token 0 is `[1,1,0,0]`; token 1 is `[0,0,1,1]`. WRITE marks an observed token;
QUERY asks about a chronological token position within the current episode.
START is an observed world boundary, never an operator `brain.reset()` call.
The vanished-cue brain receives zero external-history coordinates throughout.
Probe payload is zero, and all current inputs are identical within each pair.

The source trace updates after **every** admitted observation; it has no explicit
write/hold/replace mask. Neutral/distractor events therefore update it as well.
Replacement is a later observed WRITE, not host mutation of the retained trace.
Only the full-reset control clears neural state and trace. No control clears or
installs the live original life. These rules are measured, not assumed to provide
perfect selective hold or revision.

## Frozen episodes and curriculum

Eight continuing rows form four pairs. Each pair has opposite queried values;
nonqueried events, noise, distractors, timing and current query are identical,
except the deliberate AB/BA order and opposite-token replacement histories:
those have different earlier WRITE observations but matched multisets/current
query. They are ordered-history tests, not identical-prefix controls.
Cue assignment is independently permuted at each episode, never selected from
the previous output. The sequence is START, one/four WRITE observations with
declared interspersed events, the declared intervening observations, then QUERY.
Only the final QUERY has a label. No earlier output is clamped or labeled.

Freeze **192 training episodes**: 16 instances of each of the following twelve
conditions, with their balanced order frozen once before any brain work. Both
brains receive the same real events/labels and each has at most 192 probe lessons.
Evaluation uses a disjoint RNG and **24 fresh episodes per condition**, with no
teachers or parameter updates. Freeze observations, expected historical values,
cue times/positions, shuffle permutations, timestamp variants, uniform-random
actions and expected per-condition denominators in compressed arrays first.

| Condition | Queried evidence and intervening events | Role |
| --- | --- | --- |
| clean-0 / clean-1 / clean-2 | One complete token; 0/1/2 neutral events | Horizon prefix |
| distractor-1 / distractor-2 | One complete token; 1/2 matched distractors | Horizon prefix |
| partial-1 | Retain one of the two redundant active payload coordinates; one neutral | Required nuisance |
| noise-1 | Add independently frozen uniform noise in [-.05,.05] to all payload coordinates; one neutral | Required nuisance |
| replacement-1 | Old token, neutral, opposite replacement token, neutral; query latest | Required revision |
| order-latest-1 | Tokens AB versus BA with the same multiset, neutral; query latest | Required ordered-token recall |
| capacity-2-first-1 | Two tokens; paired first tokens differ, second is matched; neutral; query first | Fixed-state capacity limit |
| capacity-4-first-1 | Four tokens; paired first tokens differ, later three are matched; neutral; query first | Fixed-state capacity limit |
| clean-4 | One token; four neutral observations | Beyond-horizon limit |

Also evaluate clean-8 without additional lessons. Distractor payloads are sampled
independently of labels, identical within pairs, and have the distractor marker
rather than WRITE. No evaluation cue is ambiguous: corruption preserves at least
.90 separation between the appropriate signal and nonsignal payload coordinates.

An external control stores the **actual observed payload arrays** of the last
four WRITE events after START, in chronological order. At QUERY it appends those
raw payloads into the sixteen reserved input coordinates. It never reads labels,
answers or target codes; partial/noisy cues remain partial/noisy. Its history
acquisition and readout are explicitly supplied assistance, with extra storage
and transport counted separately. The control's own storage is sixteen float64
payload values and one int64 token-count pointer per row (136 bytes, 1088 bytes
for eight rows); raw buffer reads/writes and appended-coordinate transport are
reported in addition to the brain's storage/solver work. Its scored free performance
that this control is competent, rather than treating it as a mathematical bound.

## Controls, numeric gates and denominators

Every evaluation query forks the same complete pre-query checkpoint into intact,
erased trace (retain warm neural state), shuffled trace (transplant trace/last/
cold from the opposite-value paired row), and full reset (retain parameters)
conditions. Each branch has the identical current query. The intact original
alone continues. The separately trained external-history brain has its own
continuing life. A source-frozen uniform-random policy has a pre-sampled action
for every row, including rows never reached by a failed model.

Each condition/founder has **192 planned rows and 96 paired trials**. A refusal
is wrong; unrun planned rows are wrong for acceptance, separately reported from
refused/attempted rows. No successful-founder filtering or pooled-only pass.
Report correct/planned, correct/attempted, both-paired-correct/planned-pairs,
absolute uniform-random scores, and every control's raw per-row answers.

For a promoted horizon, **every founder** must have zero refused/unrun rows and:

- intact accuracy >=.80 and paired-both-correct >=.75 on all clean/distractor
  conditions in its prefix, and partial-1/noise-1/replacement-1/order-latest-1;
- external-history accuracy >=.90 on those same conditions;
- intact accuracy at least .20 above the measured uniform-random score on each
  required condition (the binary random expectation is .50);
- for each nonzero delay in the promoted prefix, pooled clean/distractor erased
  and reset accuracies <=.60, and intact exceeds each by >=.20;
- shuffled accuracy against the **transplanted history's** value >=.80 and
  accuracy against the original value <=.20 on those same causal probes.

The permitted horizon is computed by the above prewritten contiguous-prefix
rule, not by altering floors after looking at results. Any failed nuisance,
continuation, purity, source or resource gate prevents promotion. Capacity 2/4
and delays 4/8 have their complete scores/controls reported without an assumed
pass; if the history control is incompetent, a capacity/retention explanation is
explicitly inconclusive. This protocol claims at most one binary historical
variable, not arbitrary source/order records, audio competence or action credit.

## Independent trace and saved-seam checks

Record every admitted observation's source activation, prior trace/last/cold and
resulting trace. A separate literal NumPy recurrence checks the actual source
law, without calling `Trace.update`: next trace is `.8*trace + .2*h`, last
becomes `h`, and cold becomes false. The constructor's `focus=0` supplies no
movement weight or explicit write gate. Require max
absolute discrepancy <=1e-12. Report paired trace and neural-state distances,
motor margins and their delay/load curves. These establish a measured finite
history-separation envelope on frozen reachable streams, not a universal theorem.

For the first evaluation episode of every condition/founder/model, save after
the first cue, immediately before a replacement when present, and before QUERY.
From each seam, the loaded clone must execute the exact remaining observations
with equal answers, qualification reports and complete saved arrays at the end.
All additional actual solver/trace work is charged; no branch supplies state to
the live original. After the midpoint, private `imagine` must leave a full
checkpoint unchanged. A zero-budget/tight-tolerance act on a changed cue must
refuse without changing that checkpoint or manufacturing an event; restoring
the declared solver config must preserve subsequent continuation.

Fork clean-2 after its cue to replay identical event arrays under two different
irregular timestamp schedules; demand equal final complete arrays. A separate
fork varies one versus two neutral events at the same declared elapsed time and
reports the resulting recall/state difference. Timestamps never enter the brain
or correlate with labels. Charge these branches and every private/refused solve.

## Resource and artifact admission

One hard worker wall cap: **900 seconds** including I/O and checks. Total retained
output cap: **160 MiB**, enforced during work and again after summary publication.
Progress after every 16 training episodes and every completed evaluation
condition, and at least every 30 seconds. No cloud use or hidden workers.

Write protocol, frozen arrays, complete source/runtime/model manifest and initial
checkpoints before teaching. Worker admission checks all bindings first. Keep an
atomic phase/current-condition census and append-only completed-call journal;
hard timeout, exception or cap preserves incomplete current work explicitly as
unknown, never as zero work or a passed summary. All original and control teacher,
free, imagined/refused phases, residual checks, trace updates, per-row work,
checkpoint I/O, memory/storage and wall time are reported. Sweeps are not joules.

Completed receipts bind sources, arrays, every retained checkpoint, work ledger
and acceptance calculation. Independent review recomputes pairing, labels,
denominators, recurrence, control scores and continuation before any closure.
If the supported horizon or causal gates fail, retain the negative attempt and
keep #84 open. Broader durable retention (#85), source/event records (#112),
physical/event-time integration (#116), delayed actual-action credit (#111), and
audio/sequence integration (#121) remain separate owners.
