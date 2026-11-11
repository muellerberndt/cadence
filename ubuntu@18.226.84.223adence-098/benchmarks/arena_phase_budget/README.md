# Full reward-phase budget: acquired-life comparison, 2026-10-09

**The bounded behavioral screen failed.** Four of six candidate lives meet both
approach and attack floors. Tumbler103 remains ineffective, and Hexapod302 misses the
attack floor that its unchanged continuation clears. The candidate substantially
reduces damage in the previously successful Tumbler102 and Mantis201. It improves
Hexapod301's attack relative to its continued control. This is mixed evidence, not a
repair of the failed fresh nine-life confirmation or a release-readiness result.

| Founder | Control mean progress (m) | Full budget mean progress (m) | Control total damage (HP) | Full budget total damage (HP) | Candidate progress / attack floor |
| --- | ---: | ---: | ---: | ---: | --- |
| Tumbler103 | −0.495 | −0.495 | 0.000 | 0.000 | Fail / Fail |
| Mantis202 | 16.787 | 13.411 | 228.686 | 145.366 | Pass / Pass |
| Hexapod302 | 1.000 | 3.798 | 36.658 | 12.983 | Pass / Fail |
| Tumbler102 | 25.023 | 22.701 | 389.333 | 54.073 | Pass / Pass |
| Mantis201 | 27.183 | 16.043 | 762.302 | 81.556 | Pass / Pass |
| Hexapod301 | 20.256 | 19.139 | 3.328 | 205.809 | Pass / Pass |

Progress is the mean across two frozen 2,000-moment worlds; damage is their sum.
Each candidate must beat both uniform-random and frozen-newborn mean progress by
more than `1e-9`, exceed both damage totals by more than `1e-9`, and deal at least
30 HP. All six must pass both for this bounded screen to pass. Five pass progress;
four pass attack. Passing would still require a separate fresh confirmation.

The three original behavioral failures and one previously successful founder per body
were fixed before launch. Ordinary continued training itself changes competence:
Hexapod302's control now exceeds 30 HP, while Hexapod301's control falls below it.
The table therefore compares matched continuation, without attributing every difference
from the original 80,000-moment checkpoints to the candidate. Exact control values,
target-channel ablations, all pair differences and failure status are retained in
[summary.json](summary.json).

## Single source intervention

Candidate commit `83b645726faa1ffef93eb05cfb9df423a57e2c5b` changes only reward
`ActorCritic._nudged_groups` settling from the ordinary absolute state-change tolerance
to `tolerance=0.0`. Both reward phases consume their already configured step budget;
the change does not hardcode a new step count or change the free settle, genes, rates,
temperature, memory setting or architecture. These arena checkpoints use a 12-step
reward budget. The control runtime is byte-identical to `750c8ef`; its new-work tree
has evidence-only commit `e20afe1`. Admission checks all 55 core source files.

The motivating numerical defect is concrete: absolute state movement can become small
before a local reward perturbation has propagated to an upstream sensory synapse.
Independent sensory-gradient regressions distinguish that defect from small useful
credit. The candidate passes those regressions and 114 existing System 1 preservation
tests, whose source-bound receipts are retained under `reviews/`. That mathematical
and test evidence does not establish the robot behavior that failed here.

## Fixed continuation and custody

All six candidate lives and two new Hexapod controls load the original checkpoints,
complete saved worlds and real outcomes still owed to their last executed commands.
Original inputs are admitted against the original frozen receipts, themselves pinned by
hash. Re-saving a loaded brain must match every original serialized array and metadata
field, including pending phase state, eligibility, working trace and random state.
Memory stays disabled. No old eligibility is recomputed: the source change affects
newly computed future reward phases only.

Each new arm executes exactly 4,000 additional actual moments, then freezes the saved
brain for worlds 2001 and 2002, 2,000 moments each. Each world includes intact, four
target-bearing/proximity-channel ablation, newborn and uniform-random controls. All new
physics uses corrected arena `af439c18b40fba05e6410b51a7ca12d05f626077`; all 15 arena
source files are admitted and checked afterward. Four earlier no-memory controls from
the [memory comparison](../arena_memory/README.md) have the identical runtime source,
initial state, physics and schedule and are imported by exact receipt hash. They were
not rerun, selected by outcome, or counted as new work.

The final actual outcome remains owed alongside the saved world, wrapper and brain.
Every saved life reproduces 32 physical steps in an unbroken/reloaded fork, with exact
commands, outcomes, non-timing work, world and final brain arrays. Outcomes received
advance from 79,999 to 83,999 during training and to 84,031 per fork. Fingerprints and
evaluations use private frozen copies, preserve working state, never learn and never
consume the real continuation's pending outcome. No checkpoint selection or horizon
adjustment occurred. All eight new jobs completed with no refusals.

## Complete work accounting

| Work category | Actual sweeps | Settle calls | Failed calls |
| --- | ---: | ---: | ---: |
| Eight new comparison arms | 4,330,227 | 309,025 | 0 |
| Two separate Hexapod preflight smokes | 10,429 | 684 | 0 |
| Four imported controls, previously executed | 1,838,178 | 148,964 | 0 |
| Six original acquisitions, previously executed | 21,025,048 | 2,066,931 | 0 |

The new comparison arms include 32,000 training moments, 128,000 evaluation moments,
512 physical fork moments and 64 fingerprint reads. Preflights add 336 physical moments
and 16 fingerprint reads. Original acquisitions account for 480,000 earlier training
moments; their probes and evaluations are included in the inherited work row. The four
imported controls account for 80,256 earlier physical moments. Neither inherited row
was executed again or included in the new totals.

Production wall time was 312.822 seconds with at most two workers and numerical
threads fixed to one. Summed overlapping per-arm elapsed time was 606.758 seconds;
it is not CPU time. Across all six matched lives, measured training learning sweeps
increase from 241,113 to 499,732. Total work including evaluation increases from
2,963,387 control sweeps to 3,205,018 candidate sweeps. These are measured totals;
behavioral paths and arousal also change.

## Retained evidence

The fixed protocol, both source admissions, exact historical producer, launches,
all 12 paired receipts, raw training outcome streams, preflight receipts and independent
reviews are retained. [manifest.json](manifest.json) checks compressed and raw bytes.
Large checkpoints/worlds remain local at the paths and hashes in the receipts. As with
the preceding acquired-life comparison, historical scripts retain their actual paths;
the compact archive does not claim to reconstruct those original brains independently.

The collator was hardened after launch to report missing, running or failed arms as a
negative complete census, retain partial errors/work, check unique arms and horizons,
and separate preflight cost. Its incomplete-census check is retained. This changed no
producer, source, protocol, behavioral threshold or trial outcome. Final collation
returns failure for the behavioral screen. Prior memory-preservation failures and the
original fresh-confirmation failures remain intact.
