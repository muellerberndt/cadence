# Acquired-life memory control, 2026-10-09

**The memory candidate failed preservation.** It recovered target-dependent approach in
the failed Tumbler103, but reduced the previously successful Mantis201's attack damage
below its untrained control. These results do not justify changing the arena's memory
default or claiming that the failed nine-life readiness confirmation is repaired.

This is a bounded diagnostic continuation of four original 80,000-moment lives, not a
fresh confirmation cohort. Each control and candidate received exactly 4,000 additional
physical moments, followed by two independent frozen 2,000-moment evaluations at worlds
2001 and 2002. No intermediate checkpoint was selected. The first stage used Tumbler103
and Mantis202; the conditional preservation stage used the previously successful
Tumbler102 and Mantis201. All eight planned arms completed and remain here.

| Founder | Control mean progress (m) | Memory mean progress (m) | Control total damage (HP) | Memory total damage (HP) | Memory with target channels removed (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tumbler103 | −0.495 | 16.923 | 0.000 | 18.171 | −22.061 |
| Mantis202 | 16.787 | 9.081 | 228.686 | 286.230 | 2.149 |
| Tumbler102 | 25.023 | 26.358 | 389.333 | 169.830 | −0.047 |
| Mantis201 | 27.183 | 12.062 | 762.302 | 6.117 | −37.205 |

Progress is the mean across the two frozen worlds; damage is their sum. Each arm
recomputes uniform-random and frozen-newborn controls on corrected physics. Mantis201's
newborn deals 58.211 HP, exceeding its memory candidate. Tumbler103's recovered approach
does not clear the original 30 HP attack floor. Mantis202 also recovers under ordinary
continuation, and memory reduces its progress while increasing damage. Tumbler102 keeps
both capabilities but has substantially lower damage with memory. No combat-win claim
is made. Full controls and exact values are in [summary.json](summary.json).

## The single intervention

Both arms load the same original checkpoint, full saved world, wrapper counters, random
state and actual outcome still owed to the final executed action. The control remains
`episodic=False`. The candidate attaches the existing compose-default component:

```python
brain.hippocampus = SynapticMemory(
    brain.sensory_index, brain.motor_index, consolidation=0.05
)
```

It begins with zero persistent and fast memory, zero writes and no invented history.
Every shared serialized array and metadata field matches before the first brain call.
The old pending action's forecast is preserved. No brain, working trace, critic,
eligibility or random state is reset. This is exactly the existing component and its
defaults, with no new learning rule or gain search.

The first owed outcome was physically produced by the original arena. All new physical
steps, including controls, use corrected arena commit
`af439c18b40fba05e6410b51a7ca12d05f626077`, which removes collision-separation phantom
weapon velocity. Core source is `750c8ef`; all 55 core and 15 arena source hashes are
admitted and checked afterward. The inherited original eligibility remains valid state;
neither arm recomputes old learning from unobserved history.

The terminal actual outcome remains owed in the saved world alongside the brain and
wrapper. Each saved life then reproduces 32 physical steps exactly against an unbroken
fork, including actions, outcomes, work, world and final brain arrays. Outcomes received
advance from 79,999 to 83,999 in training and to 84,031 in either continuation fork.
Frozen evaluation loads private checkpoint copies, preserves both memory timescales and
working trace, calls `act(greedy=True)`, and supplies no learning feedback. Target ablation
zeros the four target bearing/proximity channels only.

## Diagnosis and limits

An earlier observational microscope continued each of the same four lives for 1,024
moments on the original arena, without changing behavior. Tumbler103's routine outcomes
were poorly calibrated relative to sampled outcomes; most negative routine outcomes did
not modify the actor or critic, as the calm contract requires. The arena's disabled
memory removed the existing route for remembering wake-causing routine outcomes.
Critic clipping was rare. Passive replay with the composed default critic rate improved
one failure's calibration but not the other cases, so it did not support a general
critic-rate repair. [microscope-summary.json](microscope-summary.json) retains the mixed
summary; full captured feature streams remain local, with exact hashes and paths in
[microscope-local-artifacts.json](microscope-local-artifacts.json). Those larger local
streams are not included in this archive, and their work was not fully metered.

The memory arm wrote 84 actual routine outcomes for Tumbler103, including 78 negative
outcomes. It also wrote 2,747 sampled outcomes; this experiment does not isolate which
writes caused the recovery. Mantis202 wrote 31 routine outcomes and 2,361 sampled
outcomes. Earlier arena grids had also reported an unsuccessful episodic-memory arm
under a different founder recipe; this comparison does not erase that prior negative
evidence. The preservation failure here rules out a blanket improvement claim.

## Custody and cost

The eight production arms include 32,000 additional training moments, 128,000 frozen
evaluation moments, 512 physical continuation-fork moments and 64 fingerprint reads.
Every phase is included in 3,763,016 actual sweeps and 302,227 settle calls, with zero
failed settles or refusals. Summed per-arm elapsed time is 553.565 seconds; arms overlap
at two workers, so that sum is not wall time or CPU time. Smoke receipts are separate
and are excluded from these production totals.

[protocol.json](protocol.json), [sources.json](sources.json), launch records, exact
historical producers, compressed receipts and all training outcome streams are retained.
[manifest.json](manifest.json) binds archived bytes to original paths and hashes. Full
checkpoint/world artifacts remain under the local paths named in receipts, with hashes;
the compact archive cannot reconstruct those brains on its own. Historical scripts retain the experimental directory structure with normalized
storage prefixes; they are not portable clean-clone reproduction commands. The
[publication manifest](../PUBLICATION.md) maps original fingerprints to the published copies.
The original frozen helper comes from confirmation commit `9ecbd0f`; its hash is bound
in admission and receipts. The readiness gate and original failed census are unchanged.

Independent reviews check source admission, initial equivalence, real outcome chains,
reward arithmetic, memory write counts, exact saved continuation and all work totals.
An initial review-script mismatch compared wall-clock timing across forks; the corrected
review excludes only timing and retains that initial failed review log. No world trial
was rerun to obtain a preferred result.
