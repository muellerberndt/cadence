# Prospective robot training confirmation

**Result, 2026-10-08: the readiness gate failed.** All nine fixed lives completed
with valid source custody, no refusals and exact saved continuation. Seven of
nine passed the progress requirement; the preregistered requirement was eight.
The remaining greedy-transfer limitation prevents a readiness claim from this
confirmation. No source, recipe, horizon or gate was changed after launch.

This fixes a fresh cohort before running it, on the repaired source `750c8ef`.
The complete [protocol](protocol.json) names every body, founder, training world,
evaluation world, horizon and practical acceptance gate. It retains the existing
arena founder recipe without tuning. The previous mixed [development results](../arena_policy/README.md)
remain evidence: some early learned robots stood still, and the combined repair
underperformed the preceding source in the retained long-run and acquired-life
comparisons.

Nine continuing lives cover wheeled spike (Tumbler), four-legged hammer (Mantis),
and six-legged spinner (Hexapod) bodies. Each trains for exactly 80,000 nursery
moments. No checkpoint selection or interim behavioral probes are allowed.
Private frozen copies then act for 2,000 moments in each of two fresh worlds.
The controls are same-founder frozen newborns and same-body uniform random.
The target ablation zeros only the four target bearing/proximity channels;
other senses remain. All nine attempts count, including refusals and incomplete
attempts. The retained receipts are not replaced by retries.

For each life, average each metric across the two evaluation worlds. Its progress
gain is trained progress minus the larger of newborn and random progress; damage
gain is defined the same way. A positive gain must exceed `1e-9` in the metric's
units. At least eight of nine lives must have positive progress gain and at least
seven must have positive damage gain. A damage success also requires at least
30 hit points dealt in total across both evaluations (half a dummy's health),
so an incidental tiny collision does not pass. Each body must have at least
two of its three lives succeed at progress and at least two succeed at damage.

The sensory fraction is the sum over nine lives of intact minus ablated progress,
divided by the sum of progress gains. The denominator must exceed `1e-9`, or the
gate fails. The fraction must reach 0.5. No near-zero denominator is replaced or
clipped. Zero refusals, the complete cohort, source custody and exact saved
continuation are additional gates. These are fixed practical demo requirements,
not a statistical confidence bound on arbitrary future robots or combat wins.

The nursery's final real reward remains owed to its last executed action. The
receipt stores it, the next observation, every robot and arena state, both world
random generators, the dummy relocation counter, and the wrapper state alongside
the brain checkpoint. There is no synthetic reward, terminal flag or extra
unexecuted action. Before evaluation, a separate 32-moment fork compares the
unbroken brain with its loaded checkpoint in identically restored worlds. The
first call delivers the actual owed reward, once. Actions, rewards, complete
physics snapshots and resulting brain checkpoint arrays must match exactly.
These fork actions and all diagnostic work are charged separately; the training
checkpoint used for evaluations is unchanged.

Numerical work records unrounded answer/forecast sweeps and learning sweeps for
every training, continuation and frozen call. An observational wrapper counts
the returned sweeps from each actual `NeuralGraph.settle_batch` call, including
failed qualification and automatic retries, without changing its arithmetic.
Refusal attempts fail even when the arena wrapper's automatic retry succeeds;
the receipt retains completed partial totals and the refused call's work.
The eight-situation final probe
reports its own sweeps and elapsed time; there are no uncounted interim probes.
Wall-clock sections include policy calls and physics, and the outer receipt times
the full attempt. Sweeps are numerical work, not an energy or memory-I/O ledger.
Policy-dependent dummy destruction can change relocation times despite shared
initial world seeds. Frozen probes use private copies, receive no feedback, and
do not change the trained checkpoint.

Run each case in a fresh process, with at most three workers and one BLAS thread:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /path/to/arena/.venv/bin/python \
  benchmarks/arena_confirmation/confirm.py \
  --cadence /path/to/fixed-cadence --arena /path/to/cadence-robot-arena \
  --output /path/to/ignored-attempts --case tumbler-101
```

`--smoke` runs a separate declared non-cohort founder for a short lifecycle check;
its result cannot enter the confirmation census. The producer refuses to overwrite
an existing case directory and binds the exact source files, protocol, producer,
command, and checkpoint artifacts by SHA-256. [sources.json](sources.json) admits
only the fixed 55 Cadence and 15 arena Python source files before any training.
Large checkpoints and world
snapshots stay outside the repository.

Run `summarize.py /path/to/ignored-attempts` to check the complete census and all
gates. It checks required artifacts, their hashes, actual continuation records
and checkpoint arrays, source admission, horizons and the original metric math.
Missing and invalid cases fail. `--smoke --refusal-smoke` restricts only a separate
smoke brain's settling budget to cause a real qualification refusal and exercise
the failure receipt; it cannot enter the confirmation cohort.

The retained [preflight receipts](preflight/manifest.json) cover native-nursery
parity and exact 32-step continuation for all three bodies on non-cohort founders.
Every successful call's counted sweeps equal its answer plus learning sweeps.
A real one-step qualification refusal retains one attempted sweep and its partial
ledger, and a different runtime is rejected before training. The gate collator
rejects missing cohort cases and exits unsuccessfully when its gate is false.

## Complete confirmation result

Progress is the mean metres across the two 2,000-moment worlds. Damage is the
trained brain's total hit points dealt across both worlds. P/A records whether
the life passes its progress and attack requirements, respectively; attack
requires both beating the controls and dealing at least 30 hit points.

| Life | Trained progress | Newborn progress | Random progress | Ablated progress | Trained damage | P/A |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Tumbler 101 | 8.482 | -17.611 | 0.397 | 0.289 | 247.621 | pass / pass |
| Tumbler 102 | 20.257 | -0.178 | -1.138 | -0.047 | 180.637 | pass / pass |
| Tumbler 103 | -0.495 | -9.844 | 0.820 | -0.495 | 0.000 | fail / fail |
| Mantis 201 | 25.683 | 2.591 | -0.111 | -0.563 | 693.219 | pass / pass |
| Mantis 202 | -0.672 | -8.971 | 0.741 | 2.149 | 79.006 | fail / pass |
| Mantis 203 | 7.356 | 1.228 | -1.229 | -12.581 | 188.185 | pass / pass |
| Hexapod 301 | 12.781 | -2.864 | 0.277 | -0.233 | 43.359 | pass / pass |
| Hexapod 302 | 13.030 | -2.770 | -0.068 | 1.322 | 0.106 | pass / fail |
| Hexapod 303 | 17.427 | -1.135 | 0.575 | -3.836 | 49.077 | pass / pass |

| Gate | Measured | Required | Result |
| --- | --- | --- | --- |
| Valid completed census, source and custody | 9 of 9 | 9 of 9 | pass |
| Progress successes | 7 of 9 | at least 8 of 9 | **fail** |
| Attack successes | 7 of 9 | at least 7 of 9 | pass |
| Progress successes by Tumbler / Mantis / Hexapod | 2 / 2 / 3 | at least 2 each | pass |
| Attack successes by Tumbler / Mantis / Hexapod | 2 / 3 / 2 | at least 2 each | pass |
| Aggregate sensory fraction | 1.209077 | at least 0.5, positive denominator | pass |

The fraction can exceed one because some ablated policies perform worse than
the newborn/random comparator used in the denominator. It is a causal ablation
ratio with that declared denominator, not a percentage of intelligence or a
probability. The full frozen verifier returned `pass: false` and exit status 1;
there were no invalid or missing receipts. Six of nine lives pass both the
progress and attack conditions. An independent rerun of the unchanged verifier
returns the exact same JSON and failure status.

The means do not imply success in every world. Tumbler 101 makes 26.952 m in one
world and -9.988 m in the other, and takes substantial damage. Mantis 202 does
more damage than the controls but fails progress; removing its target channels
improves progress. Hexapod 302 approaches but deals only 0.106 hit points, so its
tiny collision does not count as successful attacking.

Tumbler 103 retains the strongest greedy-transfer failure: it gives the same
commands in all eight fingerprint situations, deals no damage, and its intact
and ablated world outcome totals match. The retained read-only diagnostics check
source and checkpoint identity, command decoding, current-observation settlement,
working-trace use and saved actual-temperature eligibility. They find no further
implementation defect. Sampled training behavior with a small motor probability
margin does not guarantee useful greedy behavior. These diagnostics neither
remove the failure nor establish a new repair; no extra world trials were run.

## Work, custody and archived bytes

The census ran from 11:48:34 to 12:36:05 UTC on 2026-10-08, about 47 minutes
31 seconds, with at most three workers and one numerical thread per process.
The sum of receipt per-case elapsed times is 7,586.863 seconds; those intervals overlap
and are not CPU time. All nine complete the same 80,000 training moments
(720,000 total), followed by 144,000 evaluation world moments, 576 continuation
world steps across the paired forks, and 72 individual fingerprint answers.
The observational meter counts 31,812,258 actual settling sweeps in 3,117,555
calls, with zero failed settle calls or policy refusals.

Each training checkpoint owns its final executed action with 79,999 outcomes
already received and the last real reward stored beside its world. Each resumed
32-step fork finishes with 80,031 received outcomes. The unbroken and reloaded
forks match in commands, rewards, physical state and resulting checkpoint arrays.
Frozen evaluations and fingerprints perform no learning and leave the training
checkpoint unchanged. There is no fabricated terminal outcome or flushed extra
action.

The [archive manifest](results/manifest.json) binds the exact raw and compressed
bytes of all nine receipts, process logs/statuses, launch commands, final summary,
completion record, independent root verification, orchestration producer and four
diagnostic files. Gzip uses a
zero timestamp; decompression restores the original bytes. Complete checkpoints,
world snapshots and progress files remain in the manifest's ignored local run
directory, with their hashes also bound by each receipt. The verifier checked
those artifacts before the evidence was archived. The pinned runtime admission,
producer and protocol remain the original frozen bytes. This archive preserves
the failed readiness result; it does not authorize or announce a release.
