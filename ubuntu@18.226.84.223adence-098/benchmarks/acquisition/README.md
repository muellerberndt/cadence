# Acquisition and retention microscope

This bounded CPU protocol tests graph acquisition on actual sensory/action rows
from the recorded no-damage Castlevania movie. Its System 1 composition has
bounded observer-like regions, local neural state, sensory/action indices,
reciprocal constraints and local plastic repair; phase readbacks and checkpoints
retain the evidence. These independent-row controls omit working and associative
memory reads. They do not test a continuing world-model lifecycle. Every issued
answer requires the original full equations at tolerance `0.003` or the explicitly
selected stricter tolerance; a teaching target is never an answer clamp.

From the library checkout:

```sh
.venv/bin/python benchmarks/acquisition/run.py \
  --out /tmp/cadence-acquisition-run \
  --recipe both --updates 128 --check-every 8 \
  --free-steps 1024 --nudged-steps 1024 --seconds 120
```

Use a new output directory for each attempt. The finite control always keeps the
0.70 configuration: `650` inputs, `36` actions, modules `(32,16)`, no observers,
seed `0`, `1024/12/12` maximum phase sweeps, beta `0.1`, synaptic rate `0.5`, bias
rate `0.02`, cross-entropy temperature `0.2` and momentum `0.9`. `--seed` changes
both founders explicitly. Rate, phase budgets, nudge and damping arguments alter
the qualified candidate. They are candidate genes; the historical finite
configuration remains the control. Both use the raw sensor boundary and
float64, without calibration or an auxiliary classifier.

The explicit candidate `--gene fixed-lateral-local-rms` starts sensory biases at
`0.6`, freezes the existing motor-to-motor lateral synapses, and uses synaptic
rate `0.005` with local RMS normalization `0.99` and floor `0.001`.
This gene overrides `--rate`: changing that argument does not create a rate
comparison for this gene. Other synapses, every bias,
reciprocal tying and the System 1 mechanisms remain present. This candidate does
not change the default library. Use `--tolerance 1e-6 --free-steps 4096
--nudged-steps 4096` to check it with the original `dt=1` model and bounded
numerical damping.

`--gene lateral0-local-rms` removes motor lateral inhibition
(`Brain.compose(lateral=0.0)`), uses local RMS normalization `0.99` with floor
`1e-4`, and takes its synaptic rate from `--rate`. It transfers those two choices
from the Transcribe keyword control; this school has different inputs,
architecture, rates and data. `--gene lateral0-resting` additionally initializes
processing-region bias to `0.5`, with sensory, working-memory and motor biases
still zero; the chamber sets these biases itself (`resting`), the arrays the composed
brain's `resting_bias` built up to 0.84.3. Neither gene changes library defaults. The effective configuration
of every recipe is frozen in `protocol.json`, alongside requested arguments.

The [2026-10-03 source-era runs](https://github.com/muellerberndt/cadence/issues/110#issuecomment-5967884860)
did not pass the 24-row screen. Their retained receipts distinguish these cases:

| Gene | Effective synaptic rate | Nudged budget | Recorded outcome |
| --- | --- | --- | --- |
| `fixed-lateral-local-rms`, seed 0 | `0.005` | `512` | 15/24 qualified free recalls after 47 accepted updates; attempted update 48 refused its nudged phase at residual `0.00739573` and ended the stage. |
| `fixed-lateral-local-rms`, seeds 0--2 | `0.005` | `128` | 6, 8 and 8 of 24 at the terminal refused teaching attempt. |
| `lateral0-local-rms`, seed 0 | `0.02`, `0.05` | `512` | Four-row stage ended at 2/4 and 3/4 after the 128-update cap. |
| `lateral0-resting`, seed 0 | `0.02`, `0.05` | `2048` | Four-row stage ended at 3/4 for both rates after the 128-update cap. |

The fixed-gene runs named for requested rates `0.02` and `0.05` used the same
effective `0.005` recipe. Their recall counts at attempts 8, 16, 24, 32, 40 and
48 were 2, 5, 9, 10, 12 and 15; the final query followed the refused attempt,
which applied no update. Zero free-recall refusals does not mean zero teaching
refusals. Earlier `lateral0-local-rms` seed runs used a source-era fixed rate
`0.003`; the current gene's rate follows `--rate`. Reproduce each historical
result with its archived harness and source, not its directory name or the
current gene name alone.

`protocol.json` freezes arguments, fixture/runtime/source hashes, gates and
resource model before numerical work. The microscope reproduces the finite
free/positive/negative phases, checks the original residual through independent
edge scatter, reads one/two further undamped steps, and checks contrast
arithmetic with independent array products. It records the 24-row diagnostic
without teaching it. Training starts with two selected relations, then four,
then 24; each stage starts from a fresh identical founder. Expansion requires
perfect qualified free recall on the preceding stage. Each gate also requires
accepted learning exposure and improvement over that stage's founder. The
24-row screen requires at least `18/24` correct, zero free-recall refusals and
no refused teaching attempt or qualification violation. A screen is acquisition
evidence on selection rows.

Refused teaching is terminal in this protocol. It applies no update, retains
the failed phases and charges the work already attempted. Skipping or retrying
would require a separately frozen schedule and comparison; an identical retry
without any state or budget change is not new evidence. The existing CLI already
allows up to `4096` sweeps per phase, declared before the run. Increasing a budget
creates a new experiment; it does not revise the earlier 512-sweep result.

Every attempted training phase has raw potential, activation and adaptation
arrays, original-equation residuals, motor readback and work reports. Initial and
final checkpoints, refused phase states, unknown failures and capped attempts
remain in their own directories. Costs include accepted row presentations,
phase/query sweeps, independent checks, reported residual transports and saved
continuation replay. Transport estimates do not measure hardware operations.
The native `run.py` default limit is 60 seconds/64 MiB; it permits at most 300
seconds/80 MiB. The other instruments freeze their own limits before work.
These are bounded local runs and launch no cloud resource.

After a passed 24-row screen the frozen model reads independent TRAIN and
development panels once, then checks save/load predictions and one additional
teaching continuation. Same-set replay does not establish retention. A separate
family-disjoint interference test and five fresh-seed held-out confirmation are
required before a promoted acquisition/generalization claim.

The small fixture preserves the exact 24 historical selected rows and their
native frame/action/input hashes. Independent TRAIN supplies only 18 of these
families: six TRAIN classes occur once. Development supplies 19 selected
families. Fourteen families have disjoint TRAIN/development/test rows; their
balanced confirmation panels are frozen separately. The runner never reads
held-out arrays. This single movie supplies no fresh gameplay or emulator-parity
claim. The parent source and native receipt stay in the owning application.

Regenerate into a new folder from the local workspace parent fixture:

```sh
.venv/bin/python benchmarks/acquisition/extract.py --out /tmp/cadence-school
.venv/bin/python -m pytest -q benchmarks/acquisition/test_protocol.py
```

The extractor checks the parent witness/native/selection hashes and each selected
row's position, frame, label and little-endian float64 sensory hash. It copies
only the small panels, not the parent corpus. Provenance records the original
selection and source boundary; new results retain their own source identity.

Runs retain source bytes under `source/library/cadence`, plus the harness and
fixture. New protocols record the imported source's version and path separately
from installed distribution metadata, which can name a different wheel when
`PYTHONPATH` selects a checkout. Source hashes identify the executed code.
Replay an artifact using its archived source:

```sh
PYTHONPATH=/absolute/run/source/library .venv/bin/python \
  benchmarks/acquisition/verify.py /absolute/run
```

The verifier checks hashes, the complete scheduled case census, independent
phase residuals, replayed parameters and optimizer/counters, work sums and final
free recall. Its receipt names checks it does not replay. `--allow-source-drift`
permits diagnostic comparison and produces no source-bound pass. Earlier recall,
microscope trajectories, development and retention need their own independent
checks. Ten adversarial mutations test false acceptance, exposure/work counts,
altered phases/checkpoints, missing cases and changed summary/protocol identity.

## Identifiable relations and durable memory

Movie action labels are future-window summaries, not invariant semantic cue
families. Keep the native results above, including weak independent recall.
`relations.py` supplies a separate association instrument: 24 disjoint cue
families, a fixed shuffled mapping to 24 of the 36 actions, and independently
drawn nuisance variants. Four TRAIN instances, two development instances and two
sealed held-out instances belong to each family. A family credit requires every
instance in the queried panel to be correct. This measures recall of learned
families under nuisance changes; it does not measure native gameplay or inference
of unseen facts. Nearest-example and centroid controls establish that the declared
relation is identifiable; their answers are never Cadence answers.

```sh
.venv/bin/python benchmarks/acquisition/relations.py --out /tmp/cadence-relations
.venv/bin/python benchmarks/acquisition/relation_development.py \
  --out /tmp/cadence-relation-development --genes canonical \
  --updates 512 --seconds-per-gene 600
.venv/bin/python benchmarks/acquisition/continual.py \
  --out /tmp/cadence-memory-controls --relations --rounds 128
.venv/bin/python benchmarks/acquisition/continual.py \
  --out /tmp/cadence-memory-confirmation --relations --confirm-memory \
  --rounds 128 --seconds-per-arm 300
.venv/bin/python benchmarks/acquisition/graph_confirmation.py \
  --out /tmp/cadence-graph-confirmation --prepare-only
.venv/bin/python -m pytest -q benchmarks/acquisition/test_relations.py
```

The graph confirmation command above freezes the complete five-founder
protocol for review without teaching. Its full run has a 1,200-second
admission bound per founder and a separately recorded mandatory readback tail;
it is longer than the native microscope. Use a new output directory for an
executing run without `--prepare-only`. The declared source census retains all
five founders, including failures, and keeps their held-out panels sealed
unless each matching development gate passes.

The `canonical` graph control uses the parameter defaults of `LearnerConfig`
with explicit qualified phases, 4,096-sweep phase budgets, tolerance `.003` and
at most three numerical halvings. It is distinct from the implicit teaching
configuration of `Brain.compose`. Graph development uses only local free/±nudge
contrasts, with no associative read or write. Accepted phases retain per-row
equation checks and exact vector/parameter hashes for source-bound replay;
refused phases retain raw states. Cold queries start with fresh graph activity.

The `continual.py` memory arms explicitly observe actual cue/teacher-label pairs
through the existing consolidating store. They perform no graph contrast updates
and do not reinterpret a teacher label as a reward. Every answer still comes from
the original graph equilibrium with recall supplying drive at its motor ports.
Queries clear live activity, working trace and the fast associative
residual, retaining consolidated weights. `Brain.act(..., greedy=True)` is checked
separately in confirmation. These arms test durable associative storage, not the
graph learner's acquisition contract.

Retention starts from an actually acquired four-family checkpoint. Twenty new
families are then taught for a frozen schedule, with a separate rehearsal arm
that pays for old-example presentations. Old and new families are queried one
cue at a time after clearing temporary state. This does not require 24 facts to
fit simultaneously in working memory. The gates are at least three old and
15 new family credits, all scheduled rounds completed, and zero refusals.
Save/load checks include the next actual write or teaching update. Five-founder
confirmation retains every founder and opens held-out data only for a matching
recipe that passed development. A memory-control pass does not discharge the
separate local-contrast gate; neither pass proves lifelong retention or an
efficiency advantage.

## 2026-10-04 receipt snapshot and portable viewer

The [small offline viewer](demo/README.md) is included in source checkouts and
source distributions. It presents compact historical summaries measured with
recorded `0.73.0` source. Release packaging does not rerun those experiments;
raw phases, checkpoints, logs and the private parent movie corpus stay external.
The bundle retains all twenty founders: half rate 1/5, quarter rate 3/5,
zero lateral 4/5 and local RMS 5/5. Source/protocol digests, every recorded
readback, failures, work and continuation outcomes remain inspectable. Optional
receipt upload checks byte identity only, not scientific replay.

The controlled local RMS pass does not replace the native gate. The simpler
existing `modules=(32,)` control passes selected native 2/2 and fresh 4/4; its
24-row extension passes 18/24, but independent TRAIN 6/18 and development 0/19
fail the unchanged 14/18 and 15/19 floors. Adding previously queried TRAIN
examples in a matched coverage diagnostic reaches 3/19 reused development while
selected recall falls 23/24 to 13/24. Selected repetitions are halved for eighteen
actions, so that comparison does not separate dilution from actual forgetting.
All 94 separate reference fits and 22 outcomes fail the original transfer gates.
No external reference becomes the brain's learned answer head. A later raw text
preview may have exposed target/held-out plaintext; future confirmation cannot
claim potentially previewed rows were analyst-unexposed.

Actual own-outcome evidence is stronger on the scoped ordered-context task:
ABNQ/BANQ reaches 12/12 unused physical instances per order after transient
reset and saved restoration. Graph-only, C food readback and integrated trace
controls support only #85's first non-speech acquisition box. Historical actor
phases retained as hashes are not independently regenerated. A separate matched
partial-cue experience continuation recalls 360/360 reserved responses versus
347/360 for full-only, uniform 192/360 and joint reset 128/360. It reuses four
acquired seed-0 specimens in eight complete lives; no fresh confirmation or
integrated historical replay is claimed. Earlier capped retention failures and
the negative amplitude-2 readout remain in the research summary.

Optional resting bias zero and `0.5` both pass five fresh matched bounded
acquisition/retention/continuation pairs. Their unequal stopping and endpoint
scores support functional qualification, not a better default or speed claim.
#106's regression and explicit functional diagnosis merged in PR #136, preserving
the original `.25` assertion and strict xfail. **#85 and #110 remain open.**
Follow-up work preserves local agreement repair into coupled equilibrium and
first checks the simplest existing state, experience, context and wiring,
using abstracted animal behavior as the reference with finite limitations.
