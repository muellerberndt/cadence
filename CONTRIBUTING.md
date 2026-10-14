# Contributing

Issues and pull requests are welcome in this repository and in
[cadence-demos](https://github.com/muellerberndt/cadence-demos). This page says
how to set up, what the checks are, and what a change needs.

**Every change must meet [the manifesto](MANIFESTO.md).** It defines what we are
building and the design and evidence bars for code, defaults, documentation,
benchmarks and infrastructure. This guide turns those bars into review checks;
keep both documents in sync when changing a requirement. The manifesto's goals
are not claims that every capability is already implemented. Declare what a
change establishes, its limits and the applicable checks; infrastructure need
not demonstrate a whole-brain capability to ship.

## Preserve the capable foundation

**Required design rule.** Local agreement repair between patches into one
coupled global equilibrium is the fixed foundation. Every added mechanism must
participate through that same local rule and equilibrium; an external learned
answer path is not a Cadence extension. Start with the simplest existing System 1
composition and check its state, experience, sensory information, memory and
implementation before adding structure. Animal and human brains are the evolved
functional reference; we build abstractions of their solutions within their
environments, with finite capacity and possible rigidity.

For any mechanism, learning or default change, the pull request must identify
the demonstrated limitation, the affected local boundaries and update law,
the simpler control, and the preservation evidence specified below. Promotion as a repair or new default also requires measured improvement on
the declared limitation against that simpler control, with information and work
disclosed. A required existing capability regression blocks promotion.
Preserve failed comparisons and original acceptance gates; neither a biological
name nor a qualified wrong equilibrium waives them.

The goal is a simulated human-like brain built from simplified biological
mechanisms. Memory, plasticity, imagination and continuing interaction are
working parts of the default System 1 foundation. Optional System 2 adds
recursive cortical feedback to that brain, whose base can already be deep and
modular. The optional mechanism can ship without a proven task advantage; do
not turn research on its benefit into a release gate.

A replacement must preserve demonstrated behavior and saved continuation before
removing its predecessor. Keep source-bound evidence, original application
checkpoints and failed comparisons. Smaller code or a newer solver is not a
capability-preservation test. State the actual learning rule and the scope of
its equilibrium guarantee; record scans, temporal repair and graph settlement
must not inherit one another's claims.

Use [one continuing equilibrium brain](docs/world-model.md) as the application
and tutorial frame: bootstrap useful relations and memory, operate with them,
repair witnessed failures, then resume the same acquired brain. Parameters and
memory support a family of equilibria under changing evidence. Deep reciprocal
modules remain part of System 1; biological role names do not assert literal
biology.

Keep independent classification, calibration and memory-isolation tests as
labeled controls. A default application must not replace the continuing brain
with a new instance for each row, silently bypass memory or supply its answer
through an external trained readout. Preserve acquired state through disruption
and test both recovery and earlier capabilities. Numerical settlement, correct
world prediction and measured cost are different observations.

For a change to learning, memory or resolved defaults, compare the released and
candidate sources on the same acquired brains and tasks. Keep three cases
distinct: an explicit historical recipe, unchanged public arguments that may
resolve to different defaults, and a proposed opt-in recipe. Compare both fresh
acquisition and continuation from acquired checkpoints. Initial-array
identity and constructor coverage cannot replace post-learning behavior. Retain
the existing acquisition, trace, associative recall and reward-reversal floors;
check actual-action custody, private imagination and pending saved continuation.
Record absolute scores, refusals and work after acquisition, interfering traffic
and recovery within the demonstrated tasks, capacity and budgets. Measure
retention of useful old associations alongside revision of obsolete ones;
retaining every old response is not the preservation target.

A qualified wrong answer satisfies the tested settlement equations. That alone
establishes neither task correctness nor the capacity to acquire a required
relation. Local learning may change its relations or encounter rigidity and
finite limits; measure that response under the declared task and budget.
Before changing the local rule, independently check its actual update direction
on the failing case. Reference derivatives are diagnostics, not a replacement
answer path or installed learning mechanism.

Document the seam between working mechanisms and the intended world model.
The current `Brain` reward loop is not universal failure-gated plasticity, and
private responses to supplied observations are not learned world transitions.
A proposal for cheap stable use or automatic mismatch repair needs an explicit
contract and evidence before a guide advertises it. Do not change defaults or
invent success thresholds to make an example read as a completed capability.

## Scope and evidence for behavior over time

Every issue, plan and pull request must declare bounded component,
integrated-behavior or infrastructure scope. Name the applicable equilibrium
boundaries, the simpler control and preservation gates. Keep one acquired brain
through use, interference and recovery; a fresh brain per observation or an
external trained answer path is a control, not the intended continuing life.
Designed brain settings remain genes, with the composed founder as the control.

The manifesto's sequence and language goals concern coherent activity governed
by retained context and learned relationships. Next-event accuracy or a
self-feeding rollout alone is component evidence. Integrated claims need earlier
contexts against identical recent events, persistence through rests or familiar
missing feedback, and recovery after disturbance, with matched information and
exposure. Disclose supplied timing, phase, goals and arrangements. Today's
`Brain.act` settles a present state with held trace and memory input; it does
not jointly solve a future phrase or trajectory. No particular new module is
required: start with the existing System 1 state and memory.

Compare behavioral claims with uniform-random, frozen-copy and conventional
online learners where applicable, at matched information and work. Count
teaching, settling, replay and refused work, including routine and repair costs.
Keep numerical residuals, meaningful surprise, task success and physical energy
separate. Preserve original failed comparisons and useful earlier skills while
testing adaptation; neither a better reward average nor a smaller residual
establishes retention or understanding. Link the relevant capability owner under
[#109](https://github.com/muellerberndt/cadence/issues/109), rather than creating
a competing roadmap or silently waiving an unmet gate.

## Set up

```bash
git clone git@github.com:muellerberndt/cadence.git
cd cadence
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`[dev]` installs pytest, ruff, mypy, torch, Numba and SciPy. Python 3.11, 3.12 and
3.13 are supported on Linux, macOS and Windows.

## Checks

Every push runs these; run them before opening a pull request:

```bash
ruff check src tests
mypy
pytest -q
```

The test suite includes `tests/test_documentation.py`, which executes every Python block
of the listed guides in order and checks that every local link and anchor in `README.md`,
`MANIFESTO.md`, `CONTRIBUTING.md`, `AGENTS.md` and `docs/` resolves. Default pytest
collection also includes the acquisition, recall, rhythm, reversal
and key-door instruments, including their provenance, refusal, source-admission,
archive and confirmation-audit tests.
A change to a guide's code is a change to a test. The
[minimal-install job](https://github.com/muellerberndt/cadence/blob/main/.github/workflows/ci.yml)
builds the wheel with NumPy alone
and runs 20 pages without optional backends: `README.md` and, under `docs/`,
`world-model.md`, `patchnet.md`, `quickstart.md`, `temporal.md`,
`temporal-memory.md`, `architecture.md`, `planning.md`, `interaction.md`,
`partitioned.md`, `record-patch.md`, `build.md`, `belief.md`, `steering.md`,
`recursive-settlement.md`, `recursive-training.md`, `api.md`, `learning.md`,
`protocols.md` and `receipts.md`. It also executes
`examples/continuing_brain.py`, `examples/memory_imagination.py` and
`examples/equilibrium/layout_learning.py`. For changes to these guides or
examples, run `python -m pytest -q tests/test_documentation.py` and
`ruff check examples/continuing_brain.py tests/test_documentation.py`.

Mathematical claims must state their assumptions and connect them to the
executed numerical model. The [settling certificates](docs/certificate.md) explain
the supported contraction checks and their limits; the
[trace recurrence checks](tests/test_trace_bridge.py) state their reference
equations inline. These numerical checks are not formal verification of Python.
Any additional formal guarantee needs a self-contained, accessible proof and an
explicit correspondence to the operation that claims it.

### Short local feedback

Run the contracts affected by a change while iterating. For teaching, phase
qualification and actual-feedback transactions:

```bash
python -m pytest -q tests/test_qualified_learning.py tests/test_damping_stagnation.py tests/test_learning_reports.py tests/test_feedback_transaction.py
```

For gain selection, bias calibration, backend continuation and motor wiring:

```bash
python -m pytest -q tests/test_learner_calibration.py tests/test_calibration_backends.py tests/test_bias_qualification.py tests/test_composed_lateral.py
```

Run the default foundation and preserved population solver separately to locate
slow cases. These two commands together retain the complete test inventory:

```bash
python -m pytest -q --ignore=tests/equilibrium --durations=10
python -m pytest -q tests/equilibrium --durations=10
```

A fresh NumPy-only environment avoids optional accelerator startup and runs
the required-dependency paths. Use a separate environment; installing fewer
packages into an existing development environment does not remove its backends.
The environment below is inside the ignored `.venv/` directory:

```bash
python -m venv .venv/numpy-only
.venv/numpy-only/bin/python -m pip install -e . pytest
.venv/numpy-only/bin/python -m pytest -q tests --ignore=tests/equilibrium -rs --durations=10
```

Focused runs and a NumPy-only run provide partial coverage. Report optional
backend skips explicitly; they do not establish accelerator parity. Before
publication, retain the full suite and installed-package checks in the declared
environments. Run one substantial local QA process at a time on a busy machine.

## What a change needs

- **A test.** A new operation gets a test of its contract, and a numerical claim gets a
  finite-difference or reference check where one exists (`tests/test_equilibrium.py`,
  `tests/test_belief.py` are the pattern).
- **A line in `CHANGELOG.md`** under `Unreleased`, saying what changed and why.
- **A guide.** A public name is described in `docs/api.md`, and a new capability gets a
  section with a runnable snippet in the guide that owns it; add that guide to
  `tests/test_documentation.py` so the snippet keeps running.
- **Contracts kept.** Imagination does not teach or change live activity;
  cost counters may count attempted computation. Invalid calls and failed
  transactions cannot partially install parameters or records. Qualification,
  continuation and device portability follow the owning API's
  [contract](docs/contracts.md); do not transfer a graph guarantee to a finite
  repair or temporal model. The audit tests (`tests/test_audit_*.py`) check
  specific cases and backends.
- **Plain prose.** State what an operation does and what it does not establish, in the
  register of the existing guides.

## Layout

| Path | What lives there |
| --- | --- |
| `src/cadence/` | the numerical library: `brain.py`, `learning.py`, `genome.py`, `regions.py` (the settling brain); `temporal.py`, `planning.py`, `temporal_memory.py`; `record_patch.py`, `records.py`, `record_stack.py`, `record_ports.py`, `ports.py`; `belief.py`, `belief_torch.py`; `steering.py`, `life.py`, `instruments.py` (composition) |
| `tests/` | unit, numerical, integration and documentation contract tests |
| `docs/` | the guides; `docs/index.md` is the map |

## Examples

The archived atlas, browser renderer and local quickstart pages live in
[`cadence-examples/viewer`](https://github.com/muellerberndt/cadence-examples/tree/main/viewer)
and [`quickstart`](https://github.com/muellerberndt/cadence-examples/tree/main/quickstart).
Their tests run in that archive's declared environment and library version:
`python -m pytest -q viewer quickstart`. The core package no longer supplies
`cadence.atlas`, `cadence.demo` or the `cadence-demo` executable.

Current applications live in [cadence-demos](https://github.com/muellerberndt/cadence-demos).
Each demo's README names its library version, mechanisms, setup and behavioral scope.
Keep receipts and reproduction checks behind measured claims, and state data rights
and known limits. The separate research archive retains its own example layout and
library pins; use those when reproducing archived results.

The primary local entry is [the continuing brain](examples/continuing_brain.py):
one brain, actual feedback through changed conditions, and saved continuation.
The [memory/planning example](examples/memory_imagination.py) separately exercises
`TemporalPatchNet` and finite response protection. Do not present those distinct
model equations as an already integrated `Brain.compose` world model.

The preserved population solver lives in `cadence.experimental.equilibrium`,
with guides in `docs/equilibrium`, examples in `examples/equilibrium` and tests
in `tests/equilibrium`. Test it separately from the default foundation when
changing either API or numerical rule. The default runtime requires NumPy;
optional backends do not make it dependency-free.

Keep [GPL-3.0](LICENSE) as the current package license and preserve historical
attribution and license notices with their source.

## Releases

Releases are tagged `vX.Y.Z`, published to PyPI as `cadence-net`, and listed in
`CHANGELOG.md`. Pin a release or a commit for reproducible work.

Before publication, complete
source-bound foundation, consumer, CI and installed-artifact checks before
publication; record any missing application parity rather than inferring it
from a successful library suite.
