# Agent guide: build one brain, measure its behavior

Follow the repository's [contributor instructions](../AGENTS.md). For API
spelling and numerical semantics, use [the reference](REFERENCE.md) and
[specification](SPECIFICATION.md). The practical explanation is
[brain design](BRAIN_DESIGN.md); start a small application from
[the quickstart](QUICKSTART.md).

## Explain the supported model first

Use `Cortex` to declare one graph, then `build()` to create its persistent
`Brain`. Inputs are supplied samples, populations contain processing patches,
and outputs expose selected patch states. Every population uses the same
bounded relation and coupled repair. There is no separate feed-forward output
network or external critic attached to each population.

The application still owns sensor meaning, action decoding and outcome units.
Choose useful connectivity and adequate information before increasing width.

## Use System 1 and System 2 for roles

**System 1** is a learned skill that keeps working with little repair: a groove,
a familiar movement or another competent routine. It can need several ordinary
layers and useful temporal information. It is not restricted to a flat map.

**System 2** is additional recursive observation and correction when that routine
cannot maintain equilibrium or meet longer-term needs. Useful correction should
restore competent behavior, preserve the skill and eventually need less work.
Unexpected success and predictable failure differ: surprise measures a missed
issued forecast; outcome valuation measures whether behavior serves the task.
Current patch error is a third quantity, recomputed from the current state and
relation. It is not a stored historical forecast miss.

These names describe the intended roles of one brain. They are not constructor
names, biological claims, or bootstrapping/live phase switches. Current
`0.60.0` supports the layouts below; it does not yet implement automatic
attention, independently progressing populations or the complete integrated
routine/correction cycle. Keep that boundary visible when writing examples.

## Choose among supported layouts

| Need | Public construction | Interpretation |
| --- | --- | --- |
| A simple direct relation | `column(..., inputs=senses)` feeding an output | Flat when it reads sensors only; unused patches do not supply hidden capacity. |
| Learned intermediate features | Another `column(..., inputs=earlier_population)` | Ordinary deep composition; all live states settle jointly, with returning influence through the energy. |
| Current internal mismatch as a learned feature | `observer(..., observes=earlier_population)` | Reads both states and exact prediction errors under the same patch rule. Measure whether this helps. |
| Specialized branches that share context | Columns and observers combined in one `Cortex`, then `build()` | One graph and body boundary. Branch structure does not give branches separate clocks. |

The [layout examples](VARIANTS.md) teach, query and resume all three patterns.
Run this command from the [tagged source checkout](../examples/README.md#get-the-example-sources):

```sh
PYTHONPATH=src python examples/layout_learning.py --layout all
```

This checks a tiny supplied relation, not matched capacity or a recursive
advantage. Previously declared sources give the public builder acyclic read
dependencies. Joint feedback is supported; explicit recurrent state cycles and
learned temporal memory are separate questions.

## Application recipe

1. **Declare one body boundary.** Define observations, output units, actual
   execution and measured outcomes. Keep preprocessing and action decoding
   explicit. Preserve enough causal context to distinguish required answers;
   future targets, unavailable teacher decisions and privileged environment
   state are not actor inputs. Past executed actions may be part of that context.
   Include measured actuator state when commands alone are ambiguous: a held
   note can be silent after its sample ends, just as a movement command can fail
   against an obstacle. Give a routine branch the signals its job needs; attach
   broader context where a controlled comparison shows it helps.
   A bounded `History` is explicit memory, not learned recurrence.
2. **Acquire a capable routine.** Start with flat or ordinary composition as the
   task requires. Check output connectivity and target range. Use public
   `observe`/`observe_batch` or `bootstrap`; measured labels are witnesses,
   derived teaching values use `source="estimate"`. A batch has private row
   states and one shared parameter admission, not an implicit sequence.
   When interpreting forecasts as probabilities, preserve the relevant event
   distribution or explicitly account for resampling. Replaying only events
   where an action was available can bias a shared predictor; supply selected-action
   value targets only where that action and its outcome were actually recorded.
3. **Test free behavior.** Disconnect the teacher and leave future outputs
   unclamped. Check `qualified` before acting and `accepted` before counting
   learning. Measure the actual body, not just prediction MAE or command flags.
   Refusals and timeouts remain outcomes. Compare independent assessment cases,
   old-skill retention, work and complete command latency.
4. **Attach consequences correctly.** Record the issued forecast before seeing
   its outcome, and preserve the action actually executed. For supported
   discrete reward learning, use the documented `Reinforcement` decision and
   execution acknowledgments. Its estimated Q targets are not automatic
   brain-wide emotion or a planning model. Keep outcome records separate from
   replaceable sensory summaries.
5. **Test correction as an addition.** Compare a competent ordinary control with
   an observer layout on the same causal information, with disclosed capacity,
   exposure and work. Establish routine, disturb the actual body, measure
   useful recovery, and recheck retention and cost after recovery. Include
   routine-plus-factual-fit as a control so consolidation alone is not called
   a planning benefit. Adding observers or changing a free output does not by
   itself establish System 2.

Use the operation that matches the intended state change:

| Intent | Operation |
| --- | --- |
| Inspect a free or conditional answer without committing it | `settle`; use `predict` for outputs only with refusal raised as an exception |
| Continue activity with parameters fixed | `step`, optionally with the same `targets`/`interventions` as `settle` |
| Admit one labeled experience and its activity | `observe` |
| Admit labeled examples while preserving current activity | `observe_batch` |

Clamps on `step` and `settle` condition only that call. They do not admit
teaching evidence; a goal-clamped future output is an intention, not a
forecast. A subsequent free call must qualify again. Label measured targets
as witnesses and derived targets as estimates; provenance labels do not
authenticate what the body actually did.

Save the brain with preprocessing, body state, data position, history and RNG.
For reward learning, save the complete learner so pending ownership survives.
Serialize calls to one brain; parallelize independent brains or collection.
`LiveController` keeps a caller responsive while one callback owns the brain.
It does not let an unfinished branch issue a new whole-brain qualified answer.

## Documentation and implementation checks

Keep examples on the existing public API. Prefer a link to one runnable example
over another wrapper or a new application-specific core abstraction. Currently
all populations use the same patch rule. Future specialized mechanisms need
explicit bounded state, ports, readback, learning and repair semantics, with
declared qualification and measured general benefit. Simplicity is a design
requirement, not proof that the current primitive can replace every memory
mechanism. Keep experimental mechanisms distinct from supported public behavior.

For future automatic fast/slow execution, state what invalidates reused work,
how a forecast miss or unmet need recruits correction, and what each
qualification covers. Separately settled or stale states cannot be presented
as one simultaneous global equilibrium. Concurrent action/learning proposals
and separately scheduled state blocks are different runtime claims. Neither
may silently exclude unresolved coordinates from current qualification.
The intended architecture should require no application attention flag or
per-population evaluator.

Keep implemented behavior separate from that target and from experimental
results. Query caching is arithmetic reuse, not learned attention; low
stationarity is not worldly success; a valid snapshot is not task competence.
The 0.60.0 release documents the implemented API; it does not establish the
future automatic routine/correction architecture described above.

After documentation changes, run the focused checks from the repository root:

```sh
PYTHONPATH=src python -m pytest -q tests/test_documentation.py tests/test_packaging.py
```

The documentation test executes every Python fence unless it has an explicit
reason to be skipped. Keep signatures, defaults, mutation semantics and refusal
behavior aligned with the reference; report behavioral evidence separately from
test counts. Runtime changes also require the full checks in the root guide.
