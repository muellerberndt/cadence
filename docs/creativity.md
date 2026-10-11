# Creativity and evolving self-reflection

Cadence aims at general intelligence: a continuing organization that learns
from experience, preserves useful skills, explores alternatives and creates
solutions across environments. Its common structure is a network of bounded
observer-like patches with local state, ports, records, readback and repair.
Applications supply observations, executable actions and evaluation conditions;
they should not require a different basic learning mechanism.

## From disagreement to a revised organization

The intended loop is:

1. Interpret actual observations in the current patch-net equilibrium.
2. Expose a consequential mismatch through local or internal readback.
3. Explore possible responses and predict their consequences privately.
4. Test selected actions in the environment and admit what actually happened.
5. Repair learned relationships while preserving unrelated useful experience.

`TemporalPatchNet` supplies temporal interpretation, local learning, private
continuation and continuous-input planning. `TemporalMemory` supplies explicit
response constraints. Choosing worthwhile experiments, forming goals and
learning which memories matter are further capabilities, not automatic effects
of calling these APIs. An internally consistent prediction is not a new fact.

## What counts as creativity

Creativity means producing a novel response that serves a meaningful purpose
or satisfies constraints. Random variation, exact recall and interpolation
alone do not establish it. Evaluation should ask whether the proposal differs
substantially from acquired examples, meets the task's requirements, improves
through revision and remains useful when its context changes. Account for any
search, supplied structure, external evaluator and additional training.

The same questions apply to a new strategy, an explanation, a design, a physical
solution or a musical work. A human assessment can be relevant, but a positive
judgment about one result does not establish general creativity. Preserve both
objective checks and the actual feedback.

Evaluate the desirability of the goal itself. Matching an externally supplied
sound, trajectory or output pattern can establish control while making the
result less useful. Compare the proposed target, the executed result and an
unchanged reference. Feedback that rejects both target and result identifies
a goal-selection problem; it does not by itself identify a missing motor or
prediction mechanism. Goal formation and preference learning need their own
observations and transfer tests.

## Self-reflection uses the same patch contract

A patch may read a bounded summary of the system's own predictions, intentions,
proposed actions, uncertainty or unresolved disagreement. It can return a
constraint through ordinary ports. Further patches can read that coordination
process in turn. The research goal is to learn and evolve which summaries,
connections and depths improve behavior, using the same repair and detuning
principles rather than adding an independent controller for every application.

Current readback APIs expose detached state and diagnostics; the planner
revises proposals under a supplied goal. They do not implement an automatically
learned recursive hierarchy. The existing `evolve` function searches supplied
graph genomes under caller-defined fitness; it does not already evolve the
temporal core's self-reflection. More levels alone are not evidence of a
benefit: linear feedback can sometimes be absorbed into recurrent weights.

Compare aligned internal readback with matched ordinary recurrence,
disconnected feedback and shuffled readback. Preserve access to goals and
observations in every arm, and count added state and computation. Test whether
self-reflection improves correction, transfer, retained skills or creative
solutions. This is a functional engineering hypothesis inspired by OPH's
self-reading observers, not a derivation of intelligence or consciousness
from a physical theory.

## Transfer across uses

| Application example | General capability to test |
| --- | --- |
| Games such as Connect Four | Learning action consequences, maintaining context, revising a strategy and transferring across situations. |
| Language | Preserving relevant context and learned knowledge, producing useful new explanations, and correcting claims from evidence. |
| Multimodal perception | Combining compatible evidence across declared sensory boundaries without discarding decisive distinctions. |
| Embodied interaction | Predicting executable actions, observing their actual effects and adapting while retaining prior skills. |
| Music | Sustaining an intention, evaluating and revising a novel structured work, and transferring acquired relationships. |

These are acceptance directions, not capabilities established by the current
bounded experiments. Categorical actions, modality encodings and partial
observability need explicit tested interfaces; the present temporal planner
optimizes continuous input ports under a learned model.

## Demonstrations and the next research phase

A public demonstration should substantiate the capability it advertises. An
example may illustrate one mechanism without being presented as accomplished
general intelligence.

The [Amen demo](https://github.com/muellerberndt/cadence-demos/tree/main/amen)
starts from silence and computes a track in the browser, with a receipt that
reports next-event accuracy on tracks it never heard. That is creation in the
plain sense of producing material that was not recorded. The standard for
original creation is higher, and stated here: learn from all available material
and create new, complete tracks with coherent structure, assessed for corpus
coverage, meaningful novelty against what it learned from, intentional
development, independent listening and retained skills. A few recalled bars or
untaught mixtures do not meet it. Meeting it is work in progress.
