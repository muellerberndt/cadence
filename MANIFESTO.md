# Cadence, in one page

These are the design points every change to this library answers to.

Every change must meet these bars. [Contributing](CONTRIBUTING.md) gives the
required review and evidence checks; keep these two documents in sync whenever
a requirement changes. The goals below do not imply that every capability is
already implemented; the [world-model guide](docs/world-model.md) names current limits.

**The goal.** One continuing brain with the functional capabilities of an animal and a
human-like learner, built as a patch-net equilibrium that is more scalable, more capable
and more efficient than a transformer. Every result is measured against that goal at
matched information, matched task and a declared resource model.

**The hypothesis.** The brain settles in global equilibria. Patches repair local
disagreement with their neighbours until the whole graph agrees; the settled state is
the learned world model in the current situation. Learning is the same local repair,
driven by witnessed mismatch with reality and failed objectives. Every part of the brain
answers with a settled state of that one rule. There is no separate answer head.

Here agreement means satisfying the declared coupled relations, not identical
neural states or guaranteed correctness about the world. A fixed-input settle
holds its parameters and memory readback fixed; the continuing brain encounters
new observations, carries context and learns across such solves. Neural equation
residuals and the experimental population model's stationarity residual have
different contracts; stationarity can retain nonzero prediction error.

**Simple first; evolved brains are the reference.** Start from the simplest existing
System 1 and check state, memory, experience and available information before adding a
mechanism. Animal and human brains are the functional reference, with biology suggesting
tests, never supplying proof. Finite capacity, rigid interpretations and failure in an
unfamiliar environment are possible limits to measure, not a promise of a perfect learner.

**One continuing life.** A brain is composed once and lives on: short-term memory in the
working trace, long-term memory in the associative stores and the plastic weights, one
reward channel that carries relief and suffering, temporal credit through eligibility over
its own recent actions. The acquired brain persists across episodes; explicit terminal
handling can reset transient state while retaining learned parameters and durable memory.
Routine is equilibrium: a calm
brain answers from one greedy settle and learns nothing; surprise, pain and want raise its
arousal, and an aroused brain explores, learns from every outcome and writes memory. The
share of routine moments and the sweeps per moment are measured, not assumed.

**System 1 is the default; System 2 is optional.** System 1 is acquired competence in the
reciprocal graph with memory, plasticity, private imagination and action. System 2 is
observing regions that read the graph and return their influence into the same
equilibrium. Ordinary depth is not recursion, and a System 2 advantage is a claim to measure.

**Routine should be cheap; restoring equilibrium can be expensive.** A familiar situation
reuses learned organisation with little work; a surprise may cost many sweeps. A small
numerical residual is neither low energy nor competence; both are measured separately.

**Coherent behaviour over time.** A groove, a phrase, a sentence, a movement or a sustained
goal is one organised activity governed by retained context and learned relationships,
expressed through successive equilibria. Next-event accuracy, or a rollout fed its own
output, is component evidence at most. Tests use earlier context against identical recent
events, persistence through rests and missing feedback, and recovery after a disturbance.

**Language in the equilibrium.** Cadence formulates whole, consistent sentences from its
settled state, as the composer forms a groove from it: meaning held in context and memory,
words emitted in time, each word answering the same settled interpretation. This is a
dedicated goal, not next-word likelihood from what was said last. The capability to speak
belongs to the learned equilibrium of one life and its experience; it is not a statistic
distilled from ten billion training views in a data centre. Pretrained language models are
scored controls at matched information, never the hidden answer path.

**The world supplies problems, never answers.** Environments pay and hurt; nothing tells
the brain which action means what. Adapters render the world and report outcomes exactly
once. A score, a phase or an arrangement supplied by an adapter is not acquired by the brain.

**What Cadence is not.** Not a classifier, not an MLP, not a transformer with an attention
matrix, not a feed-forward pass with a trained readout, not a fresh brain per observation.
Those are controls and baselines, matched in information, kept beside every claim.

**What counts as progress.** Behaviour after learning in the same continuing brain: absolute
acquired competence, free held-out recall, appropriate forgetting, transfer, retention of
old skills and recovery, against uniform-random, frozen-copy and conventional online
learners at matched information, with every unit of teaching, settling, replay and refused
work counted. A better mean reward does not establish retention; a lower residual does not
establish understanding.

**Changing the library.** Every addition stays within local agreement repair into the same
coupled equilibrium, preserves demonstrated recall, associations, context, action
ownership, private imagination and saved continuation, and passes the
[preservation review](CONTRIBUTING.md#preserve-the-capable-foundation). Every constant
that would be a designed number of the brain is a gene with the composed default as the
founder and the control. Read [the world-model guide](docs/world-model.md) before building.
