# Existing results for local settling and recovery

Use established distributed-systems and operator results when investigating
Cadence's settling, recovery and cost. The task is to identify the theorem that
fits the executed operation, check its assumptions and measure its usefulness.
This guide adds no solver, mechanism, default or behavioral result. Start from
the [continuing System 1 brain](world-model.md) and its
[preservation review](../CONTRIBUTING.md#preserve-the-capable-foundation).

## Identify the operation first

For a free neural solve with fixed parameters and no adaptation, write

```text
F_c(v) = (1 - dt) v + dt (W act(v) + c).
```

`W` means effective signed weights, including contact counts and gains. `c`
includes bias and the current sensory, trace and associative-memory drive.
`Brain.act` holds that drive while qualifying the present graph. After an
action, trace updates, actual outcomes, memory writes and learning can change
the next solve. `live`'s arousal signal is an outcome-driven broadcast, not a
coordinate jointly solved in `F_c`. See [contracts](contracts.md),
[memory](memory.md) and [continuous interaction](continuous.md).

Agreement means satisfying these relations, not giving every neuron the same
activity. Keep five questions separate: does an equilibrium exist; is it unique;
does this iteration reach it; does the continuing life preserve useful context;
does its behavior succeed? Full-equation qualification answers a numerical
question at the measured state. The [existing contraction certificate](certificate.md)
adds a sufficient convergence condition. Neither supplies the behavioral answers.

## Recovery: closure and convergence

[Dijkstra's 1974 self-stabilization paper](https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD426.html)
defines legitimate global states, requires allowed moves to preserve legitimacy,
and gives local finite-state constructions that reach legitimacy from every
initial state. A central daemon selects one enabled move at a time; the guarantee
holds regardless of that selection. Legitimacy allows continuing
activity; it need not be one frozen configuration. These are specific algorithms,
not a theorem that every locally interacting network stabilizes.

Reuse the distinction in a Cadence recovery protocol: name the admissible
disturbance, the recovery condition, allowed subsequent inputs/updates and the
continuation over which that condition must persist. State whether convergence
means finite entry into a declared tolerance or asymptotic approach. A qualified
answer alone does not show recovery from every initial state, continued skill
retention, or repair of corrupted checkpoints. A tolerance set's invariance
must be checked for the actual update. When observations or parameters keep
changing, use a tracking question rather than assuming a fixed legitimate set.

**Status:** an established specification pattern for recovery tests. Cadence's
transactional refusals and saved continuation are narrower implemented contracts;
the whole life has no established arbitrary-state self-stabilization guarantee.

## Asynchronous settling: state the schedule and the norm

[Bertsekas, 1983, sections 3–4.2](https://web.mit.edu/dimitrib/www/Distr_Comp_Fixed.pdf)
gives asynchronous fixed-point convergence, including contractions in a weighted
maximum norm. A usable finite-dimensional specialization fixes one map `F` on
the whole space or an invariant weighted maximum-norm ball about its fixed point,
with `||F(x)-F(y)||_w <= q ||x-y||_w`, `q < 1`, where
`||x||_w = max_i |x_i| / w_i` and every `w_i > 0`. Initial stored values lie in that
domain; each coordinate keeps being updated; information used in future updates
eventually comes from arbitrarily late states. No coordinate or required
communication may remain permanently stalled. Contraction in another norm alone
does not supply this asynchronous result. Uniform delay and update-gap bounds
are additional assumptions when seeking finite work or latency bounds.

**Cadence mapping:** the row-mass certificate supplies a maximum-norm contraction
for its fixed free map. A future asynchronous implementation would still need
its own stored-state, communication and full-residual correspondence. A local
ellipsoid or signed-Jacobian stability observation cannot silently replace the
required norm/domain. Finite teaching phases, changed weights and held memory
also require separate treatment. This is a numerical-execution research option,
not evidence that current `Brain.compose` implements a distributed scheduler.

## Changing inputs: track a moving equilibrium

[Simonetto, 2017, Theorem 3.1(b)](https://arxiv.org/abs/1704.07338)
separates contraction tracking from weaker averaged-operator residual results.
For a common invariant domain and norm, uniformly contractive maps `F_t` with
factor `q < 1` and fixed points `z_t`, the elementary tracking calculation is

```text
x[t+1] = F_t(x[t]) + error[t]
||error[t]|| <= eta,    ||z[t+1] - z[t]|| <= delta
e[t+1] <= q e[t] + eta + delta,    e[t] = ||x[t] - z[t]||
e[t] <= q^t e[0] + (eta + delta) (1 - q^t) / (1 - q).
```

The calculation assumes approximate iterates remain in the certified domain.
It follows by inserting `F_t(z_t)=z_t`, applying contraction and the triangle
inequality, then iterating. This is a classical tracking bound, not a new
learning theorem. The cited theorem's exact-map case has `eta=0`.

**Cadence mapping:** changes in observation, trace, recalled records and effective
weights all contribute to movement of the target. Independently certifying two
checkpoints in different local norms does not establish a common tracking bound.
Measure drive changes and actual warm-start work before proposing reuse.

[Bernstein and Dall'Anese, 2018](https://arxiv.org/abs/1804.09768)
extends tracking analysis to imperfect maps and distributed updates with delays
and dropped communications, under explicit contraction and communication
conditions. It is the next reference if both moving targets and stale reads
matter; they cannot be handled by combining unrelated guarantees informally.

## Sheaf diffusion: compatibility can preserve different local values

A linear cellular sheaf assigns vector spaces to vertices/edges and fixed
restriction maps `R_ve` from vertex values into their shared edge space. For
an oriented edge `e=(u,v)`, define

```text
(D x)[e] = R_ve x[v] - R_ue x[u],    L = D* D.
```

Finite-dimensional inner products define the adjoint `D*`. `L` is positive
semidefinite; `ker L = ker D` is the space of compatible assignments. Local
values need not be equal. These are the degree-zero constructions in
[Hansen and Ghrist, 2019, sections 2–3](https://arxiv.org/abs/1808.01513).
An orientation supplies the sign convention; it does not make the symmetric
Laplacian an arbitrary directed neural operator.

With vertex values fixed on a boundary `B`, a harmonic extension solves
`L_II x_I = -L_IB x_B` on the remaining vertices `I`. It exists for this
degree-zero least-squares problem, and is unique exactly when `L_II` has no
kernel: no nonzero compatible assignment vanishes on `B`. See that paper's
Proposition 4.1 and its degree-zero discussion. A harmonic extension can retain
nonzero edge disagreement when the prescribed boundary is incompatible.

[Hansen and Ghrist, 2021, Theorems 4.1 and 5.1](https://arxiv.org/abs/2005.12798)
show that fixed-sheaf continuous diffusion `dx/dt = -alpha L x`, `alpha > 0`,
converges to the initial state's projection onto `ker L`; holding a boundary
fixed instead yields its nearest harmonic extension. Thus convergence, boundary
compatibility and uniqueness are different properties. For the discrete step
`x <- x - h L x`, spectral decomposition gives convergence on positive modes
when `0 < h < 2/lambda_max(L)` if `L` is nonzero; kernel components stay fixed.
The paper also studies specific changing restriction-map laws. Their conclusions
belong to those laws, not arbitrary plasticity.

Nonlinear edge potentials also have relevant prior results.
[Hanks et al., 2025, Theorem 2](https://arxiv.org/abs/2504.02049v2)
uses fixed linear restrictions on an undirected graph and differentiable,
strongly convex edge potentials `U_e`. If their joint minimizer `b` lies in
`image D`, continuous flow `dx/dt = -alpha D* grad U(D x)` converges to the
initial state's projection onto `D^+ b + ker D`. Their ADMM construction adds
closed, proper convex node objectives; Theorem 3 assumes a Lagrangian saddle
point. A finite inner diffusion tolerance needs its own error control before
claiming the exact-projection ADMM guarantee.

[Zhao et al., 2025, Algorithm 1 and Theorem 1](https://arxiv.org/abs/2510.00270v1)
studies asynchronous coordinate updates of that gradient form, with fixed
restrictions and globally strongly convex, gradient-Lipschitz edge potentials.
The intended schedule updates every coordinate in each `B+1`-step window,
bounds stale reads by `B`, and uses the current own-coordinate value. For a
sufficiently small fixed step, the stated result is periodic geometric decay
of the energy gap and recent squared increments. Check the schedule against
[Zhou et al., 2018, Assumptions 4–5 and Theorem 9](https://jmlr.org/papers/v19/17-444.html),
which its proof invokes, and derive constants from the actual restrictions and
potentials: do not import the v1 scheduler quantifier or spectral error-bound
constant unchecked. This result does not imply a unique minimizer, the same
limit under different schedules, or a fixed step valid for every delay bound.

**Cadence mapping:** these supply a linear comparison model for compatibility
and held boundaries, and nonlinear extensions with a specified gradient law.
Current nonlinear activation, directed effective weights, finite nudges, reward
updates and changing memories do not establish
`F(x) = x - h D* grad U(D x)`. Writing a squared neural residual and taking its
gradient generally changes the update by introducing its Jacobian transpose.
Do not replace the capable foundation to manufacture that correspondence.
Neither a linear section nor a harmonic extension is evidence of acquired
recall, environmental truth or a future phrase solved by `Brain.act`.

## Make the results serve existing work

Use the current [roadmap](https://github.com/muellerberndt/cadence/issues/109)
and existing capability owners. These are bounded consumers, not a new serial
prerequisite for developing the brain:

| Consumer | Useful question and simpler control |
| --- | --- |
| [Useful context #84](https://github.com/muellerberndt/cadence/issues/84), feeding [sequences #121](https://github.com/muellerberndt/cadence/issues/121) | After a vanished cue, identify the state carrying the distinction and measure its persistence/recovery under identical recent input. Compare carried/reset trace and held/read memory on saved copies. Conditional graph stability does not establish trace-feedback stability. |
| [Durable learning #169](https://github.com/muellerberndt/cadence/issues/169) | Separate changed graph parameters, overwritten associative records and transient readback before selecting a stability argument. Use the source-bound [competing-skill instrument](../benchmarks/competing/README.md); preserve its distinction between table-world and arena failures. Compare retention alongside acquisition/revision. |
| [Routine repair #122](https://github.com/muellerberndt/cadence/issues/122), [work #99](https://github.com/muellerberndt/cadence/issues/99) | Declare closure and recovery conditions; test whether a bound predicts refusal, sensitivity or saved work beyond residual, task margin and observed warm-start sweeps. Charge certification, memory, teaching, waiting and refused work. |
| [Event time #116](https://github.com/muellerberndt/cadence/issues/116), [actual outcomes #111](https://github.com/muellerberndt/cadence/issues/111) | Keep solver iterations, sensed events and received outcomes distinct. `wait` advances the trace while holding action ownership and eligibility; its caller-supplied custody is not learned delay credit. |

An analysis starts with a fixed source, acquired checkpoint, observation sequence,
allowed disturbance and work budget. Freeze any confirmation cases and behavioral
gates before selecting a candidate. Preserve free recall, associations, causal
context, actual-action ownership, private imagination and saved continuation.
Changing a schedule, state equation or learning law needs the ordinary mechanism
review even if the limiting equations are unchanged.

For each proposed result, record the prior theorem and exact statement, its
assumptions, the code correspondence, any unproved premise and the useful delta.
Reusing a theorem, calculating its constants and measuring its applicability are
valid engineering contributions. A new theorem requires a new statement and
proof; a capability claim requires task evidence. Close an analysis when the
map does not fit or it adds no decision beyond simpler instruments. Keep the
negative result rather than broadening the claim or weakening an existing gate.
