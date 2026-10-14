# The settling certificate

The certificate gives a sufficient condition for convergence and bounds distance
to the equilibrium after a finite number of steps. Full-equation qualification
can succeed without this stronger guarantee. For recovery, asynchronous updates,
changing inputs and linear compatibility models, see
[existing settling results and their assumptions](settling-literature.md).

## The rule

One settling step of a brain without adaptation moves every potential toward its synaptic
input plus stimulus:

```text
v <- (1 - dt) v + dt (W act(v) + c)
```

Let `L` be the largest slope of the activation and `rho` the largest absolute incoming
effective weight sum of any neuron (the row mass). If `L rho < 1` and `0 < dt <= 1`, one step
shrinks the largest potential difference between any two states by the factor

```text
q = 1 - dt (1 - L rho) < 1
```

Then there is exactly one equilibrium, and

- after `k` steps from any start, the distance to it is at most `q^k / (1 - q)` times the
  first step's movement;
- the last step's movement `m` bounds the remaining distance by `m / (1 - q)`;
- after the stimulus changes by `delta` (largest change of any drive), a warm start from the
  old equilibrium is within `tol` of the new one after about
  `log(dt delta / ((1 - q) tol)) / log(1 / q)` steps, and after no change it is there at once;
- with no stimulus and no bias the equilibrium is rest, exactly;
- removing a set of neurons moves the equilibrium by at most `dt` times the synaptic mass
  they sent, times the activation bound, over `1 - q`.

For the library's activation the slope bound is `(slope / 4) max(1 / (1 - rest), leak / rest)`
with `rest` the rest emission. Under `learning_neuron_model()` (slope 1, threshold 0, leak
0.1) that is `1/2`, so a learning brain is certified when every neuron's absolute incoming
effective weight sum is below 2. The contraction argument is the Banach fixed-point theorem
for the update map.

## Reading it

```python
import cadence as cd

connectome = cd.layered(8, 12, 3, density=1.0, init=0.25)   # row mass 1.39: certified
brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
cert = cd.certificate(brain)
print(cert.row_mass, cert.lipschitz, cert.rate, cert.certified)
print(cert.mass_limit)                 # the row mass below which this model is certified
print(cert.error_bound(1e-4))          # remaining distance after a step that moved 1e-4
print(cert.steps_for(change=0.5, tolerance=1e-3))   # warm-start budget after a change
```

`steps_for` raises `ValueError` for an uncertified brain; check `cert.certified` first.
`cd.row_mass(brain)` and `cd.lipschitz_constant(model)` are the two ingredients. The
certificate covers the free phase without adaptation; a nudge adds a drive the argument does
not include, and adaptation adds a slow variable. For those, and for any brain whose row mass
is above the limit, `NeuralGraph.residual` and `NeuralGraph.equilibrate` remain the checks: they certify
the equations at the state they measure, and nothing about uniqueness or convergence.
The certificate concerns the settled regions; records are read and written without settling.

Recompute `certificate(net.brain)` after learning: it reports the current effective
weights. The `recursive` builder's `coupling` bounds only the initialization.
Clipping each efficacy separately does not preserve an incoming row-mass bound.
A failed certificate means this sufficient contraction argument no longer applies;
it does not demonstrate that no equilibrium exists. Even a certified contracting
network may need more steps than a particular finite budget allows.

For supported smooth reciprocal CPU graphs, `PatchNet(..., solver="hybrid")`
adds optional energy refinement and a positive local-curvature check. This is a
different, local qualification: it can accept a stable equilibrium when the global
row-mass bound is unavailable, but does not restore a uniqueness theorem. Inspect
`phase.qualified` and its `phase.refinement` diagnostics. The
[recursive-training guide](recursive-training.md) covers that policy and its cost.

## Using it

- Report `cert.to_dict()` in every receipt next to the residual.
- Keep learned brains under the limit where you can: bound efficacies (the learner clips
  each plastic efficacy at magnitude eight), scale the gain by the fan-in, and watch
  `row_mass` over a life. A brain that crosses the limit may settle in practice; the
  certificate then says nothing.
- Warm-start streams. The step bound is logarithmic in the change, and an unchanged input
  needs no repair step; checking that it is unchanged still has a cost.
- Lesions keep the certificate: cutting synapses can only lower the row mass.

The movement passed to `error_bound` is **potential** movement. To bound the
returned state directly, use its next-step potential residual; the resulting
potential-error bound times `cert.lipschitz` also bounds activation error. The ordinary
`settle_batch(tolerance=...)` option measures activation movement instead and must not be
substituted into that bound. `equilibrate` directly checks the potential and adaptation
equations. A zero contraction rate (`dt=1`, zero coupling) needs at most one step after a
changed input. The warm-start budget assumes the old state is already the old equilibrium;
an approximate warm state has its own remaining error, even when the stimulus is unchanged.

For an unmasked free phase without adaptation, `NeuralGraph.residual` reports the equation
error before multiplication by the integration step. Convert it to potential movement
with `brain.neuron_model.dt * brain.residual(drive, state)` before passing it to
`cert.error_bound`. This conversion and distance guarantee require the certificate's
conditions; a small equation residual alone does not establish contraction.

## Equilibrium-propagation scope

`ep_structure(brain, fixed_inputs=...)` checks effective weight symmetry among the free
neurons. Its `compatible` flag is a structural diagnostic, separate from the contraction
certificate. Efficacy tying alone does not establish effective symmetry when contact counts
or presynaptic gains differ.

The [original equilibrium-propagation model](https://arxiv.org/abs/1602.05179) keeps its
inputs clamped. In Cadence, source neurons with no incoming effective weight settle to
potentials set by their fixed drive and bias. After convergence, their projections into
the free neurons act as external fields. These projections need no reverse edge for the
free-state energy argument. The diagnostic checks that excluded inputs are actually
sources; calling a recurrent neuron an input does not make it clamped.

Keep the excluded inputs unchanged and unnudged between phases. The derivative statement
then concerns projection weights, recurrent weights and free-neuron biases with the source
parameters held fixed. A positive structural check alone does not establish a gradient:
phases must converge on a smooth stable branch, finite beta retains estimation bias, and
adaptation lies outside this free-state argument. Convert the raw contrast using the
contact/gain factor and the loss-temperature convention in [learning](learning.md).

[`tests/test_equilibrium.py`](../tests/test_equilibrium.py) checks the converted contrast
against finite differences of the loss on a brain with symmetric effective weights and
converged phases. [`tests/test_runtime_checks.py`](../tests/test_runtime_checks.py) checks
that `ep_structure` accepts one-way input projections from declared inputs and rejects
them undeclared, as well as unequal presynaptic gains, a hidden neuron named as an input,
and adaptation.
