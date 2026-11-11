# Local equilibrium learning on finite grammar judgments

This package compares the public centered `Learner.update` rule with an implicit
equilibrium-gradient baseline on the **same network**, a frozen network, and an
affine classifier. The task is subject/verb number agreement across a distractor
noun in a prepositional phrase. Supplied role-specific one-hot inputs identify
the words; the binary agreement judgment requires an interaction between subject
number and verb form. The held-out examples introduce unseen subject/distractor
combinations, including conflicting numbers. They do not introduce unseen
subject/verb pairings.

The neural arms instantiate observer-like self-reading patches: bounded local
activity is published at synaptic ports, recurrent readback repairs the joint
state, witnessed labels enter a distinct nudged phase, and endpoint contrasts
update persistent synapses. The public receipt records the states' equation
errors and the evidence used for the comparison. There is no input-copy readout,
external lookup table or autograd gradient in the local arm.

## Reproduce

From the repository root, in the existing development environment:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/local_equilibrium/run.py
python3 benchmarks/local_equilibrium/verify.py --self-test
```

The protocol fixes every seed, split, update count and acceptance criterion
before training. It runs a small CPU experiment and stores only a JSON receipt
and small final parameter checkpoints. It never adapts its budget to held-out
scores. A failed phase or gradient audit remains in the receipt.

The independent verifier uses only NumPy and saved graph/parameter arrays. It imports
neither the producer nor Cadence, reconstructs the data and fixed-point predictions,
checks hashes and the acceptance decision, and rejects deliberate false-green changes
to scores, hashes, seed coverage and acceptance. A zero verification exit code means
the recorded outcome was reproduced; consult `predeclared_acceptance_met` separately.
It also checks sampled final gradients by finite differences. It does not replay all
training updates. Pass `--receipt /path/to/receipt.json` to verify a separate compatible
experiment without changing the original receipt.

## Interpretation

This is a one-moment finite-grammar diagnostic on 24 training and eight test
sentences, with a fixed hand-specified architecture as an experimental control.
It tests whether actual local equilibrium contrasts learn a nonlinear judgment;
it does not establish language-model scale, useful temporal credit, discovered
word/role representations, evolved wiring, or superiority to backpropagation.

The source neurons have no incoming synapses and remain unnudged. Once their
fixed stimulus is settled, their activation contributes a fixed external field
to the hidden/output subsystem. Its symmetric reciprocal row mass is capped at
1.5. The smooth activation is `tanh(v/2)`, with Lipschitz constant 0.5; the free
subsystem therefore has contraction rate at most 0.875 at step size 0.5. For the
softmax nudge, its additional activation Lipschitz constant is at most
`abs(beta)/(2*T) = 0.075`, giving rate at most 0.89375 for either sign. Runtime
checks additionally require every full potential residual to be at most
`1e-10`. These are real-arithmetic bounds, not a formal rounding-error analysis.

The effective weight is 0.5 times the efficacy. The exact comparator applies
the same positive efficacy preconditioner and clipping/projection as the local
arm. Its loss convention is `T * cross_entropy`, matching the library nudge.
Finite differences separately verify source weights, reciprocal physical
weights and biases. Reciprocal directions share one physical parameter; they
are not counted twice. No self-synapses are present.

The asymptotic gradient identity and measured residuals alone do not establish
learning success. Only the retained, fixed-protocol held-out results can support
that bounded empirical claim. Timings describe this implementation on this
machine; they are not a hardware energy measurement.
