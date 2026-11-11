# Longer-training follow-up of the local equilibrium comparison

This is an adaptive follow-up to `../local_equilibrium/`, whose original
protocol, source, receipt and checkpoints remain unchanged. Its initial
800-update experiment retained substantial training error and did not meet its
predeclared robust generalization criterion. This follow-up changes only the
fixed training budget to 8000 updates and uses fresh seeds 5 through 9. The
architecture, task, inputs, split, hyperparameters and acceptance criterion
remain the same. Held-out scores are computed only after training finishes.

Run from the Cadence repository root:

```sh
PYTHONPATH=src .venv/bin/python benchmarks/local_equilibrium_longer/run.py
PYTHONPATH=src .venv/bin/python benchmarks/local_equilibrium/verify.py \
  --receipt benchmarks/local_equilibrium_longer/receipt.json
```

The [original package](../local_equilibrium/README.md) defines the observer-like
patch structure, exact local learning rule, mathematical conditions and limited
task interpretation. The follow-up protocol records the original producer and
protocol hashes and the reason for the extension. This is an explicitly adaptive
comparison on the same task, not a newly independent task or a replacement for
the first experiment's outcome. Every scheduled seed, including failures,
remains in the receipt. No further tuning is permitted by this protocol.
