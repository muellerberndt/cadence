"""Replay the failed admission and check the existing numerical damping fallback.

This is a diagnostic replay of the same recipe, not another candidate or seed
search. A local linearization at a REFUSED state is not an equilibrium stability
certificate. Solver counters and host time are not physical energy or clocks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import admission
import numpy as np

import cadence as cd
from cadence.receipts import Receipt


def run(protocol_path: Path) -> Receipt:
    protocol = json.loads(protocol_path.read_text())
    began = time.monotonic()
    r = admission.relevance
    results = []
    for seed in protocol["development_seeds"]:
        brain = cd.Brain.compose(r.INPUTS, 2, seed=seed, **protocol["brain"])
        graph = brain.brain
        brain.learner.brain = cd.NeuralGraph(
            graph.connectome, graph.neuron_model.replace(dt=protocol["dt"]),
            efficacy=graph.efficacy, bias=graph.bias, log_gain=graph.log_gain)
        rng = np.random.default_rng(np.random.SeedSequence([31014, seed]))
        order = np.stack([rng.permutation(len(r.CASES)) for _ in range(protocol["rounds"])])
        trained = r.train(brain, order, 1, exposed=True, frozen=False,
                          deadline=time.monotonic() + protocol["training_seconds_per_arm"])
        if trained["stop"] != "refused_query":
            raise RuntimeError("the original refused-query boundary was not reproduced")
        rows = r.CASES[order[trained["rounds"]]]
        _, query, _ = r.inputs(rows[:, :3], rows[:, 3], 1, exposed=True)
        drive, graph = brain.stimulus(query), brain.brain
        tolerance = brain.learner.config.tolerance
        budget = protocol["brain"]["learning_free_steps"]
        phase = graph.equilibrate(drive, state=brain._activity(), budget=budget,
                                  tolerance=tolerance)
        weight = np.zeros((graph.connectome.n, graph.connectome.n))
        np.add.at(weight, (graph.connectome.post, graph.connectome.pre), graph.weights)
        rates, radii = [], []
        for v in np.atleast_2d(phase.state.v):
            jacobian = (weight * graph.neuron_model.slope_at(v)[None, :]
                        - np.eye(len(v)))
            eigenvalues = np.linalg.eigvals(jacobian)
            rates.append(float(eigenvalues.real.min()))
            radii.append(float(np.abs(1 + graph.neuron_model.dt * eigenvalues).max()))
        damped = graph.equilibrate(drive, state=brain._activity(), budget=budget,
                                   tolerance=tolerance, damping=3)
        results.append({
            "seed": seed, "training_replay": trained,
            "tolerance": tolerance, "undamped_residual": phase.residual.tolist(),
            "undamped_steps": phase.steps, "undamped_checks": phase.residual_checks,
            "attempted_state_minimum_linearized_rate": min(rates),
            "attempted_state_maximum_euler_spectral_radius": max(radii),
            "damped_residual": damped.residual.tolist(), "damped_steps": damped.steps,
            "damped_checks": damped.residual_checks, "halvings": damped.damping_halvings,
            "damped_qualified": bool(damped.qualified.all()),
        })
    root = Path(__file__).resolve().parents[2]
    package = Path(cd.__file__).resolve().parent
    paths = [Path(__file__).resolve(), Path(admission.__file__), protocol_path.resolve(),
             Path(r.__file__)]
    sources = [(str(p.relative_to(root)), p) for p in paths]
    sources += [("src/cadence/" + str(p.relative_to(package)), p)
                for p in sorted(package.rglob("*.py"))]
    return Receipt.build("cadence-attention-admission-damping-readback/1", {
        "protocol": protocol, "founders": results,
        "numpy_version": np.__version__, "cadence_version": cd.__version__,
        "elapsed_host_seconds": time.monotonic() - began,
        "all_damped_boundaries_qualified": all(r["damped_qualified"] for r in results),
        "interpretation": (
            "A successful numerical retry still requires the task acquisition gate. "
            "Residuals above tolerance cannot be published as answers; this replay "
            "does not localize a retrieval failure."),
    }, sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; retain the earlier diagnostic")
    run(args.protocol).write(args.out)
    print(args.out)
