"""Frozen small grammar comparison of local EP and implicit equilibrium gradients.

Run from the repository with PYTHONPATH=src .venv/bin/python
benchmarks/local_equilibrium/run.py. No autograd is imported or used.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset():
    triples = np.array(list(itertools.product(range(4), range(4), range(2))))
    x = np.zeros((len(triples), 10))
    rows = np.arange(len(x))
    x[rows, triples[:, 0]] = 1
    x[rows, 4 + triples[:, 1]] = 1
    x[rows, 8 + triples[:, 2]] = 1
    y = (triples[:, 0] % 2 == triples[:, 2]).astype(int)
    selected = {tuple(pair) for pair in P["heldout_subject_distractor_pairs"]}
    held = np.array([tuple(row[:2]) in selected for row in triples])
    assert held.sum() == 8 and (~held).sum() == 24
    for mask in (held, ~held):
        assert np.bincount(y[mask]).tolist() == [int(mask.sum() // 2)] * 2
        assert (x[mask].sum(axis=0) > 0).all()
    assert (triples[held, 0] % 2 == triples[held, 1] % 2).sum() == 4
    return triples, x, y, held


def build(seed):
    graph = cd.layered(10, P["hidden"], 2, density=1, seed=seed)
    model = cd.NeuronModel(**P["neuron_model"])
    rng = np.random.default_rng(seed)
    weights = np.zeros(graph.synapses)
    source = graph.pre < 10
    weights[source] = rng.normal(0, P["initial_effective_source_std"], source.sum())
    lookup = {
        (int(a), int(b)): i for i, (a, b) in enumerate(zip(graph.pre, graph.post, strict=True))
    }
    for i in np.flatnonzero(~source):
        pre, post = int(graph.pre[i]), int(graph.post[i])
        if pre < post:
            weights[i] = weights[lookup[post, pre]] = rng.normal(
                0, P["initial_effective_reciprocal_std"]
            )
    bias = np.zeros(graph.n)
    bias[list(graph.populations["hidden"])] = rng.normal(
        0, P["initial_hidden_bias_std"], P["hidden"]
    )
    cfg = cd.LearnerConfig(
        beta=P["beta"],
        eta=P["eta"],
        eta_bias=P["eta_bias"],
        temperature=P["temperature"],
        scale_cap=P["efficacy_cap"],
        tolerance=None,
    )
    learner = cd.Learner(
        cd.Brain(graph, model, efficacy=weights / model.gain, bias=bias),
        graph.populations["output"],
        cfg,
        plastic_neurons=np.arange(graph.n) >= 10,
    )
    project(learner)
    assert cd.ep_structure(learner.brain, fixed_inputs=range(10)).compatible
    return learner


def project(learner):
    graph = learner.brain.connectome
    free = graph.pre >= 10
    mass = float(
        np.bincount(
            graph.post[free], weights=np.abs(learner.brain.weights[free]), minlength=graph.n
        ).max()
    )
    if mass > P["free_reciprocal_row_mass_cap"]:
        efficacy = learner.brain.efficacy.copy()
        efficacy[free] *= P["free_reciprocal_row_mass_cap"] / mass
        learner.brain = learner.brain.with_parameters(efficacy=efficacy)
    return min(mass, P["free_reciprocal_row_mass_cap"])


def drives(learner, features):
    drive = np.zeros((len(features), learner.brain.connectome.n))
    drive[:, :10] = 2 * features
    return drive


def solve(brain, drive, ledger, state=None, nudge=None):
    start = time.perf_counter()
    result = brain.equilibrate(drive, state=state, nudge=nudge, **P["solve"])
    ledger["solve_seconds"] += time.perf_counter() - start
    ledger["solve_calls"] += 1
    ledger["settling_steps"] += result.steps
    ledger["row_steps"] += result.steps * len(drive)
    worst = float(result.residual.max())
    ledger["maximum_residual"] = max(ledger["maximum_residual"], worst)
    if not np.all(result.converged):
        raise ArithmeticError(f"unconverged phase: residual={worst}, steps={result.steps}")
    return result.state


def ledger():
    return dict(
        solve_seconds=0.0, solve_calls=0, settling_steps=0, row_steps=0, maximum_residual=0.0
    )


def log_probabilities(activity):
    z = activity / P["temperature"]
    z -= z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def metrics(activity, target):
    lp = log_probabilities(activity)
    prediction = lp.argmax(axis=1)
    return dict(
        cross_entropy=float(-lp[np.arange(len(target)), target].mean()),
        accuracy=float((prediction == target).mean()),
        predictions=prediction.tolist(),
        probabilities=np.exp(lp).tolist(),
    )


def evaluate(learner, x, y, held, book):
    state = solve(learner.brain, drives(learner, x), book)
    out = state.activation[:, learner.output_index]
    result = {"train": metrics(out[~held], y[~held]), "heldout": metrics(out[held], y[held])}
    triples = dataset()[0][held]
    words = P["vocabulary"]
    result["heldout"]["wrong_sentences"] = [
        dict(
            sentence=(
                f"the {words['subjects'][s]} near the {words['distractors'][d]} {words['verbs'][v]}"
            ),
            grammatical=bool(target),
            predicted_grammatical=bool(predicted),
        )
        for (s, d, v), target, predicted in zip(
            triples, y[held], result["heldout"]["predictions"], strict=True
        )
        if target != predicted
    ]
    return result


def implicit_gradient(learner, state, target):
    """Gradient of T*CE, wrt one physical efficacy and each trainable bias."""
    brain, graph = learner.brain, learner.brain.connectome
    s, v = state.activation, state.v
    free = np.arange(10, graph.n)
    derivative = brain.neuron_model.slope_at(v[:, free])
    cost_s = np.zeros_like(s)
    probability = np.exp(log_probabilities(s[:, learner.output_index]))
    probability[np.arange(len(target)), target] -= 1
    cost_s[:, learner.output_index] = probability
    # v column = W_ff.T @ act(v) + external field.
    w = brain.dense()[np.ix_(free, free)]
    jacobian = np.eye(len(free))[None] - w.T[None] * derivative[:, None, :]
    rhs = derivative * cost_s[:, free]
    adjoint = np.zeros_like(s)
    adjoint[:, free] = np.linalg.solve(np.swapaxes(jacobian, 1, 2), rhs[..., None])[..., 0]
    factor = brain.neuron_model.gain * graph.count * np.exp(brain.log_gain[graph.pre])
    grad = factor * (s[:, graph.pre] * adjoint[:, graph.post]).mean(axis=0)
    paired = learner.reverse >= 0
    grad[paired] += grad[learner.reverse[paired]].copy()
    return grad, adjoint.mean(axis=0), factor


def detuned(learner, drive, target, free, book, beta):
    full = learner.targets(target)
    plus = solve(learner.brain, drive, book, state=free, nudge=learner.nudge_for(full, beta))
    minus = solve(learner.brain, drive, book, state=free, nudge=learner.nudge_for(full, -beta))
    for state in (plus, minus):
        if np.max(np.abs(state.activation[:, :10] - free.activation[:, :10])) > 1e-9:
            raise ArithmeticError("input source changed between phases")
    return plus, minus


def gradient_audit(learner, x, y):
    book = ledger()
    drive = drives(learner, x)
    free = solve(learner.brain, drive, book)
    exact_e, exact_b, factor = implicit_gradient(learner, free, y)
    graph, brain = learner.brain.connectome, learner.brain
    sources = np.flatnonzero(graph.pre < 10)
    pairs = np.flatnonzero((graph.pre >= 10) & (graph.pre < graph.post))
    coordinates = []
    for family, indices in (
        ("source", sources),
        ("reciprocal", pairs),
        ("bias", np.arange(10, graph.n)),
    ):
        for coordinate in indices[[0, len(indices) // 2, -1]]:
            values = []
            for sign in (-1, 1):
                if family == "bias":
                    bias = brain.bias.copy()
                    bias[coordinate] += sign * 1e-5
                    candidate = brain.with_parameters(bias=bias)
                else:
                    efficacy = brain.efficacy.copy()
                    efficacy[coordinate] += sign * 1e-5
                    reverse = learner.reverse[coordinate]
                    if reverse >= 0:
                        efficacy[reverse] += sign * 1e-5
                    candidate = brain.with_parameters(efficacy=efficacy)
                state = solve(candidate, drive, book)
                ce = metrics(state.activation[:, learner.output_index], y)["cross_entropy"]
                values.append(P["temperature"] * ce)
            finite = (values[1] - values[0]) / 2e-5
            exact = float(exact_b[coordinate] if family == "bias" else exact_e[coordinate])
            coordinates.append(
                dict(
                    family=family,
                    coordinate=int(coordinate),
                    finite_difference=finite,
                    implicit=exact,
                    absolute_error=abs(finite - exact),
                )
            )
    comparison = []
    for beta in (P["beta"], P["beta"] / 2):
        plus, minus = detuned(learner, drive, y, free, book, beta)
        # Public contrast divides by configured beta; rescale for this diagnostic only.
        local_e, local_b = learner.contrast(free, plus, minus)
        local_e *= P["beta"] / beta
        local_b *= P["beta"] / beta
        comparison.append(
            dict(
                beta=beta,
                efficacy_max_error=float(np.max(np.abs(factor * local_e + exact_e))),
                bias_max_error=float(np.max(np.abs(local_b[10:] + exact_b[10:]))),
            )
        )
    maximum = max(row["absolute_error"] for row in coordinates)
    if (
        maximum > 1e-7
        or comparison[-1]["efficacy_max_error"] > 1e-4
        or comparison[-1]["bias_max_error"] > 1e-4
    ):
        raise ArithmeticError(f"gradient audit failed: finite difference {maximum}; {comparison}")
    return dict(coordinates=coordinates, centered_comparison=comparison, work=book)


def linear(x, y, held):
    start = time.perf_counter()
    a = np.column_stack([x, np.ones(len(x))])
    weights = np.zeros((a.shape[1], 2))
    for _ in range(P["updates"]):
        logits = a[~held] @ weights
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        prob[np.arange(sum(~held)), y[~held]] -= 1
        weights -= a[~held].T @ prob / sum(~held)
    # metrics expects activations before division by the declared temperature.
    prediction = (a @ weights) * P["temperature"]
    return dict(
        status="complete",
        arm="linear_logistic",
        independent_parameters=weights.size,
        updates=P["updates"],
        wall_seconds=time.perf_counter() - start,
        final={
            "train": metrics(prediction[~held], y[~held]),
            "heldout": metrics(prediction[held], y[held]),
        },
    )


def main():
    triples, x, y, held = dataset()
    root = HERE.parents[1]
    source_paths = [
        HERE / "run.py",
        HERE / "protocol.json",
        *sorted((root / "src/cadence").rglob("*.py")),
    ]
    receipt = dict(
        schema="cadence.local-equilibrium-grammar/v1",
        protocol=P,
        source_sha256={str(path.relative_to(root)): sha(path) for path in source_paths},
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            machine=platform.machine(),
            platform=platform.platform(),
        ),
        dataset=dict(triples=triples.tolist(), labels=y.tolist(), heldout=held.tolist()),
        linear=linear(x, y, held),
        seeds=[],
    )
    destination = HERE / "receipt.json"

    def save():
        destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

    save()
    for seed in P["seeds"]:
        result = dict(seed=seed, arms=[])
        receipt["seeds"].append(result)
        try:
            result["gradient_audit"] = gradient_audit(build(seed), x[~held], y[~held])
        except Exception:
            result["gradient_audit_failure"] = traceback.format_exc()
            save()
            continue
        for arm in P["arms"][:3]:
            start = time.perf_counter()
            learner, book = build(seed), ledger()
            record = dict(
                arm=arm,
                work=book,
                updates=0,
                independent_parameters=learner.parameters(),
                status="running",
                maximum_free_row_mass=project(learner),
                free_contraction_bound=0.875,
                detuned_contraction_bound=0.89375,
            )
            result["arms"].append(record)
            try:
                record["initial"] = evaluate(learner, x, y, held, book)
                drive = drives(learner, x[~held])
                if arm != "frozen_no_nudge":
                    for update in range(P["updates"]):
                        free = solve(learner.brain, drive, book)
                        if arm == "local_centered":
                            plus, minus = detuned(learner, drive, y[~held], free, book, P["beta"])
                            learner.update(free, plus, minus)
                        else:
                            grad_e, grad_b, factor = implicit_gradient(learner, free, y[~held])
                            learner.apply(-P["eta"] * grad_e / factor, -P["eta_bias"] * grad_b)
                        record["maximum_free_row_mass"] = max(
                            record["maximum_free_row_mass"], project(learner)
                        )
                        record["updates"] = update + 1
                record["final"] = evaluate(learner, x, y, held, book)
                checkpoint = HERE / f"seed-{seed}-{arm}.npz"
                np.savez_compressed(
                    checkpoint,
                    efficacy=learner.brain.efficacy,
                    bias=learner.brain.bias,
                    pre=learner.brain.connectome.pre,
                    post=learner.brain.connectome.post,
                    count=learner.brain.connectome.count,
                )
                record["checkpoint"] = dict(path=checkpoint.name, sha256=sha(checkpoint))
                record["status"] = "complete"
            except Exception:
                record["status"], record["error"] = "failed", traceback.format_exc()
            record["wall_seconds"] = time.perf_counter() - start
            save()
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        arm=arm,
                        status=record["status"],
                        updates=record["updates"],
                        heldout=record.get("final", {}).get("heldout", {}).get("accuracy"),
                    )
                ),
                flush=True,
            )
    receipt["complete"] = all(
        len(s["arms"]) == 3 and all(a["status"] == "complete" for a in s["arms"])
        for s in receipt["seeds"]
    )
    supported = receipt["complete"]
    for seed in receipt["seeds"]:
        arms = {a["arm"]: a for a in seed["arms"]}
        if len(arms) != 3:
            supported = False
            continue
        local = arms["local_centered"].get("final", {}).get("heldout", {}).get("accuracy", 0)
        exact = arms["exact_implicit"].get("final", {}).get("heldout", {}).get("accuracy", 0)
        frozen = arms["frozen_no_nudge"].get("final", {}).get("heldout", {}).get("accuracy", 1)
        supported &= local >= 0.875 and exact >= 0.875 and local > frozen
        supported &= local > receipt["linear"]["final"]["heldout"]["accuracy"]
    receipt["predeclared_acceptance_met"] = bool(supported)
    save()


if __name__ == "__main__":
    main()
