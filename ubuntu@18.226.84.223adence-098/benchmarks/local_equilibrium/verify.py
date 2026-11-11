"""Independent NumPy replay of a local-equilibrium grammar receipt.

This verifier imports neither cadence nor the producer. It rebuilds the
declared data, solves tanh fixed points from checkpoint arrays, reconstructs
scores and the acceptance decision, and checks source/checkpoint custody.
Successful verification does not mean the experimental acceptance gate passed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(protocol):
    cases = np.asarray(list(itertools.product(range(4), range(4), range(2))))
    features = np.concatenate(
        [np.eye(4)[cases[:, 0]], np.eye(4)[cases[:, 1]], np.eye(2)[cases[:, 2]]],
        axis=1,
    )
    labels = (cases[:, 0] % 2 == cases[:, 2]).astype(int)
    pairs = {tuple(pair) for pair in protocol["heldout_subject_distractor_pairs"]}
    test = np.asarray([tuple(case[:2]) in pairs for case in cases])
    require(test.sum() == 8 and (~test).sum() == 24, "split sizes changed")
    require(
        (cases[test, 0] % 2 == cases[test, 1] % 2).sum() == 4,
        "held-out distractors must include matching and conflicting numbers",
    )
    for mask in (test, ~test):
        require(np.all(features[mask].sum(axis=0) > 0), "a token is absent from a partition")
        require(labels[mask].sum() * 2 == mask.sum(), "class imbalance")
    return cases, features, labels, test


def softmax(values, temperature):
    logits = values / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def settle(weights, bias, features, *, beta=0.0, labels=None, temperature=0.2):
    """Direct fixed-point iteration, independent of the library stepper."""
    drive = np.zeros((len(features), len(bias)))
    drive[:, :10] = 2 * features
    potential = drive.copy()
    for iteration in range(4096):
        activity = np.tanh(potential / 2)
        target = activity @ weights + drive + bias
        if beta:
            nudge = np.eye(2)[labels] - softmax(activity[:, -2:], temperature)
            target[:, -2:] += beta * nudge
        residual = float(np.max(np.abs(target - potential)))
        if residual <= 1e-12:
            return activity, residual, iteration
        potential = target
    raise AssertionError(f"independent fixed point did not converge: {residual}")


def score(activity, labels, temperature):
    probabilities = softmax(activity, temperature)
    predictions = probabilities.argmax(axis=1)
    return {
        "cross_entropy": float(-np.log(probabilities[np.arange(len(labels)), labels]).mean()),
        "accuracy": float((predictions == labels).mean()),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


def check_score(actual, reported, where):
    require(actual["predictions"] == reported["predictions"], f"predictions differ: {where}")
    require(actual["accuracy"] == reported["accuracy"], f"accuracy differs: {where}")
    require(
        abs(actual["cross_entropy"] - reported["cross_entropy"]) < 1e-8,
        f"cross-entropy differs: {where}",
    )
    require(
        np.max(np.abs(np.asarray(actual["probabilities"]) - reported["probabilities"])) < 1e-8,
        f"probabilities differ: {where}",
    )


def final_gradient_check(weights, bias, features, labels, protocol):
    """Compare actual endpoint contrasts with fresh finite differences."""
    temperature, gain = protocol["temperature"], protocol["neuron_model"]["gain"]
    beta = protocol["beta"] / 2
    plus = settle(weights, bias, features, beta=beta, labels=labels, temperature=temperature)[0]
    minus = settle(weights, bias, features, beta=-beta, labels=labels, temperature=temperature)[0]
    samples = [
        ("source", 0, 10),
        ("reciprocal", 10, len(bias) - 2),
        ("bias", len(bias) - 1, len(bias) - 1),
    ]
    results = []
    for family, pre, post in samples:
        losses = []
        for direction in (-1, 1):
            altered, altered_bias = weights.copy(), bias.copy()
            if family == "bias":
                altered_bias[post] += direction * 1e-5
            else:
                altered[pre, post] += direction * gain * 1e-5
                if family == "reciprocal":
                    altered[post, pre] += direction * gain * 1e-5
            activity = settle(altered, altered_bias, features)[0]
            losses.append(
                temperature * score(activity[:, -2:], labels, temperature)["cross_entropy"]
            )
        finite_difference = (losses[1] - losses[0]) / 2e-5
        if family == "bias":
            local_direction = (plus[:, post] - minus[:, post]).mean() / (2 * beta)
        else:
            local_direction = (
                gain
                * (plus[:, pre] * plus[:, post] - minus[:, pre] * minus[:, post]).mean()
                / (2 * beta)
            )
        results.append(
            {
                "family": family,
                "pre": pre,
                "post": post,
                "finite_difference": float(finite_difference),
                "negative_local_direction": float(-local_direction),
                "absolute_error": float(abs(finite_difference + local_direction)),
            }
        )
    return results


def verify(receipt, directory, *, audit_gradients=True):
    protocol = receipt["protocol"]
    require(receipt.get("complete") is True, "receipt is incomplete")
    for name, expected in receipt["source_sha256"].items():
        require(digest(ROOT / name) == expected, f"source hash differs: {name}")
    cases, features, labels, test = dataset(protocol)
    require(
        receipt["dataset"]
        == {
            "triples": cases.tolist(),
            "labels": labels.tolist(),
            "heldout": test.tolist(),
        },
        "receipt dataset differs from independently constructed task",
    )
    require(
        protocol["neuron_model"]["slope"] == 1.0
        and protocol["neuron_model"]["threshold"] == 0.0
        and protocol["neuron_model"]["leak"] == 1.0,
        "this independent verifier requires the declared tanh(v/2) activation",
    )
    expected_seeds = protocol["seeds"]
    require(
        [row["seed"] for row in receipt["seeds"]] == expected_seeds,
        "missing, reordered or additional seeds",
    )
    # Each subject/distractor row occurs with both verbs and opposite labels.
    # Consequently the zero affine classifier has exactly zero gradient here.
    affine = np.column_stack([features[~test], np.ones((~test).sum())])
    require(
        np.array_equal(affine.T @ (0.5 - labels[~test]), np.zeros(11)),
        "zero affine baseline is not stationary for the training split",
    )
    uniform = np.zeros((len(features), 2))
    for name, mask in (("train", ~test), ("heldout", test)):
        check_score(
            score(uniform[mask], labels[mask], protocol["temperature"]),
            receipt["linear"]["final"][name],
            f"linear/{name}",
        )
    checked, support, worst_residual = [], True, 0.0
    for seed in receipt["seeds"]:
        arms = seed["arms"]
        require(
            [arm["arm"] for arm in arms] == protocol["arms"][:3],
            f"missing arms for seed {seed['seed']}",
        )
        audit = seed["gradient_audit"]
        require(
            max(row["absolute_error"] for row in audit["coordinates"]) <= 1e-7,
            "producer finite-difference gate failed",
        )
        small_beta = audit["centered_comparison"][-1]
        require(
            small_beta["efficacy_max_error"] <= 1e-4 and small_beta["bias_max_error"] <= 1e-4,
            "producer centered-gradient gate failed",
        )
        accuracies = {}
        for arm in arms:
            where = f"seed {seed['seed']}/{arm['arm']}"
            require(arm["status"] == "complete", f"incomplete arm: {where}")
            expected_updates = 0 if arm["arm"] == "frozen_no_nudge" else protocol["updates"]
            require(arm["updates"] == expected_updates, f"update budget differs: {where}")
            require(
                arm["work"]["maximum_residual"] <= protocol["solve"]["tolerance"],
                f"recorded convergence failure: {where}",
            )
            checkpoint = directory / arm["checkpoint"]["path"]
            require(
                digest(checkpoint) == arm["checkpoint"]["sha256"],
                f"checkpoint hash differs: {where}",
            )
            with np.load(checkpoint, allow_pickle=False) as data:
                pre, post, count = data["pre"], data["post"], data["count"]
                bias, efficacy = data["bias"], data["efficacy"]
                weights = np.zeros((len(bias), len(bias)))
                np.add.at(weights, (pre, post), protocol["neuron_model"]["gain"] * count * efficacy)
            require(np.isfinite(weights).all() and np.isfinite(bias).all(), "nonfinite checkpoint")
            require(
                not weights[:, :10].any() and not bias[:10].any(), "input sources are not fixed"
            )
            require(
                np.allclose(weights[10:, 10:], weights[10:, 10:].T, atol=1e-14, rtol=0),
                "reciprocal weights differ",
            )
            mass = float(np.abs(weights[10:, 10:]).sum(axis=0).max())
            require(
                mass <= protocol["free_reciprocal_row_mass_cap"] + 1e-12,
                "free-subsystem contraction budget exceeded",
            )
            activity, residual, iterations = settle(weights, bias, features)
            worst_residual = max(worst_residual, residual)
            actual = {}
            for name, mask in (("train", ~test), ("heldout", test)):
                actual[name] = score(activity[mask, -2:], labels[mask], protocol["temperature"])
                check_score(actual[name], arm["final"][name], f"{where}/{name}")
            accuracies[arm["arm"]] = actual["heldout"]["accuracy"]
            row = {
                "seed": seed["seed"],
                "arm": arm["arm"],
                "train_accuracy": actual["train"]["accuracy"],
                "heldout_accuracy": actual["heldout"]["accuracy"],
                "heldout_cross_entropy": actual["heldout"]["cross_entropy"],
                "maximum_equation_residual": residual,
                "iterations": iterations,
                "free_row_mass": mass,
            }
            if audit_gradients and arm["arm"] == "local_centered":
                row["independent_final_gradient_check"] = final_gradient_check(
                    weights,
                    bias,
                    features[~test],
                    labels[~test],
                    protocol,
                )
            checked.append(row)
        support &= accuracies["local_centered"] >= 0.875
        support &= accuracies["exact_implicit"] >= 0.875
        support &= accuracies["local_centered"] > accuracies["frozen_no_nudge"]
        support &= accuracies["local_centered"] > 0.5
    require(
        bool(support) == receipt["predeclared_acceptance_met"],
        "reported acceptance differs from independent reconstruction",
    )
    return {
        "schema": "cadence.local-equilibrium-independent-verification/v1",
        "integrity_verified": True,
        "predeclared_acceptance_met": bool(support),
        "maximum_independent_residual": worst_residual,
        "checked": checked,
        "scope": (
            "Final checkpoints and metrics replayed independently; "
            "training trajectories not replayed."
        ),
    }


def mutation_checks(receipt, directory):
    changes = [
        (
            "acceptance",
            lambda x: x.update(predeclared_acceptance_met=not x["predeclared_acceptance_met"]),
        ),
        (
            "prediction",
            lambda x: x["seeds"][0]["arms"][0]["final"]["heldout"]["predictions"].__setitem__(0, 7),
        ),
        (
            "checkpoint hash",
            lambda x: x["seeds"][0]["arms"][0]["checkpoint"].update(sha256="0" * 64),
        ),
        (
            "source hash",
            lambda x: x["source_sha256"].update({next(iter(x["source_sha256"])): "0" * 64}),
        ),
        ("seed omission", lambda x: x["seeds"].pop()),
    ]
    rejected = []
    for name, mutate in changes:
        changed = copy.deepcopy(receipt)
        mutate(changed)
        try:
            verify(changed, directory, audit_gradients=False)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError(f"mutation was not rejected: {name}")
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    result = verify(receipt, args.receipt.parent)
    result["receipt_sha256"] = digest(args.receipt)
    result["verifier_sha256"] = digest(Path(__file__))
    if args.self_test:
        result["rejected_mutations"] = mutation_checks(receipt, args.receipt.parent)
    output = args.out or args.receipt.with_name("verification.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checked"}, indent=2))


if __name__ == "__main__":
    main()
