"""Independent saved-state arithmetic; zero new settling/learning/body operations."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import body
import lesion as L
import probe as P

from cadence import Brain


def arithmetic(graph, inputs, state, weights, biases, alpha):
    predictions, errors = [0.0] * len(state), [0.0] * len(state)
    for j in graph.residual_order:
        terms = [biases[j]]
        for edge in graph.incoming[j]:
            kind, source, _ = graph.edges[edge]
            value = (
                inputs[source]
                if kind == "input"
                else state[source]
                if kind == "state"
                else errors[source]
            )
            terms.append(weights[edge] * value)
        predictions[j] = math.tanh(math.fsum(terms))
        errors[j] = state[j] - predictions[j]
    gradient = [alpha * x for x in state]
    adjoint = list(errors)
    for j in reversed(graph.residual_order):
        gradient[j] += adjoint[j]
        slope = -adjoint[j] * (1 - predictions[j] ** 2)
        for edge in graph.incoming[j]:
            kind, source, _ = graph.edges[edge]
            if kind == "state":
                gradient[source] += slope * weights[edge]
            elif kind == "residual":
                adjoint[source] += slope * weights[edge]
    energy = 0.5 * math.fsum(e * e for e in errors) + 0.5 * alpha * math.fsum(
        x * x for x in state
    )
    return predictions, errors, gradient, energy


def analyze(root):
    protocol = L.bound(root)
    result = json.loads((root / "result.json").read_text())
    assert result["complete"] and result["admissions"] == 0
    parent = Path(protocol["parent"])
    tape = json.loads((parent / "data.json").read_text())
    artifact_pins = {
        str(p): P.sha(p)
        for p in root.rglob("*")
        if p.is_file() and p.name != "saved-state-verification.json"
    }
    summaries, checked = {}, 0
    max_gradient = 0.0
    for item in result["variants"]:
        directory = root / item["variant"]
        original = (directory / "before-intervention.json").read_text()
        modified = (directory / "after-intervention.json").read_text()
        brain = Brain.from_snapshot(original)
        assert L.intervention(brain, item["variant"]) == [
            {**r, "edge": tuple(r["edge"])} for r in item["removed"]
        ]
        assert brain.snapshot() == modified
        info = brain.inspect()
        ranges = {pop["name"]: pop["indices"] for pop in info["populations"]}
        outputs = {
            o["name"]: ranges[o["reads"]][o["indices"][0]] for o in info["outputs"]
        }
        expected = [
            (name, mode, r)
            for name in L.DESIGN["sets"]
            for mode in L.DESIGN["modes"]
            for r in tape[name]
        ]
        with gzip.open(directory / "queries.jsonl.gz", "rt") as stream:
            journal = [json.loads(line) for line in stream]
        assert len(journal) == len(expected) == 256
        work, rows = Counter(), {}
        for n, (saved, (name, mode, row)) in enumerate(
            zip(journal, expected, strict=True), 1
        ):
            inputs = body.inputs(row)
            targets = body.facts(row) if mode == "corrected" else None
            assert saved["ordinal"] == n and saved["id"] == row["id"]
            assert saved["set"] == name and saved["mode"] == mode
            assert saved["arguments_sha256"] == P.digest([inputs, targets])
            assert saved["before"] == P.digest(modified)
            r = saved["result"]
            assert r["qualified"] is True and r["parameter_sha256"] == P.digest(
                [brain.weights, brain.biases]
            )
            flat = tuple(
                x
                for port in ("coarse", "command", "prior", "fine", "present")
                for x in inputs[port]
            )
            pred, err, grad, energy = arithmetic(
                brain.graph,
                flat,
                r["state"],
                brain.weights,
                brain.biases,
                brain.config["state_prior"],
            )
            assert (
                pred == r["predictions"]
                and err == r["errors"]
                and energy == r["energy"]
            )
            clamps = {outputs[k]: v[0] for k, v in (targets or {}).items()}
            assert all(r["state"][i] == x for i, x in clamps.items())
            bound = brain.config["state_bound"]
            components = [
                0.0 if (x <= -bound and g > 0) or (x >= bound and g < 0) else abs(g)
                for i, (x, g) in enumerate(zip(r["state"], grad, strict=True))
                if i not in clamps
            ]
            stationarity = max(components)
            assert (
                stationarity == r["stationarity"]
                and stationarity <= brain.config["tolerance"]
            )
            max_gradient = max(max_gradient, stationarity)
            assert r["outputs"] == {o: [r["state"][i]] for o, i in outputs.items()}
            f = outputs["f"]
            raw, represented = [], []
            for edge in brain.graph.incoming[f]:
                kind, source, _ = brain.graph.edges[edge]
                if kind == "input":
                    raw.append(brain.weights[edge] * flat[source])
                else:
                    assert kind == "state" and source in ranges["nonlinear_head"]
                    represented.append(brain.weights[edge] * r["state"][source])
            assert math.isclose(
                pred[f],
                math.tanh(brain.biases[f] + math.fsum(raw + represented)),
                rel_tol=0,
                abs_tol=2e-16,
            )
            rows[name, mode, row["id"]] = {
                "error": abs(r["outputs"]["f"][0] - row["actual"][3]),
                "raw_drive": math.fsum(raw),
                "head_drive": math.fsum(represented),
            }
            work.update(r["work"])
            checked += 1
        assert dict(work) == item["work"]
        groups = {}
        for name in L.DESIGN["sets"]:
            for mode in L.DESIGN["modes"]:
                selected = [rows[name, mode, r["id"]] for r in tape[name]]
                mae = sum(r["error"] for r in selected) / 64
                assert mae == item["results"][f"{name}/{mode}"]["mae"]
                groups[f"{name}/{mode}"] = {
                    "mae": mae,
                    "mean_abs_raw_drive": sum(abs(r["raw_drive"]) for r in selected)
                    / 64,
                    "mean_abs_head_drive": sum(abs(r["head_drive"]) for r in selected)
                    / 64,
                }
            for r in tape[name]:
                assert (
                    rows[name, "routine", r["id"]]["raw_drive"]
                    == rows[name, "corrected", r["id"]]["raw_drive"]
                )
            groups[name + "/clamp_change"] = {
                "mean_abs_head_drive_change": sum(
                    abs(
                        rows[name, "corrected", r["id"]]["head_drive"]
                        - rows[name, "routine", r["id"]]["head_drive"]
                    )
                    for r in tape[name]
                )
                / 64
            }
        summaries[item["variant"]] = groups
    assert checked == 768 and all(P.sha(path) == h for path, h in artifact_pins.items())
    L.bound(root)
    return {
        "valid": True,
        "saved_states_checked": checked,
        "maximum_projected_gradient": max_gradient,
        "new_settlements": 0,
        "admissions": 0,
        "body_operations": 0,
        "scope": "Independent scalar prediction/error/energy/reverse-gradient and output/clamp checks at every saved final state. This verifies qualification, not numerical trajectories or wall time. Direct versus head drive magnitudes are descriptive, not causal importance percentages.",
        "groups": summaries,
        "artifact_pins": artifact_pins,
        "verifier_sha256": P.sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "saved-state-verification.json"
    if out.exists():
        raise ValueError("Preserve prior verification")
    P.write(out, analyze(root))
    print(P.sha(out))


if __name__ == "__main__":
    main()
