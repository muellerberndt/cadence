"""One explicit leak=1 operating-range gene at the frozen quarter learning rates.

This develops a new neuron gene from birth, preserving the local contrast law,
historical reciprocal graph, biases, masks and curriculum. It does not claim
that changing leak repairs an already failed brain, or choose library defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rate_development as rate

PRODUCERS = (*rate.PRODUCERS, "smooth_development.py")
QUARTER_FACTORY = rate.make_brain


def make_brain():
    model = QUARTER_FACTORY()
    graph = model.brain
    model.learner.brain = type(graph)(
        graph.connectome, graph.neuron_model.replace(leak=1.0),
        efficacy=graph.efficacy, bias=graph.bias, log_gain=graph.log_gain,
        backend=graph.backend, dense_limit=graph.dense_limit,
        layout=graph.layout, precision=graph.precision,
    )
    if np.any(model.brain.bias):
        raise AssertionError("smooth gene unexpectedly changes initial biases")
    return model


def prepare(root, reference, canonical, negative):
    if root.exists():
        raise FileExistsError(root)
    parent = root.parent if root.parent.exists() else Path.cwd()
    if shutil.disk_usage(parent).free <= 20 * 1024**3:
        raise RuntimeError("require >20 GiB free before source admission")
    prior = json.loads((reference / "protocol.json").read_text())
    previous = json.loads((reference / "summary.json").read_text())
    if previous.get("passed") is not True or previous.get("protocol_sha256") != rate.harness.sha256(
        reference / "protocol.json"
    ):
        raise ValueError("reference must be the passed, hash-bound quarter-rate development")
    if prior["relations_protocol_sha256"] != rate.relations.protocol_hash():
        raise ValueError("smooth gene changed the task")
    panels = {
        "tiny2": rate.relations.load_panel("train", families=rate.relations.TWO_FAMILIES),
        "old4": rate.relations.load_panel("train", families=rate.relations.OLD_FAMILIES),
        "mixed": rate.relations.load_panel("train"),
    }
    orders = {name: rate.online.curriculum(panel, rate.CAPS[name] + 1, stage)
              for stage, (name, panel) in enumerate(panels.items())}
    for name, key in (("tiny2", "tiny_order"), ("old4", "old_order"), ("mixed", "mixed_order")):
        if orders[name][:-1] != prior[key]:
            raise ValueError("smooth gene changed the frozen quarter-rate curriculum")
    model = make_brain()
    if model.learner.config.to_dict() != prior["learner_config"]:
        raise ValueError("smooth gene changed the fixed quarter learning configuration")
    baseline_model = prior["neuron_model"]
    selected_model = model.brain.neuron_model.to_dict()
    changed = {name for name in baseline_model if baseline_model[name] != selected_model[name]}
    if changed != {"leak"}:
        raise ValueError("smooth gene changed more than neuron leak")
    root.mkdir(parents=True, exist_ok=False)
    controls = {}
    for name, folder in (("canonical", canonical), ("quarter-rate-development", reference),
                         ("quarter-rate-confirmation", negative)):
        target = root / "controls" / name
        target.mkdir(parents=True)
        controls[name] = {"original_directory": str(folder), "artifacts": {}}
        for filename in ("protocol.json", "summary.json"):
            shutil.copyfile(folder / filename, target / filename)
            controls[name]["artifacts"][filename] = rate.harness.sha256(target / filename)
        if name == "quarter-rate-confirmation":
            custody = folder / "custody/seed-6.json"
            shutil.copyfile(custody, target / "seed-6-custody.json")
            controls[name]["artifacts"]["seed-6-custody.json"] = rate.harness.sha256(custody)
            controls[name]["scope"] = (
                "campaign snapshot; completed seed6 failure; others may be pending"
            )
    inherited = root / "controls/preserved-prior-controls"
    shutil.copytree(reference / "controls", inherited)
    controls["preserved-prior-controls"] = {
        "original_directory": str(reference / "controls"),
        "artifacts": {str(path.relative_to(inherited)): rate.harness.sha256(path)
                      for path in sorted(inherited.rglob("*")) if path.is_file()},
    }
    protocol = dict(prior)
    protocol.update(
        schema="cadence-online-smooth-quarter-rate-development-v1",
        candidate_gene={"step_scale": 0.25, "eta": 0.05, "eta_bias": 0.005, "leak": 1.0},
        neuron_model=selected_model, initial_model_identity=rate.identity(model),
        reference_protocol_sha256=rate.harness.sha256(reference / "protocol.json"),
        reference_summary_sha256=rate.harness.sha256(reference / "summary.json"),
        preserved_controls=controls, library_sources=rate.harness.sources(),
        producing_sources={name: rate.harness.sha256(Path(__file__).with_name(name))
                           for name in PRODUCERS},
        runtime_precision={"numpy": np.__version__, "dtype": "float64",
                           "eps": float(np.finfo(np.float64).eps)},
        scope="new neuron leak gene from birth at fixed quarter rates; same reciprocal graph, "
              "local contrast, zero biases, full plastic masks, normalize=0 and momentum=0; "
              "no memory, reward, classifier, default or efficiency claim",
        rationale="TRAIN-only newborn intervention substantially increases cue separation and "
                  "local contrast. On failed seed6 endpoint leak1 does not restore the missing "
                  "cue; its target was already positive. This tests acquisition from birth, "
                  "not a proved explanation or repair of the old endpoint.",
    )
    rate.harness.write_json(root / "protocol.json", protocol)
    rate.development.freeze_sources(root)
    for name in PRODUCERS:
        shutil.copyfile(Path(__file__).with_name(name), root / "source" / name)
    rate.check_sources(protocol, root)
    rate.harness.write_json(root / "summary.json", {
        "protocol_sha256": rate.harness.sha256(root / "protocol.json"),
        "status": "prepared", "passed": False, "heldout_read": False,
    })


def worker(root):
    protocol = json.loads((root / "protocol.json").read_text())
    rate.check_sources(protocol, root)
    if make_brain().brain.neuron_model.to_dict() != protocol["neuron_model"]:
        raise ValueError("worker neuron model differs from the explicit smooth gene")
    # The frozen generic worker charges the same stages and continuation. Its
    # constructor hook is the sole explicitly declared new neuron gene above.
    rate.make_brain = make_brain
    rate.worker(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--canonical-control", type=Path)
    parser.add_argument("--negative-confirmation", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker:
        worker(root)
        return
    controls = (args.reference, args.canonical_control, args.negative_confirmation)
    if any(path is None for path in controls):
        parser.error("reference, canonical-control and negative-confirmation are required")
    prepare(root, args.reference.resolve(), args.canonical_control.resolve(),
            args.negative_confirmation.resolve())
    if args.prepare_only:
        return
    env = dict(os.environ)
    env.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
               "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    env.update(PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(root / "source" / Path(__file__).name),
               "--out", str(root), "--worker"]
    with (root / "worker.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if code:
            summary = json.loads((root / "summary.json").read_text())
            summary.update(status="process_failed", passed=False, process_exit_code=code)
            rate.harness.write_json(root / "summary.json", summary)
            raise RuntimeError("smooth worker failed; worker.log and attempt preserved")
    print((root / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
