"""One explicit zero-lateral motor gene at frozen quarter learning rates.

The candidate removes only the motor-to-motor synapses through Brain.compose.
It preserves reciprocal association feedback, the common local contrast law,
neuron leak, biases, masks, and the already frozen balanced curriculum.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rate_development as rate

PRODUCERS = (*rate.PRODUCERS, "lateral_development.py")
QUARTER_FACTORY = rate.make_brain


def make_brain():
    model = rate.development.make_brain(
        "canonical", rate.SEED, SimpleNamespace(phase_steps=4096, tolerance=0.003),
        lateral=0.0,
    )
    model.learner.config = replace(model.learner.config, eta=0.05, eta_bias=0.005)
    return model


def assert_only_lateral_changed(model, baseline):
    """Admit the exact graph deletion, preserving every remaining parameter."""
    wire, prior = model.connectome, baseline.connectome
    removed = np.isin(prior.pre, baseline.motor_index) & np.isin(prior.post, baseline.motor_index)
    if int(removed.sum()) != 36 * 35:
        raise ValueError("baseline does not contain the historical motor lateral circuit")
    for name in ("pre", "post", "sign", "count"):
        np.testing.assert_array_equal(getattr(wire, name), getattr(prior, name)[~removed])
    np.testing.assert_array_equal(model.brain.efficacy, baseline.brain.efficacy[~removed])
    for name in ("bias", "log_gain"):
        np.testing.assert_array_equal(getattr(model.brain, name), getattr(baseline.brain, name))
    for name in ("sensory_index", "motor_index"):
        np.testing.assert_array_equal(getattr(model, name), getattr(baseline, name))
    np.testing.assert_array_equal(model.learner.plastic_synapses,
                                  baseline.learner.plastic_synapses[~removed])
    np.testing.assert_array_equal(model.learner.plastic_neurons, baseline.learner.plastic_neurons)
    if wire.populations.keys() != prior.populations.keys():
        raise ValueError("lateral gene changed population declarations")
    for name in wire.populations:
        np.testing.assert_array_equal(wire.populations[name], prior.populations[name])
    for name in ("backend", "dense_limit", "precision"):
        if getattr(model.brain, name) != getattr(baseline.brain, name):
            raise ValueError("lateral gene changed graph runtime: " + name)
    if model.learner.config.to_dict() != baseline.learner.config.to_dict():
        raise ValueError("lateral gene changed the quarter-rate learner configuration")
    if model.brain.neuron_model.to_dict() != baseline.brain.neuron_model.to_dict():
        raise ValueError("lateral gene changed the neuron model")
    if (np.any(model.brain.bias) or model.hippocampus.writes
            or np.any(model.hippocampus.consolidated)):
        raise ValueError("lateral gene unexpectedly altered initial biases or memory")


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
    failed = json.loads((negative / "summary.json").read_text())
    if failed.get("passed") is not False or failed.get("protocol_sha256") != rate.harness.sha256(
        negative / "protocol.json"
    ):
        raise ValueError("negative smooth development must preserve its failed source-bound result")
    if prior["relations_protocol_sha256"] != rate.relations.protocol_hash():
        raise ValueError("lateral gene changed the task")
    panels = {
        "tiny2": rate.relations.load_panel("train", families=rate.relations.TWO_FAMILIES),
        "old4": rate.relations.load_panel("train", families=rate.relations.OLD_FAMILIES),
        "mixed": rate.relations.load_panel("train"),
    }
    orders = {name: rate.online.curriculum(panel, rate.CAPS[name] + 1, stage)
              for stage, (name, panel) in enumerate(panels.items())}
    for name, key in (("tiny2", "tiny_order"), ("old4", "old_order"), ("mixed", "mixed_order")):
        if orders[name][:-1] != prior[key]:
            raise ValueError("lateral gene changed the frozen quarter-rate curriculum")
    model = make_brain()
    assert_only_lateral_changed(model, QUARTER_FACTORY())
    if model.learner.config.to_dict() != prior["learner_config"]:
        raise ValueError("lateral gene changed the fixed quarter learning configuration")
    if model.brain.neuron_model.to_dict() != prior["neuron_model"]:
        raise ValueError("lateral gene changed the quarter-rate neuron model")
    root.mkdir(parents=True, exist_ok=False)
    controls = {}
    for name, folder in (("canonical", canonical), ("quarter-rate-development", reference),
                         ("smooth-development", negative)):
        target = root / "controls" / name
        target.mkdir(parents=True)
        controls[name] = {"original_directory": str(folder), "artifacts": {}}
        for filename in ("protocol.json", "summary.json"):
            shutil.copyfile(folder / filename, target / filename)
            controls[name]["artifacts"][filename] = rate.harness.sha256(target / filename)
    inherited = root / "controls/preserved-prior-controls"
    shutil.copytree(negative / "controls", inherited)
    controls["preserved-prior-controls"] = {
        "original_directory": str(negative / "controls"),
        "artifacts": {str(path.relative_to(inherited)): rate.harness.sha256(path)
                      for path in sorted(inherited.rglob("*")) if path.is_file()},
    }
    protocol = dict(prior)
    protocol.update(
        schema="cadence-online-zero-lateral-quarter-rate-development-v1",
        candidate_gene={"step_scale": 0.25, "eta": 0.05, "eta_bias": 0.005, "lateral": 0.0},
        model_construction={**prior["model_construction"], "lateral": 0.0},
        initial_model_identity=rate.identity(model),
        reference_protocol_sha256=rate.harness.sha256(reference / "protocol.json"),
        reference_summary_sha256=rate.harness.sha256(reference / "summary.json"),
        preserved_controls=controls, library_sources=rate.harness.sources(),
        producing_sources={name: rate.harness.sha256(Path(__file__).with_name(name))
                           for name in PRODUCERS},
        runtime_precision={"numpy": np.__version__, "dtype": "float64",
                           "eps": float(np.finfo(np.float64).eps)},
        scope="one explicitly selected zero-lateral motor gene at quarter rates and leak=.1; "
              "only motor-to-motor synapses removed; reciprocal feedback and common local "
              "contrast preserved; zero biases, full masks, normalize=0 and momentum=0; "
              "no memory, reward, classifier, default or efficiency claim",
        rationale="quarter-rate fresh founders include failed acquisition and retention. This "
                  "tests removal of broad inhibitory competition at unchanged fixed rates. "
                  "Prior lateral0-local-rms failure also changed normalization and labels, "
                  "so it does not decide this unnormalized identifiable-relations candidate.",
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
    model = make_brain()
    assert_only_lateral_changed(model, QUARTER_FACTORY())
    if protocol["model_construction"]["lateral"] != 0.0:
        raise ValueError("worker graph differs from the declared zero-lateral gene")
    if model.brain.neuron_model.to_dict() != protocol["neuron_model"]:
        raise ValueError("worker neuron model differs from the unchanged quarter-rate model")
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
            raise RuntimeError("lateral worker failed; worker.log and attempt preserved")
    print((root / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
