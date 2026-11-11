"""One declared local-RMS development gene on the passed zero-lateral graph.

Only teacher eta, eta_bias and normalize change. RMS uses each synapse's own
contrast history; it is an adaptive update metric, not a protection guarantee.
No momentum, extra neurons, masks, operating-point shifts or heldout are added.
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

import lateral_development as lateral
import numpy as np
import rate_development as rate

PRODUCERS = (*lateral.PRODUCERS, "normalized_development.py")
BASE_FACTORY = lateral.make_brain


def make_brain():
    model = BASE_FACTORY()
    model.learner.config = replace(
        model.learner.config, eta=0.003, eta_bias=0.0003, normalize=0.99, momentum=0.0
    )
    return model


def assert_only_optimizer_changed(model, baseline):
    before, after = baseline.learner.config.to_dict(), model.learner.config.to_dict()
    if {name for name in before if before[name] != after[name]} != {"eta", "eta_bias", "normalize"}:
        raise ValueError("normalized gene changed more than its two rates and RMS setting")
    if (after["eta"], after["eta_bias"], after["normalize"], after["momentum"]) != (
        0.003, 0.0003, 0.99, 0.0
    ):
        raise ValueError("normalized gene does not match its declared optimizer")
    if rate.identity(model) != rate.identity(baseline):
        raise ValueError("normalized gene changed initial arrays or masks")
    np.testing.assert_array_equal(model.connectome.sign, baseline.connectome.sign)
    if model.brain.neuron_model.to_dict() != baseline.brain.neuron_model.to_dict():
        raise ValueError("normalized gene changed the neuron model")
    for name in ("backend", "dense_limit", "precision"):
        if getattr(model.brain, name) != getattr(baseline.brain, name):
            raise ValueError("normalized gene changed graph runtime: " + name)
    if model.brain.layout.to_dict() != baseline.brain.layout.to_dict():
        raise ValueError("normalized gene changed graph layout")
    if (np.any(model.brain.bias) or model.hippocampus.writes
            or np.any(model.hippocampus.consolidated)):
        raise ValueError("normalized gene unexpectedly altered initial biases or memory")


def copy_control(root, name, folder, filenames):
    target = root / "controls" / name
    target.mkdir(parents=True)
    control = {"original_directory": str(folder), "artifacts": {}}
    for filename in filenames:
        destination = target / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / filename, destination)
        control["artifacts"][filename] = rate.harness.sha256(destination)
    return control


def prepare(root, reference, canonical, negative, diagnostic):
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
        raise ValueError("reference must be the passed, hash-bound zero-lateral development")
    failure = json.loads((negative / "summary.json").read_text())
    if failure["recipe_sha256"] != rate.harness.sha256(negative / "protocol.json") or (
        failure["founder_denominator"] != 5 or failure["all_founders_passed"] is not False
        or any(row["status"] in ("scheduled", "running") for row in failure["outcomes"])
    ):
        raise ValueError("preserve the completed failed five-founder zero-lateral confirmation")
    failed_founder = next(row for row in failure["outcomes"] if row["seed"] == 19)
    if failed_founder["passed"] is not False or failed_founder["heldout_read"] is not False:
        raise ValueError("founder19 TRAIN failure and sealed heldout control differ")
    diagnosis = json.loads((diagnostic / "summary.json").read_text())
    if diagnosis["protocol_sha256"] != rate.harness.sha256(diagnostic / "protocol.json") or (
        diagnosis["heldout_read"] or diagnosis["development_read"]
        or diagnosis["work"]["learning_calls"] != 0
    ):
        raise ValueError("diagnostic must preserve its bound TRAIN-only no-learning result")
    if prior["relations_protocol_sha256"] != rate.relations.protocol_hash():
        raise ValueError("normalized gene changed the task")
    panels = {
        "tiny2": rate.relations.load_panel("train", families=rate.relations.TWO_FAMILIES),
        "old4": rate.relations.load_panel("train", families=rate.relations.OLD_FAMILIES),
        "mixed": rate.relations.load_panel("train"),
    }
    orders = {name: rate.online.curriculum(panel, rate.CAPS[name] + 1, stage)
              for stage, (name, panel) in enumerate(panels.items())}
    for name, key in (("tiny2", "tiny_order"), ("old4", "old_order"), ("mixed", "mixed_order")):
        if orders[name][:-1] != prior[key]:
            raise ValueError("normalized gene changed the frozen zero-lateral curriculum")
    baseline, model = BASE_FACTORY(), make_brain()
    assert_only_optimizer_changed(model, baseline)
    if baseline.learner.config.to_dict() != prior["learner_config"] or (
        rate.identity(baseline) != prior["initial_model_identity"]
        or baseline.brain.neuron_model.to_dict() != prior["neuron_model"]
        or prior["model_construction"]["lateral"] != 0.0
    ):
        raise ValueError("current baseline differs from the passed source-bound zero-lateral model")
    root.mkdir(parents=True, exist_ok=False)
    controls = {
        "canonical": copy_control(root, "canonical", canonical, ("protocol.json", "summary.json")),
        "lateral-development": copy_control(
            root, "lateral-development", reference, ("protocol.json", "summary.json")
        ),
        "lateral-confirmation": copy_control(
            root, "lateral-confirmation", negative,
            ("protocol.json", "summary.json", "custody/seed-19.json"),
        ),
        "founder19-train-diagnostic": copy_control(
            root, "founder19-train-diagnostic", diagnostic,
            ("protocol.json", "summary.json", "representation-comparison.json",
             "source-custody.json"),
        ),
    }
    inherited = root / "controls/preserved-prior-controls"
    shutil.copytree(reference / "controls", inherited)
    controls["preserved-prior-controls"] = {
        "original_directory": str(reference / "controls"),
        "artifacts": {str(path.relative_to(inherited)): rate.harness.sha256(path)
                      for path in sorted(inherited.rglob("*")) if path.is_file()},
    }
    protocol = dict(prior)
    protocol.update(
        schema="cadence-online-normalized-zero-lateral-development-v1",
        candidate_gene={"eta": 0.003, "eta_bias": 0.0003, "normalize": 0.99,
                        "momentum": 0.0, "lateral": 0.0, "leak": 0.1},
        learner_config=model.learner.config.to_dict(),
        initial_model_identity=rate.identity(model),
        reference_protocol_sha256=rate.harness.sha256(reference / "protocol.json"),
        reference_summary_sha256=rate.harness.sha256(reference / "summary.json"),
        preserved_controls=controls, library_sources=rate.harness.sources(),
        producing_sources={name: rate.harness.sha256(Path(__file__).with_name(name))
                           for name in PRODUCERS},
        runtime_precision={"numpy": np.__version__, "dtype": "float64",
                           "eps": float(np.finfo(np.float64).eps)},
        scope="one local-RMS teacher optimizer gene with separately declared smaller rates; "
              "same zero-lateral reciprocal graph, local contrast, zero biases, full masks, "
              "leak=.1 and momentum=0; no added capacity, memory, reward, classifier, "
              "library default, autonomous protection or efficiency claim",
        rationale="founder19 lost responsive old motor relations as association-motor weights "
                  "and shared representations changed. The conflicting cues remain distinct "
                  "above measured nuisance noise. RMS tests each contact's own adaptive step "
                  "history with appropriately coupled small rates; it is not a claim that "
                  "RMS protects old skills or that a configured rate bounds every increment.",
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
    assert_only_optimizer_changed(model, BASE_FACTORY())
    if model.brain.neuron_model.to_dict() != protocol["neuron_model"]:
        raise ValueError("worker neuron model differs from the unchanged zero-lateral model")
    rate.make_brain = make_brain
    rate.worker(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--canonical-control", type=Path)
    parser.add_argument("--negative-confirmation", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker:
        worker(root)
        return
    controls = (args.reference, args.canonical_control, args.negative_confirmation, args.diagnostic)
    if any(path is None for path in controls):
        parser.error("reference, canonical-control, negative-confirmation and diagnostic required")
    prepare(root, *(path.resolve() for path in controls))
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
            raise RuntimeError("normalized worker failed; worker.log and attempt preserved")
    print((root / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
