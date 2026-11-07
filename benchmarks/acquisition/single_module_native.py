"""One frozen native2->4 control using the existing one-module compose API.

Only modules=(32,) replaces (32,16). The first sensory projection is retained
byte for byte, but the remaining physical graph and plastic parameter count
change. This is an architectural control, not a matched capacity/speed claim.
No centering, calibration, memory writes or independent/heldout reads occur.
Preparation constructs and freezes states; execution requires separate review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import native_online as native
import numpy as np

from cadence import Brain, LearnerConfig

SCHEMA = "cadence-native-single-module-control-v1"
ORIGINAL_FACTORY = native.harness.make_brain
CONSTRUCTION = {
    "inputs": 650,
    "actions": 36,
    "modules": [32],
    "observers": [],
    "lateral": 0.0,
}
BOUNDS = {
    "teaching_seconds": 300,
    "worker_seconds": 360,
    "admission_mib": 72,
    "output_cap_mib": 80,
}
FIXED = {
    "schema": SCHEMA,
    "seed": 0,
    "stage_counts": [2, 4],
    "model_construction": CONSTRUCTION,
    "lesson_caps": {"2": 512, "4": 1024},
    "positions": {"2": [0, 9], "4": [0, 9, 17, 22]},
    "query_every": {"2": 32, "4": 32},
    "selected_correct": {"2": 2, "4": 4},
    "independent_correct": {},
    **BOUNDS,
}


def make_brain(recipe, seed, args):
    if recipe != "qualified" or seed != 0:
        raise ValueError("single-module control admits only the fixed qualified founder0")
    effective = native.harness.recipe_configuration(recipe, args)
    if (
        effective["gene"] != "lateral0-small-rms"
        or effective["lateral"] != 0
        or effective["resting_bias"]
        or effective["sensory_bias"]
        or effective["freeze_motor_lateral"]
    ):
        raise ValueError("single-module control changed the original raw teacher gene")
    return Brain.compose(
        inputs=650,
        actions=36,
        modules=(32,),
        observers=(),
        seed=seed,
        learning=LearnerConfig(**effective["learning"]),
        lateral=0.0, recall=False,
    )


def first_projection(model):
    """Canonical local endpoints bind the identical 650->32 newborn projection."""
    wire = model.connectome
    first = np.asarray(wire.populations.get("module_0", wire.populations["association"]))
    sensory = np.asarray(model.sensory_index)
    selected = np.isin(wire.pre, sensory) & np.isin(wire.post, first)
    pre = np.full(wire.n, -1, dtype=np.int64)
    post = pre.copy()
    pre[sensory] = np.arange(len(sensory))
    post[first] = np.arange(len(first))
    arrays = {
        "local_pre": pre[wire.pre[selected]],
        "local_post": post[wire.post[selected]],
        "sign": wire.sign[selected],
        "count": wire.count[selected],
        "efficacy": model.brain.efficacy[selected],
    }
    if len(first) != 32 or len(sensory) != 650 or selected.sum() != 650 * 32:
        raise ValueError("single-module first projection is not the declared dense650->32")
    return arrays


def construction_proof(candidate, original):
    a, b = first_projection(candidate), first_projection(original)
    if a.keys() != b.keys() or any(not np.array_equal(a[k], b[k]) for k in a):
        raise ValueError("single-module changed the original newborn sensory projection")
    if candidate.connectome.n != original.connectome.n or any(
        not np.array_equal(getattr(candidate, name), getattr(original, name))
        for name in ("sensory_index", "motor_index")
    ):
        raise ValueError("single-module changed the original external ports/neuron count")
    held = len(candidate.sensory_index) + len(candidate.connectome.populations["prefrontal"])
    old_held = len(original.sensory_index) + len(original.connectome.populations["prefrontal"])
    return {
        "first_projection_arrays_sha256": {
            key: native.harness.row_hash(value) for key, value in a.items()
        },
        "first_projection_byte_identical": True,
        "external_ports_byte_identical": True,
        "changed_roles": (
            "original module_0 becomes the32-node association cortex; original16-node association "
            "range becomes held prefrontal state in this cold microscope; prefrontal/trace width "
            "grows16->32, memory remains unused; free dimension84->68 with total neurons750"
        ),
        "candidate": {
            "neurons": candidate.connectome.n,
            "free_neurons": candidate.connectome.n - held,
            "synapses": candidate.connectome.synapses,
            "plastic_synapses": int(np.count_nonzero(candidate.learner.plastic_synapses)),
            "plastic_neurons": int(np.count_nonzero(candidate.learner.plastic_neurons)),
        },
        "original": {
            "neurons": original.connectome.n,
            "free_neurons": original.connectome.n - old_held,
            "synapses": original.connectome.synapses,
            "plastic_synapses": int(np.count_nonzero(original.learner.plastic_synapses)),
            "plastic_neurons": int(np.count_nonzero(original.learner.plastic_neurons)),
        },
        "interpretation": (
            "same first projection and local teacher genes; removes the intermediate module "
            "and changes association/motor/prefrontal wiring and capacity; "
            "no matched efficiency claim"
        ),
    }


def check_protocol(protocol):
    if any(protocol.get(name) != value for name, value in FIXED.items()):
        raise ValueError("single-module control changed its architecture/gene/gate/bound")
    for key in ("2", "4"):
        order = protocol["orders_including_next_lesson"][key]
        if len(order) != protocol["lesson_caps"][key] + 1 or any(
            type(row) is not int or row not in protocol["positions"][key] for row in order
        ):
            raise ValueError("single-module fixed teacher order differs")


def load_fixture(folder):
    """Decode the selected TRAIN school only; lazy independent/heldout arrays stay sealed."""
    provenance = json.loads((folder / "provenance.json").read_text())
    with np.load(folder / "school.npz", allow_pickle=False) as archive:
        x, y = archive["school_inputs"], archive["school_labels"]
    identities = provenance["panels"]["school"]
    if (
        x.shape != (24, 650)
        or x.dtype != np.dtype("float64")
        or y.shape != (24,)
        or not np.issubdtype(y.dtype, np.integer)
        or not np.isfinite(x).all()
        or np.any(y < 0)
        or np.any(y >= 36)
        or len(identities) != 24
    ):
        raise ValueError("single-module TRAIN school census differs")
    for row, label, identity in zip(x, y, identities, strict=True):
        if (
            native.harness.row_hash(row) != identity["input_sha256"]
            or int(label) != identity["label"]
        ):
            raise ValueError("single-module TRAIN row identity differs")
    return {"school_inputs": x, "school_labels": y}


def check_sources(root, protocol, *, worker_runtime=True):
    check_protocol(protocol)
    for folder, pins in (
        ("source/library/cadence", protocol["library_sources"]),
        ("source", protocol["producing_sources"]),
        ("source/fixture", protocol["fixture"]),
        ("control", protocol["control_capsule"]),
        ("prior-centered", protocol["centered_negative_capsule"]),
        ("reference", protocol["reference_capsule"]),
    ):
        for name, expected in pins.items():
            if native.harness.sha256(root / folder / name) != expected:
                raise ValueError("single-module frozen capsule changed: " + folder + "/" + name)
    for name, expected in protocol["producing_sources"].items():
        imported = sys.modules.get(Path(name).stem)
        path = Path(imported.__file__) if imported else Path(__file__).with_name(name)
        if native.harness.sha256(path) != expected:
            raise ValueError("single-module imported producer changed: " + name)
    if native.harness.sources() != protocol["library_sources"]:
        raise ValueError("single-module imported library differs")
    if native.runtime_precision() != protocol["runtime_precision"]:
        raise ValueError("single-module runtime differs from frozen source")
    if worker_runtime and any(os.environ.get(k) != v for k, v in protocol["threads"].items()):
        raise ValueError("single-module worker thread settings differ")
    control = json.loads((root / "control/protocol.json").read_text())
    control_summary = json.loads((root / "control/summary.json").read_text())
    if (
        control["seed"] != 0
        or control["library_sources"] != protocol["library_sources"]
        or control["learner_config"] != protocol["learner_config"]
        or control["neuron_model"] != protocol["neuron_model"]
        or control["numpy"] != protocol["runtime_precision"]["numpy"]
        or control["dtype"] != protocol["runtime_precision"]["dtype"]
        or control_summary["protocol_sha256"] != protocol["control_capsule"]["protocol.json"]
        or control_summary["heldout_read"]
    ):
        raise ValueError("single-module changed original raw source/genes/runtime")
    for key in ("2", "4"):
        if any(
            control[field][key] != protocol[field][key]
            for field in ("positions", "orders_including_next_lesson", "lesson_caps", "query_every")
        ):
            raise ValueError("single-module changed original teacher exposure")
    stages = control_summary["stages"]
    if [r["examples"] for r in stages] != [2, 4] or [
        (r["passed"], r["recall"][-1]["lesson"], r["recall"][-1]["correct"]) for r in stages
    ] != [(True, 512, 2), (False, 1024, 1)]:
        raise ValueError("single-module raw negative census differs")
    original = ORIGINAL_FACTORY("qualified", 0, native.ARGS)
    prior = json.loads((root / "reference/protocol.json").read_text())
    native.check_reference(
        prior,
        json.loads((root / "reference/summary.json").read_text()),
        protocol["reference_capsule"]["protocol.json"],
        original,
    )
    candidate = make_brain("qualified", 0, native.ARGS)
    if (
        native.model_identity(candidate) != protocol["initial_model_identity"]
        or construction_proof(candidate, original) != protocol["construction_proof"]
        or candidate.learner.config.to_dict() != original.learner.config.to_dict()
        or candidate.brain.neuron_model.to_dict() != original.brain.neuron_model.to_dict()
    ):
        raise ValueError("single-module initial physical graph/genes changed")
    for count in (2, 4):
        receipt = json.loads((root / f"control/receipt-{count}.json").read_text())
        old = Brain.load(root / f"control/initial-{count}.npz")
        if (
            receipt["initial_sha256"] != protocol["control_capsule"][f"initial-{count}.npz"]
            or receipt["final_sha256"] != protocol["control_capsule"][f"final-{count}.npz"]
            or native.model_identity(old) != native.model_identity(original)
        ):
            raise ValueError("single-module negative newborn/endpoint custody differs")
    negative = json.loads((root / "prior-centered/summary.json").read_text())
    if (
        negative["protocol_sha256"] != protocol["centered_negative_capsule"]["protocol.json"]
        or negative["passed"]
        or negative["independent_read"]
        or negative["heldout_read"]
        or negative["stages"][0]["recall"][-1]["correct"] != 1
        or negative["stages"][0]["recall"][-1]["lesson"] != 512
        or negative["stages"][1]["status"] != "not_run_prior_failure"
    ):
        raise ValueError("single-module centered negative census changed")
    job = json.loads((root / "job-protocol.json").read_text())
    if (
        job["protocol_sha256"] != native.harness.sha256(root / "protocol.json")
        or job["interpreter_sha256"]
        != hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
        or job["bounds"] != BOUNDS
        or native.harness.sha256(root / "prepared-founder.npz")
        != protocol["prepared_founder_sha256"]
        or native.model_identity(Brain.load(root / "prepared-founder.npz"))
        != protocol["initial_model_identity"]
    ):
        raise ValueError("single-module reviewed job/model changed")


def prepare(root, fixture, reference, control, centered):
    if root.exists():
        raise FileExistsError(root)
    original = ORIGINAL_FACTORY("qualified", 0, native.ARGS)
    prior = json.loads((reference / "protocol.json").read_text())
    control_protocol = json.loads((control / "protocol.json").read_text())
    native.check_reference(
        prior,
        json.loads((reference / "summary.json").read_text()),
        native.harness.sha256(reference / "protocol.json"),
        original,
    )
    load_fixture(fixture)
    candidate = make_brain("qualified", 0, native.ARGS)
    proof = construction_proof(candidate, original)
    root.mkdir(parents=True, exist_ok=False)
    native.harness.freeze_sources(root / "source", fixture)
    producers = (*native.PRODUCERS, "single_module_native.py")
    for name in producers:
        shutil.copyfile(Path(__file__).with_name(name), root / "source" / name)
    capsules = {
        "reference": {
            name + ".json": reference / (name + ".json") for name in ("protocol", "summary")
        },
        "control": {name + ".json": control / (name + ".json") for name in ("protocol", "summary")},
        "prior-centered": {
            name + ".json": centered / (name + ".json")
            for name in ("protocol", "job-protocol", "summary")
        },
    }
    for count in (2, 4):
        for name in ("initial.npz", "final.npz", "receipt.json"):
            stem, suffix = name.split(".")
            capsules["control"][f"{stem}-{count}.{suffix}"] = control / f"selected-{count}" / name
    for name, expected in control_protocol["producing_sources"].items():
        source = control / "source" / name
        if native.harness.sha256(source) != expected:
            raise ValueError("historical raw-control producing source changed: " + name)
        capsules["control"]["source/" + name] = source
    for name, paths in capsules.items():
        (root / name).mkdir()
        for filename, source in paths.items():
            (root / name / filename).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, root / name / filename)
    founder = candidate.save(root / "prepared-founder.npz")
    protocol = {
        **FIXED,
        "learner_config": candidate.learner.config.to_dict(),
        "neuron_model": candidate.brain.neuron_model.to_dict(),
        "initial_model_identity": native.model_identity(candidate),
        "construction_proof": proof,
        "prepared_founder_sha256": native.harness.sha256(founder),
        "library_sources": native.harness.sources(),
        "producing_sources": {
            name: native.harness.sha256(root / "source" / name) for name in producers
        },
        "fixture": {
            name: native.harness.sha256(root / "source/fixture" / name)
            for name in ("school.npz", "provenance.json")
        },
        "reference_capsule": {
            name: native.harness.sha256(root / "reference" / name) for name in capsules["reference"]
        },
        "control_capsule": {
            name: native.harness.sha256(root / "control" / name) for name in capsules["control"]
        },
        "centered_negative_capsule": {
            name: native.harness.sha256(root / "prior-centered" / name)
            for name in capsules["prior-centered"]
        },
        "runtime_precision": native.runtime_precision(),
        "threads": {name: "1" for name in native.THREAD_VARIABLES},
        "orders_including_next_lesson": {
            key: control_protocol["orders_including_next_lesson"][key] for key in ("2", "4")
        },
        "selected_gate": (
            "perfect selected2/4, positive newborn gain, zero teaching/query refusals; "
            "full next-lesson checkpoint continuation; fresh founder at each stage"
        ),
        "heldout": "sealed;24/independent/heldout not admitted",
        "fixture_decode": "TRAIN school only; independent and heldout arrays stay lazy and sealed",
        "scope": (
            "single fixed architecture control of existing raw local equilibrium teacher; "
            "selected2 then fresh selected4; no default change, whole-issue closure, "
            "native generalization/retention or continuing-animal claim"
        ),
        "hypothesis": (
            "Removing the intermediate processing projection may restore separable acquisition "
            "under the existing common local rule; no offset/regulator and no width search"
        ),
        "failure_policy": (
            "one founder/order/cap; preserve failure and skip later stages; no retry/tuning"
        ),
    }
    native.harness.write_json(root / "protocol.json", protocol)
    native.harness.write_json(
        root / "job-protocol.json",
        {
            "schema": "cadence-native-single-module-job-v1",
            "status": "prepared_not_executed",
            "protocol_sha256": native.harness.sha256(root / "protocol.json"),
            "interpreter": sys.executable,
            "interpreter_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "bounds": BOUNDS,
            "teacher_calls": 0,
            "query_rows": 0,
            "scope": protocol["scope"],
        },
    )
    protocol = json.loads((root / "protocol.json").read_text())
    check_sources(root, protocol, worker_runtime=False)
    native.harness.write_json(
        root / "admission.json",
        {
            "protocol_sha256": native.harness.sha256(root / "protocol.json"),
            "source_fixture_reference_model_preflight": True,
            "status": "prepared_not_executed",
            "teacher_calls": 0,
            "query_rows": 0,
        },
    )
    return protocol


def worker(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check_sources(root, protocol)
    with (root / "execution-started.json").open("x") as stream:
        stream.write(native.harness.sha256(root / "protocol.json") + "\n")
    native.harness.make_brain = make_brain
    native.check_sources = check_sources
    native.load_fixture = load_fixture
    result = native.worker(root)
    result.update(
        architecture_control_passed=result["passed"],
        native_24_gate_passed=False,
        whole_issue_passed=False,
        scope=protocol["scope"],
    )
    native.finalize_summary(root, result, protocol)
    return result


def launch(root):
    protocol = json.loads((root / "protocol.json").read_text())
    check_sources(root, protocol, worker_runtime=False)
    if (root / "execution-started.json").exists():
        raise ValueError("single-module frozen control already attempted")
    env = dict(os.environ, PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    env.update(protocol["threads"])
    with (root / "worker.log").open("w") as stream:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "source/single_module_native.py"),
                    "--worker",
                    "--out",
                    str(root),
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=protocol["worker_seconds"],
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            native.process_failure_summary(
                root, protocol, {"kind": "worker_timeout", "seconds": error.timeout}
            )
            raise
    if completed.returncode:
        native.process_failure_summary(
            root, protocol, {"kind": "worker_exit", "exit_code": completed.returncode}
        )
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    result = json.loads((root / "summary.json").read_text())
    if not result.get("complete"):
        raise ValueError("single-module worker exited without complete census")
    result["process_exit_code"] = completed.returncode
    native.finalize_summary(root, result, protocol)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    for name in ("reference", "fixture", "control", "centered"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker:
        worker(root)
    elif args.execute_reviewed:
        print(json.dumps(launch(root), indent=2))
    elif args.prepare_only and all(
        getattr(args, key) for key in ("reference", "fixture", "control", "centered")
    ):
        prepare(
            root,
            *(
                getattr(args, key).resolve()
                for key in ("fixture", "reference", "control", "centered")
            ),
        )
        print(
            json.dumps(
                {
                    "prepared": str(root),
                    "protocol_sha256": native.harness.sha256(root / "protocol.json"),
                    "teacher_calls": 0,
                    "query_rows": 0,
                }
            )
        )
    else:
        parser.error(
            "prepare-only needs reference/fixture/control/centered; execution awaits review"
        )


if __name__ == "__main__":
    main()
