"""A fixed-capacity actual-outcome #85 chamber, prepared before one bounded run.

Preparation performs no solves. --launch runs only the frozen producer/library.
The precise scientific contract is FIXED_CAPACITY_PROTOCOL.md; replay is a
store-only control and isolated odors do not discharge the sequence checkbox.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import chamber_inputs as inputs
import numpy as np

import cadence
import cadence.brain as graph_module
from cadence import ActorCriticConfig, Brain, LearnerConfig, SynapticMemory
from cadence.brain import Brain as NeuralGraph
from cadence.learning import LearningPhaseError
from cadence.receipts import canonical_json
from cadence.stream import Trace

SCHEMA = "cadence-actual-outcome-retention/1"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
PRODUCERS = ("actual_outcome_chamber.py", "chamber_inputs.py", "FIXED_CAPACITY_PROTOCOL.md")
FORMAL = ("MemoryBoundary.lean", "ProtectedTemporalPath.lean", "ReadableTrace.lean",
          "TemporalOverlap.lean", "StructuralMemory.lean")
BRIDGES = ("docs/continuous.md", "docs/memory.md", "docs/temporal-memory.md",
           "tests/test_continuous.py", "tests/test_actual_outcome_association.py",
           "tests/test_feedback_transaction.py", "tests/test_generic_memory_checkpoint.py",
           "tests/test_temporal_memory.py", "tests/test_temporal_memory_metric.py")
GATES = {"variants_correct": 9, "variants_planned": 10, "prototype_correct": 1,
         "rare_max_drop": 1, "obsolete_max": 1, "cohort_gap": 0.20}
REQUIRED_EXPOSURES = (0, 128)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(canonical_json({"shape": array.shape, "dtype": array.dtype.str}).encode()
                          + array.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n")
    temporary.replace(path)


def tree_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def runtime():
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        np.__config__.show()
    return {"python": sys.version, "implementation": platform.python_implementation(),
            "executable": sys.executable, "numpy": np.__version__, "cadence": cadence.__version__,
            "platform": platform.platform(), "precision": "float64",
            "eps": float(np.finfo(np.float64).eps),
            "longdouble_eps": float(np.finfo(np.longdouble).eps),
            "blas": configuration.getvalue(),
            "cpu_fused_selected": graph_module._FUSED,
            "cadence_fused_env": os.environ.get("CADENCE_FUSED", "1"),
            "numba": getattr(sys.modules.get("numba"), "__version__", None),
            "threads": {name: os.environ.get(name) for name in THREADS}}


def make_brain(seed):
    return Brain.compose(
        8, 2, modules=(32,), lateral=-0.5, seed=seed, resting_bias=0, backend="cpu",
        learning=LearnerConfig(eta=0.5, eta_bias=0.05, beta=0.1, temperature=0.2,
                               free_steps=1024, nudged_steps=12, tolerance=0.003,
                               momentum=0.9, normalize=0, qualified=False, damping=3),
        reward=ActorCriticConfig(gamma=0.9, lam=0.8, eta=1, eta_bias=0.05,
                                 eta_critic=0.3, eligibility_steps=12, normalize=0,
                                 momentum=0, dopamine_cap=1, dopamine_center=0,
                                 dopamine_floor=0, critic_normalize=True, critic_signal="auto"),
        consolidation=0.05, episodic=True, working_memory_decay=0.2,
        working_memory_amplitude=3,
        working_memory_scale=12.0,  # the composition of releases before 0.82.0
    )


def model_identity(brain):
    graph, learner = brain.brain, brain.learner
    arrays = {"pre": brain.connectome.pre, "post": brain.connectome.post,
              "count": brain.connectome.count, "sign": brain.connectome.sign,
              "efficacy": graph.efficacy, "bias": graph.bias, "log_gain": graph.log_gain,
              "plastic_synapses": learner.plastic_synapses,
              "plastic_neurons": learner.plastic_neurons}
    return {"arrays": {name: array_hash(value) for name, value in arrays.items()},
            "populations": {name: array_hash(value)
                            for name, value in brain.connectome.populations.items()},
            "learner": learner.config.to_dict(), "actor": brain.basal_ganglia.config.to_dict(),
            "critic_target": brain.basal_ganglia.config.critic_target,
            "neuron": graph.neuron_model.to_dict(), "trace": brain.working_memory.to_dict(),
            "hippocampus": brain.hippocampus.to_dict(), "layout": graph.layout.to_dict(),
            "backend": graph.backend, "precision": graph.precision,
            "dense_limit": graph.dense_limit}


def checkpoints_equal(first, second):
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        return set(left.files) == set(right.files) and all(
            np.array_equal(left[name], right[name]) for name in left.files
        )


def memory_arrays(memory):
    return {name: getattr(memory, name).copy()
            for name in ("consolidated", "strength", "mass") } | {"writes": memory.writes}


def expected_memory(previous, key, actions, reward, *, consolidation=0.05, decay=0.9, rate=1):
    """Independent longdouble masked local law; no production helper is called."""
    cue = np.asarray(key, np.longdouble)
    scale = np.max(np.abs(cue), axis=1, keepdims=True)
    cue = cue / np.where(scale > 0, scale, 1)
    norm = np.sqrt((cue * cue).sum(axis=1, keepdims=True))
    cue = cue / np.where(norm > 0, norm, 1)
    batch = len(cue)
    old = np.asarray(previous["consolidated"], np.longdouble)
    same_batch = len(previous["strength"]) == batch
    strength = np.asarray(previous["strength"], np.longdouble) if same_batch else old[None]
    strength = np.broadcast_to(old + decay * (strength - old), (batch, *old.shape)).copy()
    mass = np.asarray(previous["mass"], np.longdouble) * decay if same_batch else np.zeros(batch)
    targets, mask = np.zeros((batch, 2), np.longdouble), np.zeros((batch, 2), bool)
    targets[np.arange(batch), actions] = reward
    mask[np.arange(batch), actions] = True
    slow_rate = np.minimum(1, consolidation + np.minimum(1, consolidation * np.abs(reward)))
    error = (targets - cue @ old) * mask
    # An explicit row sum makes the independent averaging denominator visible.
    change = sum(slow_rate[row] * np.outer(cue[row], error[row])
                 for row in range(batch)) / batch
    slow = old + change
    strength += change
    prediction = np.einsum("bi,bij->bj", cue, strength)
    correction = (targets - prediction) * mask
    strength += rate * cue[:, :, None] * correction[:, None, :]
    mass += rate
    return {"consolidated": slow, "strength": strength, "mass": mass,
            "writes": previous["writes"] + batch}


def check_memory(previous, memory, key, actions, reward):
    expected = expected_memory(previous, key, actions, reward,
                               consolidation=memory.consolidation,
                               decay=memory.decay, rate=memory.rate)
    deviations = {name: float(np.abs(getattr(memory, name) - expected[name]).max(initial=0))
                  for name in ("consolidated", "strength", "mass")}
    if max(deviations.values()) > 1e-12 or memory.writes != expected["writes"]:
        raise AssertionError("actual masked store write differs from independent local law")
    cue = np.array(key, dtype=float, copy=True)
    cue /= np.linalg.norm(cue, axis=1, keepdims=True)
    readback = np.einsum("bi,bij->bj", cue, memory.strength)
    if memory.rate == 1 and np.max(np.abs(readback[np.arange(len(key)), actions] - reward)) > 1e-12:
        raise AssertionError("one-trial chosen-action read differs from executed outcome")
    return deviations


def declared_lives(founders):
    return [{"seed": seed, "condition": condition, "exposure": exposure,
             "required": condition == "orthogonal" and exposure in REQUIRED_EXPOSURES,
             "planned_batches": 768 + exposure, "status": "unrun", "complete": False,
             "passed": None, "actual_batches": 0, "unknown_current_work": False,
             "planned_cold_query_rows": {"main_and_lesions": 858, "nonlearning": 132,
                                         "frozen_acquired": 99, "store_replay": 297},
             "planned_actual_control_batches": {"nonlearning": 768 + exposure,
                                                 "frozen_acquired": 512 + exposure},
             "planned_store_control_batches": {"old": (512 + exposure) * 5 // 4,
                                                "current": (512 + exposure) * 5 // 4,
                                                "awake": 512 + exposure},
             "planned_pending_seams": 4 + int(exposure > 0)}
            for seed in founders for condition in inputs.CONDITIONS
            for exposure in inputs.EXPOSURES]


def prepare(root, *, confirmation=False, reference=None, formal_source=None):
    """Construct inputs/initial arrays; never act, learn, imagine, predict or solve."""
    began = time.monotonic()
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError("all declared numerical thread variables must equal 1")
    formal_path = None if formal_source is None else Path(formal_source).resolve()
    formal_files = {}
    if formal_path is not None:
        for name in FORMAL:
            source = formal_path / name
            if not source.is_file():
                raise FileNotFoundError("requested formal source is missing: " + str(source))
            formal_files[name] = digest(source)
    founders = [451, 452, 453, 454, 455] if confirmation else [0]
    if confirmation:
        if reference is None:
            raise ValueError("confirmation needs its passed source-bound development reference")
        summary = json.loads((reference / "summary.json").read_text())
        prior = json.loads((reference / "protocol.json").read_text())
        if not reference_passed(prior, summary, digest(reference / "protocol.json")):
            raise ValueError("development reference did not pass its complete frozen census")
    root.mkdir(parents=True)
    package = Path(cadence.__file__).resolve().parent
    manifest = {}
    for original in sorted(package.rglob("*.py")):
        relative = "source/library/cadence/" + original.relative_to(package).as_posix()
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        manifest[relative] = digest(target)
    for name in PRODUCERS:
        target = root / "source" / name
        shutil.copyfile(Path(__file__).with_name(name), target)
        manifest[target.relative_to(root).as_posix()] = digest(target)
    for name, expected in formal_files.items():
        original = formal_path / name
        target = root / "formal" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, target)
        manifest[target.relative_to(root).as_posix()] = digest(target)
        if manifest[target.relative_to(root).as_posix()] != expected:
            raise ValueError("requested formal source changed during copying: " + name)
    repo = Path(__file__).resolve().parents[2]
    for name in BRIDGES:
        target = root / "bridge" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
        manifest[target.relative_to(root).as_posix()] = digest(target)
    artifacts, models, input_bytes = {}, {}, {}
    for seed in founders:
        frozen = inputs.freeze(root / f"inputs-{seed}.npz", seed)
        input_bytes[str(seed)] = {name: value.nbytes for name, value in frozen.items()}
        brain = make_brain(seed)
        models[str(seed)] = model_identity(brain)
        brain.save(root / f"initial-{seed}.npz")
        for name in (f"inputs-{seed}.npz", f"initial-{seed}.npz"):
            artifacts[name] = digest(root / name)
    protocol = {"schema": SCHEMA, "mode": "confirmation" if confirmation else "development",
                "founders": founders, "gates": GATES, "lives": declared_lives(founders),
                "worker_seconds": 900 if confirmation else 300,
                "output_cap_mib": 160 if confirmation else 64,
                "output_reserve_bytes": (8 if confirmation else 2) * 1024**2,
                "storage_reconciliation_seconds": 5,
                "runtime": runtime(), "source_files": manifest, "initial_models": models,
                "admitted_artifacts": artifacts, "library_origin": str(package),
                "frozen_world_probe_index_logical_bytes": input_bytes,
                "main_streams": 8, "teacher_calls": 0, "terminal_done": True,
                "capacity": {"persistent_C": [8, 2], "strength_C_plus_F": [8, 8, 2],
                             "mass": [8], "external_replay_records": 64},
                "sequence_acceptance": "remaining; immediate isolated odors only",
                "replay_scope": "store-only old versus current equal masked writes",
                "required_exposures": list(REQUIRED_EXPOSURES),
                "limits_only": "orthogonal512 and all correlated arms; all retained",
                "formal_scope": ("optional source snapshots, not rechecked; "
                                 "independent finite store arithmetic"),
                "formal_evidence": {
                    "requested": formal_path is not None,
                    "status": ("source_snapshots_bound_not_rechecked" if formal_path is not None
                               else "omitted_not_checked"),
                    "source_origin": None if formal_path is None else str(formal_path),
                    "files": formal_files,
                }}
    if confirmation:
        prior = json.loads((reference / "protocol.json").read_text())
        if (prior["source_files"] != manifest or prior["gates"] != GATES
                or prior["initial_models"]["0"]["learner"] != models[str(founders[0])]["learner"]
                or prior["initial_models"]["0"]["actor"] != models[str(founders[0])]["actor"]
                or prior["runtime"] != protocol["runtime"]):
            raise ValueError("confirmation changed development source/config/runtime")
        control = root / "reference"
        control.mkdir()
        for name in ("protocol.json", "summary.json"):
            shutil.copyfile(reference / name, control / name)
            protocol.setdefault("reference", {})[name] = digest(control / name)
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "admission.json", {"protocol_sha256": digest(root / "protocol.json"),
                                         "status": "prepared-awaiting-source-job-review",
                                         "preparation_seconds": 0.0})
    preflight(root, worker=False)
    atomic_json(root / "admission.json", {"protocol_sha256": digest(root / "protocol.json"),
                                         "status": "prepared-awaiting-source-job-review",
                                         "preparation_seconds": time.monotonic() - began,
                                         "final_admission_io_allowance_seconds": 1.0})
    return protocol


def preflight(root, *, worker=True):
    protocol = json.loads((root / "protocol.json").read_text())
    admission = json.loads((root / "admission.json").read_text())
    if admission["protocol_sha256"] != digest(root / "protocol.json"):
        raise ValueError("admitted protocol changed")
    reserved = admission.get("preparation_seconds", 0) + admission.get(
        "final_admission_io_allowance_seconds", 0,
    )
    if not np.isfinite(reserved) or reserved < 0 or reserved >= protocol["worker_seconds"]:
        raise ValueError("invalid admitted preparation/resource work")
    confirmation = protocol.get("mode") == "confirmation"
    founders = [451, 452, 453, 454, 455] if confirmation else [0]
    if (protocol["schema"] != SCHEMA or protocol["mode"] not in ("development", "confirmation")
            or protocol["founders"] != founders or protocol["gates"] != GATES
            or protocol["lives"] != declared_lives(founders)
            or protocol["worker_seconds"] != (900 if confirmation else 300)
            or protocol["output_cap_mib"] != (160 if confirmation else 64)
            or protocol["output_reserve_bytes"] != (8 if confirmation else 2) * 1024**2
            or protocol["storage_reconciliation_seconds"] != 5
            or protocol["runtime"] != runtime() or protocol["terminal_done"] is not True
            or protocol["teacher_calls"] != 0 or protocol["main_streams"] != 8):
        raise ValueError("declared scientific protocol/runtime differs")
    formal = protocol.get("formal_evidence", {})
    if formal.get("requested") is False:
        formal_valid = (formal.get("status") == "omitted_not_checked"
                        and formal.get("source_origin") is None and formal.get("files") == {})
    elif formal.get("requested") is True:
        formal_valid = (formal.get("status") == "source_snapshots_bound_not_rechecked"
                        and isinstance(formal.get("source_origin"), str)
                        and set(formal.get("files", {})) == set(FORMAL))
    else:
        formal_valid = False
    formal_manifest = {"formal/" + name: expected
                       for name, expected in formal.get("files", {}).items()}
    if (not formal_valid or formal_manifest != {
            name: expected for name, expected in protocol["source_files"].items()
            if name.startswith("formal/")}):
        raise ValueError("formal snapshot status/source binding differs")
    admitted = {**protocol["source_files"], **protocol["admitted_artifacts"]}
    for relative, expected in admitted.items():
        if digest(root / relative) != expected:
            raise ValueError("frozen source/input/initial artifact changed: " + relative)
    imported = {"actual_outcome_chamber.py": Path(__file__),
                "chamber_inputs.py": Path(inputs.__file__)}
    for name, original in imported.items():
        if digest(original) != protocol["source_files"]["source/" + name]:
            raise ValueError("imported producer differs: " + name)
    package = Path(cadence.__file__).resolve().parent
    if worker and package != (root / "source/library/cadence").resolve():
        raise ValueError("worker did not import frozen library")
    current = {"source/library/cadence/" + path.relative_to(package).as_posix(): digest(path)
               for path in package.rglob("*.py")}
    frozen = {name: value for name, value in protocol["source_files"].items()
              if name.startswith("source/library/cadence/")}
    if current != frozen:
        raise ValueError("imported library source differs")
    for seed in founders:
        inputs.load(root / f"inputs-{seed}.npz", seed)
        loaded = Brain.load(root / f"initial-{seed}.npz")
        if (model_identity(loaded) != protocol["initial_models"][str(seed)]
                or model_identity(make_brain(seed)) != protocol["initial_models"][str(seed)]):
            raise ValueError("canonical initial model differs")
    for name, expected in protocol.get("reference", {}).items():
        if digest(root / "reference" / name) != expected:
            raise ValueError("development reference changed")
    if confirmation and not reference_passed(
        json.loads((root / "reference/protocol.json").read_text()),
        json.loads((root / "reference/summary.json").read_text()),
        protocol["reference"]["protocol.json"],
    ):
        raise ValueError("copied development reference failed its recomputed gates")
    if tree_bytes(root) >= protocol["output_cap_mib"] * 1024**2 - protocol["output_reserve_bytes"]:
        raise ValueError("prepared capsule exceeds hard output bound")
    return protocol


class ResourceLimit(RuntimeError):
    pass


class Journal:
    """Append completed calls; a killed current operation retains unknown work."""

    def __init__(self, root, protocol, began):
        self.root, self.protocol, self.began = root, protocol, began
        self.last_progress, self.serial, self.totals = began, 0, {}
        self.work_by_arm = {}
        self.file_sizes, self.output_bytes = {}, 0
        self.last_storage_reconciliation = began
        self.resource_accounting = {"managed_file_size_checks": 0, "full_reconciliations": 0,
                                    "files_scanned": 0, "check_seconds": 0.0,
                                    "largest_managed_operation_bytes": 0}
        self.reconcile_storage()
        self.summary = {"schema": SCHEMA, "protocol_sha256": digest(root / "protocol.json"),
                        "lives": declared_lives(protocol["founders"]), "status": "running",
                        "complete": False, "passed": False, "current_operation": None,
                        "unknown_current_work": False, "completed_calls": 0,
                        "required_denominator": 2 * len(protocol["founders"]),
                        "life_denominator": 6 * len(protocol["founders"]),
                        "sequence_acceptance": "remaining", "process_failure": None}
        self.persist()

    def persist(self, *, full=False):
        self.summary.update(work=self.totals, work_by_arm=self.work_by_arm,
                            completed_calls=self.serial,
                            seconds=time.monotonic() - self.began,
                            resource_accounting=self.resource_accounting.copy())
        if full:
            # The existing summary still occupies disk during atomic replacement.
            # Keep the complete next census inside the declared physical headroom.
            if len((canonical_json(self.summary) + "\n").encode()) > self.protocol[
                "output_reserve_bytes"
            ]:
                self.summary.update(passed=False, complete=False, status="census_reserve_exceeded",
                                    full_census_payloads_retained_in_journal=True)
                self.persist(full=False)
                raise ResourceLimit("complete census exceeds declared output reserve")
            atomic_json(self.root / "summary.json", self.summary)
            self.track_file(self.root / "summary.json")
            return
        # Completed endpoint payloads are in the append-only journal. Avoid
        # repeatedly serializing every previous query array in a learning loop.
        census = {name: value for name, value in self.summary.items() if name != "lives"}
        census["lives"] = []
        for row in self.summary["lives"]:
            small = {name: value for name, value in row.items()
                     if name not in ("endpoints", "controls", "curves", "seams", "random_world")}
            small.update(completed_endpoints=list(row.get("endpoints", {})),
                         completed_pending_seams=len(row.get("seams", [])))
            census["lives"].append(small)
        census["partial_summary_full_payloads_in_journal"] = True
        atomic_json(self.root / "summary.json", census)
        self.track_file(self.root / "summary.json")

    def track_file(self, path):
        """One stat per managed mutation; accounting never rescans earlier records."""
        began = time.monotonic()
        size = path.stat().st_size
        previous = self.file_sizes.get(path, 0)
        self.output_bytes += size - previous
        self.file_sizes[path] = size
        self.resource_accounting["managed_file_size_checks"] += 1
        self.resource_accounting["largest_managed_operation_bytes"] = max(
            self.resource_accounting["largest_managed_operation_bytes"],
            size if path.name == "summary.json" else max(0, size - previous),
        )
        self.resource_accounting["check_seconds"] += time.monotonic() - began

    def reconcile_storage(self):
        """Periodic complete scan charges unmanaged logs and detects accounting drift."""
        began = time.monotonic()
        self.file_sizes = {path: path.stat().st_size for path in self.root.rglob("*")
                           if path.is_file()}
        self.output_bytes = sum(self.file_sizes.values())
        self.last_storage_reconciliation = time.monotonic()
        self.resource_accounting["full_reconciliations"] += 1
        self.resource_accounting["files_scanned"] += len(self.file_sizes)
        self.resource_accounting["check_seconds"] += time.monotonic() - began

    def bounds(self, *, storage=False):
        if time.monotonic() - self.began >= self.protocol["worker_seconds"]:
            raise ResourceLimit("worker total wall bound reached")
        threshold = (self.protocol["output_cap_mib"] * 1024**2
                     - self.protocol["output_reserve_bytes"])
        if storage and (time.monotonic() - self.last_storage_reconciliation
                        >= self.protocol["storage_reconciliation_seconds"]):
            self.reconcile_storage()
        if storage and self.output_bytes >= threshold:
            raise ResourceLimit("retained output reserve reached")

    def begin(self, operation, context):
        # Check every operation, retaining space for its record and the failure census.
        self.bounds(storage=True)
        self.summary.update(current_operation={"kind": operation, **context},
                            unknown_current_work=True)
        self.persist()

    def completed(self, operation, context, body, work=None):
        self.serial += 1
        group = context.get("branch", context.get("arm", "custody_or_probe"))
        if operation in ("private_imagination", "deliberate_feedback_refusal",
                         "duplicate_feedback"):
            group = operation
        arm_work = self.work_by_arm.setdefault(group, {})
        for name, value in (work or {}).items():
            self.totals[name] = self.totals.get(name, 0) + value
            arm_work[name] = arm_work.get(name, 0) + value
        with gzip.open(self.root / "journal.jsonl.gz", "at") as stream:
            stream.write(canonical_json({"serial": self.serial, "operation": operation,
                                         "context": context, "body": body,
                                         "work": work or {}}) + "\n")
        self.track_file(self.root / "journal.jsonl.gz")
        self.summary.update(current_operation=None, unknown_current_work=False)
        if time.monotonic() - self.last_progress >= 25:
            print(canonical_json({"progress": context, "completed_calls": self.serial,
                                  "seconds": time.monotonic() - self.began}), flush=True)
            self.last_progress = time.monotonic()
        self.bounds(storage=True)
        self.persist()

    def save(self, brain, relative):
        self.begin("checkpoint_save", {"path": relative})
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        began = time.monotonic()
        brain.save(path)
        self.track_file(path)
        self.completed("checkpoint_save", {"path": relative}, {"sha256": digest(path)},
                       {"checkpoint_writes": 1, "io_seconds": time.monotonic() - began})
        return path

    def load(self, path):
        self.begin("checkpoint_load", {"path": str(path.relative_to(self.root))})
        began = time.monotonic()
        brain = Brain.load(path)
        self.completed("checkpoint_load", {"path": str(path.relative_to(self.root))}, {},
                       {"checkpoint_reads": 1, "io_seconds": time.monotonic() - began})
        return brain


def endpoint_equations(graph, drive, state, nudge=None):
    """Literal scatter/activation equations for the admitted unmasked CPU law."""
    model, wire = graph.neuron_model, graph.connectome
    if model.adaptation is not None:
        raise ValueError("this frozen chamber admits no adaptation")
    v = np.atleast_2d(state.v)
    sigmoid = 1 / (1 + np.exp(-model.slope * (v - model.threshold)))
    raw = sigmoid - model.rest_emission
    activity = np.where(raw > 0, raw / (1 - model.rest_emission),
                        model.leak * raw / model.rest_emission)
    weights = model.gain * wire.count * graph.efficacy * np.exp(graph.log_gain[wire.pre])
    incoming = np.stack([np.bincount(wire.post, weights=weights * row[wire.pre],
                                    minlength=wire.n) for row in activity])
    defect = incoming + drive + graph.bias - v
    if nudge is not None:
        if (nudge.softmax_temperature != 0.2 or nudge.groups is not None
                or nudge.weight is not None or nudge.anchor is not None):
            raise ValueError("unexpected reward nudge gene")
        mask = np.asarray(nudge.mask, bool)
        logits = activity[:, mask] / nudge.softmax_temperature
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        defect[:, mask] += nudge.beta * (np.asarray(nudge.target)[:, mask] - probability)
    return np.abs(defect).max(axis=1), np.abs(activity - state.activation).max(axis=1)


class Capture:
    """Observational wrappers charge internal solves/reads; never change a result."""

    def __init__(self, witness=None):
        self.witness, self.work, self.phases, self.stores = witness, {}, [], []
        self.depth = 0

    def add(self, name, value):
        self.work[name] = self.work.get(name, 0) + value

    def phase(self, graph, drive, state, nudge, *, kind, reported=None):
        residual, cache = endpoint_equations(graph, drive, state, nudge)
        self.add("independent_equation_row_checks", len(drive))
        self.phases.append({"kind": kind, "steps": int(state.steps),
                            "rows": len(drive), "equation_residual": residual,
                            "cache_defect": cache, "drive_sha256": array_hash(drive),
                            "v_sha256": array_hash(state.v),
                            "activation_sha256": array_hash(state.activation),
                            "beta": None if nudge is None else nudge.beta,
                            "reported": reported})

    def __enter__(self):
        self.originals = []

        def patch(cls, name, method):
            self.originals.append((cls, name, getattr(cls, name)))
            setattr(cls, name, method)

        old_equilibrate, old_settle = NeuralGraph.equilibrate, NeuralGraph.settle_batch
        old_residual = NeuralGraph.residual
        old_observe, old_recall = SynapticMemory.observe, SynapticMemory.recall
        old_update, old_reset = Trace.update, Trace.reset

        def equilibrate(graph, drive, **kwargs):
            self.depth += 1
            try:
                phase = old_equilibrate(graph, drive, **kwargs)
            finally:
                self.depth -= 1
            self.add("equilibrium_calls", 1)
            self.add("stagnation_checks", phase.stagnation_checks)
            self.phase(graph, drive, phase.state, kwargs.get("nudge"), kind="equilibrium",
                       reported={"qualified": phase.qualified, "steps": phase.steps,
                                 "residual": phase.residual,
                                 "residual_checks": phase.residual_checks,
                                 "damping_halvings": phase.damping_halvings,
                                 "stagnation_checks": phase.stagnation_checks})
            return phase

        def settle(graph, drive, **kwargs):
            state = old_settle(graph, drive, **kwargs)
            self.add("settle_kernel_calls", 1)
            self.add("solver_sweeps", state.steps)
            self.add("solver_row_sweeps", state.steps * len(drive))
            if self.depth == 0:
                self.add("finite_phase_calls", 1)
                self.add("finite_movement_stop_checks", state.steps)
                self.phase(graph, drive, state, kwargs.get("nudge"), kind="finite_movement")
            return state

        def residual(graph, drive, state, **kwargs):
            answer = old_residual(graph, drive, state, **kwargs)
            self.add("production_residual_calls", 1)
            self.add("production_residual_row_checks", len(np.atleast_2d(drive)))
            return answer

        def observe(memory, key, value, write=None, **kwargs):
            if self.witness is None or write is not None:
                raise AssertionError("store write lacks its witnessed executed-action owner")
            witness = self.witness
            actions, reward = witness["actions"], witness["reward"]
            expected_target = np.zeros((len(key), 2))
            expected_mask = np.zeros(expected_target.shape, bool)
            expected_target[np.arange(len(key)), actions] = reward
            expected_mask[np.arange(len(key)), actions] = True
            if (not np.array_equal(key, witness["key"])
                    or not np.array_equal(value, expected_target)
                    or not np.array_equal(kwargs.get("value_mask"), expected_mask)
                    or not np.array_equal(kwargs.get("salience"), np.abs(reward))):
                raise AssertionError("store write differs from its executed-action witness")
            previous = memory_arrays(memory)
            old_observe(memory, key, value, **kwargs)
            deviation = check_memory(previous, memory, key, actions, reward)
            self.add("store_write_calls", 1)
            self.add("store_writing_rows", len(key))
            self.add("store_independent_row_checks", len(key))
            self.add("store_dense_outer_products", 2 * len(key) * 8 * 2)
            self.stores.append({"genes": {"consolidation": memory.consolidation,
                                          "decay": memory.decay, "rate": memory.rate},
                                "after": memory_arrays(memory), "deviation": deviation,
                                "consolidated_change": (memory.consolidated
                                                        - previous["consolidated"])})

        def recall(memory, key):
            old_c, writes = memory.consolidated.copy(), memory.writes
            answer = old_recall(memory, key)
            if not np.array_equal(memory.consolidated, old_c) or memory.writes != writes:
                raise AssertionError("memory read rehearsed a guessed observation")
            self.add("store_read_calls", 1)
            self.add("store_read_rows", len(key))
            self.add("store_read_inner_products", len(key) * 8 * 2)
            return answer

        def update(trace, state):
            self.add("trace_update_rows", len(np.atleast_2d(state.activation)))
            return old_update(trace, state)

        def reset(trace, batch, rows=None):
            self.add("trace_reset_rows", batch if rows is None else len(rows))
            return old_reset(trace, batch, rows)

        patch(NeuralGraph, "equilibrate", equilibrate)
        patch(NeuralGraph, "settle_batch", settle)
        patch(NeuralGraph, "residual", residual)
        patch(SynapticMemory, "observe", observe)
        patch(SynapticMemory, "recall", recall)
        patch(Trace, "update", update)
        patch(Trace, "reset", reset)
        self.began = time.monotonic()
        return self

    def __exit__(self, *_):
        for cls, name, original in reversed(self.originals):
            setattr(cls, name, original)
        self.add("call_seconds", time.monotonic() - self.began)


def charge(journal, operation, context, callback, *, witness=None, expect_refusal=False):
    journal.begin(operation, context)
    refused, error, result = False, None, None
    with Capture(witness) as captured:
        try:
            result = callback()
        except LearningPhaseError as failure:
            refused, error = True, {"type": type(failure).__name__, "message": str(failure)}
        except RuntimeError as failure:
            # A numerical refusal is reported by Brain.act. Other errors propagate.
            if "no action issued" not in str(failure):
                raise
            refused, error = True, {"type": type(failure).__name__, "message": str(failure)}
    captured.add(operation + "_calls", 1)
    captured.add(operation + "_refusals", int(refused))
    if operation == "component_store_write" and not refused:
        # A completed callback owns its cost even if journal.completed's cap guard raises.
        counter = "reused_record_writes" if context.get("reused") else "factual_donor_record_writes"
        captured.add(counter, captured.work.get("store_writing_rows", 0))
    journal.completed(operation, context,
                      {"accepted": not refused, "expected_refusal": expect_refusal,
                       "result": result, "error": error, "phases": captured.phases,
                       "store_updates": captured.stores, "witness": witness}, captured.work)
    return result, refused


def capacity(brain):
    actor, memory, trace = brain.basal_ganglia, brain.hippocampus, brain.working_memory
    if memory.consolidated.shape != (8, 2):
        raise AssertionError("persistent memory capacity changed")
    if memory.strength.shape not in ((0, 8, 2), (8, 8, 2)):
        raise AssertionError("main-life memory streams grew or were replaced")
    for name, shape in (("trace", (8, brain.connectome.synapses)),
                        ("trace_bias", (8, brain.connectome.n)),
                        ("trace_critic", (8, 33))):
        value = getattr(actor, name)
        if value is not None and value.shape != shape:
            raise AssertionError("actor/eligibility dimensions changed: " + name)
    if trace.trace.shape not in ((0, 32), (8, 32)):
        raise AssertionError("main-life working trace streams changed")
    return {"neurons": brain.connectome.n, "synapses": brain.connectome.synapses,
            "graph_efficacy_bytes": brain.brain.efficacy.nbytes,
            "graph_bias_bytes": brain.brain.bias.nbytes,
            "graph_gain_bytes": brain.brain.log_gain.nbytes,
            "C_shape": [8, 2], "C_bytes": memory.consolidated.nbytes,
            "strength_capacity_shape": [8, 8, 2], "strength_capacity_bytes": 8 * 8 * 2 * 8,
            "mass_capacity_bytes": 8 * 8,
            "working_capacity_bytes": 8 * 32 * 8 * 2 + 8,
            "actor_trace_capacity_bytes": 8 * (brain.connectome.synapses + brain.connectome.n
                                                + 33) * 8,
            "actor_optimizer_bytes": sum(getattr(actor, name).nbytes for name in
                                         ("velocity", "velocity_bias", "second_moment",
                                          "second_moment_bias")),
            "learner_optimizer_bytes": sum(getattr(brain.learner, name).nbytes for name in
                                           ("velocity", "velocity_bias", "second_moment",
                                            "second_moment_bias")),
            "critic_bytes": actor.w_critic.nbytes + 8,
            "external_buffer_records": 64}


def sampled_event(journal, brain, key, identities, world, following, context):
    action, refused = charge(journal, "sampled_act", context,
                             lambda: brain.act(key, greedy=False))
    if refused:
        return None
    reward = inputs.actual_reward(identities, action, world)
    witness = {"key": key.copy(), "identities": identities.copy(), "actions": action.copy(),
               "reward": reward, "world": world, "done": np.ones(8, bool),
               "event_ids": [f"{context}/{row}" for row in range(8)],
               "probabilities": brain.basal_ganglia.probabilities(brain.basal_ganglia.state)}
    if (brain._moment is None or not np.array_equal(brain._moment[0], key)
            or not np.array_equal(brain._moment[1], action)):
        raise AssertionError("public pending action differs from actual executed event")
    return witness


def actual_feedback(journal, brain, witness, following, context):
    before = brain.basal_ganglia.updates
    before_weights, before_bias = brain.brain.efficacy.copy(), brain.brain.bias.copy()
    report, refused = charge(
        journal, "actual_feedback", context,
        lambda: brain.learn(witness["reward"], witness["done"], following), witness=witness,
    )
    if not refused:
        if (brain.basal_ganglia.updates != before + 1 or brain.learner.contrast_updates != 0
                or brain._moment is not None or brain.basal_ganglia._pending is not None
                or not brain.working_memory.cold.all()
                or brain.working_memory.trace.any()
                or any(getattr(brain.basal_ganglia, name).any()
                       for name in ("trace", "trace_bias", "trace_critic"))):
            raise AssertionError("actual terminal feedback violates custody/reset contract")
        journal.completed("parameter_movement", context,
                          {"efficacy_l2": float(np.linalg.norm(
                              brain.brain.efficacy - before_weights)),
                           "bias_l2": float(np.linalg.norm(brain.brain.bias - before_bias)),
                           "memory_writes": brain.hippocampus.writes,
                           "capacity": capacity(brain)},
                          {"actor_feedback_bookkeeping_updates": 1,
                           "critic_candidate_updates": 1,
                           "parameter_changing_feedback_updates": int(
                               not np.array_equal(brain.brain.efficacy, before_weights)
                               or not np.array_equal(brain.brain.bias, before_bias))})
    return report, refused


def cold_probe(journal, anchor, initial, probes, prototypes, world, context, control="intact"):
    brain = journal.load(anchor)
    if control in ("initial_graph", "joint_reset"):
        base = journal.load(initial)
        brain.learner.brain = brain.brain.with_parameters(
            efficacy=base.brain.efficacy, bias=base.brain.bias, log_gain=base.brain.log_gain,
        )
    if control in ("graph_only", "joint_reset"):
        charge(journal, "store_clear", context, brain.hippocampus.clear)
    if control not in ("intact", "initial_graph", "joint_reset", "graph_only"):
        raise ValueError("unknown causal probe control")
    keys = np.concatenate([prototypes[:3], probes.reshape(30, 8)])
    identities = np.concatenate([np.arange(3), np.repeat(np.arange(3), 10)])
    labels = inputs.MAPPING[world, identities]
    answers = np.full(33, -1, dtype=np.int64)
    read_values = np.zeros((33, 2))
    margins = [None] * 33
    slow, writes = brain.hippocampus.consolidated.copy(), brain.hippocampus.writes
    weights, bias = brain.brain.efficacy.copy(), brain.brain.bias.copy()
    for start in range(0, 33, 8):
        x = keys[start:start + 8]
        detail = {**context, "control": control, "probe_start": start, "rows": len(x)}

        def reset(x=x):
            brain.reset()
            brain.hippocampus.reset(len(x))

        charge(journal, "cold_reset", detail, reset)
        values, _ = charge(journal, "component_read", detail,
                           lambda x=x: brain.hippocampus.recall(x))
        read_values[start:start + len(x)] = values
        action, refused = charge(journal, "free_probe", detail,
                                 lambda x=x: brain.act(x, greedy=True))
        if not refused:
            report = brain.last_settlement
            if report["qualified"] is not True or report["max_residual"] > 0.003:
                raise AssertionError("public free probe published an unqualified action")
            answers[start:start + len(x)] = action
            motor = brain.basal_ganglia.state.activation[:, brain.motor_index]
            for row in range(len(x)):
                label = labels[start + row]
                margins[start + row] = float(motor[row, label] - motor[row, 1 - label])
    if (not np.array_equal(slow, brain.hippocampus.consolidated)
            or writes != brain.hippocampus.writes
            or not np.array_equal(weights, brain.brain.efficacy)
            or not np.array_equal(bias, brain.brain.bias)):
        raise AssertionError("teacher-free cold query changed durable memory or parameters")
    rows = []
    for cue in range(3):
        selected = slice(3 + 10 * cue, 3 + 10 * (cue + 1))
        variant = answers[selected]
        label = int(inputs.MAPPING[world, cue])
        rows.append({"cue": cue, "label": label, "planned": 10, "attempted": 10,
                     "unrun": 0, "refused": int((variant < 0).sum()),
                     "correct": int((variant == label).sum()),
                     "obsolete": int((variant == 1 - label).sum()) if cue < 2 else 0,
                     "prototype_prediction": int(answers[cue]),
                     "prototype_correct": int(answers[cue] == label),
                     "predictions": variant, "memory_values": read_values[selected],
                     "margins": margins[selected]})
    return {"control": control, "world": world, "planned": 33, "attempted": 33,
            "unrun": 0, "refused": int((answers < 0).sum()), "cues": rows,
            "qualified_free": bool((answers >= 0).all()),
            "teacher_calls": 0, "durable_query_pure": True}


def probe_endpoint(journal, brain, initial, arrays, condition, world, relative, context,
                   *, lesions=True):
    anchor = journal.save(brain, relative + ".npz")
    controls = ("intact", "graph_only", "initial_graph", "joint_reset") if lesions else ("intact",)
    results = {control: cold_probe(journal, anchor, initial, arrays[condition + "/probes"],
                                  arrays[condition + "/prototypes"], world, context, control)
               for control in controls}
    live = journal.save(brain, relative + "-after-queries.npz")
    if not checkpoints_equal(anchor, live):
        raise AssertionError("private endpoint probes changed their main-life anchor")
    uniform = arrays[condition + "/uniform_probes"]
    results["uniform"] = {"planned": 30, "correct": int((
        uniform == inputs.MAPPING[world, :3, None]).sum()), "actions": uniform,
        "per_cue": [int((uniform[cue] == inputs.MAPPING[world, cue]).sum()) for cue in range(3)]}
    results["main_anchor_sha256"] = digest(anchor)
    results["main_checkpoint_pure"] = True
    journal.completed("endpoint_score", {**context, "checkpoint": relative}, results)
    return anchor, results


def checkpoint_seam(journal, brain, witness, following, relative, context):
    """Reprocess one already executed witness on branches; the real life never retries."""
    anchor = journal.save(brain, relative + "-pending.npz")
    branch, reference = journal.load(anchor), journal.load(anchor)
    charge(journal, "private_imagination", context,
           lambda: [{"qualified": phase.qualified, "steps": phase.steps,
                     "residual": phase.residual} for phase in branch.imagine(
                         [following, witness["key"]], budget=1024, tolerance=0.003,
                     )])
    private = journal.save(branch, relative + "-after-private.npz")
    private_pure = checkpoints_equal(anchor, private)
    config = branch.learner.config
    branch.learner.config = replace(config, qualified=True, free_steps=0, tolerance=1e-12)
    before = journal.save(branch, relative + "-tight-before.npz")
    _, refused = charge(journal, "deliberate_feedback_refusal", context,
                         lambda: branch.learn(witness["reward"], witness["done"], following),
                         witness=witness, expect_refusal=True)
    after = journal.save(branch, relative + "-tight-after.npz")
    refusal_pure = refused and checkpoints_equal(before, after)
    branch.learner.config = config
    left, right = [], []
    for candidate, name in ((branch, "refused-restored"), (reference, "loaded-reference")):
        _, feedback_refused = actual_feedback(
            journal, candidate, witness, following,
            {**context, "branch": name, "reused_witness": True},
        )
        saved = journal.save(candidate, relative + "-" + name + ".npz")
        if name == "refused-restored":
            left = [feedback_refused, saved]
        else:
            right = [feedback_refused, saved]
    accepted_equal = not left[0] and not right[0]
    arrays_equal = checkpoints_equal(left[1], right[1])
    duplicate_before = journal.save(branch, relative + "-duplicate-before.npz")
    journal.begin("duplicate_feedback", context)
    duplicate_refused = False
    try:
        branch.learn(witness["reward"], witness["done"], following)
    except RuntimeError as error:
        if "non-greedy act first" not in str(error):
            raise
        duplicate_refused = True
    journal.completed("duplicate_feedback", context, {"refused": duplicate_refused},
                       {"duplicate_feedback_calls": 1})
    duplicate_after = journal.save(branch, relative + "-duplicate-after.npz")
    duplicate_pure = duplicate_refused and checkpoints_equal(duplicate_before, duplicate_after)
    answers, refusals = [], []
    for candidate, name in ((branch, "resumed"), (reference, "original")):
        answer, refusal = charge(journal, "continuation_next_act", {**context, "branch": name},
                                 lambda candidate=candidate: candidate.act(following))
        answers.append(answer)
        refusals.append(refusal)
    next_action_equal = not any(refusals) and np.array_equal(*answers)
    last_left = journal.save(branch, relative + "-continued.npz")
    last_right = journal.save(reference, relative + "-resumed.npz")
    live = journal.save(brain, relative + "-live-after-controls.npz")
    flags = {"private_pure": private_pure, "refusal_pure": refusal_pure,
             "feedback_accepted_equal": accepted_equal, "feedback_arrays_equal": arrays_equal,
             "duplicate_pure": duplicate_pure, "next_action_equal": next_action_equal,
             "next_pending_arrays_equal": checkpoints_equal(last_left, last_right),
             "main_life_pure": checkpoints_equal(anchor, live)}
    result = {**flags, "passed": all(flags.values()), "witness_reprocessed_twice": True,
              "next_branch_action_is_pending": True}
    journal.completed("pending_seam_result", context, result)
    return result


def freeze_control(brain):
    """Explicit component lesion: actor/critic/C fixed; fast residuals may fade."""
    brain.basal_ganglia.config = replace(brain.basal_ganglia.config,
                                         eta=0, eta_bias=0, eta_critic=0)
    brain.hippocampus.consolidation = 0
    brain.hippocampus.rate = 0
    return brain


def record_buffer(journal, witnesses, relative):
    if len(witnesses) < 8:
        raise AssertionError("last64 real acquired records are not available")
    recent = witnesses[-8:]
    buffer = {"key": np.concatenate([row["key"] for row in recent]),
              "identities": np.concatenate([row["identities"] for row in recent]),
              "actions": np.concatenate([row["actions"] for row in recent]),
              "reward": np.concatenate([row["reward"] for row in recent]),
              "done": np.concatenate([row["done"] for row in recent]),
              "event_ids": np.concatenate([np.asarray(row["event_ids"]) for row in recent])}
    path = journal.root / relative
    journal.begin("external_buffer_freeze", {"path": relative})
    np.savez_compressed(path, **buffer)
    journal.track_file(path)
    journal.completed("external_buffer_freeze", {"path": relative},
                      {"sha256": digest(path), "records": 64,
                       "logical_bytes": sum(value.nbytes for value in buffer.values())},
                      {"external_records_retained": 64,
                       "external_record_bytes": sum(value.nbytes for value in buffer.values())})
    return buffer


def replay_witness(buffer, indices):
    return {name: value[indices].copy() for name, value in buffer.items()} | {
        "world": 0, "source": "last64-actual-acquisition-records", "reused": True,
    }


def store_write(journal, brain, witness, context, *, reused=False):
    target = np.zeros((8, 2))
    mask = np.zeros((8, 2), bool)
    target[np.arange(8), witness["actions"]] = witness["reward"]
    mask[np.arange(8), witness["actions"]] = True
    _, refused = charge(
        journal, "component_store_write", {**context, "reused": reused},
        lambda: brain.hippocampus.observe(witness["key"], target,
                                          salience=np.abs(witness["reward"]), value_mask=mask),
        witness=witness,
    )
    if refused:
        raise AssertionError("a direct admitted store record was refused")


def score_gate(endpoints, seams, *, complete, required):
    """Cohort baseline gap is computed separately; no ten-item random gap gate."""
    if not complete or not {"acquired", "routine", "revision", "restored"}.issubset(endpoints):
        return {"passed": False, "required": required, "reason": "incomplete-census"}
    positive = True
    retained_c = []
    details = {}
    for name in ("acquired", "routine", "revision", "restored"):
        reading = endpoints[name]["intact"]
        cue_pass = [row["correct"] >= GATES["variants_correct"]
                    and row["prototype_correct"] == 1 and row["refused"] == row["unrun"] == 0
                    for row in reading["cues"]]
        if name == "revision":
            cue_pass = [passed and (cue == 2 or reading["cues"][cue]["obsolete"] <=
                                    GATES["obsolete_max"])
                        for cue, passed in enumerate(cue_pass)]
        details[name] = cue_pass
        retained_c.append(reading["cues"][2]["correct"])
        positive = positive and all(cue_pass) and reading["refused"] == reading["unrun"] == 0
    rare_drop = max(retained_c[0] - value for value in retained_c[1:])
    continuation = bool(seams) and all(row["passed"] is True for row in seams)
    return {"passed": bool(positive and rare_drop <= 1 and continuation), "required": required,
            "per_endpoint_cue_pass": details, "rare_max_drop": rare_drop,
            "continuation_passed": continuation, "baseline_gap_scope": "cohort-only"}


def rescore_previous(reading, world):
    """World changes re-score witnessed predictions, with no new query or answer."""
    result = deepcopy(reading)
    result["previous_world"] = reading["world"]
    result["world"] = world
    result["reused_predictions_zero_new_solves"] = True
    for cue, row in enumerate(result["cues"]):
        label = int(inputs.MAPPING[world, cue])
        prediction = np.asarray(row["predictions"])
        changed = label != row["label"]
        row.update(label=label, correct=int((prediction == label).sum()),
                   obsolete=int((prediction == 1 - label).sum()) if cue < 2 else 0,
                   prototype_correct=int(row["prototype_prediction"] == label))
        if changed:
            row["margins"] = [None if margin is None else -margin for margin in row["margins"]]
    return result


def probes_complete(value):
    """A refused lesion/reset probe cannot improve a measured baseline gap."""
    if isinstance(value, dict):
        if "qualified_free" in value and (
            value["qualified_free"] is not True or value["refused"] or value["unrun"]
            or value["attempted"] != value["planned"]
        ):
            return False
        return all(probes_complete(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(probes_complete(child) for child in value)
    return True


def life(journal, index, protocol, arrays):
    row = journal.summary["lives"][index]
    seed, condition, exposure = row["seed"], row["condition"], row["exposure"]
    prefix = f"seed-{seed}/{condition}-{exposure}"
    context = {"life": index, "seed": seed, "condition": condition, "exposure": exposure}
    row.update(status="running", capacity=None, endpoints={}, curves={}, seams=[],
               controls={"nonlearning": {}, "frozen_acquired": {}, "store_replay": {}},
               random_world={}, current_batch=None, unknown_current_work=True)
    journal.persist(full=True)
    initial = journal.root / f"initial-{seed}.npz"
    brain, untrained = journal.load(initial), freeze_control(journal.load(initial))
    row["capacity"] = capacity(brain)
    fixed_capacity = row["capacity"]
    frozen, buffer, replay = None, None, {}
    prototypes = arrays[condition + "/prototypes"]
    schedules = {phase: arrays["schedule/" + phase][:exposure if phase == "routine" else 256]
                 for phase in inputs.PHASES}
    recent, factual_batches = [], 0
    outcomes = {phase: {"events": 0, "reward_sum": 0.0, "correct_non_neutral": 0,
                        "non_neutral": 0, "rare_events": 0, "rare_penalties": 0,
                        "actions_per_cue": [[0, 0] for _ in range(4)]}
                for phase in inputs.PHASES}
    _, row["endpoints"]["newborn"] = probe_endpoint(
        journal, brain, initial, arrays, condition, 0, prefix + "/newborn", context,
    )

    def endpoint(name, world):
        anchor, result = probe_endpoint(
            journal, brain, initial, arrays, condition, world, prefix + "/" + name,
            {**context, "endpoint": name},
        )
        row["endpoints"][name] = result
        _, row["controls"]["nonlearning"][name] = probe_endpoint(
            journal, untrained, initial, arrays, condition, world,
            prefix + "/nonlearning-" + name, {**context, "endpoint": name, "arm": "nonlearning"},
            lesions=False,
        )
        if frozen is not None:
            _, row["controls"]["frozen_acquired"][name] = probe_endpoint(
                journal, frozen, initial, arrays, condition, world,
                prefix + "/frozen-" + name,
                {**context, "endpoint": name, "arm": "frozen-acquired"}, lesions=False,
            )
        for name_control, candidate in replay.items():
            _, score = probe_endpoint(
                journal, candidate, initial, arrays, condition, world,
                prefix + f"/store-{name_control}-{name}",
                {**context, "endpoint": name, "arm": "store-" + name_control}, lesions=False,
            )
            row["controls"]["store_replay"].setdefault(name_control, {})[name] = score
        journal.persist(full=True)
        return anchor

    for phase in inputs.PHASES:
        schedule = schedules[phase]
        world = inputs.WORLD[phase]
        uniform = arrays["uniform/" + phase][:len(schedule)]
        random_reward = inputs.actual_reward(schedule, uniform, world)
        row["random_world"][phase] = {
            "events": int(schedule.size), "reward_sum": float(random_reward.sum()),
            "non_neutral": int((schedule != 3).sum()),
            "correct_non_neutral": int(((uniform == inputs.MAPPING[world, schedule])
                                         & (schedule != 3)).sum()),
        }
        if phase in ("revision", "restored"):
            previous_name = "routine" if phase == "revision" else "revision"
            previous = row["endpoints"][previous_name]["intact"]
            row["curves"][phase] = {"0": {"reused_endpoint": previous_name,
                                           "reading": rescore_previous(previous, world)}}
            journal.completed("world_change_rescore", {**context, "phase": phase},
                              row["curves"][phase]["0"], {"reused_probe_comparisons": 33})
        for number, identities in enumerate(schedule, 1):
            detail = {**context, "phase": phase, "batch": number, "arm": "main"}
            row.update(current_batch={"phase": phase, "batch": number}, unknown_current_work=True)
            journal.persist()
            key = prototypes[identities]
            if number < len(schedule):
                following = prototypes[schedule[number]]
            else:
                next_phase = inputs.PHASES.index(phase) + 1
                while (next_phase < len(inputs.PHASES)
                       and not len(schedules[inputs.PHASES[next_phase]])):
                    next_phase += 1
                following = (prototypes[schedules[inputs.PHASES[next_phase]][0]]
                             if next_phase < len(inputs.PHASES)
                             else np.repeat(prototypes[[3]], 8, 0))
            witness = sampled_event(journal, brain, key, identities, world, following, detail)
            if witness is None:
                row.update(status="refused_main_action", passed=False, unknown_current_work=False)
                journal.save(brain, prefix + "/refused-main-action.npz")
                return row
            if ((number == len(schedule) and phase != "restored")
                    or phase in ("revision", "restored") and number == 128):
                row["seams"].append(checkpoint_seam(
                    journal, brain, witness, following, prefix + f"/seam-{phase}-{number}", detail,
                ))
            _, refused = actual_feedback(journal, brain, witness, following, detail)
            if refused:
                row.update(status="refused_main_feedback", passed=False, unknown_current_work=False)
                journal.save(brain, prefix + "/refused-main-feedback.npz")
                return row
            if capacity(brain) != fixed_capacity:
                raise AssertionError("continuing acquired brain capacity changed")
            row["actual_batches"] += 1
            stats = outcomes[phase]
            stats["events"] += 8
            stats["reward_sum"] += float(witness["reward"].sum())
            stats["non_neutral"] += int((identities != 3).sum())
            stats["correct_non_neutral"] += int((
                (witness["actions"] == inputs.MAPPING[world, identities]) & (identities != 3)
            ).sum())
            stats["rare_events"] += int((identities == 2).sum())
            stats["rare_penalties"] += int(((identities == 2) & (witness["reward"] < 0)).sum())
            for cue, action in zip(identities, witness["actions"], strict=True):
                stats["actions_per_cue"][cue][action] += 1
            if phase == "acquisition":
                recent.append(witness)
                if len(recent) > 8:
                    del recent[0]
            else:
                factual_batches += 1
                for name, candidate in replay.items():
                    store_write(journal, candidate, witness, {**detail, "arm": "store-" + name})
                if factual_batches % 4 == 0:
                    indices = arrays["replay/indices"][factual_batches // 4 - 1]
                    store_write(journal, replay["old"], replay_witness(buffer, indices),
                                {**detail, "arm": "store-old", "buffer_indices": indices},
                                reused=True)
                    store_write(journal, replay["current"], witness,
                                {**detail, "arm": "store-current"}, reused=True)
            for candidate, arm in ((untrained, "nonlearning"), (frozen, "frozen-acquired")):
                if candidate is None:
                    continue
                unchanged = (candidate.brain.efficacy.copy(), candidate.brain.bias.copy(),
                             candidate.hippocampus.consolidated.copy(),
                             candidate.basal_ganglia.w_critic.copy(),
                             candidate.basal_ganglia.b_critic)
                own = sampled_event(journal, candidate, key, identities, world, following,
                                    {**detail, "arm": arm})
                if own is None:
                    row.update(status="refused_frozen_control", passed=False,
                               unknown_current_work=False)
                    return row
                _, refused = actual_feedback(journal, candidate, own, following,
                                              {**detail, "arm": arm})
                if refused or not all(np.array_equal(left, right) for left, right in zip(
                    unchanged, (candidate.brain.efficacy, candidate.brain.bias,
                                candidate.hippocampus.consolidated,
                                candidate.basal_ganglia.w_critic, candidate.basal_ganglia.b_critic),
                    strict=True,
                )):
                    raise AssertionError("frozen control changed its pinned learned state")
            row.update(current_batch=None, unknown_current_work=False)
            if phase in ("revision", "restored") and number in (32, 64, 128):
                _, reading = probe_endpoint(
                    journal, brain, initial, arrays, condition, world,
                    prefix + f"/curve-{phase}-{number}", detail, lesions=False,
                )
                row["curves"][phase][str(number)] = reading["intact"]
            journal.persist()
        name = {"acquisition": "acquired", "routine": "routine",
                "revision": "revision", "restored": "restored"}[phase]
        anchor = endpoint(name, world)
        if phase == "acquisition":
            frozen = freeze_control(journal.load(anchor))
            buffer = record_buffer(journal, recent, prefix + "/actual-record-buffer.npz")
            recent.clear()
            replay = {name: journal.load(anchor) for name in ("old", "current", "awake")}
        if phase in ("revision", "restored"):
            row["curves"][phase]["256"] = row["endpoints"][name]["intact"]
    row.update(actual_world=outcomes, complete=True, status="completed",
               unknown_current_work=False, current_batch=None)
    row["gate"] = score_gate(row["endpoints"], row["seams"], complete=True,
                             required=row["required"])
    row["all_free_controls_qualified"] = probes_complete(
        [row["endpoints"], row["curves"], row["controls"]],
    )
    row["passed"] = bool(row["gate"]["passed"] and row["all_free_controls_qualified"]
                         and len(row["seams"]) == row["planned_pending_seams"])
    row["replay_work"] = {"factual_batches": factual_batches,
                           "old_extra_batches": factual_batches // 4,
                           "current_extra_batches": factual_batches // 4,
                           "awake_extra_batches": 0, "equal_store_writes_old_current": True,
                           "query_sweeps_equal_claimed": False, "integrated_replay_claimed": False}
    row["frozen_work_semantics"] = (
        "accepted feedback increments actor/learner bookkeeping in frozen controls; "
        "graph/C/critic arrays are checked unchanged and fast residuals can fade"
    )
    journal.persist(full=True)
    return row


def cohort_gate(lives):
    required = [row for row in lives if row["required"]]
    correct = uniform = joint = planned = 0
    for life_row in required:
        for name in ("acquired", "routine", "revision", "restored"):
            endpoint = life_row.get("endpoints", {}).get(name)
            planned += 30
            if endpoint is None:
                continue
            correct += sum(cue["correct"] for cue in endpoint["intact"]["cues"])
            uniform += endpoint["uniform"]["correct"]
            joint += sum(cue["correct"] for cue in endpoint["joint_reset"]["cues"])
    return {"planned": planned, "correct": correct, "uniform_correct": uniform,
            "joint_reset_correct": joint,
            "accuracy": correct / planned if planned else 0,
            "uniform_accuracy": uniform / planned if planned else 0,
            "joint_reset_accuracy": joint / planned if planned else 0,
            "passed": bool(planned and (correct - uniform) / planned >= GATES["cohort_gap"]
                           and (correct - joint) / planned >= GATES["cohort_gap"])}


def reference_passed(protocol, summary, protocol_sha256):
    """A mutable pass flag alone cannot admit a fresh confirmation."""
    expected = declared_lives([0])
    lives = summary.get("lives", [])
    if (protocol.get("schema") != SCHEMA or protocol.get("mode") != "development"
            or protocol.get("founders") != [0] or protocol.get("gates") != GATES
            or summary.get("protocol_sha256") != protocol_sha256
            or summary.get("passed") is not True or summary.get("complete") is not True
            or summary.get("process_failure") is not None or summary.get("unknown_current_work")
            or len(lives) != len(expected)):
        return False
    for row, declared in zip(lives, expected, strict=True):
        if (any(row.get(field) != declared[field]
                for field in ("seed", "condition", "exposure", "required", "planned_batches"))
                or row.get("complete") is not True
                or row.get("actual_batches") != declared["planned_batches"]):
            return False
        if row["required"] and (
            len(row.get("seams", [])) != declared["planned_pending_seams"]
            or not probes_complete([row.get("endpoints", {}), row.get("curves", {}),
                                    row.get("controls", {})])
            or not score_gate(row.get("endpoints", {}), row.get("seams", []),
                              complete=True, required=True)["passed"]
        ):
            return False
    return cohort_gate(lives)["passed"]


def finalize(journal):
    summary = journal.summary
    lives = summary["lives"]
    summary["cohort"] = cohort_gate(lives)
    summary["unrun_batches"] = sum(row["planned_batches"] - row["actual_batches"] for row in lives)
    summary["complete"] = all(row["complete"] for row in lives)
    summary["passed"] = bool(summary["complete"] and not summary["process_failure"]
                             and not summary["unknown_current_work"]
                             and all(row["passed"] is True for row in lives if row["required"])
                             and summary["cohort"]["passed"])
    if summary["process_failure"] is None:
        summary["status"] = "bounded_contract_passed" if summary["passed"] else (
            "bounded_contract_failed" if summary["complete"] else "incomplete_census")
    summary["output_bytes"] = tree_bytes(journal.root)
    summary["output_cap_exceeded"] = (
        summary["output_bytes"] >= journal.protocol["output_cap_mib"] * 1024**2
    )
    if summary["output_cap_exceeded"]:
        summary.update(passed=False, status="output_cap_exceeded")
    journal.persist(full=True)
    final_bytes = tree_bytes(journal.root)
    if final_bytes >= journal.protocol["output_cap_mib"] * 1024**2:
        summary.update(passed=False, status="output_cap_exceeded", output_bytes=final_bytes,
                       output_cap_exceeded=True)
        journal.persist(full=True)
    return summary


def worker(root):
    began = time.monotonic()
    protocol = preflight(root)
    admission = json.loads((root / "admission.json").read_text())
    began -= admission.get("preparation_seconds", 0) + admission.get(
        "final_admission_io_allowance_seconds", 0,
    )
    journal = Journal(root, protocol, began)
    try:
        for index, row in enumerate(journal.summary["lives"]):
            arrays = inputs.load(root / f"inputs-{row['seed']}.npz", row["seed"])
            life(journal, index, protocol, arrays)
        journal.bounds(storage=True)
        preflight(root)
    except BaseException as error:
        journal.summary.update(status="process_failed", process_failure={
            "type": type(error).__name__, "message": str(error),
        })
        for row in journal.summary["lives"]:
            if row["status"] == "running":
                row.update(status="interrupted", passed=None, complete=False,
                           unknown_current_work=journal.summary["unknown_current_work"])
        raise
    finally:
        finalize(journal)
    return journal.summary


def failed_process(root, protocol, failure):
    path = root / "summary.json"
    try:
        summary = json.loads(path.read_text())
        if (summary["protocol_sha256"] != digest(root / "protocol.json")
                or len(summary["lives"]) != len(protocol["lives"])):
            raise ValueError("invalid worker census")
    except (OSError, ValueError, KeyError):
        summary = {"schema": SCHEMA, "protocol_sha256": digest(root / "protocol.json"),
                   "lives": declared_lives(protocol["founders"]), "work": {},
                   "unknown_current_work": True}
    for row in summary["lives"]:
        if row["status"] == "running":
            row.update(status="interrupted", passed=None, complete=False, unknown_current_work=True)
        elif row["status"] == "unrun":
            row["status"] = "not_run_process_failure"
    summary.update(passed=False, complete=False, status="process_failed", process_failure=failure)
    atomic_json(path, summary)
    return summary


def launch(root):
    began = time.monotonic()
    protocol = preflight(root, worker=False)
    admission = json.loads((root / "admission.json").read_text())
    reserved = admission.get("preparation_seconds", 0) + admission.get(
        "final_admission_io_allowance_seconds", 0,
    )
    available = protocol["worker_seconds"] - reserved - (time.monotonic() - began)
    env = dict(os.environ, PYTHONPATH=str(root / "source/library"), PYTHONDONTWRITEBYTECODE="1")
    env.update({name: "1" for name in THREADS})
    try:
        with (root / "worker.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, str(root / "source/actual_outcome_chamber.py"),
                 "--worker", "--out", str(root)], env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False, timeout=available,
            )
    except subprocess.TimeoutExpired as error:
        failed_process(root, protocol, {"type": "hard_timeout", "seconds": error.timeout})
        raise
    except BaseException as error:
        failed_process(root, protocol, {"type": type(error).__name__, "message": str(error)})
        raise
    if result.returncode:
        failed_process(root, protocol, {"type": "worker_exit", "exit_code": result.returncode})
        raise subprocess.CalledProcessError(result.returncode, result.args)
    summary = json.loads((root / "summary.json").read_text())
    if (summary.get("protocol_sha256") != digest(root / "protocol.json")
            or summary.get("complete") is not True):
        failed_process(root, protocol, {"type": "worker_exited_without_complete_census"})
        raise RuntimeError("worker exited without the complete planned census")
    summary["process_exit_code"] = 0
    summary["launch_wall_seconds"] = time.monotonic() - began
    summary["preparation_seconds"] = admission.get("preparation_seconds", 0)
    summary["preparation_plus_launch_wall_seconds"] = (
        summary["launch_wall_seconds"] + summary["preparation_seconds"]
    )
    atomic_json(root / "summary.json", summary)
    if tree_bytes(root) >= protocol["output_cap_mib"] * 1024**2:
        failed_process(root, protocol, {"type": "final_output_cap_exceeded"})
        raise ResourceLimit("final output cap exceeded")
    if time.monotonic() - began + summary["preparation_seconds"] >= protocol["worker_seconds"]:
        failed_process(root, protocol, {"type": "final_total_wall_cap_exceeded"})
        raise ResourceLimit("final preparation/launch wall cap exceeded")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--worker", action="store_true")
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--formal-source", type=Path,
                        help="optional directory of Lean source snapshots; no proof recheck")
    args = parser.parse_args(argv)
    if args.formal_source is not None and not args.prepare:
        parser.error("--formal-source is only supported with --prepare")
    root = args.out.resolve()
    if args.prepare:
        prepare(root, confirmation=args.confirmation, reference=args.reference,
                formal_source=args.formal_source)
        print(canonical_json({"status": "prepared_no_solves", "out": root,
                              "protocol_sha256": digest(root / "protocol.json")}))
    elif args.worker:
        worker(root)
    else:
        print(canonical_json(launch(root)))


if __name__ == "__main__":
    main()
