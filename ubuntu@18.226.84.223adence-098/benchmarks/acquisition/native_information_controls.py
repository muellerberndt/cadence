"""Bounded reference-only native information controls, frozen before any fit.

Nothing here settles or teaches Cadence, supplies its answers, or reads heldout
arrays. The original school24 fitting role stays separate from the diagnostic
42-witness role, which explicitly teaches previously queried TRAIN18 to the
reference. The known action codec is a declared structural prior.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PANELS = {"school": 24, "independent_train": 18, "development": 19}
GROUPS = {"scene": [0, 192], "motion": [192, 384], "fovea": [384, 640], "body": [640, 650]}
MODES = [
    "raw_full",
    "balanced_full",
    "scene",
    "motion",
    "fovea",
    "body",
    "sensory650",
    "initial_projection32",
    "final_projection32",
]
ARMS = ["school-only", "expanded-train"]
LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0]
THREADS = {
    name: "1"
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}
FIXED = {
    "schema": "cadence-native-information-reference-v1",
    "panels": PANELS,
    "groups": GROUPS,
    "feature_modes": MODES,
    "fit_arms": ARMS,
    "ridge_lambdas": LAMBDAS,
    "primary": {"arm": "school-only", "feature": "raw_full", "readout": "factorized"},
    "kernel": "1 + .5*linear + .5*exp(-squared_distance/train_median_nonzero_distance)",
    "regularization_choice": (
        "maximum weighted TRAIN-only leave-one-witness-out composite accuracy; lowest lambda on tie"
    ),
    "loo_scope": (
        "frozen full-TRAIN input-only conditioning/kernel; each row's target excluded by "
        "exact ridge LOO formula; expanded arm retains the other same-action witness"
    ),
    "extra_categorical_controls": ["raw_full", "final_projection32"],
    "categorical_lambda": 0.1,
    "fit_budget": 94,
    "seconds_cap": 60,
    "worker_seconds_cap": 75,
    "output_cap_mib": 16,
    "final_receipt_reserve_bytes": 16384,
    "threads": THREADS,
    "random_seed": 110065032,
    "information_gates": {"school": 18, "independent_train": 14, "development": 15},
    "loo_equation": (
        "D=sqrt(diag(weights)); A=D*K*D+lambda*I; alpha=D*A^-1*D*Y; "
        "H=K*D*A^-1*D; leave-one-row score=Y-(Y-K*alpha)/(1-diag(H))"
    ),
    "heldout": "never decoded",
    "neural_queries": 0,
    "teacher_calls": 0,
    "equilibrium_solves": 0,
    "whole_issue_passed": False,
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def read_panels(root):
    provenance = json.loads((root / "source/provenance.json").read_text())
    panels = {}
    with np.load(root / "source/school.npz", allow_pickle=False) as archive:
        for name, count in PANELS.items():
            x, y = archive[name + "_inputs"], archive[name + "_labels"]
            if (
                x.shape != (count, 650)
                or y.shape != (count,)
                or y.dtype.kind not in "iu"
                or np.any((y < 0) | (y >= 36))
                or len(np.unique(y)) != count
                or not np.isfinite(x).all()
            ):
                raise ValueError("admitted native panel shape differs")
            for row, label, identity in zip(x, y, provenance["panels"][name], strict=True):
                if sha_bytes(row) != identity["input_sha256"] or int(label) != identity["label"]:
                    raise ValueError("admitted native row identity differs")
            panels[name] = {"inputs": x, "labels": y}
    return panels


def sha_bytes(row):
    return hashlib.sha256(np.asarray(row, dtype="<f8").tobytes()).hexdigest()


def guard(root, protocol, *, runtime=True):
    if any(protocol.get(key) != value for key, value in FIXED.items()):
        raise ValueError("frozen native reference scope/config/bounds changed")
    for name, expected in protocol["sources"].items():
        if sha(root / "source" / name) != expected:
            raise ValueError("frozen reference source/data changed: " + name)
    if sha(Path(__file__)) != protocol["sources"]["native_information_controls.py"]:
        raise ValueError("executed reference helper differs")
    if protocol["runtime"] != {
        "numpy": np.__version__,
        "interpreter_sha256": sha(Path(sys.executable)),
    }:
        raise ValueError("reference runtime differs")
    if runtime and any(os.environ.get(key) != value for key, value in THREADS.items()):
        raise ValueError("reference thread settings differ")
    job = json.loads((root / "job-protocol.json").read_text())
    if job["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("reference reviewed job binding differs")
    # Static feature arrays are bound to the original24-only graph/checkpoints,
    # and contain only61 admitted rows in school/TRAIN/development order.
    static_job = json.loads((root / "source/static-job.json").read_text())
    static_result = json.loads((root / "source/static-result.json").read_text())
    original = json.loads((root / "source/original-protocol.json").read_text())
    if (
        not static_result["verified"]
        or static_result["heldout_decoded"]
        or static_result["job_protocol_sha256"] != sha(root / "source/static-job.json")
        or static_result["raw_arrays_sha256"] != sha(root / "source/direct-current-arrays.npz")
        or static_job["source_data_pins"]["protocol.json"]
        != sha(root / "source/original-protocol.json")
        or static_job["source_data_pins"]["summary.json"]
        != sha(root / "source/original-summary.json")
        or static_job["source_data_pins"]["source/fixture/school.npz"]
        != sha(root / "source/school.npz")
        or static_job["source_data_pins"]["source/fixture/provenance.json"]
        != sha(root / "source/provenance.json")
        or static_job["script_sha256"] != sha(root / "source/static-script.py")
        or static_job["source_data_pins"]["selected-24/initial.npz"]
        != sha(root / "source/original-initial.npz")
        or static_job["source_data_pins"]["selected-24/final.npz"]
        != sha(root / "source/original-final.npz")
        or original["fixture"]["school.npz"] != sha(root / "source/school.npz")
    ):
        raise ValueError("reference static/current input provenance differs")


def prepare(root, original, static, semantics, *, static_script=None):
    # The producer's source is an explicit, hash-bound input. A portable capsule
    # can keep it beside its arrays; the historical adjacent workspace remains
    # a fallback, never a required repository/private-evidence dependency.
    if static_script is None:
        local_script = static / "static-script.py"
        static_script = local_script if local_script.is_file() else (
            static.parent / "verification-sources/native_modality_static_audit.py"
        )
    static_job = json.loads((static / "job-protocol.json").read_text())
    if sha(static_script) != static_job["script_sha256"]:
        raise ValueError("reference static producer source differs from its pinned job")
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    for name in ("school.npz", "provenance.json"):
        shutil.copyfile(original / "source/fixture" / name, source / name)
    for original_name, name in (
        ("protocol.json", "original-protocol.json"),
        ("summary.json", "original-summary.json"),
        ("selected-24/initial.npz", "original-initial.npz"),
        ("selected-24/final.npz", "original-final.npz"),
    ):
        shutil.copyfile(original / original_name, source / name)
    for original_name, name in (
        ("job-protocol.json", "static-job.json"),
        ("result.json", "static-result.json"),
        ("direct-current-arrays.npz", "direct-current-arrays.npz"),
    ):
        shutil.copyfile(static / original_name, source / name)
    shutil.copyfile(static_script, source / "static-script.py")
    for name in ("eye.py", "collect.py"):
        shutil.copyfile(semantics / name, source / name)
    shutil.copyfile(Path(__file__), source / "native_information_controls.py")
    shutil.copyfile(original / "source/extract.py", source / "extract.py")
    protocol = {
        **FIXED,
        "sources": {path.name: sha(path) for path in sorted(source.iterdir())},
        "runtime": {"numpy": np.__version__, "interpreter_sha256": sha(Path(sys.executable))},
        "action_codec": (
            "product((-1,0,1),(-1,0,1),(0,1),(0,1)); components decoded from admitted labels"
        ),
        "structural_prior": (
            "factorized10-score readout knows the public action codec; opaque36-score control "
            "does not share target parameters across component combinations; "
            "neither is a Cadence answer"
        ),
        "fit_roles": {
            "school-only": "24 selected witnesses, weights1; TRAIN18/development19 scoring only",
            "expanded-train": (
                "42 witnesses; selected/other weights.5 for18 duplicated actions, others1; "
                "TRAIN18 now taught; effective weighted witness mass24"
            ),
        },
        "feature_conditioning": (
            "TRAIN-only weighted centering/global RMS; balanced_full divides each modality by "
            "its TRAIN RMS first; constant groups remain zero; "
            "no validation-label feature selection"
        ),
        "projection_features": (
            "static direct sensory current into32 association neurons; initial random projection "
            "versus original selected24-only384-lesson projection, no new graph solving"
        ),
        "gate_scope": (
            "reference diagnostic only; passing school-only reveals usable current-input/"
            "declared-prior information; expandedTRAIN cannot count as independent; "
            "failing these controls proves no impossibility"
        ),
        "semantics_scope": (
            "current collector/eye bytes are semantic references; "
            "historical parent did not pin those bytes"
        ),
        "scope": (
            "one frozen reference diagnostic, no Cadence/default/acceptance promotion "
            "or neural work"
        ),
    }
    write(root / "protocol.json", protocol)
    write(
        root / "job-protocol.json",
        {
            "schema": "cadence.native-information-reference-job.v1",
            "status": "prepared_not_executed",
            "protocol_sha256": sha(root / "protocol.json"),
            "helper_sha256": sha(source / "native_information_controls.py"),
            "interpreter": sys.executable,
            "bounds": {
                key: protocol[key]
                for key in ("fit_budget", "seconds_cap", "worker_seconds_cap", "output_cap_mib")
            },
            "reference_fit_calls": 0,
            "teacher_calls": 0,
            "neural_queries": 0,
            "equilibrium_solves": 0,
        },
    )
    guard(root, protocol, runtime=False)
    read_panels(root)
    write(
        root / "admission.json",
        {
            "prepared_not_executed": True,
            "protocol_sha256": sha(root / "protocol.json"),
            "reference_fit_calls": 0,
            "heldout_decoded": False,
            "teacher_calls": 0,
            "neural_queries": 0,
            "equilibrium_solves": 0,
        },
    )


def feature_arrays(root, panels, mode):
    if mode in GROUPS:
        lo, hi = GROUPS[mode]
        return {name: value["inputs"][:, lo:hi] for name, value in panels.items()}
    if mode in ("raw_full", "balanced_full"):
        return {name: value["inputs"] for name, value in panels.items()}
    with np.load(root / "source/direct-current-arrays.npz", allow_pickle=False) as archive:
        if mode == "sensory650":
            array = archive["initial_sensory_s"]
        else:
            checkpoint = "initial" if mode == "initial_projection32" else "final"
            array = sum(archive[checkpoint + "_current_" + group] for group in GROUPS)
    if array.shape != (61, 650 if mode == "sensory650" else 32) or not np.isfinite(array).all():
        raise ValueError("static projection feature census differs")
    result, position = {}, 0
    for name, count in PANELS.items():
        result[name] = array[position : position + count]
        position += count
    return result


def target_scores(labels, readout):
    if readout == "categorical":
        return np.eye(36)[labels]
    actions = np.array(tuple(itertools.product((-1, 0, 1), (-1, 0, 1), (0, 1), (0, 1))))
    values = actions[labels]
    return np.column_stack(
        [
            np.eye(count)[values[:, column] + (1 if column < 2 else 0)]
            for column, count in enumerate((3, 3, 2, 2))
        ]
    )


def labels_from_scores(scores, readout):
    if readout == "categorical":
        return scores.argmax(axis=1)
    h, v, w, j = (scores[:, lo:hi].argmax(axis=1) for lo, hi in ((0, 3), (3, 6), (6, 8), (8, 10)))
    return ((h * 3 + v) * 2 + w) * 2 + j


def component_correct(prediction, truth):
    actions = np.array(tuple(itertools.product((-1, 0, 1), (-1, 0, 1), (0, 1), (0, 1))))
    return dict(
        zip(
            ("horizontal", "vertical", "whip", "jump"),
            (actions[prediction] == actions[truth]).sum(axis=0).tolist(),
            strict=True,
        )
    )


def condition(train, arrays, weights, mode):
    mean = np.average(train, axis=0, weights=weights)
    scale = np.ones(train.shape[1])
    groups = GROUPS.values() if mode == "balanced_full" else ((0, train.shape[1]),)
    for lo, hi in groups:
        norm = np.sqrt(
            np.average(np.square(train[:, lo:hi] - mean[lo:hi]).sum(axis=1), weights=weights)
        )
        scale[lo:hi] = norm if norm > 1e-12 else 1.0
    z = (train - mean) / scale
    test = {name: (value - mean) / scale for name, value in arrays.items()}
    d = np.maximum(
        np.square(z).sum(axis=1)[:, None] + np.square(z).sum(axis=1)[None, :] - 2 * z @ z.T, 0.0
    )
    positive = d[d > 1e-12]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0

    def kernel(first, second):
        dot = first @ second.T
        distance = np.maximum(
            np.square(first).sum(axis=1)[:, None]
            + np.square(second).sum(axis=1)[None, :]
            - 2 * dot,
            0.0,
        )
        return 1.0 + 0.5 * dot + 0.5 * np.exp(-distance / bandwidth)

    return z, test, kernel, {"mean": mean, "scale": scale, "rbf_squared_bandwidth": bandwidth}


def worker(root):
    protocol = json.loads((root / "protocol.json").read_text())
    guard(root, protocol)
    if (root / "launch-started.json").read_text().strip() != sha(root / "protocol.json"):
        raise ValueError("reference worker needs its bounded launcher admission")
    with (root / "execution-started.json").open("x") as stream:
        stream.write(sha(root / "protocol.json") + "\n")
    panels = read_panels(root)
    began, fits = time.monotonic(), 0
    output = {
        "schema": "cadence.native-information-reference-result.v1",
        "protocol_sha256": sha(root / "protocol.json"),
        "complete": False,
        "models": [],
        "heldout_decoded": False,
        "whole_issue_passed": False,
    }
    rng = np.random.default_rng(protocol["random_seed"])
    output["uniform_random_control"] = {
        "seed": protocol["random_seed"],
        "distribution": "uniform over all36 existing executed action labels",
        "expected_row_accuracy": 1 / 36,
        "scope": "one measured seeded draw per admitted row, separate from expectation",
        "panels": {},
    }
    for name, panel in panels.items():
        prediction = rng.integers(0, 36, size=PANELS[name])
        output["uniform_random_control"]["panels"][name] = {
            "correct": int((prediction == panel["labels"]).sum()),
            "rows": PANELS[name],
            "predictions": prediction.tolist(),
            "component_correct": component_correct(prediction, panel["labels"]),
        }
    retained = {}
    try:
        for arm in ARMS:
            fit_labels = panels["school"]["labels"]
            weights = np.ones(24)
            if arm == "expanded-train":
                other = panels["independent_train"]["labels"]
                weights[np.isin(fit_labels, other)] = 0.5
                fit_labels = np.concatenate((fit_labels, other))
                weights = np.concatenate((weights, np.full(18, 0.5)))
            for mode in MODES:
                features = feature_arrays(root, panels, mode)
                train = features["school"]
                if arm == "expanded-train":
                    train = np.concatenate((train, features["independent_train"]))
                z, tests, kernel, metadata = condition(train, features, weights, mode)
                train_kernel = kernel(z, z)
                for readout in ("factorized", "categorical"):
                    if readout == "categorical" and mode not in FIXED["extra_categorical_controls"]:
                        continue
                    targets = target_scores(fit_labels, readout)
                    square_root = np.sqrt(weights)
                    rhs = targets * square_root[:, None]
                    grid = LAMBDAS if readout == "factorized" else [FIXED["categorical_lambda"]]
                    candidates = []
                    for penalty in grid:
                        if (
                            time.monotonic() - began >= FIXED["seconds_cap"]
                            or fits >= FIXED["fit_budget"]
                        ):
                            raise RuntimeError("reference fixed resource bound exhausted")
                        output["pending_fit"] = {
                            "arm": arm,
                            "feature": mode,
                            "readout": readout,
                            "lambda": penalty,
                            "completed_fit_calls_before": fits,
                        }
                        write(root / "result.json", output)
                        matrix = square_root[:, None] * train_kernel * square_root[
                            None, :
                        ] + penalty * np.eye(len(weights))
                        solved = np.linalg.solve(
                            matrix, np.column_stack((rhs, np.eye(len(weights))))
                        )
                        alpha = solved[:, : targets.shape[1]] * square_root[:, None]
                        inverse = solved[:, targets.shape[1] :]
                        residual = float(
                            np.max(
                                np.abs(
                                    matrix @ solved - np.column_stack((rhs, np.eye(len(weights))))
                                )
                            )
                        )
                        if not np.isfinite(solved).all() or residual > 1e-9:
                            raise RuntimeError(
                                "reference linear system failed independent residual bound"
                            )
                        diagonal = np.diag(
                            train_kernel @ (square_root[:, None] * inverse * square_root[None, :])
                        )
                        if np.any(1 - diagonal <= 1e-12):
                            raise RuntimeError("reference LOO denominator unresolved")
                        fitted = train_kernel @ alpha
                        loo = targets - (targets - fitted) / (1 - diagonal[:, None])
                        prediction = labels_from_scores(loo, readout)
                        candidates.append(
                            {
                                "lambda": penalty,
                                "loo_correct": int((prediction == fit_labels).sum()),
                                "loo_weighted_correct": float(
                                    weights[prediction == fit_labels].sum()
                                ),
                                "loo_predictions": prediction.tolist(),
                                "linear_system_residual": residual,
                                "alpha": alpha,
                            }
                        )
                        fits += 1
                        output["reference_fit_calls"] = fits
                        output.pop("pending_fit")
                    selected = max(
                        candidates, key=lambda row: (row["loo_weighted_correct"], -row["lambda"])
                    )
                    key = arm.replace("-", "_") + "_" + mode + "_" + readout
                    retained[key + "_alpha"] = selected["alpha"]
                    retained[key + "_condition_mean"] = metadata["mean"]
                    retained[key + "_condition_scale"] = metadata["scale"]
                    retained[key + "_fit_labels"] = fit_labels
                    retained[key + "_fit_weights"] = weights
                    result = {
                        "arm": arm,
                        "feature": mode,
                        "readout": readout,
                        "fit_rows": len(fit_labels),
                        "effective_weighted_witness_mass": float(weights.sum()),
                        "known_action_codec_prior": readout == "factorized",
                        "selected_lambda": selected["lambda"],
                        "rbf_squared_bandwidth": metadata["rbf_squared_bandwidth"],
                        "train_conditioning_only": True,
                        "candidates": [
                            {name: value for name, value in candidate.items() if name != "alpha"}
                            for candidate in candidates
                        ],
                        "panels": {},
                    }
                    for name, features_test in tests.items():
                        scores = kernel(features_test, z) @ selected["alpha"]
                        prediction = labels_from_scores(scores, readout)
                        retained[key + "_" + name + "_scores"] = scores
                        result["panels"][name] = {
                            "correct": int((prediction == panels[name]["labels"]).sum()),
                            "rows": PANELS[name],
                            "predictions": prediction.tolist(),
                            "component_correct": component_correct(
                                prediction, panels[name]["labels"]
                            ),
                            "taught": name == "school"
                            or (arm == "expanded-train" and name == "independent_train"),
                        }
                    result["reference_information_gate_passed"] = bool(
                        arm == "school-only"
                        and all(
                            result["panels"][name]["correct"] >= gate
                            for name, gate in protocol["information_gates"].items()
                        )
                    )
                    output["models"].append(result)
                    write(root / "result.json", output)
        output["complete"] = True
        primary = next(
            row
            for row in output["models"]
            if all(row[key] == value for key, value in protocol["primary"].items())
        )
        output["primary"] = {
            **protocol["primary"],
            "panels": primary["panels"],
            "reference_information_gate_passed": primary["reference_information_gate_passed"],
            "whole_issue_passed": False,
        }
    except BaseException as error:
        output["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        np.savez_compressed(root / "reference-arrays.npz", **retained)
        output.update(
            seconds=time.monotonic() - began,
            reference_fit_calls=fits,
            teacher_calls=0,
            neural_queries=0,
            equilibrium_solves=0,
            reference_arrays_sha256=sha(root / "reference-arrays.npz"),
        )
        output["scope"] = protocol["scope"]
        if output["seconds"] > protocol["seconds_cap"]:
            output["complete"] = False
            output["resource_failure"] = "final reference work exceeded the60-second analysis cap"
        write(root / "result.json", output)
    return output


def launch(root):
    protocol = json.loads((root / "protocol.json").read_text())
    guard(root, protocol, runtime=False)
    if any((root / name).exists() for name in ("launch-started.json", "execution-started.json")):
        raise ValueError("reference diagnostic already attempted")
    with (root / "launch-started.json").open("x") as stream:
        stream.write(sha(root / "protocol.json") + "\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.update(THREADS)
    command = [
        sys.executable,
        str(root / "source/native_information_controls.py"),
        "--worker",
        "--out",
        str(root),
    ]
    execution = {
        "schema": "cadence.native-information-reference-execution.v1",
        "protocol_sha256": sha(root / "protocol.json"),
        "job_protocol_sha256": sha(root / "job-protocol.json"),
        "command": command,
        "complete": False,
    }
    began = time.monotonic()
    try:
        with (root / "worker.log").open("w") as stream:
            completed = subprocess.run(
                command,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=FIXED["worker_seconds_cap"],
                check=False,
            )
        execution["worker_returncode"] = completed.returncode
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, completed.args)
        result = json.loads((root / "result.json").read_text())
        if not result["complete"] or result["reference_fit_calls"] != FIXED["fit_budget"]:
            raise RuntimeError("reference final scheduled census incomplete")
        if result["seconds"] > protocol["seconds_cap"]:
            raise RuntimeError("reference final work exceeded the declared analysis cap")
        if (
            sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            > FIXED["output_cap_mib"] * 1024**2 - FIXED["final_receipt_reserve_bytes"]
        ):
            raise RuntimeError("reference retained output exceeds declared cap")
        execution["complete"] = True
        return result
    except BaseException as error:
        execution["failure"] = {"type": type(error).__name__, "message": str(error)}
        execution["unwitnessed_resource_tail"] = (
            "a pending reference fit may have started without a completion receipt; "
            "charge the whole elapsed worker attempt"
        )
        raise
    finally:
        execution["seconds"] = time.monotonic() - began
        result_path = root / "result.json"
        if result_path.exists():
            execution["retained_result_sha256"] = sha(result_path)
            retained = json.loads(result_path.read_text())
            execution["completed_reference_fit_calls"] = retained.get("reference_fit_calls", 0)
            execution["pending_fit"] = retained.get("pending_fit")
        execution["retained_bytes_before_execution_receipt"] = sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        )
        write(root / "execution.json", execution)
        final_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if final_bytes > protocol["output_cap_mib"] * 1024**2:
            execution["complete"] = False
            execution["resource_failure"] = "final receipt exceeded the declared retained-byte cap"
            execution["retained_bytes"] = final_bytes
            write(root / "execution.json", execution)
            raise RuntimeError(execution["resource_failure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument(
        "--static-script", type=Path,
        help="hash-bound static feature producer; defaults to the capsule's static-script.py",
    )
    parser.add_argument("--semantics", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker:
        worker(root)
    elif args.execute_reviewed:
        result = launch(root)
        print(
            json.dumps(
                {
                    "complete": result["complete"],
                    "reference_fit_calls": result["reference_fit_calls"],
                    "seconds": result["seconds"],
                    "passing_controls": [
                        [row["feature"], row["readout"]]
                        for row in result["models"]
                        if row["reference_information_gate_passed"]
                    ],
                }
            )
        )
    elif args.prepare_only and args.original and args.static and args.semantics:
        prepare(
            root, args.original.resolve(), args.static.resolve(), args.semantics.resolve(),
            static_script=args.static_script.resolve() if args.static_script else None,
        )
        print(
            json.dumps(
                {
                    "prepared": str(root),
                    "protocol_sha256": sha(root / "protocol.json"),
                    "reference_fit_calls": 0,
                    "teacher_calls": 0,
                    "neural_queries": 0,
                }
            )
        )
    else:
        parser.error(
            "prepare requires original/static/semantics; execution requires separate review"
        )


if __name__ == "__main__":
    main()
