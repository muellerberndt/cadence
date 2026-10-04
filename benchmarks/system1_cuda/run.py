"""One fresh process per source/backend/seed; retain failed runs and complete costs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sources(root):
    return {
        str(p.relative_to(root)).replace("\\", "/"): digest(p)
        for p in sorted((root / "src/cadence").rglob("*.py"))
    }


def memory():
    """OS process-lifetime peak RSS, including imports; no Python-only heap claim."""
    if sys.platform == "win32":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                    "PrivateUsage",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "peak_rss_bytes": counters.PeakWorkingSetSize,
            "rss_bytes": counters.WorkingSetSize,
            "private_bytes": counters.PrivateUsage,
        }
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": int(peak * (1 if sys.platform == "darwin" else 1024))}


def main():
    entered = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda64", "cuda32"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = str(protocol["threads"])
    os.environ["PYTHONUTF8"] = "1"
    source_before = sources(root)
    manifest = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
        "source_root": str(root),
        "source_sha256_lf": source_before,
        "harness_sha256_lf": digest(Path(__file__)),
        "protocol_sha256_lf": digest(args.protocol),
        "backend": args.backend,
        "seed": args.seed,
        "protocol": protocol,
        "normalization": "CRLF bytes replaced with LF; all other bytes preserved",
    }
    write(output / "manifest.json", manifest)
    # Each subprocess imports exactly one declared source tree, including the baseline.
    sys.path.insert(0, str(root / "src"))
    result = {"passed": False, "measurements": [], "warnings": []}
    event_file = (output / "events.jsonl").open("x", encoding="utf-8")
    try:
        from dataclasses import replace

        import numpy as np

        import cadence as cd

        assert Path(cd.__file__).resolve().is_relative_to(root)
        cuda = args.backend.startswith("cuda")
        torch = None
        if cuda:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("Actual CUDA required; no fallback or skipped receipt")
            torch.set_num_threads(protocol["threads"])
            # No TF32 precision substitution for the declared float32 equations.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def sync():
            if cuda:
                torch.cuda.synchronize()

        def plain(value):
            if isinstance(value, np.ndarray):
                return plain(value.tolist())
            if isinstance(value, np.generic):
                return plain(value.item())
            if isinstance(value, dict):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        work = {
            "settling_calls": 0,
            "failed_settling_calls": 0,
            "sweeps": 0,
            "row_sweeps": 0,
            "residual_calls": 0,
        }
        settle, residual = cd.NeuralGraph.settle_batch, cd.NeuralGraph.residual

        def counted_settle(graph, *a, **kw):
            work["settling_calls"] += 1
            try:
                phase = settle(graph, *a, **kw)
            except Exception:
                work["failed_settling_calls"] += 1
                raise
            work["sweeps"] += phase.steps
            work["row_sweeps"] += phase.steps * len(np.atleast_2d(a[0]))
            return phase

        def counted_residual(graph, *a, **kw):
            work["residual_calls"] += 1
            return residual(graph, *a, **kw)

        cd.NeuralGraph.settle_batch, cd.NeuralGraph.residual = counted_settle, counted_residual

        def record(name, operation, decisions=0):
            before = work.copy()
            sync()
            started = time.perf_counter()
            failure = None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    value = operation()
                except Exception:
                    failure = traceback.format_exc()
                    raise
                finally:
                    sync()
                    row = {
                        "stage": name,
                        "seconds": time.perf_counter() - started,
                        "decisions": decisions,
                        "work": {k: work[k] - before[k] for k in work},
                        "failure": failure,
                    }
                    result["warnings"].extend(str(w.message) for w in caught)
                    result["measurements"].append(row)
                    event_file.write(json.dumps(row, allow_nan=False) + "\n")
                    event_file.flush()
            return value

        result["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "cadence": cd.__version__,
            "numpy": np.__version__,
            "threads": protocol["threads"],
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "numba", "scipy", "torch")
            },
        }
        if cuda:
            props = torch.cuda.get_device_properties(0)
            result["environment"].update(
                {
                    "device": props.name,
                    "device_index": 0,
                    "vram_bytes": props.total_memory,
                    "torch_cuda_runtime": torch.version.cuda,
                    "tf32": torch.backends.cuda.matmul.allow_tf32,
                    "driver": subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,driver_version,memory.total,power.draw,temperature.gpu",
                            "--format=csv",
                        ],
                        text=True,
                    ).strip(),
                }
            )
        result["imports_and_device_init_seconds"] = time.perf_counter() - entered
        result["memory_after_imports"] = memory()

        def create():
            brain = cd.Brain.compose(
                protocol["inputs"],
                protocol["actions"],
                modules=protocol["modules"],
                observers=protocol["observers"],
                seed=args.seed,
                backend="torch" if cuda else "cpu",
                device="cuda:0" if cuda else None,
            )
            brain.learner.config = replace(brain.learner.config, **protocol["learning"])
            brain.basal_ganglia.config = replace(brain.basal_ganglia.config, **protocol["reward"])
            if args.backend == "cuda32":
                graph = brain.brain
                brain.learner.brain = cd.NeuralGraph(
                    graph.connectome,
                    graph.neuron_model,
                    backend="torch",
                    device="cuda:0",
                    precision="float32",
                    efficacy=graph.efficacy,
                    log_gain=graph.log_gain,
                    bias=graph.bias,
                    layout=graph.layout,
                )
            return brain

        brain = record("construction", create)
        result["execution"] = {
            "backend": brain.brain.backend,
            "device": "cpu",
            "state_dtype": "float64",
            "parameter_dtype": "float64",
        }
        if cuda:
            kernel = brain.brain._torch
            assert kernel.device == torch.device("cuda:0")
            assert kernel.dtype == (torch.float32 if args.backend == "cuda32" else torch.float64)
            result["execution"].update(
                {
                    "device": str(kernel.device),
                    "state_dtype": str(kernel.dtype),
                    "parameter_dtype": str(kernel.param_dtype),
                }
            )
        result["graph"] = {
            "neurons": brain.connectome.n,
            "edges": brain.connectome.synapses,
            "learning": brain.learner.config.to_dict(),
            "reward": brain.basal_ganglia.config.to_dict(),
        }

        def samples(seed, count=protocol["training_rows"]):
            rng = np.random.default_rng(seed)
            labels = np.repeat([0, 1], count // 2)
            x = np.zeros((count, 8))
            x[labels == 0, :4] = 1
            x[labels == 1, 4:] = 1
            return np.clip(x + protocol["noise"] * rng.standard_normal(x.shape), 0, 1), labels

        training, labels = samples(protocol["training_data_seed"])
        held, expected = samples(protocol["held_out_data_seed"])
        result["founder_accuracy"] = record(
            "cold_free_query", lambda: brain.accuracy(held, expected)
        )
        result["fit_accuracy"] = record(
            "bootstrap",
            lambda: brain.fit(
                training, labels, epochs=protocol["epochs"], batch=protocol["training_batch"]
            ),
        )
        result["bootstrap_accuracy"] = record(
            "bootstrap_held_out", lambda: brain.accuracy(held, expected)
        )
        brain.save(output / "acquired.npz")
        rng = np.random.default_rng(700 + args.seed)
        last_action = last_labels = None
        live_rows = []
        for stage in protocol["stages"]:
            for tick in range(stage["calls"]):

                def decision(stage=stage, tick=tick):
                    nonlocal last_action, last_labels
                    y = rng.integers(0, 2, protocol["live_batch"])
                    x = np.zeros((len(y), 8))
                    x[y == 0, :4] = stage["amplitude"]
                    x[y == 1, 4:] = stage["amplitude"]
                    x = np.clip(x + protocol["noise"] * rng.standard_normal(x.shape), 0, 1)
                    feedback = (
                        {}
                        if last_action is None
                        else {
                            "reward": (last_action == last_labels).astype(float),
                            "done": np.zeros(len(y), dtype=bool),
                        }
                    )
                    action = brain.step(x, teacher=y if stage["teach"] else None, **feedback)
                    row = {
                        "stage": stage["name"],
                        "tick": tick,
                        "correct": int((action == y).sum()),
                        "labels": y.tolist(),
                        "actions": action.tolist(),
                        "settlement": dict(brain.last_settlement),
                        "learning": dict(brain.last_learning),
                    }
                    live_rows.append(plain(row))
                    last_action, last_labels = action, y
                    return x

                current = record(stage["name"], decision, protocol["live_batch"])
        result["live"] = live_rows
        saved = record("checkpoint_save", lambda: brain.save(output / "pending.npz"))
        loaded = record(
            "checkpoint_restore",
            lambda: cd.Brain.load(
                saved,
                backend="torch" if cuda else "cpu",
                device="cuda:0" if cuda else None,
                precision="float32" if args.backend == "cuda32" else None,
            ),
        )
        feedback = {
            "reward": (last_action == last_labels).astype(float),
            "done": np.zeros(protocol["live_batch"], dtype=bool),
        }
        first = record(
            "saved_continuation",
            lambda: brain.step(current * 0.9, **feedback),
            protocol["live_batch"],
        )
        second = record(
            "restored_continuation",
            lambda: loaded.step(current * 0.9, **feedback),
            protocol["live_batch"],
        )
        result["saved_actions_equal"] = bool(np.array_equal(first, second))
        result["retained_accuracy"] = record(
            "retention_query", lambda: brain.accuracy(held, expected)
        )
        result["memory"] = memory()
        result["gpu_memory"] = (
            None
            if not cuda
            else {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
        )
        result["counts"] = {
            "graph_updates": brain.learner.updates,
            "actor_updates": brain.basal_ganglia.updates,
            "memory_writes": brain.hippocampus.writes,
            "work": work,
        }
        groups = {}
        for stage in protocol["stages"]:
            rows = [r for r in result["measurements"] if r["stage"] == stage["name"]]
            values = np.array([r["seconds"] for r in rows])
            groups[stage["name"]] = {
                "calls": len(rows),
                "seconds": float(values.sum()),
                "p50_seconds": float(np.percentile(values, 50)),
                "p95_seconds": float(np.percentile(values, 95)),
                "p99_seconds": float(np.percentile(values, 99)),
                "max_seconds": float(values.max()),
                "decisions_per_second": sum(r["decisions"] for r in rows) / float(values.sum()),
                "deadline_misses": int((values > protocol["latency"]["deadline_seconds"]).sum()),
            }
        result["latencies"] = groups
        result["passed"] = bool(
            result["bootstrap_accuracy"] >= protocol["gate"]["bootstrap_held_out_accuracy"]
            and result["bootstrap_accuracy"] > result["founder_accuracy"]
            and result["retained_accuracy"] >= protocol["gate"]["retained_held_out_accuracy"]
            and result["saved_actions_equal"]
            and all(row["settlement"]["qualified"] for row in live_rows)
        )
    except Exception:
        result["failure"] = traceback.format_exc()
    finally:
        result["memory"] = memory()
        if "torch" in locals() and torch is not None and torch.cuda.is_initialized():
            result["gpu_memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
        result["source_unchanged"] = sources(root) == source_before
        result["passed"] = result["passed"] and result["source_unchanged"]
        result["worker_seconds"] = time.perf_counter() - entered
        event_file.close()
        write(output / "result.json", result)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "passed",
                    "bootstrap_accuracy",
                    "retained_accuracy",
                    "worker_seconds",
                    "failure",
                )
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
