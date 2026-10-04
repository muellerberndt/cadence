"""Fixed-work transport controls across graph and batch sizes, separate from acquisition."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run import digest, memory, sources, write


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda64", "cuda32"], required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = str(protocol["threads"])
    before = sources(root)
    write(
        output / "manifest.json",
        {
            "source_sha256_lf": before,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "harness_sha256_lf": digest(Path(__file__)),
            "protocol": protocol,
            "protocol_sha256_lf": digest(args.protocol),
            "backend": args.backend,
        },
    )
    sys.path.insert(0, str(root / "src"))
    result = {"passed": False, "cases": []}
    try:
        import numpy as np

        import cadence as cd

        assert Path(cd.__file__).resolve().is_relative_to(root)
        cuda = args.backend.startswith("cuda")
        if cuda:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA hardware required")
            torch.set_num_threads(protocol["threads"])
            torch.backends.cuda.matmul.allow_tf32 = False

        def sync():
            if cuda:
                torch.cuda.synchronize()

        config = protocol["transport_controls"]
        for modules in config["layouts"]:
            for batch in config["batches"]:
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                body = cd.Brain.compose(8, 2, modules=modules, observers=(8,), seed=11)
                original = body.brain
                graph = cd.NeuralGraph(
                    original.connectome,
                    original.neuron_model,
                    backend="torch" if cuda else "cpu",
                    device="cuda:0" if cuda else None,
                    precision="float32" if args.backend == "cuda32" else None,
                    efficacy=original.efficacy,
                    bias=original.bias,
                    log_gain=original.log_gain,
                )
                if cuda:
                    torch.cuda.synchronize()
                construction = time.perf_counter() - tick
                if cuda:
                    assert graph._torch.device == torch.device("cuda:0")
                    assert graph._torch.dtype == (
                        torch.float32 if args.backend == "cuda32" else torch.float64
                    )
                rng = np.random.default_rng(904)
                drives = [
                    body.stimulus(rng.uniform(0, 1, (batch, 8)), memory=False)
                    for _ in range(config["warmup"] + config["repeats"])
                ]
                times, residuals, output_values = [], [], []
                cold_seconds = None
                for i, drive in enumerate(drives):
                    sync()
                    tick = time.perf_counter()
                    phase = graph.settle_batch(drive, steps=config["steps"])
                    sync()
                    duration = time.perf_counter() - tick
                    if i == 0:
                        cold_seconds = duration
                    if i >= config["warmup"]:
                        times.append(duration)
                        residuals.append(graph.residual(drive, phase, on_device=False).tolist())
                        output_values.append(phase.v.copy())
                    assert phase.steps == config["steps"]
                row = {
                    "device": str(graph._torch.device) if cuda else "cpu",
                    "state_dtype": str(graph._torch.dtype) if cuda else "float64",
                    "modules": modules,
                    "batch": batch,
                    "neurons": graph.connectome.n,
                    "edges": graph.connectome.synapses,
                    "layout_entries": graph.layout.size,
                    "steps": config["steps"],
                    "construction_seconds": construction,
                    "cold_solve_seconds": cold_seconds,
                    "samples_seconds": times,
                    "p50_seconds": float(np.percentile(times, 50)),
                    "p95_seconds": float(np.percentile(times, 95)),
                    "p99_seconds": float(np.percentile(times, 99)),
                    "host_memory": memory(),
                    "residuals": residuals,
                    "gpu_memory": None
                    if not cuda
                    else {
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    },
                }
                arrays = output / f"states-{'-'.join(map(str, modules))}-batch{batch}.npz"
                np.savez_compressed(arrays, potentials=np.asarray(output_values))
                row["states_sha256"] = hashlib.sha256(arrays.read_bytes()).hexdigest()
                row["states_file"] = arrays.name
                result["cases"].append(row)
                write(output / "progress.json", result)
                del phase, graph, body, original, output_values
        result["passed"] = True
        result["packages"] = {
            name: importlib.metadata.version(name) for name in ("numpy", "numba", "scipy", "torch")
        }
    except Exception:
        result["failure"] = traceback.format_exc()
    result["source_unchanged"] = before == sources(root)
    result["passed"] = result["passed"] and result["source_unchanged"]
    result["worker_seconds"] = time.perf_counter() - started
    write(output / "result.json", result)
    print(json.dumps({key: result.get(key) for key in ("passed", "worker_seconds", "failure")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
