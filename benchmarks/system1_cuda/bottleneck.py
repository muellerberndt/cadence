"""Profile a declared coupled graph; this is a bottleneck control, not acquisition."""

import argparse
import cProfile
import hashlib
import io
import json
import os
import pstats
import subprocess
import sys
import time
import traceback
from pathlib import Path

from run import digest, memory, sources, write

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", choices=["cpu", "cuda64", "cuda32"], required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
root = args.source_root.resolve()
before = sources(root)
sys.path.insert(0, str(root / "src"))
protocol = {
    "kind": "bottleneck profile, no acquisition claim",
    "seed": 7,
    "inputs": 8,
    "actions": 2,
    "modules": [64, 32],
    "observers": [8],
    "batch": 16,
    "warmup_calls": 4,
    "profiled_calls": 8,
    "backend": args.backend,
    "harness_sha256_lf": digest(Path(__file__)),
    "protocol_sha256_lf": digest(args.protocol),
    "source_sha256_lf": before,
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
}
write(args.out / "manifest.json", protocol)
try:
    start = time.perf_counter()
    import numpy as np
    import torch

    import cadence as cd

    assert Path(cd.__file__).resolve().is_relative_to(root)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    imports = time.perf_counter() - start
    cuda = args.backend.startswith("cuda")
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Actual CUDA required")
        torch.cuda.reset_peak_memory_stats()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    start = time.perf_counter()
    brain = cd.Brain.compose(
        8,
        2,
        modules=(64, 32),
        observers=(8,),
        seed=7,
        backend="torch" if cuda else "cpu",
        device="cuda:0" if cuda else None,
    )
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
    sync()
    construction = time.perf_counter() - start
    rng = np.random.default_rng(700)
    drives = []
    targets = []
    for _tick in range(13):
        labels = np.arange(16) % 2
        x = np.zeros((16, 8))
        x[labels == 0, :4] = 1
        x[labels == 1, 4:] = 1
        x = np.clip(x + 0.3 * rng.standard_normal(x.shape), 0, 1)
        drives.append(x)
        targets.append(labels)
    action = None
    rows = []

    def step(tick):
        global action
        feedback = (
            {}
            if action is None
            else {
                "reward": (action == targets[tick - 1]).astype(float),
                "done": np.zeros(16, dtype=bool),
            }
        )
        sync()
        start = time.perf_counter()
        action = brain.step(drives[tick], teacher=targets[tick], **feedback)
        sync()
        rows.append(
            {
                "tick": tick,
                "seconds": time.perf_counter() - start,
                "correct": int((action == targets[tick]).sum()),
                "settlement": dict(brain.last_settlement),
                "learning": brain.last_learning,
            }
        )

    for tick in range(4):
        step(tick)
    profile = cProfile.Profile()
    profile.enable()
    for tick in range(4, 12):
        step(tick)
    profile.disable()
    text = io.StringIO()
    pstats.Stats(profile, stream=text).sort_stats("cumulative").print_stats(45)
    (args.out / "python-profile.txt").write_text(text.getvalue(), encoding="utf-8")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        step(12)
    (args.out / "operators.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=35), encoding="utf-8"
    )
    operators = [
        {
            "name": event.key,
            "count": event.count,
            "self_cpu_us": event.self_cpu_time_total,
            "self_device_us": event.self_device_time_total,
            "self_cpu_memory_bytes": event.self_cpu_memory_usage,
            "self_device_memory_bytes": event.self_device_memory_usage,
        }
        for event in prof.key_averages()
    ]
    prof.export_chrome_trace(str(args.out / "trace.json"))
    result = {
        "passed": before == sources(root),
        "source_unchanged": before == sources(root),
        "imports_seconds": imports,
        "construction_seconds": construction,
        "rows": rows,
        "gpu_peak_allocated": torch.cuda.max_memory_allocated() if cuda else None,
        "gpu_peak_reserved": torch.cuda.max_memory_reserved() if cuda else None,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cadence": cd.__version__,
        "host_memory": memory(),
        "operators": operators,
        "trace_sha256": hashlib.sha256((args.out / "trace.json").read_bytes()).hexdigest(),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "backend": args.backend,
                "construction_seconds": construction,
                "warm_seconds": [r["seconds"] for r in rows[4:12]],
            }
        )
    )
    print(text.getvalue())
    if not result["passed"]:
        raise RuntimeError("Sources changed during profile")
except BaseException:
    (args.out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    raise
