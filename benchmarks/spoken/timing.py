"""Seconds per step of 500 ms of audio and per correction, one stream on one core (cadence#303).

A step is the heard window's ``act`` (a brain) or one more step of the phrase (a baseline,
which reads the phrase's steps so far again, as a causal model over the steps does without a
cache). A correction is one phrase taught: the brain's lesson before each act, the baseline's
one Adam step on the phrase. Phrases come from ``--data`` (``base_test``), or from synthetic
speech where none is given.

    PYTHONPATH=src python benchmarks/spoken/timing.py --data /data/phrases --learners fields \\
        transformer_ctc --out results/timing.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    key = f"spoken_{name}"  # a name of this chamber's own: other benchmarks have their modules
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load("chamber")
ph = _load("phrases")
sp = _load("spoken")


def measure(
    name: str, data: str, phrases: int, founder: int = 0, match: str = "fields"
) -> dict[str, Any]:
    test = ph.Phrases(Path(data) / "base_test.npz")
    hearing = ch.Hearing.measured(ph.Phrases(Path(data) / "base_train.npz"))
    learner = sp.learner_of(name, founder, match, None)
    rows = np.arange(min(phrases, len(test)))
    steps, corrections = [], []
    for r in rows:
        x = ch.stream(test, np.array([r]), hearing)
        target = test.target[[r], : x.shape[1]]
        started = time.perf_counter()
        learner.hear(x, target, [test.reference(int(r))])
        corrections.append(time.perf_counter() - started)
        if learner.kind == "brain":
            learner.brain.reset()
            for k in range(x.shape[1]):
                started = time.perf_counter()
                learner.brain.act(x[:, k], greedy=True)
                steps.append(time.perf_counter() - started)
        else:
            for k in range(1, x.shape[1] + 1):
                started = time.perf_counter()
                learner.hear(x[:, :k])
                steps.append(time.perf_counter() - started)
    skip = len(steps) // 10  # the first moments compile and warm the caches
    return {
        "learner": name,
        "parameters": learner.parameters(),
        "phrases": len(rows),
        "step_median_ms": 1000 * float(np.median(steps[skip:])),
        "step_p95_ms": 1000 * float(np.percentile(steps[skip:], 95)),
        "correction_median_s": float(np.median(corrections[1:])),
        "real_time_factor": 0.49 / float(np.median(steps[skip:])),
    }


def main() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=None)
    parser.add_argument("--learners", nargs="+", default=["fields", "transformer_ctc"])
    parser.add_argument("--phrases", type=int, default=24)
    parser.add_argument("--match", default="fields", help="the brain arm a baseline matches")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    with tempfile.TemporaryDirectory() as scratch:
        data = args.data
        if data is None:
            ph.build_synthetic(scratch)
            data = scratch
        with open(args.out, "a") as handle:
            for name in args.learners:
                handle.write(json.dumps(measure(name, data, args.phrases, match=args.match)) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
