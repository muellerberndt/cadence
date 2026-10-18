"""Run the declared local-work campaign and write one JSON per case plus a summary.

    python benchmarks/local_work/run.py --campaign skills --founders 0 1 2 --out /path/out
    python benchmarks/local_work/run.py --campaign widths --founders 0 1 2 --out /path/out

The output directory is the caller's; results do not belong in this repository. The BLAS
and OpenMP thread counts are pinned to one before NumPy is imported, so the host seconds
of both arms are single-threaded. Read LOCAL_WORK_PROTOCOL.md first: the campaign grids,
gates and budgets are declared there and frozen before a run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_work as lw  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--campaign", choices=("skills", "widths"), required=True)
    parser.add_argument("--founders", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=["dense", "sparse"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-minutes", type=float, default=30.0,
                        help="host budget per case; a case over it is flagged in its record")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="case names to run, if given")
    parser.add_argument("--local-margin", type=float, default=1.0,
                        help="amendment 1: the worklist stops at tolerance / margin")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore", RuntimeWarning)  # the library's budget warnings, retained
    out = args.out
    (out / "cases").mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(json.dumps(lw.environment(), indent=1))
    summaries = []
    specs = lw.case_specs(args.campaign, args.founders, args.arms, args.local_margin)
    if args.only:
        specs = [spec for spec in specs if spec.name in set(args.only)]
    for spec in specs:
        print(f"case {spec.name}", flush=True)
        started = time.perf_counter()
        case = lw.run_case(spec, verbose=args.verbose)
        seconds = time.perf_counter() - started
        case["over_budget"] = seconds > 60.0 * args.case_minutes
        case["environment"] = lw.environment()
        (out / "cases" / f"{spec.name}.json").write_text(json.dumps(case))
        summary = lw.summarize(case)
        summary["over_budget"] = case["over_budget"]
        summaries.append(summary)
        gates = case["gates"]
        print(
            f"  {seconds:7.1f}s acquired {case['acquisition']['at_gate']} "
            f"lessons {case['acquisition']['lessons']} same {gates['same_action']} "
            f"qualified {gates['local_qualified']} state {gates['state_within_gate']} "
            f"max dist {gates['max_distance_activation']:.3g}",
            flush=True,
        )
        suffix = "" if args.local_margin == 1.0 else f"-m{args.local_margin:g}"
        (out / f"summary-{args.campaign}{suffix}.json").write_text(
            json.dumps(summaries, indent=1)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
