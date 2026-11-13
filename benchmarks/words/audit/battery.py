"""The whole audit on a range of founders, one process a script and founder (cadence#326).

    python battery.py <out> [first founder] [founders] [jobs] [short]
    python read.py <out>; python genes.py read <out>

Writes ``<script>_<founder>.json`` for ``ablations``, ``adversarial``, ``lookup``, ``path`` and
``genes``, and ``heldout.json``; a file present is kept.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common

HERE = Path(__file__).resolve().parent
SCRIPTS = ("ablations", "adversarial", "lookup", "path", "genes")


def main(argv: list[str]) -> None:
    out = Path(argv[1])
    first = int(argv[2]) if len(argv) > 2 else common.FOUNDERS[0]
    count = int(argv[3]) if len(argv) > 3 else len(common.FOUNDERS)
    workers = int(argv[4]) if len(argv) > 4 else 8
    short = ["short"] if len(argv) > 5 and argv[5] == "short" else []
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
    python = sys.executable
    subprocess.run([python, str(HERE / "heldout.py"), str(out), str(first), str(count)],
                   cwd=HERE, env=env, check=True, capture_output=True)
    jobs = [(script, seed) for seed in range(first, first + count) for script in SCRIPTS]

    def one(job: tuple[str, int]) -> str:
        script, seed = job
        path = out / f"{script}_{seed}.json"
        if path.exists():
            return f"kept {path.name}"
        done = subprocess.run([python, str(HERE / f"{script}.py"), str(seed), str(out), *short],
                              cwd=HERE, env=env, capture_output=True, text=True)
        if done.returncode:
            (out / f"{script}_{seed}.err").write_text(done.stderr)
            return f"failed {path.name}"
        return f"wrote {path.name}"

    with ThreadPoolExecutor(workers) as pool:
        for line in pool.map(one, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main(sys.argv)
