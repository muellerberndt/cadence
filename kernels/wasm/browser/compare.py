"""Two engines' lives side by side (``lives.py`` output): how many moments chose the same action,
the first that did not, and how far the arrays a life ends on lie apart.

    python kernels/wasm/browser/compare.py wasm.txt numpy.txt [--bitwise LABEL ...]

With ``--bitwise``, the named lives must end on the same bits and every life must choose the
same actions; the exit status says whether they do.
"""
from __future__ import annotations

import base64
import json
import sys

import numpy as np


def read(path: str) -> dict[str, dict]:
    lives = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("LIFE "):
            record = json.loads(line[5:])
            lives[record["label"]] = record
    return lives


def arrays(record: dict) -> dict[str, np.ndarray]:
    return {k: np.frombuffer(base64.b64decode(record[k]), dtype=np.float64)
            for k in ("efficacy", "bias")}


def main() -> int:
    args = sys.argv[1:]
    bitwise = set()
    if "--bitwise" in args:
        at = args.index("--bitwise")
        bitwise = set(args[at + 1:])
        args = args[:at]
    a, b = read(args[0]), read(args[1])
    ok = True
    print(f"{'life':36s} {'moments':>7s} {'same actions':>14s} {'first apart':>11s} "
          f"{'arrays':>8s} {'max |diff|':>10s} {'ms a':>7s} {'ms b':>7s}")
    for label, x in a.items():
        if label not in b:
            continue
        y = b[label]
        same = sum(int(p == q) for p, q in zip(x["actions"], y["actions"], strict=True))
        first = next((i for i, (p, q) in enumerate(zip(x["actions"], y["actions"]))
                      if p != q), None)
        ax, ay = arrays(x), arrays(y)
        bits = all(np.array_equal(ax[k].view(np.uint64), ay[k].view(np.uint64)) for k in ax)
        gap = max(float(np.abs(ax[k] - ay[k]).max()) for k in ax)
        moments = len(x["actions"])
        print(f"{label:36s} {moments:7d} {same:6d} ({100 * same / moments:5.1f}%) "
              f"{'-' if first is None else first:>11} {'bitwise' if bits else 'apart':>8s} "
              f"{gap:10.2e} {x['ms_per_moment']:7.2f} {y['ms_per_moment']:7.2f}")
        ok = ok and same == moments and (label not in bitwise or bits)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
