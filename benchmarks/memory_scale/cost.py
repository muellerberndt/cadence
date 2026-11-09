"""What a large memory costs (cadence#288, instrument c): read and write time and bytes against
the records held, on one CPU thread and on a device.

One process per cell (a founder, N, a key width), one row per arm:

* ``moment``: a newborn brain of the default composition at the same widths, its memory filled
  to N records, lived for ``acts`` moments. Each act is split into the read
  (``Recall.stimulate``), the write (``Recall.witness``) and the rest (the settle and the
  brain's own work). ``rest_ms`` at the default 128 slots is the budget a read is held against.
* a candidate bank per plan entry (``bank.Bank``): ``read_ms`` and ``write_ms`` of a full bank,
  the write with its recognition read and its moment of fading.
* ``exact``: the exact nearest-neighbour store's read.
* ``device``: the recall memory's read as one product on a device (``--device cuda`` or
  ``mps``), float32: every slot scored, the softmax, the weights over the values and the nexts,
  timed after the device has finished each read.

``bytes_per_record`` is what the arm holds over N. ``map_bytes`` is the address map of one head
at the row's key width: it grows with the square of the width, which is what
``Brain.compose``'s guard at 4,096 code coordinates bounds.

    PYTHONPATH=src python benchmarks/memory_scale/cost.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _moment(founder: int, recs: Any, acts: int, options: dict[str, Any]) -> dict[str, Any]:
    from library import Slots

    from cadence.stream import Recall

    n, width, value_width = recs.n, recs.address.shape[1], recs.content.shape[1]
    timers = {"read": 0.0, "write": 0.0}
    stimulate, witness = Recall.stimulate, Recall.witness

    def timed(name: str, function: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t = time.perf_counter()
            out = function(*args, **kwargs)
            timers[name] += time.perf_counter() - t
            return out
        return wrapped

    slots = Slots(founder, width, value_width, n, **options)
    brain = slots.brain
    silent = np.zeros(value_width)
    brain.act(np.concatenate([recs.address[0], recs.content[0]])[None, :], greedy=True)
    for i in range(1, n):
        slots.write(recs.address[i], recs.content[i])
    for i in range(20):  # the first acts compile and warm the paths the timed ones take
        brain.act(np.concatenate([recs.address[i % n], silent])[None, :], greedy=True)
    Recall.stimulate, Recall.witness = timed("read", stimulate), timed("write", witness)
    try:
        sweeps = 0.0
        t0 = time.perf_counter()
        for i in range(acts):
            brain.act(np.concatenate([recs.address[i % n], silent])[None, :], greedy=True)
            sweeps += float((brain.last_settlement or {}).get("steps", 0))
        seconds = time.perf_counter() - t0
    finally:
        Recall.stimulate, Recall.witness = stimulate, witness
    w_x, w_c, b = brain.recall.head_map(0)
    return {
        "read_ms": round(1e3 * timers["read"] / acts, 4),
        "write_ms": round(1e3 * timers["write"] / acts, 4),
        "rest_ms": round(1e3 * (seconds - timers["read"] - timers["write"]) / acts, 4),
        "sweeps": round(sweeps / acts, 1),
        "bytes_per_record": round(slots.nbytes / n, 1),
        "map_bytes": int(w_x.nbytes + w_c.nbytes + b.nbytes),
        "temperature": brain.recall.temperature,
        "slots": brain.recall.slots,
    }


def _bank(founder: int, recs: Any, reads: int, settings: dict[str, Any]) -> dict[str, Any]:
    from bank import Bank

    n = recs.n
    bank = Bank(recs.address.shape[1], recs.content.shape[1], n, seed=founder, **settings)
    for i in range(n):
        bank.write(recs.address[i], recs.content[i])
        bank.moment()
    t0 = time.perf_counter()
    for i in range(reads):
        bank.value(bank.read(recs.address[i % n], credit=False))
    read = time.perf_counter() - t0
    rng = np.random.default_rng(founder)
    fresh = rng.standard_normal((reads, recs.address.shape[1]))
    fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
    t0 = time.perf_counter()
    for i in range(reads):
        bank.write(fresh[i], recs.content[i % n])
        bank.moment()
    write = time.perf_counter() - t0
    return {"read_ms": round(1e3 * read / reads, 4), "write_ms": round(1e3 * write / reads, 4),
            "bytes_per_record": round(bank.nbytes / n, 1), "settings": settings}


def _exact(recs: Any, reads: int) -> dict[str, Any]:
    from references import Exact

    store = Exact(recs.address)
    t0 = time.perf_counter()
    for i in range(reads):
        store.top(recs.address[i % recs.n])
    return {"read_ms": round(1e3 * (time.perf_counter() - t0) / reads, 4),
            "bytes_per_record": round(store.nbytes / recs.n, 1)}


def _device(recs: Any, reads: int, device: str, temperature: float) -> dict[str, Any]:
    """The recall memory's read as one product on ``device``, float32: every slot's key against
    the cue, the softmax, and the weights over the values and the nexts."""
    import torch

    where = torch.device(device)
    moments = np.concatenate([recs.address, recs.content], axis=1).astype(np.float32)
    keys = torch.from_numpy(moments).to(where)
    values, nexts = keys.clone(), torch.roll(keys, -1, dims=0)
    cues = torch.from_numpy(
        np.concatenate([recs.address, np.zeros_like(recs.content)], axis=1).astype(np.float32)
    ).to(where)

    def wait() -> None:
        if where.type == "cuda":
            torch.cuda.synchronize()
        elif where.type == "mps":
            torch.mps.synchronize()

    def read(i: int) -> Any:
        p = torch.softmax(keys @ cues[i % recs.n] / temperature, dim=0)
        return p @ values, p @ nexts

    for i in range(20):
        read(i)
    wait()
    t0 = time.perf_counter()
    for i in range(reads):
        out = read(i)
        wait()
    seconds = time.perf_counter() - t0
    del out
    return {"read_ms": round(1e3 * seconds / reads, 4), "device": device,
            "bytes_per_record": round(3 * moments.nbytes / recs.n, 1),
            "temperature": temperature}


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import world

    founder, n, width = int(spec["founder"]), int(spec["n"]), int(spec["width"])
    acts, reads = int(spec.get("acts", 200)), int(spec.get("reads", 500))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, n, width, 5]))
    recs = world.records(rng, n, width, width)
    base = {"founder": founder, "n": n, "width": width}
    rows = []
    for arm in spec["arms"]:
        name = arm if isinstance(arm, str) else arm["name"]
        if name.startswith("moment"):
            options = {} if isinstance(arm, str) else dict(arm.get("options", {}))
            row = _moment(founder, recs, acts, options)
        elif name == "exact":
            row = _exact(recs, reads)
        elif name == "device":
            row = _device(recs, reads, spec.get("device", "cuda"),
                          float(spec.get("temperature", 0.25)))
        else:
            row = _bank(founder, recs, reads, dict(arm.get("settings", {})))
        rows.append({**base, "arm": name, **row})
    return rows


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for founder in plan["founders"]:
        for n in plan["n"]:
            for width in plan.get("widths", [64]):
                arms = [a for a in plan["arms"]
                        if isinstance(a, str) or n <= int(a.get("max_n", n))]
                out.append({"founder": founder, "n": n, "width": width, "arms": arms,
                            **{k: plan[k] for k in ("acts", "reads", "device", "temperature")
                               if k in plan}})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    todo = cells(json.loads(args.plan.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink, get_context("spawn").Pool(args.workers,
                                                               maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_cell, todo, chunksize=1):
            for row in rows:
                sink.write(json.dumps(row) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
