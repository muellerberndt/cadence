"""Keys that drift (cadence#342): records written by a brain at one point of its life, read by
the same brain later in its life.

    python drift.py <out.json> <earlier brain folder> <later brain folder> [things] [heads]

The earlier brain names ``things`` things (``completion.py``'s records) and writes a record
under each read of its state. The later brain hears each word alone with nothing attended and
reads the earlier brain's records. For each key: the share of the heads' records named by the
cue's word, read by the earlier brain itself and by the later one; and how far the
association's state of one naming moment moved between the two brains (the cosine).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import completion as cp
import lane
import numpy as np

from cadence import Brain
from cadence.longterm import LongTermMemory

nt, it = lane.night, lane.introspect


def main(argv: list[str]) -> None:
    out, early, late = Path(argv[1]), Path(argv[2]), Path(argv[3])
    things = int(argv[4]) if len(argv) > 4 else 320
    heads = int(argv[5]) if len(argv) > 5 else 5
    ch = cp.Chamber(early)
    p = ch.p
    rng = np.random.default_rng(7)
    named = [(r, w) for r, t in it.scenes(p, rng, things, ch.held, test=False)
             for w in cp.names(r, t)]
    widths = {"association": len(ch.assoc), "code": p.mixed, "both": len(ch.assoc) + p.mixed}
    ch.stores = {name: LongTermMemory(width=widths[name], value_width=ch.w + ch.ear,
                                      capacity=2 * len(named), cosine=True, revise=0.9999,
                                      seed=lane.SEED) for name in cp.KEYS}
    states = []
    for chunk, m in cp.batches(named):
        keys, _ = ch.live([r for r, _ in chunk], [[None, None, w] for _, w in chunk],
                          np.random.default_rng(len(states)),
                          write=[True] * m + [False] * (32 - m))
        states.append(keys["association"][:m])
    words = [w for group in cp.WORDS.values() for w in group]
    empty = nt.Room([], heading=0.0)

    def read() -> dict[str, float]:
        row = {}
        for name in cp.KEYS:
            hits: list[bool] = []
            for chunk, m in cp.batches(words):
                ch.live([empty] * 32, [[None, None, w] for w in chunk], rng, key=name,
                        heads=heads, after=1)
                hits += [named[s][1] == w for w, slots in zip(chunk[:m], ch.told[:m],
                                                              strict=True) for s in slots]
            row[name] = round(float(np.mean(hits)), 3)
        return row

    res = {"records": len(named), "heads": heads, "earlier": early.name, "later": late.name,
           "heads' records named by the cue's word": {"read by the earlier brain": read()}}
    p.brain = Brain.load(late / "brain.npz")
    ch.brain = it.frozen(p)
    res["heads' records named by the cue's word"]["read by the later brain"] = read()
    kept, ch.stores = ch.stores, {}
    moved = []
    for k, (chunk, m) in enumerate(cp.batches(named)):
        keys, _ = ch.live([r for r, _ in chunk], [[None, None, w] for _, w in chunk],
                          np.random.default_rng(k))
        moved.append(np.sum(cp.unit(keys["association"][:m]) * cp.unit(states[k]), axis=1))
    ch.stores = kept
    res["cosine of a naming moment's association state, earlier to later"] = round(
        float(np.mean(np.concatenate(moved))), 3)
    print(json.dumps(res, indent=1))
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
