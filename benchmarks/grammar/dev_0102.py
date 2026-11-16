"""Development reading for 0.102.0 (#308): the role-by-choice store (``sentence.py``'s design
``chosen``) on four development founders, and where the recall read lands on a path's hops with
and without the bar that empties a path's start at ``GO``.

    SENTENCE_BACKEND=mlx PYTHONPATH=src python benchmarks/grammar/dev_0102.py OUT_DIR [batches]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grammar as gr  # noqa: E402
import meaning as mn  # noqa: E402
import sentence as st  # noqa: E402


def locate_barred(brain, batches, bar):
    """``grammar.locate`` with the bar applied at ``GO`` as the lessons apply it."""
    fork = copy.deepcopy(brain)
    groups = {}
    for rows in batches:
        fork.reset()
        written = 0
        for t in range(len(rows[0].pre)):
            fork.act(np.stack([r.pre[t] for r in rows]), greedy=True)
            written += 1
        if bar:
            mn.bar_meaning(fork, rows)
        expected = [gr.sources(r) for r in rows]
        for j in range(len(rows[0].target)):
            x = (np.tile(gr.concept("GO"), (len(rows), 1)) if j == 0
                 else np.stack([gr.spoken(r.target[j - 1]) for r in rows]))
            ages = None if fork.recall is None else np.asarray(fork.recall.ages())
            action = np.asarray(fork.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
            heads = fork.recall.attended_heads if fork.recall is not None else None
            p = heads[0] if heads else None
            for i, r in enumerate(rows):
                source = expected[i][j]
                if source is None:
                    continue
                landed = -2
                if p is not None and ages is not None:
                    landed = written - 1 - int(ages[int(np.argmax(p[i]))])
                g = groups.setdefault(gr._kind(r, j), [0, 0, 0])
                g[0] += int(landed in (source, source - 1))
                g[1] += int(action[i] == r.target[j])
                g[2] += 1
            written += 1
    return {k: {"n": v[2], "delivered": round(v[0] / v[2], 3), "said": round(v[1] / v[2], 3)}
            for k, v in sorted(groups.items())}


def main(out: Path, batches: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for founder in (0, 1, 2, 3):
        save = out / f"brain-{founder}"
        row = st.life({"kind": "store", "founder": founder, "batches": batches, "emulated": True,
                       "design": "chosen", "depth": 4, "held_probes": 4, "save": str(save),
                       "tag": "chosen"})
        brain = row.pop("_brain", None)
        (out / f"life-{founder}.json").write_text(json.dumps(row, default=str))
        print(json.dumps({"founder": founder, "panels": row.get("panels"),
                          "sayback": row.get("sayback"), "seconds": row.get("seconds")},
                         default=str), flush=True)
        del brain


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else mn.BATCHES)
