"""Development diagnostics of the text parrot (amendments 1 and 2 on cadence#338).

    python textdiag.py rates <scale> <form|props> <founder> <d2|d4>    # rates times scale
    python textdiag.py errors <form> <founder> '<genome json>'        # where a list form fails

``rates``: a fresh brain taught one form (or the properties stage's seven, ``props``) with the
teaching point's rates times ``scale``, read every 25 blocks. ``errors``: at the answer sweep's
first moment at each thing, teacher forced, whether a matching thing is named (right, another
word, silence) and whether another thing is passed in silence, every 50 blocks.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import textchamber as tc  # noqa: E402
import textparrot as tp  # noqa: E402
import textworld as tw  # noqa: E402


def mode(b: int, rng: np.random.Generator) -> str:
    return "teacher" if b <= 20 or rng.random() > 0.3 else "dagger"


def rates(scale: float, form: str, seed: int, depth: str, genome: dict | None = None) -> None:
    modules = [160, 128] if depth == "d2" else [128] * 4
    layout = tc.Layout.founder(seed)
    p = tp.BrainPolicy(layout, {"eta_scale": scale, "modules": modules, **(genome or {})}, seed)
    forms = dict(tp.STAGES)["properties"] if form == "props" else (form,)
    rng = np.random.default_rng(seed)
    t = time.time()
    for b in range(1, 601):
        tc.run(p, layout, tp.batch(rng, forms, 64), mode(b, rng))
        if b % 25 == 0:
            r = tp.read(p, layout, rng, forms, 64)
            print(json.dumps({"scale": scale, "form": form, "block": b,
                              "read": float(np.mean([v["right"] for v in r.values()])),
                              "forms": {k: v["right"] for k, v in r.items()},
                              "s": round(time.time() - t),
                              "max_eff": p.saturation()["max_efficacy"]}), flush=True)


def breakdown(p: tp.BrainPolicy, layout: tc.Layout, form: str, rng: np.random.Generator,
              n: int = 128) -> dict[str, int]:
    eps = [tc.Episode(layout, s, q, "teacher") for s, q in (tp.draw(form, rng) for _ in range(n))]
    p.begin(n)
    c = dict.fromkeys(("match_right", "match_other", "match_silent", "nomatch_silent",
                       "nomatch_word"), 0)
    for _ in range(120):
        mask = np.array([not e.done for e in eps])
        if not mask.any():
            break
        x = np.stack([e.observe() for e in eps])
        a = p.moment(x, None, mask, False)
        for k, e in enumerate(eps):
            if e.done:
                continue
            w, g = e.expert()
            said = layout.say(int(a[k, 0]))
            if e.phase == "answer" and not e.at:
                target = e.target_words()
                if target:
                    key = ("match_right" if said == target[0] else
                           "match_silent" if said == tw.SILENCE else "match_other")
                else:
                    key = "nomatch_silent" if said == tw.SILENCE else "nomatch_word"
                c[key] += 1
            e.advance(w, g)
    return c


def errors(form: str, seed: int, genome: dict) -> None:
    layout = tc.Layout.founder(seed)
    p = tp.BrainPolicy(layout, genome, seed)
    rng = np.random.default_rng(seed)
    t = time.time()
    for b in range(1, 401):
        tc.run(p, layout, tp.batch(rng, (form,), 64), mode(b, rng))
        if b % 50 == 0:
            print(json.dumps({"block": b, "s": round(time.time() - t),
                              **breakdown(p, layout, form, rng)}), flush=True)


def gaze(seed: int, genome: dict) -> None:
    """The words stage, then "where" alone: read free (the parrot's own hold) and with the
    teacher's focus (its own words), and the gaze's holds at the named thing."""
    layout = tc.Layout.founder(seed)
    p = tp.BrainPolicy(layout, genome, seed)
    rng = np.random.default_rng(seed)
    t = time.time()
    for b in range(1, 121):
        tc.run(p, layout, tp.batch(rng, dict(tp.STAGES)["words"], 64), mode(b, rng))
    for b in range(1, 301):
        m = mode(b, rng)
        if m == "dagger" and genome.get("dagger_gaze"):
            m = "dagger_gaze"
        tc.run(p, layout, tp.batch(rng, ("where",), 64), m)
        if b % 50 == 0:
            items = [tp.draw("where", rng) for _ in range(64)]
            free = tc.run(p, layout, items, "free", learn=False)
            focus = tc.run(p, layout, items, "dagger", learn=False)
            holds = [e.held for e in free]
            right_hold = float(np.mean([h == [e.q.referent] for h, e in zip(holds, free,
                                                                              strict=True)]))
            print(json.dumps({"block": b, "s": round(time.time() - t),
                              "free": float(np.mean([tc.score(e)["right"] for e in free])),
                              "teacher_focus": float(np.mean([tc.score(e)["right"]
                                                              for e in focus])),
                              "held_right": right_hold}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gaze":
        gaze(int(sys.argv[2]), json.loads(sys.argv[3]))
    elif sys.argv[1] == "rates":
        rates(float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5],
              json.loads(sys.argv[6]) if len(sys.argv) > 6 else None)
    else:
        errors(sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4]))
