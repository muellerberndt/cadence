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


def rates(scale: float, form: str, seed: int, depth: str) -> None:
    modules = [160, 128] if depth == "d2" else [128] * 4
    layout = tc.Layout.founder(seed)
    p = tp.BrainPolicy(layout, {"eta_scale": scale, "modules": modules}, seed)
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


if __name__ == "__main__":
    if sys.argv[1] == "rates":
        rates(float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        errors(sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4]))
