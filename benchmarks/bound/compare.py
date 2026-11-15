"""Brains of the meaning chamber's founder read the same way, glimpse from the eyes and from
the map (cadence#342).

    python compare.py <out.json> NAME=<brain folder> ...

Through the page's host (``mapped.MappedHost``): the demo room's every question, three passes
(the share answered right in all three), and fresh rooms, 300 questions on kind-colour pairs of
the training rooms and 300 on pairs kept out of training. Through the chamber, frozen and
greedy: every focus form, 96 questions each, on the kept-out pairs for the colour forms.
A brain folder holds ``brain.npz`` and ``spec.json`` (``life.py`` writes both).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lane
import mapped
import numpy as np

from cadence import Brain

nt, ds, ep, ps = lane.night, lane.demo_stage, lane.export_page, lane.pretrain_stage
FORMS = ["name", "colour_this", "material_this", "is_colour", "colour_of", "shape_of",
         "is_round", "can_eat_bal", "direction"]
TEST = ("colour_of", "colour_this", "is_colour")


def as_host(r) -> list[dict]:
    return [{"kind": t.kind, "colour": t.colour, "material": t.material,
             "direction": nt.bearing(r.heading, t.place),
             "distance": float(np.linalg.norm(t.place)), "focus": bool(t.focus)}
            for t in r.things]


def draws(held: set) -> dict[str, list]:
    rng = np.random.default_rng(5)

    def draw(test: bool):
        while True:
            form = FORMS[int(rng.integers(len(FORMS)))]
            r = nt.room(rng, int(rng.integers(3, 6)), held, test=test)
            q = nt.ask(form, r, rng)
            if q is not None and not q.scan:
                return form, as_host(r), list(q.words), list(q.answer)

    return {"seen": [draw(False) for _ in range(300)], "held": [draw(True) for _ in range(300)]}


def chamber_reads(p, held: set, n: int = 96) -> dict[str, float]:
    rng = np.random.default_rng(99)
    out = {}
    for f in [*FORMS, "shape_not", "imitate"]:
        for test in ((False, True) if f in TEST else (False,)):
            ok: list[bool] = []
            while len(ok) < n:
                ok += nt.live(p, nt.blocks_of(p, rng, f, "turn", held, test=test), rng,
                              learn=False)["right"]
            out[f + ("/held" if test else "")] = round(float(np.mean(ok[:n])), 3)
    return out


def main(argv: list[str]) -> None:
    out = Path(argv[1])
    brains = [(a.split("=")[0], Path(a.split("=")[1])) for a in argv[2:]]
    mapped.install("map")
    held = nt.founder_held(lane.SEED)
    fresh = draws(held)
    res: dict = json.loads(out.read_text()) if out.exists() else {}
    for name, folder in brains:
        for source in ("eyes", "map"):
            label = f"{name}, glimpse from the {source}"
            if label in res:
                continue
            m = mapped.MappedHost(folder / "brain.npz", folder / "spec.json", seed=lane.SEED,
                                  source=source)
            m.records = None
            row: dict = {}
            room: dict = {}
            for _ in range(3):
                for words, answer, focus, form in ps.room_questions():
                    room.setdefault((" ".join(words), focus, form), []).append(
                        ep.ask(m, ds.room_things(focus), list(words), False) == answer)
            by: dict[str, list[bool]] = {}
            for (_, _, form), v in room.items():
                by.setdefault(form, []).append(all(v))
            row["room, right in all three passes"] = round(float(np.mean(
                [all(v) for v in room.values()])), 3)
            row["room by form"] = {f: f"{sum(v)}/{len(v)}" for f, v in by.items()}
            for k, rows in fresh.items():
                byf: dict[str, list[bool]] = {}
                for form, things, words, answer in rows:
                    byf.setdefault(form, []).append(ep.ask(m, things, words, False) == answer)
                row[f"fresh rooms, {k} pairs"] = round(float(np.mean(sum(byf.values(), []))), 3)
                row[f"fresh rooms, {k} pairs, by form"] = {
                    f: round(float(np.mean(v)), 2) for f, v in byf.items()}
            p = nt.Parrot(lane.SEED, dict(lane.GENOME), streams=32)
            p.brain = Brain.load(folder / "brain.npz")
            p.records = None
            p.source = source
            row["chamber"] = chamber_reads(p, held)
            res[label] = row
            print(label, json.dumps(row), flush=True)
            out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
