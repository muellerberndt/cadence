"""A thing out of view, asked about (cadence#342): readings of a brain whose glimpse is the
map's read of the attended thing, beside the same brain fed the eyes' direct report.

    python occluded.py <brain folder> <out.json> [questions a row]

``<brain folder>`` holds ``brain.npz`` of the meaning chamber's founder. Each row asks
``n`` questions of a form on unseen kind-colour pairs for the colour forms, frozen and greedy:

- **in view:** the meaning chamber's own exchange;
- **hidden:** the eyes look over the room, then the thing asked about goes behind a screen
  (in no slot of the scene, its place unlit) before the question;
- **hidden, moved:** hidden, the thing makes a sound at another place, in another direction
  from the body; the right answer to "where" is the direction of the new place.

``source`` "eyes" feeds the glimpse the eyes' report (nothing, for a hidden thing); "map" feeds
it the map's read. ``map activity`` reads how far the map's own read regions (``where`` and
``what``) move between the settle of a look and the settle of the attended read, and whether
the sheet's carried potentials are the same after a read. Where the folder holds a ``spec.json``
the demo room is asked through the page's host as well (``mapped.MappedHost``): every question
on every thing, with the thing asked about in view and behind a screen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lane
import mapped
import numpy as np

from cadence import Brain

nt = lane.night
FORMS = ("name", "colour_of", "colour_this", "material_this", "direction", "is_round",
         "can_eat_bal")
HELD = ("colour_of", "colour_this")


def hide(exs: list, rng: np.random.Generator, moved: bool = False) -> None:
    """The thing asked about goes behind a screen; with ``moved`` it was seen first at another
    place, in another direction from the body."""
    for ex in exs:
        x = next(t for t in ex.room.things if t.focus)
        ex.extra["hidden"] = (x.kind,)
        if moved:
            taken = [t.place for t in ex.room.things]
            was = nt.free_place(rng, taken)
            for _ in range(500):
                if nt.direction_of(ex.room.heading, was) != nt.direction_of(
                        ex.room.heading, x.place):
                    break
                was = nt.free_place(rng, taken)
            ex.extra["moved"] = {x.kind: was}


def row(p, form: str, held: set, n: int, seed: int, how: str) -> float:
    rng = np.random.default_rng(seed)
    ok: list[bool] = []
    while len(ok) < n:
        exs = nt.blocks_of(p, rng, form, "turn", held, test=form in HELD)
        if how != "in view":
            hide(exs, rng, moved=how == "hidden, moved")
        ok += nt.live(p, exs, rng, learn=False)["right"]
    return round(float(np.mean(ok[:n])), 3)


def map_activity(p, held: set, rooms: int = 64, seed: int = 5) -> dict[str, float]:
    """How far ``where`` and ``what`` move from a look's settle to the attended read's, and
    between the reads of two things of one room, as shares of the look's length."""
    rng = np.random.default_rng(seed)
    me = mapped.MapEyes(p.eyes)
    moved, apart, share, same = [], [], [], True
    for _ in range(rooms):
        r = nt.room(rng, int(rng.integers(3, 6)), held)
        me.reset()
        me.look([(t.kind, t.place, r.seen_of(t)) for t in r.things])
        own = me.mine()
        look = own.observation()
        carried = own.potential.copy()
        reads = []
        for t in r.things[:2]:
            v = own._settle(kept=own.sheet(mapped.KIND[t.kind]).ravel() > 0)
            run = own._runs
            reads.append(np.maximum(np.tanh(np.concatenate(
                [v[run["where"]], v[run["what"]]]) / 2.0), 0.0))
        same &= bool(np.array_equal(carried, own.potential))  # a read changes nothing held
        scale = float(np.linalg.norm(look))
        moved.append(float(np.linalg.norm(reads[0] - look)) / scale)
        apart.append(float(np.linalg.norm(reads[0] - reads[1])) / scale)
        where = reads[0][: me.map.count]
        on = me.map.sheet(mapped.KIND[r.things[0].kind]).ravel() > 0
        share.append(float(where[on].sum() / where.sum()))
    return {"look to attended read": round(float(np.mean(moved)), 3),
            "between two attended things": round(float(np.mean(apart)), 3),
            "share of where on the attended thing's cells": round(float(np.mean(share)), 3),
            "the carried sheet is unchanged by a read": same, "rooms": rooms}


def demo_room(folder: Path) -> dict[str, float]:
    """The demo room through the page's host: the share of its questions answered right, the
    thing asked about in view and behind a screen, the glimpse from the eyes and the map."""
    ds, ep, ps = lane.demo_stage, lane.export_page, lane.pretrain_stage
    out = {}
    for source in ("eyes", "map"):
        m = mapped.MappedHost(folder / "brain.npz", folder / "spec.json", seed=lane.SEED,
                              source=source)
        m.records = None
        for how in ("in view", "behind a screen"):
            ok = []
            for words, answer, focus, _ in ps.room_questions():
                things = ds.room_things(focus)
                m.see(things)  # the eyes look over the room; then the thing is hidden
                if how != "in view":
                    things = [{**t, "hidden": bool(t["focus"])} for t in things]
                ok.append(ep.ask(m, things, list(words), False) == answer)
            out[f"{how} / {source}"] = round(float(np.mean(ok)), 3)
    return out


def main(argv: list[str]) -> None:
    folder, out = Path(argv[1]), Path(argv[2])
    n = int(argv[3]) if len(argv) > 3 else 96
    mapped.install("map")
    held = nt.founder_held(lane.SEED)
    p = nt.Parrot(lane.SEED, dict(lane.GENOME), streams=32)
    p.brain = Brain.load(folder / "brain.npz")
    p.records = None
    res: dict = {"brain": folder.name, "questions": n, "rows": {}}
    for how in ("in view", "hidden", "hidden, moved"):
        for source in ("eyes", "map"):
            p.source = source
            forms = FORMS if how != "hidden, moved" else ("direction", "colour_of")
            res["rows"][f"{how} / {source}"] = {
                f: row(p, f, held, n, 11, how) for f in forms}
            print(how, source, json.dumps(res["rows"][f"{how} / {source}"]), flush=True)
            out.write_text(json.dumps(res, indent=1))
    res["map activity"] = map_activity(p, held)
    print(json.dumps(res["map activity"]), flush=True)
    if (folder / "spec.json").exists():
        res["demo room through the host"] = demo_room(folder)
        print(json.dumps(res["demo room through the host"]), flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
