"""Rooms of things for the map of things: what there is and where, with things that arrive, leave
and are moved while the camera looks elsewhere.

A room is the chamber's plane with things at rest, each of a kind. A camera at the plane's lower
middle sees a sector of 40 degrees that turns: over the whole room for its first sweep, then over
one half of it. Things in the sector are reported with a centimetre of noise. The map is given the
reports and the sector as a function of places; the room's truth is kept beside it to score.

- ``kinds`` (S9): eight things of four kinds, two of each; asked "where is" each kind and "what
  is at" each thing's place, for things in view and things last seen 500 moments before.
- ``moved`` (S11): a thing on the unswept half is moved there while the camera looks away; read
  before the camera sweeps there again and five moments after it sees the thing anew.
- ``many`` (S12): 8, 32 and 128 things of as many kinds, one sweep, then 500 moments of the
  camera looking at the floor beside the room; read by kind and by place.
- ``arrive`` (S13): things arrive in view, a second thing of a kind arrives apart, a thing is
  taken away while unseen and later put back elsewhere; the inventory is read against the room.

    python benchmarks/tracking/rooms.py --founder 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking as tr  # noqa: E402

PLANE = (-1.0, 1.0, -0.2, 1.4)
EYE = np.array([0.0, -0.4])
NOISE = 0.01
HALF = np.deg2rad(20.0)


def sees(heading: float) -> Any:
    """The camera's sector at a heading, as a function of ``(points, 2)`` places."""
    def look(points: np.ndarray) -> np.ndarray:
        away = np.asarray(points, dtype=np.float64).reshape(-1, 2) - EYE
        turn = np.arctan2(away[:, 1], away[:, 0]) - heading
        return np.asarray(np.abs((turn + np.pi) % (2 * np.pi) - np.pi) <= HALF)
    return look


def headings(steps: int, first: int = 120, half: str = "left") -> np.ndarray:
    """A sweep over the whole room (from 15 to 165 degrees and back) in the first ``first``
    moments, then sweeps over one half only."""
    out = np.zeros(steps)
    whole = np.deg2rad(np.r_[np.linspace(15, 165, first // 2), np.linspace(165, 15, first // 2)])
    out[:first] = whole[:first]
    low, high = (95, 165) if half == "left" else (15, 85)
    t = np.arange(steps - first)
    out[first:] = np.deg2rad(low + (high - low) * 0.5 * (1 - np.cos(2 * np.pi * t / 120)))
    return out


def places(rng: np.random.Generator, count: int, apart: float, side: str | None = None
           ) -> np.ndarray:
    """``count`` places in the room at least ``apart`` from each other, on one half where given
    (left: x below -0.1, right: x above 0.1)."""
    out: list[np.ndarray] = []
    for _ in range(100_000):
        if len(out) == count:
            break
        x = rng.uniform(-0.9, 0.9) if side is None else (
            rng.uniform(-0.9, -0.1) if side == "left" else rng.uniform(0.1, 0.9))
        point = np.array([x, rng.uniform(0.1, 1.3)])
        if all(np.linalg.norm(point - other) >= apart for other in out):
            out.append(point)
    if len(out) < count:
        raise ValueError("no room for that many things that far apart")
    return np.array(out)


def run(held: Any, kinds: np.ndarray, where: np.ndarray, present: np.ndarray,
        heading: np.ndarray, rng: np.random.Generator) -> None:
    """Moments of a room: at moment t the things present (``present[t]``) at ``where[t]`` in
    the camera's sector are reported."""
    for t in range(len(heading)):
        look = sees(float(heading[t]))
        here = present[t] & look(where[t])
        seen = [(int(kinds[i]), where[t, i] + rng.normal(0.0, NOISE, 2))
                for i in np.flatnonzero(here)]
        held.moment(seen=seen, lit=look)


def match(held: Any, kinds: np.ndarray, truth: np.ndarray, within: float = 0.05
          ) -> dict[str, float]:
    """The map's answers against the things of a room: the share of things whose kind's
    ``where_is`` holds a place within ``within`` of them, the share whose place ``what_is_at``
    answers with their kind, the mean error of the nearest place, and the inventory's things
    with a place that match no thing (extra) and things it misses."""
    found, named, errors = [], [], []
    for kind, point in zip(kinds, truth, strict=True):
        got = held.where_is(int(kind))
        apart = np.linalg.norm(got - point, axis=1) if len(got) else np.array([np.inf])
        found.append(float(apart.min() <= within))
        errors.append(float(apart.min()))
        named.append(float(held.what_is_at(point) == int(kind)))
    listed = [t for t in held.things() if t["place"] is not None]
    used = np.zeros(len(listed), dtype=bool)
    for kind, point in zip(kinds, truth, strict=True):
        best = [j for j, t in enumerate(listed) if not used[j] and t["kind"] == int(kind)
                and np.linalg.norm(t["place"] - point) <= within]
        if best:
            used[best[0]] = True
    return {"where": float(np.mean(found)), "what": float(np.mean(named)),
            "error": float(np.mean(errors)), "extra": float((~used).sum()),
            "missed": float(len(kinds) - used.sum())}


def kinds_panel(rng: np.random.Generator, settings: dict[str, Any]) -> dict[str, Any]:
    """S9: eight things of four kinds, two a kind; the camera sweeps the room once, then the
    left half for 500 moments. Read the things in view (left) and those last seen 500
    moments before (right)."""
    from cadence.space.map import Map

    left, right = places(rng, 4, 0.2, "left"), places(rng, 4, 0.2, "right")
    truth = np.vstack([left, right])
    kinds = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    steps = 620
    held = Map(PLANE, **settings)
    run(held, kinds, np.broadcast_to(truth, (steps, 8, 2)), np.ones((steps, 8), dtype=bool),
        headings(steps), rng)
    return {"in view": match(held, kinds[:4], left), "seen 500 before": match(
        held, kinds[4:], right), "seen error": NOISE * np.sqrt(np.pi / 2)}


def moved_panel(rng: np.random.Generator, settings: dict[str, Any]) -> dict[str, Any]:
    """S11: a thing on the right half is moved there at moment 300 while the camera sweeps the
    left; at moment 500 the camera turns to the right half. Read at 499 (the old place held,
    nothing at the new) and five moments after the camera first sees it at the new place."""
    from cadence.space.map import Map

    truth = places(rng, 6, 0.25, None)
    right = int(np.argmax(truth[:, 0]))
    new = places(rng, 1, 0.0, "right")[0]
    while np.linalg.norm(new - truth[right]) < 0.3 or np.min(
            np.linalg.norm(np.delete(truth, right, 0) - new, axis=1)) < 0.2:
        new = places(rng, 1, 0.0, "right")[0]
    kinds = np.arange(6)
    steps = 800
    where = np.broadcast_to(truth, (steps, 6, 2)).copy()
    where[300:, right] = new
    heading = headings(steps)
    heading[500:] = headings(steps - 500 + 120, half="right")[120:]
    held = Map(PLANE, **settings)
    present = np.ones((steps, 6), dtype=bool)
    run(held, kinds, where[:500], present[:500], heading[:500], rng)
    before = held.where_is(int(kinds[right]))
    old_held = bool(len(before) and np.linalg.norm(before - truth[right], axis=1).min() < 0.05)
    first = next(t for t in range(500, steps) if sees(heading[t])(new[None])[0])
    run(held, kinds, where[500 : first + 6], present[500 : first + 6], heading[500 : first + 6],
        rng)
    after = held.where_is(int(kinds[right]))
    return {"old place held before": old_held,
            "new place within 5 moments": bool(len(after) and np.linalg.norm(
                after - new, axis=1).min() < 0.05),
            "nothing at the old place": bool(not len(after) or np.linalg.norm(
                after - truth[right], axis=1).min() > 0.15),
            "seen at": int(first)}


def many_panel(rng: np.random.Generator, count: int, settings: dict[str, Any]
               ) -> dict[str, Any]:
    """S12: ``count`` things of as many kinds; one sweep of the room, then 500 moments with the
    camera on the floor beside it (nothing seen); read by kind and by place."""
    from cadence.space.map import Map

    truth = places(rng, count, 0.10 if count > 32 else 0.2)
    kinds = np.arange(count)
    units = 16 if count <= 16 else 32
    held = Map(PLANE, units=units, kinds=max(count, units), **settings)
    steps = 120
    run(held, kinds, np.broadcast_to(truth, (steps, count, 2)),
        np.ones((steps, count), dtype=bool), headings(steps), rng)
    for _ in range(500):
        held.moment(lit=sees(-np.pi / 2))
    return match(held, kinds, truth)


def arrive_panel(rng: np.random.Generator, settings: dict[str, Any]) -> dict[str, Any]:
    """S13: four things at first; at 150 a thing of a new kind and a second thing of kind 0
    arrive on the left, where the camera sweeps; at 280 thing 1 (right) is taken away while the
    camera sweeps the left; at 500 the camera turns right, sees its place empty and sweeps over
    it; at 650 thing 1 is put back on the left. Read the inventory at 149 (four things), 279
    (six), 640 (thing 1 not known where) and the end (six, thing 1 found once)."""
    from cadence.space.map import Map

    first = np.vstack([places(rng, 2, 0.3, "left"), places(rng, 2, 0.3, "right")])
    later = places(rng, 3, 0.3, "left")
    while np.min(np.linalg.norm(later[:, None] - first[None], axis=2)) < 0.25:
        later = places(rng, 3, 0.3, "left")
    kinds = np.array([0, 1, 2, 3, 4, 0])
    steps = 800
    where = np.zeros((steps, 6, 2))
    where[:, :4] = first
    where[:, 4], where[:, 5] = later[0], later[1]
    present = np.ones((steps, 6), dtype=bool)
    present[:150, 4:] = False
    thing = int(np.argmax(first[:, 0] * (kinds[:4] == 1) - 9 * (kinds[:4] != 1)))
    present[280:650, thing] = False
    where[650:, thing] = later[2]
    heading = headings(steps)
    heading[500:650] = headings(150 + 120, half="right")[120:]
    held = Map(PLANE, **settings)
    marks: dict[str, Any] = {}
    for start, stop in ((0, 150), (150, 280), (280, 641), (641, steps)):
        run(held, kinds, where[start:stop], present[start:stop], heading[start:stop], rng)
        listed = held.things()
        here = present[stop - 1]
        score = match(held, kinds[here], where[stop - 1, here])
        marks[str(stop - 1)] = {**score, "things": len([t for t in listed
                                                        if t["place"] is not None]),
                                "not known where": [t["kind"] for t in listed
                                                    if t["state"] == "not known where"]}
    return marks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--founder", type=int, default=0)
    parser.add_argument("--rooms", type=int, default=20)
    parser.add_argument("--settings", type=json.loads, default={})
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(np.random.SeedSequence([tr.seed(args.founder, "test"), 13]))
    out: dict[str, Any] = {"founder": args.founder, "rooms": args.rooms}
    out["kinds"] = [kinds_panel(rng, args.settings) for _ in range(args.rooms)]
    out["moved"] = [moved_panel(rng, args.settings) for _ in range(args.rooms)]
    out["many"] = {str(n): [many_panel(rng, n, args.settings)
                            for _ in range(max(args.rooms // 4, 2))] for n in (8, 32, 128)}
    out["arrive"] = [arrive_panel(rng, args.settings) for _ in range(args.rooms)]
    k = out["kinds"]
    print(f"S9 kinds: in view where {np.mean([r['in view']['where'] for r in k]):.2f} what "
          f"{np.mean([r['in view']['what'] for r in k]):.2f} error "
          f"{100 * np.mean([r['in view']['error'] for r in k]):.2f} cm; seen 500 before where "
          f"{np.mean([r['seen 500 before']['where'] for r in k]):.2f} what "
          f"{np.mean([r['seen 500 before']['what'] for r in k]):.2f} error "
          f"{100 * np.mean([r['seen 500 before']['error'] for r in k]):.2f} cm", flush=True)
    m = out["moved"]
    print("S11 moved: " + ", ".join(f"{key} {np.mean([r[key] for r in m]):.2f}" for key in (
        "old place held before", "new place within 5 moments", "nothing at the old place")),
        flush=True)
    for n, rows in out["many"].items():
        print(f"S12 {n} things: where {np.mean([r['where'] for r in rows]):.2f} what "
              f"{np.mean([r['what'] for r in rows]):.2f} error "
              f"{100 * np.mean([r['error'] for r in rows]):.2f} cm", flush=True)
    a = out["arrive"]
    for mark in ("149", "279", "640", "799"):
        print(f"S13 at {mark}: where {np.mean([r[mark]['where'] for r in a]):.2f} extra "
              f"{np.mean([r[mark]['extra'] for r in a]):.2f} missed "
              f"{np.mean([r[mark]['missed'] for r in a]):.2f} things "
              f"{np.mean([r[mark]['things'] for r in a]):.2f} not known where "
              f"{np.mean([1 in r[mark]['not known where'] for r in a]):.2f}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
