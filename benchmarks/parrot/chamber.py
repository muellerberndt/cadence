"""The parrot's first chamber (#332): things around a low perch, two eyes, a map and distance.

A founder is one seed: its world (the scenes of its life and of its tests) and its brain. A life
is ``scenes`` scenes of ``moments`` moments: things are placed around the perch, the parrot looks
around by its own reflex and learns. No distance, place or kind is ever given to the brain; the
chamber keeps the truth for the readings only.

Readings, on fresh test scenes after the life (no lessons):

- **E1, seen and held**: after a look-around, the share of things the map holds within
  ``margin`` of their place; then each thing put behind the head, in the blind spot, for
  ``behind`` moments: the share still held there.
- **E2, the same thing from another view**: the head turned so that the thing is seen by the
  left eye alone, by the right eye alone, then in front by both: the share of things held as
  exactly one thing near their place, with no other thing near them.
- **E3, distance learned**: the distance read for sightings by both eyes in the region both
  eyes see, against the true distance on the table's plane, as the median of |log(read/true)|;
  the same things with one eye covered; the learning curve over the life.
- **E7, the distance neurons**: every neuron's tuning to the true distance and to the true
  disparity, the neurons that carry distance found by their information about it, and the
  readings with them silenced and with as many other neurons silenced.

``python -m benchmarks.parrot.chamber --founders 900 901 --out runs/dev`` (development founders).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from cadence.parrot.brain import GENES, Parrot, flat
from cadence.parrot.eyes import render
from cadence.parrot.world import NAMES, Scene, Thing, stand_ins

TABLE = stand_ins()
KINDS = sorted(TABLE["kinds"])
LIFE = {"scenes": 24, "moments": 60, "things": (3, 6)}
TESTS = {"scenes": 10, "moments": 50, "behind": 20, "margin": 0.08, "alone": 0.15}


def place_things(rng: np.random.Generator, count: int, panel: str = "stands") -> Scene:
    """``count`` things around the perch: 0.15 to 0.6 m from it, 0.18 m apart or more, on the
    table; ``panel`` "stands" puts each on a clear stand 0 to 0.25 m high at a scale of 0.7 to
    1.4, "table" puts each on the table."""
    out: list[Thing] = []
    while len(out) < count:
        a = float(rng.uniform(-math.pi, math.pi))
        r = float(rng.uniform(0.15, 0.6))
        x, z = r * math.sin(a), r * math.cos(a)
        if abs(x) > 0.72 or abs(z) > 0.48:
            continue
        if any(math.hypot(x - t.x, z - t.z) < 0.18 for t in out):
            continue
        stand = float(rng.uniform(0.0, 0.25)) if panel == "stands" else 0.0
        out.append(Thing(str(len(out)), KINDS[int(rng.integers(len(KINDS)))], x, z,
                         float(rng.uniform(0.0, 2.0 * math.pi)), float(rng.uniform(0.7, 1.4)),
                         stand))
    return Scene(out)


def truth_distance(scene: Scene, thing: int, origin: Any) -> float:
    """The distance on the table's plane from a sighting's origin to the thing's middle."""
    c = scene.things[thing].center(TABLE)
    return float(np.linalg.norm(flat(c - np.asarray(origin))))


def truth_disparity(scene: Scene, thing: int, parrot: Parrot) -> float:
    """The true disparity of a thing's middle (degrees): its head azimuth from the right eye
    less that from the left eye."""
    eyes = parrot.body.eyes()
    c = scene.things[thing].center(TABLE)
    out = []
    for e in (0, 1):
        local = eyes.rotation.T @ (c - eyes.places[e])
        out.append(math.atan2(local[0], local[2]))
    return math.degrees(out[1] - out[0])


def step(parrot: Parrot, scene: Scene, *, learn: bool, act: bool = True,
         cover: str | None = None) -> dict[str, Any]:
    sight = render(scene, parrot.body.eyes(), cover=cover)
    return parrot.moment(sight.images, learn=learn, ids=sight.ids, act=act)


def turn(parrot: Parrot, yaw: float) -> None:
    """Put the head at ``yaw`` degrees at once (the chamber's hand, for a test)."""
    parrot.body.set("head_yaw", yaw)
    parrot.body.velocity[:] = 0.0


def look_at(thing: Thing, side: float = 0.0) -> float:
    """The head yaw that puts a thing at ``side`` degrees of head azimuth (+ left)."""
    return math.degrees(math.atan2(thing.x, thing.z)) - side


def distance_errors(parrot: Parrot, scene: Scene, records: list[dict[str, Any]],
                    eye: int | None = None) -> list[float]:
    """|log(read / true)| for the records with a thing under them and a distance read; a
    record with no distance read counts at the default distance."""
    out = []
    for r in records:
        if r["truth"] < 0 or (eye is not None and r["eye"] != eye):
            continue
        true = truth_distance(scene, r["truth"], r["origin"])
        read = r["distance"] if np.isfinite(r["distance"]) else float(parrot.genes["default"])
        out.append(abs(math.log(read / true)))
    return out


def held(parrot: Parrot, scene: Scene, margin: float) -> list[bool]:
    """For each thing: the map holds a thing within ``margin`` of its place."""
    places = [t["place"] for t in parrot.map.things() if t["place"] is not None]
    out = []
    for thing in scene.things:
        here = np.array([thing.x, thing.z])
        out.append(any(float(np.linalg.norm(np.asarray(p) - here)) <= margin for p in places))
    return out


def one_each(parrot: Parrot, scene: Scene, margin: float, alone: float) -> list[bool]:
    """For each thing: exactly one thing held within ``alone`` of its place, and that one
    within ``margin``."""
    places = [np.asarray(t["place"]) for t in parrot.map.things() if t["place"] is not None]
    out = []
    for thing in scene.things:
        here = np.array([thing.x, thing.z])
        near = [float(np.linalg.norm(p - here)) for p in places]
        close = [d for d in near if d <= alone]
        out.append(len(close) == 1 and close[0] <= margin)
    return out


def bearing_error(parrot: Parrot, scene: Scene) -> list[float]:
    """For each thing: the angle (degrees, from the perch) to the nearest held place."""
    places = [np.asarray(t["place"]) for t in parrot.map.things() if t["place"] is not None]
    out = []
    for thing in scene.things:
        if not places:
            out.append(180.0)
            continue
        here = np.array([thing.x, thing.z])
        nearest = min(places, key=lambda p: float(np.linalg.norm(p - here)))
        a = math.atan2(here[0], here[1]) - math.atan2(nearest[0], nearest[1])
        out.append(abs(math.degrees((a + math.pi) % (2.0 * math.pi) - math.pi)))
    return out


def live(parrot: Parrot, rng: np.random.Generator, scenes: int, moments: int,
         things: tuple[int, int] = (3, 6), panel: str = "stands") -> list[dict[str, Any]]:
    """The life: each scene's distance errors (both eyes, one eye) as it went."""
    curve = []
    for _ in range(scenes):
        scene = place_things(rng, int(rng.integers(things[0], things[1] + 1)), panel)
        parrot.reset()
        parrot.record = []
        for _ in range(moments):
            step(parrot, scene, learn=True)
        both = distance_errors(parrot, scene, parrot.record, eye=2)
        one = distance_errors(parrot, scene, [r for r in parrot.record if r["eye"] != 2])
        read = [r for r in parrot.record if r["truth"] >= 0]
        curve.append({"both": float(np.median(both)) if both else math.nan,
                      "one": float(np.median(one)) if one else math.nan,
                      "read": float(np.mean([np.isfinite(r["distance"]) for r in read]))
                      if read else math.nan,
                      "lessons": parrot.lessons})
    parrot.record = None
    return curve


def front(parrot: Parrot, scene: Scene, cover: str | None) -> list[dict[str, Any]]:
    """Each thing straight in front of the beak, two moments; the sightings of the second."""
    out = []
    for k, thing in enumerate(scene.things):
        parrot.reset()
        turn(parrot, look_at(thing))
        parrot.record = []
        step(parrot, scene, learn=False, act=False, cover=cover)
        parrot.record = []
        step(parrot, scene, learn=False, act=False, cover=cover)
        mine = [r for r in parrot.record if r["truth"] == k]
        for r in mine:
            r["true"] = truth_distance(scene, k, r["origin"])
            r["true_disparity"] = truth_disparity(scene, k, parrot)
        out += mine
    parrot.record = None
    return out


def test(parrot: Parrot, rng: np.random.Generator, *, scenes: int, moments: int, behind: int,
         margin: float, alone: float, panel: str = "stands") -> dict[str, Any]:
    """The test readings on fresh scenes (no lessons)."""
    seen, kept, same, bearing = [], [], [], []
    both_err, one_err, records = [], [], []
    for _ in range(scenes):
        scene = place_things(rng, int(rng.integers(3, 7)), panel)
        # E1: a look-around by the parrot's own reflex
        parrot.reset()
        for _ in range(moments):
            step(parrot, scene, learn=False)
        seen += held(parrot, scene, margin)
        bearing += bearing_error(parrot, scene)
        # E1, behind: each thing put in the blind spot for ``behind`` moments
        for k, thing in enumerate(scene.things):
            turn(parrot, look_at(thing, 180.0))
            for _ in range(behind):
                step(parrot, scene, learn=False, act=False)
            kept.append(held(parrot, scene, margin)[k])
        # E2: the left eye alone, the right eye alone, then both, then turns
        parrot.reset()
        for thing in scene.things:
            for side in (70.0, -70.0, 0.0):
                turn(parrot, look_at(thing, side))
                for _ in range(2):
                    step(parrot, scene, learn=False, act=False)
        for _ in range(moments // 2):
            step(parrot, scene, learn=False)
        same += one_each(parrot, scene, margin, alone)
        # E3: in front, both eyes and one eye covered
        b = front(parrot, scene, None)
        o = front(parrot, scene, "left") + front(parrot, scene, "right")
        both_err += [abs(math.log((r["distance"] if np.isfinite(r["distance"])
                                   else parrot.genes["default"]) / r["true"]))
                     for r in b if r["eye"] == 2]
        one_err += [abs(math.log((r["distance"] if np.isfinite(r["distance"])
                                  else parrot.genes["default"]) / r["true"])) for r in o]
        records += b
    return {"E1_seen": float(np.mean(seen)), "E1_behind": float(np.mean(kept)),
            "E2_one": float(np.mean(same)), "bearing": float(np.median(bearing)),
            "E3_both": float(np.median(both_err)) if both_err else math.nan,
            "E3_one": float(np.median(one_err)) if one_err else math.nan,
            "E3_both_count": len(both_err), "E3_one_count": len(one_err),
            "records": records}


def probe(records: list[dict[str, Any]], parrot: Parrot, bins: int = 8) -> dict[str, Any]:
    """Every probed neuron's tuning to the true distance and disparity, and its information
    about the distance (bits, from its activity in ``bins`` bins against the distance in
    ``bins`` bins)."""
    if not records:
        return {}
    acts = np.array([np.concatenate([r["cues"], r["units"], r["look"], r["kinds"]])
                     for r in records])
    true = np.array([r["true"] for r in records])
    disp = np.array([r["true_disparity"] for r in records])
    edges = np.quantile(np.log(true), np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, np.log(true), side="right") - 1, 0, bins - 1)
    info = np.zeros(acts.shape[1])
    tuning = np.zeros((acts.shape[1], bins))
    for n in range(acts.shape[1]):
        a = acts[:, n]
        for b in range(bins):
            tuning[n, b] = float(a[which == b].mean()) if (which == b).any() else 0.0
        if np.ptp(a) <= 1e-9:
            continue
        a_edges = np.quantile(a, np.linspace(0, 1, bins + 1))
        a_bin = np.clip(np.searchsorted(a_edges, a, side="right") - 1, 0, bins - 1)
        joint = np.zeros((bins, bins))
        np.add.at(joint, (a_bin, which), 1.0)
        joint /= joint.sum()
        pa, pd = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
        nz = joint > 0
        info[n] = float((joint[nz] * np.log2(joint[nz] / (pa @ pd)[nz])).sum())
    d_edges = np.quantile(disp, np.linspace(0, 1, bins + 1))
    d_which = np.clip(np.searchsorted(d_edges, disp, side="right") - 1, 0, bins - 1)
    d_tuning = np.array([[float(acts[d_which == b, n].mean()) if (d_which == b).any() else 0.0
                          for b in range(bins)] for n in range(acts.shape[1])])
    return {"info": info, "tuning": tuning, "disparity_tuning": d_tuning,
            "distance_bins": np.exp(0.5 * (edges[:-1] + edges[1:])).tolist(),
            "disparity_bins": (0.5 * (d_edges[:-1] + d_edges[1:])).tolist()}


def found(info: np.ndarray, parrot: Parrot, share: float = 0.5) -> np.ndarray:
    """The neurons that carry distance: information at ``share`` of the most or more, as
    indices into the brain's regions (the probe's order is cues, distance, look, kind)."""
    r = parrot.regions
    order = np.concatenate([np.arange(r["disparity"].start, r["elevation"].stop),
                            np.arange(r["distance"].start, r["distance"].stop),
                            np.arange(r["look"].start, r["look"].stop),
                            np.arange(r["kind"].start, r["kind"].stop)])
    return np.asarray(order[info >= share * info.max()])


def founder(seed: int, genes: dict[str, Any] | None = None, *, life: dict[str, Any] | None = None,
            tests: dict[str, Any] | None = None, silence: bool = True,
            table: bool = True) -> dict[str, Any]:
    """One founder's life and readings."""
    life = {**LIFE, **(life or {})}
    tests = {**TESTS, **(tests or {})}
    start = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 332, 1]))
    parrot = Parrot(genes, seed=seed)
    curve = live(parrot, rng, life["scenes"], life["moments"], tuple(life["things"]))
    test_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 332, 2]))
    reading = test(parrot, test_rng, **tests)
    records = reading.pop("records")
    out: dict[str, Any] = {"seed": int(seed), "curve": curve, "tests": reading,
                           "parameters": parrot.parameters(), "lessons": parrot.lessons,
                           "kinds": parrot.kinds.used, "genes": parrot.genes}
    tuned = probe(records, parrot)
    if tuned and silence:
        group = found(tuned["info"], parrot)
        out["found"] = {"units": group.tolist(), "regions": {
            name: int(((group >= s.start) & (group < s.stop)).sum())
            for name, s in parrot.regions.items()}}
        out["probe"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in tuned.items()}
        small = {**tests, "scenes": max(2, tests["scenes"] // 2)}
        parrot.silence(group)
        out["silenced"] = test(parrot, np.random.default_rng(
            np.random.SeedSequence([int(seed), 332, 2])), **small)
        out["silenced"].pop("records")
        pick = np.random.default_rng(np.random.SeedSequence([int(seed), 332, 3]))
        others = np.setdiff1d(np.arange(parrot.neurons), group)
        parrot.silence(pick.choice(others, size=len(group), replace=False))
        out["silenced_others"] = test(parrot, np.random.default_rng(
            np.random.SeedSequence([int(seed), 332, 2])), **small)
        out["silenced_others"].pop("records")
        parrot.silence(None)
        out["intact_small"] = test(parrot, np.random.default_rng(
            np.random.SeedSequence([int(seed), 332, 2])), **small)
        out["intact_small"].pop("records")
    if table:
        table_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 332, 4]))
        out["table"] = test(parrot, table_rng, **{**tests, "scenes": max(2, tests["scenes"] // 2)},
                            panel="table")
        out["table"].pop("records")
    out["seconds"] = time.perf_counter() - start
    out["work"] = {"distance_macs": parrot.distance.work["macs"],
                   "kinds_macs": parrot.kinds.work["macs"], "moments": parrot.moments}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--founders", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--genes", type=str, default="{}", help="JSON genes over the founder's")
    parser.add_argument("--scenes", type=int, default=LIFE["scenes"])
    parser.add_argument("--moments", type=int, default=LIFE["moments"])
    parser.add_argument("--tests", type=int, default=TESTS["scenes"])
    parser.add_argument("--no-silence", action="store_true")
    args = parser.parse_args()
    genes = json.loads(args.genes)
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in args.founders:
        result = founder(seed, genes, life={"scenes": args.scenes, "moments": args.moments},
                         tests={"scenes": args.tests}, silence=not args.no_silence)
        (args.out / f"founder-{seed}.json").write_text(json.dumps(result, default=float))
        t = result["tests"]
        print(f"founder {seed}: E1 {t['E1_seen']:.3f} behind {t['E1_behind']:.3f} "
              f"E2 {t['E2_one']:.3f} E3 both {t['E3_both']:.3f} one {t['E3_one']:.3f} "
              f"bearing {t['bearing']:.2f} lessons {result['lessons']} kinds {result['kinds']} "
              f"{result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["GENES", "LIFE", "NAMES", "TESTS", "founder", "live", "place_things", "probe", "test"]
