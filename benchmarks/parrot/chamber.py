"""The parrot's first chamber (#332): things around a low perch, two eyes, a map and distance.

A founder is one seed: its world (the scenes of its life and of its tests) and its brain. A life
is ``scenes`` scenes of ``moments`` moments: things are placed around the perch, the parrot looks
around by its own reflex and learns. No distance, place or kind is ever given to the brain; the
chamber keeps the truth for the readings only.

Readings, on fresh test scenes after the life (no lessons):

- **E1, seen and held**: after a look-around, the share of things the map holds within
  ``margin`` of their place; then each thing so held that the neck can put in the blind spot,
  there for ``behind`` moments: the share held there after them.
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
from cadence.parrot.world import HIGH, LOW, NAMES, Scene, Thing, stand_ins

TABLE = stand_ins()
KINDS = sorted(TABLE["kinds"])
LIFE = {"scenes": 48, "moments": 50, "things": (3, 6)}
TESTS = {"scenes": 10, "moments": 60, "behind": 20, "margin": 0.10, "alone": 0.15}


def place_things(rng: np.random.Generator, count: int, panel: str = "stands") -> Scene:
    """``count`` things around the perch: 0.22 to 0.6 m from it, 0.18 m apart or more, on the
    table; ``panel`` "stands" puts each on a clear stand 0 to 0.25 m high at a scale of 0.7 to
    1.4, "table" puts each on the table."""
    out: list[Thing] = []
    while len(out) < count:
        a = float(rng.uniform(-math.pi, math.pi))
        r = float(rng.uniform(0.22, 0.6))
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


def truth_disparity(scene: Scene, thing: int, record: dict[str, Any]) -> float:
    """The true disparity of a thing's middle (degrees) at a record's pose: its head azimuth
    from the right eye less that from the left eye."""
    places, rotation = np.asarray(record["eyes"]), np.asarray(record["rotation"])
    c = scene.things[thing].center(TABLE)
    out = []
    for e in (0, 1):
        local = rotation.T @ (c - places[e])
        out.append(math.atan2(local[0], local[2]))
    return math.degrees(out[1] - out[0])


def truths(scene: Scene, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records with a thing under them, each given its true distance and disparity."""
    out = []
    for r in records:
        if r["truth"] < 0:
            continue
        r["true"] = truth_distance(scene, r["truth"], r["origin"])
        r["true_disparity"] = truth_disparity(scene, r["truth"], r)
        out.append(r)
    return out


RENDER = {"seconds": 0.0, "count": 0}


def step(parrot: Parrot, scene: Scene, *, learn: bool, act: bool = True,
         cover: str | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    sight = render(scene, parrot.body.eyes(), cover=cover)
    RENDER["seconds"] += time.perf_counter() - start
    RENDER["count"] += 1
    return parrot.moment(sight.images, learn=learn, ids=sight.ids, act=act)


def turn(parrot: Parrot, yaw: float) -> None:
    """Put the head at ``yaw`` degrees at once (the chamber's hand, for a test)."""
    parrot.body.set("head_yaw", yaw)
    parrot.body.velocity[:] = 0.0


def look_at(thing: Thing, side: float = 0.0) -> float | None:
    """The head yaw that puts a thing at ``side`` degrees of head azimuth (+ left), or None
    where the neck cannot turn that far."""
    yaw = (math.degrees(math.atan2(thing.x, thing.z)) - side + 180.0) % 360.0 - 180.0
    k = NAMES.index("head_yaw")
    return yaw if LOW[k] <= yaw <= HIGH[k] else None


def distance_errors(parrot: Parrot, scene: Scene, records: list[dict[str, Any]],
                    eye: int | None = None) -> list[float]:
    """|log(read / true)| for the records with a thing under them and a distance read; a
    record with no distance read counts at the default distance."""
    out = []
    for r in records:
        if r["truth"] < 0 or (eye is not None and r["eye"] != eye):
            continue
        true = truth_distance(scene, r["truth"], r["origin"])
        read = r["distance"] if np.isfinite(r["distance"]) else float(parrot.genes["place_default"])
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


def gap_to(parrot: Parrot, thing: Thing) -> float:
    """The distance from a thing to the nearest place the map holds (inf for an empty map)."""
    here = np.array([thing.x, thing.z])
    places = [np.asarray(t["place"]) for t in parrot.map.things() if t["place"] is not None]
    return min((float(np.linalg.norm(p - here)) for p in places), default=math.inf)


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
        curve.append({"errors": both,
                      "both": float(np.median(both)) if both else math.nan,
                      "one": float(np.median(one)) if one else math.nan,
                      "read": float(np.mean([np.isfinite(r["distance"]) for r in read]))
                      if read else math.nan,
                      "lessons": parrot.lessons})
    parrot.record = None
    return curve


def front(parrot: Parrot, scene: Scene, cover: str | None) -> list[dict[str, Any]]:
    """Each thing the neck can bring in front of the beak, there for two moments; the
    sightings of the second."""
    out = []
    for k, thing in enumerate(scene.things):
        yaw = look_at(thing)
        if yaw is None:
            continue
        parrot.reset()
        turn(parrot, yaw)
        parrot.record = []
        step(parrot, scene, learn=False, act=False, cover=cover)
        parrot.record = []
        step(parrot, scene, learn=False, act=False, cover=cover)
        out += [r for r in truths(scene, parrot.record) if r["truth"] == k]
    parrot.record = None
    return out


def test(parrot: Parrot, rng: np.random.Generator, *, scenes: int, moments: int, behind: int,
         margin: float, alone: float, panel: str = "stands", front_panel: bool = True
         ) -> dict[str, Any]:
    """The test readings on fresh scenes (no lessons); ``front_panel`` False leaves out the
    panel of things in front with an eye covered (the fitness never reads it)."""
    seen, kept, same, bearing = [], [], [], []
    both_err, one_err, records = [], [], []
    details: list[dict[str, Any]] = []
    after_gaps: list[float] = []
    for _ in range(scenes):
        scene = place_things(rng, int(rng.integers(3, 7)), panel)
        # E1: a look-around by the parrot's own reflex
        parrot.reset()
        parrot.record = []
        for _ in range(moments):
            step(parrot, scene, learn=False)
        looked_here = truths(scene, parrot.record)
        records += looked_here
        parrot.record = None
        seen += held(parrot, scene, margin)
        places = [(np.asarray(t["place"]), t["kind"]) for t in parrot.map.things()
                  if t["place"] is not None]
        for k, thing in enumerate(scene.things):
            here = np.array([thing.x, thing.z])
            gaps = sorted(float(np.linalg.norm(p - here)) for p, _ in places)
            mine = [r for r in looked_here if r["truth"] == k]
            details.append({"gap": gaps[0] if gaps else math.inf,
                            "near": sum(g <= alone for g in gaps),
                            "both": sum(r["eye"] == 2 for r in mine), "one": sum(
                                r["eye"] != 2 for r in mine),
                            "kinds": sorted({r["kind"] for r in mine if r["kind"] is not None}),
                            "true": math.hypot(thing.x, thing.z), "kind": thing.kind,
                            "entries": len(places), "things": len(scene.things)})
        bearing += bearing_error(parrot, scene)
        # E1, behind: each thing the neck can put in the blind spot, there for ``behind``
        # moments
        before = held(parrot, scene, margin)
        for k, thing in enumerate(scene.things):
            yaw = look_at(thing, 180.0)
            if yaw is None or not before[k]:
                continue
            turn(parrot, yaw)
            for _ in range(behind):
                step(parrot, scene, learn=False, act=False)
            kept.append(held(parrot, scene, margin)[k])
            after_gaps.append(gap_to(parrot, thing))
        # E2: the left eye alone, the right eye alone and both, as far as the neck turns (two
        # views or more), then the parrot's own looking
        parrot.reset()
        tested = []
        for k, thing in enumerate(scene.things):
            views = [y for y in (look_at(thing, side) for side in (70.0, -70.0, 0.0))
                     if y is not None]
            if len(views) < 2:
                continue
            tested.append(k)
            for yaw in views:
                turn(parrot, yaw)
                for _ in range(2):
                    step(parrot, scene, learn=False, act=False)
        for _ in range(moments // 2):
            step(parrot, scene, learn=False)
        alone_held = one_each(parrot, scene, margin, alone)
        same += [alone_held[k] for k in tested]
        # E3: in front, both eyes and one eye covered
        if not front_panel:
            continue
        b = front(parrot, scene, None)
        o = front(parrot, scene, "left") + front(parrot, scene, "right")
        both_err += [abs(math.log((r["distance"] if np.isfinite(r["distance"])
                                   else parrot.genes["place_default"]) / r["true"]))
                     for r in b if r["eye"] == 2]
        one_err += [abs(math.log((r["distance"] if np.isfinite(r["distance"])
                                  else parrot.genes["place_default"]) / r["true"])) for r in o]
        records += [r for r in b if r["eye"] == 2]
    looked = [r for r in records if r["eye"] == 2]
    looked_one = [r for r in records if r["eye"] != 2]

    def median_error(rs: list[dict[str, Any]]) -> float:
        if not rs:
            return math.nan
        return float(np.median([abs(math.log((r["distance"] if np.isfinite(r["distance"]) else
                                              parrot.genes["place_default"]) / r["true"]))
                                for r in rs]))

    return {"E3_looked_one": median_error(looked_one),
            "gap_seen": float(np.median([min(d["gap"], 0.3) for d in details])),
            "gap_behind": float(np.median([min(g, 0.3) for g in after_gaps])) if after_gaps
            else math.nan,"E1_seen": float(np.mean(seen)), "E1_behind": float(np.mean(kept)) if kept
            else math.nan, "E1_behind_count": len(kept), "E2_one": float(np.mean(same)) if same
            else math.nan, "E2_count": len(same), "bearing": float(np.median(bearing)),
            "E3_both": float(np.median(both_err)) if both_err else math.nan,
            "E3_one": float(np.median(one_err)) if one_err else math.nan,
            "E3_both_count": len(both_err), "E3_one_count": len(one_err),
            "E3_looked": float(np.median([abs(math.log((r["distance"] if np.isfinite(
                r["distance"]) else parrot.genes["place_default"]) / r["true"])) for r in looked]))
            if looked else math.nan, "E3_looked_count": len(looked),
            "details": details, "records": records}


def information(a: np.ndarray, which: np.ndarray, bins: int) -> float:
    """The information (bits) between a neuron's activity in ``bins`` quantile bins and the
    binned variable ``which``."""
    if np.ptp(a) <= 1e-9:
        return 0.0
    a_edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    a_bin = np.clip(np.searchsorted(a_edges, a, side="right") - 1, 0, len(a_edges) - 2)
    joint = np.zeros((len(a_edges) - 1, int(which.max()) + 1))
    np.add.at(joint, (a_bin, which), 1.0)
    joint /= joint.sum()
    pa, pd = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pa @ pd)[nz])).sum())


def probe_panel(parrot: Parrot, rng: np.random.Generator, things: int = 24,
                steps: int = 8) -> list[dict[str, Any]]:
    """The probe's own panel: one thing at a time in front of the beak, at ``steps`` distances
    from 0.22 to 0.6 m in a shuffled order, each time at a new scale and on a new stand, so
    that only its distance and disparity follow how far it is; the sightings by both eyes."""
    out: list[dict[str, Any]] = []
    for _ in range(things):
        kind = KINDS[int(rng.integers(len(KINDS)))]
        bearing = float(rng.uniform(-math.radians(60.0), math.radians(60.0)))
        for far in rng.permutation(np.linspace(0.22, 0.6, steps)):
            thing = Thing("probe", kind, float(far) * math.sin(bearing),
                          float(far) * math.cos(bearing), float(rng.uniform(0.0, 6.28)),
                          float(rng.uniform(0.7, 1.4)), float(rng.uniform(0.0, 0.25)))
            scene = Scene([thing])
            yaw = look_at(thing)
            if yaw is None:
                continue
            parrot.reset()
            turn(parrot, yaw)
            parrot.record = []
            step(parrot, scene, learn=False, act=False)
            parrot.record = []
            step(parrot, scene, learn=False, act=False)
            out += [r for r in truths(scene, parrot.record) if r["eye"] == 2]
    parrot.record = None
    return out


def probe(records: list[dict[str, Any]], bins: int = 6, shuffles: int = 5,
          seed: int = 0) -> dict[str, Any]:
    """Every probed neuron's tuning to the true distance and disparity, and its information
    about the distance (bits, less the mean information with the distances shuffled), over the
    sightings by both eyes."""
    records = [r for r in records if r["eye"] == 2]
    if len(records) < 4 * bins:
        return {}
    rng = np.random.default_rng(seed)
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
        base = np.mean([information(a, rng.permutation(which), bins) for _ in range(shuffles)])
        info[n] = information(a, which, bins) - float(base)
    d_edges = np.quantile(disp, np.linspace(0, 1, bins + 1))
    d_which = np.clip(np.searchsorted(d_edges, disp, side="right") - 1, 0, bins - 1)
    d_tuning = np.array([[float(acts[d_which == b, n].mean()) if (d_which == b).any() else 0.0
                          for b in range(bins)] for n in range(acts.shape[1])])
    return {"info": info, "tuning": tuning, "disparity_tuning": d_tuning, "samples": len(records),
            "distance_bins": np.exp(0.5 * (edges[:-1] + edges[1:])).tolist(),
            "disparity_bins": (0.5 * (d_edges[:-1] + d_edges[1:])).tolist()}


def found(info: np.ndarray, parrot: Parrot, share: float = 0.5) -> np.ndarray:
    """The neurons that carry distance: information at ``share`` of the most or more, as
    indices into the brain's regions (the probe's order is cues, distance, look, kind)."""
    r = parrot.regions
    order = np.concatenate([np.arange(r["disparity"].start, r["elevation_both"].stop),
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
    reading.pop("records")
    quarter = max(1, len(curve) // 4)
    first = [e for c in curve[:quarter] for e in c.pop("errors")]
    for c in curve[quarter:-quarter]:
        c.pop("errors")
    last = [e for c in curve[-quarter:] for e in c.pop("errors", [])]
    reading["E3_first_quarter"] = float(np.median(first)) if first else math.nan
    reading["E3_last_quarter"] = float(np.median(last)) if last else math.nan
    out: dict[str, Any] = {"seed": int(seed), "curve": curve, "tests": reading,
                           "parameters": parrot.parameters(), "lessons": parrot.lessons,
                           "kinds": parrot.kinds.used, "genes": parrot.genes}
    panel = probe_panel(parrot, np.random.default_rng(np.random.SeedSequence([int(seed), 332,
                                                                               5])))
    tuned = probe(panel, seed=seed)
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
        others = np.setdiff1d(parrot.settling, group)
        parrot.silence(pick.choice(others, size=min(len(group), len(others)), replace=False))
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
    out["work"] = parrot.cost()
    out["work"]["render_ms"] = 1000.0 * RENDER["seconds"] / max(RENDER["count"], 1)
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
              f"looked {t['E3_looked']:.3f} "
              f"bearing {t['bearing']:.2f} lessons {result['lessons']} kinds {result['kinds']} "
              f"{result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["GENES", "LIFE", "NAMES", "TESTS", "founder", "live", "place_things", "probe", "test"]
