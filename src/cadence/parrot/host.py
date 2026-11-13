"""The parrot's brain in a web page: the host the page's worker runs in Pyodide.

The page's worker builds ``Host(config)`` and calls ``handle(message)`` for ``describe``,
``reset``, ``silence``, ``scene``, ``place``, ``move`` and ``remove``, and ``step(meta, arrays)``
once a moment (ten a second). The host keeps the scene the page shows and renders both eyes
from it with ``cadence.parrot.eyes`` (the same file as the chamber's), lives one moment of the
brain, moves the body's joints and answers with the angles to pose the rig, the things on the
map, the focus, the distance read for the thing looked at, and every drawn neuron's activity in
``buffers``. ``config["files"]`` is a folder that may hold ``brain.npz`` (``Parrot.save``) and
``groups.json`` (the chamber's probe: the neurons that carry distance and their tuning curves);
without them the brain is a newborn.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .brain import GENES, Parrot
from .eyes import HEIGHT, WIDTH, render
from .whole import PAGE, WholeParrot
from .world import NAMES, Body, Scene, Thing

AREAS = {"retina_l": "retina", "retina_r": "retina", "disparity": "wulst", "parallax": "wulst",
         "size_one": "rotundus", "elevation_one": "tectum", "size_both": "rotundus",
         "elevation_both": "tectum", "distance": "arco", "look": "rotundus", "kind": "rotundus",
         "focus": "tectum", "map": "map"}
TITLES = {"retina_l": "Left eye: what stands out from the room",
          "retina_r": "Right eye: what stands out from the room",
          "disparity": "Disparity units: the two eyes' views compared",
          "parallax": "Parallax units: one eye before and after the head moved",
          "size_one": "Size in one eye", "elevation_one": "Height in one eye's field",
          "size_both": "Size seen by both eyes", "elevation_both": "Height seen by both eyes",
          "distance": "Distance neurons (the reach)", "look": "The look: hue, saturation, light",
          "kind": "Kinds learned", "focus": "The focus: one winner", "map": "The map of things"}
POOL = 4  # pixels a side of the retina's drawn cells


class Host:
    """The page's brain host; see the module note."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = dict(config or {})
        folder = Path(str(config.get("files", ""))) if config.get("files") else None
        brain = folder / "brain.npz" if folder is not None else None
        # ``config["genes"]``: genes over the founder's for a newborn brain (the page's
        # ``{"map_cells": 32}``: a coarser map, half the map's cost)
        self.parrot = (Parrot.load(brain) if brain is not None and brain.exists()
                       else Parrot({**GENES, **dict(config.get("genes") or {})},
                                   seed=int(config.get("seed", 0))))
        self.newborn = not (brain is not None and brain.exists())
        found = folder / "groups.json" if folder is not None else None
        self.probe: dict[str, Any] = (json.loads(found.read_text()) if found is not None
                                      and found.exists() else {})
        self.learn = bool(config.get("learn", False))
        # the whole brain (``config["whole"]``): the association, the ear, the voice and the
        # dopamine join the first brain, and a pistachio (``meta["reward"]``) pays
        whole = config.get("whole")
        if whole == "page":
            from .voice import Voice

            self.whole: WholeParrot | None = WholeParrot(self.parrot, voice=Voice(), **PAGE)
        else:
            self.whole = WholeParrot(self.parrot) if whole else None
        self.scene = Scene([])
        self.cover: str | None = None
        self.silenced: set[str] = set()
        self.buffers: dict[str, np.ndarray] = {}
        r = self.parrot.regions
        sizes = [("retina_l", (HEIGHT // POOL) * (WIDTH // POOL)),
                 ("retina_r", (HEIGHT // POOL) * (WIDTH // POOL))]
        sizes += [(name, r[name].stop - r[name].start) for name in r if name not in ("map",)]
        sizes += [("map", self.parrot.map.count)]
        self.populations: list[dict[str, Any]] = []
        start = 0
        for name, size in sizes:
            self.populations.append({"name": name, "title": TITLES.get(name, name),
                                     "region": AREAS.get(name, name), "start": start,
                                     "size": int(size)})
            start += int(size)
        self.neurons = start
        self.pop = {p["name"]: p for p in self.populations}

    # -- commands

    def handle(self, message: dict[str, Any]) -> dict[str, Any]:
        command = message.get("command")
        if command == "describe":
            return self.describe()
        if command == "reset":
            return self.reset()
        if command == "silence":
            return self.silence(str(message.get("group")), bool(message.get("on")))
        if command == "scene":
            self.scene = Scene.from_json(message["scene"])
            return {"things": len(self.scene.things)}
        if command == "place":
            self.scene.things.append(Scene.from_json({"schema": "parrot-scene/1",
                                                      "things": [message["thing"]]}).things[0])
            return {"things": len(self.scene.things)}
        if command == "move":
            for t in self.scene.things:
                if t.id == str(message["id"]):
                    t.x, t.z = float(message["pos"][0]), float(message["pos"][2])
                    t.stand = float(message.get("stand", t.stand))
            return {"things": len(self.scene.things)}
        if command == "remove":
            self.scene.things = [t for t in self.scene.things if t.id != str(message["id"])]
            return {"things": len(self.scene.things)}
        raise ValueError(f"unknown command {command!r}")

    def describe(self) -> dict[str, Any]:
        reach = self.pop["distance"]
        group: dict[str, Any] = {
            "name": "distance", "title": "Distance neurons",
            "explain": ("Each of these neurons prefers one distance. They started with no "
                        "synapses at all and learned from the parrot's own head turns: a thing "
                        "stays where it is while the head turns, so the distance at which two "
                        "views of it agree is the one they learn."),
            "units": list(range(reach["start"], reach["start"] + reach["size"])),
            "variable": "distance", "unit": "m",
            "range": [float(v) for v in self.parrot.distance.span],
            "preferred": self.parrot.distance.preferred.tolist()}
        if self.probe:
            group["tuning"] = self.probe.get("tuning", {})
            group["found"] = self.probe.get("found", {})
        return {"name": "parrot", "placeholder": False, "newborn": self.newborn,
                "title": "The parrot's brain: two eyes, a map of things, distance learned",
                "joints": list(NAMES), "eye": {"w": WIDTH, "h": HEIGHT}, "rate_hz": 10,
                "populations": self.populations, "neurons": self.neurons,
                "synapses": int(self.parrot.distance.connectome.synapses
                                + self.parrot.kinds.connectome.synapses
                                + self.parrot.focus.connectome.synapses
                                + self.parrot.map.connectome.synapses),
                "parameters": self.parrot.parameters(), "groups": [group],
                "genes": {k: v for k, v in self.parrot.genes.items() if k != "wiring"}}

    def reset(self) -> dict[str, Any]:
        self.parrot.reset()
        return {"ok": True}

    def silence(self, group: str, on: bool) -> dict[str, Any]:
        (self.silenced.add if on else self.silenced.discard)(group)
        units: list[int] = []
        if "distance" in self.silenced:
            found = self.probe.get("found", {}).get("units")
            r = self.parrot.regions["distance"]
            units = list(found) if found else list(range(r.start, r.stop))
        self.parrot.silence(units)
        return {"silenced": sorted(self.silenced)}

    # -- the brain view's block

    def _viz(self, out: dict[str, Any], looked: int | None, frames: int = 16) -> dict[str, Any]:
        """The moment's settles as each region's residual at ``frames`` points, the focus with
        its inhibition of return, the modulators and the map (VIZ_EVENTS.md)."""
        p = self.parrot
        regions: dict[str, list[list[float]]] = {}
        sweeps = 0

        def add(part: dict[str, list[float]]) -> None:
            for k, v in part.items():
                regions.setdefault(k, []).append(v)

        if looked is not None:
            d = p.distance
            drive = np.zeros((1, d.neurons))
            drive[0, : d.width_in] = out["sightings"][looked].code
            rows = _settle_rows(d.graph, drive * d.mask, d.steps, d.mask)
            sweeps = max(sweeps, len(rows) - 1)
            r = p.regions
            cue = {name: np.arange(r[name].start, r[name].stop) - r["disparity"].start
                   for name in ("disparity", "parallax", "size_one", "elevation_one",
                                "size_both", "elevation_both")}
            add(_frames(rows, {"wulst": np.concatenate([cue["disparity"], cue["parallax"]]),
                               "rotundus": np.concatenate([cue["size_one"], cue["size_both"]]),
                               "tectum": np.concatenate([cue["elevation_one"],
                                                         cue["elevation_both"]]),
                               "arco": d.output}, frames))
            k = p.kinds
            look = np.asarray(out["looks"])[looked]
            drive = np.zeros((1, k.neurons))
            drive[0, k.feature] = k.drive * look
            rows = _settle_rows(k.graph, drive * k.mask, k.steps, k.mask)
            add(_frames(rows, {"rotundus": np.arange(k.neurons)}, frames))
        f = p.focus
        if f.drive.any():
            rows = _settle_rows(f.graph, f.drive, f.steps, f.mask)
            sweeps = max(sweeps, len(rows) - 1)
            add(_frames(rows, {"tectum": np.arange(f.slots)}, frames))
        if self.whole is not None:
            brain = self.whole.association
            graph = brain.learner.brain
            drive = brain.stimulus(self.whole._observation())
            state = graph.settle_batch(drive, steps=24, tolerance=None, trajectory=True)
            rows = np.concatenate([np.zeros((1, brain.connectome.n)),
                                   np.asarray(state.trajectory)[:, 0, :]])
            sweeps = max(sweeps, len(rows) - 1)
            pops = brain.connectome.populations
            atlas = {"association": "ncl", "prefrontal": "trace", "history": "trace",
                     "recall": "records", "next": "records", "recall_1": "heads",
                     "next_1": "heads", "motor": "arco"}
            index: dict[str, list[int]] = {}
            for name, aid in atlas.items():
                if name in pops:
                    index.setdefault(aid, []).extend(int(v) for v in pops[name])
            add(_frames(rows, {k: np.asarray(v) for k, v in index.items()}, frames))
        settle = {"frames": frames, "sweeps": int(sweeps), "kind": "residual",
                  "regions": {k: np.mean(v, axis=0).round(4).tolist() for k, v in regions.items()}}
        focus = p.focused
        returned = [[float(c.place[0]), float(c.place[1]),
                     round(math.exp(-(p.moments - c.focused) / f.return_time), 3)]
                    for c in f.candidates if 0 < p.moments - c.focused < 4 * f.return_time]
        surprise: dict[str, Any] = {
            "units": [], "regions": {"tectum": round(max((c.surprise for c in f.candidates),
                                                         default=0.0), 3)},
            "focus": None if focus is None else {
                "place": [float(v) for v in focus.place], "why": focus.why,
                "held": int(f.current.held) if f.current is not None else 0,
                "returned": returned}}
        dopamine = 0.0 if self.whole is None else float(np.clip(self.whole.dopamine, -1, 1))
        modulators = {
            "dopamine": {"level": dopamine, "why": "pistachio" if self.whole is not None and
                         self.whole.reward > 0 else "prediction",
                         "paid": [] if self.whole is None else self.whole.paid},
            "acetylcholine": {"level": 1.0 if focus is not None else 0.0,
                              "at": None if focus is None else [float(v) for v in focus.place],
                              "targets": {"tectum": 1}}}
        things = [{"place": [float(v) for v in t["place"]], "seen": t["state"] == "seen"}
                  for t in p.map.things() if t["place"] is not None]
        return {"v": 1, "settle": settle, "surprise": surprise, "modulators": modulators,
                "map": {"things": things, "pose": {"heading": float(p.body.joint("head_yaw"))}}}

    # -- a moment

    def _looked_at(self, sightings: list[Any]) -> int | None:
        """The sighting nearest the beak's direction, a thing both eyes see first."""
        if not sightings:
            return None
        rotation = self.parrot.body.eyes().rotation
        best, key = None, (2, math.inf)
        for i, s in enumerate(sightings):
            head = rotation.T @ s.direction
            angle = abs(math.atan2(float(head[0]), float(head[2])))
            here = (0 if s.eye == 2 else 1, angle)
            if here < key:
                best, key = i, here
        return best

    def step(self, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        began = time.perf_counter()
        if meta.get("scene") is not None:
            self.scene = Scene.from_json(meta["scene"])
        cue = meta.get("cue")
        if isinstance(cue, dict) and cue.get("az") is not None:
            # the visitor clicked a thing: the focus goes to it at once
            self.parrot.point(float(cue["az"]), float(cue.get("el") or 0.0))
        cover = meta.get("cover", self.cover)
        size = HEIGHT * WIDTH * 3
        given = [arrays.get("left"), arrays.get("right")]
        if all(a is not None and a.size == size for a in given):
            images = np.stack([np.asarray(a, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
                               for a in given])
            if cover in ("left", "right"):
                images[("left", "right").index(cover)] = 0
        else:
            images = render(self.scene, self.parrot.body.eyes(), cover=cover).images
        rendered = time.perf_counter()
        learn = bool(meta.get("learn", self.learn))
        reward = meta.get("reward")
        if self.whole is not None:
            audio = arrays.get("audio")
            heard = None if audio is None or audio.size != 1600 else np.asarray(audio, np.float64)
            whole = self.whole.moment(images, samples=heard, learn=learn,
                                      reward=None if reward is None else float(reward))
            out = whole["first"]
        else:
            out = self.parrot.moment(images, learn=learn)
        lived = time.perf_counter()
        sightings = out["sightings"]
        looked = self._looked_at(sightings)
        activity = np.zeros(self.neurons, dtype=np.float32)
        weights = self.parrot.vision.weights
        for e, name in ((0, "retina_l"), (1, "retina_r")):
            pooled = weights[e][: HEIGHT // POOL * POOL, : WIDTH // POOL * POOL].reshape(
                HEIGHT // POOL, POOL, WIDTH // POOL, POOL).mean(axis=(1, 3))
            p = self.pop[name]
            activity[p["start"]: p["start"] + p["size"]] = pooled.reshape(-1)
        readout: dict[str, Any] = {"distance": None, "looked_at": None}
        group_activity: dict[str, list[float]] = {"distance": []}
        r = self.parrot.regions
        if looked is not None:
            s = sightings[looked]
            row = out["reach"][looked]
            cues = self.parrot.distance.width_in
            for name in ("disparity", "parallax", "size_one", "elevation_one", "size_both",
                         "elevation_both"):
                p, a = self.pop[name], r[name].start - r["disparity"].start
                activity[p["start"]: p["start"] + p["size"]] = row[a: a + p["size"]]
            p = self.pop["distance"]
            reach = row[cues: cues + p["size"]]
            activity[p["start"]: p["start"] + p["size"]] = reach
            group_activity["distance"] = [float(v) for v in reach]
            kinds = out["kind_activity"][looked]
            for name, part in (("look", self.parrot.kinds.feature),
                               ("kind", self.parrot.kinds.category)):
                p = self.pop[name]
                activity[p["start"]: p["start"] + p["size"]] = kinds[part]
            readout = {"distance": float(s.distance) if np.isfinite(s.distance) else None,
                       "looked_at": {"place": None if s.place is None else
                                     [float(v) for v in s.place],
                                     "kind": s.kind, "both_eyes": s.eye == 2}}
        focus = self.parrot.focused
        if focus is not None:
            p = self.pop["focus"]
            activity[p["start"]: p["start"] + len(focus.activity)] = focus.activity
        p = self.pop["map"]
        sheet = self.parrot.map.potential.sum(axis=1)
        activity[p["start"]: p["start"] + p["size"]] = sheet / max(float(sheet.max()), 1e-9)
        angles = self.parrot.body.angles.astype(np.float32)
        self.buffers = {"activity": activity, "angles": angles,
                        "targets": Body.scaled(self.parrot.body.targets).astype(np.float32)}
        names = ["activity", "angles", "targets"]
        if meta.get("show_eyes"):
            self.buffers["left"], self.buffers["right"] = images[0].ravel(), images[1].ravel()
            names += ["left", "right"]
        viz = self._viz(out, looked) if meta.get("viz") else None
        things = [{"place": None if t["place"] is None else [float(v) for v in t["place"]],
                   "kind": t["kind"], "state": t["state"], "since": int(t["since"])}
                  for t in self.parrot.map.things()]
        return {"buffers": names, "group_activity": group_activity, "readout": readout,
                "focus": None if focus is None else {
                    "place": [float(v) for v in focus.place], "kind": int(focus.kind),
                    "why": focus.why, "activity": [float(v) for v in focus.activity]},
                "things": things, "silenced": sorted(self.silenced),
                "reward": None if self.whole is None else self.whole.reward,
                "vta": None if self.whole is None else self.whole.dopamine,
                "paid": [] if self.whole is None else self.whole.paid,
                "said": None if self.whole is None else self.whole.said,
                "vocalized": None if self.whole is None else self.whole.vocalized,
                "ms": {"render": 1000.0 * (rendered - began),
                       "brain": 1000.0 * (lived - rendered),
                       "host": 1000.0 * (time.perf_counter() - lived)},
                "newborn": self.newborn, "viz": viz}


def _frames(rows: np.ndarray, index: dict[str, np.ndarray], frames: int) -> dict[str, list[float]]:
    """Each region's residual (mean distance of its neurons from the settled state) at ``frames``
    points spread over a settle's trajectory ``(sweeps + 1, neurons)``."""
    pick = np.linspace(0, len(rows) - 1, frames).round().astype(int)
    final = rows[-1]
    return {k: np.abs(rows[pick][:, i] - final[i]).mean(axis=1).round(4).tolist()
            for k, i in index.items() if len(i)}


def _settle_rows(graph: Any, drive: np.ndarray, steps: int, mask: np.ndarray | None = None
                 ) -> np.ndarray:
    state = graph.settle_batch(drive, steps=steps, tolerance=None, mask=mask, trajectory=True)
    rows = np.asarray(state.trajectory)[:, 0, :]
    return np.concatenate([np.zeros((1, rows.shape[1])), rows])


def scene_of(things: list[Thing]) -> dict[str, Any]:
    """A scene in the page's format, for a test or a script."""
    return Scene(list(things)).to_json()


__all__ = ["AREAS", "Host", "scene_of"]
