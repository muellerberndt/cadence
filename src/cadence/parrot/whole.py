"""The parrot's whole brain, every region present, and its export for a 3D view.

``WholeParrot`` joins the first brain (``Parrot``: the eyes' figures, the Wulst's units, the
reach, the kinds, the focus and the map of things) to the regions the later steps teach, each
present at its founder and marked untrained until a chamber teaches it:

- **the association** (the caudolateral nidopallium), its **working trace** and **heard
  history**, a **recall memory with two heads**, and **a motor** with a slot a joint and a slot
  for the voice, composed by ``Brain.compose`` with the actor and critic whose prediction error
  is the dopamine (``arousal=True``);
- **the ear**: the speech front end's log-mel frames of the moment (``cadence.speech``) as the
  association's heard input;
- **the words**: a lexicon (``cadence.words.Lexicon``) whose word neurons join the kinds.

The association hears, each moment, the ear's frames, the focus's kind and place, the reach's
distance neurons and the body's joint readings. ``export(folder)`` writes every neuron's region
and place in the atlas's boxes, each tract's synapse count with a sample, and, for each moment
recorded with ``record``, every neuron's activity at each sweep of that moment's settles, in
the order the parts settle (the format of the parrot's interfaces, D12).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .brain import Parrot
from .world import NAMES

# the atlas's regions: id, centre and half extents (the parrot's 3D atlas), and whether a region
# is drawn on both sides of the head
ATLAS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], bool]] = {
    "retina": ((2.7, -0.15, 1.1), (0.12, 0.55, 0.55), True),
    "tectum": ((1.35, -0.65, -0.35), (0.55, 0.5, 0.6), True),
    "rotundus": ((1.15, 0.0, 0.55), (0.35, 0.3, 0.45), True),
    "wulst": ((0.6, 1.35, 0.95), (0.4, 0.18, 0.5), True),
    "place": ((0.45, 1.05, -0.25), (0.2, 0.15, 0.2), False),
    "map": ((0.0, 1.25, -0.55), (0.55, 0.22, 0.55), False),
    "records": ((-0.6, 1.1, -1.0), (0.35, 0.25, 0.3), False),
    "heads": ((-0.95, 0.85, -0.85), (0.18, 0.15, 0.18), False),
    "ncl": ((1.25, 0.55, -0.95), (0.4, 0.4, 0.35), True),
    "trace": ((1.45, 0.95, -1.15), (0.2, 0.2, 0.2), True),
    "words": ((-1.35, 0.45, -0.45), (0.25, 0.22, 0.25), False),
    "ear": ((2.25, -0.75, -0.9), (0.12, 0.2, 0.2), True),
    "vocal": ((1.2, 0.15, 0.25), (0.25, 0.2, 0.3), True),
    "arco": ((1.0, -0.35, -0.75), (0.3, 0.22, 0.3), True),
    "body": ((0.0, -1.25, -2.4), (0.25, 0.18, 0.25), False),
    "striatum": ((0.7, -0.25, 0.95), (0.3, 0.22, 0.3), True),
    "vta": ((0.0, -0.95, -0.4), (0.22, 0.15, 0.2), False),
}
MELS, FRAMES = 40, 10  # the ear: 40 bands, ten frames of 10 ms a moment
VOICE = 27  # the voice's slot: a letter or a space
PLACES = 8  # tuned units an axis of the focus's place
DEFAULT_SLOTS = {"head_yaw": 7, "head_pitch": 7, "head_roll": 3, "neck_pitch": 3,
                 "beak_open": 3, "wing_l": 3, "wing_r": 3, "grip_l": 3, "grip_r": 3, "leg_l": 3,
                 "leg_r": 3, "voice": VOICE}


class WholeParrot:
    """The first brain and the regions of the later steps, as one brain; see the module note."""

    def __init__(self, parrot: Parrot | None = None, *, seed: int = 0, association: int = 256,
                 words: int = 32, backend: Any = "cpu"):
        from ..generic import Brain
        from ..words.lexicon import Lexicon

        self.parrot = parrot if parrot is not None else Parrot(seed=seed, backend=backend)
        p = self.parrot
        self.inputs = {"ear": MELS * FRAMES, "focus": p.kinds.capacity, "place": 2 * PLACES,
                       "reach": p.distance.units, "body": 2 * len(NAMES)}
        self.slots = dict(DEFAULT_SLOTS)
        self.association = Brain.compose(
            inputs=sum(self.inputs.values()), actions=sum(self.slots.values()),
            slots=list(self.slots.values()), modules=(int(association),), recall_heads=2,
            arousal=True, seed=seed, backend=backend)
        self.lexicon = Lexicon({"kind": p.kinds.capacity}, capacity=int(words), backend=backend)
        self.ear = np.zeros((FRAMES, MELS))
        self._tail = np.zeros(0)
        self.recorded: list[dict[str, Any]] = []
        self._started = False
        self.said: str | None = None
        self.reward = 0.0
        self.dopamine = 0.0  # the critic's prediction error at the latest outcome
        self.paid: list[dict[str, Any]] = []  # the tracts the outcome's dopamine changed

    # -- the ear

    def hear(self, samples: np.ndarray | None) -> np.ndarray:
        """The moment's 1,600 samples at 16 kHz as ten log-mel frames ``(10, 40)``: the frames
        run every 160 samples across moments (the stream keeps its last 240 samples)."""
        from ..speech.frontend import log_mel

        given = np.zeros(1600) if samples is None else np.asarray(samples, dtype=np.float64)
        stream = np.concatenate([self._tail, given])
        frames = log_mel(stream) if len(stream) >= 400 else np.full((1, MELS), math.log(1e-6))
        self._tail = stream[-240:]
        out = np.full((FRAMES, MELS), math.log(1e-6))
        out[-min(FRAMES, len(frames)):] = frames[-FRAMES:]
        self.ear = out
        return out

    # -- one moment of the whole brain

    def _observation(self) -> np.ndarray:
        p = self.parrot
        parts = []
        ear = (self.ear - math.log(1e-6)) / 20.0
        parts.append(np.clip(ear, 0.0, 1.0).reshape(-1))
        focus = np.zeros(self.inputs["focus"])
        place = np.zeros(self.inputs["place"])
        if p.focused is not None:
            focus[int(p.focused.kind)] = 1.0
            grid = np.linspace(-0.8, 0.8, PLACES)
            place[:PLACES] = np.exp(-0.5 * ((p.focused.place[0] - grid) / 0.2) ** 2)
            place[PLACES:] = np.exp(-0.5 * ((p.focused.place[1] - grid) / 0.2) ** 2)
        parts += [focus, place]
        reach = np.zeros(self.inputs["reach"])
        rows = p.last.get("reach")
        if rows is not None and len(rows):
            reach = np.maximum(np.asarray(rows)[:, p.distance.output].max(axis=0), 0.0)
        parts.append(reach)
        parts.append(0.5 * (p.body.proprioception() + 1.0))
        return np.concatenate(parts)[None, :]

    def moment(self, images: np.ndarray, *, samples: np.ndarray | None = None,
               reward: float | None = None, learn: bool = False, record: str | None = None,
               ids: np.ndarray | None = None) -> dict[str, Any]:
        """One moment of the whole brain: the first brain on the eyes' images, the ear on the
        moment's samples, then the association's settle and its act (the joints and the voice);
        ``reward`` (a pistachio) is the outcome of the association's last act. ``record`` names
        the moment and keeps every neuron's activity at each sweep for ``export``."""
        p = self.parrot
        before = self._snapshot() if record else None
        trace: list[np.ndarray] = []
        if record:
            assert before is not None
            trace.append(before.copy())
        out = p.moment(images, learn=learn, ids=ids)
        self.hear(samples)
        observation = self._observation()
        brain = self.association
        start_state = None
        if record:
            start_state = brain.stimulus(observation)
        weights = np.asarray(brain.learner.brain.efficacy).copy()
        if self._started:
            action = brain.live(observation,
                                reward=np.array([0.0 if reward is None else float(reward)]))
        else:
            action, self._started = brain.live(observation), True
        chosen = np.asarray(action).reshape(-1)
        report = brain.last_learning or {}
        self.reward = 0.0 if reward is None else float(reward)
        self.dopamine = float(report.get("td_error", 0.0))
        self.paid = self._paid(weights, np.asarray(brain.learner.brain.efficacy))
        letters = "abcdefghijklmnopqrstuvwxyz "
        voice = int(chosen[list(self.slots).index("voice")]) if chosen.size == len(
            self.slots) else 0
        self.said = None if voice == VOICE - 1 else letters[voice]
        if record:
            assert start_state is not None
            self._record(record, trace, start_state, out)
        return {"first": out, "action": chosen, "said": self.said}

    def _paid(self, before: np.ndarray, after: np.ndarray, most: int = 8
              ) -> list[dict[str, Any]]:
        """The association's tracts the moment's outcome changed (eligibility times dopamine):
        each from-and-to pair of its regions with its summed change, the largest ``most``."""
        change = after - before
        if not np.any(change):
            return []
        pops = self.association.connectome.populations
        owner = np.full(self.association.connectome.n, -1)
        names = [n for n in pops if "/" not in n]
        for k, name in enumerate(names):
            owner[np.asarray(pops[name], dtype=np.int64)] = k
        pre = owner[np.asarray(self.association.connectome.pre)]
        post = owner[np.asarray(self.association.connectome.post)]
        moved = np.nonzero(change)[0]
        out: dict[tuple[str, str], float] = {}
        for a, b, c in zip(pre[moved], post[moved], change[moved], strict=True):
            if a < 0 or b < 0:
                continue
            key = (names[int(a)], names[int(b)])
            out[key] = out.get(key, 0.0) + float(c)
        ranked = sorted(out.items(), key=lambda item: -abs(item[1]))[:most]
        return [{"from": a, "to": b, "change": v} for (a, b), v in ranked]

    # -- the layout of every neuron

    def layout(self) -> list[dict[str, Any]]:
        """The brain's regions in neuron order: each part's own regions, with the atlas region
        each stands in and whether it is learned, born or untrained."""
        p = self.parrot
        r = p.regions
        rows: list[dict[str, Any]] = [
            {"id": "retina", "name": "retina_l", "size": 2 * 1100 // 2, "status": "learned",
             "part": "retina"},
            {"id": "retina", "name": "retina_r", "size": 2 * 1100 // 2, "status": "learned",
             "part": "retina"}]
        for name, atlas, status in (("disparity", "wulst", "born"), ("parallax", "wulst", "born"),
                                    ("size_one", "rotundus", "born"),
                                    ("elevation_one", "tectum", "born"),
                                    ("size_both", "rotundus", "born"),
                                    ("elevation_both", "tectum", "born"),
                                    ("distance", "arco", "learned"),
                                    ("look", "rotundus", "born"), ("kind", "rotundus", "learned"),
                                    ("focus", "tectum", "born")):
            rows.append({"id": atlas, "name": name, "size": r[name].stop - r[name].start,
                         "status": status, "part": "first"})
        for name, run in p.map._runs.items():
            rows.append({"id": "map", "name": f"map.{name}", "size": run.stop - run.start,
                         "status": "born", "part": "map"})
        pops = self.association.connectome.populations
        ids = {"sensory": None, "association": "ncl", "prefrontal": "trace", "history": "trace",
               "motor": None, "recall": "records", "next": "records", "recall_1": "heads",
               "next_1": "heads"}
        for name in sorted(pops, key=lambda n: int(np.min(pops[n])) if len(pops[n]) else 0):
            if "/" in name or name not in ids:
                continue
            if name == "sensory":
                for group, many in self.inputs.items():
                    atlas = {"ear": "ear", "body": "body"}.get(group, "place")
                    rows.append({"id": atlas, "name": f"sensory.{group}", "size": many,
                                 "status": "untrained", "part": "association"})
            elif name == "motor":
                joints = sum(v for k, v in self.slots.items() if k != "voice")
                rows.append({"id": "arco", "name": "motor.joints", "size": joints,
                             "status": "untrained", "part": "association"})
                rows.append({"id": "vocal", "name": "motor.voice", "size": self.slots["voice"],
                             "status": "untrained", "part": "association"})
            else:
                rows.append({"id": ids[name], "name": name, "size": len(pops[name]),
                             "status": "untrained", "part": "association"})
        rows.append({"id": "words", "name": "words", "size": self.lexicon.neurons,
                     "status": "untrained", "part": "lexicon"})
        rows.append({"id": "striatum", "name": "critic", "size": 1, "status": "untrained",
                     "part": "reward"})
        rows.append({"id": "vta", "name": "dopamine", "size": 1, "status": "untrained",
                     "part": "reward"})
        start = 0
        for row in rows:
            row["start"] = start
            start += int(row["size"])
        return rows

    def _snapshot(self) -> np.ndarray:
        """Every neuron's activity as the parts hold it between moments (rest where a part
        carries none)."""
        rows = self.layout()
        out = np.zeros(sum(int(r["size"]) for r in rows), dtype=np.float32)
        p = self.parrot
        by = {r["name"]: r for r in rows}
        weights = p.vision.weights
        for e, name in ((0, "retina_l"), (1, "retina_r")):
            pooled = weights[e][:100, :176 if weights.shape[2] >= 176 else 172].reshape(
                25, 4, -1, 4).mean(axis=(1, 3)).reshape(-1)
            row = by[name]
            out[row["start"]: row["start"] + min(len(pooled), row["size"])] = pooled[
                : row["size"]]
        start = by["map.reference"]["start"]
        out[start + p.map._runs["map"].start: start + p.map._runs["map"].stop] = (
            p.map.potential.reshape(-1))
        return out

    def _record(self, name: str, trace: list[np.ndarray], drive: np.ndarray,
                out: dict[str, Any]) -> None:
        """The moment's settles, part after part, as rows of every neuron's activity."""
        p = self.parrot
        rows = self.layout()
        by = {r["name"]: r for r in rows}
        state = self._snapshot()
        state[: by["retina_r"]["start"] + by["retina_r"]["size"]] = self._snapshot()[
            : by["retina_r"]["start"] + by["retina_r"]["size"]]
        trace.append(state.copy())
        reach = out.get("reach")
        first = by["disparity"]["start"]
        if reach is not None and len(reach):
            codes = np.asarray(reach)[:, : p.distance.width_in]
            drive_r = np.zeros((1, p.distance.neurons))
            drive_r[0, : p.distance.width_in] = codes[0]
            steps = p.distance.graph.settle_batch(drive_r * p.distance.mask, steps=p.distance.steps,
                                                  tolerance=None, mask=p.distance.mask,
                                                  trajectory=True).trajectory
            assert steps is not None
            for row in np.asarray(steps)[:, 0, :]:
                state[first: first + p.distance.neurons] = row
                trace.append(state.copy())
        if p.focused is not None and len(p.focused.activity):
            f = by["focus"]["start"]
            state[f: f + len(p.focused.activity)] = p.focused.activity
            trace.append(state.copy())
        m = by["map.reference"]["start"]
        state[m + p.map._runs["map"].start: m + p.map._runs["map"].stop] = (
            p.map.potential.reshape(-1))
        trace.append(state.copy())
        graph = self.association.learner.brain
        settled = graph.settle_batch(drive, steps=24, tolerance=None, trajectory=True).trajectory
        assert settled is not None
        a0 = by["sensory.ear"]["start"]
        for row in np.asarray(settled)[:, 0, :]:
            state[a0: a0 + len(row)] = row
            trace.append(state.copy())
        state[by["dopamine"]["start"]] = self.dopamine
        trace.append(state.copy())
        focus = p.focused
        self.recorded.append({"name": name, "trace": np.asarray(trace, dtype=np.float16),
                              "focus": None if focus is None else {
                                  "kind": int(focus.kind), "place": [float(v) for v in focus.place],
                                  "why": focus.why},
                              "said": self.said, "reward": self.reward,
                              "dopamine": self.dopamine, "paid": self.paid})

    # -- the export

    def tracts(self) -> list[dict[str, Any]]:
        """Each part's synapses as tracts between the layout's regions: (from, to, pre, post,
        efficacy) with every synapse."""
        rows = self.layout()
        by = {r["name"]: r for r in rows}
        p = self.parrot
        out: list[dict[str, Any]] = []

        def add(pre: np.ndarray, post: np.ndarray, eff: np.ndarray, offset: int,
                kind: str) -> None:
            pre_g, post_g = np.asarray(pre) + offset, np.asarray(post) + offset
            out.append({"pre": pre_g, "post": post_g, "efficacy": np.asarray(eff), "kind": kind})

        d = p.distance
        add(np.asarray(d.connectome.pre), np.asarray(d.connectome.post), d.graph.efficacy,
            by["disparity"]["start"], "reciprocal")
        k = p.kinds
        # the kinds' graph: categories first, then the look; the layout holds look then kind
        remap = np.concatenate([by["kind"]["start"] + np.arange(k.capacity),
                                by["look"]["start"] + np.arange(k.features)])
        out.append({"pre": remap[np.asarray(k.connectome.pre)],
                    "post": remap[np.asarray(k.connectome.post)],
                    "efficacy": np.asarray(k.graph.efficacy), "kind": "reciprocal"})
        f = p.focus
        add(np.asarray(f.connectome.pre), np.asarray(f.connectome.post), f.graph.efficacy,
            by["focus"]["start"], "fixed")
        add(np.asarray(p.map.connectome.pre), np.asarray(p.map.connectome.post),
            p.map.graph.efficacy, by["map.reference"]["start"], "fixed")
        a = self.association
        add(np.asarray(a.connectome.pre), np.asarray(a.connectome.post),
            np.asarray(a.learner.brain.efficacy), by["sensory.ear"]["start"], "learned")
        lex = self.lexicon
        add(np.asarray(lex.connectome.pre), np.asarray(lex.connectome.post),
            np.asarray(lex.graph.efficacy), by["words"]["start"], "reciprocal")
        return out

    def export(self, folder: str | Path, *, sample: int = 4000) -> Path:
        """Write the brain for a 3D view: ``brain.json``, ``neurons.bin``, a sample of each
        tract and each recorded moment's trace (the parrot's interfaces, D12)."""
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        rows = self.layout()
        n = sum(int(r["size"]) for r in rows)
        region_of = np.zeros(n, dtype=np.int64)
        xyz = np.zeros((n, 3), dtype=np.float32)
        atlas_ids = list(ATLAS)
        for row in rows:
            centre, half, both = ATLAS[row["id"]]
            idx = np.arange(row["start"], row["start"] + row["size"])
            region_of[idx] = atlas_ids.index(row["id"])
            xyz[idx] = _fill(len(idx), np.asarray(centre), np.asarray(half), both,
                             seed=row["start"])
        record = np.zeros(n, dtype=[("region", "<u2"), ("x", "<f4"), ("y", "<f4"),
                                    ("z", "<f4")])
        record["region"], record["x"], record["y"], record["z"] = (
            region_of, xyz[:, 0], xyz[:, 1], xyz[:, 2])
        record.tofile(target / "neurons.bin")
        pairs: dict[tuple[str, str], dict[str, Any]] = {}
        for t in self.tracts():
            src, dst = region_of[t["pre"]], region_of[t["post"]]
            for a, b in {(int(x), int(y)) for x, y in zip(src, dst, strict=True)}:
                keep = (src == a) & (dst == b)
                key = (atlas_ids[a], atlas_ids[b])
                held = pairs.setdefault(key, {"pre": [], "post": [], "efficacy": [],
                                              "kind": t["kind"]})
                held["pre"].append(t["pre"][keep])
                held["post"].append(t["post"][keep])
                held["efficacy"].append(t["efficacy"][keep])
        tracts = []
        rng = np.random.default_rng(0)
        for k, ((origin, end), held) in enumerate(sorted(pairs.items())):
            pre = np.concatenate(held["pre"])
            post = np.concatenate(held["post"])
            eff = np.concatenate(held["efficacy"])
            pick = np.sort(rng.choice(len(pre), size=min(sample, len(pre)), replace=False))
            data = np.zeros(len(pick), dtype=[("pre", "<u4"), ("post", "<u4"),
                                              ("efficacy", "<f4")])
            data["pre"], data["post"], data["efficacy"] = pre[pick], post[pick], eff[pick]
            data.tofile(target / f"tract-{k}.bin")
            tracts.append({"from": origin, "to": end, "synapses": int(len(pre)),
                           "kind": held["kind"],
                           "sample": f"tract-{k}.bin"})
        moments = []
        for k, m in enumerate(self.recorded):
            np.asarray(m["trace"], dtype="<f2").tofile(target / f"trace-{k}.bin")
            moments.append({"name": m["name"], "sweeps": int(len(m["trace"]) - 1),
                            "trace": f"trace-{k}.bin", "focus": m["focus"], "said": m["said"],
                            "reward": m["reward"], "dopamine": m["dopamine"],
                            "paid": m["paid"]})
        regions = []
        for aid, (centre, half, both) in ATLAS.items():
            mine = [r for r in rows if r["id"] == aid]
            status = sorted({r["status"] for r in mine})
            regions.append({"id": aid, "parts": [r["name"] for r in mine],
                            "start": [int(r["start"]) for r in mine],
                            "size": int(sum(int(r["size"]) for r in mine)),
                            "status": status[0] if len(status) == 1 else status,
                            "at": list(centre), "r": list(half), "bi": both})
        index = {"format": "cadence-parrot-brain/1", "neurons": int(n),
                 "synapses": int(sum(t["synapses"] for t in tracts)),
                 "layout": [{k: r[k] for k in ("name", "id", "start", "size", "status")}
                            for r in rows],
                 "regions": regions, "tracts": tracts, "moments": moments}
        (target / "brain.json").write_text(json.dumps(index, indent=1))
        return target


def _fill(count: int, centre: np.ndarray, half: np.ndarray, both: bool, seed: int
          ) -> np.ndarray:
    """``count`` places on a grid inside a box (``centre`` +- ``half``); a region on both sides
    of the head puts its second half at the mirror of the centre (-x)."""
    if count == 0:
        return np.zeros((0, 3), dtype=np.float32)
    sides = 2 if both and count > 1 else 1
    out = []
    for side in range(sides):
        many = count // sides + (count % sides if side == 0 else 0)
        per = max(1, math.ceil(many ** (1.0 / 3.0)))
        grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, per)] * 3, indexing="ij"),
                        axis=-1).reshape(-1, 3)[:many]
        jitter = np.random.default_rng(seed + side).uniform(-0.3, 0.3, grid.shape) / per
        here = centre * np.array([1.0 if side == 0 else -1.0, 1.0, 1.0])
        out.append(here + (grid + jitter) * half)
    return np.concatenate(out).astype(np.float32)


__all__ = ["ATLAS", "WholeParrot"]
