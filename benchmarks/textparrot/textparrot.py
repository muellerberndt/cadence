"""The text parrot: one composed Cadence brain that reads a question, looks, and answers.

**The brain** (``compose``) is ``Brain.compose`` with two processing regions (``modules``), the
last the association region, whose settled activity is the concept code. Every input reaches
the lowest region: the word heard, the gap, the glimpse and the focus. The question is held by
the held message (``message``), which the gap closes (``message_boundary`` with
``message_boundary_holds``): the words heard before the gap stay in it through the sweeps, and
the association reads it through plastic synapses. The association reaches the lowest region
top-down through the apical compartment (``apical``), so the question held above shapes how a
glimpse is read below. The heard history (``history_decays``) holds the moments before, so the
parrot knows where it is in a sentence it is saying. Two motor slots: the voice (every word and
silence) and the gaze (pass or hold). The hand-set genome is ``HAND_SET``.

**The life** (``life``): pretraining by teacher forcing and DAgger through the curriculum
``STAGES`` (words, properties, relations, questions, sentences, scenes), each stage until a
frozen greedy read of its forms reaches ``MASTERY`` or its cap; then the readings.

    python textparrot.py dev <founder> <out.json> [--blocks N] [--genome k=v ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import textchamber as tc  # noqa: E402
import textworld as tw  # noqa: E402

HAND_SET: dict[str, Any] = {
    "modules": (160, 128),
    "apical": 0.25,
    "message": 48,
    "message_phases": 4,
    "message_winners": 4,
    "history_decays": (0.0, 0.6),
    "working_memory_amplitude": 0.3,
    "teaching": "deep",
    "learning_surprise": False,  # every row taught (surprise-only lessons leave small margins)
    "learning_scale_cap": 8.0,
    "sleep_scale": 1.0,  # between stages every efficacy times this (1.0: no sleep)
    # where the words read reach the brain: "module_0" (with the eyes, basal, as compose wires
    # every input), "association" (the top region alone: the question reaches the regions below
    # only top-down, through their apical compartments) or "both"
    "ear_entry": "module_0",
}


def anatomy(g: dict[str, Any], layout: tc.Layout, memory: dict[str, Any]) -> Any:
    """The layout ``Brain.compose`` develops, its sensory region split into ``eyes`` (the glimpse
    and the focus), which reach the lowest region, and ``ear`` (the word read and the gap), which
    reach the regions ``ear_entry`` names."""
    from cadence.connectome import Connectome
    from cadence.genome import Genome, Projection
    from cadence.regions import Region

    modules = tuple(int(m) for m in g["modules"])
    base = Genome.composed(layout.inputs, layout.voice + 2, modules=modules,
                           slots=(layout.voice, 2), **memory)
    ear = list(range(layout.gap + 1))
    eyes = list(range(layout.gap + 1, layout.inputs))
    senses = Connectome.from_synapses(layout.inputs, pre=[], post=[], sign=[],
                                      populations={"eyes": eyes, "ear": ear},
                                      label="eyes and ear")
    regions = tuple(Region("sensory", circuit=senses) if r.name == "sensory" else r
                    for r in base.regions)
    first = base.regions[1].name
    entry = str(g["ear_entry"])
    sensory = [Projection("sensory/eyes", first, reciprocal=False, scale=1.0)]
    if entry in ("association", "both") and first != "association":
        sensory.append(Projection("sensory/ear", "association", reciprocal=False, scale=1.0))
    if entry in ("module_0", "both") or first == "association":
        sensory.append(Projection("sensory/ear", first, reciprocal=False, scale=1.0))
    rest = tuple(p for p in base.projections if p.pre != "sensory")
    return Genome(regions=regions, projections=tuple(sensory) + rest, label="text parrot")

STAGES: list[tuple[str, tuple[str, ...]]] = [
    ("words", ("it", "it_colour", "it_shape", "echo")),
    ("properties", ("which_colour", "which_shape", "which_size", "which_material", "which_food",
                    "know_shape", "know_food")),
    ("relations", ("where", "relation")),
    ("questions", ("colour", "can_eat")),
    ("sentences", ("describe",)),
    ("scenes", ("describe", "describe_rel")),
]
ALL_FORMS = tuple(f for _, forms in STAGES for f in forms)
MASTERY = 0.9
# forms x depth never taught (held-out sentence forms)
HELD_DEPTH = frozenset({("where", 2), ("can_eat", 2)})


def things_for(form: str, rng: np.random.Generator, stage: str | None = None) -> int:
    if form in ("it", "it_colour", "it_shape", "echo"):
        return 1
    if form in ("know_shape", "know_food"):
        return 0
    if stage == "sentences":
        return int(rng.integers(1, 3))
    return int(rng.integers(2, 5))


def draw(form: str, rng: np.random.Generator, *, split: str = "train", stage: str | None = None,
         count: int | None = None, kinds: tuple[str, ...] = tw.TRAIN_KINDS,
         include: tuple[tuple[str, str | None], ...] = ()
         ) -> tuple[tw.Scene, tw.Question]:
    """One exchange of ``form``. ``split``: ``train`` (no held-out pair, relation or depth),
    ``test`` (fresh scenes alike), ``pairs`` (a held-out colour pair in the scene, asked about),
    ``relations`` (a held-out relation x referent pair), ``depth`` (a held-out form x depth)."""
    for _ in range(1000):
        n = count if count is not None else things_for(form, rng, stage)
        held = split == "pairs"
        inc = include
        if held and not inc and n > 0:  # a held-out colour pair, placed first
            pairs = sorted(tw.HELD_PAIRS)
            inc = (pairs[int(rng.integers(len(pairs)))],)
        sc = tw.scene(rng, n, kinds=kinds, held=held, include=inc)
        if split == "depth":
            depth = 2
        else:
            depth = int(rng.integers(1, 3))
            if (form, depth) in HELD_DEPTH:
                depth = 1
        q = tw.question(form, sc, rng, depth=depth, held_relations=split == "relations")
        if q is None:
            continue
        if split == "pairs":
            # the question has to be about the held-out thing (the scene's first)
            first = sc.things[0]
            if q.referent is not None and q.referent is not first:
                continue
            if q.referent is None and form not in ("describe",) and \
                    sc.things.index(first) not in q.clauses:
                continue
        return sc, q
    raise RuntimeError(f"no exchange of {form} in 1000 draws")


class BrainPolicy:
    """The composed brain as the chamber's learner."""

    def __init__(self, layout: tc.Layout, genome: dict[str, Any], seed: int,
                 backend: str = "cpu"):
        from cadence import Brain

        g = {**HAND_SET, **genome}
        self.layout, self.genome = layout, g
        options: dict[str, Any] = {}
        for name, value in g.items():
            if name.startswith("learning_"):
                options[name] = value
        if g.get("message"):
            options.update(message=int(g["message"]), message_phases=int(g["message_phases"]),
                           message_winners=int(g["message_winners"]),
                           message_boundary=layout.gap, message_boundary_holds=True)
        modules = tuple(int(m) for m in g["modules"])
        if float(g["apical"]) > 0 and len(modules) > 1:
            options["apical"] = float(g["apical"])
        if str(g["ear_entry"]) != "module_0":
            from cadence.genome import develop

            memory = {k: v for k, v in options.items() if k.startswith("message")}
            memory["history_decays"] = tuple(g["history_decays"])
            options["connectome"] = develop(anatomy(g, layout, memory), seed=int(seed))
        else:
            options["modules"] = modules
        self.brain = Brain.compose(
            layout.inputs, layout.voice + 2, slots=(layout.voice, 2),
            seed=int(seed), backend=backend, recall=False, teaching=str(g["teaching"]),
            history_decays=tuple(g["history_decays"]),
            working_memory_amplitude=float(g["working_memory_amplitude"]), **options)
        self.n = 0
        self.lessons = 0
        self.taught = 0
        self.sweeps = 0.0  # rows times settle sweeps, lessons and acts: every synapse a MAC
        self.watch = [0.0, 0.0]  # association units near saturation, units seen (reads only)
        assoc = self.brain.connectome.populations["association"]
        self.association = np.asarray(list(assoc))

    @classmethod
    def from_brain(cls, brain: Any, layout: tc.Layout) -> BrainPolicy:
        """A saved brain (``Brain.load``) as the chamber's learner."""
        self = cls.__new__(cls)
        self.layout, self.genome, self.brain = layout, dict(HAND_SET), brain
        self.n = self.lessons = self.taught = 0
        self.sweeps = 0.0
        self.watch = [0.0, 0.0]
        self.association = np.asarray(list(brain.connectome.populations["association"]))
        return self

    def begin(self, n: int) -> None:
        self.brain.reset()
        self.n = n

    def moment(self, x: np.ndarray, labels: np.ndarray | None, mask: np.ndarray,
               learn: bool) -> np.ndarray:
        if learn and labels is not None and mask.any():
            w = mask.astype(float)
            self.brain.lesson(x, labels, weights=None if w.all() else w)
            self.lessons += 1
            report = self.brain.last_learning
            self.taught += int(report.get("demonstrations", mask.sum())
                               - report.get("demonstration_agreed_presentations", 0))
            self.sweeps += float(report.get("demonstration_total_row_sweeps", 0.0))
        a = np.asarray(self.brain.act(x, greedy=True)).reshape(len(x), -1)
        self.sweeps += float(self.brain.last_settlement["steps"]) * len(x)
        if not learn:
            act = np.asarray(self.brain.basal_ganglia.state.activation)[:, self.association]
            self.watch[0] += float((np.abs(act[mask]) > 0.98).sum())
            self.watch[1] += float(act[mask].size)
        return a

    def parameters(self) -> int:
        return int(self.brain.parameters())

    def macs(self) -> float:
        """Multiply-accumulates spent so far: every synapse once a sweep, a row."""
        return float(self.sweeps * len(self.brain.connectome.pre))

    def saturation(self) -> dict[str, float]:
        """How full the brain is: the efficacies' largest and mean magnitude, and the share of
        association activities near their bound in the reads since the last call."""
        eff = np.abs(np.asarray(self.brain.learner.brain.efficacy))
        share = self.watch[0] / self.watch[1] if self.watch[1] else float("nan")
        self.watch = [0.0, 0.0]
        return {"max_efficacy": float(eff.max()), "mean_efficacy": float(eff.mean()),
                "saturated_share": float(share)}

    def sleep(self, scale: float) -> None:
        """Sleep between stages: every efficacy scaled by ``scale`` (synaptic homeostasis); each
        neuron's ranking of its inputs is kept."""
        graph = self.brain.learner.brain
        self.brain.learner.brain = graph.with_parameters(
            efficacy=np.asarray(graph.efficacy) * float(scale), bias=np.asarray(graph.bias))


def batch(rng: np.random.Generator, forms: tuple[str, ...], n: int, *, split: str = "train",
          stage: str | None = None) -> list[tuple[tw.Scene, tw.Question]]:
    return [draw(str(rng.choice(forms)), rng, split=split, stage=stage) for _ in range(n)]


def read(policy: Any, layout: tc.Layout, rng: np.random.Generator, forms: tuple[str, ...],
         per: int, *, split: str = "test", streams: int = 64) -> dict[str, Any]:
    """A frozen greedy read: ``per`` exchanges of each form, free running, nothing taught."""
    out: dict[str, Any] = {}
    for form in forms:
        items = [draw(form, rng, split=split) for _ in range(per)]
        got: list[dict[str, Any]] = []
        for i in range(0, per, streams):
            eps = tc.run(policy, layout, items[i:i + streams], "free", learn=False)
            got += [tc.score(e) for e in eps]
        out[form] = {"right": float(np.mean([g["right"] for g in got])),
                     "example": {"said": " ".join(got[0]["said"]),
                                 "want": " ".join(got[0]["want"])}}
        if "parsed" in got[0]:
            out[form]["parsed"] = float(np.mean([g["parsed"] for g in got]))
    return out


def life(seed: int, *, genome: dict[str, Any] | None = None, streams: int = 64,
         caps: dict[str, int] | None = None, dagger: float = 0.3, rehearsal: float = 0.3,
         check_every: int = 20, per: int = 64, backend: str = "cpu", log: Any = None,
         policy: Any = None) -> tuple[Any, tc.Layout, dict[str, Any]]:
    """Pretraining through the curriculum; returns the policy, the layout and the record."""
    layout = tc.Layout.founder(seed)
    rng = np.random.default_rng([seed, 1])
    if policy is None:
        policy = BrainPolicy(layout, genome or {}, seed, backend=backend)
    caps = caps or {}
    record: dict[str, Any] = {"stages": [], "seed": seed}
    seen: tuple[str, ...] = ()
    start = time.time()
    blocks = 0
    for stage, forms in STAGES:
        seen = tuple(dict.fromkeys(seen + forms))
        cap = int(caps.get(stage, caps.get("default", 400)))
        mastered = None
        curve = []
        lived = 0
        for b in range(1, cap + 1):
            lived = b
            use = forms if (rng.random() > rehearsal or seen == forms) else seen
            mode = "dagger" if (b > check_every and rng.random() < dagger) else "teacher"
            tc.run(policy, layout, batch(rng, use, streams, stage=stage), mode)
            blocks += 1
            if b % check_every == 0:
                r = read(policy, layout, rng, forms, per, split="test", streams=streams)
                mean = float(np.mean([v["right"] for v in r.values()]))
                curve.append({"block": b, "read": mean,
                              "forms": {k: v["right"] for k, v in r.items()},
                              "seconds": round(time.time() - start, 1)})
                if log is not None:
                    print(json.dumps({"stage": stage, **curve[-1]}), file=log, flush=True)
                if mean >= MASTERY:
                    mastered = b
                    break
        entry: dict[str, Any] = {"stage": stage, "mastered": mastered, "blocks": lived,
                                 "curve": curve,
                                 "dagger_own_right": tc.STATS["dagger_own_right"]
                                 / max(1, tc.STATS["dagger_moments"])}
        tc.STATS.update(dagger_moments=0, dagger_own_right=0)
        if hasattr(policy, "saturation"):
            entry["saturation"] = policy.saturation()
            scale = float(policy.genome.get("sleep_scale", 1.0))
            if scale != 1.0 and stage != STAGES[-1][0]:
                policy.sleep(scale)
                entry["slept"] = scale
        if log is not None:
            print(json.dumps({"stage_end": stage, **{k: v for k, v in entry.items()
                                                     if k != "curve"}}), file=log, flush=True)
        record["stages"].append(entry)
    record["blocks"] = blocks
    record["seconds"] = round(time.time() - start, 1)
    return policy, layout, record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["dev"])
    ap.add_argument("seed", type=int)
    ap.add_argument("out")
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--streams", type=int, default=64)
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--genome", nargs="*", default=[])
    a = ap.parse_args()
    genome: dict[str, Any] = {}
    for kv in a.genome:
        k, v = kv.split("=", 1)
        genome[k] = json.loads(v)
    policy, layout, record = life(a.seed, genome=genome, streams=a.streams,
                                  caps={"default": a.cap}, backend=a.backend, log=sys.stdout)
    rng = np.random.default_rng([a.seed, 2])
    record["final"] = read(policy, layout, rng, ALL_FORMS, 64)
    record["parameters"] = policy.parameters()
    Path(a.out).write_text(json.dumps(record, indent=1))
    policy.brain.save(Path(a.out).with_suffix(".npz"))


if __name__ == "__main__":
    main()
