"""Round 2 (cadence#338, declared protocol 2): a sparse mixed code where the question and the
thing meet, the design review's confirming experiment.

    python textround2.py life <founder> <arm> <out_dir> [--blocks 1000]
    python textround2.py control <founder> <mlp|transformer> <out_dir> [--blocks 1000]
    python textround2.py read <out_dir>

Every arm lives 1,000 blocks of 64 exchanges of the forms "what is <colour> ?", "what objects
are <shape> ?", "what is made of <material> ?" and "what colour has the X ?" (teacher forcing,
DAgger three blocks in ten from the 21st), two regions (160, 128) under the apical compartment
at .25 unless the arm says otherwise, and after each block one naming lesson (the thing alone in
view, its name taught), or in the binding arms one binding lesson in its place. Arms:

- ``A0`` the hand-set (lessons where the answer differs); ``A1`` every row taught; ``A1c`` A1 in
  a world of three colours;
- ``C`` A1 with the written question in view and a fixed sparse code of the field that holds the
  written question and the glimpse (1024 units, a tenth on, random weights on every input of the
  field, drawn per founder), the rest of the observation (the word heard, the gap, the focus)
  beside it; ``F`` C in one flat region of matched parameters, no compartment;
- ``Cw`` A1 with the written question in view and ``code=1024`` over the whole observation;
- ``L`` A1 with the written question in view and ``learned_code`` (256 units) with
  ``learned_code_fields``: the word heard and the gap, the focus, and the field of the written
  question and the glimpse; a held message refuses a learned code, so L holds the question by
  the written page alone;
- ``G`` A1 under the gated apical compartment at .5 (``apical_gate.patch``, applied to a copy of
  the source);
- ``B`` A1 with the binding lesson in place of the naming lesson: the word alone (nothing in
  view) pulled toward the association's state with the thing in view and no word, by the
  learner's own phases with the pull on the association, with optimizer moments of its own and
  the word's synapses alone plastic; ``BC`` the same on top of C.

Readings every 50 blocks: per form, hit and false alarm at the answer sweep's first moment at
each thing (teacher forced; a matching thing named right is a hit, any word at another thing a
false alarm), d' and criterion, on shown and on held-out colour pairs. At the end: whole replies;
the probe of match; the completion matrix (B, BC, A1, C at 50, 100, 200 blocks); names; the
page's gate (1, 2, 5, 10 naming lessons on one focus, then that focus's colour answer).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import textchamber as tc  # noqa: E402
import textparrot as tp  # noqa: E402
import textworld as tw  # noqa: E402

FORMS = ("which_colour", "which_shape", "which_material", "colour")
BASE: dict[str, Any] = {"modules": [160, 128], "apical": 0.25}
EVERY = {"learning_surprise": False}
ARMS: dict[str, dict[str, Any]] = {
    "A0": {"genome": {}},
    "A1": {"genome": EVERY},
    "A1c": {"genome": EVERY, "colours": ("red", "yellow", "blue")},
    "C": {"genome": EVERY, "page": True, "field": "code"},
    "F": {"genome": {**EVERY, "apical": 0.0}, "page": True, "field": "code", "flat": True},
    "Cw": {"genome": {**EVERY, "code": 1024, "page": True}, "page": True},
    "L": {"genome": {**EVERY, "message": 0}, "page": True, "field": "learned"},
    "G": {"genome": {**EVERY, "apical_gate": 0.5}},
    "B": {"genome": EVERY, "bind": True},
    "BC": {"genome": EVERY, "page": True, "field": "code", "bind": True},
    # added development arm (not of the declared protocol): L on the learned code's fix (the
    # first of several modules reads the code at unit length), a nursery of 1,024 moments of
    # the forms (acts alone, no lesson) before the first block, the code's rate divided by four
    "L2": {"genome": {**EVERY, "message": 0}, "page": True, "field": "learned", "nursery": 1024,
           "code_rate": 0.25},
}
CODE_UNITS, CODE_ON = 1024, 102
LEARNED_UNITS = 256


# --- the observation an arm's brain reads ---------------------------------------------------

class Field:
    """What the brain of an arm reads: the chamber's observation, or with ``code`` the word heard,
    the gap and the focus beside a fixed sparse code of the written question and the glimpse, or
    with ``learned`` the same parts in fields (``widths``) for a learned code."""

    def __init__(self, layout: tc.Layout, kind: str | None, seed: int):
        self.layout, self.kind = layout, kind
        L = layout
        self.rest = np.r_[np.arange(L.words + 1), np.arange(L.focus.start, L.focus.stop)]
        self.field = np.r_[np.arange(L.glimpse.start, L.glimpse.stop - 1),
                           np.arange(L.written.start, L.written.stop)] if L.page else \
            np.arange(L.glimpse.start, L.glimpse.stop - 1)
        if kind == "code":
            rng = np.random.default_rng([seed, 338, 2])
            self.weights = rng.normal(0.0, 1.0, (len(self.field), CODE_UNITS))
            self.inputs = len(self.rest) + CODE_UNITS
        elif kind == "learned":
            self.inputs = len(self.rest) + len(self.field)
            self.widths = (L.words + 1, len(self.rest) - L.words - 1, len(self.field))
        else:
            self.inputs = L.inputs

    def code(self, f: np.ndarray) -> np.ndarray:
        drive = f @ self.weights
        out = np.zeros_like(drive)
        on = np.argsort(-drive, axis=1)[:, :CODE_ON]
        live = np.abs(f).sum(1) > 0  # an empty field codes nothing
        rows = np.arange(len(f))[:, None]
        out[rows, on] = 1.0
        return out * live[:, None]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.kind is None:
            return x
        rest, field = x[:, self.rest], x[:, self.field]
        if self.kind == "code":
            return np.concatenate([rest, self.code(field)], axis=1)
        return np.concatenate([rest, field], axis=1)


class Arm(tp.BrainPolicy):
    """The brain of an arm, reading its field."""

    def __init__(self, layout: tc.Layout, genome: dict[str, Any], seed: int, field: Field):
        self.field = field
        # the brain is built for the field's width; the gap channel is at the same index
        fake = _Width(layout, field.inputs)
        g = dict(genome)
        if field.kind == "learned":
            g["learned_code"] = LEARNED_UNITS
            g["learned_code_fields"] = list(field.widths)
        super().__init__(fake, g, seed)  # type: ignore[arg-type]
        self.layout = layout

    def moment(self, x: np.ndarray, labels: Any, mask: np.ndarray, learn: bool) -> np.ndarray:
        return super().moment(self.field(x), labels, mask, learn)


class _Width:
    """A layout whose inputs are the field's width (for composing an arm's brain)."""

    def __init__(self, layout: tc.Layout, inputs: int):
        self._l, self.inputs = layout, inputs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._l, name)


def compose_arm(arm: str, seed: int) -> tuple[Arm, tc.Layout]:
    spec = ARMS[arm]
    layout = tc.Layout.founder(seed, page=bool(spec.get("page")))
    field = Field(layout, spec.get("field"), seed)
    genome = {**BASE, **spec["genome"]}
    if spec.get("flat"):
        genome["modules"] = [flat_width(seed)]
    policy = Arm(layout, genome, seed, field)
    return policy, layout


def flat_width(seed: int) -> int:
    """The width of one flat region whose parameters are nearest C's two regions."""
    target = compose_arm("C", seed)[0].parameters()
    best, width = math.inf, 128
    for w in range(64, 352, 16):
        layout = tc.Layout.founder(seed, page=True)
        field = Field(layout, "code", seed)
        p = Arm(layout, {**BASE, **EVERY, "apical": 0.0, "modules": [w]}, seed, field)
        if abs(p.parameters() - target) < best:
            best, width = abs(p.parameters() - target), w
    return width


# --- detection readings ---------------------------------------------------------------------

def detection(policy: Any, layout: tc.Layout, rng: np.random.Generator, form: str,
              split: str = "test", n: int = 128) -> dict[str, float]:
    """Hit and false alarm at the answer sweep's first moment at each thing, teacher forced."""
    items = [tp.draw(form, rng, split=split) for _ in range(n)]
    eps = [tc.Episode(layout, s, q, "teacher") for s, q in items]
    policy.begin(n)
    c = {"hit": 0, "match": 0, "fa": 0, "other": 0}
    for _ in range(160):
        mask = np.array([not e.done for e in eps])
        if not mask.any():
            break
        x = np.stack([e.observe() for e in eps])
        a = policy.moment(x, None, mask, False)
        for k, e in enumerate(eps):
            if e.done:
                continue
            w, g = e.expert()
            if e.phase == "answer" and not e.at:
                said = layout.say(int(a[k, 0]))
                target = e.target_words()
                if target:
                    c["match"] += 1
                    c["hit"] += int(said == target[0])
                else:
                    c["other"] += 1
                    c["fa"] += int(said != tw.SILENCE)
            e.advance(w, g)
    return dprime(c)


def dprime(c: dict[str, int]) -> dict[str, float]:
    from statistics import NormalDist

    z = NormalDist().inv_cdf
    nm, no = max(c["match"], 1), max(c["other"], 1)
    h = min(max(c["hit"] / nm, 1 / (2 * nm)), 1 - 1 / (2 * nm))
    f = min(max(c["fa"] / no, 1 / (2 * no)), 1 - 1 / (2 * no))
    return {"hit": c["hit"] / nm, "fa": c["fa"] / no, "d": z(h) - z(f),
            "criterion": -(z(h) + z(f)) / 2, "n_match": c["match"], "n_other": c["other"]}


def detections(policy: Any, layout: tc.Layout, seed: int, block: int) -> dict[str, Any]:
    rng = np.random.default_rng([seed, 338, 7, block])
    out: dict[str, Any] = {}
    for form in FORMS:
        out[form] = {"shown": detection(policy, layout, rng, form)}
        if form in ("which_colour", "colour"):
            out[form]["held"] = detection(policy, layout, rng, form, split="pairs")
    return out


# --- naming and binding lessons -------------------------------------------------------------

def alone(layout: tc.Layout, rng: np.random.Generator, n: int) -> list[tw.Thing]:
    out = []
    for _ in range(n):
        k = str(rng.choice(tw.TRAIN_KINDS))
        cs = [c for c in tw.KINDS[k][3] if (k, c) not in tw.HELD_PAIRS]
        c = str(rng.choice(cs))
        place = tw.PLACES[int(rng.integers(len(tw.PLACES)))]
        out.append(tw.Thing(k, c, place, tw.look(k, c, place, rng)))
    return out


def seen(layout: tc.Layout, things: list[tw.Thing]) -> np.ndarray:
    """The observation of each thing alone in view: no word, no question."""
    x = np.zeros((len(things), layout.inputs))
    for i, t in enumerate(things):
        x[i, layout.glimpse] = np.concatenate([t.look, [1.0]])
    return x


def heard(layout: tc.Layout, words: list[str]) -> np.ndarray:
    """The observation of each word alone, nothing in view."""
    x = np.zeros((len(words), layout.inputs))
    for i, w in enumerate(words):
        x[i, :layout.words] = layout.ear(w)
    return x


def naming_lesson(policy: Arm, layout: tc.Layout, things: list[tw.Thing]) -> None:
    policy.begin(len(things))
    labels = np.array([[layout.act_of(t.kind), tc.PASS] for t in things])
    policy.brain.lesson(policy.field(seen(layout, things)), labels)
    policy.brain.reset()


def settled(policy: Arm, x: np.ndarray) -> np.ndarray:
    """The association's state of each row settled from rest, nothing learned."""
    brain = policy.brain
    brain.reset()
    brain.act(policy.field(x), greedy=True)
    state = np.asarray(brain.basal_ganglia.state.activation)[:, policy.association].copy()
    brain.reset()
    return state


MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")


class Binder:
    """The binding lesson: the word alone pulled toward the association's state with the thing
    in view, by the learner's own phases with the pull on the association, its optimizer moments
    its own, the word's synapses (from the heard word's channels) alone plastic."""

    def __init__(self, policy: Arm, layout: tc.Layout, beta: float = 0.1):
        self.policy, self.layout, self.beta = policy, layout, beta
        learner = policy.brain.learner
        self.moments = {k: np.zeros_like(np.asarray(getattr(learner, k))) for k in MOMENTS}
        c = policy.brain.connectome
        sensory = np.asarray(list(c.populations["sensory"]))
        ear = sensory[:layout.words]  # the field keeps the heard word's channels first
        self.plastic = np.isin(np.asarray(c.pre), ear)
        self.lessons = 0

    def lesson(self, things: list[tw.Thing]) -> None:
        from cadence.brain import Nudge

        policy, layout = self.policy, self.layout
        brain = policy.brain
        learner = brain.learner
        target = settled(policy, seen(layout, things))
        x = policy.field(heard(layout, [t.kind for t in things]))
        main = {k: np.array(getattr(learner, k), copy=True) for k in MOMENTS}
        for k in MOMENTS:
            setattr(learner, k, self.moments[k])
        before = (np.array(learner.brain.efficacy, copy=True),
                  np.array(learner.brain.bias, copy=True))
        brain.reset()
        brain._prepare(x)
        drive = brain._moment_drive(x)
        free = learner.free(drive, None)
        n = brain.connectome.n
        full = np.zeros((len(x), n))
        full[:, policy.association] = target
        mask = np.zeros(n)
        mask[policy.association] = 1.0
        held = learner.brain.apical_gain(drive, free, own_precision=True)
        nudged = learner._nudged(drive, free, Nudge(full, mask, self.beta, apical_gain=held),
                                 "nudged")
        opposite = learner._nudged(drive, free,
                                   Nudge(full, mask, -self.beta, apical_gain=held), "opposite")
        learner.update(free, nudged, opposite)
        graph = learner.brain
        eff = np.where(self.plastic, np.asarray(graph.efficacy), before[0])
        learner.brain = graph.with_parameters(efficacy=eff, bias=before[1])
        self.moments = {k: np.array(getattr(learner, k), copy=True) for k in MOMENTS}
        for k in MOMENTS:
            setattr(learner, k, main[k])
        brain.reset()
        self.lessons += 1


def completion(policy: Arm, layout: tc.Layout, seed: int) -> dict[str, float]:
    """The word alone against the seen states: the kind decoded from each kind word's state by
    the nearest mean seen state (centred per route), of the ten kinds."""
    rng = np.random.default_rng([seed, 338, 11])
    kinds = list(tw.TRAIN_KINDS)
    means = []
    for k in kinds:
        things = [t for t in alone(layout, rng, 200) if t.kind == k][:16]
        means.append(settled(policy, seen(layout, things)).mean(0))
    m = np.stack(means)
    m = m - m.mean(0)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    h = settled(policy, heard(layout, kinds))
    h = h - h.mean(0)
    h /= np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
    near = (h @ m.T).argmax(1)
    return {"kind": float(np.mean(near == np.arange(len(kinds)))),
            "own_cosine": float(np.mean(np.diag(h @ m.T)))}


# --- the probe of match and the page's gate -------------------------------------------------

def match_probe(policy: Arm, layout: tc.Layout, seed: int, n: int = 192) -> dict[str, float]:
    """A linear read of match (the question's colour equals the glimpsed thing's) at the answer
    sweep's first moment at each thing, "what is <colour> ?", teacher forced: on the chamber's
    observation with the question in view, on its field code, on the arm's own observation, and
    on the arm's association."""
    rng = np.random.default_rng([seed, 338, 13])
    page = tc.Layout(layout.forms, True)
    field = Field(page, "code", seed)
    items = [tp.draw("which_colour", rng) for _ in range(n)]
    eps = [tc.Episode(layout, s, q, "teacher") for s, q in items]
    pages = [tc.Episode(page, s, q, "teacher") for s, q in items]
    policy.begin(n)
    rows: dict[str, list[np.ndarray]] = {"raw": [], "field_code": [], "own": [],
                                         "association": []}
    y: list[int] = []
    for _ in range(160):
        mask = np.array([not e.done for e in eps])
        if not mask.any():
            break
        x = np.stack([e.observe() for e in eps])
        xp = np.stack([e.observe() for e in pages])
        policy.moment(x, None, mask, False)
        state = np.asarray(policy.brain.basal_ganglia.state.activation)[:, policy.association]
        for k, (e, ep) in enumerate(zip(eps, pages, strict=True)):
            if e.done:
                continue
            w, g = e.expert()
            if e.phase == "answer" and not e.at:
                rows["raw"].append(xp[k])
                rows["field_code"].append(field(xp[k:k + 1])[0])
                rows["own"].append(policy.field(x[k:k + 1])[0])
                rows["association"].append(state[k].copy())
                y.append(int(bool(e.target_words())))
            e.advance(w, g)
            ep.advance(w, g)
    labels = np.array(y, float)
    out = {}
    half = len(labels) // 2
    for name, xs in rows.items():
        X = np.stack(xs)
        out[name] = _balanced(X[:half], labels[:half], X[half:], labels[half:])
    return out


def _balanced(xa: np.ndarray, ya: np.ndarray, xb: np.ndarray, yb: np.ndarray) -> float:
    mu, sd = xa.mean(0), xa.std(0) + 1e-6
    za, zb = (xa - mu) / sd, (xb - mu) / sd
    w, b = np.zeros(za.shape[1]), 0.0
    pos = max(ya.mean(), 1e-3)
    weight = np.where(ya > 0, 0.5 / pos, 0.5 / max(1 - pos, 1e-3))
    for _ in range(400):
        q = 1 / (1 + np.exp(-(za @ w + b)))
        g = (q - ya) * weight
        w -= 0.5 * (za.T @ g / len(ya) + 1e-2 * w)
        b -= 0.5 * float(np.mean(g))
    got = (zb @ w + b) > 0
    t = yb > 0
    return float((np.mean(got[t]) + np.mean(~got[~t])) / 2) if t.any() and (~t).any() else \
        float("nan")


def first_word(policy: Any, layout: tc.Layout, items: list[tuple[tw.Scene, tw.Question]]
               ) -> float:
    """The share of exchanges whose first word at the named thing (the answer sweep's first
    moment there, teacher forced) is the teacher's."""
    eps = [tc.Episode(layout, s, q, "teacher") for s, q in items]
    policy.begin(len(eps))
    right: list[int] = []
    for _ in range(160):
        mask = np.array([not e.done for e in eps])
        if not mask.any():
            break
        x = np.stack([e.observe() for e in eps])
        a = policy.moment(x, None, mask, False)
        for k, e in enumerate(eps):
            if e.done:
                continue
            w, g = e.expert()
            if e.phase == "answer" and not e.at and e.target_words():
                right.append(int(layout.say(int(a[k, 0])) == e.target_words()[0]))
            e.advance(w, g)
    return float(np.mean(right)) if right else float("nan")


def names(policy: Arm, layout: tc.Layout, seed: int, n: int = 128) -> float:
    """The name said for a thing alone in view, nothing asked (the naming lesson's moment)."""
    things = alone(layout, np.random.default_rng([seed, 338, 29]), n)
    policy.begin(n)
    a = np.asarray(policy.brain.act(policy.field(seen(layout, things)), greedy=True))
    policy.brain.reset()
    return float(np.mean([layout.say(int(v)) == t.kind for v, t in
                          zip(a.reshape(n, -1)[:, 0], things, strict=True)]))


def page_gate(policy: Arm, layout: tc.Layout, seed: int) -> dict[str, Any]:
    """After the last block: 1, 2, 5 and 10 naming lessons on one focus (one thing, its name
    taught with it alone in view), then that focus's colour answer ("what colour has the X ?"
    with the focus among two other things, the first word said at it, teacher forced): does it
    hold?"""
    rng = np.random.default_rng([seed, 338, 17])
    focus = alone(layout, rng, 1)[0]
    q = copy.copy(policy)
    q.brain = copy.deepcopy(policy.brain)

    def colour_answer() -> float:
        r = np.random.default_rng([seed, 338, 19])
        items: list[tuple[tw.Scene, tw.Question]] = []
        while len(items) < 16:  # the focus among two other things, as the form is taught
            sc = tw.scene(r, 3, include=((focus.kind, focus.colour),))
            if len(sc.find(focus.kind)) != 1:
                continue
            ref = sc.things[0]
            items.append((sc, tw.Question("colour", ["what", "colour", "has", "the", ref.kind,
                                                     "?"], ref, {0: [ref.colour]}, [])))
        return first_word(q, layout, items)

    out = {"focus": [focus.kind, focus.colour], "before": colour_answer()}
    done = 0
    for n in (1, 2, 5, 10):
        while done < n:
            naming_lesson(q, layout, [focus])
            done += 1
        out[str(n)] = colour_answer()
    return out


# --- a life --------------------------------------------------------------------------------

def life(seed: int, arm: str, out_dir: Path, blocks: int = 1000, every: int = 50) -> None:
    if (out_dir / f"skip_{arm}").exists():  # the arm runs on another box
        return
    spec = ARMS[arm]
    if spec.get("colours"):
        tw.use_colours(tuple(spec["colours"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{arm}_{seed}"
    policy, layout = compose_arm(arm, seed)
    binder = Binder(policy, layout) if spec.get("bind") else None
    rng = np.random.default_rng([seed, 338, 1])
    if spec.get("code_rate"):
        policy.brain.learned_code.rate *= float(spec["code_rate"])
    if spec.get("nursery"):  # the code region learns from the forms' moments before any lesson
        moments = 0
        while moments < int(spec["nursery"]):
            eps = tc.run(policy, layout, tp.batch(rng, FORMS, 64), "free", learn=False)
            moments += max(e.moments for e in eps)
    record: dict[str, Any] = {"arm": arm, "seed": seed, "parameters": policy.parameters(),
                              "synapses": int(len(policy.brain.connectome.pre)), "reads": [],
                              "completion": []}
    record["probe_newborn"] = match_probe(policy, layout, seed)
    start = time.time()
    log = open(f"{stem}.log", "a")
    for b in range(1, blocks + 1):
        mode = "teacher" if b <= 20 or rng.random() > 0.3 else "dagger"
        tc.run(policy, layout, tp.batch(rng, FORMS, 64), mode)
        things = alone(layout, rng, 64)
        if binder is not None:
            binder.lesson(things)
        else:
            naming_lesson(policy, layout, things)
        if b % every == 0:
            row = {"block": b, "seconds": round(time.time() - start),
                   "detection": detections(policy, layout, seed, b), "macs": policy.macs()}
            record["reads"].append(row)
            short = {f: round(v["shown"]["d"], 2) for f, v in row["detection"].items()}
            print(json.dumps({"block": b, "d": short, "s": row["seconds"]}), file=log,
                  flush=True)
            if arm in ("A1", "B", "BC", "C") and b in (50, 100, 200, blocks):
                record["completion"].append({"block": b, **completion(policy, layout, seed)})
            Path(f"{stem}.json").write_text(json.dumps(record, indent=1, default=float))
    record["seconds"] = round(time.time() - start)
    record["macs"] = policy.macs()
    record["refused"] = policy.refused
    rr = np.random.default_rng([seed, 338, 23])
    record["replies"] = {"shown": {k: v["right"] for k, v in
                                   tp.read(policy, layout, rr, FORMS, 128).items()},
                         "held": {k: v["right"] for k, v in
                                  tp.read(policy, layout, rr, ("which_colour", "colour"), 128,
                                          split="pairs").items()},
                         "names": names(policy, layout, seed)}
    record["probe_trained"] = match_probe(policy, layout, seed)
    record["page_gate"] = page_gate(policy, layout, seed)
    policy.brain.save(f"{stem}.npz")
    Path(f"{stem}.json").write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({"done": arm, "seed": seed}), file=log, flush=True)


# --- the controls --------------------------------------------------------------------------

def control(seed: int, model: str, out_dir: Path, blocks: int = 1000, every: int = 50) -> None:
    """The one-hidden-layer gradient net (``mlp``: 128 tanh units on the moment, the moment before
    and the question's words) or the transformer at the A1 brain's parameter count, online in the
    same moments: one Adam step a block."""
    import textcontrols as tcl

    if (out_dir / f"skip_{model}").exists():  # the control runs on another box
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = tc.Layout.founder(seed)
    rng = np.random.default_rng([seed, 338, 1])
    if model == "mlp":
        net: Any = MLP(layout, seed)
    else:
        target = compose_arm("A1", seed)[0].parameters()
        net = tcl.NetPolicy("transformer", layout, target, seed, 0.003)
    record: dict[str, Any] = {"arm": model, "seed": seed, "parameters": net.parameters(),
                              "reads": []}
    start = time.time()
    for b in range(1, blocks + 1):
        items = tp.batch(rng, FORMS, 64)
        if model == "mlp":
            tc.run(net, layout, items, "teacher")
            net.step()
        else:
            eps = [tc.Episode(layout, s, q, "teacher") for s, q in items]
            net.update(*tcl.sequences(eps, layout))
        if b % every == 0:
            record["reads"].append({"block": b, "seconds": round(time.time() - start),
                                    "detection": detections(net, layout, seed, b)})
            Path(out_dir / f"{model}_{seed}.json").write_text(json.dumps(record, indent=1))
    rr = np.random.default_rng([seed, 338, 23])
    record["replies"] = {"shown": {k: v["right"] for k, v in
                                   tp.read(net, layout, rr, FORMS, 128).items()},
                         "held": {k: v["right"] for k, v in
                                  tp.read(net, layout, rr, ("which_colour", "colour"), 128,
                                          split="pairs").items()}}
    record["macs"] = getattr(net, "macs", 0)
    Path(out_dir / f"{model}_{seed}.json").write_text(json.dumps(record, indent=1,
                                                                 default=float))


class MLP:
    """One hidden layer of 128 tanh units on the moment, the moment before and the question's
    words; a voice and a gaze head; trained by gradient, one Adam step a block."""

    def __init__(self, layout: tc.Layout, seed: int, hidden: int = 128):
        import textcontrols as tcl
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.torch, self.layout = torch, layout
        self.feat = tcl.ReadoutPolicy(layout, seed)
        d = 3 * layout.inputs
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, layout.voice + 2))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.003)
        self.buf: list[Any] = []
        self.macs = 0

    def parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def begin(self, n: int) -> None:
        self.feat.begin(n)

    def moment(self, x: np.ndarray, labels: Any, mask: np.ndarray, learn: bool) -> np.ndarray:
        torch, L = self.torch, self.layout
        f = torch.from_numpy(self.feat.features(x)).float()
        self.feat.prev = x.copy()
        z = self.net(f)
        self.macs += (3 if learn else 1) * len(x) * self.parameters()
        if learn and labels is not None and mask.any():
            m = torch.from_numpy(mask.astype(np.float32))
            lv = torch.nn.functional.cross_entropy(z[:, :L.voice], torch.from_numpy(labels[:, 0]),
                                                   reduction="none")
            lg = torch.nn.functional.cross_entropy(z[:, L.voice:], torch.from_numpy(labels[:, 1]),
                                                   reduction="none")
            self.buf.append(((lv + lg) * m).sum() / m.sum().clamp(min=1))
        with torch.no_grad():
            return np.stack([z[:, :L.voice].argmax(1).numpy(), z[:, L.voice:].argmax(1).numpy()], 1)

    def step(self) -> None:
        if self.buf:
            self.opt.zero_grad()
            self.torch.stack(self.buf).sum().backward()
            self.opt.step()
            self.buf = []


# --- the tables ----------------------------------------------------------------------------

def table(out_dir: Path) -> str:
    recs: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(out_dir.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        recs.setdefault(r["arm"], []).append(r)
    blocks = sorted({row["block"] for rs in recs.values() for r in rs for row in r["reads"]})
    pick = [b for b in blocks if b in (50, 100, 200, 300, 500, 750, 1000)]
    lines = ["d' on 'what is red ?' (shown pairs), mean over founders (founders reaching 2)",
             "", "| arm | n | params | " + " | ".join(str(b) for b in pick) +
             " | first block at d' 2 (median) |", "|" + "---|" * (len(pick) + 4)]
    for arm, rs in recs.items():
        cells = []
        for b in pick:
            ds = [row["detection"]["which_colour"]["shown"]["d"] for r in rs for row in
                  r["reads"] if row["block"] == b]
            cells.append(f"{np.mean(ds):.2f} ({sum(d >= 2 for d in ds)})" if ds else "")
        firsts = []
        for r in rs:
            hit = [row["block"] for row in r["reads"]
                   if row["detection"]["which_colour"]["shown"]["d"] >= 2]
            firsts.append(hit[0] if hit else math.inf)
        med = float(np.median(firsts)) if firsts else math.inf
        lines.append(f"| {arm} | {len(rs)} | {np.mean([r['parameters'] for r in rs]):.3g} | "
                     + " | ".join(cells) + f" | {med:.0f} |")
    return "\n".join(lines)


def regate(out_dir: Path, arm: str, seed: int) -> None:
    """The end readings read from the saved brain: names, the page's gate (first words) and the
    completion, written into the record."""
    from cadence import Brain

    spec = ARMS[arm]
    if spec.get("colours"):
        tw.use_colours(tuple(spec["colours"]))
    stem = out_dir / f"{arm}_{seed}"
    policy, layout = compose_arm(arm, seed)
    policy.brain = Brain.load(f"{stem}.npz", backend="cpu")
    record = json.loads(Path(f"{stem}.json").read_text())
    record["names_first"] = names(policy, layout, seed)
    record["page_gate_first"] = page_gate(policy, layout, seed)
    record["completion_end"] = completion(policy, layout, seed)
    Path(f"{stem}.json").write_text(json.dumps(record, indent=1, default=float))


def criteria(out_dir: Path) -> dict[str, Any]:
    """The declared criteria, per founder (d' on "what is red ?", shown pairs)."""
    recs: dict[str, dict[int, dict[str, Any]]] = {}
    for p in sorted(out_dir.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        recs.setdefault(r["arm"], {})[int(r["seed"])] = r

    def first(r: dict[str, Any]) -> float:
        for row in r["reads"]:
            if row["detection"]["which_colour"]["shown"]["d"] >= 2:
                return float(row["block"])
        return math.inf

    def curve(arm: str) -> dict[int, float]:
        rows: dict[int, list[float]] = {}
        for r in recs.get(arm, {}).values():
            for row in r["reads"]:
                rows.setdefault(row["block"], []).append(
                    row["detection"]["which_colour"]["shown"]["d"])
        return {b: float(np.mean(v)) for b, v in sorted(rows.items())}

    out: dict[str, Any] = {}
    a1 = recs.get("A1", {})
    for arm in ("C", "L", "Cw", "A1c", "BC", "F", "G", "B", "A0"):
        got = recs.get(arm, {})
        both = [f for f in got if f in a1]
        half = sum(first(got[f]) <= first(a1[f]) / 2 for f in both)
        out[f"{arm}_at_half_A1"] = f"{half} of {len(both)}"
        out[f"{arm}_first_median"] = float(np.median([first(got[f]) for f in got])) if got else None
    out["A1_first_median"] = float(np.median([first(r) for r in a1.values()])) if a1 else None
    c, f_, g, a = curve("C"), curve("F"), curve("G"), curve("A1")
    out["F_minus_C_max_abs"] = max((abs(f_[b] - c[b]) for b in f_ if b in c), default=None)
    out["G_above_A1_reads"] = [b for b in g if b in a and g[b] > a[b]]
    out["G_minus_A1_max"] = max((g[b] - a[b] for b in g if b in a), default=None)
    refute = 0
    for f in recs.get("G", {}):
        for row in recs["G"][f]["reads"]:
            b = row["block"]
            if row["detection"]["which_colour"]["shown"]["d"] < 2:
                continue
            others = []
            for arm in ("A1", "C", "L"):
                r = recs.get(arm, {}).get(f)
                if r:
                    others += [x["detection"]["which_colour"]["shown"]["d"] for x in r["reads"]
                               if x["block"] == b]
            if others and max(others) < 1:
                refute += 1
                break
    out["G_refute_founders"] = refute

    def gate(r: dict[str, Any]) -> tuple[float, float] | None:
        g = r.get("page_gate_first")
        return (g["before"], g["10"]) if g else None
    for arm in ("C", "A1", "BC", "B", "A0", "Cw", "F", "G", "A1c"):
        rows = [gate(r) for r in recs.get(arm, {}).values()]
        rows = [x for x in rows if x is not None]
        if rows:
            out[f"page_{arm}"] = {"before": float(np.mean([x[0] for x in rows])),
                                  "after_10": float(np.mean([x[1] for x in rows])),
                                  "holds": sum(x[1] >= 0.75 for x in rows),
                                  "flips": sum(x[1] < 0.5 for x in rows), "n": len(rows)}
    for arm in ("A1", "B", "C", "BC"):
        comp = [r["completion"] for r in recs.get(arm, {}).values() if r.get("completion")]
        if comp:
            out[f"completion_{arm}"] = {
                str(b): float(np.mean([row["kind"] for c_ in comp for row in c_
                                       if row["block"] == b])) for b in (50, 100, 200)}
        names_ = [r.get("names_first") for r in recs.get(arm, {}).values()
                  if r.get("names_first") is not None]
        if names_:
            out[f"names_{arm}"] = float(np.mean(names_))
    for arm in recs:
        probes = [r["probe_trained"] for r in recs[arm].values() if r.get("probe_trained")]
        if probes:
            out[f"probe_{arm}"] = {k: round(float(np.mean([p[k] for p in probes])), 3)
                                   for k in probes[0]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["life", "control", "read", "regate", "criteria"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--blocks", type=int, default=1000)
    ap.add_argument("--every", type=int, default=50)
    a = ap.parse_args()
    if a.what == "life":
        life(int(a.args[0]), a.args[1], Path(a.args[2]), a.blocks, a.every)
    elif a.what == "control":
        control(int(a.args[0]), a.args[1], Path(a.args[2]), a.blocks, a.every)
    elif a.what == "criteria":
        print(json.dumps(criteria(Path(a.args[0])), indent=1, default=str))
    elif a.what == "regate":
        regate(Path(a.args[0]), a.args[1], int(a.args[2]))
    else:
        print(table(Path(a.args[0])))


if __name__ == "__main__":
    main()
