"""Looking inside the text parrot: concepts as codes, read from its own regions.

A thing's **seeing code** is the settled activity of the association region at the moment the
parrot is about to name the one thing in view ("what is it ?", the answer sweep's first moment
at the thing). A word's **hearing code** is the same region's activity at the moment the parrot
reads the word alone, nothing in view. Nothing in the brain names a concept; these readings find
what it learned:

- **probes**: a logistic readout from the seeing code to a property (round, red, food, left,
  near), fitted on things of the pretraining's kinds and read on fresh things;
- **neighbours**: each kind's mean seeing code and the kinds nearest it (cosine, centred), the
  new kinds (a lemon, a pen) included;
- **a word alone**: the seeing probes read on each word's hearing code (each route centred on its
  own mean): does "apple" alone bring up round, red and food?
- **silencing**: a property's probe direction taken out of the association region's outgoing
  synapses; the answers that need the property, and others, read again;
- **new objects**: the probes read on things of the never-shown kinds;
- **a changed scene**: the same question after the thing moved or changed colour.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import textchamber as tc  # noqa: E402
import textworld as tw  # noqa: E402

PROPERTIES = ("round", "red", "food", "left", "near")


def label(t: tw.Thing, prop: str) -> int:
    if prop == "round":
        return int(t.shape == "round")
    if prop == "red":
        return int(t.colour == "red")
    if prop == "food":
        return int(t.food)
    if prop == "left":
        return int(t.direction == "left")
    if prop == "near":
        return int(not t.far)
    raise ValueError(prop)


def region_names(policy: Any) -> list[str]:
    """The processing regions, lowest first, the association region last."""
    pops = policy.brain.connectome.populations
    names = sorted((n for n in pops if n.startswith("module_")), key=lambda n: int(n[7:]))
    return names + ["association"]


class Recorder:
    """A frozen copy of a brain policy that records the association region each moment (and with
    ``every`` each processing region, ``regions[name]``)."""

    def __init__(self, policy: Any, every: bool = False):
        self.policy = copy.copy(policy)
        self.policy.brain = copy.deepcopy(policy.brain)
        self.policy.watch = [0.0, 0.0]
        self.codes: list[np.ndarray] = []
        self.every = every
        pops = policy.brain.connectome.populations
        self.index = {n: np.asarray(list(pops[n])) for n in region_names(policy)}
        self.regions: dict[str, list[np.ndarray]] = {}

    def begin(self, n: int) -> None:
        self.codes = []
        self.regions = {name: [] for name in self.index}
        self.policy.begin(n)

    def moment(self, x: np.ndarray, labels: Any, mask: np.ndarray, learn: bool) -> np.ndarray:
        a = self.policy.moment(x, None, mask, False)
        state = np.asarray(self.policy.brain.basal_ganglia.state.activation)
        self.codes.append(state[:, self.policy.association].astype(np.float64).copy())
        if self.every:
            for name, idx in self.index.items():
                self.regions[name].append(state[:, idx].astype(np.float64).copy())
        return a


def seeing_codes(policy: Any, things: list[tw.Thing], streams: int = 64) -> np.ndarray:
    """The association code of each thing, alone in view, as the parrot is about to name it."""
    layout = policy.layout
    rec = Recorder(policy)
    out = []
    q_words = ["what", "is", "it", "?"]
    at = len(q_words) + 1  # the question, the search moment, then the answer's first moment
    for i in range(0, len(things), streams):
        chunk = things[i:i + streams]
        items = [(tw.Scene([t]), tw.Question("it", list(q_words), None, {0: [t.kind]}, []))
                 for t in chunk]
        tc.run(rec, layout, items, "free", learn=False, cap=at + 1)
        out.append(rec.codes[at])
    return np.concatenate(out)


def hearing_codes(policy: Any, words: list[str], moment: int = 0) -> np.ndarray:
    """The association code as each word is read alone, nothing in view (``moment`` 0: the
    moment it is read; 1: the gap after it, the word held)."""
    layout = policy.layout
    rec = Recorder(policy)
    items = [(tw.Scene([]), tw.Question("echo", [w], None, {}, [w])) for w in words]
    tc.run(rec, layout, items, "free", learn=False, cap=moment + 1)
    return rec.codes[moment]


def things_of(rng: np.random.Generator, n: int, kinds: tuple[str, ...],
              held: bool = False) -> list[tw.Thing]:
    out = []
    for _ in range(n):
        k = str(rng.choice(kinds))
        cs = [c for c in tw.KINDS[k][3] if held or (k, c) not in tw.HELD_PAIRS or
              k in tw.NEW_KINDS]
        c = str(rng.choice(cs))
        place = tw.PLACES[int(rng.integers(len(tw.PLACES)))]
        out.append(tw.Thing(k, c, place, tw.look(k, c, place, rng)))
    return out


def fit_probe(x: np.ndarray, y: np.ndarray, steps: int = 500, l2: float = 1e-2) -> dict:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    w, b = np.zeros(z.shape[1]), 0.0
    for _ in range(steps):
        q = 1 / (1 + np.exp(-(z @ w + b)))
        w -= 0.5 * (z.T @ (q - y) / len(y) + l2 * w)
        b -= 0.5 * float(np.mean(q - y))
    return {"w": w, "b": b, "mu": mu, "sd": sd}


def probe_read(p: dict, x: np.ndarray, centre: np.ndarray | None = None) -> np.ndarray:
    """The probe's reading; with ``centre`` the codes are moved from their own mean to the mean
    the probe was fitted on (a route read against another route's probe)."""
    if centre is not None:
        x = x - centre + p["mu"]
    return 1 / (1 + np.exp(-(((x - p["mu"]) / p["sd"]) @ p["w"] + p["b"])))


def probes(policy: Any, rng: np.random.Generator, n: int = 512) -> dict[str, Any]:
    train = things_of(rng, n, tw.TRAIN_KINDS)
    test = things_of(rng, n // 2, tw.TRAIN_KINDS)
    new = things_of(rng, 128, tw.NEW_KINDS)
    xa, xb, xn = (seeing_codes(policy, s) for s in (train, test, new))
    out: dict[str, Any] = {"probe": {}, "accuracy": {}, "base_rate": {}, "new_objects": {}}
    for prop in PROPERTIES:
        ya = np.array([label(t, prop) for t in train], float)
        yb = np.array([label(t, prop) for t in test], float)
        yn = np.array([label(t, prop) for t in new], float)
        p = fit_probe(xa, ya)
        out["probe"][prop] = p
        out["accuracy"][prop] = float(np.mean((probe_read(p, xb) > .5) == (yb > .5)))
        out["base_rate"][prop] = float(max(yb.mean(), 1 - yb.mean()))
        out["new_objects"][prop] = float(np.mean((probe_read(p, xn) > .5) == (yn > .5)))
    out["seeing_mean"] = xa.mean(0)
    return out


def neighbours(policy: Any, rng: np.random.Generator, per: int = 32) -> dict[str, Any]:
    means = {}
    for k in tw.KINDS:
        means[k] = seeing_codes(policy, things_of(rng, per, (k,))).mean(0)
    keys = list(means)
    m = np.stack([means[k] for k in keys])
    m = m - m.mean(0)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    cos = m @ m.T
    nearest = {k: [keys[j] for j in np.argsort(-cos[i]) if j != i][:3] for i, k in enumerate(keys)}
    rows = {k: dict(zip(keys, np.round(cos[i], 3).tolist(), strict=True))
            for i, k in enumerate(keys)}
    return {"nearest": nearest, "cosine": rows}


def word_alone(policy: Any, pr: dict[str, Any], rng: np.random.Generator | None = None,
               per: int = 32) -> dict[str, Any]:
    """Each kind word read alone, nothing in view. ``properties``: the seeing probes read on the
    word's hearing code (each route centred on its own mean), scored as the balanced accuracy
    over the ten words of round, food and red (the kind's usual colour). ``same_thing``: the
    share of words whose hearing code lies nearest (cosine, each route centred) the mean seeing
    code of the kind it names, of the ten (chance .1)."""
    words = list(tw.TRAIN_KINDS)
    rng = rng or np.random.default_rng(0)
    seen = np.stack([seeing_codes(policy, things_of(rng, per, (k,))).mean(0) for k in words])
    seen = seen - seen.mean(0)
    seen /= np.linalg.norm(seen, axis=1, keepdims=True) + 1e-9
    truth = {"round": [tw.KINDS[w][0] == "round" for w in words],
             "food": [tw.KINDS[w][1] in tw.FOOD for w in words],
             "red": [tw.KINDS[w][3][0] == "red" for w in words]}
    out: dict[str, Any] = {}
    for moment in (0, 1):
        x = hearing_codes(policy, words, moment)
        centre = x.mean(0)
        reads = {w: {prop: float(probe_read(pr["probe"][prop], x[i:i + 1], centre)[0])
                     for prop in truth} for i, w in enumerate(words)}
        balanced = {}
        for prop, t in truth.items():
            got = np.array([reads[w][prop] > .5 for w in words])
            t_ = np.array(t)
            balanced[prop] = float((np.mean(got[t_]) + np.mean(~got[~t_])) / 2)
        h = x - centre
        h /= np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        nearest = (h @ seen.T).argmax(1)
        out[f"moment_{moment}"] = {"reads": reads, "balanced": balanced,
                                   "right": float(np.mean(list(balanced.values()))),
                                   "same_thing": float(np.mean(nearest == np.arange(len(words))))}
    return out


def lesion_direction(policy: Any, direction: np.ndarray) -> Any:
    """A copy with the component along ``direction`` (over the association's neurons) taken out
    of the association region's outgoing synapses."""
    q = copy.copy(policy)
    q.brain = copy.deepcopy(policy.brain)
    graph = q.brain.learner.brain
    eff = np.array(graph.efficacy, copy=True)
    w = q.brain.connectome
    assoc = policy.association
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    index = {int(a): i for i, a in enumerate(assoc)}
    pre, post = np.asarray(w.pre), np.asarray(w.post)
    out = np.flatnonzero(np.isin(pre, assoc) & ~np.isin(post, assoc))
    for p in np.unique(post[out]):
        rows = out[post[out] == p]
        col = np.zeros(len(assoc))
        for s in rows:
            col[index[int(pre[s])]] = eff[s]
        col = col - d * float(d @ col)
        for s in rows:
            eff[s] = col[index[int(pre[s])]]
    q.brain.learner.brain = graph.with_parameters(efficacy=eff, bias=np.asarray(graph.bias))
    return q


def targeted(prop: str) -> tuple[str, Callable[[tw.Scene, tw.Question], bool]]:
    """The form that needs ``prop`` and the exchanges of it that do."""
    if prop == "round":
        return "which_shape", lambda s, q: q.meta.get("value") == "round"
    if prop == "red":
        return "which_colour", lambda s, q: q.meta.get("value") == "red"
    if prop == "food":
        return "which_food", lambda s, q: True
    if prop == "left":
        return "where", lambda s, q: q.referent is not None and q.referent.direction == "left"
    if prop == "near":
        return "where", lambda s, q: q.referent is not None and not q.referent.far
    raise ValueError(prop)


SPARED = {"round": ("which_colour", "where"), "red": ("which_shape", "where"),
          "food": ("which_colour", "where"), "left": ("which_colour", "which_shape"),
          "near": ("which_colour", "which_shape")}


def read_where(policy: Any, rng: np.random.Generator, form: str,
               keep: Callable[[tw.Scene, tw.Question], bool], n: int = 64) -> float:
    import textparrot as tp

    items = []
    while len(items) < n:
        s, q = tp.draw(form, rng, split="test")
        if keep(s, q):
            items.append((s, q))
    eps = tc.run(policy, policy.layout, items, "free", learn=False)
    return float(np.mean([tc.score(e)["right"] for e in eps]))


def silencing(policy: Any, pr: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prop in ("round", "red", "food"):
        p = pr["probe"][prop]
        direction = p["w"] / p["sd"]
        lesioned = lesion_direction(policy, direction)
        form, keep = targeted(prop)
        row = {"form": form,
               "before": read_where(policy, np.random.default_rng([7, len(prop)]), form, keep),
               "after": read_where(lesioned, np.random.default_rng([7, len(prop)]), form, keep)}
        for other in SPARED[prop]:
            row[f"{other}_before"] = read_where(policy, np.random.default_rng([8, len(other)]),
                                                other, lambda s, q: True)
            row[f"{other}_after"] = read_where(lesioned, np.random.default_rng([8, len(other)]),
                                               other, lambda s, q: True)
        out[prop] = row
    return out


def owner_room() -> tw.Scene:
    """The owner's room: a red apple to the left, a blue block left of it (far), a white cup
    behind, a yellow key in front."""
    rng = np.random.default_rng(0)
    spec = [("apple", "red", ("left", "near")), ("block", "blue", ("left", "far")),
            ("cup", "white", ("behind", "near")), ("key", "yellow", ("front", "near"))]
    return tw.Scene([tw.Thing(k, c, p, tw.look(k, c, p, rng)) for k, c, p in spec])


OWNER = [("colour", ["what", "colour", "has", "the", "apple", "?"]),
         ("relation", ["what", "is", "left", "of", "the", "apple", "?"]),
         ("where", ["where", "is", "the", "apple", "?"]),
         ("where", ["where", "is", "the", "cup", "?"]),
         ("which_shape", ["what", "objects", "are", "round", "?"]),
         ("describe", ["what", "do", "you", "see", "?"]),
         ("describe_rel", ["what", "do", "you", "see", "left", "of", "the", "apple", "?"])]


def ask(policy: Any, scene: tw.Scene, words: list[str]) -> list[str]:
    """The parrot's reply to ``words`` about ``scene`` (greedy, nothing taught). The question's
    teacher answer is not used: the reply is the parrot's own."""
    q = tw.Question("ask", list(words), None, {}, [])
    eps = tc.run(policy, policy.layout, [(scene, q)], "free", learn=False)
    return eps[0].reply()


def teacher_reply(scene: tw.Scene, form: str, words: list[str]) -> list[str]:
    """The teacher's reply to one of the owner's questions about ``scene``."""
    rng = np.random.default_rng(0)
    for _ in range(4000):
        q = tw.question(form, scene, rng, depth=1)
        if q is not None and q.words == words:
            return q.reply(scene)
    for depth in (2,):
        for _ in range(4000):
            q = tw.question(form, scene, rng, depth=depth)
            if q is not None and q.words == words:
                return q.reply(scene)
    raise ValueError(words)


def owner_exam(policy: Any) -> list[dict[str, Any]]:
    room = owner_room()
    return [{"question": " ".join(w), "said": " ".join(ask(policy, room, w)),
             "want": " ".join(teacher_reply(room, f, w))} for f, w in OWNER]


def changed_scene(policy: Any, rng: np.random.Generator, n: int = 64) -> dict[str, float]:
    """The same question before and after its thing moved (where) or changed colour (colour):
    the share of pairs answered right both times, with different answers."""
    import textparrot as tp

    out = {}
    for form in ("where", "colour"):
        before, after = [], []
        while len(before) < n:
            s, q = tp.draw(form, rng, split="test", count=3)
            ref = q.referent
            assert ref is not None
            if form == "where":
                free = [p for p in tw.PLACES if p not in [t.place for t in s.things]]
                place = free[int(rng.integers(len(free)))]
                moved = tw.Thing(ref.kind, ref.colour, place, tw.look(ref.kind, ref.colour,
                                                                       place, rng))
            else:
                colours = [c for c in tw.KINDS[ref.kind][3]
                           if c != ref.colour and (ref.kind, c) not in tw.HELD_PAIRS]
                if not colours:
                    continue
                c = colours[int(rng.integers(len(colours)))]
                moved = tw.Thing(ref.kind, c, ref.place, tw.look(ref.kind, c, ref.place, rng))
            s2 = tw.Scene([moved if t is ref else t for t in s.things])
            if len(s2.find(moved.kind, None if q.depth == 1 else moved.colour)) != 1:
                continue
            q2 = tw.Question(form, list(q.words), moved,
                             {s2.things.index(moved): tw.place_words(moved) if form == "where"
                              else [moved.colour]}, [], q.depth)
            if q.depth == 2 and form == "colour":
                continue  # the question names the old colour
            before.append((s, q))
            after.append((s2, q2))
        e1 = tc.run(policy, policy.layout, before, "free", learn=False)
        e2 = tc.run(policy, policy.layout, after, "free", learn=False)
        out[form] = float(np.mean([tc.score(a)["right"] * tc.score(b)["right"]
                                   for a, b in zip(e1, e2, strict=True)]))
    return out


def _separation(x: np.ndarray, y: list[str]) -> float:
    """The share of the code's variance between the classes of ``y`` (a Fisher ratio)."""
    x = x - x.mean(0)
    total = float((x ** 2).sum()) + 1e-12
    between = 0.0
    for c in set(y):
        m = np.array([v == c for v in y])
        between += float(m.sum() * (x[m].mean(0) ** 2).sum())
    return between / total


def _decode(xa: np.ndarray, ya: list[str], xb: np.ndarray, yb: list[str]) -> float:
    """Nearest class mean (standardized on the fitting half): the share decoded right."""
    mu, sd = xa.mean(0), xa.std(0) + 1e-6
    za, zb = (xa - mu) / sd, (xb - mu) / sd
    classes = sorted(set(ya))
    means = np.stack([za[np.array([v == c for v in ya])].mean(0) for c in classes])
    got = [classes[int(np.argmin(((means - z) ** 2).sum(1)))] for z in zb]
    return float(np.mean([g == t for g, t in zip(got, yb, strict=True)]))


QUESTIONS = {"name": ["what", "is", "it", "?"], "colour": ["what", "colour", "is", "it", "?"],
             "shape": ["what", "shape", "is", "it", "?"]}


def regions(policy: Any, rng: np.random.Generator, n: int = 384) -> dict[str, Any]:
    """Where in the depth the concept codes live, and whether the question switches what a region
    reads out. The same things alone in view under three questions ("what is it", "what colour
    is it", "what shape is it"); each region's code at the moment the parrot is about to answer:
    the colour and the shape decoded from it (nearest class mean, fitted on half the things) and
    the share of its variance between colours and between shapes. A region the question's
    top-down input switches holds more colour under the colour question and more shape under the
    shape question, for the same things."""
    layout = policy.layout
    things = things_of(rng, n, tw.TRAIN_KINDS)
    colours = [t.colour for t in things]
    shapes = [t.shape for t in things]
    kinds = [t.kind for t in things]
    half = n // 2
    out: dict[str, Any] = {}
    codes: dict[str, dict[str, np.ndarray]] = {}
    for qname, words in QUESTIONS.items():
        rec = Recorder(policy, every=True)
        at = len(words) + 1
        got: dict[str, list[np.ndarray]] = {r: [] for r in rec.index}
        for i in range(0, n, 64):
            chunk = things[i:i + 64]
            items = [(tw.Scene([t]), tw.Question("it", list(words), None, {0: [t.kind]}, []))
                     for t in chunk]
            tc.run(rec, layout, items, "free", learn=False, cap=at + 1)
            for r in rec.index:
                got[r].append(rec.regions[r][at])
        codes[qname] = {r: np.concatenate(v) for r, v in got.items()}
    for r in region_names(policy):
        row: dict[str, Any] = {}
        for qname in QUESTIONS:
            x = codes[qname][r]
            row[qname] = {
                "colour_decoded": _decode(x[:half], colours[:half], x[half:], colours[half:]),
                "shape_decoded": _decode(x[:half], shapes[:half], x[half:], shapes[half:]),
                "kind_decoded": _decode(x[:half], kinds[:half], x[half:], kinds[half:]),
                "colour_share": _separation(x, colours),
                "shape_share": _separation(x, shapes)}
        row["switch"] = {
            "colour": row["colour"]["colour_share"] - row["shape"]["colour_share"],
            "shape": row["shape"]["shape_share"] - row["colour"]["shape_share"]}
        out[r] = row
    return out
