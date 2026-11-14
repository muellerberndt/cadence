"""Looking inside the night's brain (cadence#334): the concepts as codes, and the exam.

A thing's **code** is the settled activity of the association region (and of the two regions
below it) at the moment the parrot hears the last word of a question about it, its focus on it.
Nothing in the brain names a concept; these readings find what the brain learned:

- **probes:** a logistic readout from the code to a property (round, red, food, metal, near,
  left), fitted on scenes of the training worlds and read on held-out scenes;
- **neighbours:** each kind's mean code, and which kinds lie nearest (cosine);
- **a word alone:** the code when the parrot hears one word in an empty room, read by the probes
  (does "apple" alone bring up round, red and food?);
- **silencing:** the units most tuned to a property disconnected (their synapses at zero), or the
  property's probe direction projected out of the association region's outgoing synapses; the
  answers that need that property, and the others, read again;
- **a new object:** a kind never shown (a lemon): where its code lands, and "can you eat this?";
- **a changed scene:** the same question after a thing moved or changed colour.

``exam`` asks the owner's five exchanges in their own room, then held-out ones of the same forms.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import night as nt  # noqa: E402

PROPERTIES = ("round", "red", "food", "metal", "near", "left")
# the association and the processing regions below it, top first (a deeper brain has more)
REGIONS = ("association",) + tuple(f"module_{k}" for k in range(7, -1, -1))


def label(t: nt.Thing, heading: float, prop: str) -> int:
    if prop == "round":
        return int(t.kind in nt.ROUND)
    if prop == "red":
        return int(t.colour == "red")
    if prop == "food":
        return int(t.material in nt.EDIBLE)
    if prop == "metal":
        return int(t.material == "metal")
    if prop == "near":
        return int(float(np.linalg.norm(t.place)) <= nt.FAR)
    if prop == "left":
        return int(nt.direction_of(heading, t.place) == "left")
    raise ValueError(prop)


def frozen(p: nt.Parrot) -> Any:
    brain = copy.deepcopy(p.brain)
    brain.reset()
    return brain


def settle(p: nt.Parrot, brain: Any, rooms: list[nt.Room], heard: list[list[str | None]],
           rng: np.random.Generator) -> tuple[dict[str, np.ndarray], list[str | None]]:
    """Lives the moments of ``heard`` in each room (one a stream, lockstep, greedy, nothing
    learned) and returns the codes at the last heard moment and the first word said there."""
    n = len(rooms)
    width = max(len(h) for h in heard)
    heard = [[None] * (width - len(h)) + list(h) for h in heard]
    exs = [nt.Exchange(r, h, width - 1, None) for r, h in zip(rooms, heard, strict=True)]
    own: list[str | None] = [None] * n
    codes: dict[str, np.ndarray] = {}
    said: list[str | None] = [None] * n
    for t in range(width):
        x = p.observe(exs, t, own, np.zeros(n, dtype=bool), rng)
        action = np.asarray(brain.act(x, greedy=True)).reshape(n, -1)
        own = [None] * n
        if t == width - 1:
            state = brain.basal_ganglia.state.activation
            pops = brain.connectome.populations
            codes = {r: np.asarray(state[:, list(pops[r])], dtype=np.float64).copy()
                     for r in REGIONS if r in pops}
            said = [p.said(a) for a in action[:, 0]]
    return codes, said


def scenes(p: nt.Parrot, rng: np.random.Generator, count: int, held: set[tuple[str, str]], *,
           test: bool, kinds: tuple[str, ...] | None = None) -> list[tuple[nt.Room, nt.Thing]]:
    """``count`` rooms, each with one thing in focus (of ``kinds`` where given)."""
    out = []
    while len(out) < count:
        r = nt.room(rng, int(rng.integers(3, 6)), held, test=test)
        if kinds is not None:
            k = str(rng.choice(kinds))
            spec = {**nt.KINDS, **nt.NEW_KINDS}
            if k not in [t.kind for t in r.things]:
                r.things[0] = nt.thing(rng, k, r.things[0].place,
                                       colour=str(rng.choice(spec[k][3] or tuple(nt.COLOURS))))
        x = [t for t in r.things if kinds is None or t.kind in kinds][0]
        for t in r.things:
            t.focus = t is x
        out.append((r, x))
    return out


def codes_of(p: nt.Parrot, pairs: list[tuple[nt.Room, nt.Thing]], rng: np.random.Generator,
             words: list[str] | None = None) -> tuple[np.ndarray, list[str | None]]:
    """The association codes (with the regions below, concatenated) of each focused thing at the
    last word of ``words`` ("what is this" by default)."""
    words = words or ["what", "is", "this"]
    brain = frozen(p)
    rows, said = [], []
    for i in range(0, len(pairs), p.streams):
        chunk = pairs[i:i + p.streams]
        pad = p.streams - len(chunk)
        rooms = [r for r, _ in chunk] + [chunk[-1][0]] * pad
        got, s = settle(p, brain, rooms, [list(words)] * p.streams, rng)
        rows.append(np.concatenate([got[r] for r in REGIONS if r in got], axis=1)[:len(chunk)])
        said += s[:len(chunk)]
        brain.reset()
    return np.concatenate(rows), said


def fit_probe(x: np.ndarray, y: np.ndarray, *, steps: int = 400, l2: float = 1e-2
              ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    w, b = np.zeros(z.shape[1]), 0.0
    for _ in range(steps):
        q = 1 / (1 + np.exp(-(z @ w + b)))
        w -= 0.5 * (z.T @ (q - y) / len(y) + l2 * w)
        b -= 0.5 * float(np.mean(q - y))
    return w, b, mu, sd


def probe_read(probe: tuple[np.ndarray, float, np.ndarray, np.ndarray], x: np.ndarray
               ) -> np.ndarray:
    w, b, mu, sd = probe
    return 1 / (1 + np.exp(-(((x - mu) / sd) @ w + b)))


def probes(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
           count: int = 512) -> dict[str, Any]:
    """Each property's probe, fitted on training scenes and read on held-out ones."""
    train = scenes(p, rng, count, held, test=False)
    test = scenes(p, rng, count // 2, held, test=True)
    xa, _ = codes_of(p, train, rng)
    xb, _ = codes_of(p, test, rng)
    out: dict[str, Any] = {"probes": {}, "accuracy": {}}
    for prop in PROPERTIES:
        ya = np.array([label(t, r.heading, prop) for r, t in train], float)
        yb = np.array([label(t, r.heading, prop) for r, t in test], float)
        if ya.min() == ya.max():
            continue
        pr = fit_probe(xa, ya)
        out["probes"][prop] = pr
        out["accuracy"][prop] = float(np.mean((probe_read(pr, xb) > 0.5) == (yb > 0.5)))
        out.setdefault("base_rate", {})[prop] = float(max(yb.mean(), 1 - yb.mean()))
    return out


def probes_by_region(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
                     count: int = 512) -> dict[str, dict[str, float]]:
    """Where the concept code lives: each property's probe fitted on one region's code alone,
    read on held-out scenes (and the share a probe of that region would read by chance)."""
    train = scenes(p, rng, count, held, test=False)
    test = scenes(p, rng, count // 2, held, test=True)
    pops = p.brain.connectome.populations
    widths = [(r, len(pops[r])) for r in REGIONS if r in pops]
    xa, _ = codes_of(p, train, rng)
    xb, _ = codes_of(p, test, rng)
    out: dict[str, dict[str, float]] = {}
    at = 0
    for region, width in widths:
        part_a, part_b = xa[:, at:at + width], xb[:, at:at + width]
        at += width
        for prop in PROPERTIES:
            ya = np.array([label(t, r.heading, prop) for r, t in train], float)
            yb = np.array([label(t, r.heading, prop) for r, t in test], float)
            if ya.min() == ya.max():
                continue
            pr = fit_probe(part_a, ya)
            out.setdefault(region, {})[prop] = float(
                np.mean((probe_read(pr, part_b) > 0.5) == (yb > 0.5)))
    return out


def question_gating(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
                    count: int = 384) -> dict[str, Any]:
    """Does the question switch what a region reads of one thing? The same focused things under
    "what is this" and "what colour has the <kind>": per region, a probe of the thing's colour
    (red) and of its shape (round) fitted and read under each question (what it then says, in
    the gap, is the forms' reading of ``name`` and ``colour_of``)."""
    pairs = scenes(p, rng, count, held, test=False)
    pops = p.brain.connectome.populations
    out: dict[str, Any] = {}
    for label_q, words_of in (("what is this", lambda t: ["what", "is", "this"]),
                              ("what colour has the X",
                               lambda t: ["what", "colour", "has", "the", t.kind])):
        brain = frozen(p)
        codes: dict[str, list[np.ndarray]] = {}
        said: list[str | None] = []
        for i in range(0, len(pairs), p.streams):
            chunk = pairs[i:i + p.streams]
            pad = p.streams - len(chunk)
            rooms = [r for r, _ in chunk] + [chunk[-1][0]] * pad
            heard = [words_of(t) for _, t in chunk] + [words_of(chunk[-1][1])] * pad
            got, s = settle(p, brain, rooms, heard, rng)
            for r in got:
                codes.setdefault(r, []).append(got[r][:len(chunk)])
            said += s[:len(chunk)]
            brain.reset()
        half = len(pairs) // 2
        reads: dict[str, dict[str, float]] = {}
        for region in [r for r in REGIONS if r in pops and r in codes]:
            x = np.concatenate(codes[region])
            for prop in ("red", "round"):
                y = np.array([label(t, r.heading, prop) for r, t in pairs], float)
                if y[:half].min() == y[:half].max():
                    continue
                pr = fit_probe(x[:half], y[:half])
                reads.setdefault(region, {})[prop] = float(
                    np.mean((probe_read(pr, x[half:]) > 0.5) == (y[half:] > 0.5)))
        out[label_q] = reads
    return out


FACETS = ("kind", "colour", "round", "material", "edible", "direction", "distance")


def facet(t: nt.Thing, heading: float, name: str) -> str:
    """A thing's value of one facet, as a class label."""
    if name == "kind":
        return t.kind
    if name == "colour":
        return t.colour
    if name == "round":
        return str(t.kind in nt.ROUND)
    if name == "material":
        return t.material
    if name == "edible":
        return str(t.material in nt.EDIBLE)
    if name == "direction":
        return nt.direction_of(heading, t.place)
    if name == "distance":
        return "far" if float(np.linalg.norm(t.place)) > nt.FAR else "near"
    raise ValueError(name)


def fit_classes(x: np.ndarray, labels: list[str], l2: float = 1.0) -> tuple[Any, ...]:
    """A linear read of a class label (ridge onto one-hot), standardized on the fit's codes."""
    classes = sorted(set(labels))
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    y = np.zeros((len(labels), len(classes)))
    y[np.arange(len(labels)), [classes.index(v) for v in labels]] = 1.0
    w = np.linalg.solve(z.T @ z + l2 * len(z) * np.eye(z.shape[1]), z.T @ y)
    return classes, mu, sd, w


def read_classes(fit: tuple[Any, ...], x: np.ndarray) -> list[str]:
    classes, mu, sd, w = fit
    return [classes[int(i)] for i in np.argmax(((x - mu) / sd) @ w, axis=1)]


def focus_codes(p: nt.Parrot, pairs: list[tuple[nt.Room, nt.Thing]], rng: np.random.Generator,
                moments: int = 2) -> dict[str, np.ndarray]:
    """The settled state, per region, with a thing in focus and nothing heard (no question)."""
    brain = frozen(p)
    out: dict[str, list[np.ndarray]] = {}
    for i in range(0, len(pairs), p.streams):
        chunk = pairs[i:i + p.streams]
        rooms = [r for r, _ in chunk] + [chunk[-1][0]] * (p.streams - len(chunk))
        got, _ = settle(p, brain, rooms, [[None] * moments] * p.streams, rng)
        for r in got:
            out.setdefault(r, []).append(got[r][:len(chunk)])
        brain.reset()
    return {r: np.concatenate(v) for r, v in out.items()}


def state_completeness(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
                       count: int = 768) -> dict[str, Any]:
    """With a thing in focus and no question: each facet (kind, colour, shape, material, edible,
    direction, distance) read linearly from the one settled state, per region, fitted on
    training scenes and read on held-out ones, beside the base rate (the commonest value)."""
    train = scenes(p, rng, count, held, test=False)
    test = scenes(p, rng, count // 2, held, test=True)
    xa, xb = focus_codes(p, train, rng), focus_codes(p, test, rng)
    out: dict[str, Any] = {"base": {}, "read": {}, "fits": {}}
    for name in FACETS:
        ya = [facet(t, r.heading, name) for r, t in train]
        yb = [facet(t, r.heading, name) for r, t in test]
        out["base"][name] = max(yb.count(v) for v in set(yb)) / len(yb)
        for region in xa:
            fit = fit_classes(xa[region], ya)
            out["read"].setdefault(region, {})[name] = float(
                np.mean([a == b for a, b in zip(read_classes(fit, xb[region]), yb,
                                                strict=True)]))
            if region == "association":
                out["fits"][name] = fit
    return out


def completion(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
               fits: dict[str, Any], per: int = 16) -> dict[str, Any]:
    """Cue one facet alone and read every other facet from the association with the focus
    probes (``state_completeness``'s fits): a kind's word heard with nothing in view, and a
    colour word heard alone (which things are that colour). Each cell is the share of reads that
    give the facet the cue implies (a kind's shape, material class and edibility; a colour)."""
    empty = nt.Room([], 0.0)
    out: dict[str, Any] = {}
    brain = frozen(p)
    for word in nt.KIND_NAMES:
        got, _ = settle(p, brain, [empty] * p.streams, [[word, None]] * p.streams, rng)
        brain.reset()
        x = got["association"][:per]
        row: dict[str, Any] = {}
        spec = nt.KINDS[word]
        truth = {"kind": word, "round": str(word in nt.ROUND),
                 "edible": str("fruit" in spec[2])}
        for name in ("kind", "round", "edible", "colour", "material"):
            said = read_classes(fits[name], x)
            if name in truth:
                row[name] = float(np.mean([v == truth[name] for v in said]))
            else:
                row[name] = max(set(said), key=said.count)  # the value the word evokes most
        out[f"word {word}"] = row
    for colour in nt.COLOURS:
        got, _ = settle(p, brain, [empty] * p.streams, [[colour, None]] * p.streams, rng)
        brain.reset()
        said = read_classes(fits["colour"], got["association"][:per])
        kinds = read_classes(fits["kind"], got["association"][:per])
        out[f"word {colour}"] = {"colour": float(np.mean([v == colour for v in said])),
                                 "kind": max(set(kinds), key=kinds.count)}
    return out


def sound_in_state(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
                   count: int = 128) -> dict[str, float]:
    """On focus alone (no question): how often the state's pattern recalls a word from the word
    records (the three nearest agree), and how often that word is the thing's name."""
    if p.records is None or not p.records.count:
        return {"records": 0.0, "recalled": 0.0, "right": 0.0}
    pairs = scenes(p, rng, count, held, test=False)
    codes = focus_codes(p, pairs, rng)["association"]
    names = {w: p.forms(w) for w in nt.WORDS}
    recalled, right = 0, 0
    for (_, t), cue in zip(pairs, codes, strict=True):
        heads = min(3, int(p.records.count))
        got = p.records.read(cue, heads=heads, credit=False)
        if got.first() < 0:
            continue
        value = p.records.value(got)
        if any(got.first(h) < 0 or not np.array_equal(p.records.value(got, h), value)
               for h in range(1, heads)):
            continue
        recalled += 1
        word = next((w for w, c in names.items() if np.array_equal(c, value)), None)
        right += int(word == t.kind)
    return {"records": float(p.records.count), "recalled": recalled / len(pairs),
            "right": right / len(pairs)}


def neighbours(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
               per: int = 32) -> dict[str, Any]:
    """Each kind's mean code (the lemon's too, never shown) and its cosine to the others."""
    means = {}
    for k in nt.KIND_NAMES + tuple(nt.NEW_KINDS):
        x, _ = codes_of(p, scenes(p, rng, per, held, test=False, kinds=(k,)), rng)
        means[k] = x.mean(0)
    keys = list(means)
    m = np.stack([means[k] - np.mean(list(means.values()), axis=0) for k in keys])
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    cos = m @ m.T
    nearest = {k: [keys[j] for j in np.argsort(-cos[i]) if j != i][:3] for i, k in enumerate(keys)}
    return {"kinds": keys, "cosine": cos.round(3).tolist(), "nearest": nearest, "means": means}


def word_alone(p: nt.Parrot, rng: np.random.Generator, pr: dict[str, Any],
               words: tuple[str, ...] = ("apple", "ball", "block", "banana", "key")
               ) -> dict[str, dict[str, float]]:
    """The probes' readings when one word is heard in an empty room."""
    brain = frozen(p)
    empty = nt.Room([], heading=0.0)
    out: dict[str, dict[str, float]] = {}
    for w in words:
        got, _ = settle(p, brain, [empty] * p.streams, [[w]] * p.streams, rng)
        x = np.concatenate([got[r] for r in REGIONS if r in got], axis=1)[:1]
        out[w] = {prop: float(probe_read(probe, x)[0]) for prop, probe in pr["probes"].items()}
        brain.reset()
    return out


def lesion_units(p: nt.Parrot, units: np.ndarray) -> nt.Parrot:
    """A copy whose ``units`` (indices into the brain's neurons) are disconnected."""
    q = copy.copy(p)
    q.brain = copy.deepcopy(p.brain)
    graph = q.brain.learner.brain
    eff = np.array(graph.efficacy, copy=True)
    w = q.brain.connectome
    cut = np.isin(w.pre, units) | np.isin(w.post, units)
    eff[cut] = 0.0
    q.brain.learner.brain = graph.with_parameters(efficacy=eff)
    return q


def lesion_direction(p: nt.Parrot, direction: np.ndarray) -> nt.Parrot:
    """A copy with the component along ``direction`` (over the association's neurons) taken out
    of the association region's outgoing synapses."""
    q = copy.copy(p)
    q.brain = copy.deepcopy(p.brain)
    graph = q.brain.learner.brain
    eff = np.array(graph.efficacy, copy=True)
    w = q.brain.connectome
    assoc = np.asarray(list(w.populations["association"]))
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    index = {int(a): i for i, a in enumerate(assoc)}
    out = np.flatnonzero(np.isin(w.pre, assoc) & ~np.isin(w.post, assoc))
    for post in np.unique(w.post[out]):
        rows = out[w.post[out] == post]
        col = np.zeros(len(assoc))
        for s in rows:
            col[index[int(w.pre[s])]] = eff[s]
        col = col - d * float(d @ col)
        for s in rows:
            eff[s] = col[index[int(w.pre[s])]]
    q.brain.learner.brain = graph.with_parameters(efficacy=eff)
    return q


def forms_read(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
               forms: list[str], n: int = 64) -> dict[str, float]:
    return nt.read(p, rng, forms, held, n)


def owner_room() -> nt.Room:
    """The owner's room: a red apple to the left and far, a block left of it, a cup behind, a
    key in front."""
    rng = np.random.default_rng(0)

    def at(deg: float, r: float) -> np.ndarray:
        return np.array([r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))])

    things = [nt.thing(rng, "apple", at(70, 1.5), colour="red", material="fruit"),
              nt.thing(rng, "block", at(120, 1.0), colour="blue", material="wood"),
              nt.thing(rng, "cup", at(180, 0.8), colour="white", material="glass"),
              nt.thing(rng, "key", at(-20, 1.0), colour="yellow", material="metal")]
    return nt.Room(things, heading=0.0)


OWNER = [  # the owner's five exchanges in the owner's room, the teacher's answers, the focus
    (["what", "colour", "has", "the", "apple"], ["red"], "apple"),
    (["what", "is", "left", "of", "the", "apple"], ["block"], "apple"),
    (["where", "is", "the", "apple"], ["left", "far"], "apple"),
    (["where", "is", "the", "cup"], ["behind"], "cup"),
    (["what", "objects", "are", "round"], ["apple", "cup"], None),
]


def exam_owner(p: nt.Parrot, rng: np.random.Generator) -> list[dict[str, Any]]:
    """The owner's five, each asked on every stream of a block, read greedily."""
    out = []
    for words, answer, focus in OWNER:
        exs = []
        for _ in range(p.streams):
            r = owner_room()
            for t in r.things:  # the apple clicked (or attended to) while the question is asked
                t.focus = t.kind == focus
            ref = next((t for t in r.things if t.focus), None)
            several = words[1] == "objects"
            scan = words[1] == "objects" or words[2] == "left"  # questions over the scene
            q = nt.Question(list(words), list(answer), "owner", focus=ref, several=several,
                            scan=scan)
            exs.append(nt.make_exchange(r, q, "turn", rng))
        got = nt.live(p, exs, rng, learn=False)
        out.append({"question": " ".join(words), "answer": answer,
                    "said": got["replies"][0], "right": float(np.mean(got["right"]))})
    return out


def lemon_room(rng: np.random.Generator, held: set[tuple[str, str]]) -> nt.Room:
    """A room with a lemon (a kind never shown in training) in focus."""
    r = nt.room(rng, int(rng.integers(2, 5)), held)
    r.things[0] = nt.thing(rng, "lemon", r.things[0].place)
    for t in r.things:
        t.focus = t is r.things[0]
    return r


def new_word(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]], *,
             namings: int = 3, trials: int = 80, check: int = 16) -> dict[str, Any]:
    """A new association by pistachios alone, on a copy of the parrot, one exchange at a time:
    a visitor names the lemon ``namings`` times ("what is this?" and the nonce word, with the
    lemon in focus; the parrot's turn after it, a pistachio when it says the word back), then
    asks "what is this?" with the focus on a lemon, a pistachio when the parrot says the new
    word. No correction and no teacher's answer after the namings. Every 4 trials a frozen copy
    is read on ``check`` fresh lemons; the gate's count is the trials (and pistachios) until it
    says the word at .9. The turn opens in the gap after the visitor stops, as in the chamber."""
    q = copy.copy(p)
    q.brain = copy.deepcopy(p.brain)
    q.brain.reset()
    q.streams = 1
    word = nt.NEW_WORD
    log: list[dict[str, Any]] = []
    words = ["what", "is", "this"]

    def asked(mode: str) -> nt.Exchange:
        r = lemon_room(rng, held)
        return nt.make_exchange(r, nt.Question(list(words), [word], "new_name",
                                               focus=r.things[0]), mode, rng)

    for _ in range(namings):  # the visitor's naming: the word heard with the lemon in focus
        nt.live(q, [asked("model")], rng)
    paid = 0
    learned_at = None
    for trial in range(1, trials + 1):
        got = nt.live(q, [asked("turn")], rng)
        paid += int(got["right"][0])
        if trial % 4 == 0:
            ok = []
            for _ in range(check):
                ok += nt.live(q, [asked("turn")], rng, learn=False)["right"]
            log.append({"trial": trial, "pistachios": paid, "read": float(np.mean(ok))})
            if np.mean(ok) >= 0.9 and learned_at is None:
                learned_at = (trial, paid)
                break
    q.streams = p.streams
    q.brain.reset()
    names = nt.read(q, rng, ["name"], held, 32)["name"]
    return {"namings": namings, "learned_at_trial": None if learned_at is None else learned_at[0],
            "pistachios": None if learned_at is None else learned_at[1], "curve": log,
            "old_names_after": names, "parrot": q}
