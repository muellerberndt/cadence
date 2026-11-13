"""The audit's harness (cadence#326): a founder's life replayed with probes beside it.

The brain lives the founder's script as ``words.life`` runs it. At every test question each
probe answers the same question with one thing changed: the brain (a twin lexicon with
synapses cut, the map emptied, the body turned, a room seen in part) or the question (its words
shuffled, a word added). A probe is scored against the world's answer (``ok``), against the
answer its change predicts where it has one (``follows``), and by whether it said the word for
"I do not know" (``refused``).

Probes read the brain through its public parts only: ``name``, ``practice``, ``ask``,
``lexicon`` (``settings``, ``arrays``, ``from_arrays``, ``groups``, ``unit``, ``forms``),
``map`` and ``body``.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
CHAMBER = HERE.parent
for path in (str(CHAMBER), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import words  # noqa: E402
import world  # noqa: E402

# a short life for smoke runs on a loaded machine; the audit's readings use the defaults
SHORT = {"naming_rooms": 16, "namings": 15, "practice_rooms": 10, "asked": 10, "test_rooms": 4,
         "unseen_rooms": 3, "new_rooms": 3}
# the founders the audit reads: the declared run's twelve (a reading, never a gate)
FOUNDERS = list(range(32600000, 32600012))


def question_words(lex: world.Lexicon) -> set[str]:
    return set(lex.question.values())


def content(lex: world.Lexicon, text: str) -> list[str]:
    """The words of a question that are not question words."""
    asked = question_words(lex)
    return [w for w in text.split() if w not in asked]


class Harness:
    """A founder's brain, its world lexicon and the room it is in."""

    def __init__(self, seed: int, brain: Any, settings: dict[str, Any]):
        self.seed, self.brain, self.settings = int(seed), brain, settings
        self.lex = world.lexicon(seed)
        self.unseen = world.unseen_colours(seed)
        self.room: world.Room | None = None
        self.rooms = 0  # rooms entered, so a probe sets up once a room
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 77]))
        self._twins: dict[Any, Any] = {}

    # -- twins of the brain's lexicon, cached by the lexicon's lessons

    def twin_lexicon(self, key: Any, cut: Any) -> Any:
        """A copy of the brain's lexicon with the synapses ``cut`` (flags, or a function of
        the lexicon giving flags) at zero."""
        from cadence.words import Lexicon

        lexicon = self.brain.lexicon
        full = (key, int(lexicon.lessons))
        if full not in self._twins:
            arrays = {k: np.array(v, copy=True) for k, v in lexicon.arrays().items()}
            flags = cut(lexicon) if callable(cut) else cut
            arrays["efficacy"][np.asarray(flags, dtype=bool)] = 0.0
            self._twins[full] = Lexicon.from_arrays(lexicon.settings, arrays)
        return self._twins[full]

    def twin(self, key: Any, cut: Any, body: Any = None, map: Any = None) -> Any:
        """A brain that shares this brain's map and body (unless given) and holds a twin
        lexicon."""
        from cadence.words import Words

        return Words(self.brain.map if map is None else map, lexicon=self.twin_lexicon(key, cut),
                     body=self.brain.body if body is None else body)

    def with_lexicon(self, map: Any, body: Any) -> Any:
        """A brain with this brain's own lexicon (shared, unchanged), on another map and body."""
        from cadence.words import Words

        return Words(map, lexicon=self.brain.lexicon, body=body)


def word_synapses(lexicon: Any, form: str) -> np.ndarray:
    """Flags on the lexicon's synapses: both directions of every pair of the word's neuron."""
    flags = np.zeros(int(lexicon.connectome.synapses), dtype=bool)
    unit = lexicon.unit(form)
    if unit is not None:
        neuron = int(lexicon.word[unit])
        pre, post = np.asarray(lexicon.connectome.pre), np.asarray(lexicon.connectome.post)
        flags |= (pre == neuron) | (post == neuron)
    return flags


def group_synapses(lexicon: Any, group: str) -> np.ndarray:
    """Flags on the lexicon's synapses: every pair of a concept neuron of ``group``."""
    neurons = lexicon.concept[lexicon.groups[group]]
    pre, post = np.asarray(lexicon.connectome.pre), np.asarray(lexicon.connectome.post)
    return np.isin(pre, neurons) | np.isin(post, neurons)


# -- maps made beside the brain's

def fresh_map(seed: int) -> Any:
    """A map as ``words.make_brain`` makes it, empty."""
    from cadence.space.map import Map

    return Map(world.PLANE, world.CELLS, units=world.KINDS, values={"colour": world.COLOURS},
               seed=int(seed))


def look_around(map: Any, body: Any, r: world.Room, *, reports: Any = None, lit: Any = None,
                turns: Any = None, reset: bool = True) -> None:
    """``words.look_around`` on a given map and body: ``reports(room, heading)`` in place of
    the camera's, ``lit(heading)`` in place of its sight, ``turns`` the eighths of a turn
    looked at (all eight unless given)."""
    if reset:
        map.reset()
    eighths = list(range(8)) if turns is None else list(turns)
    for i in eighths + [None]:
        heading = r.heading + i * math.pi / 4 if i is not None else r.heading
        body.heading = heading
        seen = world.seen(r, heading) if reports is None else reports(r, heading)
        map.moment(seen=seen, lit=world.camera(heading) if lit is None else lit(heading))


def snap_map(map: Any) -> dict[str, Any]:
    keys = ("potential", "clock", "lost", "doubt", "_sensed", "reading", "_views")
    return {k: copy.deepcopy(getattr(map, k)) for k in keys}


def restore_map(map: Any, held: dict[str, Any]) -> None:
    for k, v in held.items():
        setattr(map, k, copy.deepcopy(v))


# -- the tally

def tally(book: dict[str, dict[str, Any]], gate: str, result: dict[str, Any]) -> None:
    row = book.setdefault(gate, {"n": 0, "ok": 0, "follows": 0, "follows_n": 0, "refused": 0,
                                 "same": 0, "same_n": 0, "examples": []})
    row["n"] += 1
    row["ok"] += int(bool(result.get("ok")))
    row["refused"] += int(bool(result.get("refused")))
    if result.get("follows") is not None:
        row["follows_n"] += 1
        row["follows"] += int(bool(result["follows"]))
    if result.get("same") is not None:
        row["same_n"] += 1
        row["same"] += int(bool(result["same"]))
    if "category" in result:
        counts = row.setdefault("categories", {})
        counts[result["category"]] = counts.get(result["category"], 0) + 1
    if len(row["examples"]) < 3 and not result.get("ok"):
        row["examples"].append({k: result[k] for k in ("question", "said", "answer", "predicted")
                                if k in result})


POOLS = {"W1": ("W1_practised", "W1_named_only"), "W2": ("W2_where", "W2_colour"),
         "W3": ("W3_colour", "W3_by_colour"), "W4": ("W4_never_seen", "W4_never_taught"),
         "W5": ("W5_by_colour", "W5_by_place")}


def shares(book: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Each gate's and pool's shares: right, following the change, refused, as before."""
    out: dict[str, Any] = {}
    names = dict(POOLS)
    names.update({g: (g,) for g in book})
    for name, parts in names.items():
        rows = [book[p] for p in parts if p in book]
        n = sum(r["n"] for r in rows)
        if not n:
            continue
        fn = sum(r["follows_n"] for r in rows)
        sn = sum(r["same_n"] for r in rows)
        out[name] = {"n": n, "ok": sum(r["ok"] for r in rows) / n,
                     "refused": sum(r["refused"] for r in rows) / n,
                     "follows": (sum(r["follows"] for r in rows) / fn) if fn else None,
                     "same": (sum(r["same"] for r in rows) / sn) if sn else None}
        counts: dict[str, int] = {}
        for r in rows:
            for k, v in r.get("categories", {}).items():
                counts[k] = counts.get(k, 0) + v
        if counts:
            out[name]["categories"] = {k: v / n for k, v in sorted(counts.items())}
    return out


def scored(h: Harness, q: world.Question, said: list[str], predicted: list[str] | None = None,
           base: list[str] | None = None) -> dict[str, Any]:
    return {"question": q.text, "said": said, "answer": q.answer, "ok": world.right(q, said),
            "refused": said == [h.lex.none],
            "follows": None if predicted is None else (
                sorted(said) == sorted(predicted) and len(said) == len(predicted)),
            "predicted": predicted,
            "same": None if base is None else said == base}


# -- the replay

def replay(seed: int, probes: list[Any], settings: dict[str, Any] | None = None,
           brain: Any = None, after: Any = None, practise: Any = None) -> dict[str, Any]:
    """The founder's life with ``probes`` answering beside the brain at every test question;
    ``after(harness)`` runs on the brain the life left (scenes of its own) and gives a dict;
    ``practise(harness, question)`` takes each practice question in place of the teacher's
    told answer. Gives each probe's shares by gate, and the brain's own."""
    settings = {**words.DEFAULTS, **(settings or {})}
    events = words.script(seed, **settings)
    h = Harness(seed, words.make_brain(seed) if brain is None else brain, settings)
    books: dict[str, dict[str, dict[str, Any]]] = {"brain": {}}
    books.update({p.name: {} for p in probes})
    began = time.time()
    entered: dict[str, int] = {}
    for event in events:
        kind = event[0]
        if kind == "room":
            h.room = event[1]
            h.rooms += 1
            words.look_around(h.brain, h.room)
        elif kind == "turn":
            assert h.room is not None
            words.turn(h.brain, h.room, event[1])
        elif kind == "name":
            h.brain.name(event[2], event[1])
        elif kind == "practice":
            q = event[1]
            if practise is not None:
                practise(h, q)
            else:
                h.brain.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
        else:
            gate, q = event[1], event[2]
            said = h.brain.ask(q.text)
            tally(books["brain"], gate, scored(h, q, said))
            if gate == "W5_by_place" and q.about == "several":  # a place that holds things
                tally(books["brain"], "W5_by_place_held", scored(h, q, said))
            for p in probes:
                if entered.get(p.name) != h.rooms:
                    entered[p.name] = h.rooms
                    p.room(h)
                result = p.ask(h, gate, q, said)
                if result is not None:
                    tally(books[p.name], result.get("gate", gate), result)
    out: dict[str, Any] = {"seed": int(seed), "settings": settings,
                           "seconds": round(time.time() - began, 1),
                           "words": len(h.brain.lexicon.forms)}
    out["arms"] = {name: {"shares": shares(book), "gates": book} for name, book in books.items()}
    if after is not None:
        out["after"] = after(h)
    return out


class Probe:
    """A probe: ``room(h)`` at the first test question of each room, ``ask(h, gate, q, said)``
    gives a scored answer or None (not asked of this question)."""

    name = "probe"

    def room(self, h: Harness) -> None:
        pass

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        raise NotImplementedError


def write(out: Path, name: str, seed: int, result: dict[str, Any]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{seed}.json"
    part = path.with_suffix(".part")
    part.write_text(json.dumps(result, indent=1, default=float))
    part.rename(path)
    return path


def arguments(argv: list[str]) -> tuple[int, Path, dict[str, Any]]:
    """``<founder> <out> [short]``: the founder, the folder its file goes to and the life's
    settings (the defaults, or the short life with ``short``)."""
    seed = int(argv[1]) if len(argv) > 1 else 900
    out = Path(argv[2]) if len(argv) > 2 else Path("audit_out")
    settings = dict(SHORT) if len(argv) > 3 and argv[3] == "short" else {}
    return seed, out, settings
