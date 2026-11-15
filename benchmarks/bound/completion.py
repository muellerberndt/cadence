"""Completion from one facet (cadence#342): records of whole bound things in one
``cadence.longterm.LongTermMemory``, keyed by the brain's own settled state at the moment a
thing is attended and named, read by one facet alone, and what is read given back to the
brain's senses, so that its settled state completes.

    python completion.py <brain folder> <out.json> [things] [heads] [rooms a word]

**A record.** A trainer holds up a thing and says a word that names it (its kind, its colour,
its material, where it is, "round"): two moments of looking, then the word. At the moment the
word is heard the record is written. Its value is the bound thing: the glimpse's sense fields
as the eyes report them and the heard word's code. Its key is the brain's settled state of that
moment, in one of three reads (``KEYS``):

- ``association``: the association region's settled activity;
- ``code``: the brain's sparse mixed code, the population where the attended thing's senses and
  what is heard meet, each at unit length;
- ``both``: the two side by side, each at unit length.

**A read.** A cue is the same read of the state at a later moment. ``heads`` heads read the
store (``LongTermMemory.read``), one record a head. The bound thing is compared population by
population (``LINES``: each sense's tuned cells, the word): a population comes back where every
head's record holds it alike (cosine ``AGREE`` to the first head's), as the mean of the heads;
one they hold otherwise stays silent. With one head the nearest record comes back whole. A read
whose records hold a population otherwise than the cue itself holds it (another word than the
one heard, another colour than the one glimpsed) returns nothing: the records are of something
else.

**Back into the brain.** At the next moments what was read enters the senses as a thing in
focus and a heard word do: the word to the ear (the chamber's own recall), and the thing's
populations to the glimpse, wherever the eyes report nothing there, and to an empty slot of the
scene. The brain settles on it. Three further arms return less: the thing to the glimpse and
no slot of the scene, the word alone, or the thing alone.

**The readings.** Linear probes of each facet (kind, colour, round, material, edible, direction,
distance) are fitted on the association's settled state over looks 1 to 3 at whole things in
focus with nothing heard (``rooms of 3 to 5``, the meaning chamber's scenes; ``rooms of 1 to
5``, the same scenes cut to the thing and none to four others). The completion matrix cues one
facet alone and reads the facets the cue implies from the settled state one and two moments
later: with no records (before), with the records (after), and with the records' values dealt
to other keys (the control). The cues: a word heard with nothing attended, in an empty room and
in a room of things none of which the word names; one sense of a held-out thing in the glimpse,
the thing in no slot of the scene.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import lane
import mapped
import numpy as np

from cadence import Brain
from cadence.longterm import LongTermMemory

nt, it = lane.night, lane.introspect
KEYS = ("association", "code", "both")
AGREE = 0.8  # two tuned populations within about one cell's spacing of each other
SENSES = {"colour": ("hue", "lightness"), "form": ("shape", "size"), "surface": ("surface",),
          "place": ("direction", "distance")}
# a sense's own facets and, for the form, what its kind implies in this world
OWN = {"colour": ("colour",), "form": ("kind", "round", "edible", "material"),
       "surface": ("material", "edible"), "place": ("direction", "distance")}
WORDS = {"kind": nt.KIND_NAMES, "colour": tuple(nt.COLOURS), "material": tuple(nt.MATERIALS),
         "direction": nt.DIRECTIONS, "round": ("round",)}


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def implied(word: str) -> dict[str, str]:
    """The facets a word alone implies, as the probes' class labels."""
    if word in nt.KINDS:
        spec = nt.KINDS[word]
        out = {"kind": word, "round": str(word in nt.ROUND),
               "edible": str(set(spec[2]) <= set(nt.EDIBLE))}
        if len(spec[2]) == 1:
            out["material"] = spec[2][0]
        return out
    if word in nt.COLOURS:
        return {"colour": word}
    if word in nt.MATERIALS:
        return {"material": word, "edible": str(word in nt.EDIBLE)}
    if word in nt.DIRECTIONS:
        return {"direction": word}
    return {"round": "True"} if word == "round" else {}


def names(r: Any, t: Any) -> list[str]:
    """The words that name a thing: its kind, colour, material, where it is, and its shape."""
    return [t.kind, t.colour, t.material, nt.direction_of(r.heading, t.place),
            *(["round"] if t.kind in nt.ROUND else [])]


class Chamber:
    """The frozen brain, its three reads of the state, and the stores."""

    def __init__(self, folder: Path, source: str = "eyes"):
        mapped.install(source)
        self.p = p = nt.Parrot(lane.SEED, dict(lane.GENOME), streams=32)
        p.brain = Brain.load(folder / "brain.npz")
        p.records = None
        self.brain = it.frozen(p)  # nothing is learned: every read is of this copy, cleared
        self.held = nt.founder_held(lane.SEED)
        self.assoc = np.asarray(list(p.brain.connectome.populations["association"]))
        self.eyes = e = p.eyes
        self.w, self.ear = e.slot_width, p.ear
        self.code = slice(self.w + e.width, self.w + e.width + p.mixed)
        edges = np.cumsum([0, *e.parts.values()])
        self.field = {n: slice(int(a), int(b)) for n, a, b in zip(e.parts, edges[:-1],
                                                                  edges[1:], strict=True)}
        # the bound thing's populations: each sense's tuned cells (a line a shape dimension and
        # a surface reading), and the word
        self.lines: dict[str, slice] = {}
        for name, run in self.field.items():
            per = {"shape": e.per_shape, "surface": e.per_surface}.get(name)
            if per is None:
                self.lines[name] = run
            else:
                for at in range(run.start, run.stop, per):
                    self.lines[f"{name} {(at - run.start) // per}"] = slice(at, at + per)
        self.lines["word"] = slice(self.w, self.w + self.ear)
        self.stores: dict[str, LongTermMemory] = {}
        self.part: np.ndarray | None = None  # what the eyes glimpse of a thing, a row a stream
        self.back: np.ndarray | None = None  # what the records returned, a row a stream
        self.returned: np.ndarray | None = None  # the latest read's rows
        self.mode = "whole"  # what comes back: "whole", "glimpse", "word" or "thing" (above)
        self.told: list[list[int]] = []  # each read's records, for the read's own statistics
        plain = mapped.observe
        me = self

        def observe(self_: Any, exs: list, t: int, own: list, praise: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
            x = plain(self_, exs, t, own, praise, rng)
            if me.part is None and me.back is None:
                return x
            # a thing glimpsed in part (one sense of it), in place of the eyes' report
            glimpse = x[:, : me.w].copy() if me.part is None else me.part.copy()
            if me.back is not None and me.mode != "word":
                # the recalled thing, wherever the eyes report nothing
                for name, run in me.lines.items():
                    if name != "word":
                        silent = ~glimpse[:, run].any(axis=1)
                        glimpse[silent, run] = me.back[silent, run]
                if me.mode != "glimpse":  # and as a thing of the scene, in an empty slot
                    scene = x[:, me.w: me.w + e.width].reshape(len(x), e.slots, me.w)
                    for i in np.flatnonzero(me.back[:, : me.w].any(axis=1)):
                        free = np.flatnonzero(~scene[i].any(axis=1))
                        if len(free):
                            scene[i, free[0]] = me.back[i, : me.w]
            if me.back is not None and me.mode != "thing":
                x[:, -me.ear:] += me.back[:, me.w:]
            x[:, : me.w] = glimpse
            for i in range(len(x)):
                x[i, me.code] = self_.mixed_code(glimpse[i], self_.heard_trace[i])
            return x

        nt.Parrot.observe = observe

    def keys(self, x: np.ndarray, state: np.ndarray) -> dict[str, np.ndarray]:
        a, c = state[:, self.assoc], x[:, self.code]
        return {"association": a, "code": c, "both": np.concatenate([unit(a), unit(c)], axis=1)}

    def live(self, rooms: list, heard: list, rng: np.random.Generator, *, key: str | None = None,
             heads: int = 5, after: int = 0, write: list | None = None,
             ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        """Each room's moments of ``heard`` from a cleared state (lockstep, greedy, frozen).
        Returns the keys of the last heard moment and the association's state at each of the
        ``after`` moments that follow it. With ``key`` the store of that key is read at the
        last heard moment and what it returns comes back at the moments after. With ``write``
        (a flag a room) every store takes a record at the last heard moment."""
        p, n, last = self.p, len(rooms), len(heard[0]) - 1
        brain = self.brain
        brain.reset()
        exs = [nt.Exchange(r, list(h) + [None] * after, last, None)
               for r, h in zip(rooms, heard, strict=True)]
        self.back = None
        keys: dict[str, np.ndarray] = {}
        states = []
        for t in range(last + 1 + after):
            x = p.observe(exs, t, [None] * n, np.zeros(n, dtype=bool), rng)
            brain.act(x, greedy=True)
            state = np.asarray(brain.basal_ganglia.state.activation, dtype=np.float64)
            if t == last:
                keys = self.keys(x, state)
                if write is not None:
                    value = np.concatenate([x[:, : self.w], x[:, -self.ear:]], axis=1)
                    for name, store in self.stores.items():
                        for i in range(n):
                            if write[i]:
                                store.write(keys[name][i], value[i])
                if key is not None:
                    self.told = []
                    held = np.concatenate([x[:, : self.w], x[:, -self.ear:]], axis=1)
                    self.back = np.stack([self.recall(key, keys[key][i], heads, held[i])
                                          for i in range(n)])
                    self.returned = self.back
            elif t > last:
                states.append(state[:, self.assoc].copy())
        self.back = None
        return keys, states

    def recall(self, key: str, cue: np.ndarray, heads: int, holds: np.ndarray | None = None
               ) -> np.ndarray:
        """What the store of ``key`` returns for ``cue``: each population of the bound thing
        where the heads' records hold it alike, the mean of the heads; zeros elsewhere.
        ``holds`` is what the senses hold at the cue's moment (the glimpse, then the ear): a
        read that returns a population otherwise than the senses hold it returns nothing."""
        store = self.stores[key]
        out = np.zeros(self.w + self.ear)
        self.told.append([])
        if not store.count or not np.any(cue):
            return out
        got = store.read(cue, heads=min(heads, store.count), credit=False)
        slots = [got.first(h) for h in range(min(heads, store.count))]
        slots = [s for s in slots if s >= 0]
        self.told[-1] = slots
        if not slots:
            return out
        values = [store.value(got, h) for h in range(len(slots))]
        for run in self.lines.values():
            parts = np.stack([v[run] for v in values])
            first = unit(parts[0])
            if parts[0].any() and all(float(unit(q) @ first) >= AGREE for q in parts[1:]):
                out[run] = parts.mean(axis=0)
                if holds is not None and holds[run].any() and float(
                        unit(out[run]) @ unit(holds[run])) < AGREE:
                    return np.zeros(self.w + self.ear)  # records of something else
        return out


def batches(items: list, size: int = 32) -> Any:
    for i in range(0, len(items), size):
        chunk = items[i:i + size]
        yield chunk + [chunk[-1]] * (size - len(chunk)), len(chunk)


def cut(r: Any, others: int) -> Any:
    """A room cut to its thing in focus and ``others`` of the rest."""
    return nt.Room([t for t in r.things if t.focus]
                   + [t for t in r.things if not t.focus][:others], r.heading)


def room_without(rng: np.random.Generator, held: set, word: str) -> Any:
    """A room of three to five things none of which ``word`` names, nothing in focus."""
    while True:
        r = nt.room(rng, int(rng.integers(3, 6)), held)
        if not any(word in names(r, t) for t in r.things):
            return r


def main(argv: list[str]) -> None:
    folder, out = Path(argv[1]), Path(argv[2])
    things = int(argv[3]) if len(argv) > 3 else 320
    heads = int(argv[4]) if len(argv) > 4 else 5
    per_word = int(argv[5]) if len(argv) > 5 else 16
    ch = Chamber(folder)
    p, held = ch.p, ch.held
    rng = np.random.default_rng(7)
    res: dict[str, Any] = {"brain": str(folder.name), "things": things, "heads": heads,
                           "agree": AGREE, "rooms a word": per_word}

    # -- the probes: looks 1 to 3 at whole things in focus, nothing heard
    def looks(pairs: list, others: list[int] | None = None) -> tuple[np.ndarray, list]:
        rows, labels = [], []
        for chunk, m in batches(list(zip(pairs, others or [9] * len(pairs), strict=True))):
            first, states = ch.live([cut(r, k) for (r, _), k in chunk], [[None]] * 32, rng,
                                    after=2)
            for state in (first["association"], *states):
                rows.append(state[:m])
                labels += [pair for pair, _ in chunk[:m]]
        return np.concatenate(rows), labels

    train = it.scenes(p, rng, 512, held, test=False)
    test = it.scenes(p, rng, 256, held, test=True)
    sizes = [int(k) for k in np.random.default_rng(1).integers(0, 5, len(train))]
    fits: dict[str, dict[str, Any]] = {}
    res["state completeness"] = {}
    xb, lb = looks(test)
    xl, ll = looks(test, [0] * len(test))
    for name, (xa, la) in (("rooms of 3 to 5", looks(train)),
                           ("rooms of 1 to 5", looks(train, sizes))):
        fits[name] = {f: it.fit_classes(xa, [it.facet(t, r.heading, f) for r, t in la])
                      for f in it.FACETS}
        for label, x, lab in (("whole scenes", xb, lb), ("the thing alone in its scene", xl, ll)):
            res["state completeness"][f"{name}: {label}"] = {
                f: round(float(np.mean([a == it.facet(t, r.heading, f) for a, (r, t) in zip(
                    it.read_classes(fits[name][f], x), lab, strict=True)])), 3)
                for f in it.FACETS}
    res["state completeness"]["base rate"] = {
        f: round(max(v.count(c) for c in set(v)) / len(v), 3)
        for f in it.FACETS for v in [[it.facet(t, r.heading, f) for r, t in lb]]}
    print(json.dumps(res["state completeness"]), flush=True)

    # -- the records: every thing named by each word that names it
    named = [(r, t, w) for r, t in it.scenes(p, rng, things, held, test=False)
             for w in names(r, t)]
    widths = {"association": len(ch.assoc), "code": p.mixed, "both": len(ch.assoc) + p.mixed}

    def store(name: str) -> LongTermMemory:
        return LongTermMemory(width=widths[name], value_width=ch.w + ch.ear,
                              capacity=2 * len(named), cosine=True, revise=0.9999,
                              seed=lane.SEED)

    ch.stores = {name: store(name) for name in KEYS}
    for chunk, m in batches(named):
        ch.live([r for r, _, _ in chunk], [[None, None, w] for _, _, w in chunk], rng,
                write=[True] * m + [False] * (32 - m))
    res["records"] = int(ch.stores["code"].count)
    assert res["records"] == len(named)  # a record a naming, in the order named
    dealt = {}
    for name in KEYS:  # the control: every value under another record's key
        dealt[name] = store(name)
        order = np.random.default_rng(3).permutation(len(named))
        for slot in range(len(named)):
            dealt[name].write(ch.stores[name].keys[slot], ch.stores[name].values[order[slot]])
    plans: list[tuple[str, str | None, Any, int, str]] = [
        ("before: no records", None, None, heads, "whole")]
    plans += [(f"after: key {name}, {heads} heads that agree", name, ch.stores[name], heads,
               "whole") for name in KEYS]
    plans += [("after: key code, 1 head", "code", ch.stores["code"], 1, "whole"),
              (f"control: key code, {heads} heads, values dealt to other keys", "code",
               dealt["code"], heads, "whole"),
              ("control: key code, 1 head, values dealt to other keys", "code", dealt["code"],
               1, "whole"),
              (f"after: key code, {heads} heads, no slot of the scene", "code",
               ch.stores["code"], heads, "glimpse"),
              (f"after: key code, {heads} heads, the word alone comes back", "code",
               ch.stores["code"], heads, "word"),
              (f"after: key code, {heads} heads, the thing alone comes back", "code",
               ch.stores["code"], heads, "thing")]
    kept = dict(ch.stores)

    def arms(rooms: list, heard: list, truths: list[dict[str, str]], words: list | None = None,
             parts: list | None = None) -> dict[str, Any]:
        """Every arm on one set of cues: each implied facet read from the state one and two
        moments after the cue by each probe; for a read, the share of its heads' records
        named by the cue's word and how often each population came back."""
        table: dict[str, Any] = {}
        for label, name, held_store, h, mode in plans:
            if name is not None:
                ch.stores[name] = held_store
            ch.mode = mode
            gathered: list[list[np.ndarray]] = [[], []]
            told: list[list[int]] = []
            came: dict[str, int] = dict.fromkeys(ch.lines, 0)
            cues = list(zip(rooms, heard, parts or [None] * len(rooms), strict=True))
            for chunk, m in batches(cues):
                ch.part = None if parts is None else np.stack([g for _, _, g in chunk])
                _, states = ch.live([r for r, _, _ in chunk], [x for _, x, _ in chunk], rng,
                                    key=name, heads=h, after=2)
                ch.part = None
                for k in range(2):
                    gathered[k].append(states[k][:m])
                told += ch.told[:m]
                for back in (ch.returned[:m] if name is not None else []):
                    for line, run in ch.lines.items():
                        came[line] += int(back[run].any())
            ch.stores.update(kept)
            ch.mode = "whole"
            cells: dict[str, Any] = {}
            for probe, fit in fits.items():
                for f in it.FACETS:
                    rows = [i for i, tr in enumerate(truths) if f in tr]
                    if rows:
                        cells[f"{probe}: {f}"] = [round(float(np.mean([
                            a == truths[i][f] for a, i in zip(it.read_classes(
                                fit[f], np.concatenate(g)[rows]), rows, strict=True)])), 3)
                            for g in gathered]
            cells["cues"] = len(truths)
            if name is not None and held_store is kept[name] and mode == "whole":
                if words is not None:
                    cells["heads' records named by the cue's word"] = round(float(np.mean([
                        named[s][2] == words[i] for i, slots in enumerate(told)
                        for s in slots] or [0.0])), 3)
                cells["came back"] = {line: round(n / len(truths), 2)
                                      for line, n in came.items() if n}
            table[label] = cells
        return table

    # -- a word alone, nothing attended: in an empty room, and in a room it names nothing of
    empty = nt.Room([], heading=0.0)
    res["a word alone"] = {}
    for group, words in WORDS.items():
        cues = [w for w in words for _ in range(per_word)]
        for where, rooms in (("an empty room", [empty] * len(cues)),
                             ("a room of other things", [room_without(rng, held, w)
                                                         for w in cues])):
            label = f"a {group}'s word, {where}"
            res["a word alone"][label] = arms(rooms, [[None, None, w] for w in cues],
                                              [implied(w) for w in cues], words=cues)
            print(label, json.dumps(res["a word alone"][label]), flush=True)
            out.write_text(json.dumps(res, indent=1))
    # -- one sense of a held-out thing alone in the glimpse, the thing in no slot of the scene
    fresh = it.scenes(p, rng, 128, held, test=True)
    res["one sense alone"] = {}
    for sense, own in OWN.items():
        keep = np.zeros(ch.w, dtype=bool)
        for name in SENSES[sense]:
            keep[ch.field[name]] = True
        rooms, parts = [], []
        for r, t in fresh:
            rest = [x for x in r.things if x is not t]
            rooms.append(nt.Room(rest, r.heading))
            parts.append(np.where(keep, p.eyes.slot(r.seen_of(t)), 0.0))
        res["one sense alone"][sense] = arms(
            rooms, [[None, None]] * len(rooms),
            [{f: it.facet(t, r.heading, f) for f in own
              if f != "material" or sense != "form" or len(nt.KINDS[t.kind][2]) == 1}
             for r, t in fresh], parts=parts)
        print(sense, json.dumps(res["one sense alone"][sense]), flush=True)
        out.write_text(json.dumps(res, indent=1))
    # -- a held-out thing seen whole, nothing heard: the word the records return
    res["a thing seen, nothing heard"] = {}
    for name in KEYS:
        for h in (heads, 1):
            said: list[str | None] = []
            for chunk, m in batches(fresh):
                ch.live([r for r, _ in chunk], [[None, None]] * 32, rng, key=name, heads=h,
                        after=1)
                said += [p.forms.heard_as(back[ch.lines["word"]]) for back in ch.returned[:m]]
            res["a thing seen, nothing heard"][f"key {name}, {h} head{'s' if h > 1 else ''}"] = {
                "a word comes back": round(float(np.mean([w is not None for w in said])), 3),
                "it names the thing": round(float(np.mean([
                    w in names(r, t) for w, (r, t) in zip(said, fresh, strict=True)])), 3),
                "it is the thing's kind": round(float(np.mean([
                    w == t.kind for w, (_, t) in zip(said, fresh, strict=True)])), 3)}
    print(json.dumps(res["a thing seen, nothing heard"]), flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
