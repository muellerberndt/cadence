"""A named thing draws attention (cadence#342): "what colour has the X" with the trainers'
focus on another thing or on nothing.

    python named.py <brain folder> <out.json> [questions a row] [things] [heads]

The parts are the two other instruments' (``completion.py``, ``mapped.py``):

1. **The heard name recalls the thing.** At every heard word the records of whole bound things
   are read by the ear alone: the cue is the brain's sparse mixed code of that word with
   nothing attended, the key under which a named thing's record lies. ``heads`` heads read, and
   a population of the bound thing comes back where they agree. The read counts where the word
   that comes back is the word heard.
2. **The map gives where it is.** Where the heads agree on a form (every shape line and the
   size), the recalled thing is looked for among the things the map holds: each is read with
   ``Map.thing_at`` and compared with the recalled populations (cosine, ``MATCH`` or above).
3. **Attention moves there.** From that moment on (the moment the name is heard) the glimpse is
   the map's read of that thing, in view or behind a screen. The brain's settle binds it and
   its voice answers.
4. **A thing the map does not hold comes to mind.** Where no thing on the map matches, the
   recalled thing itself returns to the senses where the eyes attend nothing: to the glimpse
   and to an empty slot of the scene (``completion.py``'s return of the thing).

Steps 1 to 3 are a born rule of this instrument (which read holds to the ear, and that a
recalled form moves the focus); nothing in them is taught. The readings, in rooms of three
things of three kinds, three colours and three directions:

- the reply to "what colour has the X" and to "where is the X": the asked thing's, the focus
  thing's, or another, with the rule off and on, the trainers' focus on the asked thing, on
  another thing or on nothing, and the asked thing in view or behind a screen;
- every focus form of the meaning chamber with the rule off and on, the focus on the thing
  asked about: what the rule costs where no attention has to move;
- "what shape has the X" of a kind that is nowhere in the room, the focus on nothing: the
  voice's answer from the thing the name brought to mind ("round", or "no" for a kind that is
  not round), with the rule off, on, and on with the recall of another kind's name in its
  place (the control: the answer then follows the other kind).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import completion as cp
import lane
import mapped
import numpy as np

from cadence.longterm import LongTermMemory

nt, it = lane.night, lane.introspect
MATCH = 0.9
FORMS = ("name", "colour_of", "colour_this", "material_this", "direction", "is_colour",
         "is_round", "can_eat_bal")


class Named(cp.Chamber):
    """The frozen brain with its map, its records of named things, and the born rule."""

    def __init__(self, folder: Path, things: int, heads: int, rng: np.random.Generator):
        super().__init__(folder, "map")
        p = self.p
        self.heads, self.draw = heads, False
        self.drawn: list[str | None] = []
        self.minded: list[np.ndarray | None] = []  # the thing a name brought to mind
        self.swap: dict[str, str] = {}  # the control: a name recalls another kind's records
        named = [(r, w) for r, t in it.scenes(p, rng, things, self.held, test=False)
                 for w in cp.names(r, t)]
        self.stores = {"code": LongTermMemory(
            width=p.mixed, value_width=self.w + self.ear, capacity=2 * len(named), cosine=True,
            revise=0.9999, seed=lane.SEED)}
        for chunk, m in cp.batches(named):
            self.live([r for r, _ in chunk], [[None, None, w] for _, w in chunk], rng,
                      write=[True] * m + [False] * (32 - m))
        self.form = [n for n in self.lines if n.startswith("shape") or n == "size"]
        self.like = [n for n in self.lines
                     if n not in ("word", "direction", "distance", "salience")]
        inner = nt.Parrot.observe
        me = self

        def observe(self_: Any, exs: list, t: int, own: list, praise: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
            x = inner(self_, exs, t, own, praise, rng)
            if not me.draw:
                return x
            if t == 0 or len(me.drawn) != len(exs):
                me.drawn = [None] * len(exs)
                me.minded = [None] * len(exs)
            e = me.eyes
            for i, ex in enumerate(exs):
                word = ex.heard[t] if t < len(ex.heard) else None
                if word is not None:  # the ear's read of the records, by this word alone
                    said = self_.forms(me.swap.get(word, word))
                    back = me.recall("code", self_.mixed_code(np.zeros(me.w), said), me.heads,
                                     np.concatenate([np.zeros(me.w), said]))
                    if back[me.lines["word"]].any() and all(
                            back[me.lines[n]].any() for n in me.form):  # a thing of that name
                        found = me.find(self_.mapped[i], back)
                        if found is not None:
                            me.drawn[i], me.minded[i] = found, None
                        else:
                            me.minded[i] = back[: me.w]
                if me.drawn[i] is not None:  # the focus the name drew: the map's read of it
                    g = self_.mapped[i].glimpse(me.drawn[i], ex.room.heading, True)
                elif me.minded[i] is not None and not x[i, : me.w].any():
                    g = me.minded[i]  # brought to mind: in the glimpse and a slot of the scene
                    scene = x[i, me.w: me.w + e.width].reshape(e.slots, me.w)
                    free = np.flatnonzero(~scene.any(axis=1))
                    if len(free):
                        scene[free[0]] = g
                else:
                    continue
                x[i, : me.w] = g
                x[i, me.code] = self_.mixed_code(g, self_.heard_trace[i])
            return x

        nt.Parrot.observe = observe

    def find(self, eyes: mapped.MapEyes, back: np.ndarray) -> str | None:
        """The kind of the thing on the map most like the recalled one, where a thing matches
        it at ``MATCH`` or above."""
        held = eyes.mine()
        on = np.zeros(self.w, dtype=bool)
        for n in self.like:
            if back[self.lines[n]].any():
                on[self.lines[n]] = True
        best, level = None, MATCH
        for thing in held.things():
            if thing["place"] is None:
                continue
            got = held.thing_at(None, cells=np.isin(np.arange(held.count), thing["cells"]))
            if got is None:
                continue
            v = got["values"]
            seen = np.concatenate([v["hue"], v["lightness"], v["shape"], v["size"],
                                   np.zeros(self.eyes.directions + self.eyes.distances + 1),
                                   v["surface"]])
            cos = float(cp.unit(seen[on]) @ cp.unit(back[: self.w][on]))
            if cos >= level:
                best, level = thing["kind"], cos
        return None if best is None else list(mapped.KIND)[int(best)]


def rooms_of_three(rng: np.random.Generator, held: set, n: int) -> list:
    """Rooms of three things of three kinds, three colours and three directions."""
    out = []
    while len(out) < n:
        r = nt.room(rng, 3, held)
        if len({t.colour for t in r.things}) == 3 and len(
                {nt.direction_of(r.heading, t.place) for t in r.things}) == 3:
            out.append(r)
    return out


def asked(ch: Named, rng: np.random.Generator, n: int, question: str, focus: str,
          screen: bool) -> dict[str, float]:
    """One row: where the reply falls. ``focus``: "asked", "another" or "nothing"."""
    p = ch.p
    tally = {"the asked thing's": 0, "the focus thing's": 0, "other": 0}
    drawn = {"to the asked thing": 0, "to another thing": 0}
    rooms = rooms_of_three(rng, ch.held, n)
    for chunk, m in cp.batches(rooms):
        exs, truth = [], []
        for r in chunk:
            r = nt.Room([nt.Thing(**{**t.__dict__, "focus": False}) for t in r.things],
                        r.heading)
            order = rng.permutation(3)
            x, y = r.things[int(order[0])], r.things[int(order[1])]
            if focus == "asked":
                x.focus = True
            elif focus == "another":
                y.focus = True
            if question == "colour":
                words, of = ["what", "colour", "has", "the", x.kind], lambda t: t.colour
            else:
                words, of = ["where", "is", "the", x.kind], lambda t, r=r: nt.direction_of(
                    r.heading, t.place)
            q = nt.Question(words, [of(x)], "colour_of" if question == "colour" else "direction")
            ex = nt.make_exchange(r, q, "turn", rng)
            if screen:
                ex.extra["hidden"] = (x.kind,)
            exs.append(ex)
            truth.append((of(x), of(y) if focus == "another" else None))
        width = max(e.turn for e in exs)
        for e in exs:
            pad = width - e.turn
            e.heard = [None] * pad + e.heard
            e.turn += pad
        got = nt.live(p, exs, rng, learn=False)
        if ch.draw:  # where the focus stood at the end of each exchange
            for ex, to in list(zip(exs, ch.drawn, strict=True))[:m]:
                if to is not None:
                    drawn["to the asked thing" if to == ex.question.words[-1]
                          else "to another thing"] += 1
        for reply, (own, other) in list(zip(got["replies"], truth, strict=True))[:m]:
            said = reply[0] if reply else None
            tally["the asked thing's" if said == own else
                  "the focus thing's" if other is not None and said == other else "other"] += 1
    out = {k: round(v / n, 3) for k, v in tally.items()}
    if ch.draw:
        out["the focus drawn"] = {k: round(v / n, 3) for k, v in drawn.items()}
    return out


def absent(ch: Named, rng: np.random.Generator, n: int) -> dict[str, float]:
    """"What shape has the X" of a kind that is nowhere in the room, the focus on nothing:
    the share of right replies, for round kinds ("round") and for the others ("no")."""
    p = ch.p
    right: dict[str, list[bool]] = {"round kinds": [], "other kinds": []}
    cues = [nt.KIND_NAMES[i % len(nt.KIND_NAMES)] for i in range(n)]
    for chunk, m in cp.batches(cues):
        exs = []
        for kind in chunk:
            while True:
                r = nt.room(rng, 3, ch.held)
                if kind not in [t.kind for t in r.things]:
                    break
            q = nt.Question(["what", "shape", "has", "the", kind],
                            ["round" if kind in nt.ROUND else "no"], "shape_of")
            exs.append(nt.make_exchange(r, q, "turn", rng))
        got = nt.live(p, exs, rng, learn=False)
        for kind, ok in list(zip(chunk, got["right"], strict=True))[:m]:
            right["round kinds" if kind in nt.ROUND else "other kinds"].append(bool(ok))
    return {k: round(float(np.mean(v)), 3) for k, v in right.items()}


def main(argv: list[str]) -> None:
    folder, out = Path(argv[1]), Path(argv[2])
    n = int(argv[3]) if len(argv) > 3 else 96
    things = int(argv[4]) if len(argv) > 4 else 320
    heads = int(argv[5]) if len(argv) > 5 else 5
    ch = Named(folder, things, heads, np.random.default_rng(7))
    res: dict[str, Any] = {"brain": folder.name, "questions": n, "things": things,
                           "heads": heads, "match": MATCH,
                           "records": int(ch.stores["code"].count), "asked": {}, "forms": {},
                           "a kind that is nowhere": {}}
    # the control of the last reading: each kind's name recalls the next kind of the other
    # shape class (a round kind's name a kind that is not round, and the reverse)
    kinds = list(nt.KIND_NAMES)
    other = {k: next(j for j in kinds[i + 1:] + kinds[: i] if (j in nt.ROUND) != (k in nt.ROUND))
             for i, k in enumerate(kinds)}
    for label, rule, swap in (("rule off", False, {}), ("rule on", True, {}),
                              ("rule on, another kind's records recalled", True, other)):
        ch.draw, ch.swap = rule, swap
        res["a kind that is nowhere"][label] = absent(ch, np.random.default_rng(31), n)
        print("a kind that is nowhere,", label, json.dumps(res["a kind that is nowhere"][label]),
              flush=True)
        out.write_text(json.dumps(res, indent=1))
    ch.swap = {}
    for question in ("colour", "where"):
        for focus in ("asked", "another", "nothing"):
            for screen in (False, True):
                if screen and focus == "asked":
                    continue
                for rule in (False, True):
                    ch.draw = rule
                    label = (f"{question}, focus on {focus}, the asked thing "
                             f"{'behind a screen' if screen else 'in view'}, rule "
                             f"{'on' if rule else 'off'}")
                    res["asked"][label] = asked(ch, np.random.default_rng(21), n, question,
                                                focus, screen)
                    print(label, json.dumps(res["asked"][label]), flush=True)
                    out.write_text(json.dumps(res, indent=1))
    held = ch.held
    for rule in (False, True):  # what the rule costs where no attention has to move
        ch.draw = rule
        row: dict[str, Any] = {}
        for f in FORMS:
            rng = np.random.default_rng(11)
            ok: list[bool] = []
            drawn: list[bool] = []
            while len(ok) < n:
                ok += nt.live(ch.p, nt.blocks_of(ch.p, rng, f, "turn", held), rng,
                              learn=False)["right"]
                drawn += [to is not None for to in ch.drawn] if rule else []
            row[f] = round(float(np.mean(ok[:n])), 3)
            if rule:
                row.setdefault("the focus drawn", {})[f] = round(float(np.mean(drawn[:n])), 3)
        res["forms"][f"rule {'on' if rule else 'off'}"] = row
        print("forms, rule", "on" if rule else "off", json.dumps(row), flush=True)
        out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv)
