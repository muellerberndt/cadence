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

Steps 1 to 3 are a born rule of this instrument (which read holds to the ear, and that a
recalled form moves the focus); nothing in them is taught. The readings, in rooms of three
things of three kinds, three colours and three directions:

- the reply to "what colour has the X" and to "where is the X": the asked thing's, the focus
  thing's, or another, with the rule off and on, the trainers' focus on the asked thing, on
  another thing or on nothing, and the asked thing in view or behind a screen;
- every focus form of the meaning chamber with the rule off and on, the focus on the thing
  asked about: what the rule costs where no attention has to move.
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
            for i, ex in enumerate(exs):
                word = ex.heard[t] if t < len(ex.heard) else None
                if word is not None:  # the ear's read of the records, by this word alone
                    cue = self_.mixed_code(np.zeros(me.w), self_.forms(word))
                    heard = np.concatenate([np.zeros(me.w), self_.forms(word)])
                    found = me.find(self_.mapped[i], me.recall("code", cue, me.heads, heard))
                    if found is not None:
                        me.drawn[i] = found
                if me.drawn[i] is not None:  # the focus the name drew: the map's read of it
                    g = self_.mapped[i].glimpse(me.drawn[i], ex.room.heading, True)
                    x[i, : me.w] = g
                    x[i, me.code] = self_.mixed_code(g, self_.heard_trace[i])
            return x

        nt.Parrot.observe = observe

    def find(self, eyes: mapped.MapEyes, back: np.ndarray) -> str | None:
        """The kind of the thing on the map most like the recalled one, where the recall holds
        a form and a thing matches it at ``MATCH`` or above."""
        if not back[self.lines["word"]].any() or not all(
                back[self.lines[n]].any() for n in self.form):
            return None
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


def main(argv: list[str]) -> None:
    folder, out = Path(argv[1]), Path(argv[2])
    n = int(argv[3]) if len(argv) > 3 else 96
    things = int(argv[4]) if len(argv) > 4 else 320
    heads = int(argv[5]) if len(argv) > 5 else 5
    ch = Named(folder, things, heads, np.random.default_rng(7))
    res: dict[str, Any] = {"brain": folder.name, "questions": n, "things": things,
                           "heads": heads, "match": MATCH,
                           "records": int(ch.stores["code"].count), "asked": {}, "forms": {}}
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
