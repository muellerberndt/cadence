"""The answer a question has over a room as a brain could hold it (cadence#326 audit).

``answer(lex, things, q)`` is the world's answer to ``q``'s form computed over ``things``, a list
of ``(kind, colour, place around the parrot)``: the room as the world holds it
(``triples(room)``), or changed the way a probe changed what the brain holds (a colour moved on,
the body turned, things never seen left out). It agrees with ``world.py``'s answers on the
unchanged room."""

from __future__ import annotations

from typing import Any

import common  # noqa: F401  (puts the chamber on the path)
import world


def triples(r: world.Room, *, only: Any = None) -> list[tuple[int, int, int]]:
    """The room's things as ``(kind, colour, place around the parrot)``; ``only(thing)`` keeps
    some."""
    return [(t.kind, t.colour, world.place_of(r.heading, t.place)) for t in r.things
            if only is None or only(t)]


def answer(lex: world.Lexicon, things: list[tuple[int, int, int]],
           q: world.Question) -> list[str]:
    said = q.text.split()
    if q.form in ("where", "colour"):
        hit = [t for t in things if q.kind is not None and q.kind >= 0 and t[0] == q.kind]
        if not hit:
            return [lex.none]
        _, c, p = hit[0]
        return [lex.place[p]] if q.form == "where" else [lex.colour[c]]
    if q.form == "by_colour":
        c = next(lex.colour.index(w) for w in said if w in lex.colour)
        return sorted(lex.kind[k] for k, cc, _ in things if cc == c) or [lex.none]
    p = next(lex.place.index(w) for w in said if w in lex.place)
    return sorted(lex.kind[k] for k, _, pp in things if pp == p) or [lex.none]
