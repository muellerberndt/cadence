"""Ablations and lies (cadence#326 audit): where does each answer come from?

Every test question of the founder's life is asked again of the brain with one part taken away or
told a lie, and each answer is scored against the world (``ok``) and against what the change
predicts (``follows``). An answer that comes from the brain's state follows the change; one that
comes from anywhere else does not.

Each probe, its change and the answer it predicts:

- ``lexicon_zero``: every synapse of the lexicon at zero (W6's twin); nothing said.
- ``map_cleared``: the map emptied before the question; "I do not know" to every question.
- ``word_cut``: the synapses of the question's content word (the kind, colour or place word)
  at zero; "I do not know".
- ``word_cut_other``: the synapses of another trained kind's word at zero (W1 questions); the
  brain's own answer.
- ``read_cut``: the synapses of the read concepts at zero; "I do not know".
- ``none_cut``: the synapses of the concept "finds nothing" at zero; no "I do not know" said.
- ``qword_cut_<w>``: the synapses of one question word at zero (where, what, colour, is, the,
  you); a reading of which words carry the read.
- ``body_turned``: the body's heading a quarter turn to the left at the question; each place
  around the body one on.
- ``colour_lie``: the room looked at with every colour reported one on; each colour one on.
- ``half_seen``: the room looked at in one direction only; "I do not know" for things never in
  view.
- ``no_reset``: the map never emptied between test rooms; the world's answer (things no longer
  there cleared by sight).

    python ablations.py <founder> <out> [short]
"""

from __future__ import annotations

import math
import sys
from typing import Any

import common
import numpy as np
import truth
import world
from common import Harness, Probe, scored


class LexiconZero(Probe):
    name = "lexicon_zero"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        twin = h.twin("zero", lambda lx: np.ones(int(lx.connectome.synapses), dtype=bool))
        return scored(h, q, twin.ask(q.text))


class MapCleared(Probe):
    name = "map_cleared"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        held = common.snap_map(h.brain.map)
        h.brain.map.reset()
        try:
            out = h.brain.ask(q.text)
        finally:
            common.restore_map(h.brain.map, held)
        return scored(h, q, out, predicted=[h.lex.none])


class WordCut(Probe):
    name = "word_cut"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        cut = common.content(h.lex, q.text)
        if not cut:
            return None
        twin = h.twin(("word", tuple(cut)), lambda lx: np.any(
            [common.word_synapses(lx, w) for w in cut], axis=0))
        return scored(h, q, twin.ask(q.text), predicted=[h.lex.none])


class WordCutOther(Probe):
    name = "word_cut_other"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        if not gate.startswith("W1") or q.kind is None or q.kind < 0:
            return None
        other = h.lex.kind[(q.kind + 1) % world.TRAINED]
        twin = h.twin(("word", (other,)), lambda lx: common.word_synapses(lx, other))
        return scored(h, q, twin.ask(q.text), predicted=said, base=said)


class GroupCut(Probe):
    def __init__(self, group: str, predicted: str | None):
        self.group, self.predicted = group, predicted
        self.name = f"{group}_cut"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        twin = h.twin(("group", self.group), lambda lx: common.group_synapses(lx, self.group))
        out = twin.ask(q.text)
        predicted = [h.lex.none] if self.predicted == "none" else None
        return scored(h, q, out, predicted=predicted)


class QuestionWordCut(Probe):
    def __init__(self, role: str):
        self.role = role
        self.name = f"qword_cut_{role}"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        form = h.lex.question[self.role]
        if form not in q.text.split():
            return None
        twin = h.twin(("word", (form,)), lambda lx: common.word_synapses(lx, form))
        return scored(h, q, twin.ask(q.text), base=said)


class BodyTurned(Probe):
    """The body's heading a quarter turn to the left (counter-clockwise) at the question: what
    was in front is to the right of the body as it believes it faces."""

    name = "body_turned"

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        from cadence.words import Body

        assert h.room is not None
        body = Body(h.brain.body.at, heading=h.brain.body.heading + math.pi / 2)
        twin = h.with_lexicon(h.brain.map, body)
        things = [(k, c, (p + 1) % 4) for k, c, p in truth.triples(h.room)]
        return scored(h, q, twin.ask(q.text), predicted=truth.answer(h.lex, things, q))


class ColourLie(Probe):
    """The room looked at by a camera that reports every colour one on (red as blue, ...)."""

    name = "colour_lie"

    def room(self, h: Harness) -> None:
        from cadence.words import Body

        assert h.room is not None
        self.map, self.body = common.fresh_map(h.seed), Body()

        def lied(r: world.Room, heading: float) -> list[Any]:
            return [(k, p, {"colour": (v["colour"] + 1) % world.COLOURS})
                    for k, p, v in world.seen(r, heading)]

        common.look_around(self.map, self.body, h.room, reports=lied)

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        assert h.room is not None
        twin = h.with_lexicon(self.map, self.body)
        things = [(k, (c + 1) % world.COLOURS, p) for k, c, p in truth.triples(h.room)]
        return scored(h, q, twin.ask(q.text), predicted=truth.answer(h.lex, things, q))


class HalfSeen(Probe):
    """The room looked at in the direction the robot faces only (the camera's 120 degrees):
    a thing never in view is not held."""

    name = "half_seen"

    def room(self, h: Harness) -> None:
        from cadence.words import Body

        assert h.room is not None
        self.map, self.body = common.fresh_map(h.seed), Body()
        common.look_around(self.map, self.body, h.room, turns=[0])

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        assert h.room is not None
        r = h.room
        twin = h.with_lexicon(self.map, self.body)
        held = truth.triples(r, only=lambda t: bool(world.in_view(r.heading, t.place)[0]))
        out = scored(h, q, twin.ask(q.text), predicted=truth.answer(h.lex, held, q))
        # the questions whose thing was never in view: the world's answer is a place or a
        # colour, the brain's held answer "I do not know"
        if q.form in ("where", "colour") and q.about in ("in_view", "out_of_view"):
            t = next(t for t in r.things if t.kind == q.kind)
            if not bool(world.in_view(r.heading, t.place)[0]):
                out["gate"] = f"{gate}_never_in_view"
        return out


class NoReset(Probe):
    """A map of its own that is never emptied between test rooms: the program's "new room"
    reset is the one change."""

    name = "no_reset"

    def __init__(self) -> None:
        self.map: Any = None

    def room(self, h: Harness) -> None:
        from cadence.words import Body

        assert h.room is not None
        if self.map is None:
            self.map, self.body = common.fresh_map(h.seed), Body()
        common.look_around(self.map, self.body, h.room, reset=False)

    def ask(self, h: Harness, gate: str, q: world.Question, said: list[str]) -> Any:
        twin = h.with_lexicon(self.map, self.body)
        return scored(h, q, twin.ask(q.text), base=said)


def probes() -> list[Probe]:
    return [LexiconZero(), MapCleared(), WordCut(), WordCutOther(), GroupCut("read", "none"),
            GroupCut("none", None), BodyTurned(), ColourLie(), HalfSeen(), NoReset()] + [
        QuestionWordCut(role) for role in world.QUESTION_WORDS]


def main(argv: list[str]) -> None:
    seed, out, settings = common.arguments(argv)
    result = common.replay(seed, probes(), settings)
    print(common.write(out, "ablations", seed, result))


if __name__ == "__main__":
    main(sys.argv)
