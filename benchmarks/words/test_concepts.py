"""The learned-concepts chamber's contracts: the teacher, the raw readings and what is held out,
the script, the given concepts, the probes' boundary read and the controls' sequences."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import concepts  # noqa: E402
import world  # noqa: E402

TAU = 2.0 * math.pi


def test_a_teacher_moves_its_boundaries_and_holds_hues_and_shapes_out() -> None:
    tea = concepts.teacher(3)
    other = concepts.teacher(4)
    assert not np.allclose(tea.bounds, other.bounds)  # each founder's teacher its own
    rng = np.random.default_rng(0)
    for c in range(world.COLOURS):
        start, width = tea.band(c)
        assert math.radians(30.0) <= width <= math.radians(90.0)
        assert tea.colour(start + 0.01) == c and tea.colour(start + width - 0.01) == c
        low, high = tea.gap(c)
        for _ in range(50):
            h = concepts.hue_for(tea, c, rng)
            assert tea.colour(h) == c and not concepts._within(h, low, high)
            g = concepts.hue_for(tea, c, rng, held=True)
            assert tea.colour(g) == c and concepts._within(g, low, high)
    for k in range(world.KINDS):
        feature, sign = tea.side[k]
        for _ in range(20):
            s = concepts.shape_for(tea, k, rng)
            assert sign * (s[feature] - tea.shapes[k][feature]) <= 1e-12
            held = concepts.shape_for(tea, k, rng, held=True)
            assert sign * (held[feature] - tea.shapes[k][feature]) >= concepts.SPREAD - 1e-12
    assert min(np.linalg.norm(a - b) for i, a in enumerate(tea.shapes)
               for b in tea.shapes[i + 1 :]) >= concepts.APART
    # the places around the body: four turns, each within its move of the quarters
    assert [tea.place(math.radians(a)) for a in (0.0, -90.0, 180.0, 90.0)] == [0, 1, 2, 3]
    assert np.all(np.abs(tea.turns - np.radians([-45.0, -135.0, 135.0, 45.0]))
                  <= concepts.TURN)


def test_the_script_gives_raw_readings_and_the_teachers_answers() -> None:
    events = concepts.script(5, naming_rooms=3, practice_rooms=2, test_rooms=2,
                             unseen_rooms=2, held_rooms=2, new_rooms=2)
    tea = concepts.teacher(5)
    kinds = {e[0] for e in events}
    assert kinds == {"room", "turn", "look", "name", "practice", "ask"}
    for e in events:
        if e[0] == "room":
            for t in e[1].things:
                assert t.shape.shape == (3,) and 0.0 <= t.hue < TAU
                assert tea.colour(t.hue) == t.colour
    lex = world.lexicon(5)
    gates = {e[1] for e in events if e[0] == "ask"}
    assert {"C2_colour", "C2_where", "W1_practised", "W2_where", "W3_colour"} <= gates
    # a "where" answer is the teacher's place of the thing as the parrot's body faces
    r = None
    for e in events:
        if e[0] == "room":
            r = e[1]
        if e[0] == "ask" and e[2].form == "where" and e[2].about in ("in_view", "out_of_view"):
            assert r is not None
            t = next(t for t in r.things if t.kind == e[2].kind)
            assert e[2].answer == [lex.place[tea.place(concepts.direction(r.heading, t.place))]]


def test_the_given_concepts_and_the_boundary_read() -> None:
    tea = concepts.teacher(6)
    given = concepts.Given(tea)
    kind, colour = given.percept(tea.shapes[3] + 0.01, float(tea.bounds[2]) + 0.3)
    assert kind == 3 and 0 <= colour < world.COLOURS
    reports = given.perceive([(tea.shapes[1], np.array([1.0, 0.0]), 0.2)])
    assert reports[0][0] == 1 and set(reports[0][2]) == {"colour"}
    sweep = np.linspace(0.0, 1.0, 11)
    said = ["a"] * 4 + ["b"] * 7
    assert concepts._switch(sweep, said, "a", "b") == (sweep[3] + sweep[4]) / 2


def test_the_controls_read_raw_readings_beside_the_words() -> None:
    import concepts_controls as controls

    items = controls.sequences(7, naming_rooms=2, practice_rooms=1, test_rooms=1,
                               unseen_rooms=1, held_rooms=1, new_rooms=1)
    tok = controls.Tokens(world.lexicon(7))
    assert not any(n.startswith(("kind", "colour", "place")) for n in tok.names)
    kind, pairs, _ = next(i for i in items if i[0] == "train")
    (tokens, values), target = pairs[0]
    assert len(tokens) == len(values) and all(v.shape == (controls.READINGS,) for v in values)
    assert target[-1] == tok("END")
    net = controls.build("transformer", 16, len(tok), 0)
    import torch

    x = torch.tensor([tokens])
    v = torch.from_numpy(np.asarray(values, dtype=np.float32)[None])
    assert tuple(net(x, v).shape) == (1, len(tokens), len(tok))
