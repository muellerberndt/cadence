"""The map of things' compiled kernels (``cadence.compiled``: the settle, the pieces, the moves,
the values a report writes and a report's claims and writes) against the map's own Python, over
lives whose things move on steady lines in and out of a sweeping camera's view: reports find
their things near and a stride off, in doubt and carried by the learned physics. Natively NumPy's
BLAS sums a few products in another order than the compiled loops, so the lives agree within
rounding; in a browser they agree to the bit (``kernels/wasm/browser/maps.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import compiled
from cadence.space.map import Map

pytest.importorskip("numba")
pytestmark = pytest.mark.skipif(not compiled.available(), reason="the map's kernels are not in use")

KERNELS = ("_map_settle", "_map_pieces", "_map_moves", "_map_write_values", "_map_report")


def wild_map() -> Map:
    return Map((0.0, 24.0, 0.0, 24.0), (8, 8), units=4, values={"hue": 4}, reach=3.0, step=1.8,
               stride=6.0, margin=0.3, physics={"noise": 0.3}, surprise=True)


def plain_map() -> Map:
    return Map((0.0, 10.0, 0.0, 10.0), (6, 5), units=3, empty="near")


def life(make, moments: int = 90) -> list[np.ndarray]:
    rng = np.random.default_rng(5)
    held = make()
    side = held.plane[1] - held.plane[0]
    kinds = len(held.codes)
    walkers = rng.random((kinds, 2)) * side
    pace = rng.normal(size=(kinds, 2)) * 0.03 * side
    out = []
    for t in range(moments):
        edge = side * (0.6 + 0.45 * np.sin(t / 2.0))
        walkers += pace
        pace[(walkers < 0.0) | (walkers > side)] *= -1.0
        walkers = np.clip(walkers, 0.0, side)
        seen = []
        for kind in range(kinds):
            if walkers[kind, 0] < edge:
                if held.values:
                    seen.append((kind, walkers[kind].copy(), {"hue": int(rng.integers(4))}))
                else:
                    seen.append((kind, walkers[kind].copy()))
        held.moment(seen=seen, lit=lambda points, e=edge: points[:, 0] < e)
        out.extend([held.potential.copy(), held.clock.copy(), held.doubt.copy(),
                    held.lost.copy(), held.observation()])
        if held.physics is not None:
            out.append(held.motion.copy())
    return out


@pytest.mark.parametrize("make", [wild_map, plain_map])
def test_the_compiled_map_lives_as_its_python(make, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    report = compiled.map_report
    monkeypatch.setattr(compiled, "map_report", lambda *a: calls.append(1) or report(*a))
    fast = life(make)
    assert calls  # the reports took the compiled call
    for name in KERNELS:  # the map's Python: the reference
        monkeypatch.setattr(compiled, name, None)
    slow = life(make)
    assert len(fast) == len(slow)
    for a, b in zip(fast, slow, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


def asked(held: Map, rng: np.random.Generator, moments: int = 30) -> list[object]:
    """A life whose every moment asks the map about each thing it holds twice."""
    out: list[object] = []
    side = held.plane[1] - held.plane[0]
    for t in range(moments):
        seen = [(k, rng.random(2) * side, {"hue": int(rng.integers(4))}) for k in range(2)]
        held.moment(seen=seen, lit=lambda points, e=0.3 * side + (t % 5): points[:, 0] < e)
        for _ in range(2):
            for thing in held.things():
                if len(thing["cells"]):
                    cells = np.zeros(held.count, dtype=bool)
                    cells[thing["cells"]] = True
                    out.append(held.thing_at(None, cells=cells, body=(1.0, 2.0), heading=0.3))
            out.append(held.thing_at(rng.random(2) * side))
    out.append(dict(held.work))
    return out


def test_the_maps_answers_read_back_are_the_answers_asked_anew(monkeypatch) -> None:
    kept = asked(wild_map(), np.random.default_rng(7))
    monkeypatch.setattr(Map, "_answers", lambda self: None)
    anew = asked(wild_map(), np.random.default_rng(7))
    assert len(kept) == len(anew)
    for a, b in zip(kept, anew, strict=True):
        if isinstance(a, dict) and "kind" in a:
            assert isinstance(b, dict) and a["kind"] == b["kind"]
            assert a["place"].tobytes() == b["place"].tobytes()
            assert (a["bearing"], a["distance"]) == (b["bearing"], b["distance"])
            for group in a["values"]:
                assert a["values"][group].tobytes() == b["values"][group].tobytes()
        else:
            assert a == b  # None, or the map's work: every question counted both ways


def test_an_answer_read_back_is_the_readers_own() -> None:
    held = wild_map()
    held.moment(seen=[(0, np.array([6.0, 6.0]), {"hue": 1})], lit=lambda p: p[:, 0] < 12.0)
    first = held.thing_at(np.array([6.0, 6.0]))
    assert first is not None
    first["place"][:] = -1.0
    first["values"]["hue"][:] = 9.0
    again = held.thing_at(np.array([6.0, 6.0]))
    assert again is not None and (again["place"] > 0).all() and (again["values"]["hue"] <= 1).all()
