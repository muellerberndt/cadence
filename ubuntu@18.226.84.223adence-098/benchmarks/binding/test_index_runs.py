"""Contracts of the declared lives' camera and retina."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runs = _load("index_runs")
tb = runs.ir.tb


def test_the_camera_gives_each_token_its_own_block() -> None:
    camera = runs.Camera(3)
    assert camera.inputs == 12_824
    assert np.all(camera.patches.sum(axis=1) == runs.CAMERA["block"] ** 2)
    assert np.all(camera.patches.sum(axis=0) <= 1.0)  # blocks never overlap
    assert not camera.patches[:, camera.pixels:].any()  # the body inputs carry no token


def test_the_retina_codes_blocks_and_drops_lone_pixels() -> None:
    camera, retina = runs.Camera(3), runs.Retina()
    assert retina.units == 744 + runs.CAMERA["body"]
    event = np.zeros((2, tb.INPUTS))
    event[0, [0, 5]] = 1.0  # two tokens
    event[1, [0, 5, 9]] = 1.0  # the same two and a third
    frames = camera([event])[0]
    code = retina(frames)
    active = (code[:, : len(retina.index)] > 0).sum(axis=1)
    assert 40 <= active[0] < active[1] <= 128
    np.testing.assert_allclose(np.linalg.norm(code[:, : len(retina.index)], axis=1), 1.0)
    assert np.sum(code[0] * code[1]) > 0.7  # two of three blocks shared
    noise = (np.random.default_rng(3).random((1, camera.inputs)) < 0.01).astype(float)
    assert not retina(noise)[:, : len(retina.index)].any()
    np.testing.assert_array_equal(retina(noise)[:, len(retina.index):], noise[:, camera.pixels:])


def test_a_query_shares_its_facts_code_and_little_else() -> None:
    camera, retina = runs.Camera(4), runs.Retina()
    trials = tb.draw(np.random.default_rng(4), 32, tb.holdout(4))
    codes = [retina(f) for f in camera(tb.events(trials))]
    q = trials["query"]
    asked = np.where(q[:, None] == 0, codes[0], codes[1])
    other = np.where(q[:, None] == 0, codes[1], codes[0])
    assert np.sum(codes[3] * asked, axis=1).mean() > 0.35
    assert np.sum(codes[3] * other, axis=1).mean() < 0.1


# -- the input schedule of the camera-width lives


def _schedule(monkeypatch, task: str, budgets: list[int], generic: int | None = None):
    """The frames a life trains on and reads, in order, with the graph and its lessons stubbed
    and one small panel: the life's own control flow, task draws, renderers and retina run
    unchanged."""
    training: list[list[np.ndarray]] = []
    reading: list[np.ndarray] = []

    class Inert:
        def reset(self) -> None:
            pass

    panel = {"held_out": (tb.draw(np.random.default_rng(125), 2, tb.holdout(0),
                                  queried_held=True), 1)}

    def act(brain, x, work):
        reading.append(np.asarray(x).copy())
        return np.zeros(len(x), dtype=int)

    monkeypatch.setattr(runs.ir.IndexBrain, "build", lambda *args, **kwargs: Inert())
    monkeypatch.setattr(runs.tb, "fork", lambda brain: Inert())
    monkeypatch.setattr(runs.tb, "panels", lambda seed, held: panel)
    monkeypatch.setattr(runs.tb, "act", act)
    monkeypatch.setattr(runs, "erase", lambda *args, **kwargs: None)
    monkeypatch.setattr(runs, "episode", lambda brain, seq, answer, work: training.append(
        [x.copy() for x in seq]))
    spec = {"seed": 0, "task": task, "arm": "index", "budgets": budgets,
            "genes": runs.genes_for(task, "index")}
    if generic is not None:
        spec["generic"] = generic
    runs.LIVES[task](spec)
    return training, reading


@pytest.mark.parametrize(("task", "generic"), [("wide", None), ("camera", None), ("camera", 96)],
                         ids=["wide", "camera", "camera_generic"])
def test_a_reading_changes_no_training_frame_and_every_condition_sees_the_same_frames(
    monkeypatch, task, generic
) -> None:
    """Training and the readings draw their noise from generators of their own: a life read at
    one round on its way to two trains on the frames of a life read at two alone, and every
    condition of a panel, at every reading, sees the same frames."""
    direct, _ = _schedule(monkeypatch, task, [2], generic)
    trained, read = _schedule(monkeypatch, task, [1, 2], generic)
    assert len(direct) == len(trained) == 2
    for first, second in zip(direct, trained, strict=True):
        for x, y in zip(first, second, strict=True):
            np.testing.assert_array_equal(x, y)
    events = 4  # two facts, the neutral event and the query
    reads = [read[k:k + events] for k in range(0, len(read), events)]
    assert len(reads) == 2 * 3  # two readings, three conditions each
    for frames in reads[1:]:
        for x, y in zip(frames, reads[0], strict=True):
            np.testing.assert_array_equal(x, y)


def test_the_lives_pass_the_index_memorys_options_through() -> None:
    """Every task's genes carry the options at their founders, and an override reaches the
    instrument unchanged."""
    for task in runs.TASKS:
        genes = runs.genes_for(task, "index")
        assert {k: genes[k] for k in ("address_step", "write", "null")} == {
            "address_step": "lms", "write": "always", "null": None,
        }
    extra = {"address_step": "toward", "recency": None, "write": "aroused", "null": 0.0}
    genes = runs.genes_for("continuing", "index", extra)
    assert {k: genes[k] for k in extra} == extra
    assert genes["capacity"] == runs.CONTINUING["capacity"]
    index = runs.ir.IndexMemory(4, 2, 4, genes)
    assert (index.address_step, index.recency, index.writing, index.null) == (
        "toward", None, "aroused", 0.0,
    )


# -- the fresh-bindings diagnostic


def test_the_competition_reports_the_attention_the_read_used() -> None:
    """Three orthogonal slots, a query that matches none, recency .25: the read weighs the
    oldest slot .2543, recency included, and the fresh-bindings diagnostic reports that
    record of the read, taken before the query moment was written."""
    index = runs.ir.IndexMemory(3, 1, 3, runs.genes_for("fresh", "index"))
    for k in range(3):
        index.slots.observe(np.eye(3)[[k]], np.eye(3)[[k]])
    q = index.address(np.zeros((1, 3)))
    used = index.weights(q)
    assert index.attended is None
    index.read(q)
    np.testing.assert_array_equal(index.attended, used)
    index.write(np.zeros((1, 3)), np.zeros((1, 1)))
    assert index.slots.slots == 4
    reported = runs.fresh_competition(SimpleNamespace(index=index), np.array([[0, 0, 0]]), 0)
    np.testing.assert_allclose(reported, [0.2542752125904656], rtol=1e-12)
    assert reported[0] < 1 / 3  # the weight without the recency charge
    with pytest.raises(ValueError):
        index.attended[0, 0] = 1.0
    index.reset()
    assert index.attended is None



def test_the_option_arms_carry_their_declared_genes() -> None:
    """Amendment 5's arms: each option beside the index arm, the power-law recency at capacity
    64 (12 in the continuing task) and the null slot at half a full match, heard^2 / 2."""
    index = runs.genes_for("token", "index")
    assert runs.genes_for("token", "toward") == dict(index, address_step="toward")
    assert runs.genes_for("token", "horizon") == dict(index, recency=None, capacity=64)
    assert runs.genes_for("token", "null") == dict(index, null=0.5)
    improved = runs.genes_for("token", "improved")
    assert improved == dict(index, address_step="toward", recency=None, capacity=64, null=0.5)
    assert runs.genes_for("continuing", "improved")["capacity"] == 12
    assert runs.genes_for("camera", "improved")["null"] == 1.125
    assert runs.genes_for("camera", "index")["null"] is None


def test_the_window_arms_carry_their_declared_genes() -> None:
    """Amendment 7's arms: the index arm with the context-window key of two and of four events."""
    index = runs.genes_for("succession", "index")
    assert runs.genes_for("succession", "window2") == dict(index, window=2)
    assert runs.genes_for("succession", "window4") == dict(index, window=4)
    assert runs.genes_for("continuing", "window4")["capacity"] == 12
