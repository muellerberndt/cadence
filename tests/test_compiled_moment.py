"""``learning_compiled_moment`` (opt-in): the work around the solve kernels of a moment on the CPU in
compiled passes (``cadence._moment``): the recall memory's read and its address lesson and the
place store's read. The same equations, other bits; without the option every bit is the ordinary
path's."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence import brain as brain_module

pytest.importorskip("numba")
pytestmark = pytest.mark.skipif(
    not brain_module._FUSED, reason="CADENCE_FUSED=0 reads the NumPy loop alone"
)


def brain(**options) -> cd.Brain:
    return cd.Brain.compose(8, 4, modules=(32,), seed=3, backend="cpu", **options)


def lived(subject: cd.Brain, moments: int = 6, seed: int = 0) -> np.random.Generator:
    """A few taught moments on four streams, so the slots and the maps hold something."""
    rng = np.random.default_rng(seed)
    for _ in range(moments):
        x = rng.random((4, 8))
        subject.lesson(x, rng.integers(0, 4, 4))
        subject.act(x, greedy=True)
    return rng


def test_the_option_is_off_by_default_and_named_by_retune() -> None:
    assert cd.LearnerConfig().compiled_moment is False
    assert "compiled_moment" not in cd.LearnerConfig().to_dict()
    subject = brain()
    assert "learning_compiled_moment" not in subject.describe()["genes"]  # off: left out
    subject.retune(learning_compiled_moment=True)
    assert subject.learner.config.compiled_moment is True
    assert subject.describe()["genes"]["learning_compiled_moment"] is True
    with pytest.raises(ValueError, match="compiled_moment"):
        cd.LearnerConfig(compiled_moment=1)  # type: ignore[arg-type]


def test_without_the_option_a_life_keeps_its_bits() -> None:
    plain, off = brain(), brain(learning_compiled_moment=False)
    lived(plain)
    lived(off)
    np.testing.assert_array_equal(plain.brain.efficacy, off.brain.efficacy)
    for a, b in zip(plain.recall.head_maps[0], off.recall.head_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    assert "compiled_moment" not in vars(plain.recall) or not plain.recall.compiled_moment


def _pair() -> tuple[cd.Brain, cd.Brain, np.random.Generator]:
    subject = brain()
    rng = lived(subject)
    other = copy.deepcopy(subject)
    other.retune(learning_compiled_moment=True)
    return subject, other, rng


def test_the_compiled_read_is_the_reads_equation() -> None:
    plain, compiled, rng = _pair()
    x = rng.random((4, 8))
    a, b = plain.stimulus(x), compiled.stimulus(x)
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-12)
    np.testing.assert_allclose(compiled.recall._attended, plain.recall._attended, atol=1e-12)
    np.testing.assert_allclose(compiled.recall.last.address, plain.recall.last.address,
                               atol=1e-12)
    np.testing.assert_allclose(compiled.recall.trace, plain.recall.trace, atol=1e-12)


def test_the_compiled_lesson_steps_the_maps_as_the_ordinary_one() -> None:
    plain, compiled, rng = _pair()
    x = rng.random((4, 8))
    labels = np.array([0, 1, 2, 3])
    plain.lesson(x, labels)
    compiled.lesson(x, labels)
    for name in ("head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps",
                 "grid_maps"):
        for ours, theirs in zip(getattr(compiled.recall, name), getattr(plain.recall, name),
                                strict=True):
            for u, v in zip(ours, theirs, strict=True):
                np.testing.assert_allclose(u, v, rtol=0, atol=1e-11)
    for key, value in plain.last_learning.items():
        if "address" in key or "position" in key:
            assert compiled.last_learning[key] == pytest.approx(value, abs=1e-10), key


def test_streams_that_began_since_read_and_learn_with_fewer_slots() -> None:
    plain, compiled, rng = _pair()
    for subject in (plain, compiled):  # two streams begin again: they hold fewer slots
        subject.recall.reset(4, rows=np.array([1, 3]))
    assert not plain.recall.held.all()
    x = rng.random((4, 8))
    np.testing.assert_allclose(compiled.stimulus(x), plain.stimulus(x), atol=1e-12)
    labels = np.array([3, 2, 1, 0])
    plain.lesson(x, labels)
    compiled.lesson(x, labels)
    for ours, theirs in zip(compiled.recall.head_maps[0], plain.recall.head_maps[0], strict=True):
        np.testing.assert_allclose(ours, theirs, atol=1e-11)


def test_a_compiled_life_saves_and_continues() -> None:
    subject = brain(learning_compiled_moment=True)
    rng = lived(subject)
    with tempfile.TemporaryDirectory() as folder:
        loaded = cd.Brain.load(subject.save(Path(folder) / "brain"))
    assert loaded.learner.config.compiled_moment is True
    x = rng.random((4, 8))
    np.testing.assert_array_equal(subject.act(x, greedy=True), loaded.act(x, greedy=True))


def test_a_read_the_kernels_do_not_take_runs_the_ordinary_path() -> None:
    plain = brain(recall_heads=2)
    rng = lived(plain)
    compiled = copy.deepcopy(plain)
    compiled.retune(learning_compiled_moment=True)
    x = rng.random((4, 8))
    np.testing.assert_array_equal(compiled.stimulus(x), plain.stimulus(x))
