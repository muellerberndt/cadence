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


# the default read, and one whose slots pass its position table, where the grid's terms read and
# the slope's lean stays
READS = [{}, {"recall_position": 2}]


def _pair(**options) -> tuple[cd.Brain, cd.Brain, np.random.Generator]:
    subject = brain(**options)
    rng = lived(subject)
    other = copy.deepcopy(subject)
    other.retune(learning_compiled_moment=True)
    return subject, other, rng


@pytest.mark.parametrize("options", READS)
def test_the_compiled_read_is_the_reads_equation(options) -> None:
    plain, compiled, rng = _pair(**options)
    assert plain.recall.slots > (2 if options else 0)
    x = rng.random((4, 8))
    a, b = plain.stimulus(x), compiled.stimulus(x)
    np.testing.assert_allclose(b, a, rtol=0, atol=1e-12)
    np.testing.assert_allclose(compiled.recall._attended, plain.recall._attended, atol=1e-12)
    np.testing.assert_allclose(compiled.recall.last.address, plain.recall.last.address,
                               atol=1e-12)
    np.testing.assert_allclose(compiled.recall.trace, plain.recall.trace, atol=1e-12)


@pytest.mark.parametrize("options", READS)
def test_the_compiled_lesson_steps_the_maps_as_the_ordinary_one(options) -> None:
    plain, compiled, rng = _pair(**options)
    x = rng.random((4, 8))
    labels = np.array([0, 1, 2, 3])
    plain.lesson(x, labels)
    compiled.lesson(x, labels)
    assert plain.last_learning["demonstration_accepted_presentations"] > 0
    # the contrast and the update of every synapse and bias
    np.testing.assert_allclose(compiled.brain.efficacy, plain.brain.efficacy, rtol=0, atol=1e-12)
    np.testing.assert_allclose(compiled.brain.bias, plain.brain.bias, rtol=0, atol=1e-12)
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


def test_the_carried_gains_step_as_the_ordinary_ones() -> None:
    plain, compiled, rng = _pair()
    assert plain.hearing is not None
    x = rng.random((4, 8))
    labels = np.array([1, 0, 3, 2])
    plain.lesson(x, labels)
    compiled.lesson(x, labels)
    for name, gain in plain.carried_gains.items():
        assert compiled.carried_gains[name] == pytest.approx(gain, abs=1e-12), name


@pytest.mark.parametrize("began", [False, True])
def test_the_compiled_write_keeps_every_bit(began: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from cadence import _moment

    taken = []
    write = _moment.recall_write

    def counted(*args: object) -> bool:
        taken.append(write(*args))  # type: ignore[arg-type]
        return taken[-1]

    monkeypatch.setattr(_moment, "recall_write", counted)
    subject = brain(recall_capacity=6)
    rng = lived(subject, moments=9)  # past the capacity: the oldest slots are dropped
    recall = subject.recall
    if began:  # two streams begin again, and one closed an episode
        recall.reset(4, rows=np.array([1, 3]))
        recall.end(4, rows=np.array([0]))
    for _ in range(4):
        subject.stimulus(rng.random((4, 8)))  # a read the write's focus follows
        recall.compiled_moment = True
        key = rng.random((4, recall.address_width))
        value = rng.random((4, recall.value_width))
        plain = copy.deepcopy(recall)  # its slots are no views of rooms: the ordinary write
        plain.compiled_moment = False
        plain._observe(key, value)
        recall._observe(key, value)
        for name in ("keys", "values", "nexts", "held", "told", "focus", "ended"):
            np.testing.assert_array_equal(getattr(recall, name), getattr(plain, name), name)
    assert any(taken)  # the compiled write ran where the rooms had space


def _chamber_life(subject: cd.Brain, seed: int, moments: int = 66) -> list[np.ndarray]:
    """Taught and acted moments on streams of four, as the grammar chamber lives them: a lesson,
    then the act of the same observations. Every third lesson shows the brain's own answer, so
    it teaches no row and the act after it reads the lesson's read and solve. The streams begin
    again after 22 moments, past the first room behind the memory's slots."""
    rng = np.random.default_rng(seed)
    actions = []
    for t in range(moments):
        if t % 22 == 0:
            subject.reset()
        x = (rng.random((4, 8)) < 0.3).astype(float)
        labels = rng.integers(0, 4, 4)
        if t % 3 == 0:
            labels = copy.deepcopy(subject).act(x, greedy=True)
        if t % 22 >= 2:
            subject.lesson(x, labels)
        actions.append(subject.act(x, greedy=True))
    return actions


# the compiled passes through the general path, with the shortcuts, and with an act and a
# lesson's judgment each as one compiled call
MODES = [(False, False), (True, False), (True, True)]


@pytest.mark.parametrize("options", READS)
def test_the_shortcuts_keep_the_compiled_moments_bits(options, monkeypatch) -> None:
    from cadence import _moment

    whole: dict[int, int] = {}
    call = _moment.one_call

    def counted(subject, x, state, mode, spans, after):
        found = call(subject, x, state, mode, spans, after)
        if found is not None and found[3][0] == 0:
            whole[mode] = whole.get(mode, 0) + 1
        return found

    monkeypatch.setattr(_moment, "one_call", counted)
    lives = []
    for shortcuts, one_call in MODES:
        monkeypatch.setattr(_moment, "SHORTCUTS", shortcuts)
        monkeypatch.setattr(_moment, "ONE_CALL", one_call)
        subject = brain(learning_compiled_moment=True, **options)
        actions = _chamber_life(subject, 5)
        lives.append((subject, actions))
    for other in lives[1:]:
        _same_life(lives[0], other)
    # most acts and every lesson's judgment ran as one call (the acts after a lesson that taught
    # nothing read the lesson's solve, and a stream's first act writes through the general path)
    assert whole[0] >= 36 and whole[1] >= 54, whole


def _same_life(first, second) -> None:
    (a, acts_a), (b, acts_b) = first, second
    for x, y in zip(acts_a, acts_b, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(a.brain.efficacy, b.brain.efficacy)
    np.testing.assert_array_equal(a.brain.bias, b.brain.bias)
    for name in ("head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps",
                 "grid_maps"):
        for ours, theirs in zip(getattr(a.recall, name), getattr(b.recall, name), strict=True):
            for u, v in zip(ours, theirs, strict=True):
                np.testing.assert_array_equal(u, v)
    for name in ("keys", "values", "nexts", "held", "told", "focus", "previous", "ended"):
        np.testing.assert_array_equal(getattr(a.recall, name), getattr(b.recall, name), name)
    assert a.recall.writes == b.recall.writes
    np.testing.assert_array_equal(a.working_memory.trace, b.working_memory.trace)
    np.testing.assert_array_equal(a.working_memory.last, b.working_memory.last)
    np.testing.assert_array_equal(a.basal_ganglia.state.v, b.basal_ganglia.state.v)
    assert a.carried_gains == b.carried_gains
    assert dict(a.last_settlement) == dict(b.last_settlement)
    assert a.last_learning == b.last_learning
