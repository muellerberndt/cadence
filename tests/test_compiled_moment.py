"""``learning_compiled_moment`` (opt-in): the work around the solve kernels of a moment on the CPU in
compiled passes (``cadence._moment``): the recall memory's read and its address lesson and the
place store's read. The same equations, other bits; without the option every bit is the ordinary
path's."""

from __future__ import annotations

import copy
import dataclasses
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


# the compiled passes through the general path, with the shortcuts, with an act and a lesson's
# judgment each as one compiled call, and with a taught lesson as one compiled call too
MODES = [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]


def _mode(monkeypatch, shortcuts: bool, one_call: bool, one_lesson: bool) -> None:
    from cadence import _moment

    monkeypatch.setattr(_moment, "SHORTCUTS", shortcuts)
    monkeypatch.setattr(_moment, "ONE_CALL", one_call)
    monkeypatch.setattr(_moment, "ONE_LESSON", one_lesson)


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
    phases = _moment.lesson_phases

    def counted_phases(*args):
        found = phases(*args)
        if found is not None:
            whole[2] = whole.get(2, 0) + 1
        return found

    monkeypatch.setattr(_moment, "lesson_phases", counted_phases)
    taught = _moment.taught_lesson

    def counted_lessons(*args):
        found = taught(*args)
        if found is not None:
            whole[3] = whole.get(3, 0) + 1
        return found

    monkeypatch.setattr(_moment, "taught_lesson", counted_lessons)
    lives = []
    for mode in MODES:
        _mode(monkeypatch, *mode)
        subject = brain(learning_compiled_moment=True, **options)
        actions = _chamber_life(subject, 5)
        lives.append((subject, actions))
    for other in lives[1:]:
        _same_life(lives[0], other)
    # every act but those after a lesson that taught nothing (they read the lesson's solve) and
    # every lesson's judgment ran as one call, the acts that begin the streams among them
    assert whole[0] >= 2 * 44 and whole[1] >= 2 * 54, whole
    # the phases of every lesson that taught a row, and every such lesson whole in the last mode
    assert whole[2] >= 30 and whole[3] >= 30, whole


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
    assert list(a.last_learning) == list(b.last_learning)
    # the optimizer's history, the gains' and the blocks the next solve reads
    for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
        np.testing.assert_array_equal(getattr(a.learner, name), getattr(b.learner, name), name)
    assert (a.learner.updates, a.learner.contrast_updates) == (
        b.learner.updates, b.learner.contrast_updates
    )
    if a.hearing is not None:
        np.testing.assert_array_equal(a.hearing.second, b.hearing.second)
        assert a.hearing.count == b.hearing.count
    np.testing.assert_array_equal(a.brain._blocks, b.brain._blocks)


# The options of the other 0.96 lanes, each named beside the compiled moment.
CROSS: dict[str, dict] = {
    "cortex_norm": {"recall_cortex_norm": True},
    "fade": {"memory_fade": 1.2e-4},
    "maturity": {"maturity": True},
    "fade+maturity": {"memory_fade": 1.2e-4, "maturity": True},
    "cortex_norm+fade": {"recall_cortex_norm": True, "memory_fade": 1.2e-4},
    "gains": {"recall_cortex_norm": True, "recall_gain": 1.0, "recall_tokens": [0, 1, 2, 3],
              "recall_gain_after": 5, "recall_gain_next": 2.0},
    "rate_zero": {},
}


def _crossed(name: str, compiled: bool) -> cd.Brain:
    subject = brain(learning_compiled_moment=compiled, **CROSS[name])
    if name == "rate_zero":  # every synapse that touches the motor region is held
        w = subject.connectome
        motor = np.zeros(w.n, dtype=bool)
        motor[np.asarray(w.populations["motor"])] = True
        rate = np.ones(w.synapses)
        rate[motor[w.post] | motor[w.pre]] = 0.0
        subject.learner.synapse_rate = rate
    return subject


def _mixed_life(subject: cd.Brain, seed: int, rounds: int = 16) -> list[np.ndarray]:
    """Every operation a life mixes, on streams of four: a lesson (every third shows the brain's
    own answer and teaches no row), the greedy act of the same observations, a sampled moment of
    life and its outcome, and an outcome reported on its own."""
    rng = np.random.default_rng(seed)
    actions = []
    for t in range(rounds):
        if t % 8 == 0:
            subject.reset()
        x = (rng.random((4, 8)) < 0.3).astype(float)
        labels = rng.integers(0, 4, 4)
        if t % 3 == 0:
            labels = copy.deepcopy(subject).act(x, greedy=True)
        if t % 8 >= 1:
            subject.lesson(x, labels)
        actions.append(subject.act(x, greedy=True))
        y = (rng.random((4, 8)) < 0.3).astype(float)
        actions.append(subject.step(y))
        actions.append(subject.step(x, reward=rng.random(4)))
        subject.learn(rng.random(4), np.zeros(4, dtype=bool), y)
    return actions


@pytest.mark.parametrize("name", CROSS)
def test_each_compiled_path_keeps_its_bits_beside_another_lanes_option(name, monkeypatch) -> None:
    lives = []
    for mode in MODES:
        _mode(monkeypatch, *mode)
        subject = _crossed(name, True)
        lives.append((subject, _mixed_life(subject, 11)))
    for other in lives[1:]:
        _same_life(lives[0], other)
        for attribute in ("consolidated", "fade_depth", "strength"):
            ours = getattr(lives[0][0].hippocampus, attribute, None)
            if ours is not None:
                np.testing.assert_array_equal(getattr(other[0].hippocampus, attribute), ours)
        np.testing.assert_array_equal(other[0].recall.gains, lives[0][0].recall.gains)


@pytest.mark.parametrize("name", CROSS)
def test_the_compiled_moment_agrees_with_the_ordinary_path_beside_another_lanes_option(
    name,
) -> None:
    plain, compiled = _crossed(name, False), _crossed(name, True)
    for ours, theirs in zip(_mixed_life(compiled, 11), _mixed_life(plain, 11), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_allclose(compiled.brain.efficacy, plain.brain.efficacy, rtol=0, atol=1e-8)
    np.testing.assert_allclose(compiled.brain.bias, plain.brain.bias, rtol=0, atol=1e-8)
    np.testing.assert_allclose(compiled.recall.previous, plain.recall.previous, atol=1e-8)
    for ours, theirs in zip(compiled.recall.head_map(0), plain.recall.head_map(0), strict=True):
        np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-8)
    store = plain.hippocampus
    np.testing.assert_allclose(compiled.hippocampus.consolidated, store.consolidated, atol=1e-8)
    if getattr(store, "fade", 0.0):
        np.testing.assert_allclose(compiled.hippocampus.fade_depth, store.fade_depth, atol=1e-8)
    if name == "rate_zero":
        held = compiled.learner.synapse_rate == 0
        fresh = _crossed(name, True)
        np.testing.assert_array_equal(
            compiled.brain.efficacy[held], fresh.brain.efficacy[held]
        )
    if name == "cortex_norm":
        np.testing.assert_allclose(np.linalg.norm(compiled.recall.previous, axis=1), 1.0)


# The lesson in one call at other settings of the learner and the wiring: each part of the update
# it composes (the founder's plain steps, the absolute floor, steps without momentum, the lesson's
# share), a brain with one carried route, one without carried gains, and a short position table.
# A name of ``LearnerConfig`` (the dict under "config") replaces the composed learner's setting.
LESSONS: dict[str, dict] = {
    "founder": {"teaching": "founder"},
    "normalized": {"teaching": "normalized"},
    "no_momentum": {"config": {"momentum": 0.0}},
    "plain_steps": {"config": {"momentum": 0.0, "normalize": 0.0, "normalize_relative": 0.0}},
    "share": {"learning_surprise_share": True},
    "one_route": {"recall_drive_gain": False},
    "no_gains": {"carried_gain": 0.0, "recall_drive_gain": False},
    "position": {"recall_position": 2},
}


def _taught_brain(name: str) -> cd.Brain:
    options = dict(LESSONS[name])
    config = options.pop("config", {})
    subject = brain(learning_compiled_moment=True, **options)
    if config:
        subject.learner.config = dataclasses.replace(subject.learner.config, **config)
    return subject


@pytest.mark.parametrize("name", LESSONS)
def test_a_lesson_in_one_call_keeps_the_bits_of_its_passes(name, monkeypatch) -> None:
    from cadence import _moment

    whole = [0, 0]
    taught = _moment.taught_lesson

    def counted(*args):
        found = taught(*args)
        whole[0] += 1
        whole[1] += found is not None
        return found

    monkeypatch.setattr(_moment, "taught_lesson", counted)
    lives = []
    for one_lesson in (False, True):
        _mode(monkeypatch, True, True, one_lesson)
        subject = _taught_brain(name)
        lives.append((subject, _chamber_life(subject, 7, moments=30)))
    _same_life(*lives)
    assert whole[0] == whole[1] >= 12, whole  # every taught lesson ran whole


def test_the_compiled_contrast_is_the_block_contrast() -> None:
    """The contrast's kernel against ``blocks.block_contrast`` and NumPy's mean: the same
    products and sums. The synapses' terms agree to the last place or two (one BLAS makes the
    products for NumPy, another may for the kernel); the neurons' terms to the bit."""
    from cadence import _moment
    from cadence.blocks import block_contrast

    subject = brain(learning_compiled_moment=True)
    layout, n = subject.brain.layout, subject.connectome.n
    rng = np.random.default_rng(0)
    for rows in (1, 2, 3, 7, 16, 40):
        minus = rng.random((rows, n))
        plus = minus + 0.1 * rng.standard_normal((rows, n))
        held = int(layout.starts[1])  # a range that is the same in both phases
        plus[:, :held] = minus[:, :held]
        span = 0.2
        synapses, neurons = _moment.contrast(layout, plus, minus, span)
        np.testing.assert_allclose(
            synapses, block_contrast(layout, plus, minus, rows * span), rtol=1e-12, atol=1e-15
        )
        np.testing.assert_array_equal(neurons, (plus - minus).mean(axis=0) / span)
    assert _moment.contrast(layout, plus.astype(np.float32), minus, span) is None


def test_a_lesson_the_update_refuses_raises_as_its_passes_do(monkeypatch) -> None:
    """An update that is not finite is refused whole: the call returns nothing, the passes in
    turn raise as they do without it, and the brain keeps its parameters and moments."""
    raised = []
    for one_lesson in (False, True):
        _mode(monkeypatch, True, True, one_lesson)
        subject = brain(learning_compiled_moment=True)
        rng = lived(subject)
        subject.learner.velocity = np.full(subject.connectome.synapses, np.inf)
        before = subject.brain.efficacy.copy(), copy.deepcopy(subject.recall.head_maps)
        x = rng.random((4, 8))
        wrong = (subject.act(x, greedy=True) + 1) % 4
        subject.reset()
        with pytest.raises(ValueError, match="finite") as refusal:
            subject.lesson(x, wrong)
        raised.append(str(refusal.value))
        np.testing.assert_array_equal(subject.brain.efficacy, before[0])
        for ours, theirs in zip(subject.recall.head_maps[0], before[1][0], strict=True):
            np.testing.assert_array_equal(ours, theirs)
        assert np.isinf(subject.learner.velocity).all()
    assert raised[0] == raised[1]


def test_a_lesson_whose_route_is_not_the_kernels_kind_runs_its_passes(monkeypatch) -> None:
    """A carried route's memory that the kernel cannot read as it is (here a trace that is not
    writeable) leaves the lesson to the passes in turn, with their bits."""
    from cadence import _moment

    found: list[object] = []
    taught = _moment.taught_lesson

    def counted(*args):
        found.append(taught(*args))
        return found[-1]

    monkeypatch.setattr(_moment, "taught_lesson", counted)
    lives = []
    for one_lesson in (False, True):
        _mode(monkeypatch, True, True, one_lesson)
        subject = brain(learning_compiled_moment=True)
        rng = lived(subject)
        x = rng.random((4, 8))
        wrong = (copy.deepcopy(subject).act(x, greedy=True) + 1) % 4
        subject.working_memory.trace.flags.writeable = False
        del found[:]
        subject.lesson(x, wrong)
        assert found == ([None] if one_lesson else [])
        subject.working_memory.trace.flags.writeable = True
        lives.append((subject, [subject.act(x, greedy=True)]))
    _same_life(*lives)
