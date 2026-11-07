"""A brain's memories on the MLX device: the three brains of the gate live, teach, save, grow, graft and
retune there at the speech size, and past the size that pays nothing of the memory, the eligibility or
the optimizer comes to the host; a moment reads as on the float64 host path within rounding."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd
from cadence import resident
from cadence.resident import _memory_on_device, _RecallOnDevice

pytestmark = pytest.mark.skipif("mlx" not in cd.available_backends(), reason="mlx not installed")

SYNAPSES = 3e-2  # a synapse's normalized step at float32's noise: the backend's bound
READS = 1e-5  # what a read adds within rounding

SPEECH = (3920, 37)
BRAINS = {
    "default": {"modules": (512,)},
    "deep_apical": {"modules": (128, 64, 64, 64), "apical": 0.25, "teaching": "deep"},
    "both": {"modules": (256, 128), "apical": 0.25, "teaching": "deep"},
}


def rows_of(rng, rows, width=SPEECH[0]):
    return rng.random((rows, width)) * 0.25


def on_the_device(brain):
    """Nothing of the memory, the eligibility or the optimizer rests on the host."""
    recall, agent, learner, graph = brain.recall, brain.basal_ganglia, brain.learner, brain.brain
    assert isinstance(recall._device, _RecallOnDevice) and "keys" not in recall.__dict__
    assert brain.working_memory._device is not None and "trace" not in brain.working_memory.__dict__
    assert brain.hippocampus._device is not None and "strength" not in brain.hippocampus.__dict__
    assert graph._efficacy is None and graph._bias is None and graph._weights_host is None
    assert learner.__dict__.get("_device_moments") is not None
    return recall, agent


@pytest.mark.parametrize("form", list(BRAINS))
def test_a_life_at_the_speech_size_rests_on_the_device(form, tmp_path):
    """``lesson``, ``step``, ``live``, a checkpoint, ``grow``, ``graft`` and ``retune`` at the speech
    size, against the host moment by moment."""
    options = BRAINS[form]
    cpu = cd.Brain.compose(*SPEECH, seed=2, **options)
    mlx = cd.Brain.compose(*SPEECH, seed=2, backend="mlx", **options)
    rng = np.random.default_rng(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            x = rows_of(rng, 2)
            np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))
        np.testing.assert_allclose(
            mlx.basal_ganglia.state.activation[:, mlx.motor_index],
            cpu.basal_ganglia.state.activation[:, cpu.motor_index], rtol=0, atol=READS,
        )
        # a lesson: the phases' reports and the answer after it
        x, labels = rows_of(rng, 2), rng.integers(SPEECH[1], size=2)
        a, b = cpu.lesson(x, labels), mlx.lesson(x, labels)
        assert (b["free_steps"], b["nudged_steps"]) == (a["free_steps"], a["nudged_steps"])
        on_the_device(mlx)
        np.testing.assert_array_equal(cpu.act(x, greedy=True), mlx.act(x, greedy=True))
        # a reward: the step, the eligibility and the read's on the device
        cpu.step(x), mlx.step(x)
        for _ in range(2):
            x, reward = rows_of(rng, 2), rng.random(2) - 0.5
            np.testing.assert_array_equal(
                cpu.step(x, reward=reward), mlx.step(x, reward=reward)
            )
        recall, agent = on_the_device(mlx)
        assert agent._trace_mlx is not None and agent.trace is None
        assert not isinstance(agent.trace_memory, np.ndarray)
        for name in ("delta", "td_error", "value", "scale_step", "bias_step", "trace"):
            assert mlx.last_learning[name] == pytest.approx(
                cpu.last_learning[name], rel=1e-2, abs=1e-4
            ), name
        # a checkpoint comes back and goes on
        path = mlx.save(tmp_path / "brain.npz", compressed=False)
        again = cd.Brain.load(path, backend="mlx")
        for _ in range(2):  # the action a checkpoint kept pending learns on the host, the next here
            x, reward = rows_of(rng, 2), rng.random(2) - 0.5
            np.testing.assert_array_equal(mlx.step(x, reward=reward), again.step(x, reward=reward))
        assert again.recall._device is not None and again.basal_ganglia._trace_mlx is not None
        np.testing.assert_allclose(again.brain.efficacy, mlx.brain.efficacy, rtol=0, atol=1e-3)
        # the settings that keep to the device, and the parts that grow on it
        if mlx.basal_ganglia._pending is not None:
            mlx.learn(np.zeros(2), np.zeros(2, dtype=bool), rows_of(rng, 2))
        mlx.retune(recall_amplitude=2.5)
        mlx.act(rows_of(rng, 2), greedy=True)
        grown = mlx.grow(history_decays=(0.5,))
        assert grown.brain.backend == "mlx"
        grown.lesson(rows_of(rng, 2), rng.integers(SPEECH[1], size=2))
        donor = cd.Brain.compose(*SPEECH, seed=3, **options)
        grafted = mlx.graft(donor, ("association", "motor"))
        assert grafted.brain.backend == "mlx"
        grafted.lesson(rows_of(rng, 2), rng.integers(SPEECH[1], size=2))


def test_a_stream_lives_at_the_speech_size_on_the_device():
    cpu = cd.Brain.compose(*SPEECH, seed=2, modules=(512,), arousal=True)
    mlx = cd.Brain.compose(*SPEECH, seed=2, modules=(512,), arousal=True, backend="mlx")
    rng = np.random.default_rng(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(12):
            x = rows_of(rng, 1)
            kwargs = {} if t == 0 else {"reward": np.array([float(t % 4 == 0)])}
            np.testing.assert_array_equal(cpu.live(x, **kwargs), mlx.live(x, **kwargs))
            assert mlx.last_arousal["mode"] == cpu.last_arousal["mode"]
    assert mlx.recall._device is not None


def test_a_moment_takes_no_arrays_back_and_forth(monkeypatch):
    """Past a brain's first moments a lesson and a step bring nothing of the memory back."""
    mlx = cd.Brain.compose(*SPEECH, seed=2, modules=(512,), backend="mlx")
    rng = np.random.default_rng(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlx.step(rows_of(rng, 2))
        mlx.step(rows_of(rng, 2), reward=rng.random(2))
        held = mlx.recall._device

        def forbidden(*_, **__):
            raise AssertionError("the memory came back to the host")

        monkeypatch.setattr(resident._RecallOnDevice, "give_back", forbidden)
        monkeypatch.setattr(resident._RecallOnDevice, "adopt", forbidden)
        monkeypatch.setattr(resident._StoreOnDevice, "give_back", forbidden)
        monkeypatch.setattr(mlx.brain._mlx, "to_edges", forbidden)
        monkeypatch.setattr(mlx.brain._mlx, "rows_to_edges", forbidden)
        for _ in range(3):
            x = rows_of(rng, 2)
            mlx.step(x, reward=rng.random(2))
        mlx.learn(rng.random(2), np.zeros(2, dtype=bool), rows_of(rng, 2))
        for _ in range(3):
            x = rows_of(rng, 2)
            mlx.lesson(x, rng.integers(SPEECH[1], size=2))
            mlx.act(x, greedy=True)
    assert mlx.recall._device is held
    on_the_device(mlx)


def test_the_size_decides_on_mlx(monkeypatch):
    class Kernel:
        mx = None
        plain = False

    monkeypatch.delenv("CADENCE_MLX_MEMORY", raising=False)
    kernel = Kernel()
    assert _memory_on_device(kernel, 1, 3920) and _memory_on_device(kernel, 16, 3920)
    assert not _memory_on_device(kernel, 1, 35)
    monkeypatch.setenv("CADENCE_MLX_MEMORY", "1")
    assert _memory_on_device(kernel, 1, 35)
    monkeypatch.setenv("CADENCE_MLX_MEMORY", "0")
    assert not _memory_on_device(kernel, 1, 3920)
    monkeypatch.delenv("CADENCE_MLX_MEMORY")
    kernel.plain = True
    assert not _memory_on_device(kernel, 1, 3920)


def test_a_slot_is_written_in_place():
    """The room's slots stay where they were: a write touches one slot of a shared buffer."""
    mlx = cd.Brain.compose(64, 8, seed=1, backend="mlx", modules=(32,), recall_capacity=16)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CADENCE_MLX_MEMORY", "1")
        rng = np.random.default_rng(6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seen = []
            for _ in range(24):
                x = rng.random((2, 64))
                mlx.act(x, greedy=True)
                seen.append(x)
        keys = np.array(mlx.recall._device.keys)
    assert keys.shape[1] == 16
    expected = np.stack(seen[-16:], axis=1) * mlx.recall.heard
    np.testing.assert_allclose(keys, expected, rtol=0, atol=1e-6)
