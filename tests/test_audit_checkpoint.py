"""Audit 2026-09-15, class 4: checkpoints carry every array and every count."""

import json

import numpy as np
import pytest

import cadence as cd
from cadence.checkpoint import _known_config


def _same(a: cd.Learner, b: cd.Learner) -> None:
    for name in ("updates", "contrast_updates", "reciprocal", "slot_count", "slot_size"):
        assert getattr(a, name) == getattr(b, name), name
    for name in (
        "velocity",
        "velocity_bias",
        "second_moment",
        "second_moment_bias",
        "plastic_synapses",
        "plastic_neurons",
        "output_index",
        "slot_sizes",
        "reverse",
        "_members",
        "_member_groups",
    ):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.tie_groups is not None and b.tie_groups is not None
    assert np.array_equal(a.tie_groups, b.tie_groups)
    for name in ("efficacy", "bias", "log_gain"):
        assert np.array_equal(getattr(a.brain, name), getattr(b.brain, name)), name
    assert a.brain.neuron_model == b.brain.neuron_model
    assert a.config == b.config
    assert a.brain.dense_limit == b.brain.dense_limit and a.brain.precision == b.brain.precision
    assert a.brain.connectome.digest() == b.brain.connectome.digest()


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_learner_round_trip_keeps_every_array_and_count(backend: str, tmp_path) -> None:
    if backend == "torch":
        pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    c, tie = cd.embedded(12, 3, 6, 24, 6, seed=0)
    mask = np.ones(c.synapses, dtype=bool)
    mask[::5] = False
    neurons = np.ones(c.n, dtype=bool)
    neurons[:12] = False
    d = np.zeros((8, c.n))
    for p in range(3):
        d[np.arange(8), p * 12 + rng.integers(0, 12, 8)] = 1.0
    labels = np.stack([rng.integers(0, 2, 8), rng.integers(0, 4, 8)], axis=1)
    brain = cd.NeuralGraph(
        c,
        cd.learning_neuron_model(dt=1.0).replace(adaptation=cd.Adaptation(20.0, 0.1)),
        backend=backend,
        device="cpu" if backend == "torch" else None,
        log_gain=rng.normal(size=c.n) * 0.1,
        dense_limit=512,
    )
    learner = cd.Learner(
        brain,
        c.populations["output"],
        cd.LearnerConfig(momentum=0.9, normalize=0.99, tolerance=1e-4),
        plastic_synapses=mask,
        plastic_neurons=neurons,
        tie_groups=tie,
        slots=[2, 4],
    )
    for _ in range(3):
        learner.step(d, labels)
    learner.apply(np.zeros(c.synapses), np.zeros(c.n))  # an external update: updates 4, contrast 3
    path = learner.save(tmp_path / "learner", compressed=backend == "cpu")
    back = cd.Learner.load(path, backend="cpu")
    _same(learner, back)
    if backend == "torch":
        on_device = cd.Learner.load(path, backend="torch", device="cpu")
        _same(learner, on_device)
        learner.step(d, labels)
        on_device.step(d, labels)
        assert np.array_equal(learner.brain.efficacy, on_device.brain.efficacy)


def test_retired_config_fields_are_dropped_and_current_ones_kept() -> None:
    saved = {**cd.LearnerConfig(tolerance=None, momentum=0.5).to_dict(), "consolidation": 0.3}
    config = cd.LearnerConfig(**_known_config(saved))
    assert config.tolerance is None and config.momentum == 0.5


def test_checkpoint_meta_names_the_format_and_version(tmp_path) -> None:
    c = cd.layered(3, 4, 2, seed=0)
    learner = cd.Learner(cd.NeuralGraph(c, cd.learning_neuron_model()), c.populations["output"])
    path = learner.save(tmp_path / "small")
    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        assert meta["format"] == "cadence-checkpoint/2" and meta["version"] == cd.__version__
        for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias", "tie_groups"):
            assert name in data.files


def test_the_first_format_is_refused_naming_the_release_that_loads_it(tmp_path) -> None:
    c = cd.layered(3, 4, 2, seed=0)
    learner = cd.Learner(cd.NeuralGraph(c, cd.learning_neuron_model()), c.populations["output"])
    path = learner.save(tmp_path / "small")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(str(arrays["meta"]))
    arrays["meta"] = np.array(json.dumps({**meta, "format": "cadence-checkpoint/1"}))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="cadence 0.94.0 loads cadence-checkpoint/1"):
        cd.Learner.load(path)


def test_generic_brain_resumes_identically(tmp_path) -> None:
    rng = np.random.default_rng(0)
    g = cd.Brain.compose(6, 3, modules=(16,), seed=1)
    g.step(rng.random((4, 6)))
    g.step(rng.random((4, 6)), reward=rng.normal(size=4), done=np.array([False, True, False, False]))
    g.step(rng.random((4, 6)), reward=rng.normal(size=4), teacher=np.array([0, 1, 2, 0]))
    g.step(rng.random((4, 6)), reward=rng.normal(size=4))  # a pending action awaits its reward
    path = g.save(tmp_path / "generic")
    h = cd.Brain.load(path)
    for _ in range(3):
        x, r = rng.random((4, 6)), rng.normal(size=4)
        assert np.array_equal(g.step(x, reward=r), h.step(x, reward=r))
    assert np.array_equal(g.brain.efficacy, h.brain.efficacy)
    assert g.working_memory is not None and h.working_memory is not None
    assert np.array_equal(g.working_memory.trace, h.working_memory.trace)
    assert isinstance(g.hippocampus, cd.SynapticMemory) and isinstance(h.hippocampus, cd.SynapticMemory)
    assert np.array_equal(g.hippocampus.consolidated, h.hippocampus.consolidated)
    assert np.array_equal(g.hippocampus.strength, h.hippocampus.strength)
    assert np.array_equal(g.basal_ganglia.w_critic, h.basal_ganglia.w_critic)
    assert g.basal_ganglia.trace is not None and h.basal_ganglia.trace is not None
    assert np.array_equal(g.basal_ganglia.trace, h.basal_ganglia.trace)
