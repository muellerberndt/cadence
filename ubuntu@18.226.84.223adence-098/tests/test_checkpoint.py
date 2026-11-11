"""A learner survives a round trip through a checkpoint, backend changes included."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import cadence as cd


def trained_learner(seed: int = 0) -> tuple[cd.Learner, np.ndarray]:
    connectome = cd.layered(6, 5, 3, density=1.0, seed=seed).with_populations(extra=[0, 1])
    config = cd.LearnerConfig(eta=0.4, eta_bias=0.1, momentum=0.5, normalize=0.5)
    tie = np.full(connectome.synapses, -1, dtype=np.int64)
    tie[:4] = 0
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        config,
        tie_groups=tie,
    )
    rng = np.random.default_rng(seed)
    drive = learner.brain.stimulus_levels(rng.random((5, connectome.n)) * 0.6)
    for _ in range(3):
        learner.step(drive, np.array([0, 1, 2, 1, 0]))
    return learner, drive


def test_round_trip_reproduces_brains_and_state(tmp_path: Path) -> None:
    learner, drive = trained_learner()
    path = learner.save(tmp_path / "net")
    assert path.suffix == ".npz"
    back = cd.Learner.load(path)
    assert np.allclose(back.brain.efficacy, learner.brain.efficacy)
    assert np.allclose(back.brain.bias, learner.brain.bias)
    assert np.allclose(back.free(drive).activation, learner.free(drive).activation)
    assert back.updates == learner.updates
    assert back.config == learner.config
    assert back.brain.connectome.populations == learner.brain.connectome.populations
    assert back.brain.connectome.digest() == learner.brain.connectome.digest()
    assert back.tie_groups is not None and np.array_equal(back.tie_groups, learner.tie_groups)
    assert np.allclose(back.velocity, learner.velocity)
    assert np.allclose(back.second_moment, learner.second_moment)
    # learning continues identically from the checkpoint
    labels = np.array([2, 2, 1, 0, 0])
    a = learner.step(drive, labels)[0].nudged.activation
    b = back.step(drive, labels)[0].nudged.activation
    assert np.allclose(a, b)
    assert np.allclose(back.brain.efficacy, learner.brain.efficacy)


def test_load_can_change_backend_and_freeze(tmp_path: Path) -> None:
    learner, drive = trained_learner()
    path = cd.save(learner, tmp_path / "net.npz")
    frozen = cd.load(path, config=cd.LearnerConfig(eta=0.0, eta_bias=0.0))
    before = frozen.brain.efficacy.copy()
    frozen.step(drive, np.array([0, 1, 2, 1, 0]))
    assert np.array_equal(frozen.brain.efficacy, before)  # a deployment that only settles
    if "torch" in cd.available_backends():
        on_torch = cd.load(path, backend="torch")
        assert on_torch.brain.backend == "torch"
        assert np.allclose(
            on_torch.free(drive).activation, learner.free(drive).activation, atol=1e-4
        )


def test_rule_with_adaptation_and_masks_survive(tmp_path: Path) -> None:
    connectome = cd.layered(4, 3, 2, density=1.0, seed=1)
    neuron_model = cd.NeuronModel(
        dt=0.5, adaptation=cd.Adaptation(tau_steps=20, strength=0.3), leak=0.1
    )
    synapses = np.zeros(connectome.synapses, dtype=bool)
    synapses[::2] = True
    neurons = np.zeros(connectome.n, dtype=bool)
    neurons[-2:] = True
    learner = cd.Learner(
        cd.NeuralGraph(connectome, neuron_model),
        connectome.populations["output"],
        plastic_synapses=synapses,
        plastic_neurons=neurons,
    )
    back = cd.load(learner.save(tmp_path / "adapt"))
    assert back.brain.neuron_model == neuron_model
    assert back.plastic_synapses is not None and np.array_equal(back.plastic_synapses, synapses)
    assert back.plastic_neurons is not None and np.array_equal(back.plastic_neurons, neurons)


def test_refuses_foreign_files(tmp_path: Path) -> None:
    path = tmp_path / "other.npz"
    np.savez(path, meta=np.array('{"format": "something-else"}'), x=np.zeros(3))
    with pytest.raises(ValueError):
        cd.load(path)


def test_a_checkpoint_from_an_earlier_release_loads_without_its_retired_knobs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A config saved with a field this version no longer has (the consolidation of 0.7)
    loads, the field dropped."""
    import json

    connectome = cd.layered(3, 4, 2, density=1.0, seed=0)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0)),
        connectome.populations["output"],
        cd.LearnerConfig(eta=0.5),
    )
    path = tmp_path / "old.npz"
    learner.save(path)
    data = dict(np.load(path, allow_pickle=False))
    meta_key = next(k for k in data if k.startswith("meta") or k == "meta_json")
    meta = json.loads(str(data[meta_key]))
    meta["config"]["consolidate"] = 0.2
    meta["config"]["restore"] = 0.1
    data[meta_key] = np.array(json.dumps(meta))
    np.savez(path, **data)
    loaded = cd.Learner.load(path)
    assert loaded.config.eta == 0.5


def test_checkpoint_preserves_explicit_precision_and_allows_override(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    connectome = cd.layered(3, 4, 2, density=1.0, seed=0)
    learner = cd.Learner(
        cd.NeuralGraph(
            connectome,
            cd.learning_neuron_model(),
            backend="torch",
            device="cpu",
            precision="float32",
        ),
        connectome.populations["output"],
    )
    path = learner.save(tmp_path / "single.npz")
    restored = cd.load(path, backend="torch", device="cpu")
    assert restored.brain.precision == "float32"
    assert restored.brain._torch.dtype == torch.float32
    overridden = cd.load(path, backend="torch", device="cpu", precision="float64")
    assert overridden.brain._torch.dtype == torch.float64

    # Old checkpoints did not store precision; their device default still applies.
    import json

    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    meta = json.loads(str(arrays["meta"]))
    meta.pop("precision", None)
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    legacy = cd.load(path, backend="torch", device="cpu")
    assert legacy.brain._torch.dtype == torch.float64
