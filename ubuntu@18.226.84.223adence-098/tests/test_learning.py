from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import cadence as cd
from cadence.learning import SCALE_CAP


def two_blobs(n_per: int = 60, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two classes of 8-pixel patterns: the left half lit, or the right half, with noise."""
    rng = np.random.default_rng(seed)
    x = np.zeros((2 * n_per, 8))
    y = np.repeat([0, 1], n_per)
    x[:n_per, :4] = 1.0
    x[n_per:, 4:] = 1.0
    x = np.clip(x + 0.3 * rng.standard_normal(x.shape), 0.0, 1.0)
    return x, y


def test_learner_separates_two_classes_and_the_rule_is_local() -> None:
    connectome = cd.layered(8, 16, 2, density=0.6, seed=1)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()),
        connectome.populations["output"],
        cd.LearnerConfig(eta=2.0),
    )
    x, y = two_blobs()
    drive = learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 8))))
    before = learner.accuracy(drive, y)
    rng = np.random.default_rng(0)
    for _ in range(3):
        order = rng.permutation(len(y))
        for start in range(0, len(y), 20):
            idx = order[start : start + 20]
            learner.step(drive[idx], y[idx])
    after = learner.accuracy(drive, y)
    assert after >= 0.95 and after > before

    # Locality: a synapse's update is a function of its own two endpoints only (and, when
    # tied, of its reverse partner's two endpoints, which are the same two neurons).
    learner.reciprocal = False
    learner.reverse[:] = -1
    free = learner.free(drive[:8])
    target = learner.targets(y[:8])
    plus = learner.nudged(drive[:8], free, target)
    minus = learner.nudged(drive[:8], free, target, sign=-1.0)
    scale_before = learner.brain.efficacy.copy()
    learner.update(free, plus, minus)
    delta = learner.brain.efficacy - scale_before
    w = connectome
    expected = (
        learner.config.eta
        / (2.0 * learner.config.beta)
        * (
            (plus.activation[:, w.pre] * plus.activation[:, w.post]).mean(axis=0)
            - (minus.activation[:, w.pre] * minus.activation[:, w.post]).mean(axis=0)
        )
    )
    clipped = np.abs(scale_before + expected) > SCALE_CAP
    assert np.allclose(delta[~clipped], expected[~clipped])


def test_free_phase_never_sees_the_target() -> None:
    connectome = cd.layered(4, 6, 2, seed=2)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"]
    )
    drive = learner.brain.stimulus_levels(np.pad(np.eye(4)[:2], ((0, 0), (0, connectome.n - 4))))
    a = learner.free(drive).activation
    b = learner.free(drive).activation
    assert np.array_equal(a, b)  # deterministic and label-free by construction
    free = learner.free(drive)
    target = learner.targets(np.array([0, 1]))
    plus = learner.nudged(drive, free, target)
    minus = learner.nudged(drive, free, target, sign=-1.0)
    out = connectome.populations["output"]
    for row, label in ((0, 0), (1, 1)):
        assert plus.activation[row, out[label]] > free.activation[row, out[label]]
        assert minus.activation[row, out[label]] < free.activation[row, out[label]]


def test_contrast_tracks_the_loss_gradient() -> None:
    """The centered contrast points along the finite-difference gradient of the nudge's loss."""
    connectome = cd.layered(8, 12, 3, density=0.7, seed=3)
    config = cd.LearnerConfig(beta=0.05, tolerance=1e-9, free_steps=400, nudged_steps=400)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"], config
    )
    rng = np.random.default_rng(0)
    x = rng.random((16, 8))
    labels = rng.integers(0, 3, 16)
    drive = learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, connectome.n - 8))))
    out = np.asarray(connectome.populations["output"])

    def loss(brain: cd.NeuralGraph) -> float:
        s = brain.settle_batch(drive, steps=400, tolerance=1e-9).activation[:, out]
        z = s / config.temperature
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return float(-np.log(p[np.arange(16), labels]).mean())

    free = learner.free(drive)
    target = learner.targets(labels)
    synapse_term, _ = learner.contrast(
        free,
        learner.nudged(drive, free, target),
        learner.nudged(drive, free, target, sign=-1.0),
    )
    tied = synapse_term + np.where(learner.reverse >= 0, synapse_term[learner.reverse], 0.0)
    base = loss(learner.brain)
    sample = rng.choice(connectome.synapses, 40, replace=False)
    finite = []
    for e in sample:
        scale = learner.brain.efficacy.copy()
        scale[e] += 1e-4
        if learner.reverse[e] >= 0:
            scale[learner.reverse[e]] += 1e-4
        finite.append(-(loss(learner.brain.with_parameters(efficacy=scale)) - base) / 1e-4)
    correlation = np.corrcoef(np.asarray(finite), tied[sample])[0, 1]
    assert correlation > 0.9


def test_leak_keeps_rest_exact_and_responds_below_rest() -> None:
    neuron_model = cd.learning_neuron_model(leak=0.1)
    assert neuron_model.activation(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]
    below = neuron_model.activation(np.array([-1.0, -3.0, -50.0]))
    assert (below < 0).all() and (below >= -0.1).all()
    assert neuron_model.activation(np.array([1.0]))[0] > 0
    assert neuron_model.slope_at(np.array([-1.0]))[0] > 0


def test_brain_stops_at_tolerance_and_reports_steps() -> None:
    connectome = cd.layered(4, 6, 2, seed=4)
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
    drive = brain.stimulus_levels(np.pad(np.eye(4)[:1], ((0, 0), (0, connectome.n - 4))))
    fixed = brain.settle_batch(drive, steps=500)
    early = brain.settle_batch(drive, steps=500, tolerance=1e-6)
    assert early.steps < 500
    assert np.abs(early.activation - fixed.activation).max() < 1e-4
    assert early.trajectory is None
    with_trace = brain.settle_batch(drive, steps=500, tolerance=1e-6, trajectory=True)
    assert with_trace.trajectory is not None and len(with_trace.trajectory) == with_trace.steps


def test_dense_and_segmented_transport_agree() -> None:
    connectome = cd.layered(6, 10, 3, seed=5)
    neuron_model = cd.learning_neuron_model()
    drive = cd.NeuralGraph(connectome, neuron_model).stimulus_levels(
        np.pad(np.random.default_rng(0).random((5, 6)), ((0, 0), (0, connectome.n - 6)))
    )
    dense = cd.NeuralGraph(connectome, neuron_model).settle_batch(drive, steps=80)
    segmented = cd.NeuralGraph(connectome, neuron_model, dense_limit=0).settle_batch(drive, steps=80)
    assert np.abs(dense.activation - segmented.activation).max() < 1e-12
    assert cd.NeuralGraph(connectome, neuron_model).to_dict()["transport"] == "dense"
    assert cd.NeuralGraph(connectome, neuron_model, dense_limit=0).to_dict()["transport"] == "segmented"


def test_weighted_nudge_pushes_each_row_its_own_way() -> None:
    connectome = cd.layered(4, 6, 2, seed=6)
    config = cd.LearnerConfig(tolerance=1e-12, free_steps=1000, nudged_steps=200)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"], config
    )
    drive = learner.brain.stimulus_levels(np.pad(np.eye(4)[:2], ((0, 0), (0, connectome.n - 4))))
    free = learner.free(drive)
    target = learner.targets(np.array([0, 0]))
    out = connectome.populations["output"]
    pulled = learner.nudged(drive, free, target, weight=np.array([1.0, -1.0]))
    assert pulled.activation[0, out[0]] > free.activation[0, out[0]]  # advantage: toward
    assert pulled.activation[1, out[0]] < free.activation[1, out[0]]  # penalty: away
    silent = learner.nudged(drive, free, target, weight=np.array([0.0, 0.0]))
    assert np.allclose(silent.activation, free.activation, atol=1e-6)


def test_normalized_steps_stay_local_and_bounded() -> None:
    connectome = cd.layered(4, 6, 2, seed=8)
    config = cd.LearnerConfig(eta=0.05, normalize=0.9)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"], config
    )
    drive = learner.brain.stimulus_levels(np.pad(np.eye(4)[:2], ((0, 0), (0, connectome.n - 4))))
    before = learner.brain.efficacy.copy()
    learner.step(drive, np.array([0, 1]))
    moved = np.abs(learner.brain.efficacy - before)
    assert moved.max() > 0
    # with the RMS floor of 1e-3 and one update, no synapse moves more than eta / (1 - rho) ** 0.5
    assert moved.max() <= config.eta / np.sqrt(1 - config.normalize) + 1e-9
    assert learner.second_moment.shape == (connectome.synapses,)


def test_adaptive_local_step_is_bias_corrected() -> None:
    """With momentum and normalization the first step of every moving synapse is eta in size
    (the running average and the RMS are corrected for their short history, as Adam's are),
    and a step is the same whether the contrast is large or small."""
    connectome = cd.layered(4, 6, 2, seed=8)
    config = cd.LearnerConfig(
        eta=0.01, eta_bias=0.0, momentum=0.9, normalize=0.999, normalize_floor=1e-12
    )
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()), connectome.populations["output"], config
    )
    drive = learner.brain.stimulus_levels(np.pad(np.eye(4)[:2], ((0, 0), (0, connectome.n - 4))))
    before = learner.brain.efficacy.copy()
    learner.step(drive, np.array([0, 1]))
    moved = np.abs(learner.brain.efficacy - before)
    moving = moved > 0
    assert moving.any()
    assert np.allclose(moved[moving], config.eta, rtol=1e-6)
    # the same net, contrasts scaled down a hundredfold by a smaller nudge: the same first step
    small = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()),
        connectome.populations["output"],
        dataclasses.replace(config, beta=0.001),
    )
    small.step(drive, np.array([0, 1]))
    moved_small = np.abs(small.brain.efficacy - before)
    assert np.allclose(moved_small[moving], config.eta, rtol=1e-6)


def test_tie_groups_share_one_scale_across_positions() -> None:
    connectome, groups = cd.embedded(vocabulary=5, positions=3, dim=2, hidden=4, outputs=2, seed=1)
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()),
        connectome.populations["output"],
        tie_groups=groups,
    )
    # every (token, unit) synapse starts equal across the positions and stays equal after an update
    tied = groups >= 0
    rng = np.random.default_rng(0)
    windows = rng.integers(0, 5, size=(6, 3))
    drive = np.zeros((6, connectome.n))
    for r in range(6):
        for p in range(3):
            drive[r, p * 5 + windows[r, p]] = 1.0
    learner.step(drive, rng.integers(0, 2, 6))
    scale = learner.brain.efficacy
    for g in np.unique(groups[tied]):
        members = scale[groups == g]
        assert members.size == 3 and np.allclose(members, members[0])
    # one embedding table, the tied dense synapses, and the biases
    assert learner.parameters() == 5 * 2 + (3 * 2) * 4 + 4 * 2 + connectome.n


def test_an_update_at_rate_zero_leaves_every_synapse_as_it_was() -> None:
    """No leak: what is not relearned keeps its efficacy. The leak (``decay``) and the
    one-sided teaching law (``centered=False``) of 0.84.3 are not settings."""
    connectome = cd.layered(4, 3, 2, density=1.0, seed=0)
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0))
    config = cd.LearnerConfig(eta=0.0, eta_bias=0.0)
    learner = cd.Learner(brain, connectome.populations["output"], config)
    before = learner.brain.efficacy.copy()
    drive = brain.stimulus_levels(np.zeros((2, connectome.n)))
    free = learner.free(drive)
    target = learner.targets(np.array([0, 1]))
    nudged = learner.nudged(drive, free, target)
    learner.update(free, nudged, learner.nudged(drive, free, target, sign=-1.0))
    assert np.array_equal(learner.brain.efficacy, before)
    for name, value in (("decay", 0.1), ("centered", False)):
        with pytest.raises(TypeError):
            cd.LearnerConfig(**{name: value})
        with pytest.raises(ValueError, match=f"learning_{name} was retired.*0.84.3"):
            cd.Brain.compose(4, 2, modules=(8,), **{"recall": False, **{f"learning_{name}": value}})


def test_a_composed_brain_learns_with_the_cross_entropy_nudge(tmp_path) -> None:
    """The quadratic nudge is a standalone learner's: a composed brain refuses it by name, in
    its learning configuration, through retune and in a checkpoint, and describes no nudge
    gene."""
    with pytest.raises(ValueError, match="learning_nudge was retired.*0.84.3"):
        cd.Brain.compose(4, 2, modules=(8,), learning_nudge="quadratic", recall=False)
    quadratic = cd.LearnerConfig(nudge="quadratic")
    with pytest.raises(ValueError, match="cross-entropy nudge"):
        cd.Brain.compose(4, 2, modules=(8,), learning=quadratic, recall=False)
    brain = cd.Brain.compose(4, 2, modules=(8,), recall=False)
    with pytest.raises(ValueError, match="cross-entropy nudge"):
        brain.retune(learning=quadratic)
    assert "learning_nudge" not in brain.describe()["genes"]
    assert brain.learner.config.nudge == "cross_entropy"
    assert "learning_nudge" not in brain.describe()["genes"]
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    meta = json.loads(str(arrays["meta"]))
    meta["config"]["nudge"] = "quadratic"
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="nudge='quadratic'.*retired in 0.85.0.*0.84.3"):
        cd.Brain.load(path)
    assert cd.Learner.load(path).config.nudge == "quadratic"  # a standalone learner keeps it


def test_trainable_masks_leave_the_rest_of_the_net_alone() -> None:
    connectome = cd.layered(4, 3, 2, density=1.0, seed=0)
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model(dt=1.0))
    synapses = np.zeros(connectome.synapses, dtype=bool)
    synapses[: connectome.synapses // 2] = True
    neurons = np.zeros(connectome.n, dtype=bool)
    neurons[list(connectome.populations["output"])] = True
    config = cd.LearnerConfig(eta=0.5, eta_bias=0.5)
    learner = cd.Learner(
        brain,
        connectome.populations["output"],
        config,
        plastic_synapses=synapses,
        plastic_neurons=neurons,
    )
    scale0, bias0 = learner.brain.efficacy.copy(), learner.brain.bias.copy()
    drive = brain.stimulus_levels(np.ones((2, connectome.n)) * 0.5)
    learner.step(drive, np.array([0, 1]))
    assert np.array_equal(learner.brain.efficacy[~synapses], scale0[~synapses])
    assert np.array_equal(learner.brain.bias[~neurons], bias0[~neurons])
    assert not np.array_equal(learner.brain.bias[neurons], bias0[neurons])


@pytest.mark.parametrize("reciprocal", [True, False])
def test_parameters_count_only_trainable_synapses_and_biases(reciprocal: bool) -> None:
    connectome = cd.Connectome.from_synapses(n=4, pre=[0, 1, 2, 3], post=[1, 0, 3, 2])
    synapses = connectome.pre % 2 == 0  # two separate pairs, one trainable side in each
    neurons = np.array([False, True, False, True])
    learner = cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()),
        [1, 3],
        reciprocal=reciprocal,
        plastic_synapses=synapses,
        plastic_neurons=neurons,
    )
    assert learner.parameters() == 4  # two independent moving synapses and two biases
    synapses[:] = True
    assert learner.parameters() == (4 if reciprocal else 6)
    neurons[:] = False
    assert learner.parameters() == (2 if reciprocal else 4)
    synapses[:] = False
    assert learner.parameters() == 0


@pytest.mark.parametrize("backend", ["torch", "mlx"])
def test_contrast_on_the_device_matches_the_host(backend: str) -> None:
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    w = cd.layered(12, 8, 4, density=1.0, seed=3)
    neuron_model = cd.learning_neuron_model(dt=1.0)
    kw = {"device": "cpu"} if backend == "torch" else {}
    device = cd.NeuralGraph(w, neuron_model, backend=backend, **kw)  # type: ignore[arg-type]
    host = cd.NeuralGraph(w, neuron_model)
    config = cd.LearnerConfig(eta=1.0, beta=0.1, temperature=0.1, tolerance=1e-4)
    drive = host.stimulus_levels(np.random.default_rng(4).random((6, w.n)) * 0.5)
    labels = np.array([0, 1, 2, 3, 0, 1])
    for brain in (device, host):
        learner = cd.Learner(brain, w.populations["output"], config)
        state, _ = learner.step(drive, labels)
        if brain is device:
            assert state.nudged.device is not None
            assert brain.contrast_on_device(state.nudged, state.opposite) is not None  # type: ignore[arg-type]
            on_device = learner.brain.efficacy.copy()
        else:
            on_host = learner.brain.efficacy.copy()
    tolerance = 1e-12 if backend == "torch" else 1e-5
    assert np.abs(on_device - on_host).max() < tolerance
    assert host.contrast_on_device(state.nudged, state.opposite) is None  # type: ignore[arg-type]


def _learner(surprise: bool, **config: object) -> cd.Learner:
    connectome = cd.layered(8, 16, 2, density=0.6, seed=1)
    settings = dict(eta=2.0, surprise=surprise, tolerance=1e-9, free_steps=4000, nudged_steps=4000)
    settings.update(config)
    return cd.Learner(
        cd.NeuralGraph(connectome, cd.learning_neuron_model()),
        connectome.populations["output"],
        cd.LearnerConfig(**settings),
    )


def _blob_drive(learner: cd.Learner) -> np.ndarray:
    x, _ = two_blobs(8)
    return learner.brain.stimulus_levels(np.pad(x, ((0, 0), (0, learner.brain.connectome.n - 8))))


@pytest.mark.parametrize("qualified", [False, True])
def test_surprise_teaches_only_the_rows_whose_answer_differs(qualified: bool) -> None:
    settings = dict(qualified=True, tolerance=1e-6) if qualified else {}
    selective, every = _learner(True, **settings), _learner(False, **settings)
    drive = _blob_drive(every)
    labels = every.predict(drive)
    labels[::2] = 1 - labels[::2]  # eight rows disagree with the net's answer, eight agree
    taught = np.zeros(16, bool)
    taught[::2] = True
    state, report = selective.step(drive, labels)
    assert state.taught is not None and state.taught.tolist() == taught.tolist()
    assert len(state.free.activation) == 8 and len(state.nudged.activation) == 8
    assert report["attempted_presentations"] == 16.0
    assert report["accepted_presentations"] == 8.0 and report["agreed_presentations"] == 8.0
    assert report["nudged_row_sweeps"] == 8 * report["nudged_steps"]
    # the update is the one the disagreeing rows teach on their own
    every.step(drive[taught], labels[taught])
    np.testing.assert_allclose(selective.brain.efficacy, every.brain.efficacy, atol=1e-6)
    np.testing.assert_allclose(selective.brain.bias, every.brain.bias, atol=1e-6)


def test_surprise_leaves_an_agreeing_batch_untaught() -> None:
    learner = _learner(True)
    drive = _blob_drive(learner)
    labels = learner.predict(drive)
    efficacy, bias = learner.brain.efficacy.copy(), learner.brain.bias.copy()
    state, report = learner.step(drive, labels)
    assert state.taught is not None and not state.taught.any()
    assert state.nudged is state.free and state.opposite is None
    assert report["accepted"] == 1.0 and report["accepted_presentations"] == 0.0
    assert report["agreed_presentations"] == 16.0 and report["attempted_presentations"] == 16.0
    assert report["nudged_steps"] == 0.0 and report["opposite_row_sweeps"] == 0.0
    assert report["free_budget_exhausted"] == 0.0 and report["free_steps"] > 0
    np.testing.assert_array_equal(learner.brain.efficacy, efficacy)
    np.testing.assert_array_equal(learner.brain.bias, bias)
    assert learner.contrast_updates == 0


def test_every_row_lesson_marks_no_rows_and_counts_none_agreed() -> None:
    learner = _learner(False)
    drive = _blob_drive(learner)
    labels = learner.predict(drive)
    state, report = learner.step(drive, labels)
    assert state.taught is None
    assert report["accepted_presentations"] == 16.0 and report["agreed_presentations"] == 0.0
    assert learner.contrast_updates == 1


def test_surprise_with_the_callers_answer_settles_only_the_differing_rows() -> None:
    selective, every = _learner(True), _learner(False)
    drive = _blob_drive(every)
    labels = every.predict(drive)
    answer = labels.copy()
    answer[:4] = 1 - answer[:4]  # the caller read a different answer on four rows
    state, report = selective.step(drive, labels, answer=answer)
    assert state.taught is not None and state.taught.tolist() == [True] * 4 + [False] * 12
    assert len(state.free.activation) == 4 and len(state.nudged.activation) == 4
    assert report["attempted_presentations"] == 16.0
    assert report["accepted_presentations"] == 4.0 and report["agreed_presentations"] == 12.0
    assert report["free_row_sweeps"] == 4 * report["free_steps"]
    every.step(drive[:4], labels[:4])
    np.testing.assert_allclose(selective.brain.efficacy, every.brain.efficacy, atol=1e-6)
    np.testing.assert_allclose(selective.brain.bias, every.brain.bias, atol=1e-6)


def test_surprise_with_an_agreeing_answer_runs_no_phase() -> None:
    learner = _learner(True)
    drive = _blob_drive(learner)
    labels = 1 - learner.predict(drive)  # the free phase would disagree on every row
    efficacy = learner.brain.efficacy.copy()
    state, report = learner.step(drive, labels, answer=labels)
    assert state.free is None and state.nudged is None and state.opposite is None
    assert state.taught is not None and not state.taught.any()
    assert report["accepted"] == 1.0 and report["accepted_presentations"] == 0.0
    assert report["agreed_presentations"] == 16.0 and report["attempted_presentations"] == 16.0
    assert report["free_steps"] == 0.0 and report["total_row_sweeps"] == 0.0
    np.testing.assert_array_equal(learner.brain.efficacy, efficacy)
    assert learner.contrast_updates == 0
    with pytest.raises(ValueError):
        learner.step(drive, labels, answer=labels[:8])
