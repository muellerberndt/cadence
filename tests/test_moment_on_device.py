"""A moment on a torch device crosses less (cadence#274): its drive is uploaded once for every
settle and check, a settle from rest uploads no zeros, the motor and critic reads bring their columns
and not the state, and a copy of a brain has a kernel of its own."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")


def brain_on_torch(seed: int = 3) -> cd.Brain:
    return cd.Brain.compose(12, 5, modules=(16,), seed=seed, backend="torch", device="cpu")


def test_a_moments_drive_reaches_the_device_once(monkeypatch):
    brain = brain_on_torch()
    kernel = brain.brain._torch
    wide = []
    upload = kernel._staging.upload

    def counted(array, dtype):
        if np.ndim(array) == 2 and np.shape(array)[1] == brain.connectome.n:
            wide.append(array)
        return upload(array, dtype)

    monkeypatch.setattr(kernel._staging, "upload", counted)
    x = np.eye(12)[[1, 7]]
    brain.act(x, greedy=True)
    assert len(wide) == 1
    wide.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain.lesson(x, np.array([0, 1]))
    assert len(wide) == 1  # the free phase starts from rest: no upload of its zeros either


def test_the_public_drive_is_a_host_array():
    brain = brain_on_torch()
    assert isinstance(brain.stimulus(np.eye(12)[[1, 7]]), np.ndarray)


def test_the_choice_and_the_critic_read_their_columns_of_a_resting_state():
    brain = brain_on_torch()
    x = np.eye(12)[[1, 7]]
    action = brain.act(x)
    state = brain.basal_ganglia.state
    assert state.__dict__.get("activation") is None and state.__dict__.get("v") is None
    motor = state.part(brain.motor_index)
    assert motor.shape == (2, 5)
    assert state.__dict__.get("activation") is None  # still no whole state on the host
    assert state.part(brain.motor_index) is motor  # read once
    np.testing.assert_array_equal(motor, state.activation[:, brain.motor_index])
    np.testing.assert_array_equal(action, np.asarray(action))


def test_a_copy_of_a_brain_on_torch_continues_alike_with_a_kernel_of_its_own():
    brain = brain_on_torch()
    x = np.eye(12)[[1, 7]]
    brain.step(x)
    twin = copy.deepcopy(brain)
    assert twin.brain._torch is not brain.brain._torch
    reward = np.array([1.0, 0.0])
    np.testing.assert_array_equal(brain.step(x, reward=reward), twin.step(x, reward=reward))
    np.testing.assert_allclose(twin.brain.efficacy, brain.brain.efficacy, rtol=0, atol=1e-12)
    before = twin.brain.efficacy.copy()
    brain.step(x, reward=reward)
    np.testing.assert_array_equal(twin.brain.efficacy, before)  # the original moved alone


@pytest.mark.parametrize("seed", range(3))
def test_a_life_on_torch_reads_as_on_the_host(seed):
    host = cd.Brain.compose(12, 5, modules=(16,), seed=seed, backend="cpu")
    device = brain_on_torch(seed)
    rng = np.random.default_rng(seed)
    answer = None
    for t in range(24):
        x = np.eye(12)[rng.integers(12, size=3)]
        if t % 3 == 2 and answer is not None:
            labels = (np.asarray(answer).reshape(-1) + 1 + t) % 5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                host.lesson(x, labels)
                device.lesson(x, labels)
        answer = host.act(x, greedy=True)
        np.testing.assert_array_equal(device.act(x, greedy=True), answer)
    np.testing.assert_allclose(device.brain.efficacy, host.brain.efficacy, rtol=0, atol=1e-9)


def _a_life_of_rewards(brain: cd.Brain, seed: int = 5, moments: int = 12) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    actions = []
    for t in range(moments):
        x = np.eye(12)[rng.integers(12, size=3)]
        kwargs = {} if t == 0 else {
            "reward": rng.random(3), "done": rng.random(3) < 0.2,
        }
        actions.append(np.asarray(brain.step(x, **kwargs)))
    return actions


@pytest.mark.parametrize("entries", [1, 1 << 14])
@pytest.mark.parametrize("kept", [0, 1 << 40])
def test_the_streams_eligibility_reads_alike_in_parts_and_made_twice(monkeypatch, entries, kept):
    """A large batch makes each stream's contrast in parts and, past what can wait for the last
    refusal, makes it again to install it: the same life within rounding."""
    from cadence import brain as graph_module
    from cadence import plasticity

    whole = brain_on_torch()
    plain = _a_life_of_rewards(whole)
    monkeypatch.setattr(graph_module, "_CONTRAST_ENTRIES", entries)
    monkeypatch.setattr(plasticity, "_KEPT_CONTRAST", kept)
    parted = brain_on_torch()
    for ours, theirs in zip(plain, _a_life_of_rewards(parted), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_allclose(parted.brain.efficacy, whole.brain.efficacy, rtol=0, atol=1e-12)
    for ours, theirs in zip(whole.basal_ganglia._trace_device, parted.basal_ganglia._trace_device,
                            strict=True):
        np.testing.assert_allclose(theirs.numpy(), ours.numpy(), rtol=0, atol=1e-12)


def test_a_refused_outcome_leaves_the_streams_eligibility_as_it_was(monkeypatch):
    from cadence import plasticity

    monkeypatch.setattr(plasticity, "_KEPT_CONTRAST", 0)
    brain = brain_on_torch()
    _a_life_of_rewards(brain, moments=6)
    x = np.eye(12)[[1, 7, 3]]
    brain.act(x)
    agent = brain.basal_ganglia
    trace, trace_bias = (tensor.clone() for tensor in agent._trace_device)
    critic = agent.trace_critic.copy()
    efficacy = brain.brain.efficacy.copy()

    def refused(*_args, **_kwargs):
        raise ValueError("refused")

    monkeypatch.setattr(brain.learner, "_apply_device", refused)
    with pytest.raises(ValueError, match="refused"):
        brain.learn(np.ones(3), np.zeros(3, dtype=bool), x)
    monkeypatch.undo()
    np.testing.assert_array_equal(agent._trace_device[0].numpy(), trace.numpy())
    np.testing.assert_array_equal(agent._trace_device[1].numpy(), trace_bias.numpy())
    np.testing.assert_array_equal(agent.trace_critic, critic)
    np.testing.assert_array_equal(brain.brain.efficacy, efficacy)


def _twins(monkeypatch, **options) -> tuple[cd.Brain, cd.Brain]:
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    options.setdefault("modules", (16,))
    return (
        cd.Brain.compose(12, 5, seed=4, backend="cpu", **options),
        cd.Brain.compose(12, 5, seed=4, backend="torch", device="cpu", **options),
    )


def _both_live(host: cd.Brain, placed: cd.Brain, rng: np.random.Generator, moments: int = 5) -> None:
    for _ in range(moments):
        x = np.eye(12)[rng.integers(12, size=2)]
        pending = host.basal_ganglia._pending is not None
        kwargs = {"reward": rng.random(2)} if pending else {}
        np.testing.assert_array_equal(host.step(x, **kwargs), placed.step(x, **kwargs))


@pytest.mark.parametrize(
    "options",
    [{}, {"modules": (16, 8), "apical": 0.25, "teaching": "deep"}],
    ids=["default", "apical"],
)
def test_retune_grow_and_graft_read_as_on_the_host(monkeypatch, tmp_path, options):
    """A brain whose memories rest on its device retunes, grows, is grafted and saves as the host's
    does, and each of them continues where the host's does."""
    host, placed = _twins(monkeypatch, **options)
    rng = np.random.default_rng(10)
    _both_live(host, placed, rng)
    assert placed.recall._device is not None
    for brain in (host, placed):
        if brain.basal_ganglia._pending is not None:
            brain.learn(np.zeros(2), np.zeros(2, dtype=bool), np.eye(12)[[0, 1]])
        brain.retune(recall_amplitude=2.5, recall_address_reward=0.5)
    assert placed.basal_ganglia.trace_memory.shape == host.basal_ganglia.trace_memory.shape
    _both_live(host, placed, rng)
    np.testing.assert_allclose(placed.brain.efficacy, host.brain.efficacy, rtol=0, atol=1e-9)
    grown = host.grow(history_decays=(0.5,)), placed.grow(history_decays=(0.5,))
    assert grown[1].brain.backend == "torch"
    _both_live(*grown, rng)
    np.testing.assert_allclose(grown[1].brain.efficacy, grown[0].brain.efficacy, rtol=0, atol=1e-9)
    if "apical" not in options:
        donor = cd.Brain.compose(12, 5, seed=5, modules=(16,))
        regions = ("association", "motor")
        grafted = host.graft(donor, regions), placed.graft(donor, regions)
        _both_live(*grafted, rng)
        np.testing.assert_allclose(
            grafted[1].brain.efficacy, grafted[0].brain.efficacy, rtol=0, atol=1e-9
        )
    path = placed.save(tmp_path / "placed.npz", compressed=False)
    on_cpu = cd.Brain.load(path, backend="cpu")
    again = cd.Brain.load(path, backend="torch", device="cpu")
    _both_live(on_cpu, again, rng)
    assert again.recall._device is not None


def test_a_salience_weighs_the_eligibility_as_on_the_host():
    host = cd.Brain.compose(12, 5, seed=2, modules=(16,), backend="cpu")
    device = brain_on_torch(2)
    rng = np.random.default_rng(11)
    for t in range(10):
        x = np.eye(12)[rng.integers(12, size=3)]
        kwargs = {} if t == 0 else {"reward": rng.random(3)}
        for brain in (host, device):
            brain.basal_ganglia.salience = (
                None if t < 2 else np.ones((3, brain.connectome.n)) * (1.0 + t % 3)
            )
        np.testing.assert_array_equal(host.step(x, **kwargs), device.step(x, **kwargs))
    np.testing.assert_allclose(device.brain.efficacy, host.brain.efficacy, rtol=0, atol=1e-10)
