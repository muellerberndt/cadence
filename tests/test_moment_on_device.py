"""A moment on a torch device crosses less (cadence#274): its drive is uploaded once for every
settle and check, a settle from rest uploads no zeros, the motor and critic reads bring their columns
and not the state, a copy of a brain has a kernel of its own, and where the memories rest on the
device the critic rests with them and nothing wider than the action comes to the host."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")


DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


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
@pytest.mark.parametrize("room", [False, True])
def test_the_streams_eligibility_reads_alike_in_parts_and_made_twice(monkeypatch, entries, room):
    """A large batch makes each stream's contrast in parts and, past what can wait for the last
    refusal, makes it again to install it: the same life within rounding."""
    from cadence import brain as graph_module

    whole = brain_on_torch()
    plain = _a_life_of_rewards(whole)
    monkeypatch.setattr(graph_module, "_CONTRAST_ENTRIES", entries)
    monkeypatch.setattr(graph_module._TorchKernel, "has_room", lambda self, like: room)
    parted = brain_on_torch()
    for ours, theirs in zip(plain, _a_life_of_rewards(parted), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_allclose(parted.brain.efficacy, whole.brain.efficacy, rtol=0, atol=1e-12)
    for ours, theirs in zip(whole.basal_ganglia._trace_device, parted.basal_ganglia._trace_device,
                            strict=True):
        np.testing.assert_allclose(theirs.numpy(), ours.numpy(), rtol=0, atol=1e-12)


def test_a_refused_outcome_leaves_the_streams_eligibility_as_it_was(monkeypatch):
    from cadence import brain as graph_module

    monkeypatch.setattr(graph_module._TorchKernel, "has_room", lambda self, like: False)
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


def _twins(monkeypatch, device: str, **options) -> tuple[cd.Brain, cd.Brain]:
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    monkeypatch.setenv("CADENCE_MLX_MEMORY", "1")
    options.setdefault("modules", (16,))
    return (
        cd.Brain.compose(12, 5, seed=4, backend="cpu", **options),
        cd.Brain.compose(12, 5, seed=4, **_placed_on(device), **options),
    )


def _placed_on(device: str) -> dict[str, str]:
    return {"backend": "mlx"} if device == "mlx" else {"backend": "torch", "device": device}


# the devices a life is read on, and how far a synapse lies from the host's on each
_DEVICES = [
    ("cpu", 1e-9),
    *([("mlx", 3e-2)] if "mlx" in cd.available_backends() else []),
]


def _both_live(host: cd.Brain, placed: cd.Brain, rng: np.random.Generator, moments: int = 5) -> None:
    for _ in range(moments):
        x = np.eye(12)[rng.integers(12, size=2)]
        pending = host.basal_ganglia._pending is not None
        kwargs = {"reward": rng.random(2)} if pending else {}
        np.testing.assert_array_equal(host.step(x, **kwargs), placed.step(x, **kwargs))


@pytest.mark.parametrize(("device", "atol"), _DEVICES, ids=[row[0] for row in _DEVICES])
@pytest.mark.parametrize(
    "options",
    [{}, {"modules": (16, 8), "apical": 0.25, "teaching": "deep"}],
    ids=["default", "apical"],
)
def test_retune_grow_and_graft_read_as_on_the_host(monkeypatch, tmp_path, options, device, atol):
    """A brain whose memories rest on its device retunes, grows, is grafted and saves as the host's
    does, and each of them continues where the host's does."""
    host, placed = _twins(monkeypatch, device, **options)
    rng = np.random.default_rng(10)
    _both_live(host, placed, rng)
    assert placed.recall._device is not None
    for brain in (host, placed):
        if brain.basal_ganglia._pending is not None:
            brain.learn(np.zeros(2), np.zeros(2, dtype=bool), np.eye(12)[[0, 1]])
        brain.retune(recall_amplitude=2.5, recall_address_reward=0.5)
    assert placed.basal_ganglia.trace_memory.shape == host.basal_ganglia.trace_memory.shape
    _both_live(host, placed, rng)
    np.testing.assert_allclose(placed.brain.efficacy, host.brain.efficacy, rtol=0, atol=atol)
    grown = host.grow(history_decays=(0.5,)), placed.grow(history_decays=(0.5,))
    assert grown[1].brain.backend == placed.brain.backend
    _both_live(*grown, rng)
    np.testing.assert_allclose(grown[1].brain.efficacy, grown[0].brain.efficacy, rtol=0, atol=atol)
    if "apical" not in options:
        donor = cd.Brain.compose(12, 5, seed=5, modules=(16,))
        regions = ("association", "motor")
        grafted = host.graft(donor, regions), placed.graft(donor, regions)
        _both_live(*grafted, rng)
        np.testing.assert_allclose(
            grafted[1].brain.efficacy, grafted[0].brain.efficacy, rtol=0, atol=atol
        )
    path = placed.save(tmp_path / "placed.npz", compressed=False)
    on_cpu = cd.Brain.load(path, backend="cpu")
    again = cd.Brain.load(path, **_placed_on(device))
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


# -- the critic rests with the memories


@pytest.fixture
def lived_twins(monkeypatch) -> tuple[cd.Brain, cd.Brain, np.random.Generator]:
    host, placed = _twins(monkeypatch)
    rng = np.random.default_rng(6)
    _both_live(host, placed, rng, moments=7)
    return host, placed, rng


def _critics_agree(host: cd.Brain, placed: cd.Brain) -> None:
    ours, theirs = host.basal_ganglia, placed.basal_ganglia
    np.testing.assert_allclose(theirs.w_critic, ours.w_critic, rtol=0, atol=1e-11)
    if ours.trace_critic is None:
        assert theirs.trace_critic is None
    else:
        np.testing.assert_allclose(theirs.trace_critic, ours.trace_critic, rtol=0, atol=1e-11)
    assert theirs.b_critic == pytest.approx(ours.b_critic, abs=1e-11)


def test_the_critic_rests_on_the_device_and_comes_back_when_the_host_reads_it(lived_twins):
    host, placed, rng = lived_twins
    agent = placed.basal_ganglia
    assert agent._critic_device is not None
    assert "w_critic" not in agent.__dict__ and "trace_critic" not in agent.__dict__
    assert agent.parameters() == host.basal_ganglia.parameters()  # without a read of the weights
    assert agent._critic_device is not None
    _critics_agree(host, placed)
    assert agent._critic_device is None
    assert isinstance(agent.w_critic, np.ndarray) and isinstance(agent.trace_critic, np.ndarray)
    _both_live(host, placed, rng)
    assert agent._critic_device is not None
    _critics_agree(host, placed)


def test_a_write_on_the_host_reaches_the_resting_critics_next_read(lived_twins):
    host, placed, rng = lived_twins
    for brain in (host, placed):
        brain.basal_ganglia.w_critic = brain.basal_ganglia.w_critic * 0.5  # the other array stays
        brain.basal_ganglia.b_critic = 0.25
    _both_live(host, placed, rng)
    assert placed.basal_ganglia._critic_device is not None
    _critics_agree(host, placed)
    for brain in (host, placed):
        brain.basal_ganglia.trace_critic[:] = 0.0  # a read and an edit of what it brought back
    _both_live(host, placed, rng)
    _critics_agree(host, placed)


def test_a_fade_and_a_reset_reach_the_resting_critic(lived_twins):
    host, placed, rng = lived_twins
    agent = placed.basal_ganglia
    ended = np.array([True, False])
    for brain in (host, placed):
        brain.basal_ganglia.fade(ended)
    assert agent._critic_device is not None
    with pytest.raises(ValueError, match="must match"):
        agent.fade(np.array([True, False, False]))
    assert not agent._critic_device.trace[0].any()
    _both_live(host, placed, rng, moments=2)
    _critics_agree(host, placed)
    _both_live(host, placed, rng, moments=2)
    for brain in (host, placed):
        brain.basal_ganglia.reset()
    assert agent._critic_device is not None and agent._critic_device.trace is None
    assert agent.trace_critic is None and agent._critic_device is None


def test_a_refused_outcome_leaves_the_resting_critic_as_it_was(monkeypatch):
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    brain = brain_on_torch()
    _a_life_of_rewards(brain, moments=6)
    x = np.eye(12)[[1, 7, 3]]
    brain.act(x)
    agent = brain.basal_ganglia
    held = agent._critic_device
    assert held is not None
    weights, trace, bias = held.weights.clone(), held.trace.clone(), agent.b_critic

    def refused(*_args, **_kwargs):
        raise ValueError("refused")

    with monkeypatch.context() as patch:
        patch.setattr(brain.learner, "_apply_device", refused)
        with pytest.raises(ValueError, match="refused"):
            brain.learn(np.ones(3), np.zeros(3, dtype=bool), x)
    assert agent._critic_device is held and "trace_critic" not in agent.__dict__
    assert torch.equal(held.weights, weights) and torch.equal(held.trace, trace)
    assert agent.b_critic == bias


def test_a_refused_critic_step_is_refused_on_the_device(monkeypatch):
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    brain = brain_on_torch()
    _a_life_of_rewards(brain, moments=4)
    x = np.eye(12)[[1, 7, 3]]
    brain.act(x)
    agent = brain.basal_ganglia
    agent._critic_device.weights[0] = float("inf")
    with pytest.raises(ValueError, match="critic parameters must remain finite"):
        agent._critic_device.candidate(np.ones(3))


# -- what a moment brings to the host


@pytest.mark.parametrize(
    "options",
    [{}, {"modules": (16, 8), "apical": 0.25, "teaching": "deep"}],
    ids=["default", "apical"],
)
@pytest.mark.parametrize("device", DEVICES)
def test_a_moment_brings_nothing_wider_than_its_action_to_the_host(monkeypatch, options, device):
    """The memories, the eligibility and the optimizer rest on the device: of a moment only the
    motor columns, a number per row and the checks come to the host."""
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    options = {"modules": (16,), **options}
    rows, motor = 3, 5
    brain = cd.Brain.compose(12, motor, seed=3, backend="torch", device=device, **options)
    rng = np.random.default_rng(0)

    def observed() -> np.ndarray:
        return np.eye(12)[rng.integers(12, size=rows)]

    reward, ended = np.ones(rows), np.zeros(rows, dtype=bool)
    for _ in range(4):  # the slots, the eligibility and the optimizer's state exist from here
        brain.act(observed())
        brain.learn(reward, ended, observed())
    reads: list[int] = []
    bring = torch.Tensor.numpy

    def counted(self, *args, **kwargs):
        reads.append(int(self.numel()))
        return bring(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "numpy", counted)
    x = observed()
    brain.act(x)
    brain.learn(reward, ended, observed())
    brain.act(x, greedy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain.lesson(x, np.arange(rows) % motor)
    brain.step(x)
    brain.step(observed(), reward=reward)
    assert reads and max(reads) <= rows * motor


@pytest.mark.parametrize("device", DEVICES)
def test_a_lived_moment_brings_nothing_wider_than_its_action_to_the_host(monkeypatch, device):
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    motor = 5
    brain = cd.Brain.compose(
        12, motor, seed=3, backend="torch", device=device, modules=(16,), arousal=True
    )
    rng = np.random.default_rng(0)
    for t in range(6):
        brain.live(np.eye(12)[[rng.integers(12)]], **({"reward": np.ones(1)} if t else {}))
    reads: list[int] = []
    bring = torch.Tensor.numpy

    def counted(self, *args, **kwargs):
        reads.append(int(self.numel()))
        return bring(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "numpy", counted)
    for t in range(12):
        x = np.eye(12)[[rng.integers(12)]]
        if t % 5 == 4 and brain.pending_feedback:
            brain.wait(x)
        else:
            brain.live(x, reward=np.array([float(t % 3 == 0)]))
    assert reads and max(reads) <= motor
