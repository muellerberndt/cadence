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
# an outcome's one read where the critic rests on the device: its eleven checks and numbers, then
# the TD error and the dopamine of every row
OUTCOME_READ = 11


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
    host = cd.Brain.compose(12, 5, modules=(16,), seed=seed, backend="cpu", learning_fast_settle=False)
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
def test_the_streams_eligibility_reads_alike_in_parts(monkeypatch, entries):
    """A large batch makes each stream's contrast in parts: the same life within rounding."""
    from cadence import brain as graph_module

    whole = brain_on_torch()
    plain = _a_life_of_rewards(whole)
    monkeypatch.setattr(graph_module, "_CONTRAST_ENTRIES", entries)
    parted = brain_on_torch()
    for ours, theirs in zip(plain, _a_life_of_rewards(parted), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_allclose(parted.brain.efficacy, whole.brain.efficacy, rtol=0, atol=1e-12)
    for ours, theirs in zip(whole.basal_ganglia._trace_device, parted.basal_ganglia._trace_device,
                            strict=True):
        np.testing.assert_allclose(theirs.numpy(), ours.numpy(), rtol=0, atol=1e-12)


def test_a_refused_outcome_leaves_the_streams_eligibility_as_it_was(monkeypatch):
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


def _twins(monkeypatch, device: str = "cpu", **options) -> tuple[cd.Brain, cd.Brain]:
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    monkeypatch.setenv("CADENCE_MLX_MEMORY", "1")
    options.setdefault("modules", (16,))
    options.setdefault("learning_fast_settle", False)  # the compiled CPU path's other solve
    return (
        cd.Brain.compose(12, 5, seed=4, backend="cpu", **options),
        cd.Brain.compose(12, 5, seed=4, **_placed_on(device), **options),
    )


def _placed_on(device: str) -> dict[str, str]:
    if device == "mlx":
        return {"backend": "mlx"}
    return {"backend": "torch", "device": device, "precision": "float64"}


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
    grown = host.grow(history_decays=(0.0, 0.5)), placed.grow(history_decays=(0.0, 0.5))
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
    host = cd.Brain.compose(12, 5, seed=2, modules=(16,), backend="cpu", learning_fast_settle=False)
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


def test_the_mean_step_of_a_dopamine_is_the_weighted_contrast():
    """The synapses' step of a dopamine reads a weighted Gram contrast as the mean over each row's
    own contrast reads it, whichever the row's sign."""
    brain = brain_on_torch()
    kernel = brain.brain._torch
    torch.manual_seed(2)
    plus = torch.rand((5, kernel.n), dtype=torch.float64)
    minus = plus + 1e-3 * torch.randn((5, kernel.n), dtype=torch.float64)
    d = torch.tensor([0.5, -1.0, 2.0, 0.0, -0.25], dtype=torch.float64)
    weighted = kernel.contrast_edges(plus, minus, d)
    rows = torch.cat([part for _, _, part in kernel.contrast_parts(plus, minus)])
    np.testing.assert_allclose(weighted.numpy(), torch.mv(rows.T, d).numpy(), rtol=1e-12, atol=1e-15)
    plain = kernel.contrast_edges(plus, minus)
    assert torch.equal(kernel.contrast_edges(plus, minus, torch.ones_like(d)), plain)


@pytest.mark.parametrize(
    "graph",
    [
        lambda: cd.Brain.compose(12, 5, modules=(16,), seed=3, backend="torch", device="cpu").brain,
        lambda: cd.Brain.compose(
            12, 5, modules=(16, 8), apical=0.25, teaching="deep", seed=3, backend="torch",
            device="cpu",
        ).brain,
        lambda: cd.NeuralGraph(
            cd.layered(6, 24, 3, density=0.4, seed=4), cd.learning_neuron_model(), backend="torch",
            device="cpu",
        ),
    ],
    ids=["default", "apical", "sparse"],
)
@pytest.mark.parametrize("entries", [1, 1 << 28])
@pytest.mark.parametrize("salient", [False, True])
def test_a_contrast_added_by_rectangles_has_the_bits_of_one_read_by_synapse(
    monkeypatch, graph, entries, salient
):
    from cadence import brain as graph_module

    monkeypatch.setattr(graph_module, "_CONTRAST_ENTRIES", entries)
    kernel = graph()._torch
    connectome = kernel._connectome
    torch.manual_seed(1)
    plus = torch.rand((3, kernel.n), dtype=torch.float64)
    minus = plus + 1e-3 * torch.randn((3, kernel.n), dtype=torch.float64)
    salience = torch.rand((3, kernel.n), dtype=torch.float64) if salient else None
    pre = torch.from_numpy(connectome.pre.copy())
    rows = torch.cat([part for _, _, part in kernel.contrast_parts(plus, minus)])
    if salient:
        rows = rows * salience[:, pre]
    rows = rows / 0.5
    start = torch.rand(rows.shape, dtype=torch.float64)
    added = start.clone()
    kernel.add_contrast(added, plus, minus, 0.5, salience)
    assert torch.equal(added, start + rows)
    if len(connectome.pre) > 100:
        assert kernel._rectangles() is not None or (np.diff(connectome.post) < 0).any()


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
        patch.setattr(brain.learner, "_device_candidate", refused)  # the update's, refused
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


@pytest.mark.parametrize("pending", [False, True], ids=["after an outcome", "with an action out"])
def test_a_large_eligibility_goes_to_a_checkpoint_as_the_actor_keeps_it(
    monkeypatch, tmp_path, pending
):
    """Past a size a checkpoint holds the read's eligibility without the columns and rows that hold
    nothing, and it continues as the full layout does: on the host to the bit."""
    from cadence import stream

    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, backend="cpu")
    rng = np.random.default_rng(0)
    x = np.eye(12)[rng.integers(12, size=2)]
    brain.step(x)
    for _ in range(4):
        x = np.eye(12)[rng.integers(12, size=2)]
        brain.step(x, reward=rng.random(2))
    if not pending:
        brain.learn(rng.random(2), np.zeros(2, dtype=bool), x)
    full = brain.save(tmp_path / "full.npz")
    monkeypatch.setattr(stream, "_SAVED_IN_FULL", 0)
    small = brain.save(tmp_path / "small.npz")
    with np.load(full) as whole, np.load(small) as kept:
        assert "memories/recall/trace" in whole.files and "memories/recall/trace" not in kept.files
        assert kept["memories/recall/trace_kept"].size < whole["memories/recall/trace"].size
        assert ("memories/recall/pending_kept" in kept.files) == pending
        assert "memories/recall/pending" not in kept.files
    one, other = cd.Brain.load(full, backend="cpu"), cd.Brain.load(small, backend="cpu")
    for name in ("trace_memory", "_pending_memory"):
        ours, theirs = getattr(one.basal_ganglia, name), getattr(other.basal_ganglia, name)
        if ours is None:
            assert theirs is None
        else:
            np.testing.assert_array_equal(ours, theirs)
    assert one.recall._address_held == other.recall._address_held
    for _ in range(4):
        x = np.eye(12)[rng.integers(12, size=2)]
        kwargs = {"reward": rng.random(2)} if one.basal_ganglia._pending is not None else {}
        np.testing.assert_array_equal(one.step(x, **kwargs), other.step(x, **kwargs))
    np.testing.assert_array_equal(one.brain.efficacy, other.brain.efficacy)
    np.testing.assert_array_equal(one.basal_ganglia.trace_memory, other.basal_ganglia.trace_memory)


def test_a_float32_brains_eligibility_goes_to_a_checkpoint_in_float32(tmp_path):
    """Half the file and no second array at a load: 256 streams at the speech size hold 6 GB in
    float32; on the host it continues in float64."""
    brain = cd.Brain.compose(
        12, 5, modules=(16,), seed=3, backend="torch", device="cpu", precision="float32"
    )
    _a_life_of_rewards(brain, moments=6)
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path) as held:
        assert held["actor/trace"].dtype == np.float32
    again = cd.Brain.load(path, backend="torch", device="cpu", precision="float32")
    assert again.basal_ganglia.trace.dtype == np.float32
    on_host = cd.Brain.load(path, backend="cpu")
    assert on_host.basal_ganglia.trace.dtype == np.float64
    np.testing.assert_array_equal(on_host.basal_ganglia.trace, again.basal_ganglia.trace)
    x = np.eye(12)[[1, 7, 3]]
    for one in (again, on_host):
        for _ in range(3):  # the saved action's outcome on the host's phases, then on the device
            one.step(x, reward=np.ones(3))
    assert again.basal_ganglia._trace_device[0].dtype == torch.float32
    assert on_host.basal_ganglia.trace.dtype == np.float64


def test_a_retune_keeps_the_eligibility_without_its_full_layout(monkeypatch):
    """A change of what can step keeps every column held, and reads no array of the full layout."""
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, backend="cpu")
    rng = np.random.default_rng(0)
    brain.step(np.eye(12)[rng.integers(12, size=2)])
    for _ in range(3):
        brain.step(np.eye(12)[rng.integers(12, size=2)], reward=rng.random(2))
    brain.learn(rng.random(2), np.zeros(2, dtype=bool), np.eye(12)[rng.integers(12, size=2)])
    recall, agent = brain.recall, brain.basal_ganglia
    before = recall.full_eligibility(agent.trace_memory)
    width = agent.trace_memory.shape[-1]
    monkeypatch.setattr(type(recall), "kept_eligibility", None)
    brain.retune(recall_amplitude=2.5)
    assert agent.trace_memory.shape[-1] == width
    brain.retune(recall_address_reward=0.5)
    assert agent.trace_memory.shape[-1] == width + recall.address_width
    monkeypatch.undo()
    np.testing.assert_array_equal(recall.full_eligibility(agent.trace_memory), before)


def test_a_kept_eligibility_of_another_width_is_refused(monkeypatch, tmp_path):
    from cadence import stream

    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, backend="cpu")
    x = np.eye(12)[[1, 7]]
    brain.step(x)
    brain.step(x, reward=np.ones(2))
    monkeypatch.setattr(stream, "_SAVED_IN_FULL", 0)
    path = brain.save(tmp_path / "small.npz", compressed=False)
    with np.load(path) as held:
        arrays = {name: held[name] for name in held.files}
    arrays["memories/recall/trace_kept"] = arrays["memories/recall/trace_kept"][:, :, :-1]
    np.savez(tmp_path / "wrong.npz", **arrays)
    with pytest.raises(ValueError, match="trace_kept"):
        cd.Brain.load(tmp_path / "wrong.npz", backend="cpu")


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
    assert reads and max(reads) <= max(rows * motor, OUTCOME_READ + 2 * rows)


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
    assert reads and max(reads) <= max(motor, OUTCOME_READ + 2)


@pytest.mark.parametrize("device", ["cpu", "torch"])
def test_grow_and_graft_leave_a_living_brain_as_it_was(device):
    """A grown or grafted brain starts its streams afresh; the brain it came from keeps them."""
    place = {"backend": "cpu"} if device == "cpu" else {"backend": "torch", "device": "cpu"}
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, **place)
    _a_life_of_rewards(brain, moments=5)
    x = np.eye(12)[[1, 7, 3]]
    brain.act(x)
    before = copy.deepcopy(brain)
    grown = brain.grow(history_decays=(0.0, 0.5))
    assert grown.basal_ganglia._pending is None
    twin = cd.Brain.compose(12, 5, modules=(16,), seed=4, **place)
    grafted = twin.graft(brain, regions=("association", "motor"), joints="kept")
    assert grafted.basal_ganglia._pending is None
    reward = np.ones(3)
    np.testing.assert_array_equal(
        brain.step(x, reward=reward), before.step(x, reward=reward)
    )
    np.testing.assert_array_equal(brain.brain.efficacy, before.brain.efficacy)


@pytest.mark.parametrize("device", ["cpu", "torch"])
def test_a_brain_grown_while_it_streams_is_the_brain_grown_after_a_reset(device, tmp_path):
    place = {"backend": "cpu"} if device == "cpu" else {"backend": "torch", "device": "cpu"}
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, **place)
    _a_life_of_rewards(brain, moments=5)
    brain.act(np.eye(12)[[1, 7, 3]])
    streaming = brain.grow(history_decays=(0.0, 0.5))
    twin = copy.deepcopy(brain)
    twin.reset()
    afresh = twin.grow(history_decays=(0.0, 0.5))
    ours, theirs = streaming.save(tmp_path / "a.npz"), afresh.save(tmp_path / "b.npz")
    with np.load(ours) as a, np.load(theirs) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


# -- an outcome's reads (cadence#274: the critic and the dopamine on the device)


def _reads_of(monkeypatch, call) -> list[str]:
    """Every read of a torch tensor's numbers on the host that ``call`` makes, by kind."""
    reads: list[str] = []
    for name in ("cpu", "item", "tolist", "__bool__", "__float__", "__int__"):
        original = getattr(torch.Tensor, name)

        def counted(self, *args, _original=original, _name=name, **kwargs):
            reads.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, name, counted)
    call()
    monkeypatch.undo()
    return reads


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_an_outcome_reads_its_checks_and_dopamine_once(monkeypatch, precision):
    """Where the critic rests on the device the dopamine is made there: the outcome's checks (the
    dopamine's, the critic's and the update's), the critic's bias step, the report's steps and the
    dopamine itself come to the host in one read before anything is installed."""
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    brain = cd.Brain.compose(12, 5, modules=(16,), seed=3, backend="torch", device="cpu",
                             precision=precision, recall=False)
    _a_life_of_rewards(brain, moments=5)
    x = np.eye(12)[[1, 7, 3]]
    brain.act(x)
    assert brain.basal_ganglia._critic_device is not None
    with monkeypatch.context() as patch:
        patch.setenv("CADENCE_TORCH_MEMORY", "1")
        reads = _reads_of(
            patch, lambda: brain.learn(np.ones(3), np.array([False, True, False]), x)
        )
    # the next state's settle (its stop, one sweep behind, and its residual check), the store's
    # write, the one read of the outcome and the report's trace
    settle = reads.count("item") + 1
    assert len(reads) - settle <= 4, reads


def test_a_refused_dopamine_leaves_the_outcome_pending(monkeypatch):
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    brain = brain_on_torch()
    _a_life_of_rewards(brain, moments=5)
    x = np.eye(12)[[1, 7, 3]]
    brain.act(x)
    agent = brain.basal_ganglia
    held = agent._critic_device
    weights, scale = held.weights.clone(), brain.brain._torch.scale.clone()
    updates = brain.learner.updates

    def unbounded(state):
        return torch.full((3,), float("inf"), dtype=held.weights.dtype)

    with monkeypatch.context() as patch:
        patch.setattr(held, "value_here", unbounded)
        with pytest.raises(ValueError, match="TD errors must remain finite"):
            brain.learn(np.ones(3), np.zeros(3, dtype=bool), x)
    assert agent._pending is not None and brain.learner.updates == updates
    assert torch.equal(held.weights, weights) and torch.equal(brain.brain._torch.scale, scale)
    brain.learn(np.ones(3), np.zeros(3, dtype=bool), x)  # the same outcome, accepted
    assert brain.learner.updates == updates + 1
