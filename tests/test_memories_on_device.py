"""A brain's memories on its device (cadence#274): the recall memory, the working trace and the
store rest there, the moment's drive with them, and a life reads as on the float64 host path within
rounding, moment by moment. ``CADENCE_TORCH_MEMORY=1`` and ``CADENCE_MLX_MEMORY=1`` place them at any
size, on a torch CPU device, on CUDA where there is one and on MLX where it is installed."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import cadence as cd
from cadence import resident as stream

torch = pytest.importorskip("torch")

DEVICES = [
    "cpu", *(["cuda"] if torch.cuda.is_available() else []),
    *(["mlx"] if "mlx" in cd.available_backends() else []),
]
TIGHT = 1e-9
# float32 on MLX: a read within rounding, a synapse's normalized step within the backend's bound
ROUNDED = 3e-2


def within(device: str) -> float:
    return ROUNDED if device == "mlx" else TIGHT


@pytest.fixture(autouse=True)
def _placed(monkeypatch):
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    monkeypatch.setenv("CADENCE_MLX_MEMORY", "1")


def on_device(device: str) -> dict[str, str]:
    return {"backend": "mlx"} if device == "mlx" else {"backend": "torch", "device": device}


def pair(
    device: str, inputs: int = 12, actions: int = 5, seed: int = 3, **options
) -> tuple[cd.Brain, cd.Brain]:
    options.setdefault("modules", (16,))
    return (
        cd.Brain.compose(inputs, actions, seed=seed, backend="cpu", **options),
        cd.Brain.compose(inputs, actions, seed=seed, **on_device(device), **options),
    )


def arrays(brain: cd.Brain) -> dict[str, np.ndarray]:
    """What a life changes, on the host: reading it brings every memory back."""
    recall, agent = brain.recall, brain.basal_ganglia
    out = {"efficacy": brain.brain.efficacy, "bias": brain.brain.bias,
           "critic": agent.w_critic, "critic_bias": np.array([agent.b_critic])}
    if recall is not None:
        for name in ("keys", "values", "nexts", "held", "previous", "told", "focus", "trace"):
            out["recall/" + name] = np.asarray(getattr(recall, name), dtype=float)
        for kind in ("head", "position", "anchor", "relative", "slope", "grid"):
            for k, maps in enumerate(getattr(recall, kind + "_maps")):
                for i, array in enumerate(maps):
                    out[f"recall/{kind}{k}/{i}"] = array
        if agent.trace_memory is not None:
            out["read eligibility"] = np.asarray(agent.trace_memory)
    if brain.working_memory is not None:
        out["trace"] = brain.working_memory.trace
    if brain.hippocampus is not None:
        out["strength"] = brain.hippocampus.strength
        out["consolidated"] = brain.hippocampus.consolidated
        out["mass"] = brain.hippocampus.mass
    if brain.hearing is not None:
        out["gains"] = np.array(list(brain.hearing.gains().values()))
    return out


def same(host: cd.Brain, device: cd.Brain, atol: float = TIGHT) -> None:
    ours, theirs = arrays(host), arrays(device)
    assert set(ours) == set(theirs)
    for name, value in ours.items():
        # what a stream wrote, at float32's rounding of what it heard, whatever its weights
        written = name in ("recall/keys", "recall/values", "recall/nexts", "recall/held",
                           "recall/told", "recall/focus")
        np.testing.assert_allclose(
            theirs[name], value, rtol=0, atol=min(atol, 1e-6) if written else atol, err_msg=name
        )


def resident(brain: cd.Brain) -> bool:
    return all(
        memory._device is not None
        for memory in (brain.recall, brain.working_memory, brain.hippocampus)
    )


def observations(rng: np.random.Generator, rows: int, inputs: int = 12) -> np.ndarray:
    return np.eye(inputs)[rng.integers(inputs, size=rows)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows", [1, 3])
def test_a_life_of_acts_and_lessons_reads_as_on_the_host(device, rows):
    host, placed = pair(device)
    rng = np.random.default_rng(0)
    answer = None
    for t in range(40):
        x = observations(rng, rows)
        if answer is not None and t % 3:
            labels = (np.asarray(answer).reshape(-1) + 1 + t) % 5
            reports = host.lesson(x, labels), placed.lesson(x, labels)
            assert resident(placed)
            assert set(reports[0]) == set(reports[1])
            for name, value in reports[0].items():
                assert reports[1][name] == pytest.approx(
                    value, rel=within(device) or 1e-9, abs=within(device)
                ), name
        answer = host.act(x, greedy=True)
        np.testing.assert_array_equal(placed.act(x, greedy=True), answer)
    assert resident(placed)
    same(host, placed, within(device))


@pytest.mark.parametrize("device", DEVICES)
def test_a_life_of_rewards_with_ended_streams_reads_as_on_the_host(device):
    host, placed = pair(device)
    rng = np.random.default_rng(1)
    for t in range(40):
        x = observations(rng, 3)
        if t == 0:
            first = host.step(x), placed.step(x)
        else:
            reward, done = rng.random(3), rng.random(3) < 0.2
            first = host.step(x, reward=reward, done=done), placed.step(x, reward=reward, done=done)
            for name, value in host.last_learning.items():
                assert placed.last_learning[name] == pytest.approx(
                    value, rel=within(device), abs=within(device)
                ), name
        np.testing.assert_array_equal(*first)
    assert resident(placed)
    assert not isinstance(placed.basal_ganglia.trace_memory, np.ndarray)
    same(host, placed, within(device))


@pytest.mark.parametrize("device", DEVICES)
def test_live_and_wait_read_as_on_the_host(device):
    host, placed = pair(device, arousal=True)
    rng = np.random.default_rng(2)
    for t in range(60):
        x = observations(rng, 1)
        if t % 11 == 5 and host.pending_feedback:
            host.wait(x)
            placed.wait(x)
            continue
        kwargs = {} if t == 0 else {"reward": np.array([float(t % 7 == 0)]),
                                    "done": np.array([t % 13 == 0])}
        np.testing.assert_array_equal(host.live(x, **kwargs), placed.live(x, **kwargs))
        assert host.last_arousal["mode"] == placed.last_arousal["mode"]
        assert placed.last_arousal["level"] == pytest.approx(
            host.last_arousal["level"], abs=within(device)
        )
    same(host, placed, atol=within(device) * 10)


@pytest.mark.parametrize("device", DEVICES)
def test_the_memories_rest_there_and_come_back_when_the_host_reads_them(device):
    _, placed = pair(device)
    rng = np.random.default_rng(3)
    for _ in range(4):
        placed.act(observations(rng, 2), greedy=True)
    recall = placed.recall
    assert resident(placed) and "keys" not in recall.__dict__
    assert "strength" not in placed.hippocampus.__dict__
    assert recall.streams == 2 and recall.slots == 5  # without a read of the slots
    assert "keys" not in recall.__dict__
    assert isinstance(recall.keys, np.ndarray) and recall.keys.shape == (2, 5, 12)
    assert recall._device is None
    placed.act(observations(rng, 2), greedy=True)
    assert recall._device is not None and recall.slots == 6
    assert isinstance(placed.working_memory.trace, np.ndarray)
    assert isinstance(placed.hippocampus.strength, np.ndarray)
    placed.act(observations(rng, 2), greedy=True)
    assert resident(placed)
    assert isinstance(placed.stimulus(observations(rng, 2)), np.ndarray)  # the public drive


@pytest.mark.parametrize("device", DEVICES)
def test_a_write_on_the_host_reaches_the_next_read(device):
    host, placed = pair(device)
    rng = np.random.default_rng(4)
    for _ in range(4):
        x = observations(rng, 2)
        host.act(x, greedy=True)
        placed.act(x, greedy=True)
    for brain in (host, placed):
        brain.recall.keys = brain.recall.keys * 0.5  # an expert's edit
    x = observations(rng, 2)
    np.testing.assert_array_equal(host.act(x, greedy=True), placed.act(x, greedy=True))
    same(host, placed, within(device))


@pytest.mark.parametrize("device", DEVICES)
def test_a_checkpoint_continues_within_rounding(device, tmp_path):
    host, placed = pair(device)
    rng = np.random.default_rng(5)
    placed.step(observations(rng, 2))
    for _ in range(12):
        placed.step(observations(rng, 2), reward=rng.random(2))
    path = placed.save(tmp_path / "placed.npz", compressed=False)
    on_cpu = cd.Brain.load(path, backend="cpu")
    again = cd.Brain.load(path, **on_device(device))
    assert not resident(again)
    for _ in range(6):
        x, reward = observations(rng, 2), rng.random(2)
        np.testing.assert_array_equal(on_cpu.step(x, reward=reward), again.step(x, reward=reward))
    assert resident(again)
    same(on_cpu, again, within(device))
    np.testing.assert_allclose(
        on_cpu.recall.full_eligibility(on_cpu.basal_ganglia.trace_memory),
        again.recall.full_eligibility(again.basal_ganglia.trace_memory), rtol=0,
        atol=within(device),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_a_deep_copy_and_an_imagined_branch_keep_the_life(device):
    host, placed = pair(device)
    rng = np.random.default_rng(6)
    placed.step(observations(rng, 2))
    host.step(observations(np.random.default_rng(6), 2))
    for _ in range(6):
        x, reward = observations(rng, 2), rng.random(2)
        host.step(x, reward=reward)
        placed.step(x, reward=reward)
    twin = copy.deepcopy(placed)
    plan = np.stack([observations(rng, 2) for _ in range(2)])
    host.imagine(plan)
    placed.imagine(plan)
    x, reward = observations(rng, 2), rng.random(2)
    actions = [brain.step(x, reward=reward) for brain in (host, placed, twin)]
    np.testing.assert_array_equal(actions[0], actions[1])
    np.testing.assert_array_equal(actions[0], actions[2])
    same(host, placed, within(device))
    same(host, twin, within(device))


@pytest.mark.parametrize("device", DEVICES)
def test_a_refused_outcome_leaves_the_memories_as_they_were(device, monkeypatch):
    _, placed = pair(device)
    rng = np.random.default_rng(7)
    placed.step(observations(rng, 2))
    for _ in range(4):
        placed.step(observations(rng, 2), reward=rng.random(2))
    placed.act(observations(rng, 2))
    agent, store = placed.basal_ganglia, placed.hippocampus
    before = store._device.strength, store._device.mass, store._device.consolidated
    trace, pending = agent.trace_memory, agent._pending_memory
    maps = placed.recall._device.position_maps

    def refused(*_args, **_kwargs):
        raise ValueError("refused")

    monkeypatch.setattr(placed.learner, "_apply_device", refused)
    monkeypatch.setattr(placed.learner, "apply", refused)
    with pytest.raises(ValueError, match="refused"):
        placed.learn(rng.random(2), np.zeros(2, dtype=bool), observations(rng, 2))
    assert agent.trace_memory is trace and agent._pending_memory is pending
    assert placed.recall._device.position_maps is maps
    after = store._device.strength, store._device.mass, store._device.consolidated
    assert all(a is b for a, b in zip(after, before, strict=True))


@pytest.mark.parametrize(
    "options",
    [{"recall_heads": 2}, {"recall_window": 2}, {"recall_action": True}, {"recall_when": True},
     {"episodic": True, "memory_choice": True}, {"maturity": True}, {"recall": False}],
    ids=["heads", "window", "action", "when", "episodic", "maturity", "no recall"],
)
def test_what_has_no_device_path_stays_on_the_host_and_reads_alike(options):
    host, placed = pair("cpu", **options)
    rng = np.random.default_rng(8)
    placed_recall = placed.recall
    for t in range(8):
        x = observations(rng, 2)
        kwargs = {} if t == 0 else {"reward": rng.random(2)}
        np.testing.assert_array_equal(host.step(x, **kwargs), placed.step(x, **kwargs))
    if "recall" not in options:
        assert placed_recall._device is None
        assert placed.hippocampus._device is None
    host_w, placed_w = host.brain.efficacy, placed.brain.efficacy
    np.testing.assert_allclose(placed_w, host_w, rtol=0, atol=TIGHT)


def test_the_size_decides_on_cuda_and_nothing_else_places_them(monkeypatch):
    class Kernel:
        def __init__(self, kind: str) -> None:
            self.device = type("Device", (), {"type": kind})()

    monkeypatch.delenv("CADENCE_TORCH_MEMORY")
    cuda, cpu, mps = Kernel("cuda"), Kernel("cpu"), Kernel("mps")
    assert stream._memory_on_device(cuda, 1, 3920) and stream._memory_on_device(cuda, 128, 35)
    assert not stream._memory_on_device(cuda, 16, 35) and not stream._memory_on_device(cuda, 1, 35)
    assert not stream._memory_on_device(cpu, 1, 3920) and not stream._memory_on_device(mps, 1, 3920)
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "1")
    assert stream._memory_on_device(cuda, 1, 35) and stream._memory_on_device(cpu, 1, 35)
    assert not stream._memory_on_device(mps, 1, 3920)
    monkeypatch.setenv("CADENCE_TORCH_MEMORY", "0")
    assert not stream._memory_on_device(cuda, 1, 3920)


@pytest.mark.parametrize("device", [d for d in DEVICES if d != "mlx"])
def test_a_float32_brain_works_in_float32_and_its_parameters_add_up_in_float64(device, tmp_path):
    """Where a float32 brain's memories rest, they, its critic and its eligibility are float32; its
    efficacies and biases stay float64 so that small steps add up. A life reads as the host's within
    float32's rounding (the panels' reading is in docs/backends.md)."""
    host, _ = pair(device)
    path = tmp_path / "brain.npz"
    host.save(path)
    placed = cd.Brain.load(path, backend="torch", device=device, precision="float32")
    rng = np.random.default_rng(1)
    for t in range(30):
        x = observations(rng, 3)
        kwargs = {} if t == 0 else {"reward": rng.random(3), "done": rng.random(3) < 0.1}
        np.testing.assert_array_equal(host.step(x, **kwargs), placed.step(x, **kwargs))
    agent, kernel = placed.basal_ganglia, placed.brain._torch
    assert resident(placed) and agent._critic_device is not None
    assert kernel.dtype == torch.float32 and kernel.scale.dtype == torch.float64
    assert agent._trace_device[0].dtype == torch.float32
    assert agent._critic_device.weights.dtype == torch.float32
    assert placed.recall._device.keys.dtype == torch.float32
    assert placed.hippocampus._device.strength.dtype == torch.float32
    same(host, placed, atol=ROUNDED)
