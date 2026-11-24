"""Behavior at the feedback, evaluation, device and training boundaries."""

import copy

import numpy as np
import pytest

import cadence as cd
from cadence.space.map import Map
from cadence.speech.align import Alignments, forced_path, soft_path


@pytest.mark.parametrize("reward", [1.0, -1.0])
def test_delayed_credit_protects_the_acted_value_from_a_near_key(reward, tmp_path):
    b = cd.Brain.compose(
        senses={"eyes": 16, "ear": 16},
        actions=2,
        modules=(8,),
        code=24,
        voice="ear",
        bound=8,
        arousal=True,
        seed=4,
        backend="cpu",
    )
    eye = {"eyes": np.eye(16)[[14]]}
    b.step({**eye, "voice": np.eye(16)[[2]]})
    b = cd.Brain.load(b.save(tmp_path / "acted.npz"))
    slot = int(b.bound.written[0])
    value = b.bound.store.values[slot].copy()
    b.wait({**eye, "voice": np.eye(16)[[3]]})
    np.testing.assert_array_equal(b.bound.store.values[slot], value)
    b.live({}, reward=[reward])
    if reward > 0:
        assert b.say(eye).argmax() == 2 and b.bound.paid[slot] == b.bound.PAY
    else:
        assert b.bound.refused[slot]


def test_two_streams_at_the_same_event_share_its_record_and_a_payment_wins():
    b = cd.Brain.compose(
        senses={"eyes": 16, "ear": 16},
        actions=2,
        modules=(8,),
        code=24,
        voice="ear",
        bound=8,
        seed=4,
        backend="cpu",
    )
    b.step({"eyes": np.eye(16)[[14, 14]], "voice": np.eye(16)[[2, 2]]})
    a, c = b.bound.written
    assert a == c >= 0 and b.bound.store.count == 1
    b.bound.pay([1.0, -1.0])
    assert b.bound.paid[a] > 0 and not b.bound.refused[a]


def test_independent_learned_code_is_cold_after_an_unrelated_stream(monkeypatch):
    brain = cd.Brain.compose(16, 4, modules=(12,), learned_code=24, seed=3, backend="cpu")
    brain.act(np.random.default_rng(3).normal(size=(64, 16)), greedy=True)
    rested = copy.deepcopy(brain)
    rested.reset()
    brain.learned_code.restep(1)
    query = np.random.default_rng(3015).normal(size=(64, 16))
    np.testing.assert_array_equal(
        brain.stimulus(query, memory=False), rested.stimulus(query, memory=False)
    )
    np.testing.assert_array_equal(brain.observation_code(query), rested.observation_code(query))
    expected = rested.predict(query)
    np.testing.assert_array_equal(brain.predict(query), expected)
    assert brain.accuracy(query, expected) == 1
    observed = []
    solve = brain._settle_code

    def counted(*args, **options):
        assert options["cold"] and not options["report"]
        phase = solve(*args, **options)
        observed.append(brain.learned_code.halvings)
        return phase

    monkeypatch.setattr(brain, "_settle_code", counted)
    live = brain.learned_code.graph
    brain.predict(query)
    # one solve from rest, at the step the dictionary allows over every state
    assert observed == [brain.learned_code.rest_stability()]
    # independent reads leave the live policy alone
    assert brain.learned_code.halvings == 1 and brain.learned_code.graph is live


def test_actor_requalifies_after_public_parameter_replacement(tmp_path):
    b = cd.Brain.compose(4, 2, modules=(8,), seed=5, backend="cpu")
    b.act(np.zeros((1, 4)), greedy=True)  # establish all composed stream state for the checkpoint
    drive = np.zeros((1, b.connectome.n))
    actor = b.basal_ganglia
    first = actor.act(drive, greedy=True)[0]
    bias = b.brain.bias.copy()
    bias[b.motor_index] = -8
    bias[b.motor_index[1 - first]] = 8
    b.brain.bias = bias
    loaded = cd.Brain.load(b.save(tmp_path / "changed.npz"))
    assert loaded.basal_ganglia.act(drive, greedy=True)[0] == 1 - first
    assert actor.act(drive, greedy=True)[0] == 1 - first


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_torch_drive_mutation_inference_and_selector(device):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    b = cd.Brain.compose(
        4, 2, modules=(8,), seed=5, backend="torch", device=device, precision="float32"
    )
    motor = list(b.motor_index)
    drive = torch.zeros((1, b.connectome.n), dtype=torch.float32, device=device)
    drive[0, motor[0]] = 8
    actor = b.basal_ganglia
    assert actor.act(drive, greedy=True)[0] == 0
    if device == "cpu":
        drive.numpy()[0, motor] = [-8, 8]  # a write outside Torch's version counter
    else:
        drive[0, motor] = torch.tensor([-8.0, 8.0], device=device)
    assert actor.act(drive, greedy=True)[0] == 1
    with torch.inference_mode():
        assert b.act(np.eye(4)[[0]], greedy=True).shape == (1,)
        bad = torch.full_like(drive, float("nan"))
        with pytest.raises(ValueError):
            b.brain.settle_batch(bad, steps=1)
    state = b.brain.settle_batch(drive, steps=3)
    n = b.connectome.n
    with pytest.raises(IndexError):
        state.part(np.array([n - 1, n]))
    expected = state.part(np.array([-1, -2]))
    np.testing.assert_array_equal(expected, state.activation[:, [-1, -2]])


@pytest.mark.parametrize(
    "method,keys,steps",
    [
        ("lesson", ["a", "missing"], 3),
        ("lesson", ["a", "b"], 2),
        ("targets", ["a", "a"], 3),
        ("targets", ["a", "b"], 2),
    ],
)
def test_failed_alignment_batch_can_be_retried(method, keys, steps):
    bank = Alignments()
    bank.even("a", 2, [1])
    bank.even("b", 3, [2])
    with pytest.raises((KeyError, ValueError)):
        if method == "lesson":
            bank.lesson(keys, steps)
        else:
            bank.targets(keys, steps, 3)
    assert "a" in bank and "b" in bank
    got = bank.targets(["a", "b"], 3, 3)
    np.testing.assert_array_equal(got, np.eye(3)[[[1, 2], [1, 2], [0, 2]]])
    assert len(bank) == 0


@pytest.mark.parametrize(
    "kind,point",
    [(-1, (0.0, 0.0)), (3, (0.0, 0.0)), (0.9, (0.0, 0.0)), (0, (np.inf, 0.0)), (0, (0.0, np.nan))],
)
def test_invalid_second_spatial_report_preserves_the_whole_moment(kind, point):
    grid = Map((0.0, 1.0, 0.0, 1.0), (5, 5), units=3)
    grid.moment(seen=[(0, (0.1, 0.2))])
    before = {name: v.copy() for name, v in grid.arrays().items()}
    with pytest.raises(ValueError):
        grid.moment(seen=[(1, (0.4, 0.6)), (kind, point)])
    for name, v in grid.arrays().items():
        np.testing.assert_array_equal(v, before[name])


def test_zero_weight_frames_do_not_train_visual_fields():
    b = cd.Brain.compose((16, 16), 2, learned_fields=4, seed=0, backend="cpu", recall=False)
    x = np.random.default_rng(42).random((4, 256))
    b.lesson(x, [0, 1, 0, 1])
    fields = b.fields
    dictionary, thresholds = fields.dictionary.copy(), fields.thresholds.copy()
    updates = fields.work["updates"]
    b.lesson(x, [1, 0, 1, 0], weights=np.zeros(4))
    np.testing.assert_array_equal(dictionary, fields.dictionary)
    np.testing.assert_array_equal(thresholds, fields.thresholds)
    assert fields.work["updates"] == updates
    twin = copy.deepcopy(b)
    b.lesson(x, [1, 0, 1, 0], weights=[1, 0, 0, 1])
    twin.fields.learn(x[[0, 3]])
    np.testing.assert_array_equal(twin.fields.dictionary, fields.dictionary)
    np.testing.assert_array_equal(twin.fields.thresholds, fields.thresholds)


@pytest.mark.parametrize("align", [forced_path, soft_path])
@pytest.mark.parametrize("symbol", [np.uint64(2**64 - 1), np.uint64(2**63), float(2**64)])
def test_transcript_rejects_unrepresentable_symbol(align, symbol):
    scores = np.log(np.array([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]))
    with pytest.raises(ValueError):
        align(scores, np.array([symbol]))


def test_mlx_residual_and_actor_see_an_edited_device_drive():
    mx = pytest.importorskip("mlx.core")
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=5, backend="mlx")
    graph, actor = brain.brain, brain.basal_ganglia
    drive = mx.zeros((1, brain.connectome.n))
    state = graph.equilibrate(drive).state
    drive[0, 0] = 1.
    np.testing.assert_allclose(graph.residual(drive, state),
                               graph.residual(np.array(drive), state, on_device=False), atol=1e-6)
    motors = brain.motor_index.tolist()
    drive = mx.zeros((1, brain.connectome.n))
    drive[0, motors[0]] = 8.
    assert actor.act(drive, greedy=True)[0] == 0
    drive[0, motors[0]] = -8.
    drive[0, motors[1]] = 8.
    assert actor.act(drive, greedy=True)[0] == 1
