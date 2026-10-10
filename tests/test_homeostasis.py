"""The readout's intrinsic plasticity: two genes, founder off, a bias step toward a target
activation at every teaching or reward update, on the host and on the device."""

import numpy as np
import pytest

import cadence as cd

INPUTS, ACTIONS = 8, 3


def compose(seed: int = 0, **options):
    return cd.Brain.compose(INPUTS, ACTIONS, modules=(16,), seed=seed, **options)


def patterns(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((4, INPUTS)) < 0.5).astype(float)


def motor_activation(brain: cd.Brain) -> np.ndarray:
    state = brain.basal_ganglia.state
    assert state is not None
    return np.atleast_2d(np.asarray(state.activation))[:, brain.motor_index]


def test_the_founder_is_off_and_leaves_the_composed_brain_unchanged(tmp_path):
    plain = compose()
    named = compose(learning_homeostasis_rate=0.0, learning_homeostasis_target=0.3)
    assert plain.describe()["learning"]["homeostasis_rate"] == 0.0
    assert plain.describe()["learning"]["homeostasis_target"] == 0.3
    assert (
        plain.save(tmp_path / "plain.npz").read_bytes()
        == named.save(tmp_path / "named.npz").read_bytes()
    )
    x = patterns()
    before = plain.brain.bias.copy()
    report = plain.learner.step(plain.stimulus(x), np.array([0, 1, 2, 0]))[1]
    assert report["homeostasis_step"] == 0.0
    twin = compose()
    twin.learner.step(twin.stimulus(x), np.array([0, 1, 2, 0]))
    assert np.array_equal(plain.brain.bias, twin.brain.bias)
    assert not np.array_equal(plain.brain.bias, before), "the ordinary lesson still moves biases"


@pytest.mark.parametrize(
    "genes, message",
    [
        ({"learning_homeostasis_rate": -0.1}, "homeostasis_rate"),
        ({"learning_homeostasis_rate": 1.5}, "homeostasis_rate"),
        ({"learning_homeostasis_target": 1.0}, "homeostasis_target"),
        ({"learning_homeostasis_target": -0.2}, "homeostasis_target"),
    ],
)
def test_the_genes_are_validated(genes, message):
    with pytest.raises(ValueError, match=message):
        compose(**genes)


def test_a_lesson_adds_the_intrinsic_step_to_the_output_biases_only():
    rate, target = 0.4, 0.3
    brain = compose(learning_homeostasis_rate=rate, learning_homeostasis_target=target)
    twin = compose()
    x, labels = patterns(), np.array([0, 1, 2, 0])
    state, report = brain.learner.step(brain.stimulus(x), labels)
    twin.learner.step(twin.stimulus(x), labels)
    free = np.atleast_2d(np.asarray(state.free.activation))[:, brain.motor_index]
    expected = rate * (target - free.mean(axis=0))
    motor = brain.motor_index
    assert np.allclose(brain.brain.bias[motor] - twin.brain.bias[motor], expected)
    others = np.setdiff1d(np.arange(brain.connectome.n), motor)
    assert np.array_equal(brain.brain.bias[others], twin.brain.bias[others])
    assert np.array_equal(brain.brain.efficacy, twin.brain.efficacy)
    assert report["homeostasis_step"] == pytest.approx(float(np.abs(expected).mean()))


def test_a_learned_outcome_adds_the_intrinsic_step_from_the_decision_state():
    rate, target = 0.25, 0.3
    brain = compose(learning_homeostasis_rate=rate, learning_homeostasis_target=target, seed=3)
    twin = compose(seed=3)
    x = patterns()[:1]
    brain.act(x)
    twin.act(x)
    free = motor_activation(brain)
    assert np.array_equal(free, motor_activation(twin))
    report = brain.learn(np.array([1.0]), np.array([False]), x)
    twin.learn(np.array([1.0]), np.array([False]), x)
    motor = brain.motor_index
    expected = rate * (target - free.mean(axis=0))
    assert np.allclose(brain.brain.bias[motor] - twin.brain.bias[motor], expected)
    assert report["homeostasis_step"] == pytest.approx(float(np.abs(expected).mean()))


def test_retune_and_describe_carry_the_genes():
    brain = compose()
    brain.retune(learning_homeostasis_rate=0.2, learning_homeostasis_target=0.4)
    learning = brain.describe()["learning"]
    assert (learning["homeostasis_rate"], learning["homeostasis_target"]) == (0.2, 0.4)
    with pytest.raises(ValueError):
        brain.retune(learning_homeostasis_target=2.0)
    assert brain.describe()["learning"]["homeostasis_target"] == 0.4


def test_a_saved_life_with_the_gene_on_continues_identically(tmp_path):
    brain = compose(learning_homeostasis_rate=0.3, seed=5)
    x, labels = patterns(), np.array([2, 1, 0, 2])
    for _ in range(3):
        brain.learner.step(brain.stimulus(x), labels)
    copy = cd.Brain.load(brain.save(tmp_path / "life.npz"))
    assert copy.describe()["learning"]["homeostasis_rate"] == 0.3
    for _ in range(3):
        brain.learner.step(brain.stimulus(x), labels)
        copy.learner.step(copy.stimulus(x), labels)
    assert np.array_equal(brain.brain.bias, copy.brain.bias)
    assert np.array_equal(brain.brain.efficacy, copy.brain.efficacy)


def _with_motor_bias(brain: cd.Brain, value: float) -> None:
    bias = brain.brain.bias.copy()
    bias[brain.motor_index] = value
    brain.learner.brain = brain.learner.brain.with_parameters(bias=bias)


@pytest.mark.parametrize("parked", [8.0, -4.0])
def test_a_parked_readout_returns_to_its_range_only_with_the_gene_on(parked):
    x, labels = patterns(), np.array([0, 1, 2, 0])
    levels = {}
    for rate in (0.0, 0.5):
        brain = compose(learning_homeostasis_rate=rate, seed=7)
        _with_motor_bias(brain, parked)
        brain.act(x, greedy=True)
        start = float(motor_activation(brain).mean())
        for _ in range(60):
            brain.learner.step(brain.stimulus(x), labels)
        brain.act(x, greedy=True)
        levels[rate] = (start, float(motor_activation(brain).mean()))
    start, still = levels[0.0]
    if parked > 0:
        assert start > 0.95 and still > 0.95, "without the gene a saturated readout stays saturated"
        assert levels[0.5][1] < 0.6, "with the gene it comes down toward the target"
    else:
        assert start < 0.05 and still < 0.05, "without the gene a silent readout stays silent"
        assert levels[0.5][1] > 0.15, "with the gene it rises toward the target"


def test_the_device_path_takes_the_same_intrinsic_step():
    pytest.importorskip("torch")
    rate, target = 0.4, 0.3
    x, labels = patterns(), np.array([0, 1, 2, 0])
    host = compose(learning_homeostasis_rate=rate, learning_homeostasis_target=target)
    device = cd.Brain.compose(
        INPUTS,
        ACTIONS,
        modules=(16,),
        seed=0,
        learning_homeostasis_rate=rate,
        learning_homeostasis_target=target,
        backend="torch",
        device="cpu",
    )
    host_report = host.learner.step(host.stimulus(x), labels)[1]
    device_report = device.learner.step(device.stimulus(x), labels)[1]
    assert device_report["homeostasis_step"] == pytest.approx(
        host_report["homeostasis_step"], rel=1e-5
    )
    assert np.allclose(host.brain.bias, device.brain.bias, atol=1e-6)
