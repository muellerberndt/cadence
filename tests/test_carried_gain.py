"""One learned gain per carried route (``carried_gain``, cadence#263): how strongly the association
hears the working trace, the efference copy and every block of the heard history, taught by the
settle's own contrast."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd
from cadence.stream import Hearing


def brain_of(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, modules=(16,), seed=5, **options)


def teach(brain: cd.Brain, seed: int = 4, lessons: int = 12) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(lessons):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))


def live(brain: cd.Brain, seed: int = 7, moments: int = 12) -> None:
    rng = np.random.default_rng(seed)
    brain.reset()
    for t in range(moments):
        brain.act(rng.random((2, 6)))
        brain.learn(rng.random(2), np.full(2, t % 5 == 4), rng.random((2, 6)))


def test_the_newborn_reads_as_without_it():
    plain = brain_of()
    assert plain.hearing is None and plain.carried_gains == {}
    assert "carried_gain" not in plain.describe()["genes"]
    assert plain.working_memory.hearing == 1.0
    # at birth every gain is the working value: the first drive is the plain brain's
    brain = brain_of(carried_gain=0.01)
    x = np.random.default_rng(1).random((2, 6))
    np.testing.assert_array_equal(plain.stimulus(x), brain.stimulus(x))
    np.testing.assert_array_equal(plain.brain.efficacy, brain.brain.efficacy)
    # at a zero rate the option is absent
    assert brain_of(carried_gain=0.0).hearing is None
    with pytest.raises(ValueError, match="carried_gain"):
        brain_of(carried_gain=-0.01)


def test_one_gain_per_carried_route_each_one_parameter():
    brain = brain_of(carried_gain=0.01)
    assert brain.carried_gains == {"trace": 1.0}
    assert brain.parameters() == brain_of().parameters() + 1
    options = {
        "history_decays": (0.5, 0.9), "efference_amplitude": 0.2, "recall": True, "recall_heads": 2,
    }
    full = brain_of(carried_gain=0.01, **options)
    assert full.carried_gains == {
        "trace": 1.0, "efference": 1.0, "history_0": 1.0, "history_1": 1.0,
    }
    assert full.parameters() == brain_of(**options).parameters() + 4
    assert full.describe()["genes"]["carried_gain"] == 0.01


def test_a_gain_scales_what_its_route_drives():
    brain = brain_of(
        carried_gain=0.01, history_decays=(0.5, 0.9), efference_amplitude=0.2, recall=True,
    )
    rng = np.random.default_rng(2)
    for _ in range(3):
        brain.act(rng.random((2, 6)), greedy=True)
    populations = brain.connectome.populations
    x = rng.random((2, 6))
    before = brain.stimulus(x)
    brain.working_memory.hearing = 0.5
    brain.efference.hearing = 0.25
    brain.history.hearing[:] = (2.0, 0.0)
    after = brain.stimulus(x)
    prefrontal, efference = populations["prefrontal"], populations["efference"]
    history = np.asarray(populations["history"])
    np.testing.assert_allclose(after[:, prefrontal], 0.5 * before[:, prefrontal])
    np.testing.assert_allclose(after[:, efference], 0.25 * before[:, efference])
    width = brain.history.width
    np.testing.assert_allclose(after[:, history[:width]], 2.0 * before[:, history[:width]])
    assert not after[:, history[width:]].any()


def test_the_contrast_is_the_energys_derivative_on_the_route():
    brain = brain_of(carried_gain=0.01)
    hearing = brain.hearing
    rng = np.random.default_rng(3)
    brain.act(rng.random((2, 6)), greedy=True)
    brain.act(rng.random((2, 6)), greedy=True)
    trace = brain.working_memory
    association = np.asarray(brain.association_index)
    movement = rng.normal(size=(2, len(association)))
    got = hearing._contrast(brain.learner.brain, movement, None, 2)[:, 0]
    # sum_j f'(v_j) a t_j sum_i W_ij m_i, summed by hand over the route's synapses
    graph, route = brain.learner.brain, hearing.routes[0]
    weights = np.asarray(graph._gain_pre) * np.asarray(graph.efficacy)
    slope = brain.brain.neuron_model.slope_at(trace.amplitude * trace.trace)
    expected = np.zeros(2)
    for synapse, j, i in zip(route["synapses"], route["rows"], route["cols"], strict=True):
        expected += (
            slope[:, j] * trace.amplitude * trace.trace[:, j] * weights[synapse] * movement[:, i]
        )
    np.testing.assert_allclose(got, expected)
    assert np.abs(expected).max() > 0


def test_a_step_is_normalized_and_bounded():
    connectome = brain_of().connectome
    association = np.asarray(connectome.populations["association"])
    model = brain_of().brain.neuron_model

    class Held:
        hearing = 1.0
        amplitude = 3.0
        trace = np.zeros((1, 16))

    hearing = Hearing(0.5, connectome, association, model)
    held = Held()
    hearing.add("held", held, None, 0, 16, np.asarray(connectome.populations["prefrontal"]))
    # the first step is the rate's whatever the contrast's size
    hearing.step(np.array([1e6]))
    assert held.hearing == pytest.approx(1.5)
    for _ in range(100):
        hearing.step(np.array([1e6]))
    assert held.hearing == Hearing.BOUND
    for _ in range(100):
        hearing.step(np.array([-1e-6]))
    assert 0.0 <= held.hearing < 8.0
    held.hearing = 0.1
    hearing.count, hearing.second = 0, np.zeros(1)
    hearing.step(np.array([-1.0]))
    assert held.hearing == 0.0


def test_a_lesson_and_the_dopamine_each_teach_the_gains():
    options = {"history_decays": (0.5, 0.9), "carried_gain": 0.01}
    brain = brain_of(**options)
    teach(brain)
    lesson = brain.carried_gains
    assert all(gain != 1.0 for gain in lesson.values()), lesson
    brain = brain_of(**options)
    live(brain)
    assert all(gain != 1.0 for gain in brain.carried_gains.values()), brain.carried_gains
    assert all(0.0 <= gain <= Hearing.BOUND for gain in brain.carried_gains.values())


def test_a_lesson_of_another_batch_reads_its_memory_as_silence():
    brain = brain_of(carried_gain=0.01)
    rng = np.random.default_rng(5)
    brain.act(rng.random((2, 6)), greedy=True)
    brain.reset()
    # a lesson of a new batch finds the trace empty: nothing raises, the gain stays bounded
    brain.lesson(rng.random((3, 6)), rng.integers(0, 4, size=3))
    assert 0.0 <= brain.carried_gains["trace"] <= Hearing.BOUND


def test_a_checkpoint_continues_exactly(tmp_path):
    brain = brain_of(carried_gain=0.01, history_decays=(0.5, 0.9), recall=True)
    teach(brain)
    live(brain)
    rng = np.random.default_rng(9)
    # saved with an outcome pending: the act's contrast and the eligibility come back
    x = rng.random((2, 6))
    brain.act(x)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert copy.carried_gains == brain.carried_gains
    np.testing.assert_array_equal(copy.hearing.second, brain.hearing.second)
    assert copy.hearing.count == brain.hearing.count
    np.testing.assert_array_equal(
        copy.basal_ganglia.trace_hearing, brain.basal_ganglia.trace_hearing
    )
    for t in range(6):
        reward, following = rng.random(2), rng.random((2, 6))
        done = np.full(2, t == 3)
        brain.learn(reward, done, following)
        copy.learn(reward, done, following)
        np.testing.assert_array_equal(brain.act(following), copy.act(following))
    brain.reset()
    copy.reset()
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
    assert copy.carried_gains == brain.carried_gains
    np.testing.assert_array_equal(brain.brain.efficacy, copy.brain.efficacy)
    assert copy.parameters() == brain.parameters()


def test_a_brain_without_it_saves_nothing_of_it(tmp_path):
    copy = cd.Brain.load(brain_of().save(tmp_path / "brain.npz"))
    assert copy.hearing is None and copy.basal_ganglia.hearing is None


def test_retune_refuses_it():
    with pytest.raises(ValueError, match="construction setting"):
        brain_of(carried_gain=0.01).retune(carried_gain=0.1)
    with pytest.raises(ValueError, match="construction setting"):
        brain_of().retune(carried_gain=0.1)


def test_grow_carries_the_gains_by_route_and_a_new_route_starts_at_one():
    brain = brain_of(carried_gain=0.01)
    teach(brain)
    trace = brain.carried_gains["trace"]
    assert trace != 1.0
    grown = brain.grow(history_decays=(0.5,))
    assert grown.carried_gains == {"trace": trace, "history_0": 1.0}
    assert grown.hearing.count == brain.hearing.count
    assert grown.hearing.second[0] == brain.hearing.second[0] > 0
    assert grown.basal_ganglia.hearing is grown.hearing
    assert grown.parameters() > brain.parameters()
    # and the next grow carries both on
    grown.hearing.set(grown.hearing.routes[1], 0.5)
    again = grown.grow(efference_amplitude=0.2)
    assert again.carried_gains == {"trace": trace, "efference": 1.0, "history_0": 0.5}


def test_graft_keeps_this_brains_gains():
    brain, donor = brain_of(carried_gain=0.01), brain_of(carried_gain=0.01)
    teach(brain)
    teach(donor, seed=8)
    gains = brain.carried_gains
    assert gains != donor.carried_gains
    grafted = brain.graft(donor, ("association", "motor"))
    assert grafted.carried_gains == gains
    assert grafted.hearing is not None and grafted.basal_ganglia.hearing is grafted.hearing


def test_an_imagined_branch_leaves_the_gains():
    brain = brain_of(carried_gain=0.01)
    teach(brain)
    gains, second = brain.carried_gains, brain.hearing.second.copy()
    rng = np.random.default_rng(5)
    brain.imagine([rng.random((2, 6)) for _ in range(2)], budget=256)
    assert brain.carried_gains == gains
    np.testing.assert_array_equal(brain.hearing.second, second)


@pytest.mark.parametrize(
    ("backend", "device", "precision", "atol"),
    [("torch", "cpu", "float64", 1e-9), ("torch", "mps", "float32", 1e-3),
     ("mlx", None, None, 1e-3)],
    ids=["torch_cpu64", "mps32", "mlx"],
)
def test_the_gains_learn_on_a_device_as_on_the_cpu(tmp_path, backend, device, precision, atol):
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    if device == "mps":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS hardware unavailable")
    cpu = brain_of(carried_gain=0.01, history_decays=(0.5, 0.9), recall=True, recall_when=True)
    cpu.save(tmp_path / "brain.npz")
    moved = cd.Brain.load(
        tmp_path / "brain.npz", backend=backend, device=device, precision=precision
    )
    assert moved.brain.backend == backend
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for brain in (cpu, moved):
            teach(brain, lessons=6)
            brain.reset()
    assert all(gain != 1.0 for gain in cpu.carried_gains.values())
    np.testing.assert_allclose(
        list(moved.carried_gains.values()), list(cpu.carried_gains.values()), rtol=0, atol=atol
    )


def test_grow_adds_no_gains_to_a_brain_without_them():
    with pytest.raises(ValueError, match="carried_gain"):
        brain_of().grow(carried_gain=0.01)


def test_every_routine_keeps_the_gains_in_their_bounds():
    brain = brain_of(
        carried_gain=0.01, recall=True, arousal=True, memory_choice=True,
        efference_amplitude=0.5, history_decays=(0.5,),
    )
    rng = np.random.default_rng(4)
    x = rng.random((1, 6))
    brain.live(x)
    for _ in range(3):
        brain.live(x, reward=[1.0])
    brain.wait(x)
    brain.reset()
    brain.act(x)
    brain.learn(np.array([1.0]), np.array([False]), rng.random((1, 6)))
    brain.lesson(rng.random((1, 6)), np.array([1]))
    brain.imagine([rng.random((1, 6)) for _ in range(2)], budget=256)
    assert all(0.0 <= gain <= Hearing.BOUND for gain in brain.carried_gains.values())
    brain.reset()
    brain.fit(rng.random((8, 6)), rng.integers(0, 4, size=8), epochs=1, batch=4)
    assert all(0.0 <= gain <= Hearing.BOUND for gain in brain.carried_gains.values())


def test_a_learned_code_predicts_from_what_the_region_holds():
    brain = brain_of(carried_gain=0.01, learned_code=32)
    rng = np.random.default_rng(6)
    brain.act(rng.random((2, 6)), greedy=True)
    trace = brain.working_memory
    assert trace.trace.any()
    trace.hearing = 0.5
    seen = []
    stimulus = brain.learned_code.stimulus

    def spy(sensory, z):
        seen.append((z, trace.trace.copy()))
        return stimulus(sensory, z)

    brain.learned_code.stimulus = spy
    brain.act(rng.random((2, 6)), greedy=True)
    z, held = seen[0]
    model, bias = brain.brain.neuron_model, brain.brain.bias
    expected = model.activation(0.5 * trace.amplitude * held + bias[brain.prefrontal_index])
    np.testing.assert_allclose(z, expected)
