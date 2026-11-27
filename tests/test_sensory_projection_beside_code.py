"""A one-way sensory projection beside a learned code (a reflex arc from the observation to the
motor region): the sensory neurons hold the observation as a source of the whole graph, the
projection's synapses carry it to the motor neurons and learn by the lessons' local rule, as in
a brain without the region, and the code's own synapses, tables and phase are what they are
without it. The neurons that read the code still read it alone, and nothing leads back into the
sensory neurons."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.genome import Genome, Projection, develop
from cadence.reference import settle_neuron_by_neuron

INPUTS, ACTIONS, UNITS, SLOTS = 12, 6, 16, (3, 3)
X = np.eye(INPUTS)[[0, 5]]  # one input on per row
LABELS = np.array([[0, 1], [2, 0]])  # one label per motor slot
# a stream of two moments in turn: a lesson never follows the act on its own observations
STREAM = ((X, LABELS), (np.eye(INPUTS)[[3, 9]], np.array([[1, 2], [0, 1]])))
DEEP = {
    "modules": (16, 12), "recall": True, "recall_next": True, "recall_window": 2,
    "recall_heads": 2, "attention": True, "memory_amplitude": 0.0, "apical": 0.25,
    "teaching": "deep", "learning_drive_budget": 4.0, "learning_credit_window": 3,
}


def anatomy(seed: int = 0, extra: tuple[Projection, ...] | None = None, **options):
    """The composed layout of a learned code with ``extra`` projections appended (one from the
    sensory to the motor region unless given), developed."""
    if extra is None:
        extra = (Projection("sensory", "motor", reciprocal=False, scale=0.1),)
    options = {"modules": (16,), **options}
    base = Genome.composed(
        INPUTS, ACTIONS, slots=SLOTS, learned_code=UNITS, seed=seed, **options
    )
    return develop(Genome(base.regions, (*base.projections, *extra), base.label), seed=seed)


def composed(seed: int = 0, extra: tuple[Projection, ...] | None = None, place=None, **options):
    """``Brain.compose`` around that anatomy; ``modules`` is the layout's, ``place`` the backend."""
    options = {"recall": False, **options}
    connectome = anatomy(seed, extra, **options)
    options.pop("modules", None)
    return cd.Brain.compose(
        INPUTS, ACTIONS, connectome=connectome, slots=SLOTS, learned_code=UNITS, seed=seed,
        **options, **(place or {}),
    )


def projection(brain: cd.Brain) -> np.ndarray:
    """The synapses from the sensory to the motor neurons."""
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    return np.isin(pre, brain.sensory_index) & np.isin(post, brain.motor_index)


def region(brain: cd.Brain) -> np.ndarray:
    """The learned code's own synapses: dictionary, lateral and predictor."""
    return np.concatenate([synapses for synapses, _, _ in brain._code_synapses.values()])


def column(brain: cd.Brain, index: np.ndarray) -> np.ndarray:
    """Each neuron's place in ``index``, -1 outside it."""
    out = -np.ones(brain.connectome.n, dtype=np.int64)
    out[index] = np.arange(len(index))
    return out


def tables(brain: cd.Brain) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in brain.learned_code.arrays().items()}


def test_the_projection_composes_beside_the_code_and_leaves_the_rest_as_it_was():
    brain, plain = composed(), composed(extra=())
    reflex = projection(brain)
    assert reflex.sum() == INPUTS * ACTIONS and not projection(plain).any()
    learner = brain.learner
    # its synapses are taught; the region's own stay with the region
    assert learner.plastic_synapses[reflex].all()
    assert not learner.plastic_synapses[region(brain)].any()
    assert not learner.plastic_neurons[brain.code_index].any()
    assert learner.plastic_synapses.sum() == plain.learner.plastic_synapses.sum() + reflex.sum()
    # appended last, the projection changes no other synapse, bias or table, and no reader
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    assert np.array_equal(pre[~reflex], plain.connectome.pre)
    assert np.array_equal(post[~reflex], plain.connectome.post)
    assert np.array_equal(np.asarray(brain.brain.efficacy)[~reflex], plain.brain.efficacy)
    assert np.array_equal(brain.brain.bias, plain.brain.bias)
    assert np.array_equal(brain._code_readers, plain._code_readers)
    assert np.array_equal(brain._code_readers, brain.association_index)
    for name, value in tables(plain).items():
        assert np.array_equal(brain.learned_code.arrays()[name], value), name
    # the region's phase reads the sensory activation alone: the same code with and without
    assert np.array_equal(brain.observation_code(X), plain.observation_code(X))
    # sources and outputs are no hidden region of the lessons' credit plan
    hidden = learner._credit_plan().hidden
    assert not hidden[brain.sensory_index].any() and not hidden[brain.motor_index].any()


def test_the_readers_read_the_code_alone_and_nothing_leads_back_into_the_senses():
    # onto the neurons that read the code: the form the region does not take
    with pytest.raises(ValueError, match="read the code alone"):
        composed(extra=(Projection("sensory", "association", reciprocal=False),))
    with pytest.raises(ValueError, match="read the code alone"):
        composed(modules=(16, 12), extra=(Projection("sensory", "module_0", reciprocal=False),))
    # a reciprocal projection leads back into the sensory neurons
    with pytest.raises(ValueError, match="sensory and prefrontal neurons that receive no"):
        composed(extra=(Projection("sensory", "motor", scale=0.1),))
    # past the readers a deeper module takes one, as the motor region does
    deep = composed(
        modules=(16, 12), extra=(Projection("sensory", "association", reciprocal=False),)
    )
    pre, post = np.asarray(deep.connectome.pre), np.asarray(deep.connectome.post)
    beside = np.isin(pre, deep.sensory_index) & np.isin(post, deep.association_index)
    assert beside.sum() == INPUTS * 12 and deep.learner.plastic_synapses[beside].all()
    assert np.array_equal(deep._code_readers, deep.connectome.populations["module_0"])


def test_the_settle_is_the_neuron_by_neuron_sum_with_the_projection_in_it():
    brain = composed()
    graph, connectome = brain.brain, brain.connectome
    drive = np.asarray(brain.stimulus(X, memory=False))  # the code held as a source
    for row in drive:
        ours = graph.settle(row, steps=30, trajectory=True)
        reference, ledger = settle_neuron_by_neuron(
            connectome, graph.neuron_model, row, steps=30, log_gain=graph.log_gain,
            bias=graph.bias, efficacy=graph.efficacy,
        )
        assert ledger.clean
        np.testing.assert_allclose(ours.trajectory, reference, rtol=0, atol=1e-12)
    # the motor neurons' potential equation by hand, the projection's term apart
    settled = graph.equilibrate(drive, budget=4000, tolerance=1e-12)
    assert settled.qualified.all()
    s, v = settled.state.activation, settled.state.v
    pre, post = np.asarray(connectome.pre), np.asarray(connectome.post)
    weight = graph.neuron_model.gain * connectome.count * np.asarray(graph.efficacy)
    reflex = projection(brain)
    model = graph.neuron_model
    held = model.activation(X * model.stimulus_amplitude + graph.bias[brain.sensory_index])
    # the sensory neurons hold the observation, a source of the settle
    np.testing.assert_allclose(s[:, brain.sensory_index], held, rtol=0, atol=1e-12)
    arc, rest = np.zeros_like(s), np.zeros_like(s)
    np.add.at(arc, (slice(None), post[reflex]),
              held[:, column(brain, brain.sensory_index)[pre[reflex]]] * weight[reflex])
    np.add.at(rest, (slice(None), post[~reflex]), s[:, pre[~reflex]] * weight[~reflex])
    motor = brain.motor_index
    assert np.abs(arc[:, motor]).max() > 1e-3
    np.testing.assert_allclose(
        v[:, motor], (arc + rest + drive + graph.bias)[:, motor], rtol=0, atol=1e-10
    )
    # so the projection is the current its source gives the motor neurons: the brain without
    # its synapses settles to the same state under that current, and to another without it
    muted = np.array(graph.efficacy, dtype=float, copy=True)
    muted[reflex] = 0.0
    cut = graph.with_parameters(efficacy=muted)
    as_current = cut.equilibrate(drive + arc, budget=4000, tolerance=1e-12)
    without = cut.equilibrate(drive, budget=4000, tolerance=1e-12)
    np.testing.assert_allclose(as_current.state.v, v, rtol=0, atol=1e-12)
    assert np.abs(without.state.v - v)[:, motor].max() > 1e-3


def test_a_motor_lesson_moves_the_projection_by_the_local_rule():
    # the plain rule, without momentum or normalized steps: a step is the rate times the contrast
    config = cd.LearnerConfig(beta=0.1, eta=0.5, free_steps=1024, nudged_steps=12, tolerance=3e-3)
    brain = composed(learning=config)
    reflex, frozen = projection(brain), region(brain)
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    before = np.array(brain.brain.efficacy, dtype=float, copy=True)
    # the learner's step on the moment's drive, its phases in hand
    drive = np.asarray(brain.stimulus(X, memory=False))
    lesson, _ = brain.learner.step(drive, LABELS)
    plus, minus = lesson.nudged.activation, lesson.opposite.activation
    # a source holds the same activation in both phases
    np.testing.assert_array_equal(plus[:, brain.sensory_index], minus[:, brain.sensory_index])
    contrast = (plus[:, pre] * plus[:, post] - minus[:, pre] * minus[:, post]).mean(axis=0)
    step = config.eta * contrast / (2.0 * config.beta)
    moved = np.asarray(brain.brain.efficacy) - before
    np.testing.assert_allclose(moved[reflex], step[reflex], rtol=0, atol=1e-12)
    assert np.abs(moved[reflex]).max() > 1e-3
    assert not moved[frozen].any() and brain._code_installed(brain.brain)


def test_a_lesson_strengthens_the_heard_inputs_onto_the_shown_actions():
    brain = composed()
    reflex = projection(brain)
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    motor = column(brain, brain.motor_index)
    before, held = np.array(brain.brain.efficacy, dtype=float, copy=True), tables(brain)
    report = brain.lesson(X, LABELS)
    assert report["accepted_presentations"] == len(X)
    moved = np.asarray(brain.brain.efficacy) - before
    offsets = np.concatenate([[0], np.cumsum(SLOTS)[:-1]])
    for row, heard in enumerate(np.argmax(X, axis=1)):
        from_heard = reflex & (pre == brain.sensory_index[heard])
        shown = np.isin(motor[post], offsets + LABELS[row])
        assert (moved[from_heard & shown] > 0).all()  # onto the shown action of every slot
        assert (moved[from_heard & ~shown] < 0).all()  # off the slot's other actions
    silent = reflex & ~np.isin(pre, brain.sensory_index[np.argmax(X, axis=1)])
    assert silent.sum() == (INPUTS - len(X)) * ACTIONS and not moved[silent].any()
    # the lesson's contrast enters no table and no synapse of the region
    assert not np.asarray(brain.brain.efficacy)[region(brain)].any()
    for name, value in held.items():
        assert np.array_equal(brain.learned_code.arrays()[name], value), name


def test_the_region_stays_its_own_through_lessons_acts_and_rewards():
    brain, plain = composed(seed=2, **DEEP), composed(seed=2, extra=(), **DEEP)
    reflex, frozen = projection(brain), region(brain)
    before, born = np.array(brain.brain.efficacy, dtype=float, copy=True), tables(brain)
    reports = []
    for lived in (brain, plain):
        for moment in range(4):
            x, labels = STREAM[moment % 2]
            report = lived.lesson(x, labels)
            lived.act(x, greedy=True)
        reports.append(report)
        lived.step(X)
        for _ in range(3):
            lived.step(X, reward=np.array([1.0, -1.0]))
    after = np.asarray(brain.brain.efficacy)
    assert np.isfinite(after).all() and (after[reflex] != before[reflex]).any()
    # the region's synapses rest at zero in the whole graph and its biases are the thresholds
    assert not after[frozen].any() and brain._code_installed(brain.brain)
    assert np.array_equal(brain.brain.bias[brain.code_index], -brain.learned_code.thresholds)
    # the acts taught the region by its own rule, as they do without the projection
    assert not np.array_equal(brain.learned_code.dictionary, born["dictionary"])
    assert not np.array_equal(plain.learned_code.dictionary, born["dictionary"])
    # the lessons' controls read the same regions with and without the projection
    ours, theirs = reports
    assert "drive_limited_projections" in ours and "credit_motor_contrast" in ours
    assert {name for name in ours if name.startswith("credit_")} == {
        name for name in theirs if name.startswith("credit_")
    }
    assert not any("sensory" in name or "credit_code" in name for name in ours)


def test_a_saved_brain_continues_with_the_projection(tmp_path):
    brain = composed(seed=1, **DEEP)
    for x, labels in STREAM:
        brain.lesson(x, labels)
        brain.act(x, greedy=True)
    path = brain.save(tmp_path / "brain.npz")
    loaded = cd.Brain.load(path)
    reflex = projection(loaded)
    assert reflex.sum() == INPUTS * ACTIONS and np.array_equal(reflex, projection(brain))
    assert np.array_equal(loaded.brain.efficacy, brain.brain.efficacy)
    assert np.array_equal(loaded.brain.bias, brain.brain.bias)
    assert np.array_equal(loaded.learner.plastic_synapses, brain.learner.plastic_synapses)
    for name, value in tables(brain).items():
        assert np.array_equal(loaded.learned_code.arrays()[name], value), name
    for x, labels in STREAM:  # the same lessons and acts from the same state
        assert brain.lesson(x, labels) == loaded.lesson(x, labels)
        assert np.array_equal(brain.act(x, greedy=True), loaded.act(x, greedy=True))
    assert np.array_equal(loaded.brain.efficacy, brain.brain.efficacy)
    assert (np.asarray(loaded.brain.efficacy)[reflex] != 0).any()


def test_the_torch_backend_reads_the_projection_as_the_host_does():
    torch = pytest.importorskip("torch")
    del torch
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    host, device = composed(seed=3, **DEEP), composed(seed=3, place=place, **DEEP)
    assert device.brain.backend == "torch"
    reflex = projection(host)
    before = np.array(host.brain.efficacy, dtype=float, copy=True)
    for moment in range(4):
        x, labels = STREAM[moment % 2]
        for brain in (host, device):
            brain.lesson(x, labels)
        assert np.array_equal(host.act(x, greedy=True), device.act(x, greedy=True))
    ours, theirs = np.asarray(host.brain.efficacy), np.asarray(device.brain.efficacy)
    np.testing.assert_allclose(theirs, ours, rtol=0, atol=1e-9)
    np.testing.assert_allclose(device.brain.bias, host.brain.bias, rtol=0, atol=1e-9)
    assert (ours[reflex] != before[reflex]).any()
    assert not theirs[region(device)].any() and device._code_installed(device.brain)
    np.testing.assert_allclose(
        device.learned_code.dictionary, host.learned_code.dictionary, rtol=0, atol=1e-9
    )
