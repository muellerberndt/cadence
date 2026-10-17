"""The heard history: the observations as heard, carried at several timescales in a region of
their own; ``history_decays=()`` is its founder and control."""

import json

import numpy as np
import pytest

import cadence as cd
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, motor_cortex

CUE = np.array([[1.0, 0.0, 0.5, 0.0]])
GO = np.array([[0.0, 0.0, 0.0, 1.0]])


def composed(seed=3, decays=(0.2, 0.8), **options):
    genes = {} if decays is None else {"history_decays": decays}
    return cd.Brain.compose(4, 2, modules=(8,), seed=seed, episodic=False, **genes, **options)


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def assert_same_checkpoint(first, second):
    a, b = saved_arrays(first), saved_arrays(second)
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def synapse_rows(connectome):
    return {
        (int(p), int(q), int(c), float(s))
        for p, q, c, s in zip(
            np.asarray(connectome.pre),
            np.asarray(connectome.post),
            np.asarray(connectome.count),
            np.asarray(connectome.sign),
            strict=True,
        )
    }


def literal(decays, observations, width=4):
    """The recurrence of every block, written out: h <- d h + (1 - d) x after each act."""
    blocks = [np.zeros((1, width)) for _ in decays]
    for x in observations:
        blocks = [d * h + (1.0 - d) * x for d, h in zip(decays, blocks, strict=True)]
    return np.concatenate(blocks, axis=1)


@pytest.mark.parametrize("options", [{}, {"arousal": True}, {"efference_amplitude": 1.0}])
def test_the_founder_builds_the_composition_without_a_history_byte_identically(tmp_path, options):
    control = composed(decays=None, **options)
    founder = composed(decays=(), **options)
    assert founder.history is None and "history" not in founder.connectome.populations
    for brain in (control, founder):
        brain.act(CUE, greedy=True)
    assert_same_checkpoint(control.save(tmp_path / "control"), founder.save(tmp_path / "founder"))
    with np.load(tmp_path / "founder.npz", allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
    assert meta["format"] != "cadence-generic/7" and "history" not in meta
    assert "history_scale" not in meta["composition"]
    described = founder.describe()
    assert described["history"] is None and "history_decays" not in described["genes"]
    assert described == control.describe()


def test_the_history_adds_only_its_blocks_and_their_projection():
    control, candidate = composed(decays=None), composed(decays=(0.2, 0.8, 0.95))
    populations = dict(candidate.connectome.populations)
    history = list(populations.pop("history"))
    assert {k: list(v) for k, v in populations.items()} == {
        k: list(v) for k, v in control.connectome.populations.items()
    }
    assert len(history) == 3 * 4 and candidate.connectome.n == control.connectome.n + 12
    # Appended last: the control's synapses are a subset, and the added ones all run from the
    # history neurons into the association region.
    old, new = synapse_rows(control.connectome), synapse_rows(candidate.connectome)
    assert old <= new
    added = new - old
    assert len(added) == 12 * len(candidate.association_index)
    assert {p for p, _, _, _ in added} == set(history)
    assert {q for _, q, _, _ in added} == set(candidate.association_index.tolist())
    # A history neuron receives no synapses; a silent history leaves the independent answer.
    assert not np.isin(np.asarray(candidate.connectome.post), history).any()
    np.testing.assert_array_equal(control.predict(CUE), candidate.predict(CUE))


def test_the_projection_starts_at_its_scale_like_the_working_trace():
    quiet = composed(decays=(0.5,), history_scale=1.2)
    loud = composed(decays=(0.5,), history_scale=12.0)
    history = np.asarray(quiet.connectome.populations["history"])
    rows = np.isin(np.asarray(quiet.connectome.pre), history)
    np.testing.assert_allclose(
        np.asarray(loud.connectome.sign)[rows], 10.0 * np.asarray(quiet.connectome.sign)[rows]
    )
    assert quiet.describe()["initialization"]["composition"]["history_scale"] == 1.2


@pytest.mark.parametrize("greedy", [True, False])
def test_each_block_traces_the_observations_acted_on_at_its_own_decay(greedy):
    decays = (0.0, 0.2, 0.8, 0.95)
    brain = composed(decays=decays, arousal=True)
    stream = [CUE, GO, np.array([[0.0, 1.0, 0.0, 0.25]]), GO, GO]
    assert brain.history.trace.shape == (0, 16)
    for t, x in enumerate(stream):
        # at the decision the history holds the past only; the drive holds the present
        if t:
            np.testing.assert_allclose(brain.history.trace, literal(decays, stream[:t]))
        brain.act(x, greedy=greedy)
        np.testing.assert_allclose(brain.history.trace, literal(decays, stream[: t + 1]))
    # the block of decay zero is exactly the last observation acted on
    np.testing.assert_array_equal(brain.history.trace[:, :4], stream[-1])


def test_streams_keep_their_own_histories_and_the_stimulus_reads_them_at_the_amplitude():
    brain = composed(decays=(0.5,), history_amplitude=2.5)
    pasts = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    brain.act(pasts, greedy=True)
    present = np.repeat(GO, 3, axis=0)
    columns = np.asarray(brain.connectome.populations["history"])
    np.testing.assert_allclose(brain.history.trace, 0.5 * pasts)
    live = brain.stimulus(present)
    np.testing.assert_allclose(live[:, columns], 2.5 * 0.5 * pasts)
    assert not brain.stimulus(present, memory=False)[:, columns].any()
    # identical present inputs settle apart because the pasts differ
    state = brain._qualified(live, brain._activity())
    association = np.asarray(state.activation)[:, brain.association_index]
    assert np.min(np.linalg.norm(association[:, None] - association[None], axis=2)[
        np.triu_indices(3, 1)
    ]) > 1e-3
    before = brain.history.trace.copy()
    brain.predict(present)
    brain.accuracy(present, [0, 1, 0])
    np.testing.assert_array_equal(brain.history.trace, before)
    # another batch size reads silence without erasing the live streams
    assert not brain.history.stimulate(np.zeros((2, brain.connectome.n)))[:, columns].any()
    np.testing.assert_array_equal(brain.history.trace, before)


def test_reset_and_ended_rows_begin_in_silence_and_a_refused_update_rolls_back(monkeypatch):
    brain = composed(decays=(0.5,))
    x = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    brain.act(x)
    kept = brain.history.trace[1].copy()
    brain.learn(np.zeros(2), np.array([True, False]), x)
    assert not brain.history.trace[0].any()
    np.testing.assert_array_equal(brain.history.trace[1], kept)
    brain.act(x)
    before = brain.history.trace.copy()

    def refused(*args, **kwargs):
        raise RuntimeError("the feedback update refused after the memories were touched")

    monkeypatch.setattr(brain.basal_ganglia, "learn", refused)
    with pytest.raises(RuntimeError, match="refused"):
        brain.learn(np.zeros(2), np.array([True, True]), x)
    np.testing.assert_array_equal(brain.history.trace, before)
    brain.reset()
    assert brain.history.trace.shape == (0, 4)


def test_a_routine_moment_that_ends_the_episode_starts_the_history_afresh(tmp_path):
    brain = composed(decays=(0.5,), working_memory_amplitude=0.0, arousal=cd.ArousalConfig(youth=0))
    brain.live(CUE)
    brain.live(GO, reward=[0.0])
    assert brain.last_arousal["mode"] == "routine" and brain.history.trace.any()
    twin = cd.Brain.load(brain.save(tmp_path / "before-terminal"))
    twin.reset()
    action = brain.live(GO, reward=[0.0], done=[True])
    np.testing.assert_array_equal(action, twin.live(GO))
    np.testing.assert_array_equal(brain.history.trace, twin.history.trace)
    np.testing.assert_allclose(brain.history.trace, 0.5 * GO)


def test_wait_hears_the_sensed_observation(tmp_path):
    brain = composed(decays=(0.5,), arousal=True)
    brain.live(CUE)
    brain.wait(GO)
    np.testing.assert_allclose(brain.history.trace, 0.25 * CUE + 0.5 * GO)
    twin = cd.Brain.load(brain.save(tmp_path / "waiting"))
    np.testing.assert_array_equal(twin.history.trace, brain.history.trace)
    np.testing.assert_array_equal(
        brain.live(GO, reward=[1.0]), twin.live(GO, reward=[1.0])
    )


def test_imagination_hears_privately_and_leaves_the_live_history(tmp_path):
    brain = composed(decays=(0.0,), history_amplitude=3.0)
    brain.act(CUE, greedy=True)
    before = brain.save(tmp_path / "before")
    branch = brain.imagine([GO, GO, GO])
    assert len(branch) == 3 and all(np.all(phase.qualified) for phase in branch)
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    # a silent history changes the first imagined solve (the live history is read there)
    brain.history.amplitude = 0.0
    silent = brain.imagine([GO, GO, GO])
    gaps = [
        float(np.max(np.abs(a.state.activation - b.state.activation)))
        for a, b in zip(branch, silent, strict=True)
    ]
    assert gaps[0] > 1e-6


@pytest.mark.parametrize("episodic", [False, True])
def test_a_saved_history_continues_identically_and_marks_its_format(tmp_path, episodic):
    options = {"memory_choice": True} if episodic else {}
    brain = cd.Brain.compose(
        4, 2, modules=(8,), seed=5, history_decays=(0.2, 0.9), arousal=True,
        efference_amplitude=1.0, **options,
    )
    for t in range(4):
        brain.live(CUE if t % 2 else GO, reward=None if t == 0 else [float(t % 3 == 0)])
    path = brain.save(tmp_path / "life")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
        assert meta["format"] == "cadence-generic/7"
        assert meta["history"] == brain.history.to_dict()
        assert ("episodic" in meta) == episodic and "efference" in meta and "arousal" in meta
        assert "history/trace" in data.files
    twin = cd.Brain.load(path)
    assert isinstance(twin.history, cd.History)
    assert twin.describe() == brain.describe()
    for t in range(8):
        x, reward = (CUE if t % 3 else GO), [float(t % 2)]
        np.testing.assert_array_equal(brain.live(x, reward=reward), twin.live(x, reward=reward))
        np.testing.assert_array_equal(brain.history.trace, twin.history.trace)
    assert_same_checkpoint(brain.save(tmp_path / "later"), twin.save(tmp_path / "twin"))


@pytest.mark.parametrize(
    "tamper",
    [
        "format_without_history",
        "history_without_format",
        "missing_trace",
        "wrong_width",
        "wrong_blocks",
        "bad_decay",
        "composition_without_scale",
    ],
)
def test_corrupt_saved_histories_are_refused(tmp_path, tamper):
    brain = composed(decays=(0.2, 0.8))
    brain.act(CUE, greedy=True)
    arrays = saved_arrays(brain.save(tmp_path / "history"))
    meta = json.loads(str(arrays["generic"]))
    if tamper == "format_without_history":
        del meta["history"]
    elif tamper == "history_without_format":
        meta["format"] = "cadence-generic/2"
    elif tamper == "missing_trace":
        del arrays["history/trace"]
    elif tamper == "wrong_width":
        arrays["history/trace"] = np.zeros((1, 7))
    elif tamper == "wrong_blocks":
        meta["history"]["decays"] = [0.2]
    elif tamper == "bad_decay":
        meta["history"]["decays"] = [0.2, 1.0]
    elif tamper == "composition_without_scale":
        del meta["composition"]["history_scale"]
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError):
        cd.Brain.load(tmp_path / "bad.npz")


def test_the_amplitude_retunes_and_the_blocks_are_construction_settings():
    brain = composed(decays=(0.5,), arousal=True)
    brain.live(CUE)
    brain.retune(history_amplitude=1.5)
    assert brain.history.amplitude == 1.5 and brain.describe()["genes"]["history_amplitude"] == 1.5
    for name, value in (("history_decays", (0.1,)), ("history_scale", 2.0)):
        with pytest.raises(ValueError, match="construction setting"):
            brain.retune(**{name: value})
    with pytest.raises(ValueError):
        brain.retune(history_amplitude=-1.0)
    with pytest.raises(ValueError, match="existing memory component"):
        composed(decays=None).retune(history_amplitude=1.0)


@pytest.mark.parametrize(
    "decays", [(1.0,), (-0.1,), (float("nan"),), (True,), ("0.5",), [[0.2]], 0.5, "0.2"]
)
def test_invalid_decays_are_refused(decays):
    with pytest.raises(ValueError):
        composed(decays=decays)


@pytest.mark.parametrize("value", [True, -0.5, float("nan"), float("inf"), [1.0], "1"])
def test_invalid_amplitudes_are_refused(value):
    with pytest.raises(ValueError):
        composed(decays=(0.5,), history_amplitude=value)


def test_a_custom_connectome_needs_one_block_of_sensory_width_per_decay():
    def graph(width):
        regions = (
            Region("sensory", 2),
            Region("association", 4),
            motor_cortex(2),
            Region("history", width),
        )
        projections = (
            Projection("sensory", "association", reciprocal=False),
            Projection("association", "motor"),
            Projection("history", "association", reciprocal=False),
        )
        return develop(Genome(regions, projections))

    assert cd.Brain(graph(4), episodic=False, history_decays=(0.2, 0.8)).history is not None
    with pytest.raises(ValueError):
        cd.Brain(graph(3), episodic=False, history_decays=(0.2, 0.8))
    with pytest.raises(ValueError):
        cd.Brain(graph(4), episodic=False)
    with pytest.raises(ValueError):
        composed(decays=None).__class__(
            composed(decays=None).connectome, episodic=False, history_decays=(0.5,)
        )


def test_the_resting_bias_leaves_the_history_neurons_at_rest():
    brain = composed(decays=(0.5,), resting_bias=0.5)
    history = np.asarray(brain.connectome.populations["history"])
    assert not np.asarray(brain.brain.bias)[history].any()
    assert np.asarray(brain.brain.bias)[brain.association_index].min() == 0.5


def test_a_surprise_lesson_reads_the_history_its_act_reads_and_the_act_then_hears():
    brain = composed(decays=(0.5,), learning_surprise=True)
    brain.act(CUE, greedy=True)
    drive = brain.stimulus(GO)
    seen = {}
    step = brain.learner.step

    def spy(drive_, labels, **kwargs):
        seen["drive"] = drive_.copy()
        return step(drive_, labels, **kwargs)

    brain.learner.step = spy
    brain.step(GO, teacher=[1])
    np.testing.assert_array_equal(seen["drive"], drive)
    np.testing.assert_allclose(brain.history.trace, 0.25 * CUE + 0.5 * GO)
