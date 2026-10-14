"""The efference copy: a trace of the issued command, a gene with zero as its control."""

import json
from dataclasses import replace

import numpy as np
import pytest

import cadence as cd
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, motor_cortex

DRIVE = np.array([[1.0, 0.0, 0.0, 0.0]])


def composed(seed=3, echo=1.0, **options):
    genes = {} if echo is None else {"efference_amplitude": echo}
    return cd.Brain.compose(4, 2, modules=(8,), seed=seed, episodic=False, **genes, **options)


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def assert_same_checkpoint(first, second):
    a, b = saved_arrays(first), saved_arrays(second)
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


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


def test_the_founder_value_builds_the_released_composition_byte_identically(tmp_path):
    control = composed(echo=None)
    founder = composed(echo=0.0)
    assert founder.efference is None and "efference" not in founder.connectome.populations
    assert_same_checkpoint(control.save(tmp_path / "control"), founder.save(tmp_path / "founder"))
    with np.load(tmp_path / "founder.npz", allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
    assert meta["format"] == "cadence-generic/2" and "efference" not in meta


def test_the_copy_adds_only_its_neurons_and_their_projection():
    control, candidate = composed(echo=None), composed(echo=1.0)
    assert candidate.efference is not None
    populations = dict(candidate.connectome.populations)
    efference = list(populations.pop("efference"))
    assert {k: list(v) for k, v in populations.items()} == {
        k: list(v) for k, v in control.connectome.populations.items()
    }
    assert len(efference) == len(candidate.motor_index) == 2
    assert candidate.connectome.n == control.connectome.n + 2
    # Appended last: the developed synapses of the control are a subset, and the additional
    # ones all run from the efference neurons into the association region.
    old, new = synapse_rows(control.connectome), synapse_rows(candidate.connectome)
    assert old <= new
    added = new - old
    assert len(added) == 2 * len(candidate.association_index)
    assert {p for p, _, _, _ in added} == set(efference)
    assert {q for _, q, _, _ in added} == set(candidate.association_index.tolist())
    # An efference neuron receives no synapses and rests silent, so an empty copy leaves the
    # independent answer of the control unchanged.
    assert not (np.isin(np.asarray(candidate.connectome.post), efference)).any()
    np.testing.assert_array_equal(control.predict(DRIVE), candidate.predict(DRIVE))


@pytest.mark.parametrize("greedy", [True, False])
def test_an_issued_command_is_written_as_its_fading_one_hot(greedy):
    brain = composed(echo=1.0, efference_decay=0.25)
    copy = brain.efference
    assert copy.cold.shape == (0,)
    first = brain.act(DRIVE, greedy=greedy)
    one_hot = np.eye(2)[first]
    np.testing.assert_allclose(copy.trace, 0.75 * one_hot)
    np.testing.assert_array_equal(copy.last, one_hot)
    assert not copy.cold.any()
    second = brain.act(DRIVE, greedy=greedy)
    np.testing.assert_allclose(copy.trace, 0.25 * 0.75 * one_hot + 0.75 * np.eye(2)[second])
    np.testing.assert_array_equal(copy.last, np.eye(2)[second])


def test_a_sampled_command_is_the_executed_action_not_the_most_active_motor_neuron():
    brain = composed(seed=11, echo=1.0, efference_decay=0.0)
    seen = set()
    for _ in range(40):
        action = brain.act(DRIVE, temperature=50.0)
        np.testing.assert_array_equal(brain.efference.last, np.eye(2)[action])
        seen.add(int(action[0]))
    assert seen == {0, 1}  # both actions were executed and each was copied as issued


def test_slotted_actions_issue_one_command_per_slot():
    brain = cd.Brain.compose(
        4, 5, modules=(8,), slots=(2, 3), seed=3, episodic=False, efference_amplitude=1.0
    )
    action = brain.act(DRIVE, greedy=True)
    assert action.shape == (1, 2)
    expected = np.zeros((1, 5))
    expected[0, action[0, 0]] = 1.0
    expected[0, 2 + action[0, 1]] = 1.0
    np.testing.assert_array_equal(brain.efference.last, expected)
    np.testing.assert_allclose(brain.efference.trace, 0.8 * expected)


def test_the_stimulus_reads_the_copy_at_its_amplitude_and_independent_answers_ignore_it():
    brain = composed(echo=2.5, efference_decay=0.0)
    brain.act(DRIVE, greedy=True)
    columns = np.asarray(brain.connectome.populations["efference"])
    live = brain.stimulus(DRIVE)
    np.testing.assert_allclose(live[:, columns], 2.5 * brain.efference.trace)
    assert not brain.stimulus(DRIVE, memory=False)[:, columns].any()
    before = brain.efference.trace.copy()
    brain.predict(DRIVE)
    brain.accuracy(DRIVE, [0])
    np.testing.assert_array_equal(brain.efference.trace, before)


def test_the_copy_changes_the_next_answer_of_a_continuing_stream():
    # Same founder weights, same observations: the only difference between the two streams is
    # that one reads its own last command. Over identical drive the sequences diverge. They do
    # for founders 1, 7, 8, 10, 11 and 12 of 1 to 12 at the 0.82 trace defaults; founder 3,
    # which this test read before, diverged only with the 0.81 composition (retention 0.2,
    # the carried interpretation read back at scale 12).
    control, candidate = composed(seed=1, echo=None), composed(seed=1, echo=3.0)
    control_actions = [int(control.act(DRIVE, greedy=True)[0]) for _ in range(8)]
    candidate_actions = [int(candidate.act(DRIVE, greedy=True)[0]) for _ in range(8)]
    assert control_actions[0] == candidate_actions[0]  # the first answer reads an empty copy
    assert control_actions != candidate_actions


def test_feedback_resets_the_copy_of_ended_rows_and_rolls_back_a_refused_update(monkeypatch):
    brain = composed(echo=1.0)
    x = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    brain.act(x)
    kept = brain.efference.trace[1].copy()
    brain.learn(np.zeros(2), np.array([True, False]), x)
    assert not brain.efference.trace[0].any() and brain.efference.cold[0]
    np.testing.assert_array_equal(brain.efference.trace[1], kept)
    assert not brain.efference.cold[1]
    brain.act(x)
    trace, last, cold = (
        getattr(brain.efference, name).copy() for name in ("trace", "last", "cold")
    )

    def refused(*args, **kwargs):
        raise RuntimeError("the feedback update refused after the memories were touched")

    monkeypatch.setattr(brain.basal_ganglia, "learn", refused)
    with pytest.raises(RuntimeError, match="refused"):
        brain.learn(np.zeros(2), np.array([True, True]), x)
    np.testing.assert_array_equal(brain.efference.trace, trace)
    np.testing.assert_array_equal(brain.efference.last, last)
    np.testing.assert_array_equal(brain.efference.cold, cold)


def test_a_routine_moment_that_ends_the_episode_starts_the_copy_afresh(tmp_path):
    brain = cd.Brain.compose(
        4,
        2,
        modules=(8,),
        seed=5,
        working_memory_amplitude=0.0,
        efference_amplitude=1.0,
        arousal=cd.ArousalConfig(youth=0),
    )
    brain.live(DRIVE)
    brain.live(DRIVE, reward=[0.0])
    assert brain.last_arousal["mode"] == "routine"
    assert brain.efference.trace.any()
    twin = cd.Brain.load(brain.save(tmp_path / "before-terminal"))
    twin.reset()
    action = brain.live(DRIVE, reward=[0.0], done=[True])
    np.testing.assert_array_equal(action, twin.live(DRIVE))
    np.testing.assert_array_equal(brain.basal_ganglia.state.v, twin.basal_ganglia.state.v)
    for memory in ("working_memory", "efference"):
        for name in ("trace", "last", "cold"):
            np.testing.assert_array_equal(
                getattr(getattr(brain, memory), name), getattr(getattr(twin, memory), name)
            )
    # the ended row was cleared before this moment's command was issued into it
    np.testing.assert_allclose(brain.efference.trace, 0.8 * np.eye(2)[action])


@pytest.mark.parametrize("refusal", ["forecast", "arousal"])
def test_refused_terminal_routine_feedback_preserves_the_copy(tmp_path, monkeypatch, refusal):
    brain = composed(echo=3.0, arousal=cd.ArousalConfig(youth=0))
    brain.live(DRIVE)
    before = brain.save(tmp_path / "before")
    config, lived = brain.learner.config, brain._lived
    outcome = brain.arousal.outcome
    if refusal == "forecast":
        brain.learner.config = replace(config, free_steps=0)
        error, message = RuntimeError, "did not settle"
    else:
        def reject(*args, **kwargs):
            raise ValueError("arousal refused the outcome")

        monkeypatch.setattr(brain.arousal, "outcome", reject)
        error, message = ValueError, "arousal refused"
    with pytest.raises(error, match=message):
        brain.live(DRIVE, reward=[0.0], done=[True])
    assert brain._lived is lived
    brain.learner.config = config
    monkeypatch.setattr(brain.arousal, "outcome", outcome)
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    twin = cd.Brain.load(before)
    np.testing.assert_array_equal(
        brain.live(DRIVE, reward=[0.0], done=[True]),
        twin.live(DRIVE, reward=[0.0], done=[True]),
    )
    assert_same_checkpoint(brain.save(tmp_path / "retried"), twin.save(tmp_path / "twin"))


def test_reset_clears_the_copy():
    brain = composed(echo=1.0)
    brain.act(DRIVE, greedy=True)
    brain.reset()
    assert brain.efference.trace.shape == (0, 2) and brain.efference.cold.shape == (0,)


def test_imagination_issues_a_private_copy_and_leaves_the_live_one(tmp_path):
    brain = composed(seed=3, echo=3.0, efference_decay=0.0)
    brain.act(DRIVE, greedy=True)
    before = brain.save(tmp_path / "before")
    branch = brain.imagine([DRIVE, DRIVE, DRIVE])
    assert len(branch) == 3 and all(np.all(phase.qualified) for phase in branch)
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    # Zeroing the copy's read gain removes its contribution from the first imagined solve on
    # (the live copy is read there) and from every later one (the private copy is issued).
    brain.efference.amplitude = 0.0
    silent = brain.imagine([DRIVE, DRIVE, DRIVE])
    gaps = [
        float(np.max(np.abs(a.state.activation - b.state.activation)))
        for a, b in zip(branch, silent, strict=True)
    ]
    assert max(gaps) > 1e-3


def test_a_saved_copy_continues_identically_and_marks_its_format(tmp_path):
    brain = composed(seed=3, echo=3.0, efference_decay=0.3)
    for _ in range(3):
        brain.act(DRIVE, greedy=True)
    path = brain.save(tmp_path / "copy")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
        assert meta["format"] == "cadence-generic/4"
        assert meta["efference"] == brain.efference.to_dict()
        assert {"efference/trace", "efference/last", "efference/cold"} <= set(data.files)
    twin = cd.Brain.load(path)
    assert isinstance(twin.efference, cd.Efference)
    assert twin.efference.to_dict() == brain.efference.to_dict()
    for name in ("trace", "last", "cold"):
        np.testing.assert_array_equal(getattr(twin.efference, name), getattr(brain.efference, name))
    for _ in range(6):
        np.testing.assert_array_equal(brain.act(DRIVE, greedy=True), twin.act(DRIVE, greedy=True))
    assert_same_checkpoint(brain.save(tmp_path / "later"), twin.save(tmp_path / "twin"))


def test_a_life_with_the_copy_and_arousal_saves_its_pending_moment(tmp_path):
    brain = cd.Brain.compose(
        4,
        2,
        modules=(8,),
        seed=5,
        working_memory_amplitude=0.0,
        efference_amplitude=1.0,
        arousal=cd.ArousalConfig(youth=2),
    )
    brain.live(DRIVE)
    brain.live(DRIVE, reward=[1.0])
    assert brain.basal_ganglia._pending is not None
    path = brain.save(tmp_path / "life")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
    assert meta["format"] == "cadence-generic/4" and "arousal" in meta and "lived" in meta
    twin = cd.Brain.load(path)
    for moment in range(8):
        reward = [float(moment % 2)]
        np.testing.assert_array_equal(
            brain.live(DRIVE, reward=reward), twin.live(DRIVE, reward=reward)
        )
        assert brain.last_arousal == twin.last_arousal


@pytest.mark.parametrize(
    "tamper",
    [
        "format_without_copy",
        "copy_without_format",
        "missing_arrays",
        "wrong_width",
    ],
)
def test_corrupt_saved_copies_are_refused(tmp_path, tamper):
    brain = composed(seed=3, echo=1.0)
    brain.act(DRIVE, greedy=True)
    path = brain.save(tmp_path / "copy")
    arrays = saved_arrays(path)
    meta = json.loads(str(arrays["generic"]))
    if tamper == "format_without_copy":
        meta["efference"] = None
    elif tamper == "copy_without_format":
        meta["format"] = "cadence-generic/2"
    elif tamper == "missing_arrays":
        del arrays["efference/trace"]
    elif tamper == "wrong_width":
        arrays["efference/trace"] = np.zeros((1, 3))
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError):
        cd.Brain.load(tmp_path / "bad.npz")


@pytest.mark.parametrize("value", [True, -0.5, float("nan"), float("inf"), [1.0], "1"])
def test_invalid_read_gains_are_refused(value):
    with pytest.raises(ValueError):
        composed(echo=value)


def test_invalid_decays_and_commands_are_refused():
    with pytest.raises(ValueError):
        composed(echo=1.0, efference_decay=1.0)
    brain = composed(echo=1.0)
    with pytest.raises(ValueError):
        brain.efference.issue(np.ones((1, 3)))
    with pytest.raises(ValueError):
        brain.efference.issue(np.array([[1.0, float("nan")]]))
    brain.act(DRIVE, greedy=True)
    with pytest.raises(TypeError):
        brain.efference.update(brain.basal_ganglia.state)


def test_a_custom_connectome_needs_one_efference_neuron_per_motor_neuron():
    regions = (
        Region("sensory", 2),
        Region("association", 4),
        motor_cortex(2),
        Region("efference", 3),
    )
    projections = (
        Projection("sensory", "association", reciprocal=False),
        Projection("association", "motor"),
        Projection("efference", "association", reciprocal=False),
    )
    with pytest.raises(ValueError):
        cd.Brain(develop(Genome(regions, projections)), episodic=False)


def alternation_after_teaching(seed, echo, bouts=24, events=12, window=48):
    """Four streams under identical drive are taught to answer the opposite of their own last
    action after a one-event cue; the free window then counts changes between successive
    actions. The steady-rhythm chamber is the frozen instrument; this is its smallest witness."""
    genes = {"efference_amplitude": echo, "efference_decay": 0.0} if echo else {}
    brain = cd.Brain.compose(
        4,
        2,
        modules=(32,),
        seed=seed,
        episodic=False,
        working_memory_decay=0.1,
        working_memory_amplitude=3.0,
        working_memory_scale=12.0,  # the steady-rhythm chamber's composition
        **genes,
    )
    rng = np.random.default_rng(seed)
    rows = np.arange(4)
    previous = np.zeros(4, dtype=np.int64)
    for _ in range(bouts):
        cues = rng.permutation([0, 0, 1, 1])
        for t in range(events):
            x = np.zeros((4, 4))
            x[:, 0] = 1.0
            if t == 0:
                x[rows, 2 + cues] = 1.0
            labels = cues if t == 0 else 1 - previous
            brain.learner.step(brain.stimulus(x), labels)
            previous = brain.act(x, greedy=True)
    cues = rng.permutation([0, 0, 1, 1])
    x = np.zeros((4, 4))
    x[:, 0] = 1.0
    x[rows, 2 + cues] = 1.0
    actions = [brain.act(x, greedy=True)]
    x = np.zeros((4, 4))
    x[:, 0] = 1.0
    actions.extend(brain.act(x, greedy=True) for _ in range(window))
    actions = np.stack(actions)
    return float(np.mean(actions[1:] != actions[:-1]))


def test_the_copy_makes_alternation_under_identical_drive_learnable():
    # Development seed 5 of the steady-rhythm chamber: the unchanged brain holds or wobbles,
    # the same founder with the copy alternates. The chamber's receipt gives the five fresh
    # seeds; this test pins one founder so the mechanism cannot silently regress.
    with_copy = alternation_after_teaching(5, 3.0)
    without = alternation_after_teaching(5, None)
    assert with_copy >= 0.9
    assert without < with_copy
