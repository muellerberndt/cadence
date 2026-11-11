"""Behavioral contracts for composing the recovered mechanisms."""

import numpy as np
import pytest

from cadence import Brain, SynapticMemory, Trace


def assert_same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


@pytest.mark.parametrize("observers", [(), (3,), (3, 2)])
def test_modular_base_and_optional_observers_share_one_connected_brain(observers):
    brain = Brain.compose(2, 2, modules=(5, 4), observers=observers, seed=3, recall=False)
    assert isinstance(brain.working_memory, Trace)
    assert isinstance(brain.hippocampus, SynapticMemory)
    graph = brain.connectome
    neighbors = [set() for _ in range(graph.n)]
    edges = set(zip(graph.pre.tolist(), graph.post.tolist(), strict=True))
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    reached, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node not in reached:
            reached.add(node)
            pending.extend(neighbors[node] - reached)
    assert len(reached) == graph.n
    # Ordinary modular depth is present without observation. It is not all-to-all.
    assert not any(
        (a, b) in edges for a in graph.populations["module_0"] for b in graph.populations["motor"]
    )
    for index in range(len(observers)):
        members = graph.populations[f"observer_{index}"]
        assert all(
            (a, b) in edges and (b, a) in edges
            for a in members
            for b in graph.populations["association"]
        )
    phase = brain.imagine([[[0.2, -0.1]]])[0]
    assert np.all(phase.qualified)
    drive = brain.stimulus([[0.2, -0.1]])
    np.testing.assert_allclose(phase.residual, brain.brain.residual(drive, phase.state))


def test_observer_feedback_changes_the_base_inside_the_same_solve():
    brain = Brain.compose(2, 2, modules=(5, 4), observers=(3,), seed=3, recall=False)
    drive = brain.stimulus([[0.2, -0.1]])
    ordinary = brain.brain.equilibrate(drive, budget=2048, tolerance=1e-8)
    perturbed = drive.copy()
    perturbed[:, brain.connectome.populations["observer_0"]] += 0.1
    feedback = brain.brain.equilibrate(perturbed, budget=2048, tolerance=1e-8)
    assert np.all(ordinary.qualified) and np.all(feedback.qualified)
    base = brain.connectome.populations["association"]
    assert (
        np.max(np.abs(ordinary.state.activation[:, base] - feedback.state.activation[:, base]))
        > 1e-4
    )


def test_imagination_preserves_memories_randomness_pending_outcomes_and_continuation(tmp_path):
    brain = Brain.compose(2, 2, modules=(5, 4), observers=(3,), seed=3, recall=False)
    brain.hippocampus.observe(np.array([[1.0, 0.0]]), np.array([[0.7, -0.2]]))
    brain.act([[0.2, -0.1]])  # A real action is still awaiting its outcome.
    before = brain.save(tmp_path / "before")
    control = Brain.load(before)
    sequence = [[[0.0, 0.1]], [[-0.1, 0.0]]]
    branch = brain.imagine(sequence)
    assert len(branch) == 2 and all(np.all(phase.qualified) for phase in branch)
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    repeated = brain.imagine(sequence)
    for first, second in zip(branch, repeated, strict=True):
        np.testing.assert_array_equal(first.state.activation, second.state.activation)
    # Real feedback still belongs to the preceding real action, with identical learning.
    reward, done, following = np.array([0.4]), np.array([False]), [[0.1, 0.2]]
    brain.learn(reward, done, following)
    control.learn(reward, done, following)
    np.testing.assert_array_equal(brain.act(following), control.act(following))
    assert_same_checkpoint(brain.save(tmp_path / "continued"), control.save(tmp_path / "control"))


def test_private_trace_is_used_between_imagined_steps_without_becoming_real_memory():
    brain = Brain.compose(2, 2, modules=(5, 4), seed=3, recall=False)
    sequence = [[[0.6, -0.3]], [[0.0, 0.0]]]
    branch = brain.imagine(sequence)
    assert len(branch) == 2 and all(np.all(phase.qualified) for phase in branch)
    assert brain.working_memory.trace.shape[0] == 0
    # Zeroing the trace's read gain removes its causal contribution to the second solve.
    brain.working_memory.amplitude = 0
    erased = brain.imagine(sequence)
    np.testing.assert_array_equal(branch[0].state.activation, erased[0].state.activation)
    assert np.max(np.abs(branch[1].state.activation - erased[1].state.activation)) > 1e-3


def test_refused_imagined_step_ends_branch_and_leaves_real_state_untouched(tmp_path):
    brain = Brain.compose(2, 2, modules=(5, 4), seed=3, recall=False)
    before = brain.save(tmp_path / "before")
    branch = brain.imagine([[[0.2, 0.1]], [[0.5, 0.3]]], budget=0)
    assert len(branch) == 1 and not np.all(branch[0].qualified)
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    with pytest.raises(ValueError, match="stream identities"):
        brain.imagine([[[0.1, 0.2]], [[0.1, 0.2], [0.2, 0.1]]])
    assert_same_checkpoint(before, brain.save(tmp_path / "invalid"))


@pytest.mark.parametrize(
    "modules,observers", [((), ()), ((0,), ()), ((True,), ()), ((4,), (-1,)), (4, ()), ((4,), "2")]
)
def test_invalid_composition_is_refused(modules, observers):
    with pytest.raises(ValueError):
        Brain.compose(2, 2, modules=modules, observers=observers, recall=False)


@pytest.mark.parametrize("controls", [{"budget": -1}, {"budget": True}])
def test_empty_imagined_branch_still_validates_solver_controls(controls):
    brain = Brain.compose(2, 2, modules=(4,), recall=False)
    with pytest.raises(ValueError):
        brain.imagine([], **controls)


def test_imagine_settles_to_the_learners_tolerance():
    """The one tolerance is the learner's: ``imagine`` takes none of its own (the argument of
    0.92.1 is an unknown keyword), and its phases are held to ``learning_tolerance``."""
    brain = Brain.compose(2, 2, modules=(4,), learning_tolerance=0.01, recall=False)
    with pytest.raises(TypeError, match="tolerance"):
        brain.imagine([[[1.0, 0.0]]], tolerance=1e-6)
    phases = brain.imagine([[[1.0, 0.0]], [[0.0, 1.0]]])
    assert len(phases) == 2 and all(phase.tolerance == 0.01 for phase in phases)
    assert all(phase.qualified.all() for phase in phases)


@pytest.mark.parametrize("options", [{"seed": True}, {"seed": -1}, {"episodic": "yes"}])
def test_invalid_composition_options_are_refused(options):
    with pytest.raises(ValueError):
        Brain.compose(2, 2, modules=(4,), **{"recall": False, **options})


def test_learning_surprise_gene_teaches_only_the_rows_the_brain_answers_differently() -> None:
    brain = Brain.compose(4, 2, modules=(8,), seed=0, learning_surprise=True, recall=False)
    assert brain.learner.config.surprise is True
    x = np.random.default_rng(0).random((3, 4))
    labels = brain.predict(x)
    brain.step(x, teacher=labels)
    report = brain.last_learning
    assert report["demonstrations"] == 3.0
    assert report["demonstration_accepted_presentations"] == 0.0
    assert report["demonstration_agreed_presentations"] == 3.0
    brain.reset()
    brain.step(x, teacher=1 - labels)
    report = brain.last_learning
    assert report["demonstration_accepted_presentations"] == 3.0
    assert report["demonstration_agreed_presentations"] == 0.0
    assert Brain.compose(4, 2, modules=(8,), seed=0, recall=False).learner.config.surprise is False


def test_learning_surprise_judges_from_the_brains_own_answer(monkeypatch) -> None:
    brain = Brain.compose(4, 2, modules=(8,), seed=0, learning_surprise=True, recall=False)
    x = np.random.default_rng(1).random((3, 4))
    labels = np.array([0, 1, 0])
    read = []

    def choices(state):  # the answer the act would read, here fixed by the test
        read.append(len(state.activation))
        return np.array([0, 1, 1])

    monkeypatch.setattr(brain, "_choices", choices)
    brain.step(x, teacher=labels)
    assert read == [3]  # the judgment settle; the act's choice is the basal ganglia's
    assert brain.last_learning["demonstration_accepted_presentations"] == 1.0
    assert brain.last_learning["demonstration_agreed_presentations"] == 2.0
    assert brain.last_learning["demonstration_free_row_sweeps"] == (
        brain.last_learning["demonstration_free_steps"]
    )


def test_compose_wraps_a_custom_anatomy_with_the_composed_founders():
    """``Genome.composed`` is the layout ``compose`` develops; ``compose(connectome=...)``
    gives a developed anatomy the composed founders, the episodic founders filled in, and
    refuses layout settings and populations that do not fit the inputs and actions."""
    from cadence import Genome, develop

    options = dict(memory_choice=True, recall=True, recall_founder=True, seed=3)
    composed = Brain.compose(6, 3, modules=(8,), **{"recall": False, **options})
    genome = Genome.composed(6, 3, modules=(8,), **options)
    wrapped = Brain.compose(6, 3, connectome=develop(genome, seed=3), **options)
    np.testing.assert_array_equal(wrapped.brain.efficacy, composed.brain.efficacy)
    np.testing.assert_array_equal(wrapped.brain.bias, composed.brain.bias)
    assert wrapped.describe()["memories"].get("episodic") == composed.describe()["memories"].get("episodic")
    plain = Brain(develop(genome, seed=3), memory_choice=True, recall=True, recall_founder=True, seed=3)
    # only compose fills them in
    assert plain.memories["episodic"].share == 0.0 < wrapped.memories["episodic"].share
    x = np.eye(6)[[1]]
    for _ in range(3):
        np.testing.assert_array_equal(wrapped.step(x), composed.step(x))
    assert wrapped.describe()["layout"]["composition"] is None  # its own anatomy
    with pytest.raises(ValueError, match="own layout"):
        Brain.compose(6, 3, connectome=develop(genome, seed=3), modules=(8,))
    with pytest.raises(ValueError, match="sensory and motor"):
        Brain.compose(7, 3, connectome=develop(genome, seed=3))
    assert Genome.observation_width(6) == 6 and Genome.observation_width(2048, 1024) == 1024
    assert not hasattr(Brain, "build") and not hasattr(Brain, "genome")
