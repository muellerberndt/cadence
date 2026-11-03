"""The episodic memory's complete greedy policy and the outcome boundaries of its replay.

With ``memory_choice`` a greedy act is the readout with the episodic vote. A lesson under
``learning_surprise`` judges a row by that act, ``live`` decides from it whether an issued action
was the brain's own best guess, and an imagined branch issues it to its efference copy and the
recall memory's predecessor key. A replay path keeps a decision whose outcome was a reward or a
punishment when the next decision is taken in the same situation, so a later reward replays only
the decisions taken since that outcome.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from checkpoints import metadata

import cadence as cd


def episodic(*args, frozen: bool = False, **options) -> cd.Brain:
    """``Brain.compose`` with the episodic memory; ``frozen``: the cortex learns nothing from
    lessons (the composed teaching founders at step zero)."""
    brain = cd.Brain.compose(*args, **{"memory_choice": True, **options})
    if frozen:
        brain.retune(learning=replace(brain.learner.config, eta=0.0, eta_bias=0.0))
    return brain


def one_hot(index: int, width: int) -> np.ndarray:
    x = np.zeros((1, width))
    x[0, index] = 1.0
    return x


def assert_same_checkpoint(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def path(brain: cd.Brain) -> list[tuple[int, float, bool]]:
    """The stream's replay path: each decision's observation, its outcome and whether the
    outcome was received."""
    return [(int(np.argmax(entry[2])), entry[3], entry[4]) for entry in brain.memories["episodic"].paths[0]]


def cued(seed: int, **options) -> tuple[cd.Brain, dict[int, np.ndarray], np.ndarray]:
    """A brain shown each cue's answer once at the go moment after it. Its cortex learns
    nothing from the lessons (``frozen``): the records alone carry them."""
    brain = episodic(3, 2, modules=(8,), seed=seed, frozen=True, **options)
    cues, go = {0: one_hot(0, 3), 1: one_hot(1, 3)}, one_hot(2, 3)
    for cue, shown in cues.items():
        brain.reset()
        brain.step(shown, teacher=[cue])
        brain.step(go, reward=[0.0], teacher=[cue])
    brain.reset()
    return brain, cues, go


def misled() -> tuple[cd.Brain, np.ndarray, int]:
    """A brain whose memory turns its cortex's answer into the other act: one demonstration of
    the other action is recorded, and the cortex learns nothing from it."""
    x = one_hot(0, 3)
    brain = episodic(3, 2, modules=(8,), seed=3, learning_surprise=True, frozen=True)
    cortical = int(brain.act(x, greedy=True)[0])
    brain.reset()
    brain.step(x, teacher=[1 - cortical])
    brain.reset()
    return brain, x, cortical


# -- one calculation of the act a greedy brain issues


@pytest.mark.parametrize(
    "options,teacher",
    [
        ({"memory_choice": False}, [1]),
        ({}, [1]),
        ({"memory_context": 0.0}, [1]),
        ({"slots": 2}, [[1, 0]]),
        ({"slots": (1, 3)}, [[0, 2]]),
    ],
)
def test_the_complete_greedy_choice_is_the_act_a_greedy_brain_issues(options, teacher):
    brain = episodic(3, 4, modules=(8,), seed=2, **options)
    for t in range(3):
        brain.step(one_hot(t, 3), reward=None if t == 0 else [float(t == 2)], teacher=teacher)
    for t in range(6):
        x = one_hot(t % 3, 3)
        _, free = brain._settled(x)
        expected = brain._greedy(free, x, brain._decision_key(x))
        np.testing.assert_array_equal(brain.act(x, greedy=True), expected)


# -- the surprise lesson judges by the act the brain would issue


def test_a_surprise_lesson_corrects_an_act_the_memory_made_wrong(tmp_path):
    brain, x, cortical = misled()
    assert int(brain.act(x, greedy=True)[0]) == 1 - cortical  # the memory overrules the cortex
    brain.reset()
    saved = brain.save(tmp_path / "misled.npz")
    writes = brain.hippocampus.writes, brain.context_memory.writes
    report = brain.lesson(x, [cortical])  # the right answer, which the cortex already gives
    assert report["accepted_presentations"] == 1.0 and report["agreed_presentations"] == 0.0
    assert (brain.hippocampus.writes, brain.context_memory.writes) == (writes[0] + 1, writes[1] + 1)
    # the lesson every row gets without the rule corrects the memory alike
    control, resumed = cd.Brain.load(saved), cd.Brain.load(saved)
    control.retune(learning_surprise=False)
    control.lesson(x, [cortical])
    assert resumed.lesson(x, [cortical]) == report  # the saved brain takes the same lesson
    for other in (control, resumed):
        for name in ("hippocampus", "context_memory"):
            np.testing.assert_array_equal(getattr(other, name).strength,
                                          getattr(brain, name).strength)
    for subject in (brain, control, resumed):
        subject.reset()
        assert int(subject.act(x, greedy=True)[0]) == cortical


def test_an_act_the_memory_makes_right_is_no_surprise_and_gets_no_lesson():
    brain, x, cortical = misled()
    shown = 1 - cortical  # what the brain does, by its memory against its cortex
    brain.retune(teaching="founder")  # the cortex could learn again
    graph = brain.brain
    parameters = [np.array(getattr(graph, name), copy=True) for name in ("efficacy", "bias")]
    writes = brain.hippocampus.writes, brain.context_memory.writes
    report = brain.lesson(x, [shown])
    assert report["accepted_presentations"] == 0.0 and report["agreed_presentations"] == 1.0
    # consolidating the cortex under an answer the memory supports is not surprise teaching
    for name, before in zip(("efficacy", "bias"), parameters, strict=True):
        np.testing.assert_array_equal(getattr(brain.brain, name), before)
    assert (brain.hippocampus.writes, brain.context_memory.writes) == writes
    assert int(brain._choices(brain._settled(x)[1])[0]) == cortical


def test_a_surprise_lesson_agrees_exactly_where_the_brain_would_act_as_shown(tmp_path):
    brain, cues, go = cued(4)
    brain.retune(learning_surprise=True)
    decided = 0
    for cue, shown in cues.items():
        for label in (0, 1):
            brain.reset()
            brain.act(shown, greedy=True)  # the cue: the context of the decision at go
            copy = cd.Brain.load(brain.save(tmp_path / f"cue{cue}-{label}.npz"))
            act = int(copy.act(go, greedy=True)[0])
            decided += act != int(copy._choices(copy.basal_ganglia.state)[0])
            report = brain.lesson(go, [label])
            assert report["agreed_presentations"] == float(act == label)
            assert report["accepted_presentations"] == float(act != label)
    assert decided  # the vote, under the decision's own context, decides some of these acts


# -- an outcome of a greedy act the vote decided is the brain's own


def test_a_greedy_act_the_vote_decided_is_owned_and_its_outcome_can_surprise(tmp_path):
    x = one_hot(0, 2)
    brain = episodic(2, 2, modules=(8,), seed=0, arousal={"youth": 0})
    cortical = int(brain.act(x, greedy=True)[0])
    brain.reset()
    brain.step(x, teacher=[1 - cortical])
    brain.reset()
    assert int(brain.live(x)[0]) == 1 - cortical  # a calm, greedy act the vote decided
    assert brain._lived[5]
    saved = brain.save(tmp_path / "issued.npz")
    with np.load(saved, allow_pickle=False) as data:
        assert metadata(data)["lived"]["own"] is True
    copy = cd.Brain.load(saved)
    assert copy._lived[5]
    writes = brain.hippocampus.writes
    for subject in (brain, copy):
        for reward in [0.0] * 5 + [1.0]:  # as forecast five times, then far beyond it
            assert subject._lived[5]
            subject.live(x, reward=[reward])
        assert subject.arousal.outcomes == 6
        reading = subject.last_arousal
        assert reading["mode"] == "aroused" and reading["surprise"] > 0 and reading["recorded"]
        assert subject.hippocampus.writes == writes + 1  # the outcome that woke it
    assert dict(copy.last_arousal) == dict(brain.last_arousal)
    for name in ("hippocampus", "context_memory"):
        np.testing.assert_array_equal(getattr(copy, name).strength, getattr(brain, name).strength)


def test_an_aroused_brain_owns_the_samples_of_its_complete_greedy_choice():
    x = one_hot(0, 2)
    # place records alone, and nothing learned or written while it lives: the vote stays
    brain = episodic(2, 2, modules=(8,), seed=0, memory_context=0.0, arousal={"youth": 1000},
                     frozen=True, actor_eta=0, actor_eta_bias=0, actor_eta_critic=0)
    cortical = int(brain.act(x, greedy=True)[0])
    brain.reset()
    brain.step(x, teacher=[1 - cortical])
    brain.reset()
    writes = brain.hippocampus.writes
    own, overruled = [], 0
    for t in range(30):
        action = int(brain.live(x, reward=None if t == 0 else [0.0])[0])
        assert brain.last_arousal["mode"] == "aroused"
        state = brain.basal_ganglia.state
        # the most probable action of the readout with the place records' vote
        greedy = int(np.argmax(brain.basal_ganglia.probabilities(state, bias=brain.memories["episodic"].vote(x, None))))
        overruled += greedy != int(brain._choices(state)[0])
        assert brain._lived[5] == (action == greedy)
        assert int(brain._greedy(state, x, None)[0]) == greedy
        own.append(brain._lived[5])
    assert brain.hippocampus.writes == writes
    assert overruled and True in own and False in own
    assert brain.arousal.outcomes == sum(own[:-1])  # the last action's outcome is still to come


# -- an imagined branch issues the act the brain would issue


def test_an_imagined_branch_issues_the_act_the_brain_would_issue(tmp_path, monkeypatch):
    brain, cues, go = cued(4, efference_amplitude=1.0)
    issued = []
    release = cd.Efference.issue

    def issue(self, command):
        issued.append(int(np.argmax(command[0])))
        release(self, command)

    monkeypatch.setattr(cd.Efference, "issue", issue)
    config = brain.learner.config
    before = brain.save(tmp_path / "before.npz")
    for shown in cues.values():
        sequence = [shown, go, go]
        issued.clear()
        phases = brain.imagine(sequence, budget=config.free_steps)
        imagined = list(issued)
        cortex = [int(np.argmax(phase.state.activation[0, brain.motor_index])) for phase in phases]
        live = cd.Brain.load(before)
        acts, states = [], []
        for x in sequence:
            acts.append(int(live.act(x, greedy=True)[0]))
            states.append(live.basal_ganglia.state.activation)
        # the vote, read under the branch's own temporal context, decides as it decides the acts
        assert imagined == acts and cortex != acts
        for phase, state in zip(phases, states, strict=True):
            # the private copy carries the act issued, so the branch settles as the stream does
            np.testing.assert_array_equal(phase.state.activation, state)
    assert_same_checkpoint(before, brain.save(tmp_path / "after.npz"))  # nothing real changed


# -- a known outcome bounds the replay, whatever follows it in the same situation


@pytest.mark.parametrize("outcome", [1.0, -1.0])
def test_an_outcome_at_an_unchanged_observation_bounds_the_next_replay(outcome, monkeypatch):
    brain = episodic(4, 2, modules=(8,), seed=0)
    x = np.eye(4)
    brain.step(x[[0]])
    brain.step(x[[1]], reward=[0.0])
    brain.step(x[[1]], reward=[outcome])  # the outcome at 1, and the next decision at 1 again
    assert path(brain) == [(0, 0.0, True), (1, outcome, True), (1, 0.0, False)]
    brain.step(x[[2]], reward=[0.0])
    keys = [entry[0].copy() for entry in brain.memories["episodic"].paths[0]]
    written = []
    observe = brain.context_memory.observe

    def recorded(key, value, write=None, **kwargs):
        rows = np.ones(len(key), bool) if write is None else np.asarray(write)
        written.extend(np.asarray(key)[rows])
        observe(key, value, write, **kwargs)

    monkeypatch.setattr(brain.context_memory, "observe", recorded)
    brain.learn([2.0], [False], x[[3]])
    # the paid decision and the one taken since the outcome, not the decision before it
    assert len(written) == 2
    np.testing.assert_array_equal(written[0], keys[3])
    np.testing.assert_array_equal(written[1], keys[2])
    assert path(brain) == [(0, 0.0, True), (1, outcome, True), (1, 0.0, True), (2, 2.0, True)]


def test_decisions_without_an_outcome_still_collapse_at_an_unchanged_observation():
    brain = episodic(4, 2, modules=(8,), seed=0)
    x = np.eye(4)
    brain.step(x[[0]])
    for _ in range(3):
        brain.step(x[[0]], reward=[0.0])  # nothing came of it: each decision replaces the last
    assert path(brain) == [(0, 0.0, False)]
    brain.act(x[[0]])  # a decision whose outcome never came is replaced as well
    assert path(brain) == [(0, 0.0, False)]
    brain.step(x[[0]], reward=[1.0])  # paid: the outcome stays and the next decision follows
    assert path(brain) == [(0, 1.0, True), (0, 0.0, False)]
    for _ in range(3):
        brain.step(x[[0]], reward=[0.0])
    assert path(brain) == [(0, 1.0, True), (0, 0.0, False)]


def test_a_saved_path_keeps_its_boundary_and_continues_exactly(tmp_path):
    brain = episodic(4, 2, modules=(8,), seed=0)
    x = np.eye(4)
    brain.step(x[[0]])
    brain.step(x[[1]], reward=[0.0])
    brain.step(x[[1]], reward=[1.0])
    assert brain.pending_feedback  # the decision taken at 1 again awaits its outcome
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert path(copy) == path(brain) == [(0, 0.0, True), (1, 1.0, True), (1, 0.0, False)]
    for subject in (brain, copy):
        subject.step(x[[2]], reward=[0.0])
        writes = subject.context_memory.writes
        subject.learn([2.0], [False], x[[3]])
        assert subject.context_memory.writes == writes + 2
    assert path(copy) == path(brain)
    for name in ("hippocampus", "context_memory"):
        np.testing.assert_array_equal(getattr(copy, name).strength, getattr(brain, name).strength)
    np.testing.assert_array_equal(copy.act(x[[3]]), brain.act(x[[3]]))


def test_an_outcome_that_comes_after_waiting_bounds_the_replay_too():
    brain = episodic(4, 2, modules=(8,), seed=0, arousal={"youth": 1000})  # every outcome learned
    x = np.eye(4)
    brain.live(x[[0]])
    brain.live(x[[1]], reward=[0.0])
    brain.live(x[[1]], reward=[1.0])
    brain.wait(x[[2]])  # the decision at 1 awaits its outcome while the stream senses on
    assert path(brain) == [(0, 0.0, True), (1, 1.0, True), (1, 0.0, False)]
    brain.live(x[[2]], reward=[0.0])
    writes = brain.context_memory.writes
    brain.live(x[[3]], reward=[2.0])
    assert brain.context_memory.writes == writes + 2
    assert path(brain)[:3] == [(0, 0.0, True), (1, 1.0, True), (1, 0.0, True)]
