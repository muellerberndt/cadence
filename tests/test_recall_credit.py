"""The recall memory's read at a forecast. A calm ``live`` moment forecasts the next moment
on the stream, and the act taken on that forecast, sampled by a brain the outcome woke or
issued greedily by one it left calm, keeps the forecast's read: its address eligibility is
the read's and ``attended`` describes it, as a fresh read of the same stream would give them.
An episode's end empties the stream's slots, so a forecast at an end reads an emptied store.
``recall_address_reward=0`` is the founder and control: no eligibility, and the checkpoint as
before."""

from dataclasses import replace
from types import MethodType

import numpy as np
import pytest

import cadence as cd

EYE = np.eye(3)
ARRAYS = ("keys", "values", "nexts", "held", "previous", "ended")


def composed(seed=1, **options):
    """The episodic vote, the recall memory and a reward-trained address map, calm from
    birth; ``options`` vary one of them."""
    settings = dict(
        memory_choice=True, recall=True, recall_address_reward=1.0, arousal={"youth": 0},
    )
    settings.update(options)
    return cd.Brain.compose(3, 2, modules=(8,), seed=seed, **settings)


def routine(brain):
    """Six calm moments whose outcomes are nothing: the brain is used to them."""
    brain.live(EYE[[0]])
    for t in range(5):
        brain.live(EYE[[t % 3]], reward=[0.0])
    assert brain.last_arousal["mode"] == "routine"
    return brain


def rereading(brain):
    """The reference: a brain that reads the recall memory again, on the stream as the moment
    is acted in, before every act. The drive it reads must be the forecast's, so the act and
    its settle are the same and only the read is taken afresh."""
    original = brain._choose

    def choose(self, observations, drive, free, **options):
        np.testing.assert_array_equal(self.stimulus(observations), drive)
        return original(observations, drive, free, **options)

    brain._choose = MethodType(choose, brain)
    return brain


def eligibility(brain):
    pending = brain.basal_ganglia._pending_memory
    return None if pending is None else pending.copy()


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def assert_same_checkpoint(first, second):
    a, b = saved_arrays(first), saved_arrays(second)
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def assert_same_memory(first, second):
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)


def test_a_reading_is_held_through_a_restore_and_an_end():
    brain = routine(composed())
    recall = brain.recall
    assert recall.last is None and recall.attended_heads is not None  # written; its attention kept
    brain.stimulus(EYE[[1]])
    reading = recall.reading()
    assert reading[0] is recall.last and reading[1] is recall.attended_heads[0] and recall.last is not None
    recall.restore(recall.snapshot())
    assert recall.reading() == (None, None)
    recall.hold(reading)
    assert recall.last is reading[0] and recall.attended_heads[0] is reading[1]
    recall.end(1, np.array([0]))
    assert recall.reading() == (None, None)
    recall.hold(reading)
    assert recall.last is reading[0] and recall.attended_heads[0] is reading[1]


def test_a_woken_brains_first_act_keeps_the_forecasts_read(tmp_path):
    """A woken brain's first act is decided on the forecast: its address eligibility is the
    forecast's read and ``attended`` describes it, as the reference, which reads the stream
    again before the act, gives them."""
    checkpoint = routine(composed()).save(tmp_path / "routine")
    readings = {}
    for name, reread in (("held", False), ("reference", True)):
        brain = cd.Brain.load(checkpoint)
        if reread:
            rereading(brain)
        action = brain.live(EYE[[1]], reward=[1.0], done=[False])
        assert brain.last_arousal["mode"] == "aroused" and brain.recall.slots == 7
        attended = brain.recall.attended_heads[0]
        assert attended is not None and attended.shape == (1, 6)  # the six slots the act read
        pending = eligibility(brain)
        assert pending is not None and pending.any()
        before = brain.recall.head_map(0)[0].copy()
        report = brain.learn([1.0], [False], EYE[[2]])
        assert report["address_reward_step"] > 0.0 and not np.array_equal(brain.recall.head_map(0)[0], before)
        readings[name] = (
            action, brain.basal_ganglia.state.activation.copy(), attended, pending,
            np.array(report["address_reward_step"]), brain.recall.head_map(0)[0].copy(),
        )
    for held, reread in zip(readings["held"], readings["reference"], strict=True):
        np.testing.assert_array_equal(held, reread)


def test_the_read_is_of_the_stream_as_the_end_leaves_it(tmp_path):
    """An episode's end empties the stream's slots, so the forecast at the end read an
    emptied store: no attention and no eligibility, as a fresh read of the emptied store
    gives, and the moment is the new episode's first slot."""
    checkpoint = routine(composed()).save(tmp_path / "routine")
    brain, reference = cd.Brain.load(checkpoint), rereading(cd.Brain.load(checkpoint))
    for subject in (brain, reference):
        subject.live(EYE[[1]], reward=[1.0], done=[True])
        assert subject.last_arousal["mode"] == "aroused"
    assert brain.recall.slots == reference.recall.slots == 1
    held, reread = eligibility(brain), eligibility(reference)
    np.testing.assert_array_equal(held, reread)
    assert not held.any()
    assert brain.recall.attended_heads is None and reference.recall.attended_heads is None
    assert_same_memory(brain.recall, reference.recall)
    assert_same_checkpoint(brain.save(tmp_path / "held"), reference.save(tmp_path / "reread"))


def test_a_calm_terminal_moment_witnesses_the_forecasts_read_without_eligibility(tmp_path):
    """An outcome the brain is used to leaves it calm: its greedy act keeps no eligibility,
    the moment is written as the new episode's first slot under the forecast's read of the
    emptied store, and the checkpoint is the one a fresh read of the same stream gives."""
    checkpoint = routine(composed()).save(tmp_path / "routine")
    brain, reference = cd.Brain.load(checkpoint), rereading(cd.Brain.load(checkpoint))
    for subject in (brain, reference):
        subject.live(EYE[[1]], reward=[0.0], done=[True])
        assert subject.last_arousal["mode"] == "routine"
        assert subject.basal_ganglia._pending is None and eligibility(subject) is None
        assert subject.recall.last is None and subject.recall.slots == 1
    assert brain.recall.attended_heads is None and reference.recall.attended_heads is None
    assert_same_memory(brain.recall, reference.recall)
    assert_same_checkpoint(brain.save(tmp_path / "held"), reference.save(tmp_path / "reread"))


@pytest.mark.parametrize(
    "memory",
    [{"memory_choice": True}, {"memory_choice": False}, {"memory_choice": False, "episodic": False}],
    ids=["episodic-vote", "associative-drive", "no-associative-memory"],
)
def test_the_first_sampled_act_is_credited_whatever_the_associative_memory(tmp_path, memory):
    """With the episodic vote, and without an associative memory, the woken brain samples on
    the forecast and keeps its read. With an associative memory in the drive the record the
    outcome wrote changes the drive, and the brain settles and reads again. Seed 3: at seeds
    1 and 2 the association of one of the three brains does not move between the sampled
    act's nudged phases, so no slot is scored. The forecast is nonterminal: an ended
    episode's slots are emptied, and its forecast reads nothing."""
    checkpoint = routine(composed(seed=3, **memory)).save(tmp_path / "routine")
    brain, reference = cd.Brain.load(checkpoint), rereading(cd.Brain.load(checkpoint))
    actions = []
    for subject in (brain, reference):
        actions.append(subject.live(EYE[[1]], reward=[1.0], done=[False]))
        assert subject.last_arousal["mode"] == "aroused"
    np.testing.assert_array_equal(actions[0], actions[1])
    held, reread = eligibility(brain), eligibility(reference)
    assert held.any()
    np.testing.assert_array_equal(held, reread)
    assert brain.recall.attended_heads is not None
    np.testing.assert_array_equal(brain.recall.attended_heads[0], reference.recall.attended_heads[0])
    np.testing.assert_array_equal(
        brain.basal_ganglia.state.activation, reference.basal_ganglia.state.activation
    )
    assert_same_checkpoint(brain.save(tmp_path / "held"), reference.save(tmp_path / "reread"))


def test_without_an_address_rate_the_read_is_held_and_no_eligibility_is_kept(tmp_path):
    """At the founder ``recall_address_reward=0`` the woken brain's act keeps no eligibility
    of the map and saves no pending contrast, as before; ``attended`` describes the forecast's
    read, and the next outcome leaves the map as it is."""
    checkpoint = routine(composed(recall_address_reward=0.0)).save(tmp_path / "routine")
    brain, reference = cd.Brain.load(checkpoint), rereading(cd.Brain.load(checkpoint))
    for subject in (brain, reference):
        subject.live(EYE[[1]], reward=[1.0], done=[False])
        assert subject.last_arousal["mode"] == "aroused"
        assert subject.basal_ganglia._pending is not None and eligibility(subject) is None
    assert brain.recall.attended_heads is not None
    np.testing.assert_array_equal(brain.recall.attended_heads[0], reference.recall.attended_heads[0])
    assert "recall/pending" not in saved_arrays(brain.save(tmp_path / "held"))
    assert_same_checkpoint(tmp_path / "held.npz", reference.save(tmp_path / "reread"))
    before = brain.recall.head_map(0)[0].copy()
    brain.learn([1.0], [False], EYE[[2]])
    np.testing.assert_array_equal(brain.recall.head_map(0)[0], before)


def test_the_held_reads_eligibility_survives_a_checkpoint_while_its_outcome_is_pending(tmp_path):
    """Saved after the woken moment, before the sampled act's outcome, the loaded brain holds
    the same pending eligibility and its next outcome steps the map by the same amount; saved
    before the woken moment, the loaded brain takes it to the same read and credit."""
    checkpoint = routine(composed()).save(tmp_path / "routine")
    brain = cd.Brain.load(checkpoint)
    brain.live(EYE[[1]], reward=[1.0], done=[False])
    pending = eligibility(brain)
    assert pending.any()
    resumed = cd.Brain.load(brain.save(tmp_path / "pending"))
    np.testing.assert_array_equal(eligibility(resumed), pending)
    twin = cd.Brain.load(checkpoint)
    twin.live(EYE[[1]], reward=[1.0], done=[False])
    np.testing.assert_array_equal(eligibility(twin), pending)
    np.testing.assert_array_equal(twin.recall.attended_heads[0], brain.recall.attended_heads[0])
    steps = [
        subject.learn([1.0], [False], EYE[[2]])["address_reward_step"]
        for subject in (brain, resumed, twin)
    ]
    assert steps[0] > 0.0 and steps == [steps[0]] * 3
    assert_same_checkpoint(brain.save(tmp_path / "learned"), resumed.save(tmp_path / "resumed"))
    assert_same_checkpoint(tmp_path / "learned.npz", twin.save(tmp_path / "twin"))


@pytest.mark.parametrize("refusal", ["forecast", "arousal"])
def test_a_refused_terminal_moment_keeps_the_outcome_and_the_retry_takes_it_once(
    tmp_path, monkeypatch, refusal
):
    """A forecast that does not settle, or an arousal update that refuses after it settled,
    changes nothing durable and keeps the outcome pending; the same call then takes it once,
    as an uninterrupted moment does, and the ended episode's slots are emptied."""
    checkpoint = routine(composed()).save(tmp_path / "routine")
    brain = cd.Brain.load(checkpoint)
    before, lived, config = brain.save(tmp_path / "before"), brain._lived, brain.learner.config
    if refusal == "forecast":
        brain.learner.config = replace(config, free_steps=0)
        error, message = RuntimeError, "did not settle"
    else:
        def reject(*args, **kwargs):
            raise ValueError("arousal refused the outcome")

        monkeypatch.setattr(brain.arousal, "outcome", reject)
        error, message = ValueError, "arousal refused"
    with pytest.raises(error, match=message):
        brain.live(EYE[[1]], reward=[1.0], done=[True])
    assert brain._lived is lived and brain.basal_ganglia._pending is None
    assert brain.recall.slots == 6 and not brain.recall.ended.any()
    brain.learner.config = config
    monkeypatch.undo()
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    twin = cd.Brain.load(checkpoint)
    np.testing.assert_array_equal(
        brain.live(EYE[[1]], reward=[1.0], done=[True]),
        twin.live(EYE[[1]], reward=[1.0], done=[True]),
    )
    assert brain.last_arousal["mode"] == "aroused" and brain.recall.slots == 1
    np.testing.assert_array_equal(eligibility(brain), eligibility(twin))
    assert brain.recall.attended_heads is None and twin.recall.attended_heads is None
    assert_same_checkpoint(brain.save(tmp_path / "retried"), twin.save(tmp_path / "twin"))
