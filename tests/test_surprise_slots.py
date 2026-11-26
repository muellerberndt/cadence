"""Surprise by slot (``surprise_slots``, ``learning_surprise_slots``): in a surprise lesson of a
learner with several motor slots, a taught row pulls only the slots whose judging answer differs
from their label. A labelled slot the answer already gives as shown is left open at that row, as
a label of ``OPEN`` leaves it: its target is what the slot says in the lesson's own free phase,
the pull on it is zero, and no synapse learns that answer again from the row. A row that agrees
in every slot stays untaught, and with the option off the learner is the learner before it."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import numpy as np
import pytest
from checkpoints import write

import cadence as cd
from cadence.learning import OPEN

INPUTS, SLOTS = 12, (3, 3, 3, 3)
ACTIONS = sum(SLOTS)
X = np.eye(INPUTS)[[0, 5, 7]]
# the plain rule, without momentum or normalized steps: a step is the rate times the contrast
PLAIN = cd.LearnerConfig(
    beta=0.1, eta=0.5, free_steps=1024, nudged_steps=12, tolerance=3e-3, surprise=True
)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def composed(**options) -> cd.Brain:
    return cd.Brain.compose(
        INPUTS, ACTIONS, slots=SLOTS, seed=0,
        **{"modules": (24,), "recall": False, "learning_surprise": True, **options},
    )


def moment(brain: cd.Brain, rows: np.ndarray = X):
    """The drive of independent rows, the learner's own answer on it, and labels that differ
    from it in slot 0 of row 0 and in slots 0 and 1 of row 1: row 2 agrees in every slot."""
    drive = np.asarray(brain.stimulus(rows, memory=False))
    answer = brain.learner._answer(brain.learner.free(drive))
    labels = answer.copy()
    labels[0, 0] = (labels[0, 0] + 1) % 3
    labels[1, :2] = (labels[1, :2] + 1) % 3
    return drive, answer, labels


def pulled(brain: cd.Brain, monkeypatch):
    """The first nudge a lesson of ``brain``'s learner makes, kept as it is built."""
    kept: dict[str, object] = {}
    build = brain.learner.nudge_for

    def nudge_for(target, beta, weight=None, apical_gain=None):
        pull = build(target, beta, weight, apical_gain)
        kept.setdefault("pull", pull)
        return pull

    monkeypatch.setattr(brain.learner, "nudge_for", nudge_for)
    return kept


def by_slot(values: np.ndarray) -> np.ndarray:
    """The largest size over each slot's motor neurons, ``(rows, slots)``."""
    return np.abs(values).reshape(len(values), len(SLOTS), SLOTS[0]).max(axis=2)


def into_motor(brain: cd.Brain, step: np.ndarray) -> np.ndarray:
    """The mean size of a step over the synapses into each slot's motor neurons."""
    post = np.asarray(brain.connectome.post)
    column = -np.ones(brain.connectome.n, dtype=np.int64)
    column[brain.motor_index] = np.arange(ACTIONS)
    slot = np.where(column[post] >= 0, column[post] // SLOTS[0], -1)
    return np.array([np.abs(step[slot == k]).mean() for k in range(len(SLOTS))])


def test_a_taught_row_pulls_the_slots_that_differ_and_leaves_the_rest_open(monkeypatch):
    brain = composed(learning=replace(PLAIN, surprise_slots=True))
    kept = pulled(brain, monkeypatch)
    drive, answer, labels = moment(brain)
    before = np.array(brain.brain.efficacy, dtype=float, copy=True)
    lesson, report = brain.learner.step(drive, labels, answer=answer)
    # the row that agrees in every slot is no surprise, as without the option
    assert lesson.taught.tolist() == [True, True, False]
    assert report["accepted_presentations"] == 2 and report["agreed_presentations"] == 1
    assert report["taught_slots"] == 3  # one slot of row 0, two of row 1
    differs = (answer != labels)[lesson.taught]
    motor = brain.motor_index
    # the pull at the lesson's free state: zero on every slot left open, to the bit
    force = by_slot(kept["pull"].drive(np.asarray(lesson.free.activation))[:, motor])
    assert force[~differs].max() < 1e-15 and force[differs].min() > 1e-4  # rounding, against a pull
    # an open slot moves with the rest of the net through the phases, far less than a taught one
    moved = by_slot((lesson.nudged.activation - lesson.opposite.activation)[:, motor])
    assert moved[differs].min() > 5 * moved[~differs].max()
    # the synapses into the slots no row was wrong in learn far less than those into the others
    step = into_motor(brain, np.asarray(brain.brain.efficacy) - before)
    assert step[:2].min() > 5 * step[2:].max()


def test_without_the_option_a_taught_row_pulls_every_slot(monkeypatch):
    by_row, by_slots = composed(learning=PLAIN), composed(learning=replace(PLAIN, surprise_slots=True))
    kept = pulled(by_row, monkeypatch)
    drive, answer, labels = moment(by_row)
    opened, shared = [], []
    targets, share = by_row.learner._open_targets, by_row.learner._share
    monkeypatch.setattr(by_row.learner, "_open_targets",
                        lambda target, free, open: (opened.append(open), targets(target, free, open))[1])
    monkeypatch.setattr(by_row.learner, "_share",
                        lambda taught, slots=None: (shared.append(slots), share(taught, slots))[1])
    start = np.array(by_row.brain.efficacy, dtype=float, copy=True)
    lesson, report = by_row.learner.step(drive, labels, answer=answer)
    # nothing of the option reaches the step: no slot is opened, the share reads the rows,
    # and the report holds the names it held
    assert opened == [None] and shared == [None] and "taught_slots" not in report
    force = by_slot(kept["pull"].drive(np.asarray(lesson.free.activation))[:, by_row.motor_index])
    assert force.min() > 1e-4  # every slot of a taught row is pulled, the agreeing ones too
    by_slots.learner.step(drive, labels, answer=answer)
    rows = into_motor(by_row, np.asarray(by_row.brain.efficacy) - start)
    slots = into_motor(by_slots, np.asarray(by_slots.brain.efficacy) - start)
    assert (rows[2:] > 5 * slots[2:]).all()  # the slots that were right are what the option spares


def test_the_free_answer_judges_the_slots_where_no_answer_is_given():
    brain = composed(learning=replace(PLAIN, surprise_slots=True))
    drive, answer, labels = moment(brain)
    given = composed(learning=replace(PLAIN, surprise_slots=True))
    _, read = brain.learner.step(drive, labels)  # judged by the lesson's own free phase
    _, told = given.learner.step(drive, labels, answer=answer)
    assert read["taught_slots"] == told["taught_slots"] == 3
    assert read["accepted_presentations"] == 2
    np.testing.assert_allclose(brain.brain.efficacy, given.brain.efficacy, rtol=0, atol=1e-12)


def test_open_labels_no_token_rows_and_weights_keep_their_meaning(monkeypatch):
    brain = composed(learning=replace(PLAIN, surprise_slots=True))
    kept = pulled(brain, monkeypatch)
    drive, answer, labels = moment(brain)
    # a slot the caller leaves open stays open, and a row whose answer is none (a brain with
    # declared tokens gives -1 for a row whose answer is no token) is taught in every slot
    labels[1, 1] = OPEN
    judged = answer.copy()
    judged[2] = -1
    lesson, report = brain.learner.step(drive, labels, answer=judged)
    assert lesson.taught.tolist() == [True, True, True]
    assert report["taught_slots"] == 1 + 1 + len(SLOTS)
    force = by_slot(kept["pull"].drive(np.asarray(lesson.free.activation))[:, brain.motor_index])
    assert force[2].min() > 1e-4 and force[1, 1] < 1e-15 and force[1, 0] > 1e-4
    # a row of weight zero pulls nothing; the slots of the others are judged as without weights
    weighed = composed(learning=replace(PLAIN, surprise_slots=True))
    drive, answer, labels = moment(weighed)
    _, report = weighed.learner.step(drive, labels, weight=np.array([1.0, 0.0, 1.0]), answer=answer)
    assert report["taught_slots"] == 3 and report["accepted_presentations"] == 2


def test_a_lesson_every_row_agrees_with_teaches_nothing():
    brain = composed(learning=replace(PLAIN, surprise_slots=True))
    drive, answer, _ = moment(brain)
    before = np.array(brain.brain.efficacy, dtype=float, copy=True)
    lesson, report = brain.learner.step(drive, answer.copy(), answer=answer)
    assert not lesson.taught.any() and report["taught_slots"] == 0
    assert report["total_steps"] == 0 and brain.learner.updates == 0
    np.testing.assert_array_equal(brain.brain.efficacy, before)


def test_the_share_is_the_share_of_the_labelled_slots_taught():
    steps = {}
    for share in (False, True):
        brain = composed(learning=replace(PLAIN, surprise_slots=True, surprise_share=share))
        drive, answer, labels = moment(brain)
        labels[2, 3] = OPEN  # an open label is no labelled slot: 11 are labelled, 3 are taught
        before = np.array(brain.brain.efficacy, dtype=float, copy=True)
        brain.learner.step(drive, labels, answer=answer)
        steps[share] = np.asarray(brain.brain.efficacy) - before
    np.testing.assert_allclose(steps[True], steps[False] * 3 / 11, rtol=0, atol=1e-15)
    # by rows the same lesson's share is two rows of three
    rows = composed(learning=replace(PLAIN, surprise_share=True))
    whole = composed(learning=PLAIN)
    drive, answer, labels = moment(rows)
    for brain in (rows, whole):
        brain.learner.step(drive, labels, answer=answer)
    born = composed(learning=PLAIN).brain.efficacy
    np.testing.assert_allclose(
        rows.brain.efficacy - born, (whole.brain.efficacy - born) * 2 / 3, rtol=0, atol=1e-15
    )


def test_the_setting_needs_surprise_and_one_slot_is_its_row():
    with pytest.raises(ValueError, match="needs surprise=True"):
        cd.LearnerConfig(surprise_slots=True)
    with pytest.raises(ValueError, match="surprise_slots must be boolean"):
        cd.LearnerConfig(surprise=True, surprise_slots=1)
    with pytest.raises(ValueError, match="needs surprise=True"):
        composed(learning_surprise=False, learning_surprise_slots=True)
    assert "surprise_slots" not in cd.LearnerConfig().to_dict()
    assert cd.LearnerConfig(surprise=True, surprise_slots=True).to_dict()["surprise_slots"] is True
    # a learner of one slot: a row is its slot, and the lesson is the lesson by rows
    lessons = []
    for option in (False, True):
        brain = cd.Brain.compose(
            INPUTS, 4, modules=(24,), seed=0, recall=False,
            learning=replace(PLAIN, surprise_slots=option),
        )
        drive = np.asarray(brain.stimulus(X, memory=False))
        answer = brain.learner._answer(brain.learner.free(drive))
        labels = answer.copy()
        labels[0] = (labels[0] + 1) % 4
        _, report = brain.learner.step(drive, labels, answer=answer)
        assert "taught_slots" not in report
        lessons.append((report, np.asarray(brain.brain.efficacy).copy()))
    assert lessons[0][0] == lessons[1][0]
    np.testing.assert_array_equal(lessons[0][1], lessons[1][1])


def lived(brain: cd.Brain, moments: int = 4) -> list[dict[str, float]]:
    """Lessons before acts on a stream of two moments in turn, labels drawn from a seed."""
    rng = np.random.default_rng(4)
    stream = (X, np.eye(INPUTS)[[3, 9, 1]])
    reports = []
    for step in range(moments):
        labels = np.stack([rng.integers(0, k, size=3) for k in SLOTS], axis=1)
        reports.append(brain.lesson(stream[step % 2], labels))
        brain.act(stream[step % 2], greedy=True)
    return reports


@pytest.mark.parametrize("options", [
    {"recall": False},
    {"modules": (24, 16), "recall": True, "recall_heads": 2, "learning_surprise_share": True},
    {"modules": (24,), "recall": True, "learned_code": 16, "attention": True},
    {"recall": True, "learning_compiled_moment": True},
])
def test_a_composed_brain_teaches_by_slot(options):
    brain, by_row = composed(learning_surprise_slots=True, **options), composed(**options)
    reports, rows = lived(brain), lived(by_row)
    for report, row in zip(reports, rows, strict=True):
        taught = report["accepted_presentations"]
        # the slots taught are the labelled slots of the lesson less the ones that agreed
        assert report["taught_slots"] == report["attempted_presentations"] * len(SLOTS) - (
            report["agreed_slots"]
        )
        assert taught <= report["taught_slots"] <= taught * len(SLOTS)
        assert "taught_slots" not in row
    assert any(report["taught_slots"] < report["accepted_presentations"] * len(SLOTS)
               for report in reports)
    assert brain.last_learning["demonstration_taught_slots"] == reports[-1]["taught_slots"]
    if brain.recall is not None:  # the address lesson reads the lesson it was given
        assert "address_rows" in reports[-1]
    assert not np.array_equal(brain.brain.efficacy, by_row.brain.efficacy)


def test_retune_switches_it_and_describe_names_it():
    brain = composed()
    assert "learning_surprise_slots" not in brain.describe()["genes"]
    assert brain.retune(learning_surprise_slots=True) is brain
    assert brain.learner.config.surprise_slots is True
    assert brain.describe()["genes"]["learning_surprise_slots"] is True
    built = composed(learning_surprise_slots=True)
    for report, other in zip(lived(brain), lived(built), strict=True):
        assert report == other
    np.testing.assert_array_equal(brain.brain.efficacy, built.brain.efficacy)
    with pytest.raises(ValueError, match="needs surprise=True"):
        brain.retune(learning_surprise=False)
    brain.retune(learning_surprise=False, learning_surprise_slots=False)
    assert brain.learner.config.surprise_slots is False
    with pytest.raises(ValueError, match="no surprise by slot"):
        cd.Lives(6, 6, lives=2, backend="cpu", modules=(8,), slots=(3, 3), recall=False,
                 episodic=False, learning_surprise=True, learning_surprise_slots=True)


def record(path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["brain"])), json.loads(str(archive["meta"]))


def test_a_saved_brain_names_the_setting_and_continues(tmp_path, monkeypatch):
    brain = composed(learning_surprise_slots=True, recall=True, recall_heads=2, modules=(24, 16))
    lived(brain, 2)
    path = brain.save(tmp_path / "brain.npz")
    saved, learner = record(path)
    # a reader without the setting refuses both names
    assert saved["format"] == "cadence-brain/5" and saved["learning_surprise_slots"] is True
    assert "learning_surprise_slots" in saved["features"]
    assert learner["format"] == "cadence-checkpoint/7" and learner["config"]["surprise_slots"]
    loaded = cd.Brain.load(path)
    assert loaded.learner.config.surprise_slots is True
    for report, other in zip(lived(brain, 2), lived(loaded, 2), strict=True):
        assert report == other
    np.testing.assert_array_equal(loaded.brain.efficacy, brain.brain.efficacy)
    # a reader that does not hold the feature refuses the file by its name
    from cadence import generic

    with monkeypatch.context() as earlier:
        earlier.setattr(generic, "FEATURES", generic.FEATURES - {"learning_surprise_slots"})
        with pytest.raises(ValueError, match="needs the feature 'learning_surprise_slots'"):
            cd.Brain.load(path)
    # the learner's own checkpoint carries the name too, and a file without the option none
    alone = cd.Learner.load(brain.learner.save(tmp_path / "learner.npz"))
    assert alone.config.surprise_slots is True
    plain, plain_learner = record(composed(recall=True).save(tmp_path / "plain.npz"))
    assert "learning_surprise_slots" not in plain and plain_learner["format"] != "cadence-checkpoint/7"
    assert "learning_surprise_slots" not in plain.get("features", [])


def test_a_file_whose_names_and_setting_disagree_is_refused(tmp_path):
    brain = composed(learning_surprise_slots=True)
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    saved, learner = record(path)

    def changed(name: str, brain_record: dict, learner_record: dict):
        write(tmp_path / name, {**data, "brain": np.array(json.dumps(brain_record)),
                                "meta": np.array(json.dumps(learner_record))})
        return tmp_path / name

    # the setting without its format name
    without = {**learner, "format": "cadence-checkpoint/2"}
    with pytest.raises(ValueError, match="surprise_slots belongs to checkpoint format"):
        cd.Brain.load(changed("format.npz", saved, without))
    # the format name without the setting
    config = {k: v for k, v in learner["config"].items() if k != "surprise_slots"}
    with pytest.raises(ValueError, match="surprise_slots belongs to checkpoint format"):
        cd.Brain.load(changed("setting.npz", saved, {**learner, "config": config}))
    # the brain's feature without the learner's setting
    plain = {**learner, "format": "cadence-checkpoint/2", "config": config}
    with pytest.raises(ValueError, match="learning_surprise_slots feature and its learner"):
        cd.Brain.load(changed("feature.npz", saved, plain))


def test_the_torch_backend_teaches_by_slot_as_the_host_does():
    pytest.importorskip("torch")
    options = {"modules": (24, 16), "recall": True, "recall_heads": 2, "apical": 0.25,
               "learning_surprise_slots": True, "learning_surprise_share": True}
    host = composed(**options)
    device = composed(backend="torch", device="cpu", precision="float64", **options)
    assert device.brain.backend == "torch"
    for report, other in zip(lived(host), lived(device), strict=True):
        assert report["taught_slots"] == other["taught_slots"]
        assert report["accepted_presentations"] == other["accepted_presentations"]
    np.testing.assert_allclose(device.brain.efficacy, host.brain.efficacy, rtol=0, atol=1e-9)
    np.testing.assert_allclose(device.brain.bias, host.brain.bias, rtol=0, atol=1e-9)
