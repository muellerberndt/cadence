"""Rehearsal from the lessons a brain was taught (cadence#238, ``learning_rehearsal``): the
reservoir keeps each lesson row with the moments its stream heard since its episode began, and
after a lesson some are replayed on fresh streams, so the recall read is computed and taught
again with the cortex, and types taught in sequence keep what earlier lessons taught; the live
streams continue as they would have without the replay."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import numpy as np
import pytest

import cadence as cd
from cadence.learning import LearnerConfig

VOCAB, CUES = 4, ("first", "last")
READ = dict(recall=True, recall_recency=0.0, recall_boundary=True, recall_anchor=True,
            learning_surprise=True)


# -- types the cortex holds


def lesson_rows(kind: int, rng: np.random.Generator, rows: int = 8):
    """Type ``kind``: an observation in its own third of 12 inputs, labelled by its own rule."""
    x = np.zeros((rows, 12))
    index = rng.integers(0, 4, size=rows)
    x[np.arange(rows), kind * 4 + index] = 1.0
    return x, (index + kind) % 4


def taught_in_sequence(rehearsal: float, lessons: int = 150) -> cd.Brain:
    brain = cd.Brain.compose(12, 4, seed=0, learning_rehearsal=rehearsal, learning_surprise=True)
    rng = np.random.default_rng(0)
    for kind in (0, 1, 2):
        for _ in range(lessons):
            brain.reset()
            brain.lesson(*lesson_rows(kind, rng))
    return brain


def kind_accuracy(brain: cd.Brain, kind: int) -> float:
    x, labels = lesson_rows(kind, np.random.default_rng(9 + kind), 64)
    brain.reset()
    return float(np.mean(brain.act(x, greedy=True) == labels))


def test_the_founder_keeps_no_reservoir_and_no_record_of_one(tmp_path):
    brain = cd.Brain.compose(12, 4, seed=0)
    brain.lesson(*lesson_rows(0, np.random.default_rng(0)))
    assert brain.learner.config.rehearsal == 0.0
    assert not brain.rehearsal_moments and not brain._episode_moments
    assert brain.rehearsal_seen == brain.rehearsal_draws == 0
    assert "learning_rehearsal" not in brain.describe()["genes"]
    with np.load(brain.save(tmp_path / "brain.npz")) as saved:
        assert not any(name.startswith("rehearsal") for name in saved.files)
        assert "rehearsal" not in json.loads(str(saved["brain"]))


def test_types_taught_in_sequence_keep_what_earlier_lessons_taught():
    plain, rehearsed = taught_in_sequence(0.0), taught_in_sequence(1.0)
    kept = [kind_accuracy(rehearsed, kind) for kind in (0, 1, 2)]
    assert min(kept) >= 0.95, kept
    assert np.mean(kept) > np.mean([kind_accuracy(plain, kind) for kind in (0, 1, 2)])


def test_the_reservoir_keeps_a_uniform_sample_within_its_capacity():
    brain = taught_in_sequence(1.0, lessons=100)
    assert len(brain.rehearsal_moments) == brain.learner.config.rehearsal_capacity == 512
    assert brain.rehearsal_seen == 3 * 100 * 8
    assert {len(moment) for moment in brain.rehearsal_moments} == {1}  # a lesson after a reset
    kinds = np.array([np.argmax(moment[-1]) // 4 for moment in brain.rehearsal_moments])
    assert all(np.bincount(kinds, minlength=3) > 100)  # every type in the sample


def test_rehearsal_settings_are_validated():
    with pytest.raises(ValueError, match="rehearsal must be"):
        cd.Brain.compose(12, 4, learning_rehearsal=-1.0)
    with pytest.raises(ValueError, match="rehearsal_capacity"):
        cd.Brain.compose(12, 4, learning_rehearsal=1.0, learning_rehearsal_capacity=0)


# -- types the recall read holds


def story(rng: np.random.Generator, cue: str | None = None, batch: int = 8, lengths=(3, 5)):
    """Symbols on ``batch`` streams, then a cue, ``first`` or ``last`` (``cue``, else one drawn
    per stream): the moments heard, the cue's observations, the answers and the cues."""
    length = int(rng.integers(lengths[0], lengths[1] + 1))
    seq = rng.integers(VOCAB, size=(batch, length))
    eye = np.eye(VOCAB + len(CUES))
    cues = np.full(batch, CUES.index(cue)) if cue else rng.integers(len(CUES), size=batch)
    answer = seq[np.arange(batch), np.where(cues == 0, 0, length - 1)]
    return [eye[seq[:, t]] for t in range(length)], eye[VOCAB + cues], answer, cues


def composed(seed: int = 0, **options) -> cd.Brain:
    return cd.Brain.compose(VOCAB + len(CUES), VOCAB, modules=(16,), seed=seed, episodic=False,
                            teaching="normalized", **options)


def taught(brain: cd.Brain, rng: np.random.Generator, episodes: int, cue: str | None = None,
           batch: int = 8) -> list[np.ndarray]:
    """Episodes on fresh streams: the story heard by greedy acts, the cue's lesson, an act on it."""
    acts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(episodes):
            brain.reset()
            moments, x, answer, _ = story(rng, cue, batch)
            acts += [np.asarray(brain.act(m, greedy=True)) for m in moments]
            brain.lesson(x, answer)
            acts.append(np.asarray(brain.act(x, greedy=True)))
    return acts


def accuracy(brain: cd.Brain, cue: str) -> float:
    rng, hits = np.random.default_rng(33), []
    for _ in range(12):
        brain.reset()
        moments, x, answer, _ = story(rng, cue)
        for m in moments:
            brain.act(m, greedy=True)
        hits.extend(np.asarray(brain.act(x, greedy=True)) == answer)
    return float(np.mean(hits))


def test_each_lesson_row_is_kept_with_its_episode():
    brain = composed(**READ, learning_rehearsal=1.0)
    taught(brain, np.random.default_rng(1), 4)
    assert len(brain.rehearsal_moments) == brain.rehearsal_seen == 4 * 8
    rng = np.random.default_rng(1)
    for _ in range(4):
        moments, x, answer, _ = story(rng)
    # the last lesson's rows: the moments each stream heard since the reset, then the moment
    # it was taught on
    for row in range(8):
        np.testing.assert_array_equal(brain.rehearsal_moments[24 + row],
                                      np.stack([m[row] for m in moments] + [x[row]]))
    np.testing.assert_array_equal(brain.rehearsal_labels[24:], answer)


def test_cues_held_by_the_read_keep_what_earlier_lessons_taught():
    """``first`` taught, then ``last``: the replay keeps ``first``, which the read holds."""
    kept = {}
    for name, options in (("plain", READ), ("replayed", {**READ, "learning_rehearsal": 1.0})):
        brain = composed(seed=1, **options)
        rng = np.random.default_rng(11)
        taught(brain, rng, 60, "first")
        taught(brain, rng, 60, "last")
        kept[name] = (accuracy(brain, "first"), accuracy(brain, "last"))
    assert kept["replayed"][1] >= 0.9 and kept["replayed"][0] >= 0.8, kept
    assert kept["replayed"][0] >= kept["plain"][0] + 0.4, kept  # 0.86 against 0.29


def test_the_replay_reports_its_rows_and_moments():
    brain = composed(**READ, learning_rehearsal=1.0)
    rng = np.random.default_rng(2)
    taught(brain, rng, 1)
    assert not any(name.startswith("rehearsal_") for name in brain.last_learning)
    taught(brain, rng, 1)
    report = brain.last_learning
    assert report["rehearsal_rows"] == 8.0 and report["demonstrations"] == 8.0
    assert report["rehearsal_moments"] == sum(
        len(m) for m in brain.rehearsal_moments[:8]
    )  # one lesson kept before: its eight rows, each with its episode
    assert report["rehearsal_attempted_presentations"] == 8.0
    assert "demonstration_attempted_presentations" in report  # the live lesson's own report


# -- the live streams continue as without the replay


def frozen(rehearsal: float, **options) -> cd.Brain:
    """A brain whose lessons change no parameter (the learner's and the read's steps at zero),
    with every memory a stream keeps."""
    return cd.Brain.compose(
        VOCAB + len(CUES), VOCAB, modules=(16,), seed=3, recall=True, recall_recency=0.0,
        recall_boundary=True, recall_anchor=True, recall_relative=True, recall_address=0.0,
        recall_position_rate=0.0, efference_amplitude=1.0, history_decays=(0.5,),
        learning=LearnerConfig(eta=0.0, eta_bias=0.0, surprise=True, rehearsal=rehearsal),
        **options,
    )


def stepped(brain: cd.Brain) -> list[np.ndarray]:
    """Lessons, greedy acts, rewarded steps with demonstrations and streams that end apart."""
    rng, acts = np.random.default_rng(4), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            brain.reset()
            moments, x, answer, _ = story(rng, batch=4)
            acts += [np.asarray(brain.act(m, greedy=True)) for m in moments]
            brain.lesson(x, answer)
            acts.append(np.asarray(brain.act(x, greedy=True)))
        brain.reset()
        moments, x, answer, _ = story(rng, batch=4, lengths=(5, 5))
        acts.append(np.asarray(brain.step(moments[0])))
        for t, m in enumerate([*moments[1:], x, *moments]):
            done = np.array([t == 2, False, t == 4, False])
            acts.append(np.asarray(brain.step(m, reward=rng.normal(size=4), done=done,
                                              teacher=answer if t in (1, 3, 5) else None)))
    return acts


def checkpoint(brain: cd.Brain, path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(brain.save(path)) as saved:
        arrays = {name: saved[name] for name in saved.files if name not in ("meta", "brain")}
        life = json.loads(str(saved["brain"]))
    for name in ("last_learning", "rehearsal"):  # the replay's report and draws
        life.pop(name, None)
    return arrays, life


@pytest.mark.parametrize(
    "options",
    [{}, {"recall_heads": 2}, {"memory_choice": True, "memory_context": 0.5},
     {"learned_code": 8, "recall_window": 2}],
    ids=["read", "heads", "episodic", "learned_code"],
)
def test_the_live_streams_continue_exactly_as_without_the_replay(tmp_path, options):
    replaying, plain = frozen(1.0, **options), frozen(0.0, **options)
    for a, b in zip(stepped(replaying), stepped(plain), strict=True):
        np.testing.assert_array_equal(a, b)
    assert replaying.rehearsal_draws > 0 and plain.rehearsal_draws == 0
    mine, theirs = checkpoint(replaying, tmp_path / "a.npz"), checkpoint(plain, tmp_path / "b.npz")
    # the replaying brain's checkpoint holds the reservoir and the episodes beside the rest
    assert {name for name in mine[0] if not name.startswith("rehearsal/")} == set(theirs[0])
    for name, value in theirs[0].items():
        np.testing.assert_array_equal(mine[0][name], value, err_msg=name)
    assert mine[1] == theirs[1]


def test_the_lessons_read_stays_readable_after_the_replay():
    twins = [frozen(rehearsal) for rehearsal in (1.0, 0.0)]
    for brain in twins:
        taught(brain, np.random.default_rng(17), 1, batch=4)
        brain.reset()
        moments, x, answer, _ = story(np.random.default_rng(18), batch=4)
        for m in moments:
            brain.act(m, greedy=True)
        brain.lesson(x, answer)
    assert twins[0].rehearsal_draws > 0
    np.testing.assert_array_equal(twins[0].recall.attended_heads[0], twins[1].recall.attended_heads[0])


def test_a_live_stream_continues_exactly_as_without_the_replay():
    """``live`` with arousal: a moment ``wait`` senses, which the episode keeps, and a lesson
    while a calm action awaits its outcome, which is reported after it."""
    twins = [frozen(rehearsal, arousal=True, arousal_youth=4, arousal_threshold=0.5)
             for rehearsal in (1.0, 0.0)]
    rng, outs = np.random.default_rng(5), [[], []]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = story(np.random.default_rng(0), batch=1)[0][0]
        for brain, out in zip(twins, outs, strict=True):
            out.append(brain.live(first))
        for _ in range(6):
            moments, x, answer, _ = story(rng, batch=1)
            reward = rng.normal(size=1)
            for brain, out in zip(twins, outs, strict=True):
                out.append(brain.live(moments[0], reward=reward))
                brain.wait(moments[1])
                out += [brain.live(m) for m in moments[2:]]
                if brain.basal_ganglia._pending is None:
                    brain.lesson(x, answer)
                out.append(brain.live(x, reward=reward, done=np.array([True])))
    for a, b in zip(*outs, strict=True):
        np.testing.assert_array_equal(a, b)
    assert twins[0].rehearsal_draws > 0
    np.testing.assert_array_equal(twins[0].basal_ganglia.state.activation,
                                  twins[1].basal_ganglia.state.activation)


def test_a_refused_replay_reports_it_and_leaves_the_streams(monkeypatch):
    replaying, plain = frozen(1.0), frozen(0.0)
    rng = np.random.default_rng(13)
    taught(replaying, np.random.default_rng(14), 1, batch=4)
    taught(plain, np.random.default_rng(14), 1, batch=4)
    moments, x, answer, _ = story(rng, batch=4)
    for brain in (replaying, plain):
        brain.reset()
        for m in moments:
            brain.act(m, greedy=True)

    def refused(*args, **kwargs):
        raise RuntimeError("brain did not settle")

    monkeypatch.setattr(replaying, "_settled", refused)  # the replay's first moment refuses
    replaying.lesson(x, answer)
    monkeypatch.undo()
    plain.lesson(x, answer)
    assert replaying.last_learning["rehearsal_refused"] == 1.0
    assert replaying.last_learning["rehearsal_rows"] == 0.0
    np.testing.assert_array_equal(replaying.act(x, greedy=True), plain.act(x, greedy=True))
    np.testing.assert_array_equal(replaying.recall.keys, plain.recall.keys)
    np.testing.assert_array_equal(replaying.recall.told, plain.recall.told)


# -- checkpoints, retune, grow and graft


def same_reservoir(a: cd.Brain, b: cd.Brain) -> None:
    for mine, theirs in zip(a.rehearsal_moments, b.rehearsal_moments, strict=True):
        np.testing.assert_array_equal(mine, theirs)
    np.testing.assert_array_equal(a.rehearsal_labels, b.rehearsal_labels)
    assert (a.rehearsal_seen, a.rehearsal_draws) == (b.rehearsal_seen, b.rehearsal_draws)


def test_a_brain_saved_inside_an_episode_continues_exactly(tmp_path):
    brain = composed(**READ, learning_rehearsal=1.0)
    rng = np.random.default_rng(6)
    taught(brain, rng, 3)
    brain.reset()
    moments, x, answer, _ = story(rng)
    brain.act(moments[0], greedy=True)
    brain.act(moments[1], greedy=True)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert len(copy.rehearsal_moments) == len(brain.rehearsal_moments) == 24
    outs = []
    for twin in (brain, copy):
        acts = [np.asarray(twin.act(m, greedy=True)) for m in moments[2:]]
        twin.lesson(x, answer)
        acts.append(np.asarray(twin.act(x, greedy=True)))
        acts += taught(twin, np.random.default_rng(7), 2)
        outs.append(acts)
    for a, b in zip(*outs, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(copy.brain.efficacy, brain.brain.efficacy)
    for a, b in zip(copy.recall.anchor_maps[0], brain.recall.anchor_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    same_reservoir(copy, brain)


def test_a_retune_above_zero_keeps_the_moments_heard_from_then_on():
    brain = composed(**READ)
    rng = np.random.default_rng(20)
    taught(brain, rng, 2)  # at zero nothing is kept
    assert not brain.rehearsal_moments and not brain._episode_moments
    brain.reset()
    moments, x, answer, _ = story(rng, lengths=(5, 5))
    for m in moments[:3]:
        brain.act(m, greedy=True)
    brain.retune(learning_rehearsal=1.0)
    for m in moments[3:]:
        brain.act(m, greedy=True)
    brain.lesson(x, answer)
    # the episode in progress is replayed from the moments heard after the retune
    assert {len(m) for m in brain.rehearsal_moments} == {3}
    np.testing.assert_array_equal(brain.rehearsal_moments[0],
                                  np.stack([moments[3][0], moments[4][0], x[0]]))
    assert brain.describe()["genes"]["learning_rehearsal"] == 1.0


def test_the_reservoir_is_saved_while_rehearsal_is_off(tmp_path):
    brain = composed(**READ, learning_rehearsal=1.0)
    rng = np.random.default_rng(21)
    taught(brain, rng, 2)
    brain.retune(learning_rehearsal=0.0)
    kept, draws = len(brain.rehearsal_moments), brain.rehearsal_draws
    taught(brain, rng, 1)  # no replay, and nothing joins the reservoir
    assert len(brain.rehearsal_moments) == kept and brain.rehearsal_draws == draws
    assert not any(name.startswith("rehearsal_") for name in brain.last_learning)
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    same_reservoir(copy, brain)
    for twin in (brain, copy):
        twin.retune(learning_rehearsal=1.0)
    outs = [taught(twin, np.random.default_rng(22), 2) for twin in (brain, copy)]
    for a, b in zip(*outs, strict=True):
        np.testing.assert_array_equal(a, b)
    assert brain.rehearsal_draws > draws


def test_the_capacity_is_a_named_integer_setting():
    brain = cd.Brain.compose(6, 4, seed=0, learning_rehearsal=1.0, learning_rehearsal_capacity=256)
    assert brain.learner.config.rehearsal_capacity == 256
    brain.retune(learning_rehearsal_capacity=128)
    assert brain.learner.config.rehearsal_capacity == 128


def test_a_smaller_capacity_trims_the_reservoir_and_the_brain_saves_and_loads(tmp_path):
    brain = composed(**READ, learning_rehearsal=1.0, learning_rehearsal_capacity=16)
    rng = np.random.default_rng(8)
    taught(brain, rng, 3)
    assert len(brain.rehearsal_moments) == 16 and brain.rehearsal_seen == 24
    brain.retune(learning_rehearsal_capacity=4)  # retune cuts the reservoir at once
    assert len(brain.rehearsal_moments) == 4 and len(brain.rehearsal_labels) == 4
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    outs = [taught(twin, np.random.default_rng(9), 2) for twin in (brain, copy)]
    for a, b in zip(*outs, strict=True):
        np.testing.assert_array_equal(a, b)
    assert len(brain.rehearsal_moments) == 4
    same_reservoir(copy, brain)


def test_save_grow_and_graft_leave_the_living_reservoir_as_it_is(tmp_path):
    brain = composed(**READ, learning_rehearsal=1.0, learning_rehearsal_capacity=16)
    taught(brain, np.random.default_rng(10), 2)
    # set without retune: the next lesson trims
    brain.learner.config = replace(brain.learner.config, rehearsal_capacity=4)
    before = [moment.copy() for moment in brain.rehearsal_moments]
    brain.save(tmp_path / "brain.npz")
    grown = brain.grow(recall_heads=2)
    assert len(brain.rehearsal_moments) == len(before) == 16
    for moment, kept in zip(brain.rehearsal_moments, before, strict=True):
        np.testing.assert_array_equal(moment, kept)
    brain.retune(learning_rehearsal_capacity=4)  # retune trims at once
    assert len(brain.rehearsal_moments) == 4
    # the grown brain starts its reservoir afresh and keeps the setting
    assert grown.learner.config.rehearsal == 1.0
    assert not grown.rehearsal_moments and grown.rehearsal_seen == grown.rehearsal_draws == 0
    taught(grown, np.random.default_rng(10), 2)
    assert len(grown.rehearsal_moments) == 4 and grown.last_learning["rehearsal_rows"] == 8.0
    # a graft keeps this brain's
    host = cd.Brain.compose(6, 4, modules=(16, 16), seed=0, learning_rehearsal=1.0)
    donor = cd.Brain.compose(6, 4, modules=(16, 16), seed=1)
    rng = np.random.default_rng(11)
    for _ in range(2):
        host.reset()
        host.act(rng.random((4, 6)), greedy=True)
        host.lesson(rng.random((4, 6)), rng.integers(0, 4, size=4))
    grafted = host.graft(donor, ["module_0", "association"])
    assert grafted.learner.config.rehearsal == 1.0 and not grafted._episode_moments
    same_reservoir(grafted, host)
    assert host.rehearsal_seen == 8


def test_an_episode_reaches_as_far_as_the_read_keeps_slots():
    brain = composed(**READ, learning_rehearsal=1.0, recall_capacity=3)
    rng = np.random.default_rng(12)
    brain.reset()
    moments, x, answer, _ = story(rng, lengths=(5, 5))
    for m in moments:
        brain.act(m, greedy=True)
    brain.lesson(x, answer)
    assert {len(m) for m in brain.rehearsal_moments} == {3}  # the latest two heard, the cue
    np.testing.assert_array_equal(brain.rehearsal_moments[0],
                                  np.stack([moments[3][0], moments[4][0], x[0]]))


@pytest.mark.parametrize(
    ("name", "value"),
    [("rehearsal/told", np.array([-1, 0, 0])), ("rehearsal/lengths", None),
     ("rehearsal/labels", np.array([9, 0, 0]))],
)
def test_an_invalid_saved_reservoir_is_refused(tmp_path, name, value):
    brain = composed(**READ, learning_rehearsal=1.0)
    taught(brain, np.random.default_rng(15), 1, batch=3)
    with np.load(brain.save(tmp_path / "brain.npz")) as saved:
        data = {key: saved[key] for key in saved.files}
    if name == "rehearsal/lengths":
        value = np.array([0, *data[name][1:-1], data[name][-1] + data[name][0]])
    data[name] = value
    np.savez(tmp_path / "bad.npz", **data)
    with pytest.raises(ValueError, match="invalid saved rehearsal episodes"):
        cd.Brain.load(tmp_path / "bad.npz")


@pytest.mark.parametrize(
    ("backend", "device", "precision", "atol"),
    [("torch", "cpu", "float64", 1e-9), ("torch", "mps", "float32", 1e-4),
     ("mlx", None, None, 1e-4)],
    ids=["torch_cpu64", "mps32", "mlx"],
)
def test_the_replay_teaches_on_a_device_as_on_the_cpu(tmp_path, backend, device, precision, atol):
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    if device == "mps":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS hardware unavailable")
    cpu = composed(seed=5, **READ, learning_rehearsal=1.0)
    cpu.save(tmp_path / "brain.npz")
    moved = cd.Brain.load(tmp_path / "brain.npz", backend=backend, device=device,
                          precision=precision)
    assert moved.brain.backend == backend
    cpu_acts, moved_acts = (taught(brain, np.random.default_rng(19), 3) for brain in (cpu, moved))
    np.testing.assert_array_equal(cpu_acts[0], moved_acts[0])
    assert cpu.last_learning["rehearsal_rows"] == moved.last_learning["rehearsal_rows"] == 8.0
    for a, b in zip(moved.recall.anchor_maps[0], cpu.recall.anchor_maps[0], strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=atol)
    np.testing.assert_allclose(np.asarray(moved.brain.efficacy, dtype=float),
                               np.asarray(cpu.brain.efficacy, dtype=float), rtol=0, atol=atol)
