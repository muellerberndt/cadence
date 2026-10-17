"""The acquisition controls' surprise rule teaches the wrong rows from their rows of the complete
pre-query stimulus: the trace the prefix built is present, and the query's own settle never
advances it before the lesson."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "benchmarks" / "relevance"
sys.path.insert(0, str(HERE))

import acquisition_controls as ac  # noqa: E402
import selective_recall as sr  # noqa: E402

try:
    import cadence  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    pytest.skip("Cadence is not importable", allow_module_level=True)

from cadence import Brain  # noqa: E402


@pytest.mark.parametrize("exposed", [False, True])
def test_surprise_lessons_read_the_pre_query_stimulus_of_their_rows(exposed: bool) -> None:
    brain = Brain.compose(sr.INPUTS, 2, seed=5, modules=[16], episodic=False)
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    original = brain.learner.step

    def spy(drive: np.ndarray, labels: np.ndarray, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured.append((np.array(drive, copy=True), np.array(labels, copy=True)))
        return original(drive, labels, *args, **kwargs)

    brain.learner.step = spy  # type: ignore[method-assign]
    rng = np.random.default_rng(0)
    order = np.stack([rng.permutation(len(sr.CASES)) for _ in range(3)])
    glow = brain.working_memory.glow
    # replay the loop by hand for the first round to know the expected lesson
    rows = sr.CASES[order[0]]
    prefix, query, _ = sr.inputs(rows[:, :3], rows[:, 3], ac.NEUTRAL_EVENTS, exposed=exposed)
    twin = Brain.compose(sr.INPUTS, 2, seed=5, modules=[16], episodic=False)
    twin.reset()
    for x in prefix:
        twin.act(x, greedy=True)
    expected_drive = twin.stimulus(query)
    answer = twin.act(query, greedy=True)
    target = ac.targets("mux", rows, rows[:, 3])
    wrong = np.flatnonzero(np.asarray(answer).reshape(-1) != target)
    work = sr.Work()
    ac.train(brain, "mux", order[:1], work, "surprise", exposed)
    if len(wrong) == 0:
        assert captured == [] and work.lessons == 0
        return
    assert len(captured) == 1 and work.lessons == 1
    drive, labels = captured[0]
    assert drive.shape == (len(wrong), expected_drive.shape[1])
    assert np.array_equal(labels, target[wrong])
    assert np.allclose(drive, expected_drive[wrong])  # the rows of the complete stimulus
    assert np.abs(drive[:, glow]).max() > 0  # the context the prefix built is present
    assert not np.allclose(drive[:, glow], twin.stimulus(query)[wrong][:, glow])  # the query advanced the twin's trace


def test_every_rule_still_teaches_all_cases_then_acts() -> None:
    brain = Brain.compose(sr.INPUTS, 2, seed=3, modules=[16], episodic=False)
    calls: list[int] = []
    original = brain.learner.step

    def spy(drive, labels, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        calls.append(len(labels))
        return original(drive, labels, *args, **kwargs)

    brain.learner.step = spy  # type: ignore[method-assign]
    rng = np.random.default_rng(1)
    order = np.stack([rng.permutation(len(sr.CASES)) for _ in range(2)])
    work = sr.Work()
    trained = ac.train(brain, "and", order, work, "every")
    assert calls == [16, 16] and trained["lessons"] == 2 and work.lessons == 2


# -- the selection readings (SELECTION_PROTOCOL.md)


@pytest.mark.parametrize("seed", list(range(16)))
def test_holdout_split_keeps_every_value_in_every_training_cell(seed: int) -> None:
    held = ac.holdout_split(seed)
    assert held.shape == (4,) and len(set(held.tolist())) == 4 and held.tolist() == sorted(held)
    assert np.array_equal(held, ac.holdout_split(seed))  # fixed by the founder seed
    cases = sr.CASES[held]
    assert len({tuple(c) for c in cases[:, 2:].tolist()}) == 4  # one per (order, query) cell
    assert len({tuple(c) for c in cases[:, :2].tolist()}) == 4  # four different fact pairs
    assert int((cases[:, 0] != cases[:, 1]).sum()) == 2  # two unequal-fact, two equal-fact
    training = np.delete(sr.CASES, held, axis=0)
    assert len(training) == 12
    for order in (0, 1):
        for query in (0, 1):
            cell = training[(training[:, 2] == order) & (training[:, 3] == query)]
            assert len(cell) == 3
            assert set(cell[:, 0].tolist()) == {0, 1} and set(cell[:, 1].tolist()) == {0, 1}
    for pair in sr.CASES[:, :2].tolist():
        assert int((training[:, :2] == pair).all(axis=1).sum()) == 3


def test_holdout_splits_differ_between_founders() -> None:
    assert len({tuple(ac.holdout_split(s).tolist()) for s in range(12)}) > 1


def test_split_scores_reads_the_held_out_cells_of_the_answer_grid() -> None:
    held = ac.holdout_split(3)
    expected = np.stack(
        [ac.targets("mux", sr.FACTS, np.full(8, q, dtype=np.int64)) for q in range(2)], axis=1
    )
    intact = expected.copy()
    case = int(held[1])
    intact[case // 2, case % 2] ^= 1  # one held-out case wrong
    erased = 1 - expected  # everything wrong
    evaluation = {"expected": expected.tolist(), "answers": intact.tolist(),
                  "erased": {"answers": erased.tolist()}}
    scores = ac.split_scores(evaluation, held)
    assert scores["held_out_index"] == held.tolist()
    assert scores["held_out_cases"] == sr.CASES[held].tolist()
    assert scores["intact"] == {"training": 1.0, "held_out": 0.75,
                                "held_out_unequal_fact": pytest.approx(
                                    0.5 if (sr.CASES[case, 0] != sr.CASES[case, 1]) else 1.0)}
    assert scores["erased"] == {"training": 0.0, "held_out": 0.0, "held_out_unequal_fact": 0.0}


def decode(observation: np.ndarray) -> np.ndarray:
    """The symbol of each row of a one-hot event: A0, A1, B0, B1, neutral, ask-A, ask-B."""
    assert np.array_equal(observation.sum(axis=1), np.ones(len(observation)))
    return observation[:, :7].argmax(axis=1)


def test_holdout_training_presents_the_twelve_training_cases_only() -> None:
    held = ac.holdout_split(7)
    training = np.delete(sr.CASES, held, axis=0)
    brain = Brain.compose(sr.INPUTS, 2, seed=7, modules=[16], episodic=False)
    events: list[np.ndarray] = []
    original_act, original_step = brain.act, brain.learner.step
    sizes: list[int] = []

    def act(observation, **kwargs):  # noqa: ANN001, ANN003, ANN202
        events.append(decode(np.asarray(observation)))
        return original_act(observation, **kwargs)

    def step(drive, labels, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        sizes.append(len(labels))
        return original_step(drive, labels, *args, **kwargs)

    brain.act, brain.learner.step = act, step  # type: ignore[method-assign]
    rng = np.random.default_rng(0)
    order = np.stack([rng.permutation(len(training)) for _ in range(2)])
    work = sr.Work()
    trained = ac.train(brain, "mux", order, work, "every", False, cases=training)
    assert trained["completed"] and sizes == [12, 12]
    assert len(events) == 2 * (ac.NEUTRAL_EVENTS + 3)
    for round_events in (events[:4], events[4:]):
        first, second, neutral, query = round_events
        assert (neutral == 4).all() and (query >= 5).all()
        seen = np.zeros((12, 4), dtype=np.int64)
        for event, position in ((first, 0), (second, 1)):
            seen[np.arange(12), event // 2] = event % 2
            seen[event >= 2, 2] = 1 - position  # order 0: A first
        seen[:, 3] = query - 5
        assert sorted(map(tuple, seen.tolist())) == sorted(map(tuple, training.tolist()))


class Spy:
    """A fake brain for the replacement reading: it decodes the facts of every prefix it is
    shown and answers the named fact from the last prefix (``last``, a selecting brain) or
    from the first (``first``, a brain the replacement does not reach)."""

    def __init__(self, memory: str) -> None:
        self.memory = memory
        self.last_settlement: dict[str, float] | None = None
        self.reset()

    def reset(self) -> None:
        self.episodes: list[np.ndarray] = []
        self.pending: np.ndarray | None = None

    def act(self, observation: np.ndarray, *, greedy: bool = False) -> np.ndarray:
        self.last_settlement = {"steps": 1, "residual_checks": 1, "max_residual": 0.0,
                                "qualified": True}
        symbols = decode(np.asarray(observation))
        if (symbols < 4).all():
            if self.pending is None:
                self.pending = symbols
            else:
                facts = np.zeros((len(symbols), 2), dtype=np.int64)
                for event in (self.pending, symbols):
                    facts[np.arange(len(event)), event // 2] = event % 2
                self.episodes.append(facts)
                self.pending = None
            return np.zeros(len(symbols), dtype=np.int64)
        if (symbols == 4).all():
            return np.zeros(len(symbols), dtype=np.int64)
        facts = self.episodes[-1 if self.memory == "last" else 0]
        return facts[np.arange(len(symbols)), symbols - 5]


@pytest.mark.parametrize(
    "memory,relevant,irrelevant",
    [("last", {"follow": 1.0, "stay": 0.0}, {"follow": 1.0, "stay": 1.0}),
     ("first", {"follow": 0.0, "stay": 1.0}, {"follow": 1.0, "stay": 1.0})],
)
def test_replacement_reading_counts_follow_and_stay(
    memory: str, relevant: dict[str, float], irrelevant: dict[str, float]
) -> None:
    from contextlib import nullcontext  # noqa: PLC0415

    work = sr.Work()
    reading = ac.replacement(Spy(memory), "mux", work, forks=nullcontext(lambda: Spy(memory)))
    assert reading["completed"] and reading["intact_accuracy"] == 1.0
    assert reading["relevant"] == relevant and reading["irrelevant"] == irrelevant
    assert reading["identical"] == {"follow": 1.0, "stay": 1.0}
    assert not any(name.endswith("_transplant") for name in reading)  # no working trace
    assert work.actions == 4 + 3 * (2 * (ac.NEUTRAL_EVENTS + 2) + 1) and work.lessons == 0


def test_replacement_reading_forks_the_trained_brain_and_leaves_it_unchanged() -> None:
    brain = Brain.compose(sr.INPUTS, 2, seed=2, modules=[16], episodic=False)
    rng = np.random.default_rng(2)
    order = np.stack([rng.permutation(len(sr.CASES)) for _ in range(2)])
    work = sr.Work()
    ac.train(brain, "mux", order, work, "surprise", False)
    evaluation = ac.evaluate(brain, "mux", work, False)
    before = brain.learner.updates
    reading = ac.replacement(brain, "mux", work)  # raises if the checkpoint changed
    assert reading["completed"] and brain.learner.updates == before
    assert np.array(reading["answers"]["intact"]).reshape(8, 2).tolist() == evaluation["answers"]
    conditions = {"relevant", "irrelevant", "identical"}
    assert set(reading) == {"completed", "answers", "intact_accuracy"} | conditions | {
        c + "_transplant" for c in conditions
    }
    # the trace of the same prefix from a fresh fork, put into the host, is the intact read
    assert reading["answers"]["identical_transplant"] == reading["answers"]["intact"]
    assert reading["identical_transplant"]["stay"] == 1.0


def test_readings_off_keep_the_row_and_on_add_their_keys() -> None:
    plain = ac.job(("mux", "m32", 2, 0, "surprise", None, False, {}))
    off = ac.job(("mux", "m32", 2, 0, "surprise", None, False, {}, False, False))
    for row in (plain, off):
        assert "holdout" not in row and "replacement" not in row
        assert list(row)[-2:] == ["work", "elapsed_seconds"]
    assert {k: v for k, v in plain.items() if k != "elapsed_seconds"} == {
        k: v for k, v in off.items() if k != "elapsed_seconds"
    }
    on = ac.job(("mux", "m32", 2, 0, "surprise", None, False, {}, True, True))
    assert on["holdout"]["held_out_index"] == ac.holdout_split(0).tolist()
    assert on["training"]["completed"] and on["replacement"]["completed"]
    assert list(on)[-4:] == ["holdout", "replacement", "work", "elapsed_seconds"]
