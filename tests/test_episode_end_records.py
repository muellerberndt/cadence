"""An episode's end and the long-term records: the stream whose episode ended holds no turn,
so its next episode's turn writes its own record, however the end is delivered (``step``,
``learn``, ``live`` aroused and in routine); a stream that goes on keeps its turn; every record
stays; and a refused outcome leaves the turn as it was."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cadence import Brain

EYE = np.eye(15)
TURN = 14  # the message's boundary: the moment after it is the answer the record holds
FIRST, SECOND = (0, 6, TURN, 12), (1, 7, TURN, 13)


def teller(**options) -> Brain:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Brain.compose(15, 4, modules=(32,), message=70, message_boundary=TURN,
                             message_boundary_holds=True, longterm=64, seed=0, backend="cpu",
                             **options)


def answers(brain: Brain, row: int = 0) -> list[int]:
    store = brain.longterm.stores[row]
    return store.values[: store.count].argmax(axis=1).tolist()


def told(brain: Brain, ends: str) -> None:
    """Two episodes, the second's first moment carrying the first's end."""
    moments = [EYE[[channel]] for channel in FIRST + SECOND]
    if ends == "reset":
        for k, x in enumerate(moments):
            if k == len(FIRST):
                brain.reset()
            brain.step(x)
    elif ends == "step":
        for k, x in enumerate(moments):
            brain.step(x, done=[True] if k == len(FIRST) else None)
    elif ends == "learn":
        brain.act(moments[0])
        for k, x in enumerate(moments[1:], start=1):
            brain.learn([0.0], [k == len(FIRST)], x)
            brain.act(x)
    else:  # live
        for k, x in enumerate(moments):
            brain.live(x, done=[True] if k == len(FIRST) else None)


@pytest.mark.parametrize(("ends", "options"), [
    ("reset", {}),
    ("step", {}),
    ("learn", {}),
    ("live", {"arousal": {"youth": 1000}}),  # aroused: every outcome is learned from
    ("live", {"arousal": {"youth": 0}}),  # routine: a calm brain's outcome
])
def test_the_episode_after_an_end_writes_its_own_record(ends, options):
    brain = teller(**options)
    told(brain, ends)
    if ends == "live":
        assert brain.last_arousal["mode"] == ("aroused" if options["arousal"]["youth"] else
                                              "routine")
    assert answers(brain) == [12, 13]


@pytest.mark.parametrize("beside", [
    (2, 8, 3, TURN, 11, 4, 5, 9),  # at its turn: its answer comes at the other's end
    (2, TURN, 11, 3, 4, 5, 8, 9),  # between turns: its record is written, its turn goes on
])
def test_a_stream_that_goes_on_keeps_its_turn_beside_one_that_ended(beside):
    both, lone = teller(), teller()
    for k in range(len(beside)):
        ended = k == len(FIRST)
        both.step(np.stack([EYE[(FIRST + SECOND)[k]], EYE[beside[k]]]),
                  done=[ended, False] if ended else None)
        lone.step(EYE[[beside[k]]])
        assert both.longterm.since[1] == lone.longterm.since[0]
    assert answers(both, 0) == [12, 13]
    assert answers(both, 1) == answers(lone) == [11]
    np.testing.assert_array_equal(both.longterm.read_slots[1], lone.longterm.read_slots[0])


def test_a_refused_outcome_leaves_the_turn_and_a_retry_ends_the_episode(monkeypatch):
    brain = teller()
    for channel in FIRST:
        brain.step(EYE[[channel]])
    records = brain.longterm
    before = {"since": records.since.copy(), "read_slots": records.read_slots.copy(),
              "window": records.window.copy(), "key": records.key.copy(),
              "sets": [set(held) for held in records._wsets],
              "cues": [held.copy() for held in records._cues]}
    learn = brain.basal_ganglia.learn

    def refuse(*args, **kwargs):
        raise RuntimeError("the feedback solve did not settle")

    monkeypatch.setattr(brain.basal_ganglia, "learn", refuse)
    with pytest.raises(RuntimeError, match="did not settle"):
        brain.step(EYE[[SECOND[0]]], done=[True])
    assert brain.pending_feedback  # the outcome is still awaited
    np.testing.assert_array_equal(records.since, before["since"])
    np.testing.assert_array_equal(records.read_slots, before["read_slots"])
    np.testing.assert_array_equal(records.window, before["window"])
    np.testing.assert_array_equal(records.key, before["key"])
    assert records._wsets == before["sets"]
    for held, kept in zip(records._cues, before["cues"], strict=True):
        np.testing.assert_array_equal(held, kept)
    assert answers(brain) == [12]
    monkeypatch.setattr(brain.basal_ganglia, "learn", learn)
    for k, channel in enumerate(SECOND):  # the same outcome again, then the episode
        brain.step(EYE[[channel]], done=[True] if k == 0 else None)
    assert answers(brain) == [12, 13]
