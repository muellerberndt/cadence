"""The planning window (cadence#257): k motor slots of one menu settled jointly under the
message, taught together, and advanced through the efference copy; the founder, one slot, is
unchanged."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import Brain

MENU, K = 6, 4


def message(m: int) -> np.ndarray:
    return np.eye(MENU)[[m]]


def phrase(m: int) -> list[int]:
    """The window a message asks for: four words that follow from one message."""
    return [(m + i) % MENU for i in range(K)]


def test_a_window_is_k_slots_of_one_menu_settled_in_one_settle():
    speaker = Brain.compose(MENU, K * MENU, modules=(16,), seed=0, slots=K, episodic=False)
    assert speaker.describe()["layout"]["slots"] == [MENU] * K
    window = speaker.act(message(2), greedy=True)
    assert window.shape == (1, K) and (0 <= window).all() and (window < MENU).all()
    settled = speaker.last_settlement
    assert settled is not None and settled["operation"] == "act" and settled["qualified"]


def test_a_lesson_teaches_the_k_slots_at_once_and_the_window_is_learned():
    speaker = Brain.compose(MENU, K * MENU, modules=(16,), seed=1, slots=K, episodic=False,
                            teaching="normalized", learning_surprise=True)
    with pytest.raises(ValueError):
        speaker.lesson(message(0), np.array([0]))  # a slotted brain takes (batch, k) labels
    batch = np.eye(MENU)
    labels = np.array([phrase(m) for m in range(MENU)])
    speaker.reset()
    report = speaker.lesson(batch, labels)
    assert report["accepted"] == 1.0 and report["attempted_presentations"] == MENU
    for _ in range(300):
        speaker.reset()
        speaker.lesson(batch, labels)
        speaker.reset()
        said = speaker.act(batch, greedy=True)
        if (said == labels).all():
            break
    assert (said == labels).all()  # every message's four words, each in one settle
    assert speaker.last_settlement is not None and speaker.last_settlement["operation"] == "act"


def test_the_efference_copy_advances_the_window():
    speaker = Brain.compose(MENU, K * MENU, modules=(16,), seed=0, slots=K, episodic=False,
                            efference_amplitude=1.0)
    window = speaker.act(message(1), greedy=True)
    copy = speaker.efference.trace[0].reshape(K, MENU)
    assert copy.argmax(axis=1).tolist() == window[0].tolist()  # the k commands issued
    assert np.allclose(copy.max(axis=1), 0.8)  # each at 1 - decay
    drive = speaker.stimulus(message(1))
    efference = speaker.connectome.populations["efference"]
    assert drive[0, efference].reshape(K, MENU).argmax(axis=1).tolist() == window[0].tolist()


def test_live_and_step_return_the_window_and_take_its_outcome():
    speaker = Brain.compose(MENU, K * MENU, modules=(16,), seed=0, slots=K, episodic=False,
                            arousal=True)
    first = speaker.live(message(0))
    assert first.shape == (1, K)
    second = speaker.live(message(1), reward=[1.0])  # one outcome for the whole window
    assert second.shape == (1, K)
    taught = Brain.compose(MENU, K * MENU, modules=(16,), seed=0, slots=K, episodic=False)
    taught.reset()
    assert taught.step(message(0), teacher=np.array([phrase(0)])).shape == (1, K)


def test_the_founder_is_one_slot_and_a_window_is_a_described_construction():
    founder = Brain.compose(MENU, MENU, modules=(16,), seed=0)
    assert founder.describe()["layout"]["slots"] == [MENU]
    assert founder.act(message(0), greedy=True).shape == (1,)
    speaker = Brain.compose(MENU, K * MENU, modules=(16,), seed=0, slots=K)
    assert speaker.describe()["layout"]["composition"]["slots"] == [MENU] * K
    grown = founder.grow(slots=(MENU,) * K)  # the whole grouping, extending the brain's own
    assert grown.describe()["layout"]["slots"] == [MENU] * K
    assert grown.act(message(0), greedy=True).shape == (1, K)
