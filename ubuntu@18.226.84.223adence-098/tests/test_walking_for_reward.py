"""The runnable reward example reports outcomes on consistent event boundaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "examples" / "walking_for_reward.py"
SPEC = importlib.util.spec_from_file_location("walking_for_reward_example", PATH)
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


@pytest.mark.parametrize(
    ("actions", "income", "beat"),
    [([0, 1, 0, 1], 0.75, 1.0), ([0] + [1] * 64, 0.02, 0.0)],
)
def test_report_uses_event_income_and_pairs_inside_the_last_window(actions, income, beat):
    class Walker:
        last_arousal = {"mode": "routine", "sweeps": 1, "learning_sweeps": 0}

        def __init__(self):
            self.actions = iter(actions)
            self.feedback = []

        def live(self, observation, **feedback):
            self.feedback.append(feedback.get("reward"))
            return np.array([next(self.actions)])

    walker = Walker()
    report = example.live(walker, len(actions))
    assert report["income_by_block"] == [income]
    assert report["changed_foot_by_block"] == [income]
    assert report["last_64_beat"] == beat
    assert report["work"]["routine"] == len(actions)
    assert walker.feedback[:3] == [None, [0.0], [1.0]]
