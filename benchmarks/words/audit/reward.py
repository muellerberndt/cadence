"""The pistachio (cadence#326 W11 audit): is the reward the one teacher of the reward-taught
reads, and is it the reward's tie to a right answer that teaches?

Each arm lives the founder's script with practice by reward (``ask(explore=True)``, then
``reward``), the answers never told, and is read on the test as ``words.life`` reads it:

- ``reward``: a pistachio (1) for a right answer, nothing for a wrong one (protocol 2's arm);
- ``withheld``: no pistachio (protocol 2's arm);
- ``random``: a pistachio at random, at the rate the ``reward`` arm earned them on the same
  founder, whatever the answer;
- ``wrong``: a pistachio for a wrong answer, nothing for a right one;
- ``late``: the pistachio for each answer given after the next question is asked (it lands on
  the next answer);
- ``told``: the teacher's answers (protocol 1's practice), the reference.

A reward that teaches by its tie to a right answer reads high in ``reward``, near chance (a read
of three) in ``random`` and ``late``, and low in ``wrong``. The script also checks the reward
path's code: a practice by reward hands the brain no told answer (the teacher's answer is read
only by the scorer).

    python reward.py <founder> <out> [short] [practice rooms]
"""

from __future__ import annotations

import sys
from typing import Any

import common
import numpy as np
import world

ARMS = ("reward", "withheld", "random", "wrong", "late", "told")


class Practice:
    """How a practice question teaches in an arm, and the pistachios given."""

    def __init__(self, arm: str, rate: float, seed: int):
        self.arm, self.rate = arm, float(rate)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 80]))
        self.right: list[bool] = []
        self.given: list[float] = []
        self.pending: float | None = None

    def __call__(self, h: common.Harness, q: world.Question) -> None:
        if self.arm == "told":
            h.brain.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
            return
        right = world.right(q, h.brain.ask(q.text, explore=True))
        self.right.append(right)
        if self.arm == "withheld":
            return
        if self.arm == "reward":
            amount = 1.0 if right else 0.0
        elif self.arm == "wrong":
            amount = 0.0 if right else 1.0
        elif self.arm == "random":
            amount = 1.0 if self.rng.random() < self.rate else 0.0
        else:  # late: the previous answer's pistachio lands on this answer
            amount = self.pending if self.pending is not None else 0.0
            self.pending = 1.0 if right else 0.0
        self.given.append(amount)
        h.brain.reward(amount)


def life(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"seed": seed, "settings": settings, "arms": {}}
    rate = 0.0
    for arm in ARMS:
        practice = Practice(arm, rate, seed)
        result = common.replay(seed, [], settings, practise=practice)
        row = {"shares": result["arms"]["brain"]["shares"],
               "gates": result["arms"]["brain"]["gates"],
               "answers_right_in_practice": float(np.mean(practice.right))
               if practice.right else None,
               "pistachios": float(np.sum(practice.given)) if practice.given else 0.0,
               "pistachio_rate": float(np.mean(practice.given)) if practice.given else 0.0}
        if arm == "reward":
            rate = row["pistachio_rate"]
        out["arms"][arm] = row
    return out


def told_answer_on_the_path() -> dict[str, Any]:
    """The reward path's code: does ``Words.ask``, ``Words.reward`` or ``Words.choose`` take an
    answer?"""
    import inspect

    from cadence.words import Words

    out = {}
    for name in ("ask", "reward", "choose", "asking"):
        fn = getattr(Words, name, None)
        out[name] = None if fn is None else list(inspect.signature(fn).parameters)
    return out


def main(argv: list[str]) -> None:
    seed, out, settings = common.arguments(argv[:4])
    rooms = next((int(a) for a in argv[3:] if a.isdigit()), 60)
    settings = {**settings, "practice_rooms": rooms}
    result = life(seed, settings)
    result["signatures"] = told_answer_on_the_path()
    print(common.write(out, "reward", seed, result))


if __name__ == "__main__":
    main(sys.argv)
