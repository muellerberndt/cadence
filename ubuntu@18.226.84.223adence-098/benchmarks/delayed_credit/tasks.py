"""The delayed-credit chamber's two worlds (cadence#283): reward only at the end of an episode.

**Delayed T-maze** (``tmaze``). The episode opens at the junction, which shows the cue
(``C0`` or ``C1``); the act taken there is the choice. A corridor of ``delay`` moments
follows, each the corridor token or, with probability ``DISTRACTION``, one of six distractor
tokens drawn uniformly; the corridor is the same whichever way the choice went. The outcome,
+1 for the cue's arm and -1 for the other, arrives after the corridor's last act. Nine
one-hot inputs: ``C0``, ``C1``, the corridor, ``D0`` to ``D5``.

**Delayed key-door** (``keydoor``). The episode opens in the key room, which shows the key on
the left or on the right (``KL``, ``KR``); the reach to the key's side takes it. A corridor of
``delay - 1`` moments follows (distractors as above), then the door, where one action opens it.
From the key's moment on, a pouch input shows the key held. Opening with the key pays +1,
opening without it -1, leaving the door shut 0, after the door's act: taking the key pays only
when the door opens ``delay`` moments later. Eleven inputs: ``KL``, ``KR``, the corridor,
``D0`` to ``D5``, the door and the pouch.

In both worlds the act whose credit is read (the choice, the key) is followed by ``delay``
further acts before its outcome arrives, so its eligibility has decayed ``delay`` times when
the dopamine meets it. Which action is right is counterbalanced across founders (bit 0 of the
founder flips the choice and the reach; bit 1 flips the door's opening action).

The exact solver (``solve``) answers every episode from what the brain senses at the moment
it acts: every outcome is reachable from the observations alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TASKS = ("tmaze", "keydoor")
DELAYS = (10, 50, 200)
ACTIONS = 2
DISTRACTORS = 6
DISTRACTION = 0.25  # the share of corridor moments that show a distractor
CORRIDOR = 2
DISTRACTOR = 3  # D0 is input 3
DOOR, POUCH = 9, 10
INPUTS = {"tmaze": 9, "keydoor": 11}
RIGHT, WRONG, SHUT = 1.0, -1.0, 0.0
SEEDS = {"schedule": 28301, "panel": 28302, "yoked": 28303, "shuffled": 28304}


def right_action(cue: int, founder: int) -> int:
    """The choice (T-maze) or the reach that takes the key (key-door) for a cue."""
    return int(cue) ^ (int(founder) & 1)


def open_action(founder: int) -> int:
    """The key-door action that opens the door."""
    return 1 ^ ((int(founder) >> 1) & 1)


def corridor_tokens(rng: np.random.Generator, moments: int) -> tuple[int, ...]:
    """One corridor: the corridor token, or with probability ``DISTRACTION`` a uniform
    distractor, per moment."""
    shown = rng.random(moments) < DISTRACTION
    drawn = DISTRACTOR + rng.integers(DISTRACTORS, size=moments)
    return tuple(int(t) for t in np.where(shown, drawn, CORRIDOR))


@dataclass
class Episode:
    """One episode of a world. ``observation()`` is the present moment's input row, ``act``
    takes the brain's action and advances; after the last act ``done`` is true and ``reward``
    holds the outcome."""

    task: str
    founder: int
    delay: int
    cue: int
    corridor: tuple[int, ...]
    moment: int = 0
    reward: float = 0.0
    key: bool = False
    choice: int | None = None
    door: int | None = None
    actions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}")
        if self.delay < 1 or (self.task == "keydoor" and self.delay < 2):
            raise ValueError("delay must be at least one moment (two with a door)")
        wanted = self.delay if self.task == "tmaze" else self.delay - 1
        if len(self.corridor) != wanted:
            raise ValueError("the corridor must hold delay moments (delay - 1 with a door)")

    @property
    def moments(self) -> int:
        """Acts per episode: the decision and ``delay`` more."""
        return self.delay + 1

    @property
    def done(self) -> bool:
        return self.moment >= self.moments

    def token(self) -> int:
        if self.moment == 0:
            return int(self.cue)
        if self.task == "keydoor" and self.moment == self.moments - 1:
            return DOOR
        return int(self.corridor[self.moment - 1])

    def observation(self) -> np.ndarray:
        if self.done:
            raise RuntimeError("the episode has ended")
        x = np.zeros(INPUTS[self.task])
        x[self.token()] = 1.0
        if self.task == "keydoor" and self.key:
            x[POUCH] = 1.0
        return x

    def act(self, action: int) -> None:
        if self.done:
            raise RuntimeError("the episode has ended")
        action = int(action)
        if action not in range(ACTIONS):
            raise ValueError("action out of range")
        self.actions.append(action)
        if self.moment == 0:
            self.choice = action
            if self.task == "keydoor":
                self.key = action == right_action(self.cue, self.founder)
        if self.task == "keydoor" and self.moment == self.moments - 1:
            self.door = action
        self.moment += 1
        if self.done:
            self.reward = self._outcome()

    def _outcome(self) -> float:
        if self.task == "tmaze":
            return RIGHT if self.choice == right_action(self.cue, self.founder) else WRONG
        if self.door != open_action(self.founder):
            return SHUT
        return RIGHT if self.key else WRONG

    @property
    def right(self) -> bool:
        """Whether the decision whose credit is read (the choice, the key) was right."""
        return self.choice == right_action(self.cue, self.founder)


def corridor_length(task: str, delay: int) -> int:
    return delay if task == "tmaze" else delay - 1


def schedule(task: str, founder: int, delay: int, episodes: int) -> list[tuple[int, tuple]]:
    """The life's episodes: both cues in a random order in every block of two, each with
    its own corridor."""
    rng = np.random.default_rng(
        np.random.SeedSequence([SEEDS["schedule"], founder, delay, TASKS.index(task)])
    )
    out: list[tuple[int, tuple]] = []
    while len(out) < episodes:
        for cue in rng.permutation(2):
            out.append((int(cue), corridor_tokens(rng, corridor_length(task, delay))))
    return out[:episodes]


def panel(task: str, founder: int, delay: int, corridors: int = 4) -> list[tuple[int, tuple]]:
    """The reading panel: each cue with ``corridors`` corridors drawn from the panel seed."""
    rng = np.random.default_rng(
        np.random.SeedSequence([SEEDS["panel"], founder, delay, TASKS.index(task)])
    )
    tuples = [corridor_tokens(rng, corridor_length(task, delay)) for _ in range(corridors)]
    return [(cue, t) for t in tuples for cue in (0, 1)]


def solve(task: str, founder: int, x: np.ndarray) -> int:
    """The exact solver's action from the present observation alone."""
    token = int(np.argmax(x[:POUCH] if task == "keydoor" else x))
    if token in (0, 1):
        return right_action(token, founder)
    if task == "keydoor" and token == DOOR:
        return open_action(founder) if x[POUCH] > 0 else 1 - open_action(founder)
    return 0
