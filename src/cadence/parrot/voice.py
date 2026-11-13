"""When to speak, learned: the voice's onset as an act paid by a pistachio (#331, rung T0).

No part here detects a question, a silence or a pause, and no timer says when to reply.

- **Born, the ear's front end**: the speech front end's log-mel frames, pooled into ``bands``
  bands, less each band's slow running level (the cochlea's and brainstem's adaptation to
  stationary sound): what rose above the room's level drives the ear, and a sound that goes on
  unchanged fades from it.
- **Born, the held sound**: the heard history's fading blocks (``history``): a sound that just
  ended, and how long it has been quiet since, is decaying activity in the state. Nothing labels
  it.
- **Born, the babble**: the actor samples its act (``temperature``), so the voice starts on its
  own now and then.
- **Learned, when to speak**: each moment the actor chooses between silence and the voice's
  onset from the settled state. A pistachio is the reward of the act before it; the critic's
  prediction error (the dopamine) times each synapse's eligibility trace is its change: the
  three-factor rule of the library's actor and critic. ``lam`` sets how long an act stays
  eligible (``actor_lam`` and ``actor_gamma``; at 0.97 its credit halves in about a second of
  ten moments a second).

``Voice.moment(frames, reward=, done=)`` hears one moment's frames and answers whether the voice
starts; ``done`` ends an exchange (the critic expects nothing across it).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SILENT = math.log(1e-6)  # the front end's log-mel of no sound


class Voice:
    """The voice's onset, learned; see the module note."""

    def __init__(self, *, bands: int = 4, association: int = 16,
                 history: tuple[float, ...] = (0.0, 0.5, 0.8), adapt: float = 0.97,
                 gain: float = 0.15, lam: float = 0.97, temperature: float = 0.2,
                 seed: int = 0, backend: Any = "cpu"):
        from ..generic import Brain

        if 40 % int(bands) != 0:
            raise ValueError("the ear's 40 mel bands pool evenly into its bands")
        self.bands, self.adapt, self.gain = int(bands), float(adapt), float(gain)
        self.brain = Brain.compose(inputs=self.bands, actions=2, modules=(int(association),),
                                   history_decays=tuple(history), recall=False, episodic=False,
                                   actor_lam=float(lam), actor_gamma=float(lam),
                                   temperature=float(temperature), seed=int(seed),
                                   backend=backend)
        self.level: np.ndarray | None = None
        self.started = False
        self.vocalized = False
        self.heard = np.zeros(self.bands)

    def hear(self, frames: np.ndarray | None) -> np.ndarray:
        """One moment's log-mel frames ``(k, 40)`` as the ear's drive ``(bands,)``: each band's
        rise above its slow running level, scaled by ``gain`` and held in [0, 1]."""
        given = (np.full((1, 40), SILENT) if frames is None
                 else np.asarray(frames, dtype=np.float64).reshape(-1, 40))
        x = given.reshape(len(given), self.bands, 40 // self.bands).mean(axis=(0, 2))
        if self.level is None:
            self.level = np.full(self.bands, SILENT)
        out = np.clip((x - self.level) * self.gain, 0.0, 1.0)
        self.level = self.adapt * self.level + (1.0 - self.adapt) * x
        self.heard = out
        return np.asarray(out)

    def moment(self, frames: np.ndarray | None, *, reward: float | None = None,
               done: bool = False) -> bool:
        """Hear a moment and choose: True where the voice starts. ``reward`` pays the act of
        the moment before; ``done`` says that act ended an exchange."""
        x = self.hear(frames)[None, :]
        if not self.started:
            act = self.brain.step(x)
            self.started = True
        else:
            act = self.brain.step(x, reward=np.array([0.0 if reward is None else float(reward)]),
                                  done=np.array([bool(done)]))
        self.vocalized = int(np.asarray(act).reshape(-1)[0]) == 1
        return self.vocalized

    def dopamine(self) -> float:
        """The critic's prediction error at the latest outcome."""
        report = self.brain.last_learning or {}
        return float(report.get("td_error", 0.0))


__all__ = ["SILENT", "Voice"]
