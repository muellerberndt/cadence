"""Players of the Doom body (cadence#285): the brain, its controls and a convolutional baseline.

Every player answers ``act(observation, previous)`` with one button index. ``previous`` is the
transition that led to ``observation`` (``None`` at the life's first moment); when it ended an
episode, ``observation`` is the first of the next one. ``finish(previous)`` takes the life's
last transition. A player reads the ``Observation`` alone: the screen, the own-body ports and
the history port.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from doom_body import RESOLUTIONS, Observation, Transition

# the brain's retina: the screen's mean over square blocks of this many pixels a side
POOL = {"84x84": 4, "160x120": 8}


class Player(Protocol):
    actions: int

    def act(self, observation: Observation, previous: Transition | None) -> int: ...

    def finish(self, previous: Transition) -> None: ...


def retina(screen: np.ndarray, pool: int) -> np.ndarray:
    """The screen's mean over ``pool`` by ``pool`` blocks, in [0, 1], flattened."""
    h, w, c = screen.shape
    if h % pool or w % pool:
        raise ValueError("the pool must divide the screen")
    blocks = screen.reshape(h // pool, pool, w // pool, pool, c).astype(np.float64)
    return np.asarray(blocks.mean(axis=(1, 3)).ravel() / 255.0)


def inputs(resolution: str, actions: int) -> int:
    """The width of the brain's observation: the retina, the three body ports, the history."""
    h, w = RESOLUTIONS[resolution]
    pool = POOL[resolution]
    return (h // pool) * (w // pool) * 3 + 3 + actions


class BrainPlayer:
    """``Brain.compose(inputs, actions, seed=seed, **options)`` playing through the body.

    It reads one vector: the retina, the body ports and the history port. ``mode`` is
    ``"learn"`` (``Brain.step`` with every outcome: the reward, the episode's end, and a
    truncated episode's value as its bootstrap), ``"blind"`` (the same with the retina dark: a
    control that reads the body and history ports alone) or ``"frozen"`` (the newborn's sampled
    acts, nothing learned: a control)."""

    def __init__(
        self,
        resolution: str,
        actions: int,
        *,
        seed: int = 0,
        mode: str = "learn",
        brain: Any = None,
        **options: Any,
    ) -> None:
        from cadence import Brain

        if mode not in ("learn", "blind", "frozen"):
            raise ValueError("mode is 'learn', 'blind' or 'frozen'")
        self.resolution, self.actions, self.mode, self.seed = resolution, actions, mode, seed
        self.pool = POOL[resolution]
        self.width = inputs(resolution, actions)
        self.options = options
        self.brain = brain if brain is not None else Brain.compose(
            self.width, actions, seed=seed, **options
        )
        self._last: np.ndarray | None = None

    def encode(self, observation: Observation) -> np.ndarray:
        """The brain's observation: the retina (dark when blind), the body and history ports."""
        seen = retina(observation.screen, self.pool)
        if self.mode == "blind":
            seen = np.zeros_like(seen)
        x = np.concatenate([seen, observation.body, observation.history])
        if x.shape != (self.width,):
            raise ValueError(f"an observation of width {x.shape} for a brain of {self.width}")
        return x[None, :]

    def _value(self, observation: Observation) -> float:
        drive = self.brain.stimulus(self.encode(observation), memory=False)
        return float(np.asarray(self.brain.basal_ganglia.value_of(drive))[0])

    def act(self, observation: Observation, previous: Transition | None) -> int:
        x = self.encode(observation)
        if self.mode == "frozen":
            action = self.brain.act(x)
        elif previous is None:
            action = self.brain.step(x)
        else:
            bootstrap = None
            if previous.outcome == "truncation" and previous.observation is not None:
                bootstrap = [self._value(previous.observation)]
            action = self.brain.step(
                x, reward=[previous.reward], done=[previous.done], bootstrap=bootstrap
            )
        self._last = x
        return int(np.asarray(action)[0])

    def finish(self, previous: Transition) -> None:
        """Learns the life's last outcome, so no action is left pending."""
        if self.mode == "frozen" or self._last is None:
            return
        x = self._last if previous.observation is None else self.encode(previous.observation)
        bootstrap = None
        if previous.outcome == "truncation" and previous.observation is not None:
            bootstrap = [self._value(previous.observation)]
        self.brain.learn([previous.reward], [previous.done], x, bootstrap=bootstrap)

    def parameters(self) -> int:
        return int(self.brain.parameters())

    def save(self, path: str | Path) -> None:
        """The brain and this player's own state: the brain's file and a JSON beside it."""
        path = Path(path)
        self.brain.save(path.with_suffix(".npz"))
        state = {
            "resolution": self.resolution, "actions": self.actions, "mode": self.mode,
            "seed": self.seed, "last": None if self._last is None else self._last.tolist(),
        }
        path.with_suffix(".json").write_text(json.dumps(state))

    @classmethod
    def load(cls, path: str | Path, **load: Any) -> BrainPlayer:
        from cadence import Brain

        path = Path(path)
        state = json.loads(path.with_suffix(".json").read_text())
        brain = Brain.load(path.with_suffix(".npz"), **load)
        player = cls(
            state["resolution"], state["actions"], seed=state["seed"], mode=state["mode"],
            brain=brain,
        )
        player._last = None if state["last"] is None else np.asarray(state["last"])
        return player


class RandomPlayer:
    """Uniform random buttons: a control."""

    def __init__(self, actions: int, *, seed: int = 0) -> None:
        self.actions = actions
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Observation, previous: Transition | None) -> int:
        return int(self.rng.integers(self.actions))

    def finish(self, previous: Transition) -> None:
        return None

    def parameters(self) -> int:
        return 0


class ConvActorCritic:
    """A small convolutional actor-critic (advantage actor-critic, n-step returns), the
    conventional learner at the brain's information: the full screen, the body ports and the
    history port, the same rewards and episode ends.

    Two convolutions (16 maps of 8 by 8 at stride 4, 32 of 4 by 4 at stride 2) and a 256-unit
    layer that also reads the ports, then a softmax policy and a value. RMSprop at ``lr``,
    discount ``gamma``, ``steps``-step returns, entropy weight ``entropy``. A truncated
    episode's return bootstraps from the value of its last frame; an ended one from zero."""

    def __init__(
        self,
        resolution: str,
        actions: int,
        *,
        seed: int = 0,
        device: str = "cpu",
        lr: float = 7e-4,
        gamma: float = 0.99,
        steps: int = 5,
        entropy: float = 0.01,
        learn: bool = True,
    ) -> None:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch, self.actions, self.device = torch, actions, torch.device(device)
        self.gamma, self.steps, self.entropy, self.learn = gamma, steps, entropy, learn
        h, w = RESOLUTIONS[resolution]
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            flat = int(self.features(torch.zeros(1, 3, h, w)).shape[1])
        self.hidden = nn.Sequential(nn.Linear(flat + 3 + actions, 256), nn.ReLU())
        self.policy = nn.Linear(256, actions)
        self.value = nn.Linear(256, 1)
        self.net = nn.ModuleList([self.features, self.hidden, self.policy, self.value])
        self.net.to(self.device)
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=lr, alpha=0.99, eps=1e-5)
        self.generator = torch.Generator().manual_seed(seed)
        self.buffer: list[tuple[Any, Any, int, float]] = []
        self._last: tuple[Any, Any] | None = None
        self._action = -1

    def _tensors(self, observation: Observation) -> tuple[Any, Any]:
        torch = self.torch
        screen = torch.as_tensor(observation.screen, dtype=torch.float32).permute(2, 0, 1) / 255.0
        ports = torch.as_tensor(
            np.concatenate([observation.body, observation.history]), dtype=torch.float32
        )
        return screen.to(self.device), ports.to(self.device)

    def _forward(self, screens: Any, ports: Any) -> tuple[Any, Any]:
        torch = self.torch
        z = self.hidden(torch.cat([self.features(screens), ports], dim=1))
        return self.policy(z), self.value(z).squeeze(1)

    def act(self, observation: Observation, previous: Transition | None) -> int:
        torch = self.torch
        current = self._tensors(observation)
        if previous is not None and self._last is not None and self.learn:
            self.buffer.append((*self._last, self._action, previous.reward))
            if previous.done or len(self.buffer) >= self.steps:
                self._update(previous, None if previous.done else current)
        with torch.no_grad():
            logits, _ = self._forward(current[0][None], current[1][None])
            p = torch.softmax(logits, dim=1).cpu()
        self._action = int(torch.multinomial(p, 1, generator=self.generator)[0, 0])
        self._last = current
        return self._action

    def finish(self, previous: Transition) -> None:
        if self._last is None or not self.learn:
            return
        self.buffer.append((*self._last, self._action, previous.reward))
        self._update(previous, None)

    def _update(self, previous: Transition, following: tuple[Any, Any] | None) -> None:
        torch = self.torch
        cut = previous.observation if previous.outcome == "truncation" else None
        if following is None and cut is not None:
            following = self._tensors(cut)
        if following is None:
            tail = 0.0
        else:
            with torch.no_grad():
                tail = float(self._forward(following[0][None], following[1][None])[1][0])
        screens = torch.stack([b[0] for b in self.buffer])
        ports = torch.stack([b[1] for b in self.buffer])
        actions = torch.as_tensor([b[2] for b in self.buffer], device=self.device)
        returns, running = [], tail
        for b in reversed(self.buffer):
            running = b[3] + self.gamma * running
            returns.append(running)
        target = torch.as_tensor(returns[::-1], dtype=torch.float32, device=self.device)
        logits, values = self._forward(screens, ports)
        logp = torch.log_softmax(logits, dim=1)
        advantage = target - values
        policy_loss = -(logp.gather(1, actions[:, None]).squeeze(1) * advantage.detach()).mean()
        entropy = -(logp.exp() * logp).sum(dim=1).mean()
        loss = policy_loss + 0.5 * advantage.pow(2).mean() - self.entropy * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()
        self.buffer = []

    def parameters(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))
