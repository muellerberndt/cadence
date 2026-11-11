"""The matched baseline of the pixel chamber: a small convolutional actor-critic trained
online by gradient on the same frames, the same lessons or rewards and the same number of
frames as a brain's life (cadence#284).

Two convolutions as in the Atari network of Mnih and colleagues (2013): 16 filters of 8 by 8 at
stride 4, then 32 of 4 by 4 at stride 2, each rectified, a rectified hidden layer of ``hidden``
units, a policy head over the actions and a value head. ``hidden`` sets the parameter count,
to match a brain's. Under lessons the policy head learns the label by cross-entropy; under
reward the one-step actor-critic learns from the reward of its sampled action, the critic's
value as the baseline, with an entropy bonus. Every update is one Adam step on the batch of
frames the brain sees at that moment. It needs PyTorch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RATE = {"lesson": 1e-3, "reward": 3e-4}
ENTROPY = 0.01


def _torch() -> Any:
    import torch

    return torch


def conv_out(size: int, kernel: int, stride: int) -> int:
    return (size - kernel) // stride + 1


def geometry(shape: tuple[int, int, int]) -> dict[str, int]:
    height, width, channels = shape
    h1, w1 = conv_out(height, 8, 4), conv_out(width, 8, 4)
    h2, w2 = conv_out(h1, 4, 2), conv_out(w1, 4, 2)
    return {"channels": channels, "h1": h1, "w1": w1, "h2": h2, "w2": w2, "flat": 32 * h2 * w2}


def parameters(shape: tuple[int, int, int], actions: int, hidden: int) -> int:
    g = geometry(shape)
    conv = 16 * (64 * g["channels"] + 1) + 32 * (16 * 16 + 1)
    return conv + hidden * (g["flat"] + 1) + (actions + 1) * (hidden + 1)


def macs(shape: tuple[int, int, int], actions: int, hidden: int) -> int:
    """Multiply-accumulates of one forward pass of one frame."""
    g = geometry(shape)
    conv = g["h1"] * g["w1"] * 16 * 64 * g["channels"] + g["h2"] * g["w2"] * 32 * 16 * 16
    return conv + g["flat"] * hidden + hidden * (actions + 1)


def hidden_for(shape: tuple[int, int, int], actions: int, target: int, least: int = 16) -> int:
    """The hidden width whose parameter count is nearest ``target`` (at least ``least``)."""
    g = geometry(shape)
    fixed = parameters(shape, actions, 0)
    return max(least, round((target - fixed) / (g["flat"] + 1 + actions + 1)))


class ConvActorCritic:
    def __init__(self, shape: tuple[int, int, int], actions: int, hidden: int, seed: int,
                 condition: str = "lesson") -> None:
        torch = _torch()
        if condition not in RATE:
            raise ValueError(f"unknown condition {condition!r}")
        torch.manual_seed(int(seed))
        nn = torch.nn
        g = geometry(shape)
        self.shape, self.actions, self.hidden, self.condition = shape, actions, hidden, condition
        self.body = nn.Sequential(
            nn.Conv2d(g["channels"], 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(), nn.Flatten(),
            nn.Linear(g["flat"], hidden), nn.ReLU(),
        ).double()
        self.policy = nn.Linear(hidden, actions).double()
        self.value = nn.Linear(hidden, 1).double()
        modules = [self.body, self.policy, self.value]
        self.optimizer = torch.optim.Adam(
            [p for m in modules for p in m.parameters()], lr=RATE[condition]
        )
        self.rng = np.random.default_rng(np.random.SeedSequence([28404, int(seed)]))
        self.frames = 0

    def parameters(self) -> int:
        return sum(
            int(p.numel()) for m in (self.body, self.policy, self.value) for p in m.parameters()
        )

    def _input(self, batch: np.ndarray) -> Any:
        torch = _torch()
        x = np.asarray(batch, dtype=float).reshape(len(batch), *self.shape)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))

    def logits(self, batch: np.ndarray) -> Any:
        return self.policy(self.body(self._input(batch)))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            return np.asarray(self.logits(batch).argmax(dim=1).numpy())

    def lesson(self, batch: np.ndarray, labels: np.ndarray) -> None:
        torch = _torch()
        loss = torch.nn.functional.cross_entropy(
            self.logits(batch), torch.as_tensor(np.asarray(labels), dtype=torch.long)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.frames += len(batch)

    def act(self, batch: np.ndarray) -> np.ndarray:
        """A sampled action per frame, from the policy."""
        torch = _torch()
        with torch.no_grad():
            p = torch.softmax(self.logits(batch), dim=1).numpy()
        return np.array([self.rng.choice(self.actions, p=row / row.sum()) for row in p])

    def reward(self, batch: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        """One actor-critic step on the frames, the actions taken and their rewards."""
        torch = _torch()
        hidden = self.body(self._input(batch))
        logp = torch.log_softmax(self.policy(hidden), dim=1)
        value = self.value(hidden).squeeze(1)
        r = torch.as_tensor(np.asarray(rewards, dtype=float))
        a = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        advantage = (r - value).detach()
        chosen = logp.gather(1, a.unsqueeze(1)).squeeze(1)
        entropy = -(logp.exp() * logp).sum(dim=1)
        loss = (-(advantage * chosen) + 0.5 * (r - value) ** 2 - ENTROPY * entropy).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.frames += len(batch)
