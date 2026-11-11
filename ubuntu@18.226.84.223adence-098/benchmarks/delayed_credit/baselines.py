"""The delayed-credit chamber's conventional learners (cadence#283), outside the library.

Both live the brain's episodes in the same order, with the same observations and the same
outcome after the episode's last act, and both are sized to the brain's parameter count.

- ``LSTMActorCritic``: an LSTM whose state carries across the moments of an episode and is
  cleared when an episode begins; a softmax policy and a linear value read its state. One
  update per episode, after its outcome: generalized advantage estimation (``gamma``,
  ``lam``) over the episode's moments, the policy gradient with an entropy bonus and the
  value's squared error, back-propagated through every moment of the episode (exact
  gradients through time), Adam with the gradient's norm clipped. Torch on the CPU.
- ``TDLambda``: the online TD(lambda) actor-critic of ``benchmarks/competitors.py`` (an
  Elman cell, eligibility traces over every weight, an update at every moment, the gradient
  through the present moment only), loaded by path.

Work is counted as multiply-accumulates, analytically: the LSTM's forward pass per moment,
its backward pass at twice the forward and Adam at ten per parameter per update; the TD
learner's own ledger.
"""

from __future__ import annotations

from typing import Any

import numpy as np

LSTM_SETTINGS = {"lr": 3e-3, "gamma": 0.99, "lam": 0.95, "entropy": 0.01, "value": 0.5,
                 "clip": 1.0}
TD_SETTINGS = {"alpha": 0.02, "alpha_value": 0.04, "gamma": 0.99, "lam": 0.95,
               "temperature": 1.0, "clip": 5.0}


def lstm_parameters(inputs: int, actions: int, hidden: int) -> int:
    """Torch's LSTM (two bias vectors per gate) with a policy and a value head."""
    return 4 * hidden * (inputs + hidden + 2) + (hidden + 1) * (actions + 1)


def elman_parameters(inputs: int, actions: int, hidden: int) -> int:
    return hidden * (inputs + hidden + 1) + (hidden + 1) * (actions + 1)


def matched_hidden(count: int, inputs: int, actions: int, kind: str) -> int:
    """The hidden width whose parameter count is closest to ``count``."""
    size = lstm_parameters if kind == "lstm" else elman_parameters
    return min(range(1, 1024), key=lambda h: abs(size(inputs, actions, h) - count))


class LSTMActorCritic:
    """One LSTM actor-critic life: ``begin`` an episode, ``act`` each moment, ``finish``
    with the outcome (one update). ``greedy`` reads without learning."""

    def __init__(self, inputs: int, actions: int, hidden: int, seed: int,
                 **settings: float) -> None:
        import torch

        self.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(int(seed))
        self.settings = {**LSTM_SETTINGS, **settings}
        self.inputs, self.actions, self.hidden = inputs, actions, hidden
        self.cell = torch.nn.LSTM(inputs, hidden)
        self.policy_head = torch.nn.Linear(hidden, actions)
        self.value_head = torch.nn.Linear(hidden, 1)
        self.modules = torch.nn.ModuleList([self.cell, self.policy_head, self.value_head])
        self.optimizer = torch.optim.Adam(self.modules.parameters(), lr=self.settings["lr"])
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 283]))
        self.work = {"updates": 0, "moments": 0, "multiply_accumulates": 0.0}
        self._episode: list[Any] = []
        self._state: Any = None

    @property
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.modules.parameters()))

    @property
    def forward_macs(self) -> int:
        return 4 * self.hidden * (self.inputs + self.hidden) + self.hidden * (self.actions + 1)

    def describe(self) -> dict[str, Any]:
        return {"kind": "lstm", "hidden": self.hidden, "parameters": self.parameter_count,
                **self.settings}

    def begin(self) -> None:
        self._episode = []
        self._state = None

    def _forward(self, x: np.ndarray) -> tuple[Any, Any]:
        torch = self.torch
        row = torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, 1, -1)
        out, self._state = self.cell(row, self._state)
        h = out[0, 0]
        self.work["moments"] += 1
        self.work["multiply_accumulates"] += self.forward_macs
        return self.policy_head(h), self.value_head(h)[0]

    def act(self, x: np.ndarray) -> int:
        logits, value = self._forward(x)
        logp = self.torch.log_softmax(logits, dim=0)
        probs = logp.detach().exp().numpy().astype(float)
        action = int(min(np.searchsorted(np.cumsum(probs), self.rng.random()), len(probs) - 1))
        self._episode.append((logp, value, action))
        return action

    def finish(self, reward: float) -> float:
        """The episode's outcome after its last act: one update; returns the loss."""
        torch, s = self.torch, self.settings
        steps = len(self._episode)
        values = torch.stack([v for _, v, _ in self._episode])
        with torch.no_grad():
            v = values.numpy().astype(float)
            rewards = np.zeros(steps)
            rewards[-1] = float(reward)
            advantage = np.zeros(steps)
            running = 0.0
            for t in reversed(range(steps)):
                following = v[t + 1] if t + 1 < steps else 0.0
                delta = rewards[t] + s["gamma"] * following - v[t]
                running = delta + s["gamma"] * s["lam"] * running
                advantage[t] = running
            returns = advantage + v
        logps = torch.stack([lp[a] for lp, _, a in self._episode])
        entropy = torch.stack([-(lp.exp() * lp).sum() for lp, _, _ in self._episode])
        adv = torch.as_tensor(advantage, dtype=torch.float32)
        ret = torch.as_tensor(returns, dtype=torch.float32)
        loss = (-(adv * logps).sum() + s["value"] * ((ret - values) ** 2).sum()
                - s["entropy"] * entropy.sum()) / steps
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.modules.parameters(), s["clip"])
        self.optimizer.step()
        self.work["updates"] += 1
        self.work["multiply_accumulates"] += 2 * steps * self.forward_macs
        self.work["multiply_accumulates"] += 10 * self.parameter_count
        self._episode = []
        self._state = None
        return float(loss.detach())

    def greedy(self, observations: list[np.ndarray], respond: Any) -> list[int]:
        """Greedy answers along an episode without learning: ``respond(action)`` returns the
        next observation, or None when the episode has ended."""
        torch = self.torch
        answers: list[int] = []
        saved, self._state = self._state, None
        x: np.ndarray | None = observations[0]
        with torch.no_grad():
            while x is not None:
                logits, _ = self._forward(x)
                answers.append(int(torch.argmax(logits)))
                x = respond(answers[-1])
        self._state = saved
        return answers

    def probability(self, x: np.ndarray, action: int) -> float:
        """The policy's probability of ``action`` at an episode's first observation."""
        torch = self.torch
        saved, self._state = self._state, None
        with torch.no_grad():
            logits, _ = self._forward(x)
            p = float(torch.softmax(logits, dim=0)[action])
        self._state = saved
        return p


class TDLambda:
    """The online TD(lambda) actor-critic of ``benchmarks/competitors.py`` (Elman cell), in the
    chamber's episode protocol."""

    def __init__(self, inputs: int, actions: int, hidden: int, seed: int,
                 **settings: float) -> None:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "competitors.py"
        spec = importlib.util.spec_from_file_location("competitors", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.settings = {**TD_SETTINGS, **settings}
        self.learner = module.ActorCritic(inputs, actions, int(seed), cell="elman",
                                          hidden=int(hidden), **self.settings)
        self.hidden = hidden
        self._owed: float | None = None
        self._first = True

    @property
    def parameter_count(self) -> int:
        return int(self.learner.parameter_count)

    @property
    def work(self) -> dict[str, Any]:
        return dict(self.learner.work)

    def describe(self) -> dict[str, Any]:
        return {"kind": "tdlambda", "hidden": self.hidden, "parameters": self.parameter_count,
                **self.settings}

    def act(self, x: np.ndarray, first: bool) -> int:
        """The moment's action; at an episode's first moment the previous outcome arrives
        with the end of its episode."""
        if first and self._owed is not None:
            return self.learner.act(x, self._owed, done=True)
        if first:
            return self.learner.act(x, None)
        return self.learner.act(x, 0.0)

    def finish(self, reward: float) -> None:
        self._owed = float(reward)

    def greedy(self, observations: list[np.ndarray], respond: Any) -> list[int]:
        twin = self.learner.clone(cleared=True)
        answers: list[int] = []
        x: np.ndarray | None = observations[0]
        while x is not None:
            answers.append(twin.step(x))
            x = respond(answers[-1])
        return answers

    def probability(self, x: np.ndarray, action: int) -> float:
        return float(self.learner.policy(x)[action])


def lstm_gradient_check(seed: int = 0) -> float:
    """The largest relative difference between the LSTM learner's autograd gradient of its
    episode loss and a finite difference, on a tiny learner (for the tests)."""
    import torch

    learner = LSTMActorCritic(3, 2, 4, seed)
    learner.modules.double()
    rng = np.random.default_rng(seed)
    xs = [np.eye(3)[rng.integers(3)] for _ in range(5)]

    def loss_of() -> Any:
        learner.begin()
        state = None
        total = torch.zeros((), dtype=torch.float64)
        for t, x in enumerate(xs):
            row = torch.as_tensor(x, dtype=torch.float64).reshape(1, 1, -1)
            out, state = learner.cell(row, state)
            logits = learner.policy_head(out[0, 0])
            value = learner.value_head(out[0, 0])[0]
            total = total + torch.log_softmax(logits, 0)[t % 2] + 0.5 * value**2
        return total

    weight = learner.cell.weight_hh_l0
    loss = loss_of()
    (grad,) = torch.autograd.grad(loss, [weight])
    worst = 0.0
    eps = 1e-6
    for index in [(0, 0), (3, 2), (7, 1)]:
        with torch.no_grad():
            weight[index] += eps
            up = float(loss_of())
            weight[index] -= 2 * eps
            down = float(loss_of())
            weight[index] += eps
        numeric = (up - down) / (2 * eps)
        worst = max(worst, abs(numeric - float(grad[index])) / max(1e-8, abs(numeric)))
    return worst


__all__ = ["LSTMActorCritic", "TDLambda", "lstm_parameters", "elman_parameters",
           "matched_hidden", "lstm_gradient_check"]
