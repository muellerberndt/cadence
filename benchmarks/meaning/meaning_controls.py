"""The matched controls of the meaning chamber (cadence#334): learners that live the parrot's
very exchanges through the calls the night's engine makes on the brain (``act``, ``learn``,
``lesson``), with the same observations, the same pistachios at the same moments and the same
lessons (the self-hearing lesson and the pistachio-gated lesson on their own replies).

- ``lstm`` and ``transformer`` (torch): a recurrent network, or a causal transformer over the
  moments of the block, a head per motor slot and a value head. Each learns once a block, when its
  exchanges end: REINFORCE with the value as baseline (discount ``GAMMA`` a moment), the value's
  squared error, and every lesson of the block as cross-entropy on the acts it names; Adam at
  ``RATE``, gradients clipped at 1.
- ``delta``, the audit's plain readout: linear, from the moment and the two before it, to each
  slot; a lesson moves the named act's row by the delta rule; a reward moves the acts of the
  moments before it toward themselves, at the discount.
- ``table``, the audit's table: the key is the set of word forms heard in the last ``KEEP``
  moments, with no order among them; a key counts the acts that lessons named and that rewards
  paid; the most counted act is said, or silence where the key holds none.
"""

from __future__ import annotations

from typing import Any

import numpy as np

GAMMA = 0.9
RATE = 0.001
KEEP = 8


class Network:
    """An LSTM or a causal transformer with the brain's calls."""

    def __init__(self, kind: str, inputs: int, slots: tuple[int, ...], width: int, seed: int = 0):
        import torch

        torch.manual_seed(int(seed))
        self.kind, self.slots, self.width = kind, tuple(slots), int(width)
        nn = torch.nn
        self.embed = nn.Linear(int(inputs), self.width)
        if kind == "lstm":
            self.core: Any = nn.LSTM(self.width, self.width, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(self.width, nhead=4, dim_feedforward=2 * self.width,
                                               batch_first=True, dropout=0.0)
            self.core = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.position = nn.Parameter(torch.zeros(1, 64, self.width))
        self.heads = nn.ModuleList([nn.Linear(self.width, s) for s in self.slots])
        self.value_head = nn.Linear(self.width, 1)
        self.params = [*self.embed.parameters(), *self.core.parameters(),
                       *self.heads.parameters(), *self.value_head.parameters()]
        if kind != "lstm":
            self.params.append(self.position)
        self.optimizer = torch.optim.Adam(self.params, lr=RATE)
        self.rng = np.random.default_rng(int(seed))
        self.last_settlement: dict[str, Any] = {"steps": 0}
        self.macs = 0.0
        self.reset()

    @property
    def torch(self) -> Any:
        import torch

        return torch

    def parameters(self) -> int:
        return int(sum(p.numel() for p in self.params))

    def reset(self) -> None:
        self.xs: list[np.ndarray] = []
        self.acts: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.lessons: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.pending = False

    @property
    def pending_feedback(self) -> bool:
        return self.pending

    def _forward(self, xs: np.ndarray) -> tuple[list[Any], Any]:
        torch = self.torch
        h = torch.relu(self.embed(torch.as_tensor(xs, dtype=torch.float32)))
        if self.kind == "lstm":
            h, _ = self.core(h)
        else:
            moments = h.shape[1]
            mask = torch.triu(torch.full((moments, moments), float("-inf")), diagonal=1)
            h = self.core(h + self.position[:, :moments], mask=mask)
        self.macs += float(xs.shape[0] * xs.shape[1] * self.parameters())
        return [head(h) for head in self.heads], self.value_head(h).squeeze(-1)

    def act(self, x: Any, greedy: bool = False, temperature: Any = None) -> np.ndarray:
        self.xs.append(np.asarray(x, dtype=np.float32))
        with self.torch.no_grad():
            logits, _ = self._forward(np.stack(self.xs, axis=1))
        out = []
        for lg in logits:
            last = lg[:, -1].double().numpy()
            if greedy:
                out.append(np.argmax(last, axis=1))
            else:
                prob = np.exp(last - last.max(axis=1, keepdims=True))
                prob /= prob.sum(axis=1, keepdims=True)
                out.append(np.array([self.rng.choice(len(q), p=q) for q in prob]))
        action = np.stack(out, axis=1).astype(np.int64)
        self.acts.append(action)
        self.pending = True
        return action

    def learn(self, reward: Any, done: Any, x: Any, **_: Any) -> dict[str, float]:
        if self.acts:  # reward j pays act j; an act left from the block before takes none
            self.rewards.append(np.asarray(reward, dtype=np.float64))
        self.pending = False
        if np.asarray(done).all():
            self._update()
        return {"free_steps": 0.0}

    def lesson(self, x: Any, labels: Any, weights: Any = None) -> dict[str, float]:
        labels = np.asarray(labels).reshape(len(labels), -1)
        w = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
        self.lessons.append((np.asarray(x, dtype=np.float32), labels, w))
        return {"total_row_sweeps": 0.0}

    def _update(self) -> None:
        torch = self.torch
        steps = min(len(self.acts), len(self.rewards))
        if not steps:
            self.reset()
            return
        xs = np.stack(self.xs[:steps], axis=1)
        logits, value = self._forward(xs)
        rewards = np.stack(self.rewards[:steps], axis=1)
        returns = np.zeros_like(rewards)
        running = np.zeros(rewards.shape[0])
        for t in range(steps - 1, -1, -1):
            running = rewards[:, t] + GAMMA * running
            returns[:, t] = running
        g = torch.as_tensor(returns, dtype=torch.float32)
        acts = torch.as_tensor(np.stack(self.acts[:steps], axis=1))
        advantage = (g - value).detach()
        loss = ((g - value) ** 2).mean()
        for s, lg in enumerate(logits):
            logp = torch.log_softmax(lg, dim=-1).gather(-1, acts[..., s:s + 1]).squeeze(-1)
            loss = loss - (logp * advantage).mean()
        if self.lessons:  # each lesson's rows, read as one moment with no context
            xs_l = np.stack([x for x, _, _ in self.lessons], axis=1)
            lg_l, _ = self._forward(xs_l)
            for j, (_, labels, w) in enumerate(self.lessons):
                weight = torch.as_tensor(w, dtype=torch.float32)
                for s in range(labels.shape[1]):
                    ce = torch.nn.functional.cross_entropy(
                        lg_l[s][:, j], torch.as_tensor(labels[:, s]), reduction="none")
                    loss = loss + (ce * weight).sum() / max(1.0, float(w.sum())) / len(self.lessons)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.optimizer.step()
        self.macs += 2.0 * xs.shape[0] * xs.shape[1] * self.parameters()
        self.reset()


class Delta:
    """The audit's plain readout: linear, from the moment and the two before it."""

    def __init__(self, inputs: int, slots: tuple[int, ...], seed: int = 0, rate: float = 0.05):
        self.slots, self.rate = tuple(slots), float(rate)
        self.w = [np.zeros((3 * int(inputs) + 1, s)) for s in self.slots]
        self.rng = np.random.default_rng(int(seed))
        self.last_settlement: dict[str, Any] = {"steps": 0}
        self.reset()

    def parameters(self) -> int:
        return int(sum(w.size for w in self.w))

    def reset(self) -> None:
        self.history: list[np.ndarray] = []
        self.taken: list[tuple[np.ndarray, np.ndarray]] = []
        self.pending = False

    @property
    def pending_feedback(self) -> bool:
        return self.pending

    def _z(self, x: np.ndarray) -> np.ndarray:
        zero = np.zeros_like(x)
        before = self.history[-1] if self.history else zero
        earlier = self.history[-2] if len(self.history) > 1 else zero
        return np.concatenate([x, before, earlier, np.ones((len(x), 1))], axis=1)

    def act(self, x: Any, greedy: bool = False, temperature: Any = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = self._z(x)
        out = []
        for w in self.w:
            q = z @ w
            if greedy:
                out.append(np.argmax(q, axis=1))
            else:
                prob = np.exp((q - q.max(axis=1, keepdims=True)) / 0.2)
                prob /= prob.sum(axis=1, keepdims=True)
                out.append(np.array([self.rng.choice(len(r), p=r) for r in prob]))
        action = np.stack(out, axis=1).astype(np.int64)
        self.history.append(x)
        self.taken = (self.taken + [(z, action)])[-12:]
        self.pending = True
        return action

    def learn(self, reward: Any, done: Any, x: Any, **_: Any) -> dict[str, float]:
        reward = np.asarray(reward, dtype=np.float64)
        if reward.any():
            for back, (z, action) in enumerate(reversed(self.taken)):
                credit = reward * GAMMA ** back
                for s, w in enumerate(self.w):
                    for i in np.flatnonzero(credit):
                        w[:, action[i, s]] += self.rate * credit[i] * z[i]
        if np.asarray(done).all():
            self.history, self.taken = [], []
        self.pending = False
        return {"free_steps": 0.0}

    def lesson(self, x: Any, labels: Any, weights: Any = None) -> dict[str, float]:
        z = self._z(np.asarray(x, dtype=np.float64))
        labels = np.asarray(labels).reshape(len(z), -1)
        w8 = np.ones(len(z)) if weights is None else np.asarray(weights, dtype=np.float64)
        for s, w in enumerate(self.w):
            error = np.eye(w.shape[1])[labels[:, s]] - z @ w
            w += self.rate * (z * w8[:, None]).T @ error / max(1.0, float(w8.sum()))
        return {"total_row_sweeps": 0.0}


class Table:
    """The audit's table: counts by the set of word forms heard in the last ``KEEP`` moments."""

    def __init__(self, decode: Any, slots: tuple[int, ...], seed: int = 0):
        self.decode, self.slots = decode, tuple(slots)
        self.counts: list[dict[frozenset[str], np.ndarray]] = [{} for _ in self.slots]
        self.last_settlement: dict[str, Any] = {"steps": 0}
        self.reset()

    def parameters(self) -> int:
        return int(sum(len(c) * s for c, s in zip(self.counts, self.slots, strict=True)))

    def reset(self) -> None:
        self.heard: list[list[str | None]] = []
        self.taken: list[tuple[list[frozenset[str]], np.ndarray]] = []
        self.pending = False

    @property
    def pending_feedback(self) -> bool:
        return self.pending

    def _keys(self, x: np.ndarray, *, keep: bool = True) -> list[frozenset[str]]:
        heard = self.heard + [[self.decode(row) for row in x]]
        if keep:
            self.heard = heard
        return [frozenset(w for moment in heard[-KEEP:] for w in [moment[i]] if w is not None)
                for i in range(len(x))]

    def act(self, x: Any, greedy: bool = False, temperature: Any = None) -> np.ndarray:
        keys = self._keys(np.asarray(x))
        out = []
        for s, size in enumerate(self.slots):
            got = []
            for k in keys:
                c = self.counts[s].get(k)
                got.append(int(np.argmax(c)) if c is not None and c.max() > 0 else (
                    size - 1 if s == 0 else 0))
            out.append(np.array(got))
        action = np.stack(out, axis=1).astype(np.int64)
        self.taken = (self.taken + [(keys, action)])[-12:]
        self.pending = True
        return action

    def _count(self, keys: list[frozenset[str]], action: np.ndarray, amount: np.ndarray) -> None:
        for s, size in enumerate(self.slots):
            for i, k in enumerate(keys):
                if amount[i] > 0:
                    self.counts[s].setdefault(k, np.zeros(size))[action[i, s]] += amount[i]

    def learn(self, reward: Any, done: Any, x: Any, **_: Any) -> dict[str, float]:
        reward = np.asarray(reward, dtype=np.float64)
        if reward.any():
            for back, (keys, action) in enumerate(reversed(self.taken)):
                self._count(keys, action, reward * GAMMA ** back)
        if np.asarray(done).all():
            self.heard, self.taken = [], []
        self.pending = False
        return {"free_steps": 0.0}

    def lesson(self, x: Any, labels: Any, weights: Any = None) -> dict[str, float]:
        keys = self._keys(np.asarray(x), keep=False)
        labels = np.asarray(labels).reshape(len(keys), -1)
        w = np.ones(len(keys)) if weights is None else np.asarray(weights, dtype=np.float64)
        self._count(keys, labels, w)
        return {"total_row_sweeps": 0.0}


class Wiring:
    """The ``connectome.synapses`` the engine reads: the control's weights."""

    def __init__(self, count: int):
        self.synapses = int(count)


def control(kind: str, parrot: Any, *, parameters: int, seed: int = 0) -> Any:
    """A control of ``kind`` placed as ``parrot``'s brain: its senses and its motor slots (the
    voice and the beak), the networks near ``parameters`` weights."""
    slots = (len(parrot.voice) + 1, 2)
    agent: Any
    if kind in ("lstm", "transformer"):
        width = 32
        while Network(kind, parrot.inputs, slots, width + 32, seed).parameters() <= parameters:
            width += 32
        agent = Network(kind, parrot.inputs, slots, width, seed)
    elif kind == "delta":
        agent = Delta(parrot.inputs, slots, seed)
    elif kind == "table":
        ear = parrot.eyes.width

        def decode(row: np.ndarray) -> str | None:
            return parrot.forms.heard_as(np.asarray(row)[ear:ear + parrot.forms.width])

        agent = Table(decode, slots, seed)
    else:
        raise ValueError(kind)
    agent.connectome = Wiring(agent.parameters())
    parrot.brain = agent
    parrot.synapses = agent.connectome.synapses
    return agent
