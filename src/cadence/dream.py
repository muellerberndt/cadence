"""Learning by dreaming: an off-policy actor and critic in one settlement net.

The three-factor rule of ``plasticity`` learns from the stream as it comes. A
brain also learns from what it keeps: recent experience is replayed, and a
critic that reads the state and the action taken tells the actor which way
to move. Here that is one wiring with five sets of owners:

    input          the state, clamped
    hidden_actor   the actor's hidden owners
    action         a population code of the action (``Population``)
    hidden_critic  the critic's hidden owners, reading the state and the action
    q              one owner whose activation, times ``q_scale``, is the value of (state, action)

Acting is a free settlement with the state clamped: the action owners come
to rest at the actor's proposal. Around that proposal the net imagines a few
candidate actions, feels each in the critic (state and candidate clamped),
and takes the best, or draws among them by value while exploring. Learning
replays recent experience:

* **the critic's dream**: clamp the state and the action that was taken,
  settle, then nudge the ``q`` owner toward the target ``r + gamma * Q'``,
  where ``Q'`` is the best imagined candidate of the state that followed,
  felt with the *slow* strengths (the consolidated copy, which is what a
  target network is); every critic seam moves on its own two endpoints.
* **the actor imitates**: clamp the state, settle, then nudge the action
  owners toward the bump of the action that was taken; the actor's seams move
  by the same contrast. The critic chose the action; the actor learns to
  propose it next time.

Slow strengths follow the fast ones by a small fraction per update
(consolidation). No gradient is transported anywhere; every seam reads its
own two ends and one broadcast target.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .learning import Learner, LearnerConfig
from .plasticity import Population
from .rules import GradedRule
from .settle import Backend, Nudge, SettledState, Settlement
from .wiring import Wiring

__all__ = ["DreamConfig", "DreamActorCritic", "actor_critic_wiring", "DiscreteCode"]


@dataclass(frozen=True)
class DiscreteCode:
    """A discrete action as one owner per choice: written as a one-hot, read as the most
    active owner.

    Every choice is a candidate, so imagine-and-feel evaluates all of them.
    """

    actions: int
    dims: int = 1
    sigma: float = 0.0

    @property
    def size(self) -> int:
        return self.actions

    def read(self, activation: np.ndarray) -> np.ndarray:
        return np.argmax(activation.reshape(len(activation), self.actions), axis=1)[:, None].astype(
            float
        )

    def write(self, value: np.ndarray) -> np.ndarray:
        out = np.zeros((len(value), self.actions))
        out[np.arange(len(value)), np.asarray(value).reshape(-1).astype(int)] = 1.0
        return out

    def candidates(
        self, proposal: np.ndarray, rng: np.random.Generator, k: int, sigma: float
    ) -> np.ndarray:
        return np.broadcast_to(
            np.arange(self.actions, dtype=float)[None, :, None], (len(proposal), self.actions, 1)
        ).copy()


def actor_critic_wiring(
    inputs: int,
    hidden_actor: int,
    action_owners: int,
    hidden_critic: int,
    *,
    seed: int = 0,
    density: float = 1.0,
    init: float = 1.0,
) -> tuple[Wiring, np.ndarray, np.ndarray]:
    """One wiring for actor and critic; returns it with the actor's and the critic's overlap masks.

    Seams (each a tied symmetric pair): input-hidden_actor, hidden_actor-action,
    input-hidden_critic, action-hidden_critic, hidden_critic-q. Sets: ``input``,
    ``hidden_actor``, ``action``, ``hidden_critic``, ``q``.
    """
    rng = np.random.default_rng(seed)
    i0 = 0
    ha0 = inputs
    a0 = ha0 + hidden_actor
    hc0 = a0 + action_owners
    q0 = hc0 + hidden_critic
    n = q0 + 1
    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    sign: list[np.ndarray] = []

    def block(a_start: int, a_size: int, b_start: int, b_size: int, dens: float) -> None:
        mask = rng.random((a_size, b_size)) < dens
        rows, cols = np.nonzero(mask)
        magnitude = (
            rng.uniform(0.0, 1.0, size=len(rows))
            * np.sqrt(6.0 / (a_size * dens + b_size * dens))
            * init
        )
        s = rng.choice([-1.0, 1.0], size=len(rows)) * magnitude
        pre.append(a_start + rows)
        post.append(b_start + cols)
        sign.append(s)
        pre.append(b_start + cols)  # the feedback direction, tied to the forward one
        post.append(a_start + rows)
        sign.append(s.copy())

    block(i0, inputs, ha0, hidden_actor, density)
    block(ha0, hidden_actor, a0, action_owners, 1.0)
    block(i0, inputs, hc0, hidden_critic, density)
    block(a0, action_owners, hc0, hidden_critic, 1.0)
    block(hc0, hidden_critic, q0, 1, 1.0)
    wiring = Wiring.from_edges(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        sign=np.concatenate(sign),
        sets={
            "input": range(i0, ha0),
            "hidden_actor": range(ha0, a0),
            "action": range(a0, hc0),
            "hidden_critic": range(hc0, q0),
            "q": [q0],
        },
        label=f"actor-critic:{inputs}x{hidden_actor}x{action_owners}|{hidden_critic}",
    )
    actor_set = set(range(i0, hc0))  # input, hidden_actor, action
    critic_set = set(range(i0, ha0)) | set(range(a0, n))  # input, action, hidden_critic, q
    actor_mask = np.array(
        [
            int(p) in actor_set and int(q) in actor_set
            for p, q in zip(wiring.pre, wiring.post, strict=True)
        ]
    )
    critic_mask = np.array(
        [
            int(p) in critic_set and int(q) in critic_set
            for p, q in zip(wiring.pre, wiring.post, strict=True)
        ]
    )
    critic_mask &= (
        ~actor_mask
    )  # input-hidden_actor belongs to the actor, input-hidden_critic to the critic
    return wiring, actor_mask, critic_mask


@dataclass(frozen=True, slots=True)
class DreamConfig:
    gamma: float = 0.99
    q_scale: float = 100.0  # the value is q_scale times (activation of the q owner minus q_offset)
    q_offset: float = (
        0.5  # the activation that stands for a value of zero, so negative values have room
    )
    beta: float = 0.1  # nudge strength of both dreams
    eta_critic: float = 0.5
    eta_actor: float = 0.2
    eta_bias: float = 0.02
    consolidate: float = 0.005  # fraction of the fast strengths the slow ones take on per update
    memory: int = 100_000  # transitions kept
    batch: int = 128
    warmup: int = 2_000  # transitions before the first dream
    sigma: float = 0.2  # exploration noise on the action
    # >0: imagine this many actions around the actor's proposal, feel each in the critic,
    # take the best
    candidates: int = 0
    candidate_temperature: float = (
        0.0  # >0: draw among candidates by softmax of their values instead of the best
    )
    action_clamp: float = 3.0  # drive per unit bump level when an action is clamped for the critic
    free_steps: int = 60
    nudged_steps: int = 20
    tolerance: float = 3e-3
    centered: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class DreamActorCritic:
    def __init__(
        self,
        wiring: Wiring,
        actor_mask: np.ndarray,
        critic_mask: np.ndarray,
        population: Population,
        rule: GradedRule,
        config: DreamConfig | None = None,
        seed: int = 0,
        backend: Backend = "cpu",
        encode: Callable[[np.ndarray], np.ndarray] | None = None,
        state_dim: int | None = None,
    ) -> None:
        """``encode(states) -> drives`` lets the memory hold raw states of ``state_dim``
        numbers instead of drives."""
        self.config = config or DreamConfig()
        self.encode = encode
        self.population = population
        self.wiring = wiring
        self.actor_mask = np.asarray(actor_mask, dtype=bool)
        self.critic_mask = np.asarray(critic_mask, dtype=bool)
        self.rng = np.random.default_rng(seed)
        engine = Settlement(wiring, rule, backend=backend)
        self.learner = Learner(
            engine,
            wiring.sets["action"],
            LearnerConfig(
                beta=self.config.beta,
                eta=1.0,
                eta_bias=self.config.eta_bias,
                nudge="quadratic",
                tolerance=self.config.tolerance,
                free_steps=self.config.free_steps,
                nudged_steps=self.config.nudged_steps,
            ),
        )
        self.input_index = np.asarray(wiring.sets["input"])
        self.action_index = np.asarray(wiring.sets["action"])
        self.q_index = int(wiring.sets["q"][0])
        self.q_mask = np.zeros(wiring.n)
        self.q_mask[self.q_index] = 1.0
        self.no_actor = np.ones(wiring.n)
        self.no_actor[np.asarray(wiring.sets["hidden_actor"])] = (
            0.0  # the critic's dream silences the actor: the action owners hold the clamped action
        )
        self.action_clamp = self.config.action_clamp
        self.slow_scale = engine.edge_scale.copy()
        self.slow_bias = engine.bias.copy()
        n = wiring.n
        m = self.config.memory
        if encode is None:
            width = n
        else:
            assert state_dim is not None, "state_dim goes with encode"
            width = int(state_dim)
        self.mem_state = np.zeros((m, width))
        self.mem_next = np.zeros((m, width))
        self.mem_action = np.zeros((m, population.dims))
        self.mem_reward = np.zeros(m)
        self.mem_done = np.zeros(m, dtype=bool)
        self.size = 0
        self.cursor = 0
        self.updates = 0
        self.settle_steps: list[int] = []

    # -- engines

    @property
    def engine(self) -> Settlement:
        return self.learner.engine

    def target_engine(self) -> Settlement:
        return self.engine.with_parameters(edge_scale=self.slow_scale, bias=self.slow_bias)

    # -- acting

    def act(
        self, drive: np.ndarray, greedy: bool = False, warm: SettledState | None = None
    ) -> tuple[np.ndarray, SettledState]:
        """The actor's proposal; with ``candidates`` the net imagines that many actions around it,
        feels each in the critic, and takes the best (greedy) or draws by value (exploring)."""
        cfg = self.config
        state = self.engine.settle_batch(
            drive, steps=cfg.free_steps, state=warm, tolerance=cfg.tolerance
        )
        self.settle_steps.append(state.steps)
        proposal = self.population.read(state.activation[:, self.action_index])
        if cfg.candidates <= 0 or self.size < cfg.warmup:
            if hasattr(
                self.population, "candidates"
            ):  # discrete: the proposal, or a uniform draw while warming up
                action = (
                    proposal
                    if greedy
                    else self.rng.integers(0, self.population.size, size=(len(proposal), 1)).astype(
                        float
                    )
                )
            else:
                action = (
                    proposal
                    if greedy
                    else np.clip(
                        proposal + cfg.sigma * self.rng.standard_normal(proposal.shape), -1.0, 1.0
                    )
                )
            return action, state
        batch, dims = proposal.shape
        if hasattr(self.population, "candidates"):
            cands = self.population.candidates(proposal, self.rng, cfg.candidates, cfg.sigma)
        else:
            noise = cfg.sigma * self.rng.standard_normal((batch, cfg.candidates, dims))
            noise[:, 0] = 0.0  # the proposal itself is a candidate
            cands = np.clip(proposal[:, None, :] + noise, -1.0, 1.0)
        options: np.ndarray = np.asarray(cands, dtype=float)
        k = int(options.shape[1])
        drives = self._clamp_action(np.repeat(drive, k, axis=0), options.reshape(batch * k, dims))
        felt = self.engine.settle_batch(
            drives, steps=cfg.free_steps, tolerance=cfg.tolerance, mask=self.no_actor
        )
        values = self.value(felt).reshape(batch, k)
        pick: np.ndarray
        if greedy or cfg.candidate_temperature <= 0:
            pick = np.argmax(values, axis=1)
        else:
            z = values / cfg.candidate_temperature
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            u = np.asarray(self.rng.random(batch))
            pick = np.minimum((p.cumsum(axis=1) < u[:, None]).sum(axis=1), k - 1)
        chosen: np.ndarray = options[np.arange(batch), pick]
        return chosen, state

    def remember(
        self,
        drive: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_drive: np.ndarray,
        done: np.ndarray,
    ) -> None:
        for i in range(len(reward)):
            self.mem_state[self.cursor] = drive[i]
            self.mem_next[self.cursor] = next_drive[i]
            self.mem_action[self.cursor] = action[i]
            self.mem_reward[self.cursor] = reward[i]
            self.mem_done[self.cursor] = done[i]
            self.cursor = (self.cursor + 1) % self.config.memory
            self.size = min(self.size + 1, self.config.memory)

    # -- dreaming

    def _clamp_action(self, drive: np.ndarray, action: np.ndarray) -> np.ndarray:
        out = drive.copy()
        out[:, self.action_index] = self.action_clamp * self.population.write(action)
        return out

    def value(self, state: SettledState) -> np.ndarray:
        return (state.activation[:, self.q_index] - self.config.q_offset) * self.config.q_scale

    def _contrast(
        self, plus: SettledState, minus: SettledState, span: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch-mean contrast per overlap and per owner: the pairwise products as one matrix
        product, then read at the overlaps."""
        w = self.wiring
        sp, sm = plus.activation, minus.activation
        gram = (sp.T @ sp - sm.T @ sm) / (len(sp) * span)
        return gram[w.pre, w.post], (sp - sm).mean(axis=0) / span

    def _dream(
        self,
        drive: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        eta: float,
        owners: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Settle free under ``drive`` (``owners`` zeroes silenced ones), nudge the q owner
        toward ``target`` (in activation units), move the masked seams."""
        cfg = self.config
        free = self.engine.settle_batch(
            drive, steps=cfg.free_steps, tolerance=cfg.tolerance, mask=owners
        )
        full = np.zeros((len(drive), self.wiring.n))
        full[:, self.q_index] = target
        plus = self.engine.settle_batch(
            drive,
            steps=cfg.nudged_steps,
            state=free,
            nudge=Nudge(full, self.q_mask, cfg.beta),
            tolerance=cfg.tolerance,
            mask=owners,
        )
        if cfg.centered:
            minus = self.engine.settle_batch(
                drive,
                steps=cfg.nudged_steps,
                state=free,
                nudge=Nudge(full, self.q_mask, -cfg.beta),
                tolerance=cfg.tolerance,
                mask=owners,
            )
            overlap_term, owner_term = self._contrast(plus, minus, 2.0 * cfg.beta)
        else:
            overlap_term, owner_term = self._contrast(plus, free, cfg.beta)
        self.learner.trainable_overlaps = mask.copy()
        owners = np.zeros(self.wiring.n, dtype=bool)
        owners[np.unique(self.wiring.post[mask])] = True
        owners[self.input_index] = False
        owners[self.action_index] = (
            False  # the action owners' rest level is shared by actor and critic: fixed
        )
        self.learner.trainable_owners = owners
        report = self.learner.apply(eta * overlap_term, cfg.eta_bias * owner_term)
        report["q"] = float(self.value(free).mean())
        return report

    def _imitate(self, drive: np.ndarray, action: np.ndarray) -> dict[str, float]:
        cfg = self.config
        free = self.engine.settle_batch(drive, steps=cfg.free_steps, tolerance=cfg.tolerance)
        target = np.zeros((len(drive), self.wiring.n))
        target[:, self.action_index] = self.population.write(action)
        mask = np.zeros(self.wiring.n)
        mask[self.action_index] = 1.0
        plus = self.engine.settle_batch(
            drive,
            steps=cfg.nudged_steps,
            state=free,
            nudge=Nudge(target, mask, cfg.beta),
            tolerance=cfg.tolerance,
        )
        minus = self.engine.settle_batch(
            drive,
            steps=cfg.nudged_steps,
            state=free,
            nudge=Nudge(target, mask, -cfg.beta),
            tolerance=cfg.tolerance,
        )
        overlap_term, owner_term = self._contrast(plus, minus, 2.0 * cfg.beta)
        self.learner.trainable_overlaps = self.actor_mask.copy()
        owners = np.zeros(self.wiring.n, dtype=bool)
        owners[np.unique(self.wiring.post[self.actor_mask])] = True
        owners[self.input_index] = False
        owners[self.action_index] = False
        self.learner.trainable_owners = owners
        report = self.learner.apply(cfg.eta_actor * overlap_term, cfg.eta_bias * owner_term)
        report["q"] = float(
            np.abs(self.population.read(free.activation[:, self.action_index]) - action).mean()
        )
        return report

    def update(self) -> dict[str, float] | None:
        cfg = self.config
        if self.size < max(cfg.warmup, cfg.batch):
            return None
        idx = self.rng.integers(0, self.size, size=cfg.batch)
        state, nxt, action, reward, done = (
            self.mem_state[idx],
            self.mem_next[idx],
            self.mem_action[idx],
            self.mem_reward[idx],
            self.mem_done[idx],
        )
        if self.encode is not None:
            state, nxt = self.encode(state), self.encode(nxt)
        # the target, read with the slow strengths: the next state's own action and its value
        target_engine = self.target_engine()
        nxt_free = target_engine.settle_batch(nxt, steps=cfg.free_steps, tolerance=cfg.tolerance)
        next_action = self.population.read(nxt_free.activation[:, self.action_index])
        if cfg.candidates > 0:  # the next state's value is the best of its imagined candidates
            if hasattr(self.population, "candidates"):
                cands = self.population.candidates(next_action, self.rng, cfg.candidates, cfg.sigma)
            else:
                noise = cfg.sigma * self.rng.standard_normal(
                    (cfg.batch, cfg.candidates, next_action.shape[1])
                )
                noise[:, 0] = 0.0
                cands = np.clip(next_action[:, None, :] + noise, -1.0, 1.0)
            k = cands.shape[1]
            drives = self._clamp_action(np.repeat(nxt, k, axis=0), cands.reshape(cfg.batch * k, -1))
            felt = target_engine.settle_batch(
                drives, steps=cfg.free_steps, tolerance=cfg.tolerance, mask=self.no_actor
            )
            q_next = self.value(felt).reshape(cfg.batch, k).max(axis=1)
        else:
            nxt_q_state = target_engine.settle_batch(
                self._clamp_action(nxt, next_action),
                steps=cfg.free_steps,
                tolerance=cfg.tolerance,
                mask=self.no_actor,
            )
            q_next = self.value(nxt_q_state)
        y = reward + cfg.gamma * np.where(done, 0.0, q_next)
        y_activation = np.clip(y / cfg.q_scale + cfg.q_offset, 0.02, 0.98)
        # the critic's dream: state and action clamped, q pulled toward the target
        critic = self._dream(
            self._clamp_action(state, action),
            y_activation,
            self.critic_mask,
            cfg.eta_critic,
            owners=self.no_actor,
        )
        # the actor learns what was taken: the taken action's bump nudged onto the action owners
        actor = self._imitate(state, action)
        # consolidation: the slow strengths follow
        self.slow_scale += cfg.consolidate * (self.engine.edge_scale - self.slow_scale)
        self.slow_bias += cfg.consolidate * (self.engine.bias - self.slow_bias)
        self.updates += 1
        return {
            "critic_step": critic["scale_step"],
            "actor_step": actor["scale_step"],
            "q": critic["q"],
            "target": float(y.mean()),
            "td": float(np.abs(y - critic["q"]).mean()),
        }

    def parameters(self) -> int:
        return self.learner.parameters()

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "updates": self.updates,
            "learner": self.learner.to_dict(),
        }
