"""A long reward life outside VizDoom: one action pays, the others do not (cadence#296).

The chamber keeps the Doom body's shape and drops its engine. A player reads one vector per
moment: a retina (a fixed scene with noise, the statistics of a Doom basic screen), three
constant body ports and the history port (the episode's previous action, one-hot). It answers
one of ``actions`` buttons. Each moment costs ``cost``; the paying button ends the episode with
``+1`` with probability ``pay``; an episode that reaches ``length`` moments ends at its deadline
with nothing more. At the founders (``pay=0.04``, ``length=75``, three buttons) uniform random
buttons complete .63 of episodes, as random buttons complete Doom basic.

    PYTHONPATH=src python benchmarks/actor_latch/chamber.py --episodes 400 --seed 0 \\
        > results/founder-0.json
    PYTHONPATH=src python benchmarks/actor_latch/chamber.py --read results/*.json

``--set name=value`` composes the brain with a named setting (``--set actor_eta=0.1``);
``--player random`` and ``--player frozen`` are the controls. ``--read`` prints each life by
quarters of its episodes: the button shares, the paying button's share, completion, return, the
motor activations, saturation, the actor's synapse and bias steps, the critic's error and the
trace.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# the Doom body's 84 by 84 retina: 21 by 21 blocks of 3 colours
RETINA = 21 * 21 * 3
# the learn report's fields kept per moment
FIELDS = ("saturation", "scale_step", "bias_step", "td_error", "delta", "dopamine", "value",
          "trace", "capped")


@dataclass
class Transition:
    reward: float
    done: bool
    outcome: str | None
    observation: np.ndarray


class Bandit:
    """One paying button among ``actions``, in episodes with a cost per moment and a deadline.

    The scene is fixed per world seed (a Doom basic screen's mean and spread per pixel); each
    moment adds independent noise. The paying button is ``paying``; it ends the episode with
    ``+1`` at probability ``pay`` per press."""

    def __init__(
        self,
        *,
        actions: int = 3,
        paying: int = 0,
        width: int = RETINA,
        pay: float = 0.04,
        cost: float = 0.01,
        length: int = 75,
        noise: float = 0.025,
        seed: int = 0,
        history: bool = True,
    ) -> None:
        if not 0 <= paying < actions:
            raise ValueError("the paying button is one of the buttons")
        self.history = history  # False: the history port stays dark
        self.actions, self.paying, self.width = actions, paying, width
        self.pay, self.cost, self.length, self.noise = pay, cost, length, noise
        self.rng = np.random.default_rng([seed, 296])
        self.scene = np.clip(self.rng.normal(0.238, 0.112, width), 0.0, 1.0)
        self.ports = np.array([1.0, 0.0, 0.96])  # health, armour, ammunition as in basic
        self.moment = 0
        self.previous = -1

    @property
    def inputs(self) -> int:
        return self.width + 3 + self.actions

    def observe(self) -> np.ndarray:
        seen = np.clip(self.scene + self.rng.normal(0.0, self.noise, self.width), 0.0, 1.0)
        history = np.zeros(self.actions)
        if self.previous >= 0 and self.history:
            history[self.previous] = 1.0
        return np.concatenate([seen, self.ports, history])

    def reset(self) -> np.ndarray:
        self.moment, self.previous = 0, -1
        return self.observe()

    def step(self, action: int) -> Transition:
        if not 0 <= action < self.actions:
            raise ValueError("an action is one of the buttons")
        self.moment += 1
        self.previous = action
        reward = -self.cost
        if action == self.paying and self.rng.random() < self.pay:
            return Transition(reward + 1.0, True, "completion", self.observe())
        if self.moment >= self.length:
            return Transition(reward, True, "deadline", self.observe())
        return Transition(reward, False, None, self.observe())


class Survival(Bandit):
    """Health gathering's shape: every moment alive pays ``alive``, health drains by ``drain``
    a moment and the paying button finds a medkit (``+heal`` health, at most 1) with
    probability ``pay``; at no health the episode ends in death (``death``), and at ``length``
    moments in completion. Health is the first body port. At the founders uniform random
    buttons live about 190 moments and die, and the paying button held lives to the length."""

    def __init__(
        self,
        *,
        actions: int = 3,
        paying: int = 0,
        width: int = RETINA,
        pay: float = 0.0625,
        heal: float = 0.25,
        drain: float = 0.0104,
        alive: float = 0.01,
        death: float = -1.0,
        length: int = 525,
        noise: float = 0.025,
        seed: int = 0,
        history: bool = True,
    ) -> None:
        super().__init__(actions=actions, paying=paying, width=width, pay=pay, cost=-alive,
                         length=length, noise=noise, seed=seed, history=history)
        # a health gathering screen's mean and spread per pixel
        self.scene = np.clip(self.rng.normal(0.112, 0.092, width), 0.0, 1.0)
        self.heal, self.drain, self.alive, self.death = heal, drain, alive, death
        self.ports = np.array([1.0, 0.0, 0.0])

    def reset(self) -> np.ndarray:
        self.ports = np.array([1.0, 0.0, 0.0])
        return super().reset()

    def step(self, action: int) -> Transition:
        if not 0 <= action < self.actions:
            raise ValueError("an action is one of the buttons")
        self.moment += 1
        self.previous = action
        health = self.ports[0] - self.drain
        if action == self.paying and self.rng.random() < self.pay:
            health = min(1.0, health + self.heal)
        self.ports = np.array([max(health, 0.0), 0.0, 0.0])
        if health <= 0.0:
            return Transition(self.death, True, "death", self.observe())
        if self.moment >= self.length:
            return Transition(self.alive, True, "completion", self.observe())
        return Transition(self.alive, False, None, self.observe())


class Aim(Bandit):
    """Doom basic's shape: a target at an offset from the crosshair, drawn per episode in
    ``-reach..reach``; button 0 strafes left, button 1 right (the offset moves by one, within
    the reach), button 2 attacks, which completes the episode with ``+1`` on target and misses
    otherwise. Each moment costs ``cost``; ``length`` moments end the episode at its deadline.
    The retina shows the target as a bright band of columns whose place follows the offset. The
    paying button is the one the offset asks for: strafing toward the target, then attacking."""

    def __init__(
        self,
        *,
        actions: int = 3,
        paying: int = 2,
        width: int = RETINA,
        reach: int = 6,
        cost: float = 0.01,
        length: int = 75,
        noise: float = 0.025,
        seed: int = 0,
        history: bool = True,
    ) -> None:
        if actions != 3:
            raise ValueError("the aim chamber has three buttons: left, right, attack")
        super().__init__(actions=3, paying=2, width=width, pay=1.0, cost=cost, length=length,
                         noise=noise, seed=seed, history=history)
        self.reach = reach
        self.offset = 0
        columns = 21
        self._column = np.arange(width) // 3 % columns  # the retina's block column of a value

    def observe(self) -> np.ndarray:
        x = super().observe()
        band = np.abs(self._column - (10 + self.offset)) <= 1
        x[: self.width][band] = np.minimum(x[: self.width][band] + 0.5, 1.0)
        return x

    def reset(self) -> np.ndarray:
        self.offset = int(self.rng.integers(-self.reach, self.reach + 1))
        return super().reset()

    def step(self, action: int) -> Transition:
        if not 0 <= action < self.actions:
            raise ValueError("an action is one of the buttons")
        self.moment += 1
        self.previous = action
        if action == 0:
            self.offset = min(self.offset + 1, self.reach)
        elif action == 1:
            self.offset = max(self.offset - 1, -self.reach)
        elif self.offset == 0:
            return Transition(1.0 - self.cost, True, "completion", self.observe())
        if self.moment >= self.length:
            return Transition(-self.cost, True, "deadline", self.observe())
        return Transition(-self.cost, False, None, self.observe())


CHAMBERS = {"bandit": Bandit, "survival": Survival, "aim": Aim}


def compose(
    inputs: int, actions: int, seed: int, settings: dict[str, Any],
    rates: dict[str, float] | None = None,
) -> Any:
    """The default composition at ``settings``; ``rates`` multiplies the step of each named
    projection (``"sensory->association"``) through ``brain.learner.synapse_rate``."""
    from cadence import Brain

    brain = Brain.compose(inputs, actions, seed=seed, backend="cpu", **settings)
    if rates:
        connectome = brain.connectome
        populations = connectome.populations
        rate = brain.learner.synapse_rate
        rate = np.ones(connectome.synapses) if rate is None else np.array(rate, dtype=float)
        for name, value in rates.items():
            source, target = name.split("->")
            pre = np.zeros(connectome.n, dtype=bool)
            post = np.zeros(connectome.n, dtype=bool)
            pre[np.asarray(populations[source], dtype=np.int64)] = True
            post[np.asarray(populations[target], dtype=np.int64)] = True
            chosen = pre[connectome.pre] & post[connectome.post]
            if not chosen.any():
                raise ValueError(f"no synapse from {source} to {target}")
            rate[chosen] *= float(value)
        brain.learner.synapse_rate = rate
    return brain


def live(
    episodes: int,
    seed: int,
    *,
    player: str = "brain",
    settings: dict[str, Any] | None = None,
    world: dict[str, Any] | None = None,
    rates: dict[str, float] | None = None,
    chamber: str = "bandit",
) -> dict[str, Any]:
    """One life of ``episodes`` episodes in ``chamber`` (``"bandit"`` or ``"survival"``); the
    paying button is ``seed % actions``.

    ``player`` is ``brain`` (``step`` from every outcome), ``frozen`` (the newborn's sampled
    acts) or ``random``. Returns the life's declaration and one summary per episode."""
    settings = dict(settings or {})
    world = dict(world or {})
    actions = int(world.pop("actions", 3))
    paying = {} if chamber == "aim" else {"paying": seed % actions}
    bandit = CHAMBERS[chamber](actions=actions, seed=seed, **paying, **world)
    brain = None if player == "random" else compose(
        bandit.inputs, actions, seed, settings, rates
    )
    rng = np.random.default_rng([seed, 2960])
    out_index = None if brain is None else brain.learner.output_index
    x = bandit.reset()
    previous: Transition | None = None
    rows: list[dict[str, Any]] = []
    counts = np.zeros(actions, dtype=np.int64)
    sums: dict[str, float] = {name: 0.0 for name in FIELDS}
    learned = 0
    motor = np.zeros(actions)
    returned = 0.0
    started = time.perf_counter()
    while True:
        if brain is None:
            action = int(rng.integers(actions))
        elif player == "frozen":
            action = int(brain.act(x[None])[0])
        elif previous is None:
            action = int(brain.step(x[None])[0])
        else:
            action = int(brain.step(x[None], reward=[previous.reward], done=[previous.done])[0])
        if brain is not None:
            report = brain.last_learning or {}
            if "saturation" in report:
                learned += 1
                for name in FIELDS:
                    sums[name] += abs(float(report.get(name, 0.0))) if name != "dopamine" \
                        else float(report.get(name, 0.0))
            assert out_index is not None
            motor += np.asarray(brain.basal_ganglia.state.part(out_index))[0]
        counts[action] += 1
        transition = bandit.step(action)
        returned += transition.reward
        previous = transition
        if not transition.done:
            x = transition.observation
            continue
        moments = int(counts.sum())
        row: dict[str, Any] = {
            "episode": len(rows), "outcome": transition.outcome, "return": returned,
            "moments": moments, "actions": counts.tolist(),
            "motor": (motor / moments).tolist(),
        }
        row.update({name: (sums[name] / learned if learned else None) for name in FIELDS})
        rows.append(row)
        counts[:] = 0
        motor[:] = 0.0
        sums = {name: 0.0 for name in FIELDS}
        learned, returned = 0, 0.0
        if len(rows) == episodes:
            break
        x = bandit.reset()
    if brain is not None and player == "brain":
        brain.learn([previous.reward], [previous.done], x[None])
    return {
        "chamber": chamber, "player": player, "seed": seed, "settings": settings,
        "rates": dict(rates or {}),
        "world": {"actions": actions, "paying": bandit.paying, "width": bandit.width,
                  "pay": bandit.pay, "length": bandit.length, "noise": bandit.noise,
                  **({"cost": bandit.cost} if chamber != "survival" else {
                      "heal": bandit.heal, "drain": bandit.drain, "alive": bandit.alive,
                      "death": bandit.death})},
        "inputs": bandit.inputs,
        "parameters": 0 if brain is None else int(brain.parameters()),
        "seconds": time.perf_counter() - started,
        "machine": platform.platform(),
        "episodes": rows,
    }


def quarters(life: dict[str, Any]) -> list[dict[str, Any]]:
    """The life's four quarters of episodes, each summed: button shares, the paying button's
    share, completion, return, the motor activations and the learn report's means."""
    rows = life["episodes"]
    size = max(len(rows) // 4, 1)
    out = []
    for q in range(4):
        part = rows[q * size:(q + 1) * size] if q < 3 else rows[3 * size:]
        if not part:
            continue
        buttons = np.array([r["actions"] for r in part]).sum(axis=0)
        moments = np.array([r["moments"] for r in part], dtype=float)
        summary: dict[str, Any] = {
            "quarter": q + 1,
            "moments": int(moments.sum()),
            "shares": (buttons / buttons.sum()).tolist(),
            "paying": float(buttons[life["world"]["paying"]] / buttons.sum()),
            "completion": float(np.mean([r["outcome"] == "completion" for r in part])),
            "return": float(np.mean([r["return"] for r in part])),
            "motor": (np.array([r["motor"] for r in part]).T @ moments / moments.sum()).tolist(),
        }
        for name in FIELDS:
            values = [(r[name], r["moments"]) for r in part if r.get(name) is not None]
            summary[name] = (
                float(sum(v * m for v, m in values) / sum(m for _, m in values)) if values
                else None
            )
        out.append(summary)
    return out


def held(life: dict[str, Any], threshold: float = 0.9) -> int | None:
    """The first episode from which a non-paying button holds at least ``threshold`` of the
    presses in every later window of 20 episodes; None when no button does."""
    rows = life["episodes"]
    paying = life["world"]["paying"]
    window = 20
    shares = []
    for i in range(0, max(len(rows) - window + 1, 0)):
        buttons = np.array([r["actions"] for r in rows[i:i + window]]).sum(axis=0)
        shares.append(buttons / buttons.sum())
    for i in range(len(shares)):
        tail = np.array(shares[i:])
        for b in range(len(rows[0]["actions"])):
            if b != paying and (tail[:, b] >= threshold).all():
                return i
    return None


def read(paths: list[Path]) -> None:
    print("| Arm | Founder | Paying | Quarter | Button shares | Paying share | Completion | Return "
          "| Motor | Saturation | Synapse step | Bias step | Critic error | Trace |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: "
          "| ---: |")
    for path in paths:
        life = json.loads(path.read_text())
        named = {**life["settings"], **{f"rate {k}": v for k, v in life.get("rates", {}).items()},
                 **{f"world {k}": v for k, v in life.get("changed", {}).items()}}
        arm = life["player"] if not named else ", ".join(f"{k}={v}" for k, v in named.items())
        arm = f"{life.get('chamber', 'bandit')}: {arm}"
        for q in quarters(life):
            def num(name: str, fmt: str, q: dict[str, Any] = q) -> str:
                value = q[name]
                return "" if value is None else format(value, fmt)
            print(
                f"| {arm} | {life['seed']} | {life['world']['paying']} | {q['quarter']} "
                f"| {', '.join(f'{v:.2f}' for v in q['shares'])} | {q['paying']:.2f} "
                f"| {q['completion']:.2f} | {q['return']:.3f} "
                f"| {', '.join(f'{v:.2f}' for v in q['motor'])} | {num('saturation', '.2f')} "
                f"| {num('scale_step', '.1e')} | {num('bias_step', '.1e')} "
                f"| {num('td_error', '.3f')} | {num('trace', '.1e')} |"
            )


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chamber", choices=sorted(CHAMBERS), default="bandit")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--player", choices=("brain", "frozen", "random"), default="brain")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="a named setting of Brain.compose")
    parser.add_argument("--world", action="append", default=[], metavar="NAME=VALUE",
                        help="a setting of the bandit (width, pay, cost, length, noise, actions)")
    parser.add_argument("--rate", action="append", default=[], metavar="PRE->POST=VALUE",
                        help="multiply one projection's step (sensory->association=0.03)")
    parser.add_argument("--read", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if args.read is not None:
        read(args.read)
        return
    settings = {k: _value(v) for k, v in (s.split("=", 1) for s in args.set)}
    world = {k: _value(v) for k, v in (s.split("=", 1) for s in args.world)}
    rates = {k: float(v) for k, v in (s.rsplit("=", 1) for s in args.rate)}
    life = live(args.episodes, args.seed, player=args.player, settings=settings, world=world,
                rates=rates, chamber=args.chamber)
    life["changed"] = world
    json.dump(life, sys.stdout)


if __name__ == "__main__":
    main()
