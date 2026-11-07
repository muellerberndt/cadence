"""The embedding test: does a learned code carry an answer from taught tokens to untaught ones?

A token chamber with hidden structure. Member tokens fall into ``GROUPS`` groups. In heard
sentences ``pre + member + post`` a group's members follow and precede that group's own
context tokens; nothing else marks a group. A question ``ask + member`` asks for the
member's group answer (one of ``GROUPS`` actions, a founder-drawn permutation of the groups).
Lessons come only at questions about the taught members; each group's held-out members are
heard in sentences as often as the taught ones and are never asked or taught before a reading.
Every token is its own input coordinate, so the raw observation and a fixed random code of it
give an untaught member nothing in common with its group: they answer it at chance by
construction. A code that learns from the contexts a token is heard in can place a member
near its group, and a brain that computes on that code can carry the group's answer to it.

Arms, one life per founder and arm, on the same schedule of sentences and questions:

* ``raw``: ``Brain.compose(INPUTS, GROUPS, modules=(64,))`` on the observation;
* ``random``: the same with the library's fixed random code, ``code=UNITS``;
* ``learned``: a ``CodeRegion`` of ``UNITS`` units with both boundaries (top-down weight 4)
  settles each moment; ``Brain.compose(UNITS, GROUPS, modules=(64,))`` receives the settled
  code at unit length, and the region's context is that brain's settled association activity
  of the moment before;
* ``learned_g2``: the same at top-down weight 2;
* ``bottom_up``: the same region without its top-down boundary;
* ``shuffled``: ``learned`` in a chamber whose context tokens are drawn from a group
  independent of the member's (its own random stream), so the contexts carry no group;
* ``uniform``: uniform random answers;
* ``gru``: a supervised online GRU on the observation (``competitors.GRUCell``, 64 hidden
  units, a softmax over the answers, cross-entropy at the question through the round's
  events, one step per round);
* ``gru_lm``: the same GRU that also predicts every next token (a softmax over the tokens),
  its embedding of the inputs learned by gradient from the sentences as well;
* ``library``: the library's own code region, ``Brain.compose(INPUTS, GROUPS, modules=(64,),
  learned_code=UNITS)``, every part of it the region's constants: the region settles
  with the brain, its top-down boundary reads the working trace inside the settle, and it
  learns at every accepted moment at the per-row rates that give the prototype's effective
  step at this chamber's batch;
* ``library_bottom_up``: the same region without its top-down boundary
  (``brain.learned_code.top_down = 0``, the control).

Every brain arm teaches with ``learning_surprise`` (rows the brain already answers as shown
are not taught); every other gene is the founder's. Readings run on copies of the life with
the code's learning off; they never change the life. ``EMBEDDING_PROTOCOL`` in the evidence
declares the gates.

    PYTHONPATH=src python benchmarks/learned_code/embedding.py run --out DIR --workers 2

Each life is written to ``DIR/lives`` as it ends, so a run that stops keeps every finished
life and a second call skips them; ``DIR/receipt.json`` holds every life found there.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
COMPETITORS_PATH = HERE.parent / "competitors.py"


def _sibling(name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cr = _sibling("code_region")

# -- the chamber's vocabulary

GROUPS = 4
MEMBERS = 8  # member tokens per group
HELD = 3  # held-out members per group: heard in sentences, never asked before a reading
PRE = 2  # context tokens heard before a group's members
POST = 2  # context tokens heard after them
LATE = 1  # per group: a member heard only after the change of input statistics
MEMBER0 = 0
PRE0 = MEMBER0 + GROUPS * MEMBERS
POST0 = PRE0 + GROUPS * PRE
ASK = POST0 + GROUPS * POST
LATE0 = ASK + 1
INPUTS = LATE0 + GROUPS * LATE  # 53
ACTIONS = GROUPS
UNITS = 64
WIDTH = 64  # the association region
SENTENCES = 2  # heard sentences per round, before the question
BATCH = 16  # streams per round
NURSERY = 1024  # rounds of sentences alone before the first question
BUDGETS = (256, 1024)  # rounds with a question and its lesson, read at each
PANEL = 128  # trials per reading panel
CHANGE_ROUNDS = (0, 16, 64, 256)  # rounds of the changed statistics, read at each
FOUNDERS = tuple(range(6))
ARMS = ("raw", "random", "learned", "learned_g2", "bottom_up", "shuffled", "uniform", "gru",
        "gru_lm", "library", "library_bottom_up")
# the order lives start in: the gated arm and the gates' controls first
ORDER = ("library", "library_bottom_up", "learned", "bottom_up", "raw", "random", "gru",
         "uniform", "shuffled", "gru_lm", "learned_g2")
# the code arms and their top-down weight; ``shuffled`` also draws its contexts apart
LIBRARY_TOP_DOWN = 2.0  # the library region's top-down weight: the smallest at which the
# taught members' context-free codes reach a within-group cosine of .5 with between-group
# at most .1 on both pilot founders (900, 901) at the nursery's end; 4 passes as well
TOP_DOWN = {"learned": 4.0, "learned_g2": 2.0, "bottom_up": 0.0, "shuffled": 4.0,
            "library": LIBRARY_TOP_DOWN, "library_bottom_up": 0.0}
CODE_ARMS = tuple(TOP_DOWN)
LIBRARY_ARMS = ("library", "library_bottom_up")  # the library's region, inside the brain's settle
# the library region runs at its constants, the prototype's rates per moment as per-row rates
# at this chamber's batch: the prototype's rates are means over the BATCH streams, and its
# predictor closed its whole gap per update on a unit-length context with a bias input
# (rate 0.5 at energy 2), so the library's share per row is 1 / BATCH
SCHEDULE_SEED, HOLDOUT_SEED, PANEL_SEED, GRU_SEED, UNIFORM_SEED, CHANGE_SEED, SHUFFLE_SEED = (
    51101, 51102, 51103, 51104, 51105, 51106, 51107,
)
# the declared code: rates per moment, each update the mean over the streams
CODE = cr.CodeSettings(units=UNITS, top_down=4.0, threshold=0.1, step=0.1, budget=400,
                       tolerance=1e-4, dictionary_rate=2.0, predictor_rate=0.5,
                       threshold_rate=0.01, target=0.1)
GRU = {"hidden": 64, "rate": 1.0, "clip": 5.0}
GATE = {"held_out": 0.90, "founders": 5, "control_margin": 0.15}


def member(group: np.ndarray, index: np.ndarray) -> np.ndarray:
    return MEMBER0 + np.asarray(group) * MEMBERS + np.asarray(index)


def onehot(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.int64)
    out = np.zeros((len(tokens), INPUTS))
    out[np.arange(len(tokens)), tokens] = 1.0
    return out


def holdout(seed: int) -> np.ndarray:
    """``(GROUPS, MEMBERS)`` True where a member is held out, ``HELD`` per group from the
    founder's own draw."""
    rng = np.random.default_rng(np.random.SeedSequence([HOLDOUT_SEED, seed]))
    held = np.zeros((GROUPS, MEMBERS), bool)
    for group in range(GROUPS):
        held[group, rng.permutation(MEMBERS)[:HELD]] = True
    return held


def answers_of(seed: int) -> np.ndarray:
    """Each group's answer: a permutation of the actions drawn from the founder's seed."""
    rng = np.random.default_rng(np.random.SeedSequence([HOLDOUT_SEED, seed, 1]))
    return rng.permutation(ACTIONS).astype(np.int64)


def sentence(rng: np.random.Generator, n: int, *, late: bool = False,
             shuffle: np.random.Generator | None = None) -> list[np.ndarray]:
    """``n`` heard sentences ``pre + member + post`` of uniformly drawn groups; any member of
    the group, held out or not; with ``late`` a group's late member is one of its members.
    With ``shuffle`` the context tokens belong to a group drawn from that generator,
    independent of the member's; the members are drawn as without it."""
    groups = rng.integers(GROUPS, size=n)
    count = MEMBERS + (LATE if late else 0)
    index = rng.integers(count, size=n)
    members = np.where(
        index < MEMBERS, member(groups, np.minimum(index, MEMBERS - 1)),
        LATE0 + groups * LATE + np.maximum(index - MEMBERS, 0),
    )
    pre_index, post_index = rng.integers(PRE, size=n), rng.integers(POST, size=n)
    context = groups if shuffle is None else shuffle.integers(GROUPS, size=n)
    return [PRE0 + context * PRE + pre_index, members, POST0 + context * POST + post_index]


def question(rng: np.random.Generator, n: int, held: np.ndarray, *,
             asked: str = "taught") -> tuple[np.ndarray, np.ndarray]:
    """``n`` questions: the asked member tokens and their groups. ``asked`` "taught" draws a
    taught member of a uniform group, "held_out" a held-out one, "late" the group's late
    member."""
    groups = rng.integers(GROUPS, size=n)
    tokens = np.zeros(n, np.int64)
    for i, group in enumerate(groups):
        if asked == "late":
            tokens[i] = LATE0 + group * LATE + rng.integers(LATE)
            continue
        pool = np.flatnonzero(held[group] if asked == "held_out" else ~held[group])
        tokens[i] = member(group, rng.choice(pool))
    return tokens, groups


# -- a life: the composed brain, with or without a code region before it


class Life:
    """The brain of an arm and, for the code arms, its code region. ``hear`` lets a moment
    pass with a greedy act, ``teach`` gives the moment's lesson and then acts; the code
    settles once per moment and learns from it. Without a code region the brain receives
    the observation as given: the released ``Brain.compose`` driven directly. The library
    arms hold the region inside the brain (``learned_code``): the brain receives the
    observation, settles the region with itself and learns it at every accepted moment."""

    def __init__(self, arm: str, seed: int, *, code: Any = CODE) -> None:
        from cadence import Brain

        self.arm, self.seed = arm, seed
        self.code = None
        self.library = arm in LIBRARY_ARMS
        if self.library:
            self.brain = Brain.compose(
                INPUTS, ACTIONS, backend="cpu", modules=(WIDTH,), seed=seed, learning_surprise=True,
                learned_code=UNITS, recall=False,
            )
            assert self.brain.learned_code is not None
            self.brain.learned_code.top_down = TOP_DOWN[arm]  # 0: the bottom-up control
        elif arm in CODE_ARMS:
            settings = cr.CodeSettings(**{**code.to_dict(), "top_down": TOP_DOWN[arm]})
            self.brain = Brain.compose(settings.units, ACTIONS, backend="cpu", modules=(WIDTH,), seed=seed,
                                       learning_surprise=True, recall=False)
            self.code = cr.CodeRegion(INPUTS, WIDTH, settings, seed=seed)
        elif arm == "random":
            self.brain = Brain.compose(INPUTS, ACTIONS, backend="cpu", modules=(WIDTH,), seed=seed,
                                       code=code.units, learning_surprise=True, recall=False)
        elif arm == "raw":
            self.brain = Brain.compose(INPUTS, ACTIONS, backend="cpu", modules=(WIDTH,), seed=seed,
                                       learning_surprise=True, recall=False)
        else:
            raise ValueError(arm)
        self.context: np.ndarray | None = None
        self.top_down: float | None = None  # a reading's override of the code's boundary
        self.last: dict[str, np.ndarray] = {}
        self.work = {"acts": 0, "act_rows": 0, "act_sweeps": 0, "act_refusals": 0,
                     "lessons": 0, "lesson_rows": 0, "rows_taught": 0, "lesson_sweeps": 0,
                     "lesson_refusals": 0, "code_sweeps": 0}

    @property
    def coded(self) -> bool:
        """Whether a code stands between the observation and the cortex (either kind)."""
        return self.code is not None or self.library

    def freeze(self) -> None:
        """The code's learning off: a reading's copy keeps the life's tables."""
        if self.code is not None:
            self.code.learning = False
        elif self.library:
            self.brain.learned_code.learning = False

    def reset(self) -> None:
        self.brain.reset()
        self.context = None

    def _coded(self, x: np.ndarray) -> np.ndarray:
        """The moment's input to the brain: the observation, or its settled code at unit
        length after the code region settled (and learned) on it; a library brain takes the
        observation and settles its own region."""
        if self.code is None:
            return x
        p = self.code.predict(self.context, len(x))
        a, _ = self.code.settle(x, p, top_down=self.top_down)
        self.last = {"x": x, "a": a, "p": p}
        self.code.learn(x, a, p, self.context)
        return cr.unit_rows(a)

    def _act(self, y: np.ndarray) -> np.ndarray | None:
        self.work["acts"] += 1
        self.work["act_rows"] += len(y)
        if self.library and self.top_down is not None:
            # a reading's override of the boundary at this moment alone, on a copy
            self.brain.learned_code.top_down = self.top_down
        try:
            action = self.brain.act(y, greedy=True)
        except RuntimeError:
            self.work["act_refusals"] += 1
            self.context = None
            return None
        self.work["act_sweeps"] += int(self.brain.last_settlement["steps"])
        if self.library:
            self.work["code_sweeps"] += int(self.brain.last_settlement["code_steps"])
            last = self.brain.learned_code.last
            self.last = {"x": last["sensory"], "a": np.maximum(last["activation"], 0.0),
                         "p": np.zeros_like(last["activation"]) if last["prediction"] is None
                         else last["prediction"], "context": last["context"]}
        state = self.brain.basal_ganglia.state
        self.context = np.asarray(state.activation[:, self.brain.association_index]).copy()
        return np.asarray(action)

    def residuals(self) -> dict[str, np.ndarray]:
        """The last moment's residuals on both boundaries, per row, for either kind."""
        if self.library:
            return self.brain.learned_code.residuals(
                self.last["x"], self.last["a"], self.last["context"]
            )
        assert self.code is not None
        return self.code.residuals(self.last["x"], self.last["a"], self.last["p"])

    @property
    def region_work(self) -> dict[str, int]:
        if self.library:
            return dict(self.brain.learned_code.work)
        assert self.code is not None
        return dict(self.code.work)

    @property
    def thresholds(self) -> np.ndarray:
        if self.library:
            return np.asarray(self.brain.learned_code.thresholds)
        assert self.code is not None
        return np.asarray(self.code.thresholds)

    def region_bytes(self) -> int:
        if self.library:
            return int(self.brain.learned_code.nbytes())
        assert self.code is not None
        return int(self.code.nbytes())

    def hear(self, x: np.ndarray) -> np.ndarray | None:
        return self._act(self._coded(x))

    def teach(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
        from cadence.learning import LearningPhaseError

        y = self._coded(x)
        self.work["lessons"] += 1
        self.work["lesson_rows"] += len(y)
        try:
            report = self.brain.lesson(y, labels)
        except LearningPhaseError:
            self.work["lesson_refusals"] += 1
        else:
            self.work["rows_taught"] += int(report.get("accepted_presentations", 0))
            self.work["lesson_sweeps"] += int(report.get("total_steps", 0))
        return self._act(y)

    def code_of(self, tokens: np.ndarray) -> np.ndarray:
        """The context-free code of tokens as the brain would receive it: the observation, the
        random code, or the region's settle on the observation alone (no prediction); for a
        library brain the region's rectified activation settled without a context."""
        x = onehot(tokens)
        if self.library:
            phase = self.brain._settle_code(x, context=False)
            return np.maximum(np.asarray(phase.state.activation), 0.0)
        if self.code is None:
            return self.brain.observation_code(x)
        a, _ = self.code.settle(x, np.zeros((len(x), self.code.settings.units)), top_down=0.0)
        return a

    def residual_of(self, tokens: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """The bottom-up residual of context-free ``codes`` of ``tokens``, per row."""
        x = onehot(tokens)
        if self.library:
            model = self.brain.brain.neuron_model
            sensory = model.activation(
                x * model.stimulus_amplitude + self.brain.brain.bias[self.brain.sensory_index]
            )
            return self.brain.learned_code.residuals(sensory, codes, None)["bottom_up"]
        assert self.code is not None
        return self.code.residuals(x, codes, np.zeros_like(codes))["bottom_up"]


def fork(life: Any) -> Any:
    """An independent copy of a life: every array, store and random state copied, the
    immutable connectome shared; read-only mapping proxies restored as proxies."""
    brain = getattr(life, "brain", None)
    memo = {} if brain is None else {id(brain.connectome): brain.connectome}
    region = None if brain is None else getattr(brain, "learned_code", None)
    if region is not None:  # the region's own immutable connectome is shared as well
        memo[id(region.connectome)] = region.connectome
    proxies = {} if brain is None else {
        k: v for k, v in vars(brain).items() if isinstance(v, MappingProxyType)}
    for key, value in proxies.items():
        setattr(brain, key, dict(value))
    try:
        twin = copy.deepcopy(life, memo)
    finally:
        for key, value in proxies.items():
            setattr(brain, key, value)
    if brain is not None:
        for key in proxies:
            setattr(twin.brain, key, MappingProxyType(getattr(twin.brain, key)))
    return twin


# -- the GRU controls


def _competitors() -> Any:
    if "competitors" in sys.modules:
        return sys.modules["competitors"]
    spec = importlib.util.spec_from_file_location("competitors", COMPETITORS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["competitors"] = module
    spec.loader.exec_module(module)
    return module


class Recurrent:
    """A GRU read by a softmax over the answers at the question; with ``predict`` a second
    softmax over the tokens reads every moment and learns the next token. One online step per
    round on the mean cross-entropy, through every event of the round, bounded in norm."""

    def __init__(self, seed: int, *, predict: bool = False, hidden: int = GRU["hidden"],
                 rate: float = GRU["rate"], clip: float = GRU["clip"]) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([GRU_SEED, seed]))
        self.hidden, self.rate, self.clip, self.predicts = hidden, rate, clip, predict
        self.cell = _competitors().GRUCell(INPUTS, hidden, rng)
        self.readout = {"W": np.zeros((ACTIONS, hidden)), "b": np.zeros(ACTIONS)}
        if predict:
            self.readout.update({"V": np.zeros((INPUTS, hidden)), "c": np.zeros(INPUTS)})
        self.work = {"updates": 0, "rows": 0, "clipped": 0, "moments": 0,
                     "forward_moments": 0}

    def parameters(self) -> dict[str, np.ndarray]:
        return {**self.readout, **self.cell.parameters()}

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return np.asarray(p / p.sum(axis=-1, keepdims=True))

    def run(self, seq: list[np.ndarray]) -> tuple[list[np.ndarray], list[list[dict[str, Any]]]]:
        n = len(seq[0])
        self.work["forward_moments"] += n * len(seq)
        h = np.zeros((n, self.hidden))
        states, caches = [], [[] for _ in range(n)]
        for x in seq:
            for i in range(n):
                h[i], cache = self.cell.forward(x[i], h[i].copy())
                caches[i].append(cache)
            states.append(h.copy())
        return states, caches

    def answer(self, seq: list[np.ndarray]) -> np.ndarray:
        states, _ = self.run(seq)
        p = self._softmax(states[-1] @ self.readout["W"].T + self.readout["b"])
        return np.argmax(p, axis=-1).astype(np.int64)

    def loss(self, seq: list[np.ndarray], target: np.ndarray | None) -> float:
        """The round's objective: the mean cross-entropy of the answer at the last event and,
        predicting, the mean cross-entropy of every next token."""
        states, _ = self.run(seq)
        n, steps, total = len(seq[0]), len(seq), 0.0
        if target is not None:
            p = self._softmax(states[-1] @ self.readout["W"].T + self.readout["b"])
            total -= float(np.mean(np.log(p[np.arange(n), target])))
        if self.predicts:
            for t in range(steps - 1):
                q = self._softmax(states[t] @ self.readout["V"].T + self.readout["c"])
                total -= float(np.sum(np.log(np.sum(q * seq[t + 1], axis=1)))) / (
                    n * (steps - 1))
        return total

    def round(self, seq: list[np.ndarray], target: np.ndarray | None) -> None:
        """One step on a round's events: the answer at the last event (when ``target`` is
        given) and, predicting, every next token; the gradient bounded in norm."""
        if target is None and not self.predicts:
            return
        grads = self.gradients(seq, target)
        n = len(seq[0])
        norm = float(np.sqrt(sum(float(np.sum(v * v)) for v in grads.values())))
        scale = min(1.0, self.clip / norm) if norm > 0 else 1.0
        self.work["clipped"] += int(scale < 1.0)
        for key, value in self.parameters().items():
            value -= self.rate * scale * grads[key]
        self.work["updates"] += 1
        self.work["rows"] += n
        self.work["moments"] += n * len(seq)

    def gradients(self, seq: list[np.ndarray], target: np.ndarray | None,
                  ) -> dict[str, np.ndarray]:
        """The gradient of ``loss`` by backpropagation through the round's events."""
        states, caches = self.run(seq)
        n, steps = len(seq[0]), len(seq)
        grads = {k: np.zeros_like(v) for k, v in self.parameters().items()}
        inject = [np.zeros((n, self.hidden)) for _ in range(steps)]
        if target is not None:
            p = self._softmax(states[-1] @ self.readout["W"].T + self.readout["b"])
            d = p.copy()
            d[np.arange(n), target] -= 1.0
            d /= n
            grads["W"] += d.T @ states[-1]
            grads["b"] += d.sum(axis=0)
            inject[-1] += d @ self.readout["W"]
        if self.predicts:
            for t in range(steps - 1):
                q = self._softmax(states[t] @ self.readout["V"].T + self.readout["c"])
                d = q - seq[t + 1]
                d /= n * (steps - 1)
                grads["V"] += d.T @ states[t]
                grads["c"] += d.sum(axis=0)
                inject[t] += d @ self.readout["V"]
        w = self.cell.weights
        for i in range(n):
            g = np.zeros(self.hidden)
            for t in reversed(range(steps)):
                g = g + inject[t][i]
                cache = caches[i][t]
                part = self.cell.backward(g, cache)
                for key, value in part.items():
                    grads[key] += value
                g = (g * cache["z"] + w["Uz"].T @ part["bz"] + w["Ur"].T @ part["br"]
                     + (w["Un"].T @ part["bn"]) * cache["r"])
        return grads


# -- rounds


def draw_round(rng: np.random.Generator, held: np.ndarray, answers: np.ndarray, *,
               asked: bool, shuffle: np.random.Generator | None = None,
               ) -> tuple[list[np.ndarray], np.ndarray | None]:
    """One round's events for ``BATCH`` streams: ``SENTENCES`` heard sentences, then (with
    ``asked``) the ask marker and a taught member; the answer of the asked member's group.
    ``shuffle`` draws the sentences' context groups apart (the ``shuffled`` arm)."""
    seq: list[np.ndarray] = []
    for _ in range(SENTENCES):
        seq += [onehot(t) for t in sentence(rng, BATCH, shuffle=shuffle)]
    if not asked:
        return seq, None
    tokens, groups = question(rng, BATCH, held)
    seq += [onehot(np.full(BATCH, ASK)), onehot(tokens)]
    return seq, answers[groups]


def live_round(life: Life, seq: list[np.ndarray], target: np.ndarray | None) -> bool:
    life.reset()
    last = len(seq) - 1
    for t, x in enumerate(seq):
        if t == last and target is not None:
            out = life.teach(x, target)
        else:
            out = life.hear(x)
        if out is None:
            return False
    return True


# -- readings


def panels(seed: int, held: np.ndarray, *, shuffled: bool = False,
           ) -> dict[str, dict[str, Any]]:
    """The reading panels of a founder: ``PANEL`` trials each of one heard sentence, the ask
    marker and the asked member; ``train`` asks taught members, ``held_out`` held-out ones.
    Returns the events, the asked groups and tokens. ``shuffled`` draws the sentence's
    contexts apart, from a stream of the panel's own."""
    rng = np.random.default_rng(np.random.SeedSequence([PANEL_SEED, seed]))
    shuffle = (np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, seed, 1]))
               if shuffled else None)
    out = {}
    for name in ("train", "held_out"):
        seq = [onehot(t) for t in sentence(rng, PANEL, shuffle=shuffle)]
        tokens, groups = question(rng, PANEL, held, asked=name if name == "held_out"
                                  else "taught")
        seq += [onehot(np.full(PANEL, ASK)), onehot(tokens)]
        out[name] = {"events": seq, "groups": groups, "tokens": tokens}
    return out


def read_brain(life: Life, seq: list[np.ndarray], *, top_down: float | None = None,
               ) -> tuple[np.ndarray, dict[str, Any]]:
    """A copy's answers on a panel, its code frozen; ``top_down`` overrides the code's
    top-down boundary at the question alone. Returns the answers and, for a code arm, the
    question's codes and residuals."""
    copy_ = fork(life)
    copy_.freeze()
    copy_.reset()
    for x in seq[:-1]:
        copy_.hear(x)
    copy_.top_down = top_down
    out = copy_.hear(seq[-1])
    extra: dict[str, Any] = {}
    if copy_.coded and copy_.last:
        extra = {"codes": copy_.last["a"], "residual": copy_.residuals()}
    work = copy_.work
    if copy_.coded:
        work = {**work, "code": copy_.region_work}
    extra["work"] = work
    return (np.full(len(seq[0]), -1) if out is None else out), extra


def cosines(codes: np.ndarray) -> np.ndarray:
    unit = cr.unit_rows(np.asarray(codes, dtype=float))
    return np.asarray(unit @ unit.T)


def structure(codes: np.ndarray, held: np.ndarray) -> dict[str, float]:
    """The similarity structure of the members' context-free codes against the hidden groups:
    mean cosine within a group (taught pairs, held-out to taught) and between groups, and the
    share of held-out members whose code lies nearest their own group's taught centroid."""
    tokens = np.arange(GROUPS * MEMBERS)
    group = tokens // MEMBERS
    is_held = held.reshape(-1)
    sim = cosines(codes)
    same = group[:, None] == group[None, :]
    off = ~np.eye(len(tokens), dtype=bool)
    taught = ~is_held
    pick = {
        "within_taught": same & off & taught[:, None] & taught[None, :],
        "within_held_taught": same & is_held[:, None] & taught[None, :],
        "between": ~same,
        "between_held_taught": ~same & is_held[:, None] & taught[None, :],
    }
    out = {name: float(sim[mask].mean()) for name, mask in pick.items()}
    unit = cr.unit_rows(np.asarray(codes, dtype=float))
    centroids = np.stack([unit[(group == g) & taught].mean(axis=0) for g in range(GROUPS)])
    centroids = cr.unit_rows(centroids)
    nearest = np.argmax(unit[is_held] @ centroids.T, axis=1)
    out["nearest_group_held"] = float(np.mean(nearest == group[is_held]))
    return out


def code_report(life: Life, held: np.ndarray, seed: int, *,
                before: np.ndarray | None = None) -> dict[str, Any]:
    """Reading 6 on a copy, the code frozen: residuals at rest over heard sentences (of the
    life's own chamber), units in use and sparsity, the members' code structure
    (context-free codes) and, given ``before``, their drift since: the mean cosine of each
    member's code to its code then."""
    copy_ = fork(life)
    report: dict[str, Any] = {}
    members = np.arange(GROUPS * MEMBERS)
    if not copy_.coded:
        report["structure"] = structure(copy_.code_of(members), held)
        return report
    copy_.freeze()
    rng = np.random.default_rng(np.random.SeedSequence([PANEL_SEED, seed, 2]))
    shuffle = (np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, seed, 2]))
               if life.arm == "shuffled" else None)
    copy_.reset()
    seq = []
    for _ in range(SENTENCES):
        seq += [onehot(t) for t in sentence(rng, PANEL, shuffle=shuffle)]
    bottom, top, codes, sweeps = [], [], [], []
    for x in seq:
        start = copy_.region_work["sweeps"]
        copy_.hear(x)
        sweeps.append(copy_.region_work["sweeps"] - start)
        r = copy_.residuals()
        bottom.append(r["bottom_up"])
        top.append(r["top_down"])
        codes.append(copy_.last["a"])
    a = np.concatenate(codes)
    active = a > 0
    report["rest"] = {
        "bottom_up_residual": float(np.mean(np.concatenate(bottom))),
        "top_down_residual": float(np.mean(np.concatenate(top))),
        "sweeps_per_settle": float(np.mean(sweeps)),
    }
    report["units_in_use"] = int(np.sum(active.any(axis=0)))
    report["units_over_one_percent"] = int(np.sum(active.mean(axis=0) >= 0.01))
    report["active_per_code"] = float(active.sum(axis=1).mean())
    report["silent_codes"] = int(np.sum(~active.any(axis=1)))
    report["thresholds"] = {"mean": float(copy_.thresholds.mean()),
                            "max": float(copy_.thresholds.max()),
                            "zero": int(np.sum(copy_.thresholds == 0.0))}
    free = copy_.code_of(members)
    report["structure"] = structure(free, held)
    report["free_active_per_code"] = float((free > 0).sum(axis=1).mean())
    report["free_silent"] = int(np.sum(~(free > 0).any(axis=1)))
    report["free_codes"] = free
    report["remapping"] = remapping(copy_, free)
    if before is not None:
        report["drift_cosine"] = float(np.mean(np.sum(cr.unit_rows(free)
                                                      * cr.unit_rows(before), axis=1)))
    return report


def remapping(life: Life, free: np.ndarray) -> dict[str, float]:
    """The soft form of a remapping switch: every member's code in context after a context
    token of its own group (A), after the group's other context token (a small change) and
    after a context token of another group (a large change), as mean cosines to the code
    under A; the context-free code's cosine to A beside them. On a copy with the code's
    learning off, one stream per member."""
    members = np.arange(GROUPS * MEMBERS)
    groups = members // MEMBERS

    def after(pre_tokens: np.ndarray) -> np.ndarray:
        twin = fork(life)
        twin.freeze()
        twin.reset()
        twin.hear(onehot(pre_tokens))
        out = twin.hear(onehot(members))
        if out is None or not twin.last:
            return np.full((len(members), UNITS), np.nan)
        return np.asarray(twin.last["a"], dtype=float)

    own = after(PRE0 + groups * PRE)
    small = after(PRE0 + groups * PRE + (PRE - 1))
    large = after(PRE0 + ((groups + 1) % GROUPS) * PRE)

    def cosine(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.nanmean(np.sum(cr.unit_rows(u) * cr.unit_rows(v), axis=1)))

    return {
        "small_change": cosine(own, small),
        "large_change": cosine(own, large),
        "context_free": cosine(own, free),
        "active_in_context": float(np.nanmean((own > 0).sum(axis=1))),
    }


def change_report(life: Life, held: np.ndarray, seed: int) -> dict[str, Any]:
    """Reading 6's change of input statistics, on a copy whose code keeps learning and whose
    brain is not taught: each group's late member, a coordinate never heard before, joins the
    group's sentences. Read at each of ``CHANGE_ROUNDS``: the bottom-up residual of the late
    members' and the old members' context-free codes, the drift of the old members' codes
    from before the change (mean cosine), and the late members' nearest taught group."""
    copy_ = fork(life)
    assert copy_.coded
    rng = np.random.default_rng(np.random.SeedSequence([CHANGE_SEED, seed]))
    shuffle = (np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, seed, 3]))
               if life.arm == "shuffled" else None)
    members = np.arange(GROUPS * MEMBERS)
    late = LATE0 + np.arange(GROUPS * LATE)
    before = copy_.code_of(members)
    unit = cr.unit_rows(before)
    taught = ~held.reshape(-1)
    centroids = cr.unit_rows(np.stack([unit[(members // MEMBERS == g) & taught].mean(axis=0)
                                       for g in range(GROUPS)]))
    readings, done = [], 0
    for target in CHANGE_ROUNDS:
        while done < target:
            copy_.reset()
            for _ in range(SENTENCES):
                for x in (onehot(t) for t in sentence(rng, BATCH, late=True,
                                                      shuffle=shuffle)):
                    copy_.hear(x)
            done += 1
        old = copy_.code_of(members)
        new = copy_.code_of(late)
        r_old = copy_.residual_of(members, old)
        r_new = copy_.residual_of(late, new)
        drift = np.sum(cr.unit_rows(old) * unit, axis=1)
        nearest = np.argmax(cr.unit_rows(new) @ centroids.T, axis=1)
        readings.append({
            "rounds": target,
            "late_residual": float(r_new.mean()),
            "old_residual": float(r_old.mean()),
            "old_code_cosine_to_before": float(drift.mean()),
            "late_nearest_group": float(np.mean(nearest == np.arange(GROUPS * LATE) // LATE)),
            "late_active_per_code": float((new > 0).sum(axis=1).mean()),
        })
    return {"readings": readings, "code_work": copy_.region_work}


def accuracy(given: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.asarray(given) == np.asarray(truth)))


def read_life(life: Any, arm: str, seed: int, held: np.ndarray,
              answers: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rng = np.random.default_rng(np.random.SeedSequence([UNIFORM_SEED, seed]))
    for name, panel in panels(seed, held, shuffled=arm == "shuffled").items():
        truth = answers[panel["groups"]]
        row: dict[str, Any] = {}
        if arm == "uniform":
            row["intact"] = accuracy(rng.integers(ACTIONS, size=PANEL), truth)
        elif arm in ("gru", "gru_lm"):
            row["intact"] = accuracy(life.answer(panel["events"]), truth)
        else:
            given, extra = read_brain(life, panel["events"])
            row["intact"] = accuracy(given, truth)
            row["refused"] = int(np.sum(given < 0))
            row["work"] = extra["work"]
            if arm in CODE_ARMS:
                row["residual_bottom_up"] = float(extra["residual"]["bottom_up"].mean())
                row["residual_top_down"] = float(extra["residual"]["top_down"].mean())
                row["active_per_code"] = float((extra["codes"] > 0).sum(axis=1).mean())
            if arm in ("learned", "learned_g2", "shuffled", "library"):
                given, extra = read_brain(life, panel["events"], top_down=0.0)
                row["top_down_off_at_question"] = accuracy(given, truth)
                row["work_top_down_off"] = extra["work"]
        out[name] = row
    return out


# -- one life


def life_run(spec: dict[str, Any]) -> dict[str, Any]:
    """One founder and arm: the nursery, then lesson rounds read at every budget."""
    seed, arm = int(spec["seed"]), spec["arm"]
    code = cr.CodeSettings(**{**CODE.to_dict(), **spec.get("code", {})})
    held, answers = holdout(seed), answers_of(seed)
    rng = np.random.default_rng(np.random.SeedSequence([SCHEDULE_SEED, seed]))
    shuffle = (np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, seed]))
               if arm == "shuffled" else None)
    wall, cpu = time.monotonic(), time.process_time()
    if arm in ("gru", "gru_lm"):
        life: Any = Recurrent(seed, predict=arm == "gru_lm")
    elif arm == "uniform":
        life = None
    else:
        life = Life(arm, seed, code=code)
    failed = 0
    nursery = int(spec.get("nursery", NURSERY))
    for _ in range(nursery):
        seq, target = draw_round(rng, held, answers, asked=False, shuffle=shuffle)
        if isinstance(life, Recurrent):
            life.round(seq, None)
        elif life is not None:
            failed += int(not live_round(life, seq, None))
    readings, done = [], 0
    before = None
    if isinstance(life, Life) and life.coded:
        before = life.code_of(np.arange(GROUPS * MEMBERS))
        nursery_code = code_report(life, held, seed)
        nursery_code.pop("free_codes")
        readings.append({"rounds": 0, "panels": None, "code": nursery_code,
                         "cpu": time.process_time() - cpu, "wall": time.monotonic() - wall})
    for budget in sorted(spec.get("budgets", BUDGETS)):
        while done < budget:
            seq, target = draw_round(rng, held, answers, asked=True, shuffle=shuffle)
            if isinstance(life, Recurrent):
                life.round(seq, target)
            elif life is not None:
                failed += int(not live_round(life, seq, target))
            done += 1
        started = time.process_time()
        reading: dict[str, Any] = {"rounds": budget,
                                   "panels": read_life(life, arm, seed, held, answers)}
        if isinstance(life, Life):
            reading["code"] = code_report(life, held, seed, before=before)
            before = reading["code"].pop("free_codes", None)
        reading["cpu"] = time.process_time() - cpu
        reading["read_cpu"] = time.process_time() - started
        reading["wall"] = time.monotonic() - wall
        readings.append(reading)
    out: dict[str, Any] = {**spec, "held": held.tolist(), "answers": answers.tolist(),
                           "readings": readings, "failed_rounds": failed}
    if isinstance(life, Life):
        out["work"] = dict(life.work)
        if life.coded:
            out["code_work"] = life.region_work
            out["code_bytes"] = life.region_bytes()
            if spec.get("change", True):
                out["change"] = change_report(life, held, seed)
    elif isinstance(life, Recurrent):
        out["work"] = dict(life.work)
    out["cpu"] = time.process_time() - cpu
    out["wall"] = time.monotonic() - wall
    return out


def gate(row: dict[str, Any]) -> bool:
    return bool(row["readings"][-1]["panels"]["held_out"]["intact"] >= GATE["held_out"])


def lesson_readings(row: dict[str, Any]) -> list[dict[str, Any]]:
    """A life's readings after lessons began (the code arms also read the nursery's end)."""
    return [r for r in row["readings"] if r["panels"] is not None]


def digests() -> dict[str, str]:
    files = [Path(__file__), HERE / "code_region.py", COMPETITORS_PATH]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| arm | founder | rounds | train | held-out | held-out, top-down off at the "
             "question | gate |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        for reading in lesson_readings(row):
            p = reading["panels"]
            off = p["held_out"].get("top_down_off_at_question")
            lines.append(
                f"| {row['arm']} | {row['seed']} | {reading['rounds']} | "
                f"{p['train']['intact']:.3f} | {p['held_out']['intact']:.3f} | "
                f"{'' if off is None else f'{off:.3f}'} | "
                f"{'pass' if reading['panels']['held_out']['intact'] >= GATE['held_out'] else ''}"
                " |")
    return "\n".join(lines)


def life_name(spec: dict[str, Any]) -> str:
    return f"{spec['arm']}-s{spec['seed']}.json"


def kept_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life, written to ``spec["lives"]`` when it ends (through a temporary file)."""
    row = life_run({k: v for k, v in spec.items() if k != "lives"})
    row["digests"] = digests()
    target = Path(spec["lives"]) / life_name(spec)
    partial = target.with_suffix(".partial")
    partial.write_text(json.dumps(row, indent=1, default=float) + "\n")
    partial.replace(target)
    return {"arm": spec["arm"], "seed": spec["seed"], "cpu": row["cpu"]}


def receipt(out: Path, args: argparse.Namespace, elapsed: float) -> list[dict[str, Any]]:
    """Every life found in ``out/lives``, in the arms' and founders' order, as one receipt."""
    rows = []
    for arm in ARMS:
        for seed in args.seeds:
            path = out / "lives" / life_name({"arm": arm, "seed": seed})
            if path.exists():
                rows.append(json.loads(path.read_text()))
    body = {"schema": "learned-code-embedding/3", "digests": digests(), "rows": rows,
            "elapsed_seconds": elapsed,
            "constants": {"GROUPS": GROUPS, "MEMBERS": MEMBERS, "HELD": HELD, "PRE": PRE,
                          "POST": POST, "LATE": LATE, "INPUTS": INPUTS, "UNITS": UNITS,
                          "WIDTH": WIDTH, "SENTENCES": SENTENCES, "BATCH": BATCH,
                          "NURSERY": args.nursery, "PANEL": PANEL,
                          "CHANGE_ROUNDS": CHANGE_ROUNDS, "CODE": CODE.to_dict(), "GRU": GRU,
                          "GATE": GATE, "TOP_DOWN": TOP_DOWN}}
    (out / "receipt.json").write_text(json.dumps(body, indent=1, default=float) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("run", "receipt"))
    parser.add_argument("--out", type=Path, required=True, help="a directory")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    parser.add_argument("--nursery", type=int, default=NURSERY)
    args = parser.parse_args(argv)
    lives = args.out / "lives"
    began = time.monotonic()
    if args.command == "run":
        lives.mkdir(parents=True, exist_ok=True)
        specs = [{"arm": arm, "seed": seed, "budgets": sorted(args.budgets),
                  "nursery": args.nursery, "lives": str(lives)}
                 for arm in sorted(args.arms, key=ORDER.index) for seed in args.seeds]
        specs = [spec for spec in specs if not (lives / life_name(spec)).exists()]
        with get_context("spawn").Pool(max(1, min(args.workers, len(specs) or 1))) as pool:
            for done in pool.imap_unordered(kept_life, specs, chunksize=1):
                print(f"{done['arm']} {done['seed']} {done['cpu']:.0f} s", flush=True)
    rows = receipt(args.out, args, time.monotonic() - began)
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
