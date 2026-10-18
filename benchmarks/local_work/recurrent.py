"""Conventional recurrent baselines on the local-work skills (protocol amendment 3).

The same workload as the brain's: S block-independent cue-to-action skills, the same lesson
stream (16 streams of uniform cues per lesson, from the same seeds), the same probe rows and
gate, and the same life (each Cadence case's own sequence of moments, disturbances and
lessons, replayed moment by moment). Three learners:

* ``SkillGRU``: one gated recurrent unit over all S blocks with one softmax readout per skill
  slot. The cell is ``benchmarks/competitors.py``'s ``GRUCell`` (its equations and its
  initialization); the rule is the stream chamber's online rule (``OnlineGRU``): at every
  lesson the cross-entropy of the present frame is taken back through the last
  ``truncation`` frames (their cached forward passes, the current weights), its norm clipped
  at ``clip``, and one Adam step is made. Two extensions, declared: a batch of streams (one
  hidden state each; the loss is the mean over the streams and the slots), and one softmax
  per slot.
* the same GRU at the brain's association width (``sum(widths)`` units): a size-matched row.
* ``Bank``: one small ``SkillGRU`` per skill, each reading only its own block and answering
  only its own slot: the skill's block is supplied as routing (a privileged control). In the
  life an oracle router steps a module only when its block's input changed or its own last
  step still moved its state by more than ``tolerance``; a skipped module holds its answer.
  The routing decisions are free.

Work per decision is counted as multiply-accumulates: input contacts ``3 H D``, recurrent
contacts ``3 H H`` and readout contacts ``A H`` per stream and step (a GRU recomputes every
unit at every step), beside the host seconds of the step. Learning work (forward, truncated
backward and Adam) is counted separately. NumPy only, outside the library.

    python benchmarks/local_work/recurrent.py search --out /path/out
    python benchmarks/local_work/recurrent.py report --search /path/out/search.json \\
        --cases /path/cadence/cases --founders 3 4 5 --out /path/out
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


competitors = _load("competitors", HERE.parent / "competitors.py")
lw = sys.modules["local_work"] if "local_work" in sys.modules else _load(
    "local_work", HERE / "local_work.py"
)

HIDDEN_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
SKILLS = (1, 2, 4, 8)


def labels2d(cues: np.ndarray, maps: Sequence[np.ndarray]) -> np.ndarray:
    """The teacher's labels, one column per skill slot, ``(batch, S)``."""
    cues = np.asarray(cues, dtype=np.int64)
    return np.stack([maps[k][cues[:, k]] for k in range(cues.shape[1])], axis=1)


class SkillGRU:
    """A gated recurrent unit with one softmax readout per skill slot, learning online."""

    def __init__(
        self,
        inputs: int,
        slots: int,
        actions: int,
        hidden: int,
        seed: Sequence[int] | int,
        *,
        truncation: int = 16,
        rate: float = 0.003,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        clip: float = 5.0,
        readout_init: float = 0.1,
    ) -> None:
        if min(inputs, slots, actions, hidden, truncation) < 1 or rate <= 0 or clip <= 0:
            raise ValueError("sizes, truncation, rate and clip must be positive")
        rng = np.random.default_rng(seed)
        self.inputs, self.slots, self.actions, self.hidden = inputs, slots, actions, hidden
        self.truncation, self.rate, self.betas, self.eps, self.clip = (
            truncation, rate, betas, eps, clip,
        )
        self.cell = competitors.GRUCell(inputs, hidden, rng)
        self.W = rng.normal(0.0, readout_init, (slots * actions, hidden))
        self.b = np.zeros(slots * actions)
        self.h = np.zeros((1, hidden))
        self.history: deque[dict[str, np.ndarray]] = deque(maxlen=truncation)
        self.moments = {
            name: (np.zeros_like(value), np.zeros_like(value))
            for name, value in self.params().items()
        }
        self.updates = 0
        self.learning_macs = 0

    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b, **self.cell.parameters()}

    @property
    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.params().values()))

    def macs(self) -> dict[str, int]:
        """Multiply-accumulates of one step of one stream, by contact kind."""
        h, d = self.hidden, self.inputs
        return {"input": 3 * h * d, "recurrent": 3 * h * h,
                "readout": self.slots * self.actions * h}

    @property
    def step_macs(self) -> int:
        return int(sum(self.macs().values()))

    def reset(self, batch: int = 1) -> None:
        self.h = np.zeros((batch, self.hidden))
        self.history.clear()

    def _step(self, x: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        w = self.cell.weights
        z = 1.0 / (1.0 + np.exp(-(x @ w["Wz"].T + h_prev @ w["Uz"].T + w["bz"])))
        r = 1.0 / (1.0 + np.exp(-(x @ w["Wr"].T + h_prev @ w["Ur"].T + w["br"])))
        reset = r * h_prev
        n = np.tanh(x @ w["Wn"].T + reset @ w["Un"].T + w["bn"])
        h = (1.0 - z) * n + z * h_prev
        return h, {"x": x, "h_prev": h_prev, "z": z, "r": r, "reset": reset, "n": n}

    def _probabilities(self, h: np.ndarray) -> np.ndarray:
        logits = (h @ self.W.T + self.b).reshape(len(h), self.slots, self.actions)
        logits = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return np.asarray(p / p.sum(axis=-1, keepdims=True))

    def answer(self, x: np.ndarray) -> np.ndarray:
        """One step of the streams on ``x``, carrying the state; the greedy action per slot."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if self.h.shape[0] != len(x):
            self.reset(len(x))
        h, cache = self._step(x, self.h)
        self.history.append(cache)
        self.h = h
        return np.asarray(np.argmax(self._probabilities(h), axis=-1))

    def probe(self, x: np.ndarray) -> np.ndarray:
        """Independent samples: one step from a zero state, nothing carried or learned."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        h, _ = self._step(x, np.zeros((len(x), self.hidden)))
        return np.asarray(np.argmax(self._probabilities(h), axis=-1))

    def gradients(
        self, labels: np.ndarray, p: np.ndarray, h: np.ndarray
    ) -> dict[str, np.ndarray]:
        """The mean cross-entropy gradient of the present frame through the cached history."""
        batch = len(h)
        onehot = np.zeros_like(p)
        onehot[np.arange(batch)[:, None], np.arange(self.slots)[None, :], labels] = 1.0
        dlogits = ((p - onehot) / (batch * self.slots)).reshape(batch, -1)
        grads = {"W": dlogits.T @ h, "b": dlogits.sum(axis=0)}
        for name, value in self.cell.parameters().items():
            grads[name] = np.zeros_like(value)
        w = self.cell.weights
        g = dlogits @ self.W
        for cache in reversed(self.history):
            x, h_prev, z, r, n, reset = (
                cache[k] for k in ("x", "h_prev", "z", "r", "n", "reset")
            )
            gn = g * (1.0 - z) * (1.0 - n * n)
            gz = g * (h_prev - n) * z * (1.0 - z)
            through_n = gn @ w["Un"]
            gr = through_n * h_prev * r * (1.0 - r)
            grads["Wn"] += gn.T @ x
            grads["Un"] += gn.T @ reset
            grads["bn"] += gn.sum(axis=0)
            grads["Wz"] += gz.T @ x
            grads["Uz"] += gz.T @ h_prev
            grads["bz"] += gz.sum(axis=0)
            grads["Wr"] += gr.T @ x
            grads["Ur"] += gr.T @ h_prev
            grads["br"] += gr.sum(axis=0)
            g = g * z + gz @ w["Uz"] + gr @ w["Ur"] + through_n * r
        return grads

    def lesson(self, x: np.ndarray, labels: np.ndarray, *, advance: bool = True) -> np.ndarray:
        """One online lesson on a batch of frames; the greedy answers before the step.

        With ``advance`` false the streams do not move on: the state and the cached history
        are those before the lesson, as a lesson that is not a moment of the life.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        labels = np.asarray(labels, dtype=np.int64).reshape(len(x), self.slots)
        if self.h.shape[0] != len(x):
            self.reset(len(x))
        kept = list(self.history)
        h, cache = self._step(x, self.h)
        p = self._probabilities(h)
        self.history.append(cache)
        frames = len(self.history)
        grads = self.gradients(labels, p, h)
        norm = float(np.sqrt(sum(float(np.sum(v * v)) for v in grads.values())))
        scale = min(1.0, self.clip / max(norm, 1e-300))
        self.updates += 1
        t = self.updates
        b1, b2 = self.betas
        for name, value in self.params().items():
            m, v = self.moments[name]
            grad = scale * grads[name]
            m *= b1
            m += (1 - b1) * grad
            v *= b2
            v += (1 - b2) * grad * grad
            value -= self.rate * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + self.eps)
        self.learning_macs += len(x) * (
            self.step_macs + frames * (self.cell.backward_macs + 3 * self.hidden**2)
        ) + 6 * self.parameter_count
        if advance:
            self.h = h
        else:
            self.history = deque(kept, maxlen=self.truncation)
        return np.asarray(np.argmax(p, axis=-1))


class Bank:
    """One ``SkillGRU`` per skill, each reading its own block: the routing is supplied."""

    def __init__(
        self, S: int, hidden: int, founder: int, *, d: int = 4, a: int = 2, **options: Any
    ) -> None:
        self.S, self.d, self.a, self.hidden = S, d, a, hidden
        self.modules = [
            SkillGRU(d, 1, a, hidden, [founder, 23, S, k, hidden], **options) for k in range(S)
        ]

    def block(self, cues: np.ndarray, k: int) -> np.ndarray:
        return np.asarray(lw.observations(np.asarray(cues)[:, k : k + 1], self.d))

    def lesson(self, cues: np.ndarray, labels: np.ndarray) -> None:
        for k, module in enumerate(self.modules):
            module.lesson(self.block(cues, k), np.asarray(labels)[:, k : k + 1])

    def probe(self, cues: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [m.probe(self.block(cues, k)) for k, m in enumerate(self.modules)], axis=1
        )

    @property
    def parameter_count(self) -> int:
        return int(sum(m.parameter_count for m in self.modules))

    @property
    def learning_macs(self) -> int:
        return int(sum(m.learning_macs for m in self.modules))


def _train(
    learner: Any, lesson: Any, probe: Any, S: int, founder: int, maps: Sequence[np.ndarray],
    *, d: int, budget: int, batch: int, every: int, gate: float, probe_rows: int,
) -> dict[str, Any]:
    """The Cadence instrument's acquisition schedule, for a conventional learner."""
    rng = np.random.default_rng([founder, 5, S])
    cues_probe = np.random.default_rng([founder, 7]).integers(0, d, size=(probe_rows, S))
    y_probe = labels2d(cues_probe, maps)
    seen: set[tuple[int, ...]] = set()
    history: list[dict[str, Any]] = []
    accuracy, predicted, taken = np.zeros(S), np.zeros_like(y_probe), 0
    started = time.perf_counter()
    for index in range(budget):
        cues = rng.integers(0, d, size=(batch, S))
        seen.update(tuple(row) for row in cues.tolist())
        lesson(cues, labels2d(cues, maps))
        taken = index + 1
        if taken % every == 0 or taken == budget:
            predicted = np.asarray(probe(cues_probe)).reshape(probe_rows, S)
            accuracy = (predicted == y_probe).mean(axis=0)
            history.append({"lessons": taken, "accuracy": [float(v) for v in accuracy]})
            if bool((accuracy >= gate).all()):
                break
    unseen = np.array([tuple(row) not in seen for row in cues_probe.tolist()], dtype=bool)
    within = [h for h in history if h["lessons"] <= 400 and min(h["accuracy"]) >= gate]
    return {
        "S": S, "founder": founder, "lessons": taken,
        "at_gate": bool((accuracy >= gate).all()),
        "at_gate_within_400": bool(within),
        "accuracy": [float(v) for v in accuracy],
        "heldout_rows": int(unseen.sum()),
        "heldout_accuracy": (
            [float(v) for v in (predicted[unseen] == y_probe[unseen]).mean(axis=0)]
            if unseen.any() else None
        ),
        "history": history,
        "parameters": int(learner.parameter_count),
        "learning_macs": int(learner.learning_macs),
        "seconds": time.perf_counter() - started,
    }


def train_gru(
    S: int, hidden: int, founder: int, *, d: int = 4, a: int = 2, budget: int = 2000,
    batch: int = 16, every: int = 20, gate: float = 0.95, probe_rows: int = 256,
    **options: Any,
) -> tuple[SkillGRU, dict[str, Any]]:
    maps = lw.skill_maps(S, founder, d, a)
    net = SkillGRU(S * d, S, a, hidden, [founder, 21, S, hidden], **options)
    record = _train(
        net, lambda c, y: net.lesson(lw.observations(c, d), y),
        lambda c: net.probe(lw.observations(c, d)), S, founder, maps, d=d, budget=budget,
        batch=batch, every=every, gate=gate, probe_rows=probe_rows,
    )
    record.update({"hidden": hidden, "learner": "gru", "step_macs": net.step_macs,
                   "macs": net.macs()})
    return net, record


def train_bank(
    S: int, hidden: int, founder: int, *, d: int = 4, a: int = 2, budget: int = 2000,
    batch: int = 16, every: int = 20, gate: float = 0.95, probe_rows: int = 256,
    **options: Any,
) -> tuple[Bank, dict[str, Any]]:
    maps = lw.skill_maps(S, founder, d, a)
    bank = Bank(S, hidden, founder, d=d, a=a, **options)
    record = _train(
        bank, bank.lesson, bank.probe, S, founder, maps, d=d, budget=budget, batch=batch,
        every=every, gate=gate, probe_rows=probe_rows,
    )
    record.update({"hidden": hidden, "learner": "bank",
                   "step_macs": bank.modules[0].step_macs, "macs": bank.modules[0].macs()})
    return bank, record


# -- the life, replayed from a Cadence case


def schedule_from_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    """The moments of a Cadence case: kind, episode, skill 0's cue, and for the relation
    change moments the probe index whose lesson precedes them."""
    probes = int(case["spec"]["probes"])
    out = []
    for m in case["moments"]:
        item = {"index": m["index"], "kind": m["kind"], "episode": m["episode"],
                "cue0": int(m["cue0"])}
        if m["kind"] == "relation":
            item["relation"] = int(m["episode"]) - probes
        out.append(item)
    return out


def _cues(S: int, cue0: int) -> np.ndarray:
    cues = np.zeros((1, S), dtype=np.int64)
    cues[0, 0] = cue0
    return cues


def _shown(cues: np.ndarray, maps: Sequence[np.ndarray], relation: int, a: int) -> np.ndarray:
    """The relation lesson's labels: skill 0's label moved on by ``1 + relation``, as the
    Cadence case's own lesson, the other slots their maps."""
    y = labels2d(cues, maps).copy()
    y[0, 0] = (y[0, 0] + 1 + relation) % a
    return y


def live_gru(
    net: SkillGRU, schedule: Sequence[dict[str, Any]], maps: Sequence[np.ndarray], *,
    d: int = 4, a: int = 2,
) -> list[dict[str, Any]]:
    """The GRU through the case's moments: every moment one step of every unit."""
    S = net.slots
    net.reset(1)
    out = []
    for item in schedule:
        cues = _cues(S, item["cue0"])
        x = np.asarray(lw.observations(cues, d))
        record: dict[str, Any] = {"index": item["index"], "kind": item["kind"],
                                  "episode": item["episode"]}
        if "relation" in item:
            before = {k: v.copy() for k, v in net.params().items()}
            macs = net.learning_macs
            net.lesson(x, _shown(cues, maps, item["relation"], a), advance=False)
            record["lesson_changed_parameters"] = int(sum(
                np.count_nonzero(v - before[k]) for k, v in net.params().items()
            ))
            record["lesson_macs"] = int(net.learning_macs - macs)
        started = time.perf_counter()
        action = net.answer(x)
        record["seconds"] = time.perf_counter() - started
        record["macs"] = net.step_macs
        record["units"] = net.hidden
        record["action"] = action.ravel().tolist()
        record["correct"] = bool(np.array_equal(action.ravel(), labels2d(cues, maps).ravel()))
        out.append(record)
    return out


def live_bank(
    bank: Bank, schedule: Sequence[dict[str, Any]], maps: Sequence[np.ndarray], *,
    tolerance: float = 3e-3,
) -> list[dict[str, Any]]:
    """The oracle-routed bank through the case's moments: a module steps when its block's
    input changed, its weights changed, or its last step moved its state beyond
    ``tolerance``; a skipped module holds its answer."""
    S, a = bank.S, bank.a
    for module in bank.modules:
        module.reset(1)
    moving = [True] * S
    answers = [0] * S
    previous: np.ndarray | None = None
    out = []
    for item in schedule:
        cues = _cues(S, item["cue0"])
        record: dict[str, Any] = {"index": item["index"], "kind": item["kind"],
                                  "episode": item["episode"]}
        if "relation" in item:
            module = bank.modules[0]
            macs = module.learning_macs
            shown = _shown(cues, maps, item["relation"], a)
            module.lesson(bank.block(cues, 0), shown[:, :1], advance=False)
            record["lesson_macs"] = int(module.learning_macs - macs)
            record["lesson_changed_parameters"] = module.parameter_count
            moving[0] = True  # its function changed: the router steps it again
        started = time.perf_counter()
        stepped = []
        for k, module in enumerate(bank.modules):
            changed = previous is None or cues[0, k] != previous[0, k]
            if changed or moving[k]:
                before = module.h.copy()
                answers[k] = int(module.answer(bank.block(cues, k))[0, 0])
                moving[k] = bool(np.max(np.abs(module.h - before)) > tolerance)
                stepped.append(k)
        record["seconds"] = time.perf_counter() - started
        record["stepped"] = stepped
        record["macs"] = int(sum(bank.modules[k].step_macs for k in stepped))
        record["units"] = bank.hidden * len(stepped)
        record["action"] = list(answers)
        record["correct"] = bool(np.array_equal(
            np.asarray(answers), labels2d(cues, maps).ravel()
        ))
        previous = cues
        out.append(record)
    return out


def life_summary(moments: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per change moment, per held moment and per episode (change plus its held moments)."""
    out: dict[str, Any] = {}
    for kind in ("input", "relation"):
        change = [m for m in moments if m["kind"] == kind]
        held = [m for m in moments if m["kind"] == f"{kind}-hold"]
        episodes = sorted({m["episode"] for m in change})
        per_episode = [
            sum(m["macs"] for m in moments
                if m["episode"] == e and m["kind"] in (kind, f"{kind}-hold"))
            for e in episodes
        ]
        out[kind] = {
            "change_macs": float(np.mean([m["macs"] for m in change])) if change else None,
            "held_macs": float(np.mean([m["macs"] for m in held])) if held else None,
            "episode_macs": float(np.mean(per_episode)) if per_episode else None,
            "episode_moments": (
                float(np.mean([sum(1 for m in moments if m["episode"] == e
                                   and m["kind"] in (kind, f"{kind}-hold"))
                               for e in episodes])) if episodes else None
            ),
        }
    seconds = [m["seconds"] for m in moments]
    out["seconds_median"] = float(np.median(seconds))
    out["correct"] = int(sum(m["correct"] for m in moments))
    out["moments"] = len(moments)
    lessons = [m for m in moments if "lesson_macs" in m]
    out["lesson_macs_mean"] = (
        float(np.mean([m["lesson_macs"] for m in lessons])) if lessons else None
    )
    out["lesson_changed_parameters_mean"] = (
        float(np.mean([m["lesson_changed_parameters"] for m in lessons])) if lessons else None
    )
    return out


# -- the declared campaigns


def search(
    skills: Sequence[int], hidden: Sequence[int], founders: Sequence[int], out: Path, *,
    budget: int = 2000,
) -> dict[str, Any]:
    """For each S, the smallest hidden width at which every founder reaches the gate."""
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"budget": budget, "hidden_grid": list(hidden),
                              "founders": list(founders), "runs": [], "selected": {}}
    for S in skills:
        for width in hidden:
            records = []
            for founder in founders:
                _, record = train_gru(S, width, founder, budget=budget)
                record.pop("history")
                records.append(record)
                print(json.dumps({k: record[k] for k in (
                    "S", "hidden", "founder", "lessons", "at_gate", "accuracy", "seconds"
                )}), flush=True)
            result["runs"].extend(records)
            (out / "search.json").write_text(json.dumps(result, indent=1))
            if all(r["at_gate"] for r in records):
                result["selected"][str(S)] = width
                break
        (out / "search.json").write_text(json.dumps(result, indent=1))
    return result


def report(
    selected: dict[str, int], case_dirs: Sequence[Path], founders: Sequence[int], out: Path,
    *, budget: int = 2000,
) -> None:
    """The baselines on every block-anatomy, declared-stop, cone-check Cadence case of the
    given founders: the GRU at the selected width, the size-matched GRU, and the bank."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    cases = []
    for folder in case_dirs:
        for path in sorted(Path(folder).glob("sparse-S*-cone*.json")):
            case = json.loads(path.read_text())
            spec = case["spec"]
            if (spec["founder"] in founders and spec.get("local_margin", 1.0) == 1.0
                    and spec.get("check") == "cone" and spec.get("lesson", "library") == "library"
                    and spec.get("momentum") is None):
                cases.append(case)
    trained: dict[tuple[str, int, int, int], Any] = {}
    for case in sorted(cases, key=lambda c: (c["spec"]["S"], c["spec"]["widths"],
                                             c["spec"]["founder"])):
        spec = case["spec"]
        S, founder, widths = spec["S"], spec["founder"], spec["widths"]
        maps = lw.skill_maps(S, founder, spec["d"], spec["a"])
        schedule = schedule_from_case(case)
        record: dict[str, Any] = {"cadence_case": spec["name"], "S": S, "widths": widths,
                                  "founder": founder, "learners": {}}
        arms = (
            ("gru", "gru", selected[str(S)]),
            ("gru_matched", "gru", int(sum(widths))),
            ("bank", "bank", selected["1"]),
        )
        for label, kind, width in arms:
            key = (kind, S, width, founder)
            if key not in trained:
                maker = train_gru if kind == "gru" else train_bank
                learner, training = maker(S, width, founder, budget=budget)
                trained[key] = (learner, training)
                print(json.dumps({"trained": label, "S": S, "hidden": width,
                                  "founder": founder, "lessons": training["lessons"],
                                  "at_gate": training["at_gate"],
                                  "seconds": round(training["seconds"], 1)}), flush=True)
            learner, training = trained[key]
            life = copy.deepcopy(learner)
            moments = (live_gru(life, schedule, maps) if kind == "gru"
                       else live_bank(life, schedule, maps))
            record["learners"][label] = {
                "hidden": width, "training": training, "life": life_summary(moments),
                "moments": moments,
            }
        name = f"baselines-{spec['name']}"
        (out / "cases" / f"{name}.json").write_text(json.dumps(record))
        print(json.dumps({"case": spec["name"], **{
            k: {"macs": v["life"]["input"]["change_macs"], "correct": v["life"]["correct"]}
            for k, v in record["learners"].items()
        }}), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search")
    s.add_argument("--skills", type=int, nargs="+", default=list(SKILLS))
    s.add_argument("--hidden", type=int, nargs="+", default=list(HIDDEN_GRID))
    s.add_argument("--founders", type=int, nargs="+", default=[0, 1, 2])
    s.add_argument("--budget", type=int, default=2000)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--search", type=Path, required=True)
    r.add_argument("--cases", type=Path, nargs="+", required=True)
    r.add_argument("--founders", type=int, nargs="+", default=[3, 4, 5])
    r.add_argument("--budget", type=int, default=2000)
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "search":
        search(args.skills, args.hidden, args.founders, args.out, budget=args.budget)
    else:
        selected = json.loads(args.search.read_text())["selected"]
        report(selected, args.cases, args.founders, args.out, budget=args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
