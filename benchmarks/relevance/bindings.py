"""Named-fact bindings drawn fresh every trial, read on value pairs never shown in training.

Two named facts A and B of ``bits`` bits each are drawn fresh for every trial and shown at two
prefix events in the instrument's encoding (one coordinate per name, bit and value; with one
bit it is ``selective_recall.inputs`` exactly), followed by a neutral event; the query names one
of them, and the answer is the queried fact's bits, one output slot of two neurons per bit. A
founder's held-out pairs, a quarter of all value pairs, are never shown in training, while
every single value of each fact is; a table over trained trace patterns cannot answer them.
The protocol and its gate are declared in ``BINDINGS_PROTOCOL.md``.

Learners on the same trials: the composed brain at its founder genes (the trace arm, facts
absent at the query; the exposed arm, both facts present at the query) under lessons judged by
the brain's own act (the learner's opt-in ``surprise`` rule, given the act's answer) or on
every row, and a supervised online GRU control (``benchmarks/competitors.py``'s cell, a softmax
per bit, cross-entropy through the trial by backpropagation through time). Each life is read
at every budget on forks: intact, with the working memory erased before the query, and with
the working memory of another trial transplanted before the query. Nothing here changes the
library.

    PYTHONPATH=src python benchmarks/relevance/bindings.py --out DIR --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
import tempfile
import time
import warnings
from collections.abc import Iterator
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import selective_recall as sr  # noqa: E402

BITS = 3
BATCH = 16  # trials per round, one stream each, one lesson per round
BUDGETS = (512, 2048, 8192)  # rounds; each life is read at every budget
FOUNDERS = tuple(range(12))
BRAINS = {"m32": [32], "m64": [64]}
ARMS = ("trace", "exposed")
RULES = ("act", "every")
LEARNERS = ("graph", "gru")
THROUGH = ("trial", "query")  # the GRU's gradient: through the trial, or the query event alone
CONDITIONS = ("intact", "erased", "transplant")
NEUTRAL_EVENTS = 1
SCHEDULE_SEED, HOLDOUT_SEED, GRU_SEED = 31017, 31018, 31019
GRU = {"hidden": 64, "rate": 1.0, "clip": 5.0}
WINDOW = 256  # rounds per entry of the training log
GATE = {
    "held_out_exact": 0.75,
    "erased_per_bit_margin": 0.1,
    "erased_exact_factor": 2.0,
    "founders": 8,
}
COMPETITORS_PATH = HERE.parent / "competitors.py"


def _competitors() -> Any:
    spec = importlib.util.spec_from_file_location("competitors", COMPETITORS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the task


def width(bits: int) -> int:
    """Inputs: A's bits, B's bits (two coordinates per bit, one per value), neutral, ask-A,
    ask-B, then the exposed copy of A's and B's bits."""
    return 8 * bits + 3


def unpack(values: np.ndarray, bits: int) -> np.ndarray:
    """The bits of each value, least significant first: ``(n, bits)``."""
    return (np.asarray(values)[:, None] >> np.arange(bits)) & 1


def inputs(
    rows: np.ndarray,
    bits: int,
    neutral_events: int = NEUTRAL_EVENTS,
    *,
    exposed: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Prefix events, the query event and the answer of trials ``(A, B, order, query)``.

    The first event shows A (order 0) or B, the second the other fact: the coordinate
    ``2 i + v`` for A's bit ``i`` of value ``v``, ``2 bits + 2 i + v`` for B's. The neutral
    events follow. The query names A or B; the exposed arm adds both facts to it. The answer
    is the queried fact's bits, ``(n, bits)``.
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 4 or not len(rows) or rows.dtype.kind not in "iu":
        raise ValueError("rows hold integer A, B, order and query")
    a, b, order, query = rows.T
    if ((a < 0) | (a >= 2**bits) | (b < 0) | (b >= 2**bits)).any():
        raise ValueError(f"facts are values of {bits} bits")
    if not (np.isin(order, (0, 1)).all() and np.isin(query, (0, 1)).all()):
        raise ValueError("order and query are binary")
    n, columns = len(rows), np.arange(bits)
    fa, fb = unpack(a, bits), unpack(b, bits)
    shown_a, shown_b = np.zeros((n, width(bits))), np.zeros((n, width(bits)))
    np.put_along_axis(shown_a, 2 * columns + fa, 1.0, axis=1)
    np.put_along_axis(shown_b, 2 * bits + 2 * columns + fb, 1.0, axis=1)
    first = np.where(order[:, None] == 0, shown_a, shown_b)
    second = np.where(order[:, None] == 0, shown_b, shown_a)
    prefix = [first, second]
    neutral = np.zeros((n, width(bits)))
    neutral[:, 4 * bits] = 1.0
    prefix.extend(neutral.copy() for _ in range(neutral_events))
    ask = np.zeros((n, width(bits)))
    ask[np.arange(n), 4 * bits + 1 + query] = 1.0
    if exposed:
        ask[:, 4 * bits + 3 :] = shown_a[:, : 4 * bits] + shown_b[:, : 4 * bits]
    target = np.where(query[:, None] == 0, fa, fb)
    return prefix, ask, target


def holdout(seed: int, bits: int) -> np.ndarray:
    """The founder's held-out value pairs, ``held[A, B]``: a quarter of all pairs, drawn from
    the founder seed as ``(p(A) + q(B)) mod 4 == 0`` for two permutations ``p, q`` of the
    values. Every value of A keeps three quarters of the values of B in training and every
    value of B three quarters of the values of A (with at least two bits)."""
    rng = np.random.default_rng(np.random.SeedSequence([HOLDOUT_SEED, seed]))
    values = 2**bits
    p, q = rng.permutation(values), rng.permutation(values)
    return (p[:, None] + q[None, :]) % 4 == 0


def trials(seed: int, bits: int, rounds: int, held: np.ndarray) -> Iterator[np.ndarray]:
    """The founder's training trials, ``BATCH`` per round, each drawn fresh: a value pair
    uniformly from the training pairs, an order and a query. One round's draws follow the
    previous round's, so a shorter budget's trials are the start of a longer one's."""
    rng = np.random.default_rng(np.random.SeedSequence([SCHEDULE_SEED, seed]))
    training = np.argwhere(~held)
    for _ in range(rounds):
        pair = training[rng.integers(0, len(training), BATCH)]
        order = rng.integers(0, 2, BATCH)
        query = rng.integers(0, 2, BATCH)
        yield np.column_stack([pair, order, query]).astype(np.int64)


def histories(bits: int) -> np.ndarray:
    """The evaluation panel: every value pair in either order, ``(A, B, order)`` at index
    ``2 (A 2**bits + B) + order``."""
    values = range(2**bits)
    return np.array(list(itertools.product(values, values, (0, 1))), dtype=np.int64)


def donors(panel: np.ndarray, bits: int) -> np.ndarray:
    """The transplant's donor of each history: both facts complemented, the same order, so
    every bit of the donor's queried fact differs from the host's."""
    mask = 2**bits - 1
    return 2 * (((panel[:, 0] ^ mask) << bits) + (panel[:, 1] ^ mask)) + panel[:, 2]


def asked(panel: np.ndarray, query: int) -> np.ndarray:
    return np.column_stack([panel, np.full(len(panel), query)]).astype(np.int64)


def truth(panel: np.ndarray, bits: int) -> np.ndarray:
    """The answer of every history under each query: ``(histories, 2, bits)``."""
    return np.stack([inputs(asked(panel, q), bits)[2] for q in (0, 1)], axis=1)


# -- readings


def score(answers: dict[str, np.ndarray], held: np.ndarray, bits: int) -> dict[str, Any]:
    """Exact-match and per-bit accuracy of each condition on the training and the held-out
    pairs; -1 marks an unanswered case. ``other_fact_unequal`` counts answers equal to the
    fact that was not asked, on pairs whose facts differ. The transplant adds ``follow``
    (the donor's queried fact) and ``stay`` (the host's intact answer)."""
    panel = histories(bits)
    expected = truth(panel, bits)
    donor = expected[donors(panel, bits)]
    held_rows = held[panel[:, 0], panel[:, 1]]
    unequal = panel[:, 0] != panel[:, 1]
    out: dict[str, Any] = {}
    for condition, answer in answers.items():
        answer = np.asarray(answer)
        valid = (answer >= 0).all(axis=-1)
        right = (answer == expected) & valid[..., None]
        exact = right.all(axis=-1)
        other = (answer == expected[:, ::-1]).all(axis=-1) & valid
        out[condition] = {"unanswered": int((~valid).sum())}
        for name, rows in (("training", ~held_rows), ("held_out", held_rows)):
            entry = {
                "cases": int(2 * rows.sum()),
                "exact": float(exact[rows].mean()),
                "per_bit": float(right[rows].mean()),
                "exact_per_query": [float(v) for v in exact[rows].mean(axis=0)],
                "exact_unequal": float(exact[rows & unequal].mean()),
                "other_fact_unequal": float(other[rows & unequal].mean()),
            }
            if condition == "transplant":
                follow = (answer == donor) & valid[..., None]
                entry["follow_exact"] = float(follow.all(axis=-1)[rows].mean())
                entry["follow_per_bit"] = float(follow[rows].mean())
                stay = (answer == np.asarray(answers["intact"])).all(axis=-1) & valid
                entry["stay_exact"] = float(stay[rows].mean())
            out[condition][name] = entry
    return out


def gate(reading: dict[str, Any], bits: int) -> bool:
    """The declared single-founder gate: held-out exact match at least 0.75, with the erased
    reading at chance on the held-out pairs (per-bit within 0.1 of 0.5, exact match at most
    twice the chance of ``2**-bits``)."""
    if not reading["completed"]:
        return False
    intact = reading["scores"]["intact"]["held_out"]
    erased = reading["scores"]["erased"]["held_out"]
    return bool(
        intact["exact"] >= GATE["held_out_exact"]
        and abs(erased["per_bit"] - 0.5) <= GATE["erased_per_bit_margin"]
        and erased["exact"] <= GATE["erased_exact_factor"] * 2.0**-bits
    )


def lesion(memory: Any, condition: str, donor: np.ndarray) -> None:
    """The working trace before the query: kept, erased, or replaced row by row with the
    trace another trial built (``donor``), under the host's own warm state."""
    if condition == "erased":
        memory.reset(len(donor))
    elif condition == "transplant":
        for name in ("trace", "last", "cold"):
            setattr(memory, name, getattr(memory, name)[donor].copy())
    elif condition != "intact":
        raise ValueError(condition)


# -- the composed brain


def teach(
    work: sr.Work,
    brain: Any,
    drive: np.ndarray,
    target: np.ndarray,
    answer: np.ndarray | None = None,
) -> dict[str, float]:
    """One lesson from the complete pre-query stimulus, through the learner; under the
    learner's ``surprise`` rule ``answer`` (the brain's act) selects the rows it teaches."""
    from cadence.learning import LearningPhaseError  # noqa: PLC0415

    work.lessons += 1
    try:
        _, report = brain.learner.step(drive, target, answer=answer)
    except LearningPhaseError as error:
        work.lesson_refusals += 1
        report = error.report
    for field, reported in (
        ("presentations", "attempted_presentations"),
        ("accepted_presentations", "accepted_presentations"),
        ("teaching_sweeps", "total_steps"),
        ("teaching_row_sweeps", "total_row_sweeps"),
        ("teaching_residual_checks", "total_residual_checks"),
        ("teaching_row_residual_checks", "total_row_residual_checks"),
    ):
        setattr(work, field, getattr(work, field) + int(report.get(reported, 0)))
    return report


def compose(brain_name: str, rule: str, seed: int, bits: int) -> Any:
    """The founder at its composed genes; the act rule is the learner's opt-in ``surprise``."""
    from cadence import Brain  # noqa: PLC0415

    genes = {"learning_surprise": True} if rule == "act" else {}
    return Brain.compose(
        width(bits),
        2 * bits,
        seed=seed,
        modules=BRAINS[brain_name],
        episodic=False,
        slots=(2,) * bits,
        **genes,
    )


def graph_round(
    brain: Any,
    work: sr.Work,
    rows: np.ndarray,
    bits: int,
    *,
    exposed: bool,
    rule: str,
) -> tuple[np.ndarray | None, dict[str, float] | None, str | None]:
    """One round: every stream from rest through its prefix, the drive of the query, the
    brain's own act, then the lesson from that drive: the rows the act got wrong in any bit
    (``act``) or every row (``every``)."""
    prefix, query, target = inputs(rows, bits, exposed=exposed)
    brain.reset()  # the same learned parameters, a trial boundary for every stream
    for observation in prefix:
        if work.act(brain, observation) is None:
            return None, None, "refused_prefix"
    drive = brain.stimulus(query)  # the complete pre-query stimulus, the trace the prefix built
    answer = work.act(brain, query)
    if answer is None:
        return None, None, "refused_query"
    answer = np.asarray(answer).reshape(len(rows), bits)
    report = teach(work, brain, drive, target, answer if rule == "act" else None)
    return answer, report, None


def graph_reading(
    brain: Any,
    bits: int,
    held: np.ndarray,
    work: sr.Work,
    *,
    exposed: bool,
) -> dict[str, Any]:
    """Every history through its prefix on a fork of the trained brain, one saved pre-query
    state, then each condition and query on its own fork of that state; no lesson. The
    trained brain's checkpoint must read the same afterwards."""
    from cadence import Brain  # noqa: PLC0415

    panel = histories(bits)
    donor = donors(panel, bits)
    answers = {c: np.full((len(panel), 2, bits), -1, dtype=np.int64) for c in CONDITIONS}
    complete = True
    with tempfile.TemporaryDirectory(prefix="cadence-bindings-") as directory:
        directory = Path(directory)
        initial = brain.save(directory / "initial.npz")
        original = sr.checkpoint_digest(initial)
        walker = Brain.load(initial)
        walker.reset()
        prefix, _, _ = inputs(asked(panel, 0), bits, exposed=exposed)
        for observation in prefix:
            if work.act(walker, observation) is None:
                complete = False
                break
        state = walker.save(directory / "prequery.npz")
        if complete:
            for condition in CONDITIONS:
                for q in (0, 1):
                    branch = Brain.load(state)
                    lesion(branch.working_memory, condition, donor)
                    _, query, _ = inputs(asked(panel, q), bits, exposed=exposed)
                    answer = work.act(branch, query)
                    if answer is None:
                        complete = False
                    else:
                        answers[condition][:, q] = np.asarray(answer).reshape(len(panel), bits)
        if sr.checkpoint_digest(brain.save(directory / "after.npz")) != original:
            raise AssertionError("the reading changed the trained brain")
    return {
        "completed": complete,
        "answers": {c: a.tolist() for c, a in answers.items()},
        "scores": score(answers, held, bits),
    }


# -- the supervised GRU control


class Supervised:
    """A GRU (``competitors.GRUCell``) read by one softmax per bit, trained online on the
    same rounds: one step per round on the mean cross-entropy of its trials, bounded in norm
    by ``clip``. ``through`` ``trial`` takes the gradient through every event of the trial
    (backpropagation through time); ``query`` through the query event alone, the state
    entering it held fixed (the credit path of the brain's lesson at the query). The hidden
    state starts at zero in every trial."""

    def __init__(
        self,
        bits: int,
        seed: int,
        *,
        hidden: int = GRU["hidden"],
        rate: float = GRU["rate"],
        clip: float = GRU["clip"],
        through: str = "trial",
    ) -> None:
        if through not in THROUGH:
            raise ValueError(f"through is one of {THROUGH}")
        rng = np.random.default_rng(np.random.SeedSequence([GRU_SEED, seed]))
        self.bits, self.hidden, self.rate, self.clip = bits, hidden, rate, clip
        self.through = through
        self.cell = _competitors().GRUCell(width(bits), hidden, rng)
        self.readout = {"W": np.zeros((2 * bits, hidden)), "b": np.zeros(2 * bits)}
        self.work = {"updates": 0, "rows": 0, "clipped": 0, "forward_events": 0}

    def parameters(self) -> dict[str, np.ndarray]:
        return {**self.readout, **self.cell.parameters()}

    def describe(self) -> dict[str, Any]:
        return {
            "cell": "gru",
            "inputs": width(self.bits),
            "hidden": self.hidden,
            "rate": self.rate,
            "clip": self.clip,
            "through": self.through,
            "parameters": int(sum(v.size for v in self.parameters().values())),
        }

    def run(
        self, events: list[np.ndarray], start: np.ndarray | None = None
    ) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        """Each row's events from ``start`` (zero when absent): the last hidden states and
        every row's caches."""
        n = len(events[0])
        h = np.zeros((n, self.hidden)) if start is None else np.array(start, dtype=float)
        caches: list[list[dict[str, Any]]] = [[] for _ in range(n)]
        for x in events:
            for i in range(n):
                h[i], cache = self.cell.forward(x[i], h[i].copy())
                caches[i].append(cache)
            self.work["forward_events"] += n
        return h, caches

    def probabilities(self, h: np.ndarray) -> np.ndarray:
        logits = (h @ self.readout["W"].T + self.readout["b"]).reshape(len(h), self.bits, 2)
        logits = logits - logits.max(axis=-1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=-1, keepdims=True)

    def answer(self, h: np.ndarray) -> np.ndarray:
        return np.argmax(self.probabilities(h), axis=-1).astype(np.int64)

    def state_gradient(
        self, g: np.ndarray, cache: dict[str, Any], part: dict[str, np.ndarray]
    ) -> np.ndarray:
        """The gradient of ``g . h`` with respect to ``h_prev`` (the cell's ``backward``
        holds it fixed), from that backward's gate gradients."""
        w = self.cell.weights
        return (
            g * cache["z"]
            + w["Uz"].T @ part["bz"]
            + w["Ur"].T @ part["br"]
            + (w["Un"].T @ part["bn"]) * cache["r"]
        )

    def gradients(
        self, events: list[np.ndarray], target: np.ndarray, start: np.ndarray | None = None
    ) -> tuple[float, dict[str, np.ndarray], np.ndarray]:
        """The mean cross-entropy over the rows and bits, its gradient (through the trial or
        the query event), and the answers of the forward pass from ``start``."""
        h, caches = self.run(events, start)
        p = self.probabilities(h)
        n = len(target)
        loss = float(-np.log(np.take_along_axis(p, target[..., None], -1) + 1e-300).mean())
        onehot = np.zeros_like(p)
        np.put_along_axis(onehot, target[..., None], 1.0, axis=-1)
        dlogits = (p - onehot).reshape(n, 2 * self.bits) / (n * self.bits)
        grads = {k: np.zeros_like(v) for k, v in self.parameters().items()}
        grads["W"] = dlogits.T @ h
        grads["b"] = dlogits.sum(axis=0)
        for i in range(n):
            g = self.readout["W"].T @ dlogits[i]
            for cache in reversed(caches[i] if self.through == "trial" else caches[i][-1:]):
                part = self.cell.backward(g, cache)
                for name, value in part.items():
                    grads[name] += value
                g = self.state_gradient(g, cache, part)
        return loss, grads, np.argmax(p, axis=-1).astype(np.int64)

    def lesson(self, events: list[np.ndarray], target: np.ndarray) -> tuple[float, np.ndarray]:
        loss, grads, answer = self.gradients(events, target)
        norm = float(np.sqrt(sum(float(np.sum(v * v)) for v in grads.values())))
        scale = min(1.0, self.clip / norm) if norm > 0 else 1.0
        self.work["clipped"] += int(scale < 1.0)
        for name, value in self.parameters().items():
            value -= self.rate * scale * grads[name]
        self.work["updates"] += 1
        self.work["rows"] += len(target)
        return loss, answer


def gru_reading(model: Supervised, bits: int, held: np.ndarray, *, exposed: bool) -> dict[str, Any]:
    """The GRU's reading: every history through its prefix, then the query from the kept,
    the erased (zero) or the donor's hidden state; no lesson."""
    panel = histories(bits)
    donor = donors(panel, bits)
    counted = dict(model.work)  # the ledger counts training alone
    prefix, _, _ = inputs(asked(panel, 0), bits, exposed=exposed)
    h, _ = model.run(prefix)
    starts = {"intact": h, "erased": np.zeros_like(h), "transplant": h[donor]}
    answers = {c: np.full((len(panel), 2, bits), -1, dtype=np.int64) for c in CONDITIONS}
    for condition in CONDITIONS:
        for q in (0, 1):
            _, query, _ = inputs(asked(panel, q), bits, exposed=exposed)
            last, _ = model.run([query], starts[condition])
            answers[condition][:, q] = model.answer(last)
    model.work.update(counted)
    return {
        "completed": True,
        "answers": {c: a.tolist() for c, a in answers.items()},
        "scores": score(answers, held, bits),
    }


# -- one life


def job(spec: dict[str, Any]) -> dict[str, Any]:
    """One life to the longest budget, read at every budget; ``spec`` names the learner,
    brain, arm, rule, founder, bits and budgets."""
    began = time.monotonic()
    learner, arm, seed, bits = spec["learner"], spec["arm"], spec["seed"], spec["bits"]
    budgets = sorted(spec["budgets"])
    exposed = arm == "exposed"
    held = holdout(seed, bits)
    row: dict[str, Any] = {
        "learner": learner,
        "brain": spec.get("brain"),
        "arm": arm,
        "rule": spec.get("rule"),
        "seed": seed,
        "bits": bits,
        "batch": BATCH,
        "neutral_events": NEUTRAL_EVENTS,
        "budgets": budgets,
        "held_out_pairs": np.argwhere(held).tolist(),
    }
    # per WINDOW rounds: rows taught, the act's exact and per-bit accuracy on its own training
    # trials, and lessons whose free phase used its whole budget (the learner's report)
    log = {
        "window": WINDOW,
        "taught_rows": [],
        "act_exact": [],
        "act_per_bit": [],
        "lessons_exhausted": [],
    }
    # counted in the log instead of printed once per worker
    warnings.filterwarnings("ignore", "finite teaching free phase", RuntimeWarning)
    readings: dict[str, Any] = {}
    reading_seconds: dict[str, float] = {}
    rounds, stop = 0, None
    if learner == "graph":
        brain = compose(spec["brain"], spec["rule"], seed, bits)
        described = brain.describe()
        row["modules"] = BRAINS[spec["brain"]]
        row["resolved_working_memory"] = described["working_memory"]
        row["resolved_learning"] = described["learning"]
        work, reading_work = sr.Work(), sr.Work()
    else:
        model = Supervised(bits, seed, through=spec["rule"], **spec.get("gru", {}))
        row["gru"] = model.describe()
    window = np.zeros(4)
    for rows in trials(seed, bits, budgets[-1], held):
        if learner == "graph":
            answer, report, stop = graph_round(
                brain, work, rows, bits, exposed=exposed, rule=spec["rule"]
            )
            if stop is not None:
                break
            assert answer is not None and report is not None
            taught = float(report.get("accepted_presentations", 0.0))
            exhausted = float(report.get("free_budget_exhausted", 0.0))
        else:
            prefix, query, target = inputs(rows, bits, exposed=exposed)
            _, answer = model.lesson([*prefix, query], target)
            taught, exhausted = float(len(rows)), 0.0
        target = inputs(rows, bits)[2]
        window += (
            taught,
            float((answer == target).all(axis=1).sum()),
            float((answer == target).mean() * len(rows)),
            exhausted,
        )
        rounds += 1
        if rounds % WINDOW == 0 or rounds == budgets[-1]:
            size = (rounds - 1) % WINDOW + 1
            log["taught_rows"].append(int(window[0]))
            log["act_exact"].append(round(window[1] / (size * BATCH), 4))
            log["act_per_bit"].append(round(window[2] / (size * BATCH), 4))
            log["lessons_exhausted"].append(int(window[3]))
            window[:] = 0
        if rounds in budgets:
            start = time.monotonic()
            readings[str(rounds)] = (
                graph_reading(brain, bits, held, reading_work, exposed=exposed)
                if learner == "graph"
                else gru_reading(model, bits, held, exposed=exposed)
            )
            readings[str(rounds)]["gate"] = gate(readings[str(rounds)], bits)
            reading_seconds[str(rounds)] = round(time.monotonic() - start, 3)
    row["training"] = {
        "rounds": rounds,
        "completed": rounds == budgets[-1],
        "stop": stop,
        "log": log,
    }
    row["readings"] = readings
    if learner == "graph":
        row["work"] = asdict(work)
        row["reading_work"] = asdict(reading_work)
    else:
        row["work"] = dict(model.work)
    row["reading_seconds"] = reading_seconds
    row["elapsed_seconds"] = round(time.monotonic() - began, 3)
    return row


def name(spec: dict[str, Any]) -> str:
    if spec["learner"] == "graph":
        return f"graph-{spec['brain']}-{spec['arm']}-{spec['rule']}-{spec['seed']}"
    return f"gru-{spec['rule']}-{spec['arm']}-{spec['seed']}"


# -- tables


def cells(rows: list[dict[str, Any]]) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    """Runs grouped by learner, brain, arm and rule, founders ascending."""

    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (
            LEARNERS.index(r["learner"]),
            ARMS.index(r["arm"]),
            list(BRAINS).index(r["brain"]) if r["brain"] else -1,
            (RULES if r["learner"] == "graph" else THROUGH).index(r["rule"]),
        )

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in sorted(rows, key=lambda r: (key(r), r["seed"])):
        groups.setdefault((r["learner"], r["brain"], r["arm"], r["rule"]), []).append(r)
    return sorted(groups.items(), key=lambda item: key(item[1][0]))


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| learner | brain | arm | lessons | rounds | training exact | held-out exact "
        "[per founder] | held-out per-bit | erased held-out exact | erased held-out per-bit "
        "| transplant follow, held-out | founders at the gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (learner, brain, arm, rule), cell in cells(rows):
        budgets = sorted({int(b) for r in cell for b in r["readings"]})
        for budget in budgets:
            got = [r["readings"][str(budget)] for r in cell if str(budget) in r["readings"]]
            s = [g["scores"] for g in got]
            held = [v["intact"]["held_out"]["exact"] for v in s]
            lines.append(
                f"| {learner} | {brain or ''} | {arm} "
                f"| {rule if learner == 'graph' else 'every row, through the ' + rule} "
                f"| {budget} | {np.mean([v['intact']['training']['exact'] for v in s]):.3f} "
                f"| {np.mean(held):.3f} [{' '.join(f'{h:.2f}' for h in held)}] "
                f"| {np.mean([v['intact']['held_out']['per_bit'] for v in s]):.3f} "
                f"| {np.mean([v['erased']['held_out']['exact'] for v in s]):.3f} "
                f"| {np.mean([v['erased']['held_out']['per_bit'] for v in s]):.3f} "
                f"| {np.mean([v['transplant']['held_out']['follow_exact'] for v in s]):.3f} "
                f"| {sum(bool(g['gate']) for g in got)}/{len(got)} |"
            )
    return "\n".join(lines)


def digests() -> dict[str, str]:
    paths = {
        "benchmarks/relevance/bindings.py": Path(__file__).resolve(),
        "benchmarks/relevance/selective_recall.py": HERE / "selective_recall.py",
        "benchmarks/competitors.py": COMPETITORS_PATH,
    }
    out = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    out.update(
        {
            k: hashlib.sha256(p.read_bytes()).hexdigest()
            for k, p in sr.sources()
            if k.startswith("src/")
        }
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learners", nargs="*", default=list(LEARNERS), choices=LEARNERS)
    parser.add_argument("--brains", nargs="*", default=list(BRAINS), choices=list(BRAINS))
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    parser.add_argument("--rules", nargs="*", default=list(RULES), choices=RULES)
    parser.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--bits", type=int, default=BITS)
    parser.add_argument("--gru-hidden", type=int, default=GRU["hidden"])
    parser.add_argument("--gru-rate", type=float, default=GRU["rate"])
    parser.add_argument("--gru-through", nargs="*", default=list(THROUGH), choices=THROUGH)
    args = parser.parse_args(argv)
    if not 2 <= args.bits <= 6:
        parser.error("bits must lie in [2, 6]")
    if not args.budgets or min(args.budgets) < 1:
        parser.error("budgets are positive round counts")
    args.out.mkdir(parents=True, exist_ok=True)
    gru = {"hidden": args.gru_hidden, "rate": args.gru_rate, "clip": GRU["clip"]}
    common = {"bits": args.bits, "budgets": sorted(set(args.budgets))}
    specs = []
    if "graph" in args.learners:
        # the longest lives first, so the workers finish together
        for brain, arm, rule, seed in itertools.product(
            sorted(args.brains, key=lambda b: -BRAINS[b][0]), args.arms, args.rules, args.seeds
        ):
            specs.append(
                {
                    "learner": "graph",
                    "brain": brain,
                    "arm": arm,
                    "rule": rule,
                    "seed": seed,
                    **common,
                }
            )
    if "gru" in args.learners:
        specs.extend(
            {"learner": "gru", "rule": through, "arm": arm, "seed": seed, "gru": gru, **common}
            for through, arm, seed in itertools.product(args.gru_through, args.arms, args.seeds)
        )
    done = {p.stem for p in args.out.glob("*.json") if p.stem != "summary"}
    todo = [s for s in specs if name(s) not in done]
    print(f"{len(todo)} lives to go, {len(done)} done", flush=True)
    import cadence  # noqa: PLC0415

    if todo:
        with get_context("spawn").Pool(max(1, min(args.workers, len(todo)))) as pool:
            for row in pool.imap_unordered(job, todo):
                spec = {
                    "learner": row["learner"],
                    "brain": row["brain"],
                    "arm": row["arm"],
                    "rule": row["rule"],
                    "seed": row["seed"],
                }
                (args.out / f"{name(spec)}.json").write_text(json.dumps(row, indent=1))
                readings = " ".join(
                    f"{b}: {r['scores']['intact']['training']['exact']:.2f}"
                    f"/{r['scores']['intact']['held_out']['exact']:.2f}"
                    f"/{r['scores']['erased']['held_out']['exact']:.2f}"
                    f"{' gate' if r['gate'] else ''}"
                    for b, r in row["readings"].items()
                )
                print(f"  {name(spec)}: {readings} ({row['elapsed_seconds']}s)", flush=True)
    rows = [json.loads(p.read_text()) for p in args.out.glob("*.json") if p.stem != "summary"]
    summary = {
        "cadence_version": cadence.__version__,
        "numpy_version": np.__version__,
        "sources": digests(),
        "runs": len(rows),
        "bits": args.bits,
        "batch": BATCH,
        "gru": gru,
        "gate": GATE,
        "table": table(rows),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary["table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
