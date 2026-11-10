"""Answers on tokens no lesson used (R1 of the attention tracker, cadence#272).

Six of the battery's tasks (``battery.py``) with some tokens held out of every answer in
training: ``lookback``, ``induction``, ``nback`` and ``twohop`` hear the held tokens in training and
never answer them (``unanswered``), ``repeat`` (copying unique tokens, closed loop) and
``periodic`` never hear them in training. Tests draw from every token, and a moment whose answer
is a held token is scored under ``held``, one whose answer is taught under ``taught``. A held
answer can only be said through the read: a learned output has never been raised for it.

Each answer moment is read beside the read's own prediction: every port's read of the slots'
values and successors as the act used it, and the token it drives most (``read_best`` is the port
that names the answer most often). With ``votes`` each scored moment also reads what drives the
answer's motor neuron and the neuron the brain chose: the association's synapses, the read's
learned gains (``recall_gain``) and the rest (the motor's own synapses and its bias), so a held
answer the read names and the brain does not is located in the vote.

Arms (``ARMS``) are settings of ``Brain.compose(..., recall=True)``, the default composition:
``default`` and ``gain`` (``recall_gain=1.0`` on every token). ``support`` names what the harness
does to a life to locate a term, no option of the library: ``no_association_vote`` reads the tests
with the association's synapses onto the tokens' motor neurons at zero, so the read and its gains
vote alone. ``gain_start`` in a life's settings is an emulation: every port's gain at that value
at birth in place of zero.

    PYTHONPATH=src python benchmarks/sequence/held.py --tasks nback lookback --arms gain \\
        --founders 0 1 2 --out DIR/held.jsonl --workers 6
    PYTHONPATH=src python benchmarks/sequence/held.py --table DIR/held.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = _module("sequence_battery", HERE / "battery.py")

# the six held tasks of #262's confirmation, at its lengths
TASKS: dict[str, tuple[str, dict[str, Any]]] = {
    "lookback": ("lookback", {"lengths": [5, 12], "test_lengths": [24, 60], "held": [3],
                              "held_mode": "unanswered"}),
    "induction": ("induction", {"lengths": [8, 16], "test_lengths": [24, 60], "held": [6, 7],
                                "held_mode": "unanswered"}),
    "nback": ("nback", {"lengths": [6, 12], "test_lengths": [24, 60], "held": [6, 7],
                        "held_mode": "unanswered"}),
    "twohop": ("twohop", {"think": True, "held": [12, 13, 14, 15], "held_mode": "unanswered"}),
    "copying": ("repeat", {"unique": True, "lengths": [3, 6], "test_lengths": [8, 12],
                           "held": [12, 13, 14, 15]}),
    "periodic": ("periodic", {"held": [5]}),
}
ARMS: dict[str, dict[str, Any]] = {
    "default": {},
    "gain": {"recall_gain": 1.0},
}
SUPPORTS = ("no_association_vote",)
FOUNDERS = (0, 1, 2)


# ------------------------------------------------------------------------ held-out episodes


def tokens(task: Any, train: bool) -> np.ndarray:
    held = set(task.held)
    full = np.arange(task.vocab)
    return np.array([t for t in full if t not in held]) if train else full


def draw(rng: np.random.Generator, toks: np.ndarray, size: Any) -> np.ndarray:
    return toks[rng.integers(len(toks), size=size)]


def heard(task: Any, train: bool) -> np.ndarray:
    """The tokens an episode may hear: every token where held tokens are heard in training
    (``unanswered``), else the answer tokens."""
    return np.arange(task.vocab) if task.held_mode == "unanswered" else tokens(task, train)


def ep_lookback(self: Any, rng: np.random.Generator, train: bool) -> list[Any]:
    lo, hi = self.lengths if train else self.test_lengths
    length = int(rng.integers(lo, hi + 1))
    seq = draw(rng, heard(self, train), (B.BATCH, length))
    c = rng.integers(len(self.cues), size=B.BATCH)
    where = {"first": 0, "last": length - 1, "second": 1}
    pos = np.array([where[self.cues[i]] for i in c])
    seq[np.arange(B.BATCH), pos] = draw(rng, tokens(self, train), B.BATCH)
    moments = [B.Moment(B.onehot(seq[:, t], self.inputs)) for t in range(length)]
    label = seq[np.arange(B.BATCH), pos]
    moments.append(B.Moment(B.onehot(self.vocab + c, self.inputs), label, length - 1 - pos,
                            [f"{self.cues[i]}" for i in c]))
    return moments


def ep_nback(self: Any, rng: np.random.Generator, train: bool) -> list[Any]:
    lo, hi = self.lengths if train else getattr(self, "test_lengths", self.lengths)
    length = int(rng.integers(lo, hi + 1))
    seq = draw(rng, heard(self, train), (B.BATCH, length))
    k_index = rng.integers(len(self.ks), size=B.BATCH)
    ks = np.array(self.ks)[k_index]
    seq[np.arange(B.BATCH), length - ks] = draw(rng, tokens(self, train), B.BATCH)
    moments = [B.Moment(B.onehot(seq[:, t], self.inputs)) for t in range(length)]
    label = seq[np.arange(B.BATCH), length - ks]
    moments.append(B.Moment(B.onehot(self.vocab + k_index, self.inputs), label, ks - 1,
                            [f"k{k}" for k in ks]))
    return moments


def ep_induction(self: Any, rng: np.random.Generator, train: bool) -> list[Any]:
    lo, hi = self.lengths if train else self.test_lengths
    length = int(rng.integers(lo, hi + 1))
    toks = heard(self, train)
    answers = tokens(self, train)
    rows, ages = [], np.zeros(B.BATCH, dtype=int)
    for r in range(B.BATCH):
        a = int(rng.choice(toks))
        others = [t for t in toks if t != a]
        seq = list(rng.choice(others, size=length - 1))
        i = int(rng.integers(0, length - 1 - self.gap_min))
        seq[i] = a
        seq[i + 1] = int(rng.choice([t for t in answers if t != a]))
        seq.append(a)
        rows.append(seq)
        ages[r] = (length - 1) - 1 - i
    rows_a = np.array(rows)
    moments = [B.Moment(B.onehot(rows_a[:, t], self.inputs)) for t in range(length - 1)]
    label = np.array([rows_a[r][np.where(rows_a[r][:-1] == rows_a[r][-1])[0][0] + 1]
                      for r in range(B.BATCH)])
    moments.append(B.Moment(B.onehot(rows_a[:, -1], self.inputs), label, ages - 1,
                            [f"L{length}"] * B.BATCH))
    return moments


def ep_repeat(self: Any, rng: np.random.Generator, train: bool) -> list[Any]:
    lo, hi = self.lengths if train else self.test_lengths
    length = int(rng.integers(lo, hi + 1))
    toks = tokens(self, train)
    if self.unique:
        seq = np.stack([toks[rng.permutation(len(toks))[:length]] for _ in range(B.BATCH)])
    else:
        seq = draw(rng, toks, (B.BATCH, length))
    moments = [B.Moment(B.onehot(np.full(B.BATCH, self.START), self.inputs))]
    moments += [B.Moment(B.onehot(seq[:, t], self.inputs)) for t in range(length)]
    moments.append(B.Moment(B.onehot(np.full(B.BATCH, self.GO), self.inputs), seq[:, 0],
                            np.full(B.BATCH, length - 1), [f"L{length}:p0"] * B.BATCH))
    for j in range(length - 1):
        moments.append(B.Moment(B.onehot(seq[:, j], self.inputs), seq[:, j + 1],
                                np.full(B.BATCH, length - 1), [f"L{length}:p{j + 1}"] * B.BATCH,
                                feedback=True))
    return moments


def ep_twohop(self: Any, rng: np.random.Generator, train: bool) -> list[Any]:
    n = self.pairs
    toks = tokens(self, train)
    hear = heard(self, train)
    firsts = np.zeros((B.BATCH, n), dtype=int)
    seconds = np.zeros((B.BATCH, n), dtype=int)
    a = np.zeros(B.BATCH, dtype=int)
    b = np.zeros(B.BATCH, dtype=int)
    c = np.zeros(B.BATCH, dtype=int)
    for r in range(B.BATCH):
        if self.held_mode == "unanswered":
            bc = toks[rng.permutation(len(toks))[:2]]
            rest = [t for t in hear if t not in bc]
            rest = list(np.array(rest)[rng.permutation(len(rest))])
            perm = np.array([rest[0], bc[0], bc[1]] + rest[1:])
        else:
            perm = toks[rng.permutation(len(toks))]
        a[r], b[r], c[r] = perm[:3]
        fs = [a[r], b[r]] + list(perm[3:3 + n - 2])
        ss = [b[r], c[r]] + list(draw(rng, hear, n - 2))
        order = rng.permutation(n)
        firsts[r], seconds[r] = np.array(fs)[order], np.array(ss)[order]
    moments = []
    for i in range(n):
        moments.append(B.Moment(B.onehot(firsts[:, i], self.inputs)))
        moments.append(B.Moment(B.onehot(seconds[:, i], self.inputs)))
    moments.append(B.Moment(B.onehot(a, self.inputs), b, None, ["hop1"] * B.BATCH))
    moments.append(B.Moment(B.onehot(np.full(B.BATCH, self.THINK), self.inputs), c, None,
                            ["hop2"] * B.BATCH))
    return moments


def ep_periodic(self: Any, rng: np.random.Generator, train: bool) -> list[Any]:
    toks = tokens(self, train)
    p = int(rng.choice(self.periods))
    motif = draw(rng, toks, (B.BATCH, p))
    length = p * self.repeats
    seq = np.tile(motif, (1, self.repeats))
    vary = rng.random((B.BATCH, length)) < self.variation
    vary[:, :p] = False
    seq = np.where(vary, draw(rng, toks, (B.BATCH, length)), seq)
    target = np.tile(motif, (1, self.repeats + 1))
    moments = []
    for t in range(length):
        if t < p:
            moments.append(B.Moment(B.onehot(seq[:, t], self.inputs)))
            continue
        moments.append(B.Moment(B.onehot(seq[:, t], self.inputs), target[:, t + 1],
                                np.full(B.BATCH, p - 2), [f"P{p}"] * B.BATCH))
    return moments


EPISODES = {"lookback": ep_lookback, "nback": ep_nback, "induction": ep_induction,
            "repeat": ep_repeat, "twohop": ep_twohop, "periodic": ep_periodic}


def make_task(name: str) -> Any:
    """The battery's task with its held tokens."""
    kind, args = TASKS[name]
    args = dict(args)
    held = tuple(args.pop("held"))
    mode = args.pop("held_mode", "unheard")
    longer = args.pop("test_lengths", None) if kind == "nback" else None
    task = B.TASKS[kind](**args)
    if longer:  # n-back's tests at lengths no lesson reached
        task.test_lengths = tuple(longer)
        task.settings["test_lengths"] = list(longer)
    task.held, task.held_mode = held, mode
    task.settings.update(held=list(held), held_mode=mode)
    task.episode = EPISODES[kind].__get__(task)
    return task


def make_brain(task: Any, arm: str, founder: int, settings: dict[str, Any] | None = None,
               module: int = B.MODULE) -> Any:
    """The default composition for ``task``, each action ``j`` the token of observation ``j``."""
    import cadence as cd

    options = {"backend": "cpu", "recall": True, **ARMS[arm], **(settings or {})}
    start = options.pop("gain_start", None)  # an emulation: every port's gain at birth
    if options.get("recall_gain") and "recall_tokens" not in options:
        options["recall_tokens"] = list(range(task.actions))
    brain = cd.Brain.compose(task.inputs, task.actions, modules=(module,), seed=int(founder),
                             **options)
    if start is not None:
        brain.recall.gains = np.full_like(np.asarray(brain.recall.gains), float(start))
    return brain


# ------------------------------------------------------------------------------ the reading


def silence_association_vote(brain: Any, actions: int) -> None:
    """The association's synapses onto the tokens' motor neurons at zero (a support)."""
    graph = brain.connectome
    pre, post = np.asarray(graph.pre), np.asarray(graph.post)
    motor = np.asarray(graph.populations["motor"])[:actions]
    association = np.asarray(graph.populations["association"])
    cut = np.isin(pre, association) & np.isin(post, motor)
    learner = brain.learner
    efficacy = np.where(cut, 0.0, np.asarray(learner.brain.efficacy))
    learner.brain = learner.brain.with_parameters(efficacy=efficacy)


def motor_votes(brain: Any, neurons: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """What drives each named motor neuron per row at the act just settled: the association's
    synapses, the gains' drive and the rest (the motor's other synapses and its bias)."""
    graph = brain.connectome
    pre, post = np.asarray(graph.pre), np.asarray(graph.post)
    association = np.zeros(graph.n, dtype=bool)
    association[np.asarray(graph.populations["association"])] = True
    weights = np.asarray(brain.brain.weights)
    activation = np.atleast_2d(np.asarray(brain.basal_ganglia.state.activation))
    drive = np.atleast_2d(np.asarray(brain.basal_ganglia._drive))
    bias = np.asarray(brain.brain.bias)
    out: dict[str, np.ndarray] = {}
    rows = np.arange(len(activation))
    for name, neuron in neurons.items():
        cortex, rest = np.zeros(len(rows)), np.zeros(len(rows))
        for r in rows:
            into = post == neuron[r]
            parts = weights[into] * activation[r, pre[into]]
            cortex[r] = parts[association[pre[into]]].sum()
            rest[r] = parts[~association[pre[into]]].sum() + bias[neuron[r]]
        out[f"{name}_association"] = cortex
        out[f"{name}_gain"] = drive[rows, neuron[rows]]
        out[f"{name}_rest"] = rest
    return out


def run_episode(brain: Any, task: Any, moments: list[Any], teach: bool, closed: bool,
                stats: dict[str, Any], votes: bool = False) -> None:
    """The battery's ``run_episode``, with the read's prediction (and the vote) at scored
    moments."""
    brain.reset()
    previous = None
    held = set(task.held)
    vocab = task.vocab
    motor = np.asarray(brain.connectome.populations["motor"])
    for m in moments:
        x = m.x
        if m.feedback and closed and previous is not None:
            x = B.onehot(previous, task.inputs)
        if teach and m.label is not None:
            try:
                brain.lesson(x, m.label)
            except RuntimeError:
                stats["refused"] += 1
        scored = not teach and m.label is not None
        if scored:
            values = np.array(brain.recall.values, copy=True)
            nexts = np.array(brain.recall.nexts, copy=True)
        try:
            action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
        except RuntimeError:
            stats["exhausted"] += 1
            action = np.full(len(x), -1)
        if scored:
            hit = action == m.label
            conds = [("held" if int(m.label[r]) in held else "taught") for r in range(len(x))]
            heads = brain.recall.attended_heads or ()
            reads = {}
            for h, p in enumerate(heads):
                if p is None or p.shape[1] != values.shape[1]:
                    continue
                reads[f"recall{h}"] = np.argmax(np.einsum("bs,bsw->bw", p, values)[:, :vocab], 1)
                reads[f"next{h}"] = np.argmax(np.einsum("bs,bsw->bw", p, nexts)[:, :vocab], 1)
            voted = None
            if votes:
                chosen = np.where(action >= 0, action, 0)
                voted = motor_votes(brain, {"answer": motor[m.label], "chosen": motor[chosen]})
            for r, cond in enumerate(conds):
                cell = stats["cells"].setdefault(cond, {"n": 0, "answer": 0, "ports": {},
                                                        "right_wrong": {}, "votes": {}})
                cell["n"] += 1
                cell["answer"] += int(hit[r])
                for port, pred in reads.items():
                    ok = int(pred[r] == m.label[r])
                    cell["ports"][port] = cell["ports"].get(port, 0) + ok
                    cell["right_wrong"][port] = cell["right_wrong"].get(port, 0) + int(
                        ok and not hit[r])
                if voted is not None and not hit[r]:
                    for key, value in voted.items():
                        cell["votes"][key] = cell["votes"].get(key, 0.0) + float(value[r])
                    cell["votes"]["missed"] = cell["votes"].get("missed", 0) + 1
        previous = action if m.label is not None else previous


def evaluate(brain: Any, task: Any, rng: np.random.Generator, episodes: int, closed: bool,
             votes: bool = False) -> dict[str, Any]:
    stats: dict[str, Any] = {"cells": {}, "refused": 0, "exhausted": 0}
    for _ in range(episodes):
        run_episode(brain, task, task.episode(rng, train=False), teach=False, closed=closed,
                    stats=stats, votes=votes)
    out: dict[str, Any] = {"exhausted": stats["exhausted"]}
    for cond, cell in sorted(stats["cells"].items()):
        n = max(cell["n"], 1)
        ports = {k: v / n for k, v in cell["ports"].items()}
        best = max(ports, key=lambda k: ports[k]) if ports else None
        row: dict[str, Any] = {
            "n": cell["n"], "answer": round(cell["answer"] / n, 4), "best_port": best,
            "read_best": round(ports[best], 4) if best else None,
            "read_right_answer_wrong": round(cell["right_wrong"][best] / n, 4) if best else None,
        }
        missed = cell["votes"].pop("missed", 0)
        if missed:
            row["vote_missed"] = {k: round(v / missed, 3) for k, v in cell["votes"].items()}
            row["missed"] = missed
        out[cond] = row
    return out


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``task`` (a key of ``TASKS``), ``arm``, ``founder``, with ``settings`` over the
    arm's, ``lessons``, ``tests``, ``module``, ``support`` and ``votes``."""
    warnings.simplefilter("ignore")
    import copy

    import cadence as cd

    task = make_task(spec["task"])
    arm, founder = spec["arm"], int(spec["founder"])
    lessons, tests = int(spec.get("lessons", B.LESSONS)), int(spec.get("tests", B.TESTS))
    module = int(spec.get("module", B.MODULE))
    started = time.monotonic()
    brain = make_brain(task, arm, founder, spec.get("settings"), module)
    rng = np.random.default_rng(np.random.SeedSequence([founder, B.SEEDS["train"]]))
    stats: dict[str, Any] = {"cells": {}, "refused": 0, "exhausted": 0}
    for _ in range(lessons):
        run_episode(brain, task, task.episode(rng, train=True), teach=True, closed=False,
                    stats=stats)
    closed = task.name == "repeat"
    votes = bool(spec.get("votes", True))

    def test() -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([founder, B.SEEDS["test"]]))

    row: dict[str, Any] = {
        "task": spec["task"], "task_settings": task.settings, "arm": arm, "founder": founder,
        "settings": {**ARMS[arm], **spec.get("settings", {})}, "lessons": lessons,
        "tests": tests, "module": module, "cadence": cd.__version__, "tag": spec.get("tag"),
        "refused_lessons": stats["refused"], "exhausted_train": stats["exhausted"],
        "test": evaluate(copy.deepcopy(brain), task, test(), tests, closed, votes),
    }
    if brain.recall is not None and getattr(brain.recall, "gain", 0):
        row["gains"] = [round(float(g), 4) for g in brain.recall.gains]
    if "no_association_vote" in spec.get("support", ()):
        fork = copy.deepcopy(brain)
        silence_association_vote(fork, task.actions)
        row["no_association_vote"] = evaluate(fork, task, test(), tests, closed)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| task | arm | founder | held | read held | taught | association vote, answer / "
             "chosen | gain vote, answer / chosen | without the association's vote: held, "
             "taught |", "|" + " --- |" * 9]
    for r in sorted(rows, key=lambda r: (r["task"], r["arm"], r["founder"])):
        held, taught = r["test"].get("held", {}), r["test"].get("taught", {})
        v = held.get("vote_missed") or {}
        cut = r.get("no_association_vote", {})
        lines.append(
            f"| {r['task']} | {r['arm']} | {r['founder']} | {held.get('answer', float('nan')):.2f}"
            f" | {held.get('read_best') or 0:.2f} | {taught.get('answer', float('nan')):.2f} | "
            f"{v.get('answer_association', float('nan')):.2f} / "
            f"{v.get('chosen_association', float('nan')):.2f} | "
            f"{v.get('answer_gain', float('nan')):.2f} / {v.get('chosen_gain', float('nan')):.2f}"
            f" | {cut.get('held', {}).get('answer', float('nan')):.2f}, "
            f"{cut.get('taught', {}).get('answer', float('nan')):.2f} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--arms", nargs="*", default=["gain"], choices=list(ARMS))
    parser.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--settings", type=json.loads, default={}, help="over the arms'")
    parser.add_argument("--support", nargs="*", default=[], choices=SUPPORTS)
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--table", type=Path)
    args = parser.parse_args(argv)
    if args.table:
        rows = [json.loads(line) for line in args.table.read_text().splitlines() if line.strip()]
        print("\n".join(table(rows)))
        return 0
    if args.out is None:
        parser.error("--out is required")
    specs = json.loads(args.plan.read_text()) if args.plan else [
        {"task": t, "arm": a, "founder": f, "settings": args.settings, "support": args.support,
         "tag": f"{t}/{a}/{f}"} for t in args.tasks for a in args.arms for f in args.founders]
    done = set()
    if args.out.exists():
        done = {json.loads(line).get("tag") for line in args.out.read_text().splitlines() if line}
    specs = [s for s in specs if s.get("tag") is None or s.get("tag") not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink, get_context("spawn").Pool(
            max(args.workers, 1), maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(run_life, specs, chunksize=1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
