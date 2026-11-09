"""The crossing chamber: a question whose fact depends on an earlier context (cadence#267).

The answer is the value of the fact ``c xor k``: neither the context ``c`` nor the question ``k``
names the fact on its own, and a held read's scores add over what it reads (the event, the cortex
of the moment before), so no held address built from main effects selects the fact. The rows
(``keyed``) use the grammar chamber's words on its brain. A stream witnesses two records, a fact's
name and its value (``dog V0``, ``cat V1``: the names fixed, the values drawn from four nouns and
unequal, the order drawn per stream so that a position names no fact), then the context ``X_c``
(an open bracket) and the question ``Q_k`` (a close bracket). With ``think`` the answer comes at a
``GO`` moment after the question: the crossing can form in the cortex at the question and a held
address read it from there, the second moment a read on a read takes. Without it the question is
answered at its own moment. A third of the taught rows ask a fact by its name in place of the
question, which teaches the association to say what the read holds. A reader that ignores the
crossing says one of the two values: .50.

Each life teaches ``batches`` batches of sixteen rows, one lesson a batch on the answer moment
before the greedy act on it, and reads at its ``checkpoints`` and at the end, on fresh rows: the
answers (``right``, and ``other_value``, the share answering the other fact's value), the read's
attention on the fact asked for and on the other (a fact's name and value slots together:
``relevant``, ``other``), the facts asked by name (``direct``), and the controls (the fact asked
for erased from the slots before the answer, the other fact erased, every slot erased).

``decode`` reads where the conjunction stands at the moment the answer's read is addressed from
(the cortex of the question's moment with ``think``, of the context's moment without): a ridge
decoder of ``c``, ``k`` and ``c xor k`` fit on half of a set of fresh rows and read on the other
half, and the crossing term of the four cells' means (``|m00 - m01 - m10 + m11| / 4``) against the
spread of the rows around their cell's mean. A conjunction the cortex holds linearly can be
selected by a held address on it; one decoded at chance cannot.

Arms (``ARMS``) are settings of the grammar chamber's ``recipe`` brain; ``Fd`` is the read's rule
at the deep teaching point. ``support`` names what the harness does to a life to locate a term,
no option of the library:

- ``frozen_cortex``: the synapses into the association and prefrontal regions from anything but
  the read's ports, and those regions' biases, are restored after every lesson, so the lessons
  move the readout, the read's synapses and the read's maps alone;
- ``cortex_unit``: the maps read the cortex of the moment before at unit norm per stream.

The matched transformer and LSTM (``run_control``) hear the same rows, one Adam step per batch,
and are read on the same fresh rows; independently scored controls, never part of a brain's
answer path.

    PYTHONPATH=src python benchmarks/attention/crossing.py --arms Fd --think --founders 0 1 2 \\
        --batches 3000 --checkpoints 1000 --out DIR/keyed.jsonl --workers 3
    PYTHONPATH=src python benchmarks/attention/crossing.py --table DIR/keyed.jsonl
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
import warnings
from dataclasses import dataclass
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


gr = _module("grammar", HERE.parent / "grammar" / "grammar.py")

ARMS: dict[str, dict[str, Any]] = {
    "recipe": {},
    "Fd": {"teaching": "deep"},
}
SUPPORTS = ("frozen_cortex", "cortex_unit")
FOUNDERS = (0, 1, 2)
BATCH = 16
SEED = 228
NAMES = (0, 1)  # the facts' names, ``dog`` and ``cat``: a record's key
VALUES = (2, 3, 6, 7)  # the words a fact can hold: ``fox owl foxes owls``
DIRECT = 3  # one taught row in this many asks a fact by its name
CONTROLS = ("erase_relevant", "erase_other", "memory_off")
FROZEN_REGIONS = ("association", "prefrontal")
READ_PORTS = ("recall", "next")


@dataclass
class Ask:
    """One stream of a keyed batch: the heard moments, the question and its answer."""

    pre: list[np.ndarray]
    question: np.ndarray
    answer: int
    c: int
    k: int  # the question's bit; -1 for a fact asked by its name
    fact: int  # the fact asked for
    values: tuple[int, int]  # what fact 0 and fact 1 hold
    first: int  # the fact witnessed first
    think: bool = False  # the answer comes at a moment after the question


def keyed(
    rng: np.random.Generator, n: int = BATCH, direct: float = 0.0, think: bool = False
) -> list[Ask]:
    """``n`` streams of the crossing; with ``direct`` that share asks a fact by its name."""
    rows = []
    for _ in range(n):
        c, k = int(rng.integers(2)), int(rng.integers(2))
        drawn = rng.permutation(VALUES)[:2]
        values = (int(drawn[0]), int(drawn[1]))
        first = int(rng.integers(2))
        order = (first, 1 - first)
        pre = [gr.heard(word) for fact in order for word in (NAMES[fact], values[fact])]
        pre.append(gr.heard(gr.OPENS[c]))
        if rng.random() < direct:
            fact = int(rng.integers(2))
            rows.append(
                Ask(pre, gr.heard(NAMES[fact]), values[fact], c, -1, fact, values, first, think)
            )
        else:
            fact = c ^ k
            question = gr.heard(gr.MATCH[gr.OPENS[k]])
            rows.append(Ask(pre, question, values[fact], c, k, fact, values, first, think))
    return rows


def new_work() -> dict[str, float]:
    return dict.fromkeys(("lessons", "rows", "lesson_row_sweeps", "acts", "act_sweeps",
                          "refused_lessons", "refused_acts"), 0.0)


# ------------------------------------------------------------------------------- the supports


class Support:
    """What the harness does to a life to locate a term (``SUPPORTS``)."""

    def __init__(self, brain: Any, names: Any = ()) -> None:
        self.names = tuple(names)
        if not set(self.names) <= set(SUPPORTS):
            raise ValueError(f"unknown support: {self.names}")
        self.mask: np.ndarray | None = None
        self.neurons: np.ndarray | None = None
        if "frozen_cortex" in self.names:
            populations = brain.connectome.populations
            inside = np.zeros(brain.connectome.n, dtype=bool)
            for region in FROZEN_REGIONS:
                inside[np.asarray(populations[region])] = True
            ports = np.zeros(brain.connectome.n, dtype=bool)
            for region in READ_PORTS:
                if region in populations:
                    ports[np.asarray(populations[region])] = True
            pre, post = brain.connectome.pre, brain.connectome.post
            self.mask = inside[post] & ~ports[pre]
            self.neurons = inside

    def before(self, brain: Any) -> None:
        """Before a lesson or an act: what the maps read."""
        if "cortex_unit" in self.names and brain.recall is not None:
            raw = np.atleast_2d(np.asarray(brain.recall.previous, dtype=float))
            norm = np.linalg.norm(raw, axis=1, keepdims=True)
            brain.recall.previous = np.where(norm > 0, raw / np.where(norm > 0, norm, 1.0), raw)

    def lesson(self, brain: Any, x: np.ndarray, labels: np.ndarray) -> Any:
        """A lesson, with what a frozen cortex keeps restored after it."""
        self.before(brain)
        if self.mask is None:
            return brain.lesson(x, labels)
        graph = brain.learner.brain
        efficacy, bias = np.array(graph.efficacy), np.array(graph.bias)
        try:
            return brain.lesson(x, labels)
        finally:
            after = brain.learner.brain
            kept = np.where(self.mask, efficacy, np.asarray(after.efficacy))
            kept_bias = np.where(self.neurons, bias, np.asarray(after.bias))
            brain.learner.brain = after.with_parameters(efficacy=kept, bias=kept_bias)


def make_brain(arm: str, founder: int, settings: dict[str, Any] | None = None) -> Any:
    return gr.make_brain("recipe", founder, {**ARMS[arm], **(settings or {})})


def fact_slots(brain: Any, row: Ask) -> tuple[int, int]:
    """The slots of the names of fact 0 and fact 1: the boundary, then the records in order."""
    start = 1 if brain.recall.boundary else 0
    return (start, start + 2) if row.first == 0 else (start + 2, start)


def _act(brain: Any, x: np.ndarray, work: dict[str, float], support: Support) -> np.ndarray:
    support.before(brain)
    try:
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
    except RuntimeError:
        work["refused_acts"] += 1
        return np.full(len(x), -1)
    work["acts"] += 1
    work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
    return action


def bring(brain: Any, rows: list[Ask], work: dict[str, float], support: Support) -> np.ndarray:
    """Fresh streams brought to their answer: the records and the context heard, and with
    ``think`` the question too. Returns the moment that is answered."""
    brain.reset()
    for t in range(len(rows[0].pre)):
        _act(brain, np.stack([r.pre[t] for r in rows]), work, support)
    question = np.stack([r.question for r in rows])
    if not rows[0].think:
        return question
    _act(brain, question, work, support)
    return np.tile(gr.concept("GO"), (len(rows), 1))


def teach(brain: Any, rows: list[Ask], work: dict[str, float], support: Support) -> None:
    """One taught batch: a lesson on the answer moment before the greedy act on it."""
    answered = bring(brain, rows, work, support)
    try:
        report = support.lesson(brain, answered, np.array([r.answer for r in rows]))
        work["lessons"] += 1
        work["rows"] += report.get("attempted_presentations", 0.0)
        work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
    except RuntimeError:
        work["refused_lessons"] += 1
    _act(brain, answered, work, support)


def read_rows(brain: Any, rows: list[Ask], support: Support,
              control: str | None = None) -> dict[str, list[Any]]:
    """The rows answered on a fork of ``brain``, one list per reading, one entry per row."""
    fork = copy.deepcopy(brain)
    recall = fork.recall
    work = new_work()
    answered = bring(fork, rows, work, support)
    if control is not None:
        keys, values, nexts = recall.keys.copy(), recall.values.copy(), recall.nexts.copy()
        for i, r in enumerate(rows):
            if control == "memory_off":
                slots = list(range(recall.slots))
            else:
                fact = r.fact if control == "erase_relevant" else 1 - r.fact
                name = fact_slots(fork, r)[fact]
                slots = [name, name + 1]  # the name, whose next holds the value, and the value
            for slot in slots:
                keys[i, slot], values[i, slot], nexts[i, slot] = 0.0, 0.0, 0.0
        recall.keys, recall.values, recall.nexts = keys, values, nexts
    answer = _act(fork, answered, work, support)
    attended = recall.attended_heads[0] if recall.attended_heads else None
    out: dict[str, list[Any]] = {k: [] for k in ("right", "other_value", "relevant", "other",
                                                 "pair")}
    for i, r in enumerate(rows):
        out["right"].append(bool(answer[i] == r.answer))
        out["other_value"].append(bool(answer[i] == r.values[1 - r.fact]))
        names = fact_slots(fork, r)
        mass = (0.0, 0.0) if attended is None else (
            float(attended[i, names[0]] + attended[i, names[0] + 1]),
            float(attended[i, names[1]] + attended[i, names[1] + 1]),
        )
        out["relevant"].append(mass[r.fact])
        out["other"].append(mass[1 - r.fact])
        out["pair"].append(f"c{r.c}k{r.k}")
    return out


def panel(brain: Any, batches: list[list[Ask]], direct: list[list[Ask]], support: Support,
          controls: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for control in (None, *(CONTROLS if controls else ())):
        reads = [read_rows(brain, rows, support, control) for rows in batches]
        summary: dict[str, Any] = {
            key: round(float(np.mean([v for r in reads for v in r[key]])), 4)
            for key in ("right", "other_value", "relevant", "other")
        }
        pairs: dict[str, list[bool]] = {}
        for r in reads:
            for pair, right in zip(r["pair"], r["right"], strict=True):
                pairs.setdefault(pair, []).append(right)
        summary["conditions"] = {k: round(float(np.mean(v)), 4) for k, v in sorted(pairs.items())}
        out["intact" if control is None else control] = summary
    reads = [read_rows(brain, rows, support) for rows in direct]
    out["direct"] = {key: round(float(np.mean([v for r in reads for v in r[key]])), 4)
                     for key in ("right", "relevant")}
    return out


# ----------------------------------------------------------------------------- the decoder


def cortex_at_address(brain: Any, rows: list[Ask], support: Support) -> np.ndarray:
    """The cortex the answer's read is addressed from, per row: the association of the moment
    before the answer, as the maps read it."""
    fork = copy.deepcopy(brain)
    answered = bring(fork, rows, new_work(), support)
    del answered
    support.before(fork)
    return np.array(fork.recall.previous, dtype=float)


def _ridge_accuracy(x: np.ndarray, y: np.ndarray, half: int, alpha: float = 1e-2) -> float:
    mean = x[:half].mean(axis=0)
    a = np.hstack([x - mean, np.ones((len(x), 1))])
    train, test = a[:half], a[half:]
    w = np.linalg.solve(train.T @ train + alpha * np.eye(a.shape[1]), train.T @ y[:half])
    return float(np.mean(np.sign(test @ w) == y[half:]))


def decode(brain: Any, rng: np.random.Generator, support: Support, think: bool,
           batches: int = 32) -> dict[str, float]:
    """Where the conjunction stands at the cortex the answer's read is addressed from."""
    rows = [r for _ in range(batches) for r in keyed(rng, think=think)]
    cortex = np.vstack([cortex_at_address(brain, rows[i:i + BATCH], support)
                        for i in range(0, len(rows), BATCH)])
    c = np.array([r.c for r in rows])
    k = np.array([r.k for r in rows])
    half = len(rows) // 2
    out = {name: round(_ridge_accuracy(cortex, 2.0 * bit - 1.0, half), 4)
           for name, bit in (("c", c), ("k", k), ("crossing", c ^ k))}
    means = {(i, j): cortex[(c == i) & (k == j)].mean(axis=0) for i in (0, 1) for j in (0, 1)}
    cross = (means[0, 0] - means[0, 1] - means[1, 0] + means[1, 1]) / 4.0
    cells = np.array([means[int(ci), int(kj)] for ci, kj in zip(c, k, strict=True)])
    out["crossing_term"] = round(float(np.linalg.norm(cross)), 4)
    out["c_term"] = round(float(np.linalg.norm(
        (means[1, 0] + means[1, 1] - means[0, 0] - means[0, 1]) / 4.0)), 4)
    out["k_term"] = round(float(np.linalg.norm(
        (means[0, 1] + means[1, 1] - means[0, 0] - means[1, 0]) / 4.0)), 4)
    out["spread"] = round(float(np.linalg.norm(cortex - cells, axis=1).mean()), 4)
    out["energy"] = round(float(np.linalg.norm(cortex, axis=1).mean()), 4)
    return out


# ----------------------------------------------------------------------------- one life


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One brain's life: ``arm``, ``founder``, ``batches``, ``checkpoints``, ``settings`` (over the
    arm's), ``support``, ``think``, ``controls``, ``panel_batches``, ``decode``; ``init_seed``
    (the brain's seed, on the founder's rows)."""
    warnings.simplefilter("ignore")
    import cadence as cd

    arm, founder = spec["arm"], int(spec["founder"])
    batches = int(spec.get("batches", 3000))
    checkpoints = {int(c) for c in spec.get("checkpoints", [])}
    think = bool(spec.get("think", True))
    started = time.monotonic()
    brain = make_brain(arm, int(spec.get("init_seed", founder)), spec.get("settings"))
    support = Support(brain, spec.get("support", ()))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"], SEED, 1]))
    test = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["test"], SEED, 1]))
    panels = int(spec.get("panel_batches", 8))
    crossing_rows = [keyed(test, think=think) for _ in range(panels)]
    direct_rows = [keyed(test, direct=1.0, think=think) for _ in range(max(panels // 4, 1))]
    decoding = bool(spec.get("decode", True))

    def reading(b: int, controls: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "batches": b, "work": {**work, "seconds": round(time.monotonic() - started, 1)},
            "panels": panel(brain, crossing_rows, direct_rows, support, controls)}
        if decoding:
            probe = np.random.default_rng(np.random.SeedSequence([founder, 228, 5]))
            out["decode"] = decode(brain, probe, support, think)
        return out

    work = new_work()
    readings = [reading(0, False)]
    for b in range(1, batches + 1):
        teach(brain, keyed(rng, direct=1.0 / DIRECT, think=think), work, support)
        if b in checkpoints or b == batches:
            readings.append(reading(b, bool(spec.get("controls", True)) and b == batches))
    return {
        "rows": "keyed", "arm": arm, "founder": founder, "think": think,
        "init_seed": int(spec.get("init_seed", founder)), "tag": spec.get("tag"),
        "settings": {**ARMS[arm], **spec.get("settings", {})},
        "support": list(support.names), "batches": batches,
        "parameters": int(brain.parameters()), "cadence": cd.__version__,
        "readings": readings, "seconds": round(time.monotonic() - started, 1),
    }


def run_control(spec: dict[str, Any]) -> dict[str, Any]:
    """One online life of a matched control (``model``: ``transformer`` or ``lstm``) on the same
    rows, its width the nearest to the brain's parameters, one Adam step per batch at the grammar
    chamber's online rate, read by the first word it says on the same fresh rows."""
    import torch

    ct = _module("grammar_controls", HERE.parent / "grammar" / "controls.py")
    torch.set_num_threads(1)
    model, founder = spec["model"], int(spec["founder"])
    batches = int(spec.get("batches", 3000))
    checkpoints = {int(c) for c in spec.get("checkpoints", [])}
    think = bool(spec.get("think", True))
    started = time.monotonic()
    target = int(make_brain("Fd", founder).parameters())
    width = ct.matched_width(model, target)
    learner = ct.Learner(model, width, founder, float(spec.get("rate", ct.RATES["online"])))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"], SEED, 1]))
    test = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["test"], SEED, 1]))
    panels = int(spec.get("panel_batches", 8))
    crossing_rows = [keyed(test, think=think) for _ in range(panels)]

    def rows_of(asks: list[Ask]) -> list[Any]:
        return [gr.Row("keyed", [*a.pre, a.question], [a.answer, gr.END]) for a in asks]

    def score(batches_: list[list[Ask]]) -> float:
        right = []
        learner.net.eval()
        for asks in batches_:
            for a, said in zip(asks, learner.speak(rows_of(asks)), strict=True):
                right.append(bool(said) and said[0] == a.answer)
        learner.net.train()
        return round(float(np.mean(right)), 4)

    readings = []
    for b in range(1, batches + 1):
        learner.update(rows_of(keyed(rng, direct=1.0 / DIRECT, think=think)))
        if b in checkpoints or b == batches:
            readings.append({"batches": b, "right": score(crossing_rows),
                             "seconds": round(time.monotonic() - started, 1)})
    return {
        "rows": "keyed_control", "model": model, "founder": founder, "width": width,
        "parameters": int(ct.count(learner.net)), "brain_parameters": target,
        "batches": batches, "readings": readings, "seconds": round(time.monotonic() - started, 1),
    }


def run_any(spec: dict[str, Any]) -> dict[str, Any]:
    return run_control(spec) if "model" in spec else run_life(spec)


def table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| arm | support | founder | batches | right | other value | relevant | other | "
             "direct | erase relevant | erase other | memory off | decode c, k, crossing | "
             "crossing term / spread | seconds |", "|" + " --- |" * 15]
    for r in sorted((r for r in rows if r.get("rows") == "keyed"),
                    key=lambda r: (r["arm"], r["support"], r["founder"])):
        last = r["readings"][-1]
        p = last["panels"]
        intact = p["intact"]
        d = last.get("decode", {})

        def cell(name: str, panels: dict[str, Any] = p) -> str:
            return f"{panels[name]['right']:.2f}" if name in panels else ""

        lines.append(
            f"| {r['arm']} | {'+'.join(r['support']) or '-'} | {r['founder']} | "
            f"{last['batches']} | {intact['right']:.2f} | {intact['other_value']:.2f} | "
            f"{intact['relevant']:.2f} | {intact['other']:.2f} | {p['direct']['right']:.2f} | "
            f"{cell('erase_relevant')} | {cell('erase_other')} | {cell('memory_off')} | "
            f"{d.get('c', float('nan')):.2f} {d.get('k', float('nan')):.2f} "
            f"{d.get('crossing', float('nan')):.2f} | "
            f"{d.get('crossing_term', float('nan')):.3f} / {d.get('spread', float('nan')):.2f} | "
            f"{r['seconds']:.0f} |")
    for r in sorted((r for r in rows if r.get("rows") == "keyed_control"),
                    key=lambda r: (r["model"], r["founder"])):
        lines.append(f"| {r['model']} | | {r['founder']} | " + ", ".join(
            f"{x['batches']}: {x['right']:.2f}" for x in r["readings"]) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--arms", nargs="*", default=["Fd"], choices=list(ARMS))
    parser.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--batches", type=int, default=3000)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[])
    parser.add_argument("--settings", type=json.loads, default={}, help="over the arms'")
    parser.add_argument("--support", nargs="*", default=[], choices=SUPPORTS)
    parser.add_argument("--own-moment", action="store_true", help="answer at the question")
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path, help="JSON lines, one row per life, appended")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--table", type=Path, help="print the table of a results file")
    args = parser.parse_args(argv)
    if args.table:
        rows = [json.loads(line) for line in args.table.read_text().splitlines() if line.strip()]
        print("\n".join(table(rows)))
        return 0
    if args.out is None:
        parser.error("--out is required")
    if args.plan:
        specs = json.loads(args.plan.read_text())
    else:
        specs = [{"arm": arm, "founder": f, "batches": args.batches,
                  "checkpoints": args.checkpoints, "settings": args.settings,
                  "support": args.support, "think": not args.own_moment,
                  "tag": f"{arm}/{'+'.join(args.support)}/{f}"}
                 for arm in args.arms for f in args.founders]
    done = set()
    if args.out.exists():
        done = {json.loads(line).get("tag") for line in args.out.read_text().splitlines() if line}
    specs = [s for s in specs if s.get("tag") is None or s.get("tag") not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink, get_context("spawn").Pool(
            max(args.workers, 1), maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(run_any, specs, chunksize=1):
            sink.write(json.dumps(row) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
