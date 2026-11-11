"""Acquisition controls for the named-fact selection instrument: can the same graph, taught
the same way, acquire AND, OR, XOR and the selection task itself when both facts are exposed?

A declared development comparison beside ``selective_recall.py``: same encoding, trial
structure, teaching path and work accounting; the target function of the two exposed facts
is the only change. Selection on the unequal-fact cases is XOR of the query bit and fact A,
so a graph that cannot acquire XOR cannot acquire selection. Width, depth and rounds vary as
declared comparisons; nothing here is a runtime change or a protocol of the instrument.

    PYTHONPATH=src python benchmarks/relevance/acquisition_controls.py --out DIR --workers 3
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import selective_recall as sr  # noqa: E402

TASKS = ("and", "or", "xor", "mux")
BRAINS = {"m32": [32], "m64": [64], "m32x16": [32, 16], "m64x32": [64, 32]}
ROUNDS = (128, 512, 2048)
FOUNDERS = (0, 1, 2, 3, 4, 5)
NEUTRAL_EVENTS = 1
HOLDOUT_SEED = 31016  # the held-out panel's draw, beside the schedule's 31014


def targets(task: str, rows: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """The answer the task requires for facts (A, B) and the query that names one of them."""
    a, b = rows[:, 0], rows[:, 1]
    if task == "mux":
        return rows[np.arange(len(rows)), queries].copy()
    if task == "and":
        return a & b
    if task == "or":
        return a | b
    if task == "xor":
        return a ^ b
    raise ValueError(task)


def train(
    brain: Any, task: str, order: np.ndarray, work: sr.Work, teaching: str = "every",
    exposed: bool = True, cases: np.ndarray | None = None,
) -> dict[str, Any]:
    """The instrument's training loop with the task's targets; both facts exposed at the query
    (``exposed``), or absent, the instrument's trace arm.

    ``teaching`` ``every`` is the instrument's: one lesson on every case of the round, then the
    free action. ``surprise`` is the recall chamber's rule: the complete pre-query stimulus of
    every row is taken first, with the trace the prefix built; the brain acts; the rows whose
    answer was wrong are taught from their rows of that stimulus, so a lesson never reads zero
    context from a smaller batch and never the trace the query's own settle advanced; a round
    without a wrong row has no lesson. ``cases`` are the training cases each round permutes,
    all sixteen when omitted (the held-out panel passes the twelve it trains on).
    """
    rounds = 0
    stop = None
    lessons = 0
    training = sr.CASES if cases is None else cases
    for permutation in order:
        rows = training[permutation]
        prefix, query, _ = sr.inputs(rows[:, :3], rows[:, 3], NEUTRAL_EVENTS, exposed=exposed)
        target = targets(task, rows, rows[:, 3])
        brain.reset()
        for observation in prefix:
            if work.act(brain, observation) is None:
                stop = "refused_prefix"
                break
        if stop is not None:
            break
        if teaching == "surprise":
            drive = brain.stimulus(query)  # every row's complete pre-query stimulus
            answer = work.act(brain, query)
            if answer is None:
                stop = "refused_query"
                break
            wrong = np.flatnonzero(np.asarray(answer).reshape(-1) != target)
            if len(wrong):
                teach_drive(work, brain, drive[wrong], target[wrong])
                lessons += 1
        else:
            work.teach(brain, query, target)
            lessons += 1
            if work.act(brain, query) is None:
                stop = "refused_query"
                break
        rounds += 1
    return {"completed": rounds == len(order), "rounds": rounds, "stop": stop, "lessons": lessons}


def teach_drive(work: sr.Work, brain: Any, drive: np.ndarray, target: np.ndarray) -> None:
    """The instrument's lesson accounting for a drive taken before the query's act (the recall
    chamber's ``teach(..., drive=)``): the rows of the complete stimulus, never a new one."""
    from cadence.learning import LearningPhaseError  # noqa: PLC0415

    work.lessons += 1
    try:
        _, report = brain.learner.step(drive, target)
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
        setattr(work, field, getattr(work, field) + int(report[reported]))


def evaluate(brain: Any, task: str, work: sr.Work, exposed: bool = True) -> dict[str, Any]:
    """The instrument's forked evaluation: every history to the pre-query state, both queries,
    intact and with the working trace erased before the query (the instrument's lesion)."""
    answers = np.full((8, 2), -1, dtype=np.int64)
    erased = np.full((8, 2), -1, dtype=np.int64)
    expected = np.stack(
        [targets(task, sr.FACTS, np.full(8, q, dtype=np.int64)) for q in range(2)], axis=1
    )
    with tempfile.TemporaryDirectory(prefix="cadence-relevance-controls-") as directory:
        directory = Path(directory)
        from cadence import Brain  # noqa: PLC0415

        prefix_brain = Brain.load(brain.save(directory / "initial.npz"))
        prefix_brain.reset()
        prefix, _, _ = sr.inputs(sr.FACTS, np.zeros(8, dtype=np.int64), NEUTRAL_EVENTS)
        complete = True
        for observation in prefix:
            if work.act(prefix_brain, observation) is None:
                complete = False
                break
        state = prefix_brain.save(directory / "prequery.npz")
        if complete:
            for lesion, store in (("intact", answers), ("erased", erased)):
                for q in range(2):
                    branch = Brain.load(state)
                    if lesion == "erased":
                        branch.working_memory.reset(len(sr.FACTS))
                    _, query, _ = sr.inputs(
                        sr.FACTS, np.full(8, q, dtype=np.int64), NEUTRAL_EVENTS, exposed=exposed
                    )
                    answer = work.act(branch, query)
                    if answer is None:
                        complete = False
                    else:
                        store[:, q] = answer
    valid = answers >= 0
    correct = (answers == expected) & valid
    unequal = sr.FACTS[:, 0] != sr.FACTS[:, 1]
    erased_correct = (erased == expected) & (erased >= 0)
    return {
        "completed": complete,
        "answers": answers.tolist(),
        "expected": expected.tolist(),
        "accuracy": float(correct.mean()),
        "unequal_fact_accuracy": float(correct[unequal].mean()),
        "per_query_accuracy": correct.mean(axis=0).tolist(),
        "unanswered": int((~valid).sum()),
        "erased": {"answers": erased.tolist(), "accuracy": float(erased_correct.mean()),
                   "unequal_fact_accuracy": float(erased_correct[unequal].mean())},
    }


def holdout_split(seed: int) -> np.ndarray:
    """The founder's held-out panel: four case indices, one per (order, query) cell, each a
    different fact pair, assigned by a permutation drawn from the founder seed. Every training
    cell keeps three fact pairs, so both values of each fact occur with every order and query
    in training; only the four conjunctions are new. Case ``(A, B, order, query)`` has the
    index ``8A + 4B + 2 order + query`` in ``CASES``."""
    rng = np.random.default_rng(np.random.SeedSequence([HOLDOUT_SEED, seed]))
    pairs = rng.permutation(4)  # the fact pair 2A + B held out of each cell 2 order + query
    return np.sort(4 * pairs + np.arange(4))


def split_scores(evaluation: dict[str, Any], held_out: np.ndarray) -> dict[str, Any]:
    """The evaluation's intact and erased accuracy on the training and the held-out cases;
    case ``8A + 4B + 2 order + query`` is the answer row ``4A + 2B + order``, column ``query``.
    """
    expected = np.asarray(evaluation["expected"])
    mask = np.zeros(expected.shape, dtype=bool)
    mask[held_out // 2, held_out % 2] = True
    unequal = (sr.FACTS[:, 0] != sr.FACTS[:, 1])[:, None] & mask
    out: dict[str, Any] = {
        "held_out_index": held_out.tolist(), "held_out_cases": sr.CASES[held_out].tolist(),
    }
    for name, answers in (("intact", evaluation["answers"]),
                          ("erased", evaluation["erased"]["answers"])):
        answers = np.asarray(answers)
        correct = (answers == expected) & (answers >= 0)
        out[name] = {
            "training": float(correct[~mask].mean()), "held_out": float(correct[mask].mean()),
            "held_out_unequal_fact": float(correct[unequal].mean()),
        }
    return out


@contextmanager
def checkpoint_forks(brain: Any) -> Iterator[Callable[[], Any]]:
    """Fresh copies of the brain from one saved checkpoint, the instrument's own snapshot;
    the brain's checkpoint must read the same after the forks were used."""
    from cadence import Brain  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="cadence-relevance-controls-") as directory:
        initial = brain.save(Path(directory) / "initial.npz")
        original = sr.checkpoint_digest(initial)
        yield lambda: Brain.load(initial)
        if sr.checkpoint_digest(brain.save(Path(directory) / "after.npz")) != original:
            raise AssertionError("the reading changed the acquired brain")


def walk(brain: Any, work: sr.Work, prefixes: list[list[np.ndarray]]) -> bool:
    """Fresh streams through one or more prefixes in turn; False on a refusal."""
    brain.reset()
    for prefix in prefixes:
        for observation in prefix:
            if work.act(brain, observation) is None:
                return False
    return True


def present(
    brain: Any, work: sr.Work, prefixes: list[list[np.ndarray]], query: np.ndarray,
) -> np.ndarray | None:
    """``walk``, then the query; None on a refusal."""
    if not walk(brain, work, prefixes):
        return None
    answer = work.act(brain, query)
    return None if answer is None else np.asarray(answer).reshape(-1)


def transplant(
    load: Callable[[], Any], work: sr.Work, prefix: list[np.ndarray], second: list[np.ndarray],
    query: np.ndarray,
) -> np.ndarray | None:
    """The host walks the prefix; a donor fork walks the second prefix from its own reset;
    the donor's working trace replaces the host's before the query. The replacement reaches
    the trace content alone, under the host's warm state from its own prefix."""
    host, donor = load(), load()
    if not walk(host, work, [prefix]) or not walk(donor, work, [second]):
        return None
    for name in ("trace", "last", "cold"):
        setattr(host.working_memory, name, getattr(donor.working_memory, name).copy())
    answer = work.act(host, query)
    return None if answer is None else np.asarray(answer).reshape(-1)


def replacement(
    brain: Any, task: str, work: sr.Work, forks: Any = None,
) -> dict[str, Any]:
    """The replacement intervention on the trace arm: every case's prefix, then without a
    reset a second prefix with the queried fact flipped (relevant; the correct answer changes)
    or the other fact flipped (irrelevant; it does not), then the query, each on a fork of the
    trained brain; no lesson. ``follow`` counts answers equal to the correct answer of the
    replaced facts, ``stay`` answers equal to the intact answer of the same case.

    Two controls, added after the declared runs of the reading: ``identical`` presents the
    same prefix twice (a second episode without a boundary, the content unchanged), and the
    ``_transplant`` conditions put the trace a fresh fork builds from the second prefix alone
    into the host before the query (brains with a working trace only).
    """
    rows = sr.CASES
    queries = rows[:, 3]
    index = np.arange(len(rows))
    replaced = {"relevant": rows.copy(), "irrelevant": rows.copy(), "identical": rows.copy()}
    replaced["relevant"][index, queries] ^= 1
    replaced["irrelevant"][index, 1 - queries] ^= 1
    prefix, query, _ = sr.inputs(rows[:, :3], queries, NEUTRAL_EVENTS)
    traced = getattr(brain, "working_memory", None) is not None
    answers: dict[str, np.ndarray | None] = {}
    with (checkpoint_forks(brain) if forks is None else forks) as load:
        answers["intact"] = present(load(), work, [prefix], query)
        for name, other in replaced.items():
            second, _, _ = sr.inputs(other[:, :3], queries, NEUTRAL_EVENTS)
            answers[name] = present(load(), work, [prefix, second], query)
            if traced:
                answers[name + "_transplant"] = transplant(load, work, prefix, second, query)
    out: dict[str, Any] = {
        "completed": all(a is not None for a in answers.values()),
        "answers": {k: None if a is None else a.tolist() for k, a in answers.items()},
    }
    if out["completed"]:
        intact = answers["intact"]
        out["intact_accuracy"] = float(np.mean(intact == targets(task, rows, queries)))
        for name, answer in answers.items():
            if name != "intact":
                other = replaced[name.removesuffix("_transplant")]
                out[name] = {
                    "follow": float(np.mean(answer == targets(task, other, queries))),
                    "stay": float(np.mean(answer == intact)),
                }
    return out


def job(
    spec: tuple[str, str, int, int, str, float | None, bool, dict[str, float]]
    | tuple[str, str, int, int, str, float | None, bool, dict[str, float], bool, bool],
) -> dict[str, Any]:
    task, brain_name, rounds, seed, teaching, eta, exposed, trace, *readings = spec
    holdout, replace = (list(readings) + [False, False])[:2]
    from cadence import Brain, LearnerConfig  # noqa: PLC0415

    began = time.monotonic()
    held = holdout_split(seed) if holdout else None
    cases = sr.CASES if held is None else np.delete(sr.CASES, held, axis=0)
    rng = np.random.default_rng(np.random.SeedSequence([31014, seed]))
    order = np.stack([rng.permutation(len(cases)) for _ in range(rounds)])
    options: dict[str, Any] = {}
    if eta is not None:
        # the teaching contrast's step and its bias step at the founder's ratio of one tenth,
        # the composed teaching founders otherwise
        options["learning"] = LearnerConfig(
            beta=0.1, eta=eta, eta_bias=eta / 10, temperature=0.2, tolerance=3e-3,
            free_steps=1024, nudged_steps=12, momentum=0.9,
        )
    options.update(
        {f"working_memory_{name}": value for name, value in trace.items() if name != "scale"}
    )
    if "scale" in trace:
        options["scales"] = {"trace": trace["scale"]}
    brain = Brain.compose(
        sr.INPUTS, 2, seed=seed, modules=BRAINS[brain_name], episodic=False, **options
    )
    work = sr.Work()
    trained = train(brain, task, order, work, teaching, exposed, cases=cases)
    evaluation = evaluate(brain, task, work, exposed)
    row = {
        "task": task, "brain": brain_name, "modules": BRAINS[brain_name], "rounds": rounds,
        "seed": seed, "neutral_events": NEUTRAL_EVENTS, "teaching": teaching, "exposed": exposed,
        "learning_eta": eta, "trace_genes": trace, "resolved_working_memory": {
            k: v for k, v in brain.describe()["memories"]["working_memory"].items()
            if k in ("decay", "amplitude", "scale")
        }, "resolved_learning": {
            k: v for k, v in brain.learner.config.to_dict().items() if k in ("eta", "eta_bias")
        },
        "training": trained, "evaluation": evaluation,
    }
    if held is not None:
        row["holdout"] = split_scores(evaluation, held)
    if replace:
        row["replacement"] = replacement(brain, task, work)
    row["work"] = asdict(work)
    row["elapsed_seconds"] = round(time.monotonic() - began, 3)
    return row


def digests() -> dict[str, str]:
    out = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sr.sources()}
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out["benchmarks/relevance/acquisition_controls.py"] = me
    return out


def table(rows: list[dict[str, Any]]) -> str:
    lines = ["| task | brain | rounds | accuracy, mean [per founder] | unequal-fact accuracy "
             "| founders at or above 0.95 |", "| --- | --- | --- | --- | --- | --- |"]
    keys = sorted({(r["task"], r["brain"], r["rounds"]) for r in rows},
                  key=lambda k: (TASKS.index(k[0]), list(BRAINS).index(k[1]), k[2]))
    for task, brain_name, rounds in keys:
        cell = sorted(
            (r for r in rows if (r["task"], r["brain"], r["rounds"]) == (task, brain_name, rounds)),
            key=lambda r: r["seed"],
        )
        acc = [r["evaluation"]["accuracy"] for r in cell]
        une = [r["evaluation"]["unequal_fact_accuracy"] for r in cell]
        lines.append(
            f"| {task} | {brain_name} | {rounds} | {np.mean(acc):.3f} "
            f"[{' '.join(f'{a:.2f}' for a in acc)}] | {np.mean(une):.3f} "
            f"[{' '.join(f'{u:.2f}' for u in une)}] | {sum(a >= 0.95 for a in acc)}/{len(acc)} |"
        )
    return "\n".join(lines)


def cells(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, str, int], list[dict[str, Any]]]]:
    """The runs grouped by task, brain and rounds in the tables' order, founders ascending."""
    keys = sorted({(r["task"], r["brain"], r["rounds"]) for r in rows},
                  key=lambda k: (TASKS.index(k[0]), list(BRAINS).index(k[1]), k[2]))
    return [(key, sorted((r for r in rows if (r["task"], r["brain"], r["rounds"]) == key),
                         key=lambda r: r["seed"])) for key in keys]


def acquires_through_trace(row: dict[str, Any]) -> bool:
    """The trace-acquisition reading: right on every unequal-fact case, chance once erased."""
    e = row["evaluation"]
    return bool(e["unequal_fact_accuracy"] == 1.0 and e["erased"]["accuracy"] == 0.5)


def holdout_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| task | brain | rounds | training intact, mean [per founder] | training erased "
             "| held-out intact, mean [per founder] | held-out erased, mean [per founder] "
             "| held-out unequal-fact | founders at held-out 1.00 with erased at most 0.50 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (task, brain_name, rounds), cell in cells(rows):
        parts = {name: [r["holdout"][lesion][part] for r in cell]
                 for name, lesion, part in (("ti", "intact", "training"),
                                            ("te", "erased", "training"),
                                            ("hi", "intact", "held_out"),
                                            ("he", "erased", "held_out"),
                                            ("hu", "intact", "held_out_unequal_fact"))}
        passed = sum(hi > 0.75 and he <= 0.5 for hi, he in zip(parts["hi"], parts["he"],
                                                                strict=True))
        lines.append(
            f"| {task} | {brain_name} | {rounds} | {np.mean(parts['ti']):.3f} "
            f"[{' '.join(f'{v:.2f}' for v in parts['ti'])}] | {np.mean(parts['te']):.3f} "
            f"| {np.mean(parts['hi']):.3f} [{' '.join(f'{v:.2f}' for v in parts['hi'])}] "
            f"| {np.mean(parts['he']):.3f} [{' '.join(f'{v:.2f}' for v in parts['he'])}] "
            f"| {np.mean(parts['hu']):.3f} | {passed}/{len(cell)} |"
        )
    return "\n".join(lines)


def replacement_table(rows: list[dict[str, Any]]) -> str:
    columns = (("relevant", "follow"), ("irrelevant", "stay"), ("irrelevant", "follow"),
               ("identical", "stay"), ("relevant_transplant", "follow"),
               ("irrelevant_transplant", "stay"), ("identical_transplant", "stay"))
    lines = ["| task | brain | rounds | founder | accuracy | unequal-fact | erased "
             "| relevant follow | irrelevant stay | irrelevant correct | identical stay "
             "| transplant relevant follow | transplant irrelevant stay "
             "| transplant identical stay | acquires through the trace |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- "
             "| --- | --- |"]
    for (task, brain_name, rounds), cell in cells(rows):
        for r in cell:
            e, p = r["evaluation"], r["replacement"]
            counts = " | ".join(
                f"{p[name][count]:.2f}" if p["completed"] and name in p else ""
                for name, count in columns
            )
            lines.append(
                f"| {task} | {brain_name} | {rounds} | {r['seed']} | {e['accuracy']:.2f} "
                f"| {e['unequal_fact_accuracy']:.2f} | {e['erased']['accuracy']:.2f} | {counts} "
                f"| {'yes' if acquires_through_trace(r) else 'no'} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), choices=TASKS)
    parser.add_argument("--brains", nargs="*", default=list(BRAINS), choices=list(BRAINS))
    parser.add_argument("--rounds", nargs="*", type=int, default=list(ROUNDS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(FOUNDERS))
    parser.add_argument("--teaching", choices=("every", "surprise"), default="every")
    parser.add_argument(
        "--learning-eta", type=float, default=None,
        help="the teaching contrast's step; the composed founder when omitted",
    )
    parser.add_argument("--query", choices=("exposed", "trace"), default="exposed",
                        help="both facts exposed at the query, or absent (the trace arm)")
    for gene in ("decay", "amplitude", "scale"):
        parser.add_argument(f"--trace-{gene}", type=float, default=None,
                            help=f"working_memory_{gene}; the composed founder when omitted")
    parser.add_argument("--holdout", action="store_true",
                        help="train on twelve cases and read the founder's four held-out ones")
    parser.add_argument("--replace", action="store_true",
                        help="the replacement intervention after training (trace arm only)")
    args = parser.parse_args(argv)
    if args.replace and args.query != "trace":
        parser.error("--replace reads the trace arm; use --query trace")
    args.out.mkdir(parents=True, exist_ok=True)
    trace = {g: v for g, v in (("decay", args.trace_decay), ("amplitude", args.trace_amplitude),
                               ("scale", args.trace_scale)) if v is not None}
    specs = [(t, b, r, s, args.teaching, args.learning_eta, args.query == "exposed", trace,
              args.holdout, args.replace)
             for t, b, r, s in itertools.product(args.tasks, args.brains, args.rounds, args.seeds)]
    done = {p.stem for p in args.out.glob("*.json") if p.stem != "summary"}
    todo = [s for s in specs if f"{s[0]}-{s[1]}-{s[2]}-{s[3]}" not in done]
    print(f"{len(todo)} runs to go, {len(done)} done", flush=True)
    import cadence  # noqa: PLC0415

    with get_context("spawn").Pool(max(1, min(args.workers, len(todo) or 1))) as pool:
        for row in pool.imap_unordered(job, todo):
            name = f"{row['task']}-{row['brain']}-{row['rounds']}-{row['seed']}"
            (args.out / f"{name}.json").write_text(json.dumps(row, indent=1))
            e = row["evaluation"]
            readings = ""
            if "holdout" in row:
                h = row["holdout"]
                readings += (f" held-out {h['intact']['held_out']:.2f}"
                             f"/{h['erased']['held_out']:.2f}")
            if row.get("replacement", {}).get("completed"):
                p = row["replacement"]
                readings += (f" follow {p['relevant']['follow']:.2f}"
                             f" stay {p['irrelevant']['stay']:.2f}")
            print(
                f"  {name}: accuracy {e['accuracy']:.3f} unequal {e['unequal_fact_accuracy']:.3f} "
                f"answers {e['answers']}{readings} ({row['elapsed_seconds']}s)",
                flush=True,
            )
    rows = [json.loads(p.read_text()) for p in args.out.glob("*.json") if p.stem != "summary"]
    summary = {"cadence_version": cadence.__version__, "numpy_version": np.__version__,
               "sources": digests(), "runs": len(rows), "teaching": args.teaching,
               "learning_eta": args.learning_eta, "query": args.query, "trace_genes": trace,
               "table": table(rows)}
    if args.holdout:
        summary["holdout"] = True
        summary["holdout_table"] = holdout_table([r for r in rows if "holdout" in r])
    if args.replace:
        summary["replace"] = True
        summary["replacement_table"] = replacement_table([r for r in rows if "replacement" in r])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(summary["table"])
    for extra in ("holdout_table", "replacement_table"):
        if extra in summary:
            print(summary[extra])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
