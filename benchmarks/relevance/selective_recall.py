"""Bounded named-fact selection with current-query-only teaching and saved forks.

This instruments the existing graph and trace; it adds no runtime mechanism.
All sixteen cases are taught: evaluation measures finite acquisition, not novel
compositional generalization. Trial resets are explicit component boundaries.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import cadence
from cadence import Brain
from cadence.learning import LearningPhaseError
from cadence.receipts import Receipt, canonical_json, canonical_sha256

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"
SCHEMA = "cadence-selective-recall/1"
# A value, B value, presentation order. Every query is evaluated from the same
# saved pre-query state; row identity is permuted independently in each lesson.
FACTS = np.array(list(itertools.product(range(2), repeat=3)), dtype=np.int64)
CASES = np.array(list(itertools.product(range(2), repeat=4)), dtype=np.int64)
INPUTS = 11


def inputs(
    rows: np.ndarray, queries: np.ndarray, neutral_events: int, *, exposed: bool = False
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Encode witnessed named facts and an identity-only query, with no selected slot.

    Coordinates: A0,A1,B0,B1,neutral,ask-A,ask-B,then four exposed-control
    coordinates containing BOTH named facts. The trace arm leaves the last four zero.
    """
    rows, queries = np.asarray(rows), np.asarray(queries)
    if rows.ndim != 2 or rows.shape[1] != 3 or not len(rows):
        raise ValueError("rows must contain A, B and presentation order")
    if rows.dtype.kind not in "iu" or not np.isin(rows, (0, 1)).all():
        raise ValueError("facts and order must be binary integers")
    if queries.shape != (len(rows),) or queries.dtype.kind not in "iu":
        raise ValueError("one integer query per row is required")
    if not np.isin(queries, (0, 1)).all():
        raise ValueError("query names A or B")
    if isinstance(neutral_events, bool) or not isinstance(neutral_events, int):
        raise ValueError("neutral_events must be a nonnegative integer")
    if neutral_events < 0:
        raise ValueError("neutral_events must be a nonnegative integer")
    a, b, order = rows.T
    symbols = np.eye(INPUTS)
    first = np.where(order[:, None] == 0, symbols[a], symbols[2 + b])
    second = np.where(order[:, None] == 0, symbols[2 + b], symbols[a])
    prefix = [first, second]
    prefix.extend(np.tile(symbols[4], (len(rows), 1)) for _ in range(neutral_events))
    query = symbols[5 + queries].copy()
    if exposed:
        query[np.arange(len(rows)), 7 + a] = 1.0
        query[np.arange(len(rows)), 9 + b] = 1.0
    targets = rows[np.arange(len(rows)), queries].copy()
    return prefix, query, targets


def scores(answers: Any) -> dict[str, Any]:
    """Score factual answers and counterfactual pairs; -1 means refused or unrun.

    Invariance alone can be achieved by a constant policy. Pair correctness is
    reported beside it. Unequal facts exclude the 75%-accurate AND/OR shortcut.
    """
    answers = np.asarray(answers)
    if answers.shape != (8, 2) or not np.isin(answers, (-1, 0, 1)).all():
        raise ValueError("answers must be eight two-query rows in {-1,0,1}")
    expected = FACTS[:, :2]
    valid, correct = answers >= 0, answers == expected
    unequal = FACTS[:, 0] != FACTS[:, 1]
    index = {tuple(row): i for i, row in enumerate(FACTS.tolist())}
    comparisons: dict[str, list[bool]] = {
        "relevant_flip": [], "relevant_pair_correct": [],
        "irrelevant_invariance": [], "irrelevant_pair_correct": [],
        "order_invariance": [],
    }
    for i, row in enumerate(FACTS.tolist()):
        for q in range(2):
            for name, field in (("relevant", q), ("irrelevant", 1 - q), ("order", 2)):
                if row[field] != 0:
                    continue
                other = list(row)
                other[field] = 1
                j = index[tuple(other)]
                both = bool(valid[i, q] and valid[j, q])
                if name == "relevant":
                    comparisons["relevant_flip"].append(both and answers[i, q] != answers[j, q])
                else:
                    comparisons[name + "_invariance"].append(
                        both and answers[i, q] == answers[j, q]
                    )
                if name != "order":
                    comparisons[name + "_pair_correct"].append(
                        bool(correct[i, q] and correct[j, q])
                    )
    return {
        "accuracy": float(correct.mean()),
        "unequal_fact_accuracy": float(correct[unequal].mean()),
        "per_query_accuracy": correct.mean(axis=0).tolist(),
        "query_flip_when_facts_differ": float(
            (valid[unequal].all(axis=1)
             & (answers[unequal, 0] != answers[unequal, 1])).mean()
        ),
        "query_pair_correct_when_facts_differ": float(correct[unequal].all(axis=1).mean()),
        "unanswered": int((~valid).sum()),
        **{name: float(np.mean(values)) for name, values in comparisons.items()},
    }


@dataclass
class Work:
    """Charge actual attempts, including refusals; numerical work is not energy."""

    actions: int = 0
    action_rows: int = 0
    action_refusals: int = 0
    action_sweeps: int = 0
    action_row_sweeps: int = 0
    action_residual_checks: int = 0
    action_row_residual_checks: int = 0
    max_action_residual: float = 0.0
    lessons: int = 0
    lesson_refusals: int = 0
    presentations: int = 0
    accepted_presentations: int = 0
    teaching_sweeps: int = 0
    teaching_row_sweeps: int = 0
    teaching_residual_checks: int = 0
    teaching_row_residual_checks: int = 0

    def act(self, brain: Brain, observation: np.ndarray) -> np.ndarray | None:
        self.actions += 1
        self.action_rows += len(observation)
        previous = brain.last_settlement
        try:
            answer = brain.act(observation, greedy=True)
        except RuntimeError:
            report = brain.last_settlement
            if report is None or report is previous or report["qualified"]:
                raise
            self.action_refusals += 1
            answer = None
        report = brain.last_settlement
        assert report is not None
        self.action_sweeps += int(report["steps"])
        self.action_row_sweeps += len(observation) * int(report["steps"])
        self.action_residual_checks += int(report["residual_checks"])
        self.action_row_residual_checks += len(observation) * int(report["residual_checks"])
        residual = float(report["max_residual"])
        # JSON cannot represent an infinite refused residual; counts retain that refusal.
        if np.isfinite(residual):
            self.max_action_residual = max(self.max_action_residual, residual)
        return answer

    def teach(self, brain: Brain, observation: np.ndarray, target: np.ndarray) -> None:
        self.lessons += 1
        try:
            _, report = brain.learner.step(brain.stimulus(observation), target)
        except LearningPhaseError as error:
            self.lesson_refusals += 1
            report = error.report
        for field, reported in (
            ("presentations", "attempted_presentations"),
            ("accepted_presentations", "accepted_presentations"),
            ("teaching_sweeps", "total_steps"),
            ("teaching_row_sweeps", "total_row_sweeps"),
            ("teaching_residual_checks", "total_residual_checks"),
            ("teaching_row_residual_checks", "total_row_residual_checks"),
        ):
            setattr(self, field, getattr(self, field) + int(report[reported]))


def parameter_digest(brain: Brain) -> str:
    return canonical_sha256({
        name: np.asarray(getattr(brain.brain, name)).tolist()
        for name in ("efficacy", "log_gain", "bias")
    })


def checkpoint_digest(path: Path) -> str:
    """Array content rather than ZIP timestamps; includes all continuation fields."""
    with np.load(path, allow_pickle=False) as archive:
        return canonical_sha256({name: archive[name].tolist() for name in sorted(archive.files)})


def evaluate(
    brain: Brain, neutral_events: int, *, exposed: bool = False
) -> dict[str, Any]:
    """Use a private copy; both queries and lesions fork identical complete state."""
    work = Work()
    predictions = {name: np.full((8, 2), -1, dtype=np.int64)
                   for name in ("intact", "erased")}
    with tempfile.TemporaryDirectory(prefix="cadence-relevance-") as directory:
        directory = Path(directory)
        initial = brain.save(directory / "initial.npz")
        original = checkpoint_digest(initial)
        prefix_brain = Brain.load(initial)
        prefix_brain.reset()
        prefix, _, _ = inputs(FACTS, np.zeros(8, dtype=np.int64), neutral_events)
        complete = True
        for observation in prefix:
            if work.act(prefix_brain, observation) is None:
                complete = False
                break
        state = prefix_brain.save(directory / "prequery.npz")
        origin = checkpoint_digest(state)
        if complete:
            for condition, answers in predictions.items():
                for q in range(2):
                    branch = Brain.load(state)
                    if condition == "erased":
                        assert branch.working_memory is not None
                        branch.working_memory.reset(len(FACTS))
                    _, query, _ = inputs(
                        FACTS, np.full(8, q, dtype=np.int64), neutral_events, exposed=exposed
                    )
                    answer = work.act(branch, query)
                    if answer is not None:
                        answers[:, q] = answer
                    else:
                        complete = False
        preserved = checkpoint_digest(brain.save(directory / "after.npz")) == original
        if not preserved:
            raise AssertionError("evaluation changed the acquired brain")
    return {
        "completed": complete, "original_preserved": preserved,
        "fork_origin": origin,
        "branches": {name: {"answers": answers.tolist(), "scores": scores(answers)}
                     for name, answers in predictions.items()},
        "work": asdict(work),
    }


def train(
    brain: Brain, order: np.ndarray, neutral_events: int, *, exposed: bool, frozen: bool,
    deadline: float,
) -> dict[str, Any]:
    work = Work()
    rounds = 0
    stop = None
    for permutation in order:
        if time.monotonic() > deadline:
            stop = "wall_budget"
            break
        rows = CASES[permutation]
        prefix, query, target = inputs(rows[:, :3], rows[:, 3], neutral_events, exposed=exposed)
        brain.reset()  # same learned parameters, explicit trial/credit boundary
        for observation in prefix:
            if work.act(brain, observation) is None:
                stop = "refused_prefix"
                break
        if stop is not None:
            break
        if not frozen:
            work.teach(brain, query, target)
        if work.act(brain, query) is None:
            stop = "refused_query"
            break
        rounds += 1
    return {"completed": rounds == len(order), "rounds": rounds,
            "stop": stop, "work": asdict(work)}


def acquisition_gate(evaluation: dict[str, Any], floor: float) -> bool:
    values = evaluation["branches"]["intact"]["scores"]
    return bool(evaluation["completed"] and min(
        values["accuracy"], values["unequal_fact_accuracy"], *values["per_query_accuracy"]
    ) >= floor)


def sources() -> list[tuple[str, Path]]:
    package = Path(cadence.__file__).resolve().parent
    return [
        ("benchmarks/relevance/" + name, HERE / name)
        for name in ("selective_recall.py", "protocol.json")
    ] + [("src/cadence/" + str(path.relative_to(package)), path)
         for path in sorted(package.rglob("*.py"))]


def revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return None  # Installed source archives need not have Git available.
    return result.stdout.strip() if result.returncode == 0 else None


def run(protocol: dict[str, Any]) -> Receipt:
    began = time.monotonic()
    deadline = began + protocol["seconds"]
    founders = []
    for seed in protocol["founders"]:
        rng = np.random.default_rng(np.random.SeedSequence([31014, seed]))
        order = np.stack([rng.permutation(len(CASES)) for _ in range(protocol["rounds"])])
        schedule = {"cases": CASES.tolist(), "round_order": order.tolist()}
        result: dict[str, Any] = {"seed": seed, "schedule": schedule,
                                 "schedule_sha256": canonical_sha256(schedule), "arms": {}}
        for arm in ("trace", "exposed", "frozen"):
            brain = Brain.compose(INPUTS, 2, seed=seed, **protocol["brain"])
            exposed = arm == "exposed"
            initial = parameter_digest(brain)
            declared = brain.describe()
            trained = train(brain, order, protocol["neutral_events"], exposed=exposed,
                            frozen=arm == "frozen", deadline=deadline)
            evaluation = evaluate(brain, protocol["neutral_events"], exposed=exposed)
            result["arms"][arm] = {
                "resolved": declared, "initial_parameters": initial,
                "final_parameters": parameter_digest(brain),
                "training": trained, "evaluation": evaluation,
            }
        random = np.random.default_rng(np.random.SeedSequence([31015, seed]))
        answers = random.integers(0, 2, size=(8, 2))
        result["random"] = {"answers": answers.tolist(), "scores": scores(answers),
                            "expected_accuracy": 0.5}
        exposed = result["arms"]["exposed"]
        gate = exposed["training"]["completed"] and acquisition_gate(
            exposed["evaluation"], protocol["acquisition_floor"]
        )
        result["acquisition_control_passed"] = gate
        result["interpretation"] = (
            "acquisition_control_passed; inspect paired trace and frozen readings"
            if gate else "acquisition_control_failed; memory failure is not localized"
        )
        founders.append(result)
    return Receipt.build(SCHEMA, {
        "protocol": protocol,
        "frozen_protocol": protocol == json.loads(PROTOCOL.read_text()),
        "source_revision": revision(), "cadence_version": cadence.__version__,
        "source_identity": (
            "The manifest hashes imported code and this instrument, including uncommitted "
            "bytes; revision alone is not their identity."
        ),
        "numpy_version": np.__version__, "elapsed_seconds": time.monotonic() - began,
        "evaluation_facts": FACTS.tolist(), "founders": founders,
        "limitations": [
            "All 16 cases were in training: finite acquisition, no held-out compositional claim.",
            "Trials reset transient state; this is not an integrated continuing life.",
            "The trace is held during each qualified present solve; no future trajectory solve.",
            "No attention mechanism, trainable memory gate or runtime change is introduced.",
            "Teacher labels only the current query; no retrospective encoding credit is added.",
            "Wall budget is checked between training trials; bounded evaluation still runs.",
            "Elapsed host seconds and solver work do not measure physical energy.",
        ],
    }, sources())


def check(body: dict[str, Any]) -> str | None:
    """Recompute reported behavioral scores and gates from retained decisions."""
    for founder in body["founders"]:
        if canonical_sha256(founder["schedule"]) != founder["schedule_sha256"]:
            return "training schedule digest differs"
        for arm in founder["arms"].values():
            for branch in arm["evaluation"]["branches"].values():
                if scores(branch["answers"]) != branch["scores"]:
                    return "behavioral scores differ from retained decisions"
        if scores(founder["random"]["answers"]) != founder["random"]["scores"]:
            return "random scores differ from retained decisions"
        exposed = founder["arms"]["exposed"]
        gate = exposed["training"]["completed"] and acquisition_gate(
            exposed["evaluation"], body["protocol"]["acquisition_floor"]
        )
        if gate != founder["acquisition_control_passed"]:
            return "acquisition gate differs from readings"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="new JSON receipt; stdout when omitted")
    parser.add_argument("--verify", type=Path, help="verify a receipt against current sources")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--neutral-events", type=int)
    args = parser.parse_args(argv)
    if args.verify:
        valid, reason = Receipt.verify(args.verify, sources=sources(), check=check)
        print(reason)
        return 0 if valid else 1
    protocol = json.loads(PROTOCOL.read_text())
    for name, value in (("rounds", args.rounds), ("founders", args.seeds),
                        ("neutral_events", args.neutral_events)):
        if value is not None:
            protocol[name] = value
    if not 1 <= protocol["rounds"] <= 1024:
        parser.error("rounds must lie in [1,1024]")
    if not 0 <= protocol["neutral_events"] <= 8:
        parser.error("neutral-events must lie in [0,8]")
    if (not 1 <= len(protocol["founders"]) <= 8 or min(protocol["founders"]) < 0
            or len(set(protocol["founders"])) != len(protocol["founders"])):
        parser.error("supply one to eight distinct nonnegative seeds")
    if args.out and args.out.exists():
        parser.error("out already exists; retained results are never overwritten")
    receipt = run(protocol)
    if args.out:
        receipt.write(args.out)
        for founder in receipt.body["founders"]:
            print(f"seed {founder['seed']}: {founder['interpretation']}")
        print(f"receipt: {args.out}")
    else:
        print(canonical_json(receipt.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
