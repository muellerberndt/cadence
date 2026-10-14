"""Bounded vanished-cue protocol with frozen paired inputs and checkpoint-forked controls.

Run from an environment with the intended Cadence checkout installed. This measures
one selected graph/trace recipe; it does not certify a retention horizon or select defaults.
Use --verify RUN_DIRECTORY to check a completed run's source and artifact custody.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import cadence
from cadence import Brain, LearnerConfig
from cadence.learning import LearningPhaseError
from cadence.receipts import Receipt

NEUTRAL, PROBE = 0, 1
CONTROLS = ("intact", "erased", "shuffled", "reset")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pictures(cues: int, distractors: int) -> tuple[np.ndarray, int]:
    n = 2 + cues + distractors
    return np.eye(n), n


def make_brain(inputs: int, cues: int, seed: int, decay: float, amplitude: float,
               modules: tuple[int, ...], qualified: bool) -> Brain:
    learning = LearnerConfig(
        beta=0.1, temperature=0.2, tolerance=3e-3, free_steps=1024,
        nudged_steps=128 if qualified else 12, eta=0.05, eta_bias=0.005,
        momentum=0.9, normalize=0.99, normalize_floor=1e-4,
        qualified=qualified, damping=3,
    )
    return Brain.compose(
        inputs, cues, modules=modules, lateral=0.0, seed=seed, learning=learning,
        episodic=False, working_memory_decay=decay, working_memory_amplitude=amplitude,
        working_memory_scale=12.0,  # the composition of releases before 0.82.0
    )


def episode(rng: np.random.Generator, pics: np.ndarray, cues: int, distractors: int,
            delay: int, streams: int) -> tuple[np.ndarray, np.ndarray]:
    """Each group sees every cue once and identical subsequent distractor/probe inputs.

    Cue assignment is permuted independently each episode: row identity is not a label.
    All rows retain their own continuing state between episodes.
    """
    if cues < 2 or streams % cues or streams < cues or delay < 0 or distractors < 0:
        raise ValueError("paired episodes need >=2 cues, complete cue groups and nonnegative delay")
    groups = streams // cues
    labels = np.concatenate([rng.permutation(cues) for _ in range(groups)])
    steps = [pics[2 + labels]]
    for _ in range(delay):
        which = (2 + cues + np.repeat(rng.integers(distractors, size=groups), cues)
                 if distractors else np.zeros(streams, dtype=int))
        steps.append(pics[which])
    steps.append(np.tile(pics[PROBE], (streams, 1)))
    return np.stack(steps), labels


def freeze_streams(path: Path, *, seed: int, delay: int, cues: int, distractors: int,
                   streams: int, lessons: int, test_episodes: int) -> dict[str, np.ndarray]:
    """Freeze observations, labels and lesion permutations before any model runs."""
    pics, _ = pictures(cues, distractors)
    arrays = {}
    for phase, count, salt in (("train", lessons, 0), ("test", test_episodes, 1)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, delay, salt]))
        episodes = [episode(rng, pics, cues, distractors, delay, streams) for _ in range(count)]
        obs = np.stack([item[0] for item in episodes])
        arrays[phase + "_observations"] = np.pad(obs, ((0, 0), (0, 0), (0, 0), (0, cues)))
        arrays[phase + "_labels"] = np.stack([item[1] for item in episodes])
    rng = np.random.default_rng(np.random.SeedSequence([seed, delay, 2]))
    permutations = []
    for _ in range(test_episodes):
        # No row keeps its own trace, and every transplanted trace saw a different cue.
        permutations.append(np.concatenate([
            np.roll(np.arange(start, start + cues), rng.integers(1, cues))
            for start in range(0, streams, cues)
        ]))
    arrays["test_permutations"] = np.stack(permutations)
    np.savez_compressed(path, **arrays)
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


@dataclass
class Work:
    """Actual solver work; transport checks are separate from sweeps, not added as joules."""

    action_attempts: int = 0
    action_refusals: int = 0
    action_sweeps: int = 0
    action_row_sweeps: int = 0
    action_residual_checks: int = 0
    action_row_residual_checks: int = 0
    action_stagnation_checks: int = 0
    teacher_attempts: int = 0
    teacher_refusals: int = 0
    teacher_presentations: int = 0
    teacher_accepted_presentations: int = 0
    teacher_sweeps: int = 0
    teacher_row_sweeps: int = 0
    teacher_residual_checks: int = 0
    teacher_row_residual_checks: int = 0
    teacher_stagnation_checks: int = 0
    trace_row_updates: int = 0
    resets_after_refusal: int = 0
    calls_seconds: float = 0.0
    reports: list[dict] = field(default_factory=list)

    def act(self, brain: Brain, x: np.ndarray) -> np.ndarray | None:
        start = time.perf_counter()
        self.action_attempts += 1
        prior_report = brain.last_settlement
        try:
            answer = brain.act(x, greedy=True)
        except RuntimeError:
            # Catch only a reported numerical refusal; unrelated errors must not be hidden.
            report = brain.last_settlement
            if report is None or report is prior_report or report["qualified"]:
                raise
            self.action_refusals += 1
            answer = None
        finally:
            self.calls_seconds += time.perf_counter() - start
        report = dict(brain.last_settlement)
        self.reports.append({"operation": "act", **report})
        for name in ("sweeps", "residual_checks", "stagnation_checks"):
            amount = int(report["steps" if name == "sweeps" else name])
            setattr(self, "action_" + name, getattr(self, "action_" + name) + amount)
            if name != "stagnation_checks":
                field_name = "action_row_" + name
                setattr(self, field_name, getattr(self, field_name) + len(x) * amount)
        if answer is not None:
            self.trace_row_updates += len(x)
        return answer

    def teach(self, brain: Brain, x: np.ndarray, labels: np.ndarray) -> None:
        start = time.perf_counter()
        self.teacher_attempts += 1
        try:
            _, report = brain.learner.step(brain.stimulus(x), labels)
        except LearningPhaseError as error:
            self.teacher_refusals += 1
            report = error.report
        finally:
            self.calls_seconds += time.perf_counter() - start
        self.reports.append({"operation": "teach", **report})
        for dest, source in (
            ("presentations", "attempted_presentations"),
            ("accepted_presentations", "accepted_presentations"),
            ("sweeps", "total_steps"), ("row_sweeps", "total_row_sweeps"),
            ("residual_checks", "total_residual_checks"),
            ("row_residual_checks", "total_row_residual_checks"),
            ("stagnation_checks", "total_stagnation_checks"),
        ):
            setattr(self, "teacher_" + dest, getattr(self, "teacher_" + dest) + int(report[source]))


def run_stream(brain: Brain, obs: np.ndarray, work: Work,
               teacher: np.ndarray | None = None) -> np.ndarray | None:
    """A refused teacher is charged and skipped; a refused action stops this life.

    No reward is fabricated and no reset/retry turns an unobserved transition into experience.
    Teachers label the current probe; all issued actions are greedy and leave no reward pending.
    """
    answer = None
    for t, x in enumerate(obs):
        if t == len(obs) - 1 and teacher is not None:
            work.teach(brain, x, teacher)
        answer = work.act(brain, x)
        if answer is None:
            break
    return answer


def append_history(obs: np.ndarray, labels: np.ndarray, cues: int) -> np.ndarray:
    out = obs.copy()
    out[-1, :, -cues:] = np.eye(cues)[labels]
    return out


def probe_controls(checkpoint: Path, x: np.ndarray, permutation: np.ndarray,
                   order: tuple[str, ...] = CONTROLS) -> dict[str, dict]:
    """Every branch begins at the same complete pre-probe state, with learning disabled."""
    if set(order) != set(CONTROLS) or len(order) != len(CONTROLS):
        raise ValueError("each declared control must run exactly once")
    result = {}
    for control in order:
        branch = Brain.load(checkpoint)
        trace = branch.working_memory
        assert trace is not None
        if control == "erased":
            trace.reset(len(x))
        elif control == "shuffled":
            for name in ("trace", "last", "cold"):
                setattr(trace, name, getattr(trace, name)[permutation].copy())
        elif control == "reset":
            branch.reset()
        work = Work()
        answer = work.act(branch, x)
        result[control] = {
            "answers": None if answer is None else answer.tolist(),
            "work": asdict(work),
        }
    return result


def train(brain: Brain, frozen: dict[str, np.ndarray], *, appended: bool,
          cues: int) -> tuple[Work, bool]:
    work = Work()
    for obs, labels in zip(frozen["train_observations"], frozen["train_labels"], strict=True):
        inputs = append_history(obs, labels, cues) if appended else obs
        if run_stream(brain, inputs, work, teacher=labels) is None:
            return work, False
    return work, True


def evaluate(brain: Brain, frozen: dict[str, np.ndarray], directory: Path,
             *, appended: bool, cues: int) -> dict:
    prefix_work, continuation_work = Work(), Work()
    trials = []
    completed = True
    for index, (obs, labels, permutation) in enumerate(zip(
        frozen["test_observations"], frozen["test_labels"], frozen["test_permutations"],
        strict=True,
    )):
        obs = append_history(obs, labels, cues) if appended else obs
        if run_stream(brain, obs[:-1], prefix_work) is None:
            completed = False
            break
        anchor = brain.save(directory / f"probe-{index:04d}.npz")
        if appended:
            answer = continuation_work.act(brain, obs[-1])
            branches = {"appended": {"answers": None if answer is None else answer.tolist()}}
        else:
            branches = probe_controls(anchor, obs[-1], permutation)
            # Only the intact life continues. Forks never install their state into it.
            answer = continuation_work.act(brain, obs[-1])
            expected = branches["intact"]["answers"]
            assert (None if answer is None else answer.tolist()) == expected
            restored = Brain.load(anchor)
            # Save/load parity of all state (including optimizer/RNG), not just the answer.
            restored_work = Work()
            restored_answer = restored_work.act(restored, obs[-1])
            assert (None if restored_answer is None else restored_answer.tolist()) == expected
            continued = brain.save(directory / f"continued-{index:04d}.npz")
            resumed = restored.save(directory / f"resumed-{index:04d}.npz")
            with (
                np.load(continued, allow_pickle=False) as a,
                np.load(resumed, allow_pickle=False) as b,
            ):
                assert set(a.files) == set(b.files)
                assert all(np.array_equal(a[name], b[name]) for name in a.files)
            branches["resume_check"] = {"work": asdict(restored_work), "equal_checkpoint": True}
        trials.append({"labels": labels.tolist(), "branches": branches})
        if answer is None:
            completed = False
            break
    names = ("appended",) if appended else CONTROLS
    scores = {}
    for name in names:
        answers = [(trial["branches"][name]["answers"], trial["labels"]) for trial in trials]
        rows = sum(len(labels) for _, labels in answers)
        correct = sum(int(np.sum(np.asarray(a) == labels))
                      for a, labels in answers if a is not None)
        refused = sum(len(labels) for a, labels in answers if a is None)
        scores[name] = {"correct": correct, "attempted_rows": rows, "refused_rows": refused,
                        "not_run_rows": len(frozen["test_labels"].ravel()) - rows,
                        "recall_including_refusal": correct / rows if rows else None}
    brain.save(directory / "final.npz")
    return {"completed": completed, "scores": scores, "trials": trials,
            "prefix_work": asdict(prefix_work), "continuation_work": asdict(continuation_work)}


def _verify(directory: Path) -> tuple[bool, str]:
    path = directory / "summary.json"
    raw = json.loads(path.read_text())
    sources = [(item["path"], directory / item["path"]) for item in raw["source"]["files"]]

    def check(body: dict) -> str | None:
        for name, expected in body["artifacts"].items():
            if sha256(directory / name) != expected:
                return f"artifact differs: {name}"
        return None

    valid, reason = Receipt.verify(path, sources=sources, check=check)
    return valid, "canonical form, digest, sources and artifact hashes agree" if valid else reason


def verify(directory: Path) -> tuple[bool, str]:
    try:
        return _verify(directory)
    except (OSError, ValueError, KeyError) as error:
        return False, f"cannot verify artifact: {error}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--out", type=Path)
    destination.add_argument("--verify", type=Path)
    parser.add_argument("--cues", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=0)
    parser.add_argument("--delays", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--decays", type=float, nargs="+", default=[0.2, 0.5, 0.8, 0.9])
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--modules", default="32")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--lessons", type=int, default=150)
    parser.add_argument("--test-episodes", type=int, default=12)
    parser.add_argument("--finite", action="store_true")
    args = parser.parse_args(argv)
    if args.verify is not None:
        valid, reason = verify(args.verify)
        print(json.dumps({"valid": valid, "reason": reason}))
        return 0 if valid else 1
    try:
        modules = tuple(int(value) for value in args.modules.split(","))
    except ValueError:
        parser.error("--modules must be comma-separated positive integers")
    if (args.cues < 2 or args.streams < args.cues or args.streams % args.cues
            or args.distractors < 0 or args.lessons < 1 or args.test_episodes < 1
            or any(delay < 0 for delay in args.delays) or any(seed < 0 for seed in args.seeds)
            or not modules or any(value < 1 for value in modules)
            or not np.isfinite(args.amplitude) or args.amplitude < 0
            or any(not np.isfinite(decay) or not 0 <= decay < 1 for decay in args.decays)):
        parser.error("invalid dimensions/budgets: use complete cue groups, positive lessons/tests, "
                     "nonnegative seeds/delays/amplitude and finite decays in [0, 1)")
    for values in (args.delays, args.decays, args.seeds):
        if len(values) != len(set(values)):
            parser.error("duplicate delays, decays or seeds would duplicate a condition")
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    protocol = {**vars(args), "out": str(args.out), "verify": None, "modules": list(modules),
                "schema": "vanished-cue-recall/2", "cadence": cadence.__version__,
                "numpy": np.__version__, "python": platform.python_version(),
                "platform": platform.platform(),
                "cadence_import": str(Path(cadence.__file__).resolve()),
                "command": [Path(__file__).name, *(sys.argv[1:] if argv is None else argv)],
                "policy": "greedy life; probe-only teacher; skip refused lessons; stop refused act",
                "selection": "no acceptance horizon or default selection is claimed"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    sources = []
    source_origins = []
    package = Path(cadence.__file__).resolve().parent
    for source in [Path(__file__).resolve(), *sorted(package.rglob("*.py"))]:
        relative = ("source/vanished_cue.py" if source == Path(__file__).resolve()
                    else "source/cadence/" + source.relative_to(package).as_posix())
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources.append((relative, target))
        source_origins.append((source, target))
    frozen_sets = {}
    for delay in args.delays:
        for seed in args.seeds:
            frozen_sets[delay, seed] = freeze_streams(
                args.out / f"streams-delay{delay}-seed{seed}.npz", seed=seed, delay=delay,
                cues=args.cues, distractors=args.distractors, streams=args.streams,
                lessons=args.lessons, test_episodes=args.test_episodes,
            )
    rows = []
    for decay in args.decays:
        for delay in args.delays:
            for seed in args.seeds:
                row = {"decay": decay, "delay": delay, "seed": seed, "models": {}}
                frozen = frozen_sets[delay, seed]
                for name, appended in (("vanished", False), ("appended", True)):
                    directory = args.out / f"decay{decay}-delay{delay}-seed{seed}" / name
                    directory.mkdir(parents=True)
                    brain = make_brain(frozen["train_observations"].shape[-1], args.cues,
                                       seed, decay, args.amplitude, modules, not args.finite)
                    brain.save(directory / "initial.npz")
                    training, complete = train(brain, frozen, appended=appended, cues=args.cues)
                    brain.save(directory / "trained.npz")
                    outcome = {"training": asdict(training), "training_completed": complete,
                               "evaluation": None}
                    if complete:
                        outcome["evaluation"] = evaluate(
                            brain, frozen, directory, appended=appended, cues=args.cues)
                    row["models"][name] = outcome
                rows.append(row)
                with (args.out / "rows.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps({key: row[key] for key in ("decay", "delay", "seed")}), flush=True)
    assert all(sha256(original) == sha256(copy) for original, copy in source_origins), (
        "source changed during the run; retain this incomplete attempt and rerun from fixed source")
    artifacts = {path.relative_to(args.out).as_posix(): sha256(path)
                 for path in sorted(args.out.rglob("*")) if path.is_file()
                 and "source" not in path.relative_to(args.out).parts}
    summary = {"protocol": protocol, "chance": 1 / args.cues, "rows": rows,
               "artifacts": artifacts, "seconds": time.perf_counter() - began,
               "work_scope": "Every actual teacher/free solve, fork and continuation check; "
               "calls_seconds excludes checkpoint IO; run seconds includes IO. "
               "No reward/eligibility/episodic writes occur. Sweeps are not joules."}
    Receipt.build("vanished-cue-recall/2", summary, sources).write(args.out / "summary.json")
    valid, reason = verify(args.out)
    assert valid, reason
    print(json.dumps({"verified": valid, "summary": str(args.out / "summary.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
