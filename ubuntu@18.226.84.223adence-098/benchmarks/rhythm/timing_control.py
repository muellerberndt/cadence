"""Physical timing supplement for the existing learned recurrent control.

Prepare freezes the declaration, sources and already-used development artifacts
before run executes any action. This measures a conventional baseline only; it
cannot admit fresh confirmation or change the original brain timing result.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import time
from pathlib import Path

import steady_rhythm as chamber  # Sets thread limits before importing NumPy.

# isort: split
import numpy as np

from cadence.receipts import Receipt, source_manifest

ROOT = Path(__file__).resolve().parent
SCHEMA = "steady-rhythm/physical-control-1"
BASE_SHA = "751c2e151806103537693be145a54aae805d4e62c737f1e0cb75365da8805132"
PROTOCOL_SHA = "fc714c04712eb588811bd7a44e58d3f098a75ea4bc87e9a0d528387292fcafa7"
SEEDS = (0, 1)
SCHEDULES = {
    "paced_regular": "regular",
    "paced_regular50": "regular50",
    "paced_regular200": "regular200",
    "paced_extra": "extra",
    "paced_skipped": "skipped",
    "paced_regular_load": "regular",
    "paced_ordered": "ordered",
    "paced_shuffled_time": "shuffled_time",
}
PLANNED = [[seed, name] for seed in SEEDS for name in SCHEDULES]


def sources() -> list[tuple[str, Path]]:
    package = Path(chamber.cadence.__file__).resolve().parent
    return [
        ("source/" + name, ROOT / name)
        for name in (
            "timing_control.py",
            "steady_rhythm.py",
            "rhythm_inputs.py",
            "timing_acceptance.py",
        )
    ] + [
        ("source/cadence/" + path.relative_to(package).as_posix(), path)
        for path in sorted(package.rglob("*.py"))
    ]


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "threads": {name: os.environ.get(name) for name in chamber.THREAD_VARIABLES},
    }


def basis(directory: Path) -> dict:
    """Admit only the preserved second development receipt and its consumed artifacts."""
    parent = directory / "basis"
    assert chamber.sha256(parent / "summary.json") == BASE_SHA, "different baseline receipt"
    valid, reason = Receipt.verify(parent / "summary.json")
    assert valid, reason
    raw = json.loads((parent / "summary.json").read_text())
    body = raw["body"]
    expected = {"protocol.json"}
    for seed in SEEDS:
        expected.update(
            (f"inputs-seed{seed}.npz", f"seed{seed}-efference-every/probe-flipflop.npz")
        )
    for name in expected:
        assert chamber.sha256(parent / name) == body["artifacts"][name], name
    assert chamber.sha256(parent / "protocol.json") == PROTOCOL_SHA
    assert json.loads((parent / "protocol.json").read_text()) == body["protocol"]
    return body


def prepare(baseline: Path, directory: Path) -> None:
    """Freeze an immutable input/source declaration; no actor is called here."""
    directory.mkdir(parents=True, exist_ok=False)
    names = ["summary.json", "protocol.json"]
    for seed in SEEDS:
        names += [f"inputs-seed{seed}.npz", f"seed{seed}-efference-every/probe-flipflop.npz"]
    for name in names:
        target = directory / "basis" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((baseline / name).read_bytes())
    original = basis(directory)
    old_sources = {
        item["path"]: item["sha256"]
        for item in json.loads((baseline / "summary.json").read_text())["source"]["files"]
    }
    for name in ("steady_rhythm.py", "rhythm_inputs.py", "timing_acceptance.py"):
        assert chamber.sha256(ROOT / name) == old_sources["source/" + name], (
            "control source changed"
        )
    env = environment()
    assert all(env[key] == original["declaration"][key] for key in env), "resource setup differs"
    for relative, source in sources():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    declaration = {
        "planned": PLANNED,
        "baseline_sha256": BASE_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "bounds": original["protocol"]["timing_acceptance"]["bounds"],
        "burners": original["declaration"]["burners"],
        "environment": env,
        "cap_seconds": 180,
        "fresh_confirmation": False,
        "scope": "Control-only supplement on spent development seeds; all 16 cases required. "
        "Original brain results stay unchanged. Sequential runs cannot identify "
        "host-stall causality.",
        "work_scope": "All attempted policy calls and checkpoint I/O are charged. Dense MACs cover "
        "successful 4x7 by 7x2 policy products only, excluding feature creation, argmax, memory "
        "traffic and verification. Failed-call arithmetic work is unknown; measured time remains.",
        "artifacts": {
            "basis/" + name: chamber.sha256(directory / "basis" / name) for name in names
        },
    }
    Receipt.build(SCHEMA + "/declaration", declaration, sources()).write(
        directory / "declaration.json"
    )


def declaration(directory: Path, *, current: bool = False) -> dict:
    receipt = Receipt.read(directory / "declaration.json")
    listed = [(item["path"], directory / item["path"]) for item in receipt.source["files"]]
    valid, reason = Receipt.verify(directory / "declaration.json", sources=listed)
    assert valid, reason
    assert receipt.kind == SCHEMA + "/declaration"
    data = receipt.body
    original = basis(directory)
    assert data["planned"] == PLANNED and data["fresh_confirmation"] is False
    assert data["baseline_sha256"] == BASE_SHA and data["protocol_sha256"] == PROTOCOL_SHA
    assert data["bounds"] == original["protocol"]["timing_acceptance"]["bounds"]
    assert data["burners"] == original["declaration"]["burners"]
    assert data["cap_seconds"] == 180
    assert all(data["environment"][k] == original["declaration"][k] for k in environment())
    expected = {
        p.relative_to(directory).as_posix() for p in (directory / "basis").rglob("*") if p.is_file()
    }
    assert set(data["artifacts"]) == expected
    assert all(
        chamber.sha256(directory / name) == digest for name, digest in data["artifacts"].items()
    )
    if current:
        assert receipt.source == source_manifest(sources()), (
            "producing source changed after declaration"
        )
        assert data["environment"] == environment(), "resource setup changed after declaration"
    return data


class ControlWork:
    """Actual conventional-policy calls; no fabricated Cadence solver counters."""

    def __init__(self):
        self.records = []

    def act(self, model, observation):
        began, cpu = time.perf_counter(), time.process_time()
        action, error = None, None
        try:
            action = model.act(observation)
            return action
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            self.records.append(
                {
                    "answer": None if action is None else action.tolist(),
                    "error": error,
                    "cpu_seconds": time.process_time() - cpu,
                    "wall_seconds": time.perf_counter() - began,
                }
            )


def work_totals(records: list[dict]) -> dict:
    for record in records:
        if record["answer"] is None:
            assert isinstance(record["error"], str) and record["error"]
        else:
            assert record["error"] is None
            assert len(record["answer"]) == 4 and all(a in (0, 1) for a in record["answer"])
        assert all(
            isinstance(record[key], (float, int)) and np.isfinite(record[key]) and record[key] >= 0
            for key in ("cpu_seconds", "wall_seconds")
        )
    return {
        "action_calls": len(records),
        "action_rows": 4 * len(records),
        "failed_calls": sum(r["error"] is not None for r in records),
        "successful_dense_policy_macs": 4 * 7 * 2 * sum(r["error"] is None for r in records),
        "cpu_seconds": sum(r["cpu_seconds"] for r in records),
        "wall_seconds": sum(r["wall_seconds"] for r in records),
    }


def measure(directory: Path, seed: int, name: str, config: dict, work, io) -> dict:
    began = time.perf_counter()
    probe = directory / "basis" / f"seed{seed}-efference-every/probe-flipflop.npz"
    model = io.load(chamber.FlipFlop, probe)
    with np.load(directory / "basis" / f"inputs-seed{seed}.npz", allow_pickle=False) as frozen:
        x = frozen["window/observations"][1]
        due = frozen[f"cadence/{SCHEDULES[name]}/due_ms"]
    observations = np.repeat(x[None], len(due), axis=0)
    processes, stop = [], None
    burners = config["burners"] if name == "paced_regular_load" else 0
    try:
        if burners:
            context = multiprocessing.get_context("spawn")
            stop = context.Event()
            for _ in range(burners):
                process = context.Process(target=chamber._burn, args=(stop,), daemon=True)
                process.start()
                processes.append(process)
            time.sleep(0.5)
            assert all(p.is_alive() for p in processes), "host-load worker failed"
        measured = chamber.real_time_run(model, observations, due, work)
        assert all(p.is_alive() for p in processes), "host-load worker failed"
    finally:
        if stop is not None:
            stop.set()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join()
    target = directory / f"seed{seed}-{name}-after.npz"
    io.save(model, target)
    return {
        "seed": seed,
        "name": name,
        "burners": burners,
        **measured,
        "records": work.records,
        "work": work_totals(work.records),
        "checkpoint_io": io.operations,
        "elapsed_seconds": time.perf_counter() - began,
    }


def assess(case: dict, bounds: dict) -> dict:
    """The original timing inequalities, applied to the conventional control only."""
    timing = chamber.timing_acceptance.timing_readings(case)
    rates = chamber.timing_acceptance.alternation(case["actions"])
    passed = (
        min(rates) >= bounds["alternation_min"]
        and timing["missed_deadlines"] <= bounds["missed_deadlines"]
        and timing["max_lateness_fraction"] <= bounds["lateness_fraction"]
        and timing["phase_drift_fraction"] <= bounds["phase_drift_fraction"]
        and all(
            p["event_period"] == 2
            and p["max_error_ms"] is not None
            and p["max_error_ms"] <= bounds["period_error_ms"]
            for p in timing["periods"]
        )
    )
    return {
        "seed": case["seed"],
        "name": case["name"],
        "alternation": rates,
        "timing": timing,
        "passed": bool(passed),
    }


def evaluate(cases: list[dict], config: dict, error: str | None) -> dict:
    keys = [[case["seed"], case["name"]] for case in cases]
    assert keys == PLANNED[: len(keys)], "missing, duplicate or out-of-order control case"
    readings = [assess(case, config["bounds"]) for case in cases]
    complete = keys == PLANNED and error is None
    return {
        "planned_cases": 16,
        "completed_cases": len(cases),
        "complete": complete,
        "readings": readings,
        "passed_cases": sum(r["passed"] for r in readings),
        "competent_physical_control": complete and all(r["passed"] for r in readings),
        "fresh_confirmation": False,
        "brain_timing_qualified": False,
    }


def replay_actions(actions, saved, observation) -> np.ndarray:
    """Independent saved linear-policy recurrence; never invoke the original actor."""
    weights, previous = saved["weights"], saved["previous"].copy()
    assert weights.shape == (7, 2) and previous.shape == (4,)
    for action in actions:
        scores = weights[0] + observation @ weights[1:5] + weights[5 + previous]
        previous = np.argmax(scores, axis=1)
        assert previous.tolist() == action, "action differs from saved recurrent policy"
    return previous


def verify_case(case: dict, directory: Path, config: dict) -> None:
    seed, name = case["seed"], case["name"]
    parent = directory / "basis"
    probe = parent / f"seed{seed}-efference-every/probe-flipflop.npz"
    with np.load(parent / f"inputs-seed{seed}.npz", allow_pickle=False) as frozen:
        due = frozen[f"cadence/{SCHEDULES[name]}/due_ms"]
        assert case["timing"]["due_ms"] == due.tolist(), "schedule differs"
        # Every post-cue observation is identical; the clock is never a feature.
        x = frozen["window/observations"][1]
        assert np.all(frozen["window/observations"][1:] == x)
    records = case["records"]
    assert len(records) == len(due) and [r["answer"] for r in records] == case["actions"]
    assert all(r["error"] is None for r in records)
    assert case["work"] == work_totals(records), "control work differs"
    assert case["burners"] == (config["burners"] if name == "paced_regular_load" else 0)
    after = directory / f"seed{seed}-{name}-after.npz"
    with np.load(probe, allow_pickle=False) as saved, np.load(after, allow_pickle=False) as final:
        previous = replay_actions(case["actions"], saved, x)
        assert set(final.files) == set(saved.files)
        for key in saved.files:
            assert np.array_equal(final[key], previous if key == "previous" else saved[key])
    operations = case["checkpoint_io"]
    assert len(operations) == 2
    for operation, path, kind in zip(operations, (probe, after), ("read", "write"), strict=True):
        assert operation["operation"] == kind and operation["file"] == path.name
        assert operation["bytes"] == path.stat().st_size
        assert np.isfinite(operation["seconds"]) and operation["seconds"] >= 0
    begin = np.asarray(case["timing"]["begin_ms"])
    end = np.asarray(case["timing"]["end_ms"])
    assert np.all(begin[1:] >= end[:-1]), "overlapping sequential actions"
    for key, raw in (("solve_ms", end - begin), ("lateness_ms", begin - due)):
        displayed = np.asarray(case[key], dtype=float)
        assert displayed.shape == raw.shape and np.isfinite(displayed).all()
        assert np.array_equal(displayed, np.round(displayed, 3))
        # The scheduler rounds absolute-clock differences to .001ms. Subtracting
        # relative timestamps can land across the same rounding midpoint by a
        # few floating-point ulps. This checks display precision only; every
        # physical acceptance bound still uses the unchanged unrounded events.
        assert np.all(np.abs(displayed - raw) <= 0.000501)
    assert all(
        r["wall_seconds"] * 1000 <= elapsed + 1e-5
        for r, elapsed in zip(records, end - begin, strict=True)
    )
    assert np.isfinite(case["wall_seconds"]) and case["wall_seconds"] >= end[-1] / 1000
    assert np.isfinite(case["elapsed_seconds"])
    assert case["elapsed_seconds"] >= case["wall_seconds"] + sum(o["seconds"] for o in operations)
    assess(case, config["bounds"])


def verify_failed(failed: dict, directory: Path) -> None:
    assert failed["outer_timing_complete"] is False
    records = failed["records"]
    assert failed["work"] == work_totals(records)
    errors = [i for i, record in enumerate(records) if record["error"] is not None]
    assert not errors or errors == [len(records) - 1]
    parent = directory / "basis"
    seed, name = failed["seed"], failed["name"]
    with (
        np.load(parent / f"inputs-seed{seed}.npz", allow_pickle=False) as frozen,
        np.load(
            parent / f"seed{seed}-efference-every/probe-flipflop.npz", allow_pickle=False
        ) as saved,
    ):
        assert len(records) <= len(frozen[f"cadence/{SCHEDULES[name]}/due_ms"])
        replay_actions(
            [r["answer"] for r in records if r["error"] is None],
            saved,
            frozen["window/observations"][1],
        )
    operations = failed["checkpoint_io"]
    assert len(operations) <= 2
    for operation, kind in zip(operations, ("read", "write"), strict=False):
        path = (
            parent / f"seed{seed}-efference-every/probe-flipflop.npz"
            if kind == "read"
            else directory / f"seed{seed}-{name}-after.npz"
        )
        assert operation["operation"] == kind and operation["file"] == path.name
        assert operation["bytes"] == path.stat().st_size
        assert np.isfinite(operation["seconds"]) and operation["seconds"] >= 0
    assert np.isfinite(failed["elapsed_seconds"])
    assert failed["elapsed_seconds"] >= sum(r["wall_seconds"] for r in records) + sum(
        o["seconds"] for o in operations
    )


def verify(directory: Path) -> tuple[bool, str]:
    try:
        config = declaration(directory)
        recorded = Receipt.read(directory / "summary.json")
        declared = Receipt.read(directory / "declaration.json")
        assert recorded.kind == SCHEMA and recorded.source == declared.source
        source_files = [
            (item["path"], directory / item["path"]) for item in recorded.source["files"]
        ]
        valid, reason = Receipt.verify(directory / "summary.json", sources=source_files)
        assert valid, reason
        body = recorded.body
        assert body["declaration_sha256"] == chamber.sha256(directory / "declaration.json")
        marker = json.loads((directory / "run-start.json").read_text())
        assert marker["declaration_sha256"] == body["declaration_sha256"]
        assert np.isfinite(marker["started_unix_seconds"]) and marker["started_unix_seconds"] > 0
        assert body["cases"] == [
            json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()
        ]
        assert all(
            chamber.sha256(directory / name) == digest for name, digest in body["artifacts"].items()
        )
        expected = {"run-start.json", "cases.jsonl"}
        expected.update(f"seed{c['seed']}-{c['name']}-after.npz" for c in body["cases"])
        failed = body["failed_case"]
        if failed is not None:
            assert body["error"] is not None
            assert [failed["seed"], failed["name"]] == PLANNED[len(body["cases"])]
            verify_failed(failed, directory)
            after = f"seed{failed['seed']}-{failed['name']}-after.npz"
            if (directory / after).exists():
                expected.add(after)
        assert set(body["artifacts"]) == expected
        for case in body["cases"]:
            verify_case(case, directory, config)
        elapsed = body["elapsed_seconds"]
        spent = sum(c["elapsed_seconds"] for c in body["cases"])
        spent += failed["elapsed_seconds"] if failed is not None else 0
        assert np.isfinite(elapsed) and elapsed >= spent
        assert elapsed <= config["cap_seconds"] or body["error"] is not None
        assert body["assessment"] == evaluate(body["cases"], config, body["error"])
        return (
            True,
            "source/input/checkpoint custody, control arithmetic, work and all timing bounds agree",
        )
    except (OSError, ValueError, KeyError, AssertionError, TypeError, IndexError) as error:
        return False, str(error)


def run(directory: Path) -> int:
    config = declaration(directory, current=True)
    assert not (directory / "summary.json").exists(), "preserve the earlier attempt"
    assert not list(directory.glob("*-after.npz")), "preserve interrupted attempt artifacts"
    with (directory / "run-start.json").open("x") as marker:
        json.dump(
            {
                "declaration_sha256": chamber.sha256(directory / "declaration.json"),
                "started_unix_seconds": time.time(),
            },
            marker,
        )
    (directory / "cases.jsonl").touch(exist_ok=False)
    cases, error, failed = [], None, None
    began = time.perf_counter()
    try:
        for seed, name in PLANNED:
            if time.perf_counter() - began > config["cap_seconds"]:
                raise TimeoutError("declared wall-time cap reached")
            work, io = ControlWork(), chamber.CheckpointIO()
            attempt_start = time.perf_counter()
            try:
                case = measure(directory, seed, name, config, work, io)
            except Exception:
                failed = {
                    "seed": seed,
                    "name": name,
                    "records": work.records,
                    "work": work_totals(work.records),
                    "checkpoint_io": io.operations,
                    "outer_timing_complete": False,
                    "elapsed_seconds": time.perf_counter() - attempt_start,
                }
                raise
            cases.append(case)
            with (directory / "cases.jsonl").open("a") as journal:
                journal.write(json.dumps(case, allow_nan=False) + "\n")
            print(
                json.dumps(
                    {"seed": seed, "case": name, "passed": assess(case, config["bounds"])["passed"]}
                ),
                flush=True,
            )
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    # Source/input drift refuses the attempt rather than recertifying edited artifacts.
    declaration(directory, current=True)
    elapsed = time.perf_counter() - began
    if elapsed > config["cap_seconds"] and error is None:
        error = "declared wall-time cap reached"
    artifacts = ["run-start.json", "cases.jsonl"] + [p.name for p in directory.glob("*-after.npz")]
    body = {
        "declaration_sha256": chamber.sha256(directory / "declaration.json"),
        "cases": cases,
        "error": error,
        "failed_case": failed,
        "elapsed_seconds": elapsed,
        "assessment": evaluate(cases, config, error),
        "artifacts": {name: chamber.sha256(directory / name) for name in artifacts},
    }
    Receipt.build(SCHEMA, body, sources()).write(directory / "summary.json")
    valid, reason = verify(directory)
    print(
        json.dumps(
            {
                "verified": valid,
                "reason": reason,
                "error": error,
                "passed_cases": body["assessment"]["passed_cases"],
            }
        ),
        flush=True,
    )
    return 0 if valid and error is None else 3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", type=Path, metavar="ORIGINAL_RUN")
    modes.add_argument("--run", type=Path)
    modes.add_argument("--verify", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.prepare is not None:
        if args.out is None:
            parser.error("--prepare requires --out")
        prepare(args.prepare, args.out)
        return 0
    if args.run is not None:
        return run(args.run)
    valid, reason = verify(args.verify)
    print(json.dumps({"verified": valid, "reason": reason}))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
