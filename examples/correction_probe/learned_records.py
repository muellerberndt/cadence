"""Bounded learned-routine integration check for private RecordSession.

No correction efficacy, automatic learning or observer advantage is tested.
"""

import argparse
import gzip
import hashlib
import json
import math
import signal
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import cadence
from cadence import Cortex
from cadence import _incremental as I
from cadence import _repair as R
from cadence._records import RecordSession
from cadence.brain import IMPLEMENTATION

KINDS = ("ordinary", "observer")
SEEDS = (31, 37, 43)
FAMILIAR = tuple((a, b) for a in (-0.4, -0.2, 0.2, 0.4) for b in (-0.3, 0.3))
HELDOUT = ((0.1, -0.2), (-0.3, 0.1), (0.3, 0.2), (-0.1, -0.25))
DESIGN = {
    "schema": "learned-temporal-record-integration/1",
    "final_goal": "A patch-net equilibrium brain that is more scalable, more capable and more efficient than a transformer. Every Cadence result is measured against that goal at matched information, matched task and a declared resource model. The Amen jungle composer is the first test platform.",
    "principles": "The main hypothesis stays: a brain that settles in global equilibria. The building block stays as simple as possible, like in nature, and every part of the brain answers with a settled state of that same patch rule; a feed-forward readout or a copied input is a baseline, never a result. Natural evolution is preferred to design: any parameter that looks designed is a gene, picked by selection against the hand-set value, which stays as the control. Within a life only the patch rule learns; across lives the genome evolves.",
    "core_commit": "7628762964b5cfaed2af38d8f3bc5b63edeef336",
    "kinds": KINDS,
    "seeds": SEEDS,
    "familiar": FAMILIAR,
    "heldout": HELDOUT,
    "training": {"updates": 4, "batch": 8, "budget": 512, "source": "witness"},
    "session": {
        "budget": 2048,
        "seconds": 5.0,
        "surprise_threshold": 0.05,
        "goal_tolerance": 0.0,
    },
    "gates": {
        "heldout_mae": 0.05,
        "newborn_relative": 0.5,
        "quiet_slow_evaluations": 0,
        "quiet_slow_jobs": 0,
        "quiet_new_cache_constructors": 0,
    },
    "shift": {"inputs": HELDOUT[0], "add_to_actual_future": 0.2},
    "goal": {"inputs": HELDOUT[0], "required_value": 0.8, "terminal": True},
    "receipt_encoding": "Lossless encoding of the collector summary in result.json.gz plus compact result.json; native solve/ticket/ACK payloads are represented by hashes, not full replay evidence.",
    "outer_seconds": 120,
    "source_plus_output_bytes": 100_000,
    "information": "Same original two sensory inputs. Two appended owner-only actual-outcome input slots have NO graph contacts. Actual equation values are ACKs, never substituted forecasts. No observation-to-slow efficacy path is claimed.",
    "scope": "Development runtime check on an existing learned twelve-patch fixture. RecordSession lowers cross-block live contacts into explicit historical ports, changing the objective. It tests retained acquired accuracy, zero slow work on changing familiar inputs and causal wake requests, not useful slow correction, consolidation, native body control or all-three integration.",
    "branches": "Main owner:8 familiar then4 heldout actual ACK cycles. Shift and terminal-goal owners are separately reconstructed from identical acquired parameters/activity and genesis, with all startup work counted; they are not a public owner snapshot/fork API.",
    "selection": "All six fixed cases retained, no threshold/dose/seed changes or retry after failure.",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(canonical(value) + "\n")
    tmp.replace(path)


def pins():
    directory = Path(cadence.__file__).resolve().parent
    return {
        str(p): sha(p)
        for p in [
            Path(__file__).resolve(),
            *(directory / name for name in IMPLEMENTATION),
            *(
                directory / name
                for name in ("_records.py", "_attention.py", "_incremental.py")
            ),
            directory.parent.parent / "tests/test_attention.py",
            directory.parent.parent / "tests/test_records.py",
        ]
    }


def actual(values):
    a, b = values
    return {
        "physical": (math.tanh(0.6 * a + 0.4 * b), math.tanh(-0.5 * a + 0.3 * b)),
        "future": (math.tanh(0.3 * a + 0.5 * b), math.tanh(-0.4 * a + 0.2 * b)),
    }


def make(kind, seed):
    c = Cortex(seed=seed)
    u = c.input("senses", shape=2)
    p = c.column("physical_relation", patches=2, inputs=u)
    h = c.column("representation", patches=4, inputs=(u, p))
    m = (
        c.observer("readback", patches=4, observes=h)
        if kind == "observer"
        else c.column("readback", patches=4, inputs=h)
    )
    f = c.column("future_relation", patches=2, inputs=(u, m))
    c.output("physical", shape=2, reads=p)
    c.output("future", shape=2, reads=f)
    return c.build()


class Instrument:
    """Count actual cache construction and incremental tanh evaluations only."""

    def __init__(self):
        self.counts, self.lock = Counter(), threading.Lock()

    def __enter__(self):
        self.original_init, self.original_math = I.ActivityCache.__init__, I.math

        def init(instance, *args, **kwargs):
            with self.lock:
                self.counts["cache_constructors"] += 1
            try:
                result = self.original_init(instance, *args, **kwargs)
            except BaseException:
                with self.lock:
                    self.counts["constructor_unknown_work"] += 1
                raise
            with self.lock:
                self.counts["cache_constructor_returns"] += 1
                self.counts["constructor_full_prediction_evaluations"] += args[
                    0
                ].n_patches
                self.counts["constructor_slow_prediction_evaluations"] += 4
            return result

        def tanh(value):
            target = sys._getframe(1).f_locals["target"]
            with self.lock:
                self.counts["incremental_prediction_evaluations"] += 1
                self.counts["incremental_slow_prediction_evaluations"] += (
                    6 <= target < 10
                )
            return math.tanh(value)

        I.ActivityCache.__init__ = init
        I.math = SimpleNamespace(tanh=tanh, fsum=math.fsum, isfinite=math.isfinite)
        return self

    def __exit__(self, *_):
        I.ActivityCache.__init__, I.math = self.original_init, self.original_math

    def snapshot(self):
        with self.lock:
            return dict(self.counts)


def difference(after, before):
    return {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)}


def reference(session):
    active = session._active
    cache = active._cache
    started = time.perf_counter()
    visits = Counter()
    evaluated = R._evaluate(
        session.graph,
        cache.inputs,
        cache.state,
        session.weights,
        session.biases,
        active.state_prior,
        None,
        None,
        0.1,
        visits=visits,
        parameter_gradients=False,
    )
    residual = R._stationarity(
        cache.state,
        session.weights,
        session.biases,
        evaluated,
        {},
        False,
        active.state_bound,
        active.parameter_bound,
    )
    result = session.result()
    assert not active._clamps and result["qualified"]
    assert result["state"] == cache.state
    raw = active.result()
    assert raw["predictions"] == evaluated["predictions"]
    assert raw["errors"] == evaluated["errors"]
    assert result["energy"] == evaluated["energy"]
    assert result["stationarity"] == residual <= active.tolerance
    assert all(math.isfinite(v) and abs(v) <= active.state_bound for v in cache.state)
    return {
        "stationarity": residual,
        "energy": result["energy"],
        "state_sha256": digest(cache.state),
        "evaluations": 1,
        "prediction_evaluations": session.graph.n_patches,
        "work": dict(visits),
        "seconds": time.perf_counter() - started,
        "scope": "Separate reference validation work, excluded from runtime sleeping counts but explicitly charged.",
    }


def compact(result):
    return {
        k: result[k]
        for k in (
            "qualified",
            "stationarity",
            "energy",
            "record_version",
            "record_event",
            "counts",
            "work",
            "jobs",
            "unknown_work",
            "work_complete",
            "seconds",
        )
    }


def create(brain):
    assert brain.graph.n_inputs == 2 and brain.graph.n_patches == 12
    graph = R.Graph(4, 12, brain.graph.edges)
    assert not any(kind == "input" and source >= 2 for kind, source, _ in graph.edges)
    config = {
        k: brain.config[k]
        for k in (
            "state_prior",
            "state_bound",
            "parameter_bound",
            "tolerance",
            "step",
            "backtracks",
        )
    }
    return RecordSession(
        graph,
        (0, 1, 2, 3, 4, 5, 10, 11),
        (0.0,) * 4,
        brain.state,
        brain.weights,
        brain.biases,
        forecast_indices=(10, 11),
        outcome_inputs=(2, 3),
        value_index=10,
        **DESIGN["session"],
        **config,
    )


def public_checks(brain, owned):
    work = Counter()
    snapshot = brain.snapshot()
    started = time.perf_counter()
    owned.update(rows=[], work={}, status="started", unknown_calls=0)
    try:
        for values in HELDOUT:
            record = {
                "inputs": values,
                "actual": actual(values)["future"],
                "status": "started",
            }
            owned["rows"].append(record)
            try:
                result = brain.settle({"senses": values}, budget=512)
            except BaseException:
                record["status"] = "interrupted_or_error"
                owned["unknown_calls"] += 1
                raise
            work.update(result["work"])
            owned["work"] = dict(work)
            record.update(
                status="returned",
                forecast=result["outputs"]["future"],
                qualified=result["qualified"],
                stationarity=result["stationarity"],
                result_sha256=digest(result),
                work=result["work"],
            )
            assert result["qualified"] and brain.snapshot() == snapshot
        owned["mae"] = (
            sum(
                abs(a - b)
                for r in owned["rows"]
                for a, b in zip(r["forecast"], r["actual"], strict=True)
            )
            / 8
        )
        owned["status"] = "complete"
    finally:
        owned["seconds"] = time.perf_counter() - started
    return owned


def drain(session):
    if session._pending is None:
        return None
    session._pending[1].result(timeout=6)
    return session.poll()


def session_branch(brain, name, meter):
    starting = meter.snapshot()
    started = time.perf_counter()
    session = None
    report = {"status": "started", "branch": name, "references": [], "rows": []}
    try:
        session = create(brain)
        report["genesis"] = asdict(session._record)
        report["startup"] = compact(session.result())
        report["graph"] = {
            "original": session.graph_sha256,
            "lowered": session.lowered_sha256,
            "ports": session.ports,
        }
        report["references"].append(reference(session))
        startup_end = meter.snapshot()
        report["startup_instrumentation"] = difference(startup_end, starting)
        values_list = (*FAMILIAR, *HELDOUT) if name == "routine" else (HELDOUT[0],)
        for index, values in enumerate(values_list):
            kwargs = {"required_value": 0.8} if name == "goal" else {}
            row_record = {"index": index, "inputs": values, "status": "started"}
            report["rows"].append(row_record)
            issued = session.step(values, **kwargs)
            row_record.update(
                status="returned",
                qualified=issued["qualified"],
                reason=issued.get("reason"),
            )
            assert issued["qualified"]
            report["references"].append(reference(session))
            truth = actual(values)["future"]
            observed = tuple(x + 0.2 for x in truth) if name == "shift" else truth
            ack = session.acknowledge(
                issued["ticket"],
                observed,
                **(
                    {"task_value": truth[0], "terminal": True} if name == "goal" else {}
                ),
            )
            row_record.update(
                {
                    "index": index,
                    "inputs": values,
                    "forecast": issued["forecast"],
                    "actual": observed,
                    "unshifted_actual": truth,
                    "surprise": ack["surprise"],
                    "goal_deficit": ack["goal_deficit"],
                    "wake": ack["wake"],
                    "ticket_sha256": digest(asdict(issued["ticket"])),
                    "ack_sha256": digest(ack),
                }
            )
            # Drain every requested job; quiet cycles start none. This adds no
            # intervention and prevents unfinished work being claimed as zero.
            committed = drain(session)
            if committed is not None:
                report["rows"][-1]["proposal_status"] = committed
                report["references"].append(reference(session))
        report["final"] = compact(session.result())
        report["ledger_sha256"] = digest(session.ledger())
        report["after_startup_instrumentation"] = difference(
            meter.snapshot(), startup_end
        )
        if name == "routine":
            rows = report["rows"][-4:]
            report["heldout_mae"] = (
                sum(
                    abs(a - b)
                    for r in rows
                    for a, b in zip(r["forecast"], r["actual"], strict=True)
                )
                / 8
            )
            delta = report["after_startup_instrumentation"]
            report["sleeps"] = (
                not any(r["wake"] for r in report["rows"])
                and report["final"]["counts"]["requests"] == 0
                and report["final"]["jobs"]["foreground"][1] == 0
                and delta.get("cache_constructors", 0) == 0
                and delta.get("incremental_slow_prediction_evaluations", 0) == 0
            )
        else:
            row = report["rows"][0]
            report["wake_gate"] = (
                row["wake"] and report["final"]["counts"]["requests"] == 1
            )
            if name == "shift":
                report["wake_gate"] &= (
                    row["surprise"] > 0.05 and not row["goal_deficit"]
                )
            else:
                report["wake_gate"] &= row["surprise"] <= 0.05 and row["goal_deficit"]
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        report["unknown_work"] = session is None or bool(
            report["rows"] and report["rows"][-1]["status"] == "started"
        )
    finally:
        if session is not None:
            try:
                report["drained"] = compact(session.close())
            except BaseException:
                report["status"] = "failed"
                report["unknown_work"] = True
                report["close_exception"] = traceback.format_exc()
        report["seconds"] = time.perf_counter() - started
        report["total_instrumentation"] = difference(meter.snapshot(), starting)
    return report


def run_case(kind, seed, expected_founder):
    started = time.perf_counter()
    brain = make(kind, seed)
    assert digest(brain.snapshot()) == expected_founder
    report = {
        "kind": kind,
        "seed": seed,
        "status": "started",
        "founder_sha256": digest(brain.snapshot()),
        "training": [],
        "branches": {},
    }
    try:
        report["newborn"] = {}
        public_checks(brain, report["newborn"])
        rows = [({"senses": values}, actual(values)) for values in FAMILIAR]
        for event in range(4):
            before = brain.snapshot()
            t = time.perf_counter()
            report["admission_intent"] = {
                "event": event,
                "before": digest(before),
                "status": "started",
            }
            taught = brain.observe_batch(
                rows, event_id=event, source="witness", budget=512
            )
            report["training"].append(
                {
                    "event": event,
                    "accepted": taught["accepted"],
                    "qualified": taught["qualified"],
                    "before": digest(before),
                    "after": digest(brain.snapshot()),
                    "result_sha256": digest(taught),
                    "work": taught["work"],
                    "seconds": time.perf_counter() - t,
                }
            )
            report["admission_intent"]["status"] = "returned"
            assert taught["accepted"] and taught["qualified"]
            assert json.loads(before)["state"] == json.loads(brain.snapshot())["state"]
        report["acquired"] = {}
        public_checks(brain, report["acquired"])
        report["acquisition_gate"] = (
            report["acquired"]["mae"] <= 0.05
            and report["acquired"]["mae"] <= 0.5 * report["newborn"]["mae"]
        )
        report["model"] = json.loads(brain.snapshot())
        snapshot = brain.snapshot()
        with Instrument() as meter:
            for name in ("routine", "shift", "goal"):
                report["branches"][name] = session_branch(brain, name, meter)
            report["total_runtime_instrumentation"] = meter.snapshot()
        assert brain.snapshot() == snapshot
        b = report["branches"]
        report["all_branches_complete"] = all(
            r["status"] == "complete" for r in b.values()
        )
        report["integration_gate"] = bool(
            report["all_branches_complete"]
            and report["acquisition_gate"]
            and b["routine"]["heldout_mae"] <= 0.05
            and b["routine"]["heldout_mae"] <= 0.5 * report["newborn"]["mae"]
            and b["routine"]["sleeps"]
            and b["shift"]["wake_gate"]
            and b["goal"]["wake_gate"]
            and all(
                not r["drained"]["unknown_work"] and r["drained"]["work_complete"]
                for r in b.values()
            )
        )
        report["status"] = "complete" if report["all_branches_complete"] else "failed"
    except BaseException:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
    report["unknown_admission_work"] = (
        report.get("admission_intent", {}).get("status") == "started"
    )
    report["seconds"] = time.perf_counter() - started
    return report


def write_detail(root, value):
    path = root / "result.json.gz"
    tmp = root / "result.json.gz.tmp"
    with gzip.open(tmp, "wt") as stream:
        stream.write(canonical(value) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "check"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "check":
        for kind in KINDS:
            b = make(kind, 31)
            assert b.graph.n_patches == 12 and b.graph.n_inputs == 2
            assert tuple(
                p["indices"]
                for p in b.inspect()["populations"]
                if p["name"] == "readback"
            ) == ((6, 7, 8, 9),)
        assert len(FAMILIAR) == 8 and len(HELDOUT) == 4
        print("No-query construction/protocol checks passed")
        return
    if args.command == "prepare":
        if root.exists():
            raise ValueError("Preserve prior attempt")
        root.mkdir(parents=True)
        write(
            root / "protocol.json",
            {
                "design": DESIGN,
                "sources": pins(),
                "jobs": [(k, s) for k in KINDS for s in SEEDS],
                "founder_sha256": {
                    f"{k}-{s}": digest(make(k, s).snapshot())
                    for k in KINDS
                    for s in SEEDS
                },
            },
        )
        print(sha(root / "protocol.json"))
        return
    p = json.loads((root / "protocol.json").read_text())
    assert canonical(p["design"]) == canonical(DESIGN) and p["sources"] == pins()
    if (root / "result.json").exists() or (root / "result.json.gz").exists():
        raise ValueError("Preserve prior run")
    jobs = [
        {"kind": kind, "seed": seed, "status": "not_started"}
        for kind in KINDS
        for seed in SEEDS
    ]
    write_detail(root, {"status": "started", "cases": jobs})
    started = time.perf_counter()

    def cap(*_):
        raise TimeoutError("frozen outer elapsed cap")

    signal.signal(signal.SIGALRM, cap)
    signal.setitimer(signal.ITIMER_REAL, DESIGN["outer_seconds"])
    for i, job in enumerate(jobs):
        if time.perf_counter() - started >= DESIGN["outer_seconds"]:
            break
        jobs[i]["status"] = "started"
        write_detail(root, {"status": "running", "cases": jobs})
        jobs[i] = run_case(
            job["kind"],
            job["seed"],
            p["founder_sha256"][f"{job['kind']}-{job['seed']}"],
        )
        write_detail(root, {"status": "running", "cases": jobs})
    signal.setitimer(signal.ITIMER_REAL, 0)
    result = {
        "status": "complete",
        "cases": jobs,
        "protocol_sha256": sha(root / "protocol.json"),
        "sources_unchanged": pins() == p["sources"],
        "seconds": time.perf_counter() - started,
        "scope": DESIGN["scope"],
        "useful_correction_demonstrated": False,
        "all_three_capabilities": False,
    }
    result["complete"] = all(j["status"] == "complete" for j in jobs)
    result["integration_gate"] = (
        result["complete"]
        and result["sources_unchanged"]
        and all(j["integration_gate"] for j in jobs)
        and result["seconds"] < 120
    )
    result["behavior_gate_before_storage"] = result.pop("integration_gate")
    write_detail(root, result)
    overview = {
        "complete": result["complete"],
        "integration_gate": result["behavior_gate_before_storage"],
        "sources_unchanged": result["sources_unchanged"],
        "seconds": result["seconds"],
        "protocol_sha256": result["protocol_sha256"],
        "detail_sha256": sha(root / "result.json.gz"),
        "useful_correction_demonstrated": False,
        "all_three_capabilities": False,
        "cases": [
            {
                "kind": j["kind"],
                "seed": j["seed"],
                "status": j["status"],
                "acquisition_gate": j.get("acquisition_gate", False),
                "integration_gate": j.get("integration_gate", False),
                "newborn_mae": j.get("newborn", {}).get("mae"),
                "acquired_public_mae": j.get("acquired", {}).get("mae"),
                "record_heldout_mae": j.get("branches", {})
                .get("routine", {})
                .get("heldout_mae"),
                "sleeps": j.get("branches", {}).get("routine", {}).get("sleeps", False),
                "shift_wake": j.get("branches", {})
                .get("shift", {})
                .get("wake_gate", False),
                "quiet_failed_goal_wake": j.get("branches", {})
                .get("goal", {})
                .get("wake_gate", False),
            }
            for j in jobs
        ],
        "scope": DESIGN["scope"],
    }
    write(root / "result.json", overview)
    overview["source_plus_output_bytes"] = Path(__file__).stat().st_size + sum(
        f.stat().st_size for f in root.iterdir() if f.is_file()
    )
    overview["storage_gate"] = overview["source_plus_output_bytes"] + 256 <= 100_000
    overview["integration_gate"] &= overview["storage_gate"]
    write(root / "result.json", overview)
    print(canonical(overview))


if __name__ == "__main__":
    main()
