"""The ceiling of a first guess (cadence#290, Step 0): free solves that start from the state the
same input settled to the last time it was seen, against the default start.

A free solve of the default brain starts warm, from the stream's activity (rest after a reset).
The oracle start keeps, per observation row, the state the act on that observation last settled
to, and starts every free solve of a row whose observation it has seen from it: the act's, the
judging solve of a surprise lesson and the lesson's free phase. Rows it has not seen start as
the default starts them. The ``context`` start keys the same states on the observation and the
one the stream heard before it (a reset hears none), a guess that sees one moment of context. It
reads what a first guess learned from the brain's own settled
states could save at best (a guess that maps an input to its last settled state without error);
it is an instrument and no option of the library.

Two workloads, each at the association widths ``--widths``:

- ``chamber``: one online pass of the grammar chamber's first grammar (``grammar.speak`` with
  lessons, ``--batches`` batches of sixteen), the brain ``Brain.compose(INPUTS, ACTIONS,
  modules=(width,))`` on the CPU backend;
- ``battery``: the sequence battery's tasks ``--tasks`` (``--lessons`` training episodes and
  ``--tests`` frozen test episodes each), the arm ``default`` (what a new brain composes).

Each life is lived twice, once per start, from the same founder and the same batches. Counted
per life: the row sweeps of every free solve (the acts' and the judging solves', each solve
once, and the lessons' free phases), the row sweeps of the lessons' two nudged phases (which
start from the free state and run a fixed count), and the CPU seconds of the life
(``time.thread_time``). The oracle's own bookkeeping is inside those seconds.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src \\
        python benchmarks/settle/oracle.py --workload chamber --founders 0 --widths 64 \\
        --out DIR/oracle.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
for _folder in ("grammar", "sequence"):
    _path = str(HERE.parent / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

STARTS = ("default", "oracle", "context")
TASKS = ("lookback", "induction", "nback", "twohop", "repeat", "periodic")


def new_count() -> dict[str, float]:
    return {"solves": 0, "free_row_sweeps": 0, "lessons": 0, "lesson_free_row_sweeps": 0,
            "nudged_row_sweeps": 0, "refused": 0, "exhausted": 0, "oracle_rows": 0,
            "rows": 0}


class Counted:
    """Counts the work of every solve a brain reports, and with ``oracle`` starts every free
    solve of a seen observation row from the state that row's act last settled to.

    The brain's class is swapped for a subclass whose ``_last_settlement`` counts each report it
    is given once (a judged solve read again by the act that follows is the same solve); the
    instance's ``act``, ``lesson`` and ``_activity`` are wrapped."""

    def __init__(self, brain: Any, start: str) -> None:
        self.brain, self.oracle, self.context = brain, start != "default", start == "context"
        self.previous: np.ndarray | None = None
        self.count = new_count()
        self.held: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        self.x: np.ndarray | None = None
        self.made: tuple[Any, bytes, Any] | None = None
        self.reread = False
        counter = self

        def get(owner: Any) -> Any:
            return owner.__dict__.get("_counted_settlement")

        def put(owner: Any, report: Any) -> None:
            owner.__dict__["_counted_settlement"] = report
            if report is None:
                return
            if counter.reread:
                counter.reread = False
                return
            counter.count["solves"] += 1
            counter.count["free_row_sweeps"] += float(report.get("row_sweeps", 0))

        base = type(brain)
        brain.__dict__["_counted_settlement"] = brain.__dict__.pop("_last_settlement", None)
        brain.__class__ = type(base.__name__, (base,), {
            "_last_settlement": property(get, put), "__module__": base.__module__,
        })
        activity, act, lesson, judged, reset = (
            brain._activity, brain.act, brain.lesson, brain._read_judged, brain.reset
        )

        def cleared(*args: Any, **kwargs: Any) -> Any:
            counter.previous = None
            return reset(*args, **kwargs)

        def read_judged(drive: Any, state: Any, operation: str) -> Any:
            found = judged(drive, state, operation)
            if found is not None:
                counter.reread = True  # the report it sets is the judging solve's, counted
            return found

        def warm() -> Any:
            return counter.start(activity())

        def acted(x: Any, *args: Any, **kwargs: Any) -> Any:
            counter.x = np.asarray(x, dtype=float)
            try:
                action = act(x, *args, **kwargs)
            except RuntimeError:
                counter.x = None
                raise
            counter.x = None
            counter.keep(np.asarray(x, dtype=float))
            return action

        def taught(x: Any, *args: Any, **kwargs: Any) -> Any:
            counter.x = np.asarray(x, dtype=float)
            try:
                report = lesson(x, *args, **kwargs)
            finally:
                counter.x = None
            counter.count["lessons"] += 1
            counter.count["lesson_free_row_sweeps"] += float(report.get("free_row_sweeps", 0))
            counter.count["nudged_row_sweeps"] += float(
                report.get("nudged_row_sweeps", 0) + report.get("opposite_row_sweeps", 0)
            )
            return report

        brain._read_judged = read_judged
        brain._activity = warm
        brain.act = acted
        brain.lesson = taught
        brain.reset = cleared

    def _key(self, x: np.ndarray, i: int) -> bytes:
        """A row's key: its observation, with ``context`` the one heard before it too."""
        if not self.context:
            return x[i].tobytes()
        before = self.previous
        heard = b"" if before is None or len(before) != len(x) else before[i].tobytes()
        return x[i].tobytes() + b"|" + heard

    def start(self, state: Any) -> Any:
        """The warm start of a free solve on the observations of the moment: the stream's
        state, with each seen row's last settled state in place of its own. One moment's
        lesson and act read one start object, as a judged solve's reuse requires."""
        if not self.oracle or self.x is None:
            return state
        x = self.x
        key = x.tobytes()
        if self.made is not None and self.made[0] is state and self.made[1] == key:
            return self.made[2]
        from cadence.brain import BrainState

        rows, n = len(x), self.brain.connectome.n
        if state is None or len(np.atleast_2d(state.v)) != rows:
            v, a, steps = np.zeros((rows, n)), np.zeros((rows, n)), 0
        else:
            v = np.array(np.atleast_2d(state.v), dtype=float, copy=True)
            a = np.array(np.atleast_2d(state.adaptation), dtype=float, copy=True)
            steps = state.steps
        seen = 0
        for i in range(rows):
            found = self.held.get(self._key(x, i))
            if found is not None:
                v[i], a[i] = found
                seen += 1
        self.count["rows"] += rows
        self.count["oracle_rows"] += seen
        made = state if not seen else BrainState(
            v, self.brain.brain.neuron_model.activation(v), a, steps
        )
        self.made = (state, key, made)
        return made

    def keep(self, x: np.ndarray) -> None:
        """Each row's settled state after its act, under its observation."""
        if not self.oracle:
            return
        state = self.brain.basal_ganglia.state
        v = a = np.zeros((0, 0))
        if state is not None:
            v, a = np.atleast_2d(state.v), np.atleast_2d(state.adaptation)
        if len(v) == len(x):
            for i in range(len(x)):
                self.held[self._key(x, i)] = (v[i].copy(), a[i].copy())
        self.previous = x


def chamber_life(founder: int, width: int, start: str, batches: int) -> dict[str, Any]:
    import grammar as gr

    import cadence as cd

    brain = cd.Brain.compose(gr.INPUTS, gr.ACTIONS, modules=(width,), seed=founder,
                             backend="cpu")
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    stream = [gr.train_batch(rng, split, "g1") for _ in range(batches)]
    counted = Counted(brain, start)
    work = gr.new_work()
    began = time.thread_time()
    for batch in stream:
        gr.speak(brain, batch, teach=True, work=work)
    seconds = time.thread_time() - began
    counted.count["refused"] += work["refused"]
    counted.count["exhausted"] += work["exhausted"]
    return {"workload": "chamber", "founder": founder, "width": width, "start": start,
            "batches": batches, "parameters": int(brain.parameters()), "seconds": seconds,
            **counted.count}


def battery_life(founder: int, width: int, start: str, task_name: str, lessons: int,
                 tests: int) -> dict[str, Any]:
    import battery as bt

    task = bt.TASKS[task_name]()
    brain = bt.make_brain(task, "default", founder, None, width)
    counted = Counted(brain, start)
    rng = np.random.default_rng(np.random.SeedSequence([founder, bt.SEEDS["train"]]))
    stats = bt.new_stats()
    began = time.thread_time()
    for _ in range(lessons):
        bt.run_episode(brain, task, task.episode(rng, train=True), teach=True, closed=False,
                       stats=stats)
    test = bt.evaluate(brain, task, np.random.default_rng(
        np.random.SeedSequence([founder, bt.SEEDS["test"]])), tests, task.name == "repeat")
    seconds = time.thread_time() - began
    counted.count["refused"] += stats["refused"]
    counted.count["exhausted"] += stats["exhausted"] + test["exhausted"]
    return {"workload": "battery", "task": task_name, "founder": founder, "width": width,
            "start": start, "lessons": lessons, "tests": tests,
            "parameters": int(brain.parameters()), "seconds": seconds,
            "accuracy": test["accuracy"], **counted.count}


def life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life of ``spec``, after a short one of the same kind outside its clock: the kernels it
    calls compile or load from Numba's cache before its seconds count."""
    warnings.simplefilter("ignore")
    if not spec.get("warm"):
        life({**spec, "warm": True, "batches": 3, "lessons": 3, "tests": 2})
    if spec["workload"] == "chamber":
        return chamber_life(spec["founder"], spec["width"], spec["start"], spec["batches"])
    return battery_life(spec["founder"], spec["width"], spec["start"], spec["task"],
                        spec["lessons"], spec["tests"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload", choices=("chamber", "battery"), required=True)
    parser.add_argument("--founders", nargs="*", type=int, default=[0])
    parser.add_argument("--widths", nargs="*", type=int, default=[64, 256])
    parser.add_argument("--starts", nargs="*", default=list(STARTS), choices=STARTS)
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--lessons", type=int, default=400)
    parser.add_argument("--tests", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    specs: list[dict[str, Any]] = []
    for founder in args.founders:
        for width in args.widths:
            for start in args.starts:
                if args.workload == "chamber":
                    specs.append({"workload": "chamber", "founder": founder, "width": width,
                                  "start": start, "batches": args.batches})
                else:
                    specs.extend({"workload": "battery", "founder": founder, "width": width,
                                  "start": start, "task": task, "lessons": args.lessons,
                                  "tests": args.tests} for task in args.tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as out:
        if args.workers > 1:
            with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
                for row in pool.imap_unordered(life, specs):
                    out.write(json.dumps(row) + "\n")
                    out.flush()
        else:
            for spec in specs:
                out.write(json.dumps(life(spec)) + "\n")
                out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
