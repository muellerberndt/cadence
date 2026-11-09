"""What a chamber pass costs on one thread: the default brain, the same with
``learning_compiled_moment=True`` and the chamber's matched transformer, in one process.

One online pass of the first grammar (4,000 batches of 16 rows by default): the brains are
``Brain.compose`` at the library's defaults on the CPU backend, taught by ``grammar.speak``; the
transformer takes one Adam step per batch (``controls.Learner``, online). The three learners
take each batch in turn and every batch's CPU seconds (``time.thread_time``) are summed per
learner, so they share whatever else the machine does. The kernels compile or load on copies
that live the first 20 batches before the clock starts. Pin the process to one core
(``taskset -c 3`` on Linux) and set one thread per library:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src \\
        python benchmarks/grammar/cost.py --founder 0 --out DIR/cost-f0.json

``torch`` is required for the transformer alone; ``--learners plain,compiled`` leaves it out.

``--parts plain`` (or ``compiled``) lives one brain's pass alone and reads where its time goes:
the functions of ``PARTS`` are timed, each call's seconds less those of the timed calls inside it
charged to its part; the kernels are timed at the functions that launch them. A timer costs about
a microsecond a call, a few tenths of a second over a pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import grammar as gr
import numpy as np

LEARNERS = ("plain", "compiled", "transformer")
TARGET = 24384  # the parameters the chamber's controls match (the recipe's brain)
WARM = 20  # the batches a copy of each brain lives before the clock starts


def make(name: str, founder: int) -> Any:
    """One learner of the pass."""
    if name == "transformer":
        import controls as ct

        return ct.Learner("transformer", ct.matched_width("transformer", TARGET), founder,
                          ct.RATES["online"])
    import cadence as cd

    settings = {"learning_compiled_moment": True} if name == "compiled" else {}
    return cd.Brain.compose(gr.INPUTS, gr.ACTIONS, modules=(64,), seed=founder, backend="cpu",
                            **settings)


def _warm(brain: Any, stream: list[Any], batches: int = WARM) -> None:
    """A copy of ``brain`` lives the first batches, so every kernel the pass calls compiles or
    loads from Numba's cache before the clock starts."""
    fork = copy.deepcopy(brain)
    for batch in stream[:batches]:
        gr.speak(fork, batch, teach=True)


def run(founder: int, batches: int, learners: tuple[str, ...] = LEARNERS,
        every: int = 500) -> dict[str, Any]:
    """One pass of every learner, batch by batch in turn; the seconds per learner, the brains'
    work and a mark every ``every`` batches."""
    warnings.simplefilter("ignore")
    if "transformer" in learners:
        import torch

        torch.set_num_threads(1)
    made = {name: make(name, founder) for name in learners}
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    stream = [gr.train_batch(rng, split, "g1") for _ in range(batches)]
    for name, learner in made.items():  # the kernels compile or load outside the clock
        if name != "transformer":
            _warm(learner, stream)
    seconds = dict.fromkeys(learners, 0.0)
    work = {name: gr.new_work() for name in learners if name != "transformer"}
    marks = []
    for i, batch in enumerate(stream, 1):
        for name, learner in made.items():
            started = time.thread_time()
            if name == "transformer":
                learner.update(batch)
            else:
                gr.speak(learner, batch, teach=True, work=work[name])
            seconds[name] += time.thread_time() - started
        if i % every == 0 or i == batches:
            marks.append({"batches": i, **seconds})
    out: dict[str, Any] = {"founder": founder, "batches": batches, "seconds": seconds,
                           "work": work, "marks": marks}
    if "transformer" in seconds:
        out["against_transformer"] = {
            name: seconds[name] / seconds["transformer"] for name in seconds
        }
    return out


# The timed functions, each with its part: the kernels at the functions that launch them, then the
# parts of a pass around them.
PARTS = (
    ("cadence.fused", "_fast_solve", "kernels: the solves"),
    ("cadence.fused", "fused_solve", "kernels: the solves"),
    ("cadence.fused", "fused_settle", "kernels: the lessons' phases"),
    ("cadence.fused", "fused_residual", "kernels: the checks"),
    ("cadence.generic", "Brain._stimulus", "the read"),
    ("cadence.generic", "Brain._compiled_drive", "the read"),
    ("cadence.stream", "Recall.learn", "the address lesson"),
    ("cadence.stream", "Hearing.learn", "the carried gains' lesson"),
    ("cadence.learning", "Learner.update", "the lessons' update"),
    ("cadence.learning", "Learner.step", "the lessons' phases around their kernels"),
    ("cadence.generic", "Brain._qualified", "the solves around their kernels"),
    ("cadence.generic", "Brain._choose", "the choice and the write"),
    ("cadence.generic", "Brain.lesson", "the rest of the lesson"),
    ("cadence.generic", "Brain.act", "the rest of the act"),
)
OUTSIDE = "the chamber's own loop"


def _timed(spent: Counter[str], stack: list[float], part: str, function: Any) -> Any:
    clock = time.perf_counter
    # a settle of no sweeps publishes a solve's start and launches no kernel
    publish = "the solves around their kernels" if function.__name__ == "fused_settle" else None

    def timed(*args: Any, **kwargs: Any) -> Any:
        stack.append(0.0)
        started = clock()
        try:
            return function(*args, **kwargs)
        finally:
            took = clock() - started
            inside = stack.pop()
            spent[publish if publish and args[9] == 0 else part] += took - inside
            if stack:
                stack[-1] += took

    timed.__wrapped__ = function  # type: ignore[attr-defined]
    return timed


def parts(name: str, founder: int, batches: int) -> dict[str, Any]:
    """One brain's pass alone, its seconds charged to the parts of ``PARTS``; the functions are
    restored after it."""
    import importlib

    warnings.simplefilter("ignore")
    brain = make(name, founder)
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    stream = [gr.train_batch(rng, split, "g1") for _ in range(batches)]
    _warm(brain, stream)
    spent: Counter[str] = Counter()
    stack: list[float] = []
    held = []
    for module_name, path, part in PARTS:
        owner: Any = importlib.import_module(module_name)
        *outer, attribute = path.split(".")
        for step in outer:
            owner = getattr(owner, step)
        original = (
            owner.__dict__[attribute] if isinstance(owner, type) else getattr(owner, attribute)
        )
        held.append((owner, attribute, original))
        setattr(owner, attribute, _timed(spent, stack, part, original))
    started = time.perf_counter()
    try:
        for batch in stream:
            gr.speak(brain, batch, teach=True)
    finally:
        for owner, attribute, original in held:
            setattr(owner, attribute, original)
    wall = time.perf_counter() - started
    spent[OUTSIDE] += wall - sum(spent.values())
    return {"learner": name, "founder": founder, "batches": batches, "wall": wall,
            "parts": dict(spent.most_common())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--founder", type=int, default=0)
    parser.add_argument("--batches", type=int, default=4000)
    parser.add_argument("--learners", default=",".join(LEARNERS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--parts", choices=("plain", "compiled"), default=None,
                        help="one brain's pass alone, by part")
    args = parser.parse_args(argv)
    if args.parts:
        found = parts(args.parts, args.founder, args.batches)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(found, indent=1))
        for part, seconds in found["parts"].items():
            print(f"{seconds:8.1f} s  {part}")
        return 0
    learners = tuple(name for name in args.learners.split(",") if name)
    unknown = set(learners) - set(LEARNERS)
    if unknown:
        parser.error(f"unknown learners: {sorted(unknown)}")
    result = run(args.founder, args.batches, learners)
    text = json.dumps(result, indent=1, default=float)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(json.dumps({"seconds": result["seconds"],
                      "against_transformer": result.get("against_transformer")}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
