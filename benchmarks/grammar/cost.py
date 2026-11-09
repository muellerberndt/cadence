"""What a chamber pass costs on one thread: the default brain, the same with
``learning_compiled_moment=True`` and the chamber's matched transformer, in one process.

One online pass of the first grammar (4,000 batches of 16 rows by default): the brains are
``Brain.compose`` at the library's defaults on the CPU backend, taught by ``grammar.speak``; the
transformer takes one Adam step per batch (``controls.Learner``, online). The three learners
take each batch in turn and every batch's CPU seconds (``time.thread_time``) are summed per
learner, so they share whatever else the machine does. The kernels compile or load on copies
before the clock starts. Pin the process to one core (``taskset -c 3`` on Linux) and set one
thread per library:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src \\
        python benchmarks/grammar/cost.py --founder 0 --out DIR/cost-f0.json

``torch`` is required for the transformer alone; ``--learners plain,compiled`` leaves it out.

``--parts plain`` (or ``compiled``) lives one brain's pass alone and reads where its time goes: a
thread samples the main thread's stack every 5 ms and charges the time since its previous sample
to the part the stack is in (``PARTS``). A compiled kernel holds the interpreter until it returns,
so its time is charged to the stack the kernel returns into, the function that called it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import grammar as gr
import numpy as np

LEARNERS = ("plain", "compiled", "transformer")
TARGET = 24384  # the parameters the chamber's controls match (the recipe's brain)


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
            gr.speak(copy.deepcopy(learner), stream[0], teach=True)
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


# The kernels by the function that calls them, and the parts of a pass around them by the first
# function of the sampled stack, from the inside out, that names one.
KERNELS = {
    "fused_fast_solve": "kernels: the solves", "fused_solve": "kernels: the solves",
    "_launch": "kernels: the lessons' phases", "fused_residual": "kernels: the checks",
}
PARTS = {
    "Brain._stimulus": "the read",
    "Recall.learn": "the address lesson",
    "Hearing.learn": "the carried gains' lesson",
    "Learner.update": "the lessons' update",
    "Learner.step": "the lessons' phases around their kernels",
    "Brain._qualified": "the solves around their kernels",
    "Brain._choose": "the choice and the write",
    "Brain.lesson": "the rest of the lesson",
    "Brain.act": "the rest of the act",
}


def _part(frame: Any) -> str:
    """The part of the pass a sampled stack is in."""
    if frame is not None and frame.f_code.co_qualname in KERNELS:
        return KERNELS[frame.f_code.co_qualname]
    while frame is not None:
        part = PARTS.get(frame.f_code.co_qualname)
        if part is not None:
            return part
        frame = frame.f_back
    return "the chamber's own loop"


def parts(name: str, founder: int, batches: int) -> dict[str, Any]:
    """One brain's pass alone, its wall seconds charged to the parts of ``PARTS``."""
    warnings.simplefilter("ignore")
    brain = make(name, founder)
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    stream = [gr.train_batch(rng, split, "g1") for _ in range(batches)]
    gr.speak(copy.deepcopy(brain), stream[0], teach=True)
    main = threading.main_thread().ident
    spent: Counter[str] = Counter()
    done = threading.Event()

    def sample() -> None:
        last = time.perf_counter()
        while not done.wait(0.005):
            frame = sys._current_frames().get(main)
            now = time.perf_counter()
            if frame is not None:
                spent[_part(frame)] += now - last
            last = now

    sampler = threading.Thread(target=sample, daemon=True)
    started = time.perf_counter()
    process = time.process_time()
    sampler.start()
    for batch in stream:
        gr.speak(brain, batch, teach=True)
    done.set()
    sampler.join()
    wall = time.perf_counter() - started
    total = sum(spent.values()) or 1.0
    return {"learner": name, "founder": founder, "batches": batches, "wall": wall,
            "process": time.process_time() - process,
            "parts": {part: wall * seconds / total for part, seconds in spent.most_common()}}


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
