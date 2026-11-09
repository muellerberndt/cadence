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
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--founder", type=int, default=0)
    parser.add_argument("--batches", type=int, default=4000)
    parser.add_argument("--learners", default=",".join(LEARNERS))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
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
