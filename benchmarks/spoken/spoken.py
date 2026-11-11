"""Lives in the spoken-phrase chamber (cadence#303): a brain or a baseline learns the base
vocabulary's phrases, is read at each budget, and then adapts to new speakers.

A life is one learner (a brain arm of ``chamber.ARMS`` or a baseline of ``chamber.CONTROLS``)
and one founder. It hears ``base_train`` in batches of 32 phrases of equal steps, in an order
the founder fixes for every learner, one pass after another, and is read at each budget
(phrases taught) on samples of ``base_validation`` and ``base_test``. A baseline's parameter
count is matched to the brain arm named by ``--match``.

After the last budget, each named new speaker's ``adapt`` phrases are heard one at a time from
a copy of the life, plain and with replay, and read at each of ``--minutes`` on that speaker's
``evaluate`` phrases and on the retention panel (phrases of the base corpus's held-out speakers).
``--probe`` adds ridge readouts at the word steps (what the window, the pools and the
association hold), readings that locate a failure and never enter a life.

    PYTHONPATH=src python benchmarks/spoken/spoken.py --data /data/phrases --learners fields \\
        transformer_ctc --founders 0 1 --budgets 6000 24000 --speakers speaker/<id> \\
        --out results/step0.jsonl --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    key = f"spoken_{name}"  # a name of this chamber's own: other benchmarks have their modules
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load("chamber")
ph = _load("phrases")

SEED = 30312
READ = 300  # held-out phrases per reading
RETENTION = 300  # retention-panel phrases
PROBE_FIT, PROBE_READ = 3000, 1500


_SETS: dict[tuple[str, str], Any] = {}


def phrases(data: str, name: str) -> Any:
    key = (data, name)
    if key not in _SETS:
        _SETS[key] = ph.Phrases(Path(data) / f"{name.replace('/', '__')}.npz")
    return _SETS[key]


def sample(n: int, count: int, tag: str) -> np.ndarray:
    rng = np.random.default_rng([SEED, sum(map(ord, tag))])
    return np.sort(rng.choice(n, min(count, n), replace=False))


def learner_of(name: str, founder: int, match: str, extra: dict[str, Any] | None) -> Any:
    if name in ch.CONTROLS:
        count = ch.compose(match, founder, extra).parameters()
        return ch.control(name, founder, count)
    return ch.BrainLearner(ch.compose(name, founder, extra), name)


def schedule(train: Any, founder: int, budget: int) -> list[np.ndarray]:
    """The founder's batches up to ``budget`` phrases: passes over the set in shuffled orders."""
    rng = np.random.default_rng([SEED, founder])
    every = np.arange(len(train))
    out: list[np.ndarray] = []
    taught = 0
    while taught < budget:
        for chunk in ch.groups(train, every, ch.BATCH, rng):
            if taught >= budget:
                break
            out.append(chunk)
            taught += len(chunk)
    return out


def probe(
    learner: Any,
    data: str,
    hearing: Any,
    speakers: list[str],
    founder: int,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    train = phrases(data, "base_train")
    fit_x, fit_y = ch.word_steps(train, sample(len(train), PROBE_FIT, "probe"), hearing, PROBE_FIT)
    test = phrases(data, "base_test")
    panels = {
        "base_test": ch.word_steps(
            test, sample(len(test), PROBE_READ, "probe-test"), hearing, PROBE_READ
        )
    }
    for name in speakers:
        own = phrases(data, f"{name}/evaluate")
        panels[name] = ch.word_steps(own, np.arange(len(own)), hearing, PROBE_READ)
    out: dict[str, Any] = {"window": ch.ridge(fit_x, fit_y, panels)}
    if learner.kind != "brain":
        return out
    brain = learner.reader().brain
    views = {
        "pools": lambda x: ch.pools(brain, x),
        "association": lambda x: ch.association(brain, x),
    }
    if brain.fields is not None:
        drawn = ch.compose(learner.arm, founder, extra)
        views["pools_drawn"] = lambda x: ch.pools(drawn, x)
    for view, f in views.items():
        out[view] = ch.ridge(f(fit_x), fit_y, {k: (f(x), y) for k, (x, y) in panels.items()})
    return out


def life(spec: dict[str, Any]) -> dict[str, Any]:
    warnings.simplefilter("ignore", RuntimeWarning)
    data, name, founder = spec["data"], spec["learner"], int(spec["founder"])
    extra = spec.get("extra")
    started = time.process_time()
    train = phrases(data, "base_train")
    hearing = ch.Hearing.measured(train)
    learner = learner_of(name, founder, spec.get("match", "fields"), extra)
    validation = phrases(data, "base_validation")
    test = phrases(data, "base_test")
    v_rows = sample(len(validation), READ, "validation")
    t_rows = sample(len(test), READ, "test")
    budgets = sorted(int(b) for b in spec["budgets"])
    batches = schedule(train, founder, budgets[-1])
    readings: list[dict[str, Any]] = []
    taught = 0
    at = 0
    for budget in budgets:
        while taught < budget and at < len(batches):
            ch.train(learner, train, [batches[at]], hearing)
            taught += len(batches[at])
            at += 1
        readings.append(
            {
                "phrases": taught,
                "validation": ch.read(learner, validation, v_rows, hearing),
                "test": ch.read(learner, test, t_rows, hearing),
                "work": dict(learner.work),
            }
        )
    out: dict[str, Any] = {
        "learner": name,
        "founder": founder,
        "extra": extra,
        "parameters": learner.parameters(),
        "readings": readings,
    }
    if learner.kind == "control":
        out["width"] = learner.model.width
    if spec.get("save"):
        path = Path(spec["save"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if learner.kind == "brain":
            learner.brain.save(path.with_suffix(".npz"))
        else:
            import torch

            torch.save(learner.model, path.with_suffix(".pt"))
    speakers = list(spec.get("speakers") or [])
    if spec.get("probe"):
        out["probe"] = probe(learner, data, hearing, speakers, founder, extra)
    retention_rows = sample(len(test), RETENTION, "retention")
    adaptation = []
    for speaker in speakers:
        own = phrases(data, f"{speaker}/adapt")
        evaluate = phrases(data, f"{speaker}/evaluate")
        for mode in spec.get("modes", ["plain"]):
            for rate in spec.get("rates", [1.0]):
                curve = ch.adapt(
                    learner,
                    own,
                    evaluate,
                    test,
                    retention_rows,
                    hearing,
                    spec.get("minutes", [0, 1, 2, 5]),
                    replay=train if mode == "replay" else None,
                    replays=int(spec.get("replays", 3)),
                    rate=float(rate),
                    seed=founder,
                )
                adaptation.append({"speaker": speaker, "mode": mode, "rate": rate, "curve": curve})
    if adaptation:
        out["adaptation"] = adaptation
    out["process_seconds"] = time.process_time() - started
    return out


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ.setdefault(name, "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    try:
        return {"spec": spec, **life(spec)}
    except Exception as exc:  # a failed life is a reading too
        import traceback

        return {
            "spec": spec,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-2000:],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--learners", nargs="+", default=["fields"])
    parser.add_argument("--founders", nargs="+", type=int, default=[0])
    parser.add_argument("--budgets", nargs="+", type=int, default=[2400])
    parser.add_argument("--match", default="fields", help="the brain arm a baseline matches")
    parser.add_argument("--speakers", nargs="*", default=[])
    parser.add_argument("--modes", nargs="+", default=["plain", "replay"])
    parser.add_argument("--rates", nargs="+", type=float, default=[1.0])
    parser.add_argument("--minutes", nargs="+", type=float, default=[0, 1, 2, 5])
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--extra", default=None, help="JSON of compose settings")
    parser.add_argument("--save", default=None, help="folder for each life's final learner")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    specs = [
        {
            "data": args.data,
            "learner": name,
            "founder": f,
            "budgets": args.budgets,
            "match": args.match,
            "speakers": args.speakers,
            "modes": args.modes,
            "rates": args.rates,
            "minutes": args.minutes,
            "replays": args.replays,
            "probe": args.probe,
            "extra": json.loads(args.extra) if args.extra else None,
            "save": str(Path(args.save) / f"{name}-{f}") if args.save else None,
        }
        for name in args.learners
        for f in args.founders
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        if args.workers > 1:
            with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
                for result in pool.imap_unordered(_run, specs):
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
        else:
            for spec in specs:
                handle.write(json.dumps(_run(spec)) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
