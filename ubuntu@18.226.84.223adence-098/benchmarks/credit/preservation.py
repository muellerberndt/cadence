"""The reward chambers a reward-credit setting must keep (cadence#208).

A setting is a set of ``Brain.compose`` options. Each chamber composes its own brains, and
``composed_with`` adds the setting's options to every ``Brain.compose`` call of the chamber's
own code, which is otherwise unchanged; the same chamber without the setting is its control,
on the same seeds. A setting breaks a chamber when the lives (or blocks, or seeds) passing
its reading fall by two or more.

- ``nursery``: the odour nursery (``benchmarks/reversal``) at its operating point, arms
  ``live`` (five checks: acquired, reversed, returned, stable pair kept, calm) and ``step``
  (the four learning checks), with the reversal lag.
- ``keydoor``: the key-door nursery (``benchmarks/keydoor``, ``protocol-3.json``), arm
  ``live``: rule A acquired (fed on .90 of its last 50 trips), rule B adapted, the return of
  rule A retained, frugal and calm.
- ``bandit``: a two-armed bandit with reversal, ``Brain.compose(1, 2, seed)`` through
  ``step``: one constant input, the paying arm +1 and the other -1, the paying arm moving
  every 200 of 1,000 trials unannounced; a block is won when the paying arm takes at least
  .90 of its last 50 executed actions.
- ``contextual`` and ``delayed``: the library's two reward worlds of
  ``tests/test_context_defaults.py``, a continuing contextual bandit (seeds 0 to 5) and a
  delayed response from reward (seeds 8 to 13), each against the tests' .90.

    python benchmarks/credit/preservation.py --chamber nursery --seeds 0 1 2 3 4 5 \\
        --settings '{"memory_choice": true}' --out results/nursery.jsonl --workers 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CHAMBERS = ("nursery", "keydoor", "bandit", "contextual", "delayed")
BANDIT = {"trials": 1000, "block": 200, "last": 50, "won": 0.9, "hold": 20}


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def composed_with(settings: dict[str, Any] | None) -> Iterator[None]:
    """Every ``Brain.compose`` call within takes ``settings`` beside its own options; an
    option the caller names with another value is refused."""
    import cadence as cd

    if not settings:
        yield
        return
    original = cd.Brain.__dict__["compose"]
    bound = cd.Brain.compose

    def compose(*args: Any, **options: Any) -> Any:
        clash = [k for k in settings if k in options and options[k] != settings[k]]
        if clash:
            raise ValueError(f"the chamber sets {clash[0]!r} itself")
        return bound(*args, **{**options, **settings})

    cd.Brain.compose = staticmethod(compose)  # type: ignore[method-assign]
    try:
        yield
    finally:
        cd.Brain.compose = original  # type: ignore[method-assign]


# -------------------------------------------------------------------------------- chambers


def nursery_life(seed: int, settings: dict[str, Any], arm: str, exposure: int) -> dict[str, Any]:
    nursery = _module("odour_nursery", ROOT / "benchmarks" / "reversal" / "odour_nursery.py")
    protocol = json.loads(nursery.PROTOCOL.read_text())
    with composed_with(settings):
        life = nursery.run_life(arm, seed, exposure, protocol)
    out: dict[str, Any] = {"arm": arm, "exposure": exposure}
    if "error" in life:
        return {**out, "error": life["error"], "passed": False}
    g, phases = protocol["gates"], life["phases"]
    checks = {
        "acquired": phases[0]["final"] >= g["final"],
        "reversed": phases[1]["final"] >= g["final"],
        "returned": phases[2]["final"] >= g["final"],
        "stable_kept": all((p["stable"] or 0.0) >= g["stable"] for p in phases),
        "calm": all(p["aroused_late"] <= g["aroused_late"] for p in phases),
    }
    learning = ("acquired", "reversed", "returned", "stable_kept")
    passed = all(checks.values()) if arm == "live" else all(checks[k] for k in learning)
    return {
        **out, "checks": checks, "passed": passed,
        "final": [p["final"] for p in phases], "lag": [p["lag"] for p in phases],
        "stable": [p["stable"] for p in phases],
        "aroused_late": [p["aroused_late"] for p in phases],
    }


def keydoor_life(seed: int, settings: dict[str, Any], delay: int) -> dict[str, Any]:
    keydoor = _module("key_door", ROOT / "benchmarks" / "keydoor" / "key_door.py")
    protocol = json.loads(keydoor.PROTOCOL.read_text())
    with composed_with(settings):
        life = keydoor.run_life("live", seed, delay, protocol)
    out: dict[str, Any] = {"arm": "live", "delay": delay}
    if "error" in life:
        return {**out, "error": life["error"], "passed": False}
    g, phases = protocol["gates"], life["phases"]

    def fed(i: int) -> bool:
        return len(phases) > i and phases[i]["fed"] is not None and phases[i]["fed"] >= g["fed"]

    checks = {
        "acquired": fed(0),
        "adapted": fed(1),
        "retained": fed(2) and phases[2]["lag"] is not None
        and phases[2]["lag"] <= g["retained_lag"],
        "frugal": all(p["wrong"] is not None and p["wrong"] <= g["wrong"] for p in phases),
        "calm": all(p["aroused_late"] <= g["aroused_late"] for p in phases),
    }
    return {
        **out, "checks": checks, "passed": checks["acquired"], "joint": all(checks.values()),
        "fed": [p["fed"] for p in phases], "lag": [p["lag"] for p in phases],
        "wrong": [p["wrong"] for p in phases],
        "aroused_late": [p["aroused_late"] for p in phases],
    }


def bandit_life(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    """The two-armed bandit with reversal through ``step``; arm 0 pays in the first block."""
    import cadence as cd

    b = BANDIT
    with composed_with(settings):
        brain = cd.Brain.compose(1, 2, backend="cpu", seed=int(seed), recall=False)
    x = np.ones((1, 1))
    actions: list[int] = []
    action = brain.step(x)
    for trial in range(b["trials"]):
        paying = (trial // b["block"]) % 2
        actions.append(int(action[0]))
        reward = 1.0 if int(action[0]) == paying else -1.0
        action = brain.step(x, reward=[reward])
    blocks = []
    for k in range(b["trials"] // b["block"]):
        chosen = np.asarray(actions[k * b["block"] : (k + 1) * b["block"]]) == k % 2
        lag = next(
            (t for t in range(len(chosen) - b["hold"] + 1) if chosen[t : t + b["hold"]].mean()
             >= b["won"]),
            None,
        )
        blocks.append({"share": float(chosen[-b["last"] :].mean()), "lag": lag})
    won = sum(block["share"] >= b["won"] for block in blocks)
    return {"blocks": blocks, "won": won, "passed": won == len(blocks)}


def _worlds() -> Any:
    """The test module that holds the two worlds; it imports the tests' own helpers."""
    if str(ROOT / "tests") not in sys.path:
        sys.path.insert(0, str(ROOT / "tests"))
    return _module("context_defaults_worlds", ROOT / "tests" / "test_context_defaults.py")


def contextual_life(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    import cadence as cd

    with composed_with(settings):
        brain = cd.Brain.compose(4, 4, backend="cpu", seed=int(seed), actor_gamma=0.0, actor_lam=0.0, recall=False)
    score = _worlds().contextual_bandit(brain, int(seed))
    return {"score": score, "passed": score >= 0.9}


def delayed_life(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    import cadence as cd

    with composed_with(settings):
        brain = cd.Brain.compose(3, 2, backend="cpu", seed=int(seed), recall=False)
    score = _worlds().delayed_response(brain, 1500, int(seed))
    return {"score": score, "passed": score >= 0.9}


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life of a chamber: ``spec`` names ``chamber``, ``seed``, ``settings`` (the setting's
    options; empty for the control) and ``label``, with ``arm`` and ``exposure`` (nursery) or
    ``delay`` (key-door)."""
    warnings.simplefilter("ignore")
    import cadence as cd

    started = time.monotonic()
    chamber, seed, settings = spec["chamber"], int(spec["seed"]), dict(spec.get("settings", {}))
    row = {"chamber": chamber, "seed": seed, "label": spec.get("label"), "settings": settings,
           "cadence": cd.__version__}
    try:
        if chamber == "nursery":
            row.update(nursery_life(seed, settings, spec["arm"], int(spec["exposure"])))
        elif chamber == "keydoor":
            row.update(keydoor_life(seed, settings, int(spec["delay"])))
        elif chamber == "bandit":
            row.update(bandit_life(seed, settings))
        elif chamber == "contextual":
            row.update(contextual_life(seed, settings))
        elif chamber == "delayed":
            row.update(delayed_life(seed, settings))
        else:
            raise ValueError(f"unknown chamber {chamber!r}")
    except Exception as error:  # a life that could not run is a recorded outcome
        row["error"] = f"{type(error).__name__}: {error}"[:300]
        row["passed"] = False
    row["seconds"] = round(time.monotonic() - started, 2)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs")
    parser.add_argument("--chamber", choices=CHAMBERS)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(6)))
    parser.add_argument("--settings", type=json.loads, default={})
    parser.add_argument("--label", default=None)
    parser.add_argument("--arm", default="live", choices=("live", "step"))
    parser.add_argument("--exposure", type=int, default=300)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.plan is not None:
        specs = json.loads(args.plan.read_text())
    elif args.chamber is not None:
        extra = {"nursery": {"arm": args.arm, "exposure": args.exposure},
                 "keydoor": {"delay": args.delay}}.get(args.chamber, {})
        specs = [{"chamber": args.chamber, "seed": s, "settings": args.settings,
                  "label": args.label, **extra} for s in args.seeds]
    else:
        parser.error("--plan or --chamber is required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for row in map(run_life, specs):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                for row in pool.imap_unordered(run_life, specs, chunksize=1):
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
