"""Exposure parity on the grammar chamber (cadence#270): the default composition with its
rehearsal, taught the same 4,000 batches the transformer's offline control replays, read after
every pass beside that control at equal compute.

A life draws the first grammar's 4,000 training batches once, from the stream every arm and
control of a founder hears (``SeedSequence([founder, 11])``), and teaches them in passes: the
first in the stream's order (the chamber's online life), every later one in the order of the
transformer's next offline epoch (``SeedSequence([founder, 99, 0])``, one permutation per
epoch), with lessons as in ``grammar.speak``. After each pass every panel of the first grammar is
read on a fork, with ``located`` (where a word is lost: at the recall read or at the readout) on
the last pass. A life stops after ``passes`` passes, or once a pass raises no reading's mean
(``READINGS``) by more than ``SATURATED`` over the pass before it (saturation).

Compute is counted analytically: every row sweep of a settle (a lesson's phases, every act's
sweeps times its rows, the rehearsal's replayed moments and lessons at the pass's own sweeps per
moment and per taught row) multiplies each of the brain's synapses once, which counts an act whose
rows stop early at the batch's longest and leaves out the memory's reads; the controls' counts are
``controls.py``'s (a backward pass at twice the forward).

    python benchmarks/grammar/exposure.py plan --out DIR/plan.json
    python benchmarks/grammar/exposure.py run --plan DIR/plan.json --out DIR/brain.jsonl --workers 6
    python benchmarks/grammar/controls.py --plan DIR/controls.json --out DIR/controls.jsonl
    python benchmarks/grammar/exposure.py read DIR/brain.jsonl DIR/controls.jsonl
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grammar as gr  # noqa: E402

BATCHES = 4000
PASSES = 4
SATURATED = 0.01
FOUNDERS = tuple(range(9627000, 9627006))
ARMS: dict[str, dict[str, Any]] = {
    "default": {},
    "rehearsal": {"learning_rehearsal": 1.0},
}
# panel, score: the readings compared with the control (two hops and three hops are the path
# sentences of two hops and the three-hop panel)
READINGS: dict[str, tuple[str, str]] = {
    "clauses": ("clause", "exact"),
    "held_out": ("clause_held", "exact"),
    "long_clauses": ("clause_long", "exact"),
    "roles": ("roles", "both"),
    "brackets": ("dyck", "exact"),
    "deep_brackets": ("dyck_deep", "exact"),
    "paths": ("path", "exact"),
    "three_hops": ("path_long", "exact"),
}
MARGIN = 0.05


def synapses(brain: Any) -> int:
    return int(brain.describe()["layout"]["synapses"])


def macs(work: dict[str, float], count: int, rows: int = gr.BATCH) -> float:
    """The analytic multiply-accumulates of ``work`` on a brain of ``count`` synapses. The
    rehearsal reports its replayed moments and taught rows, not its sweeps: each replayed moment
    is counted at the pass's sweeps per act and row, each replayed taught row at the pass's
    lesson sweeps per taught row."""
    taught = work["rows"] - work["agreed_rows"]
    per_moment = work["act_sweeps"] / work["acts"] if work["acts"] else 0.0
    per_row = work["lesson_row_sweeps"] / taught if taught else 0.0
    rehearsal = (work.get("rehearsal_moments", 0.0) * per_moment
                 + work.get("rehearsal_rows", 0.0) * per_row)
    work["rehearsal_row_sweeps"] = rehearsal
    return float((work["lesson_row_sweeps"] + rehearsal + work["act_sweeps"] * rows) * count)


def order(founder: int, passes: int, count: int = BATCHES) -> list[np.ndarray]:
    """The batches' order per pass: the stream's, then the transformer's offline epochs'
    (``controls.py`` draws one permutation per epoch from this generator)."""
    epochs = np.random.default_rng(np.random.SeedSequence([int(founder), 99, 0]))
    return [np.arange(count)] + [epochs.permutation(count) for _ in range(1, passes)]


def scores(panels: dict[str, Any]) -> dict[str, float | None]:
    return {name: panels[panel][key] if panel in panels else None
            for name, (panel, key) in READINGS.items()}


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``arm``, ``founder``, ``passes`` (at most), ``module``, ``panel_batches``,
    ``batches`` (per pass) and ``settings`` over the arm's."""
    warnings.simplefilter("ignore")
    import cadence as cd

    arm, founder = spec["arm"], int(spec["founder"])
    passes, count = int(spec.get("passes", PASSES)), int(spec.get("batches", BATCHES))
    settings = {**ARMS[arm], **spec.get("settings", {})}
    started = time.monotonic()
    brain = gr.make_brain("default", founder, settings, int(spec.get("module", 64)))
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    batches = [gr.train_batch(rng, split, "g1") for _ in range(count)]
    size = synapses(brain)
    row: dict[str, Any] = {
        "arm": arm, "founder": founder, "settings": settings, "batches": count,
        "module": int(spec.get("module", 64)), "parameters": int(brain.parameters()),
        "synapses": size, "cadence": cd.__version__, "tag": spec.get("tag"), "passes": [],
    }
    total = 0.0
    previous: dict[str, float | None] | None = None
    for index, sequence in enumerate(order(founder, passes, count)):
        work = gr.new_work()
        clock = time.monotonic()
        for b in sequence:
            gr.speak(brain, batches[int(b)], teach=True, work=work)
        work["seconds"] = round(time.monotonic() - clock, 1)
        work["macs"] = macs(work, size)
        total += work["macs"]
        panels = gr.reading(brain, tests, gr.G1_PANELS, components=False)
        now = scores(panels)
        saturated = previous is not None and all(
            now[k] is None or previous[k] is None or now[k] - previous[k] <= SATURATED
            for k in READINGS)
        last = saturated or index == passes - 1
        entry = {"pass": index + 1, "work": work, "macs": total, "readings": now,
                 "panels": {k: {kk: vv for kk, vv in v.items() if kk != "examples"}
                            for k, v in panels.items()}}
        if last and brain.recall is not None:
            entry["located"] = {name: gr.locate(brain, tests[name]) for name in gr.LOCATE
                                if name in tests}
        row["passes"].append(entry)
        previous = now
        if last:
            row["saturated"] = saturated
            break
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def plan(founders: tuple[int, ...] = FOUNDERS, passes: int = PASSES,
         panel_batches: int = 16) -> list[dict[str, Any]]:
    return [{"arm": arm, "founder": f, "passes": passes, "panel_batches": panel_batches,
             "tag": f"exposure/{arm}/f{f}"} for arm in ARMS for f in founders]


def control_plan(parameters: int, founders: tuple[int, ...] = FOUNDERS,
                 panel_batches: int = 16) -> list[dict[str, Any]]:
    """The transformer's offline control on the same batches, matched to the brain's
    parameters (``controls.run_life``)."""
    return [{"model": "transformer", "mode": "offline", "founder": f,
             "phases": [["g1", BATCHES]], "parameters": parameters,
             "panel_batches": panel_batches} for f in founders]


def compare(brains: list[dict[str, Any]], controls: list[dict[str, Any]],
            arm: str = "rehearsal") -> dict[str, Any]:
    """E1: at the transformer's offline compute (each founder's last pass whose compute is
    within the control's, else the first), every reading's mean at least the control's less
    ``MARGIN``. E2: the same at the last pass read (saturation). Per reading, the means and
    each pass's mean."""
    ctrl = {int(r["founder"]): r for r in controls if r["mode"] == "offline"}
    lives = {int(r["founder"]): r for r in brains if r["arm"] == arm}
    founders = sorted(set(ctrl) & set(lives))
    if not founders:
        return {}

    def at(life: dict[str, Any], budget: float) -> dict[str, Any]:
        within = [p for p in life["passes"] if p["macs"] <= budget]
        return within[-1] if within else life["passes"][0]

    out: dict[str, Any] = {"founders": founders, "arm": arm}
    reference = {k: float(np.mean([scores(ctrl[f]["readings"][0]["panels"])[k]
                                   for f in founders])) for k in READINGS}
    for gate, pick in (("E1 equal compute", lambda f: at(lives[f], ctrl[f]["readings"][0][
            "work"]["macs"])), ("E2 saturation", lambda f: lives[f]["passes"][-1])):
        chosen = {f: pick(f) for f in founders}
        mean = {k: float(np.mean([chosen[f]["readings"][k] for f in founders]))
                for k in READINGS}
        apart = [k for k in READINGS if mean[k] < reference[k] - MARGIN]
        out[gate] = {"pass": not apart, "apart": apart,
                     "brain": {k: round(v, 3) for k, v in mean.items()},
                     "control": {k: round(v, 3) for k, v in reference.items()},
                     "passes": {f: chosen[f]["pass"] for f in founders},
                     "brain_macs": float(np.mean([chosen[f]["macs"] for f in founders])),
                     "control_macs": float(np.mean([ctrl[f]["readings"][0]["work"]["macs"]
                                                    for f in founders]))}
    curve: dict[int, dict[str, float]] = {}
    for p in range(1, max(len(lives[f]["passes"]) for f in founders) + 1):
        have = [lives[f]["passes"][p - 1] for f in founders if len(lives[f]["passes"]) >= p]
        curve[p] = {k: round(float(np.mean([h["readings"][k] for h in have])), 3)
                    for k in READINGS}
        curve[p]["lives"] = len(have)
    out["curve"] = curve
    return out


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--controls", type=Path, help="the controls' plan")
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=1)
    c = sub.add_parser("read")
    c.add_argument("brains", type=Path)
    c.add_argument("controls", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan(), indent=1))
        if args.controls:
            parameters = int(gr.make_brain("default", FOUNDERS[0]).parameters())
            args.controls.write_text(json.dumps(control_plan(parameters), indent=1))
        return 0
    if args.command == "run":
        specs = json.loads(args.plan.read_text())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as sink:
            if args.workers <= 1:
                results = map(run_life, specs)
                for result in results:
                    sink.write(json.dumps(result) + "\n")
                    sink.flush()
            else:
                with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                    for result in pool.imap_unordered(run_life, specs, chunksize=1):
                        sink.write(json.dumps(result) + "\n")
                        sink.flush()
        return 0
    brains, controls = _rows(args.brains), _rows(args.controls)
    print(json.dumps({arm: compare(brains, controls, arm) for arm in ARMS}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
