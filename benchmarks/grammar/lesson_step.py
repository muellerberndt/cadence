"""The lesson's step on the grammar chamber (cadence#258): does a step that follows the brain's
running error settle a life, with no count of lessons?

Every life is ``grammar.run_life`` on the default composition (the ``default`` arm: what
``Brain.compose`` gives a new brain, nothing named), the first grammar for 4,000 batches at width
64 or 128, with 256-sentence panels (``panel_batches=16``) read at every 500 batches and in full
at 4,000. The candidates read for #258 (``CANDIDATES``) were ``learning_running_error``: the
lesson's step scaled by the running share of lesson rows answered wrong, to the power
``learning_running_error_power``. That option failed the declared confirmation
(``LESSON_STEP_PROTOCOL.md``) and is not in the library; its lives run on the lane's commit
968a02b. The ``control`` arm, the swing reading and the gates run on any library.

Readings per life:

- ``swing``: the largest minus the smallest exact-clause reading at 3,000, 3,500 and 4,000
  batches (the validation reading settles where its swing is small).
- ``clauses``, ``held_out``, ``roles``, ``two_hops``: the full reading at 4,000 (``clause``,
  ``clause_held``, ``roles`` both right, ``path`` sentences of two hops).
- ``brackets``, ``paths``: the means of the last five readings (2,000 to 4,000), since either
  moves by more than .1 at one reading on last bits alone.

``plan`` writes the life specifications of a stage, ``read`` the table and the gates of a
results file (``--protocol`` names the stage's gates; see ``LESSON_STEP_PROTOCOL.md``).

    python benchmarks/grammar/lesson_step.py plan --stage dev --out DIR/dev.json
    python benchmarks/grammar/grammar.py --plan DIR/dev.json --out DIR/dev.jsonl --workers 30
    python benchmarks/grammar/lesson_step.py read DIR/dev.jsonl --stage dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

BATCHES = 4000
CHECKPOINTS = tuple(range(500, BATCHES, 500))
SWING_AT = (3000, 3500, 4000)
LATE = (2000, 2500, 3000, 3500, 4000)
WIDTHS = (64, 128)
PANEL_BATCHES = 16

# the candidate steps read for #258 (on commit 968a02b, whose library held the option): the
# forgetting factor of the running error and its power
ARMS: dict[str, dict[str, Any]] = {
    "control": {},
    "e99p5": {"learning_running_error": 0.99, "learning_running_error_power": 0.5},
    "e999p5": {"learning_running_error": 0.999, "learning_running_error_power": 0.5},
    "e99p1": {"learning_running_error": 0.99, "learning_running_error_power": 1.0},
    "e999p1": {"learning_running_error": 0.999, "learning_running_error_power": 1.0},
}
CANDIDATES = tuple(a for a in ARMS if a != "control")
STAGES = {
    "dev": {"founders": (0, 1, 2), "arms": tuple(ARMS)},
    # the confirmation's arms are the control and the candidate the development stage selects
    "confirm": {"founders": tuple(range(9625800, 9625806)), "arms": ("control",)},
}
READINGS = ("swing", "clauses", "held_out", "roles", "brackets", "paths", "two_hops")
LEVELS = ("clauses", "held_out", "brackets", "paths")  # the selection's eligibility
HOLD = ("clauses", "held_out", "roles", "brackets", "paths", "two_hops")  # the chamber gate


def plan(stage: str, arms: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    spec = STAGES[stage]
    return [{"arm": "default", "founder": f, "module": w, "phases": [["g1", BATCHES]],
             "settings": ARMS[a], "checkpoints": list(CHECKPOINTS),
             "panel_batches": PANEL_BATCHES, "tag": f"{stage}/{a}/w{w}/f{f}"}
            for a in (arms or spec["arms"]) for w in WIDTHS for f in spec["founders"]]


def life(row: dict[str, Any]) -> dict[str, Any]:
    """The readings of one life (a row of ``grammar.run_life``)."""
    at = {rd["batches"]: rd["panels"] for rd in row["readings"] if rd["phase"] == 0}
    final = at[BATCHES]
    hops = final["path"]["conditions"].get("hops2")
    return {
        "swing": max(at[b]["clause"]["exact"] for b in SWING_AT)
        - min(at[b]["clause"]["exact"] for b in SWING_AT),
        "clauses": final["clause"]["exact"],
        "held_out": final["clause_held"]["exact"],
        "roles": final["roles"]["both"],
        "brackets": float(np.mean([at[b]["dyck"]["exact"] for b in LATE])),
        "paths": float(np.mean([at[b]["path"]["exact"] for b in LATE])),
        "two_hops": hops,
        "seconds": row.get("seconds"),
    }


def table(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[int, dict[str, Any]]]:
    """``(arm, width) -> founder -> readings``, the arm and width from each row's tag."""
    out: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for row in rows:
        _, arm, width, _ = row["tag"].split("/")
        out.setdefault((arm, int(width[1:])), {})[int(row["founder"])] = life(row)
    return out


def means(cell: dict[int, dict[str, Any]]) -> dict[str, float]:
    return {k: float(np.mean([v[k] for v in cell.values() if v[k] is not None]))
            for k in READINGS}


def select(cells: dict[tuple[str, int], dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """The development stage's declared selection: among the candidates whose mean clauses,
    held-out clauses, brackets and paths are each at least the control's less .05 at both
    widths, the lowest mean clause swing over the two widths (ties: the smaller power, then the
    larger forgetting factor). With none eligible, the lowest swing overall, flagged."""
    score, eligible = {}, {}
    for arm in CANDIDATES:
        if not all((arm, w) in cells and ("control", w) in cells for w in WIDTHS):
            continue
        m = {w: means(cells[(arm, w)]) for w in WIDTHS}
        c = {w: means(cells[("control", w)]) for w in WIDTHS}
        score[arm] = float(np.mean([m[w]["swing"] for w in WIDTHS]))
        eligible[arm] = all(m[w][k] >= c[w][k] - 0.05 for w in WIDTHS for k in LEVELS)

    def key(arm: str) -> tuple[float, float, float]:
        s = ARMS[arm]
        return (score[arm], s["learning_running_error_power"], -s["learning_running_error"])

    pool = [a for a in score if eligible[a]] or list(score)
    chosen = min(pool, key=key) if pool else None
    return {"scores": score, "eligible": eligible, "chosen": chosen,
            "flagged": bool(chosen) and not eligible.get(chosen, False)}


def gates(cells: dict[tuple[str, int], dict[int, dict[str, Any]]], arm: str) -> dict[str, Any]:
    """The confirmation's chamber gates for ``arm`` against the control (the battery's gates
    are read by ``battery_gates``)."""
    out: dict[str, Any] = {}
    for w in WIDTHS:
        cand, ctrl = cells[(arm, w)], cells[("control", w)]
        within = sum(v["swing"] <= 0.05 for v in cand.values())
        out[f"C1 settles, width {w}"] = {
            "pass": within >= 5, "within": f"{within} of {len(cand)}",
            "swings": [round(cand[f]["swing"], 3) for f in sorted(cand)],
            "control_within": sum(v["swing"] <= 0.05 for v in ctrl.values())}
        m, c = means(cand), means(ctrl)
        out[f"C2 levels hold, width {w}"] = {
            "pass": all(m[k] >= c[k] - 0.03 for k in HOLD),
            "differences": {k: round(m[k] - c[k], 3) for k in HOLD}}
    return out


# the battery's tasks and bars (#256: S1, S2, S4 to S7, T1, U1, U2), as ``battery.py`` reads them
BATTERY: dict[str, tuple[str, dict[str, Any], str, float]] = {
    "lookback_first": ("lookback", {"lengths": [5, 12], "test_lengths": [24, 60]}, "first", .90),
    "lookback_second": ("lookback", {"lengths": [5, 12], "test_lengths": [24, 60]}, "second",
                        .90),
    "copy_unique": ("repeat", {"unique": True, "lengths": [3, 6], "test_lengths": [8, 12]},
                    "accuracy", .95),
    "copy_repeated": ("repeat", {"unique": False, "lengths": [3, 6], "test_lengths": [8, 12]},
                      "accuracy", .90),
    "periodic40": ("periodic", {"periods": [40]}, "accuracy", .85),
    "before": ("before", {}, "accuracy", .90),
    "twohop": ("twohop", {"think": True}, "accuracy", .90),
    "induction": ("induction", {"lengths": [8, 16], "test_lengths": [24, 60]}, "accuracy", .90),
    # the battery's n-back reads the lengths it teaches (its task takes no test lengths)
    "nback": ("nback", {"lengths": [6, 12]}, "accuracy", .90),
}
BATTERY_FOUNDERS = tuple(range(9625810, 9625816))


def battery_plan(arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Battery lives (``benchmarks/sequence/battery.py``) of the default composition: the
    tasks a ``lookback`` life reads twice share one life."""
    tasks = {"lookback": BATTERY["lookback_first"][:2]}
    tasks.update({k: v[:2] for k, v in BATTERY.items() if v[0] != "lookback"})
    return [{"task": task, "task_args": args, "arm": "default", "founder": f,
             "settings": settings, "tag": f"battery/{name}/{key}/f{f}"}
            for name, settings in arms.items() for key, (task, args) in tasks.items()
            for f in BATTERY_FOUNDERS]


def battery_reading(row: dict[str, Any], key: str) -> float:
    _, _, reading, _ = BATTERY[key]
    test = row["test"]
    return float(test["accuracy"] if reading == "accuracy" else test["conditions"][reading])


def battery_gates(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """C3: on every task the arm passes its bar on at least as many founders as the control
    (or on 5 of 6), and its mean is at least the control's less .03."""
    found: dict[tuple[str, str], dict[int, float]] = {}
    for row in rows:
        _, name, key, _ = row["tag"].split("/")
        keys = ("lookback_first", "lookback_second") if key == "lookback" else (key,)
        for k in keys:
            found.setdefault((name, k), {})[int(row["founder"])] = battery_reading(row, k)
    out = {}
    for key, (_, _, _, bar) in BATTERY.items():
        a, c = found.get((arm, key), {}), found.get(("control", key), {})
        if not a or not c:
            continue
        pa, pc = sum(v >= bar for v in a.values()), sum(v >= bar for v in c.values())
        ma, mc = float(np.mean(list(a.values()))), float(np.mean(list(c.values())))
        out[key] = {"pass": (pa >= pc or pa >= 5) and ma >= mc - 0.03,
                    "bar": bar, "arm": f"{pa} of {len(a)}", "control": f"{pc} of {len(c)}",
                    "means": [round(ma, 3), round(mc, 3)]}
    return out


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--stage", choices=list(STAGES), required=True)
    p.add_argument("--arms", nargs="*", choices=list(ARMS))
    p.add_argument("--battery", action="store_true", help="the battery lives of the arms")
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("read")
    r.add_argument("results", type=Path)
    r.add_argument("--stage", choices=list(STAGES), required=True)
    r.add_argument("--arm", choices=CANDIDATES, help="the confirmation's candidate")
    r.add_argument("--battery", type=Path, help="the battery's results")
    args = parser.parse_args(argv)
    if args.command == "plan":
        arms = tuple(args.arms) if args.arms else None
        specs = (battery_plan({a: ARMS[a] for a in (arms or ("control",))}) if args.battery
                 else plan(args.stage, arms))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(specs, indent=1))
        print(f"{len(specs)} lives")
        return 0
    cells = table(_rows(args.results))
    for (arm, w), cell in sorted(cells.items()):
        m = means(cell)
        print(f"{arm:8s} w{w:<4d} n{len(cell)} " + " ".join(f"{k} {m[k]:.3f}" for k in READINGS)
              + " | swing " + " ".join(f"{cell[f]['swing']:.3f}" for f in sorted(cell)))
    result: dict[str, Any] = {}
    if args.stage == "dev":
        result["selection"] = select(cells)
    elif args.arm:
        result["gates"] = gates(cells, args.arm)
        if args.battery:
            result["battery"] = battery_gates(_rows(args.battery), args.arm)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
