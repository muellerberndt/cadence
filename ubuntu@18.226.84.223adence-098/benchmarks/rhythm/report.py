"""Print the Markdown tables of a steady-rhythm receipt; no model runs.

Usage: python benchmarks/rhythm/report.py RUN_DIRECTORY/summary.json
The tables are what the README and the issue comment quote; they are recomputed from
the stored per-event actions, not copied from the stored aggregate.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import rhythm_inputs as inputs
import steady_rhythm as chamber


def load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    return raw["body"] if "body" in raw else raw


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def rescored(body: dict) -> list[dict]:
    """Recompute every window score from the stored actions and anchors."""
    block = body["protocol"]["window"]["block"]
    rows = []
    for run in body["runs"]:
        for name in chamber.CONTROLS:
            branch = run["window"]["branches"][name]
            anchor = np.asarray(branch["anchor"])
            score = chamber.score_window(np.asarray(branch["actions"]), anchor, block)
            assert score == branch["score"], name
            rows.append(
                {
                    "recipe": run["recipe"],
                    "arm": run["arm"],
                    "seed": run["seed"],
                    "branch": name,
                    **score,
                }
            )
    return rows


def controls_table(body: dict, rows: list[dict]) -> str:
    out = []
    founders = sorted({(r["recipe"], r["arm"]) for r in rows})
    out.append("| Branch | " + " | ".join(f"{recipe}/{arm}" for recipe, arm in founders) + " |")
    out.append("| --- |" + " --- |" * len(founders))
    for name in chamber.CONTROLS + ("shuffled_donor",):
        cells = []
        for recipe, arm in founders:
            if name == "shuffled_donor":
                scores = [
                    run["window"]["branches"]["shuffled"]["donor_score"]
                    for run in body["runs"]
                    if run["recipe"] == recipe and run["arm"] == arm
                ]
            else:
                scores = [
                    r
                    for r in rows
                    if r["recipe"] == recipe and r["arm"] == arm and r["branch"] == name
                ]
            alternation = np.mean([s["alternation_rate"] or 0.0 for s in scores])
            agreement = np.mean([s["agreement"] for s in scores])
            refusals = sum(s["refusals"] for s in scores)
            cells.append(f"{alternation:.2f} / {agreement:.2f} / {refusals}")
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def founder_table(rows: list[dict]) -> str:
    out = [
        "| Recipe | Arm | Seed | intact alternation | intact agreement | blocks of 8 | "
        "erased | reset | static | flipflop | random |",
        "| --- |" * 11,
    ]
    out[1] = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    index = {(r["recipe"], r["arm"], r["seed"], r["branch"]): r for r in rows}
    for recipe, arm, seed in sorted({(r["recipe"], r["arm"], r["seed"]) for r in rows}):
        intact = index[recipe, arm, seed, "intact"]
        blocks = " ".join(fmt(b) for b in intact["blocks"])
        others = [
            fmt(index[recipe, arm, seed, name]["alternation_rate"] or 0.0)
            for name in ("erased", "reset", "static", "flipflop", "random")
        ]
        out.append(
            f"| {recipe} | {arm} | {seed} | {fmt(intact['alternation_rate'])} | "
            f"{fmt(intact['agreement'])} | {blocks} | " + " | ".join(others) + " |"
        )
    return "\n".join(out)


def disturbance_table(body: dict) -> str:
    out = [
        "| Disturbance | Model | pre alternation | post alternation | agreement hold | "
        "agreement continue | recovered rows / rows |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    names = list(body["runs"][0]["disturbances"])
    for recipe, arm in sorted({(run["recipe"], run["arm"]) for run in body["runs"]}):
        runs = [run for run in body["runs"] if run["recipe"] == recipe and run["arm"] == arm]
        for name in names:
            for model in ("brain", "flipflop", "random"):
                items = [run["disturbances"][name][model] for run in runs]
                pre = np.mean([i["pre"]["alternation_rate"] or 0.0 for i in items])
                post = np.mean([i["post_hold"]["alternation_rate"] or 0.0 for i in items])
                hold = np.mean([i["post_hold"]["agreement"] for i in items])
                cont = np.mean([i["post_continue"]["agreement"] for i in items])
                recovered = sum(sum(v == 0 for v in i["recovery_events"]) for i in items)
                label = f"{recipe}/{arm} {model}" if model == "brain" else model
                out.append(
                    f"| {name} | {label} | {pre:.2f} | {post:.2f} | {hold:.2f} | "
                    f"{cont:.2f} | {recovered}/{inputs.ROWS * len(items)} |"
                )
    return "\n".join(out)


def cadence_table(body: dict) -> str:
    out = [
        "| Variant | founders identical to reference | refusals | sweeps per event | "
        "solve ms mean / max | lateness ms max | missed deadlines | slot agreement |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    runs = [run for run in body["runs"] if run["cadence"] is not None]
    if not runs:
        return "cadence runs were not part of this receipt"
    for name in runs[0]["cadence"]["variants"]:
        items = [run["cadence"]["variants"][name] for run in runs]
        identical = sum(i.get("identical_to_reference", True) for i in items)
        refusals = sum(i["score"]["refusals"] for i in items)
        sweeps = np.mean(
            [i["work"]["action_sweeps"] / max(1, i["work"]["action_attempts"]) for i in items]
        )
        if "solve_ms" in items[0]:
            solve = (
                f"{np.mean([np.mean(i['solve_ms']) for i in items]):.2f} / "
                f"{max(max(i['solve_ms']) for i in items):.2f}"
            )
            lateness = f"{max(max(i['lateness_ms']) for i in items):.2f}"
            missed = str(sum(i["missed_deadlines"] for i in items))
            slot = f"{np.mean([i['slot_agreement'] for i in items]):.2f}"
        else:
            solve = lateness = missed = slot = "unpaced"
        out.append(
            f"| {name} | {identical}/{len(items)} | {refusals} | {sweeps:.1f} | {solve} | "
            f"{lateness} | {missed} | {slot} |"
        )
    return "\n".join(out)


def custody_lines(body: dict) -> str:
    runs = body["runs"]
    continuation = sum(
        run["window"]["continuation"]["actions_equal"]
        and run["window"]["continuation"]["saved_arrays_equal"]
        for run in runs
    )
    pause = sum(
        run["disturbances"]["pause2"]["custody"]["actions_equal"]
        and run["disturbances"]["pause2"]["custody"]["saved_arrays_equal"]
        for run in runs
    )
    teaching = {}
    for run in runs:
        key = f"{run['recipe']}/{run['arm']}"
        teaching.setdefault(key, []).append(run["teaching"])
    identity = sum(sum(run["window"]["branches"]["shuffled"]["donor_identity"]) for run in runs)
    erased = sum(run["window"]["identities"]["erased_equals_reset"] for run in runs)
    cue = sum(sum(run["window"]["cue_followed"]) for run in runs)
    lines = [
        f"- shuffled rows reproducing the donor row's intact sequence: {identity}/"
        f"{inputs.ROWS * len(runs)}; erased equals reset in {erased}/{len(runs)} founders; "
        f"window cue followed in {cue}/{inputs.ROWS * len(runs)} rows",
        f"- founders planned {len(body['planned_founders'])}, completed "
        f"{len(body['completed_founders'])}, capped {body['capped']}, "
        f"run seconds {body['seconds']:.0f}",
        f"- probe continuation equal (actions and saved arrays): {continuation}/{len(runs)}",
        f"- mid-pause continuation equal: {pause}/{len(runs)}",
        f"- protocol sha256 {body['protocol_sha256']}, frozen {body['frozen_protocol']}",
    ]
    for key, items in sorted(teaching.items()):
        lessons = sum(t["lessons_attempted"] for t in items)
        refused = sum(t["work"]["teacher_refusals"] for t in items)
        exhausted = sum(t["work"]["teacher_budget_exhausted"] for t in items)
        agreement = np.mean([t["last_bout_agreement"] for t in items])
        lines.append(
            f"- {key}: lessons {lessons}, refused {refused}, free budget exhausted "
            f"{exhausted}, last-bout agreement with the label {agreement:.2f}, "
            f"flip-flop lessons {sum(t['flipflop_lessons'] for t in items)}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args(argv)
    body = load(args.summary)
    rows = rescored(body)
    print("## Window controls (alternation / agreement / refusals, means over founders)\n")
    print(controls_table(body, rows))
    print("\n## Founders\n")
    print(founder_table(rows))
    print("\n## Disturbances (means over founders)\n")
    print(disturbance_table(body))
    print("\n## Cadence, repair speed and host load\n")
    print(cadence_table(body))
    print("\n## Custody and work\n")
    print(custody_lines(body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
