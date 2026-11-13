"""The night's tables (cadence#334) from the lives' and the exams' result files.

    python report.py <results_dir> > tables.md

``<results_dir>/qc`` and ``<results_dir>/exam`` hold the lives (``<arm>_<founder>.json``) and
``<results_dir>/exam/exams`` the exam files (``exam_<founder>.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

FORMS = ["imitate", "name", "colour_of", "by_colour", "where", "left_of", "next_to", "opposite",
         "material", "can_eat", "can_eat_this", "round", "eat"]


def lives(folder: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for f in sorted(folder.glob("*.json")):
        arm = f.stem.rsplit("_", 1)[0]
        try:
            out.setdefault(arm, []).append(json.loads(f.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def cell(values: list[float]) -> str:
    v = [x for x in values if x == x]
    if not v:
        return ""
    return f"{np.mean(v):.2f} ({min(v):.2f})" if len(v) > 1 else f"{v[0]:.2f}"


def step_table(arms: dict[str, list[dict[str, Any]]], title: str) -> str:
    rows = [f"### {title}", "", "Mean over founders (least founder) of the last step each life "
            "read, frozen greedy, 64 exchanges a form.", "",
            "| Arm | Founders | Parameters | " + " | ".join(FORMS) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in FORMS) + " |"]
    for arm, got in sorted(arms.items()):
        last = [g["steps"][sorted(g["steps"])[-1]] for g in got if g.get("steps")]
        params = int(np.mean([g.get("parameters", 0) for g in got]))
        rows.append(f"| {arm} | {len(got)} | {params:,} | " + " | ".join(
            cell([r.get(f, float("nan")) for r in last]) for f in FORMS) + " |")
    return "\n".join(rows)


def exam_table(exams: list[dict[str, Any]]) -> str:
    if not exams:
        return ""
    out = ["### The owner's five", "", "| Question | Answer | " + " | ".join(
        str(e["seed"]) for e in exams) + " |", "| --- | --- | " + " | ".join(
        "---" for _ in exams) + " |"]
    for i, q in enumerate(exams[0]["owner"]):
        out.append(f"| {q['question']} | {' '.join(q['answer'])} | " + " | ".join(
            f"{' '.join(e['owner'][i]['said']) or '-'} ({e['owner'][i]['right']:.2f})"
            for e in exams) + " |")
    out += ["", "### Probes of the concept code (held-out scenes)", "",
            "| Property | Accuracy | Base rate |", "| --- | --- | --- |"]
    for prop in exams[0]["probes"]["accuracy"]:
        out.append(f"| {prop} | {cell([e['probes']['accuracy'].get(prop, np.nan) for e in exams])}"
                   f" | {cell([e['probes']['base_rate'].get(prop, np.nan) for e in exams])} |")
    out += ["", "### A word heard alone, read by the probes", "",
            "| Word | " + " | ".join(exams[0]["word_alone"]["apple"]) + " |",
            "| --- | " + " | ".join("---" for _ in exams[0]["word_alone"]["apple"]) + " |"]
    for w in exams[0]["word_alone"]:
        out.append(f"| {w} | " + " | ".join(cell([e["word_alone"][w][p] for e in exams])
                                            for p in exams[0]["word_alone"][w]) + " |")
    out += ["", "### Silencing (drop in the forms that need the property / in the others)", "",
            "| Property | Its most tuned units | Its probe direction | As many random units |",
            "| --- | --- | --- | --- |"]
    for prop in exams[0]["silencing"]:
        cols = []
        for m in ("units", "direction", "random"):
            need = [e["silencing"][prop][m]["needed"] for e in exams]
            other = [e["silencing"][prop][m]["others"] for e in exams]
            cols.append(f"{np.nanmean(need):.2f} / {np.nanmean(other):.2f}")
        out.append(f"| {prop} | " + " | ".join(cols) + " |")
    out += ["", "### Neighbours, the new object, changed scenes, the new word", ""]
    for e in exams:
        out.append(f"- founder {e['seed']}: apple's nearest {e['neighbours']['nearest']['apple']}; "
                   f"lemon's nearest {e['neighbours']['nearest']['lemon']}; lemon probes "
                   f"{ {k: round(v, 2) for k, v in e['lemon_probes'].items()} }; "
                   f"'can you eat this?' on a lemon {e['lemon_can_eat_this']:.2f}; changed scene "
                   f"{ {k: round(v, 2) for k, v in e['changed_scene'].items()} }; new word: "
                   f"learned at trial {e['new_word']['learned_at_trial']} after "
                   f"{e['new_word']['pistachios']} pistachios, old names after "
                   f"{e['new_word']['old_names_after']:.2f}")
    return "\n".join(out)


def main(argv: list[str]) -> None:
    root = Path(argv[1])
    parts = []
    if (root / "qc").exists():
        parts.append(step_table(lives(root / "qc"), "QC founders"))
    if (root / "exam").exists():
        parts.append(step_table(lives(root / "exam"), "Exam founders"))
        exams = [json.loads(f.read_text()) for f in sorted((root / "exam" / "exams").glob(
            "exam_*.json"))] if (root / "exam" / "exams").exists() else []
        parts.append(exam_table(exams))
    print("\n\n".join(p for p in parts if p))


if __name__ == "__main__":
    main(sys.argv)
