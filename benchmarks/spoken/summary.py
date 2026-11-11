"""Tables of the spoken-phrase chamber's readings (cadence#303), from ``spoken.py`` results.

    python benchmarks/spoken/summary.py results/*.jsonl

Base readings at each budget (means over founders, test word accuracy and exact phrases, the
share at word steps said right, another word or silence), the probes at the word steps, and the
adaptation curves: each speaker's own word accuracy and the retention panel at each minute,
with the change from 0 minutes paired by founder and speaker.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any

import numpy as np


def load(paths: list[str]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        with open(path) as handle:
            out += [json.loads(line) for line in handle if line.strip()]
    return out


def mean(values: list[float]) -> str:
    return f"{np.mean(values):.3f}" if values else ""


def label(r: dict[str, Any]) -> str:
    extra = r.get("extra")
    return r["learner"] + (f" {json.dumps(extra, sort_keys=True)}" if extra else "")


def base_table(results: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| Learner | Parameters | Phrases | Founders | Word | Exact | At word steps: right / "
        "other / silence | Validation word |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    by: dict[tuple[str, int], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for r in results:
        for reading in r.get("readings", []):
            by[(label(r), reading["phrases"])].append((r["parameters"], reading))
    for (name, phrases), entries in sorted(by.items()):
        tests = [e[1]["test"] for e in entries]
        at = [t["at_word"] for t in tests if "at_word" in t]
        shares = (
            " / ".join(mean([a[k] for a in at]) for k in ("right", "other", "silence"))
            if at
            else ""
        )
        params = sorted({e[0] for e in entries})
        rows.append(
            f"| {name} | {', '.join(f'{p:,}' for p in params)} | {phrases:,} | {len(entries)} | "
            f"{mean([t['word_accuracy'] for t in tests])} | {mean([t['exact'] for t in tests])} | "
            f"{shares} | {mean([e[1]['validation']['word_accuracy'] for e in entries])} |"
        )
    return rows


def probe_table(results: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| Learner | Founder | Read of | Width | Fit (held-out quarter) | Base test "
        "| New speakers |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda r: (label(r), r["founder"])):
        for view, p in r.get("probe", {}).items():
            others = [v for k, v in p["panels"].items() if k != "base_test"]
            rows.append(
                f"| {label(r)} | {r['founder']} | {view} | {p['width']} | {p['validation']:.3f} | "
                f"{p['panels']['base_test']:.3f} | {mean(others)} |"
            )
    return rows


def adaptation_table(results: list[dict[str, Any]]) -> list[str]:
    rows = [
        "| Learner | Parameters | Speakers | Mode | Rate | Minutes | Phrases | Own word "
        "| Own change | Retention word | Retention change | s per correction |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    by: dict[
        tuple[str, str, str, float, float], list[tuple[dict[str, Any], dict[str, Any], int]]
    ] = defaultdict(list)
    for r in results:
        for a in r.get("adaptation", []):
            group = a["speaker"].split("/")[0]
            start = a["curve"][0]
            for c in a["curve"]:
                by[(label(r), group, a["mode"], a["rate"], c["minutes"])].append(
                    (c, start, r["parameters"])
                )
    for (name, group, mode, rate, minutes), entries in sorted(by.items()):
        own = [c["own"]["word_accuracy"] for c, _, _ in entries]
        own_change = [c["own"]["word_accuracy"] - s["own"]["word_accuracy"] for c, s, _ in entries]
        kept = [c["retention"]["word_accuracy"] for c, _, _ in entries]
        kept_change = [
            c["retention"]["word_accuracy"] - s["retention"]["word_accuracy"] for c, s, _ in entries
        ]

        def spread(values: list[float]) -> str:
            if len(values) < 2:
                return mean(values)
            return (
                f"{np.mean(values):+.3f} ± {2 * np.std(values, ddof=1) / np.sqrt(len(values)):.3f}"
            )

        params = ", ".join(f"{p:,}" for p in sorted({p for _, _, p in entries}))
        taught = mean([c["phrases"] for c, _, _ in entries])
        per = mean([c["seconds_per_correction"] for c, _, _ in entries])
        rows.append(
            f"| {name} | {params} | {group} ({len(entries)}) | {mode} | {rate:g} | {minutes:g} "
            f"| {taught} | {mean(own)} | {spread(own_change)} | {mean(kept)} "
            f"| {spread(kept_change)} | {per} |"
        )
    return rows


def main() -> None:
    results = [r for r in load(sys.argv[1:]) if "error" not in r]
    errors = [r for r in load(sys.argv[1:]) if "error" in r]
    for title, table in (
        ("Base readings", base_table(results)),
        ("Probes at word steps", probe_table(results)),
        ("Adaptation", adaptation_table(results)),
    ):
        print(f"\n### {title}\n")
        print("\n".join(table))
    for r in errors:
        print(f"\nerror: {r['spec']['learner']} {r['spec']['founder']}: {r['error']}")


if __name__ == "__main__":
    main()
