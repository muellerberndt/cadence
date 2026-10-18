"""Markdown tables from the campaign summaries: work per change moment and per episode.

    python benchmarks/local_work/tables.py /path/out/summary-skills.json

Every number is the mean over the founders of the cases that share an arm and an anatomy
size; the founders' spread is given as the smallest and largest value.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _fmt(values: list[float], digits: int = 0) -> str:
    if not values:
        return "n/a"
    mean = _mean(values)
    low, high = min(values), max(values)
    if digits == 0:
        return f"{mean:,.0f} ({low:,.0f} to {high:,.0f})"
    return f"{mean:.{digits}f} ({low:.{digits}f} to {high:.{digits}f})"


def groups(summaries: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        key = (s["arm"], "x".join(str(w) for w in s["widths"]))
        out[key].append(s)
    return out


def episode_totals(s: dict[str, Any], kind: str) -> dict[str, float]:
    """Change moment plus the ripple held after it, per episode of that kind."""
    change, hold = s["by_kind"][kind]["change"], s["by_kind"][kind]["hold"]
    if not change:
        return {}
    episodes = change["moments"]
    out = {}
    for key in change:
        if key == "moments":
            continue
        if key.endswith("largest_cone"):
            out[key] = max(change[key], hold.get(key, 0))
        else:
            out[key] = (change[key] + hold.get(key, 0)) / episodes
    out["held"] = hold.get("moments", 0) / episodes
    return out


def table(summaries: list[dict[str, Any]], kind: str, scope: str) -> str:
    rows = [
        "| arm | widths | n | E | block 0 E | acquired | full sweeps | full updates | "
        "full contacts (effective) | full checks | local rounds | local updates | "
        "local messages | local check contacts | local cone | sync sweeps | "
        "updates local/full | contacts local/full | seconds local/full |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (arm, widths), cases in sorted(groups(summaries).items(), key=lambda kv: (
        kv[0][0], [int(w) for w in kv[0][1].split("x")]
    )):
        totals = []
        for s in cases:
            if scope == "change":
                t = dict(s["by_kind"][kind]["change"])
                if not t:
                    continue
                m = t.pop("moments")
                totals.append({k: (v / m if not k.endswith("largest_cone") else v)
                               for k, v in t.items()})
            else:
                t = episode_totals(s, kind)
                if t:
                    totals.append(t)
        if not totals:
            continue

        def col(key: str, rows: list[dict[str, float]] = totals) -> list[float]:
            return [float(t[key]) for t in rows]

        ratio_u = [t["local_updates"] / t["full_updates"] for t in totals]
        ratio_c = [t["local_contacts"] / t["full_contacts_effective"] for t in totals]
        ratio_s = [t["local_seconds"] / t["full_seconds"] for t in totals]
        rows.append(
            f"| {arm} | {widths} | {cases[0]['n']} | {cases[0]['E']:,} | "
            f"{cases[0]['block0_contacts']:,} | "
            f"{sum(1 for s in cases if s['acquired'])}/{len(cases)} | "
            f"{_fmt(col('full_sweeps'), 1)} | {_fmt(col('full_updates'))} | "
            f"{_fmt(col('full_contacts_effective'))} | {_fmt(col('full_checks'), 1)} | "
            f"{_fmt(col('local_rounds'), 1)} | {_fmt(col('local_updates'))} | "
            f"{_fmt(col('local_messages'))} | {_fmt(col('local_check_contacts'))} | "
            f"{_fmt(col('local_largest_cone'))} | {_fmt(col('sync_sweeps'), 1)} | "
            f"{_fmt(ratio_u, 3)} | {_fmt(ratio_c, 3)} | {_fmt(ratio_s, 2)} |"
        )
    return "\n".join(rows)


def gate_table(summaries: list[dict[str, Any]]) -> str:
    rows = [
        "| case | acquired (lessons) | accuracy | row mass | moments | same action | "
        "local qualified | state within gate | max activation distance | settle-in held "
        "(full, local stationary) | over budget |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for s in sorted(summaries, key=lambda s: (s["arm"], s["S"], s["widths"], s["founder"])):
        g = s["gates"]
        settle = next(e for e in s["episodes"] if e["kind"] == "settle-in")
        rows.append(
            f"| {s['name']} | {s['acquired']} ({s['lessons']}) | "
            f"{' '.join(f'{a:.2f}' for a in s['accuracy'])} | {s['row_mass']:.1f} | "
            f"{g['moments']} | {g['same_action']} | {g['local_qualified']} | "
            f"{g['state_within_gate']} | {g['max_distance_activation']:.3g} | "
            f"{settle['held']} ({settle['stationary_full']}, {settle['stationary_local']}) | "
            f"{s.get('over_budget', False)} |"
        )
    return "\n".join(rows)


def stationary_table(summaries: list[dict[str, Any]]) -> str:
    rows = [
        "| arm | widths | stationary moments | full checks per moment | full contacts per "
        "moment | local updates per moment | local check contacts per moment |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (arm, widths), cases in sorted(groups(summaries).items(), key=lambda kv: (
        kv[0][0], [int(w) for w in kv[0][1].split("x")]
    )):
        st = [s["stationary"] for s in cases if s["stationary"]]
        if not st:
            continue
        m = [t["moments"] for t in st]
        rows.append(
            f"| {arm} | {widths} | {_fmt([float(v) for v in m])} | "
            f"{_fmt([t['full_checks'] / t['moments'] for t in st], 2)} | "
            f"{_fmt([t['full_contacts_effective'] / t['moments'] for t in st])} | "
            f"{_fmt([t['local_updates'] / t['moments'] for t in st], 2)} | "
            f"{_fmt([t['local_check_contacts'] / t['moments'] for t in st])} |"
        )
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    summaries: list[dict[str, Any]] = []
    for path in argv:
        summaries.extend(json.loads(Path(path).read_text()))
    for kind in ("input", "relation"):
        for scope in ("change", "episode"):
            print(f"\n### {kind} disturbance, {scope}\n")
            print(table(summaries, kind, scope))
    print("\n### stationary moments\n")
    print(stationary_table(summaries))
    print("\n### gates per case\n")
    print(gate_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
