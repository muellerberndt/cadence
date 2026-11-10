"""The gate of cadence#296 over a run's lives: a learning brain against its own frozen newborn.

    python benchmarks/actor_latch/gate.py --doom DIR/doom --chamber DIR/chamber

Doom lives are ``doom_life.py`` output files named ``{scenario}-{arm}-s{founder}.jsonl``;
chamber lives are ``chamber.py`` output files named ``{world}-{arm}-founder-{founder}.json``.
For each founder, world and learning arm the reading is the mean return over the life's last
quarter of episodes against the frozen newborn's (``frozen``) over the same episodes: the arm
passes on that founder when its mean exceeds the newborn's by two standard errors of the
difference (``sqrt(s1^2 / n1 + s2^2 / n2)``). A world passes when 5 of 6 founders pass, and the
arm passes when at least two worlds of the Doom body pass. Beside it: each life's button shares
over its last quarter and the largest of them.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CONTROLS = ("frozen", "random", "conv")


def quarter(values: list[float]) -> list[float]:
    """The last quarter of a life's episodes."""
    return values[-max(len(values) // 4, 1):]


def doom_lives(directory: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out = {}
    for path in sorted(directory.glob("*.jsonl")):
        match = re.fullmatch(r"(.+)-([a-z0-9.]+)-s(\d+)", path.stem)
        if match is None:
            continue
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        rows = [line for line in lines[1:] if "episode" in line]
        out[(match[1], match[2], int(match[3]))] = {
            "returns": [r["return"] for r in rows],
            "parameters": lines[0]["run"]["parameters"],
            "shares": None,
        }
    return out


def chamber_lives(directory: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    out = {}
    for path in sorted(directory.glob("*.json")):
        match = re.fullmatch(r"(.+?)-([a-z0-9.]+)-founder-(\d+)", path.stem)
        if match is None:
            continue
        life = json.loads(path.read_text())
        rows = life["episodes"]
        last = rows[-max(len(rows) // 4, 1):]
        buttons = np.array([r["actions"] for r in last]).sum(axis=0)
        out[(match[1], match[2], int(match[3]))] = {
            "returns": [r["return"] for r in rows],
            "parameters": life["parameters"],
            "shares": (buttons / buttons.sum()).tolist(),
            "paying": life["world"]["paying"] if life["chamber"] != "aim" else None,
        }
    return out


def compare(lives: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per world, arm and founder: the last quarter's mean, the newborn's, the
    difference over its standard error and whether it passes."""
    rows = []
    for (world, arm, founder), life in sorted(lives.items()):
        if arm == "frozen":
            continue
        newborn = lives.get((world, "frozen", founder))
        mine = quarter(life["returns"])
        row: dict[str, Any] = {
            "world": world, "arm": arm, "founder": founder, "parameters": life["parameters"],
            "return": float(np.mean(mine)), "whole": float(np.mean(life["returns"])),
            "shares": life["shares"], "paying": life.get("paying"),
        }
        if newborn is not None:
            theirs = quarter(newborn["returns"])
            se = float(np.sqrt(np.var(mine, ddof=1) / len(mine)
                               + np.var(theirs, ddof=1) / len(theirs)))
            difference = row["return"] - float(np.mean(theirs))
            row.update(frozen=float(np.mean(theirs)), z=difference / se if se > 0 else 0.0)
            row["passes"] = bool(difference > 2 * se)
        rows.append(row)
    return rows


def table(rows: list[dict[str, Any]], doom: bool) -> None:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by[(row["arm"], row["world"])].append(row)
    print("| Arm | World | Parameters | Founders above the newborn by 2 SE | Return, last quarter "
          "(mean ± sd over founders) | Newborn | Whole life | Largest button share, last quarter |")
    print("| --- | --- | ---: | ---: | --- | ---: | ---: | --- |")
    worlds: dict[str, int] = defaultdict(int)
    for (arm, world), part in sorted(by.items()):
        returns = [r["return"] for r in part]
        passed = [r["passes"] for r in part if "passes" in r]
        frozen = [r["frozen"] for r in part if "frozen" in r]
        largest = [max(r["shares"]) for r in part if r["shares"] is not None]
        shares = "" if not largest else ", ".join(f"{v:.2f}" for v in largest)
        if passed and sum(passed) >= 5:
            worlds[arm] += 1
        print(
            f"| {arm} | {world} | {part[0]['parameters']:,} | "
            f"{f'{sum(passed)} of {len(passed)}' if passed else ''} | "
            f"{np.mean(returns):+.3f} ± {np.std(returns, ddof=1) if len(returns) > 1 else 0:.3f} | "
            f"{f'{np.mean(frozen):+.3f}' if frozen else ''} | "
            f"{np.mean([r['whole'] for r in part]):+.3f} | {shares} |"
        )
    if doom:
        print()
        for arm in sorted({r["arm"] for r in rows} - set(CONTROLS)):
            print(f"- {arm}: {worlds[arm]} of 3 scenarios at 5 of 6 founders; the gate "
                  f"{'passes' if worlds[arm] >= 2 else 'fails'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--doom", type=Path)
    parser.add_argument("--chamber", type=Path)
    parser.add_argument("--founders", action="store_true", help="one row per founder as well")
    args = parser.parse_args(argv)
    for directory, read, doom in ((args.doom, doom_lives, True),
                                  (args.chamber, chamber_lives, False)):
        if directory is None:
            continue
        rows = compare(read(directory))
        table(rows, doom)
        if args.founders:
            print()
            for row in rows:
                shares = "" if row["shares"] is None else ", ".join(
                    f"{v:.2f}" for v in row["shares"])
                frozen, z = row.get("frozen", float("nan")), row.get("z", float("nan"))
                print(f"| {row['world']} | {row['arm']} | {row['founder']} | {row['return']:+.3f} "
                      f"| {frozen:+.3f} | {z:+.1f} | {shares} |")
        print()


if __name__ == "__main__":
    main()
