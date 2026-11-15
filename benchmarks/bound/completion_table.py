"""The completion matrix of ``completion.py`` as text tables.

    python completion_table.py <out.json> [probe]

A row an arm, a column an implied facet: the share read right from the settled state one
moment and two moments after the cue. ``probe`` names the probes' fit ("rooms of 3 to 5", the
default, or "rooms of 1 to 5").
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> None:
    res = json.load(open(argv[1]))
    probe = argv[2] if len(argv) > 2 else "rooms of 3 to 5"
    print(f"records {res.get('records')}, heads {res['heads']}, probes fitted on {probe}")
    for row, cells in res["state completeness"].items():
        print(f"  {row:48s}", " ".join(f"{f} {v:.2f}" for f, v in cells.items()))
    for section in ("a word alone", "one sense alone"):
        for cue, table in res.get(section, {}).items():
            first = next(iter(table.values()))
            facets = [k.split(": ")[1] for k in first if k.startswith(probe + ": ")]
            print(f"\n{section}: {cue} ({first['cues']} cues)")
            print(f"  {'arm':58s}" + "".join(f"{f:>14s}" for f in facets) + "   heads' records")
            for arm, cells in table.items():
                share = cells.get("heads' records named by the cue's word")
                print(f"  {arm:58s}" + "".join(
                    f"{cells[f'{probe}: {f}'][0]:>8.2f} {cells[f'{probe}: {f}'][1]:.2f} "
                    for f in facets) + ("" if share is None else f"   {share:.2f}"))
            for arm, cells in table.items():
                if "came back" in cells and "code" in arm and "agree" in arm:
                    print(f"  came back ({arm.split(': ')[1]}):",
                          ", ".join(f"{k} {v:.2f}" for k, v in cells["came back"].items()))
    for arm, cells in res.get("a thing seen, nothing heard", {}).items():
        print(f"a thing seen, nothing heard: {arm}:",
              ", ".join(f"{k} {v:.2f}" for k, v in cells.items()))


if __name__ == "__main__":
    main(sys.argv)
