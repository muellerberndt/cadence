"""The brain's hand-set values, each beside its control (cadence#326 audit).

The lane's rules: a designed part of the brain is a gene with the hand-set value as its
control. Each variant below is one life of the founder's script with one value changed, read
on the chamber's gates as ``words.life`` reads them:

- ``default``: no value changed;
- ``shares_equal``: a naming's shares of kind, colour and place at (1/3, 1/3, 1/3), against
  (.4, .3, .3);
- ``shares_colour``: (.3, .4, .3), the colour first;
- ``rate_one``: a lesson closes all of its error, so a word holds its latest naming (against
  1/n, no less than .05);
- ``threshold_zero``: the word neurons' threshold at 0 (against .15).

    python genes.py <founder> <out> [short] [variant ...]
"""

from __future__ import annotations

import sys
from typing import Any

import common

VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "shares_equal": {"shares": (1 / 3, 1 / 3, 1 / 3)},
    "shares_colour": {"shares": (0.3, 0.4, 0.3)},
    "rate_one": {"rate": 1.0},
    "threshold_zero": {"threshold": 0.0},
}
GATES = ["W1", "W1_named_only", "W2", "W3", "W4_never_seen", "W4_never_taught", "W5_by_colour",
         "W5_by_place", "W6_ablated"]


def life(seed: int, settings: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    import words

    from cadence.words import Lexicon

    given = dict(variant)
    brain = None
    if "threshold" in given:
        brain = words.make_brain(seed)
        concepts = {g: s.stop - s.start for g, s in brain.lexicon.groups.items()}
        brain.lexicon = Lexicon(concepts, threshold=float(given.pop("threshold")))
    return words.life(seed, brain=brain, **settings, **given)


def main(argv: list[str]) -> None:
    seed, out, settings = common.arguments(argv[:4])
    chosen = [a for a in argv[3:] if a in VARIANTS] or list(VARIANTS)
    result: dict[str, Any] = {"seed": seed, "settings": settings, "variants": {}}
    for name in chosen:
        read = life(seed, settings, VARIANTS[name])
        result["variants"][name] = {g: read.get(g) for g in GATES}
    print(common.write(out, "genes", seed, result))


def table(lives: list[dict[str, Any]]) -> None:
    import numpy as np

    names = list(lives[0]["variants"])
    print("| variant | " + " | ".join(GATES) + " |")
    print("|" + " --- |" * (len(GATES) + 1))
    for name in names:
        row = [name]
        for g in GATES:
            values = [life["variants"][name][g] for life in lives
                      if life["variants"][name].get(g) is not None]
            row.append(f"{np.mean(values):.3f} ({min(values):.2f})" if values else "")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "read":
        import json
        from pathlib import Path

        table([json.loads(p.read_text()) for p in sorted(Path(sys.argv[2]).glob("genes_*.json"))])
    else:
        main(sys.argv)
