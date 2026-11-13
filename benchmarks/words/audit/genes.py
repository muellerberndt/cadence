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
- ``threshold_zero``: the word neurons' threshold at 0 (against .15);
- ``say_once``: the motor says the most active word without hearing it first (``say(tries=1)``,
  at a head whose ``Lexicon.say`` takes ``tries``; against 3).

    python genes.py <founder> <out> [short] [variant ...]
"""

from __future__ import annotations

import sys
from typing import Any

import common

VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "shares_equal": {"shares": {"kind": 1 / 3, "colour": 1 / 3, "place": 1 / 3}},
    "shares_colour": {"shares": {"kind": 0.3, "colour": 0.4, "place": 0.3}},
    "rate_one": {"rate": 1.0},
    "threshold_zero": {"threshold": 0.0},
    "say_once": {"tries": 1},
}


def _shares(given: dict[str, Any]) -> dict[str, Any]:
    """``shares`` as the head's ``Words`` takes it: a dict by group, or (kind, colour, place)."""
    import inspect

    from cadence.words import Words

    if "shares" in given and inspect.signature(Words).parameters["shares"].default is not None:
        given = {**given, "shares": tuple(given["shares"][g] for g in ("kind", "colour",
                                                                       "place"))}
    return given
GATES = ["W1", "W1_named_only", "W2", "W3", "W4_never_seen", "W4_never_taught", "W5_by_colour",
         "W5_by_place", "W6_ablated"]


def life(seed: int, settings: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    import words

    from cadence.words import Lexicon

    given = _shares(dict(variant))
    brain = None
    if "threshold" in given:
        brain = words.make_brain(seed)
        concepts = {g: s.stop - s.start for g, s in brain.lexicon.groups.items()}
        brain.lexicon = Lexicon(concepts, threshold=float(given.pop("threshold")))
    if "tries" in given:
        import functools
        import inspect

        tries = int(given.pop("tries"))
        brain = brain if brain is not None else words.make_brain(seed)
        if "tries" not in inspect.signature(brain.lexicon.say).parameters:
            return {"absent": "say takes no tries at this head"}
        brain.lexicon.say = functools.partial(brain.lexicon.say, tries=tries)
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
