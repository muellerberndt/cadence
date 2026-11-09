"""Where a saved brain's whole sentences go wrong, on the wordpiece language.

Closed loop, each sentence's first wrong word: its kind (a word's first or later piece, its number
piece, ``near``, ``end``) and its slot, and what was said instead. Teacher-forced, every word by
kind, the ones the read has no slot for among them (``near``, ``end``, a verb's agreeing piece).

    python benchmarks/vocabulary/diagnose.py --brain DIR/brain --lexicon LEX --swap --founder 0
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import life
import numpy as np
from pieces import PieceLanguage


def kinds_of(lang: PieceLanguage, row: object) -> list[str]:
    out = []
    target = row.target  # type: ignore[attr-defined]
    meta = row.meta  # type: ignore[attr-defined]
    for j, word in enumerate(target):
        name = "grammar"
        for slot, a, b in meta.get("slots", []):
            if a <= j < b:
                place = "number" if j == b - 1 else "first" if j == a else "later"
                name = f"{slot}_{place}"
        if word == lang.near:
            name = "near"
        if word == lang.end:
            name = "end"
        out.append(name)
    return out


def said_as(lang: PieceLanguage, words: list[int], at: int) -> str:
    """What kind of word stands where a sentence first goes wrong."""
    if at >= len(words):
        return "(stopped)"
    word = words[at]
    if word == lang.near:
        return "near"
    if word == lang.end:
        return "end"
    if word in (lang.s, lang.zero):
        return "##s or ##0"
    if word >= lang.tokens_pieces:
        return "bracket"
    return "later piece" if lang.piece_names[word].startswith("##") else "first piece"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--brain", type=Path, required=True)
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--pseudo", type=float, default=0.0)
    parser.add_argument("--founder", type=int, default=0)
    parser.add_argument("--panels", nargs="*", default=["taught", "novel_all", "swap_novel"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")
    import cadence as cd

    lang = PieceLanguage(args.lexicon, "heard", args.swap, None, False, args.pseudo)
    tests = lang.panels(args.founder, 8)
    report: dict[str, object] = {}
    for name in args.panels:
        brain = cd.Brain.load(args.brain)
        first: Counter[str] = Counter()
        instead: Counter[str] = Counter()
        by_shape: dict[str, list[int]] = {}
        for rows in tests[name]:
            said = life.speak(lang, brain, rows, teach=False)
            for row, words in zip(rows, said, strict=True):
                kinds = kinds_of(lang, row)
                key = f"L{row.meta.get('L', 0)}"
                by_shape.setdefault(key, []).append(int(words == row.target))
                wrong = next((j for j, w in enumerate(row.target)
                              if j >= len(words) or words[j] != w), None)
                if wrong is None:
                    first["right"] += 1
                    continue
                first[kinds[wrong]] += 1
                instead[f"{kinds[wrong]} -> {said_as(lang, words, wrong)}"] += 1
        forced: dict[str, list[int]] = {}
        for rows in tests[name]:
            brain.reset()
            for t in range(len(rows[0].pre)):
                brain.act(lang.dense([r.pre[t] for r in rows]), greedy=True)
            for j in range(len(rows[0].target)):
                x = (lang.dense([(lang.go,)] * len(rows)) if j == 0
                     else lang.dense([lang.spoken(r.target[j - 1]) for r in rows]))
                action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
                for i, r in enumerate(rows):
                    k = kinds_of(lang, r)[j]
                    forced.setdefault(k, []).append(int(action[i] == r.target[j]))
        report[name] = {
            "sentences": sum(first.values()),
            "first_wrong": first.most_common(),
            "instead": instead.most_common(16),
            "exact_by_shape": {k: round(float(np.mean(v)), 3) for k, v in sorted(by_shape.items())},
            "forced": {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(forced.items())},
        }
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
