"""The question's path through the brain (cadence#326 audit): which steps settle, which are
written by the program, and what the lexicon holds after a life.

1. **The steps** of ``Words.ask`` and ``Lexicon.say`` in the source, each line that chooses
   (an argmax, a dispatch on a group or a read, a stored table), printed with its place.
2. **The motor**: for every word the lexicon holds, the letters the motor says are compared
   with the form it was heard by; then one word's stored letters are written over and the
   brain is asked a question whose answer is that word.
3. **The synapses against a count table**: the lexicon's word-to-concept synapses after the
   life, against the running mean of each word's lesson targets (``lookup.Table`` fed the
   same lessons): their correlation, and whether each word's most active concept on hearing
   is the table's.

    python path.py <founder> <out> [short]
"""

from __future__ import annotations

import inspect
import re
import sys
from typing import Any

import common
import numpy as np
import world
from lookup import Table

CHOOSES = re.compile(r"argmax|READS\[|group == |if READS|max\(\(|max\(self\.answers|letters\[|"
                     r"\.letters\b|np\.eye")


def steps() -> list[str]:
    from cadence.words import lexicon, room

    out = []
    for module, names in ((room, ("Words._heard", "Words.hear", "Words.find", "Words.answer",
                                  "Words.ask", "Words.practice", "Senses.describe",
                                  "Senses.find")),
                          (lexicon, ("Lexicon.say", "Lexicon._enter"))):
        for name in names:
            owner, _, attr = name.partition(".")
            obj = getattr(getattr(module, owner, None), attr, None)
            if obj is None:
                continue
            lines, start = inspect.getsourcelines(obj)
            for i, line in enumerate(lines):
                if CHOOSES.search(line) and not line.strip().startswith(("#", '"')):
                    out.append(f"{module.__name__}:{start + i} ({name}): {line.strip()}")
    return out


def motor(brain: Any, lex: world.Lexicon) -> dict[str, Any]:
    from cadence.words import LETTERS

    lexicon = brain.lexicon
    said = []
    for u, form in enumerate(lexicon.forms):
        letters = lexicon.letters[u].argmax(axis=1)
        said.append("".join(LETTERS[c] for c in letters if c < len(LETTERS)) == form)
    # one word's stored letters written over: the motor says what is stored
    target = lex.colour[0]
    unit = lexicon.unit(target)
    saved = lexicon.letters[unit].copy()
    lexicon.letters[unit] = 0.0
    for place, letter in enumerate("zzzz"):
        lexicon.letters[unit, place, LETTERS.index(letter)] = 1.0
    lexicon.letters[unit, 4:, len(LETTERS)] = 1.0
    asked = lexicon.say(("colour", 0))
    lexicon.letters[unit] = saved
    return {"words": len(lexicon.forms), "letters_equal_heard_form": float(np.mean(said)),
            "said_after_letters_written_over": asked, "word_written_over": target}


def compare(brain: Any, table: Table, lex: world.Lexicon) -> dict[str, Any]:
    lexicon = brain.lexicon
    pre, post = np.asarray(lexicon.connectome.pre), np.asarray(lexicon.connectome.post)
    efficacy = np.asarray(lexicon.graph.efficacy, dtype=np.float64)
    where = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(pre, post, strict=True))}
    named = lex.kind + lex.colour + lex.place
    cols = np.concatenate([np.arange(lexicon.groups[g].start, lexicon.groups[g].stop)
                           for g in ("kind", "colour", "place")])
    english = lex.english()
    synapses, rows, agree, margin, who = [], [], [], [], []
    for w in named:
        u = lexicon.unit(w)
        if u is None or table.unit(w) is None:
            continue
        synapses.append([efficacy[where[(int(lexicon.word[u]), int(lexicon.concept[c]))]]
                         for c in cols])
        rows.append(table.rows[table.unit(w), cols])
        heard = lexicon.hear(w)[cols]
        agree.append(int(np.argmax(heard)) == int(np.argmax(rows[-1])))
        top = np.sort(heard)[::-1]
        margin.append(float(top[0] - top[1]))
        who.append(english[w])
    s, t = np.asarray(synapses).ravel(), np.asarray(rows).ravel()
    return {"words": len(rows), "synapses_vs_table_correlation": float(np.corrcoef(s, t)[0, 1]),
            "most_active_concept_is_tables": float(np.mean(agree)),
            "least_margin_first_to_second_concept": float(min(margin)) if margin else None,
            "least_margin_word": who[int(np.argmin(margin))] if margin else None,
            "margins": {w: round(m, 4) for w, m in zip(who, margin, strict=True)}}


def main(argv: list[str]) -> None:
    import words

    seed, out, settings = common.arguments(argv)
    settings = {**words.DEFAULTS, **settings}
    brain = words.make_brain(seed)
    concepts = {g: s.stop - s.start for g, s in brain.lexicon.groups.items()}
    table = Table(concepts)
    lessons = brain.lexicon.name

    def name(words_: Any, concepts_: Any, *, groups: Any = None) -> float:
        table.name(words_, concepts_, groups=groups)
        return float(lessons(words_, concepts_, groups=groups))

    brain.lexicon.name = name  # every lesson also fed to the table
    words.life(seed, brain=brain, **settings)
    del brain.lexicon.name
    lex = world.lexicon(seed)
    result = {"seed": seed, "settings": settings, "steps": steps(), "motor": motor(brain, lex),
              "table": compare(brain, table, lex)}
    print(common.write(out, "path", seed, result))


if __name__ == "__main__":
    main(sys.argv)
