"""The page's interference test (cadence#334): paid namings of one focus, then its colour answer.

    python page_test.py <page folder> [amount] [first]

Through the page's own host (``MeaningParrot``) in the owner's room: "what colour has the
apple?" before, then 1, 2, 5 and 10 paid namings ("what is this?" with the apple clicked, a
pistachio of ``amount`` for the reply), and after each count the colour answer again, the name,
and the cup's colour. A brain whose question-dependent answers share one readout loses the
colour answer to the naming lessons; one whose question and thing meet in a sparse mixed code
keeps it (review-topdown, section 6.6).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import introspect as ii  # noqa: E402
import night as nt  # noqa: E402

from cadence.meaning.host import MeaningParrot  # noqa: E402


def things(focus: str | None) -> list[dict[str, Any]]:
    room = ii.owner_room()
    return [{"kind": t.kind, "colour": t.colour, "material": t.material,
             "direction": nt.bearing(room.heading, t.place),
             "distance": float(math.hypot(*t.place)), "focus": t.kind == focus}
            for t in room.things]


def ask(m: MeaningParrot, words: list[str], focus: str | None) -> list[str]:
    m.brain.reset()
    m.own = None
    m.turn_over()
    m.see(things(focus))
    for w in words:
        m.hear(w)
        m.moment(more=True)
    said: list[str] = []
    for _ in range(nt.TURN):
        w = m.moment()
        if w is None and said:
            break
        if w is not None and (not said or said[-1] != w):
            said.append(w)
    return said


def main(argv: list[str]) -> None:
    folder = Path(argv[1])
    amount = float(argv[2]) if len(argv) > 2 else 1.0
    first = len(argv) > 3 and argv[3] == "first"
    m = MeaningParrot(folder / "brain.npz", folder / "spec.json", seed=1)
    colour = ["what", "colour", "has", "the", "apple"]
    out: dict[str, Any] = {"amount": amount, "first": first,
                           "before": {"colour": ask(m, colour, "apple"),
                                      "name": ask(m, ["what", "is", "this"], "apple")},
                           "after": {}}
    paid = 0
    for count in (1, 2, 5, 10):
        while paid < count:
            ask(m, ["what", "is", "this"], "apple")
            m.pistachio(amount, first=first)
            paid += 1
        out["after"][count] = {"colour": ask(m, colour, "apple"),
                               "name": ask(m, ["what", "is", "this"], "apple"),
                               "cup colour": ask(m, ["what", "colour", "has", "the", "cup"],
                                                 "cup")}
    (folder / f"page_test_{amount}{'_first' if first else ''}.json").write_text(
        json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv)
