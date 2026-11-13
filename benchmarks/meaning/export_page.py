"""The files a page loads, from a trained brain, checked through the page's own host (cadence#334).

    python export_page.py <brain.npz> <founder> <out_dir>

Writes ``brain.npz`` and ``spec.json``, then loads them with ``cadence.meaning.host`` and asks
the owner's five questions in the owner's room as a page would (the thing asked about clicked,
the words one a moment with the voice held, then the parrot's turn in the gap: three moments, or
one a thing for a question about the scene), and writes the replies to ``page_check.json``.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import introspect as ii  # noqa: E402
import night as nt  # noqa: E402

from cadence.meaning.host import MeaningParrot  # noqa: E402


def ask(m: MeaningParrot, things: list[dict[str, Any]], words: list[str], scan: bool
        ) -> list[str]:
    """One exchange through the host: the words one a moment, then the turn."""
    m.brain.reset()
    m.own = None
    m.see(things)
    for w in words:  # the trainer speaks; the voice is held
        m.hear(w)
        m.moment(more=True)
    said: list[str] = []
    looked = [t for t in things if not t.get("focus")] if scan else [None] * nt.TURN
    for t in looked:  # the turn: the gap after the trainer stops (a scan: one thing a moment)
        if t is not None:
            m.see(things, glimpse=t)
        w = m.moment()
        if w is None and said and not scan:
            break
        if w is not None and (not said or said[-1] != w):
            said.append(w)
    m.turn_over()
    return said


def main(argv: list[str]) -> None:
    brain, seed, out = argv[1], int(argv[2]), Path(argv[3])
    out.mkdir(parents=True, exist_ok=True)
    p = nt.Parrot(seed, streams=1)
    nt.export_spec(p, out / "spec.json")
    shutil.copyfile(brain, out / "brain.npz")
    m = MeaningParrot(out / "brain.npz", out / "spec.json", seed=seed)
    room = ii.owner_room()
    check = []
    for words, answer, focus in ii.OWNER:
        things = [{"kind": t.kind, "colour": t.colour, "material": t.material,
                   "direction": nt.bearing(room.heading, t.place),
                   "distance": float(math.hypot(*t.place)), "focus": t.kind == focus}
                  for t in room.things]
        scan = words[1] == "objects" or words[2] == "left"
        said = ask(m, things, list(words), scan)
        check.append({"question": " ".join(words), "answer": answer, "said": said})
    (out / "page_check.json").write_text(json.dumps(check, indent=1))
    print(json.dumps(check))


if __name__ == "__main__":
    main(sys.argv)
