"""The exchanges of a step's demo, as ``demo.py`` takes them (cadence#334).

    python imitation_demo.py <brain.npz> <founder> <out.json> ['<json genome>'] [count] [form]

``imitate`` (N0, the default): the trainer says one word in a room of things and the parrot says
it back at once; the question's audio is the trainer's word (``take_<word>.wav``). ``name`` (N1):
the thing in focus (clicked), "What is this?" (``q05.wav``), and the parrot's turn in the gap.
Each exchange is read greedily on a frozen copy of the saved brain; the parrot's voice replays its
take of each word it said.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exam as ex  # noqa: E402
import night as nt  # noqa: E402


def main(argv: list[str]) -> None:
    brain, seed, out = argv[1], int(argv[2]), Path(argv[3])
    genome = json.loads(argv[4]) if len(argv) > 4 else None
    count = int(argv[5]) if len(argv) > 5 else 12
    form = argv[6] if len(argv) > 6 else "imitate"
    p = ex.load(brain, seed, genome, streams=count)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 77]))
    held = nt.founder_held(seed)
    exs = nt.imitate_block(p, rng, held) if form == "imitate" else nt.block(
        p, rng, form, "turn", held)
    got = nt.live(p, exs, rng, learn=False)
    rows = []
    for e, reply, right in zip(exs, got["replies"], got["right"], strict=True):
        word = e.heard[0] if form == "imitate" else None
        assert e.question is not None
        text = f"(the trainer says) {word}" if word else " ".join(
            w for w in e.heard if w).capitalize() + "?"
        audio = f"take_{word}.wav" if word else "q05.wav"
        rows.append({
            "things": [{"kind": t.kind, "colour": t.colour, "material": t.material,
                        "direction": round(math.degrees(nt.bearing(e.room.heading, t.place)), 1),
                        "distance": round(float(np.linalg.norm(t.place)), 2),
                        "focus": bool(t.focus)}
                       for t in e.room.things],
            "question_text": text, "question_audio": audio, "answer": list(e.question.answer),
            "said": list(reply), "right": bool(right)})
    out.write_text(json.dumps(rows, indent=1))
    print(json.dumps({"right": float(np.mean(got["right"])), "count": len(rows)}))


if __name__ == "__main__":
    main(sys.argv)
