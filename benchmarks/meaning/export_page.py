"""The files a page loads, from a trained brain, checked through the page's own host (cadence#334).

    python export_page.py <brain.npz> <founder> <out_dir> [genome json]

Writes ``brain.npz`` and ``spec.json`` (and ``brain.records.npz``, the word records saved beside
the brain, where it keeps them) and ``probes.json``: the association's linear read of each facet
of the thing in focus (kind, colour, round, material, edible, direction, distance), fitted on the
settled state with a thing in focus and nothing heard (``introspect.state_completeness``): for
each facet its classes, the standardization (mean, sd per association neuron) and the weights
(neurons x classes), with the held-out accuracy and base rate; the page shows "In its mind" as
``argmax(((a - mean) / sd) @ weights)`` on the association's activity ``a``. Then loads them
with ``cadence.meaning.host`` and asks
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
    m.turn_over()
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


REHEARSED = ("name", "colour_of", "where", "material", "can_eat")


def rehearsal_items(seed: int, per: int = 40) -> list[dict[str, Any]]:
    """The night's exchanges a page revisits after a paid lesson (``MeaningParrot.
    rehearse_one``): rooms of the founder's training world, the trainer's words, the right
    answer, ``per`` of each form a turn answers in the gap (no scan)."""
    import numpy as np

    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 55]))
    p = nt.Parrot(seed, streams=1)
    held = nt.founder_held(seed)
    out = []
    for form in REHEARSED:
        for _ in range(per):
            ex = nt.exchange(p, rng, form, "turn", held)
            q = ex.question
            assert q is not None
            out.append({"form": form, "words": list(q.words), "answer": list(q.answer),
                        "things": [{"kind": t.kind, "colour": t.colour, "material": t.material,
                                    "direction": nt.bearing(ex.room.heading, t.place),
                                    "distance": float(math.hypot(*t.place)),
                                    "focus": bool(t.focus)} for t in ex.room.things]})
    return out


def main(argv: list[str]) -> None:
    brain, seed, out = argv[1], int(argv[2]), Path(argv[3])
    genome = json.loads(argv[4]) if len(argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    p = nt.Parrot(seed, genome, streams=1)
    nt.export_spec(p, out / "spec.json")
    shutil.copyfile(brain, out / "brain.npz")
    kept = Path(brain).with_suffix(".records.npz")
    if p.records is not None and kept.exists():
        shutil.copyfile(kept, out / "brain.records.npz")
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
    (out / "rehearsal.json").write_text(json.dumps(rehearsal_items(seed)))
    import exam as ex
    import numpy as np

    q = ex.load(brain, seed, genome, streams=16)  # the trained brain, read on a frozen copy
    sc = ii.state_completeness(q, np.random.default_rng(seed), nt.founder_held(seed), count=512)
    probes = {name: {"classes": list(fit[0]), "mean": fit[1].tolist(), "sd": fit[2].tolist(),
                     "weights": fit[3].tolist(),
                     "accuracy": sc["read"]["association"][name], "base_rate": sc["base"][name]}
              for name, fit in sc["fits"].items()}
    population = [int(i) for i in q.brain.connectome.populations["association"]]
    (out / "probes.json").write_text(json.dumps({"population": "association",
                                                 "neurons": population, "facets": probes}))
    print(json.dumps(check))


if __name__ == "__main__":
    main(sys.argv)
