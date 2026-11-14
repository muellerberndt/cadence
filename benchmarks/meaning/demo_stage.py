"""The owner's demo stage (cadence#334): a short stage on a trained brain that adds "what shape
has the X?" and "where is the X?" (the direction alone), with revisits of names, colour and
imitation, read every ``every`` blocks on held-out rooms and in the page's own room.

    python demo_stage.py <brain.npz> <founder> '<genome>' <out dir> [blocks] [every]

The brain sleeps (efficacies x .5) before the stage. Each block is one of the stage's mix.
Every ``every`` blocks:
- held-out reads (32 a form): shape_of, shape_not, direction, name, colour_of, imitate, and
  shape_of_other (how often "round" is said of a thing that is not round);
- the demo room through the page's host (``MeaningParrot``): the owner's room with a green ball
  to the parrot's right. "what colour has the apple?" -> red, "what shape has the apple?" ->
  round, "where is the ball?" -> right, "what is this?" (the apple) -> apple, and "what shape
  has the block?" -> anything but "round".
A thing that is not round is asked its shape too ("shape_not"), its answer "no": the voice has
no word for its shape. The brain and its records are saved at each read
(``<out>/brain_<founder>_demo_<block>``). The first brain that answers all five in the room,
with "round" said of at most 15% of things not round (held-out), is exported for the page to
``<page>/code_<founder>_demo`` (export_page.py, then the four in ``demo_check.json`` and the
page test at 0.05 first).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exam as ex  # noqa: E402
import export_page as ep  # noqa: E402
import night as nt  # noqa: E402

from cadence.meaning.host import MeaningParrot  # noqa: E402

MIX = {"shape_of:model": .10, "shape_of:dagger": .14, "shape_not:model": .07,
       "shape_not:dagger": .11, "direction:model": .10, "direction:dagger": .14,
       "name:dagger": .10, "colour_of:model": .04, "colour_of:dagger": .12, "imitate:turn": .08}
READS = ["shape_of", "shape_not", "shape_of_other", "direction", "name", "colour_of", "imitate"]
OTHER_MAX = 0.15  # the demo brain says "round" of at most this share of things not round
DEMO = [  # the owner's four and the contrast, the answer, the thing in focus
    (["what", "colour", "has", "the", "apple"], ["red"], "apple"),
    (["what", "shape", "has", "the", "apple"], ["round"], "apple"),
    (["where", "is", "the", "ball"], ["right"], "ball"),
    (["what", "is", "this"], ["apple"], "apple"),
    (["what", "shape", "has", "the", "block"], None, "block"),  # right: anything but "round"
]


def demo_room() -> nt.Room:
    """The owner's room (a red apple left and far, a blue block, a white cup behind, a yellow
    key in front) with a green ball to the parrot's right."""
    rng = np.random.default_rng(0)

    def at(deg: float, r: float) -> np.ndarray:
        return np.array([r * math.cos(math.radians(deg)), r * math.sin(math.radians(deg))])

    things = [nt.thing(rng, "apple", at(70, 1.5), colour="red", material="fruit"),
              nt.thing(rng, "block", at(120, 1.0), colour="blue", material="wood"),
              nt.thing(rng, "cup", at(180, 0.8), colour="white", material="glass"),
              nt.thing(rng, "key", at(-20, 1.0), colour="yellow", material="metal"),
              nt.thing(rng, "ball", at(-80, 1.0), colour="green", material="plastic")]
    return nt.Room(things, heading=0.0)


def room_things(focus: str | None) -> list[dict[str, Any]]:
    room = demo_room()
    return [{"kind": t.kind, "colour": t.colour, "material": t.material,
             "direction": nt.bearing(room.heading, t.place),
             "distance": float(math.hypot(*t.place)), "focus": t.kind == focus}
            for t in room.things]


def demo_check(folder: Path, seed: int) -> list[dict[str, Any]]:
    m = MeaningParrot(folder / "brain.npz", folder / "spec.json", seed=seed)
    out = []
    for words, answer, focus in DEMO:
        said = ep.ask(m, room_things(focus), list(words), False)
        ok = "round" not in said if answer is None else said == answer
        out.append({"question": " ".join(words), "answer": answer or "not round", "said": said,
                    "right": ok})
    return out


def main(argv: list[str]) -> None:
    path, seed, genome, out = argv[1], int(argv[2]), json.loads(argv[3]), Path(argv[4])
    blocks = int(argv[5]) if len(argv) > 5 else 600
    every = int(argv[6]) if len(argv) > 6 else 25
    page = Path(genome.pop("page_dir", "/data/meaning334/night/page"))
    out.mkdir(parents=True, exist_ok=True)
    p = ex.load(path, seed, genome, streams=32)
    held = nt.founder_held(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 120]))
    nt.sleep(p, 0.5)  # a night's sleep before the stage
    keys, weights = list(MIX), np.array(list(MIX.values()))
    began, exported = time.time(), False

    def reading(at: int) -> None:
        nonlocal exported
        forms = nt.read(p, rng, READS, held, 32)
        tmp = out / f"brain_{seed}_demo_{at}"
        tmp.mkdir(exist_ok=True)
        p.brain.save(tmp / "brain.npz")
        nt.save_records(p, tmp / "brain.records.npz")
        nt.export_spec(p, tmp / "spec.json")
        check = demo_check(tmp, seed)
        row = {"block": at, "seconds": round(time.time() - began), "forms": forms,
               "room": [" ".join(c["said"]) for c in check],
               "room_right": sum(c["right"] for c in check)}
        print(json.dumps(row), flush=True)
        if row["room_right"] == len(DEMO) and forms["shape_of_other"] <= OTHER_MAX \
                and not exported:
            exported = True
            dest = page / f"code_{seed}_demo"
            subprocess.run([sys.executable, str(HERE / "export_page.py"), str(tmp / "brain.npz"),
                            str(seed), str(dest), json.dumps(genome)], check=True)
            (dest / "demo_check.json").write_text(json.dumps(
                {"block": at, "forms": forms, "room": check}, indent=1))
            subprocess.run([sys.executable, str(HERE / "page_test.py"), str(dest), "0.05",
                            "first"], check=True, stdout=subprocess.DEVNULL)
            print(json.dumps({"exported": str(dest), "block": at}), flush=True)

    reading(0)
    for b in range(1, blocks + 1):
        key = keys[int(rng.choice(len(keys), p=weights / weights.sum()))]
        form, mode = key.split(":")
        nt.live(p, nt.blocks_of(p, rng, form, mode, held), rng)
        if b % every == 0:
            reading(b)


if __name__ == "__main__":
    main(sys.argv)
