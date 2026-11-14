"""The owner's pretraining stage (cadence#334): every focus question at once, so the parrot answers
all of them right before its owner trains it further.

    python pretrain_stage.py <brain.npz> <founder> '<genome>' <out dir>

Every block draws its 32 exchanges from the mix, one a stream (the forms interleaved within
the block), each question by a modelling trainer (the answer said first) or with a correction:
- "what is this?" -> the name; "what colour has the X?" and "what colour is this?" -> the colour;
- "is this <colour>?", "is this round?", "can you eat this?" -> yes or no, as often one as the
  other;
- "what shape has the X?" -> "round", or "no" for a thing that is not round (the voice's one
  shape word is "round");
- "where is the X?" -> left, right, front or behind; "what material is this?" -> the material;
- imitation kept.
The lesson weight follows ``phases`` ([[weight, blocks, room check every], ...]; default .25 for
300 blocks, then .10 and .05 for 50 each). At each room check, every question on every thing of
the demo room (the owner's room with a green ball to the right) through the page's host; where
the room reaches 90% (and at every 25th block), held-out reads of every form (32 each, unseen
kind-colour pairs for the colour forms). The first brain with the room at 90% or more and every
held-out form at .85 or more is exported to ``<page>/code_<founder>_pretrain[_<tag>]``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import demo_stage as ds  # noqa: E402
import exam as ex  # noqa: E402
import export_page as ep  # noqa: E402
import night as nt  # noqa: E402

from cadence.meaning.host import MeaningParrot  # noqa: E402

SHARES = {"name": .10, "colour_of": .08, "colour_this": .08, "is_colour": .10, "shape_of": .06,
          "shape_not": .05, "is_round": .08, "can_eat_bal": .10, "direction": .17,
          "material_this": .12}
MIX = {**{f"{f}:{m}": w * share for f, w in SHARES.items()
          for m, share in (("model", .4), ("dagger", .6))}, "imitate:turn": .06}
READS = [*SHARES, "imitate"]
HELD_OUT = ("colour_of", "colour_this", "is_colour")  # read on unseen kind-colour pairs
ROOM_MIN, FORM_MIN = 0.90, 0.85


def room_questions() -> list[tuple[list[str], list[str], str, str]]:
    """Every question on every thing of the demo room: (words, answer, thing in focus, form)."""
    out = []
    colours = list(nt.COLOURS)
    for t in ds.demo_room().things:
        other = colours[(colours.index(t.colour) + 1) % len(colours)]
        round_ = t.kind in nt.ROUND
        out += [(["what", "is", "this"], [t.kind], t.kind, "name"),
                (["what", "colour", "has", "the", t.kind], [t.colour], t.kind, "colour_of"),
                (["what", "colour", "is", "this"], [t.colour], t.kind, "colour_this"),
                (["is", "this", t.colour], ["yes"], t.kind, "is_colour"),
                (["is", "this", other], ["no"], t.kind, "is_colour"),
                (["what", "shape", "has", "the", t.kind], ["round" if round_ else "no"], t.kind,
                 "shape"),
                (["is", "this", "round"], ["yes" if round_ else "no"], t.kind, "is_round"),
                (["can", "you", "eat", "this"], ["yes" if t.material in nt.EDIBLE else "no"],
                 t.kind, "can_eat"),
                (["where", "is", "the", t.kind], [nt.direction_of(0.0, t.place)], t.kind,
                 "direction"),
                (["what", "material", "is", "this"], [t.material], t.kind, "material_this")]
    return out


def room_check(folder: Path, seed: int) -> list[dict[str, Any]]:
    m = MeaningParrot(folder / "brain.npz", folder / "spec.json", seed=seed)
    out = []
    for words, answer, focus, form in room_questions():
        said = ep.ask(m, ds.room_things(focus), list(words), False)
        out.append({"question": " ".join(words), "thing": focus, "form": form,
                    "answer": answer, "said": said, "right": said == answer})
    return out


def read(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
         n: int = 32) -> dict[str, float]:
    out: dict[str, float] = {}
    for form in READS:
        ok: list[bool] = []
        while len(ok) < n:
            exs = nt.blocks_of(p, rng, form, "turn", held, test=form in HELD_OUT)
            ok += nt.live(p, exs, rng, learn=False)["right"]
        out[form] = float(np.mean(ok[:n]))
    return out


def main(argv: list[str]) -> None:
    path, seed, genome, out = argv[1], int(argv[2]), json.loads(argv[3]), Path(argv[4])
    page = Path(genome.pop("page_dir", "/data/meaning334/night/page"))
    slept = bool(genome.pop("slept", 0))
    tag = str(genome.pop("tag", ""))
    phases = [(float(w), int(b), int(c)) for w, b, c in genome.pop(
        "phases", [[0.25, 300, 25], [0.10, 50, 5], [0.05, 50, 5]])]
    out.mkdir(parents=True, exist_ok=True)
    p = ex.load(path, seed, genome, streams=32)
    held = nt.founder_held(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 140]))
    if not slept:
        nt.sleep(p, 0.5)
    keys, weights = list(MIX), np.array(list(MIX.values()), dtype=float)
    began, exported = time.time(), False

    def reading(at: int, full: bool) -> None:
        nonlocal exported
        tmp = out / f"brain_{seed}_pretrain_{at}"
        tmp.mkdir(exist_ok=True)
        p.brain.save(tmp / "brain.npz")
        nt.save_records(p, tmp / "brain.records.npz")
        nt.export_spec(p, tmp / "spec.json")
        room = room_check(tmp, seed)
        share = float(np.mean([c["right"] for c in room]))
        forms = read(p, rng, held) if full or share >= ROOM_MIN else None
        by_form: dict[str, list[bool]] = {}
        for c in room:
            by_form.setdefault(c["form"], []).append(c["right"])
        row = {"block": at, "seconds": round(time.time() - began),
               "lesson": float(p.genome["gated_lesson"]), "room": round(share, 3),
               "room_by_form": {f: f"{sum(v)}/{len(v)}" for f, v in by_form.items()},
               "wrong": [f"{c['question']} ({c['thing']}): {' '.join(c['said']) or '-'}"
                         for c in room if not c["right"]], "forms": forms}
        print(json.dumps(row), flush=True)
        if forms is not None and share >= ROOM_MIN and min(forms.values()) >= FORM_MIN \
                and not exported:
            exported = True
            dest = page / (f"code_{seed}_pretrain" + (f"_{tag}" if tag else ""))
            subprocess.run([sys.executable, str(HERE / "export_page.py"), str(tmp / "brain.npz"),
                            str(seed), str(dest), json.dumps(genome)], check=True)
            (dest / "room_check.json").write_text(json.dumps(
                {"block": at, "forms": forms, "room": room}, indent=1))
            subprocess.run([sys.executable, str(HERE / "page_test.py"), str(dest), "0.05",
                            "first"], check=True, stdout=subprocess.DEVNULL)
            print(json.dumps({"exported": str(dest), "block": at}), flush=True)
        elif forms is None:
            shutil.rmtree(tmp)

    reading(0, True)
    b = 0
    for weight, count, check in phases:
        p.genome["gated_lesson"] = p.genome["correction_lesson"] = weight
        for _ in range(count):
            b += 1
            nt.live(p, ds.mixed_block(p, rng, held, keys, weights), rng)
            if b % 25 == 0 or b % check == 0:
                reading(b, b % 25 == 0)


if __name__ == "__main__":
    main(sys.argv)
