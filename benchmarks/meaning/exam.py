"""The night's exam and introspection on a saved brain (cadence#334).

    python exam.py <brain.npz> <founder> <out.json> ['<json genome>']

Reads every question form on fresh rooms (held-out kind-colour pairs for the colour forms), the
owner's five exchanges, the concept codes (probes, neighbours, a word alone), silencing (a
property's most tuned units, its probe direction, and as many random units), the new object (the
lemon's code and "can you eat this?"), changed scenes, and the new-word gate by pistachios alone.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import introspect as ii  # noqa: E402
import night as nt  # noqa: E402

from cadence import Brain  # noqa: E402

FORMS = ["imitate", "name", "colour_of", "by_colour", "where", "left_of", "next_to", "opposite",
         "material", "can_eat", "can_eat_this", "round", "eat"]
# the forms that need each property, and the forms read beside them
NEEDS = {"round": ["round"], "red": ["colour_of", "by_colour"], "food": ["eat", "can_eat"],
         "metal": ["material"], "near": ["where"], "left": ["where", "left_of"]}


def load(path: str, seed: int, genome: dict[str, Any] | None = None,
         streams: int = 16) -> nt.Parrot:
    p = nt.Parrot(seed, genome, streams=streams)
    p.brain = Brain.load(path)
    p.synapses = int(p.brain.connectome.synapses)
    nt.load_records(p, Path(path).with_suffix(".records.npz"))  # where the brain keeps records
    return p


def changed_scene(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
                  n: int = 64) -> dict[str, float]:
    """The same question after the scene changed: the focused thing's colour, or a thing moved
    to the other side; read on the changed scene against the changed scene's answer."""
    out: dict[str, float] = {}
    for form, change in (("colour_of", "recolour"), ("where", "move"), ("left_of", "move")):
        ok: list[bool] = []
        while len(ok) < n:
            exs = nt.block(p, rng, form, "turn", held)
            for ex in exs:
                r, q = ex.room, ex.question
                assert q is not None
                x = next(t for t in r.things if t.kind == q.words[-1])
                if change == "recolour":
                    others = [c for c in nt.COLOURS if c != x.colour]
                    new = nt.thing(rng, x.kind, x.place, colour=str(rng.choice(others)),
                                   material=x.material)
                    r.things[r.things.index(x)] = new
                else:
                    x.place = -x.place  # across the parrot
                if form == "colour_of":
                    q.answer = [next(t for t in r.things if t.kind == q.words[-1]).colour]
                elif form == "where":
                    t2 = next(t for t in r.things if t.kind == q.words[-1])
                    q.answer = [nt.direction_of(r.heading, t2.place)] + (
                        ["far"] if float(np.linalg.norm(t2.place)) > nt.FAR else [])
                else:
                    found = [t for t in r.things if t.kind != q.words[-1] and nt.left_of(
                        r, t, next(u for u in r.things if u.kind == q.words[-1]))]
                    q.answer = [found[0].kind] if len(found) == 1 else ["<none>"]
            got = nt.live(p, exs, rng, learn=False)
            ok += [g for g, ex in zip(got["right"], exs, strict=True)
                   if ex.question is not None and ex.question.answer != ["<none>"]]
        out[f"{form}_{change}"] = float(np.mean(ok[:n]))
    return out


def silencing(p: nt.Parrot, rng: np.random.Generator, held: set[tuple[str, str]],
              pr: dict[str, Any], base: dict[str, float], k: int = 12,
              n: int = 32) -> dict[str, Any]:
    """Each property: its ``k`` most tuned association units cut, its probe direction projected
    out, and ``k`` random association units cut; every form read again."""
    assoc = np.asarray(list(p.brain.connectome.populations["association"]))
    width = len(assoc)
    out: dict[str, Any] = {}
    for prop, probe in pr["probes"].items():
        weights = probe[0][:width]  # the association part of the code comes first
        units = assoc[np.argsort(-np.abs(weights))[:k]]
        random_units = rng.choice(assoc, size=k, replace=False)
        row: dict[str, Any] = {"units": units.tolist()}
        for name, q in (("units", ii.lesion_units(p, units)),
                        ("direction", ii.lesion_direction(p, weights)),
                        ("random", ii.lesion_units(p, random_units))):
            got = nt.read(q, rng, FORMS, held, n)
            needed = [f for f in NEEDS.get(prop, []) if f in base]
            row[name] = {"needed": float(np.mean([base[f] - got[f] for f in needed]))
                         if needed else float("nan"),
                         "others": float(np.mean([base[f] - got[f] for f in FORMS
                                                  if f not in needed and f in base])),
                         "forms": got}
        out[prop] = row
    return out


def main(argv: list[str]) -> None:
    path, seed, target = argv[1], int(argv[2]), argv[3]
    genome = json.loads(argv[4]) if len(argv) > 4 else None
    p = load(path, seed, genome)
    held = nt.founder_held(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 77]))
    out: dict[str, Any] = {"brain": path, "seed": seed, "parameters": int(p.brain.parameters()),
                           "synapses": p.synapses}
    out["forms"] = nt.read(p, rng, FORMS, held, 96)
    print("forms", json.dumps(out["forms"]), flush=True)
    out["owner"] = ii.exam_owner(p, rng)
    print("owner", json.dumps(out["owner"]), flush=True)
    room = ii.owner_room()
    texts = ["What colour has the apple?", "What is left of the apple?", "Where is the apple?",
             "Where is the cup?", "What objects are round?"]
    focus = [f for _, _, f in ii.OWNER]
    out["demo"] = [{"things": [{"kind": t.kind, "colour": t.colour, "material": t.material,
                                "direction": round(math.degrees(nt.bearing(room.heading,
                                                                           t.place)), 1),
                                "distance": round(float(np.linalg.norm(t.place)), 2),
                                "focus": t.kind == focus[i]} for t in room.things],
                    "question_text": texts[i], "question_audio": f"q{i:02d}.wav",
                    "said": q["said"], "right": q["right"] >= 0.5}
                   for i, q in enumerate(out["owner"])]
    nt.export_spec(p, Path(target).with_suffix(".spec.json"))
    pr = ii.probes(p, rng, held, count=768)
    out["probes"] = {"accuracy": pr["accuracy"], "base_rate": pr.get("base_rate")}
    print("probes", json.dumps(out["probes"]), flush=True)
    out["probes_by_region"] = ii.probes_by_region(p, rng, held, count=512)
    print("probes_by_region", json.dumps(out["probes_by_region"]), flush=True)
    nb = ii.neighbours(p, rng, held, per=48)
    out["neighbours"] = {"nearest": nb["nearest"], "kinds": nb["kinds"], "cosine": nb["cosine"]}
    print("neighbours", json.dumps(nb["nearest"]), flush=True)
    out["word_alone"] = ii.word_alone(p, rng, pr, words=nt.KIND_NAMES)
    print("word_alone", json.dumps(out["word_alone"]), flush=True)
    lemons = [(ii.lemon_room(rng, held), None) for _ in range(64)]
    pairs = [(r, r.things[0]) for r, _ in lemons]
    x, _ = ii.codes_of(p, pairs, rng)
    out["lemon_probes"] = {prop: float(np.mean(ii.probe_read(probe, x) > 0.5))
                           for prop, probe in pr["probes"].items()}
    ok = []
    for r, _ in lemons[:48]:
        words = ["can", "you", "eat", "this"]
        ex = nt.Exchange(copy.deepcopy(r), list(words), 3, nt.Question(list(words), ["yes"],
                                                                     "can_eat_this"))
        ok.append(ex)
    got: list[bool] = []
    for i in range(0, len(ok), p.streams):
        chunk = ok[i:i + p.streams]
        if len(chunk) == p.streams:
            got += nt.live(p, chunk, rng, learn=False)["right"]
    out["lemon_can_eat_this"] = float(np.mean(got)) if got else float("nan")
    print("lemon", json.dumps(out["lemon_probes"]), out["lemon_can_eat_this"], flush=True)
    out["changed_scene"] = changed_scene(p, rng, held)
    print("changed", json.dumps(out["changed_scene"]), flush=True)
    out["silencing"] = silencing(p, rng, held, pr, out["forms"])
    print("silencing", json.dumps({k: {m: (v[m]["needed"], v[m]["others"]) for m in
                                       ("units", "direction", "random")}
                                   for k, v in out["silencing"].items()}), flush=True)
    nw = ii.new_word(p, rng, held)
    nw.pop("parrot")
    out["new_word"] = nw
    print("new_word", json.dumps(nw), flush=True)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    Path(target).write_text(json.dumps(out, default=float, indent=1))


if __name__ == "__main__":
    main(sys.argv)
