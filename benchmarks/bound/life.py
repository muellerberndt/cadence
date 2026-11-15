"""One life of the meaning chamber's founder with its glimpse read from the map (cadence#342).

    python life.py <N0 brain.npz> <out dir> [source]

From the founder's checkpoint after the night's first step (imitation), the steps N1 (names),
N2 (colour) and N3 (where) as ``night.life`` lives them, then the pretraining stage's mix
(``pretrain_stage.MIX``; lesson weights .25 for 300 blocks, .10 for 50, .05 for 50), as the
equal-senses brain lived them: the same draws, the sparse mixed code taking each sense field at
unit length, word records off. ``source`` "map" (the default) reads the glimpse from the map
(``mapped.observe``); "eyes" is the meaning chamber's own glimpse.

A checkpoint every 50 blocks of the mix (25 in the last 100): the brain and its spec saved,
the demo room's every question through the page's host (its glimpse from the same source) and
every focus form read frozen (kind-colour pairs kept out of training for the colour forms).
A run that stops resumes from its last night checkpoint; with a checkpoint of the night's last
step in ``<out dir>/ckpt`` the life begins at the mix. The thread count follows the machine's
load: 6, and 4 while the one-minute load is above 40.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import lane
import mapped
import numpy as np

from cadence import Brain

nt, ds, ep, ps = lane.night, lane.demo_stage, lane.export_page, lane.pretrain_stage
FORMS = ["name", "colour_this", "material_this", "is_colour", "colour_of", "is_round",
         "can_eat_bal", "shape_of", "shape_not", "direction", "imitate"]
TEST = ("colour_of", "colour_this", "is_colour")
PHASES = [[0.25, 300, 50], [0.10, 50, 25], [0.05, 50, 25]]


def threads() -> None:
    """The thread count for the machine's load: 6, and 4 while the one-minute load is above 40.
    A second life running beside this one (the process named by the file in the environment
    variable ``BOUND_SIBLING``) takes half of them."""
    try:
        import numba
    except ImportError:
        return
    allowed = 4 if os.getloadavg()[0] > 40 else 6
    sibling = os.environ.get("BOUND_SIBLING")
    if sibling and Path(sibling).exists():
        try:
            os.kill(int(Path(sibling).read_text().split()[0]), 0)  # is it alive
            allowed //= 2
        except (OSError, ValueError, IndexError):
            pass
    want = max(1, min(allowed, int(numba.config.NUMBA_NUM_THREADS)))
    if numba.get_num_threads() != want:
        numba.set_num_threads(want)


def reads(p, held: set, n: int = 48) -> dict[str, float]:
    rng = np.random.default_rng(99)
    out = {}
    for f in FORMS:
        for test in ((False, True) if f in TEST else (False,)):
            ok: list[bool] = []
            while len(ok) < n:
                ok += nt.live(p, nt.blocks_of(p, rng, f, "turn", held, test=test), rng,
                              learn=False)["right"]
            out[f + ("/held" if test else "")] = round(float(np.mean(ok[:n])), 3)
    return out


def room_check(folder: Path, source: str) -> list[tuple[str, bool, str]]:
    m = mapped.MappedHost(folder / "brain.npz", folder / "spec.json", seed=lane.SEED,
                          source=source)
    m.records = None
    room = []
    for words, answer, focus, form in ps.room_questions():
        said = ep.ask(m, ds.room_things(focus), list(words), False)
        room.append((form, said == answer, f"{' '.join(words)} ({focus}): "
                                           f"{' '.join(said) or '-'}"))
    return room


def checkpoint(p, out: Path, b: int, t0: float, held: set, source: str) -> None:
    d = out / f"block_{b:04d}"
    d.mkdir(exist_ok=True)
    p.brain.save(d / "brain.npz")
    nt.export_spec(p, d / "spec.json")
    spec = json.loads((d / "spec.json").read_text())
    spec["mixed"]["fields"] = True
    spec["glimpse_from"] = source
    (d / "spec.json").write_text(json.dumps(spec))
    room = room_check(d, source)
    by: dict[str, list[bool]] = {}
    for form, ok, _ in room:
        by.setdefault(form, []).append(ok)
    row = {"block": b, "seconds": round(time.time() - t0),
           "lesson": float(p.genome["gated_lesson"]),
           "room": round(float(np.mean([r[1] for r in room])), 3),
           "room_by_form": {f: f"{sum(v)}/{len(v)}" for f, v in by.items()},
           "wrong": [r[2] for r in room if not r[1]], "reads": reads(p, held)}
    print(json.dumps(row), flush=True)


def main(argv: list[str]) -> None:
    n0, out = Path(argv[1]), Path(argv[2])
    source = argv[3] if len(argv) > 3 else "map"
    seed = lane.SEED
    out.mkdir(parents=True, exist_ok=True)
    mapped.install(source)
    plain = nt.live

    def live(*args, **kwargs):  # the thread count follows the load, block by block
        threads()
        return plain(*args, **kwargs)

    nt.live = live
    held = nt.founder_held(seed)
    ckpt = out / "ckpt"
    ckpt.mkdir(exist_ok=True)
    meta = ckpt / f"ckpt_{seed}.json"
    if not meta.exists():  # the night resumes at N1's first block from the N0 brain
        shutil.copy(n0, ckpt / f"ckpt_{seed}.npz")
        meta.write_text(json.dumps({
            "step": 2, "block": 0, "seen": ["imitate"], "paid": [], "work": {},
            "out": {"seed": seed, "steps": {"N0": {"imitate": 1.0}}, "streams": 32}}))
    state = out / "mix.json"  # the mix's own resume point
    done = json.loads(state.read_text())["block"] if state.exists() else None
    if done is None and json.loads(meta.read_text()).get("step", 0) < 5:  # N1 to N3
        res = nt.life(seed, dict(lane.GENOME), streams=32, reads=48,
                      steps=["N0", "N1", "N2", "N3"], ckpt=str(ckpt), save=str(out / "night"),
                      keep=True, log=lambda s: print(json.dumps({"night": s}), flush=True))
        p = res.pop("parrot")
        (out / "night.json").write_text(json.dumps(res, default=float, indent=1))
    else:
        p = nt.Parrot(seed, dict(lane.GENOME), streams=32)
        p.brain = Brain.load(out / "mix.npz" if done else ckpt / f"ckpt_{seed}.npz")
    t0 = time.time()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 140]))
    keys, weights = list(ps.MIX), np.array(list(ps.MIX.values()), dtype=float)
    if done:  # the mix's draws so far, without living them, then the saved generator state
        rng.bit_generator.state = json.loads(state.read_text())["rng"]
    else:
        checkpoint(p, out, 0, t0, held, source)
    b = 0
    for weight, count, check in PHASES:
        p.genome["gated_lesson"] = p.genome["correction_lesson"] = weight
        for _ in range(count):
            b += 1
            if done and b <= done:
                continue
            nt.live(p, ds.mixed_block(p, rng, held, keys, weights), rng)
            if b % 10 == 0:
                print(json.dumps({"block": b, "seconds": round(time.time() - t0)}), flush=True)
                p.brain.save(out / "mix.part.npz")
                (out / "mix.part.npz").replace(out / "mix.npz")
                state.write_text(json.dumps({"block": b, "rng": rng.bit_generator.state}))
            if b % check == 0:
                checkpoint(p, out, b, t0, held, source)
    print(json.dumps({"finished": b, "seconds": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main(sys.argv)
