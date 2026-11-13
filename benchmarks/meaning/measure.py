"""Run 0 of the meaning plan (cadence#334): what a moment costs, and how fast the first rungs learn.

    python measure.py rate <size> <streams> <backend> <moments> <out.json>
    python measure.py life <founder> '<json settings>' <out.json>
    python measure.py turns <founder> '<json settings>' <out.json>

``rate`` lives ``moments`` moments of a brain of ``size`` (``words``, ``grammar``, ``v500``,
``v2000``) on ``streams`` rows: every moment an act, a reward step, and a self-hearing lesson on a
third of the rows every third moment, as the curriculum does; it writes the seconds, the sweeps
and the multiply-accumulates a second. ``life`` is a curriculum life (``curriculum.life``).
``turns`` lives a rung of turns (``T0`` by default) with its yoked twin and reads both every
``every`` blocks.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SIZES: dict[str, tuple[dict[str, Any], int, int, int]] = {
    # genome over the hand-set, eye slots, ear width, voice words
    "words": ({}, 5, 64, 40),
    "grammar": ({"module_width": 256, "association": 256, "recall_heads": 4}, 5, 128, 140),
    "v500": ({"depth": 3, "module_width": 384, "association": 384, "recall_heads": 4,
              "recall_capacity": 64}, 8, 192, 520),
    "v2000": ({"depth": 3, "module_width": 512, "association": 512, "recall_heads": 8,
               "recall_capacity": 64}, 8, 256, 2010),
}


def rate(size: str, streams: int, backend: str, moments: int) -> dict[str, Any]:
    from cadence.meaning import Eyes, compose

    genome, slots, ear, words = SIZES[size]
    inputs = Eyes(slots=slots).width + ear
    began = time.time()
    brain = compose(genome, inputs, words, ear=ear, seed=0, backend=backend)
    built = time.time() - began
    rng = np.random.default_rng(0)
    synapses = int(brain.connectome.synapses)
    sweeps = {"act": 0.0, "reward_rows": 0.0, "lesson_rows": 0.0}
    times: list[float] = []
    x = rng.random((streams, inputs)) * (rng.random((streams, inputs)) < 0.2)
    for t in range(moments):
        x = rng.random((streams, inputs)) * (rng.random((streams, inputs)) < 0.2)
        start = time.time()
        if brain.pending_feedback:
            report = brain.learn((rng.random(streams) < 0.3).astype(float),
                                 np.zeros(streams, bool), x)
            sweeps["reward_rows"] += streams * (float(report.get("free_steps", 0.0)) + 24.0)
        if t % 3 == 2:
            weights = (rng.random(streams) < 0.33).astype(float)
            if weights.any():
                report = brain.lesson(x, rng.integers(words + 1, size=streams), weights=weights)
                sweeps["lesson_rows"] += float(report.get("total_row_sweeps", 0.0))
        brain.act(x)
        sweeps["act"] += streams * float((brain.last_settlement or {}).get("steps", 0))
        times.append(time.time() - start)
    warm = times[min(3, len(times) - 1):]  # the first moments compile the kernels
    total = sum(sweeps.values())
    seconds = float(np.sum(warm))
    share = len(warm) / len(times)
    return {"size": size, "streams": streams, "backend": backend, "moments": moments,
            "synapses": synapses, "parameters": int(brain.parameters()),
            "placement": dict(brain.placement) if hasattr(brain, "placement") else None,
            "compose_seconds": built, "seconds_per_moment": seconds / len(warm),
            "seconds_per_stream_moment": seconds / len(warm) / streams,
            "sweeps_per_stream_moment": total / moments / streams,
            "macs_per_second": total * share * synapses / seconds if seconds else None}


def turns_life(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    import turns as tn

    rung = str(settings.get("rung", "T0"))
    blocks, every = int(settings.get("blocks", 600)), int(settings.get("every", 50))
    streams = int(settings.get("streams", 8))
    genome = settings.get("genome")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 9]))
    master = tn.Parrot(seed, genome, streams=streams)
    yoked = tn.Parrot(seed, genome, streams=streams)
    curve: list[dict[str, Any]] = []
    began = time.time()
    for b in range(blocks):
        trials = tn.block(master, rng, rung)
        lived = tn.live(master, trials, rng)
        tn.live(yoked, trials, rng, given=lived["earned"])
        if (b + 1) % every == 0 or b == 0:
            row: dict[str, Any] = {"block": b + 1, "seconds": time.time() - began}
            for name, p in (("brain", master), ("yoked", yoked)):
                test = tn.block(p, rng, rung)
                got = tn.read(p, test, rng)
                spoken = got["in_gap"] + got["over"] + got["late"]
                row[name] = {"replied": float(np.mean(got["replied"])),
                             "in_gap_share": got["in_gap"] / spoken if spoken else 0.0,
                             "over": got["over"], "late": got["late"]}
                if rung != "T0":
                    asked = [tr.question for tr in test]
                    row[name]["after_questions"] = float(np.mean(
                        [r for r, q in zip(got["replied"], asked, strict=True) if q] or [0]))
                    row[name]["after_statements"] = float(np.mean(
                        [r for r, q in zip(got["replied"], asked, strict=True) if not q] or [0]))
            curve.append(row)
            print(json.dumps(row), flush=True)
    return {"seed": seed, "rung": rung, "streams": streams, "curve": curve,
            "work": master.work, "synapses": master.synapses}


def main(argv: list[str]) -> None:
    mode = argv[1]
    if mode == "rate":
        size, streams, backend, moments, out = argv[2], int(argv[3]), argv[4], int(argv[5]), argv[6]
        result = rate(size, streams, backend, moments)
    elif mode == "life":
        import curriculum

        seed, given, out = int(argv[2]), json.loads(argv[3]), argv[4]
        result = curriculum.life(seed, given.get("genome"), blocks=given.get("blocks"),
                                 streams=int(given.get("streams", 8)),
                                 backend=str(given.get("backend", "cpu")),
                                 log=lambda s: print(s, flush=True))
        result["settings"] = given
    elif mode == "turns":
        seed, given, out = int(argv[2]), json.loads(argv[3]), argv[4]
        result = turns_life(seed, given)
        result["settings"] = given
    else:
        raise SystemExit(__doc__)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, default=float, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "curve"}, default=float)[:2000])


if __name__ == "__main__":
    main(sys.argv)
