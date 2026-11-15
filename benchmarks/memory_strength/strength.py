"""A record's strength (cadence#336): the dose curve, the one-shot write and the clean form.

A small chamber of sensory records in one ``cadence.LongTermMemory``. A pattern is a unit code
of ``WIDTH`` coordinates; a take of it is the pattern with noise (cosine about .86); its value
is its own label. A pattern is told n times as n takes in a row, then the store lives an idle
span of moments (``moment()``, no read), then each pattern is asked once from a fresh noisy take
(cosine about .70) and the read is right where it holds the pattern's label.

Two stores read the same draws:

* ``main``: the store as 0.99.0 makes it (read by cosine, a take at cosine ``REVISE`` or above
  told again). A record fades only toward replacement, so with room to spare every record stays
  readable, and a told-again write keeps the last take.
* ``strength``: the same with ``blend=True`` (a told-again take moves its record toward it by
  1 / (1 + depth)), ``keep=KEEP`` (a record whose depth reaches it is kept for good),
  ``forget=FORGET`` (a record below it that fades that far is forgotten) and ``familiar``.

Readings: right by tellings n after each span (the dose curve; G1, G3), right after the longest
span for one write at each surprise beside a pattern told eight times (G2), and how near the
record lies to the clean pattern beside the nearest of its takes (G4).

    PYTHONPATH=src python benchmarks/memory_strength/strength.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

WIDTH = 64
REVISE = 0.6  # two takes of one pattern meet at about .74
KEEP, FORGET, FAMILIAR = 8.0, 50.0, 0.02
DOSES = (1, 2, 4, 8, 16)
SPANS = (0, 60, 300, 1000, 3000, 10000)
SURPRISES = (0.0, 1.0, 3.0, 7.0)
TAKE, CUE = 0.6, 1.0  # the noise of a take and of a cue, over the pattern's unit length


def unit(x: np.ndarray) -> np.ndarray:
    return np.asarray(x / np.linalg.norm(x, axis=-1, keepdims=True))


def noisy(rng: np.random.Generator, clean: np.ndarray, spread: float) -> np.ndarray:
    return unit(clean + spread * rng.standard_normal(clean.shape) / np.sqrt(clean.shape[-1]))


def store(arm: str, capacity: int) -> Any:
    from cadence.longterm import LongTermMemory

    settings: dict[str, Any] = {"cosine": True, "revise": REVISE, "hold": float("inf")}
    if arm == "strength":
        settings.update(blend=True, keep=KEEP, forget=FORGET, familiar=FAMILIAR)
    return LongTermMemory(WIDTH, capacity, capacity, **settings)


def right(memory: Any, rng: np.random.Generator, cleans: np.ndarray, labels: np.ndarray) -> float:
    hits = []
    for clean, label in zip(cleans, labels, strict=True):
        read = memory.read(noisy(rng, clean, CUE), credit=False)
        hits.append(read.first() >= 0 and int(np.argmax(memory.value(read))) == int(label))
    return float(np.mean(hits))


def dose(seed: int, arm: str, patterns: int = 20,
         spans: tuple[int, ...] = SPANS) -> tuple[dict[int, list[float]], list[float]]:
    """Right by tellings n after each idle span, and by n the median moments until a record is
    forgotten (inf: never, within the spans)."""
    rng = np.random.default_rng([seed, 336, 1])
    cleans = unit(rng.standard_normal((len(DOSES) * patterns, WIDTH)))
    takes = [[noisy(rng, c, TAKE) for _ in range(DOSES[i // patterns])]
             for i, c in enumerate(cleans)]
    cues = np.random.default_rng([seed, 336, 2])
    memory = store(arm, len(cleans) + 8)
    for i, told in enumerate(takes):
        for t in told:
            memory.write(t, np.eye(memory.value_width)[i])
    out, lived = {}, 0
    gone_at = np.full(len(cleans), np.inf)
    for span in spans:
        for moment in range(lived, span):
            memory.moment()
            if len(memory.gone):
                fresh = memory.gone[: len(cleans)] & ~np.isfinite(gone_at)
                gone_at[fresh] = moment + 1
        lived = span
        out[span] = [right(memory, cues, cleans[d * patterns:(d + 1) * patterns],
                           np.arange(d * patterns, (d + 1) * patterns))
                     for d in range(len(DOSES))]
    return out, [float(np.median(gone_at[d * patterns:(d + 1) * patterns]))
                 for d in range(len(DOSES))]


def law(n: float) -> float:
    """The moments until a record told n times and never read is forgotten: ``FORGET`` over its
    fade a moment, ``1 / (1 + (n / depth) ** 4)`` at the founder depth 2 (inf where kept)."""
    return float("inf") if n >= KEEP else FORGET * (1.0 + (n / 2.0) ** 4)


def one_shot(seed: int, arm: str, patterns: int = 20, span: int = SPANS[-1]) -> dict[str, float]:
    """Right after ``span`` for one write at each surprise, beside patterns told eight times."""
    rng = np.random.default_rng([seed, 336, 3])
    kinds = [f"one write, surprise {s:g}" for s in SURPRISES] + ["told 8 times"]
    cleans = unit(rng.standard_normal((len(kinds) * patterns, WIDTH)))
    cues = np.random.default_rng([seed, 336, 4])
    memory = store(arm, len(cleans) + 8)
    for i, clean in enumerate(cleans):
        kind = i // patterns
        label = np.eye(memory.value_width)[i]
        if kind < len(SURPRISES):
            memory.write(noisy(rng, clean, TAKE), label, surprise=SURPRISES[kind])
        else:
            for _ in range(8):
                memory.write(noisy(rng, clean, TAKE), label)
    for _ in range(span):
        memory.moment()
    return {kind: right(memory, cues, cleans[k * patterns:(k + 1) * patterns],
                        np.arange(k * patterns, (k + 1) * patterns))
            for k, kind in enumerate(kinds)}


def clean_form(seed: int, arm: str, patterns: int = 20) -> dict[int, tuple[float, float]]:
    """By takes n: the record's cosine to the clean pattern, and the nearest take's."""
    rng = np.random.default_rng([seed, 336, 5])
    out = {}
    for n in DOSES:
        record, nearest = [], []
        for _ in range(patterns):
            clean = unit(rng.standard_normal(WIDTH))
            memory = store(arm, 4)
            takes = [noisy(rng, clean, TAKE) for _ in range(n)]
            for t in takes:
                memory.write(t, np.eye(memory.value_width)[0])
            key = np.asarray(memory.keys[0], dtype=float)
            record.append(float(key @ clean / np.linalg.norm(key)))
            nearest.append(max(float(t @ clean) for t in takes))
        out[n] = (float(np.mean(record)), float(np.mean(nearest)))
    return out


def span_of(values: list[float]) -> str:
    low, high = min(values), max(values)
    return f"{low:.2f}" if f"{low:.2f}" == f"{high:.2f}" else f"{low:.2f} to {high:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    for arm in ("main", "strength"):
        runs = [dose(s, arm) for s in args.seeds]
        curves = [run[0] for run in runs]
        print(f"\n### Right after an idle span, by tellings ({arm}; seeds {args.seeds})\n")
        print("| Idle moments | " + " | ".join(f"n = {n}" for n in DOSES) + " |\n"
              + "| --- " * (len(DOSES) + 1) + "|")
        for span in SPANS:
            print(f"| {span:,} | " + " | ".join(
                span_of([c[span][d] for c in curves]) for d in range(len(DOSES))) + " |")
        if arm == "strength":
            print("| moments until forgotten, median | " + " | ".join(
                "never" if not np.isfinite(runs[0][1][d]) else
                " to ".join(sorted({f"{run[1][d]:,.0f}" for run in runs}))
                for d in range(len(DOSES))) + " |")
            print("| the law: forget x (1 + (n / 2) ** 4) | " + " | ".join(
                "kept" if not np.isfinite(law(n)) else f"{law(n):,.0f}" for n in DOSES) + " |")
        shots = [one_shot(s, arm) for s in args.seeds]
        print(f"\n### One write after {SPANS[-1]:,} idle moments ({arm})\n")
        print("| Told | right |\n| --- | --- |")
        for kind in shots[0]:
            print(f"| {kind} | {span_of([shot[kind] for shot in shots])} |")
        forms = [clean_form(s, arm) for s in args.seeds]
        print(f"\n### The record against its takes ({arm})\n")
        print("| Takes | the record's cosine to the pattern | the nearest take's |\n"
              "| --- | --- | --- |")
        for n in DOSES:
            print(f"| {n} | {span_of([f[n][0] for f in forms])} | "
                  f"{span_of([f[n][1] for f in forms])} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
