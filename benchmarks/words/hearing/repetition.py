"""A word learned better with every repetition (cadence#329, #336's running mean): the record a
heard word is kept in moves to the running mean of its takes, so its template comes out
cleaner the more often the word is heard.

For a founder, ``words`` of its kind words are each said by one training voice: one clean take
(the voice's own, nothing added) and ``takes`` noisy ones (other takes of the same voice, each
with white noise at ``snr`` decibels below the take's power and a level drawn over 12
decibels). A fresh ear listens to every noisy take once (its fields settle), then hears each
word's noisy takes one by one; after takes 1, 2, 4, 8 and 16 the word's record (the one most of
its takes were told in) is read against the clean take: the mean absolute difference of their
log-mel frames, each resampled to the record's length and less its own mean level (natural log
units). Also read: how many records a word's takes made.

Usage: ``repetition.py <seed> <out.json>`` (on a box with the voices).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import voices  # noqa: E402
import world  # noqa: E402

READ_AT = (1, 2, 4, 8, 16)


def run(seed: int, *, words: int = 6, takes: int = 16, snr: float = 10.0,
        folder: str = "/data/voices", **ear: Any) -> dict[str, Any]:
    import hearing

    from cadence.speech.frontend import log_mel
    from cadence.words.hearing import Ear

    lex = world.lexicon(seed)
    trainers = hearing.voice_sets(seed, count=24, new=1, other=1)["train"]
    made = voices.Voices(folder)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 329, 21]))
    said: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
    for k in range(words):
        word, voice = lex.kind[k], trainers[k % len(trainers)]
        text = voices.sentence([word], question=False)
        clean = made.say(text, voice, rng).astype(np.float64)
        noisy = []
        for _ in range(takes):
            take = made.say(text, voice, rng).astype(np.float64)
            power = float(np.mean(take**2))
            take = take + rng.standard_normal(len(take)) * np.sqrt(power / 10 ** (snr / 10))
            noisy.append(take * 10 ** (rng.uniform(-6.0, 6.0) / 20))
        said[word] = (clean, noisy)
    settings = {"settle": 1500, **ear}
    ear_ = Ear(1, **settings)
    for _ in range(2):
        for _, noisy in said.values():
            for take in noisy:
                ear_.hear(take, learn=True)
    out: dict[str, Any] = {"seed": seed, "words": words, "takes": takes, "snr": snr,
                           "settings": settings}
    distances: dict[int, list[float]] = {n: [] for n in READ_AT}
    records: list[int] = []
    for clean, noisy in said.values():
        clean_frames = log_mel(clean)
        energy = np.log(np.exp(clean_frames).sum(axis=1))
        loud = np.flatnonzero(energy > energy.max() - 4.0)
        clean_frames = clean_frames[loud[0]: loud[-1] + 1]
        told: list[int] = []
        for n, take in enumerate(noisy, start=1):
            _, kept = ear_.listen(take, learn=True)
            told += kept
            if n in READ_AT and told:
                record = max(set(told), key=told.count)
                template = ear_.template(record)
                ref = np.asarray([np.interp(np.linspace(0, len(clean_frames) - 1,
                                                        len(template)),
                                            np.arange(len(clean_frames)), clean_frames[:, b])
                                  for b in range(clean_frames.shape[1])]).T
                a, b = template - template.mean(), ref - ref.mean()
                distances[n].append(float(np.mean(np.abs(a - b))))
        records.append(len(set(told)))
    out["distance"] = {str(n): float(np.mean(v)) for n, v in distances.items() if v}
    out["distance_each"] = {str(n): v for n, v in distances.items()}
    out["records_a_word"] = records
    return out


if __name__ == "__main__":
    result = run(int(sys.argv[1]), **(json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}))
    Path(sys.argv[2]).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "distance_each"}, indent=1))
