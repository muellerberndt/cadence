"""Rung T0 of the parrot's turns: a pistachio for a reply in the gap after the trainer speaks.

A trainer speaks for ``speech`` moments and is quiet for ``gap`` moments: one exchange. The
exchange's first sound from the parrot decides it: in the gap at ``window`` (moments after the
trainer stops, 0.3 to 2 s in the declared rung) it earns a pistachio; over the trainer, too
early or too late, nothing. The yoked control receives the same pistachios at the same moments,
whatever it does (the yoked design of contingent social response). Readings: the share of the
parrot's sounds over the trainer (overlap), the share of exchanges whose first reply falls in
the window, and the reply latency.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cadence.parrot.voice import SILENT, Voice


def frames(rng: np.random.Generator, speaking: bool) -> np.ndarray:
    """One moment's ten log-mel frames: speech (energy in every band, varying), or silence."""
    if not speaking:
        return np.full((10, 40), SILENT)
    return rng.uniform(-3.0, 2.0, (10, 40))


def exchanges(voice: Voice, rng: np.random.Generator, count: int, *, speech: int = 3,
              gap: int = 9, window: tuple[int, int] = (2, 6),
              yoked: set[int] | None = None) -> dict[str, Any]:
    """``count`` exchanges; with ``yoked`` the pistachios come at those moments and the
    parrot's sounds pay nothing. Gives the log (exchange, moment, sound) and the moments paid."""
    log: list[tuple[int, int, bool]] = []
    paid: list[int] = []
    reward, moment = 0.0, 0
    for exchange in range(count):
        replied = False
        for k in range(speech + gap):
            speaking = k < speech
            sound = voice.moment(frames(rng, speaking), reward=reward, done=k == 0)
            after = k - speech + 1  # moments since the trainer stopped
            reward = 0.0
            if yoked is None:
                if sound and not replied:
                    replied = True
                    if not speaking and window[0] <= after <= window[1]:
                        reward = 1.0
                        paid.append(moment)
            elif moment in yoked:
                reward = 1.0
            log.append((exchange, k, sound))
            moment += 1
    return {"log": log, "paid": paid, "speech": speech, "window": window}


def readings(life: dict[str, Any], start: int = 0) -> dict[str, float]:
    """Overlap (sounds over the trainer), first replies in the window, the median latency (moments
    after the trainer stopped; exchanges with no reply count as the gap's end)."""
    speech, (low, high) = life["speech"], life["window"]
    rows = [x for x in life["log"] if x[0] >= start]
    over = [s for e, k, s in rows if k < speech]
    first: dict[int, int] = {}
    for e, k, s in rows:
        if s and e not in first:
            first[e] = k - speech + 1
    exchanges_ = sorted({e for e, _, _ in rows})
    latency = [first.get(e, 99) for e in exchanges_]
    return {"overlap": float(np.mean(over)) if over else 0.0,
            "in_window": float(np.mean([low <= x <= high for x in latency])),
            "latency": float(np.median(latency))}


__all__ = ["exchanges", "frames", "readings"]
