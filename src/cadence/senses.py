"""Two senses a brain declares by name: a sense for things and an ear (cadence 0.101).

``Things`` reports a seen thing as tuned cells, one population a field (a hue, a shape's
dimensions, a surface, a direction, a distance), each a ``PopulationCode``: the reading lights
the cells whose preferred values lie near it. ``Ear`` hears an utterance (a word or a phrase,
from its onset to the quiet after it) as one fixed code: its log-mel frames resampled to a few
frames, each band kept within a range of the frame's loudest, less the utterance's mean level, at
unit length. Both are born: they learn nothing and a brain does not save them. A brain declares
them by their widths::

    things = Things({"hue": (0.0, 1.0), "shape": [(0.0, 1.0)] * 3})
    ear = Ear()
    brain = Brain.compose(senses={"eyes": things.widths, "ear": ear.width}, actions=2, code=512)
    brain.live({"eyes": things.rows({"hue": [0.1], "shape": [[0.9, 0.1, 0.5]]}),
                "ear": ear.code(frames)})

Sources: a population of tuned cells a feature (Pouget, Dayan and Zemel 2000); an auditory
template of a whole song or call, as a songbird's (Konishi 1965; Bolhuis and Gahr 2006).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .tuning import PopulationCode

__all__ = ["Ear", "Things"]


class Things:
    """A sense for things: each field of a seen thing as tuned cells.

    ``fields`` maps a field's name to its range ``(low, high)``, or to a sequence of ranges for
    a field of several readings (a shape of three dimensions). ``per`` is the cells a reading
    (``PopulationCode``'s ``per``). ``widths`` is what ``Brain.compose(senses={"eyes":
    things.widths})`` takes, and ``rows`` turns readings into the rows a brain is given."""

    def __init__(self, fields: Mapping[str, Any], per: int = 8, floor: float = 0.05) -> None:
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("fields maps each field's name to its range or ranges")
        self.codes: dict[str, PopulationCode] = {}
        for name, spec in fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError("a field has a name")
            ranges = np.asarray(spec, dtype=np.float64)
            ranges = ranges.reshape(1, 2) if ranges.shape == (2,) else ranges
            pairs = [(float(low), float(high)) for low, high in ranges]
            self.codes[name] = PopulationCode(pairs, per=per, floor=floor)

    @property
    def widths(self) -> dict[str, int]:
        """Each field's cells, in the fields' order: the sense's layout for ``senses=``."""
        return {name: int(code.inputs) for name, code in self.codes.items()}

    @property
    def width(self) -> int:
        """The sense's width: every field's cells."""
        return int(sum(self.widths.values()))

    def rows(self, readings: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """The tuned cells of each field read, ``{field: (rows, cells)}``: ``readings`` maps a
        field to its readings, one value (or one row of values) a thing. A field left out is
        silent; an unknown field is refused by name."""
        out: dict[str, np.ndarray] = {}
        for name, value in readings.items():
            if name not in self.codes:
                raise ValueError(f"no field {name!r}: the fields are " + ", ".join(
                    repr(known) for known in self.codes))
            code = self.codes[name]
            values = np.asarray(value, dtype=np.float64).reshape(-1, len(code.ranges))
            out[name] = code(values)
        return out


class Ear:
    """An ear that hears an utterance as one code.

    ``frames`` is the frames an utterance is resampled to and ``bands`` the log-mel bands a
    frame holds (``cadence.speech.frontend.log_mel``'s 40). ``range`` is how far below a frame's
    loudest band a band is heard (quieter bands read as the room's floor). ``width`` is the
    sense's width. ``code(frames)`` gives one utterance's code and ``hear(audio)`` the code of
    raw samples. A longer and a shorter take of one word read alike (the resampling), and so do
    a louder and a softer one (the mean level is taken out)."""

    def __init__(self, frames: int = 16, bands: int = 40, range: float = 8.0) -> None:
        for name, value in (("frames", frames), ("bands", bands)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f"{name} is an integer of at least 2")
        if not np.isfinite(range) or range <= 0:
            raise ValueError("range is a positive number of nats")
        self.frames, self.bands, self.range = int(frames), int(bands), float(range)

    @property
    def width(self) -> int:
        """The sense's width: ``frames`` times ``bands``."""
        return self.frames * self.bands

    def code(self, frames: Any) -> np.ndarray:
        """The code of one utterance's log-mel frames ``(time, bands)``, ``(1, width)`` at unit
        length; silence (no frames, or frames of one level) reads zeros."""
        f = np.asarray(frames, dtype=np.float64)
        if f.ndim != 2 or f.shape[1] != self.bands:
            raise ValueError(f"an utterance is frames of {self.bands} bands, (time, bands)")
        if not np.isfinite(f).all():
            raise ValueError("an utterance's frames are finite")
        out = np.zeros((1, self.width))
        if len(f) < 1:
            return out
        at = np.linspace(0.0, len(f) - 1.0, self.frames)
        low = np.floor(at).astype(int)
        high = np.minimum(low + 1, len(f) - 1)
        w = (at - low)[:, None]
        x = f[low] * (1.0 - w) + f[high] * w
        x = np.maximum(x, x.max(axis=1, keepdims=True) - self.range)
        x = x - x.mean()
        norm = float(np.linalg.norm(x))
        if norm > 1e-9:
            out[0] = (x / norm).ravel()
        return out

    def hear(self, audio: Any, rate: int = 16000) -> np.ndarray:
        """The code of raw ``audio`` samples at ``rate`` a second (one utterance), through
        ``cadence.speech.frontend.log_mel``."""
        from .speech.frontend import log_mel

        return self.code(log_mel(np.asarray(audio, dtype=np.float64), rate=rate,
                                 mels=self.bands))

    def codes(self, utterances: Sequence[Any]) -> np.ndarray:
        """The codes of several utterances, ``(len(utterances), width)``."""
        return np.concatenate([self.code(u) for u in utterances]) if len(utterances) else (
            np.zeros((0, self.width)))
