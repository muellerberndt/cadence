"""Hearing words and saying them: a question asked aloud, the raw sound into the brain; each
heard word kept as a record of the brain's memory beside the thing in focus; a voice that learns
to say a word by practising it against how it was heard (cadence#329, #333, #336).

**Born and learned.** Born: the cochlea (``cadence.speech.frontend.log_mel``: log-mel energies
of 25 ms frames every 10 ms), and the mechanisms below: a word's onset and offset against a
running level, a record a heard word, a record strengthened by repetition, and the voice's
practice. Learned: the receptive fields of the ear, the words (their records, their forms and
what they are heard with) and the voice's motor programs.

**The ear** (``Ear``).

- **Field L** (``cadence.fields.LearnedFields``): at each moment of the ear (every ``step``
  frames, 40 ms) the latest ``window`` frames (240 ms) are an image of 40 bands, read through
  ``features`` fields a local rule shapes from the sounds heard (a Hebbian dictionary on the
  residual, homeostatic thresholds) and pooled over overlapping regions. The fields learn from
  the ear's first ``settle`` moments of sound and hold still after.
- **The moment held:** the moment's code beside its leaky traces at ``decays`` (the heard
  history), at unit length: the last moments integrated and held for a few.
- **A word's onset and offset:** the ear keeps a running level of the sound, the quiet it
  hears between sounds (it follows a quieter frame at once and a louder one by ``rise`` a
  frame). A word begins where the sound rises ``onset`` above the level and ends where it has
  stayed under that for ``gap`` frames; a stretch of ``least`` frames or more is a word heard.
- **A record a heard word** (``cadence.longterm.LongTermMemory``): its key the moment held at
  the word's end, beside its context; its value the context and the word's frames (its
  template). A word whose held code a record holds at a cosine of ``revise`` or more is that
  record told again: the record moves to the running mean of its takes (its ``n``-th take
  closes ``1 / n``, no less than ``rate``), so its template comes out cleaner with every
  repetition.
- **The context is learned** (``teach``): each lesson moves a record's context toward the
  concepts present, its ``n``-th lesson closing ``1 / n``, so the context is the mean of what
  the word was heard with: the thing in focus at a naming, what a question's answer asks for
  and points to.
- **Sound to thing** (``recall``): every moment of a word heard cues the memory by sound; the
  read's context counts by how far its best record's cosine lies above ``threshold``, and the
  sound's concepts are those contexts' weighted mean.
- **Thing to sound** (``choose``): the concepts in focus cue the records by their context; each
  record read is heard back (its held code recalls its concepts), and the one whose heard-back
  concepts match the focus best is the word to say. A thing no word was heard with has none.

**The voice** (``Voice``): a motor program a word, frames in the format the ear reads, said
through a vocal tract (``tract``: frames to samples; ``sound`` unless given). A program starts
immature: the word's template's mean frame held over its length. Each practice ``attempt`` says
the program with a variation, the ear hears its own output, the program moves by ``rate`` of
the error toward the template, and a variation that sounded better than the recent attempts
(the self-reward: the match of the heard output to the template, against its running mean) is
kept at ``keep``. A drive to call (``boredom``) grows every moment; saying a word relieves it,
so the parrot practises on its own (``bored``).

**A brain with ears and a voice** (``Hearing``) joins an ``Ear`` and a ``Voice`` to a
``cadence.words.Words`` brain: ``name(samples, at)`` (a naming by ear), ``practice(samples,
answer)`` (a question asked aloud with the teacher's written answer), ``ask(samples)`` (the
written answer to a question asked aloud), ``say(concepts)`` (the voice's frames for the
concepts in focus; none for a thing whose word was never heard), ``practise()`` (one attempt
at the word heard last) and ``read(word, samples)`` (a word read aloud: each form taught the
concepts the other brings back).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..fields import LearnedFields
from ..longterm import LongTermMemory
from ..speech.frontend import MELS, log_mel, mel_bank

FRAME, HOP, FFT = 400, 160, 512  # the front end's frame, hop and spectrum, in samples


def sound(frames: Any, *, rounds: int = 16, peak: float = 0.5, seed: int = 0) -> np.ndarray:
    """A vocal tract: 16 kHz samples whose log-mel frames are near ``frames`` ``(frames,
    MELS)``. Each frame's band powers spread over the spectrum's bins through the mel filters
    (per unit of filter area), the phase found by Griffin and Lim's iteration (``rounds``
    passes at the front end's frame and hop), the samples scaled to a peak of ``peak``."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != MELS or len(frames) < 2:
        raise ValueError(f"frames is (frames, {MELS}) with 2 frames or more")
    bank = mel_bank(MELS)
    power = np.maximum(np.exp(frames) - 1e-6, 0.0)
    weight = bank / np.maximum(bank.sum(axis=1, keepdims=True), 1e-12)
    cover = bank.sum(axis=0)
    magnitude = np.sqrt(np.maximum((power @ weight) / np.maximum(cover, 1e-12), 0.0)) * (
        cover > 0)
    window = np.hanning(FRAME)
    count = len(frames)
    length = (count - 1) * HOP + FRAME
    index = np.arange(FRAME)[None, :] + HOP * np.arange(count)[:, None]

    def inverse(spec: np.ndarray) -> np.ndarray:
        pieces = np.fft.irfft(spec, FFT)[:, :FRAME] * window
        out, norm = np.zeros(length), np.zeros(length)
        np.add.at(out, index, pieces)
        np.add.at(norm, index, np.broadcast_to(window**2, pieces.shape))
        return out / np.maximum(norm, 1e-8)

    rng = np.random.default_rng(seed)
    samples = inverse(magnitude * np.exp(2j * np.pi * rng.random(magnitude.shape)))
    for _ in range(int(rounds)):
        spec = np.fft.rfft(samples[index] * window, FFT)
        samples = inverse(magnitude * np.exp(1j * np.angle(spec)))
    top = float(np.abs(samples).max())
    return np.asarray(samples * (peak / top if top > 0 else 1.0), dtype=np.float32)


def _resample(rows: np.ndarray, count: int) -> np.ndarray:
    where = np.linspace(0.0, len(rows) - 1, count)
    low = np.floor(where).astype(np.int64)
    high = np.minimum(low + 1, len(rows) - 1)
    share = (where - low)[:, None]
    return np.asarray(rows[low] * (1.0 - share) + rows[high] * share)


class Ear:
    """The ear's learned fields, its running level and its records of heard words; see the
    module note. ``concepts`` is the width of a record's context (the brain's concepts) and
    ``capacity`` the records the memory holds."""

    def __init__(self, concepts: int, *, window: int = 24, step: int = 4, features: int = 24,
                 decays: tuple[float, ...] = (0.6, 0.85), settle: int = 4000,
                 onset: float = 3.0, rise: float = 0.005, gap: int = 12, least: int = 15,
                 revise: float = 0.9, rate: float = 0.05, threshold: float = 0.5,
                 frames: int = 48, capacity: int = 8192, temperature: float = 0.05,
                 seed: int = 0):
        if int(concepts) < 1 or int(window) < 8 or int(step) < 1 or int(capacity) < 1:
            raise ValueError("concepts and capacity of 1 or more, a window of 8 frames or more")
        if any(not 0.0 <= float(d) < 1.0 for d in decays):
            raise ValueError("each decay lies in [0, 1)")
        self.concepts, self.window, self.step = int(concepts), int(window), int(step)
        self.decays = tuple(float(d) for d in decays)
        self.settle, self.onset, self.rise = int(settle), float(onset), float(rise)
        self.gap, self.least, self.revise = int(gap), int(least), float(revise)
        self.rate, self.threshold, self.frames = float(rate), float(threshold), int(frames)
        self.capacity, self.temperature = int(capacity), float(temperature)
        self.settings = {"concepts": self.concepts, "window": self.window, "step": self.step,
                         "features": int(features), "decays": self.decays,
                         "settle": self.settle, "onset": self.onset, "rise": self.rise,
                         "gap": self.gap, "least": self.least, "revise": self.revise,
                         "rate": self.rate, "threshold": self.threshold,
                         "frames": self.frames, "capacity": self.capacity,
                         "temperature": self.temperature, "seed": int(seed)}
        self.fields = LearnedFields(MELS, self.window, 1, int(features), seed=int(seed))
        self.code_width = self.fields.outputs * (1 + len(self.decays))
        self.memory = LongTermMemory(
            self.code_width + self.concepts, self.concepts + self.frames * MELS + 1,
            self.capacity, temperature=self.temperature, cosine=True, revise=1.0,
            direct=self.capacity, dtype=np.float32)
        self.bank = mel_bank(MELS)
        self.counts = np.zeros(self.capacity, dtype=np.int64)  # each record's takes
        self.told = np.zeros(self.capacity, dtype=np.int64)  # each record's context lessons
        self.level: float | None = None  # the running level of the sound (natural log)
        self.moments = 0  # the moments the fields have learned from
        self.work = {"sounds": 0, "moments": 0, "macs": 0}
        self.last: dict[str, Any] = {}

    @property
    def records(self) -> int:
        return int(self.memory.count)

    def parameters(self) -> int:
        """The learned values: the fields (dictionary and thresholds) and every record's key
        and value."""
        return int(self.fields.parameters() + self.records * (
            self.memory.width + self.memory.value_width))

    # -- a sound: its moments and its words

    def hear(self, samples: Any, *, learn: bool = False) -> tuple[np.ndarray, list[tuple[int,
                                                                                         int]],
                                                                    np.ndarray]:
        """A sound's moments held ``(moments, code_width)``, its words as ``(first frame,
        end frame)`` and its log-mel frames ``(frames, MELS)``. With ``learn`` the fields
        learn from the moments of words until they have learned from ``settle``, and the
        running level follows the sound."""
        frames = log_mel(np.asarray(samples, dtype=np.float64), bank=self.bank)
        return self.read(frames, learn=learn)

    def words(self, frames: np.ndarray, *, learn: bool = False) -> list[tuple[int, int]]:
        """The words of log-mel frames by their onset and offset against the running level
        (which moves only with ``learn``)."""
        energy = np.log(np.exp(frames).sum(axis=1))
        level = float(energy.min()) if self.level is None else self.level
        loud = np.zeros(len(frames), dtype=bool)
        for t, e in enumerate(energy):
            level = float(e) if e < level else level + self.rise
            loud[t] = e > level + self.onset
        if learn:
            self.level = level
        out: list[tuple[int, int]] = []
        held = np.flatnonzero(loud)
        if not len(held):
            return out
        begin = prev = int(held[0])
        for t in held[1:]:
            if int(t) - prev > self.gap:
                out.append((begin, prev + 1))
                begin = int(t)
            prev = int(t)
        out.append((begin, prev + 1))
        return [(a, b) for a, b in out if b - a >= self.least]

    def read(self, frames: Any, *, learn: bool = False) -> tuple[np.ndarray, list[tuple[int,
                                                                                         int]],
                                                                  np.ndarray]:
        """``hear`` from log-mel frames, ``(frames, MELS)``, as ``log_mel`` gives them."""
        frames = np.asarray(frames, dtype=np.float64)
        count = len(frames) // self.step
        if count == 0:
            return np.zeros((0, self.code_width)), [], frames
        words = self.words(frames, learn=learn)
        padded = np.concatenate([np.repeat(frames[:1], self.window, axis=0), frames])
        ends = self.step * (np.arange(count) + 1) + self.window
        index = ends[:, None] - self.window + np.arange(self.window)[None, :]
        windows = padded[index]  # (moments, window, MELS)
        windows = windows - windows.mean(axis=(1, 2), keepdims=True)
        images = np.ascontiguousarray(windows.transpose(0, 2, 1)).reshape(count, -1)
        inside = np.zeros(count, dtype=bool)
        for a, b in words:
            inside[a // self.step: (b + self.step - 1) // self.step] = True
        if learn and self.fields.learning and self.moments < self.settle and inside.any():
            self.fields.learn(images[inside])
            self.moments += int(inside.sum())
            if self.moments >= self.settle:
                self.fields.learning = False
        code = self.fields.code(images)
        self.work["macs"] += int(self.fields.macs()["code"] * count)
        held = [code]
        for decay in self.decays:
            trace = np.zeros_like(code)
            h = np.zeros(code.shape[1])
            for t in range(count):
                h = decay * h + (1.0 - decay) * code[t]
                trace[t] = h
            held.append(trace)
        out = np.concatenate(held, axis=1)
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        self.work["moments"] += count
        return out, words, frames

    # -- records

    def _key(self, code: np.ndarray, context: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(context))
        return np.concatenate([code, context / norm if norm > 0 else np.zeros(self.concepts)])

    def parts(self, record: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """A record's held code, context, template ``(frames, MELS)`` (resampled) and its
        length in frames."""
        key = np.asarray(self.memory.keys[record], dtype=np.float64)
        value = np.asarray(self.memory.values[record], dtype=np.float64)
        template = value[self.concepts: -1].reshape(self.frames, MELS)
        return (key[: self.code_width].copy(), value[: self.concepts].copy(),
                template.copy(), float(value[-1]))

    def template(self, record: int) -> np.ndarray:
        """How the record's word sounded: its frames at its own length, ``(frames, MELS)``."""
        _, _, template, length = self.parts(record)
        return _resample(template, max(int(round(length)), 2))

    def _write(self, record: int, code: np.ndarray, context: np.ndarray,
               template: np.ndarray, length: float) -> int:
        value = np.concatenate([context, template.reshape(-1), [length]])
        slot, _ = self.memory.write(self._key(code, context), value, recognised=record)
        return int(slot)

    def _cue(self, code: np.ndarray) -> np.ndarray:
        return np.concatenate([code, np.zeros(self.concepts)])

    def listen(self, samples: Any, *, learn: bool) -> tuple[np.ndarray, list[int]]:
        """A sound's moments of words ``(moments, code_width)`` and (with ``learn``) the
        records its words were written to or told again in, one a word."""
        code, words, frames = self.hear(samples, learn=learn)
        self.work["sounds"] += 1
        moments = np.zeros(len(code), dtype=bool)
        records: list[int] = []
        for a, b in words:
            first, end = a // self.step, min((b + self.step - 1) // self.step, len(code))
            moments[first:end] = True
            if learn and end > first:
                records.append(self._tell(code[end - 1], _resample(frames[a:b], self.frames),
                                          float(b - a)))
        self.last = {"words": words, "records": records}
        return code[moments], records

    def _tell(self, code: np.ndarray, template: np.ndarray, length: float) -> int:
        """A heard word written, or the record that holds its code told again: the record moves
        to the running mean of its takes."""
        if self.records:
            best = self.memory.read(self._cue(code), credit=False).first(0)
            if best >= 0:
                held, context, was, size = self.parts(best)
                if float(held @ code) >= self.revise * float(np.linalg.norm(held)):
                    self.counts[best] += 1
                    share = max(1.0 / float(self.counts[best]), self.rate)
                    held += share * (code - held)
                    was += share * (template - was)
                    size += share * (length - size)
                    self.memory.deep[best] += 1.0  # a take told again deepens its record
                    return self._write(best, held / max(np.linalg.norm(held), 1e-9), context,
                                       was, size)
        slot = self._write(-1, code, np.zeros(self.concepts), template, length)
        self.counts[slot], self.told[slot] = 1, 0
        return slot

    def teach(self, records: list[int], concepts: np.ndarray) -> None:
        """A context lesson: each record's context moves toward ``concepts``, its ``n``-th
        lesson closing ``1 / n`` (no less than ``rate``)."""
        for record in dict.fromkeys(records):
            self.told[record] += 1
            share = max(1.0 / float(self.told[record]), self.rate)
            code, context, template, length = self.parts(record)
            context += share * (np.asarray(concepts, dtype=np.float64) - context)
            self._write(record, code, context, template, length)

    def recall(self, codes: np.ndarray, *, credit: bool = True) -> np.ndarray:
        """The concepts a sound's moments bring back: each moment's held code reads the memory
        by sound, and its read counts by how far its best record lies above ``threshold``."""
        out, total = np.zeros(self.concepts), 0.0
        for code in codes:
            read = self.memory.read(self._cue(code), credit=credit)
            best = read.first(0)
            if best < 0:
                continue
            held = np.asarray(self.memory.keys[best, : self.code_width], dtype=np.float64)
            weight = float(held @ code) / max(float(np.linalg.norm(held)), 1e-9) \
                - self.threshold
            if weight > 0:
                out += weight * np.asarray(self.memory.value(read)[: self.concepts])
                total += weight
        self.work["macs"] += int(len(codes) * self.records * self.memory.width)
        return out / total if total > 0 else out

    def choose(self, concepts: np.ndarray, *, top: int = 32) -> tuple[int, float]:
        """The record of the word to say for the concepts in focus and how well it names them:
        the records the concepts read by context, each heard back (its held code recalls its
        concepts), the best match of the heard-back concepts with the focus; -1 for none."""
        cue = np.asarray(concepts, dtype=np.float64)
        norm = float(np.linalg.norm(cue))
        if not self.records or norm == 0:
            return -1, 0.0
        n = self.records
        keys = np.asarray(self.memory.keys[:n], dtype=np.float64)
        near = keys[:, self.code_width:] @ (cue / norm)
        self.work["macs"] += int(n * self.concepts)
        best, match = -1, 0.0
        for record in np.argsort(-near)[: max(int(top), 1)]:
            if near[record] <= 0:
                break
            meaning = self.recall(keys[record: record + 1, : self.code_width], credit=False)
            held = float(np.linalg.norm(meaning))
            cosine = float(meaning @ cue / (held * norm)) if held > 0 else 0.0
            if cosine > match:
                best, match = int(record), cosine
        return best, match


class Voice:
    """The parrot's voice: a motor program a word, practised against the ear's template; see
    the module note. ``tract`` turns frames into samples (``sound`` unless given)."""

    def __init__(self, ear: Ear, *, tract: Callable[[np.ndarray], np.ndarray] | None = None,
                 rate: float = 0.4, explore: float = 0.4, keep: float = 0.5,
                 recent: float = 0.7, boredom: float = 0.05, relief: float = 1.0,
                 seed: int = 0):
        self.ear = ear
        self.tract = tract if tract is not None else sound
        self.rate, self.explore, self.keep, self.recent = (float(rate), float(explore),
                                                           float(keep), float(recent))
        self.boredom, self.relief = float(boredom), float(relief)
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 329, 9]))
        self.programs: dict[int, np.ndarray] = {}
        self.matches: dict[int, float] = {}  # each word's running match: the self-reward's mean
        self.attempts: dict[int, int] = {}
        self.drive = 0.0  # the drive to call
        self.last: dict[str, Any] = {}

    def program(self, record: int) -> np.ndarray:
        """The word's motor program: immature at first (its template's mean frame held over
        its length)."""
        if record not in self.programs:
            template = self.ear.template(record)
            self.programs[record] = np.repeat(template.mean(axis=0, keepdims=True),
                                              len(template), axis=0)
        return self.programs[record]

    def say(self, record: int) -> np.ndarray:
        """The frames the voice says for a word: its program as it stands."""
        return np.asarray(self.program(record)).copy()

    def heard_back(self, frames: np.ndarray) -> np.ndarray:
        """The ear's frames of the voice's own output."""
        return np.asarray(log_mel(np.asarray(self.tract(frames), dtype=np.float64),
                                  bank=self.ear.bank))

    def match(self, frames: np.ndarray, record: int) -> float:
        """How the voice's output sounds against the word's template: the cosine of their log-mel
        frames (each less its own mean level), at the template's length."""
        template = self.ear.template(record)
        heard = _resample(self.heard_back(frames), len(template))
        a, b = heard - heard.mean(), template - template.mean()
        return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    def attempt(self, record: int) -> tuple[np.ndarray, float]:
        """One practice attempt: the program said with a variation and heard back; the program
        moves toward the template by the error, and keeps a variation that sounded better than
        the recent attempts. Gives the frames said and their match; relieves the drive."""
        program = self.program(record)
        template = self.ear.template(record)
        noise = self.rng.standard_normal(program.shape)
        variation = self.explore * (noise + np.roll(noise, 1, axis=0)
                                    + np.roll(noise, -1, axis=0)) / 3.0
        said = program + variation
        heard = _resample(self.heard_back(said), len(template))
        a, b = heard - heard.mean(), template - template.mean()
        match = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        moved = program + self.rate * (b - a)
        recent = self.matches.get(record, match)
        if match > recent:  # the self-reward: better than the recent attempts
            moved = moved + self.keep * variation
        self.programs[record] = moved
        self.matches[record] = self.recent * recent + (1.0 - self.recent) * match
        self.attempts[record] = self.attempts.get(record, 0) + 1
        self.drive = max(self.drive - self.relief, 0.0)
        self.last = {"record": record, "match": match}
        return said, match

    def moment(self) -> None:
        """One moment: the drive to call grows."""
        self.drive += self.boredom

    @property
    def bored(self) -> bool:
        return self.drive >= 1.0


class Hearing:
    """A brain with ears and a voice: ``brain`` (a ``cadence.words.Words``), an ``Ear`` over its
    concepts and a ``Voice`` (made here from ``settings`` and ``voice`` unless given).
    ``things`` is the share at which a question answered teaches its words the concepts of the
    things the answer points to, and ``say_at`` how well a word must name the concepts in focus
    for the voice to say it."""

    def __init__(self, brain: Any, ear: Ear | None = None, *, things: float = 1.0,
                 say_at: float = 0.5, voice: dict[str, Any] | None = None, **settings: Any):
        self.brain = brain
        self.ear = ear if ear is not None else Ear(brain.lexicon.concepts, **settings)
        if self.ear.concepts != brain.lexicon.concepts:
            raise ValueError("the ear's records hold the brain's concepts")
        self.voice = Voice(self.ear, **(voice or {}))
        self.things, self.say_at = float(things), float(say_at)
        self.heard_last: int | None = None  # the record of the word heard last
        self.last: dict[str, Any] = {}

    def listen(self, samples: Any) -> None:
        """A sound heard with nothing to learn from but the sound: the fields and the running
        level learn."""
        self.ear.hear(samples, learn=True)

    def heard(self, samples: Any, *, learn: bool) -> tuple[np.ndarray, list[int]]:
        """The concepts a sound brings back (before its lessons) and its words' records."""
        codes, records = self.ear.listen(samples, learn=learn)
        concepts = self.ear.recall(codes, credit=not learn) if len(codes) else np.zeros(
            self.ear.concepts)
        if records:
            self.heard_last = records[-1]
        self.last = {"records": records, "concepts": concepts}
        return concepts, records

    def question(self, concepts: np.ndarray) -> Any:
        """What a question's concepts ask, as ``Words.hear`` gives it for written words: what
        is asked about (the group of the senses whose most active concept is the most active of
        all, above the brain's floor) and each read's activity."""
        from .room import Heard

        brain = self.brain
        groups = brain.lexicon.groups
        about: tuple[str, np.ndarray] | None = None
        top = brain.floor
        for g in brain.senses.groups:
            part = concepts[groups[g]]
            if part.max() > top:
                top, about = float(part.max()), (g, part)
        reads = concepts[groups["read"]]
        return Heard([], concepts, about, {g: float(max(reads[i], 0.0))
                                           for i, g in enumerate(brain.answers)})

    def name(self, samples: Any, at: Any) -> int:
        """A naming by ear: the thing at ``at`` in focus and a word heard."""
        focus = self.brain.attend(at)
        _, records = self.heard(samples, learn=True)
        if focus is not None and records:
            self.ear.teach(records, focus)
        return len(records)

    def practice(self, samples: Any, answer: str | list[str]) -> None:
        """A question asked aloud with the teacher's written answer: its words are taught what
        the answer asks for (its read) and, at ``things``, the things it points to; an answer
        that names nothing the brain holds, given where the brain's read finds nothing, is the
        written word for the read that finds nothing (as ``Words.practice`` teaches it)."""
        brain = self.brain
        lexicon, groups = brain.lexicon, brain.lexicon.groups
        concepts, records = self.heard(samples, learn=True)
        found = brain.find(self.question(concepts))
        said = [answer] if isinstance(answer, str) else list(answer)
        known = [w for w in said if lexicon.unit(w) is not None]
        told = lexicon.hear(known) if known else np.zeros(lexicon.concepts)
        named = max(brain.answers, key=lambda g: float(told[groups[g]].max()))
        if told[groups[named]].max() > brain.floor:
            lesson = np.zeros(lexicon.concepts)
            lesson[groups["read"].start + brain.answers.index(named)] = 1.0
            places: list[np.ndarray] = []
            for word in known:
                one = lexicon.hear(word)
                group = max(brain.senses.groups, key=lambda g: float(one[groups[g]].max()))
                if one[groups[group]].max() > brain.floor:
                    places += brain.senses.find(group, one[groups[group]])
            things = [c for c in (brain.attend(p) for p in places) if c is not None]
            if things:
                lesson += self.things * np.mean(things, axis=0)
            if records:
                self.ear.teach(records, lesson)
        elif not found and len(said) == 1:
            lexicon.name(said[0], {("none", 0): 1.0}, groups=("none",))

    def read(self, word: str, samples: Any) -> None:
        """A word read aloud: its written form seen while its sound is heard, nothing attended;
        the written word is taught the concepts the sound brings back, the sound's records the
        concepts the written word names (each the more active of the two at each concept)."""
        brain = self.brain
        lexicon = brain.lexicon
        by_ear, records = self.heard(samples, learn=True)
        by_eye = (lexicon.hear(word) / lexicon.level if lexicon.unit(word) is not None
                  else np.zeros(lexicon.concepts))
        sensed = np.zeros(lexicon.concepts, dtype=bool)
        for g in brain.senses.groups:
            sensed[lexicon.groups[g]] = True
        active = np.where(sensed, np.maximum(np.maximum(by_ear, by_eye), 0.0), 0.0)
        if active.max() <= brain.floor:
            return
        lexicon.name(word, active, groups=tuple(brain.shares))
        if records:
            self.ear.teach(records, active)

    def ask(self, samples: Any) -> list[str]:
        """The words the brain says, in writing, to a question asked aloud."""
        concepts, _ = self.heard(samples, learn=False)
        heard = self.question(concepts)
        said: list[str] = []
        vectors: list[np.ndarray] = []
        for _place, described in self.brain.find(heard):
            vector = self.brain.answer(heard, described)
            word = self.brain.lexicon.say(vector) if vector is not None else None
            if word is not None and vector is not None:
                said.append(word)
                vectors.append(vector)
        if not said:
            said = [self.brain.lexicon.say(("none", 0), floor=0.02) or ""]
        self.last.update({"heard": heard, "said": said, "answers": vectors})
        return said

    def say(self, concepts: np.ndarray) -> np.ndarray:
        """The voice's frames for the concepts in focus, ``(frames, MELS)`` log-mel frames every
        10 ms (the format the ear reads): the program of the word that names them at
        ``say_at`` or more; none for a thing whose word was never heard."""
        record, match = self.ear.choose(concepts)
        self.last["said_record"] = record
        if record < 0 or match < self.say_at:
            return np.zeros((0, MELS))
        return self.voice.say(record)

    def practise(self, record: int | None = None) -> float | None:
        """One practice attempt at a word (the word heard last unless given); its match."""
        record = self.heard_last if record is None else record
        if record is None:
            return None
        return self.voice.attempt(record)[1]

    def moment(self) -> None:
        """One moment of the voice's drive: a bored parrot practises the word heard last."""
        self.voice.moment()
        if self.voice.bored:
            self.practise()

    def parameters(self) -> int:
        """The learned parameters: the lexicon's (one efficacy a reciprocal pair), the ear's
        fields and records, and the voice's programs."""
        return int(self.brain.lexicon.connectome.synapses // 2 + self.ear.parameters()
                   + sum(p.size for p in self.voice.programs.values()))
