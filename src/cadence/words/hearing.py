"""Hearing words: a question asked aloud, the raw sound into the brain; every heard moment kept
as a record of the brain's memory beside the thing in focus; a voice that says a word back as
it was heard (cadence#329, #333).

**What is born and what is learned.** Born: the cochlea, the speech front end
(``cadence.speech.frontend.log_mel``: log-mel energies of 25 ms frames every 10 ms). Learned:
everything after it.

- **Field L** (``LearnedFields``, the library's learned receptive fields): at each moment of
  the ear (every ``step`` frames, 40 ms) the latest ``window`` frames (240 ms) are an image of
  40 bands by ``window`` frames, read through ``features`` fields that a local rule shapes from
  the sounds heard (``cadence.fields``: a Hebbian dictionary on the residual and homeostatic
  thresholds), pooled over overlapping regions of the image. The fields learn while the ear
  listens (``listen(learn=True)``) for its first ``settle`` moments and hold still after.
- **The moment held** (the heard history): the moment's code beside its leaky traces at
  ``decays`` (each ``h = d h + (1 - d) code``), together at unit length: the sound of the last
  moments, integrated and held for a few moments, is what the moment cues the memory with.
- **Records** (``cadence.longterm.LongTermMemory``): every moment of sound heard while the brain
  lives is written as a record: its key the held code (and the context, below), its value the
  context and, for the moment that ends a sound, the frames heard since the sound began (up to
  ``span`` seconds), the voice's memory of how it sounded. A moment whose held code a record
  holds at a cosine of ``revise`` or more is that record told again: its code moves toward the
  new one by ``1 / n`` (no less than ``rate``) and the memory deepens it.
- **The context is learned** (``teach``): each lesson moves a record's context toward the
  concepts present, its ``n``-th lesson closing ``1 / n`` (no less than ``rate``), so a
  record's context is the mean of what its moments were heard with.

**Fetched by association, both ways.** A heard moment's held code cues the memory by its sound
(``recall``): the read (``temperature``, over the cosine) gives the records' contexts, and a
sound's concepts are their mean over its moments of sound. The concepts in focus cue the memory
by context among the records that end a sound (``voice``): the record read gives its frames.
No record is told which word it holds.

**A brain with ears and a voice** (``Hearing``) joins an ``Ear`` to a ``cadence.words.Words``
brain:

- ``name(samples, at)``: a naming by ear: the thing at ``at`` in focus and a word heard; the
  moments' records are taught the thing's concepts, each group at the brain's naming share;
- ``practice(samples, answer)``: a question asked aloud with the teacher's written answer: the
  moments' records are taught what the answer asks for (its read) and, at the share ``things``,
  the concepts of the things the answer points to;
- ``ask(samples)``: the brain's written answer to a question asked aloud: the concepts the
  moments bring back are what is asked about and what is asked, and the map is asked as for a
  written question;
- ``say(concepts)``: the voice's frames for the concepts in focus (the format the ear reads),
  where a record's context holds them at ``say_at`` or more; nothing otherwise;
- ``read(word, samples)``: a word read aloud: its written form seen while its sound is heard,
  nothing attended. The written word is taught the concepts the sound brings back, and the
  moments' records the concepts the written word names.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fields import LearnedFields
from ..longterm import LongTermMemory
from ..speech.frontend import MELS, log_mel, mel_bank


class Ear:
    """The ear's learned fields and its records; see the module note. ``concepts`` is the width
    of a record's context (the brain's concepts) and ``capacity`` the records the memory
    holds."""

    def __init__(self, concepts: int, *, window: int = 24, step: int = 4, features: int = 24,
                 decays: tuple[float, ...] = (0.6, 0.85), settle: int = 4000,
                 drop: float = 4.0, revise: float = 0.97, rate: float = 0.05,
                 span: float = 1.2, voice_frames: int = 32, capacity: int = 16384,
                 temperature: float = 0.05, seed: int = 0):
        if int(concepts) < 1 or int(window) < 8 or int(step) < 1 or int(capacity) < 1:
            raise ValueError("concepts and capacity of 1 or more, a window of 8 frames or more")
        if any(not 0.0 <= float(d) < 1.0 for d in decays):
            raise ValueError("each decay lies in [0, 1)")
        self.concepts, self.window, self.step = int(concepts), int(window), int(step)
        self.decays = tuple(float(d) for d in decays)
        self.settle, self.drop, self.revise = int(settle), float(drop), float(revise)
        self.rate, self.span, self.voice_frames = float(rate), float(span), int(voice_frames)
        self.capacity, self.temperature = int(capacity), float(temperature)
        self.settings = {"concepts": self.concepts, "window": self.window, "step": self.step,
                         "features": int(features), "decays": self.decays,
                         "settle": self.settle, "drop": self.drop, "revise": self.revise,
                         "rate": self.rate, "span": self.span,
                         "voice_frames": self.voice_frames, "capacity": self.capacity,
                         "temperature": self.temperature, "seed": int(seed)}
        self.fields = LearnedFields(MELS, self.window, 1, int(features), seed=int(seed))
        self.code_width = self.fields.outputs * (1 + len(self.decays))
        # a key: the held code, the context, and whether the record ends a sound
        self.memory = LongTermMemory(
            self.code_width + self.concepts + 1,
            self.concepts + self.voice_frames * MELS + 1, self.capacity,
            temperature=self.temperature, cosine=True, revise=1.0, direct=self.capacity,
            dtype=np.float32)
        self.bank = mel_bank(MELS)
        self.counts = np.zeros(self.capacity, dtype=np.int64)  # each record's moments
        self.told = np.zeros(self.capacity, dtype=np.int64)  # each record's context lessons
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

    # -- the moments of a sound

    def hear(self, samples: Any, *, learn: bool = False) -> tuple[np.ndarray, np.ndarray,
                                                                    np.ndarray]:
        """A sound's moments: the held codes ``(moments, code_width)``, which moments hold
        sound, and the log-mel frames ``(frames, MELS)``. With ``learn`` the fields learn from
        the moments of sound until they have learned from ``settle``."""
        frames = log_mel(np.asarray(samples, dtype=np.float64), bank=self.bank)
        return self.read(frames, learn=learn)

    def read(self, frames: Any, *, learn: bool = False) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray]:
        """``hear`` from log-mel frames, ``(frames, MELS)``, as ``log_mel`` gives them."""
        frames = np.asarray(frames, dtype=np.float64)
        count = len(frames) // self.step
        if count == 0:
            return np.zeros((0, self.code_width)), np.zeros(0, dtype=bool), frames
        energy = np.log(np.exp(frames).sum(axis=1))
        loud_frames = energy > energy.max() - self.drop
        padded = np.concatenate([np.repeat(frames[:1], self.window, axis=0), frames])
        ends = self.step * (np.arange(count) + 1) + self.window
        index = ends[:, None] - self.window + np.arange(self.window)[None, :]
        windows = padded[index]  # (moments, window, MELS)
        windows = windows - windows.mean(axis=(1, 2), keepdims=True)
        images = np.ascontiguousarray(windows.transpose(0, 2, 1)).reshape(count, -1)
        loud = loud_frames[: count * self.step].reshape(count, self.step).any(axis=1)
        if learn and self.fields.learning and self.moments < self.settle and loud.any():
            self.fields.learn(images[loud])
            self.moments += int(loud.sum())
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
        return out, loud, frames

    # -- records

    def _key(self, code: np.ndarray, context: np.ndarray, ending: float) -> np.ndarray:
        norm = float(np.linalg.norm(context))
        return np.concatenate([code, context / norm if norm > 0 else np.zeros(self.concepts),
                               [ending]])

    def parts(self, record: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
        """A record's held code, context, whether it ends a sound, voice frames ``(frames,
        MELS)`` and length in frames."""
        key = np.asarray(self.memory.keys[record], dtype=np.float64)
        value = np.asarray(self.memory.values[record], dtype=np.float64)
        voice = value[self.concepts: -1].reshape(self.voice_frames, MELS)
        return (key[: self.code_width].copy(), value[: self.concepts].copy(), float(key[-1]),
                voice.copy(), float(value[-1]))

    def _write(self, record: int, code: np.ndarray, context: np.ndarray, ending: float,
               voice: np.ndarray, length: float) -> int:
        value = np.concatenate([context, voice.reshape(-1), [length]])
        slot, _ = self.memory.write(self._key(code, context, ending), value, recognised=record)
        return int(slot)

    @staticmethod
    def _resample(rows: np.ndarray, count: int) -> np.ndarray:
        where = np.linspace(0.0, len(rows) - 1, count)
        low = np.floor(where).astype(np.int64)
        high = np.minimum(low + 1, len(rows) - 1)
        share = (where - low)[:, None]
        return np.asarray(rows[low] * (1.0 - share) + rows[high] * share)

    def _sound_cue(self, code: np.ndarray) -> np.ndarray:
        return np.concatenate([code, np.zeros(self.concepts + 1)])

    def listen(self, samples: Any, *, learn: bool) -> tuple[np.ndarray, list[int]]:
        """A sound's moments of sound, their held codes ``(moments, code_width)``, and (with
        ``learn``) the records they were written to or told again in."""
        code, loud, frames = self.hear(samples, learn=learn)
        self.work["sounds"] += 1
        held = code[loud]
        records: list[int] = []
        if learn and len(held):
            last = int(np.flatnonzero(loud)[-1])
            began = int(np.flatnonzero(loud)[0]) * self.step
            for t in np.flatnonzero(loud):
                ending = 1.0 if t == last else 0.0
                said = frames[max(began, (t + 1) * self.step - int(self.span * 100)):
                              (t + 1) * self.step]
                voice = (self._resample(said, self.voice_frames) if ending and len(said) > 1
                         else np.zeros((self.voice_frames, MELS)))
                records.append(self._tell(code[t], ending, voice, float(len(said) * ending)))
        self.last = {"moments": len(held), "records": records}
        return held, records

    def _tell(self, code: np.ndarray, ending: float, voice: np.ndarray, length: float) -> int:
        """A moment written, or the record that holds its code told again."""
        if self.records:
            read = self.memory.read(self._sound_cue(code), credit=False)
            best = read.first(0)
            if best >= 0:
                held, context, end, said, was = self.parts(best)
                if float(held @ code) >= self.revise * float(np.linalg.norm(held)) and (
                        end == ending):
                    self.counts[best] += 1
                    share = max(1.0 / float(self.counts[best]), self.rate)
                    held += share * (code - held)
                    if ending:
                        said += share * (voice - said)
                        was += share * (length - was)
                    return self._write(best, held / max(np.linalg.norm(held), 1e-9), context,
                                       end, said, was)
        slot = self._write(-1, code, np.zeros(self.concepts), ending, voice, length)
        self.counts[slot], self.told[slot] = 1, 0
        return slot

    def teach(self, records: list[int], concepts: np.ndarray) -> None:
        """A context lesson: each record's context moves toward ``concepts``, its ``n``-th
        lesson closing ``1 / n`` (no less than ``rate``)."""
        for record in dict.fromkeys(records):
            self.told[record] += 1
            share = max(1.0 / float(self.told[record]), self.rate)
            code, context, ending, voice, length = self.parts(record)
            context += share * (np.asarray(concepts, dtype=np.float64) - context)
            self._write(record, code, context, ending, voice, length)

    def recall(self, codes: np.ndarray, *, credit: bool = True) -> np.ndarray:
        """The concepts a sound's moments bring back: each moment's held code reads the
        memory by sound, and the contexts read are averaged over the moments."""
        out = np.zeros(self.concepts)
        for code in codes:
            read = self.memory.read(self._sound_cue(code), credit=credit)
            out += np.asarray(self.memory.value(read)[: self.concepts])
        self.work["macs"] += int(len(codes) * self.records * self.memory.width)
        return out / max(len(codes), 1)

    def voice(self, concepts: np.ndarray, *, credit: bool = True) -> tuple[np.ndarray, float]:
        """The frames of the record that ends a sound and whose context the concepts in focus
        fetch: ``(frames, MELS)`` at its own length, and the cosine of its context with the
        cue."""
        cue = np.asarray(concepts, dtype=np.float64)
        norm = float(np.linalg.norm(cue))
        if not self.records or norm == 0:
            return np.zeros((0, MELS)), 0.0
        key = np.concatenate([np.zeros(self.code_width), cue / norm, [1.0]])
        read = self.memory.read(key, credit=credit)
        self.work["macs"] += int(self.records * self.memory.width)
        best = read.first(0)
        if best < 0:
            return np.zeros((0, MELS)), 0.0
        _, context, ending, voice, length = self.parts(best)
        held = float(np.linalg.norm(context))
        match = float(context @ cue / (held * norm)) if held > 0 and ending else 0.0
        return self._resample(voice, max(int(round(length)), 2)), match


class Hearing:
    """A brain with ears and a voice: ``brain`` (a ``cadence.words.Words``) and an ``Ear`` over
    its concepts (made here from ``settings`` unless given). ``things`` is the share at which a
    question answered teaches its moments the concepts of the things the answer points to, and
    ``say_at`` the cosine a record's context needs with the focus for the voice to say it."""

    def __init__(self, brain: Any, ear: Ear | None = None, *, things: float = 1.0,
                 say_at: float = 0.5, **settings: Any):
        self.brain = brain
        self.ear = ear if ear is not None else Ear(brain.lexicon.concepts, **settings)
        if self.ear.concepts != brain.lexicon.concepts:
            raise ValueError("the ear's records hold the brain's concepts")
        self.things, self.say_at = float(things), float(say_at)
        self.last: dict[str, Any] = {}

    def listen(self, samples: Any) -> None:
        """A sound heard with nothing to learn from but the sound: the fields learn."""
        self.ear.hear(samples, learn=True)

    def heard(self, samples: Any, *, learn: bool) -> tuple[np.ndarray, list[int]]:
        """The concepts a sound brings back (before its lessons) and its records."""
        codes, records = self.ear.listen(samples, learn=learn)
        concepts = self.ear.recall(codes, credit=not learn) if len(codes) else np.zeros(
            self.ear.concepts)
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
        """A question asked aloud with the teacher's written answer (see the module note); an
        answer that names nothing the brain holds, given where the brain's read finds nothing,
        is the written word for the read that finds nothing (as ``Words.practice`` teaches
        it)."""
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
        """A word read aloud (see the module note)."""
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
            said = [self.brain.lexicon.say(("none", 0)) or ""]
        self.last.update({"heard": heard, "said": said, "answers": vectors})
        return said

    def say(self, concepts: np.ndarray) -> np.ndarray:
        """The voice's frames for the concepts in focus, ``(frames, MELS)`` log-mel frames every
        10 ms (the format the ear reads); none for concepts no record holds at ``say_at``."""
        frames, match = self.ear.voice(concepts)
        return frames if match >= self.say_at else np.zeros((0, MELS))

    def parameters(self) -> int:
        """The learned parameters: the lexicon's (one efficacy a reciprocal pair) and the
        ear's fields and records."""
        return int(self.brain.lexicon.connectome.synapses // 2 + self.ear.parameters())
