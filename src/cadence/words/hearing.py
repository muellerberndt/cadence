"""Hearing words: a question asked aloud, the raw sound into the brain, and the sound of a word
bound to the concept its written form names (cadence#329).

A heard word is a second form of a word. The sound and the letters each reach the lexicon's
concept neurons through word neurons of their own, so a word learned by ear and the same word
learned in writing name one concept: the concepts are where the two forms meet.

**The ear** (``Ear``) hears samples through the speech front end (``cadence.speech.frontend``:
log-mel energies of 25 ms frames every 10 ms) as an image of 20 ms slices: the slices within
``drop`` of the loudest frame are the sound, and each slice is its spectrum's shape (the first
``cepstra`` cosine terms of its log-mel energies, beside their change over 40 ms), each term
standardized by the statistics of the sound the ear has heard (its first ``settle`` slices),
at unit length. The ear holds **heard forms**, units each holding the sound of a word, or of a
few words said together, as a sequence of slices: its field.

- **The units explain the sound together.** The sound is read as a chain of units, each unit's
  field laid along a stretch of the sound slice by slice, holding a slice or skipping one where
  the sound is slower or faster than the field (each at a cost of ``stay`` or ``skip``). A
  slice explained by a unit scores its cosine with the field's slice less ``threshold``, each
  unit entered costs ``enter``, and a slice no unit explains scores zero. The chain heard is
  the one of the highest score: the units it holds are what the ear heard, in their order.
- **Learning**, on every sound the brain hears while it lives (never on a test question). A
  unit that explained a stretch moves each slice of its field toward the slices of sound laid
  on it, its ``n``-th hearing closing ``1 / n`` of the difference (no less than ``rate``), so
  its field is the mean of what it heard, where its mean cosine on the stretch is ``vigilance``
  or more. Under ``vigilance`` the stretch takes a new unit instead, and that unit is what was
  heard: a word heard in a new voice, or in a new context, gains a form of its own. A run of
  sound no unit explained, of ``least`` seconds or more (split evenly above ``most``), takes a
  new unit whose field is that run: a word heard for the first time enters the ear as a
  written word enters the lexicon.

No unit is told which word it heard: a unit enters from the sound alone and means what the
lessons of the brain's lexicon bind it to.

**A brain with ears** (``Hearing``) joins an ``Ear`` to a ``cadence.words.Words`` brain. Each
unit is a word neuron of the brain's lexicon with no letters (its key is ``"~"`` and the unit's
number): the lexicon binds it to concepts by the lessons it binds a written word with, and the
motor says written words only.

- ``name(samples, at)``: a naming by ear: the brain attends to the thing at ``at`` and hears a
  word said; the units heard are taught the thing's concepts.
- ``practice(samples, answer)``: a question asked aloud with the teacher's written answer: the
  brain's own practice, and a question answered names the thing the answer points to for the
  units heard in it (``Answered``, a part of the brain): a word heard inside a question is
  bound to its concept as a word heard alone is.
- ``ask(samples)``: the words the brain says, in writing, to a question asked aloud.
- ``read(word, samples)``: a word read aloud: its written form seen while its sound is heard,
  with nothing attended. Each form is taught the concepts active at that moment: at each concept
  the more active of the two forms' settles. A word learned by ear alone is so understood and
  said in writing, and a word learned in writing alone is understood by ear.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import numpy as np

from ..speech.frontend import MELS, log_mel, mel_bank
from .lexicon import Lexicon

SLICE = 2  # frames of 10 ms to a slice of the image: 20 ms
_NONE = -(10.0**18)


class Ear:
    """Heard forms over the log-mel image; see the module note."""

    def __init__(self, *, threshold: float = 0.5, vigilance: float = 0.7, least: float = 0.2,
                 most: float = 1.6, drop: float = 4.0, cepstra: int = 13,
                 stay: float = 0.05, skip: float = 0.05, enter: float = 1.0,
                 rate: float = 0.05, settle: int = 3000):
        if not -1.0 < float(threshold) < 1.0 or float(vigilance) < float(threshold):
            raise ValueError("a threshold in (-1, 1) and a vigilance at or above it")
        if not 1 <= int(cepstra) < MELS or float(least) <= 0 or float(most) < float(least):
            raise ValueError(f"cepstra 1 to {MELS - 1}, and 0 < least <= most")
        self.threshold, self.vigilance = float(threshold), float(vigilance)
        self.least, self.most, self.drop = float(least), float(most), float(drop)
        self.cepstra, self.stay, self.skip = int(cepstra), float(stay), float(skip)
        self.enter, self.rate, self.settle = float(enter), float(rate), int(settle)
        self.settings = {"threshold": self.threshold, "vigilance": self.vigilance,
                         "least": self.least, "most": self.most, "drop": self.drop,
                         "cepstra": self.cepstra, "stay": self.stay, "skip": self.skip,
                         "enter": self.enter, "rate": self.rate, "settle": self.settle}
        self.width = 2 * self.cepstra  # a slice: the spectrum's shape and its change
        self.bank = mel_bank(MELS)
        k = np.arange(1, self.cepstra + 1)[:, None]
        i = np.arange(MELS)[None, :]
        self._cosines = np.cos(np.pi * k * (2 * i + 1) / (2 * MELS))
        self.fields = np.zeros((0, self.width))  # every unit's slices, unit after unit
        self.owner = np.zeros(0, dtype=np.int64)  # the unit of each slice
        self.first = np.zeros(0, dtype=np.int64)  # each unit's first slice
        self.lengths = np.zeros(0, dtype=np.int64)  # each unit's slices
        self.counts = np.zeros(0, dtype=np.int64)  # each unit's hearings
        # the statistics of the sound heard, each term's mean and square, over ``heard`` slices
        self.sums = np.zeros((2, self.width))
        self.heard = 0
        self.work = {"sounds": 0, "slices": 0, "macs": 0}
        self.last: dict[str, Any] = {}

    @property
    def units(self) -> int:
        return len(self.first)

    def parameters(self) -> int:
        """The learned values of the ear: every slice of every unit's field."""
        return int(self.fields.size)

    # -- the image

    def image(self, samples: Any, *, learn: bool = False) -> np.ndarray:
        """The sound's slices, ``(slices, width)``: each term standardized by the statistics
        of the sound the ear has heard (``learn`` adds this sound's until ``settle`` slices are
        heard; before any, the sound's own), at unit length. None where the samples hold no
        sound."""
        frames = log_mel(np.asarray(samples, dtype=np.float64), bank=self.bank)
        energy = np.log(np.exp(frames).sum(axis=1))
        if not np.isfinite(energy).all():
            return np.zeros((0, self.width))
        loud = np.flatnonzero(energy > energy.max() - self.drop)
        frames = frames[loud[0]: loud[-1] + 1]
        shape = frames @ self._cosines.T
        padded = np.concatenate([shape[:1], shape[:1], shape, shape[-1:], shape[-1:]])
        change = ((padded[3:-1] - padded[1:-3]) + 2.0 * (padded[4:] - padded[:-4])) / 5.0
        both = np.concatenate([shape, change], axis=1)
        count = len(both) // SLICE
        if count == 0:
            return np.zeros((0, self.width))
        image = both[: count * SLICE].reshape(count, SLICE, self.width).mean(axis=1)
        if learn and self.heard < self.settle:
            self.sums += np.stack([image.sum(axis=0), (image**2).sum(axis=0)])
            self.heard += count
        sums, total = (self.sums, self.heard) if self.heard else (
            np.stack([image.sum(axis=0), (image**2).sum(axis=0)]), count)
        mean = sums[0] / total
        spread = np.sqrt(np.maximum(sums[1] / total - mean**2, 1e-6))
        image = (image - mean) / spread
        return np.asarray(image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True),
                                             1e-9))

    def _slices(self, seconds: float) -> int:
        return max(int(round(seconds * 100 / SLICE)), 1)

    # -- the chain of units that explains a sound

    def chain(self, image: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
        """The best chain of units and unexplained slices over ``image``: a list of ``(start,
        end, unit, places)``, ``unit`` -1 for a run no unit explains and ``places`` the slice
        of the unit's field each slice of the stretch was laid on."""
        count = len(image)
        if count == 0:
            return []
        if self.units == 0:
            return [(0, count, -1, np.zeros(0, dtype=np.int64))]
        n = len(self.fields)
        score = image @ self.fields.T - self.threshold
        self.work["macs"] += int(count * n * (self.width + 8))
        first = np.zeros(n, dtype=bool)
        first[self.first] = True
        second = np.zeros(n, dtype=bool)
        second[(self.first + 1)[self.lengths > 1]] = True
        last = self.first + self.lengths - 1
        held = np.full(n, _NONE)
        held[first] = score[0, first] - self.enter
        choice = np.zeros((count, n), dtype=np.int8)  # 0 on, 1 held, 2 skipped, 3 entered
        ends = np.zeros(count, dtype=np.int64)  # the best end at each slice: -1 unexplained
        unexplained = 0.0

        def best_end(held: np.ndarray, unexplained: float) -> tuple[float, int]:
            j = int(np.argmax(held[last]))
            return (unexplained, -1) if unexplained >= held[last[j]] else (
                float(held[last[j]]), int(last[j]))

        top, ends[0] = best_end(held, unexplained)
        for t in range(1, count):
            on = np.full(n, _NONE)
            on[1:] = held[:-1]
            on[first] = _NONE
            kept = held - self.stay
            skipped = np.full(n, _NONE)
            skipped[2:] = held[:-2] - self.skip
            skipped[first | second] = _NONE
            options = np.stack([on, kept, skipped, np.where(first, top - self.enter, _NONE)])
            choice[t] = options.argmax(axis=0)
            held = options.max(axis=0) + score[t]
            unexplained = top  # an unexplained slice scores zero
            top, ends[t] = best_end(held, unexplained)
        # back along the chain, slice by slice
        laid = np.full(count, -1, dtype=np.int64)  # the field slice each slice was laid on
        began = np.zeros(count, dtype=bool)  # a slice where a unit's stretch began
        state, t = int(ends[count - 1]), count - 1
        while t >= 0:
            if state < 0:
                laid[t] = -1
                state = int(ends[t - 1]) if t > 0 else -1
            else:
                laid[t] = state
                c = int(choice[t, state])
                if c == 3 or t == 0:
                    began[t] = True
                    state = int(ends[t - 1]) if t > 0 else -1
                else:
                    state -= (1, 0, 2)[c]
            t -= 1
        out: list[tuple[int, int, int, np.ndarray]] = []
        t = 0
        while t < count:
            if laid[t] < 0:
                end = t
                while end < count and laid[end] < 0:
                    end += 1
                out.append((t, end, -1, np.zeros(0, dtype=np.int64)))
            else:
                unit = int(self.owner[laid[t]])
                end = t + 1
                while end < count and laid[end] >= 0 and not began[end]:
                    end += 1
                out.append((t, end, unit, laid[t:end] - self.first[unit]))
            t = end
        return out

    # -- hearing

    def listen(self, samples: Any, *, learn: bool) -> list[int]:
        """The units heard in a sound, in their order. With ``learn`` the units learn from it
        and new units enter (see the module note)."""
        image = self.image(samples, learn=learn)
        self.work["sounds"] += 1
        self.work["slices"] += len(image)
        chain = self.chain(image)
        heard: list[int] = []
        made: list[tuple[int, int]] = []
        for start, end, unit, places in chain:
            if unit >= 0:
                cosine = float(np.mean(np.sum(
                    image[start:end] * self.fields[self.first[unit] + places], axis=1)))
                if not learn:
                    heard.append(unit)
                elif cosine >= self.vigilance:
                    heard.append(unit)
                    self._learn(unit, image[start:end], places)
                elif end - start >= self._slices(self.least):
                    made.append((start, end))
            elif learn and end - start >= self._slices(self.least):
                made += self._split(start, end)
        if learn:
            for start, end in made:
                heard.append(self._enter(image[start:end]))
        self.last = {"chain": [(s, e, u) for s, e, u, _ in chain], "slices": len(image)}
        return heard

    def _learn(self, unit: int, stretch: np.ndarray, places: np.ndarray) -> None:
        self.counts[unit] += 1
        share = max(1.0 / float(self.counts[unit]), self.rate)
        target = np.zeros((self.lengths[unit], self.width))
        laid = np.zeros(self.lengths[unit])
        np.add.at(target, places, stretch)
        np.add.at(laid, places, 1.0)
        on = laid > 0
        moved = self.fields[self.first[unit]: self.first[unit] + self.lengths[unit]].copy()
        moved[on] += share * (target[on] / laid[on, None] - moved[on])
        moved /= np.maximum(np.linalg.norm(moved, axis=1, keepdims=True), 1e-9)
        self.fields[self.first[unit]: self.first[unit] + self.lengths[unit]] = moved

    def _enter(self, stretch: np.ndarray) -> int:
        unit = self.units
        self.first = np.concatenate([self.first, [len(self.fields)]])
        self.lengths = np.concatenate([self.lengths, [len(stretch)]])
        self.counts = np.concatenate([self.counts, [1]])
        self.owner = np.concatenate([self.owner, np.full(len(stretch), unit)])
        self.fields = np.concatenate([self.fields, stretch])
        return unit

    def _split(self, start: int, end: int) -> list[tuple[int, int]]:
        parts = int(np.ceil((end - start) / self._slices(self.most)))
        edges = np.linspace(start, end, parts + 1).round().astype(int)
        return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]

    # -- the state

    def arrays(self) -> dict[str, np.ndarray]:
        return {"fields": self.fields.copy(), "lengths": self.lengths.copy(),
                "counts": self.counts.copy(), "sums": self.sums.copy(),
                "heard": np.array(self.heard)}

    @classmethod
    def from_arrays(cls, settings: dict[str, Any], arrays: Any) -> Ear:
        out = cls(**settings)
        out.lengths = np.asarray(arrays["lengths"], dtype=np.int64).copy()
        out.counts = np.asarray(arrays["counts"], dtype=np.int64).copy()
        out.fields = np.asarray(arrays["fields"], dtype=np.float64).reshape(-1, out.width).copy()
        out.first = np.concatenate([[0], np.cumsum(out.lengths)[:-1]]).astype(np.int64)
        out.owner = np.repeat(np.arange(len(out.lengths)), out.lengths)
        out.sums = np.asarray(arrays["sums"], dtype=np.float64).copy()
        out.heard = int(arrays["heard"])
        return out


class HeardLexicon(Lexicon):
    """A lexicon whose word neurons hold heard forms beside written ones: ``enter_heard(key)``
    gives a key (a ``"~"`` and a number) the next free word neuron with no letters, and the
    motor says written words only. A heard form's neuron has the threshold ``heard`` (a written
    word's unless given).

    A lesson moves the synapses of the words heard in it alone. A word neuron not heard rests
    a little below zero, and a lesson's contrast would move its synapses by that rest; a
    word's first lesson closes its whole error at a large rate, so the hundreds of heard forms
    a life enters would each nudge every other word's synapses, the written words' among
    them."""

    def __init__(self, concepts: dict[str, int], *, heard: float | None = None,
                 boost: float = 1.0, **settings: Any):
        super().__init__(concepts, **settings)
        self.written = np.zeros(self.capacity, dtype=bool)
        self.heard = self.threshold if heard is None else float(heard)
        self._moving: np.ndarray | None = None  # the synapses the lesson under way may move
        self.boost = float(boost)  # the lesson's rate, against the lexicon's own
        self.settings.update({"heard": self.heard, "boost": self.boost})

    def _enter(self, word: str) -> int:
        form = self._form(word)
        if form in self._units:
            return self._units[form]
        unit = super()._enter(word)
        self.written[unit] = True
        return unit

    def enter_heard(self, key: str) -> int:
        """A heard form's word neuron (entered the first time), with no letters."""
        if key in self._units:
            return self._units[key]
        if not key.startswith("~"):
            raise ValueError("a heard form's key starts with '~'")
        if len(self.forms) == self.capacity:
            raise ValueError(f"the lexicon holds {self.capacity} words: a larger capacity")
        unit = len(self.forms)
        self.forms.append(key)
        self._units[key] = unit
        if self.heard != self.threshold:
            self._bias[self.word[unit]] = -self.heard
            self.graph.bias = self._bias
        return unit

    def name(self, words: Any, concepts: Any, *, groups: Any = None) -> float:
        heard = [words] if isinstance(words, str) else list(words)
        units = {self._enter(w) if not str(w).startswith("~") else self.enter_heard(str(w))
                 for w in heard}
        moving = np.zeros(self.connectome.synapses, dtype=bool)
        for u in units:
            moving[self._touch[u]] = True
        self._moving = moving
        try:
            return float(super().name(words, concepts, groups=groups))
        finally:
            self._moving = None

    def _learner(self, groups: tuple[str, ...]) -> Any:
        return _HeardOnly(super()._learner(groups), self)

    def words_for(self, concept: Any) -> np.ndarray:
        """The written word neurons settled on a concept (a heard form reads as silent: the
        motor says letters)."""
        active = super().words_for(concept)
        return np.where(self.written[: len(active)], active, -np.inf)

    def arrays(self) -> dict[str, np.ndarray]:
        out = super().arrays()
        out["written"] = self.written[: len(self.forms)].copy()
        return out


class _HeardOnly:
    """A lexicon's learner as its lesson sees it: the rates the lesson gives reach the synapses
    of the words heard in it alone."""

    def __init__(self, learner: Any, lexicon: HeardLexicon):
        object.__setattr__(self, "_learner", learner)
        object.__setattr__(self, "_lexicon", lexicon)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._learner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        moving = self._lexicon._moving
        if name == "synapse_rate" and moving is not None:
            value = np.where(moving, value, 0.0)
        if name == "config" and moving is not None:
            value = dataclasses.replace(value, eta=value.eta * self._lexicon.boost)
        setattr(self._learner, name, value)


class Answered:
    """A part of a ``Words`` brain: a question heard and answered names the thing the teacher's
    answer points to, for the heard forms of the question. Each answer word names a group of
    the senses (its most active concept's group); the senses find the things it names there,
    and the heard forms are taught the mean of those things' concepts at the naming shares, as
    a naming teaches them. Nothing is taught where the answer names no concept above the
    brain's floor or the senses find nothing."""

    groups: dict[str, int] = {}

    def practice(self, brain: Any, words: list[str], answer: list[str], heard: Any) -> None:
        keys = [w for w in words if str(w).startswith("~")]
        if not keys:
            return
        lexicon, groups = brain.lexicon, brain.lexicon.groups
        places: list[np.ndarray] = []
        for word in answer:
            if lexicon.unit(word) is None:
                continue
            told = lexicon.hear(word)
            group = max(brain.senses.groups, key=lambda g: float(told[groups[g]].max()))
            if told[groups[group]].max() <= brain.floor:
                continue
            places += brain.senses.find(group, told[groups[group]])
        concepts = [c for c in (brain.attend(p) for p in places) if c is not None]
        if concepts:
            lexicon.name(keys, np.mean(concepts, axis=0), groups=tuple(brain.shares))


class Hearing:
    """A brain with ears: ``brain`` (a ``cadence.words.Words`` with no lesson yet) and an
    ``Ear`` (made here unless given). The brain's lexicon becomes a ``HeardLexicon`` over the
    same concepts and settings (a heard form's threshold ``heard``), and ``Answered`` joins
    its parts; see the module note."""

    def __init__(self, brain: Any, ear: Ear | None = None, *, answered: bool = True,
                 heard: float | None = None, boost: float = 1.0):
        if brain.lexicon.lessons:
            raise ValueError("a brain gains ears before its first lesson")
        told = dict(brain.lexicon.settings)
        brain.lexicon = HeardLexicon(told.pop("concepts"), heard=heard, boost=boost, **told)
        # a question already heard is a list of words (heard keys): the brain takes it as it is
        brain._words = _words
        if answered:
            brain.parts.append(Answered())
        self.brain = brain
        self.ear = ear if ear is not None else Ear()
        self.last: dict[str, Any] = {}

    @staticmethod
    def key(unit: int) -> str:
        return f"~{int(unit)}"

    def heard(self, samples: Any, *, learn: bool) -> list[str]:
        """The lexicon's keys of the units the ear heard (a unit new to the ear enters the
        lexicon), in the order heard."""
        units = self.ear.listen(samples, learn=learn)
        for u in units:
            self.brain.lexicon.enter_heard(self.key(u))
        keys = [self.key(u) for u in units]
        self.last = {"keys": keys}
        return keys

    def name(self, samples: Any, at: Any) -> float:
        """A naming by ear: the brain attends to the thing at ``at`` and hears the word said."""
        concepts = self.brain.attend(at)
        keys = self.heard(samples, learn=True)
        if not keys or concepts is None:
            return float("nan")
        return float(self.brain.lexicon.name(keys, concepts, groups=tuple(self.brain.shares)))

    def ask(self, samples: Any) -> list[str]:
        """The words the brain says, in writing, to a question asked aloud."""
        return list(self.brain.ask(cast(Any, self.heard(samples, learn=False))))

    def practice(self, samples: Any, answer: str | list[str]) -> None:
        """A question asked aloud with the teacher's written answer."""
        self.brain.practice(cast(Any, self.heard(samples, learn=True)), answer)

    def read(self, word: str, samples: Any) -> None:
        """A word read aloud: its written form seen while its sound is heard, nothing attended;
        each form is taught the concepts active at that moment (at each concept the more
        active of the two forms' settles)."""
        lexicon = self.brain.lexicon
        keys = self.heard(samples, learn=True)
        by_ear = lexicon.hear(keys) if keys else np.zeros(lexicon.concepts)
        by_eye = lexicon.hear(word) if lexicon.unit(word) is not None else np.zeros(
            lexicon.concepts)
        active = np.maximum(np.maximum(by_ear, by_eye), 0.0) / lexicon.level
        if active.max() <= self.brain.floor / lexicon.level:
            return
        groups = tuple(self.brain.shares)  # the senses' groups, as a naming teaches them
        lexicon.name(word, active, groups=groups)
        if keys:
            lexicon.name(keys, active, groups=groups)

    def parameters(self) -> int:
        """The learned parameters: the lexicon's (one efficacy a reciprocal pair) and the
        ear's fields."""
        return int(self.brain.lexicon.connectome.synapses // 2 + self.ear.parameters())


def _words(question: Any) -> list[str]:
    """A question's words: a list as it is (words already heard), a string split as
    ``Words`` splits it."""
    from .room import Words

    return list(question) if isinstance(question, list) else Words._words(question)
