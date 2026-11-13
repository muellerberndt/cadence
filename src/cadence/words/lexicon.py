"""A lexicon: words bound to the concepts a brain holds, learned from naming.

The lexicon is a graph of two populations and the synapses between them:

- ``word``, a neuron a word form. A form takes the next free neuron the first time it is heard:
  the word enters the lexicon as a thing enters the map, and the neuron is that form's from then
  on. A form never heard has no neuron.
- ``concept``, a neuron a concept the brain holds, in named groups (the map's kinds, a colour,
  a place around the body, a read of the map, the read that finds nothing).

Every word neuron and every concept neuron are joined by a reciprocal pair of synapses that
starts at zero. A naming lesson is the library's free and nudged phases with a quadratic nudge
(``cadence.learning.Learner``): the heard word's neuron is driven, the concept neurons settle,
and the nudge pulls them toward the concepts active at that moment, each at its share. A word's
``n``-th lesson closes ``1 / n`` of its error, and no less than ``rate``: its synapses hold the
mean of the concepts of its namings. A word heard with a concept at every naming holds it at its
share; a concept present at some of its namings is held at the share of namings it was present
at, so a word's meaning is what its namings share.

- **Hearing** drives the neurons of the heard words and the concept neurons settle: the
  concepts the words name. A word never heard drives nothing: the concepts settle as they do
  in silence.
  A word neuron has a threshold (``threshold``): it is active where it is heard or where a
  driven concept names it, and silent under the weak drive of concepts another word named.
- **Saying** drives a concept's neuron and the word neurons settle; the motor chooses the most
  active word and says its letters, one letter a part. The motor's letters are written when the
  form enters, from the letters it was heard by.

The synapses that join a word and a concept are one reciprocal pair, so the lessons of hearing
teach saying as well: no lesson teaches the way back.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz '"  # the motor's letters; one more value is the word's end


class Lexicon:
    """Words for ``concepts``, given as named groups and their sizes (``{"kind": 16,
    "colour": 6}``), in that order. ``capacity`` word neurons, ``rate`` the share of a lesson's
    error the lesson closes, ``steps`` the sweeps of a settle, ``length`` the motor's letters a
    word.

    ``name(words, concepts)`` is a naming lesson, ``hear(words)`` the concepts the words name,
    ``say(concept)`` the word the motor says for a concept. ``index(group, value)`` is a
    concept's neuron among the concepts."""

    def __init__(self, concepts: dict[str, int], *, capacity: int = 256, rate: float = 0.05,
                 steps: int = 12, length: int = 16, level: float = 0.5, drive: float = 2.0,
                 threshold: float = 0.15, backend: Any = "cpu"):
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        if not concepts or any(int(n) < 1 for n in concepts.values()):
            raise ValueError("concepts names each group with its size, 1 or more")
        if not 0.0 < float(rate) <= 1.0 or int(capacity) < 1 or int(steps) < 1:
            raise ValueError("a lexicon takes a rate in (0, 1], a capacity and steps of 1 or more")
        self.groups: dict[str, slice] = {}
        start = 0
        for name, size in concepts.items():
            self.groups[str(name)] = slice(start, start + int(size))
            start += int(size)
        self.concepts = start
        self.capacity, self.rate, self.steps = int(capacity), float(rate), int(steps)
        self.length, self.level, self.drive = int(length), float(level), float(drive)
        self.threshold = float(threshold)
        self.settings = {"concepts": {k: s.stop - s.start for k, s in self.groups.items()},
                         "capacity": self.capacity, "rate": self.rate, "steps": self.steps,
                         "length": self.length, "level": self.level, "drive": self.drive,
                         "threshold": self.threshold}
        self.word = np.arange(self.capacity)
        self.concept = self.capacity + np.arange(self.concepts)
        self.neurons = n = self.capacity + self.concepts
        pairs = np.stack(np.meshgrid(self.word, self.concept, indexing="ij"),
                         axis=-1).reshape(-1, 2)
        self.connectome = Connectome.from_synapses(
            n, label="lexicon", pre=np.concatenate([pairs[:, 0], pairs[:, 1]]),
            post=np.concatenate([pairs[:, 1], pairs[:, 0]]),
            populations={"word": self.word.tolist(), "concept": self.concept.tolist()})
        self._model = learning_neuron_model(dt=1.0)
        # a word neuron's threshold: a word is active where it is heard, or where a driven
        # concept names it; the weak drive a word gets from the concepts another word named
        # leaves it silent, so a settle does not spread from word to word
        self._bias = np.zeros(n)
        self._bias[self.word] = -self.threshold
        self.graph = Graph(self.connectome, self._model,
                           efficacy=np.zeros(self.connectome.synapses), bias=self._bias,
                           backend=backend)
        self._pre = np.asarray(self.connectome.pre)
        self._post = np.asarray(self.connectome.post)
        self._learners: dict[tuple[str, ...], Any] = {}
        self.counts: dict[tuple[str, ...], np.ndarray] = {}  # each word's lessons, by outputs
        # each word neuron's synapses, both directions, as numbers into the connectome
        owner = np.where(np.isin(self._pre, self.word), self._pre, self._post)
        order = np.argsort(owner, kind="stable")
        self._touch = np.split(order, np.searchsorted(owner[order], self.word[1:]))
        self.forms: list[str] = []  # the form of each word neuron that has entered
        self._units: dict[str, int] = {}
        self.letters = np.zeros((self.capacity, self.length, len(LETTERS) + 1))
        self.lessons = 0
        # settles run and their multiply-accumulates: a sweep reads every synapse once
        self.work = {"settles": 0, "macs": 0}

    # -- concepts and words

    def index(self, group: str, value: int) -> int:
        """A concept's neuron among the concepts: ``value`` of ``group``."""
        held = self.groups[group]
        if not 0 <= int(value) < held.stop - held.start:
            raise ValueError(f"no concept {value!r} in a group {group!r} of this lexicon")
        return int(held.start) + int(value)

    def unit(self, word: str) -> int | None:
        """The word neuron of a form, None for a form never heard."""
        return self._units.get(self._form(word))

    @staticmethod
    def _form(word: str) -> str:
        return " ".join(str(word).lower().split())

    def _enter(self, word: str) -> int:
        """A form heard for the first time takes the next free word neuron, and the motor's
        letters for it are written from the letters it was heard by."""
        form = self._form(word)
        if form in self._units:
            return self._units[form]
        if len(self.forms) == self.capacity:
            raise ValueError(f"the lexicon holds {self.capacity} words: a larger capacity")
        if not form or len(form) > self.length or any(c not in LETTERS for c in form):
            raise ValueError(f"a word is 1 to {self.length} of the letters {LETTERS!r}")
        unit = len(self.forms)
        self.forms.append(form)
        self._units[form] = unit
        for place in range(self.length):
            letter = LETTERS.index(form[place]) if place < len(form) else len(LETTERS)
            self.letters[unit, place, letter] = 1.0
        return unit

    def _drive(self, rows: list[list[str]], *, enter: bool) -> np.ndarray:
        """The drive of heard words, ``(rows, neurons)``: each row's words' neurons."""
        out = np.zeros((len(rows), self.neurons))
        for row, words in enumerate(rows):
            for word in words:
                unit = self._enter(word) if enter else self.unit(word)
                if unit is not None:
                    out[row, self.word[unit]] = self.drive
        return out

    # -- lessons

    def _learner(self, groups: tuple[str, ...]) -> Any:
        """The learner whose outputs are ``groups``' concept neurons and whose synapses are the
        pairs between the words and those concepts."""
        from ..learning import Learner, LearnerConfig

        if groups not in self._learners:
            outputs = np.concatenate([self.concept[self.groups[g]] for g in groups])
            plastic = np.isin(self._pre, outputs) | np.isin(self._post, outputs)
            config = LearnerConfig(nudge="quadratic", beta=0.1, eta=1.0, eta_bias=0.0,
                                   free_steps=self.steps, nudged_steps=self.steps,
                                   tolerance=None)
            options: dict[str, Any] = {
                "config": config, "reciprocal": True, "plastic_synapses": plastic,
                "plastic_neurons": np.zeros(self.neurons, dtype=bool)}
            if self._learners:
                first = next(iter(self._learners.values()))
                learner = Learner.sharing(first, list(outputs), **options)
            else:
                learner = Learner(self.graph, list(outputs), **options)
            self._learners[groups] = learner
        return self._learners[groups]

    def name(self, words: Any, concepts: Any, *, groups: Any = None) -> float:
        """A naming lesson: ``words`` heard (one form, or the forms heard together) while
        ``concepts`` are active: an array over the concepts, or ``{(group, value): share}``.
        The lesson teaches the concept neurons of ``groups`` (every group unless given) and
        leaves the others' synapses as they are. A form heard for the first time enters. Gives
        the error before the lesson, as a share of the target's sum."""
        rows = [[words] if isinstance(words, str) else list(words)]
        target_shares = self._concepts(concepts)
        groups = tuple(self.groups) if groups is None else tuple(groups)
        learner = self._learner(groups)
        drive = self._drive(rows, enter=True)
        lit = np.tanh(drive[:, self.word] / 2.0)
        shared = float((lit.sum(axis=0) ** 2).sum()) / len(drive)
        # each heard word's share of its error: 1 / n at its n-th lesson, no less than rate
        counts = self.counts.setdefault(groups, np.zeros(self.capacity, dtype=np.int64))
        heard = sorted({int(u) for u in np.flatnonzero(drive[0, self.word])})
        shares = {u: max(1.0 / (counts[u] + 1), self.rate) for u in heard}
        top = max(shares.values()) if shares else self.rate
        steps = np.ones(self.connectome.synapses)
        for u, share in shares.items():
            steps[self._touch[u]] = share / top
            counts[u] += 1
        learner.synapse_rate = steps
        learner.config = dataclasses.replace(
            learner.config, eta=top / (0.25 * max(shared, 1e-12)))
        target = np.zeros_like(drive)
        target[0, self.concept] = self.level * target_shares
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        learner.update(free, plus, minus)
        self.graph = learner.brain
        self.lessons += 1
        self._count(3)  # the free phase and the two nudged ones
        outputs = np.concatenate([self.groups[g].start + np.arange(
            self.groups[g].stop - self.groups[g].start) for g in groups])
        answer = np.asarray(free.activation)[0, self.concept[outputs]]
        wanted = target[0, self.concept[outputs]]
        return float(np.abs(answer - wanted).sum() / max(np.abs(wanted).sum(), 1e-12))

    def _concepts(self, concepts: Any) -> np.ndarray:
        if isinstance(concepts, dict):
            out = np.zeros(self.concepts)
            for (group, value), share in concepts.items():
                out[self.index(group, value)] += float(share)
            return out
        out = np.asarray(concepts, dtype=np.float64).reshape(self.concepts)
        return out

    # -- hearing and saying, by settling

    def hear(self, words: Any) -> np.ndarray:
        """The concepts heard words name: the words' neurons are driven and the concept neurons
        settle, ``(concepts,)``. Words never heard drive nothing."""
        rows = [[words] if isinstance(words, str) else list(words)]
        state = self.graph.settle_batch(self._drive(rows, enter=False), steps=self.steps,
                                        tolerance=None)
        self._count(1)
        return np.asarray(np.asarray(state.activation)[0, self.concept], dtype=np.float64)

    def words_for(self, concept: Any) -> np.ndarray:
        """The word neurons settled on a concept: its neuron (a number, a ``(group, value)``, or
        an array over the concepts) is driven, ``(words entered,)``."""
        if isinstance(concept, tuple):
            concept = self.index(*concept)
        drive = np.zeros((1, self.neurons))
        if isinstance(concept, (int, np.integer)):
            drive[0, self.concept[int(concept)]] = self.drive
        else:
            drive[0, self.concept] = self.drive * np.asarray(concept, dtype=np.float64)
        state = self.graph.settle_batch(drive, steps=self.steps, tolerance=None)
        self._count(1)
        return np.asarray(np.asarray(state.activation)[0, self.word[: len(self.forms)]],
                          dtype=np.float64)

    def _count(self, settles: int) -> None:
        self.work["settles"] += settles
        self.work["macs"] += settles * self.steps * int(self.connectome.synapses)

    def say(self, concept: Any, floor: float = 1e-3) -> str | None:
        """The word the motor says for a concept: the motor chooses the most active word neuron
        of the settle and says its letters, one a part; None where no word neuron is above
        ``floor``."""
        active = self.words_for(concept)
        if not len(active) or active.max() <= floor:
            return None
        letters = self.letters[int(np.argmax(active))].argmax(axis=1)
        return "".join(LETTERS[c] for c in letters if c < len(LETTERS))

    # -- the state

    def arrays(self) -> dict[str, np.ndarray]:
        """What a file keeps of a lexicon: its synapses, the motor's letters and the forms."""
        return {"efficacy": np.asarray(self.graph.efficacy, dtype=np.float64).copy(),
                "letters": self.letters[: len(self.forms)].copy(),
                "forms": np.array(self.forms, dtype=str)}

    @classmethod
    def from_arrays(cls, settings: dict[str, Any], arrays: Any, backend: Any = "cpu") -> Lexicon:
        """A lexicon from its ``settings`` and ``arrays``."""
        from ..brain import Brain as Graph

        told = dict(settings)
        out = cls(told.pop("concepts"), backend=backend, **told)
        for form in [str(f) for f in np.asarray(arrays["forms"])]:
            out._enter(form)
        efficacy = np.asarray(arrays["efficacy"], dtype=np.float64)
        if efficacy.shape != (out.connectome.synapses,):
            raise ValueError("the lexicon's efficacy does not fit its words and concepts")
        out.graph = Graph(out.connectome, out._model, efficacy=efficacy.copy(), bias=out._bias,
                          backend=backend)
        return out
