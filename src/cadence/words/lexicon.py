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
  active word and says its letters, one letter a part. Before it says a word the brain hears
  it: a word whose hearing settles on another concept of the groups said gives way to the next
  most active. The motor's letters are written when the
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
                 threshold: float = 0.15, keep: float = 0.5, backend: Any = "cpu"):
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
        self.keep = float(keep)  # the share of a mark a moment leaves
        self.settings = {"concepts": {k: s.stop - s.start for k, s in self.groups.items()},
                         "capacity": self.capacity, "rate": self.rate, "steps": self.steps,
                         "length": self.length, "level": self.level, "drive": self.drive,
                         "threshold": self.threshold, "keep": self.keep}
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
        self.marks: dict[tuple[str, ...], np.ndarray] = {}  # eligibility, by outputs (``mark``)
        self.marked: dict[tuple[str, ...], dict[int, float]] = {}  # each marked word's strength
        self.heard_last: int | None = None  # the word neuron a mark heard last
        # each word neuron's synapses, both directions, as numbers into the connectome
        owner = np.where(np.isin(self._pre, self.word), self._pre, self._post)
        order = np.argsort(owner, kind="stable")
        self._touch = np.split(order, np.searchsorted(owner[order], self.word[1:]))
        self.forms: list[str] = []  # the form of each word neuron that has entered
        self._units: dict[str, int] = {}
        self.letters = np.zeros((self.capacity, self.length, len(LETTERS) + 1))
        self.written = np.zeros(self.capacity, dtype=bool)  # a form with letters the motor says
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

    def enter(self, key: str, *, written: bool = True) -> int:
        """A form heard for the first time takes the next free word neuron. A written form
        (``written``) is 1 to ``length`` of the letters, and the motor's letters for it are
        written from the letters it was heard by. A form heard with no letters (the sound of a
        word, keyed by a character outside the letters) takes a neuron the motor never says.
        Gives the form's neuron."""
        form = self._form(key)
        if form in self._units:
            return self._units[form]
        if len(self.forms) == self.capacity:
            raise ValueError(f"the lexicon holds {self.capacity} words: a larger capacity")
        if written and (not form or len(form) > self.length or any(
                c not in LETTERS for c in form)):
            raise ValueError(f"a word is 1 to {self.length} of the letters {LETTERS!r}")
        if not written and (not form or all(c in LETTERS for c in form)):
            raise ValueError("a form with no letters is keyed by a character outside the "
                             "letters")
        unit = len(self.forms)
        self.forms.append(form)
        self._units[form] = unit
        self.written[unit] = bool(written)
        if written:
            for place in range(self.length):
                letter = LETTERS.index(form[place]) if place < len(form) else len(LETTERS)
                self.letters[unit, place, letter] = 1.0
        return unit

    def _enter(self, word: str) -> int:
        return self.enter(word)

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

    def name(self, words: Any, concepts: Any, *, groups: Any = None,
             weight: float = 1.0) -> float:
        """A naming lesson: ``words`` heard (one form, or the forms heard together) while
        ``concepts`` are active: an array over the concepts, or ``{(group, value): share}``.
        The lesson teaches the concept neurons of ``groups`` (every group unless given) and
        leaves the others' synapses as they are. ``weight`` (0 to 1) scales the share of its
        error the lesson closes: a lesson of reward takes the reward's surprise. A form heard
        for the first time enters. Gives the error before the lesson, as a share of the
        target's sum."""
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
        weight = min(max(float(weight), 0.0), 1.0)
        shares = {u: weight * max(1.0 / (counts[u] + 1), self.rate) for u in heard}
        top = max(shares.values()) if shares else self.rate
        if top <= 0.0:
            return float("nan")  # a lesson of no weight teaches nothing
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

    # -- three factors: a mark, a pistachio, a link

    def mark(self, words: Any, concepts: Any, *, groups: Any = None) -> None:
        """Hearing ``words`` while ``concepts`` are active leaves a mark on their links: the
        step a naming lesson would take (the library's free and nudged phases and their
        contrast, ``Learner.contrast``), held as an eligibility trace and not applied; a word's
        new mark replaces its old one. A pistachio (``confirm``) makes the marks lasting links, in
        both directions; with none they fade (``fade``). A word's share is 1 / n at its n-th
        link, no less than ``rate``."""
        rows = [[words] if isinstance(words, str) else list(words)]
        target_shares = self._concepts(concepts)
        groups = tuple(self.groups) if groups is None else tuple(groups)
        learner = self._learner(groups)
        drive = self._drive(rows, enter=True)
        lit = np.tanh(drive[:, self.word] / 2.0)
        shared = float((lit.sum(axis=0) ** 2).sum()) / len(drive)
        links = self.counts.setdefault(groups, np.zeros(self.capacity))
        heard = sorted({int(u) for u in np.flatnonzero(drive[0, self.word])})
        if not heard:
            return
        shares = {u: max(1.0 / (links[u] + 1), self.rate) for u in heard}
        top = max(shares.values())
        steps = np.zeros(self.connectome.synapses)
        for u, share in shares.items():
            steps[self._touch[u]] = share / top
        target = np.zeros_like(drive)
        target[0, self.concept] = self.level * target_shares
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        edges, _ = learner.contrast(free, plus, minus)
        eta = top / (0.25 * max(shared, 1e-12))
        held = self.marks.setdefault(groups, np.zeros(self.connectome.synapses))
        step = eta * np.asarray(edges, dtype=np.float64) * steps
        marked = self.marked.setdefault(groups, {})
        for u in heard:  # a word's mark is its latest co-activity: a new mark replaces its old
            held[self._touch[u]] = step[self._touch[u]]
            marked[u] = 1.0
        self.heard_last = heard[-1]
        self._count(3)

    def confirm(self, amount: float = 1.0) -> int:
        """A pistachio: the marks of this moment, times ``amount``, become lasting links
        (``Learner.apply``: the pair of synapses moves as one, in both directions). Gives the
        words whose links it made; the marks are spent."""
        made = 0
        for groups, held in list(self.marks.items()):
            learner = self._learner(groups)
            learner.synapse_rate = None
            learner.apply(float(amount) * held, np.zeros(self.neurons))
            self.graph = learner.brain
            links = self.counts.setdefault(groups, np.zeros(self.capacity))
            for u, strength in self.marked.get(groups, {}).items():
                links[u] += float(amount) * strength  # a faded mark makes a weaker link
                made += 1
        if made:
            self.lessons += 1
        self.marks.clear()
        self.marked.clear()
        return made

    def fade(self) -> None:
        """One moment passes: every mark keeps ``keep`` of itself; a faded mark is gone."""
        for groups in list(self.marks):
            self.marks[groups] *= self.keep
            marked = self.marked.get(groups, {})
            for u in list(marked):
                marked[u] *= self.keep
                if marked[u] < 1e-3:
                    self.marks[groups][self._touch[u]] = 0.0
                    del marked[u]
            if not marked:
                del self.marks[groups]
                self.marked.pop(groups, None)

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

    def choose(self, concept: Any, floor: float = 1e-3, tries: int = 3, *,
               among: str = "written") -> int | None:
        """The word neuron the brain chooses to say for a concept: the most active word
        neuron of the settle among ``among`` (``"written"``: forms with letters, said by the
        letters motor; ``"heard"``: forms heard as sound, said by replaying them; ``"all"``);
        None where none is above ``floor``. The brain hears a word it is about to say: a word
        whose hearing settles on another concept of the groups said gives way to the next most
        active, up to ``tries`` words."""
        if among not in ("written", "heard", "all"):
            raise ValueError("among is 'written', 'heard' or 'all'")
        active = self.words_for(concept)
        keep = {"written": self.written[: len(active)], "heard": ~self.written[: len(active)],
                "all": np.ones(len(active), dtype=bool)}[among]
        active = np.where(keep, active, -np.inf)
        if not len(active) or active.max() <= floor:
            return None
        vector = self._vector(concept)
        chosen = int(np.argmax(active))
        if vector.max() > 0:
            meant = int(np.argmax(vector))
            # the groups the concept drives: a word is heard back among them
            groups = np.concatenate([np.arange(g.start, g.stop) for g in self.groups.values()
                                     if vector[g].max() > 0])
            for unit in np.argsort(-active)[: max(int(tries), 1)]:
                if active[unit] <= floor:
                    break
                heard = self.hear(self.forms[int(unit)])
                if int(groups[np.argmax(heard[groups])]) == meant:
                    chosen = int(unit)
                    break
        return chosen

    def say(self, concept: Any, floor: float = 1e-3, tries: int = 3) -> str | None:
        """The word the motor says for a concept: the written word neuron ``choose`` gives,
        its letters said one a part; None where no written word neuron is above ``floor``."""
        chosen = self.choose(concept, floor, tries, among="written")
        if chosen is None:
            return None
        letters = self.letters[chosen].argmax(axis=1)
        return "".join(LETTERS[c] for c in letters if c < len(LETTERS))

    def _vector(self, concept: Any) -> np.ndarray:
        """A concept to say as a vector over the concepts."""
        if isinstance(concept, tuple):
            concept = self.index(*concept)
        if isinstance(concept, (int, np.integer)):
            return np.asarray(np.eye(self.concepts)[int(concept)])
        return np.asarray(concept, dtype=np.float64).reshape(self.concepts)

    # -- the state

    def arrays(self) -> dict[str, np.ndarray]:
        """What a file keeps of a lexicon: its synapses, the motor's letters, the forms, and
        each word's lessons (a reloaded word goes on at its next lesson's share)."""
        keys = sorted(self.counts)
        return {"efficacy": np.asarray(self.graph.efficacy, dtype=np.float64).copy(),
                "letters": self.letters[: len(self.forms)].copy(),
                "forms": np.array(self.forms, dtype=str),
                "written": self.written[: len(self.forms)].copy(),
                # each word's lessons by the groups they taught, and the lessons in all
                "counted": np.array([",".join(k) for k in keys], dtype=str),
                "counts": (np.stack([self.counts[k] for k in keys]) if keys
                           else np.zeros((0, self.capacity), dtype=np.int64)),
                "lessons": np.array(self.lessons, dtype=np.int64)}

    @classmethod
    def from_arrays(cls, settings: dict[str, Any], arrays: Any, backend: Any = "cpu") -> Lexicon:
        """A lexicon from its ``settings`` and ``arrays``."""
        from ..brain import Brain as Graph

        told = dict(settings)
        out = cls(told.pop("concepts"), backend=backend, **told)
        forms = [str(f) for f in np.asarray(arrays["forms"])]
        written = (np.asarray(arrays["written"], dtype=bool) if "written" in arrays
                   else np.ones(len(forms), dtype=bool))
        for form, has_letters in zip(forms, written, strict=True):
            out.enter(form, written=bool(has_letters))
        efficacy = np.asarray(arrays["efficacy"], dtype=np.float64)
        if efficacy.shape != (out.connectome.synapses,):
            raise ValueError("the lexicon's efficacy does not fit its words and concepts")
        out.graph = Graph(out.connectome, out._model, efficacy=efficacy.copy(), bias=out._bias,
                          backend=backend)
        if "counts" in arrays:  # a file without them restarts every word at its first lesson
            counted = [str(k) for k in np.asarray(arrays["counted"])]
            for key, row in zip(counted, np.asarray(arrays["counts"]), strict=True):
                out.counts[tuple(key.split(","))] = np.asarray(row, dtype=np.int64).copy()
            out.lessons = int(np.asarray(arrays["lessons"]))
        return out
