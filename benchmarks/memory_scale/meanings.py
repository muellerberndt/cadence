"""Records keyed by held meanings (cadence#288, the second round): facts whose keys and values
``cadence.Message`` makes, in place of the random and sparse stand-ins of ``world.py``.

**The lexicon.** The grammar chamber's concept moments at a larger lexicon: ``nouns`` noun stems
and ``verbs`` verbs, each with a concept coordinate and a word coordinate, and the flags
``PLURAL``, ``AGENT`` and ``PATIENT``. A moment is one observation, as in the chamber: an agent
is its stem's concept with ``AGENT`` (and ``PLURAL``), a patient the same with ``PATIENT``, an
act its verb's concept; a heard word is its word coordinate alone.

**A fact** says of an agent and a verb which patient: *the dogs chase the owl*. The key says
what the fact is about and the value what it says, in the record format the meaning lane gives
(cadence#286):

* the **key** is ``Message.key()`` after the act and the agent are heard, the union of their
  codes, ``units`` zeros and ones;
* the **value** is ``Message.state`` after the patient is heard too, ``phases * units`` zeros
  and ones: the encoded meaning, with the patient's code in the third block.

A revised fact keeps its key and takes another patient. Two facts about one agent, or with one
verb, share that moment's code, so every record has siblings; nothing is added to make families.

**The code.** ``Message`` at the meaning lane's declared settings: 4 phases, 5 winners and
``units`` units (280 declared, 70 beside it). Its fan-in of 16 is 16 of the chamber's 70
channels; a lexicon of more inputs takes the same share of its channels (``SHARE``), and the
class's own 16 is an arm (``fan_in=16``).

**Cues.** ``exact``: the fact's key. ``noisy``: the key with three in ten of its active units
exchanged for silent ones (cosine .7). ``partial``: the key with half of its coordinates silent.
``reworded``: the same act and agent heard as their words, the form whose code the concept's
does not share. For several heads, ``agent``: the agent's moment alone, relevant to every
stored fact about that agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NOUNS, VERBS = 512, 128
PHASES, WINNERS = 4, 5
SHARE = 16 / 70  # the fan-in of the declared code over the chamber's channels
NOISE = 0.3  # the share of a noisy cue's active units exchanged for silent ones
CUES = ("exact", "noisy", "partial", "reworded")


@dataclass(frozen=True)
class Lexicon:
    """The coordinates of a lexicon's observations: a word per noun form and per verb, then a
    concept per stem and per verb, then the flags."""

    nouns: int = NOUNS
    verbs: int = VERBS

    @property
    def forms(self) -> int:
        """Noun forms, a stem in the singular and in the plural: the agents and the patients."""
        return 2 * self.nouns

    @property
    def abouts(self) -> int:
        """What a fact can be about: every agent with every verb."""
        return self.forms * self.verbs

    @property
    def inputs(self) -> int:
        return self.forms + self.verbs + self.nouns + self.verbs + 3

    def _flag(self, name: str) -> int:
        return self.inputs - 3 + ("PLURAL", "AGENT", "PATIENT").index(name)

    def noun(self, form: np.ndarray, role: str) -> np.ndarray:
        """Concept moments of noun forms in a role, ``(rows, inputs)``: form ``f`` is stem
        ``f mod nouns``, plural from ``nouns`` on."""
        form = np.asarray(form, dtype=np.int64)
        x = np.zeros((len(form), self.inputs))
        rows = np.arange(len(form))
        x[rows, self.forms + self.verbs + form % self.nouns] = 1.0
        x[rows, self._flag("PLURAL")] = (form >= self.nouns).astype(float)
        x[rows, self._flag(role)] = 1.0
        return x

    def act(self, verb: np.ndarray) -> np.ndarray:
        """Concept moments of verbs."""
        verb = np.asarray(verb, dtype=np.int64)
        x = np.zeros((len(verb), self.inputs))
        x[np.arange(len(verb)), self.forms + self.verbs + self.nouns + verb] = 1.0
        return x

    def noun_word(self, form: np.ndarray) -> np.ndarray:
        """Noun forms as heard words."""
        form = np.asarray(form, dtype=np.int64)
        x = np.zeros((len(form), self.inputs))
        x[np.arange(len(form)), form] = 1.0
        return x

    def verb_word(self, verb: np.ndarray) -> np.ndarray:
        """Verbs as heard words."""
        verb = np.asarray(verb, dtype=np.int64)
        x = np.zeros((len(verb), self.inputs))
        x[np.arange(len(verb)), self.forms + verb] = 1.0
        return x


def fan_in(lexicon: Lexicon) -> int:
    """The declared code's fan-in at this lexicon: the chamber's share of the channels."""
    return max(16, round(SHARE * 2 * lexicon.inputs))


def message(lexicon: Lexicon, units: int, seed: int, fan: int | None = None) -> Any:
    """``cadence.Message`` over the lexicon's observations at the declared settings."""
    import cadence as cd

    return cd.Message(lexicon.inputs, phases=PHASES, units=int(units), winners=WINNERS,
                      fan_in=fan_in(lexicon) if fan is None else int(fan), seed=int(seed))


def held(coder: Any, moments: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The key and the state of episodes heard moment by moment through a ``Message`` of
    ``coder``'s settings: ``moments[t]`` is ``(rows, inputs)``, row ``i`` episode ``i``'s
    ``t``-th moment."""
    import cadence as cd

    hearer = cd.Message(**coder.to_dict())
    hearer.reset(len(moments[0]))
    for x in moments:
        hearer.hear(x)
    return hearer.key(), hearer.state.copy()


class Coder:
    """The codes of a lexicon's moments under one ``Message``, and the keys and values of facts.

    A ``Message`` codes a moment by itself and joins codes by their union, so a fact's key and
    state follow from its moments' codes; ``held`` hears the same moments through the class,
    and the two agree (a contract test)."""

    def __init__(self, lexicon: Lexicon, units: int, seed: int, fan: int | None = None) -> None:
        self.lexicon, self.units = lexicon, int(units)
        self.message = message(lexicon, units, seed, fan)
        forms, verbs = np.arange(lexicon.forms), np.arange(lexicon.verbs)
        self.agent = self._code(lexicon.noun(forms, "AGENT"))
        self.patient = self._code(lexicon.noun(forms, "PATIENT"))
        self.acted = self._code(lexicon.act(verbs))
        self.noun_word = self._code(lexicon.noun_word(forms))
        self.verb_word = self._code(lexicon.verb_word(verbs))

    def _code(self, moments: np.ndarray, rows: int = 64) -> np.ndarray:
        parts = [self.message.code(moments[i : i + rows]) for i in range(0, len(moments), rows)]
        return np.concatenate(parts).astype(np.float32)

    def key(self, agent: np.ndarray, verb: np.ndarray) -> np.ndarray:
        """What facts are about, ``(rows, units)``: the union of the act's and the agent's
        codes."""
        return np.maximum(self.acted[verb], self.agent[agent]).astype(np.float64)

    def state(self, agent: np.ndarray, verb: np.ndarray, patient: np.ndarray) -> np.ndarray:
        """Whole facts as held messages, ``(rows, phases * units)``: the act, the agent and the
        patient in the first three blocks."""
        blocks = [self.acted[verb], self.agent[agent], self.patient[patient],
                  np.zeros((len(agent), self.units), dtype=np.float32)]
        return np.concatenate(blocks, axis=1).astype(np.float64)

    def about_agent(self, agent: np.ndarray) -> np.ndarray:
        """The agent's moment alone as a cue."""
        return self.agent[agent].astype(np.float64)

    def reworded(self, agent: np.ndarray, verb: np.ndarray) -> np.ndarray:
        """The act and the agent heard as their words."""
        return np.maximum(self.verb_word[verb], self.noun_word[agent]).astype(np.float64)

    def said(self, value: np.ndarray) -> np.ndarray:
        """The patient a value holds, per row: the form whose code its third block matches
        best (-1 where the block is empty)."""
        block = np.asarray(value)[:, 2 * self.units : 3 * self.units]
        best = np.argmax(block @ self.patient.T.astype(np.float64), axis=1)
        return np.where(block.any(axis=1), best, -1)


@dataclass
class Facts:
    agent: np.ndarray  # (n,), each fact's agent form
    verb: np.ndarray  # (n,)
    patient: np.ndarray  # (n,), the patient form the fact says
    key: np.ndarray  # (n, units)
    value: np.ndarray  # (n, phases * units)

    @property
    def n(self) -> int:
        return int(len(self.agent))


def facts(rng: np.random.Generator, coder: Coder, n: int) -> Facts:
    """``n`` facts about different agent-verb pairs, drawn without replacement, each with a
    patient drawn at random."""
    lexicon = coder.lexicon
    if n > lexicon.abouts:
        raise ValueError(f"the lexicon holds {lexicon.abouts} agent-verb pairs")
    about = rng.choice(lexicon.abouts, size=n, replace=False)
    agent, verb = about // lexicon.verbs, about % lexicon.verbs
    patient = rng.integers(lexicon.forms, size=n)
    return Facts(agent, verb, patient, coder.key(agent, verb), coder.state(agent, verb, patient))


def cue(rng: np.random.Generator, coder: Coder, found: Facts, record: int, kind: str) -> np.ndarray:
    """One cue of ``kind`` for the fact ``record``."""
    key = found.key[record]
    if kind == "exact":
        return key.copy()
    if kind == "noisy":
        active, silent = np.flatnonzero(key), np.flatnonzero(key == 0)
        swap = max(int(round(NOISE * len(active))), 1)
        out = key.copy()
        out[rng.choice(active, size=swap, replace=False)] = 0.0
        out[rng.choice(silent, size=swap, replace=False)] = 1.0
        return out
    if kind == "partial":
        out = key.copy()
        out[rng.permutation(len(key))[: len(key) // 2]] = 0.0
        return out
    if kind == "reworded":
        at = np.array([record])
        return coder.reworded(found.agent[at], found.verb[at])[0]
    raise ValueError(f"a cue is one of {CUES}")
