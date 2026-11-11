"""Conversations heard as words (cadence#305): a synthetic world the instrument renders, and
adapters for bAbI question answering and bAbI dialog.

A conversation is a list of sessions, a session a list of turns. A turn is told (a statement) or
asked (a question and its answer). The brain hears a turn one word a moment, each word one input
channel of the lexicon. The world marks a turn with the input channel ``TURN``: a told turn
before the word it says (the value), an asked turn after the question's last word, followed by a
silent moment at which the answer is said. The answer is one word, said whole as one action.

**The synthetic world.** ``people`` names and ``places`` places. A fact puts a person in a place,
told in a template of ``TELL``; a later fact about the same person moves them (a revision). A
question asks where a person is, in a template of ``ASK`` (the wordings lessons use) or of
``ASK_HELD`` (wordings no lesson uses, made of words the lessons use), or in the fact's own words
up to its turn (``prefix``: the question hears what its fact did, the records' exact key). The
templates are frozen here, before any life. A conversation tells ``facts`` facts a session and
asks about a fact told earlier in the same session and about one told ``gap`` sessions earlier.
A share of the person and place pairs (``HELD_PAIRS``) appears in no lesson.

**bAbI question answering** (Weston et al. 2016, CC BY 3.0): a story is one session; its
statements are told, the world marking the turn before the statement's last word, and its
questions asked. **bAbI dialog** (Bordes, Boureau and Weston 2017, CC BY 3.0): ``dialog_turns``
reads a dialog's user turns, knowledge-base lines and bot turns; ``dialog_answers`` reads which
bot turns a closed set of answers learned from the training dialogs can hold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TURN = "<turn>"
PEOPLE = ("mary", "john", "sandra", "daniel", "fred", "bill", "julie", "emily", "jeff", "lily",
          "greg", "yann", "anna", "paul", "nora", "omar")
PLACES = ("kitchen", "garden", "office", "hallway", "bathroom", "bedroom", "cellar", "garage")
TELL = ("{p} went to the {l} .", "{p} is in the {l} .")  # four words before the place
ASK = ("where is {p} ?", "where did {p} go ?")
ASK_HELD = ("{p} is where ?", "where did {p} go to ?")  # words the lessons use, new orders
HELD_PAIRS = 0.25  # of the person and place pairs, told and asked in no lesson


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_']+|[?.,]", text.lower())


@dataclass
class Turn:
    """One turn: ``said`` the words heard; ``answer`` the word said at the silent moment (None
    for a told turn); ``turn_at`` the word the world marks the turn before (a told turn)."""

    said: list[str]
    answer: str | None = None
    turn_at: int | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def asked(self) -> bool:
        return self.answer is not None


Session = list[Turn]
Conversation = list[Session]


class Lexicon:
    """Words to input channels (one channel a word, and ``TURN``) and answers to actions."""

    def __init__(self, vocabulary: list[str], answers: list[str]) -> None:
        self.words = sorted(set(vocabulary) - {TURN}) + [TURN]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.answers = sorted(set(answers))
        self.action = {a: i for i, a in enumerate(self.answers)}

    @property
    def inputs(self) -> int:
        return len(self.words)

    @property
    def actions(self) -> int:
        return len(self.answers)

    def moment(self, word: str | None) -> np.ndarray:
        x = np.zeros(self.inputs)
        if word is not None:
            x[self.index[word]] = 1.0
        return x

    def moments(self, turn: Turn) -> list[tuple[np.ndarray, int | None]]:
        """The moments of a turn and the action taught or scored at each (None where none)."""
        out: list[tuple[np.ndarray, int | None]] = []
        for i, w in enumerate(turn.said):
            if turn.turn_at == i and not turn.asked:
                out.append((self.moment(TURN), None))
            out.append((self.moment(w), None))
        if turn.asked:
            out.append((self.moment(TURN), None))
            out.append((self.moment(None), self.action[str(turn.answer)]))
        return out


# --------------------------------------------------------------------------- the synthetic world


class World:
    """The synthetic conversation world of one founder: its held-out pairs and its draws."""

    def __init__(self, founder: int, people: int = 12, places: int = 8) -> None:
        self.people, self.places = PEOPLE[:people], PLACES[:places]
        rng = np.random.default_rng(np.random.SeedSequence([int(founder), 305, 1]))
        pairs = [(p, q) for p in self.people for q in self.places]
        held = rng.choice(len(pairs), size=int(round(HELD_PAIRS * len(pairs))), replace=False)
        self.held = {pairs[int(i)] for i in held}
        vocabulary = [w for t in (*TELL, *ASK, *ASK_HELD)
                      for w in words(t.format(p="x", l="x")) if w != "x"]
        self.lexicon = Lexicon([*vocabulary, *self.people, *self.places], list(self.places))

    def tell(self, person: str, place: str, template: int) -> Turn:
        said = words(TELL[template].format(p=person, l=place))
        return Turn(said, None, said.index(place), {"about": person, "place": place})

    def ask(self, person: str, place: str, wording: str, template: int, told: Turn) -> Turn:
        if wording == "prefix":
            said = told.said[: told.turn_at]
        else:
            said = words((ASK if wording == "trained" else ASK_HELD)[template].format(p=person))
        return Turn(said, place, None, {"about": person, "wording": wording})

    def conversations(self, rng: np.random.Generator, streams: int, sessions: int, facts: int,
                      lesson: bool, wording: str = "trained") -> list[Conversation]:
        """``streams`` conversations of one shape (their turns align word for word), each of
        ``sessions`` sessions of ``facts`` facts and two questions: about a fact of the same
        session and about the latest fact of a person told in an earlier session. In a lesson's
        conversations no held-out pair is told; in a reading's every pair is."""
        shape_tell = rng.integers(len(TELL), size=(sessions, facts))
        shape_ask = rng.integers(len(ASK), size=(sessions, 2))
        out: list[Conversation] = []
        for _ in range(streams):
            where: dict[str, tuple[str, int, Turn]] = {}  # person -> place, session, told turn
            before: dict[str, str] = {}  # person -> the place a revision replaced
            conversation: Conversation = []
            for s in range(sessions):
                session: Session = []
                told_here = []
                order = rng.permutation(self.people)[:facts]
                for k, person in enumerate(order):
                    person = str(person)
                    choices = [q for q in self.places
                               if not (lesson and (person, q) in self.held)
                               and q != where.get(person, ("", 0, None))[0]]
                    place = str(choices[int(rng.integers(len(choices)))])
                    if person in where:
                        before[person] = where[person][0]
                    turn = self.tell(person, place, int(shape_tell[s, k]))
                    turn.tags["revised"] = person in before
                    where[person] = (place, s, turn)
                    session.append(turn)
                    told_here.append(person)
                first = told_here[int(rng.integers(len(told_here)))]
                place, _, turn = where[first]
                ask = self.ask(first, place, wording, int(shape_ask[s, 0]), turn)
                ask.tags.update(gap=0, revised=first in before, stale=before.get(first),
                                held=(first, place) in self.held)
                session.append(ask)
                earlier = [p for p, (_, at, _) in where.items() if at < s]
                if s:  # every stream asks a second question from the second session on
                    person = (earlier[int(rng.integers(len(earlier)))] if earlier
                              else told_here[int(rng.integers(len(told_here)))])
                    place, at, turn = where[person]
                    ask = self.ask(person, place, wording, int(shape_ask[s, 1]), turn)
                    ask.tags.update(gap=s - at, revised=person in before,
                                    stale=before.get(person), held=(person, place) in self.held)
                    session.append(ask)
                conversation.append(session)
            out.append(conversation)
        return out


# ------------------------------------------------------------------------- bAbI question answering


def babi_qa(path: Path) -> list[Conversation]:
    """A bAbI question-answering file as conversations of one session (one story each). A
    statement's turn is marked before its last word; an answer of several words (``a,b``) is
    kept whole as one answer."""
    stories: list[Conversation] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        index, text = line.split(" ", 1)
        if index == "1":
            stories.append([[]])
        session = stories[-1][0]
        if "\t" in text:
            question, answer, support = text.split("\t")
            session.append(Turn(words(question), answer.strip(), None,
                                {"support": [int(i) for i in support.split()]}))
        else:
            said = words(text)
            last = max(i for i, w in enumerate(said) if w not in "?.,")
            session.append(Turn(said, None, last, {}))
    return stories


def babi_lexicon(*files: list[Conversation]) -> Lexicon:
    vocabulary, answers = set(), set()
    for stories in files:
        for conversation in stories:
            for session in conversation:
                for turn in session:
                    vocabulary.update(turn.said)
                    if turn.asked:
                        answers.add(str(turn.answer))
    return Lexicon(sorted(vocabulary), sorted(answers))


# ---------------------------------------------------------------------------------- bAbI dialog


def dialog_turns(path: Path) -> list[list[tuple[str, str]]]:
    """A bAbI dialog file as dialogs of ``(user, bot)`` turns; a knowledge-base line inside a
    dialog is a user turn with the bot silent (``""``)."""
    dialogs: list[list[tuple[str, str]]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index, text = line.split(" ", 1)
        if index == "1":
            dialogs.append([])
        user, bot = text.split("\t") if "\t" in text else (text, "")
        dialogs[-1].append((user, bot))
    return dialogs


def dialog_answers(train: Path, test: Path) -> dict[str, float]:
    """Of the test dialogs' bot turns: the share a closed set of the training dialogs' bot
    utterances holds whole, and the share whose every word a training bot turn says."""
    said = {b for d in dialog_turns(train) for _, b in d if b}
    heard = {w for b in said for w in b.split()}
    bots = [b for d in dialog_turns(test) for _, b in d if b]
    return {"turns": float(len(bots)),
            "whole": float(np.mean([b in said for b in bots])),
            "words": float(np.mean([all(w in heard for w in b.split()) for b in bots]))}
