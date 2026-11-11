"""The memories of a composed brain, under one contract.

A composed brain's memories are the members of ``brain.memories``: an ordered, read-only
mapping from a name to the memory, present memories only, in the order the brain reads and
writes them (``working_memory``, ``efference``, ``history``, ``recall``, ``hippocampus``,
``episodic``). Every member keeps one contract (``Memory``):

- ``stimulus(drive, moment)``: the drive of a moment before the settle with what the memory
  adds to it; a memory that adds nothing returns the drive. ``Moment`` gives the moment's
  observations, their code, the key the associative store reads under and whether the
  stores read into the drive at all.
- ``branch()``: a private copy for an imagined branch: it reads the live state and writes
  none of it.
- ``begin(keep=False)``: the streams begin afresh. What the memory keeps across streams
  stays: the associative records, and with ``keep`` the recall memory's slots.
- ``reset(batch, rows=None)`` and ``keep(rows)``: fresh streams, all of them or ``rows``;
  the state of ``rows`` kept and the other streams dropped.
- ``to_dict()`` and ``arrays()``: its settings and counts, and the arrays a checkpoint keeps
  of it (under ``memories/<name>/``).

How a memory is written after the settle follows its own inputs: the working trace moves
toward the settled state, the efference copy takes the issued command, the heard history the
observation, the recall memory witnesses the moment, the stores record outcomes, the
episodic records take the decision into the stream's temporal context and path. The brain
writes them in the order of the list.

With ``memory_choice=True`` the member ``episodic`` is an ``EpisodicRecords``: the
associative store's place records and, with a temporal context, its own context records,
whose vote joins the motor readout after the settle (its ``stimulus`` adds nothing), with
the paths along which a reward is replayed. ``snapshot()`` and ``restore()`` keep its state
around an outcome's write.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple, Protocol

import numpy as np

from .memory import SynapticMemory
from .stream import FastSynapses, _stream_rows

__all__ = ["EpisodicRecords", "Memories", "Memory", "Moment"]


class Moment(NamedTuple):
    """What a memory reads of a moment before its settle."""

    observations: np.ndarray  # the observations, one row per stream
    code: np.ndarray  # their code: the observations themselves without a code
    place: np.ndarray | None  # the key the associative store reads; None reads the drive
    stores: bool = True  # whether the stores read into the drive (not under an episodic vote)


class Memory(Protocol):
    """The contract every member of ``brain.memories`` keeps."""

    def stimulus(self, drive: np.ndarray, moment: Moment) -> np.ndarray: ...

    def branch(self) -> Memory: ...

    def begin(self, keep: bool = False) -> None: ...

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None: ...

    def keep(self, rows: np.ndarray) -> None: ...

    def to_dict(self) -> dict[str, Any]: ...

    def arrays(self) -> dict[str, np.ndarray]: ...


class Memories(Mapping[str, Any]):
    """The present memories of a brain by name, in the order the brain reads them."""

    ORDER = ("working_memory", "efference", "history", "recall", "hippocampus", "episodic")

    def __init__(self, members: Mapping[str, Any]) -> None:
        unknown = set(members) - set(self.ORDER)
        if unknown:
            raise ValueError(f"unknown memory: {sorted(unknown)[0]!r}")
        self._members = {
            name: members[name]
            for name in self.ORDER
            if name in members and members[name] is not None
        }

    def __getitem__(self, name: str) -> Any:
        return self._members[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"Memories({', '.join(self._members)})"

    def stimulus(self, drive: np.ndarray, moment: Moment) -> np.ndarray:
        """The drive with what every member adds, in order."""
        for memory in self._members.values():
            drive = memory.stimulus(drive, moment)
        return drive


# One decision on a stream's path: (its context key, the action, the place key it was taken
# on, its outcome, whether the outcome was received, how far it was the best guess)
Entry = tuple[np.ndarray | None, np.ndarray, np.ndarray, float, bool, float]


def command(action: np.ndarray, motor: int, offsets: np.ndarray) -> np.ndarray:
    """The ``(batch, motor)`` one-hot of an issued action: one per row, or one per slot, the
    slots' motor neurons starting at ``offsets``."""
    action = np.asarray(action, dtype=np.int64)
    out = np.zeros((len(action), motor))
    rows = np.arange(len(action))
    if action.ndim == 2:
        out[rows[:, None], action + offsets[None, :]] = 1.0
    else:
        out[rows, action] = 1.0
    return out


def record_action(
    memory: FastSynapses,
    keys: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    importance: np.ndarray,
    known: np.ndarray | None,
    offsets: np.ndarray,
) -> None:
    """Write the outcome of a chosen action into a store of records for the situation it was
    chosen in (``keys``); with ``known``, only for the streams where it is set. The chosen
    motor neurons are one per row, or one per slot starting at ``offsets``."""
    # Every stream writes when ``known`` is unset: a write without a mask.
    write = None if known is None else np.asarray(known, bool)
    if write is not None and not write.any():
        return
    # The chosen motor neurons in motor order: one per row, or one per slot.
    rows: np.ndarray = np.arange(len(action))
    value: np.ndarray = reward
    chosen: np.ndarray = action
    if action.ndim == 2:
        chosen = action + offsets[None, :]
        rows, value = rows[:, None], reward[:, None]
    if isinstance(memory, SynapticMemory):
        target = np.zeros((len(action), len(memory.post)))
        target[rows, chosen] = value
        observed = np.zeros(target.shape, bool)
        observed[rows, chosen] = True
        memory.observe(keys, target, write, salience=importance, value_mask=observed)
    else:
        target = memory.recall(keys)
        target[rows, chosen] = value
        memory.observe(keys, target, write)


class EpisodicRecords:
    """The episodic memory whose records vote in the choice (``memory_choice=True``).

    Its place records are the associative store's (``brain.hippocampus``, the member
    ``hippocampus``), keyed on the place key of a moment. With a temporal context
    (``memory_context`` above zero) its context records (``temporal``, ``brain.context_memory``)
    are keyed on the stream's temporal context, the fading code of what it sensed and did,
    with the code of the decision's moment entered at the share ``1 - share``. The vote, the
    mean of the two reads, joins the motor readout after the settle, so the memory adds
    nothing to the drive. Per stream it keeps the temporal context after its last decision
    (``context``), that decision's key (``decision``) and the recent path (``paths``), one
    decision per situation the stream left, along which a reward is replayed (``steps``
    decisions back). A demonstration is recorded at the value ``demonstrations``."""

    def __init__(
        self,
        place: FastSynapses,
        temporal: SynapticMemory | None,
        *,
        width: int,
        offsets: np.ndarray,
        share: float,
        steps: int,
        demonstrations: float,
    ) -> None:
        self.place = place
        self.temporal = temporal
        self.width = int(width)  # the place key's width; a context key adds the motor neurons
        self.motor = int(len(place.post))
        self.offsets = np.asarray(offsets, dtype=np.int64)  # each motor slot's first neuron
        self.share = float(share)  # the share the past keeps per moment (``memory_context``)
        self.steps = int(steps)  # the decisions a reward replays (``memory_replay``)
        self.demonstrations = float(demonstrations)  # (``memory_demonstrations``)
        self.context: np.ndarray | None = None
        self.decision: np.ndarray | None = None
        self.paths: list[list[Entry]] = []

    # -- the memory contract

    def stimulus(self, drive: np.ndarray, moment: Moment) -> np.ndarray:
        """The vote joins the readout after the settle: the drive as it is."""
        return drive

    def branch(self) -> EpisodicRecords:
        """An imagined branch: the records are read and none is written; the temporal
        context is the branch's own copy, moved on by the branch's acts."""
        out = copy.copy(self)
        out.context = None if self.context is None else self.context.copy()
        out.decision = None if self.decision is None else self.decision.copy()
        out.paths = []
        return out

    def begin(self, keep: bool = False) -> None:
        """The streams begin afresh: no temporal context, no decision, no path; the records
        stay."""
        self.context = None
        self.decision = None
        self.paths = []

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """The streams in ``rows`` begin their next attempt with no temporal context and no
        path (all streams afresh without ``rows``); the records stay."""
        if rows is None:
            self.begin()
            return
        rows = _stream_rows(rows, batch)
        if self.context is not None and len(self.context) == batch:
            self.context[rows] = 0.0
        if len(self.paths) == batch:
            for row in rows:
                self.paths[row] = []

    def keep(self, rows: np.ndarray) -> None:
        """The state of the streams in ``rows``, the others dropped."""
        batch = (
            len(self.context) if self.context is not None
            else len(self.decision) if self.decision is not None
            else len(self.paths)
        )
        rows = _stream_rows(rows, batch)
        if self.context is not None:
            self.context = self.context[rows]
        if self.decision is not None:
            self.decision = self.decision[rows]
        if self.paths:
            self.paths = [self.paths[row] for row in rows]

    def snapshot(self) -> tuple[Any, ...]:
        """What an outcome's write changes, for ``restore`` when the write is refused: the
        context records' arrays (a write replaces them), the temporal contexts and the
        paths."""
        temporal = self.temporal
        return (
            None
            if temporal is None
            else {
                name: getattr(temporal, name)
                for name in ("strength", "mass", "writes", "consolidated")
            },
            None if self.context is None else self.context.copy(),
            [list(path) for path in self.paths],
        )

    def restore(self, saved: tuple[Any, ...]) -> None:
        """The state ``snapshot`` kept."""
        records, self.context, self.paths = saved
        if records is not None:
            for name, value in records.items():
                setattr(self.temporal, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": True,
            "context": float(self.share),
            "replay": int(self.steps),
            "demonstrations": float(self.demonstrations),
            "context_memory": None if self.temporal is None else self.temporal.to_dict(),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """The context records' arrays, the temporal contexts, the decisions' keys and the
        replay paths as padded arrays with each stream's length; the place records are the
        associative store's."""
        out: dict[str, np.ndarray] = {}
        if self.temporal is not None:
            out.update({"context_memory/" + k: v for k, v in self.temporal.arrays().items()})
            if self.context is not None:
                out["context"] = self.context
            if self.decision is not None:
                out["decision"] = self.decision
        if self.paths and any(self.paths):
            streams, longest = len(self.paths), max(len(path) for path in self.paths)
            slots = len(self.offsets)
            keys = np.zeros((streams, longest, self.width + self.motor))
            observations = np.zeros((streams, longest, self.width))
            actions = np.zeros(
                (streams, longest, slots) if slots > 1 else (streams, longest), int
            )
            rewards = np.zeros((streams, longest))
            received = np.zeros((streams, longest), bool)
            guesses = np.zeros((streams, longest))
            for row, path in enumerate(self.paths):
                for index, (key, action, observation, reward, known, guess) in enumerate(path):
                    if key is not None:
                        keys[row, index] = key
                    actions[row, index], observations[row, index] = action, observation
                    rewards[row, index], received[row, index] = reward, known
                    guesses[row, index] = guess
            out.update(
                {
                    "path/actions": actions,
                    "path/observations": observations,
                    "path/rewards": rewards,
                    "path/received": received,
                    "path/guess": guesses,
                    "path/lengths": np.array([len(path) for path in self.paths]),
                }
            )
            if self.temporal is not None:
                out["path/keys"] = keys
        return out

    # -- the vote

    def vote(self, code: np.ndarray, key: np.ndarray | None) -> np.ndarray:
        """The records' vote for every motor neuron: the mean of the place read of the place
        key ``code`` and the context read of the decision's ``key``."""
        place = np.asarray(self.place.recall(np.asarray(code, dtype=float)), dtype=float)
        if self.temporal is None or key is None:
            return place
        context = np.asarray(self.temporal.recall(key), dtype=float)
        return np.asarray(0.5 * (place + context))

    def key(self, code: np.ndarray) -> np.ndarray | None:
        """The temporal key of a decision taken now on this observation code, before the
        decision moves the stream's context on; None without context records."""
        return None if self.temporal is None else self.keyed(code, self.context)

    def keyed(self, code: np.ndarray, context: np.ndarray | None) -> np.ndarray:
        """The key of a decision on this observation code in a temporal ``context``,
        ``(batch, width + motor)``: the code takes the share ``1 - share``; the context
        before it keeps the rest."""
        x = np.asarray(code, dtype=float)
        keep = self.share
        fresh = np.zeros((len(x), self.width + self.motor))
        fresh[:, : self.width] = x
        if context is None or len(context) != len(x) or not keep:
            return (1.0 - keep) * fresh if keep else fresh
        return keep * context + (1.0 - keep) * fresh

    def decided(self, batch: int) -> np.ndarray | None:
        """The context key of each stream's last decision, when it belongs to this batch."""
        decision = self.decision
        return decision if decision is not None and len(decision) == batch else None

    def sense(self, code: np.ndarray) -> None:
        """What was sensed without a decision enters the temporal context."""
        if self.temporal is not None:
            self.context = self.keyed(code, self.context)

    def move(self, key: np.ndarray, action: np.ndarray) -> None:
        """A decision of ``key``: what the stream did joins its temporal context after what it
        saw, the issued command at the share ``1 - share``."""
        self.decision = key
        context = key.copy()
        context[:, self.width:] += (1.0 - self.share) * command(
            action, self.motor, self.offsets
        )
        self.context = context

    def decide(
        self,
        place: np.ndarray,
        key: np.ndarray | None,
        action: np.ndarray,
        guess: np.ndarray | None = None,
    ) -> None:
        """A decision on the place key ``place`` under ``key``: the temporal context moves on,
        and with replay the decision joins the stream's path with ``guess``, how far it was
        the brain's best guess."""
        if key is not None:
            self.move(key, action)
        if self.steps:
            assert guess is not None
            self.extend(place, key, action, guess)

    # -- the paths and the replay

    def extend(
        self, x: np.ndarray, key: np.ndarray | None, action: np.ndarray, guess: np.ndarray
    ) -> None:
        """Add each stream's decision to its recent path; ``x`` is the place key. A decision
        taken on the code of the stream's previous decision replaces it: that decision left
        the situation as it was, and only the decision that leaves a situation lies on the
        path onwards. A previous decision whose own outcome was known, a reward or a
        punishment, stays on the path: it bounds what a later reward replays, whatever is
        decided after it in the same situation."""
        if len(self.paths) != len(x):
            self.paths = [[] for _ in range(len(x))]
        for row, path in enumerate(self.paths):
            entry = (
                None if key is None else key[row].copy(),
                np.array(action[row], copy=True),
                x[row].copy(),
                0.0,
                False,
                float(guess[row]),
            )
            # the replay stops at a received outcome that is not nothing
            bound = bool(path) and path[-1][4] and path[-1][3] != 0.0
            if path and not bound and np.array_equal(path[-1][2], entry[2]):
                path[-1] = entry
            else:
                path.append(entry)
                del path[: -(self.steps + 1)]

    def received(self, reward: np.ndarray) -> None:
        """The outcome of each stream's last decision joins its path."""
        if len(self.paths) != len(reward):
            return
        for row, path in enumerate(self.paths):
            if path:
                key, action, observation, _, _, guess = path[-1]
                path[-1] = (key, action, observation, float(reward[row]), True, guess)

    def outcome(
        self,
        observations: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        learned: np.ndarray,
        *,
        gamma: float,
        woken: bool = False,
        salience: np.ndarray | None = None,
    ) -> None:
        """Record the outcome of the last decision for the streams in ``learned``, where it is
        known: a reward or a punishment, the end of an attempt, a surprise that woke a calm
        brain, or an outcome the caller marked with a positive ``salience``. A zero outcome
        in the middle of an attempt is not yet one. A supplied salience also sets how fast
        the outcome consolidates; without one every write moves a record equally. A reward
        replays the decisions that led to it, discounted by ``gamma`` per step. The records
        key on ``observations``, the place key the decision was taken on."""
        marked = np.zeros(len(reward), bool) if salience is None else salience > 0
        known = learned & ((reward != 0) | done | woken | marked)
        keys = self.decided(len(action))
        # what the memory expected of the decision, read before its outcome is written; an
        # attempt that ends is a closed episode, whose reward is replayed whole
        surprise = np.where(done, reward, reward - self.held(observations, keys, action))
        if known.any():
            self.remember(observations, keys, action, reward, known, salience)
        self.replay(reward, surprise, done, learned, gamma=gamma)

    def held(
        self,
        observations: np.ndarray,
        keys: np.ndarray | None,
        action: np.ndarray,
        memory: FastSynapses | None = None,
    ) -> np.ndarray:
        """Each stream's record of its chosen action in reward units: the records' vote for
        the chosen motor neuron, or the read of ``memory`` alone, averaged over the slots of a
        slotted readout."""
        place = self.place
        if not place.amplitude:
            return np.zeros(len(action))
        if memory is None:
            read = self.vote(observations, keys)
        else:
            assert keys is not None  # one store is read under the keys it is keyed by
            read = memory.recall(keys)
        vote = np.asarray(read) / place.amplitude
        rows = np.arange(len(action))
        if action.ndim == 2:
            chosen = action + self.offsets[None, :]
            return np.asarray(vote[rows[:, None], chosen].mean(axis=1))
        return np.asarray(vote[rows, action])

    def remember(
        self,
        observations: np.ndarray,
        keys: np.ndarray | None,
        action: np.ndarray,
        value: np.ndarray,
        streams: np.ndarray,
        salience: np.ndarray | None = None,
    ) -> None:
        """The write for the streams in ``streams``: the place records under the place keys
        and, with a temporal context, the context records under the decision's keys. The
        stream's record moves its store's ``rate`` toward ``value`` at once, and the shared
        record ``consolidation`` toward it, faster with a supplied ``salience``; without one
        every write moves a record equally."""
        sized = np.zeros(len(action)) if salience is None else salience
        record_action(self.place, observations, action, value, sized, streams, self.offsets)
        if self.temporal is not None and keys is not None:
            record_action(self.temporal, keys, action, value, sized, streams, self.offsets)

    def replay(
        self,
        reward: np.ndarray,
        surprise: np.ndarray,
        done: np.ndarray,
        learned: np.ndarray,
        *,
        gamma: float,
    ) -> None:
        """Reverse replay: a reward credits, backwards along the stream's recent path (one
        decision per situation, the one that left it), the decisions whose own outcome was
        nothing, back to the last decision whose outcome was known (a reward or a
        punishment), each discounted by ``gamma`` per step of the path and up to the
        discounted reward. A reward that ends an attempt (``done``) closes an episode: it is
        replayed whole, into both kinds of record. In a continuing life, where it is not
        known where the paid effort began, a reward is replayed in proportion to its surprise
        (the reward less what the memory expected of the paid decision): into each decision's
        context record, that decision in the history it was taken in, and into its place
        record only as far as it was an exploration rather than the brain's best guess.
        There an expected reward replays nothing. Replay only adds credit."""
        if not self.steps or len(self.paths) != len(reward):
            return
        gamma = float(gamma)
        temporal = self.temporal
        paid = (reward > 0) & (surprise > 0) & learned
        if temporal is None:
            paid &= done  # without context records, a continuing life replays nothing
        rows = [row for row in np.flatnonzero(paid) if self.paths[row]]
        width = len(reward)
        for k in range(1, self.steps + 1):
            discount = gamma**k
            if not discount:
                break  # with gamma zero no credit flows back
            # the decision k steps before the paid one (each path's last entry), while the
            # outcomes on the way back were nothing
            rows = [
                row
                for row in rows
                if len(self.paths[row]) > k
                and self.paths[row][-1 - k][4]
                and self.paths[row][-1 - k][3] == 0.0
            ]
            if not rows:
                break
            earlier = [self.paths[row][-1 - k] for row in rows]
            observations = np.zeros((width, self.width))  # the path keeps the codes
            observations[rows] = [entry[2] for entry in earlier]
            keys = None
            if earlier[0][0] is not None:
                keys = np.zeros((width, len(earlier[0][0])))
                keys[rows] = [entry[0] for entry in earlier]
            action = np.zeros((width, *np.shape(earlier[0][1])), dtype=np.int64)
            action[rows] = [entry[1] for entry in earlier]
            ended = np.zeros(width, bool)
            ended[rows] = done[rows]
            guess = np.ones(width)
            guess[rows] = [entry[5] for entry in earlier]
            # what the written records hold: both kinds at the end of an attempt, the context
            # records in a continuing life
            held = self.held(observations, keys, action)
            if temporal is not None and keys is not None and not ended[rows].all():
                held = np.where(ended, held, self.held(keys, keys, action, temporal))
            credit = np.zeros(width)
            credit[rows] = np.minimum(
                discount * surprise[rows], discount * reward[rows] - held[rows]
            )
            write = credit > 0
            if (write & ended).any():
                self.remember(observations, keys, action, held + credit, write & ended)
            if (write & ~ended).any():
                assert temporal is not None and keys is not None
                steady = np.zeros(width)  # every write moves a record equally
                record_action(
                    temporal, keys, action, held + credit, steady, write & ~ended, self.offsets
                )
                # the place record only as far as the decision was an exploration: the best
                # guess tells nothing about what paid apart from its history
                place = self.held(observations, observations, action, self.place)
                surprising = credit * (1.0 - guess)
                gain = write & ~ended & (surprising > 0)
                if gain.any():
                    target = place + np.minimum(surprising, discount * reward - place)
                    gain &= target > place
                    if gain.any():
                        record_action(
                            self.place, observations, action, target, steady, gain, self.offsets
                        )
