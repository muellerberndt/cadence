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
- ``reset(batch, rows=None)`` and ``keep(rows)``: fresh streams, all of them or ``rows``;
  the state of ``rows`` kept and the other streams dropped. ``Brain.reset`` begins every
  stream afresh so; the stores keep their records, and the recall memory keeps its slots
  with ``keep=("slots",)``, each stream ending an episode there (``Recall.end``).
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


def _chosen(action: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows and the motor neurons of chosen actions, to index a ``(batch, motor)`` array:
    one per row, or one per slot, the slots' motor neurons starting at ``offsets``."""
    rows = np.arange(len(action))
    if action.ndim == 2:
        return rows[:, None], action + offsets[None, :]
    return rows, action


def command(action: np.ndarray, motor: int, offsets: np.ndarray) -> np.ndarray:
    """The ``(batch, motor)`` one-hot of an issued action: one per row, or one per slot, the
    slots' motor neurons starting at ``offsets``."""
    action = np.asarray(action, dtype=np.int64)
    out = np.zeros((len(action), motor))
    out[_chosen(action, offsets)] = 1.0
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
    rows, chosen = _chosen(action, offsets)
    value = reward.reshape(rows.shape)
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
    decision per situation the stream left, along which a reward is replayed (``REPLAY``
    decisions back). A demonstration is recorded at the value ``DEMONSTRATIONS``.

    With an extinction threshold (``memory_extinction``, above zero) a third store, ``paid``,
    keeps which actions were paid directly in which places: a known outcome with a reward,
    never a replayed or demonstrated credit. A zero outcome then writes a learned zero for
    the chosen action where the records' vote for it and its direct payment in this place
    both reach the threshold, so a paid record is extinguished when failures pay nothing.
    The rule is Sean McClane's (cadence#210).

    With a hold (``memory_context_hold``) some decisions stay out of the temporal context:
    ``"unchanged"`` keys a decision taken on the observation of the stream's previous decision
    as if that decision had not been taken, whatever its outcome (the replay path drops such a
    decision only when its outcome was nothing); ``"punished"``
    takes a decision whose outcome was below zero, in the middle of an attempt, back out of
    the context. Will Glynn measured both (cadence#202). Per stream it then also keeps the
    context before the last decision (``before``) and that decision's observation code
    (``seen``)."""

    # the decisions a reward replays and the value a demonstration is recorded at: constants
    # of the composed brain since 0.93.0 (``memory_replay``, ``memory_demonstrations``), the
    # founders every reading used
    REPLAY = 8
    DEMONSTRATIONS = 1.0
    # every stream's state beside its path, each None before its first decision: the temporal
    # context, the decision's key, and with the hold the context before it and its code
    _STATE = ("context", "decision", "before", "seen")

    def __init__(
        self,
        place: FastSynapses,
        temporal: SynapticMemory | None,
        *,
        width: int,
        offsets: np.ndarray,
        share: float,
        paid: SynapticMemory | None = None,
        extinction: float = 0.0,
        hold: str | None = None,
    ) -> None:
        if hold not in (None, "unchanged", "punished"):
            raise ValueError('memory_context_hold is None, "unchanged" or "punished"')
        if hold is not None and temporal is None:
            raise ValueError("memory_context_hold needs a temporal context (memory_context)")
        if (paid is None) != (not extinction):
            raise ValueError(
                "an extinction threshold reads a store of direct payments, and only then"
            )
        self.place = place
        self.temporal = temporal
        self.paid = paid  # the places where each action was paid directly
        self.extinction = float(extinction)  # (``memory_extinction``)
        self.hold = hold  # the decisions kept out of the temporal context
        self.before: np.ndarray | None = None
        self.seen: np.ndarray | None = None
        self.width = int(width)  # the place key's width; a context key adds the motor neurons
        self.motor = int(len(place.post))
        self.offsets = np.asarray(offsets, dtype=np.int64)  # each motor slot's first neuron
        self.share = float(share)  # the share the past keeps per moment (``memory_context``)
        self.steps = self.REPLAY  # the decisions a reward replays
        self.demonstrations = self.DEMONSTRATIONS
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
        for name in self._STATE:
            state = getattr(self, name)
            setattr(out, name, None if state is None else state.copy())
        out.paths = []
        return out

    def reset(self, batch: int, rows: np.ndarray | None = None) -> None:
        """The streams in ``rows`` begin their next attempt with no temporal context and no
        path (all streams afresh without ``rows``: no temporal context, no decision, no
        path); the records stay."""
        if rows is None:
            for name in self._STATE:
                setattr(self, name, None)
            self.paths = []
            return
        rows = _stream_rows(rows, batch)
        if self.context is not None and len(self.context) == batch:
            self.context[rows] = 0.0
        if self.before is not None and len(self.before) == batch:
            self.before[rows] = 0.0  # with the context at zero a hold changes nothing there
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
        for name in self._STATE:
            state = getattr(self, name)
            if state is not None:
                setattr(self, name, state[rows])
        if self.paths:
            self.paths = [self.paths[row] for row in rows]

    def snapshot(self) -> tuple[Any, ...]:
        """What an outcome's write changes, for ``restore`` when the write is refused: the
        context records' arrays (a write replaces them), the temporal contexts and the
        paths."""
        def kept(store: SynapticMemory | None) -> dict[str, Any] | None:
            names = ("strength", "mass", "writes", "consolidated")
            return None if store is None else {name: getattr(store, name) for name in names}

        return (
            kept(self.temporal),
            None if self.context is None else self.context.copy(),
            [list(path) for path in self.paths],
            kept(self.paid),
        )

    def restore(self, saved: tuple[Any, ...]) -> None:
        """The state ``snapshot`` kept."""
        records, self.context, self.paths, paid = saved
        for store, state in ((self.temporal, records), (self.paid, paid)):
            if state is not None:
                for name, value in state.items():
                    setattr(store, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": True,
            "context": float(self.share),
            "replay": int(self.steps),
            "demonstrations": float(self.demonstrations),
            "context_memory": None if self.temporal is None else self.temporal.to_dict(),
            **(
                {}
                if self.paid is None
                else {
                    "extinction": self.extinction,
                    # its separator is the place records', saved once with them
                    "paid_memory": {
                        k: v for k, v in self.paid.to_dict().items() if k != "separator"
                    },
                }
            ),
            **({} if self.hold is None else {"hold": self.hold}),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """The context records' arrays, the temporal contexts, the decisions' keys and the
        replay paths as padded arrays with each stream's length; the place records are the
        associative store's."""
        out: dict[str, np.ndarray] = {}
        if self.paid is not None:
            out.update({
                "paid_memory/" + k: v
                for k, v in self.paid.arrays().items()
                if not k.startswith("separator/")
            })
        if self.temporal is not None:
            out.update({"context_memory/" + k: v for k, v in self.temporal.arrays().items()})
            for name in self._STATE:
                if getattr(self, name) is not None:
                    out[name] = getattr(self, name)
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
        return None if self.temporal is None else self.keyed(code, self.held_context(code))

    def held_context(self, code: np.ndarray) -> np.ndarray | None:
        """The temporal context a decision on this observation code is keyed in: with the
        hold ``"unchanged"``, a stream whose previous decision was taken on this same code
        keys in the context before that decision, which left its observation as it was."""
        context, seen = self.context, self.seen
        if (
            self.hold != "unchanged"
            or context is None
            or seen is None
            or len(seen) != len(code)
            or self.before is None
        ):
            return context
        same = np.all(np.asarray(code, dtype=float) == seen, axis=1)
        if not same.any():
            return context
        out = context.copy()
        out[same] = self.before[same]
        return out

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
        """What was sensed without a decision enters the temporal context, and ends the run of
        decisions on one unchanged observation the hold keys out."""
        if self.temporal is not None:
            self.context = self.keyed(code, self.context)
            self.seen = None

    def issue(self, key: np.ndarray, action: np.ndarray, code: np.ndarray) -> None:
        """A decision of ``key`` taken on the observation code ``code``: with the hold, the
        context it was keyed in and the code are kept for the next decision; then the
        temporal context moves on (``move``). The stream's decisions and an imagined
        branch's share this bookkeeping."""
        if self.hold is not None:
            context = self.held_context(code)
            self.before = (
                np.zeros_like(key) if context is None or len(context) != len(key)
                else context.copy()
            )
            self.seen = np.array(code, dtype=float, copy=True)
        self.move(key, action)

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
        code: np.ndarray | None = None,
    ) -> None:
        """A decision on the place key ``place`` under ``key``, keyed on the observation code
        ``code`` (``place`` when omitted): the temporal context moves on, and with replay the
        decision joins the stream's path with ``guess``, how far it was the brain's best
        guess."""
        if key is not None:
            self.issue(key, action, place if code is None else code)
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
        held = self.held(observations, keys, action)
        surprise = np.where(done, reward, reward - held)
        if self.paid is not None:
            # a zero outcome extinguishes a record the action earned by its own payments here
            zero = learned & ~known & (reward == 0)
            if zero.any():
                direct = self.held(observations, observations, action, self.paid)
                known |= zero & (held >= self.extinction) & (direct >= self.extinction)
        if known.any():
            self.remember(observations, keys, action, reward, known, salience)
        if self.hold == "punished" and self.context is not None and self.before is not None:
            # a decision punished in the middle of an attempt leaves the temporal context
            punished = learned & (reward < 0) & ~done
            if punished.any() and len(self.before) == len(reward):
                self.context[punished] = self.before[punished]
        if self.paid is not None and (learned & (reward > 0)).any():
            record_action(
                self.paid, observations, action, np.ones(len(action)), np.zeros(len(action)),
                learned & (reward > 0), self.offsets,
            )
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
        # in reward units: the read divided by the amplitude of the store that was read (the
        # vote's, the place store's), which a retune of memory_amplitude leaves the paid store
        scale = self.place.amplitude if memory is None else memory.amplitude
        if not scale:
            return np.zeros(len(action))
        if memory is None:
            read = self.vote(observations, keys)
        else:
            assert keys is not None  # one store is read under the keys it is keyed by
            read = memory.recall(keys)
        vote = np.asarray(read) / scale
        chosen = vote[_chosen(action, self.offsets)]
        return np.asarray(chosen.mean(axis=1) if action.ndim == 2 else chosen)

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
