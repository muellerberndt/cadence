"""Records of bound moments: what a brain's senses held together, kept by their strength and
read back by a part (``Brain.compose(bound=records)``, cadence 0.101, #342).

A brain with named senses and a fixed code (``senses=``, ``code=``) codes each moment where its
senses meet. With ``bound`` it keeps records of its moments in one ``LongTermMemory``:

- **A record** is written at every moment the brain lives (``act``, ``step``, ``live``,
  ``wait``) in which a sense holds something: its key is the brain's code of what the senses
  gave, each sense at unit length and its fields alike within it; its value what each sense
  gave. A moment told again (a key the store holds at cosine ``REVISE``, with a voice the same
  sound said) moves its record toward the new take (``blend``) and deepens it; another sound is
  another event and takes a record of its own; the brain's surprise (``surprise=True``) counts as
  tellings; a record told past ``KEEP`` tellings is kept for good, one told once fades and is
  forgotten in time (#336's strength law).
- **A reward** given to ``live`` or ``step`` pays the record of the moment the brain acted at:
  ``PAY`` tellings a unit of reward deepen it and are counted as its pay (``paid``). A reward of
  zero or less refuses a record of something said (``refused``).
- **A read by a part.** The records a read reaches are the known ones (depth ``KNOWN`` or kept;
  a record of something said only once paid or refused) whose key lies at cosine ``MATCH`` or
  more to the moment's code. For a field (a sense, or a field of a sense): the ``HEADS`` nearest
  records that hold it, those within ``NEAR`` of the nearest's cosine, fall into groups that hold
  it alike (cosine ``AGREE``), each weighed by its records' cosines; the heaviest comes back, as
  its mean, where it weighs more than ``MOST`` of them.
- **Completion.** Before every settle a field the senses given by name leave silent is given
  what came back (the voice apart), and the brain settles on it: a word heard alone brings back
  the thing it was heard with.
- **A voice** (``voice=sense``): ``Brain.say`` gives, in that sense's format, the paid sound
  whose moments share fields nearest the moment's (the paid records with one sound stand for the
  fields all of them hold alike, a field where they differ left out; a match of ``ANSWER`` or
  more, above every other sound's; a refused record of that sound at a moment like this one
  refuses it), else the heard sound the records bring back where the moment holds none, else a
  sound drawn from the distinct sounds the records hold, each one chance, passing over sounds
  refused at moments like this one (a babble). A moment like another holds more than half of the
  other's fields alike. What it says joins the moment under the sense ``voice`` when the caller
  passes it back, so the moment's record binds it and a reward that follows pays it.
- **A heard name draws the focus** (with a voice, ``surprise=True`` and a map with values):
  the records read by the heard sense alone bring back a thing's fields, and each thing on the
  map whose values hold more than half of them alike takes a salience of ``NAMED`` times their
  mean cosine in the focus's competition.

Born: the rule and its constants. Learned: every record, its strength and so what a part brings
back, what the brain says, and which thing a heard sound draws the focus to.

Sources: one record binds what the senses held at once and a part retrieves the whole (pattern
completion in the hippocampus: Marr 1971; Nakazawa et al. 2002); a song learned by a bird is
its own heard template refined by practice and reward (Konishi 1965; Tchernichovski et al.
2001); a target held in memory biases the competition for attention (Desimone and Duncan 1995).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .longterm import LongTermMemory

__all__ = ["BoundRecords"]


def _unit(a: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(a))
    return a / norm if norm > 1e-12 else a


def _fields(named: Mapping[str, Any], skip: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """Each field of the named senses as ``name`` or ``sense/field`` -> ``(start, stop)``,
    the senses in ``skip`` left out."""
    out: dict[str, tuple[int, int]] = {}
    for sense, columns in named.items():
        if sense in skip:
            continue
        if isinstance(columns, Mapping):
            for field, span in columns.items():
                out[f"{sense}/{field}"] = (int(span[0]), int(span[1]))
        else:
            out[sense] = (int(columns[0]), int(columns[1]))
    return out


class BoundRecords:
    """A brain's records of its bound moments (the module's docstring gives the laws).

    ``named`` is the brain's senses by name (``brain._senses["named"]``: a sense's columns, or
    its fields' by name), ``key_width`` its code's width, ``inputs`` the observation's width,
    ``capacity`` the records. ``voice`` names the sense the brain speaks in (``"voice"``) and
    ``heard`` the sense whose format it takes (the ear); both None without a voice."""

    HEADS = 8
    NEAR = 0.7
    MOST = 0.3
    ANSWER = 0.5  # the least match of a said sound's group that answers
    AGREE = 0.8
    MATCH = 0.15
    KNOWN = 0.5
    PAY = 4.0
    NAMED = 8.0
    REVISE = 0.95
    KEEP = 8.0
    SKIP = ("space",)  # the map's read: the map is its own store
    strength = True  # the brain's surprise counts as tellings (``Brain._moment_surprise``)

    def __init__(self, named: Mapping[str, Any], key_width: int, inputs: int, capacity: int, *,
                 voice: str | None = None, heard: str | None = None, seed: int = 0) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or (
                capacity < 1):
            raise ValueError(
                "bound is how many records the brain keeps: pass a positive integer, such as "
                "bound=4096")
        self.capacity = int(capacity)
        self.voice, self.heard, self.seed = voice, heard, int(seed)
        self.fields = _fields(named, self.SKIP)
        if not self.fields:
            raise ValueError("bound records need a sense to hold")
        self.inputs = int(inputs)
        self.store = LongTermMemory(
            int(key_width), self.inputs, self.capacity, cosine=True, revise=self.REVISE,
            blend=True, keep=self.KEEP, forget=8.0 * self.capacity, seed=self.seed,
        )
        self.written = np.full(0, -1, dtype=np.int64)  # the slot each stream's moment took
        # records of something said that a reward of zero or less followed, and each record's
        # pay: the tellings rewards added to it
        self.refused = np.zeros(self.capacity, dtype=bool)
        self.paid = np.zeros(self.capacity)
        self.draws = 0  # the babbles drawn, for the next draw's generator
        self.said_from: list[str | None] = []  # where each stream's latest said sound came from

    # -- what a description and a file hold

    def settings(self) -> dict[str, Any]:
        """The records' construction: ``records``, the ``voice`` sense, the ``heard`` sense it
        takes its format from, and the ``seed`` of the store's address."""
        return {"records": self.capacity, "voice": self.voice, "heard": self.heard,
                "seed": self.seed}

    def to_dict(self) -> dict[str, Any]:
        """What a checkpoint's metadata keeps: the settings and the babbles drawn."""
        return {**self.settings(), "draws": int(self.draws)}

    def arrays(self) -> dict[str, np.ndarray]:
        """What a checkpoint's arrays keep: the store's (``store/<name>``), the moment a reward
        pays (``written``), the refused records and each record's pay."""
        out = {f"store/{name}": value for name, value in self.store.arrays().items()}
        out["written"] = self.written.copy()
        out["refused"] = self.refused.copy()
        out["paid"] = self.paid.copy()
        return out

    @classmethod
    def from_parts(cls, named: Mapping[str, Any], key_width: int, inputs: int,
                   record: Mapping[str, Any], arrays: Mapping[str, Any]) -> BoundRecords:
        """The records a checkpoint kept (``to_dict`` and ``arrays``), for a brain of the senses
        ``named``; refused by name where they do not fit."""
        try:
            out = cls(named, key_width, inputs, int(record["records"]),
                      voice=record.get("voice"), heard=record.get("heard"),
                      seed=int(record.get("seed", 0)))
            out.draws = int(record.get("draws", 0))
            out.store.restore({name[len("store/"):]: np.asarray(value)
                               for name, value in arrays.items() if name.startswith("store/")})
            if "written" in arrays:
                out.written = np.asarray(arrays["written"], dtype=np.int64)
            if "refused" in arrays:
                out.refused = np.asarray(arrays["refused"], dtype=bool).reshape(out.capacity)
            if "paid" in arrays:
                out.paid = np.asarray(arrays["paid"], dtype=np.float64).reshape(out.capacity)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid saved bound records: {exc}") from exc
        return out

    # -- the read

    def _active(self, row: np.ndarray, field: str) -> bool:
        first, last = self.fields[field]
        return bool(np.any(row[first:last]))

    def holds(self, row: np.ndarray) -> bool:
        """Whether a row holds something in a field the records keep."""
        return any(self._active(row, field) for field in self.fields)

    def _known_slots(self, slots: np.ndarray) -> np.ndarray:
        """Whether each of ``slots`` is read: a record at depth ``KNOWN`` or kept for good; a
        record of something said only once a reward paid it or refused it; none forgotten."""
        store = self.store
        known = np.asarray(store.deep[slots], dtype=np.float64) >= self.KNOWN
        if store.keep:
            known |= np.asarray(store.kept[slots], dtype=bool)
        if self.voice is not None:
            first, last = self.fields[self._voice_field()]
            said = np.any(store.values[slots, first:last], axis=1)
            known = np.where(said, (self.paid[slots] > 0) | self.refused[slots], known)
        if store.forget:
            known &= ~np.asarray(store.gone[slots], dtype=bool)
        return np.asarray(known, dtype=bool)

    def _voice_field(self) -> str:
        assert self.voice is not None
        return next(f for f in self.fields if f == self.voice or f.startswith(self.voice + "/"))

    def _taking(self, code: np.ndarray) -> list[tuple[int, float]]:
        """The records a read of ``code`` reaches, each with its cosine to the cue: the known
        records among the store's candidates for the cue (every record of a store its read
        scans whole), at cosine ``MATCH`` or above."""
        store = self.store
        if not store.count or not np.any(code):
            return []
        slots, keys = store._candidates(np.asarray(code, dtype=np.float64))
        slots = np.asarray(slots, dtype=np.int64)
        if not len(slots):
            return []
        known = self._known_slots(slots)
        slots, keys = slots[known], np.asarray(keys, dtype=np.float64)[known]
        if not len(slots):
            return []
        norms = np.linalg.norm(keys, axis=1) * float(np.linalg.norm(code))
        cosines = np.where(norms > 0, keys @ code / np.maximum(norms, 1e-12), 0.0)
        return [(int(s), float(c)) for s, c in zip(slots, cosines, strict=True)
                if c >= self.MATCH]

    def _agreed(self, taking: list[tuple[int, float]], row: np.ndarray
                ) -> dict[str, np.ndarray]:
        """Each field the reached records hold, where most of the nearest of them hold it alike
        (the voice apart, whose read is ``_answer``). For a field: the ``HEADS`` nearest records
        that hold it, those within ``NEAR`` of the nearest's cosine, fall into groups that hold
        it alike (cosine ``AGREE`` to a group's first), each weighed by its records' cosines; the
        heaviest group comes back, as its mean, where it weighs more than ``MOST`` of the
        records that hold the field. ``row`` is the moment read (unused by this rule)."""
        out: dict[str, np.ndarray] = {}
        if not taking:
            return out
        slots = np.array([slot for slot, _ in taking], dtype=np.int64)
        weights = np.array([weight for _, weight in taking])
        values = np.asarray(self.store.values[slots], dtype=np.float64)
        voice = None if self.voice is None else self._voice_field()
        for field, (first, last) in self.fields.items():
            if field == voice:
                continue
            parts = values[:, first:last]
            holding = np.flatnonzero(np.any(parts, axis=1))
            if not len(holding):
                continue
            holding = holding[np.argsort(-weights[holding], kind="stable")][: self.HEADS]
            holding = holding[weights[holding] >= self.NEAR * weights[holding[0]]]
            groups: list[list[int]] = []
            for k in holding:
                for group in groups:
                    if float(_unit(parts[k]) @ _unit(parts[group[0]])) >= self.AGREE:
                        group.append(int(k))
                        break
                else:
                    groups.append([int(k)])
            mass = [float(weights[g].sum()) for g in groups]
            best = int(np.argmax(mass))
            if mass[best] > self.MOST * float(weights[holding].sum()):
                out[field] = parts[groups[best]].mean(axis=0)
        return out

    def _read(self, code: np.ndarray, row: np.ndarray) -> dict[str, np.ndarray]:
        """``_agreed(_taking(code), row)``, in one compiled call where the kernels run
        (``cadence.compiled.bound_recall``, the same read)."""
        from . import compiled

        # a cue of another width than the keys takes the Python, which refuses it
        if (compiled.available("_bound_recall") and self.store.keys.dtype == np.float64
                and np.shape(code) == self.store.keys.shape[1:]):
            return compiled.bound_recall(self, code)
        return self._agreed(self._taking(code), row)

    def complete(self, rows: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """``rows`` with each field they leave silent given what the records bring back for the
        moment's ``codes``; the voice is left as given. Changes nothing in the records."""
        out = np.array(rows, dtype=np.float64)
        for i in range(len(out)):
            if not self.holds(out[i]):
                continue
            for field, part in self._read(codes[i], out[i]).items():
                first, last = self.fields[field]
                if not np.any(out[i, first:last]):
                    out[i, first:last] = part
        return out

    def recall(self, row: np.ndarray, code: np.ndarray) -> dict[str, np.ndarray]:
        """What the records bring back for one moment: each field held alike, as its mean."""
        return self._read(code, np.asarray(row, dtype=np.float64))

    # -- the voice

    def say(self, rows: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """What the brain says at each row, in the format of the voice: the paid sound that
        stands for the moment (``_answer``), else the heard sound the records bring back where
        the moment holds none, else a sound drawn from the distinct sounds the records hold
        (``_babble``); zeros where the records hold no sound. ``said_from`` names each choice:
        ``"said"``, ``"heard"``, ``"babble"`` or None."""
        assert self.voice is not None and self.heard is not None
        voice, heard = self._voice_field(), self.heard
        v0, v1 = self.fields[voice]
        h0, h1 = self.fields[heard]
        out = np.zeros((len(rows), v1 - v0))
        self.said_from = []
        for i in range(len(rows)):
            row = np.asarray(rows[i], dtype=np.float64)
            taking = self._taking(codes[i]) if self.holds(row) else []
            back = self._agreed(taking, row)
            answer = self._answer(taking, row)
            source: str | None
            if answer is not None:
                out[i], source = answer, "said"
            elif heard in back and not np.any(row[h0:h1]):
                out[i], source = back[heard], "heard"
            else:
                drawn = self._babble(self._refused_sounds(codes[i], row))
                source = None if drawn is None else "babble"
                if drawn is not None:
                    out[i] = drawn
            self.said_from.append(source)
        return out

    def _answer(self, taking: list[tuple[int, float]], row: np.ndarray) -> np.ndarray | None:
        """The said sound that stands for the moment ``row``: the paid records the read reaches
        fall into groups by the sound they hold (cosine ``AGREE``). A group stands for what its
        moments share: the fields every one of its records holds alike. Its match to the moment
        is the mean cosine, over those fields the moment holds, of the moment's field to the
        group's mean, a field where its records differ (a place) left out; a refused record
        whose sound the group holds and which holds those fields as the moment does (``AGREE``)
        refuses the group for it. The best group answers where its match is ``ANSWER`` or more
        and above every other's; None where none does."""
        assert self.voice is not None
        store = self.store
        v0, v1 = self.fields[self._voice_field()]
        own = [f for f in self.fields if f != self._voice_field()]
        held = [f for f in own if np.any(row[self.fields[f][0]:self.fields[f][1]])]
        if not taking or not held:
            return None
        slots = np.array([slot for slot, _ in taking], dtype=np.int64)
        values = np.asarray(store.values[slots], dtype=np.float64)
        sounds = values[:, v0:v1]
        said = np.any(sounds, axis=1)
        paid = np.flatnonzero(said & (self.paid[slots] > 0) & ~self.refused[slots])
        refused = np.flatnonzero(said & self.refused[slots])
        groups: list[list[int]] = []
        for k in paid:
            for group in groups:
                if float(_unit(sounds[k]) @ _unit(sounds[group[0]])) >= self.AGREE:
                    group.append(int(k))
                    break
            else:
                groups.append([int(k)])
        best, best_match, second = None, -1.0, -1.0
        for group in groups:
            shared, alike = [], []
            for f in held:
                a, b = self.fields[f]
                parts = values[group, a:b]
                if not np.all(np.any(parts, axis=1)):
                    continue
                lead = _unit(parts[0])
                if all(float(_unit(q) @ lead) >= self.AGREE for q in parts[1:]):
                    shared.append(f)
                    alike.append(float(_unit(row[a:b]) @ _unit(parts.mean(axis=0))))
            if not shared:
                continue
            match = float(np.mean(alike))
            for k in refused:
                if float(_unit(sounds[k]) @ _unit(sounds[group[0]])) < self.AGREE:
                    continue
                same = [float(_unit(values[k, self.fields[f][0]:self.fields[f][1]])
                              @ _unit(row[self.fields[f][0]:self.fields[f][1]])) for f in shared]
                if np.mean(same) >= self.AGREE:
                    match = -1.0  # said here and not paid
                    break
            if match > best_match:
                best, best_match, second = group, match, best_match
            elif match > second:
                second = match
        if best is None or best_match < self.ANSWER or best_match <= second:
            return None
        return np.asarray(sounds[best].mean(axis=0))

    def _like(self, value: np.ndarray, row: np.ndarray) -> bool:
        """Whether a record's ``value`` is of a moment like ``row``: of the fields the moment
        holds (the voice apart), more than half held alike (cosine ``AGREE``)."""
        held = [f for f in self.fields if f != self._voice_field()
                and np.any(row[self.fields[f][0]:self.fields[f][1]])]
        if not held:
            return False
        alike = [float(_unit(value[a:b]) @ _unit(row[a:b])) >= self.AGREE
                 for a, b in (self.fields[f] for f in held)]
        return float(np.mean(alike)) > 0.5

    def _refused_sounds(self, code: np.ndarray, row: np.ndarray) -> np.ndarray:
        """The sounds said and not paid at moments like ``row`` (the refused records the read of
        ``code`` reaches that are of moments like it), one row each."""
        assert self.voice is not None
        first, last = self.fields[self._voice_field()]
        values = self.store.values
        slots = [slot for slot, _ in self._taking(code)
                 if self.refused[slot] and self._like(np.asarray(values[slot]), row)]
        return np.asarray(values[slots, first:last], dtype=np.float64).reshape(
            len(slots), last - first)

    def _babble(self, avoid: np.ndarray | None = None) -> np.ndarray | None:
        """A sound the records hold, each distinct sound one chance (records whose sounds lie at
        cosine ``AGREE`` are one sound): a record's said sound, else its heard one; a sound like
        one in ``avoid`` (said at moments like this and not paid) is passed over while another
        is held. None where no record holds a sound."""
        store = self.store
        n = store.count
        if not n:
            return None
        assert self.voice is not None and self.heard is not None
        v0, v1 = self.fields[self._voice_field()]
        h0, h1 = self.fields[self.heard]
        values = store.values[:n]
        said, heard = np.any(values[:, v0:v1], axis=1), np.any(values[:, h0:h1], axis=1)
        alive = ~store.gone[:n] if store.forget else np.ones(n, dtype=bool)
        holding = np.flatnonzero((said | heard) & alive)
        if not len(holding):
            return None
        sounds = np.where(said[holding, None], values[holding, v0:v1], values[holding, h0:h1])
        sounds = np.asarray(sounds, dtype=np.float64)
        units = sounds / np.maximum(np.linalg.norm(sounds, axis=1, keepdims=True), 1e-12)
        distinct: list[int] = []
        for k in range(len(units)):
            if not distinct or float((units[distinct] @ units[k]).max()) < self.AGREE:
                distinct.append(k)
        choices = np.array(distinct)
        if avoid is not None and len(avoid):
            shunned = avoid / np.maximum(np.linalg.norm(avoid, axis=1, keepdims=True), 1e-12)
            fresh = (units[choices] @ shunned.T).max(axis=1) < self.AGREE
            if fresh.any():
                choices = choices[fresh]
        rng = np.random.default_rng((self.seed, self.draws))
        self.draws += 1
        return np.asarray(sounds[int(choices[int(rng.integers(len(choices)))])])

    # -- the write

    def hear(self, rows: np.ndarray, codes: np.ndarray, surprise: Any = None,
             acted: bool = True) -> None:
        """The moment's records: each row that holds something written under its code (a moment
        told again deepens its record), with the moment's surprise in tellings; then the
        records fade by one moment. A moment the brain ``acted`` at becomes the one a reward
        pays; a moment it only sensed (``wait``) is written and leaves that as it was.

        A moment told again is the same event: with a voice, a record that holds another
        sound (or a sound where the moment holds none, or the reverse) is another event and
        is not revised, so a reward pays the sound the moment said. A record a reward is
        awaited for keeps its value and is neither replaced nor forgotten before the reward:
        a moment only sensed revises it only where it holds that moment exactly and else takes
        another slot, and in a store whose every record awaits a reward the moment is not
        written. The streams of one moment that act at one event share its record; the slots
        the earlier streams took are not replaced by the later ones."""
        rows = np.asarray(rows, dtype=np.float64)
        written = np.full(len(rows), -1, dtype=np.int64)
        awaited = np.array([] if acted else [int(s) for s in self.written if s >= 0],
                           dtype=np.int64)
        voice = None if self.voice is None else self.fields[self._voice_field()]
        store = self.store
        for i in range(len(rows)):
            if not self.holds(rows[i]) or not np.any(codes[i]):
                continue
            value = np.zeros(self.inputs)
            for first, last in self.fields.values():
                value[first:last] = rows[i, first:last]
            told = None if surprise is None else float(max(surprise[i], 0.0))
            revisable = None
            if voice is not None or len(awaited):
                revisable = self._same_event(value, voice, awaited)
            spare = awaited if not acted else written[:i][written[:i] >= 0]
            written[i], revised = store.write(
                np.asarray(codes[i], dtype=np.float64), value, surprise=told,
                spare=spare, revisable=revisable)
            if written[i] < 0:  # every record awaits a reward
                continue
            if not revised:  # a record of its own: no pay, nothing refused
                self.refused[written[i]] = False
                self.paid[written[i]] = 0.0
        if acted:
            self.written = written
        store.moment(awaited=None if acted else awaited)

    def _same_event(self, value: np.ndarray, voice: tuple[int, int] | None,
                    awaited: np.ndarray) -> Any:
        """The judge of which candidate records a moment of ``value`` may revise: with a
        ``voice`` field those whose said sound is the moment's (both silent, or alike at cosine
        ``AGREE``), and of the ``awaited`` records those that hold the moment exactly."""
        store = self.store
        own = value.astype(store.dtype)

        def revisable(slots: np.ndarray) -> np.ndarray:
            held = np.asarray(store.values[slots])
            out = np.ones(len(slots), dtype=bool)
            if voice is not None:
                first, last = voice
                sounds = np.asarray(held[:, first:last], dtype=np.float64)
                sound = np.asarray(value[first:last], dtype=np.float64)
                silent = ~np.any(sounds, axis=1)
                if not np.any(sound):
                    out &= silent
                else:
                    norms = np.linalg.norm(sounds, axis=1) * float(np.linalg.norm(sound))
                    alike = sounds @ sound >= self.AGREE * norms
                    out &= ~silent & alike
            if len(awaited):
                waiting = np.isin(slots, awaited)
                if waiting.any():
                    out[waiting] &= np.all(held[waiting] == own, axis=1)
            return out

        return revisable

    def pay(self, reward: Any) -> None:
        """A reward for the moment the brain acted at: each stream's record of it deepens by
        ``PAY`` tellings a unit of reward, counted as its pay, and its refusal is lifted (and a
        record forgotten while it awaited the reward is held again); a reward of zero or less
        refuses a record of something said (it then answers nothing and its sound is passed
        over at moments like it, until a reward pays it). Where the streams of one moment share
        a record, a reward any of them is paid pays it and lifts the refusal of the others."""
        r = np.asarray(reward, dtype=np.float64).reshape(-1)
        said = None
        if self.voice is not None:
            first, last = self.fields[self._voice_field()]
            said = (first, last)
        store = self.store
        paid: set[int] = set()
        refused: list[int] = []
        for i, slot in enumerate(self.written):
            if slot < 0 or i >= len(r) or not np.isfinite(r[i]):
                continue
            if r[i] > 0:
                tellings = self.PAY * float(r[i])
                store.deep[slot] += tellings
                if store.keep and store.deep[slot] >= store.keep:
                    store.kept[slot] = True
                store.faded[slot] = 0.0
                if store.forget:
                    store.gone[slot] = False
                self.paid[slot] += tellings
                self.refused[slot] = False
                paid.add(int(slot))
            elif said is not None and np.any(store.values[slot, said[0]:said[1]]):
                refused.append(int(slot))
        for held in refused:
            if held not in paid:
                self.refused[held] = True

    # -- the focus a heard name draws

    def named(self, row: np.ndarray, code_of: Any) -> dict[str, np.ndarray]:
        """The fields a heard sound alone brings back (the records read by the heard sense of
        ``row`` alone; ``code_of`` codes a row), the heard and the voice left out; empty where
        nothing is heard or nothing comes back."""
        if self.heard is None:
            return {}
        first, last = self.fields[self.heard]
        if not np.any(row[first:last]):
            return {}
        alone = np.zeros_like(row)
        alone[first:last] = row[first:last]
        back = self.recall(alone, np.asarray(code_of(alone[None]))[0])
        back.pop(self.heard, None)
        if self.voice is not None:
            back.pop(self._voice_field(), None)
        return back

    def salience(self, template: Mapping[str, np.ndarray], things: list[dict[str, Any]]
                 ) -> np.ndarray:
        """Each thing's salience in the focus's competition: for the groups of its values on the
        map whose name is a field of the ``template`` (what a heard name brings back), the
        share held alike (cosine ``AGREE``); where more than half are, ``NAMED`` times the mean
        cosine over them; zero elsewhere. A thing need not match in every field: a name brings
        back a kind's form, and its colour may be another."""
        out = np.zeros(len(things))
        for k, thing in enumerate(things):
            values = thing.get("codes") or {}
            alike = []
            for field, part in template.items():
                held = values.get(field.split("/")[-1])
                if held is not None and len(held) == len(part):
                    alike.append(float(_unit(np.asarray(held, dtype=np.float64)) @ _unit(part)))
            if alike and np.mean(np.asarray(alike) >= self.AGREE) > 0.5:
                out[k] = self.NAMED * float(np.mean(alike))
        return out
