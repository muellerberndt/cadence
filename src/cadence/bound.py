"""Records of bound moments: what a brain's senses held together, kept by their strength and
read back by a part (``Brain.compose(bound=records)``, cadence 0.101, #342).

A brain with named senses and a fixed code (``senses=``, ``code=``) codes each moment where its
senses meet, every sense at unit length. With ``bound`` it keeps records of its moments in one
``LongTermMemory``:

- **A record** is written at every moment the brain lives in which a sense holds something: its
  key is the moment's code, its value what each sense held. A moment told again (a key the
  store holds at cosine ``REVISE``) moves its record toward the new take (``blend``) and
  deepens it; the brain's surprise (``surprise=True``) counts as tellings; a record told past
  ``KEEP`` tellings is kept for good, one told once fades and is forgotten in time (#336's
  strength law).
- **A reward** given to ``live`` or ``step`` deepens the record of the moment it pays by
  ``PAY`` tellings a unit of reward.
- **A read by a part.** At every moment the brain reads its records with the code of what its
  senses hold. Up to ``HEADS`` heads each take a record; a head takes part where its record is
  known (``KNOWN`` tellings, or ``PAID`` for a record of something the brain said, which only a
  reward makes deep) and its key lies within ``NEAR`` of the nearest's cosine (and above
  ``MATCH``). A field (a sense, or a field of a sense) comes back where every taking part that
  holds it holds it alike (cosine ``AGREE``), as their mean. Where a field that comes back
  disagrees with what the senses hold, the records are of something else and nothing comes
  back.
- **Completion.** A field the senses leave silent is given what came back, and the brain
  settles on it: a word heard alone brings back the thing it was heard with.
- **A voice** (``voice=sense``): the brain says, in the format of a sense it hears, what its
  records bind to what it senses (``Brain.say``): what it said and was paid for, else what it
  heard with it; with neither, a sound drawn from the sounds it holds, the deeper the likelier
  (a babble). What it says joins the moment under the sense ``voice`` when the caller passes it
  back, so the moment's record binds the sound said to what was sensed and a reward that
  follows deepens it.
- **A heard name draws the focus** (with ``surprise=True`` and a map with values): the records
  read by the heard sense alone bring back a thing's fields, and the things on the map whose
  values hold those fields alike take a salience of ``NAMED`` in the focus's competition.

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
    PAID = 3.0
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
            raise ValueError("bound is the records a brain keeps: a positive integer")
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
        # records of something said that a reward of zero or less followed
        self.refused = np.zeros(self.capacity, dtype=bool)
        self.draws = 0  # the babbles drawn, for the next draw's generator
        self.said_from: list[str | None] = []  # where each stream's latest said sound came from
        self.recalled: np.ndarray | None = None  # the latest completion's rows

    # -- what a description and a file hold

    def settings(self) -> dict[str, Any]:
        return {"records": self.capacity, "voice": self.voice, "heard": self.heard,
                "seed": self.seed}

    def to_dict(self) -> dict[str, Any]:
        return {**self.settings(), "draws": int(self.draws)}

    def arrays(self) -> dict[str, np.ndarray]:
        out = {f"store/{name}": value for name, value in self.store.arrays().items()}
        out["written"] = self.written.copy()
        out["refused"] = self.refused.copy()
        return out

    @classmethod
    def from_parts(cls, named: Mapping[str, Any], key_width: int, inputs: int,
                   record: Mapping[str, Any], arrays: Mapping[str, Any]) -> BoundRecords:
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
        """``_known`` of each of ``slots``."""
        store = self.store
        bars = np.full(len(slots), self.KNOWN)
        if self.voice is not None:
            first, last = self.fields[self._voice_field()]
            bars[np.any(store.values[slots, first:last], axis=1)] = self.PAID
        known = (np.asarray(store.deep[slots], dtype=np.float64) >= bars) | self.refused[slots]
        if store.forget:
            known &= ~np.asarray(store.gone[slots], dtype=bool)
        return np.asarray(known)

    def _known(self, slot: int) -> bool:
        store = self.store
        if store.forget and bool(store.gone[slot]):
            return False
        if self.refused[slot]:
            return True
        bar = self.KNOWN
        if self.voice is not None and self._active(store.values[slot], self._voice_field()):
            bar = self.PAID  # something said is known once it was paid for
        return float(store.deep[slot]) >= bar

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
        """Each field the reached records hold, where most of the nearest of them hold it alike.
        For a field: the ``HEADS`` nearest records that hold it, those within ``NEAR`` of the
        nearest's cosine, fall into groups that hold it alike (cosine ``AGREE`` to a group's
        first), each weighed by its records' cosines; a refused record (something said that no
        reward followed) within that reach weighs against the group whose value it holds. The
        heaviest group comes back, as its mean, where it weighs more than ``MOST`` of the
        records that hold the field and more than nothing."""
        out: dict[str, np.ndarray] = {}
        if not taking:
            return out
        slots = np.array([slot for slot, _ in taking], dtype=np.int64)
        weights = np.array([weight for _, weight in taking])
        refused = self.refused[slots]
        values = np.asarray(self.store.values[slots], dtype=np.float64)
        for field, (first, last) in self.fields.items():
            parts = values[:, first:last]
            held = np.any(parts, axis=1)
            holding = np.flatnonzero(held & ~refused)
            if not len(holding):
                continue
            holding = holding[np.argsort(-weights[holding], kind="stable")][: self.HEADS]
            reach = self.NEAR * weights[holding[0]]
            holding = holding[weights[holding] >= reach]
            groups: list[list[int]] = []
            for k in holding:
                for group in groups:
                    if float(_unit(parts[k]) @ _unit(parts[group[0]])) >= self.AGREE:
                        group.append(int(k))
                        break
                else:
                    groups.append([int(k)])
            mass = [float(weights[g].sum()) for g in groups]
            for k in np.flatnonzero(held & refused & (weights >= reach)):
                for g, group in enumerate(groups):
                    if float(_unit(parts[k]) @ _unit(parts[group[0]])) >= self.AGREE:
                        mass[g] -= float(weights[k])
                        break
            best = int(np.argmax(mass))
            if mass[best] > 0.0 and mass[best] > self.MOST * float(weights[holding].sum()):
                out[field] = parts[groups[best]].mean(axis=0)
        return out

    def complete(self, rows: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """``rows`` with each field they leave silent given what the records bring back for the
        moment's ``codes``; the voice is left as given. Changes nothing in the records."""
        out = np.array(rows, dtype=np.float64)
        for i in range(len(out)):
            if not self.holds(out[i]):
                continue
            for field, part in self._agreed(self._taking(codes[i]), out[i]).items():
                if self.voice is not None and field == self._voice_field():
                    continue
                first, last = self.fields[field]
                if not np.any(out[i, first:last]):
                    out[i, first:last] = part
        self.recalled = out
        return out

    def recall(self, row: np.ndarray, code: np.ndarray) -> dict[str, np.ndarray]:
        """What the records bring back for one moment: each field held alike, as its mean."""
        return self._agreed(self._taking(code), np.asarray(row, dtype=np.float64))

    # -- the voice

    def say(self, rows: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """What the brain says at each row, in the format of the voice: what its records bind to
        what it senses (what it said and was paid for, else the heard sense's record where the
        senses hold no sound), else a sound drawn from the sounds the records hold, the deeper
        the likelier; zeros where the records hold no sound. ``said_from`` names each choice:
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
        paid = np.flatnonzero(said & ~self.refused[slots])
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

    def hear(self, rows: np.ndarray, codes: np.ndarray, surprise: Any = None) -> None:
        """The moment's records: each row that holds something written under its code (a moment
        told again deepens its record), with the moment's surprise in tellings; then the
        records fade by one moment."""
        rows = np.asarray(rows, dtype=np.float64)
        written = np.full(len(rows), -1, dtype=np.int64)
        for i in range(len(rows)):
            if not self.holds(rows[i]) or not np.any(codes[i]):
                continue
            value = np.zeros(self.inputs)
            for first, last in self.fields.values():
                value[first:last] = rows[i, first:last]
            told = None if surprise is None else float(max(surprise[i], 0.0))
            written[i], revised = self.store.write(np.asarray(codes[i], dtype=np.float64), value,
                                                   surprise=told)
            if not revised:
                self.refused[written[i]] = False
        self.written = written
        self.store.moment()

    def pay(self, reward: Any) -> None:
        """A reward for the latest moment: each stream's record of it deepens by ``PAY``
        tellings a unit of reward; a record of something said that a reward of zero or less
        follows is refused (it weighs against what it holds in a read, until a reward pays
        it)."""
        r = np.asarray(reward, dtype=np.float64).reshape(-1)
        said = None
        if self.voice is not None:
            first, last = self.fields[self._voice_field()]
            said = (first, last)
        for i, slot in enumerate(self.written):
            if slot < 0 or i >= len(r) or not np.isfinite(r[i]):
                continue
            if r[i] > 0:
                self.store.retell(int(slot), surprise=max(self.PAY * float(r[i]) - 1.0, 0.0))
                self.refused[slot] = False
            elif said is not None and np.any(self.store.values[slot, said[0]:said[1]]):
                self.refused[slot] = True

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
