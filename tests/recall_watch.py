"""What every read of a recall memory with several heads holds, checked as a brain lives.

``watch(brain)`` wraps the memory's read and its lesson and notes every read that breaks one of
these, with the read's number:

- a head's attention is finite, nonnegative, and zero at a slot its stream does not hold;
- no slot is read above one in total over the heads;
- head 0's mass is one where its stream holds a slot, and a later head's is one or everything
  the heads before it left, whichever is less;
- a later head's shares are the water-filling of its own attention under what is left
  (``reference_share``, found by bisection, apart from the library's closed form);
- the drive on a head's ports is the amplitude times its attention over the values and the
  nexts;
- a later head's lesson takes as its prior the attention the head read the moment with.

``tests/places.py`` names the backends; this module names what a read must hold on each.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reference_share(logits: np.ndarray, left: np.ndarray, mass: float) -> np.ndarray:
    """One row of a later head's shares, by bisection: ``w = min(left, lam * exp(logit))`` over
    the slots something is left of, with ``lam`` giving the head ``mass`` or everything left."""
    open_ = (left > 0) & np.isfinite(logits)
    out = np.zeros_like(left, dtype=float)
    if not open_.any():
        return out
    z, room = logits[open_], left[open_]
    target = min(float(mass), float(room.sum()))
    if target <= 0:
        return out
    if target >= room.sum() * (1 - 1e-15):
        out[open_] = room
        return out
    low = np.log(target) - (z.max() + np.log(len(z))) - 5.0
    high = float((np.log(room) - z).max()) + 5.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if np.minimum(room, np.exp(np.minimum(z + middle, 50.0))).sum() < target:
            low = middle
        else:
            high = middle
    out[open_] = np.minimum(room, np.exp(np.minimum(z + 0.5 * (low + high), 50.0)))
    return out


class Watch:
    """The reads and head lessons of one brain's recall memory, each checked as it happens."""

    def __init__(self, brain: Any, tolerance: float = 1e-9) -> None:
        self.recall, self.tolerance = brain.recall, tolerance
        self.broken: list[str] = []
        self.reads = self.lessons = 0
        self.largest_total = 0.0
        self.heads_reading = 0
        self._latest: tuple[Any, ...] | None = None
        recall = self.recall
        stimulate, row_steps = recall.stimulate, recall._row_steps

        def watched(drive: Any, observations: Any) -> Any:
            out = stimulate(drive, observations)
            self._read(np.asarray(out, dtype=float))
            return out

        def stepped(moment: Any, rows: Any, d_value: Any, d_next: Any, head: int = 0,
                    read: Any = None) -> Any:
            self._lesson(head, read)
            return row_steps(moment, rows, d_value, d_next, head, read)

        recall.stimulate = watched
        recall._row_steps = stepped

    def _note(self, text: str) -> None:
        if len(self.broken) < 20:
            self.broken.append(f"read {self.reads}: {text}")

    def _read(self, out: np.ndarray) -> None:
        recall, tolerance = self.recall, self.tolerance
        self.reads += 1
        moment = recall.last
        if recall._head_reads is not None:
            addresses, attended, _ = recall._head_reads
        else:
            attended = (recall._attended,)
            addresses = (None if moment is None else moment.address,)
        self._latest = attended
        if attended[0] is None:
            return
        held = np.asarray(recall.held, dtype=bool)
        streams, slots = held.shape
        total, left = np.zeros((streams, slots)), np.ones((streams, slots))
        has = held.any(axis=1)
        reading = 0
        for h, p in enumerate(attended):
            if p is None:
                continue
            p = np.asarray(p, dtype=float)
            if p.shape != held.shape:
                self._note(f"head {h} read {p.shape}, the bank holds {held.shape}")
                continue
            if not np.isfinite(p).all() or (p < -tolerance).any():
                self._note(f"head {h}'s attention is not finite or negative")
            if (np.abs(p[~held]) > tolerance).any():
                self._note(f"head {h} read a slot its stream does not hold")
            mass = p.sum(axis=1)
            due = has.astype(float) if h == 0 else (
                np.minimum(1.0, np.where(held, left, 0.0).sum(axis=1)) * has
            )
            if np.abs(mass - due).max() > 1e-7:
                self._note(f"head {h}'s mass is {mass.tolist()}, where {due.tolist()} is due")
            if h and moment is not None and addresses[h] is not None:
                position = recall.position_term(h, moment.inputs, moment.cortex)
                logits = np.asarray(recall._attention(
                    addresses[h], recall.keys, None if held.all() else held, position
                ), dtype=float)
                for row in range(streams):
                    wanted = reference_share(
                        logits[row], np.where(held[row], left[row], 0.0), float(has[row])
                    )
                    gap = float(np.abs(wanted - p[row]).max())
                    if gap > 1e-7:
                        self._note(f"head {h}, stream {row}: {gap:.3g} off the water-filling")
            amplitude = recall.amplitude * (recall.hearing if recall.drive_gain else 1.0)
            recall_columns, next_columns = recall._head_columns[h]
            ports = [(recall_columns, recall.values)]
            if recall.next:
                ports.append((next_columns, recall.nexts))
            for columns, held_values in ports:
                wanted = amplitude * np.einsum("bm,bmc->bc", p, held_values)
                gap = float(np.abs(out[:, columns] - wanted).max())
                if gap > 1e-9 * max(1.0, float(np.abs(wanted).max())):
                    self._note(f"head {h}'s port is {gap:.3g} off its attention over the slots")
            total += p
            left = np.clip(left - p, 0.0, 1.0)
            reading += int(mass.max() > 1e-12)
        if total.size:
            self.largest_total = max(self.largest_total, float(total.max()))
            if total.max() > 1.0 + tolerance:
                self._note(f"a slot was read {total.max():.6f} in total over the heads")
        self.heads_reading = max(self.heads_reading, reading)

    def _lesson(self, head: int, read: Any) -> None:
        self.lessons += 1
        if head == 0 or self._latest is None or len(self._latest) <= head:
            return
        kept = self._latest[head]
        if read is None or kept is None or not np.array_equal(kept, read):
            self._note(f"head {head}'s lesson did not take the attention the head read with")


def watch(brain: Any, tolerance: float = 1e-9) -> Watch:
    """Check every read and head lesson of ``brain``'s recall memory from here on."""
    return Watch(brain, tolerance)
