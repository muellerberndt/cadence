"""Opt-in recordings of the actual settling steps, for inspection and replay."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .brain import Brain, BrainState, Nudge
    from .connectome import Connectome
    from .neuron import NeuronModel

__all__ = ["SettlementRecord", "record_settlements"]


def _copy(value: np.ndarray) -> np.ndarray:
    out = np.array(value, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class SettlementRecord:
    """A complete call, with the initial state followed by every actual repair.

    State arrays are ``(steps + 1, batch, neurons)``. Their differences give
    signed potential repairs and activation changes without re-simulating a
    trajectory. Weights are effective synaptic weights in connectome order.
    Arrays own their storage; subsequent learning cannot alter the recording.
    ``label`` is supplied by the application, not inferred from brain activity.
    """

    label: str
    connectome: Connectome
    neuron_model: NeuronModel
    drive: np.ndarray
    mask: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    nudge: Nudge | None
    potential: np.ndarray
    activation: np.ndarray
    adaptation: np.ndarray
    steps: int


_observer: ContextVar[tuple[Callable[[SettlementRecord], None], str] | None] = ContextVar(
    "cadence_settlement_observer", default=None
)


@contextmanager
def record_settlements(
    callback: Callable[[SettlementRecord], None], *, label: str = ""
) -> Iterator[None]:
    """Send each actual settlement in this context to ``callback`` once.

    This includes settlements inside learning, reward updates and imagination.
    The callback runs synchronously after a call, with recording suspended so
    its own diagnostic computations cannot recursively record themselves.
    Nested contexts replace the outer callback and restore it on exit. No
    background thread or unbounded global history is created.

    Recording costs memory and device-to-host copies at every iteration. CPU
    recordings use the inspectable NumPy path instead of the fused kernel;
    arithmetic can differ at round-off. Recordings describe the actual observed
    run. They are not reconstructions of an earlier unrecorded run.
    """
    if not callable(callback):
        raise TypeError("callback must be callable")
    token = _observer.set((callback, str(label)))
    try:
        yield
    finally:
        _observer.reset(token)


class _Capture:
    def __init__(self, observer: tuple[Callable[[SettlementRecord], None], str]) -> None:
        self.callback, self.label = observer
        self.states: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def step(self, v: np.ndarray, activation: np.ndarray, adaptation: np.ndarray) -> None:
        self.states.append((_copy(v), _copy(activation), _copy(adaptation)))

    def finish(
        self,
        brain: Brain,
        drive: np.ndarray,
        mask: np.ndarray,
        nudge: Nudge | None,
        state: BrainState,
    ) -> None:
        potential, activation, adaptation = (
            _copy(np.stack([row[i] for row in self.states])) for i in range(3)
        )
        record = SettlementRecord(
            self.label,
            brain.connectome,
            brain.neuron_model,
            _copy(drive),
            _copy(mask),
            _copy(brain.weights),
            _copy(brain.bias),
            copy.deepcopy(nudge),
            potential,
            activation,
            adaptation,
            state.steps,
        )
        token = _observer.set(None)
        try:
            self.callback(record)
        finally:
            _observer.reset(token)


def _capture() -> _Capture | None:
    observer = _observer.get()
    return None if observer is None else _Capture(observer)
