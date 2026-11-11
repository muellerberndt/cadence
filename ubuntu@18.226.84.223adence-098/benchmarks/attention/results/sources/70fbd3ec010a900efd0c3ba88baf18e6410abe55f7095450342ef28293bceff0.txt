"""Brain regions: functional parts of one brain, blank or designed.

A ``Region`` is a named group of neurons. A blank region is only a size: a ``Genome`` gives
it projections to and from other regions, and evolution may change the size. A designed
region carries its own circuit, a ``Connectome`` of its neurons, the synapses among them and
named populations, and keeps its size under mutation. ``inputs`` names the population that
receives projections and ``outputs`` the population that sends them; either defaults to the
whole region.

Every region of a brain settles in the same equilibrium as every other region: a region is
a part of one connectome and one ``Brain``, never a separately settled model.

The standard regions are engineering analogues named for their intended functions:

* ``visual_cortex``: a retinotopic input layer and feature maps of neurons with local
  receptive fields, as in primary visual cortex.
* ``cortex``: a cortical area, optionally with lateral inhibition among its neurons; the
  association cortex of a generic brain.
* ``motor_cortex``: one neuron per action, optionally with lateral inhibition, read out as a
  choice.
* ``prefrontal_cortex``: a working-memory population with one neuron per neuron of the region
  it holds, driven by a ``Trace`` of that region's activity.

Two parts of a brain keep their state outside the connectome. The basal ganglia
(``ActorCritic``) read cortex through learned corticostriatal weights and broadcast a dopamine
prediction error. The hippocampus (``FastSynapses``) keeps one-trial associations per stream.
``Brain`` composes the regions, the basal ganglia and the hippocampus into one ready brain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .connectome import Connectome

__all__ = ["Region", "cortex", "motor_cortex", "prefrontal_cortex", "visual_cortex"]


@dataclass(frozen=True)
class Region:
    """A named group of neurons: blank (``size`` only) or designed (a ``circuit``)."""

    name: str
    size: int = 0
    circuit: Connectome | None = None
    inputs: str | None = None
    outputs: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "/" in self.name:
            raise ValueError("a region needs a nonempty name without '/'")
        if self.circuit is None:
            if isinstance(self.size, bool) or not isinstance(self.size, (int, np.integer)):
                raise ValueError("a blank region needs an integer size")
            if self.size < 1:
                raise ValueError("a blank region needs at least one neuron")
            if self.inputs is not None or self.outputs is not None:
                raise ValueError("a blank region has no populations to name as inputs or outputs")
            object.__setattr__(self, "size", int(self.size))
            return
        if self.size not in (0, self.circuit.n):
            raise ValueError("a designed region's size is its circuit's neuron count")
        object.__setattr__(self, "size", int(self.circuit.n))
        for side in (self.inputs, self.outputs):
            if side is not None and side not in self.circuit.populations:
                raise ValueError(f"population {side!r} is not in the circuit of {self.name!r}")

    @property
    def designed(self) -> bool:
        return self.circuit is not None

    def neurons(self, population: str | None = None) -> tuple[int, ...]:
        """Region-local indices of ``population``, or of every neuron when it is ``None``."""
        if population is None:
            return tuple(range(self.size))
        if self.circuit is None or population not in self.circuit.populations:
            raise KeyError(f"{self.name}/{population}")
        return self.circuit.populations[population]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "size": self.size}
        if self.circuit is not None:
            out.update(
                circuit=self.circuit.label,
                digest=self.circuit.digest(),
                inputs=self.inputs,
                outputs=self.outputs,
            )
        return out


def _lateral(
    n: int, weight: float, groups: Sequence[int] | None = None
) -> tuple[list[int], list[int], list[float]]:
    """Every ordered pair of distinct neurons, within each group when ``groups`` is given."""
    if weight == 0.0 or n < 2:
        return [], [], []
    pre: list[int] = []
    post: list[int] = []
    start = 0
    for size in groups or [n]:
        index = np.arange(start, start + size)
        a, b = np.meshgrid(index, index, indexing="ij")
        keep = a.ravel() != b.ravel()
        pre.extend(int(i) for i in a.ravel()[keep])
        post.extend(int(i) for i in b.ravel()[keep])
        start += size
    return pre, post, [weight] * len(pre)


def slot_sizes(actions: int, slots: int | Sequence[int]) -> list[int]:
    """The motor slots: ``slots`` equal groups, or one size per group, covering ``actions``.

    One slot is one choice over every action. Several slots settle together and are read as
    one softmax each: an action with several parts has one slot per part.
    """
    if isinstance(slots, (bool, np.bool_, str, bytes)):
        raise ValueError("slots is a count of equal groups or a sequence of group sizes")
    if isinstance(slots, (int, np.integer)):
        if slots < 1 or actions % int(slots):
            raise ValueError("slots must divide the number of actions")
        return [actions // int(slots)] * int(slots)
    try:
        items = list(slots)
    except TypeError as exc:
        raise ValueError("slots is a count of equal groups or a sequence of group sizes") from exc
    if not items or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1
        for k in items
    ):
        raise ValueError("slot sizes must be positive integers")
    sizes = [int(k) for k in items]
    if sum(sizes) != actions:
        raise ValueError("slot sizes must add up to the number of actions")
    return sizes


def cortex(size: int, *, lateral: float = 0.0, name: str = "association") -> Region:
    """A cortical area of ``size`` neurons.

    With ``lateral`` zero it is a blank region whose size a genome may evolve. A nonzero
    ``lateral`` connects every pair of its neurons both ways with that weight and fixes the
    size; a negative weight is lateral inhibition, a subtractive competition. No cortical
    layers are modeled.
    """
    Region(name, size)  # validate capacity even when a designed lateral circuit is requested
    if not np.isfinite(lateral):
        raise ValueError("lateral must be finite")
    if lateral == 0.0:
        return Region(name, size)
    pre, post, sign = _lateral(size, lateral)
    circuit = Connectome.from_synapses(
        size, pre=pre, post=post, sign=sign, label=f"cortex:{size}:lateral{lateral:g}"
    )
    return Region(name, circuit=circuit)


def motor_cortex(
    actions: int, *, lateral: float = 0.0, name: str = "motor", slots: int | Sequence[int] = 1
) -> Region:
    """One neuron per action, with optional lateral inhibition; population ``actions``.

    A body or an environment reads the choice from the most active neuron or from a softmax
    over them.
    With several ``slots`` (see ``slot_sizes``) the lateral inhibition stays within each slot:
    the slots are separate competitions that settle together.
    """
    if isinstance(actions, bool) or not isinstance(actions, (int, np.integer)) or actions < 1:
        raise ValueError("actions must be a positive integer")
    if not np.isfinite(lateral):
        raise ValueError("lateral must be finite")
    sizes = slot_sizes(int(actions), slots)
    pre, post, sign = _lateral(actions, lateral, sizes if len(sizes) > 1 else None)
    circuit = Connectome.from_synapses(
        actions,
        pre=pre,
        post=post,
        sign=sign,
        populations={"actions": range(actions)},
        label=f"motor-cortex:{actions}",
    )
    return Region(name, circuit=circuit)


def prefrontal_cortex(holds: Region | int, *, name: str = "prefrontal") -> Region:
    """A working-memory population with one neuron per neuron of the region it holds.

    It receives no synapses; a ``Trace`` with ``source`` the held region and ``target`` this
    one drives it with the fading activity of the moments before. Keep its size tied to the
    held region under mutation: ``mutate(..., tied=((held, name),))``.
    """
    size = holds.size if isinstance(holds, Region) else holds
    return Region(name, size)


def visual_cortex(
    height: int,
    width: int,
    *,
    channels: int = 1,
    features: int = 8,
    field: int = 3,
    stride: int = 1,
    init: float = 1.0,
    seed: int = 0,
    name: str = "visual",
) -> Region:
    """A retinotopic input layer and ``features`` maps of neurons with local receptive fields.

    Population ``input`` holds one neuron per pixel and channel, in the order of an image
    array ``(height, width, channels)`` flattened row by row. Population ``output`` holds the
    feature maps: every neuron of a map receives synapses from one ``field`` by ``field``
    window of the input, the windows stepping by ``stride``, so neighbouring neurons see
    neighbouring parts of the picture. Initial efficacies are random and fan-scaled; learning
    shapes each receptive field.
    """
    for label, value in (
        ("height", height),
        ("width", width),
        ("channels", channels),
        ("field", field),
        ("stride", stride),
        ("features", features),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if field > min(height, width):
        raise ValueError("field must fit within the image")
    if not np.isfinite(init) or init < 0:
        raise ValueError("init must be finite and nonnegative")
    rows = (height - field) // stride + 1
    cols = (width - field) // stride + 1
    n_in = height * width * channels
    n = n_in + features * rows * cols
    f, y, x, dy, dx, c = np.meshgrid(
        np.arange(features),
        np.arange(rows),
        np.arange(cols),
        np.arange(field),
        np.arange(field),
        np.arange(channels),
        indexing="ij",
    )
    pre = ((y * stride + dy) * width + (x * stride + dx)) * channels + c
    post = n_in + (f * rows + y) * cols + x
    rng = np.random.default_rng(seed)
    fan_in, fan_out = channels * field * field, features * field * field
    magnitude = rng.uniform(0.0, 1.0, size=pre.size) * np.sqrt(6.0 / (fan_in + fan_out)) * init
    sign = rng.choice([-1.0, 1.0], size=pre.size) * magnitude
    circuit = Connectome.from_synapses(
        n,
        pre=pre.ravel(),
        post=post.ravel(),
        sign=sign,
        populations={"input": range(n_in), "output": range(n_in, n)},
        label=f"visual-cortex:{height}x{width}x{channels}:{features}x{field}/{stride}",
    )
    return Region(name, circuit=circuit, inputs="input", outputs="output")
