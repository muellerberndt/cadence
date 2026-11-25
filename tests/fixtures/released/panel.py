"""The panel of brains whose files a release saved: each brain's composition, the life it lived
before the save and the continuation read after a load. The file runs under the release that
writes the fixtures (``make.py``) and under the version that reads them
(``tests/test_released_files.py``), so it uses only calls every release on the panel holds."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

import cadence as cd

SENSES = {"eyes": {"hue": 8, "shape": 8}, "ear": 16}
EAR = 16  # the width of the bound brain's heard sense and of its voice


def version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3])


def small_map(**options: Any) -> Any:
    from cadence.space.map import Map

    return Map((-1.0, 1.0, -1.0, 1.0), (6, 6), units=4, **options)


# name: (the first release that composes it, its keywords); the map brains are built by
# ``compose`` below. Every brain but the default holds a recall memory of 16 slots, to keep the
# files small.
PANEL: dict[str, tuple[str, dict[str, Any]]] = {
    "default": ("0.99.0", {"inputs": 8}),
    "arousal": ("0.99.0", {"inputs": 8, "arousal": True}),
    "code": ("0.99.0", {"inputs": 40, "code": 64}),
    "heads": ("0.99.0", {"inputs": 8, "recall_heads": 2}),
    "choice": ("0.99.0", {"inputs": 8, "memory_choice": True, "arousal": True}),
    "maturity": ("0.99.0", {"inputs": 8, "arousal": True, "maturity": True, "memory_fade": 1.2e-4}),
    "records": (
        "0.99.0",
        {
            "inputs": 15,
            "message": 24,
            "message_boundary": 14,
            "message_boundary_holds": True,
            "longterm": 16,
        },
    ),
    "map_by_hand": ("0.99.0", {}),
    "senses": ("0.100.0", {"senses": SENSES, "code": 64}),
    "map_named": ("0.100.0", {"senses": {"touch": 8}}),
    "map_values": ("0.100.0", {"senses": {"touch": 8}}),
    "surprise": ("0.100.0", {"inputs": 8, "surprise": True}),
    "recall_strength": ("0.100.0", {"inputs": 8, "recall_strength": True}),
    "records_strength": (
        "0.100.0",
        {
            "inputs": 15,
            "message": 24,
            "message_boundary": 14,
            "message_boundary_holds": True,
            "longterm": 16,
            "longterm_strength": True,
        },
    ),
    "bound": (
        "0.101.0",
        {"senses": {"eyes": {"hue": 8, "shape": 8}, "ear": EAR}, "code": 64, "bound": 16,
         "voice": "ear"},
    ),
    "cortex_reads": ("0.102.0", {"inputs": 40, "code": 64, "cortex_reads": "both"}),
    # the learned value path of one motor slot, on the panel from the files of 0.103.1 on: the
    # three cues are the three actions' tokens, and the life's lessons and rewards step its gains
    "path": ("0.103.1", {"inputs": 8, "recall_gain": 1.0, "recall_tokens": [0, 1, 2]}),
}


# a patch release writes the files of its minor release: a few of them, one a format
PATCH = {"0.99.1": ("default", "arousal", "maturity", "records", "map_by_hand")}


def names(release: str) -> list[str]:
    held = [name for name, (first, _) in PANEL.items() if version(release) >= version(first)]
    return [name for name in held if name in PATCH.get(release, held)]


def compose(name: str) -> Any:
    options = dict(PANEL[name][1])
    common = {"modules": (8,), "seed": 1, "backend": "cpu"}
    if name != "default":
        common["recall_capacity"] = 16
    if name == "map_by_hand":
        held = small_map()
        brain = cd.Brain.compose(held.count + held.units, 3, **common)
        brain.space = held
        return brain
    if name == "map_named":
        return cd.Brain.compose(actions=3, trained=[small_map()], **common, **options)
    if name == "map_values":
        held = small_map(values={"hue": 6}, surprise=True)
        return cd.Brain.compose(actions=3, trained=[held], surprise=True, **common, **options)
    return cd.Brain.compose(actions=3, **common, **options)


def observation(brain: Any, name: str, moment: int) -> Any:
    """The moment's observation, and what the eyes tell a held map."""
    rng = np.random.default_rng((11, moment))
    space = getattr(brain, "space", None)
    if space is not None:
        place = [0.2 * (moment % 5) - 0.4, 0.3]
        if name == "map_values":
            space.moment(seen=[(1, place, {"hue": moment % 6})])
        else:
            space.moment(seen=[(1, place)])
        if name == "map_by_hand":
            return space.observation()[None]
        return {"touch": np.eye(8)[[moment % 8]]}
    if name == "senses":
        row = rng.random((1, 32))
        return {"eyes": {"hue": row[:, :8], "shape": row[:, 8:16]}, "ear": row[:, 16:]}
    if name == "bound":  # one of three things, heard named at every other moment
        thing = moment % 3
        seen = {"hue": np.eye(8)[[thing]], "shape": np.eye(8)[[thing + 3]]}
        return {"eyes": seen, "ear": np.eye(EAR)[[thing]]} if moment % 2 else {"eyes": seen}
    row = 0.2 * rng.random((1, brain.inputs))
    row[0, moment % 3] = 1.0  # one of three cues, the label a lesson gives it
    if name.startswith("records"):
        row[0, 14] = 1.0 if moment % 4 == 2 else 0.0  # a turn every fourth moment
    return row


def moment(brain: Any, name: str, at: int, reward: float | None, greedy: bool) -> int:
    x = observation(brain, name, at)
    if greedy:
        out = brain.act(x, greedy=True)
    elif brain.arousal is not None:
        out = brain.live(x, reward=None if reward is None else [reward])
    else:
        out = brain.step(x, reward=None if reward is None else [reward])
    return int(np.asarray(out).ravel()[0])


def life(brain: Any, name: str) -> list[int]:
    """The life before the save: a lesson on each of nine moments, greedy acts, then sampled
    moments with rewards. A sampled action awaits its outcome when the brain is saved."""
    for t in range(9):
        brain.lesson(observation(brain, name, t), np.array([t % 3]))
    plan = [(None, True), (None, True), (None, False), (1.0, False), (0.0, False), (None, False)]
    return [moment(brain, name, 9 + t, reward, greedy) for t, (reward, greedy) in enumerate(plan)]


def continuation(brain: Any, name: str) -> list[int]:
    """The life after a load: the outcome of the pending action, a paid moment, a greedy act."""
    plan = [
        (1.0, False),
        (0.5, False),
        (None, True),
        (None, False),
        (1.0, False),
        (None, True),
        (None, True),
        (None, True),
    ]
    return [moment(brain, name, 15 + t, reward, greedy) for t, (reward, greedy) in enumerate(plan)]


def learned(brain: Any) -> str:
    """SHA-256 of what the life taught: every synapse's efficacy and every neuron's bias."""
    h = hashlib.sha256()
    for array in (brain.learner.brain.efficacy, brain.learner.brain.bias):
        h.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return h.hexdigest()


def record(path: Any) -> dict[str, Any]:
    """The format and features a checkpoint names."""
    with np.load(path, allow_pickle=False) as archive:
        saved = json.loads(str(archive["brain"]))
    return {"format": saved.get("format"), "features": saved.get("features", [])}
