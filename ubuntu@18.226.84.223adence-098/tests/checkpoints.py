"""A brain checkpoint in the layout the reader validates, for the tests that read one or corrupt it.

``Brain.save`` writes one group ``memories/<name>/...`` per memory and the brain's metadata in the
``brain`` member, and ``Brain.load`` validates it in a flat layout: each memory's settings at the top
of the metadata and its arrays under a prefix of its own (``_flat_layout``). ``flat`` gives a test
that layout, under the member ``generic``, and ``write`` writes it back as a checkpoint.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cadence.generic import _MEMORY_PREFIXES, _flat_layout

# the top-level members of the flat layout a saved brain holds apart
_EPISODIC_PREFIXES = {
    "temporal/": "memories/episodic/context_memory/",
    "paid/": "memories/episodic/paid_memory/",
    "episode/": "memories/episodic/",
}


def flat(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """A checkpoint's members in the layout the reader validates; a file of that layout already
    is returned as it is."""
    members = {name: arrays[name] for name in arrays}
    if "brain" not in members:
        return members
    meta, data = _flat_layout(json.loads(str(members["brain"])), members)
    return {**data, "generic": np.array(json.dumps(meta))}


def metadata(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """The brain's metadata in the layout the reader validates."""
    return json.loads(str(flat(arrays)["generic"]))


def saved(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Members in the flat layout as the members of a checkpoint: the inverse of ``flat``."""
    members = {name: arrays[name] for name in arrays}
    if "generic" not in members:
        return members
    meta = json.loads(str(members.pop("generic")))
    memories = {}
    for name in (*_MEMORY_PREFIXES, "episodic"):
        if meta.get(name) is not None:
            memories[name] = meta[name]
        meta.pop(name, None)
    if "reward" in meta:
        meta["actor"] = meta.pop("reward")
    meta["memories"] = memories
    out: dict[str, Any] = {"brain": np.array(json.dumps(meta))}
    prefixes = {prefix: f"memories/{group}/" for group, prefix in _MEMORY_PREFIXES.items()}
    prefixes.update(_EPISODIC_PREFIXES)
    # the longest prefix first: an episodic array is its store's, not the store's group
    ordered = sorted(prefixes, key=len, reverse=True)
    for name, value in members.items():
        target = next(
            (prefixes[prefix] + name[len(prefix):] for prefix in ordered if name.startswith(prefix)),
            name,
        )
        out[target] = value
    return out


def write(path: str | Path, arrays: Mapping[str, Any]) -> None:
    """Write members, in the flat layout or a checkpoint's own, as a checkpoint file."""
    np.savez(path, **saved(arrays))
