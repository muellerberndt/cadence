"""A map or module saved where a file is: that file stays whole until the new one is complete."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cadence as cd
from cadence.space import Interface, Space
from cadence.space.map import Map

PLANE = (-1.0, 1.0, -0.2, 1.4)
SMALL = dict(place=8, moved=5, motion=9, counted=3, unseen=4, heard=(6, 4), felt=4, hidden=16,
             shift=12, spread=6, found=8, keep=4)


def a_map(path: Any, card: dict[str, int]) -> Path:
    held = Map(PLANE, cells=(4, 4), units=4)
    held.moment(seen=[(card["round"], [0.2, 0.5])])
    return held.save(path, card=card)


def a_brains_module(path: Any, card: dict[str, int]) -> Path:
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=card["round"], backend="cpu")
    return brain.save_module(path, name="example", inputs=("sensory",), published=("motor",),
                             card=card)


def a_module_by_the_function(path: Any, card: dict[str, int]) -> Path:
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=card["round"], backend="cpu")
    return cd.modules.save(brain, path, name="example", inputs=("sensory",),
                           published=("motor",), card=card)


def a_space(path: Any, card: dict[str, int]) -> Path:
    space = Space(Interface(plane=PLANE, **SMALL), seed=card["round"], backend="cpu")
    saved: Path = space.save(path, card=card)
    return saved


def round_of(path: Path) -> int:
    """The round on the card of the file at ``path``, read by loading the whole file."""
    held = cd.load_module(path)  # every array against its checksum
    card = Map.card_of(path) if isinstance(held, Map) else held.card
    return int(card["round"])


WRITERS = {"Map.save": a_map, "Brain.save_module": a_brains_module,
           "modules.save": a_module_by_the_function, "Space.save": a_space}


@pytest.mark.parametrize("name", ["held.npz", "held", "held.map"])
@pytest.mark.parametrize("write", WRITERS.values(), ids=WRITERS)
def test_a_save_that_fails_keeps_the_file_there_and_the_next_save_replaces_it(
        write, name, tmp_path, monkeypatch):
    path = tmp_path / name
    assert write(path, {"round": 1}) == path  # the path as given: no suffix is added
    assert list(tmp_path.iterdir()) == [path]
    before = path.read_bytes()

    def fail(handle, **arrays):
        handle.write(b"part of an archive")
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(np, "savez_compressed", fail)
        with pytest.raises(OSError, match="disk full"):
            write(path, {"round": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]  # and no part of the failed file beside it
    assert round_of(path) == 1
    assert write(str(path), {"round": 2}) == path
    assert list(tmp_path.iterdir()) == [path] and round_of(path) == 2
    with pytest.raises(FileNotFoundError):  # no folder is made for the file
        write(tmp_path / "absent" / name, {"round": 3})
    assert list(tmp_path.iterdir()) == [path]
