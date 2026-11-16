"""Files the released versions saved load in this one and continue as their release continued
them; a file that needs what this version lacks is refused by name; and a brain without this
version's new options saves a file the previous release reads.

The files, and each brain's continuation after a load as its release read it, are written by
``tests/fixtures/released/make.py`` from each release's own source (``panel.py`` holds the
brains, their lives and their continuations).
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence import brain as kernel

HERE = Path(__file__).resolve().parent / "fixtures" / "released"
RELEASED = json.loads((HERE / "released.json").read_text())
SOLVER = "fused" if kernel._FUSED else "numpy"

_spec = importlib.util.spec_from_file_location("released_panel", HERE / "panel.py")
assert _spec is not None and _spec.loader is not None
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)

# What 0.100.0, the release before this one, reads (its ``Brain.load``)
PREVIOUS = "0.100.0"
PREVIOUS_FORMATS = frozenset(
    {"cadence-brain/1", "cadence-brain/2", "cadence-brain/3", "cadence-brain/4", "cadence-brain/5"}
)
PREVIOUS_FEATURES = frozenset({"strength", "heads", "recall_strength", "surprise", "map"})

FILES = [
    pytest.param(version, name, id=f"{version}-{name}")
    for version, entry in sorted(RELEASED.items())
    for name in sorted(entry["brains"])
]


def _load(path: Path) -> cd.Brain:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cd.Brain.load(path)


def test_the_fixtures_cover_every_released_version_from_0_99_0():
    assert set(RELEASED) == {"0.99.0", "0.99.1", PREVIOUS}
    assert set(RELEASED[PREVIOUS]["brains"]) == set(panel.PANEL)
    size = sum(path.stat().st_size for path in HERE.rglob("*") if path.is_file())
    assert size < 2_000_000


@pytest.mark.parametrize("version,name", FILES)
def test_a_released_file_loads_and_continues_as_its_release_did(version, name, tmp_path):
    entry = RELEASED[version]
    recorded = entry["brains"][name]
    path = HERE / entry["tag"] / f"{name}.npz"
    assert panel.record(path) == {"format": recorded["format"], "features": recorded["features"]}
    brain = _load(path)
    again = panel.record(brain.save(tmp_path / "again.npz"))
    assert again == panel.record(path), "a loaded file saves under another format"
    acts = panel.continuation(brain, name)
    assert acts == recorded["acts"][SOLVER]
    if SOLVER == "numpy":  # the bits were recorded on one machine's NumPy loop, as elsewhere
        assert panel.learned(brain) == recorded["learned"]["numpy"]


def _with_record(source: Path, target: Path, change) -> Path:
    """A copy of ``source`` whose brain record ``change`` edits."""
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    saved = json.loads(str(arrays["brain"]))
    change(saved)
    arrays["brain"] = np.array(json.dumps(saved))
    np.savez_compressed(target, **arrays)
    return target


def test_a_file_naming_a_feature_this_version_lacks_is_refused_by_name(tmp_path):
    source = HERE / "v0.100.0" / "surprise.npz"

    def later(saved):
        saved["features"] = sorted([*saved["features"], "telepathy"])

    with pytest.raises(ValueError, match="'telepathy'"):
        cd.Brain.load(_with_record(source, tmp_path / "later.npz", later))


F3 = (
    "a file of a later format name is refused as an 'unsupported Brain checkpoint format', "
    "without the name it carries"
)


@pytest.mark.xfail(strict=True, reason=F3)
def test_a_file_of_a_later_format_is_refused_by_its_name(tmp_path):
    source = HERE / "v0.100.0" / "default.npz"

    def later(saved):
        saved["format"] = "cadence-brain/6"

    with pytest.raises(ValueError) as error:
        cd.Brain.load(_with_record(source, tmp_path / "later.npz", later))
    assert "cadence-brain/6" in str(error.value)


@pytest.mark.parametrize("name", sorted(panel.PANEL))
def test_a_brain_without_the_new_options_saves_what_the_previous_release_reads(name, tmp_path):
    brain = panel.compose(name)
    panel.life(brain, name)
    saved = panel.record(brain.save(tmp_path / f"{name}.npz"))
    assert saved["format"] in PREVIOUS_FORMATS
    assert set(saved["features"]) <= PREVIOUS_FEATURES
    assert saved == {
        "format": RELEASED[PREVIOUS]["brains"][name]["format"],
        "features": RELEASED[PREVIOUS]["brains"][name]["features"],
    }


def test_a_brain_with_bound_records_names_the_feature_the_previous_release_refuses(tmp_path):
    from cadence.senses import Ear, Things

    things = Things({"hue": (0.0, 1.0)}, per=8)
    ear = Ear(frames=4, bands=8)
    brain = cd.Brain.compose(
        senses={"eyes": things.widths, "ear": ear.width},
        actions=2,
        code=64,
        bound=16,
        voice="ear",
        modules=(8,),
        seed=0,
        backend="cpu",
    )
    brain.act({"eyes": things.rows({"hue": [0.4]})}, greedy=True)
    saved = panel.record(brain.save(tmp_path / "bound.npz"))
    assert saved["format"] == "cadence-brain/5"
    assert "bound" in saved["features"] and "bound" not in PREVIOUS_FEATURES


def test_the_previous_release_job_saves_brains_that_continue_as_recorded(tmp_path):
    """The CI job's script (tests/compat/previous_release.py): what it saves, this version
    reads back with the continuation it recorded."""
    spec = importlib.util.spec_from_file_location(
        "previous_release", Path(__file__).resolve().parent / "compat" / "previous_release.py"
    )
    assert spec is not None and spec.loader is not None
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    job.save(tmp_path)
    table = json.loads((tmp_path / "saved.json").read_text())
    assert set(table["brains"]) == set(panel.PANEL)
    for name in ("default", "records", "map_values"):
        brain = _load(tmp_path / f"{name}.npz")
        assert panel.continuation(brain, name) == table["brains"][name]["acts"]
    assert "bound" in panel.record(tmp_path / "refused-bound.npz")["features"]
