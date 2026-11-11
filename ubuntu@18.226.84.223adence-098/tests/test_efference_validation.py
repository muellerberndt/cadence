"""Efference boundaries stay silent until a command and retain valid saved custody."""

import json

import checkpoints
import numpy as np
import pytest
from checkpoints import flat, write

import cadence as cd

DRIVE = np.array([[1.0, 0.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    "tamper",
    [
        "redirected_target",
        "wrong_command_width",
        "negative_amplitude",
        "boolean_amplitude",
        "downgraded_with_arrays",
        "downgraded_without_arrays",
    ],
)
def test_saved_efference_cannot_change_ports_or_silently_lose_its_state(tmp_path, tamper):
    brain = cd.Brain.compose(4, 2, modules=(8,), episodic=False, efference_amplitude=1.0, recall=False)
    brain.act(DRIVE, greedy=True)
    path = brain.save(tmp_path / "original")
    with np.load(path, allow_pickle=False) as data:
        arrays = flat({name: data[name].copy() for name in data.files})
    metadata = checkpoints.metadata(arrays)
    if tamper == "redirected_target":
        metadata["efference"]["target"] = "motor"
    elif tamper == "wrong_command_width":
        metadata["efference"].update(source="association", target="prefrontal")
        for name in ("trace", "last"):
            arrays["efference/" + name] = np.zeros((1, 8))
    elif tamper == "negative_amplitude":
        metadata["efference"]["amplitude"] = -1.0
    elif tamper == "boolean_amplitude":
        metadata["efference"]["amplitude"] = True
    else:
        del metadata["efference"]
        if tamper == "downgraded_without_arrays":
            for name in ("trace", "last", "cold"):
                del arrays["efference/" + name]
    arrays["generic"] = np.array(json.dumps(metadata))
    broken = tmp_path / "broken.npz"
    write(broken, arrays)
    with pytest.raises(ValueError, match="efference|memory's settings"):
        cd.Brain.load(broken)


def test_orphaned_efference_arrays_are_not_ignored(tmp_path):
    brain = cd.Brain.compose(4, 2, modules=(8,), episodic=False, recall=False)
    path = brain.save(tmp_path / "original")
    with np.load(path, allow_pickle=False) as data:
        arrays = flat({name: data[name].copy() for name in data.files})
    arrays["efference/trace"] = np.zeros((0, 2))
    broken = tmp_path / "broken.npz"
    write(broken, arrays)
    with pytest.raises(ValueError, match="saved memory arrays require their memory's settings"):
        cd.Brain.load(broken)
