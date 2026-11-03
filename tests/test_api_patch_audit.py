"""Effective expert settings and the custody of saved pending actions."""

from dataclasses import replace

import numpy as np
import pytest
from checkpoints import earlier

import cadence as cd


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def same_arrays(left, right):
    assert left.keys() == right.keys()
    for name, value in left.items():
        np.testing.assert_array_equal(value, right[name], err_msg=name)


@pytest.mark.parametrize("learned", [None, 8])
@pytest.mark.parametrize("legacy", [False, True])
def test_pending_checkpoint_requires_the_decision_that_owns_its_outcome(tmp_path, learned, legacy):
    brain = cd.Brain.compose(4, 2, modules=(6,), seed=3, learned_code=learned)
    observations = np.eye(4)
    brain.act(observations[[0]])
    path = brain.save(tmp_path / "pending.npz")
    saved = arrays(path)
    if legacy:
        saved = earlier(saved)
        np.savez(path, **saved)
    # An intact pending checkpoint keeps the actual action's associative write.
    twin = cd.Brain.load(path)
    brain.learn([1.0], [False], observations[[1]])
    twin.learn([1.0], [False], observations[[1]])
    assert brain.hippocampus.writes == twin.hippocampus.writes == 1
    np.testing.assert_array_equal(brain.hippocampus.consolidated, twin.hippocampus.consolidated)
    for name in ("moment/observations", "moment/action", "moment/code"):
        saved.pop(name, None)
    np.savez(path, **saved)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="missing saved state: moment/observations"):
        cd.Brain.load(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("defect", ["reordered", "other_population"])
def test_composed_checkpoint_readout_must_keep_the_motor_action_mapping(tmp_path, defect):
    brain = cd.Brain.compose(4, 2, modules=(6,), seed=3)
    brain.act(np.eye(4)[[0]])
    path = brain.save(tmp_path / "readout.npz")
    saved = arrays(path)
    saved["outputs"] = (
        brain.motor_index[::-1] if defect == "reordered" else brain.association_index[:2]
    )
    np.savez(path, **saved)
    with pytest.raises(ValueError, match="motor population"):
        cd.Brain.load(path)


def test_expert_teaching_description_keeps_effective_steps_and_is_observational(tmp_path):
    brain = cd.Brain.compose(4, 2, modules=(6,), seed=3)
    config = replace(brain.learner.config, eta=0.017, eta_bias=0.004,
                     normalize=0.4, normalize_floor=0.005)
    brain.retune(learning=config)
    brain.act(np.eye(4)[[0]])
    before = arrays(brain.save(tmp_path / "before.npz"))
    description = brain.describe()
    assert description["genes"]["teaching"] == "custom"
    assert description["genes"]["learning"] == {
        "eta": 0.017, "eta_bias": 0.004, "normalize": 0.4, "normalize_floor": 0.005,
        "normalize_relative": 0.0, "nudged_fixed": False,
    }
    description["genes"]["learning"]["eta"] = 99.0
    assert brain.learner.config == config
    same_arrays(before, arrays(brain.save(tmp_path / "after.npz")))
    restored = cd.Brain.load(tmp_path / "after.npz")
    assert restored.describe() == brain.describe()
    restored.retune(learning=replace(config, eta=0.019))
    assert restored.describe() != brain.describe()
    np.testing.assert_array_equal(
        brain.step(np.eye(4)[[1]], reward=[0.5]),
        restored.step(np.eye(4)[[1]], reward=[0.5]),
    )  # the described teaching rate does not govern reward learning


@pytest.mark.parametrize("point", ["founder", "normalized"])
def test_named_teaching_points_keep_their_compact_description(point):
    brain = cd.Brain.compose(4, 2, modules=(6,), teaching=point)
    genes = brain.describe()["genes"]
    assert genes["teaching"] == point and "learning" not in genes
