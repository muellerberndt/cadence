"""A selectable motor circuit preserves the continuing composition around it."""

import numpy as np
import pytest

import cadence as cd


def test_explicit_default_motor_lateral_preserves_the_complete_default_composition(tmp_path):
    implicit = cd.Brain.compose(2, 3, modules=(5, 4), observers=(3,), seed=7, recall=False)
    explicit = cd.Brain.compose(2, 3, modules=(5, 4), observers=(3,), lateral=0.0, seed=7, recall=False)
    for name in ("pre", "post", "count", "sign"):
        np.testing.assert_array_equal(
            getattr(implicit.connectome, name), getattr(explicit.connectome, name),
        )
    first, second = implicit.save(tmp_path / "implicit"), explicit.save(tmp_path / "explicit")
    with np.load(first, allow_pickle=False) as aa, np.load(second, allow_pickle=False) as bb:
        assert set(aa.files) == set(bb.files)
        for name in aa.files:
            np.testing.assert_array_equal(aa[name], bb[name], err_msg=name)


@pytest.mark.parametrize("lateral", [0., -0.125, 0.1])
def test_motor_lateral_is_the_requested_circuit_with_reciprocal_feedback(lateral):
    brain = cd.Brain.compose(2, 4, modules=(5, 3), observers=(2,), lateral=lateral, seed=3, recall=False)
    graph = brain.connectome
    motor = brain.motor_index
    mask = np.isin(graph.pre, motor) & np.isin(graph.post, motor)
    assert int(mask.sum()) == (0 if lateral == 0 else 4 * 3)
    np.testing.assert_array_equal(graph.sign[mask], np.full(mask.sum(), lateral))
    assert not np.any(graph.pre[mask] == graph.post[mask])
    edges = set(zip(graph.pre, graph.post, strict=True))
    for region in (brain.association_index, graph.populations["observer_0"]):
        assert all((a, b) in edges and (b, a) in edges for a in motor for b in region)
    assert isinstance(brain.working_memory, cd.Trace)
    assert isinstance(brain.hippocampus, cd.SynapticMemory)


def test_lateral_zero_qualified_teaching_and_saved_pending_feedback_continue(tmp_path):
    brain = cd.Brain.compose(2, 3, modules=(5, 4), lateral=0, seed=3,
        learning=cd.LearnerConfig(
            qualified=True, free_steps=512, nudged_steps=512, tolerance=3e-3,
        ), recall=False,
    )
    observed = np.array([[0.3, 0.], [0., 0.3]])
    brain.step(observed, teacher=np.array([0, 1]))
    assert brain.last_learning["demonstration_qualified"] == 1
    clone = cd.Brain.load(brain.save(tmp_path / "pending"))
    feedback = {"reward": np.array([0.2, -0.1]), "done": np.zeros(2, bool),
                "teacher": np.array([1, 2])}
    np.testing.assert_array_equal(brain.step(observed, **feedback), clone.step(observed, **feedback))
    with np.load(brain.save(tmp_path / "live"), allow_pickle=False) as aa, np.load(
        clone.save(tmp_path / "restored"), allow_pickle=False,
    ) as bb:
        assert set(aa.files) == set(bb.files)
        for name in aa.files:
            np.testing.assert_array_equal(aa[name], bb[name], err_msg=name)


@pytest.mark.parametrize("lateral", [np.nan, np.inf, -np.inf, True, np.bool_(False), "zero", [0]])
def test_invalid_motor_lateral_is_rejected(lateral):
    with pytest.raises(ValueError, match="lateral must be a finite signed weight"):
        cd.Brain.compose(2, 3, lateral=lateral, recall=False)
