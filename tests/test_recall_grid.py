"""A position code past the exact table at a fixed size (``recall_grid``, cadence#268): every head's
position term also reads each slot past ``recall_position`` by its age modulo each of a few coprime
numbers, the lesson steps those terms with the table's gradient over its peak, and once the slots
pass the table the slope takes no step."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

GRID = (3, 4, 5)
FLAT = {"recall_recency": 0.0}


def grid(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, modules=(16,), seed=5, recall=True, recall_founder=True, **options)


def fill(brain: cd.Brain, slots: int, seed: int = 1) -> cd.Recall:
    """``slots`` one-hot moments in two streams and a moment addressed."""
    rng = np.random.default_rng(seed)
    for _ in range(slots):
        brain.act(np.eye(6)[rng.integers(6, size=2)], greedy=True)
    brain.recall.stimulate(np.zeros((2, brain.connectome.n)), np.eye(6)[[0, 1]])
    return brain.recall


def test_the_grid_is_an_option_beside_the_position_term():
    plain = grid()
    assert plain.recall.grid == () and plain.recall.grid_maps == []
    brain = grid(recall_grid=GRID)
    width = sum(GRID)
    assert [m.shape for m in brain.recall.grid_maps[0]] == [(6, width), (16, width), (width,)]
    assert brain.parameters() == plain.parameters() + (6 + 16 + 1) * width
    assert brain.describe()["memories"]["recall"]["grid"] == list(GRID)
    assert "grid" not in plain.describe()["memories"]["recall"]
    with pytest.raises(ValueError, match="needs position"):
        grid(recall_position=0, recall_grid=GRID)
    for bad in ((4, 6), (1, 3), "35", (2.5, 3), (True, 3)):
        with pytest.raises(ValueError, match="coprime"):
            grid(recall_grid=bad)
    with pytest.raises(ValueError):
        plain.retune(recall_grid=GRID)


def test_two_ages_share_every_residue_only_at_the_product_apart():
    brain = grid(recall_grid=GRID, recall_capacity=128)
    recall = fill(brain, 70)
    buckets = recall.grid_buckets()
    ages = recall.ages().astype(int)
    assert buckets.shape == (3, recall.slots)
    assert buckets[0].max() < 3 <= buckets[1].min() and buckets[1].max() < 7 <= buckets[2].min()
    same = (buckets[:, None] == buckets[:, :, None]).all(axis=0)
    apart = np.abs(ages[:, None] - ages[None, :])
    np.testing.assert_array_equal(same, apart % 60 == 0)


def test_the_term_is_the_sum_of_the_residues_terms_past_the_table():
    brain = grid(recall_grid=GRID, recall_position=8, recall_capacity=32)
    recall = fill(brain, 20)
    event, cortex = np.zeros((2, 6)), np.zeros((2, 16))
    before = recall.position_term(0, event, cortex)
    g_x, g_c, _ = recall.grid_maps[0]
    bias = np.random.default_rng(2).random(sum(GRID))
    recall.grid_maps = [(g_x, g_c, bias)]
    after = recall.position_term(0, event, cortex)
    far = recall.ages() >= 8
    expected = bias[recall.grid_buckets()].sum(axis=0) * far
    np.testing.assert_allclose(after - before, np.repeat(expected[None], 2, axis=0))
    assert far.any() and not far.all()


def test_a_brain_reads_and_learns_as_without_it_until_a_slot_passes_the_table():
    rng = np.random.default_rng(3)
    options = {"recall_position": 32, "recall_slope": 0.1, **FLAT}
    plain, brain = grid(**options), grid(recall_grid=GRID, **options)
    for t in range(30):
        x = np.eye(6)[rng.integers(6, size=2)]
        label = rng.integers(0, 4, size=2)
        if t % 3 == 2:
            plain.lesson(x, label)
            brain.lesson(x, label)
        np.testing.assert_array_equal(plain.act(x, greedy=True), brain.act(x, greedy=True))
    assert not any(a.any() for a in brain.recall.grid_maps[0])
    for a, b in zip(plain.recall.position_maps[0], brain.recall.position_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(plain.recall.attended_heads[0], brain.recall.attended_heads[0])
    np.testing.assert_array_equal(plain.learner.brain.efficacy, brain.learner.brain.efficacy)


def test_the_lesson_steps_the_grid_with_the_tables_gradient_over_its_peak_past_the_table():
    options = {"recall_position": 8, "recall_capacity": 64, **FLAT}
    brain = grid(recall_grid=GRID, **options)
    recall = fill(brain, 40)
    found = recall._row_steps(recall.last, np.arange(2), np.eye(6)[[1, 2]], None, 0, None)
    assert found is not None
    x, c, target, prior = found[1], found[2], found[6], found[7]
    recall._position_step(np.arange(2), 0, found)
    attended = np.exp(prior)
    peak = np.maximum(target.max(axis=1, keepdims=True), attended.max(axis=1, keepdims=True))
    assert (peak < 1.0).all()
    far = (target - attended) / peak * (recall.ages() >= 8)
    rate = recall.position_rate
    energy = (x * x).sum(axis=1) + (c * c).sum(axis=1) + 1.0
    bound = np.minimum(1.0, 1.0 / (rate * energy))[:, None]
    total = np.zeros((2, sum(GRID)))
    for residues in recall.grid_buckets():
        np.add.at(total, (np.arange(2)[:, None], residues[None, :]), far)
    np.testing.assert_allclose(recall.grid_maps[0][2], rate * (total * bound).mean(axis=0))
    # the table's own step is as without the option
    plain = grid(**options)
    stored = fill(plain, 40)
    same = stored._row_steps(stored.last, np.arange(2), np.eye(6)[[1, 2]], None, 0, None)
    assert same is not None
    stored._position_step(np.arange(2), 0, same)
    np.testing.assert_array_equal(stored.position_maps[0][2], recall.position_maps[0][2])


def test_a_read_on_a_wrong_slot_takes_at_most_a_whole_step():
    """The attention's peak bounds the step where the read is concentrated and the target spread."""
    options = {"recall_position": 8, "recall_capacity": 64, "recall_temperature": 0.25, **FLAT}
    brain = grid(recall_grid=GRID, **options)
    recall = fill(brain, 40)
    found = recall._row_steps(recall.last, np.arange(2), np.eye(6)[[1, 2]], None, 0, None)
    assert found is not None
    moving, x, c, key, error, step, target, prior = found
    # a read held on one far slot, the target named across the slots
    held = np.full_like(prior, -40.0)
    held[:, 2] = 0.0
    recall._position_step(np.arange(2), 0, (moving, x, c, key, error, step, target, held))
    rate = recall.position_rate
    energy = (x * x).sum(axis=1) + (c * c).sum(axis=1) + 1.0
    bound = np.minimum(1.0, 1.0 / (rate * energy))
    assert np.abs(recall.grid_maps[0][2]).max() <= rate * bound.max() + 1e-12
    assert recall.grid_maps[0][2].any()


def test_past_the_table_the_slope_takes_no_step():
    options = {"recall_slope": 0.1, "recall_position": 8, "recall_capacity": 128, **FLAT}
    plain, brain = grid(**options), grid(recall_grid=GRID, **options)
    for memory in (plain, brain):
        fill(memory, 60)
    found = plain.recall._row_steps(plain.recall.last, np.arange(2), np.eye(6)[[1, 2]], None, 0, None)
    assert found is not None
    plain.recall._position_step(np.arange(2), 0, found)
    brain.recall._position_step(np.arange(2), 0, found)
    assert plain.recall.slope_maps[0][2].any()
    assert not brain.recall.slope_maps[0][2].any()


def test_within_the_table_the_slopes_step_is_as_it_was():
    options = {"recall_slope": 0.1, "recall_position": 32, **FLAT}
    plain, brain = grid(**options), grid(recall_grid=GRID, **options)
    for memory in (plain, brain):
        fill(memory, 30)
    found = plain.recall._row_steps(plain.recall.last, np.arange(2), np.eye(6)[[1, 2]], None, 0, None)
    assert found is not None
    plain.recall._position_step(np.arange(2), 0, found)
    brain.recall._position_step(np.arange(2), 0, found)
    assert brain.recall.slope_maps[0][2].any()
    np.testing.assert_array_equal(plain.recall.slope_maps[0][2], brain.recall.slope_maps[0][2])


def _taught(brain: cd.Brain, slots: int = 14, lessons: int = 8, seed: int = 4) -> np.random.Generator:
    rng = np.random.default_rng(seed)
    for _ in range(slots):
        brain.act(rng.random((2, 6)), greedy=True)
    for _ in range(lessons):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))
    return rng


def test_rewards_do_not_step_the_grid():
    brain = grid(recall_grid=GRID, recall_position=4, recall_position_reward=1.0,
                 recall_address_reward=1.0)
    rng = np.random.default_rng(5)
    for _ in range(10):
        brain.act(rng.random((2, 6)), greedy=True)
    brain.step(rng.random((2, 6)))
    for _ in range(5):
        brain.step(rng.random((2, 6)), reward=np.array([1.0, -1.0]))
    assert not any(a.any() for a in brain.recall.grid_maps[0])


def test_a_checkpoint_keeps_the_grid_and_continues(tmp_path):
    brain = grid(recall_grid=GRID, recall_position=4)
    rng = _taught(brain)
    assert brain.recall.grid_maps[0][2].any()
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert copy.recall.grid == GRID
    for kept, loaded in zip(brain.recall.grid_maps[0], copy.recall.grid_maps[0], strict=True):
        np.testing.assert_array_equal(kept, loaded)
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))
    for kept, loaded in zip(brain.recall.grid_maps[0], copy.recall.grid_maps[0], strict=True):
        np.testing.assert_array_equal(kept, loaded)
    path = grid().save(tmp_path / "plain.npz")
    assert cd.Brain.load(path).recall.grid_maps == []


def test_a_grown_head_and_a_graft_keep_it():
    brain = grid(recall_grid=GRID, recall_position=4)
    _taught(brain)
    grown = brain.grow(recall_heads=2)
    assert grown.recall.grid == GRID and len(grown.recall.grid_maps) == 2
    np.testing.assert_array_equal(grown.recall.grid_maps[0][2], brain.recall.grid_maps[0][2])
    assert not grown.recall.grid_maps[1][2].any()
    grafted = brain.graft(grid(), ["association", "motor"])
    assert grafted.recall.grid == GRID
    np.testing.assert_array_equal(grafted.recall.grid_maps[0][2], brain.recall.grid_maps[0][2])


def test_several_heads_each_have_their_own():
    brain = grid(recall_grid=GRID, recall_position=4, recall_heads=2)
    _taught(brain)
    first, second = brain.recall.grid_maps
    assert first[2].any() and second[2].any()
    assert not np.array_equal(first[2], second[2])
