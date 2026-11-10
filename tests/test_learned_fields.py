"""Learned receptive fields (``learned_fields``, cadence#284): the stage, its local rule, its
place in a composed brain, its checkpoint and its refusals."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cadence import Brain
from cadence.fields import LearnedFields


def _frame(height: int = 24, width: int = 24, at: tuple[int, int] = (6, 6), size: int = 6,
           background: float = 0.1) -> np.ndarray:
    frame = np.full((height, width), background)
    y, x = at
    frame[y : y + size, x : x + size] = 0.9
    return frame


def test_the_stage_pools_every_field_over_overlapping_regions_at_unit_length():
    fields = LearnedFields(24, 24, 1, features=4, seed=0)
    assert (fields.rows, fields.cols) == (9, 9) and fields.outputs == 4 * 9
    # each pool is 1.5 times its share of the map, so neighbouring pools overlap
    assert [(w.start, w.stop) for w in fields._windows(9)] == [(0, 4), (2, 6), (5, 9)]
    code = fields.code(_frame().reshape(1, -1))
    assert code.shape == (1, 36) and np.isclose(np.linalg.norm(code), 1.0)
    assert (code >= 0).all()


def test_a_uniform_screen_reads_nothing_at_any_brightness():
    fields = LearnedFields(16, 16, 1, features=4, seed=0)
    for level in (0.0, 0.4, 1.0):
        assert not fields.code(np.full((1, 256), level)).any()


def test_one_set_of_synapses_reads_every_position():
    fields = LearnedFields(24, 24, 1, features=4, seed=1)
    here = fields.responses(fields.retina(_frame(at=(4, 4)).reshape(1, -1)))
    there = fields.responses(fields.retina(_frame(at=(4 + 2 * fields.STRIDE, 4)).reshape(1, -1)))
    here = here.reshape(fields.rows, fields.cols, -1)
    there = there.reshape(fields.rows, fields.cols, -1)
    np.testing.assert_allclose(there[2:], here[:-2], atol=1e-12)  # the map moves with the image


def test_the_fields_learn_from_structured_patches_alone():
    fields = LearnedFields(16, 16, 1, features=4, seed=0)
    before = fields.dictionary.copy(), fields.thresholds.copy()
    assert not fields.learn(np.full((2, 256), 0.5))  # a uniform screen moves nothing
    np.testing.assert_array_equal(fields.dictionary, before[0])
    assert fields.learn(_frame(16, 16).reshape(1, -1))
    assert not np.array_equal(fields.dictionary, before[0])
    np.testing.assert_allclose(np.linalg.norm(fields.dictionary, axis=0), 1.0)
    assert (fields.thresholds >= 0).all() and fields.work["update_patches"] > 0
    fields.learning = False
    held = fields.dictionary.copy()
    assert not fields.learn(_frame(16, 16).reshape(1, -1))
    np.testing.assert_array_equal(fields.dictionary, held)


def test_the_rule_shapes_fields_that_reconstruct_what_the_eye_sees():
    rng = np.random.default_rng(0)
    fields = LearnedFields(16, 16, 1, features=8, seed=0)

    def frames(n: int) -> np.ndarray:
        out = []
        for _ in range(n):
            y, x = rng.integers(0, 10, size=2)
            out.append(_frame(16, 16, at=(int(y), int(x)), size=int(rng.integers(3, 7))))
        return np.stack(out).reshape(n, -1)

    def residual() -> float:
        patches = fields.retina(frames(32)).reshape(-1, fields.patch)
        patches = patches[(patches * patches).mean(axis=1) >= fields.STRUCTURE]
        a = fields.responses(patches)
        return float(np.mean(np.sum((patches - a @ fields.dictionary.T) ** 2, axis=1)
                             / np.sum(patches * patches, axis=1)))

    start = residual()
    for _ in range(200):
        fields.learn(frames(4))
    assert residual() < 0.8 * start


def test_compose_reads_the_image_through_the_fields_and_keys_the_memories_on_their_pools():
    brain = Brain.compose(inputs=(84, 84), actions=3, seed=0, learned_fields=16)
    assert brain.recall is not None  # the key is the pools: 144 coordinates, not 7,056
    assert len(brain.connectome.populations["sensory"]) == 16 * 9
    assert "visual" not in brain.connectome.populations
    layout = brain.describe()["layout"]
    assert layout["code"]["kind"] == "fields" and layout["code"]["width"] == 144
    assert layout["composition"]["learned_fields"] == 16
    assert layout["learned_fields"]["features"] == 16
    assert brain.parameters() == (
        Brain.compose(inputs=144, actions=3, seed=0).parameters() + brain.fields.parameters()
    )
    with pytest.raises(ValueError, match="observation coordinates"):
        Brain.compose(inputs=(84, 84), actions=3, seed=0)  # the visual cortex: 7,056 keys


def test_the_fields_learn_at_accepted_moments_and_lessons_and_never_in_a_reading():
    rng = np.random.default_rng(0)
    brain = Brain.compose(inputs=(16, 16), actions=2, seed=0, learned_fields=4, recall=False)
    x = np.stack([_frame(16, 16, at=(2, 2)), _frame(16, 16, at=(8, 8))]).reshape(2, -1)
    held = brain.fields.dictionary.copy()
    brain.predict(x)
    brain.observation_code(x)
    np.testing.assert_array_equal(brain.fields.dictionary, held)
    brain.lesson(x, [0, 1])
    assert brain.fields.work["updates"] == 1
    brain.act(x, greedy=True)
    assert brain.fields.work["updates"] == 2
    brain.step(x + 0.01 * rng.random(x.shape))
    assert brain.fields.work["updates"] == 3
    assert not np.array_equal(brain.fields.dictionary, held)


def test_a_checkpoint_continues_the_fields_and_the_life(tmp_path):
    rng = np.random.default_rng(1)
    brain = Brain.compose(inputs=(16, 16, 3), actions=3, seed=2, learned_fields=6)
    x = rng.random((2, 16 * 16 * 3))
    brain.lesson(x, [0, 2])
    brain.step(x)
    twin = Brain.load(brain.save(tmp_path / "brain"))
    np.testing.assert_array_equal(twin.fields.dictionary, brain.fields.dictionary)
    np.testing.assert_array_equal(twin.fields.thresholds, brain.fields.thresholds)
    y = rng.random((2, 16 * 16 * 3))
    np.testing.assert_array_equal(twin.step(y, reward=[1.0, 0.0]), brain.step(y, reward=[1.0, 0.0]))
    np.testing.assert_array_equal(twin.fields.dictionary, brain.fields.dictionary)
    assert twin.fields.work == brain.fields.work
    copied = copy.deepcopy(brain)
    np.testing.assert_array_equal(copied.fields.dictionary, brain.fields.dictionary)


@pytest.mark.parametrize("settings, message", [
    ({"inputs": 64, "learned_fields": 4}, "need an image"),
    ({"inputs": (16, 16), "learned_fields": 4, "code": 64}, "code and learned_code"),
    ({"inputs": (16, 16), "learned_fields": 4, "learned_code": 8}, "learned code needs a vector"),
    ({"inputs": (16, 16), "learned_fields": 1}, "at least 2 fields"),
    ({"inputs": (6, 6), "learned_fields": 4}, "at least 8 by 8"),
])
def test_refusals(settings, message):
    with pytest.raises(ValueError, match=message):
        Brain.compose(actions=2, recall=False, **settings)


def test_grow_and_graft_refuse_a_brain_with_fields():
    brain = Brain.compose(inputs=(16, 16), actions=2, seed=0, learned_fields=4, recall=False)
    with pytest.raises(ValueError, match="learned fields"):
        brain.grow(recall=True)
    with pytest.raises(ValueError, match="learned fields"):
        brain.graft(copy.deepcopy(brain), ["association"])


def test_ports_are_read_beside_the_pools_in_one_brain(tmp_path):
    rng = np.random.default_rng(3)
    brain = Brain.compose(inputs=(16, 16, 3), actions=3, seed=1, learned_fields=6, ports=5)
    plain = Brain.compose(inputs=(16, 16, 3), actions=3, seed=1, learned_fields=6)
    assert brain.inputs == 16 * 16 * 3 + 5 and brain.ports == 5
    assert len(brain.connectome.populations["sensory"]) == 6 * 9 + 5
    layout = brain.describe()["layout"]
    assert layout["ports"] == 5 and layout["composition"]["ports"] == 5
    assert layout["code"]["width"] == 6 * 9 + 5
    assert "ports" not in plain.describe()["layout"]
    frames, ports = rng.random((2, 16 * 16 * 3)), rng.random((2, 5))
    rows = np.concatenate([frames, ports], axis=1)  # the flattened image, then the ports
    key = brain.observation_code(rows)
    np.testing.assert_array_equal(key[:, :54], plain.observation_code(frames))  # the pools
    np.testing.assert_array_equal(key[:, 54:], ports)  # the ports as given
    brain.lesson(rows, [0, 2])
    plain.lesson(frames, [0, 2])
    np.testing.assert_array_equal(brain.fields.dictionary, plain.fields.dictionary)
    assert brain.step(rows).shape == (2,)
    twin = Brain.load(brain.save(tmp_path / "ported"))
    assert twin.ports == 5 and twin.inputs == brain.inputs
    later = np.concatenate([rng.random((2, 16 * 16 * 3)), rng.random((2, 5))], axis=1)
    np.testing.assert_array_equal(
        twin.step(later, reward=[1.0, 0.0]), brain.step(later, reward=[1.0, 0.0])
    )
    np.testing.assert_array_equal(twin.fields.dictionary, brain.fields.dictionary)
    with pytest.raises(ValueError, match="values per row"):
        brain.act(frames)  # a row without its ports


def test_the_ports_change_the_answer():
    brain = Brain.compose(inputs=(16, 16), actions=2, seed=0, learned_fields=4, ports=2,
                          recall=False)
    frame = _frame(16, 16).reshape(1, -1)
    low = np.concatenate([frame, [[1.0, 0.0]]], axis=1)
    high = np.concatenate([frame, [[0.0, 1.0]]], axis=1)
    for _ in range(40):  # the same screen, the answer in the ports alone
        brain.lesson(np.concatenate([low, high]), [0, 1])
    assert list(brain.predict(np.concatenate([low, high]))) == [0, 1]


@pytest.mark.parametrize("settings, message", [
    ({"inputs": 64, "ports": 3}, "beside learned fields"),
    ({"inputs": (16, 16), "ports": 3}, "beside learned fields"),
    ({"inputs": (16, 16), "learned_fields": 4, "ports": -1}, "ports must be an integer"),
    ({"inputs": (16, 16), "learned_fields": 4, "ports": True}, "ports must be an integer"),
])
def test_ports_refusals(settings, message):
    with pytest.raises(ValueError, match=message):
        Brain.compose(actions=2, recall=False, **settings)


def test_a_colour_image_is_read_through_luminance_fields_and_colour_fields():
    fields = LearnedFields(16, 16, 3, features=8, seed=0)
    assert fields.colour == "opponent"
    groups = [(list(m), d.stop - d.start, f.stop - f.start) for m, d, f in fields._groups]
    assert groups == [([0], 64, 6), ([1, 2], 128, 2)]  # three fields in four read luminance
    # a field reads its own group's channels alone
    assert not fields.dictionary[64:, :6].any() and not fields.dictionary[:64, 6:].any()
    np.testing.assert_allclose(np.linalg.norm(fields.dictionary, axis=0), 1.0)
    # the opponent channels are orthonormal: luminance, red against green, blue against yellow
    np.testing.assert_allclose(fields.OPPONENT @ fields.OPPONENT.T, np.eye(3), atol=1e-12)
    # one colour everywhere reads nothing, as one brightness does
    flat = np.zeros((1, 16, 16, 3))
    flat[..., 0], flat[..., 1] = 0.9, 0.2
    assert not fields.code(flat.reshape(1, -1)).any()
    # the luminance fields read a shape the same whatever its hue
    red, blue = np.full((16, 16, 3), 0.1), np.full((16, 16, 3), 0.1)
    red[4:10, 4:10] = (0.9, 0.3, 0.3)
    blue[4:10, 4:10] = (0.3, 0.3, 0.9)
    a = fields.responses(fields.retina(red.reshape(1, -1)))[..., :6]
    b = fields.responses(fields.retina(blue.reshape(1, -1)))[..., :6]
    np.testing.assert_allclose(a, b, atol=1e-9)
    assert a.any()
    # each group learns from its own patches
    before = fields.dictionary.copy()
    assert fields.learn(red.reshape(1, -1))
    assert not np.array_equal(fields.dictionary[:64, :6], before[:64, :6])
    assert not np.array_equal(fields.dictionary[64:, 6:], before[64:, 6:])
    assert not fields.dictionary[64:, :6].any() and not fields.dictionary[:64, 6:].any()


def test_the_colour_controls_and_other_channel_counts():
    assert LearnedFields(16, 16, 3, 9, colour="channels")._groups[2][2] == slice(6, 9)
    assert len(LearnedFields(16, 16, 3, 8, colour="joint")._groups) == 1
    assert LearnedFields(16, 16, 2, 8).colour == "joint"  # one group over any other channels
    assert LearnedFields(16, 16, 1, 8).colour == "joint"
    with pytest.raises(ValueError, match="colour is one of"):
        LearnedFields(16, 16, 3, 8, colour="grey")
    with pytest.raises(ValueError, match="at least 3 fields"):
        LearnedFields(16, 16, 3, 2, colour="channels")


def test_a_field_steps_less_the_more_it_has_been_used(tmp_path):
    frames = np.stack([_frame(16, 16, at=(k, k)) for k in (2, 4, 6, 8)]).reshape(4, -1)
    young, settled = LearnedFields(16, 16, 1, 4, seed=0), LearnedFields(16, 16, 1, 4, seed=0)
    assert young.maturity == LearnedFields.MATURITY > 0
    settled.use[:] = 100 * settled.maturity  # as after a long life
    start = young.dictionary.copy()
    young.learn(frames)
    settled.learn(frames)
    moved = np.abs(young.dictionary - start).sum(axis=0)
    slow = np.abs(settled.dictionary - start).sum(axis=0)
    used = moved > 0
    assert used.any() and (slow[used] < 0.02 * moved[used]).all()
    assert (young.use[used] == 1).all() and (young.use[~used] == 0).all()
    whole = LearnedFields(16, 16, 1, 4, seed=0)
    whole.maturity = 0.0  # every step whole: the control
    whole.use[:] = 1e6
    whole.learn(frames)
    np.testing.assert_allclose(whole.dictionary, young.dictionary, atol=1e-12)
    brain = Brain.compose(inputs=(16, 16), actions=2, seed=0, learned_fields=4, recall=False)
    brain.lesson(frames[:2], [0, 1])
    twin = Brain.load(brain.save(tmp_path / "used"))
    np.testing.assert_array_equal(twin.fields.use, brain.fields.use)
    assert twin.fields.use.any() and twin.fields.maturity == brain.fields.maturity
