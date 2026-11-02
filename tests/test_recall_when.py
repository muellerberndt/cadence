"""A when-channel (cadence#237): every head's read also delivers when what it read happened,
the attention-weighted features of its slots' ages on a region of its own."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def when(**options) -> cd.Brain:
    return cd.Brain.compose(6, 4, seed=5, recall=True, **options)


def test_when_is_an_option_with_a_region_per_head():
    plain = when()
    assert not plain.recall.when and "recall_when" not in plain.connectome.populations
    brain = when(recall_when=True, recall_heads=2)
    recall = brain.recall
    assert recall.when_regions(2) == ("recall_when", "recall_1_when")
    assert [len(brain.connectome.populations[n]) for n in recall.when_regions(2)] == [2, 2]
    assert recall.neurons == plain.recall.neurons * 2 + 4
    assert brain.describe()["memories"]["recall"]["when"] is True
    anchored = when(recall_when=True, recall_anchor=True)
    assert anchored.recall.when_width == 4
    assert len(anchored.connectome.populations["recall_when"]) == 4


def test_the_features_of_when_follow_age_and_place():
    brain = when(recall_when=True, recall_anchor=True)
    recall = brain.recall
    rng = np.random.default_rng(0)
    for _ in range(5):
        brain.act(rng.random((2, 6)))
    features = recall.when_features()
    ages = recall.ages()
    assert features.shape == (2, recall.slots, 4)
    np.testing.assert_allclose(features[0, :, 0], np.log1p(ages) / np.log1p(8.0))
    np.testing.assert_allclose(features[0, :, 1], np.exp(-ages / 8.0))
    places = recall.told[0] - 1 - ages
    np.testing.assert_allclose(features[0, :, 2], np.log1p(places) / np.log1p(8.0))
    np.testing.assert_allclose(features[0, :, 3], np.exp(-places / 8.0))


def test_the_read_drives_the_when_region_with_its_attention():
    brain = when(recall_when=True)
    recall = brain.recall
    rng = np.random.default_rng(1)
    for _ in range(4):
        brain.act(rng.random((2, 6)))
    drive = recall.stimulate(np.zeros((2, brain.connectome.n)), rng.random((2, 6)))
    neurons = np.asarray(brain.connectome.populations["recall_when"])
    expected = recall.amplitude * np.einsum("bm,bmw->bw", recall.attended, recall.when_features())
    np.testing.assert_allclose(drive[:, neurons], expected)


@pytest.mark.parametrize(
    "options",
    [{}, {"recall_copy": 0.5}, {"recall_copy": 0.5, "recall_heads": 2}],
    ids=["recall", "copy", "copy_heads"],
)
def test_a_newborn_when_channel_settles_as_the_memory_without_it(options):
    rng = np.random.default_rng(2)
    plain, brain = when(**options), when(recall_when=True, **options)
    n = plain.connectome.n

    def synapses(b: cd.Brain, below: int) -> list[tuple[int, int, float]]:
        c = b.connectome
        pre, post = np.asarray(c.pre), np.asarray(c.post)
        kept = (pre < below) & (post < below)  # the when regions' neurons are numbered last
        return sorted(zip(pre[kept].tolist(), post[kept].tolist(),
                          np.asarray(b.brain.efficacy)[kept].tolist(), strict=True))

    assert synapses(plain, n) == synapses(brain, n)
    for _ in range(4):
        x = rng.random((2, 6))
        np.testing.assert_array_equal(plain.act(x, greedy=True), brain.act(x, greedy=True))
        np.testing.assert_array_equal(
            np.asarray(plain.basal_ganglia.state.activation),
            np.asarray(brain.basal_ganglia.state.activation)[:, :n],
        )


def test_lessons_give_the_when_synapses_weight_and_a_checkpoint_continues(tmp_path):
    brain = when(recall_when=True)
    neurons = np.asarray(brain.connectome.populations["recall_when"])
    synapses = np.flatnonzero(np.isin(np.asarray(brain.connectome.pre), neurons))
    assert len(synapses) and not np.asarray(brain.brain.efficacy)[synapses].any()
    rng = np.random.default_rng(3)
    for _ in range(12):
        brain.step(rng.random((2, 6)), teacher=rng.integers(0, 4, size=2))
    assert np.asarray(brain.brain.efficacy)[synapses].any()
    copy = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    assert copy.recall.when
    for _ in range(3):
        x, label = rng.random((2, 6)), rng.integers(0, 4, size=2)
        np.testing.assert_array_equal(brain.step(x, teacher=label), copy.step(x, teacher=label))


def test_when_reads_into_the_recall_regions():
    with pytest.raises(ValueError, match="reads into the recall regions"):
        cd.Recall(when(recall_when=True).connectome, target="next", next=False, when=True)
