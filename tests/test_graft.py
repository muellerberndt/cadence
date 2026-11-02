"""``Brain.graft``: trained regions of one brain in another.

The grafted neurons take the donor's biases and gains and the synapses among them the donor's
efficacies; every other synapse onto or from a grafted neuron (a joint) starts at zero, or keeps
this brain's weight with ``joints="kept"``. A whole donor grafted into a larger anatomy settles
as the donor until learning gives the joints weight.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import cadence as cd
from cadence import Genome, Projection, Region, develop


def taught(brain, lessons=20, inputs=6, seed=0):
    rng = np.random.default_rng(seed)
    slots = brain.learner.slot_sizes
    for _ in range(lessons):
        x = rng.random((3, inputs))
        brain.step(x, teacher=rng.integers(0, int(slots[0]), size=3))
    return brain


def larger_anatomy(fly, *, inputs=None):
    """``fly`` as one region of a larger anatomy, a language region wired to its association;
    the brain's named populations are the fly's."""
    regions = [Region("fly", circuit=fly), Region("language", 12)]
    projections = [Projection("fly/association", "language")]
    if inputs:
        regions.append(Region("words", inputs))
        projections.append(Projection("words", "language", reciprocal=False))
    bigger = develop(Genome(tuple(regions), tuple(projections)), seed=1)
    p = bigger.populations
    return bigger.with_populations(
        sensory=p["fly/sensory"],
        association=p["fly/association"],
        prefrontal=p["fly/prefrontal"],
        motor=p["fly/motor"],
    )


def fly_connectome():
    """A stand-in for a measured connectome: a developed layout with named populations."""
    return develop(Genome.composed(6, 4, modules=(16,)), seed=0)


def test_a_whole_donor_grafted_into_a_larger_anatomy_acts_as_the_donor():
    fly = fly_connectome()
    donor = taught(cd.Brain.compose(6, 4, connectome=fly, seed=0, episodic=False))
    bigger = larger_anatomy(fly)
    host = cd.Brain.compose(6, 4, connectome=bigger, seed=0, episodic=False)
    grafted = host.graft(donor, into="fly")
    neurons = list(bigger.populations["fly"])
    twin = copy.deepcopy(donor)
    twin.reset()
    rng = np.random.default_rng(1)
    for _ in range(10):
        x = rng.random((3, 6))
        np.testing.assert_array_equal(twin.act(x, greedy=True), grafted.act(x, greedy=True))
        np.testing.assert_allclose(  # bit for bit where the larger graph sums in the same order
            np.asarray(twin.basal_ganglia.state.activation),
            np.asarray(grafted.basal_ganglia.state.activation)[:, neurons],
            rtol=0,
            atol=1e-12,
        )
    language = bigger.populations["language"]
    joints = np.isin(grafted.connectome.pre, language) | np.isin(grafted.connectome.post, language)
    assert joints.any() and not grafted.brain.efficacy[joints].any()


def test_joints_learn_where_the_new_region_has_a_drive_of_its_own():
    """At zero, a joint learns when one end is active and the other moves between the phases:
    a language region reading its own words is active. A region with no drive of its own is
    silent at zero and needs joints that start with weight (``joints="kept"``)."""
    fly = fly_connectome()
    donor = taught(cd.Brain.compose(6, 4, connectome=fly, seed=0, episodic=False))
    rng = np.random.default_rng(2)

    def lessons(brain, words):
        for _ in range(15):
            x = rng.random((3, 6 + words))
            brain.step(x, teacher=rng.integers(0, 4, size=3))

    def joint_weight(brain, anatomy):
        language = anatomy.populations["language"]
        fly_neurons = anatomy.populations["fly"]
        joints = np.isin(brain.connectome.pre, language) & np.isin(
            brain.connectome.post, fly_neurons
        )
        return float(np.abs(brain.brain.efficacy[joints]).mean())

    # with words of its own, the language region drives the joints into the fly at zero start
    worded = larger_anatomy(fly, inputs=5)
    worded = worded.with_populations(
        sensory=list(worded.populations["fly/sensory"]) + list(worded.populations["words"])
    )
    host = cd.Brain.compose(11, 4, connectome=worded, seed=0, episodic=False)
    grafted = host.graft(donor, into="fly")
    assert joint_weight(grafted, worded) == 0.0
    lessons(grafted, 5)
    assert joint_weight(grafted, worded) > 0.0
    # without a drive of its own, joints at zero stay silent; kept joints start with weight
    silent = larger_anatomy(fly)
    host = cd.Brain.compose(6, 4, connectome=silent, seed=0, episodic=False)
    zero, kept = host.graft(donor, into="fly"), host.graft(donor, into="fly", joints="kept")
    lessons(zero, 0)
    lessons(kept, 0)
    assert joint_weight(zero, silent) == 0.0
    assert joint_weight(kept, silent) > 0.0


VISION = {"visual": "visual", "module_0": "module_0", "association": "module_1"}


def vision_pair(seed=0):
    donor = cd.Brain.compose((6, 6), 4, modules=(32, 16), seed=seed)
    host = cd.Brain.compose((6, 6), 2, modules=(32, 16, 24), seed=seed + 10)
    return donor, host


def test_grafted_regions_carry_the_donors_weights_and_the_joints_start_at_zero():
    donor, host = vision_pair()
    rng = np.random.default_rng(0)
    for _ in range(5):
        donor.lesson(rng.random((8, 6, 6)), rng.integers(0, 4, size=8))
    grafted = host.graft(donor, VISION)
    d, h = donor.connectome, grafted.connectome
    where = np.full(d.n, -1)
    for theirs, mine in VISION.items():
        where[list(d.populations[theirs])] = h.populations[mine]
    inner = (where[d.pre] >= 0) & (where[d.post] >= 0)
    key = h.post * h.n + h.pre
    at = np.searchsorted(key, where[d.post[inner]] * h.n + where[d.pre[inner]])
    np.testing.assert_array_equal(grafted.brain.efficacy[at], donor.brain.efficacy[inner])
    source = np.flatnonzero(where >= 0)
    np.testing.assert_array_equal(grafted.brain.bias[where[source]], donor.brain.bias[source])
    np.testing.assert_array_equal(
        grafted.brain.log_gain[where[source]], donor.brain.log_gain[source]
    )
    grafted_neurons = np.zeros(h.n, dtype=bool)
    grafted_neurons[where[source]] = True
    copied = np.zeros(h.synapses, dtype=bool)
    copied[at] = True
    joints = (grafted_neurons[h.pre] | grafted_neurons[h.post]) & ~copied
    assert joints.any() and not grafted.brain.efficacy[joints].any()
    rest = ~(grafted_neurons[h.pre] | grafted_neurons[h.post])
    np.testing.assert_array_equal(grafted.brain.efficacy[rest], host.brain.efficacy[rest])
    kept = host.graft(donor, VISION, joints="kept")
    np.testing.assert_array_equal(kept.brain.efficacy[joints], host.brain.efficacy[joints])


def test_a_frozen_graft_keeps_its_weights_while_its_joints_learn():
    donor, host = vision_pair()
    grafted = host.graft(donor, VISION, freeze=True)
    h = grafted.connectome
    inside = np.zeros(h.n, dtype=bool)
    for mine in VISION.values():
        inside[list(h.populations[mine])] = True
    graft = inside[h.pre] & inside[h.post]
    joints = inside[h.pre] ^ inside[h.post]
    before = np.array(grafted.brain.efficacy)
    rng = np.random.default_rng(3)
    for _ in range(10):
        grafted.lesson(rng.random((8, 6, 6)), rng.integers(0, 2, size=8))
    after = np.asarray(grafted.brain.efficacy)
    np.testing.assert_array_equal(after[graft], before[graft])
    assert np.abs(after[joints]).sum() > 0
    neurons = np.flatnonzero(inside)
    assert not grafted.learner.plastic_neurons[neurons].any()


def test_a_grafted_trained_vision_learns_a_related_task():
    """A host given a donor's trained vision answers top or bottom within 40 lessons."""
    rng = np.random.default_rng(2)

    def images(n):
        where = rng.integers(0, 4, size=n)
        x = 0.1 * rng.random((n, 6, 6))
        for i, k in enumerate(where):
            r, c = 3 * (k // 2), 3 * (k % 2)
            x[i, r : r + 3, c : c + 3] += 1.0
        return x, where

    donor, host = vision_pair(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(400):
            x, where = images(16)
            donor.lesson(x, where)
        test, answer = images(400)
        assert donor.accuracy(test, answer) > 0.95
        grafted = host.graft(donor, VISION)
        for _ in range(40):
            x, where = images(16)
            grafted.lesson(x, where // 2)
    assert grafted.accuracy(test, answer // 2) > 0.9


def test_the_host_keeps_its_own_memories_and_the_donor_is_unchanged(tmp_path):
    donor, host = vision_pair()
    rng = np.random.default_rng(4)
    for _ in range(4):
        host.step(rng.random((2, 6, 6)), teacher=rng.integers(0, 2, size=2))
        donor.step(rng.random((2, 6, 6)), teacher=rng.integers(0, 4, size=2))
    donor_before = np.array(donor.brain.efficacy)
    grafted = host.graft(donor, VISION)
    np.testing.assert_array_equal(donor.brain.efficacy, donor_before)
    np.testing.assert_array_equal(grafted.hippocampus.consolidated, host.hippocampus.consolidated)
    assert grafted.describe()["genes"] == host.describe()["genes"]
    grafted.save(tmp_path / "grafted.npz")
    loaded = cd.Brain.load(tmp_path / "grafted.npz")
    x = rng.random((2, 6, 6))
    np.testing.assert_array_equal(grafted.act(x, greedy=True), loaded.act(x, greedy=True))


def test_a_graft_keeps_this_brains_backend_and_precision(tmp_path):
    pytest.importorskip("torch")
    donor, host = vision_pair()
    host.save(tmp_path / "host.npz")
    on_torch = cd.Brain.load(
        tmp_path / "host.npz", backend="torch", device="cpu", precision="float32"
    )
    grafted = on_torch.graft(donor, VISION)
    assert (grafted.brain.backend, grafted.brain.precision) == ("torch", "float32")
    grafted.act(np.random.default_rng(0).random((2, 6, 6)), greedy=True)


@pytest.mark.parametrize(
    "arguments, message",
    [
        (dict(), "regions, or into="),
        (dict(regions=["association"], into="fly"), "regions, or into="),
        (dict(regions="association"), "not be one name"),
        (dict(regions=[]), "at least one region"),
        (dict(regions=["cerebellum"]), "the donor has no population"),
        (dict(regions={"association": "cerebellum"}), "this brain has no population"),
        (dict(regions=["association"]), "holds 16 neurons"),
        (dict(regions={"module_0": "association"}), "holds 32 neurons"),
        (dict(into="association"), "holds 24 neurons"),
        (dict(regions=VISION, joints="random"), "'zero' or 'kept'"),
        (dict(regions=VISION, freeze=1), "freeze must be boolean"),
    ],
)
def test_graft_refuses(arguments, message):
    donor, host = vision_pair()
    with pytest.raises(ValueError, match=message):
        host.graft(donor, **arguments)


def test_graft_refuses_wiring_the_donor_has_and_this_brain_lacks():
    """The donor's sensory region feeds its first module; an observer reads no sensory input."""
    donor = cd.Brain.compose(6, 4, modules=(16, 16), seed=0)
    host = cd.Brain.compose(6, 4, modules=(16,), observers=(16,), seed=0)
    with pytest.raises(ValueError, match="this brain lacks"):
        host.graft(donor, {"sensory": "sensory", "module_0": "observer_0"})


def test_graft_refuses_another_neuron_model_or_compartment():
    plain = cd.Brain.compose(6, 4, modules=(16, 16), seed=0)
    deep = cd.Brain.compose(6, 4, modules=(16, 16), apical=0.5, seed=0)
    deeper = cd.Brain.compose(6, 4, modules=(16, 16), apical=0.25, seed=0)
    with pytest.raises(ValueError, match="neuron model"):
        plain.graft(deep, ["module_0"])
    with pytest.raises(ValueError, match="apical compartment"):
        deeper.graft(deep, ["module_0"])


def test_graft_refuses_a_learned_code_region():
    donor = cd.Brain.compose(40, 4, learned_code=16, modules=(16,), seed=0)
    host = cd.Brain.compose(40, 4, learned_code=16, modules=(16,), seed=1)
    with pytest.raises(ValueError, match="learned code"):
        host.graft(donor, ["code"])


def test_graft_refuses_a_donor_that_is_not_a_brain():
    _, host = vision_pair()
    with pytest.raises(TypeError, match="a Brain"):
        host.graft(host.connectome, VISION)


def test_the_graft_starts_a_fresh_optimizer_history():
    """The donor's moments belong to the donor's count of lessons: carried under this brain's
    count, the first steps of the grafted synapses would be many times too large."""
    donor, host = vision_pair()
    rng = np.random.default_rng(5)
    for _ in range(30):
        donor.lesson(rng.random((8, 6, 6)), rng.integers(0, 4, size=8))
    assert np.abs(donor.learner.velocity).sum() > 0
    grafted = host.graft(donor, VISION)
    h = grafted.connectome
    inside = np.zeros(h.n, dtype=bool)
    for mine in VISION.values():
        inside[list(h.populations[mine])] = True
    graft = inside[h.pre] & inside[h.post]
    assert not grafted.learner.velocity[graft].any()
    assert not grafted.learner.second_moment[graft].any()
    assert not grafted.learner.velocity_bias[inside].any()


def test_graft_refuses_a_one_way_synapse_on_half_of_a_pair():
    """The donor's sensory region feeds its association one way; this brain's module_0 and
    association read each other through pairs that share one efficacy."""
    donor = cd.Brain.compose(6, 4, modules=(16,), seed=0)
    host = cd.Brain.compose(6, 4, modules=(6, 16), seed=1)
    with pytest.raises(ValueError, match="half of a reciprocal pair"):
        host.graft(donor, {"sensory": "module_0", "association": "association"})


def test_graft_refuses_regions_that_share_no_synapse():
    donor, host = vision_pair()
    with pytest.raises(ValueError, match="share no synapse"):
        host.graft(donor, {"association": "module_1"})


def test_graft_refuses_another_apical_role():
    donor = cd.Brain.compose(6, 4, modules=(16, 16), apical=0.5, seed=0)
    host = cd.Brain.compose(6, 4, modules=(16, 16), apical=0.5, seed=1)
    with pytest.raises(ValueError, match="apical roles"):
        host.graft(donor, {"module_0": "association", "association": "module_0"})


def test_a_learned_code_host_takes_a_graft_its_code_reads_into():
    """Synapses from the code region are ordinary synapses; only a joint landing on the code
    region, which lives in its tables, is refused."""
    donor = taught(cd.Brain.compose(40, 4, modules=(16, 16), seed=0), inputs=40)
    host = cd.Brain.compose(40, 4, learned_code=16, modules=(16, 16), seed=1)
    grafted = host.graft(donor, ["module_0", "association"])
    grafted.act(np.random.default_rng(0).random((2, 40)), greedy=True)
    coded = cd.Brain.compose(40, 4, learned_code=16, modules=(16,), seed=2)
    with pytest.raises(ValueError, match="land on its code region"):
        host.graft(coded, ["prefrontal", "association"])


def test_wiring_the_donor_lacks_starts_at_zero_even_with_kept_joints():
    """This brain's motor neurons inhibit one another; the donor's do not."""
    donor = taught(cd.Brain.compose(6, 4, modules=(16,), seed=0))
    host = cd.Brain.compose(6, 4, modules=(16,), lateral=-0.2, seed=1)
    grafted = host.graft(donor, ["association", "motor"], joints="kept")
    motor = host.connectome.populations["motor"]
    lateral = np.isin(grafted.connectome.pre, motor) & np.isin(grafted.connectome.post, motor)
    assert lateral.any() and not grafted.brain.efficacy[lateral].any()
    assert host.brain.efficacy[lateral].any()


def test_a_graft_keeps_this_brains_contact_counts_and_signs(tmp_path):
    """A recall copy pathway with tokens is laid out again from the brain's construction on
    load; with the donor's signs carried, the brain's connectome and its learner's disagreed."""
    tokens = dict(recall=True, recall_copy=2.0)
    host = cd.Brain.compose(6, 4, seed=1, modules=(16,), recall_tokens=[0, 1, 2, 3], **tokens)
    donor = cd.Brain.compose(6, 4, seed=7, modules=(16,), **tokens)
    grafted = host.graft(donor, ["recall", "next", "motor", "association"])
    np.testing.assert_array_equal(grafted.connectome.sign, host.connectome.sign)
    np.testing.assert_array_equal(grafted.connectome.count, host.connectome.count)
    grafted.save(tmp_path / "grafted.npz")
    loaded = cd.Brain.load(tmp_path / "grafted.npz")
    assert loaded.connectome.digest() == loaded.learner.brain.connectome.digest()
    np.testing.assert_array_equal(loaded.brain.efficacy, grafted.brain.efficacy)
