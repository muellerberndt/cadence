"""Tied projections (``tied``, cadence#308, #284): equal blocks of a projection's regions share
one set of synapses. The layout, the mean step, the copies' equality through lessons, rewards,
checkpoints, growth and every backend, the parameter count, and the refusals."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import cadence as cd
from cadence import Brain, Genome, Projection, Region, develop
from cadence.genome import mutate, tie_groups

READ = {("sensory", "association"): (3, 1)}


def _brain(tied=READ, inputs=12, seed=1, **settings):
    return Brain.compose(inputs, 3, seed=seed, tied=tied,
                         **{"recall": False, "backend": "cpu", **settings})


def _copies_equal(brain) -> bool:
    """Whether every tie group of the brain's learner holds one efficacy."""
    learner = brain.learner
    values = np.asarray(brain.brain.efficacy)[learner._members]
    groups = learner._member_groups
    low = np.full(len(learner._member_count), np.inf)
    high = np.full(len(learner._member_count), -np.inf)
    np.minimum.at(low, groups, values)
    np.maximum.at(high, groups, values)
    return bool(len(values)) and bool((low == high).all())


def _live(brain, moments=6, rows=4, seed=0):
    """A few lessons, then a few rewarded moments, the last outcome reported."""
    rng = np.random.default_rng(seed)
    width = brain.inputs
    for _ in range(moments):
        brain.lesson(rng.random((rows, width)), rng.integers(3, size=rows))
    brain.step(rng.random((rows, width)))
    for _ in range(moments):
        brain.step(rng.random((rows, width)), reward=rng.random(rows),
                   done=np.zeros(rows, bool))
    brain.learn(rng.random(rows), np.ones(rows, bool), rng.random((rows, width)))


def _anatomy(tied, reciprocal=True, density=1.0):
    """A custom anatomy whose projection between ``slots`` and the association is tied."""
    regions = (Region("sensory", 6), Region("association", 8), Region("slots", 8),
               Region("prefrontal", 8), Region("motor", 3))
    ends = ("association", "slots") if tied[0] == 1 else ("slots", "association")
    if tied[0] == tied[1]:
        ends = ("association", "slots")
    return Genome(regions, (
        Projection("sensory", "association", reciprocal=False),
        Projection("sensory", "slots", reciprocal=False),
        Projection(*ends, tied=tied, reciprocal=reciprocal, density=density),
        Projection("association", "motor"),
        Projection("prefrontal", "association", reciprocal=False),
    ))


# -- the layout


def test_a_tied_projection_repeats_its_first_block_and_leaves_the_other_draws():
    plain = Genome((Region("a", 6), Region("b", 4), Region("c", 3)),
                   (Projection("a", "b", reciprocal=False, density=0.7), Projection("b", "c")))
    shared = Genome(plain.regions, (
        Projection("a", "b", reciprocal=False, density=0.7, tied=(3, 1)), Projection("b", "c")))
    one, three = develop(plain, seed=3), develop(shared, seed=3)

    def synapses(connectome, pre, post):
        return {(int(a), int(b)): float(s)
                for a, b, s in zip(connectome.pre, connectome.post, connectome.sign, strict=True)
                if a in pre and b in post}

    first = synapses(one, range(2), range(6, 10))
    assert first and synapses(three, range(2), range(6, 10)) == first
    for block in (1, 2):  # every block of the source: the first block's synapses
        moved = {(a + 2 * block, b): s for (a, b), s in first.items()}
        assert synapses(three, range(2 * block, 2 * block + 2), range(6, 10)) == moved
    later = range(6, 13)
    assert synapses(three, later, later) == synapses(one, later, later)
    assert shared.tied == {("a", "b"): (3, 1)} and plain.tied == {}


@pytest.mark.parametrize("tied,copies", [((4, 1), 4), ((1, 2), 2), ((4, 4), 4)])
@pytest.mark.parametrize("reciprocal", [False, True])
def test_the_groups_hold_one_synapse_per_block(tied, copies, reciprocal):
    genome = _anatomy(tied, reciprocal, density=0.6)
    connectome = develop(genome, seed=2)
    groups = tie_groups(connectome, genome.tied)
    assert groups is not None and groups.dtype == np.dtype(np.intp)
    counts = np.bincount(groups[groups >= 0])
    assert len(counts) and (counts == copies).all()
    for group in range(len(counts)):  # one start for every copy
        assert len(set(connectome.sign[groups == group])) == 1
    ((source, target),) = genome.tied  # the synapses from its source to its target, no other
    forward = np.isin(connectome.pre, connectome.populations[source]) & np.isin(
        connectome.post, connectome.populations[target]
    )
    assert np.array_equal(groups >= 0, forward)
    assert tie_groups(connectome, {}) is None


def test_block_onto_block_joins_no_two_different_blocks_and_scales_to_its_fan():
    genome = _anatomy((4, 4), reciprocal=False)
    connectome = develop(genome, seed=2)
    association = np.asarray(connectome.populations["association"])
    slots = np.asarray(connectome.populations["slots"])
    joins = np.isin(connectome.pre, association) & np.isin(connectome.post, slots)
    block_pre = (connectome.pre[joins] - association[0]) // 2
    block_post = (connectome.post[joins] - slots[0]) // 2
    assert joins.sum() == 4 * 2 * 2 and np.array_equal(block_pre, block_post)
    assert np.abs(connectome.sign[joins]).max() <= np.sqrt(6.0 / (2 + 2))
    assert np.abs(connectome.sign[joins]).max() > np.sqrt(6.0 / (8 + 8))


def test_a_genome_records_a_tie_and_mutation_keeps_its_blocks():
    genome = _anatomy((4, 1))
    record = json.loads(json.dumps(genome.to_dict()))
    assert [p.get("tied") for p in record["projections"]] == [None, None, [4, 1], None, None]
    assert Genome.from_dict(record) == genome
    child = mutate(genome, np.random.default_rng(0), size_step=0.5)
    assert child.region("slots").size == 8 and child.projections[2].tied == (4, 1)
    composed = Genome.composed(12, 3, tied=READ)
    assert composed.tied == READ
    assert Genome.composed(12, 3).to_dict() == Genome.composed(12, 3, tied={}).to_dict()


# -- the step


def test_a_tied_synapse_steps_by_the_mean_of_its_copies_steps():
    brain = _brain({("sensory", "association"): (2, 1)}, inputs=4)
    groups = brain.learner.tie_groups
    first, second = np.flatnonzero(groups == 0)
    before = brain.brain.efficacy.copy()
    step = np.zeros(brain.connectome.synapses)
    step[first], step[second] = 0.5, 0.25  # by hand: both copies move by 0.375
    brain.learner.apply(step, np.zeros(brain.connectome.n))
    moved = brain.brain.efficacy - before
    assert moved[first] == moved[second] == 0.375
    assert np.count_nonzero(moved) == 2 and _copies_equal(brain)


@pytest.mark.parametrize("teaching", ["founder", "normalized", "deep"])
@pytest.mark.parametrize("blocks", [2, 3])
def test_a_lesson_moves_a_tie_by_the_mean_of_the_untied_steps(teaching, blocks, monkeypatch):
    shared = {("sensory", "association"): (blocks, 1)}
    brain = _brain(shared, teaching=teaching)
    # the same synapses and starts without the tie
    twin = Brain.compose(12, 3, seed=1, connectome=brain.connectome, recall=False,
                         backend="cpu", teaching=teaching)
    assert twin.learner.tie_groups is None
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    steps = {}
    apply = type(brain.learner).apply

    def recorded(learner, delta_scale, delta_bias):
        steps[id(learner)] = np.array(delta_scale, dtype=float)
        return apply(learner, delta_scale, delta_bias)

    monkeypatch.setattr(type(brain.learner), "apply", recorded)
    rng = np.random.default_rng(5)
    x, labels = rng.random((4, 12)), rng.integers(3, size=4)
    before = brain.brain.efficacy.copy()
    brain.lesson(x, labels)
    twin.lesson(x, labels)
    untied = steps[id(twin.learner)]  # each copy's own step, as the untied twin takes it
    np.testing.assert_array_equal(steps[id(brain.learner)], untied)
    groups = brain.learner.tie_groups
    tied = groups >= 0
    assert np.abs(untied[tied]).max() > 0 and len(set(untied[groups == 0])) > 1
    # the twin moves each copy by its own step, the tied brain by the mean of the copies' steps
    np.testing.assert_array_equal(twin.brain.efficacy[tied], (before + untied)[tied])
    mean = np.bincount(groups[tied], weights=untied[tied]) / blocks
    expected = before[tied] + mean[groups[tied]]
    if blocks == 2:  # a sum of two and a halving are exact
        np.testing.assert_array_equal(brain.brain.efficacy[tied], expected)
    else:
        np.testing.assert_allclose(brain.brain.efficacy[tied], expected, rtol=1e-13, atol=1e-16)
    # and every other synapse and every bias as in the twin
    np.testing.assert_array_equal(brain.brain.efficacy[~tied], twin.brain.efficacy[~tied])
    np.testing.assert_array_equal(brain.brain.bias, twin.brain.bias)
    assert _copies_equal(brain)


# -- the copies stay one efficacy


@pytest.mark.parametrize("teaching", ["founder", "normalized", "deep"])
def test_tied_copies_stay_equal_through_lessons_and_rewards(teaching):
    brain = _brain(teaching=teaching, arousal=True)
    start = brain.brain.efficacy.copy()
    assert _copies_equal(brain)
    _live(brain)
    tied = brain.learner.tie_groups >= 0
    assert _copies_equal(brain) and np.abs(brain.brain.efficacy - start)[tied].max() > 0


@pytest.mark.parametrize("tied", [(4, 1), (1, 4), (4, 4)])
@pytest.mark.parametrize("reciprocal", [False, True])
def test_a_custom_anatomy_takes_its_genomes_ties(tied, reciprocal):
    genome = _anatomy(tied, reciprocal)
    brain = Brain.compose(6, 3, connectome=develop(genome, seed=4), seed=4, tied=genome.tied,
                          recall=False, backend="cpu")
    assert brain.tied == genome.tied
    start = brain.brain.efficacy.copy()
    _live(brain)
    learner = brain.learner
    assert _copies_equal(brain)
    assert np.abs(brain.brain.efficacy - start)[learner._members].max() > 0
    # a reciprocal projection's reverse synapses are its forward ones' partners: one group
    copies = max(tied) * (2 if reciprocal else 1)
    assert (learner._member_count == copies).all()


def test_the_fields_pool_rows_share_one_read():
    brain = Brain.compose((24, 24), 3, seed=0, learned_fields=4, recall=False, backend="cpu",
                          tied=READ)
    rng = np.random.default_rng(0)
    for _ in range(4):
        brain.lesson(rng.random((2, 576)), rng.integers(3, size=2))
    sensory = np.asarray(brain.connectome.populations["sensory"])
    assert len(sensory) == 3 * 3 * 4  # pool rows, then pool columns, then fields
    dense = brain.brain.dense()[np.ix_(sensory, brain.association_index)]
    np.testing.assert_array_equal(dense[:12], dense[12:24])
    np.testing.assert_array_equal(dense[:12], dense[24:])
    assert np.abs(dense).max() > 0 and brain.fields.work["updates"] == 4


# -- counts and reports


def test_parameters_count_a_tied_synapse_once():
    plain, tied = _brain(None), _brain()
    shared = 4 * 64  # one block of the sensory region onto the association
    assert plain.connectome.synapses == tied.connectome.synapses
    assert plain.learner.parameters() - tied.learner.parameters() == 2 * shared
    assert plain.parameters() - tied.parameters() == 2 * shared
    described = tied.describe()["layout"]
    assert described["tied"] == [
        {"pre": "sensory", "post": "association", "blocks": [3, 1], "synapses": shared}
    ]
    assert described["composition"]["tied"] == [["sensory", "association", 3, 1]]
    assert "tied" not in plain.describe()["layout"]
    assert "tied" not in plain.describe()["layout"]["composition"]
    json.dumps(described)
    reciprocal = _anatomy((1, 4))
    brain = Brain.compose(6, 3, connectome=develop(reciprocal, seed=4), seed=4,
                          tied=reciprocal.tied, recall=False, backend="cpu")
    untied = Brain.compose(6, 3, connectome=brain.connectome, seed=4, recall=False,
                           backend="cpu")
    # a reciprocal pair counts once without the tie; four pairs count once with it
    assert untied.learner.parameters() - brain.learner.parameters() == 3 * 8 * 2


# -- with the option off


@pytest.mark.parametrize("off", [None, {}, {("sensory", "association"): (1, 1)}])
def test_without_a_tie_the_brain_is_the_brain_without_the_option(off):
    plain = Brain.compose(12, 3, seed=1, recall=False, backend="cpu")
    same = _brain(off)
    assert same.tied == {} and same.learner.tie_groups is None
    assert same.connectome.digest() == plain.connectome.digest()
    assert same.describe() == plain.describe()
    for brain in (plain, same):
        _live(brain)
    np.testing.assert_array_equal(same.brain.efficacy, plain.brain.efficacy)
    np.testing.assert_array_equal(same.brain.bias, plain.brain.bias)


def test_a_file_without_a_tie_holds_no_word_of_it(tmp_path):
    plain = Brain.compose(12, 3, seed=1, recall=False, backend="cpu")
    plain.save(tmp_path / "plain.npz")
    with np.load(tmp_path / "plain.npz") as saved:
        meta = json.loads(str(saved["brain"]))
        assert "tied" not in meta and "tied" not in meta["composition"]
        assert saved["tie_groups"].size == 0


# -- checkpoints


def test_a_tied_brain_saves_and_resumes_to_the_bit(tmp_path):
    brain = _brain(teaching="normalized")
    _live(brain)
    brain.save(tmp_path / "tied.npz")
    loaded = Brain.load(tmp_path / "tied.npz")
    assert loaded.tied == brain.tied and loaded.parameters() == brain.parameters()
    np.testing.assert_array_equal(loaded.learner.tie_groups, brain.learner.tie_groups)
    np.testing.assert_array_equal(loaded.brain.efficacy, brain.brain.efficacy)
    assert loaded.describe() == brain.describe()
    for one in (brain, loaded):
        _live(one, seed=3)
    np.testing.assert_array_equal(loaded.brain.efficacy, brain.brain.efficacy)
    assert _copies_equal(loaded)
    twin = copy.deepcopy(brain)
    _live(twin, seed=4)
    assert _copies_equal(twin) and twin.tied == brain.tied


def test_a_file_whose_copies_differ_is_refused(tmp_path):
    brain = _brain()
    _live(brain)
    brain.save(tmp_path / "tied.npz")
    with np.load(tmp_path / "tied.npz") as saved:
        data = {name: saved[name] for name in saved.files}
    apart = dict(data)
    efficacy = np.array(data["efficacy"])
    efficacy[np.flatnonzero(data["tie_groups"] == 0)[0]] += 0.5
    apart["efficacy"] = efficacy
    np.savez(tmp_path / "apart.npz", **apart)
    with pytest.raises(ValueError, match="tied projections disagree"):
        Brain.load(tmp_path / "apart.npz")
    untied = dict(data)
    untied["tie_groups"] = np.zeros(0, np.int64)
    np.savez(tmp_path / "untied.npz", **untied)
    with pytest.raises(ValueError, match="tied projections disagree"):
        Brain.load(tmp_path / "untied.npz")
    wrong = dict(data)
    meta = json.loads(str(data["brain"]))
    meta["tied"] = [["sensory", "association", 3]]
    wrong["brain"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "wrong.npz", **wrong)
    with pytest.raises(ValueError, match="invalid saved tied projections"):
        Brain.load(tmp_path / "wrong.npz")


# -- growth


def test_a_tied_brain_grows_with_its_ties():
    brain = _brain(teaching="normalized")
    _live(brain)
    grown = brain.grow(efference_amplitude=1.0, slots=(3, 2))
    assert grown.tied == brain.tied and _copies_equal(grown)
    assert grown.describe()["layout"]["composition"]["tied"] == [["sensory", "association", 3, 1]]
    tied = brain.learner.tie_groups >= 0
    kept = grown.learner.tie_groups >= 0
    np.testing.assert_array_equal(
        np.sort(grown.brain.efficacy[kept]), np.sort(brain.brain.efficacy[tied])
    )
    rng = np.random.default_rng(2)
    for _ in range(4):
        grown.lesson(rng.random((3, 12)), np.stack(
            [rng.integers(3, size=3), rng.integers(2, size=3)], axis=1))
    assert _copies_equal(grown)
    assert not np.array_equal(np.sort(grown.brain.efficacy[kept]),
                              np.sort(brain.brain.efficacy[tied]))
    assert grown.parameters() > brain.parameters()


def test_grow_adds_no_neurons_to_a_tied_region():
    motor = _brain({("association", "motor"): (1, 3)})
    with pytest.raises(ValueError, match="grow adds no neurons to 'motor'"):
        motor.grow(slots=(3, 2))
    heard = _brain({("history", "association"): (2, 1)}, history_decays=(0.5, 0.8))
    with pytest.raises(ValueError, match="grow adds no neurons to 'history'"):
        heard.grow(history_decays=(0.5, 0.8, 0.9))
    assert _copies_equal(heard.grow(efference_amplitude=1.0))
    # the whole side of a tie takes new neurons: their synapses start at zero in every block
    read = _brain({("association", "motor"): (4, 1)})
    _live(read)
    wider = read.grow(slots=(3, 2))
    assert _copies_equal(wider) and wider.connectome.synapses > read.connectome.synapses
    assert (wider.learner._member_count == 8).all()  # four blocks, each pair both ways
    with pytest.raises(ValueError, match="tied is a construction choice"):
        _brain().grow(tied=READ)
    with pytest.raises(ValueError, match="construction setting"):
        _brain().retune(tied=READ)


# -- backends


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_tied_copies_stay_equal_on_torch(tmp_path, precision):
    pytest.importorskip("torch")
    brain = _brain(backend="torch", device="cpu", precision=precision, arousal=True)
    host = _brain(arousal=True)
    _live(brain)
    _live(host)
    assert _copies_equal(brain)
    np.testing.assert_allclose(brain.brain.efficacy, host.brain.efficacy, rtol=0,
                               atol=1e-9 if precision == "float64" else 1e-3)
    brain.save(tmp_path / "torch.npz")
    for place in ({"backend": "torch", "device": "cpu", "precision": precision}, {}):
        loaded = Brain.load(tmp_path / "torch.npz", **place)
        _live(loaded, seed=6)
        assert loaded.tied == brain.tied and _copies_equal(loaded)


@pytest.mark.skipif("mlx" not in cd.available_backends(), reason="mlx not installed")
def test_tied_copies_stay_equal_on_mlx(tmp_path):
    brain = _brain(backend="mlx", arousal=True)
    assert brain.brain.backend == "mlx"
    _live(brain)
    assert _copies_equal(brain)
    brain.save(tmp_path / "mlx.npz")
    loaded = Brain.load(tmp_path / "mlx.npz", backend="mlx")
    _live(loaded, seed=6)
    assert loaded.tied == brain.tied and _copies_equal(loaded)


# -- the index type


@pytest.fixture
def index_type_int32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np, "intp", np.int32)


def test_tie_groups_take_the_index_type(index_type_int32: None) -> None:
    brain = _brain()
    assert brain.learner.tie_groups.dtype == np.dtype(np.int32)
    genome = _anatomy((1, 4))
    assert tie_groups(develop(genome, seed=1), genome.tied).dtype == np.dtype(np.int32)
    _live(brain, moments=2)
    assert _copies_equal(brain)


# -- refusals


@pytest.mark.parametrize("tied,message", [
    ([("sensory", "association", 3, 1)], "tied maps a projection"),
    ({"sensory": 3}, "tied maps a projection"),
    ({("sensory", "association"): 3}, "blocks of pre, blocks of post"),
    ({("sensory", "association"): (2, 3)}, "blocks of pre, blocks of post"),
    ({("sensory", "association"): (0, 1)}, "blocks of pre, blocks of post"),
    ({("sensory", "association"): (True, 1)}, "blocks of pre, blocks of post"),
    ({("sensory", "association"): (5, 1)}, "which 5 blocks do not divide"),
    ({("sensory", "motor"): (3, 1)}, "none runs from 'sensory' to 'motor'"),
])
def test_compose_refuses_a_tie_it_cannot_build(tied, message):
    with pytest.raises(ValueError, match=message):
        _brain(tied)


def test_a_tie_needs_blocks_that_start_as_copies():
    untied = develop(_anatomy((1, 1)), seed=4)
    with pytest.raises(ValueError, match="start as copies of one block"):
        Brain.compose(6, 3, connectome=untied, seed=4, recall=False, backend="cpu",
                      tied={("slots", "association"): (4, 1)})
    sparse = develop(Genome(_anatomy((1, 1)).regions, (
        Projection("slots", "association", density=0.5),
        Projection("association", "motor"),
    )), seed=4)
    with pytest.raises(ValueError, match="hold different synapses"):
        tie_groups(sparse, {("slots", "association"): (4, 1)})
    with pytest.raises(ValueError, match="a population the connectome lacks"):
        tie_groups(untied, {("slots", "nowhere"): (4, 1)})
    with pytest.raises(ValueError, match="a synapse between two different blocks"):
        tie_groups(untied, {("slots", "association"): (4, 4)})
    tied = develop(_anatomy((4, 1)), seed=4)
    twice = tied.with_populations(all=tied.populations["association"])
    with pytest.raises(ValueError, match="two ties name one synapse"):
        tie_groups(twice, {("slots", "association"): (4, 1), ("slots", "all"): (4, 1)})


def test_a_genome_refuses_a_tie_it_cannot_develop():
    regions = (Region("a", 6), Region("b", 4))
    with pytest.raises(ValueError, match="which 4 blocks do not divide"):
        Genome(regions, (Projection("a", "b", tied=(4, 1)),))
    with pytest.raises(ValueError, match="joined twice"):
        Genome(regions, (Projection("a", "b", tied=(3, 1), reciprocal=False),
                         Projection("b", "a", reciprocal=False)))
    for bad in ((2, 3), (0, 1), (1,), "ab", 3, (True, 1), (1.0, 2)):
        with pytest.raises(ValueError, match="blocks of pre, blocks of post"):
            Projection("a", "b", tied=bad)


def test_parts_that_do_not_hold_a_tie_refuse_it_by_name():
    with pytest.raises(ValueError, match="a learned code's own projections take no tie"):
        Brain.compose(12, 3, seed=1, learned_code=16, recall=False, backend="cpu",
                      tied={("sensory", "code"): (3, 1)})
    with pytest.raises(ValueError, match="a tie on it needs ports=0"):
        Brain.compose((24, 24), 3, seed=0, learned_fields=4, ports=3, recall=False,
                      backend="cpu", tied=READ)
    with pytest.raises(ValueError, match="no tied projections"):
        cd.Lives(12, 3, lives=2, recall=False, tied=READ)
    host, donor = _brain(), _brain(None)
    with pytest.raises(ValueError, match="without tied synapses"):
        donor.graft(host, regions=("association", "motor"))
