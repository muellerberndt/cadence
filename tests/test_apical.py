"""The apical compartment and the deep teaching point: the projection-relative normalization
floor and nudged phases that run their whole budget."""

from dataclasses import replace

import numpy as np
import pytest

from cadence import Brain, Learner
from cadence.brain import Apical
from cadence.brain import Brain as Graph
from cadence.genome import Genome, Projection, Region, develop
from cadence.learning import LearnerConfig


def region_names(connectome):
    """The region (a population named without a slash) of every neuron."""
    names = np.empty(connectome.n, dtype=object)
    for name, members in connectome.populations.items():
        if "/" not in name:
            names[list(members)] = name
    return names


def apical_brain(**options):
    options = {"modules": (6, 5), "seed": 2, "apical": 0.25, **options}
    return Brain.compose(4, 3, **options)


def test_compose_lands_the_top_down_half_of_every_reciprocal_pair_apically():
    brain = apical_brain(observers=(3,))
    graph, connectome = brain.brain, brain.connectome
    names = region_names(connectome)
    pre, post = connectome.pre, connectome.post
    mask = graph.apical.synapses
    apical = {(names[a], names[b]) for a, b in zip(pre[mask], post[mask], strict=True)}
    assert apical == {
        ("association", "module_0"), ("motor", "association"),
        ("observer_0", "module_0"), ("observer_0", "association"), ("observer_0", "motor"),
    }
    basal = {(names[a], names[b]) for a, b in zip(pre[~mask], post[~mask], strict=True)}
    for pair in [("sensory", "module_0"), ("module_0", "association"), ("association", "motor"),
                 ("prefrontal", "association"), ("module_0", "observer_0")]:
        assert pair in basal
    assert not apical & basal  # a projection lands in one compartment
    # each region that hears the regions above it is one pool
    pools = {names[np.arange(connectome.n)[members]][0] for members in graph.apical.members}
    assert pools == {"module_0", "association", "motor"}
    assert graph.neuron_model.dt == 0.5 and graph.apical.fraction == 0.25


def test_apical_zero_is_the_founder_brain():
    founder = Brain.compose(4, 3, modules=(6, 5), seed=2)
    zero = Brain.compose(4, 3, modules=(6, 5), seed=2, apical=0.0)
    assert founder.brain.apical is None and zero.brain.apical is None
    assert founder.describe() == zero.describe()
    assert "apical" not in founder.describe()["genes"]
    assert "apical" not in founder.describe()["layout"]["composition"]
    x = np.random.default_rng(0).normal(size=(5, 4))
    np.testing.assert_array_equal(founder.act(x), zero.act(x))


def test_apical_is_a_construction_setting_with_reciprocal_pairs_and_no_mlx():
    brain = apical_brain()
    assert brain.describe()["genes"]["apical"] == 0.25
    assert brain.describe()["layout"]["composition"]["apical"] == 0.25
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(apical=0.5)
    with pytest.raises(ValueError, match="apical"):
        apical_brain(apical=-0.1)
    with pytest.raises(ValueError, match="mlx"):
        Graph(brain.connectome, brain.brain.neuron_model, backend="mlx", apical=brain.brain.apical)
    one_way = Genome(
        (Region("sensory", 4), Region("association", 5), Region("motor", 3)),
        (Projection("sensory", "association", reciprocal=False),
         Projection("association", "motor", reciprocal=False)),
    )
    with pytest.raises(ValueError, match="reciprocal pairs"):
        Brain(develop(one_way, seed=0), apical=0.25)


def test_the_spec_is_validated():
    with pytest.raises(ValueError):
        Apical(np.zeros(3), np.zeros(4, dtype=int), 0.25)  # not boolean
    with pytest.raises(ValueError):
        Apical(np.zeros(3, bool), np.full(4, -2), 0.25)
    with pytest.raises(ValueError):
        Apical(np.zeros(3, bool), np.zeros(4, dtype=int), 0.0)
    with pytest.raises(ValueError):
        Apical(np.zeros(3, bool), np.zeros(4, dtype=int), 0.25, floor=-1.0)
    brain = apical_brain()
    spec = brain.brain.apical
    groups = spec.groups.copy()
    groups[groups >= 0] = -1  # apical synapses landing on no pool
    with pytest.raises(ValueError, match="pool"):
        Graph(brain.connectome, brain.brain.neuron_model,
              apical=Apical(spec.synapses, groups, 0.25))


def test_the_drive_keeps_the_direction_and_is_bounded_by_the_basal_input():
    rng = np.random.default_rng(1)
    spec = Apical(np.zeros(0, bool), np.array([0, 0, 0, -1]), 0.25)
    basal, raw = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    members = slice(0, 3)
    weak = spec.drive(basal, 1e-6 * raw)  # a weak top-down input passes at fraction / floor
    np.testing.assert_allclose(weak[:, members], 0.25e-6 * raw[:, members], rtol=1e-6)
    strong = spec.drive(basal, 1e6 * raw)  # a strong one saturates at fraction * rms(basal)
    rms = lambda a: np.sqrt(np.mean(a[:, members] ** 2, axis=1))  # noqa: E731
    np.testing.assert_allclose(rms(strong), 0.25 * rms(basal), rtol=1e-6)
    cosine = (strong[:, members] * raw[:, members]).sum(1) / (
        np.linalg.norm(strong[:, members], axis=1) * np.linalg.norm(raw[:, members], axis=1))
    np.testing.assert_allclose(cosine, 1.0)
    assert not weak[:, 3].any() and not strong[:, 3].any()  # no pool, no apical drive
    assert not spec.drive(np.zeros((2, 4)), raw[:2]).any()  # no basal drive, no apical drive


def test_strong_top_down_weights_never_take_a_region_over():
    brain = apical_brain(seed=4)
    graph = brain.brain
    efficacy = graph.efficacy.copy()
    efficacy[graph.apical.synapses] *= 50.0  # top-down halves far stronger than bottom-up
    graph = graph.with_parameters(efficacy=efficacy)
    drive = brain.stimulus(np.random.default_rng(3).normal(size=(8, 4)))
    state = graph.settle_batch(drive, steps=20_000, tolerance=1e-12)
    assert graph.residual(drive, state).max() < 1e-8
    basal, raw = graph._split_input(state.activation, *graph._apical_transports())
    basal = basal + drive + graph.bias
    top_down = graph.apical.drive(basal, raw)
    for members in graph.apical.members:
        drive_rms = np.sqrt(np.mean(top_down[:, members] ** 2, axis=1))
        basal_rms = np.sqrt(np.mean(basal[:, members] ** 2, axis=1))
        assert np.all(drive_rms <= 0.25 * basal_rms + 1e-12)
    np.testing.assert_allclose(state.v, basal + top_down, atol=1e-8)  # its fixed point


def test_blocked_and_segmented_settles_agree():
    brain = apical_brain(seed=5)
    graph = brain.brain
    segmented = Graph(brain.connectome, graph.neuron_model, efficacy=graph.efficacy,
                      bias=graph.bias, dense_limit=1, apical=graph.apical)
    assert graph._blocked and not segmented._blocked
    drive = brain.stimulus(np.random.default_rng(6).normal(size=(4, 4)))
    a = graph.settle_batch(drive, steps=200)
    b = segmented.settle_batch(drive, steps=200)
    np.testing.assert_allclose(a.activation, b.activation, atol=1e-13)
    np.testing.assert_allclose(graph.residual(drive, a), segmented.residual(drive, b), atol=1e-13)


def test_torch_settles_and_learns_as_the_host():
    pytest.importorskip("torch")
    options = {"teaching": "deep", "episodic": False}
    host = apical_brain(seed=7, **options)
    device = apical_brain(seed=7, backend="torch", device="cpu", **options)
    x = np.random.default_rng(8).normal(size=(6, 4))
    labels = np.array([0, 1, 2, 0, 1, 2])
    drive = host.stimulus(x)
    a = host.brain.settle_batch(drive, steps=100)
    b = device.brain.settle_batch(drive, steps=100)
    np.testing.assert_allclose(a.activation, b.activation, atol=1e-12)
    np.testing.assert_allclose(host.brain.residual(drive, a), device.brain.residual(drive, b),
                               atol=1e-12)
    for _ in range(3):
        host.lesson(x, labels)
        device.lesson(x, labels)
    np.testing.assert_allclose(host.brain.efficacy, device.brain.efficacy, atol=1e-11)
    np.testing.assert_allclose(host.brain.bias, device.brain.bias, atol=1e-11)


def test_checkpoints_keep_the_compartment_in_a_format_of_their_own(tmp_path):
    brain = apical_brain(teaching="deep")
    rng = np.random.default_rng(9)
    for _ in range(3):
        brain.lesson(rng.normal(size=(4, 4)), rng.integers(0, 3, 4))
    path = brain.learner.save(tmp_path / "learner.npz")
    with np.load(path, allow_pickle=False) as data:
        import json

        meta = json.loads(str(data["meta"]))
        assert meta["format"] == "cadence-checkpoint/5"
        assert meta["apical"] == {"fraction": 0.25, "floor": 1.0}
        assert meta["config"]["normalize_relative"] == 0.1
        arrays = {name: data[name] for name in data.files}
    learner = Learner.load(path)
    np.testing.assert_array_equal(learner.brain.apical.synapses, brain.brain.apical.synapses)
    np.testing.assert_array_equal(learner.brain.apical.groups, brain.brain.apical.groups)
    assert learner.config == brain.learner.config
    # a file claiming the format without its compartment is refused
    broken = dict(arrays)
    del broken["apical_synapses"]
    np.savez(tmp_path / "broken.npz", **broken)
    with pytest.raises(ValueError, match="cadence-checkpoint/5"):
        Learner.load(tmp_path / "broken.npz")
    # the whole brain round-trips, and a founder learner keeps the ordinary format
    whole = brain.save(tmp_path / "brain.npz")
    loaded = Brain.load(whole)
    assert loaded.describe() == brain.describe()
    x = rng.normal(size=(3, 4))
    np.testing.assert_array_equal(loaded.act(x), brain.act(x))
    founder = Brain.compose(4, 3, modules=(6, 5), seed=2).learner.save(tmp_path / "founder.npz")
    with np.load(founder, allow_pickle=False) as data:
        assert "apical_synapses" not in data.files
        assert json.loads(str(data["meta"]))["format"] == "cadence-checkpoint/2"


def test_a_learned_code_keeps_its_full_step_beside_the_compartment(tmp_path):
    brain = Brain.compose(6, 3, modules=(8, 5), learned_code=8, seed=1, apical=0.25)
    assert brain.brain.neuron_model.dt == 0.5
    assert brain.learned_code.model.dt == 1.0
    x = np.random.default_rng(2).normal(size=(2, 6))
    first = brain.act(x)
    loaded = Brain.load(brain.save(tmp_path / "coded.npz"))
    assert loaded.brain.apical is not None and loaded.learned_code.model.dt == 1.0
    np.testing.assert_array_equal(loaded.act(x), brain.act(x))
    assert first.shape == (2,)


# -- the relative floor


def test_the_relative_floor_is_validated_and_neutral_when_off():
    base = LearnerConfig(normalize=0.99, eta=0.003, eta_bias=0.0003)
    assert "normalize_relative" not in base.to_dict()
    assert LearnerConfig(**base.to_dict()) == base
    on = replace(base, normalize_relative=0.1)
    assert on.to_dict()["normalize_relative"] == 0.1
    assert LearnerConfig(**on.to_dict()) == on
    with pytest.raises(ValueError, match="normalize_relative"):
        replace(base, normalize_relative=-0.1)
    with pytest.raises(ValueError, match="normalize above zero"):
        LearnerConfig(normalize_relative=0.1)
    founder = Brain.compose(4, 3, modules=(6, 5), seed=2, teaching="normalized")
    assert founder.learner.config.normalize_relative == 0.0
    assert founder.describe()["genes"]["teaching"] == "normalized"
    deep = Brain.compose(4, 3, modules=(6, 5), seed=2, teaching="deep")
    assert deep.learner.config.normalize_relative == 0.1
    assert deep.describe()["genes"]["teaching"] == "deep"
    assert "learning_normalize_relative" not in deep.describe()["genes"]


def test_each_step_is_divided_by_its_projection_scale():
    brain = Brain.compose(4, 3, modules=(6, 5), seed=3, teaching="deep", episodic=False)
    learner = brain.learner
    rng = np.random.default_rng(4)
    rms = rng.uniform(0.0, 1.0, learner.brain.connectome.synapses)
    block = learner.brain.layout.block_of_edge
    rms[block == block[0]] *= 1e-6  # one faint projection
    rms_bias = rng.uniform(0.0, 1.0, learner.brain.connectome.n)
    divisor, divisor_bias = learner._relative_rms(rms, rms_bias)
    for k in np.unique(block):
        members = block == k
        expected = rms[members] + 0.1 * np.sqrt(np.mean(rms[members] ** 2)) + 1e-12
        np.testing.assert_allclose(divisor[members], expected, rtol=1e-12)
    # the faint projection's steps are as large, relative to their own scale, as any other's
    faint = block == block[0]
    ratio = rms[faint] / divisor[faint]
    assert ratio.max() > 0.5
    starts = learner.brain.layout.starts
    for r in range(len(starts) - 1):
        members = slice(starts[r], starts[r + 1])
        expected = rms_bias[members] + 0.1 * np.sqrt(np.mean(rms_bias[members] ** 2)) + 1e-12
        np.testing.assert_allclose(divisor_bias[members], expected, rtol=1e-12)


def test_fixed_nudged_phases_run_their_whole_budget():
    rng = np.random.default_rng(10)
    x, labels = rng.normal(size=(4, 4)), np.array([0, 1, 2, 0])
    stopping = apical_brain(teaching="normalized", learning_nudged_steps=40)
    fixed = apical_brain(teaching="deep", learning_nudged_steps=40)  # the budget is the brain's
    early = stopping.lesson(x, labels)
    whole = fixed.lesson(x, labels)
    assert early["nudged_steps"] < 40 and whole["nudged_steps"] == 40
    assert whole["free_steps"] == early["free_steps"]  # the free phase keeps its tolerance
    assert "nudged_fixed" not in LearnerConfig().to_dict()
    assert fixed.describe()["genes"]["teaching"] == "deep"
    assert fixed.describe()["genes"]["learning_nudged_steps"] == 40
    with pytest.raises(ValueError, match="nudged_fixed"):
        LearnerConfig(nudged_fixed=1)


def test_a_nudged_settle_holds_the_free_phase_gain():
    brain = apical_brain(seed=11, teaching="normalized")
    graph, learner = brain.brain, brain.learner
    x = np.random.default_rng(12).normal(size=(5, 4))
    drive = brain.stimulus(x)
    free = learner.free(drive)
    held = graph.apical_gain(drive, free)
    assert held.shape == (5, len(graph.apical.members)) and (held >= 0).all()
    assert (held <= graph.apical.fraction / graph.apical.floor + 1e-12).all()
    target = learner.targets(np.array([0, 1, 2, 0, 1]))
    nudge = learner.nudge_for(target, 0.1, None, held)
    a = graph.settle_batch(drive, steps=30, state=free, nudge=nudge)
    # the same settle by hand: every sweep's apical drive is the held gain times the raw input
    v = np.array(free.v, copy=True)
    s = np.asarray(free.activation)
    model = graph.neuron_model
    for _ in range(30):
        basal, raw = graph._split_input(s, *graph._apical_transports())
        total = basal + drive + graph.bias + nudge.drive(s) + graph.apical.drive(basal, raw, held)
        v = v + model.dt * (total - v)
        s = model.activation(v)
    np.testing.assert_allclose(a.activation, s, atol=1e-12)
    # the learner's nudged phase passes the same gain
    np.testing.assert_allclose(learner.nudged(drive, free, target).activation,
                               graph.settle_batch(drive, steps=learner.config.nudged_steps, state=free,
                                                  nudge=nudge, tolerance=learner.config.tolerance).activation)
    with pytest.raises(ValueError, match="apical_gain"):
        graph.settle_batch(drive, steps=2, state=free,
                           nudge=learner.nudge_for(target, 0.1, None, held[:2]))
    founder = Brain.compose(4, 3, modules=(6, 5), seed=11)
    assert founder.brain.apical_gain(drive, founder.learner.free(drive)) is None


def test_a_zero_floor_is_refused():
    with pytest.raises(ValueError, match="floor"):
        Apical(np.zeros(3, bool), np.zeros(4, dtype=int), 0.25, floor=0.0)


def test_the_residual_refuses_a_held_gain_of_another_shape():
    brain = apical_brain(seed=11, teaching="normalized")
    graph, learner = brain.brain, brain.learner
    drive = brain.stimulus(np.random.default_rng(12).normal(size=(3, 4)))
    free = learner.free(drive)
    held = graph.apical_gain(drive, free)
    target = learner.targets(np.array([0, 1, 2]))
    assert graph.residual(drive, free, nudge=learner.nudge_for(target, 0.1, None, held)).shape == (3,)
    with pytest.raises(ValueError, match="apical_gain"):
        graph.residual(drive, free, nudge=learner.nudge_for(target, 0.1, None, held[:1]))


def test_tools_that_read_the_plain_sum_refuse_the_compartment():
    from cadence.certificate import certificate
    from cadence.reference import conformance

    graph = apical_brain().brain
    with pytest.raises(ValueError, match="apical"):
        graph.dense()
    with pytest.raises(ValueError, match="apical"):
        conformance(graph, None)
    with pytest.raises(ValueError, match="apical"):
        certificate(graph)
    founder = Brain.compose(4, 3, modules=(6, 5), seed=2).brain
    assert founder.dense().shape == (founder.connectome.n, founder.connectome.n)
    assert certificate(founder) is not None


def test_a_settlement_record_carries_the_compartment():
    from cadence.recording import record_settlements

    brain = apical_brain()
    records = []
    with record_settlements(records.append):
        brain.learner.free(brain.stimulus(np.ones((1, 4))))
    assert records and records[0].apical is not None
    np.testing.assert_array_equal(records[0].apical.synapses, brain.brain.apical.synapses)
    founder, plain = Brain.compose(4, 3, modules=(6, 5), seed=2), []
    with record_settlements(plain.append):
        founder.learner.free(founder.stimulus(np.ones((1, 4))))
    assert plain and plain[0].apical is None


def test_the_new_learner_settings_take_the_new_format_without_the_compartment(tmp_path):
    import json

    brain = Brain.compose(4, 3, modules=(6, 5), seed=2, teaching="deep")
    path = brain.learner.save(tmp_path / "learner.npz")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        arrays = {name: data[name] for name in data.files}
    assert meta["format"] == "cadence-checkpoint/5" and "apical" not in meta
    assert Learner.load(path).config == brain.learner.config
    # the settings under the ordinary format are refused, never dropped
    np.savez(tmp_path / "old.npz", **{**arrays, "meta": np.array(json.dumps(
        {**meta, "format": "cadence-checkpoint/2"}))})
    with pytest.raises(ValueError, match="cadence-checkpoint/5"):
        Learner.load(tmp_path / "old.npz")
    # and the new format without either is refused too
    plain = {**meta["config"]}
    plain.pop("normalize_relative")
    plain.pop("nudged_fixed")
    np.savez(tmp_path / "empty.npz", **{**arrays, "meta": np.array(json.dumps(
        {**meta, "config": plain}))})
    with pytest.raises(ValueError, match="cadence-checkpoint/5"):
        Learner.load(tmp_path / "empty.npz")


def test_retune_moves_between_the_teaching_points_with_their_floors_and_phases():
    brain = Brain.compose(4, 3, modules=(6, 5), seed=2, teaching="deep")
    brain.retune(teaching="founder")
    config = brain.learner.config
    assert config.normalize == 0.0 and config.normalize_relative == 0.0 and config.eta == 0.5
    assert not config.nudged_fixed and brain.describe()["genes"]["teaching"] == "founder"
    brain.retune(teaching="deep")
    deep = Brain.compose(4, 3, modules=(6, 5), seed=2, teaching="deep")
    assert brain.learner.config == deep.learner.config


def test_the_folded_names_are_refused_and_a_depth_checkpoint_of_0921_describes_as_deep(tmp_path):
    """``learning_normalize_relative`` and ``learning_nudged_fixed`` left in 0.93.0 for the
    point ``teaching="deep"``; the expert route to the same configuration, which 0.92.1 composed
    by the three names, saves and loads as the point."""
    for name, value in (("learning_normalize_relative", 0.1), ("learning_nudged_fixed", True)):
        with pytest.raises(ValueError, match=f"{name} was retired in 0.93.0.*teaching='deep'"):
            Brain.compose(4, 3, modules=(6, 5), seed=2, teaching="normalized", **{name: value})
        with pytest.raises(ValueError, match=f"{name} was retired in 0.93.0"):
            apical_brain(teaching="deep").retune(**{name: value})
    normalized = Brain.compose(4, 3, modules=(6, 5), seed=2, teaching="normalized").learner.config
    expert = Brain.compose(
        4, 3, modules=(6, 5), seed=2, apical=0.25,
        learning=replace(normalized, normalize_relative=0.1, nudged_steps=32, nudged_fixed=True),
    )
    deep = apical_brain(teaching="deep")
    assert expert.learner.config == deep.learner.config
    assert expert.describe() == deep.describe()
    assert deep.describe()["genes"]["teaching"] == "deep"
    twin = Brain.load(expert.save(tmp_path / "depth.npz"))
    assert twin.describe()["genes"]["teaching"] == "deep"
    assert twin.learner.config == deep.learner.config
    custom = Brain.compose(4, 3, modules=(6, 5), seed=2, apical=0.25,
                           learning=replace(normalized, normalize_relative=0.2))
    genes = custom.describe()["genes"]
    assert genes["teaching"] == "custom" and genes["learning"]["normalize_relative"] == 0.2
    assert set(genes["learning"]) == {"eta", "eta_bias", "normalize", "normalize_floor",
                                      "normalize_relative", "nudged_fixed"}
