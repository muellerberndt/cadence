"""The brain's own graph on exact gradients: each gradient against finite differences, the unrolled
forward against the library's, and each rung changing only what it says."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gr = _load("grammar")
gx = _load("gradients")


@pytest.fixture(scope="module")
def newborn():
    return gr.make_brain("recipe", 0)


@pytest.fixture(scope="module")
def taught():
    brain = gr.make_brain("recipe", 0)
    rng = np.random.default_rng(np.random.SeedSequence([0, gr.SEEDS["train"]]))
    split = gr.Split.of(0)
    for _ in range(40):
        gr.speak(brain, gr.train_batch(rng, split, "g1"), teach=True)
    return brain


@pytest.fixture(scope="module")
def tests():
    return gr.panels(0, 1)


@pytest.mark.parametrize("panel", ["clause", "dyck", "path"])
def test_the_moments_gradient_is_the_fixed_points(taught, tests, panel):
    report = gx.check_moment(copy.deepcopy(taught), tests[panel][0][:8])
    assert report["worst_error"] < 1e-7 and report["largest_gradient"] > 1e-3
    assert report["polish"]["unpolished"] == 0
    # the local contrast is the same gradient as its nudge shrinks, a third path to it
    limit = report["contrast_at_a_full_settle"][-1]
    assert limit["beta"] == 0.001 and limit["cosine"] > 0.9999 and abs(limit["ratio"] - 1) < 1e-2


def test_the_moments_gradient_holds_the_compartments_gain(tests):
    brain = gr.make_brain("recipe", 0, settings=gx.DEEP)
    assert brain.learner.brain.apical is not None and brain.learner.config.nudged_fixed
    rng = np.random.default_rng(np.random.SeedSequence([0, gr.SEEDS["train"]]))
    split = gr.Split.of(0)
    for _ in range(20):
        gr.speak(brain, gr.train_batch(rng, split, "g1"), teach=True)
    report = gx.check_moment(copy.deepcopy(brain), tests["clause"][0][:8])
    assert report["worst_error"] < 1e-6 and report["largest_gradient"] > 1e-3
    assert report["polish"]["unpolished"] == 0


@pytest.mark.parametrize("arm, base", [("observed", "recipe"), ("obsdeep", "deep")])
def test_each_reading_beside_the_lessons_leaves_its_brain_to_the_bit(arm, base):
    a = gx.run_life({"arm": base, "founder": 0, "batches": 8, "module": 16, "panel_batches": 1,
                     "panels": ["clause"]})
    b = gx.run_life({"arm": arm, "founder": 0, "batches": 8, "module": 16, "panel_batches": 1,
                     "panels": ["clause"]})
    assert a["readings"][0]["panels"] == b["readings"][0]["panels"]
    seen = b["notes"]["observed"]["all"]
    assert {"sensory", "prefrontal", "recall", "next", "motor", "all", "bias_association",
            "moved_association", "moved_motor", "nudged_steps"} <= set(seen)


@pytest.mark.parametrize("arm", ["recipe", "deep"])
def test_the_settledness_audit_leaves_the_life_to_the_bit(arm):
    spec = {"arm": arm, "founder": 0, "batches": 6, "module": 16, "panel_batches": 1,
            "panels": ["clause"]}
    plain = gx.run_life(spec)
    audited = gx.run_settled({**spec, "sample_every": 2})
    assert plain["readings"][0]["panels"] == audited["readings"][0]["panels"]
    assert plain["readings"][0]["work"]["lessons"] == audited["readings"][0]["work"]["lessons"]
    audit = audited["audit"]
    assert audit["act"]["life"]["rows"] > 0
    assert 0.0 <= audit["act"]["life"]["answer_changed"] <= 1.0
    assert audit["act"]["life"]["motor_max"] >= audit["act"]["life"]["motor_p50"] >= 0.0
    assert set(audited["sentences_changed"]) == {"clause"}
    assert "clause" in audited["notes"]["panels_tight"]


def test_a_replica_differs_at_the_last_bits_only():
    brain, other = gr.make_brain("recipe", 0, module=16), gr.make_brain("recipe", 0, module=16)
    gx.replicate(other, 3)
    a, b = brain.learner.brain.efficacy, other.learner.brain.efficacy
    assert not np.array_equal(a, b) and np.abs(a - b).max() < 1e-12
    paired = other.learner.reverse >= 0
    assert np.array_equal(b[paired], b[other.learner.reverse[paired]])
    third = gr.make_brain("recipe", 0, module=16)
    gx.replicate(third, 0)
    assert np.array_equal(a, third.learner.brain.efficacy)


@pytest.mark.parametrize("panel", ["clause", "dyck", "path"])
def test_the_gradient_through_time_is_the_sentences(taught, tests, panel):
    unrolled = gx.Unrolled(copy.deepcopy(taught))
    report = gx.check_time(unrolled, tests[panel][0][:8])
    assert report["worst_error"] < 1e-7 and report["largest_gradient"] > 1e-3
    families = {e["family"] for e in report["entries"]}
    assert {"sa", "pa", "ra", "na", "am", "ba", "bm", "qx", "qc", "qb", "posx", "ancx", "relx",
            "slox"} <= families
    # the gradient reaches the address through the cortex of the moment before
    assert max(abs(e["exact"]) for e in report["entries"] if e["family"] == "qc") > 1e-4


@pytest.mark.parametrize("panel", ["clause", "dyck", "path"])
def test_the_unrolled_forward_is_the_librarys(newborn, taught, tests, panel):
    rows = tests[panel][0][:8]
    fresh = gx.agreement(newborn, rows)
    assert fresh["read"] < 1e-9 and fresh["association"] < 1e-3 and fresh["motor"] < 1e-3
    assert fresh["words_equal"] == 1.0
    later = gx.agreement(taught, rows)
    assert later["read"] < 0.02 and later["association"] < 0.02 and later["motor"] < 0.02
    assert later["words_equal"] == 1.0


def test_at_the_fixed_point_the_two_forwards_are_one_function(taught, tests):
    for panel in ("clause", "dyck", "path"):
        tight = gx.agreement(gx.tightened(taught), tests[panel][0][:8])
        assert tight["read"] < 1e-7 and tight["association"] < 1e-7 and tight["motor"] < 1e-7
        assert tight["words_equal"] == 1.0


def test_the_numpy_polish_holds_at_the_activations_kink(newborn):
    """The moment's polish on states left near zero, Newton's direction stuck across the kink."""
    brain = copy.deepcopy(newborn)
    rng = np.random.default_rng(3)
    graph = brain.connectome
    state = brain.learner.brain.settle_batch(
        rng.normal(scale=0.02, size=(32, graph.n)), steps=4)
    system = gx.Settling(brain.learner.brain, state)
    v, report = gx.polish(system, state)
    assert report["unpolished"] == 0 and np.abs(system.residual(v)).max() < 1e-9


def test_the_polish_holds_at_the_activations_kink(newborn):
    """Many neurons near zero, where the slope jumps tenfold and a full Newton step cycles."""
    unrolled = gx.Unrolled(copy.deepcopy(newborn))
    rng = np.random.default_rng(11)
    units, words = unrolled.params["am"].shape
    for gain in (0.5, 0.9, 1.1):
        a = rng.normal(size=(units, words))
        a *= gain * 2 / np.linalg.svd(a, compute_uv=False)[0]  # the loop's gain at the slope 1/2
        a = torch.from_numpy(a)
        field = torch.from_numpy(rng.normal(scale=0.02, size=(64, units)))
        bm = torch.from_numpy(rng.normal(scale=0.02, size=words))
        rest = torch.zeros(64, units, dtype=torch.float64), torch.zeros(64, words,
                                                                       dtype=torch.float64)
        _, _, _, error, _ = unrolled._fixed_point(field, field, a, bm, *rest)
        assert error < 1e-9, gain


def test_the_unrolled_parameters_are_the_lessons_and_return_to_a_brain(taught, tests):
    brain = copy.deepcopy(taught)
    unrolled = gx.Unrolled(brain)
    graph = brain.connectome
    sources = graph.n - len(graph.populations["association"]) - len(graph.populations["motor"])
    critic = len(graph.populations["association"]) + 1
    assert unrolled.count() + sources + critic == brain.parameters()
    before = brain.learner.brain.efficacy.copy(), brain.recall.head_map(0)[1].copy()
    unrolled.write_back()
    assert np.array_equal(brain.learner.brain.efficacy, before[0])
    assert np.array_equal(brain.recall.head_map(0)[1], before[1])
    with torch.no_grad():
        unrolled.params["am"][3, 2] += 0.25
        unrolled.params["qc"][1, 4] += 0.5
    unrolled.write_back()
    index = unrolled.index
    moved = brain.learner.brain.efficacy
    expected = before[0][index["am"][3, 2]] + 0.25
    assert moved[index["am"][3, 2]] == moved[index["ma"][2, 3]] == expected
    assert brain.recall.head_map(0)[1][1, 4] == before[1][1, 4] + 0.5
    assert brain.brain is brain.learner.brain


def test_a_read_beside_the_lessons_leaves_the_recipe_to_the_bit(tests):
    rng_a = np.random.default_rng(5)
    split = gr.Split.of(0)
    recipe, observed = gr.make_brain("recipe", 0, module=16), gr.make_brain("recipe", 0, module=16)
    gx.install_exact(observed, "observe")
    for _ in range(12):
        batch = gr.train_batch(rng_a, split, "g1")
        gr.speak(recipe, batch, teach=True)
        gr.speak(observed, batch, teach=True)
    assert np.array_equal(recipe.learner.brain.efficacy, observed.learner.brain.efficacy)
    assert np.array_equal(recipe.learner.brain.bias, observed.learner.brain.bias)
    assert np.array_equal(recipe.recall.head_map(0)[0], observed.recall.head_map(0)[0])
    assert observed.learner.exact_notes["gradients"] > 0


@pytest.mark.parametrize("step", ["normalized", "adam"])
def test_the_moments_rung_replaces_the_step_and_nothing_else(step):
    rng = np.random.default_rng(5)
    split = gr.Split.of(0)
    recipe, exact = gr.make_brain("recipe", 0, module=16), gr.make_brain("recipe", 0, module=16)
    gx.install_exact(exact, step)
    lessons = {"recipe": gr.new_work(), "exact": gr.new_work()}
    for _ in range(6):
        batch = gr.train_batch(rng, split, "g1")
        gr.speak(recipe, batch, teach=True, work=lessons["recipe"])
        gr.speak(exact, batch, teach=True, work=lessons["exact"])
    learner = exact.learner
    assert not np.array_equal(recipe.learner.brain.efficacy, learner.brain.efficacy)
    assert learner.exact_notes["gradients"] > 0 and learner.exact_notes["unpolished_rows"] == 0
    assert lessons["exact"]["lessons"] > 0
    # the address map still takes its lesson, the sources' biases never move, the pairs stay tied
    assert exact.recall.head_map(0)[0].any()
    assert not learner.brain.bias[learner.exact_sources].any()
    paired = learner.reverse >= 0
    assert np.array_equal(learner.brain.efficacy[paired],
                          learner.brain.efficacy[learner.reverse[paired]])


@pytest.mark.parametrize("arm", ["G1n", "G1a", "G1d", "G2a", "G2s", "G2n", "G2t", "G2r", "G2m",
                                 "G2c", "G2o", "G2nt", "G2nr", "G2nm", "G2nc", "G2no", "G2ns",
                                 "G2np", "G2nms", "G2nmp", "G2nop", "recipe", "deep", "deepna",
                                 "Ln", "Ld", "Lda", "Lb", "whole", "Fn", "Fn2", "Fd", "Fn3",
                                 "Fn100", "Lf", "FnT", "FnC", "FnR", "FnX", "Lf3", "Lf1", "Fu3",
                                 "Fu10", "Fcap", "FdT", "FdC", "FdR", "FdX", "Fd128", "FdT128f",
                                 "FdT256f", "FdA64", "FdA80f", "Fd2", "FdL"])
def test_a_short_life_of_every_arm(arm):
    life = gx.run_life({"arm": arm, "founder": 0, "batches": 3, "module": 16, "panel_batches": 1,
                        "panels": ["clause", "path"]})
    panels = life["readings"][0]["panels"]
    assert life["row"] == arm and set(panels) == {"clause", "path"}
    assert 0.0 <= panels["clause"]["exact"] <= 1.0 and "next_word" in panels["clause"]
    if gx.ARMS[arm].get("rung") == 2:
        assert life["notes"]["taught"] > 0 and life["notes"]["worst_residual"] < 1e-8


def test_a_wide_arm_runs_the_steps_it_names():
    composed = gx.newborn(gx.ARMS["Fd128"], 0, {})
    fan_in = gx.newborn(gx.ARMS["Fd128f"], 0, {})
    narrow = gr.make_brain("recipe", 0, settings={"teaching": "deep"}, module=64)
    c = fan_in.connectome
    assert len(c.populations["association"]) == 128
    assert composed.learner.synapse_rate is None
    rates = fan_in.learner.synapse_rate
    motor = np.isin(c.post, c.populations["motor"])
    association = np.isin(c.post, c.populations["association"])
    assert (motor | association).all()
    assert np.allclose(rates[motor], 64 / 128)  # a motor neuron hears the whole module
    trace = np.isin(c.pre, c.populations["prefrontal"])
    assert trace.sum() == 128 * 128  # width's square is the trace's
    assert np.allclose(rates[association], (64 + 123) / (128 + 123))
    assert np.array_equal(gx.fan_in_rates(narrow, narrow), np.ones(len(narrow.connectome.pre)))


def test_an_absent_region_leaves_nothing_of_it():
    silent = gx.newborn(gx.ARMS["FdT"], 0, {})
    absent = gx.newborn(gx.ARMS["FdA64"], 0, {})
    assert "prefrontal" not in absent.connectome.populations and absent.working_memory is None
    assert "prefrontal" in gr.make_brain("recipe", 0).connectome.populations  # none leaks
    gone = 64 * 64 + 64  # its synapses and its neurons' frozen biases
    assert absent.parameters() == silent.parameters() - gone
    assert len(absent.connectome.pre) == len(silent.connectome.pre) - 64 * 64
    fan = np.bincount(absent.connectome.post, minlength=absent.connectome.n)
    assert fan[np.asarray(absent.connectome.populations["association"])].max() == 187 - 64


def test_a_brains_passes_stop_by_the_controls_rule_and_read_each_epoch(monkeypatch, tmp_path):
    monkeypatch.setattr(gx, "EPOCHS", 3)
    log = tmp_path / "epochs.jsonl"
    life = gx.run_life({"arm": "Fd", "founder": 0, "mode": "offline", "batches": 2, "epochs": 0,
                        "module": 16, "panel_batches": 1, "panels": ["clause"],
                        "epoch_log": str(log)})
    history = life["notes"]["epochs"]
    assert 1 <= len(history) <= 3
    assert [h["epoch"] for h in history] == list(range(1, len(history) + 1))
    assert all(0.0 <= h["error"] <= 1.0 and "clause" in h["panels"] for h in history)
    assert sum(h["work"]["rows"] for h in history) == life["readings"][0]["work"]["rows"]
    assert len(log.read_text().splitlines()) == len(history)
    assert life["readings"][0]["epochs"] == len(history)


@pytest.mark.parametrize("arm, heads", [("Fd", 1), ("Fd2", 2)])
def test_the_verbs_read_names_each_head(arm, heads):
    life = gx.run_life({"arm": arm, "founder": 0, "batches": 2, "module": 16, "panel_batches": 1,
                        "panels": ["clause"], "verb_heads": True, "verb_batches": 1})
    reading = life["notes"]["verb_heads"]
    assert set(reading) == {"L0", "L1", "L2"}
    for cell in reading.values():
        assert cell["n"] == gr.BATCH and cell["said"] <= cell["n"]
        assert cell["said"] + sum(cell[k] for k in ("other_number", "other_verb", "near", "noun",
                                                     "other")) == cell["n"]
        assert len(cell["heads"]) == heads
        for head in cell["heads"].values():
            parts = head["verb"] + head["agent"] + head["mods"]
            assert 0.0 <= parts <= 1.0 + 1e-9 and head["slots"] >= 1.0 - 1e-9
        assert ("shared" in cell) == (heads == 2)


def _gradient(brain, rows, **cuts):
    unrolled = gx.Unrolled(copy.deepcopy(brain), **cuts)
    xs, ys = gx.encode(rows)
    # a word taught at the last moment alone: what reaches a parameter came through the carries
    ys = ys.clone()
    ys[:-1] = -1
    loss, _ = unrolled.forward(xs, ys)
    loss.backward()
    return unrolled, {name: p.grad for name, p in unrolled.params.items()}


def test_each_cut_stops_the_path_it_names(taught, tests):
    rows = tests["clause"][0][:8]
    _, whole = _gradient(taught, rows)
    _, trace = _gradient(taught, rows, carry_address=False)
    _, address = _gradient(taught, rows, carry_trace=False)
    _, none = _gradient(taught, rows, carry_address=False, carry_trace=False)
    # the cuts change the gradient, one path at a time, and neither cut equals the whole
    for name in ("sa", "pa", "am"):
        assert not torch.equal(whole[name], trace[name])
        assert not torch.equal(whole[name], address[name])
        assert not torch.equal(trace[name], none[name])
        assert not torch.equal(address[name], none[name])
    # the forward's value is untouched by a cut: the same moments, the same fixed points
    xs, ys = gx.encode(rows)
    with torch.no_grad():
        a, _ = gx.Unrolled(copy.deepcopy(taught)).forward(xs, ys)
        b, _ = gx.Unrolled(copy.deepcopy(taught), carry_address=False, carry_trace=False).forward(
            xs, ys)
    assert float(a) == float(b)


def test_frozen_maps_stay_at_birth_and_the_read_stays_content(tests):
    brain = gr.make_brain("recipe", 0, module=16)
    unrolled = gx.Unrolled(brain, frozen_maps=True, carry_address=False)
    stepper = gx.Stepper(unrolled, "adam", 0.003)
    before = {name: p.detach().clone() for name, p in unrolled.params.items()}
    for _ in range(3):
        loss, _ = unrolled.forward(*gx.encode(tests["clause"][0]))
        stepper(loss)
    for name in gx.Unrolled.MAPS:
        assert torch.equal(unrolled.params[name], before[name]), name
    assert not torch.equal(unrolled.params["am"], before["am"])
    assert unrolled.count() > sum(p.numel() for p in unrolled.parameters())


def test_the_reads_local_credit_is_the_descent_as_the_nudge_shrinks(taught, tests):
    """The lesson's readback in the gradient's form against the gradient of a moment's cost through
    the read, by finite differences of each row's read over the slots: a fourth path to one
    direction. The step a row takes moves the slots' logits along its centred scores."""
    brain = copy.deepcopy(taught)
    gx.install_read_credit(brain)
    rows = tests["clause"][0][:8]
    recall, learner = brain.recall, brain.learner
    drive, labels = gx._settled_drive(brain, rows)
    graph = learner.brain
    free = graph.equilibrate(drive, budget=200000, chunk=64, tolerance=1e-12).state
    target = learner.targets(labels)
    beta = 1e-4
    plus = graph.equilibrate(drive, state=free, budget=200000, chunk=64, tolerance=1e-12,
                             nudge=learner.nudge_for(target, beta)).state
    minus = graph.equilibrate(drive, state=free, budget=200000, chunk=64, tolerance=1e-12,
                              nudge=learner.nudge_for(target, -beta)).state
    movement = (plus.activation - minus.activation)[:, recall.association]
    p = recall.attended_heads[0]
    found = recall._row_steps(recall.last, np.arange(len(labels)),
                              recall.readback(graph, "recall", movement),
                              recall.readback(graph, "next", movement))
    moving, _, _, _, _, _, chosen, _ = found
    assert moving.all()
    local = (chosen - p)  # each row's centred scores over the slots, the logits' direction
    regions = (np.asarray(brain.connectome.populations["recall"]),
               np.asarray(brain.connectome.populations["next"]))

    def cost_at(row: int, slot: int, step: float) -> float:
        logits = np.log(np.maximum(p, 1e-300))
        logits[row, slot] += step
        moved = np.exp(logits)
        moved /= moved.sum(axis=1, keepdims=True)
        changed = np.array(drive, dtype=float)
        changed[:, regions[0]] = recall.amplitude * np.einsum("bm,bmc->bc", moved, recall.values)
        changed[:, regions[1]] = recall.amplitude * np.einsum("bm,bmc->bc", moved, recall.nexts)
        return gx._cost_at(learner, changed, labels, graph, free)

    cosines = []
    for row in range(3):
        exact = np.array([(cost_at(row, slot, 1e-5) - cost_at(row, slot, -1e-5)) / 2e-5
                          for slot in range(p.shape[1])])
        cosines.append(gx.cosine(local[row], -exact))
    assert min(cosines) > 0.99, cosines


def test_the_named_rule_is_the_lanes_and_tends_to_the_first_order(taught, tests):
    """At the lesson's temperature the rule's error is ``K* - sum p K``, the library's own. As its
    temperature rises, scaled to hold the first order, the rule's direction tends to the first-order
    form on the same scores."""
    rows = tests["clause"][0][:8]
    brain = copy.deepcopy(taught)
    drive, labels = gx._settled_drive(brain, rows)
    moment = brain.recall.last
    rng = np.random.default_rng(4)
    movement = rng.normal(size=(len(rows), len(brain.recall.association)))
    graph = brain.learner.brain
    d_value = brain.recall.readback(graph, "recall", movement)
    d_next = brain.recall.readback(graph, "next", movement)
    every = np.arange(len(rows))
    plain = brain.recall._row_steps(moment, every, d_value, d_next)
    _, _, _, _, error_plain, _, named, attention = plain
    keys = brain.recall.keys
    mean_key = np.einsum("bm,bmk->bk", np.exp(attention), keys)
    # the library's error and the rule's are K* - sum p K
    assert np.allclose(error_plain, np.einsum("bm,bmk->bk", named, keys) - mean_key)

    def error_at(temperature: float, install) -> np.ndarray:
        fork = copy.deepcopy(brain)
        install(fork, temperature)
        found = fork.recall._row_steps(moment, every, d_value, d_next)
        return found[4]

    rule = error_at(0.1, gx.install_named_credit)
    assert np.allclose(rule, np.einsum("bm,bmk->bk", named, keys) - mean_key)
    first = copy.deepcopy(brain)
    gx.install_read_credit(first, no_slope=True)
    first_order = first.recall._row_steps(moment, every, d_value, d_next)[4]
    far = error_at(1000.0, gx.install_named_credit)
    assert np.allclose(far, first_order, atol=1e-2)
    assert gx.cosine(rule.ravel(), first_order.ravel()) < gx.cosine(far.ravel(),
                                                                    first_order.ravel())


def test_the_named_arm_is_the_librarys_lesson():
    """``Fd``'s patch and the library's address lesson live alike, to the bit."""

    def life(arm: str):
        brain = gx.newborn(gx.ARMS[arm], 0, {"module": 16})
        gx.install_arm(brain, gx.ARMS[arm])
        rng = np.random.default_rng(np.random.SeedSequence([0, gr.SEEDS["train"]]))
        split = gr.Split.of(0)
        for _ in range(20):
            gr.speak(brain, gr.train_batch(rng, split, "g1"), teach=True)
        return brain

    patched, option = life("Fd"), life("FdL")
    assert np.array_equal(patched.learner.brain.efficacy, option.learner.brain.efficacy)
    assert np.array_equal(patched.learner.brain.bias, option.learner.brain.bias)
    for name in ("head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps"):
        for a, b in zip(getattr(patched.recall, name)[0], getattr(option.recall, name)[0],
                        strict=True):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("part, read", [("trace", lambda b: b.working_memory.trace),
                                        ("cortex", lambda b: b.recall.previous),
                                        ("focus", lambda b: b.recall.focus + 1)])
def test_a_cut_clears_what_crosses_moments_before_every_lesson_and_act(tests, part, read):
    rows = tests["clause"][0]
    seen = []
    brain = gr.make_brain("recipe", 0, module=16)
    gx.install_cut(brain, [part])
    base = type(brain).__mro__[1]
    original = base.act

    def spy(self, observations, **options):
        seen.append(float(np.abs(read(self)).sum()))
        return original(self, observations, **options)

    base.act = spy
    try:
        gr.speak(brain, rows, teach=False)
        gr.speak(copy.deepcopy(brain), rows, teach=False)
    finally:
        base.act = original
    assert len(seen) > 8 and max(seen) == 0.0


def test_the_reads_local_credit_replaces_the_address_lesson_only():
    rng = np.random.default_rng(5)
    split = gr.Split.of(0)
    recipe, credited = gr.make_brain("recipe", 0, module=16), gr.make_brain("recipe", 0, module=16)
    gx.install_read_credit(credited)
    for _ in range(4):
        batch = gr.train_batch(rng, split, "g1")
        gr.speak(recipe, batch, teach=True)
        gr.speak(credited, batch, teach=True)
    a, b = recipe.recall.head_map(0), credited.recall.head_map(0)
    assert b[0].any() and not np.array_equal(a[0], b[0])
    assert credited.recall.position_maps[0][2].any()
    assert np.isfinite(b[0]).all() and np.isfinite(credited.recall.anchor_maps[0][0]).all()
    assert isinstance(copy.deepcopy(credited).recall, type(credited.recall))


def test_a_step_per_moment_teaches_each_moments_wrong_words_at_that_moments_parameters(tests):
    rows = tests["clause"][0]
    brain = gr.make_brain("recipe", 0, module=16)
    unrolled = gx.Unrolled(brain)
    stepper = gx.Stepper(unrolled, "normalized", 0.003)
    xs, ys = gx.encode(rows)
    before = unrolled.params["am"].detach().clone()
    went = gx.teach_by_moment(unrolled, stepper, xs, ys)
    # a newborn says most words wrong: a step for each moment that has one, within one sentence
    assert 1 < went["steps"] <= int((ys >= 0).any(dim=1).sum())
    assert 0 < went["wrong"] <= went["taught"] == int((ys >= 0).sum())
    assert stepper.count == went["steps"]
    assert not torch.equal(unrolled.params["am"], before)
    # a sentence with no wrong word takes no step
    said = gx.Unrolled(gr.make_brain("recipe", 0, module=16))
    idle = gx.Stepper(said, "normalized", 0.003)
    with torch.no_grad():
        _, seen = said.forward(xs, ys, record=True)
    right = ys.clone()
    for t in range(len(right)):
        taught = right[t] >= 0
        right[t][taught] = torch.from_numpy(seen["motor"][t].argmax(axis=1))[taught]
    assert gx.teach_by_moment(said, idle, xs, right)["steps"] == 0


@pytest.mark.parametrize("arm", ["G2n", "G2nm", "G2np", "G1a", "recipe"])
def test_an_arms_life_on_the_crossing(arm):
    life = gx.run_crossing({"arm": arm, "founder": 0, "batches": 2, "module": 16,
                            "panel_batches": 1})
    panel = life["readings"][0]["panels"]["crossing"]
    assert life["mode"] == "crossing" and life["row"] == arm
    assert 0.0 <= panel["exact"] <= 1.0 and set(panel["conditions"]) <= {"c0k0", "c0k1", "c1k0",
                                                                          "c1k1"}


def test_the_keyed_rows_ask_the_fact_the_context_and_question_cross():
    rng = np.random.default_rng(228)
    rows = gx.keyed(rng, 64)
    seen = set()
    for r in rows:
        seen.add((r.c, r.k, r.first))
        assert r.answer == r.values[r.c ^ r.k] and r.fact == r.c ^ r.k
        assert r.values[0] != r.values[1]
        order = (r.first, 1 - r.first)
        assert [int(np.argmax(x)) for x in r.pre[:4]] == [
            gx.NAMES[order[0]], r.values[order[0]], gx.NAMES[order[1]], r.values[order[1]]]
        assert np.argmax(r.pre[4]) == gr.OPENS[r.c]
        assert np.argmax(r.question) == gr.MATCH[gr.OPENS[r.k]]
    assert len(seen) == 8  # every context, question and order
    by_name = gx.keyed(rng, 32, direct=1.0)
    assert all(r.k == -1 and r.answer == r.values[r.fact] for r in by_name)
    xs, ys = gx.keyed_moments(rows, "own")
    assert xs.shape[0] == 6 and (ys[:5] == -1).all() and (ys[5] >= 0).all()
    xs, ys = gx.keyed_moments(rows, "go")
    assert xs.shape[0] == 8 and (ys[:6] == -1).all() and (ys[7] == gr.END).all()
    assert xs[6, 0, gr.C["GO"]] == 1.0


@pytest.mark.parametrize("arm", ["recipe", "Ld"])
def test_a_brains_lessons_on_the_keyed_crossing(arm):
    for timing in ("own", "go"):
        life = gx.run_keyed({"arm": arm, "founder": 0, "batches": 2, "timing": timing,
                             "module": 16, "panel_batches": 1})
        assert 0.0 <= life["readings"]["intact"]["right"] <= 1.0


@pytest.mark.parametrize("timing", ["own", "go"])
def test_an_arms_life_on_the_keyed_crossing(timing):
    life = gx.run_keyed({"arm": "G2n", "founder": 0, "batches": 2, "timing": timing, "module": 16,
                         "panel_batches": 1})
    readings = life["readings"]
    assert life["mode"] == "keyed_" + timing
    assert {"intact", "direct", "erase_relevant", "erase_other", "memory_off", "trace_off",
            "cortex_off", "both_off"} == set(readings)
    r = readings["intact"]
    assert 0.0 <= r["right"] <= 1.0 and 0.0 <= r["relevant"] <= 1.0 and r["slots"] >= 1.0
    assert all(len(row) == 5 and sum(row) <= 1.01 for row in r["mass"].values())
    assert r["relevant"] <= r["relevant_record"] and r["context_relevant"] >= 0.0


def test_a_resumed_session_skips_the_lives_it_holds(tmp_path):
    out = tmp_path / "lives.jsonl"
    specs = [{"arm": "G1n", "founder": 0, "mode": "online", "replica": 0},
             {"arm": "G1n", "founder": 0, "mode": "online", "replica": 0, "init_seed": 1000},
             {"arm": "G1n", "founder": 1, "mode": "online", "replica": 0},
             {"arm": "G1a", "founder": 0, "mode": "online", "replica": 0}]
    assert gx.pending(specs, out) == specs
    out.write_text('{"row": "G1n", "mode": "online", "founder": 0, "init_seed": 1000, '
                   '"replica": 0}\n')
    assert gx.pending(specs, out) == [specs[0], specs[2], specs[3]]


def test_an_arms_steps_move_its_cost_down(tests):
    rows = tests["clause"][0]
    for kind, rate in (("adam", 0.003), ("normalized", 0.003)):
        brain = gr.make_brain("recipe", 0, module=16)
        unrolled = gx.Unrolled(brain)
        stepper = gx.Stepper(unrolled, kind, rate)
        xs, ys = gx.encode(rows)
        first, _ = unrolled.forward(xs, ys)
        for _ in range(6):
            loss, _ = unrolled.forward(xs, ys)
            stepper(loss)
        last, _ = unrolled.forward(xs, ys)
        assert float(last) < float(first)
