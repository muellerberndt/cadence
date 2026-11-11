"""Keep the confirmed school recipe independent of changing library defaults."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cadence import Brain

ROOT = Path(__file__).parent


@pytest.fixture
def scripts(monkeypatch):
    """Load the standalone benchmark imports without retaining module overrides."""
    modules = {}
    for name in (
        "extract",
        "run",
        "relations",
        "relation_development",
        "continual",
        "online_curriculum",
        "online_half_step_confirmation",
        "rate_development",
        "smooth_development",
        "lateral_development",
        "normalized_development",
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


def assert_same_initial_graph(actual, expected):
    """Compare wiring and starting parameters, including preserved memory regions."""
    for name in ("pre", "post", "sign", "count"):
        np.testing.assert_array_equal(
            getattr(actual.connectome, name), getattr(expected.connectome, name)
        )
    assert set(actual.connectome.populations) == set(expected.connectome.populations)
    for name in actual.connectome.populations:
        np.testing.assert_array_equal(
            actual.connectome.populations[name], expected.connectome.populations[name]
        )
    for name in ("efficacy", "bias", "log_gain"):
        np.testing.assert_array_equal(getattr(actual.brain, name), getattr(expected.brain, name))
    for name in ("sensory_index", "motor_index"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("plastic_synapses", "plastic_neurons"):
        np.testing.assert_array_equal(
            getattr(actual.learner, name), getattr(expected.learner, name)
        )
    for name in ("backend", "dense_limit", "precision"):
        assert getattr(actual.brain, name) == getattr(expected.brain, name)
    assert actual.brain.layout.to_dict() == expected.brain.layout.to_dict()
    assert actual.brain.neuron_model.to_dict() == expected.brain.neuron_model.to_dict()
    assert actual.working_memory is not None
    assert actual.hippocampus is not None
    assert actual.hippocampus.writes == 0
    assert not np.any(actual.hippocampus.consolidated)
    assert_same_memory(actual, expected)


def assert_same_memory(actual, expected):
    for name in ("amplitude", "decay", "source", "target"):
        assert getattr(actual.working_memory, name) == getattr(expected.working_memory, name)
    for name in ("cold", "glow", "hidden", "last", "trace"):
        np.testing.assert_array_equal(
            getattr(actual.working_memory, name), getattr(expected.working_memory, name)
        )
    for name in ("amplitude", "consolidation", "decay", "key_width", "normalize",
                 "rate", "replace", "rule", "separator", "writes"):
        assert getattr(actual.hippocampus, name) == getattr(expected.hippocampus, name)
    for name in ("consolidated", "mass", "post", "pre", "strength"):
        np.testing.assert_array_equal(
            getattr(actual.hippocampus, name), getattr(expected.hippocampus, name)
        )


@pytest.mark.parametrize("seed", (0, 3))
def test_canonical_school_preserves_historical_motor_wiring(scripts, seed):
    args = SimpleNamespace(phase_steps=4096, tolerance=0.003)
    school = scripts.relation_development.make_brain("canonical", seed, args)
    # A current wide compose readout deliberately has no lateral inhibition.
    # The already selected school must retain its historical explicit -0.5.
    reference = Brain.compose(
        650, 36, backend="cpu", modules=(32, 16), observers=(), lateral=-0.5,
        seed=seed, learning=school.learner.config, recall=False,
    )
    assert_same_initial_graph(school, reference)
    motor = school.motor_index
    wire = school.connectome
    lateral = np.isin(wire.pre, motor) & np.isin(wire.post, motor)
    assert int(lateral.sum()) == 36 * 35
    assert np.all(school.brain.efficacy[lateral] == -0.5)


def test_half_step_changes_only_selected_learning_rates(scripts):
    canonical = scripts.relation_development.make_brain(
        "canonical", 2, SimpleNamespace(phase_steps=4096, tolerance=0.003)
    )
    candidate = scripts.online_half_step_confirmation.make_brain(2)
    original, selected = canonical.learner.config.to_dict(), candidate.learner.config.to_dict()
    assert (original["eta"], original["eta_bias"]) == (0.2, 0.02)
    assert (selected["eta"], selected["eta_bias"]) == (0.1, 0.01)
    assert {name for name in original if original[name] != selected[name]} == {"eta", "eta_bias"}
    assert selected["free_steps"] == selected["nudged_steps"] == 4096
    assert selected["qualified"] is True
    assert selected["tolerance"] == 0.003
    assert selected["damping"] == 3
    assert selected["normalize"] == selected["momentum"] == 0.0
    assert_same_initial_graph(candidate, canonical)


def test_default_half_step_accepts_archived_three_argument_constructor(scripts, monkeypatch):
    original = scripts.relation_development.make_brain
    calls = []

    def historical(name, seed, args):
        calls.append((name, seed))
        return original(name, seed, args)

    canonical = original("canonical", 3, SimpleNamespace(phase_steps=4096, tolerance=0.003))
    monkeypatch.setattr(scripts.relation_development, "make_brain", historical)
    selected = scripts.online_half_step_confirmation.make_brain(3)
    assert calls == [("canonical", 3)]
    assert (selected.learner.config.eta, selected.learner.config.eta_bias) == (0.1, 0.01)
    assert_same_initial_graph(selected, canonical)


def test_quarter_rate_changes_exactly_the_two_fixed_rates(scripts):
    canonical = scripts.relation_development.make_brain(
        "canonical", 0, SimpleNamespace(phase_steps=4096, tolerance=0.003)
    )
    quarter = scripts.rate_development.make_brain()
    original, selected = canonical.learner.config.to_dict(), quarter.learner.config.to_dict()
    assert (selected["eta"], selected["eta_bias"]) == (0.05, 0.005)
    assert {name for name in original if original[name] != selected[name]} == {"eta", "eta_bias"}
    assert_same_initial_graph(quarter, canonical)
    confirmed = scripts.online_half_step_confirmation.make_brain(0, eta=0.05, eta_bias=0.005)
    assert confirmed.learner.config.to_dict() == selected
    assert_same_initial_graph(confirmed, quarter)


def test_smooth_gene_changes_only_leak_and_matches_confirmation(scripts):
    quarter = scripts.rate_development.make_brain()
    smooth = scripts.smooth_development.make_brain()
    before, after = quarter.brain.neuron_model.to_dict(), smooth.brain.neuron_model.to_dict()
    assert {name for name in before if before[name] != after[name]} == {"leak"}
    assert (before["leak"], after["leak"]) == (0.1, 1.0)
    assert smooth.learner.config.to_dict() == quarter.learner.config.to_dict()
    # Reset the declared neuron gene only for comparison of all preserved state.
    graph = smooth.brain
    smooth.learner.brain = type(graph)(
        graph.connectome, quarter.brain.neuron_model, efficacy=graph.efficacy,
        bias=graph.bias, log_gain=graph.log_gain, backend=graph.backend,
        dense_limit=graph.dense_limit, layout=graph.layout, precision=graph.precision,
    )
    assert_same_initial_graph(smooth, quarter)
    smooth = scripts.smooth_development.make_brain()
    confirmed = scripts.online_half_step_confirmation.make_brain(
        0, eta=0.05, eta_bias=0.005, leak=1.0
    )
    assert confirmed.learner.config.to_dict() == smooth.learner.config.to_dict()
    assert_same_initial_graph(confirmed, smooth)


@pytest.mark.parametrize("constructor", ("smooth", "confirmation"))
def test_leak_replacement_preserves_explicit_graph_runtime(scripts, monkeypatch, constructor):
    brain = scripts.relation_development.make_brain(
        "canonical", 0, SimpleNamespace(phase_steps=4096, tolerance=0.003)
    )
    if constructor == "smooth":
        brain = scripts.rate_development.make_brain()
    graph = brain.brain
    brain.learner.brain = type(graph)(
        graph.connectome, graph.neuron_model, efficacy=graph.efficacy,
        bias=graph.bias, log_gain=graph.log_gain, backend=graph.backend,
        dense_limit=17, layout=graph.layout, precision="float32",
    )
    if constructor == "smooth":
        monkeypatch.setattr(scripts.smooth_development, "QUARTER_FACTORY", lambda: brain)
        actual = scripts.smooth_development.make_brain()
    else:
        monkeypatch.setattr(scripts.relation_development, "make_brain", lambda *a, **kw: brain)
        actual = scripts.online_half_step_confirmation.make_brain(
            0, eta=0.05, eta_bias=0.005, leak=1.0
        )
    assert actual.brain.dense_limit == 17
    assert actual.brain.precision == "float32"
    assert actual.brain.layout is graph.layout
    assert actual.brain.neuron_model.leak == 1.0


def test_zero_lateral_gene_removes_exactly_motor_competition(scripts):
    quarter = scripts.rate_development.make_brain()
    selected = scripts.lateral_development.make_brain()
    scripts.lateral_development.assert_only_lateral_changed(selected, quarter)
    assert len(selected.connectome.pre) == len(quarter.connectome.pre) - 36 * 35
    assert not np.any(np.isin(selected.connectome.pre, selected.motor_index)
                      & np.isin(selected.connectome.post, selected.motor_index))
    assert_same_memory(selected, quarter)
    confirmed = scripts.online_half_step_confirmation.make_brain(
        0, eta=0.05, eta_bias=0.005, lateral=0.0
    )
    assert confirmed.learner.config.to_dict() == selected.learner.config.to_dict()
    assert_same_initial_graph(confirmed, selected)


def test_normalized_gene_preserves_graph_and_changes_only_declared_optimizer(scripts):
    baseline = scripts.lateral_development.make_brain()
    selected = scripts.normalized_development.make_brain()
    scripts.normalized_development.assert_only_optimizer_changed(selected, baseline)
    before, after = baseline.learner.config.to_dict(), selected.learner.config.to_dict()
    assert {name for name in before if before[name] != after[name]} == {
        "eta", "eta_bias", "normalize"
    }
    assert (after["eta"], after["eta_bias"], after["normalize"], after["momentum"]) == (
        0.003, 0.0003, 0.99, 0.0
    )
    assert_same_initial_graph(selected, baseline)
    for name in ("second_moment", "second_moment_bias", "velocity", "velocity_bias"):
        np.testing.assert_array_equal(
            getattr(selected.learner, name), getattr(baseline.learner, name)
        )
    assert selected.learner.contrast_updates == baseline.learner.contrast_updates == 0
    confirmed = scripts.online_half_step_confirmation.make_brain(
        0, eta=0.003, eta_bias=0.0003, normalize=0.99, momentum=0.0, lateral=0.0
    )
    assert confirmed.learner.config.to_dict() == after
    assert_same_initial_graph(confirmed, selected)


@pytest.fixture
def frozen_sources(scripts, tmp_path):
    scripts.relation_development.freeze_sources(tmp_path)
    source = tmp_path / "source"
    protocol = {
        "schema": "cadence-online-half-step-five-founder-confirmation-v2",
        "library_sources": scripts.run.sources(),
        "producing_sources": {
            "run.py": scripts.run.sha256(source / "run.py"),
            "relation_development.py": scripts.run.sha256(source / "relation_development.py"),
        },
        "relations_protocol_sha256": scripts.relations.protocol_hash(),
    }
    return source, protocol


@pytest.mark.parametrize("mutation", ("library", "producer", "imported", "relations"))
def test_confirmation_rejects_frozen_or_imported_source_drift(
    scripts, frozen_sources, monkeypatch, mutation
):
    source, protocol = frozen_sources
    scripts.online_half_step_confirmation.check_sources(protocol, source)
    if mutation == "library":
        path = source / "library/cadence" / next(iter(protocol["library_sources"]))
        path.write_bytes(path.read_bytes() + b"\n# changed after freezing\n")
        error = "frozen library source changed"
    elif mutation == "producer":
        path = source / "run.py"
        path.write_bytes(path.read_bytes() + b"\n# changed after freezing\n")
        error = "frozen producing source changed"
    elif mutation == "imported":
        imported = dict(protocol["library_sources"])
        imported[next(iter(imported))] = "0" * 64
        monkeypatch.setattr(scripts.run, "sources", lambda: imported)
        error = "imported library differs"
    else:
        monkeypatch.setattr(scripts.relations, "protocol_hash", lambda: "0" * 64)
        error = "relations protocol differs"
    with pytest.raises(ValueError, match=error):
        scripts.online_half_step_confirmation.check_sources(protocol, source)


@pytest.mark.parametrize("worker", ("confirmation", "rate", "smooth", "lateral", "normalized"))
def test_worker_checks_source_before_creating_founder_or_reading_panel(
    scripts, frozen_sources, tmp_path, monkeypatch, worker
):
    source, protocol = frozen_sources
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    path = source / "run.py"
    path.write_bytes(path.read_bytes() + b"\n# changed after freezing\n")

    def forbidden_panel(*args, **kwargs):
        raise AssertionError("worker read a panel before source admission")

    monkeypatch.setattr(scripts.relations, "load_panel", forbidden_panel)
    with pytest.raises(ValueError, match="frozen produc(?:ing|er) source changed"):
        if worker == "confirmation":
            scripts.online_half_step_confirmation.worker(tmp_path, 1)
        else:
            getattr(scripts, worker + "_development").worker(tmp_path)
    assert not (tmp_path / "seed-1").exists()
