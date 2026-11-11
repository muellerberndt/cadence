"""The simpler architecture control keeps the original local rule and gates."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent


@pytest.fixture
def single(monkeypatch):
    for name in (
        "extract",
        "run",
        "relations",
        "relation_development",
        "continual",
        "online_curriculum",
        "native_online",
        "single_module_native",
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["single_module_native"]


def test_exact_first_projection_external_ports_and_original_learning(single):
    original = single.ORIGINAL_FACTORY("qualified", 0, single.native.ARGS)
    candidate = single.make_brain("qualified", 0, single.native.ARGS)
    proof = single.construction_proof(candidate, original)
    assert proof["first_projection_byte_identical"]
    assert proof["external_ports_byte_identical"]
    assert proof["candidate"]["neurons"] == proof["original"]["neurons"] == 750
    assert proof["candidate"]["free_neurons"] == 68
    assert proof["original"]["free_neurons"] == 84
    assert candidate.learner.config.to_dict() == original.learner.config.to_dict()
    assert candidate.brain.neuron_model.to_dict() == original.brain.neuron_model.to_dict()
    assert candidate.learner.contrast_updates == original.learner.contrast_updates == 0
    assert not hasattr(candidate, "_local_centering")
    assert set(candidate.connectome.populations) == {
        "sensory",
        "association",
        "prefrontal",
        "motor",
        "motor/actions",
    }


def test_first_projection_detects_actual_numeric_change(single):
    original = single.ORIGINAL_FACTORY("qualified", 0, single.native.ARGS)
    candidate = single.make_brain("qualified", 0, single.native.ARGS)
    efficacy = candidate.brain.efficacy.copy()
    first = np.flatnonzero(np.isin(candidate.connectome.pre, candidate.sensory_index))[0]
    efficacy[first] += 1e-12
    candidate.learner.brain = candidate.brain.with_parameters(efficacy=efficacy)
    with pytest.raises(ValueError, match="sensory projection"):
        single.construction_proof(candidate, original)


@pytest.mark.parametrize(
    "change",
    [
        {"stage_counts": [2, 4, 24]},
        {
            "model_construction": {
                "inputs": 650,
                "actions": 36,
                "modules": [64],
                "observers": [],
                "lateral": 0.0,
            }
        },
        {"selected_correct": {"2": 1, "4": 3}},
        {"lesson_caps": {"2": 1024, "4": 2048}},
        {"output_cap_mib": 160},
    ],
)
def test_protocol_rejects_gene_gate_or_resource_search(single, change):
    protocol = deepcopy(single.FIXED)
    protocol["orders_including_next_lesson"] = {"2": [0] * 513, "4": [0] * 1025}
    protocol.update(change)
    with pytest.raises(ValueError, match="architecture/gene/gate/bound"):
        single.check_protocol(protocol)


def test_factory_rejects_different_founder(single):
    with pytest.raises(ValueError, match="founder0"):
        single.make_brain("qualified", 1, single.native.ARGS)


def test_factory_keeps_original_teacher_method(single):
    original = single.ORIGINAL_FACTORY("qualified", 0, single.native.ARGS)
    candidate = single.make_brain("qualified", 0, single.native.ARGS)
    assert candidate.learner.update.__func__ is original.learner.update.__func__
