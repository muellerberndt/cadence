"""Direct tests for the protocol instrument: levels, codes, the shuffled control, serialisation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

HERE = Path(__file__).resolve().parent


def _load(name: str):
    """The instrument module ``name`` of this folder, loaded once under its own module name."""
    key = "instruments_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


instrument = _load("protocol")
PREDICATES, Levels, shared_code = instrument.PREDICATES, instrument.Levels, instrument.shared_code


def reading(mean: float, fraction: float, **extra: float) -> dict[str, float]:
    return {"mean": mean, "fraction": fraction, **extra}


def test_levels_defaults_and_immutability() -> None:
    levels = Levels()
    assert (levels.active, levels.inactive, levels.margin) == (0.5, 0.2, 0.15)
    assert levels.code_level == 0.5
    with pytest.raises(AttributeError):
        levels.active = 0.9  # type: ignore[misc]


def test_predicates_hold_at_their_declared_boundaries() -> None:
    L = Levels()
    assert instrument.evaluate_predicate("active", reading(L.active, 0), reading(0, 0))
    assert instrument.evaluate_predicate("inactive", reading(L.inactive, 0), reading(0, 0))
    assert instrument.evaluate_predicate("sparse", reading(0, L.sparse_min), reading(0, 0))
    assert instrument.evaluate_predicate("sparse", reading(0, L.sparse_max), reading(0, 0))
    above = reading(0, L.sparse_max + 1e-9)
    assert not instrument.evaluate_predicate("sparse", above, reading(0, 0))


def test_a_dead_net_cannot_pass_reference_predicates_vacuously() -> None:
    dead = reading(0.0, 0.0)
    for predicate in ("reduced", "retained", "densified", "sparse"):
        assert not instrument.evaluate_predicate(predicate, dead, dead), predicate


def test_specific_needs_both_codes_and_a_small_overlap() -> None:
    apart = reading(0.5, 0.1, shared=0.1)
    together = reading(0.5, 0.1, shared=0.9)
    other = reading(0.5, 0.1)
    assert instrument.evaluate_predicate("specific", apart, other)
    assert not instrument.evaluate_predicate("specific", together, other)
    # no overlap reading
    assert not instrument.evaluate_predicate("specific", reading(0.5, 0.1), other)
    assert not instrument.evaluate_predicate("specific", apart, reading(0.0, 0.0))  # versus is dead
    assert "specific" in PREDICATES


def test_shared_code_is_the_share_of_the_union() -> None:
    a = np.array([1.0, 1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 1.0, 0.0])
    assert shared_code(a, b, range(4), 0.5) == pytest.approx(1 / 3)
    assert shared_code(a, a, range(4), 0.5) == 1.0
    assert shared_code(np.zeros(4), np.zeros(4), range(4), 0.5) == 0.0
    assert shared_code(a, b, [3], 0.5) == 0.0  # members restrict the code


def ring(n: int = 12) -> cd.Connectome:
    return cd.Connectome.from_synapses(
        n,
        pre=list(range(n)) * 2,
        post=[(i + 1) % n for i in range(n)] + [(i + 3) % n for i in range(n)],
        count=list(range(1, 2 * n + 1)),
        populations={"all": range(n)},
    )


def edges(c: cd.Connectome) -> list[tuple[int, int]]:
    return sorted(zip(c.pre.tolist(), c.post.tolist(), strict=True))


def test_shuffled_keeps_counts_signs_degrees_and_populations() -> None:
    c = ring()
    s = instrument.shuffled(c, seed=3)
    assert s.synapses == c.synapses
    for ends in ("pre", "post"):
        shuffled_ends, original_ends = getattr(s, ends), getattr(c, ends)
        np.testing.assert_array_equal(
            np.bincount(shuffled_ends, minlength=c.n), np.bincount(original_ends, minlength=c.n)
        )
    assert sorted(s.count.tolist()) == sorted(c.count.tolist())
    assert sorted(s.sign.tolist()) == sorted(c.sign.tolist())
    assert not (s.pre == s.post).any()  # no autapse is made
    assert edges(s) != edges(c)
    assert s.populations == c.populations
    assert s.label.endswith(":shuffled:3")


def test_shuffled_is_deterministic_per_seed() -> None:
    c = ring()
    assert edges(instrument.shuffled(c, seed=1)) == edges(instrument.shuffled(c, seed=1))
    assert edges(instrument.shuffled(c, seed=1)) != edges(instrument.shuffled(c, seed=2))


def test_shuffled_leaves_kept_synapses_in_place() -> None:
    c = ring()
    keep = np.zeros(c.synapses, bool)
    keep[:5] = True
    kept = set(zip(c.pre[:5].tolist(), c.post[:5].tolist(), strict=True))
    assert kept <= set(edges(instrument.shuffled(c, seed=0, keep=keep)))
    with pytest.raises(ValueError, match="keep"):
        instrument.shuffled(c, seed=0, keep=np.ones(c.synapses - 1, bool))


def test_to_dict_is_json_and_carries_every_row_field() -> None:
    row = instrument.Row("r", "touch", "tail", "reduced", ablate=("head",), relative_to="touch")
    protocol = instrument.Protocol(
        stimuli={"touch": ("head",)}, rows=[row], training=[("touch", "head", "active")]
    )
    d = json.loads(json.dumps(protocol.to_dict()))
    assert d["rows"][0] == {
        "id": "r",
        "stimulus": "touch",
        "readout": "tail",
        "predicate": "reduced",
        "reference": "",
        "ablate": ["head"],
        "relative_to": "touch",
        "tier": "experiment",
        "versus": "",
    }
    assert d["training"] == [["touch", "head", "active"]]
    assert d["levels"]["active"] == Levels().active
    assert d["predicates"] == PREDICATES


def test_score_counts_passes_by_tier() -> None:
    connectome = cd.Connectome.from_synapses(2, pre=[], post=[], populations={"a": [0], "b": [1]})
    protocol = instrument.Protocol(
        stimuli={"none": ()},
        rows=[
            instrument.Row("quiet a", "none", "a", "inactive", tier="core"),
            instrument.Row("quiet b", "none", "b", "inactive", tier="extra"),
            instrument.Row("loud b", "none", "b", "active", tier="extra"),
        ],
        steps=5,
    )
    result = protocol.score(cd.NeuralGraph(connectome, cd.NeuronModel(dt=1.0)))
    assert result["total"] == 3 and result["passed"] == 2
    assert result["passed_by_tier"] == {"core": 1, "extra": 1}


def head_tail_ring(n: int = 6, count: float = 120.0) -> cd.Connectome:
    pre = list(range(n))
    post = [(i + 1) % n for i in range(n)]
    return cd.Connectome.from_synapses(
        n, pre=pre, post=post, count=[count] * n, populations={"head": [0], "tail": [n - 1]}
    )


def test_protocol_scores_and_shuffle_keeps_counts() -> None:
    w = head_tail_ring(8)
    protocol = instrument.Protocol(
        stimuli={"baseline": (), "poke": ("head",)},
        training=[("poke", "tail", "active")],
        rows=[
            instrument.Row("R1", "baseline", "tail", "inactive", "nothing in, nothing out"),
            instrument.Row("R2", "poke", "tail", "reduced", "cutting the ring", ablate=("head",)),
        ],
        steps=80,
    )
    brain = cd.NeuralGraph(w, cd.NeuronModel(gain=0.03))
    report = protocol.score(brain)
    assert report["training_passed"] == 1 and report["passed"] == 2
    control = instrument.shuffled(w, seed=0)
    assert (
        control.count.sum() == w.count.sum()
        and control.out_degree().tolist() == w.out_degree().tolist()
    )
    assert control.populations == w.populations
    gain, table = instrument.select_gain(
        lambda g: cd.NeuralGraph(w, cd.NeuronModel(gain=g)),
        protocol,
        (0.005, 0.03, 0.2),
        sparsity_cap=None,  # a toy ring lights entirely; the cap is for large nets
    )
    assert gain == 0.03 and [row["admissible"] for row in table] == [True, True, True]


def test_shuffled_keeps_degrees_and_changes_connectome() -> None:
    w = cd.layered(6, 4, 2, density=0.6, seed=2)
    s = instrument.shuffled(w, seed=1)
    assert s.synapses == w.synapses and s.n == w.n
    assert s.digest() != w.digest()
    keep = np.zeros(w.synapses, dtype=bool)
    keep[:2] = True
    kept = instrument.shuffled(w, seed=1, keep=keep)
    assert kept.synapses == w.synapses


def test_protocol_stimulates_all_neurons_of_a_two_neuron_brain():
    graph = cd.Connectome.from_synapses(2, pre=[], post=[], populations={"all": [0, 1]})
    brain = cd.NeuralGraph(graph, cd.learning_neuron_model(stimulus_amplitude=3))
    row = instrument.Row("all-on", "on", "all", "active")
    protocol = instrument.Protocol({"on": ("all",)}, [row])
    report = protocol.score(brain)
    assert report["passed"] == 1
    assert report["rows"][0]["reading"]["fraction"] == 1
