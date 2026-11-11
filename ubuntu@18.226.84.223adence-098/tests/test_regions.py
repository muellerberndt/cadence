"""Regions: blank and designed, developed into one connectome, and the standard regions."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd
from cadence.genome import Genome, Projection, develop, mutate
from cadence.regions import Region, cortex, motor_cortex, prefrontal_cortex, visual_cortex


def test_a_blank_genome_develops_exactly_as_before_designed_regions() -> None:
    genome = Genome(
        regions=(Region("input", 6), Region("hidden", 8), Region("output", 3)),
        projections=(
            Projection("input", "hidden", density=0.5),
            Projection("hidden", "output"),
            Projection("output", "output", sign=-1.0, reciprocal=False),
        ),
        label="toy",
    )
    # digests of the same genome developed by cadence 0.8.1
    assert develop(genome, seed=0).digest() == (
        "8f1bb761bc9d8c3c66186449ef39603c5f3639f2e29f71335b16c990a910ce32"
    )
    assert develop(genome, seed=3).digest() == (
        "19c0286b3312c72e1ce86d950b5f73997efd3681e61a5a09b78647bf5f411c72"
    )


def test_visual_cortex_has_local_receptive_fields() -> None:
    v1 = visual_cortex(6, 5, channels=2, features=3, field=3, stride=1)
    assert v1.designed and v1.inputs == "input" and v1.outputs == "output"
    circuit = v1.circuit
    assert circuit is not None
    rows, cols = 4, 3
    assert len(circuit.populations["input"]) == 6 * 5 * 2
    assert len(circuit.populations["output"]) == 3 * rows * cols
    indegree = circuit.in_degree()[list(circuit.populations["output"])]
    assert (indegree == 2 * 3 * 3).all()
    assert circuit.in_degree()[list(circuit.populations["input"])].sum() == 0
    # feature 1 at row 2, column 1 reads pixels rows 2..4, columns 1..3 of both channels
    neuron = 60 + (1 * rows + 2) * cols + 1
    pixels = circuit.pre[circuit.post == neuron]
    y, x = (pixels // 2) // 5, (pixels // 2) % 5
    assert set(y.tolist()) == {2, 3, 4} and set(x.tolist()) == {1, 2, 3}


def test_designed_regions_develop_into_one_connectome_and_keep_their_size() -> None:
    genome = Genome(
        regions=(visual_cortex(4, 4, features=2), cortex(10), motor_cortex(3, lateral=-0.5)),
        projections=(Projection("visual", "association"), Projection("association", "motor")),
    )
    connectome = develop(genome, seed=1)
    p = connectome.populations
    assert connectome.n == 16 + 2 * 4 + 10 + 3
    assert p["visual/input"] == tuple(range(16)) and p["visual/output"] == tuple(range(16, 24))
    assert p["motor/actions"] == tuple(range(34, 37))
    # the projection starts at the visual outputs, never at the pixels
    to_association = connectome.post >= 24
    assert set(connectome.pre[to_association & (connectome.pre < 24)].tolist()) <= set(
        range(16, 24)
    )
    # the motor cortex keeps its lateral inhibition
    lateral = (connectome.pre >= 34) & (connectome.post >= 34)
    assert lateral.sum() == 6 and np.allclose(connectome.sign[lateral], -0.5)
    rng = np.random.default_rng(0)
    child = mutate(genome, rng, size_step=1.0)
    assert child.region("visual").size == 24 and child.region("motor").size == 3
    assert child.region("association").size != 10 or mutate(child, rng, size_step=1.0) != child


def test_a_projection_can_name_one_population_of_a_designed_region() -> None:
    genome = Genome(
        regions=(visual_cortex(4, 4, features=1), Region("pulvinar", 2)),
        projections=(Projection("visual/input", "pulvinar", reciprocal=False),),
    )
    connectome = develop(genome, seed=0)
    into = connectome.post >= 25
    assert set(connectome.pre[into].tolist()) <= set(range(16))


def test_region_records_round_trip_with_the_designed_regions_supplied() -> None:
    v1 = visual_cortex(4, 4, features=1)
    genome = Genome((v1, Region("association", 5)), (Projection("visual", "association"),))
    record = genome.to_dict()
    assert record["regions"][0]["circuit"].startswith("visual-cortex")
    with pytest.raises(ValueError, match="supplied"):
        Genome.from_dict(record)
    assert Genome.from_dict(record, designed={"visual": v1}) == genome
    with pytest.raises(ValueError, match="differs"):
        Genome.from_dict(record, designed={"visual": visual_cortex(4, 4, features=1, seed=9)})


def test_region_validation() -> None:
    with pytest.raises(ValueError):
        Region("a/b", 3)
    with pytest.raises(ValueError):
        Region("a", 0)
    with pytest.raises(ValueError):
        Region("a", 3, inputs="x")
    with pytest.raises(ValueError):
        Region("a", 5, circuit=cd.Connectome.from_synapses(3, pre=[], post=[]))
    with pytest.raises(ValueError):
        motor_cortex(0)
    assert prefrontal_cortex(cortex(7)).size == 7 and cortex(7, lateral=-1.0).designed
