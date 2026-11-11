"""A genome grows into a connectome; mutation and selection change it toward a fitness."""

from __future__ import annotations

import numpy as np

import cadence as cd
from cadence.genome import Genome, Projection, Region, develop, evolve, mutate


def two_region() -> Genome:
    return Genome(
        regions=(Region("input", 6), Region("hidden", 8), Region("output", 3)),
        projections=(Projection("input", "hidden", density=0.5), Projection("hidden", "output")),
        label="toy",
    )


def test_grow_is_deterministic_and_names_contiguous_sets() -> None:
    c = two_region()
    a, b = develop(c, seed=3), develop(c, seed=3)
    assert a.digest() == b.digest() and a.n == 17
    assert a.populations["hidden"] == tuple(range(6, 14)) and a.populations["output"] == tuple(
        range(14, 17)
    )
    assert (
        a.in_degree()[list(a.populations["input"])].sum() > 0
    )  # symmetric: inputs hear the hidden neurons
    brain = cd.NeuralGraph(a, cd.learning_neuron_model())
    assert brain.layout.ranges == 3
    assert develop(c, seed=4).digest() != a.digest()


def test_mutate_keeps_the_shape_and_respects_fixed_regions() -> None:
    rng = np.random.default_rng(0)
    child = mutate(two_region(), rng, fixed=("input", "output"))
    assert [r.name for r in child.regions] == ["input", "hidden", "output"]
    assert child.region("input").size == 6 and child.region("output").size == 3
    assert len(child.projections) == 2 and 0.01 <= child.projections[0].density <= 1.0


def test_evolve_moves_toward_the_fitness() -> None:
    # fitness: a hidden region of about twelve neurons with dense input projections
    def fitness(w: cd.Connectome, seed: int) -> float:
        hidden = len(w.populations["hidden"])
        density = w.in_degree()[list(w.populations["hidden"])].mean() / 6
        return -abs(hidden - 12) + density

    lineage = evolve(
        fitness,
        two_region(),
        generations=6,
        population=6,
        keep=2,
        seed=1,
        fixed=("input", "output"),
    )
    assert lineage.best is not None and len(lineage.generations) == 6
    first, last = lineage.generations[0]["best_fitness"], lineage.generations[-1]["best_fitness"]
    assert last >= first and abs(lineage.best.region("hidden").size - 12) <= 3


def _hidden_size(w: cd.Connectome, seed: int) -> float:
    return -abs(len(w.populations["hidden"]) - 12)


def test_evolve_runs_the_lives_through_the_mapper() -> None:
    from multiprocessing.pool import ThreadPool

    sequential = evolve(
        _hidden_size,
        two_region(),
        generations=3,
        population=4,
        keep=2,
        seed=3,
        fixed=("input", "output"),
    )
    with ThreadPool(2) as pool:
        parallel = evolve(
            _hidden_size,
            two_region(),
            generations=3,
            population=4,
            keep=2,
            seed=3,
            fixed=("input", "output"),
            mapper=pool.map,
        )
    assert [g["best_fitness"] for g in parallel.generations] == [
        g["best_fitness"] for g in sequential.generations
    ]
    assert parallel.best == sequential.best


def test_mutate_keeps_tied_regions_the_same_size() -> None:
    c = Genome(
        regions=(
            Region("input", 6),
            Region("context", 8),
            Region("hidden", 8),
            Region("output", 3),
        ),
        projections=(
            Projection("input", "hidden"),
            Projection("context", "hidden", reciprocal=False),
            Projection("hidden", "output"),
        ),
    )
    rng = np.random.default_rng(5)
    for _ in range(20):
        child = mutate(c, rng, fixed=("input", "output"), tied=(("hidden", "context"),))
        assert child.region("context").size == child.region("hidden").size
        c = child
    assert c.region("hidden").size != 8


def test_evolve_reports_after_every_generation() -> None:
    seen: list[int] = []
    lineage = evolve(
        _hidden_size,
        two_region(),
        generations=3,
        population=4,
        keep=2,
        seed=3,
        fixed=("input", "output"),
        report=lambda lin: seen.append(len(lin.generations)),
    )
    assert seen == [1, 2, 3] and len(lineage.generations) == 3


def test_genome_round_trips_through_its_dict() -> None:
    c = two_region()
    again = Genome.from_dict(c.to_dict())
    assert again == c
    assert develop(again, seed=4).synapses == develop(c, seed=4).synapses


def test_evolve_selects_any_genome_with_its_own_mutation() -> None:
    from cadence.genome import genes

    # a dict genome: a threshold on a log scale, an integer budget, a categorical choice
    space = {"k": ("log", 0.3, 0.5, 50.0), "n": ("int", 1, 20), "mode": ("choice", "a", "b")}
    genome = {"k": 1.0, "n": 3, "mode": "a", "fixed": "kept"}

    def fitness(g: dict, seed: int) -> float:
        return (
            -(np.log(g["k"] / 8.0) ** 2)
            - abs(g["n"] - 12) / 10
            + (0.5 if g["mode"] == "b" else 0.0)
        )

    lineage = evolve(
        fitness, genome, mutate=genes(space), generations=12, population=10, keep=3, seed=3
    )
    best = lineage.best
    assert isinstance(best, dict) and best["fixed"] == "kept"
    assert 8.0 / 1.6 <= best["k"] <= 8.0 * 1.6 and best["n"] > 3 and best["mode"] == "b"
    assert lineage.best_fitness > fitness(genome, 0)  # the hand-set start is the control
    assert lineage.generations[-1]["best_fitness"] >= lineage.generations[0]["best_fitness"]
    assert isinstance(lineage.generations[0]["best"], dict)


def test_evolve_grows_a_foreign_genome_when_told_how() -> None:
    from cadence.genome import genes

    def grow(g: dict, seed: int) -> float:
        return float(g["x"]) * 2.0

    def fitness(grown: float, seed: int) -> float:
        return -abs(grown - 6.0)

    lineage = evolve(
        fitness,
        {"x": 1.0},
        mutate=genes({"x": ("linear", 0.5, 0.0, 10.0)}),
        grow=grow,
        generations=10,
        population=8,
        keep=2,
        seed=5,
    )
    assert abs(lineage.best["x"] - 3.0) < 0.5


def test_evolve_refuses_a_foreign_genome_without_mutate_and_mixed_mutation_arguments() -> None:
    import pytest

    from cadence.genome import genes

    with pytest.raises(ValueError):
        evolve(lambda g, s: 0.0, {"x": 1.0}, generations=1, population=1, keep=1)
    with pytest.raises(ValueError):
        evolve(
            _hidden_size,
            two_region(),
            mutate=genes({"x": ("int", 0, 1)}),
            fixed=("input",),
            generations=1,
            population=1,
            keep=1,
        )
    with pytest.raises(ValueError):
        genes({"x": ("log", -1.0, 0.0, 1.0)})
