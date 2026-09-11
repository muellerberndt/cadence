"""A constitution grows into a wiring; mutation and selection change it toward a fitness."""

from __future__ import annotations

import numpy as np

import cadence as cd
from cadence.constitution import Constitution, Projection, Region, evolve, grow, mutate


def two_region() -> Constitution:
    return Constitution(
        regions=(Region("input", 6), Region("hidden", 8), Region("output", 3)),
        projections=(Projection("input", "hidden", density=0.5), Projection("hidden", "output")),
        label="toy",
    )


def test_grow_is_deterministic_and_names_contiguous_sets() -> None:
    c = two_region()
    a, b = grow(c, seed=3), grow(c, seed=3)
    assert a.digest() == b.digest() and a.n == 17
    assert a.sets["hidden"] == tuple(range(6, 14)) and a.sets["output"] == tuple(range(14, 17))
    assert (
        a.in_degree()[list(a.sets["input"])].sum() > 0
    )  # symmetric: inputs hear the hidden owners
    engine = cd.Settlement(a, cd.learning_rule())
    assert engine.layout.ranges == 3
    assert grow(c, seed=4).digest() != a.digest()


def test_mutate_keeps_the_shape_and_respects_fixed_regions() -> None:
    rng = np.random.default_rng(0)
    child = mutate(two_region(), rng, fixed=("input", "output"))
    assert [r.name for r in child.regions] == ["input", "hidden", "output"]
    assert child.region("input").size == 6 and child.region("output").size == 3
    assert len(child.projections) == 2 and 0.01 <= child.projections[0].density <= 1.0


def test_evolve_moves_toward_the_fitness() -> None:
    # fitness: a hidden region of about twelve owners with dense input projections
    def fitness(w: cd.Wiring, seed: int) -> float:
        hidden = len(w.sets["hidden"])
        density = w.in_degree()[list(w.sets["hidden"])].mean() / 6
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
