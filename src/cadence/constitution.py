"""Where a wiring comes from: a constitution, grown into a net, selected over generations.

The wirings in this library are designed (``layered``, ``embedded``, ``stateful``) or read
from a connectome, which is a wiring evolution designed. An animal's specialised parts are
a mix: the genome lays down which regions exist, how large they are, which project to
which and with what sign and density; learning and development do the rest. This module is
the simplest version of that first step. A ``Constitution`` is a handful of numbers per
region and per projection; ``grow`` turns it into a ``Wiring`` with named sets, the same
deterministic way every time for a given seed (development); ``mutate`` perturbs it; and
``evolve`` keeps, over generations, the constitutions whose grown nets score best under a
fitness the caller supplies (a protocol score, a learning curve, a held-out accuracy).
Nothing here is a sixth primitive: the result is a wiring, and everything after it is
settlement under the same rule.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .wiring import Wiring

__all__ = ["Region", "Projection", "Constitution", "grow", "mutate", "evolve"]


@dataclass(frozen=True)
class Region:
    name: str
    size: int


@dataclass(frozen=True)
class Projection:
    """Overlaps from every owner of ``pre`` to a fraction ``density`` of the owners of ``post``."""

    pre: str
    post: str
    density: float = 1.0
    sign: float = 0.0  # mean sign of the overlaps: -1 all inhibitory, +1 all excitatory, 0 mixed
    scale: float = 1.0  # magnitude, fan-scaled
    count: float = 1.0
    symmetric: bool = True  # also the reverse overlaps, with the same weights (one seam)


@dataclass(frozen=True)
class Constitution:
    regions: tuple[Region, ...]
    projections: tuple[Projection, ...]
    label: str = "constitution"

    def region(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": [{"name": r.name, "size": r.size} for r in self.regions],
            "projections": [vars(p) for p in self.projections],
            "label": self.label,
        }


def grow(constitution: Constitution, seed: int = 0) -> Wiring:
    """Development: the constitution as a wiring, region by region, deterministic in the seed."""
    rng = np.random.default_rng(seed)
    starts: dict[str, int] = {}
    n = 0
    for r in constitution.regions:
        starts[r.name] = n
        n += r.size
    pre, post, count, sign = [], [], [], []
    for p in constitution.projections:
        a, b = constitution.region(p.pre), constitution.region(p.post)
        mask = rng.random((a.size, b.size)) < p.density
        i, j = np.nonzero(mask)
        magnitude = rng.uniform(0.0, 1.0, size=len(i)) * np.sqrt(6.0 / (a.size + b.size)) * p.scale
        signs = np.where(rng.random(len(i)) < (1.0 + p.sign) / 2.0, 1.0, -1.0) * magnitude
        pre.append(starts[p.pre] + i)
        post.append(starts[p.post] + j)
        count.append(np.full(len(i), p.count))
        sign.append(signs)
        if p.symmetric:
            pre.append(starts[p.post] + j)
            post.append(starts[p.pre] + i)
            count.append(np.full(len(i), p.count))
            sign.append(signs)
    sets = {r.name: range(starts[r.name], starts[r.name] + r.size) for r in constitution.regions}
    if not pre:
        return Wiring.from_edges(n, pre=[], post=[], sets=sets, label=constitution.label)
    return Wiring.from_edges(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        count=np.concatenate(count),
        sign=np.concatenate(sign),
        sets=sets,
        label=constitution.label,
    )


def mutate(
    constitution: Constitution,
    rng: np.random.Generator,
    *,
    size_step: float = 0.25,
    fixed: tuple[str, ...] = (),
    tied: tuple[tuple[str, str], ...] = (),
) -> Constitution:
    """One offspring: every region not in ``fixed`` may change size by about ``size_step`` of
    itself, every projection may change density, sign and scale a little; nothing is added or
    removed. A pair in ``tied`` keeps the second region the size of the first (a context
    range with one owner per hidden owner)."""
    regions = tuple(
        r
        if r.name in fixed
        else replace(r, size=max(1, int(round(r.size * float(np.exp(rng.normal(0.0, size_step)))))))
        for r in constitution.regions
    )
    for leader, follower in tied:
        size = next(r.size for r in regions if r.name == leader)
        regions = tuple(replace(r, size=size) if r.name == follower else r for r in regions)
    projections = tuple(
        replace(
            p,
            density=float(np.clip(p.density * np.exp(rng.normal(0.0, 0.2)), 0.01, 1.0)),
            sign=float(np.clip(p.sign + rng.normal(0.0, 0.2), -1.0, 1.0)),
            scale=float(np.clip(p.scale * np.exp(rng.normal(0.0, 0.2)), 0.05, 20.0)),
        )
        for p in constitution.projections
    )
    return replace(constitution, regions=regions, projections=projections)


class _Life:
    """One life as a picklable callable: grow the constitution at its seed, score the wiring."""

    def __init__(self, fitness: Callable[[Wiring, int], float]) -> None:
        self.fitness = fitness

    def __call__(self, job: tuple[Constitution, int]) -> float:
        constitution, seed = job
        return float(self.fitness(grow(constitution, seed=seed), seed))


@dataclass
class Lineage:
    """What selection did: the best constitution of every generation and its fitness."""

    generations: list[dict[str, Any]] = field(default_factory=list)
    best: Constitution | None = None
    best_fitness: float = -np.inf


def evolve(
    fitness: Callable[[Wiring, int], float],
    constitution: Constitution,
    *,
    generations: int = 10,
    population: int = 8,
    keep: int = 2,
    seed: int = 0,
    mapper: Callable[..., Iterable[float]] = map,
    **mutation: Any,
) -> Lineage:
    """Selection over constitutions: each generation grows ``population`` offspring of the
    ``keep`` best so far, scores each grown wiring with ``fitness(wiring, seed)``, and keeps
    the best. The fitness is the caller's: a protocol score, a learning curve, an accuracy.
    ``mapper`` runs a generation's lives: ``map`` one after another, a pool's ``map`` side
    by side (``fitness`` must then be picklable, so a module-level function)."""
    rng = np.random.default_rng(seed)
    lineage = Lineage()
    parents = [constitution]
    for g in range(generations):
        offspring = list(parents) if g == 0 else []
        while len(offspring) < population:
            offspring.append(mutate(parents[rng.integers(len(parents))], rng, **mutation))
        seeds = [seed + 1000 * g + k for k in range(len(offspring))]
        scores = mapper(_Life(fitness), zip(offspring, seeds, strict=True))
        scored = [(float(f), k, child) for k, (f, child) in enumerate(zip(scores, offspring, strict=True))]
        scored.sort(key=lambda s: -s[0])
        parents = [c for _, _, c in scored[:keep]]
        top = scored[0]
        lineage.generations.append(
            {
                "generation": g,
                "best_fitness": top[0],
                "best": top[2].to_dict(),
                "mean_fitness": float(np.mean([s[0] for s in scored])),
            }
        )
        if top[0] > lineage.best_fitness:
            lineage.best, lineage.best_fitness = top[2], top[0]
    return lineage
