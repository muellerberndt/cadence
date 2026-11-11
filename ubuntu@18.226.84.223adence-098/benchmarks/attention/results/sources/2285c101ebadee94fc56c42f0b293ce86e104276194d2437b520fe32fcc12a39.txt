"""Where a connectome comes from: a genome, developed into a brain, selected over generations.

The connectomes in this library are designed (``layered``, ``embedded``) or read
from a measured animal connectome. An animal's specialised parts are
a mix: the genome lays down which regions exist, how large they are, which project to
which and with what sign and density; learning and development do the rest. This module is
the simplest version of that first step. A ``Genome`` is a handful of numbers per
region and per projection; ``develop`` turns it into a ``Connectome`` with named populations,
the same deterministic way every time for a given seed (development); ``mutate`` perturbs it; and
``evolve`` keeps, over generations, the genomes whose developed brains score best under a
fitness the caller supplies. The result is a connectome, and everything after it is settling
under the same neuron model.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .connectome import Connectome
from .regions import Region

__all__ = ["Region", "Projection", "Genome", "develop", "mutate", "evolve"]


@dataclass(frozen=True)
class Projection:
    """Synapses from ``pre`` to a fraction ``density`` of ``post``.

    An end names a region, meaning its ``outputs`` (for ``pre``) or ``inputs`` (for ``post``)
    population and every neuron of a blank region, or one population as ``region/population``.
    """

    pre: str
    post: str
    density: float = 1.0
    sign: float = 0.0  # mean sign of the synapses: -1 all inhibitory, +1 all excitatory, 0 mixed
    scale: float = 1.0  # magnitude, fan-scaled
    count: float = 1.0
    reciprocal: bool = True  # also the reverse synapses, with the same weights: reciprocal pairs

    def __post_init__(self) -> None:
        if not self.pre or not self.post:
            raise ValueError("projection endpoints must name regions or region/population ports")
        if not 0 <= self.density <= 1 or not -1 <= self.sign <= 1:
            raise ValueError("density must lie in [0, 1] and sign in [-1, 1]")
        if not np.isfinite([self.scale, self.count]).all() or min(self.scale, self.count) < 0:
            raise ValueError("projection scale and count must be finite and nonnegative")


@dataclass(frozen=True)
class Genome:
    regions: tuple[Region, ...]
    projections: tuple[Projection, ...]
    label: str = "genome"

    def __post_init__(self) -> None:
        names = [r.name for r in self.regions]
        if not names or len(set(names)) != len(names):
            raise ValueError("a genome needs uniquely named regions")
        for projection in self.projections:
            for endpoint, side in ((projection.pre, "outputs"), (projection.post, "inputs")):
                name, _, population = endpoint.partition("/")
                if name not in names:
                    raise ValueError(f"unknown projection region: {name!r}")
                region = self.region(name)
                region.neurons(population or getattr(region, side))

    @classmethod
    def composed(cls, inputs: Any, actions: int, **settings: Any) -> Genome:
        """The layout ``Brain.compose(inputs, actions, **settings)`` develops: the sensory
        region (an image's visual cortex), the code region with ``learned_code``, the modules
        ending in the association region, the prefrontal region of the working trace, the
        motor region, the observers, and the efference, history, recall and next regions its
        memory settings add, with their projections. Settings that do not shape the layout
        change nothing here. Mutate it with the sensory region fixed and the prefrontal region
        tied to the association region, then wrap it with
        ``Brain.compose(inputs, actions, connectome=develop(genome, seed=seed), seed=seed)``."""
        from .generic import _layout

        keywords = (
            "modules", "observers", "lateral", "scales", "features", "field", "code",
            "learned_code", "seed", "slots",
        )
        layout = {name: settings.pop(name) for name in keywords if name in settings}
        return _layout(inputs, actions, settings, **layout)[0]

    @staticmethod
    def observation_width(
        inputs: Any, code: int | None = None, learned_code: int | None = None
    ) -> int:
        """The width of the observation code of a brain with these ``inputs`` (a vector
        length or an image shape), this ``code`` and this ``learned_code`` (as
        ``Brain.compose`` takes them), before the brain is built: with ``recall=True`` a
        custom anatomy's ``recall`` and ``next`` regions hold that many neurons. A learned
        code's key is its units and the inputs, the code joined to the observation's
        identity, for an observation no wider than the code; a wider one is keyed on the
        code alone, its units."""
        from .generic import _observation_width

        return _observation_width(inputs, code, learned_code)

    def region(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": [r.to_dict() for r in self.regions],
            "projections": [vars(p) for p in self.projections],
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], designed: Mapping[str, Region] | None = None) -> Genome:
        """The inverse of ``to_dict``: a genome read back from a lineage's record.

        A record keeps a designed region's circuit label and digest only; ``designed`` supplies
        the region itself by name, and its digest must match the record.
        """
        regions = []
        for r in d["regions"]:
            if "circuit" not in r:
                regions.append(Region(str(r["name"]), int(r["size"])))
                continue
            region = (designed or {}).get(str(r["name"]))
            if region is None or region.circuit is None:
                raise ValueError(f"designed region {r['name']!r} must be supplied by name")
            if (
                region.circuit.digest() != r["digest"]
                or region.name != r["name"]
                or region.size != r["size"]
                or region.inputs != r.get("inputs")
                or region.outputs != r.get("outputs")
            ):
                raise ValueError(f"designed region {r['name']!r} differs from the record")
            regions.append(region)
        return cls(
            regions=tuple(regions),
            projections=tuple(Projection(**p) for p in d["projections"]),
            label=str(d.get("label", "genome")),
        )


def develop(genome: Genome, seed: int = 0) -> Connectome:
    """Development: the genome as a connectome, region by region, deterministic in the seed.

    Regions are laid out in order. A designed region contributes its circuit's synapses and
    its populations as ``region/population``; every projection is then drawn between its two
    ends. The result is one connectome for one ``Brain``.
    """
    rng = np.random.default_rng(seed)
    starts: dict[str, int] = {}
    n = 0
    for r in genome.regions:
        starts[r.name] = n
        n += r.size
    pre: list[np.ndarray] = []
    post: list[np.ndarray] = []
    count: list[np.ndarray] = []
    sign: list[np.ndarray] = []
    populations: dict[str, Iterable[int]] = {}
    for r in genome.regions:
        s = starts[r.name]
        populations[r.name] = range(s, s + r.size)
        if r.circuit is None:
            continue
        populations.update(
            {f"{r.name}/{k}": [s + i for i in v] for k, v in r.circuit.populations.items()}
        )
        if r.circuit.synapses:
            pre.append(s + r.circuit.pre)
            post.append(s + r.circuit.post)
            count.append(r.circuit.count)
            sign.append(r.circuit.sign)

    def end(name: str, side: str) -> np.ndarray:
        region_name, _, population = name.partition("/")
        region = genome.region(region_name)
        local = region.neurons(population or getattr(region, side))
        return starts[region_name] + np.asarray(local, dtype=np.int64)

    for p in genome.projections:
        a, b = end(p.pre, "outputs"), end(p.post, "inputs")
        mask = rng.random((len(a), len(b))) < p.density
        i, j = np.nonzero(mask)
        magnitude = rng.uniform(0.0, 1.0, size=len(i)) * np.sqrt(6.0 / (len(a) + len(b))) * p.scale
        signs = np.where(rng.random(len(i)) < (1.0 + p.sign) / 2.0, 1.0, -1.0) * magnitude
        pre.append(a[i])
        post.append(b[j])
        count.append(np.full(len(i), p.count))
        sign.append(signs)
        if p.reciprocal:
            pre.append(b[j])
            post.append(a[i])
            count.append(np.full(len(i), p.count))
            sign.append(signs)
    if not pre:
        return Connectome.from_synapses(
            n, pre=[], post=[], populations=populations, label=genome.label
        )
    return Connectome.from_synapses(
        n,
        pre=np.concatenate(pre),
        post=np.concatenate(post),
        count=np.concatenate(count),
        sign=np.concatenate(sign),
        populations=populations,
        label=genome.label,
    )


def mutate(
    genome: Genome,
    rng: np.random.Generator,
    *,
    size_step: float = 0.25,
    fixed: tuple[str, ...] = (),
    tied: tuple[tuple[str, str], ...] = (),
) -> Genome:
    """One offspring: every blank region not in ``fixed`` may change size by about ``size_step`` of
    itself, every projection may change density, sign and scale a little; nothing is added or
    removed. A pair in ``tied`` keeps the second region the size of the first (a context
    range with one neuron per hidden neuron)."""
    if not np.isfinite(size_step) or size_step < 0:
        raise ValueError("size_step must be finite and nonnegative")
    for name in (*fixed, *(name for pair in tied for name in pair)):
        genome.region(name)
    regions = tuple(
        r
        if r.name in fixed or r.circuit is not None
        else replace(r, size=max(1, int(round(r.size * float(np.exp(rng.normal(0.0, size_step)))))))
        for r in genome.regions
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
        for p in genome.projections
    )
    return replace(genome, regions=regions, projections=projections)


_mutate_genome = mutate


class _Life:
    """One life as a picklable callable: grow the genome at its seed, score what grew."""

    def __init__(
        self, fitness: Callable[[Any, int], float], grow: Callable[[Any, int], Any]
    ) -> None:
        self.fitness = fitness
        self.grow = grow

    def __call__(self, job: tuple[Any, int]) -> float:
        genome, seed = job
        return float(self.fitness(self.grow(genome, seed), seed))


def _develop_at(genome: Genome, seed: int) -> Connectome:
    return develop(genome, seed=seed)


def _as_is(genome: Any, seed: int) -> Any:
    return genome


def _record(genome: Any) -> Any:
    return genome.to_dict() if hasattr(genome, "to_dict") else genome


def genes(
    space: Mapping[str, tuple[Any, ...]], *, rate: float = 1.0
) -> Callable[[Mapping[str, Any], np.random.Generator], dict[str, Any]]:
    """A mutation for dict genomes over a declared space, for ``evolve(..., mutate=genes(space))``.

    Every key of ``space`` names a gene and its kind: ``("log", step, low, high)`` multiplies by
    ``exp(normal(0, step))`` and clips to the bounds, for scales and thresholds;
    ``("linear", step, low, high)`` adds ``normal(0, step)`` and clips; ``("int", low, high)``
    moves by one, up or down, within the bounds; ``("choice", option, ...)`` redraws among the
    options. Each gene mutates independently with probability ``rate``; genes of the genome that
    ``space`` does not name are copied. An arousal threshold, an imagination budget, a port's
    width or a cortex size can all be genes this way, with the hand-set value as the control.
    """
    kinds = {}
    for name, spec in space.items():
        if not spec or spec[0] not in ("log", "linear", "int", "choice"):
            raise ValueError(f"gene {name!r} needs a kind: log, linear, int or choice")
        if spec[0] in ("log", "linear") and (len(spec) != 4 or spec[1] <= 0 or spec[2] > spec[3]):
            raise ValueError(
                f"gene {name!r} needs (kind, step, low, high) with step > 0 and low <= high"
            )
        if spec[0] == "int" and (len(spec) != 3 or spec[1] > spec[2]):
            raise ValueError(f"gene {name!r} needs (int, low, high) with low <= high")
        if spec[0] == "choice" and len(spec) < 2:
            raise ValueError(f"gene {name!r} needs at least one option")
        kinds[str(name)] = tuple(spec)
    if not 0.0 < float(rate) <= 1.0:
        raise ValueError("rate must lie in (0, 1]")

    def mutate_genes(genome: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        child = dict(genome)
        for name, spec in kinds.items():
            if name not in child or rng.random() > rate:
                continue
            kind, value = spec[0], child[name]
            if kind == "log":
                child[name] = float(
                    np.clip(value * np.exp(rng.normal(0.0, spec[1])), spec[2], spec[3])
                )
            elif kind == "linear":
                child[name] = float(np.clip(value + rng.normal(0.0, spec[1]), spec[2], spec[3]))
            elif kind == "int":
                child[name] = int(np.clip(int(value) + int(rng.choice([-1, 1])), spec[1], spec[2]))
            else:
                child[name] = spec[1 + int(rng.integers(len(spec) - 1))]
        return child

    return mutate_genes


@dataclass
class Lineage:
    """What selection did: the best genome of every generation and its fitness."""

    generations: list[dict[str, Any]] = field(default_factory=list)
    best: Any = None
    best_fitness: float = -np.inf


def evolve(
    fitness: Callable[[Any, int], float],
    genome: Any,
    *,
    generations: int = 10,
    population: int = 8,
    keep: int = 2,
    seed: int = 0,
    mapper: Callable[..., Iterable[float]] = map,
    report: Callable[[Lineage], None] | None = None,
    mutate: Callable[[Any, np.random.Generator], Any] | None = None,
    grow: Callable[[Any, int], Any] | None = None,
    **mutation: Any,
) -> Lineage:
    """Selection over genomes: the first generation holds the starting genome and its mutated
    offspring; every later generation holds ``population`` mutated offspring of the ``keep``
    best genomes of the previous generation. Each genome is grown at its seed and scored with
    ``fitness(grown, seed)``, and the lineage records the best genome ever scored. The fitness
    is the caller's: a protocol score, a learning curve, an accuracy, a return per unit of
    compute.

    A ``Genome`` grows into a connectome by ``develop`` and mutates by ``mutate`` of this module,
    whose keyword arguments arrive through ``mutation``. Any other genome, a dict of
    thresholds, a port topology, a patch's sizes, needs its own ``mutate(genome, rng)`` (``genes``
    supplies one over a declared space) and is passed to the fitness as it is unless ``grow``
    says how to build from it. ``mapper`` runs a generation's lives: ``map`` one after another, a
    pool's ``map`` side by side (``fitness`` must then be picklable, so a module-level function).
    ``report`` is called with the lineage so far after every generation, so a long run can be
    written out as it goes."""
    for name, value in (("generations", generations), ("population", population), ("keep", keep)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if keep > population:
        raise ValueError("keep cannot exceed population")
    if mutate is None:
        if not isinstance(genome, Genome):
            raise ValueError("a genome that is not a Genome needs its own mutate")
        step: Callable[[Any, np.random.Generator], Any] = functools.partial(
            _mutate_genome, **mutation
        )
    elif mutation:
        raise ValueError(
            "mutation keywords belong to the module's mutate; a custom mutate takes none"
        )
    else:
        step = mutate
    build = grow if grow is not None else (_develop_at if isinstance(genome, Genome) else _as_is)
    rng = np.random.default_rng(seed)
    lineage = Lineage()
    parents = [genome]
    for g in range(generations):
        offspring = list(parents) if g == 0 else []
        while len(offspring) < population:
            offspring.append(step(parents[rng.integers(len(parents))], rng))
        seeds = [seed + 1000 * g + k for k in range(len(offspring))]
        scores = mapper(_Life(fitness, build), zip(offspring, seeds, strict=True))
        scored = [
            (float(f), k, child) for k, (f, child) in enumerate(zip(scores, offspring, strict=True))
        ]
        if not all(np.isfinite(f) for f, _, _ in scored):
            raise ValueError("fitness must return finite scores")
        scored.sort(key=lambda s: -s[0])
        parents = [c for _, _, c in scored[:keep]]
        top = scored[0]
        lineage.generations.append(
            {
                "generation": g,
                "best_fitness": top[0],
                "best": _record(top[2]),
                "mean_fitness": float(np.mean([s[0] for s in scored])),
            }
        )
        if top[0] > lineage.best_fitness:
            lineage.best, lineage.best_fitness = top[2], top[0]
        if report is not None:
            report(lineage)
    return lineage
