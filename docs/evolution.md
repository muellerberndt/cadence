# Evolve a brain

This guide covers `Genome`, `develop`, `mutate`, `genes` and `evolve`: selection
over a settling brain's genome, or over any declared genome, across lives. The
application supplies the mutation choices and the fitness. Growth or
specialization of a temporal patch under selection is a research goal that needs
its own evidence.

A genome decides which regions a brain has, how large they are and how they project.
Evolution changes the genome between lives: `mutate` draws an offspring, and `evolve`
keeps the genomes whose developed brains score best under a fitness that runs a short
life. Inside a life, learning changes synapses and records, and bounded drive
perturbations sample candidate states for one decision. That state search is
distinct from the centered equilibrium contrast used by
[temporal learning](temporal.md) and [input planning](planning.md).

## Any genome

`evolve` selects over any genome that has its own mutation. A `Genome` grows into a
connectome by `develop` and mutates by `mutate` below; any other genome, a dict of a
governor's thresholds, a port topology, a patch's sizes, a habit's motor coding, is passed
to the fitness as it is (or grown by `grow(genome, seed)` when supplied) and mutates by the
`mutate(genome, rng)` you pass. `cadence.genes(space)` supplies one over a declared space:

```python
import numpy as np
from cadence import evolve, genes

space = {"threshold": ("log", 0.3, 0.5, 50.0), "budget": ("int", 1, 20), "law": ("choice", "a", "b")}
start = {"threshold": 4.0, "budget": 3, "law": "a"}


def fitness(genome, seed):  # the caller's: here a threshold near 8, a budget near 12, law b
    return -np.log(genome["threshold"] / 8.0) ** 2 - abs(genome["budget"] - 12) / 10 + (genome["law"] == "b")


lineage = evolve(fitness, start, mutate=genes(space), generations=12, population=10, keep=3, seed=5)
assert lineage.best["law"] == "b" and lineage.best["budget"] > 3  # moved by one each step
assert lineage.best_fitness > fitness(start, 0)  # the hand-set start is the control
```

`log` genes multiply by `exp(normal(0, step))` and clip, for scales and thresholds; `linear`
genes add `normal(0, step)` and clip; `int` genes move by one within bounds; `choice` genes
redraw among their options. Each gene mutates independently with probability `rate`. Where a
rule looks designed, this is how it becomes a gene: the hand-set value starts the lineage and
stays as the control.

## Mutation

`cadence.genome.mutate(genome, rng, *, size_step=0.25, fixed=(), tied=())` returns one
offspring with the same regions and projections:

- each blank region outside `fixed` takes the size `round(size * exp(normal(0, size_step)))`,
  at least one; designed regions and the regions in `fixed` keep their size;
- each `(leader, follower)` pair in `tied` gives the follower the leader's size, for a
  `prefrontal_cortex` that holds one neuron per neuron of its region;
- each projection's `density` is multiplied by `exp(normal(0, 0.2))` and clipped to
  `[0.01, 1]`, its `sign` moves by `normal(0, 0.2)` within `[-1, 1]`, and its `scale` is
  multiplied by `exp(normal(0, 0.2))` and clipped to `[0.05, 20]`;
- `count` and `reciprocal` stay as declared.

Fix the regions whose width the environment sets. In `Brain.genome` the motor
cortex is designed and keeps its size; the `sensory` region is blank and belongs in
`fixed`.

```python
import numpy as np
import cadence as cd
from cadence.genome import mutate

genome = cd.Brain.genome(2, 2, hidden=12)
child = mutate(genome, np.random.default_rng(3), fixed=("sensory",))
for parent, offspring in zip(genome.regions, child.regions):
    print(parent.name, parent.size, "->", offspring.size)     # association 12 -> 20
for projection in child.projections:
    print(projection.pre, "to", projection.post, round(projection.density, 2),
          round(projection.sign, 2), round(projection.scale, 2))
```

## Selection

`evolve(fitness, genome, *, generations=10, population=8, keep=2, seed=0, mapper=map,
report=None, **mutation)` selects over genomes. Generation 0 scores the starting genome and
`population - 1` offspring of it. Each later generation scores `population` offspring,
each of a parent drawn at random from the `keep` best genomes of the generation before.
Every life develops its genome at the seed `seed + 1000 * generation + index` and calls
`fitness(connectome, seed)` with that connectome and seed; the score must be finite, and
higher is better. The remaining keyword arguments go to `mutate`.

A fitness runs a short life and returns its score. This one wraps the developed connectome
in a `Brain`, which learns over forty moments which of two actions each cue pays
for; the score is the reward of the last twenty moments less a small cost per synapse.

```python
def life(connectome, seed):
    brain = cd.Brain(connectome, seed=seed)
    rng = np.random.default_rng(seed)
    cue = int(rng.integers(2))
    action = brain.step([np.eye(2)[cue]])
    earned = []
    for moment in range(40):
        reward = float(action[0] == cue)
        earned.append(reward)
        cue = int(rng.integers(2))
        action = brain.step([np.eye(2)[cue]], reward=[reward])
    return float(np.mean(earned[-20:])) - 1e-4 * connectome.synapses


lineage = cd.evolve(life, genome, generations=3, population=4, keep=2, seed=0, fixed=("sensory",))
for record in lineage.generations:
    print(record["generation"], round(record["best_fitness"], 3), round(record["mean_fitness"], 3))
print([region.size for region in lineage.best.regions], round(lineage.best_fitness, 3))
```

`Lineage` (`cadence.genome.Lineage`) records what selection did. `generations` holds one
record per generation with `generation`, `best_fitness`, `mean_fitness` and `best`, that
generation's best genome as `Genome.to_dict()`. `best` is the best genome over all
generations, the earliest on a tie, and `best_fitness` its score. `report=callback`
receives the lineage after every generation, so a long run writes its record while it runs.
`Genome.from_dict` reads a record back; a designed region is supplied by name and must match
the circuit digest in the record.

```python
import json
from pathlib import Path

log = Path("lineage.json")
cd.evolve(
    life, genome, generations=2, population=4, keep=2, seed=0, fixed=("sensory",),
    report=lambda so_far: log.write_text(json.dumps(so_far.generations)),
)
record = json.loads(log.read_text())[-1]["best"]
best = cd.Genome.from_dict(record, designed={"motor": genome.region("motor")})
connectome = cd.develop(best, seed=0)
```

### The selection protocol

Select every genome the same way. The hand-set genome is the lineage's first
member, so selection starts from the control and must beat it on held-out
worlds, never on the worlds it was selected on. A random search over the same
space at the same number of evaluations is the second control: what the lineage
found must beat what luck finds. The fitness reads only what the genome cannot
reweight, and prices both compute and surprise, since a price on compute alone
switches learning off — a lineage under that price stops imagining and stops
learning. Keep every outcome: lineages do switch learning off at their prices,
drop probes and residuals from their readback, and evolve a threshold rule that
wins the ordinary error and loses the exception. A genome that prunes what the
designer put in is a result, not a failure of the run.

```python
space = {"span": ("log", 0.3, 0.5, 8.0), "retain": ("linear", 0.1, 0.0, 1.0), "reads_probe": ("choice", 0, 1)}
hand_set = {"span": 2.0, "retain": 1.0, "reads_probe": 1}
mutate = cd.genes(space, rate=0.35)          # the lineage; the random search draws from the same space
```

## Parallel lives

`mapper` runs the lives of one generation. It receives a callable and an iterable of
`(genome, seed)` jobs, as `map` does, and returns their scores in order. A pool's `map`
runs the lives side by side and gives the same lineage whenever the fitness depends only
on its connectome and seed. A thread pool runs them in one process. A process pool
(`multiprocessing.Pool.map`, `concurrent.futures.ProcessPoolExecutor.map`) runs them on
separate cores; its workers import the fitness, so the fitness is a module-level function
of a file, and the script creates the pool under `if __name__ == "__main__":`.

```python
from multiprocessing.pool import ThreadPool

with ThreadPool(4) as pool:
    parallel = cd.evolve(
        life, genome, generations=3, population=4, keep=2, seed=0, fixed=("sensory",),
        mapper=pool.map,
    )
assert parallel.best == lineage.best and parallel.best_fitness == lineage.best_fitness
```

## Detuning inside a life

A settled brain gives one answer under one drive. Equilibrium detuning samples
alternatives inside one decision:

1. Repeat the drive in a batch, one row per candidate.
2. Add a bounded random drive to the latent neurons of each row; a uniform draw in
   `[-a, a]` stays within `a`.
3. Settle every row with `equilibrate` and keep the rows whose residual passed.
4. Read each kept row's candidate action and rank the candidates by a read of the records
   or by a critic.

The parameters, the records and the genome stay as they are; only the drive of this
decision changes.

```python
brain = cd.Brain(cd.develop(lineage.best, seed=0), seed=0)
records = cd.Records(2 + 2, {"reward": 1}, valued=["reward"], cells=2000, active=20, seed=0)
for cue in range(2):                   # witnessed in this life: the cue's own action pays
    for action in range(2):
        code = records.code(np.r_[np.eye(2)[cue], np.eye(2)[action]], adapt=True)[:, 0]
        records.write(code, {"reward": np.array([float(action == cue)])})

observation = np.eye(2)[1]
drive = np.repeat(brain.stimulus([observation]), 8, axis=0)                        # 1.
latent = brain.association_index
drive[:, latent] += np.random.default_rng(0).uniform(-0.5, 0.5, (8, len(latent)))  # 2.
result = brain.brain.equilibrate(drive, budget=512, chunk=32, tolerance=1e-6)      # 3.
candidates = result.state.activation[:, brain.motor_index].argmax(axis=1)          # 4.
imagined = np.stack([np.r_[observation, np.eye(2)[a]] for a in candidates])
score = records.read(records.code(imagined))["reward"][:, 0]
score[~result.converged] = -np.inf     # a row that did not converge offers no candidate
choice = int(candidates[np.argmax(score)])
value = brain.basal_ganglia.value(result.state)   # the critic's ranking of the same rows
print(candidates, score.round(2), choice)
```

## What selection and detuning change

| Mechanism | Changes | Leaves as it is |
|---|---|---|
| Selection (`evolve`) | The genome of the next generation: region sizes and projection densities, signs and scales, and so the developed connectome and the efficacies a life starts from | Every synapse and record inside a life |
| Learning inside a life | Synapses (`Learner`, `ActorCritic`) and records (`Records.write`) | The genome |
| Detuning | The drive of one decision, and so the equilibrium the brain reads | Parameters, records and genome |

The two combine inside the fitness. A life that chooses its actions by detuning scores the
genome together with that sampling, so selection favours genomes whose detuned equilibria
contain good actions and pass their residual checks within the budget. Give every genome
the same detuning bound, candidate count and settling budget, and count the settling steps
of every candidate in the cost of the life.
