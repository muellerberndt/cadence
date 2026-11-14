"""A diagnostic arm for cross-modal binding (cadence#334, for review-topdown): no library hook.

    python binding_arm.py <brain.npz> <seed> '<genome>' <arm> <lessons> <out.json>

The library's lesson can only nudge the motor outputs. This arm computes the contrast itself,
between two settles of the same frozen brain whose inputs differ, and moves the synapses into the
association region by it. It is contrastive Hebbian learning between a free phase and a second
phase, the family of the brain's own rule, written in the chamber. The two phases:

* ``bind``: free = a kind's word heard with nothing in view; second = the word heard with the
  thing in view. The mirror lesson in the same call: free = the glimpse alone; second = the
  glimpse with the word.
* ``control``: the same number of ordinary naming lessons (the brain's own lesson: "what is
  this?" with the thing in focus, its name the label at the gap).

Every 50 lessons it reads the completion matrix (a word alone read through the focus probes),
the state completeness of the association, names and imitation, and by_colour ("what is red?").
The cost is two settles a lesson plus one outer product over the association's inbound synapses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exam as ex  # noqa: E402
import introspect as ii  # noqa: E402
import night as nt  # noqa: E402


def settled(p: nt.Parrot, rooms: list[nt.Room], heard: list[list[str | None]],
            rng: np.random.Generator) -> np.ndarray:
    """Every neuron's settled activity at the last moment of ``heard`` (greedy, a frozen copy)."""
    brain = ii.frozen(p)
    n = len(rooms)
    exs = [nt.Exchange(r, h, len(h) - 1, None) for r, h in zip(rooms, heard, strict=True)]
    for t in range(len(heard[0])):
        x = p.observe(exs, t, [None] * n, np.zeros(n, dtype=bool), rng)
        brain.act(x, greedy=True)
    return np.asarray(brain.basal_ganglia.state.activation, dtype=np.float64).copy()


def contrast_step(p: nt.Parrot, free: np.ndarray, second: np.ndarray, rate: float) -> float:
    """Move the synapses into the association by the contrast of the two phases' co-activity
    (second minus free), scaled so the largest step is ``rate``; returns the step's RMS."""
    c = p.brain.connectome
    pre, post = np.asarray(c.pre), np.asarray(c.post)
    assoc = np.zeros(c.n, dtype=bool)
    assoc[np.asarray(list(c.populations["association"]))] = True
    into = assoc[post]
    d = (second[:, post[into]] * second[:, pre[into]]
         - free[:, post[into]] * free[:, pre[into]]).mean(0)
    top = float(np.abs(d).max()) or 1.0
    step = np.zeros(len(pre))
    step[into] = rate * d / top
    g = p.brain.learner.brain
    p.brain.learner.brain = g.with_parameters(efficacy=np.asarray(g.efficacy) + step,
                                              bias=np.asarray(g.bias))
    return float(np.sqrt(np.mean(step[into] ** 2)))


def main(argv: list[str]) -> None:
    path, seed, genome, arm = argv[1], int(argv[2]), json.loads(argv[3]), argv[4]
    lessons, out = int(argv[5]), Path(argv[6])
    rate = float(genome.pop("bind_rate", 0.01))
    p = ex.load(path, seed, genome, streams=32)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 99]))
    held = nt.founder_held(seed)
    empty = nt.Room([], 0.0)
    log: list[dict] = []
    began = time.time()

    def reading(at: int) -> None:
        sc = ii.state_completeness(p, rng, held, count=384)
        fits = sc.pop("fits")
        comp = ii.completion(p, rng, held, fits)
        words = [k for k in comp if k.split()[1] in nt.KIND_NAMES]
        row = {"lessons": at, "seconds": round(time.time() - began),
               "word_alone_kind": float(np.mean([comp[k]["kind"] for k in words])),
               "word_alone_round": float(np.mean([comp[k]["round"] for k in words])),
               "word_alone_edible": float(np.mean([comp[k]["edible"] for k in words])),
               "colour_word_alone": float(np.mean([comp[k]["colour"] for k in comp
                                                   if k.split()[1] in nt.COLOURS])),
               "state_association": sc["read"]["association"],
               "forms": nt.read(p, rng, ["name", "imitate", "by_colour"], held, 32)}
        log.append(row)
        print(json.dumps(row), flush=True)
        out.write_text(json.dumps({"arm": arm, "rate": rate, "log": log}, indent=1))

    reading(0)
    for k in range(1, lessons + 1):
        pairs = ii.scenes(p, rng, p.streams, held, test=False)
        rooms = [r for r, _ in pairs]
        words = [t.kind for _, t in pairs]
        if arm == "bind":
            both = settled(p, rooms, [[None, w] for w in words], rng)  # the word, the thing seen
            alone = settled(p, [empty] * len(rooms), [[None, w] for w in words], rng)
            contrast_step(p, alone, both, rate)  # the word alone pulled toward word + sight
            look = settled(p, rooms, [[None, None] for _ in words], rng)
            contrast_step(p, look, both, rate)  # the glimpse alone pulled toward glimpse + word
        else:  # the control: the brain's own naming lesson, the name the label at the gap
            exs = nt.block(p, rng, "name", "dagger", held)
            nt.live(p, exs, rng)
        if k % 50 == 0:
            reading(k)


if __name__ == "__main__":
    main(sys.argv)
