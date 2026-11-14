"""A diagnostic arm for cross-modal binding inside the brain's own rule (cadence#334, for
review-topdown). It makes no change to the library: it calls the learner's own phases.

    python state_target_arm.py <brain.npz> <seed> '<genome>' <arm> <rounds> <out.json>

The library's lesson nudges only the motor outputs. Here the same learner takes a nudge on the
association instead:
- the free phase settles the observation with a kind's word heard and nothing in view;
- the nudged and opposite phases add ``beta * (T - s)`` (and its negative) on the association's
  neurons, where ``T`` is the association's settled state under the word with the thing in view;
- ``Learner.update`` moves the synapses on the free/nudged contrast with its normalized step.
The mirror lesson in the same round: the glimpse alone, pulled toward the same both-senses state.

Each round (``bind``) is one binding lesson pair on 32 things, then one ordinary name:dagger
block, and every third round an imitation block, so names and imitation are rehearsed. The
``control`` arm runs the same rounds without the binding pair. Every 25 rounds it reads the
completion matrix (a word alone through the focus probes), the association's state completeness,
names, imitation and by_colour. Each binding lesson costs the target's settle (the free phase of
word + sight) plus the lesson's three settles (free, nudged, opposite).
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

from cadence.brain import Nudge  # noqa: E402


def observation(p: nt.Parrot, rooms: list[nt.Room], words: list[str | None],
                rng: np.random.Generator) -> np.ndarray:
    exs = [nt.Exchange(r, [w], 0, None) for r, w in zip(rooms, words, strict=True)]
    n = len(rooms)
    return p.observe(exs, 0, [None] * n, np.zeros(n, dtype=bool), rng)


def free_state(p: nt.Parrot, x: np.ndarray) -> object:
    brain = p.brain
    brain.reset()
    brain._prepare(x)
    drive = brain._moment_drive(x)
    return drive, brain.learner.free(drive, None)


MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")


OWN: dict[str, np.ndarray] = {}  # the binding lesson's own optimizer moments (review E4)


def pull(p: nt.Parrot, x: np.ndarray, target: np.ndarray, beta: float,
         masked: bool | str = False, own: bool = False) -> dict[str, float]:
    """One lesson of the brain's own rule with the association pulled toward ``target``. With
    ``masked`` the step (and the optimizer's moments) is kept only on the synapses into the
    association (its inputs and its own) and on the association's biases."""
    learner = p.brain.learner
    if own:  # the binding learner's moments in place of the brain's, for this lesson alone
        held_moments = {k: np.array(getattr(learner, k), copy=True) for k in MOMENTS}
        held_count = learner.contrast_updates
        for k in MOMENTS:
            setattr(learner, k, OWN.get(k, np.zeros_like(held_moments[k])))
        learner.contrast_updates = int(OWN.get("count", np.array(0)))
    if masked:
        graph = learner.brain
        before = (np.array(graph.efficacy, copy=True), np.array(graph.bias, copy=True))
        moments = {k: np.array(getattr(learner, k), copy=True) for k in MOMENTS}
    n = p.brain.connectome.n
    assoc = np.asarray(list(p.brain.connectome.populations["association"]))
    drive, free = free_state(p, x)
    full = np.zeros((len(x), n))
    full[:, assoc] = target
    mask = np.zeros(n)
    mask[assoc] = 1.0
    held = learner.brain.apical_gain(drive, free, own_precision=True)
    nudged = learner._nudged(drive, free, Nudge(full, mask, beta, apical_gain=held), "nudged")
    opposite = learner._nudged(drive, free, Nudge(full, mask, -beta, apical_gain=held),
                               "opposite")
    report = learner.update(free, nudged, opposite)
    if masked:
        c = p.brain.connectome
        neurons = np.zeros(c.n, dtype=bool)
        neurons[assoc] = True
        keep = neurons[np.asarray(c.post)]  # the synapses into the association
        if masked == "ear":  # the word's synapses alone: from the ear into the association
            ear = np.zeros(c.n, dtype=bool)
            ear[np.asarray(list(c.populations["sensory/ear"]))] = True
            keep = keep & ear[np.asarray(c.pre)]
        graph = learner.brain
        eff = np.where(keep, np.asarray(graph.efficacy), before[0])
        bias = np.where(neurons, np.asarray(graph.bias), before[1])
        learner.brain = graph.with_parameters(efficacy=eff, bias=bias)
        for k in MOMENTS:  # the moments of the synapses held still stay as they were
            mask = keep if k in ("velocity", "second_moment") else neurons
            setattr(learner, k, np.where(mask, np.asarray(getattr(learner, k)), moments[k]))
    if own:
        for k in MOMENTS:
            OWN[k] = np.array(getattr(learner, k), copy=True)
            setattr(learner, k, held_moments[k])
        OWN["count"] = np.array(learner.contrast_updates)
        learner.contrast_updates = held_count
    p.brain.reset()
    return report


def main(argv: list[str]) -> None:
    path, seed, genome, arm = argv[1], int(argv[2]), json.loads(argv[3]), argv[4]
    rounds, out = int(argv[5]), Path(argv[6])
    beta = float(genome.pop("state_beta", 0.1))
    masked = genome.pop("state_masked", False)
    own = bool(genome.pop("state_own", False))
    from_glimpse = bool(genome.pop("state_from_glimpse", False))
    p = ex.load(path, seed, genome, streams=32)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 334, 98]))
    held = nt.founder_held(seed)
    empty = nt.Room([], 0.0)
    assoc = np.asarray(list(p.brain.connectome.populations["association"]))
    log: list[dict] = []
    began = time.time()

    def reading(at: int) -> None:
        sc = ii.state_completeness(p, rng, held, count=384)
        fits = sc.pop("fits")
        comp = ii.completion(p, rng, held, fits)
        kinds = [k for k in comp if k.split()[1] in nt.KIND_NAMES]
        colours = [k for k in comp if k.split()[1] in nt.COLOURS]
        row = {"rounds": at, "seconds": round(time.time() - began),
               "word_alone_kind": float(np.mean([comp[k]["kind"] for k in kinds])),
               "word_alone_round": float(np.mean([comp[k]["round"] for k in kinds])),
               "word_alone_edible": float(np.mean([comp[k]["edible"] for k in kinds])),
               "colour_word_alone": float(np.mean([comp[k]["colour"] for k in colours])),
               "state_association": sc["read"]["association"],
               "forms": nt.read(p, rng, ["name", "imitate", "by_colour"], held, 32)}
        log.append(row)
        print(json.dumps(row), flush=True)
        out.write_text(json.dumps({"arm": arm, "beta": beta, "log": log}, indent=1))

    reading(0)
    for k in range(1, rounds + 1):
        if arm == "bind":
            pairs = ii.scenes(p, rng, p.streams, held, test=False)
            rooms = [r for r, _ in pairs]
            words = [t.kind for _, t in pairs]
            if from_glimpse:  # the review's form: the word alone pulled toward what the
                # thing in view settles with no word (the other sense alone)
                _, seen = free_state(p, observation(p, rooms, [None] * len(rooms), rng))
                target = np.asarray(seen.activation, dtype=np.float64)[:, assoc].copy()
                pull(p, observation(p, [empty] * len(rooms), words, rng), target, beta, masked,
                     own)
            else:
                _, both = free_state(p, observation(p, rooms, words, rng))
                target = np.asarray(both.activation, dtype=np.float64)[:, assoc].copy()
                pull(p, observation(p, [empty] * len(rooms), words, rng), target, beta, masked,
                     own)
                pull(p, observation(p, rooms, [None] * len(rooms), rng), target, beta, masked,
                     own)
        nt.live(p, nt.block(p, rng, "name", "dagger", held), rng)
        if k % 3 == 0:
            nt.live(p, nt.blocks_of(p, rng, "imitate", "turn", held), rng)
        if k % 25 == 0:
            reading(k)


if __name__ == "__main__":
    main(sys.argv)
