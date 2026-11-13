"""One deep brain that learns what words mean (cadence#334): its composition as genes.

The brain is ``Brain.compose`` with a chain of processing regions (``modules=``) ending in the
association region. The eyes reach the lowest region and the ear reaches the association region,
so that what is heard (the question's words, the present one from the ear, the earlier ones held
by the working trace and fetched back by the recall memory's heads) reaches the regions below
top-down, through the apical compartment (``apical=``). The brain holds the recall memory with
``recall_heads`` heads and the actor-critic that the pistachio teaches. The motor region has a
slot for the voice: one neuron for each word the parrot has heard (each says that word's
remembered sound) and one for silence; with ``gaze`` a second slot turns the head to a direction
around the body.

Every value that shapes the composition is a gene: ``HAND_SET`` is the hand-set founder, the
control every selected genome is read against, and ``SPACE`` declares each gene for
``cadence.genes``. Gene names follow the parrot's wiring study where they overlap
(``association``, ``working_memory_amplitude``, ``carried_gain``, ``history_decays``,
``recall``, ``recall_heads``, ``efference_amplitude``, ``actor_eta``, ``actor_lam``). Two genes
are of the body's born learning and not of ``compose``: ``babble_lesson`` (the parrot hears
itself and learns which of its acts made the sound: the inverse model of its voice, from
babbling) and ``self_reward`` (the born comparison of the parrot's own voice with the sound it
heard just before: the songbird's performance error, a reward the brain gives itself).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..connectome import Connectome
from ..generic import Brain
from ..genome import Genome, Projection, develop
from ..regions import Region

# The hand-set composition: the founder of every lineage and the control.
HAND_SET: dict[str, Any] = {
    "depth": 2,                     # processing regions below the association region
    "module_width": 128,            # the first (lowest) region's neurons
    "module_taper": 1.0,            # each region above holds this share of the one below
    "association": 128,             # the association region's neurons
    "apical": 0.25,                 # the apical compartment's fraction (0: summed alike)
    "teaching": "deep",
    "recall": 1,
    "recall_heads": 2,
    "recall_capacity": 32,
    "history_decays": (0.0,),
    "working_memory_amplitude": 3.0,
    "carried_gain": 0.01,
    "efference_amplitude": 0.0,
    "sensory_scale": 1.0,
    "temperature": 0.2,
    "actor_eta": 1.0,
    "actor_lam": 0.8,
    "actor_gamma": 0.9,
    "actor_eta_critic": 0.3,
    "actor_hold": 0.0,              # 0: absent (the actor's trace fades by gamma * lam)
    "recall_address_reward": 0.0,
    "ear_entry": "association",     # where the ear reaches: the association region (top)
    "babble_lesson": 1,
    "self_reward": 1.0,
}

SPACE: dict[str, tuple[Any, ...]] = {
    "depth": ("int", 0, 4),
    "module_width": ("log", 0.25, 32.0, 1024.0),
    "module_taper": ("log", 0.2, 0.25, 2.0),
    "association": ("log", 0.25, 32.0, 1024.0),
    "apical": ("linear", 0.05, 0.0, 0.6),
    "teaching": ("choice", "deep", "normalized", "founder"),
    "recall": ("choice", 0, 1),
    "recall_heads": ("int", 1, 8),
    "recall_capacity": ("log", 0.3, 8.0, 256.0),
    "history_decays": ("choice", (), (0.0,), (0.0, 0.5), (0.0, 0.8)),
    "working_memory_amplitude": ("linear", 0.3, 0.0, 6.0),
    "carried_gain": ("log", 0.5, 0.001, 0.1),
    "efference_amplitude": ("linear", 0.2, 0.0, 2.0),
    "sensory_scale": ("log", 0.2, 0.25, 4.0),
    "temperature": ("log", 0.2, 0.05, 1.0),
    "actor_eta": ("log", 0.3, 0.01, 4.0),
    "actor_lam": ("linear", 0.1, 0.0, 0.99),
    "actor_gamma": ("linear", 0.05, 0.5, 0.99),
    "actor_eta_critic": ("log", 0.3, 0.01, 1.0),
    "actor_hold": ("linear", 0.1, 0.0, 1.0),
    "recall_address_reward": ("log", 0.5, 0.01, 8.0),
    "ear_entry": ("choice", "association", "module_0", "both"),
    "babble_lesson": ("choice", 0, 1),
    "self_reward": ("linear", 0.2, 0.0, 2.0),
}

BODY_GENES = ("babble_lesson", "self_reward")


def modules(genome: dict[str, Any]) -> tuple[int, ...]:
    """The processing chain: ``depth`` regions from ``module_width`` tapering by
    ``module_taper``, then the association region."""
    out: list[int] = []
    width = float(genome["module_width"])
    for _ in range(int(genome["depth"])):
        out.append(max(4, int(round(width))))
        width *= float(genome["module_taper"])
    out.append(max(4, int(round(float(genome["association"])))))
    return tuple(out)


def anatomy(genome: dict[str, Any], inputs: int, ear: int, actions: int,
            slots: Any) -> Genome:
    """The layout ``Brain.compose`` develops for ``genome``, with its sensory region split into
    two populations: ``eyes`` (the first ``inputs - ear`` inputs), which reach the lowest
    processing region, and ``ear`` (the last ``ear``), which reaches the association region
    (``ear_entry="association"``, the hand-set: the auditory forebrain reaches the
    caudolateral nidopallium, and what is heard reaches the regions below top-down, through the
    apical compartment), the lowest region (``"module_0"``, as ``compose`` wires every input)
    or both."""
    g = {**HAND_SET, **genome}
    memory: dict[str, Any] = {"history_decays": tuple(g["history_decays"])}
    if int(g["recall"]):
        memory.update(recall=True, recall_heads=int(g["recall_heads"]))
    if float(g["efference_amplitude"]) > 0:
        memory["efference_amplitude"] = float(g["efference_amplitude"])
    base = Genome.composed(int(inputs), int(actions), modules=modules(g), slots=slots, **memory)
    eyes = int(inputs) - int(ear)
    senses = Connectome.from_synapses(
        int(inputs), pre=[], post=[], sign=[],
        populations={"eyes": list(range(eyes)), "ear": list(range(eyes, int(inputs)))},
        label="eyes and ear")
    regions = tuple(Region("sensory", circuit=senses) if r.name == "sensory" else r
                    for r in base.regions)
    scale = float(g["sensory_scale"])
    first = base.regions[1].name  # the lowest processing region (the association when flat)
    entry = str(g["ear_entry"])
    sensory = [Projection("sensory/eyes", first, reciprocal=False, scale=scale)]
    if str(g.get("eyes_entry", "module_0")) == "both" and first != "association":
        # what the eyes see reaches the association region directly too (as the tectofugal
        # pathway's entopallium reaches the nidopallium), beside the chain below it
        sensory.append(Projection("sensory/eyes", "association", reciprocal=False, scale=scale))
    if entry in ("association", "both") and first != "association":
        sensory.append(Projection("sensory/ear", "association", reciprocal=False, scale=scale))
    if entry in ("module_0", "both") or first == "association":
        sensory.append(Projection("sensory/ear", first, reciprocal=False, scale=scale))
    if int(g.get("ear_motor", 0)):
        # the vocal motor pathway: what the ear hears reaches the voice directly (the song
        # system's core, auditory input to the motor nuclei), beside the association's route
        sensory.append(Projection("sensory/ear", "motor", reciprocal=False, scale=scale))
    rest = tuple(p for p in base.projections if p.pre != "sensory")
    return Genome(regions=regions, projections=tuple(sensory) + rest, label="parrot")


def compose(genome: dict[str, Any], inputs: int, words: int, *, ear: int = 0, gaze: int = 0,
            seed: int = 0, backend: str = "cpu", **extra: Any) -> Brain:
    """The deep brain of ``genome`` for observations of ``inputs`` values whose last ``ear`` are
    what the ear hears, a voice of ``words`` words and silence, and (with ``gaze``) a head that
    turns to ``gaze`` directions. With ``ear`` the layout is ``anatomy``'s; without it every
    input reaches the lowest region, as ``Brain.compose`` wires it."""
    g = {**HAND_SET, **genome}
    slots: Any = (words + 1, gaze) if gaze else 1
    actions = words + 1 + gaze
    settings: dict[str, Any] = {
        "teaching": str(g["teaching"]),
        "slots": slots,
        "seed": int(seed),
        "backend": backend,
        "history_decays": tuple(g["history_decays"]),
        "working_memory_amplitude": float(g["working_memory_amplitude"]),
        "temperature": float(g["temperature"]),
        "actor_eta": float(g["actor_eta"]),
        "actor_lam": float(g["actor_lam"]),
        "actor_gamma": float(g["actor_gamma"]),
        "actor_eta_critic": float(g["actor_eta_critic"]),
    }
    if float(g["actor_hold"]) > 0:
        settings["actor_hold"] = float(g["actor_hold"])
    if ear:
        layout = anatomy(g, inputs, ear, actions, slots)
        settings["connectome"] = develop(layout, seed=int(seed))
    else:
        settings.update(modules=modules(g), scales={"sensory": float(g["sensory_scale"])})
    if float(g["apical"]) > 0 and len(modules(g)) > 1:
        settings["apical"] = float(g["apical"])
    if float(g["efference_amplitude"]) > 0:
        settings["efference_amplitude"] = float(g["efference_amplitude"])
    if int(g["recall"]):
        settings.update(recall=True, recall_heads=int(g["recall_heads"]),
                        recall_capacity=max(2, int(round(float(g["recall_capacity"])))),
                        carried_gain=float(g["carried_gain"]))
        if float(g["recall_address_reward"]) > 0.0:
            settings["recall_address_reward"] = float(g["recall_address_reward"])
        for name in ("recall_position_reward", "recall_window", "recall_founder"):
            if name in genome:  # named by the genome alone (no gene of the space)
                settings[name] = genome[name]
    else:
        settings["recall"] = False
        if float(g["carried_gain"]) > 0:
            settings["carried_gain"] = float(g["carried_gain"])
    for name in genome:  # the learner's own settings, named by the genome alone
        if name.startswith("learning_"):
            settings[name] = genome[name]
    settings.update(extra)
    return Brain.compose(int(inputs), int(actions), **settings)


def random_genome(rng: np.random.Generator) -> dict[str, Any]:
    """One uniform draw over ``SPACE`` (the random-search control); log genes uniform in the
    logarithm."""
    out: dict[str, Any] = {}
    for name, spec in SPACE.items():
        kind = spec[0]
        if kind == "log":
            out[name] = float(math.exp(rng.uniform(math.log(spec[2]), math.log(spec[3]))))
        elif kind == "linear":
            out[name] = float(rng.uniform(spec[2], spec[3]))
        elif kind == "int":
            out[name] = int(rng.integers(spec[1], spec[2] + 1))
        else:
            out[name] = spec[1 + int(rng.integers(len(spec) - 1))]
    return out
