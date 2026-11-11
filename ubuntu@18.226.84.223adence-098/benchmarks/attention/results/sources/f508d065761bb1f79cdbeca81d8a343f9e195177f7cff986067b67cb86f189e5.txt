"""Experimental independent memory reads in one coupled settlement.

Every head reads the same witnessed bank through its own live query/read ports.
The memory energy is the sum of the existing per-head energies, with the total
coupling split equally. There is no answer vote, head assignment teacher or
diversity penalty. These are benchmark classes, not supported library options.
"""

from __future__ import annotations

import importlib.util
import sys
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from cadence import Brain
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region

SPEC = importlib.util.spec_from_file_location(
    "attention_heads_coupled", Path(__file__).with_name("coupled.py")
)
assert SPEC is not None and SPEC.loader is not None
coupled = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = coupled
SPEC.loader.exec_module(coupled)
AttentionBrain, Graph, Memory = coupled.AttentionBrain, coupled.Graph, coupled.Memory


def positive_count(name: str, value: Any) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class MultiMemory:
    """Disjoint live ports over one immutable witnessed bank.

    ``weights`` has shape ``(heads, streams, slots)``. Arrays of witnessed data
    are shared by identity across the stored members, not copied per head.
    Different amplitudes permit explicit head lesions; all selectors otherwise
    use the same mode, temperature, null and recency. No diversity is imposed.
    """

    memories: tuple[Memory, ...]

    def __post_init__(self) -> None:
        members = tuple(self.memories)
        if not members or any(not isinstance(member, Memory) for member in members):
            raise ValueError("heads require a nonempty tuple of memory boundaries")
        first = members[0]
        data = ("keys", "values", "held", "ages")
        settings = ("mode", "temperature", "null", "recency", "empty_anchor")
        for member in members[1:]:
            if (any(not np.array_equal(getattr(first, name), getattr(member, name))
                    for name in data)
                    or any(getattr(first, name) != getattr(member, name) for name in settings)):
                raise ValueError("heads must read the same witnessed bank and selector settings")
        ports = np.concatenate([
            index for member in members for index in (member.query, member.read)
        ])
        if len(np.unique(ports)) != len(ports):
            raise ValueError("heads require disjoint query and read ports")
        # Own one snapshot, then share its readonly arrays across copied port descriptors.
        owned = replace(first)
        shared = [owned]
        for member in members[1:]:
            head = copy(member)
            for name in data:
                object.__setattr__(head, name, getattr(owned, name))
            shared.append(head)
        object.__setattr__(self, "memories", tuple(shared))

    @property
    def heads(self) -> int:
        return len(self.memories)

    @property
    def streams(self) -> int:
        return self.memories[0].streams

    @property
    def keys(self) -> np.ndarray:
        return self.memories[0].keys

    @property
    def values(self) -> np.ndarray:
        return self.memories[0].values

    @property
    def held(self) -> np.ndarray:
        return self.memories[0].held

    @property
    def ages(self) -> np.ndarray:
        return self.memories[0].ages

    @property
    def mode(self) -> str:
        return str(self.memories[0].mode)

    @property
    def empty_anchor(self) -> bool:
        return self.memories[0].empty_anchor

    def take(self, rows: Any) -> MultiMemory:
        """Restrict the one bank once, retaining every head's stream identity."""
        bank = self.memories[0].take(rows)
        members = [bank]
        for member in self.memories[1:]:
            head = copy(member)
            for name in ("keys", "values", "held", "ages"):
                object.__setattr__(head, name, getattr(bank, name))
            members.append(head)
        return MultiMemory(tuple(members))

    def weights(self, state: np.ndarray) -> np.ndarray:
        return np.stack([member.weights(state) for member in self.memories])

    def energy(self, state: np.ndarray) -> np.ndarray:
        total = self.memories[0].energy(state)
        for member in self.memories[1:]:
            total += member.energy(state)
        return total

    def drive(self, state: np.ndarray) -> np.ndarray:
        total = self.memories[0].drive(state)
        for member in self.memories[1:]:
            total += member.drive(state)
        return total


class MultiHeadBrain(AttentionBrain):
    """Benchmark anatomy with independent live reads and no external answer path.

    ``heads=1, port_copies=1`` preserves the existing prototype. ``port_copies``
    is a capacity control: repeat the SAME key/value coordinates isometrically,
    and enlarge every head's ports accordingly, retaining one selector per head.
    It adds no observed information. With ``heads=2, port_copies=1`` versus
    ``heads=1, port_copies=2``, total live port coordinates and leading memory
    dot-product work match; softmax work and actual graph work remain measured.

    All heads receive the same current cue, rather than supplied per-head roles.
    Read/cortex seams are ordinary independent seeded draws; query seams start
    at zero, as in the single-head prototype. Symmetric heads may still collapse.
    Set ``recall_next=False`` explicitly for tasks requiring two distinct facts.
    """

    heads: int = 1
    port_copies: int = 1
    head_queries: tuple[np.ndarray, ...]
    head_reads: tuple[np.ndarray, ...]
    head_attended: np.ndarray | None = None

    @classmethod
    def compose(
        cls, inputs: int, actions: int, *, heads: int = 1, port_copies: int = 1,
        mode: str = "agreement", dt: float = 0.1, seed: int = 0,
        empty_anchor: bool = False, **options: Any,
    ) -> MultiHeadBrain:
        heads = positive_count("heads", heads)
        port_copies = positive_count("port_copies", port_copies)
        empty_anchor = coupled.anchor_setting(empty_anchor, mode)
        if heads == port_copies == 1:
            brain = super().compose(inputs, actions, mode=mode, dt=dt, seed=seed,
                                    empty_anchor=empty_anchor, **options)
            brain.heads, brain.port_copies = heads, port_copies
            populations = brain.connectome.populations
            assert brain.recall is not None
            brain.head_queries = (coupled.frozen(populations["query"], np.int64),)
            brain.head_reads = (coupled.frozen(np.concatenate((
                brain.recall.recall_neurons, brain.recall.next_neurons,
            )), np.int64),)
            return brain
        if options.get("backend", "cpu") != "cpu":
            raise ValueError("attention instrument supports CPU float64 only")
        if options.get("learning_surprise", False) or options.get("recall_window", 1) != 1:
            raise ValueError("heads require saved-row teaching and one-event keys")
        if mode not in ("agreement", "linear", "competitive"):
            raise ValueError("unknown memory equation")
        options.update(recall=True, recall_address=0.0, recall_null=0.0,
                       learning_qualified=True, learning_damping=0)
        options.setdefault("learning_nudged_steps", options.get("learning_free_steps", 1024))
        genome = Genome.composed(inputs, actions, seed=seed, **options)
        width = Genome.observation_width(inputs, options.get("code"), options.get("learned_code"))
        read_names = ("recall", "next") if options.get("recall_next", True) else ("recall",)
        read_width = width * len(read_names)
        projections = [
            replace(p, reciprocal=True) if p.pre in read_names and p.post == "association"
            else p for p in genome.projections
        ]
        read_scale = next(p.scale for p in projections if p.pre == "recall")
        regions = [*genome.regions, Region("query", port_copies * width)]
        projections.append(Projection("query", "association", scale=0.0))
        if port_copies > 1:
            regions.append(Region("read_0_extra", (port_copies - 1) * read_width))
            projections.append(Projection("read_0_extra", "association", scale=read_scale))
        for head in range(1, heads):
            regions.extend((Region(f"query_{head}", port_copies * width),
                            Region(f"read_{head}", port_copies * read_width)))
            projections.extend((Projection(f"query_{head}", "association", scale=0.0),
                                Projection(f"read_{head}", "association", scale=read_scale)))
        anatomy = Genome(tuple(regions), tuple(projections), label="experimental-memory-heads")
        settings = {key: value for key, value in options.items()
                    if key not in ("modules", "observers", "lateral", "scales")}
        brain = Brain.compose.__func__(
            cls, inputs, actions, connectome=develop(anatomy, seed=seed), seed=seed, **settings,
        )
        brain.mode, brain.heads, brain.port_copies = mode, heads, port_copies
        brain.empty_anchor = empty_anchor
        original = brain.brain
        brain.learner.brain = Graph(
            original.connectome, original.neuron_model.replace(dt=dt),
            efficacy=original.efficacy, bias=original.bias, log_gain=original.log_gain,
        )
        populations = brain.connectome.populations
        queries = [populations["query"]]
        reads = [np.concatenate([np.asarray(populations[name]) for name in read_names])]
        if port_copies > 1:
            reads[0] = np.concatenate((reads[0], populations["read_0_extra"]))
        for head in range(1, heads):
            queries.append(populations[f"query_{head}"])
            reads.append(np.asarray(populations[f"read_{head}"]))
        brain.head_queries = tuple(coupled.frozen(port, np.int64) for port in queries)
        brain.head_reads = tuple(coupled.frozen(port, np.int64) for port in reads)
        return brain

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        if self.heads == self.port_copies == 1:
            drive = super().stimulus(observations, memory=memory)
            if self.brain.memory is not None:
                self.brain.memory = MultiMemory((self.brain.memory,))
            return drive
        cfg = self.learner.config
        if not cfg.qualified or cfg.damping or cfg.surprise:
            raise ValueError("heads require qualified saved-row teaching and damping=surprise=0")
        recall = self.recall
        if recall is None or recall.address != 0 or recall.window != 1:
            raise ValueError("heads require one-event keys and no address lesson")
        drive = Brain.stimulus(self, observations, memory=memory)
        self.brain.memory = None
        if not memory:
            return drive
        assert recall.last is not None
        copies = self.port_copies
        cue = np.tile(recall.last.key, (1, copies)) / np.sqrt(copies)
        batch = len(drive)
        if recall.streams == batch:
            keys = np.tile(recall.keys, (1, 1, copies)) / np.sqrt(copies)
            content = (np.concatenate((recall.values, recall.nexts), axis=2)
                       if recall.next else recall.values)
            values = np.tile(content, (1, 1, copies)) / np.sqrt(copies)
            held, ages = recall.held, recall.ages()
        else:
            keys = np.zeros((batch, 0, len(self.head_queries[0])))
            values = np.zeros((batch, 0, len(self.head_reads[0])))
            held, ages = np.zeros((batch, 0), dtype=bool), np.zeros(0)
        members = []
        for query, read in zip(self.head_queries, self.head_reads, strict=True):
            drive[:, query] = cue
            drive[:, read] = 0.0
            members.append(Memory(
                query, read, keys, values, held, ages,
                amplitude=recall.amplitude / self.heads, temperature=recall.temperature,
                recency=recall.recency, mode=self.mode, empty_anchor=self.empty_anchor,
            ))
        recall._attended = None
        self.brain.memory = MultiMemory(tuple(members))
        return drive

    def _choose(self, x: np.ndarray, drive: np.ndarray, free: Any, **options: Any) -> np.ndarray:
        memory = self.brain.memory
        if memory is not None and self.recall is not None:
            attended = None if memory.mode == "linear" else memory.weights(
                np.atleast_2d(free.activation)
            )
            self.head_attended = None if attended is None else coupled.frozen(attended)
            # Keep the legacy single-read diagnostic's documented stream-by-slot shape.
            self.recall._attended = (
                None if attended is None else coupled.frozen(attended[0])
            )
        return Brain._choose(self, x, drive, free, **options)
