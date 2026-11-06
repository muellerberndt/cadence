"""A controlled ordinary query-seam initializer, outside the library interface.

The frozen factory supplies every neuron, contact and runtime equation. This
initializer changes only fresh query/association efficacies. A separate seeded
child stream draws each reciprocal projection through the existing Genome API;
it uses the ordinary projection distribution, not the full factory's RNG replay.
No head is assigned a name, output slot or memory record. No behavioral benefit
or supported lifecycle is established by this construction control.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from cadence.genome import Genome, Projection, develop
from cadence.receipts import canonical_sha256
from cadence.regions import Region

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "query_founder_admission", HERE / "heads_admission.py"
)
assert SPEC is not None and SPEC.loader is not None
admission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = admission
SPEC.loader.exec_module(admission)


def scale_value(query_scale: Any) -> float:
    if isinstance(query_scale, (bool, np.bool_)) or not isinstance(query_scale, Real):
        raise ValueError("query_scale must be the experimental initializer value 0 or 1")
    try:
        value = float(query_scale)
    except (ValueError, OverflowError) as error:
        raise ValueError("query_scale must be 0 or 1") from error
    if not np.isfinite(value) or value not in (0.0, 1.0):
        raise ValueError("query_scale must be the experimental initializer value 0 or 1")
    return value


def founder_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative founder integer")
    return int(seed)


def parameter_digest(graph: Any) -> str:
    return canonical_sha256(
        {name: getattr(graph, name) for name in ("efficacy", "log_gain", "bias")}
    )


def outside_digest(graph: Any, seam: np.ndarray) -> str:
    arrays = {"efficacy": graph.efficacy[~seam], "bias": graph.bias, "log_gain": graph.log_gain}
    return canonical_sha256(
        {
            name: {
                "shape": array.shape,
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
            for name, array in arrays.items()
        }
    )


def initialize(brain: Any, *, seed: int, query_scale: float = 0) -> dict[str, Any]:
    """Initialize once before any moment; attach and return an explicit record.

    At zero, the original Brain and Graph are returned untouched apart from this
    observational record. A plain exposed control has no query ports and is
    unchanged at either setting. Counters, optimizer state and action RNG never
    participate in these separate initializer streams.
    """
    scale, seed = scale_value(query_scale), founder_seed(seed)
    learner, actor, graph = brain.learner, brain.basal_ganglia, brain.brain
    recall = brain.recall
    if (
        hasattr(brain, "query_initialization")
        or learner.updates
        or learner.contrast_updates
        or actor.state is not None
        or getattr(actor, "_free_brain", None) is not None
        or (recall is not None and (recall.writes or recall.last is not None))
        or any(
            np.any(getattr(learner, name))
            for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias")
        )
    ):
        raise ValueError("query initialization requires a fresh brain before its first moment")
    if graph.backend != "cpu":
        raise ValueError("query initializer follows the CPU benchmark factory")
    queries = tuple(getattr(brain, "head_queries", ()))
    association = np.asarray(brain.association_index, dtype=np.int64)
    connectome = graph.connectome
    if not np.array_equal(association, connectome.populations["association"]):
        raise ValueError("association ports differ from the frozen anatomy")
    if queries and (
        len(queries) != brain.heads
        or any(
            not np.array_equal(
                query, connectome.populations["query" if head == 0 else f"query_{head}"]
            )
            for head, query in enumerate(queries)
        )
    ):
        raise ValueError("query port descriptors differ from the existing anatomy")
    lookup = {
        (int(pre), int(post)): index
        for index, (pre, post) in enumerate(zip(connectome.pre, connectome.post, strict=True))
    }
    efficacy, seam, heads = graph.efficacy.copy(), np.zeros(connectome.synapses, dtype=bool), []
    for head, query in enumerate(queries):
        query = np.asarray(query, dtype=np.int64)
        initializer_seed = np.random.SeedSequence([seed, head]).spawn(1)[0]
        draw = develop(
            Genome(
                (Region("query", len(query)), Region("association", len(association))),
                (Projection("query", "association"),),
            ),
            seed=initializer_seed,
        )
        ports = np.concatenate((query, association))
        for pre, post, count, value in zip(draw.pre, draw.post, draw.count, draw.sign, strict=True):
            edge = lookup.get((int(ports[pre]), int(ports[post])))
            if edge is None or connectome.count[edge] != count:
                raise ValueError("ordinary query projection does not match existing contacts")
            seam[edge] = True
            if scale:
                efficacy[edge] = value
        heads.append(
            {
                "head": head,
                "query": query.tolist(),
                "association": association.tolist(),
                "seed_entropy": [seed, head],
                "seed_spawn_key": list(initializer_seed.spawn_key),
                "ordinary_projection_digest": draw.digest(),
            }
        )
    record = {
        "schema": "cadence-attention-query-founder/1",
        "query_scale": scale,
        "applicable": bool(queries),
        "applied": bool(scale and queries),
        "heads": heads,
        "seed_rule": (
            "SeedSequence([founder, head]).spawn(1)[0]; "
            "ordinary isolated reciprocal Projection defaults"
        ),
        "scope": (
            "Initial efficacies only; separate initializer distribution, "
            "no roles or new runtime neurons"
        ),
        "parameters_before": parameter_digest(graph),
        "outside_query_before": outside_digest(graph, seam),
    }
    if scale and queries:
        candidate = graph.with_parameters(efficacy=efficacy)
        if (
            type(candidate) is not type(graph)
            or candidate.connectome is not connectome
            or candidate.memory is not graph.memory
        ):
            raise ValueError(
                "query initialization must preserve graph class, anatomy and memory boundary"
            )
        learner.brain = candidate
    record.update(
        parameters_after=parameter_digest(brain.brain),
        outside_query_after=outside_digest(brain.brain, seam),
        query_synapses=int(seam.sum()),
    )
    brain.query_initialization = record
    return record


def make_brain(
    arm: str,
    seed: int,
    protocol: dict[str, Any],
    mode: str = "agreement",
    *,
    query_scale: float = 0,
    factory: Any = None,
) -> Any:
    """Wrap an explicitly supplied frozen factory without patching its namespace."""
    query_scale, seed = scale_value(query_scale), founder_seed(seed)
    factory = admission.make_brain if factory is None else factory
    brain = factory(arm, seed, protocol, mode)
    initialize(brain, seed=seed, query_scale=query_scale)
    return brain
