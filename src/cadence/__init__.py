"""Cadence: machine learning by patch-net settlement.

A patch net is a set of owners, each holding one patch of state, joined by
declared overlaps. Nothing is computed globally: every owner repairs its
own patch from what arrives over its overlaps, and the state the net comes
to rest in is the answer. Cadence gives you the wiring, the owner rule,
the settlement engine on CPU or an accelerator, an owner-by-owner
reference engine with a message ledger to check the accelerated one
against, a protocol layer for declared held-out tests with a shuffled
control, and receipts that bind every result to the code and data that
produced it.

    >>> import cadence as cd
    >>> wiring = cd.Wiring.from_edges(4, pre=[0, 1, 2, 3], post=[1, 2, 3, 0], count=[120] * 4)
    >>> engine = cd.Settlement(wiring, cd.GradedRule(gain=0.03))
    >>> engine.settle(clamp={0: 3.0}, steps=60).activation.round(2)
    array([1., 1., 1., 1.])
"""

from __future__ import annotations

from .checkpoint import load, save
from .constitution import Constitution, Projection, Region, evolve, grow, mutate
from .custody import Source, fetch, manifest
from .dream import DiscreteCode, DreamActorCritic, DreamConfig, actor_critic_wiring
from .learning import Learner, LearnerConfig, embedded, layered, learning_rule
from .plasticity import (
    ActorCritic,
    ActorCriticConfig,
    Bins,
    Population,
    Rehearsal,
    RehearsalConfig,
    ValueConfig,
    ValueNet,
)
from .protocol import Protocol, Row, evaluate_predicate, select_gain, shuffled
from .receipts import Receipt, canonical_json, canonical_sha256, source_manifest
from .reference import Ledger, conformance, settle_owner_by_owner
from .rules import Adaptation, GradedRule
from .settle import Nudge, SettledState, Settlement, available_backends
from .stream import Echo, stateful
from .structure import Seams, SleepConfig
from .wiring import Wiring

__all__ = [
    "ActorCritic",
    "DiscreteCode",
    "DreamActorCritic",
    "DreamConfig",
    "actor_critic_wiring",
    "ActorCriticConfig",
    "Bins",
    "ValueConfig",
    "ValueNet",
    "Population",
    "Rehearsal",
    "RehearsalConfig",
    "Seams",
    "SleepConfig",
    "Adaptation",
    "Region",
    "Projection",
    "Constitution",
    "Echo",
    "GradedRule",
    "Learner",
    "LearnerConfig",
    "Ledger",
    "Nudge",
    "Protocol",
    "Receipt",
    "Row",
    "Settlement",
    "SettledState",
    "Source",
    "Wiring",
    "available_backends",
    "canonical_json",
    "canonical_sha256",
    "conformance",
    "embedded",
    "mutate",
    "grow",
    "evolve",
    "evaluate_predicate",
    "fetch",
    "layered",
    "learning_rule",
    "load",
    "manifest",
    "save",
    "select_gain",
    "settle_owner_by_owner",
    "shuffled",
    "source_manifest",
    "stateful",
]

__version__ = "0.7.0"
