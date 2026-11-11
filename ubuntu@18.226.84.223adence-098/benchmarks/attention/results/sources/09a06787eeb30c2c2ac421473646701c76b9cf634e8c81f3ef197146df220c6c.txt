"""Experimental brains built from local state, memory and repair.

Use ``Brain.compose`` for the default System 1 brain, adding ``observers`` for
optional System 2 feedback in the same neural graph. ``NeuralGraph`` exposes
lower-level neuronal dynamics. See ``docs/contracts.md`` for the numerical and
learning boundaries.
"""

from __future__ import annotations

from . import regions
from .arousal import Arousal, ArousalConfig
from .brain import Brain as NeuralGraph
from .brain import BrainState, Equilibrium, Nudge, available_backends
from .certificate import (
    Certificate,
    EPStructure,
    certificate,
    ep_structure,
    lipschitz_constant,
    row_mass,
)
from .checkpoint import load, save
from .code import LearnedCode
from .connectome import Connectome
from .generic import Brain
from .genome import Genome, Projection, develop, evolve, genes
from .learning import (
    GraphHolder,
    Learner,
    LearnerConfig,
    LearningPhaseError,
    calibrate_bias,
    embedded,
    layered,
    learning_neuron_model,
    naive_efficacy,
    preflight,
    seam_report,
)
from .memory import SynapticMemory
from .neuron import Adaptation, NeuronModel
from .plasticity import ActorCritic, ActorCriticConfig
from .receipts import Receipt, canonical_json
from .recording import SettlementRecord, record_settlements
from .reference import conformance
from .regions import Region
from .stream import (
    Efference,
    FastSynapses,
    History,
    PatternSeparator,
    Recall,
    Trace,
)

__all__ = [
    "Arousal",
    "ArousalConfig",
    "Certificate",
    "EPStructure",
    "certificate",
    "ep_structure",
    "lipschitz_constant",
    "row_mass",
    "PatternSeparator",
    "ActorCritic",
    "ActorCriticConfig",
    "Adaptation",
    "Region",
    "Brain",
    "regions",
    "Projection",
    "Genome",
    "Trace",
    "Efference",
    "History",
    "Recall",
    "LearnedCode",
    "FastSynapses",
    "SynapticMemory",
    "NeuronModel",
    "GraphHolder",
    "Learner",
    "LearnerConfig",
    "LearningPhaseError",
    "Nudge",
    "Receipt",
    "SettlementRecord",
    "record_settlements",
    "NeuralGraph",
    "BrainState",
    "Equilibrium",
    "Connectome",
    "available_backends",
    "canonical_json",
    "conformance",
    "embedded",
    "develop",
    "evolve",
    "genes",
    "layered",
    "learning_neuron_model",
    "calibrate_bias",
    "naive_efficacy",
    "preflight",
    "seam_report",
    "load",
    "save",
]

__version__ = "0.85.1"
