"""Experimental brains built from local state, memory and repair.

Use ``Brain.compose`` for the default System 1 brain, adding ``observers`` for
optional System 2 feedback in the same neural graph. ``NeuralGraph`` exposes
lower-level neuronal dynamics. Record, belief and temporal models provide
specialized memory, inference and planning operations with their own contracts.
See ``docs/contracts.md`` for their numerical and learning boundaries.
"""

from __future__ import annotations

from . import regions
from .arousal import Arousal, ArousalConfig
from .belief import BeliefObservation, BeliefPatch, BeliefPath, BeliefReadback
from .brain import Brain as NeuralGraph
from .brain import BrainState, Equilibrium, Nudge, RefinementReport, available_backends
from .certificate import (
    Certificate,
    EPStructure,
    certificate,
    ep_structure,
    lipschitz_constant,
    row_mass,
)
from .checkpoint import load, save
from .connectome import Connectome
from .generic import Brain
from .genome import Genome, Projection, develop, evolve, genes
from .instruments import dishabituation, orienting
from .learning import (
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
from .life import (
    AlwaysAwake,
    Decision,
    Governor,
    Life,
    LifeConfig,
    NeverWakes,
    PatchGovernor,
    Signals,
    ThresholdGovernor,
)
from .memory import SynapticMemory
from .neuron import Adaptation, NeuronModel
from .nudged_settle import NudgedSettle
from .patch import PatchNet, PatchObservation
from .planning import TemporalPlan
from .plasticity import (
    ActorCritic,
    ActorCriticConfig,
    Bins,
    Valence,
)
from .ports import DenseBlock, MapBlock, StructuredPort
from .protocol import Protocol, Row, evaluate_predicate, select_gain, shuffled
from .receipts import Receipt, canonical_json
from .record_patch import (
    RecordContrast,
    RecordObservation,
    RecordPatchNet,
    RecordPath,
    RecordReadback,
)
from .record_ports import JointObservation, JointRecordPatches, Port
from .record_stack import RecordPatchStack, StackObservation
from .recording import SettlementRecord, record_settlements
from .records import Mulberry32, Records
from .reference import conformance
from .regions import Region
from .steering import Boundary, Gaze, Rule, Softmax, Steered, SteeredPath
from .stream import Afterglow, Echo, Efference, FastSynapses, PatternSeparator, Trace, stateful
from .temporal import TemporalObservation, TemporalPatchNet, TemporalPhase, TemporalReadback
from .temporal_memory import ConstraintReport, TemporalMemory

__all__ = [
    "Arousal",
    "ArousalConfig",
    "BeliefObservation",
    "BeliefPatch",
    "BeliefPath",
    "BeliefReadback",
    "Steered",
    "SteeredPath",
    "Life",
    "LifeConfig",
    "Decision",
    "Signals",
    "Governor",
    "PatchGovernor",
    "ThresholdGovernor",
    "AlwaysAwake",
    "NeverWakes",
    "orienting",
    "dishabituation",
    "Boundary",
    "Softmax",
    "Gaze",
    "Rule",
    "DenseBlock",
    "MapBlock",
    "StructuredPort",
    "ConstraintReport",
    "TemporalMemory",
    "TemporalPlan",
    "RecordPatchNet",
    "RecordPatchStack",
    "StackObservation",
    "RecordPath",
    "RecordObservation",
    "RecordContrast",
    "RecordReadback",
    "TemporalPatchNet",
    "TemporalPhase",
    "TemporalObservation",
    "TemporalReadback",
    "PatchNet",
    "PatchObservation",
    "Certificate",
    "EPStructure",
    "certificate",
    "ep_structure",
    "lipschitz_constant",
    "row_mass",
    "PatternSeparator",
    "Records",
    "Mulberry32",
    "ActorCritic",
    "ActorCriticConfig",
    "Valence",
    "Bins",
    "Adaptation",
    "Region",
    "Brain",
    "regions",
    "Projection",
    "Genome",
    "Echo",
    "Afterglow",
    "Trace",
    "Efference",
    "FastSynapses",
    "SynapticMemory",
    "NeuronModel",
    "Learner",
    "LearnerConfig",
    "LearningPhaseError",
    "NudgedSettle",
    "Nudge",
    "Protocol",
    "Receipt",
    "SettlementRecord",
    "record_settlements",
    "JointRecordPatches",
    "JointObservation",
    "Port",
    "Row",
    "NeuralGraph",
    "BrainState",
    "Equilibrium",
    "RefinementReport",
    "Connectome",
    "available_backends",
    "canonical_json",
    "conformance",
    "embedded",
    "develop",
    "evolve",
    "genes",
    "evaluate_predicate",
    "layered",
    "learning_neuron_model",
    "calibrate_bias",
    "naive_efficacy",
    "preflight",
    "seam_report",
    "load",
    "save",
    "select_gain",
    "shuffled",
    "stateful",
]

__version__ = "0.78.0"
