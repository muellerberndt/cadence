"""Checkpoints: a trained learner to one file and back, for deployment.

``save`` writes a single ``.npz`` holding the connectome (neurons, synapses, contacts, signs,
populations), the brain's parameters (every synapse's efficacy, every neuron's gain and bias),
the neuron model, the learner's configuration, plasticity masks, tie groups, momentum and
normalisation state, and the update count, plus the library version that wrote it. ``load``
rebuilds a ``Learner`` on any backend, so a brain trained on an accelerator runs on a CPU in a
body and keeps learning where it left off. Checkpoints in the first format load as well.
These are learner checkpoints; a complete composed life uses ``Brain.save/load`` instead.
When changing machine, backend or precision, check numerical and behavioural parity on the
intended workload. Checkpoint portability does not guarantee identical cross-machine results.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .brain import Apical, Backend, Brain
from .connectome import Connectome
from .learning import Learner, LearnerConfig
from .neuron import Adaptation, NeuronModel
from .receipts import canonical_json

FORMAT = "cadence-checkpoint/2"
FIRST_FORMAT = "cadence-checkpoint/1"
# A learner saved with the readout's intrinsic plasticity on (homeostasis_rate above 0).
# This version has no such setting, so such a file is refused instead of loaded without it.
HOMEOSTASIS_FORMAT = "cadence-checkpoint/3"
# Non-default numerical policy; older loaders must refuse instead of dropping it.
INITIAL_DAMPING_FORMAT = "cadence-checkpoint/4"
_INITIAL_DAMPING_POLICY = "initial-damping/1"
# A graph with an apical compartment, or a learner with the settings that came with it
# (``normalize_relative``, ``nudged_fixed``): older loaders must refuse it instead of settling or
# teaching the brain without them. Carries the initial-damping policy too when it has one.
APICAL_FORMAT = "cadence-checkpoint/5"

__all__ = ["FORMAT", "load", "save"]

# The first format's names for the entries that the second format renamed.
_FIRST_META = {"populations": "sets", "neuron_model": "rule", "reciprocal": "symmetric"}
_FIRST_ARRAYS = {
    "efficacy": "edge_scale",
    "plastic_synapses": "trainable_overlaps",
    "plastic_neurons": "trainable_owners",
}
_FIRST_MODEL: dict[str, str] = {"clamp_amplitude": "stimulus_amplitude"}


def _learner_data(learner: Learner) -> dict[str, np.ndarray]:
    from . import __version__

    brain = learner.brain
    if learner.config.initial_damping and (
        brain.backend != "cpu" or brain.precision not in (None, "float64")
    ):
        raise ValueError("nonzero initial_damping requires CPU float64")
    config = learner.config.to_dict()
    if not learner.config.initial_damping:
        # A neutral new setting must not change ordinary checkpoint metadata.
        config.pop("initial_damping")
    c = brain.connectome
    meta = {
        "format": INITIAL_DAMPING_FORMAT if learner.config.initial_damping else FORMAT,
        "version": __version__,
        "n": int(c.n),
        "label": c.label,
        "populations": {k: [int(i) for i in v] for k, v in c.populations.items()},
        "neuron_model": brain.neuron_model.to_dict(),
        "config": config,
        "reciprocal": bool(learner.reciprocal),
        "slots": [int(k) for k in learner.slot_sizes],
        "updates": int(learner.updates),
        "contrast_updates": int(learner.contrast_updates),
        "backend": brain.backend,
        "precision": brain.precision,
        "dense_limit": int(brain.dense_limit),
    }
    if learner.config.initial_damping:
        meta["numerical_policy"] = {
            "kind": _INITIAL_DAMPING_POLICY,
            "initial_damping": learner.config.initial_damping,
        }
    compartment: dict[str, np.ndarray] = {}
    if learner.config.normalize_relative or learner.config.nudged_fixed:
        meta["format"] = APICAL_FORMAT
    if brain.apical is not None:
        meta["format"] = APICAL_FORMAT
        meta["apical"] = {"fraction": brain.apical.fraction, "floor": brain.apical.floor}
        compartment = {
            "apical_synapses": brain.apical.synapses,
            "apical_groups": brain.apical.groups,
        }
    assert learner.plastic_synapses is not None and learner.plastic_neurons is not None
    return dict(
        **compartment,
        meta=np.array(canonical_json(meta)),
        pre=c.pre,
        post=c.post,
        count=c.count,
        sign=c.sign,
        efficacy=brain.efficacy,
        log_gain=brain.log_gain,
        bias=brain.bias,
        outputs=learner.output_index,
        plastic_synapses=learner.plastic_synapses,
        plastic_neurons=learner.plastic_neurons,
        tie_groups=learner.tie_groups if learner.tie_groups is not None else np.zeros(0, np.int64),
        synapse_rate=learner.synapse_rate if learner.synapse_rate is not None else np.zeros(0),
        velocity=learner.velocity,
        velocity_bias=learner.velocity_bias,
        second_moment=learner.second_moment,
        second_moment_bias=learner.second_moment_bias,
    )


def _write(data: dict[str, Any], path: str | Path, *, compressed: bool = True) -> Path:
    """Replace only after a complete archive has been written, keeping the previous checkpoint.

    ``compressed=False`` writes a plain archive: float parameters compress little, and a
    brain of tens of millions of synapses takes minutes to deflate but seconds to write."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            (np.savez_compressed if compressed else np.savez)(handle, **data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def save(learner: Learner, path: str | Path, *, compressed: bool = True) -> Path:
    """Atomically write ``learner`` to ``path`` (``.npz``); returns the path written."""
    return _write(_learner_data(learner), path, compressed=compressed)


# Settings of 0.84.3 that this version removed, with the value at which a saved config's law
# is the law of this version: the contrast of +beta against -beta, and no leak.
_REMOVED = {"centered": True, "decay": 0.0}


def _known_config(saved: dict[str, Any]) -> dict[str, Any]:
    """The saved config's fields that ``LearnerConfig`` has; any other saved field is dropped
    silently, so a checkpoint carrying settings this version lacks loads without them. A
    setting of 0.84.3 that this version removed loads at the value its law has here and is
    refused otherwise, with a message naming 0.84.3."""
    for name, kept in _REMOVED.items():
        if name in saved:
            value = saved[name]
            if isinstance(value, bool) != isinstance(kept, bool) or value != kept:
                raise ValueError(
                    f"this checkpoint was saved with the learner's {name}={value!r}, which "
                    "this version removed; load it with cadence-net 0.84.3"
                )
    fields = {f.name for f in dataclasses.fields(LearnerConfig)}
    return {k: v for k, v in saved.items() if k in fields}


def load(
    path: str | Path,
    *,
    backend: Backend | None = None,
    device: str | None = None,
    config: LearnerConfig | None = None,
    precision: str | None = None,
) -> Learner:
    """Rebuild a learner from a checkpoint; ``backend`` and ``device`` may differ from the saved.

    ``config`` replaces the saved learner configuration (a deployment may learn at another rate,
    or not at all: set ``eta`` and ``eta_bias`` to zero and the brain only settles).
    """
    with np.load(Path(path), allow_pickle=False) as data:
        meta: dict[str, Any] = json.loads(str(data["meta"]))
        first = meta.get("format") == FIRST_FORMAT
        if meta.get("format") == HOMEOSTASIS_FORMAT:
            raise ValueError(
                "this checkpoint was saved by cadence-net 0.81 with the readout's intrinsic "
                "plasticity (homeostasis_rate above 0), which 0.82.0 removed; load it with 0.81"
            )
        apical_format = meta.get("format") == APICAL_FORMAT
        initial_policy = meta.get("format") == INITIAL_DAMPING_FORMAT or (
            apical_format and "numerical_policy" in meta
        )
        known = (FORMAT, INITIAL_DAMPING_FORMAT, APICAL_FORMAT)
        if not first and meta.get("format") not in known:
            raise ValueError(f"not a cadence checkpoint: {meta.get('format')!r}")
        compartment_arrays = {"apical_synapses", "apical_groups"} & set(data.files)
        if (
            ("apical" in meta and compartment_arrays != {"apical_synapses", "apical_groups"})
            or ("apical" not in meta and compartment_arrays)
            or ("apical" in meta and not apical_format)
        ):
            raise ValueError(
                "an apical compartment belongs to checkpoint format cadence-checkpoint/5"
            )
        depth_settings = bool(
            meta["config"].get("normalize_relative") or meta["config"].get("nudged_fixed")
        )
        if apical_format and "apical" not in meta and not depth_settings:
            raise ValueError(
                "a cadence-checkpoint/5 file holds an apical compartment or the learner's "
                "normalize_relative or nudged_fixed"
            )
        if depth_settings and not apical_format:
            raise ValueError(
                "normalize_relative and nudged_fixed belong to checkpoint format "
                "cadence-checkpoint/5"
            )
        if initial_policy:
            saved_config = LearnerConfig(**_known_config(meta["config"]))
            if not saved_config.initial_damping or meta.get("numerical_policy") != {
                "kind": _INITIAL_DAMPING_POLICY,
                "initial_damping": saved_config.initial_damping,
            }:
                raise ValueError("invalid initial-damping checkpoint policy")
        elif meta["config"].get("initial_damping", 0) != 0 or "numerical_policy" in meta:
            raise ValueError("initial_damping requires checkpoint format cadence-checkpoint/4")
        else:
            # An ordinary checkpoint keeps the documented complete-config override,
            # including when the saved config contains unsupported old settings.
            saved_config = config or LearnerConfig(**_known_config(meta["config"]))
        effective_config = config or saved_config
        target_backend = backend or meta["backend"]
        target_precision = precision if precision is not None else meta.get("precision")
        if effective_config.initial_damping and (
            target_backend != "cpu" or target_precision not in (None, "float64")
        ):
            raise ValueError("nonzero initial_damping requires CPU float64")

        def entry(name: str) -> Any:
            return meta[_FIRST_META[name] if first else name]

        def array(name: str) -> np.ndarray:
            return np.asarray(data[_FIRST_ARRAYS[name] if first else name])

        connectome = Connectome(
            n=int(meta["n"]),
            pre=data["pre"],
            post=data["post"],
            count=data["count"],
            sign=data["sign"],
            populations={k: tuple(v) for k, v in entry("populations").items()},
            label=str(meta["label"]),
        )
        saved_model: dict[str, Any] = dict(entry("neuron_model"))
        model = {(_FIRST_MODEL[k] if k in _FIRST_MODEL else k): v for k, v in saved_model.items()}
        adaptation = model.pop("adaptation", None)
        neuron_model = NeuronModel(
            **model, adaptation=Adaptation(**adaptation) if adaptation else None
        )
        apical = None
        if "apical" in meta:
            spec = meta["apical"]
            if not isinstance(spec, dict) or set(spec) != {"fraction", "floor"}:
                raise ValueError("invalid checkpoint apical compartment")
            apical = Apical(
                np.asarray(data["apical_synapses"]),
                np.asarray(data["apical_groups"]),
                spec["fraction"],
                spec["floor"],
            )
        brain = Brain(
            connectome,
            neuron_model,
            backend=target_backend,
            efficacy=array("efficacy"),
            log_gain=data["log_gain"],
            bias=data["bias"],
            device=device,
            dense_limit=int(meta["dense_limit"]),
            precision=target_precision,
            apical=apical,
        )
        tie = data["tie_groups"]
        rate = data["synapse_rate"] if "synapse_rate" in data.files else np.zeros(0)
        learner = Learner(
            brain,
            data["outputs"].tolist(),
            effective_config,
            plastic_synapses=array("plastic_synapses"),
            plastic_neurons=array("plastic_neurons"),
            reciprocal=bool(entry("reciprocal")),
            tie_groups=tie if len(tie) else None,
            synapse_rate=rate.astype(float) if len(rate) else None,
            slots=meta.get("slots", 1),
            updates=int(meta["updates"]),
            contrast_updates=int(meta.get("contrast_updates", meta["updates"])),
        )
        if learner.updates < 0 or not 0 <= learner.contrast_updates <= learner.updates:
            raise ValueError("checkpoint update count must be nonnegative")
        for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
            value = data[name].astype(float)
            size = connectome.n if name.endswith("bias") else connectome.synapses
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"invalid checkpoint optimizer array: {name}")
            if name.startswith("second_moment") and (value < 0).any():
                raise ValueError(f"negative checkpoint second moment: {name}")
            setattr(learner, name, value)
    return learner
