"""Checkpoints: a trained learner to one file and back, for deployment.

``save`` writes a single ``.npz`` holding the wiring (owners, overlaps, contacts, signs, sets),
the settlement's parameters (every seam's scale, every owner's gain and bias), the rule, the
learner's configuration, masks, tie groups, momentum and normalisation state, and the update
count, plus the library version that wrote it. ``load`` rebuilds a ``Learner`` on any backend,
so a net trained on an accelerator runs on a CPU in a body and keeps learning where it left off.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .learning import Learner, LearnerConfig
from .rules import Adaptation, GradedRule
from .settle import Backend, Settlement
from .wiring import Wiring

FORMAT = "cadence-checkpoint/1"

__all__ = ["FORMAT", "load", "save"]


def save(learner: Learner, path: str | Path) -> Path:
    """Write ``learner`` to ``path`` (``.npz``); returns the path written."""
    from . import __version__

    engine = learner.engine
    w = engine.wiring
    meta = {
        "format": FORMAT,
        "version": __version__,
        "n": int(w.n),
        "label": w.label,
        "sets": {k: [int(i) for i in v] for k, v in w.sets.items()},
        "rule": engine.rule.to_dict(),
        "config": learner.config.to_dict(),
        "symmetric": bool(learner.symmetric),
        "updates": int(learner.updates),
        "backend": engine.backend,
        "dense_limit": int(engine.dense_limit),
    }
    assert learner.trainable_overlaps is not None and learner.trainable_owners is not None
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=np.array(json.dumps(meta, sort_keys=True)),
        pre=w.pre,
        post=w.post,
        count=w.count,
        sign=w.sign,
        edge_scale=engine.edge_scale,
        log_gain=engine.log_gain,
        bias=engine.bias,
        outputs=learner.output_index,
        trainable_overlaps=learner.trainable_overlaps,
        trainable_owners=learner.trainable_owners,
        tie_groups=learner.tie_groups if learner.tie_groups is not None else np.zeros(0, np.int64),
        velocity=learner.velocity,
        velocity_bias=learner.velocity_bias,
        second_moment=learner.second_moment,
        second_moment_bias=learner.second_moment_bias,
    )
    return path


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
    or not at all: set ``eta`` and ``eta_bias`` to zero and the net only settles).
    """
    with np.load(Path(path), allow_pickle=False) as data:
        meta: dict[str, Any] = json.loads(str(data["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"not a cadence checkpoint: {meta.get('format')!r}")
        wiring = Wiring(
            n=int(meta["n"]),
            pre=data["pre"],
            post=data["post"],
            count=data["count"],
            sign=data["sign"],
            sets={k: tuple(v) for k, v in meta["sets"].items()},
            label=str(meta["label"]),
        )
        rule_dict = dict(meta["rule"])
        adaptation = rule_dict.pop("adaptation", None)
        rule = GradedRule(**rule_dict, adaptation=Adaptation(**adaptation) if adaptation else None)
        engine = Settlement(
            wiring,
            rule,
            backend=backend or meta["backend"],
            edge_scale=data["edge_scale"],
            log_gain=data["log_gain"],
            bias=data["bias"],
            device=device,
            dense_limit=int(meta["dense_limit"]),
            precision=precision,
        )
        tie = data["tie_groups"]
        learner = Learner(
            engine,
            [int(i) for i in data["outputs"]],
            config or LearnerConfig(**meta["config"]),
            trainable_overlaps=data["trainable_overlaps"].astype(bool),
            trainable_owners=data["trainable_owners"].astype(bool),
            symmetric=bool(meta["symmetric"]),
            tie_groups=tie if len(tie) else None,
            updates=int(meta["updates"]),
        )
        learner.velocity = data["velocity"].astype(float)
        learner.velocity_bias = data["velocity_bias"].astype(float)
        learner.second_moment = data["second_moment"].astype(float)
        learner.second_moment_bias = data["second_moment_bias"].astype(float)
    return learner
