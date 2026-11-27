"""The files this version saves, read by the previous release.

    python tests/compat/previous_release.py save DIR    # with this checkout installed
    python tests/compat/previous_release.py load DIR    # with the previous release installed

``save`` writes every brain of the released-files panel (``tests/fixtures/released/panel.py``)
after its life, with each brain's continuation after a load in this version. ``load`` reads
them under the previous release: each brain of the panel loads and continues with the same
actions and weights to the bit. Brains that use unit recall keys or optional learning credit
controls are refused by feature name, and a brain whose sensory neurons project beside its
learned code by the connectome the release rebuilds. CI runs both on one machine on the NumPy
loop (``CADENCE_FUSED=0``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import cadence as cd

PANEL = Path(__file__).resolve().parents[1] / "fixtures" / "released" / "panel.py"
_spec = importlib.util.spec_from_file_location("released_panel", PANEL)
assert _spec is not None and _spec.loader is not None
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)


def save(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    table = {"version": cd.__version__, "brains": {}}
    for name in sorted(panel.PANEL):
        path = directory / f"{name}.npz"
        brain = panel.compose(name)
        panel.life(brain, name)
        brain.save(path)
        loaded = cd.Brain.load(path)
        table["brains"][name] = {
            "acts": panel.continuation(loaded, name),
            "learned": panel.learned(loaded),
        }
    unit_keys = cd.Brain.compose(8, 3, modules=(8,), recall_key_norm=True, backend="cpu", seed=1)
    unit_keys.save(directory / "recall_key_norm.npz")
    credit = cd.Brain.compose(8, 3, modules=(8,), learning_drive_budget=4, backend="cpu", seed=1)
    credit.save(directory / "learning_credit.npz")
    credit.learner.save(directory / "credit_learner.npz")
    # a sensory projection beside a learned code names no feature: the file holds its connectome
    layout = cd.Genome.composed(8, 3, modules=(8,), learned_code=8, seed=1)
    arc = cd.Projection("sensory", "motor", reciprocal=False, scale=0.1)
    anatomy = cd.develop(cd.Genome(layout.regions, (*layout.projections, arc)), seed=1)
    beside = cd.Brain.compose(
        8, 3, connectome=anatomy, learned_code=8, recall=False, backend="cpu", seed=1
    )
    beside.save(directory / "sensory_projection.npz")
    (directory / "saved.json").write_text(json.dumps(table, indent=1))
    print(f"cadence {cd.__version__} saved {len(table['brains'])} brains")


def load(directory: Path) -> None:
    pinned = os.environ.get("PREVIOUS_RELEASE")
    if pinned and cd.__version__ != pinned:
        raise SystemExit(f"cadence {cd.__version__} is installed, the job pins {pinned}")
    table = json.loads((directory / "saved.json").read_text())
    failures = []
    for name, expected in sorted(table["brains"].items()):
        try:
            brain = cd.Brain.load(directory / f"{name}.npz")
            acts = panel.continuation(brain, name)
            learned = panel.learned(brain)
        except Exception as error:  # noqa: BLE001 - every failure is reported by brain
            failures.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if acts != expected["acts"]:
            failures.append(f"{name}: acts {acts} against {expected['acts']}")
        elif learned != expected["learned"]:
            failures.append(f"{name}: the weights after the continuation differ")
    for feature in ("recall_key_norm", "learning_credit"):
        try:
            cd.Brain.load(directory / f"{feature}.npz")
        except ValueError as error:
            if f"feature '{feature}'" not in str(error):
                failures.append(f"{feature}: not refused by feature name: {error}")
        else:
            failures.append(f"{feature}: previous release silently loaded an unsupported feature")
    try:
        cd.Brain.load(directory / "sensory_projection.npz")
    except ValueError as error:
        if "project to the code region alone" not in str(error):
            failures.append(f"sensory projection: not refused by its connectome: {error}")
    else:
        failures.append("sensory projection: previous release built a connectome it does not hold")
    try:
        cd.Learner.load(directory / "credit_learner.npz")
    except ValueError as error:
        if "cadence-checkpoint/6" not in str(error):
            failures.append(f"credit learner: not refused by format: {error}")
    else:
        failures.append("credit learner: previous release silently dropped learning settings")
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"cadence {cd.__version__} read {len(table['brains'])} brains of cadence "
        f"{table['version']}"
    )


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    {"save": save, "load": load}[sys.argv[1]](Path(sys.argv[2]))
