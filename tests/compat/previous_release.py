"""The files this version saves, read by the previous release.

    python tests/compat/previous_release.py save DIR    # with this checkout installed
    python tests/compat/previous_release.py load DIR    # with the previous release installed

``save`` writes every brain of the released-files panel (``tests/fixtures/released/panel.py``,
brains without this version's new options) after its life, with each brain's continuation
after a load in this version, and a brain whose cortex reads the senses beside its code
(``cortex_reads="both"``, 0.102.0) that the previous release refuses. ``load`` reads them under
the previous release: each brain loads and continues with the same actions and the same weights
to the bit, and the brain with ``cortex_reads`` is refused by its feature's name. CI runs both
on one machine on the NumPy loop (``CADENCE_FUSED=0``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

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
    reads = cd.Brain.compose(inputs=40, actions=2, code=64, cortex_reads="both", modules=(8,),
                             seed=0, backend="cpu")
    reads.act(np.full((1, 40), 0.5), greedy=True)
    reads.save(directory / "refused-cortex_reads.npz")
    (directory / "saved.json").write_text(json.dumps(table, indent=1))
    print(f"cadence {cd.__version__} saved {len(table['brains'])} brains and a brain that "
          "reads both")


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
    try:
        cd.Brain.load(directory / "refused-cortex_reads.npz")
    except ValueError as error:
        if "'cortex_reads'" not in str(error):
            failures.append(
                f"the brain that reads both is refused without its feature's name: {error}")
    else:
        failures.append("the brain that reads both loads in a release without cortex_reads")
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"cadence {cd.__version__} read {len(table['brains'])} brains of cadence "
        f"{table['version']} and refused the brain that reads both by name"
    )


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    {"save": save, "load": load}[sys.argv[1]](Path(sys.argv[2]))
