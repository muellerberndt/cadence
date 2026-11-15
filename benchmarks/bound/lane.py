"""The meaning chamber, read from its own branch (cadence#334).

The instruments of this chamber drive the brain of the meaning chamber, which lives on the
branch ``lane-0100-meaning`` (its chamber under ``benchmarks/meaning``, its senses and its
genome under ``src/cadence/meaning``). A checkout of that branch is named by the environment
variable ``CADENCE_MEANING_LANE``, or lies beside this repository as ``wt-meaning``:

    git worktree add ../wt-meaning origin/lane-0100-meaning

The library itself is this branch's: only the package ``cadence.meaning`` and the chamber's
modules (``night``, ``introspect``, ``demo_stage``, ``pretrain_stage``, ``export_page``) are read
from the checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANE = Path(os.environ.get("CADENCE_MEANING_LANE", HERE.parents[2] / "wt-meaning"))
if not (LANE / "benchmarks" / "meaning" / "night.py").exists():
    raise ImportError(f"no checkout of lane-0100-meaning at {LANE}: set CADENCE_MEANING_LANE")

import cadence  # noqa: E402

if str(LANE / "src" / "cadence") not in cadence.__path__:
    cadence.__path__.append(str(LANE / "src" / "cadence"))  # cadence.meaning, from the lane
for folder in (LANE / "benchmarks" / "meaning", HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import demo_stage  # noqa: E402
import export_page  # noqa: E402
import introspect  # noqa: E402
import night  # noqa: E402
import pretrain_stage  # noqa: E402

SEED = 33400121  # the founder of the equal-senses brain
GENOME = {"mixed_code": 1024, "mixed_on": .1, "mixed_decay": .9, "mixed_balance": 1,
          "records_key": "input", "depth": 4, "apical": 0.5, "word_records": 0}

__all__ = ["GENOME", "LANE", "SEED", "demo_stage", "export_page", "introspect", "night",
           "pretrain_stage"]
