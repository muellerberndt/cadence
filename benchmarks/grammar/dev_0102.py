"""Development reading for 0.102.0 (#308): the role-by-choice store (``sentence.py``'s design
``chosen``) on four development founders, and where the recall read lands on a path's hops with
and without the bar that empties a path's start at ``GO``.

    SENTENCE_BACKEND=mlx PYTHONPATH=src python benchmarks/grammar/dev_0102.py OUT_DIR [batches]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import grammar as gr  # noqa: E402
import meaning as mn  # noqa: E402
import sentence as st  # noqa: E402


def main(out: Path, batches: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for founder in (0, 1, 2, 3):
        save = out / f"brain-{founder}"
        row = st.life({"kind": "store", "founder": founder, "batches": batches, "emulated": True,
                       "design": "chosen", "depth": 4, "held_probes": 4, "save": str(save),
                       "landing": True, "tag": "chosen"})
        (out / f"life-{founder}.json").write_text(json.dumps(row, default=str))
        print(json.dumps({"founder": founder, "panels": row.get("panels"),
                          "sayback": row.get("sayback"), "seconds": row.get("seconds")},
                         default=str), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else mn.BATCHES)
