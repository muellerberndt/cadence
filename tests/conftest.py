"""The tests' own helper modules (``checkpoints``) import from this directory. A test names its
backend, or composes on the CPU: a brain the rule would place reads as on the machine it ran on."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("CADENCE_BACKEND", "cpu")
