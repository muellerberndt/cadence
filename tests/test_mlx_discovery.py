"""MLX discovery runs in a child interpreter (#105): an MLX whose import aborts the process, as an
unsupported Metal or system configuration can, leaves the caller running, absent from
``available_backends()``, and ``backend="mlx"`` refuses with the child's reason."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

PROGRAM = textwrap.dedent(
    """
    import numpy as np
    import cadence as cd

    print("backends", sorted(cd.available_backends()))
    connectome = cd.Connectome.from_synapses(2, pre=[0], post=[1], populations={"output": [1]})
    try:
        cd.NeuralGraph(connectome, cd.learning_neuron_model(), backend="mlx")
    except RuntimeError as exc:
        print("refused", exc)
    """
)


def fake_mlx(root: Path, core: str) -> Path:
    package = root / "mlx"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "core.py").write_text(core)
    return root


def run_with(fake: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(fake), str(SRC)]))
    return subprocess.run(
        [sys.executable, "-c", PROGRAM],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize(
    "core, reason",
    [
        ("import os\nos.abort()\n", "exit"),
        ("raise ImportError('no Metal device')\n", "no Metal device"),
    ],
)
def test_an_mlx_that_fails_on_import_leaves_the_caller_running(tmp_path, core, reason):
    done = run_with(fake_mlx(tmp_path / "fake", core))
    assert done.returncode == 0, done.stderr
    assert "backends ['cpu'" in done.stdout and "'mlx'" not in done.stdout
    assert "refused the mlx backend is unavailable here" in done.stdout
    assert reason in done.stdout
