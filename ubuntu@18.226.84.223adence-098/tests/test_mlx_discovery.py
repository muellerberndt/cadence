"""MLX discovery runs in a child interpreter (#105): an MLX whose import aborts the process, as an
unsupported Metal or system configuration can, leaves the caller running, absent from
``available_backends()``, and ``backend="mlx"`` refuses with the child's reason. The child sees
the caller's import path and nothing else; an MLX already imported, or a frozen application,
answers in the caller."""

from __future__ import annotations

import importlib.util
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
    except ImportError as exc:
        print("refused", type(exc).__name__, exc)
    """
)

# a working MLX as far as discovery asks: its default device; every import leaves a line in a log
WORKING = textwrap.dedent(
    """
    import os, sys
    if "PROBE_LOG" in os.environ:
        with open(os.environ["PROBE_LOG"], "a") as log:
            log.write(f"imported in pid {os.getpid()}\\n")
    DEVICE = "gpu"
    class _Type:
        @property
        def name(self):
            return DEVICE
    class _Device:
        type = _Type()
    def default_device():
        return _Device()
    def set_default_device(name):
        global DEVICE
        DEVICE = name
    """
)


def fake_mlx(root: Path, core: str) -> Path:
    package = root / "mlx"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "core.py").write_text(core)
    return root


def run(
    program: str,
    *paths: Path,
    cwd: Path | None = None,
    script: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """``program`` in a fresh interpreter with ``paths`` and this checkout's source before the
    caller's ``PYTHONPATH``, as ``-c`` or, with ``script``, from that file."""
    path = [str(p) for p in paths] + [str(SRC)] + [os.environ.get("PYTHONPATH", "")]
    environment = dict(os.environ, **(env or {}))
    environment["PYTHONPATH"] = os.pathsep.join(p for p in path if p)
    if script is not None:
        script.write_text(program)
    command = [sys.executable, str(script)] if script else [sys.executable, "-c", program]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        cwd=cwd,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize(
    "core, reason",
    [
        ("import os\nos.abort()\n", "exit"),
        ("raise ImportError('no Metal device')\n", "no Metal device"),
        # a failing native library's message in another encoding
        ("import os\nos.write(2, b'Metal: \\xe9chec\\n')\nos._exit(1)\n", "Metal: �chec"),
    ],
)
def test_an_mlx_that_fails_on_import_leaves_the_caller_running(tmp_path, core, reason):
    done = run(PROGRAM, fake_mlx(tmp_path / "fake", core))
    assert done.returncode == 0, done.stderr
    assert "backends ['cpu'" in done.stdout and "'mlx'" not in done.stdout
    assert "refused ImportError the mlx backend is unavailable here" in done.stdout
    assert reason in done.stdout


def test_without_mlx_the_backend_refuses_as_a_missing_module(tmp_path):
    hidden = textwrap.dedent(
        """
        import importlib.util
        found = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *a: None if name == "mlx" else found(name, *a)
        """
    )
    done = run(hidden + PROGRAM)
    assert done.returncode == 0, done.stderr
    assert "refused ModuleNotFoundError the mlx backend is unavailable here" in done.stdout
    assert "mlx is not installed" in done.stdout


def test_the_child_imports_from_the_callers_path_added_at_run_time(tmp_path):
    fake = fake_mlx(tmp_path / "later", WORKING)
    program = f"import sys; sys.path.append({str(fake)!r})\n" + textwrap.dedent(
        """
        import cadence as cd
        print("backends", cd.available_backends())
        """
    )
    done = run(program)
    assert done.returncode == 0, done.stderr
    assert "'mlx': 'gpu float32'" in done.stdout


def test_the_child_does_not_import_from_its_working_directory(tmp_path):
    """A script's imports exclude the directory it runs in; the child's do too."""
    planted = tmp_path / "planted"
    fake_mlx(planted, "import os\nos.abort()\n")
    (planted / "mlx" / "__init__.py").write_text(
        "open(__file__ + '.ran', 'w').close()\n"
    )
    program = "import cadence as cd\nprint('backends', cd.available_backends())\n"
    (tmp_path / "program").mkdir()
    done = run(
        program,
        fake_mlx(tmp_path / "working", WORKING),
        cwd=planted,
        script=tmp_path / "program" / "probe.py",
    )
    assert done.returncode == 0, done.stderr
    assert "'mlx': 'gpu float32'" in done.stdout
    assert not (planted / "mlx" / "__init__.py.ran").exists()


@pytest.mark.parametrize(
    "setting",
    [
        "sys.frozen = True; sys.executable = '/nonexistent/app'",
        "sys.executable = None",
        "sys.executable = ''",
    ],
)
def test_a_frozen_application_imports_mlx_in_process(tmp_path, setting):
    """A frozen application's executable is the application itself: discovery stays in it.
    The program marks itself frozen after its imports, which read ``sys.frozen`` themselves."""
    program = f"import sys\nimport cadence as cd\n{setting}\n" + textwrap.dedent(
        """
        print("backends", cd.available_backends())
        print("imported", "mlx.core" in sys.modules)
        """
    )
    done = run(program, fake_mlx(tmp_path / "fake", WORKING))
    assert done.returncode == 0, done.stderr
    assert "'mlx': 'gpu float32'" in done.stdout and "imported True" in done.stdout


def test_an_imported_mlx_reports_its_present_default_device(tmp_path):
    program = textwrap.dedent(
        """
        import sys
        import mlx.core as mx
        mx.set_default_device("cpu")
        sys.executable = "/nonexistent/python"  # no child is asked
        import cadence as cd
        print("backends", cd.available_backends())
        """
    )
    done = run(program, fake_mlx(tmp_path / "fake", WORKING))
    assert done.returncode == 0, done.stderr
    assert "'mlx': 'cpu float32'" in done.stdout


def test_discovery_runs_once_for_concurrent_first_calls(tmp_path):
    log = tmp_path / "probe.log"
    program = textwrap.dedent(
        """
        import threading
        import cadence as cd
        found = []
        calls = [threading.Thread(target=lambda: found.append(cd.available_backends()))
                 for _ in range(8)]
        for call in calls:
            call.start()
        for call in calls:
            call.join()
        print("found", sorted({str(backends) for backends in found}))
        """
    )
    done = run(program, fake_mlx(tmp_path / "fake", WORKING), env={"PROBE_LOG": str(log)})
    assert done.returncode == 0, done.stderr
    assert "'mlx': 'gpu float32'" in done.stdout
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.skipif(
    importlib.util.find_spec("mlx") is not None, reason="needs an interpreter without MLX"
)
def test_an_mlx_installed_later_in_the_process_is_found(tmp_path):
    fake = fake_mlx(tmp_path / "later", WORKING)
    program = textwrap.dedent(
        f"""
        import importlib, sys
        import cadence as cd
        print("before", "mlx" in cd.available_backends())
        sys.path.append({str(fake)!r})
        importlib.invalidate_caches()
        print("after", cd.available_backends().get("mlx"))
        """
    )
    done = run(program)
    assert done.returncode == 0, done.stderr
    assert "before False" in done.stdout and "after gpu float32" in done.stdout


def test_a_child_that_crashes_at_its_exit_still_answers(tmp_path):
    crashing = "import atexit, os\natexit.register(os.abort)\n" + WORKING
    program = "import cadence as cd\nprint('backends', cd.available_backends())\n"
    done = run(program, fake_mlx(tmp_path / "fake", crashing))
    assert done.returncode == 0, done.stderr
    assert "'mlx': 'gpu float32'" in done.stdout
