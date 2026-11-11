"""Performance admission charges unequal phases and fails on parameter mismatch."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

numba = pytest.importorskip("numba")
PATH = Path(__file__).resolve().parents[1] / "benchmarks/settle_threads.py"
spec = importlib.util.spec_from_file_location("thread_benchmark", PATH)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def test_phase_work_counts_the_actual_opposite_phase(monkeypatch):
    brain = SimpleNamespace(
        lesson=lambda x, y: dict(free_steps=8, nudged_steps=6, opposite_steps=4),
        brain=SimpleNamespace(efficacy=np.ones(2), bias=np.zeros(3)),
        connectome=SimpleNamespace(n=3, synapses=2),
        learner=SimpleNamespace(velocity=np.zeros(2), velocity_bias=np.zeros(3),
                                second_moment=np.zeros(2), second_moment_bias=np.zeros(3),
                                updates=1, contrast_updates=1),
    )
    monkeypatch.setattr(subject, "compose", lambda *args: brain)
    args = SimpleNamespace(inputs=1, modules=(2,), classes=2, warmup=0, lessons=1,
                           batch=1, split=False)
    assert subject.run("flat", 1, args)["sweeps_per_lesson"] == 18


def test_cli_exits_nonzero_after_writing_a_failed_comparison(tmp_path, monkeypatch):
    if numba.config.NUMBA_NUM_THREADS < 2:
        pytest.skip("the CLI comparison needs two permitted thread counts")
    def run(kind, threads, args):
        return dict(neurons=3, synapses=2, seconds_min=1.0, seconds_median=1.0,
                    sweeps_per_lesson=18, weights=np.array([float(threads)]), bias=np.zeros(1),
                    optimizer={"velocity": np.zeros(1)}, phases=[[8, 6, 4]], updates=1,
                    contrast_updates=1)

    path = tmp_path / "comparison.json"
    monkeypatch.setattr(subject, "run", run)
    monkeypatch.setattr("sys.argv", [str(PATH), "--threads", "1", "2", "--out", str(path)])
    with pytest.raises(SystemExit) as error:
        subject.main()
    assert error.value.code == 1
    assert '"identical": false' in path.read_text()
