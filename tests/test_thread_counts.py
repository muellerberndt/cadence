"""The thread counts the backends guide promises bits for (docs/backends.md): the compiled
kernel settles a batch on Numba's threads to the same bits at any count, and acts alike at any
BLAS thread count (its saved arrays can differ in the last bits on some machines, #315); the
recall memory's threaded products keep the summation of one
thread; and the NumPy loop at one BLAS thread repeats a life, its save and its continuation
after a load, to the bit. Each life runs in a child interpreter whose thread variables are set
before NumPy is imported."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cadence

SOURCE = str(Path(cadence.__file__).resolve().parents[1])

LIFE = r"""
import hashlib, json, sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("ignore")
import numpy as np
import cadence as cd
from cadence import brain as kernel

STREAMS, INPUTS = int(sys.argv[2]), 64
brain = cd.Brain.compose(INPUTS, 4, modules=(int(sys.argv[3]),), seed=5, backend="cpu")
rng = np.random.default_rng(9)

def moments(count):
    return [rng.random((STREAMS, INPUTS)) for _ in range(count)]

def digest(path):
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            value = data[name]
            if value.dtype.kind in "US":
                saved = json.loads(str(value))
                saved.pop("version", None) if isinstance(saved, dict) else None
                h.update(name.encode() + json.dumps(saved, sort_keys=True).encode())
            else:
                h.update(name.encode() + np.ascontiguousarray(value).tobytes())
    return h.hexdigest()

acts = []
xs = moments(8)
brain.lesson(xs[0], np.arange(STREAMS) % 4)
for i, x in enumerate(xs[1:]):
    acts.append(np.asarray(brain.step(x, reward=None if i == 0 else np.ones(STREAMS) * (i % 2)))
                .tolist())
folder = Path(tempfile.mkdtemp())
saved = brain.save(folder / "life.npz")
loaded = cd.Brain.load(saved)
for i, x in enumerate(moments(3)):
    acts.append(np.asarray(loaded.step(x, reward=np.ones(STREAMS) * 0.5)).tolist())
threads = False
if kernel._FUSED:
    from cadence import fused
    threads = bool(fused._in_parallel(STREAMS, int(brain.connectome.n)))
print(json.dumps({"fused": bool(kernel._FUSED), "threads": threads,
                  "neurons": int(brain.connectome.n),
                  "acts": acts, "saved": digest(saved),
                  "continued": digest(loaded.save(folder / "continued.npz"))}))
"""


def life(streams: int, width: int, **threads: str) -> dict:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.endswith("_NUM_THREADS") and key not in ("VECLIB_MAXIMUM_THREADS",)
    }
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
            **threads,
        }
    )
    done = subprocess.run(
        [sys.executable, "-c", LIFE, SOURCE, str(streams), str(width)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _numba() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return os.environ.get("CADENCE_FUSED", "1") != "0"


needs_numba = pytest.mark.skipif(not _numba(), reason="the compiled kernel needs Numba")


@needs_numba
def test_the_compiled_kernel_settles_a_batch_to_the_same_bits_at_any_numba_thread_count():
    # 192 rows of a brain of 388 neurons: a sweep of more than 65,536 neurons starts
    # Numba's threads
    one, two = life(192, 64, NUMBA_NUM_THREADS="1"), life(192, 64, NUMBA_NUM_THREADS="2")
    assert one["fused"] and one["neurons"] * 192 >= 65_536
    assert not one.pop("threads") and two.pop("threads")  # the second life settles on two
    assert one == two


@needs_numba
def test_the_compiled_kernel_acts_alike_at_any_blas_thread_count():
    blas = {
        name: "2"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    one, two = life(8, 64), life(8, 64, **blas)
    assert one["fused"] and two["fused"]
    assert one["acts"] == two["acts"]  # the same actions at either count, strictly
    if (one["saved"], one["continued"]) != (two["saved"], two["continued"]):
        # some machines' compiled kernel saves other last bits at two BLAS threads than at one
        pytest.xfail(
            "#315: the compiled kernel's checkpoint at one and two BLAS threads differs in the "
            "last bits on some machines (Ubuntu, Python 3.12 on CI), with the same actions"
        )


def test_the_recall_memorys_threaded_products_keep_the_bits_of_one_thread():
    # 64 streams of 128 slots of a code of 64 values: products of more than 262,144 values
    assert life(64, 64, CADENCE_RECALL_THREADS="1") == life(64, 64, CADENCE_RECALL_THREADS="2")


def test_the_numpy_loop_at_one_blas_thread_repeats_a_life_its_save_and_its_continuation():
    first, second = (life(8, 64, CADENCE_FUSED="0") for _ in range(2))
    assert not first["fused"]
    assert first == second
