"""Optional recall threads preserve independent row arithmetic and process ownership."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cadence import stream


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("spec", ["bk,bmk->bm", "bm,bmc->bc", "bm,bmk->bk"])
def test_threaded_products_keep_numpy_bits(monkeypatch, workers, dtype, spec):
    monkeypatch.setenv("CADENCE_RECALL_THREADS", str(workers))
    monkeypatch.setattr(stream, "_THREADED", 1)
    rng = np.random.default_rng(12)
    a = rng.normal(size=(8, 17 if spec.startswith("bk") else 23)).astype(dtype)
    b = rng.normal(size=(8, 23, 17)).astype(dtype)
    np.testing.assert_array_equal(stream._einsum(spec, a, b), np.einsum(spec, a, b))


def test_noncontiguous_operands_keep_the_original_traversal(monkeypatch):
    monkeypatch.setenv("CADENCE_RECALL_THREADS", "4")
    monkeypatch.setattr(stream, "_THREADED", 1)
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=(8, 7)), rng.normal(size=(8, 11, 7))
    for query, keys in ((a[:, ::-1], b[:, :, ::-1]),
                        (np.asfortranarray(a), np.asfortranarray(b))):
        np.testing.assert_array_equal(stream._einsum("bk,bmk->bm", query, keys),
                                      np.einsum("bk,bmk->bm", query, keys))


def test_changing_worker_counts_reuses_one_bounded_pool(monkeypatch):
    monkeypatch.setattr(stream, "_THREADED", 1)
    pool = None
    for count in (2, 4, 3, 1000, 8):
        monkeypatch.setenv("CADENCE_RECALL_THREADS", str(count))
        parts = stream._by_rows(lambda start, stop: list(range(start, stop)), 64, 64)
        assert [row for block in parts for row in block] == list(range(64))
        current = stream._recall_pool()
        assert pool is None or pool is current
        pool = current
        assert len(pool._threads) <= stream._RECALL_WORKERS


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is unavailable")
def test_a_forked_child_uses_its_own_threads_without_inherited_locks():
    script = """
import os
from cadence import stream
stream._THREADED = 1
os.environ['CADENCE_RECALL_THREADS'] = '2'
assert stream._by_rows(lambda a, b: b - a, 4, 4) == [2, 2]
parent_pool = stream._recall_pool()
pid = os.fork()
if pid == 0:
    good = stream._by_rows(lambda a, b: b - a, 4, 4) == [2, 2]
    good = good and stream._recall_pool() is not parent_pool
    os._exit(0 if good else 1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
"""
    env = dict(os.environ, PYTHONPATH=str(Path(stream.__file__).resolve().parents[1]))
    subprocess.run([sys.executable, "-c", script], env=env, check=True, timeout=15)
