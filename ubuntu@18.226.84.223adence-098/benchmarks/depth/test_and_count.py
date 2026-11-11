"""The AND-count benchmark's task, arms and receipt, on a few lessons."""

import importlib.util
from pathlib import Path

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("and_count", HERE / "and_count.py")
assert spec is not None and spec.loader is not None
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_the_answer_is_the_capped_count_of_pairs_on():
    x, y = bench.task(rows=500)
    assert x.shape == (500, 48) and set(np.unique(x)) <= {0.0, 1.0}
    pairs = x.reshape(500, 24, 2)
    on = (pairs[..., 0] * pairs[..., 1]).sum(axis=1)
    np.testing.assert_array_equal(y, np.clip(on - 3, 0, 6))
    again, labels = bench.task(rows=500)
    np.testing.assert_array_equal(again, x)  # a fixed pool: the data seed alone draws it
    np.testing.assert_array_equal(labels, y)
    held, train = bench.split(500, seed=0, held=50)
    assert len(set(held) & set(train)) == 0 and len(held) + len(train) == 500


def test_every_arm_runs_and_the_receipt_verifies(tmp_path):
    runs = [bench.run_arm(arm, 0, lessons=4, every=2, rows=600, held=50)
            for arm in ("one", "two", "mlp_one", "random")]
    for run in runs:
        assert [row["lesson"] for row in run["curve"]] == [2, 4]
        assert all(0.0 <= row["accuracy"] <= 1.0 for row in run["curve"])
    assert runs[1]["apical"]["pools"] == 2  # module_0 and the association hear the regions above
    body = {"schema": bench.KIND, "runs": runs, "summary": bench.summarize(runs),
            "configuration": bench.configuration(4, 2, [0], [r["arm"] for r in runs], 600, 50),
            "execution": bench.evidence["environment"]()}
    path = cd.Receipt.build(bench.KIND, body, bench.sources()).write(tmp_path / "summary.json")
    valid, message = cd.Receipt.verify(path, sources=bench.sources(), check=bench.check)
    assert valid, message
    body["summary"]["final_accuracy"]["one"] += 0.5
    forged = cd.Receipt.build(bench.KIND, body, bench.sources()).write(tmp_path / "forged.json")
    assert not cd.Receipt.verify(forged, sources=bench.sources(), check=bench.check)[0]


def test_the_retained_receipt_is_source_bound_and_recomputed():
    path = HERE / "results" / "and-count.json"
    valid, message = bench.verify_historical(path)
    assert valid, message
    body = cd.Receipt.read(path).body
    assert body["lessons"] == 4000 and body["seeds"] == [0, 1, 2]
    means = body["summary"]["final_accuracy"]
    # every added module raises the accuracy; four come within 0.05 of two hidden layers
    assert means["one"] < means["two"] < means["three"] < means["four"]
    assert means["four"] > means["mlp_two"] - 0.05 and means["one"] > means["mlp_one"]


def test_the_memory_receipt_is_source_bound_and_recomputed():
    spec = importlib.util.spec_from_file_location("depth_memory", HERE / "memory.py")
    assert spec is not None and spec.loader is not None
    memory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(memory)
    path = HERE / "results" / "memory.json"
    valid, message = memory.verify_historical(path)
    assert valid, message
    summary = cd.Receipt.read(path).body["summary"]
    assert summary["look_back"] == {d: {"passed": 4, "founders": 4} for d in ("2", "3")}
    assert len(memory.cells([2, 3], [0, 1, 2, 3])) == 32
