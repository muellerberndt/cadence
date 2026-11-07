"""A stored transport plan: chosen once from measurements, kept per shape, repeatable by name."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd
from cadence import brain as module

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
ATOL = 1e-10


def devices() -> list[str]:
    return ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def stored_plans(tmp_path, monkeypatch):
    monkeypatch.setenv("CADENCE_PLAN_CACHE", str(tmp_path / "plans.json"))
    monkeypatch.setattr(module, "_PLANS", {})
    return tmp_path / "plans.json"


def graph(device: str, seed: int = 1, apical: bool = False, **options: object) -> cd.NeuralGraph:
    brain = cd.Brain.compose(
        12, 6, modules=(10, 8), slots=(3, 3), seed=seed, recall=False,
        **({"apical": 0.25} if apical else {}),
    )
    host = brain.learner.brain
    return cd.NeuralGraph(
        host.connectome, host.neuron_model, efficacy=host.efficacy, bias=host.bias,
        log_gain=host.log_gain, backend="torch", device=device, apical=host.apical,
        **options,  # type: ignore[arg-type]
    )


def same(host: cd.BrainState, dev: cd.BrainState) -> None:
    assert host.steps == dev.steps
    np.testing.assert_allclose(dev.activation, host.activation, rtol=0, atol=ATOL)
    np.testing.assert_allclose(dev.v, host.v, rtol=0, atol=ATOL)


def test_a_tile_never_exceeds_what_the_batch_fills():
    for batch, wide in ((1, 1), (2, 2), (3, 4), (5, 8), (16, 16), (17, 16), (63, 16)):
        assert module._tile_batch(batch, True) == wide
        assert module._tile_batch(batch, False) == wide
    assert module._tile_batch(64, False) == 64
    assert module._tile_batch(4096, False) == 64
    assert module._tile_batch(4096, True) == 16


def test_the_fused_update_takes_its_tile_from_size_batch_and_precision():
    assert module._update_tile(251, 1, True) == (32, 1)
    assert module._update_tile(251, 7, True) == (32, 8)
    assert module._update_tile(251, 16, True) == (8, 16)
    assert module._update_tile(251, 16, False) == (16, 16)
    assert module._update_tile(12821, 64, True) == (8, 64)
    assert module._update_tile(12821, 64, False) == (16, 64)


def test_the_launch_rule_reads_the_group_the_batch_and_the_precision():
    class Bucket:
        def __init__(self, count: int, rows: int, cols: int) -> None:
            self.count, self.rows, self.cols = count, rows, cols

    rule = module._launch_rule
    assert rule(Bucket(3, 8, 5000), 1, "float32") and rule(Bucket(3, 8, 5000), 1, "float64")
    assert rule(Bucket(1, 64, 128), 1, "float64") and not rule(Bucket(1, 64, 129), 1, "float64")
    assert rule(Bucket(1, 64, 256), 1, "float32") and not rule(Bucket(1, 64, 257), 4096, "float32")
    assert not rule(Bucket(1, 64, 512), 7, "float64") and rule(Bucket(1, 64, 512), 8, "float64")
    assert rule(Bucket(1, 64, 4096), 8, "float64") and not rule(Bucket(1, 64, 4097), 127, "float64")
    assert rule(Bucket(1, 64, 4097), 128, "float64")


def test_a_new_graph_runs_the_rules_and_says_so():
    first = graph("cpu")
    report = first.to_dict()["torch_transport"]
    assert report["plan"] == "rules"
    assert "plan_batches" not in report
    assert "torch_transport" not in cd.NeuralGraph(first.connectome, first.neuron_model).to_dict()


def test_the_cpu_backend_has_no_plan():
    host = graph("cpu")
    bare = cd.NeuralGraph(host.connectome, host.neuron_model)
    with pytest.raises(ValueError, match="torch backend"):
        bare.calibrate_transport()


@pytest.mark.parametrize("device", devices())
def test_a_plan_is_reported_stored_and_repeated_by_name(device, stored_plans):
    first = graph(device)
    plan = first.calibrate_transport(batches=(1, 8))
    assert set(plan["batches"]) == {"1", "8"}
    assert first.to_dict()["torch_transport"]["plan"] == plan["name"]
    assert first.to_dict()["torch_transport"]["plan_batches"] == plan["batches"]
    assert plan["precision"] == "float64"
    assert plan["hardware"] in plan["name"].replace("_", " ")
    assert json.loads(stored_plans.read_text())[plan["name"]]["batches"] == plan["batches"]
    # another process: the file serves, nothing is timed, the same object by name
    module._PLANS.clear()
    second = graph(device, seed=2)
    held = second.calibrate_transport(name=plan["name"])
    assert held["batches"] == plan["batches"]
    assert second.to_dict()["torch_transport"]["plan"] == plan["name"]
    # the rules again, by name
    assert second.calibrate_transport(name="rules") == {"name": "rules"}
    assert second.to_dict()["torch_transport"]["plan"] == "rules"


@pytest.mark.parametrize("device", devices())
def test_a_second_brain_of_one_shape_pays_nothing(device, monkeypatch):
    plan = graph(device, seed=1).calibrate_transport(batches=(4,))

    def timed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a stored plan is not timed again")

    monkeypatch.setattr(module._TorchKernel, "_seconds", timed)
    other = graph(device, seed=9)
    again = other.calibrate_transport(batches=(4,))
    assert again["name"] == plan["name"]
    assert again["calibration_seconds"] == plan["calibration_seconds"]


@pytest.mark.parametrize("device", devices())
def test_a_new_batch_size_is_timed_and_joins_the_plan(device):
    first = graph(device)
    plan = first.calibrate_transport(batches=(4,))
    more = first.calibrate_transport(batches=(4, 32))
    assert more["name"] == plan["name"]
    assert set(more["batches"]) == {"4", "32"}
    assert more["batches"]["4"] == plan["batches"]["4"]


@pytest.mark.parametrize("device", devices())
def test_a_plan_for_another_shape_or_precision_is_refused(device):
    plan = graph(device).calibrate_transport(batches=(4,))
    with pytest.raises(ValueError, match="no stored transport plan"):
        graph(device).calibrate_transport(name="nothing of that name")
    anatomy = cd.layered(3, 12, 3, seed=2)
    other = cd.NeuralGraph(anatomy, cd.learning_neuron_model(), backend="torch", device=device)
    with pytest.raises(ValueError, match="another graph or device"):
        other.calibrate_transport(name=plan["name"])
    with pytest.raises(ValueError, match="another graph or device"):
        graph(device, precision="float32").calibrate_transport(name=plan["name"])
    with pytest.raises(ValueError, match="batches"):
        graph(device).calibrate_transport(batches=())


@pytest.mark.parametrize("device", devices())
def test_a_graph_on_new_parameters_keeps_its_plan(device):
    first = graph(device)
    plan = first.calibrate_transport(batches=(4,))
    changed = first.with_parameters(efficacy=np.asarray(first.efficacy) * 0.5)
    assert changed.to_dict()["torch_transport"]["plan"] == plan["name"]


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("apical", [False, True])
def test_a_named_plan_matches_the_host_and_itself(device, apical):
    dev = graph(device, apical=apical)
    host = cd.NeuralGraph(dev.connectome, dev.neuron_model, efficacy=dev.efficacy, bias=dev.bias,
                          log_gain=dev.log_gain, apical=dev.apical)
    plan = dev.calibrate_transport(batches=(1, 5))
    drive = np.random.default_rng(3).uniform(-0.5, 1.0, (5, dev.connectome.n))
    expected = host.settle_batch(drive, steps=40, tolerance=1e-7)
    same(expected, dev.settle_batch(drive, steps=40, tolerance=1e-7))
    repeat = graph(device, apical=apical)
    repeat.calibrate_transport(name=plan["name"])
    a, b = dev.settle_batch(drive, steps=33), repeat.settle_batch(drive, steps=33)
    np.testing.assert_array_equal(a.activation, b.activation)
    same(host.settle_batch(drive, steps=33), a)


@cuda
@pytest.mark.parametrize("launched", [(), None])
def test_every_choice_of_a_launch_matches_the_host(launched):
    pytest.importorskip("triton")
    dev = graph("cuda:0", apical=True)
    host = cd.NeuralGraph(dev.connectome, dev.neuron_model, efficacy=dev.efficacy, bias=dev.bias,
                          log_gain=dev.log_gain, apical=dev.apical)
    plan = dev._torch.plan
    chosen = [list(part.buckets) if launched is None else [] for part in plan.parts]
    plan.choose({"name": "test", "batches": {"5": chosen}})
    assert plan.on_launch(plan.moving, 5) == tuple(chosen[0])
    drive = np.random.default_rng(4).uniform(-0.5, 1.0, (5, dev.connectome.n))
    same(host.settle_batch(drive, steps=40), dev.settle_batch(drive, steps=40))


@cuda
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("batch", [1, 3, 16, 70])
def test_groups_of_every_height_in_one_launch_match_the_host(precision, batch):
    pytest.importorskip("triton")
    # groups of 1 to 33 rows, three of each height, over windows of 7 to 40 presynaptic neurons
    pre, post = [], []
    first = 80
    for height in (1, 2, 3, 5, 8, 9, 16, 17, 33):
        for g in range(3):
            for r in range(height):
                for c in range(7 + height):
                    pre.append((g * 5 + c) % 80)
                    post.append(first + r)
            first += height
    connectome = cd.Connectome.from_synapses(
        first, pre=pre, post=post, sign=np.random.default_rng(0).uniform(-0.3, 0.3, len(pre)),
        populations={"input": range(80), "hidden": range(80, first)},
    )
    model = cd.learning_neuron_model()
    host = cd.NeuralGraph(connectome, model)
    dev = cd.NeuralGraph(connectome, model, backend="torch", device="cuda:0", precision=precision)
    assert dev._torch.plan.small is not None
    drive = np.random.default_rng(1).uniform(0, 1, (batch, connectome.n))
    a, b = host.settle_batch(drive, steps=9), dev.settle_batch(drive, steps=9)
    tolerance = 1e-10 if precision == "float64" else 2e-3
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=tolerance)


@cuda
def test_a_measured_plan_names_its_readings_and_cost():
    pytest.importorskip("triton")
    plan = graph("cuda:0").calibrate_transport(batches=(1, 16))
    assert plan["calibration_seconds"] > 0
    for batch in ("1", "16"):
        reading = plan["seconds"][batch]
        assert reading["chosen"] <= reading["rules"]
