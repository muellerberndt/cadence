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
def test_a_measured_plan_names_its_readings_and_cost():
    pytest.importorskip("triton")
    plan = graph("cuda:0").calibrate_transport(batches=(1, 16))
    assert plan["calibration_seconds"] > 0
    for batch in ("1", "16"):
        reading = plan["seconds"][batch]
        assert reading["chosen"] <= reading["rules"]
