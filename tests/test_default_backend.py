"""The default backend (cadence#276): a composed brain takes the fastest backend for its size and
rows, states what it took and why, and a device that refuses it never fails it."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import cadence as cd
from cadence import generic, resident
from cadence.resident import _Candidate, _fastest_backends


@pytest.fixture(autouse=True)
def _nothing_named(monkeypatch):
    monkeypatch.delenv("CADENCE_BACKEND", raising=False)


def on_a_machine(monkeypatch, *, card: str | None = None, mlx: bool = False) -> None:
    monkeypatch.setattr(resident, "_cuda_card", lambda: card)
    monkeypatch.setattr(resident, "_mlx_present", lambda: mlx)


def choose(rows: int, synapses: int, **options) -> tuple[str, ...]:
    return tuple(c.backend for c in _fastest_backends(rows, synapses, **options))


CHAMBER, DEEP, SPEECH = 13_120, 50_048, 6_321_152


def test_a_machine_with_neither_device_takes_the_cpu_at_every_size(monkeypatch):
    on_a_machine(monkeypatch)
    for rows, synapses in ((1, CHAMBER), (256, CHAMBER), (64, DEEP), (1, SPEECH), (256, SPEECH)):
        assert choose(rows, synapses) == ("cpu",)


def test_cuda_takes_the_brains_its_table_read_faster(monkeypatch):
    on_a_machine(monkeypatch, card="NVIDIA A10G")
    assert choose(1, CHAMBER) == ("cpu",)
    assert choose(16, CHAMBER) == ("cpu",)
    assert choose(1, SPEECH) == ("torch", "cpu")
    assert choose(256, SPEECH) == ("torch", "cpu")
    picked = _fastest_backends(1, SPEECH)[0]
    assert (picked.device, picked.precision) == ("cuda", "float32")
    assert _fastest_backends(1, SPEECH, precision="float64")[0].precision == "float64"
    assert choose(1, SPEECH, learned_code=True) == ("cpu",)


def test_a_compartment_passes_one_core_sooner_on_cuda(monkeypatch):
    on_a_machine(monkeypatch, card="NVIDIA A10G")
    assert choose(1, DEEP, apical=True) == ("cpu",)
    assert choose(4, DEEP, apical=True) == ("torch", "cpu")
    assert choose(4, DEEP) == ("cpu",)
    assert choose(64, CHAMBER) == ("torch", "cpu")


def test_mlx_takes_the_brains_its_table_read_faster(monkeypatch):
    on_a_machine(monkeypatch, mlx=True)
    assert choose(16, CHAMBER) == ("cpu",)
    assert choose(1, SPEECH) == ("mlx", "cpu")
    assert choose(1, SPEECH, precision="float64") == ("cpu",)


def test_a_brain_no_launch_takes_passes_one_core_at_its_rows_on_mlx(monkeypatch):
    on_a_machine(monkeypatch, mlx=True)
    deep_anatomy = {"settling": 7_000, "apical": True}
    assert choose(16, 903_520, **deep_anatomy) == ("cpu",)
    assert choose(64, 903_520, **deep_anatomy) == ("mlx", "cpu")
    assert choose(1, SPEECH, settling=549) == ("mlx", "cpu")
    assert choose(1, SPEECH, settling=3_000) == ("cpu",)


def test_a_card_the_table_has_not_read_takes_a_stated_margin(monkeypatch):
    work = resident._AUTO_CUDA_WORK
    on_a_machine(monkeypatch, card="NVIDIA A10G")
    assert choose(1, work) == ("torch", "cpu")
    on_a_machine(monkeypatch, card="Some Other Card")
    assert choose(1, work) == ("cpu",)
    assert choose(1, resident._AUTO_UNSEEN * work) == ("torch", "cpu")


def test_memories_that_stay_on_the_host_need_more_work(monkeypatch):
    work = resident._AUTO_CUDA_WORK
    on_a_machine(monkeypatch, card="NVIDIA A10G")
    assert choose(1, work) == ("torch", "cpu")
    assert choose(1, work, host_memory=True) == ("cpu",)
    assert choose(1, resident._AUTO_HOST_MEMORY * work, host_memory=True) == ("torch", "cpu")


def test_the_rule_imports_nothing_below_a_devices_work():
    code = (
        "import sys, numpy as np, cadence as cd\n"
        "brain = cd.Brain.compose(12, 3, modules=(16,), seed=0)\n"
        "brain.act(np.eye(12)[[1, 2]], greedy=True)\n"
        "assert brain.placement['backend'] == 'cpu', brain.placement\n"
        "assert 'torch' not in sys.modules and 'mlx' not in sys.modules\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        env={**__import__("os").environ, "CADENCE_BACKEND": ""},
    )
    assert done.returncode == 0, done.stderr


def small(**options) -> cd.Brain:
    return cd.Brain.compose(12, 5, **{"modules": (16,), "seed": 3, **options})


def a_life(brain: cd.Brain, rows: int = 3, moments: int = 8) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    out = []
    for t in range(moments):
        x = np.eye(12)[rng.integers(12, size=rows)]
        kwargs = {} if t == 0 else {"reward": rng.random(rows), "done": rng.random(rows) < 0.2}
        out.append(np.asarray(brain.step(x, **kwargs)))
    return out


def test_the_default_on_a_machine_without_a_device_has_the_bits_of_the_cpu(monkeypatch):
    on_a_machine(monkeypatch)
    named, default = small(backend="cpu"), small()
    assert default.placement["pending"] is True
    assert default.placement["chosen_by"] == "auto" and default.placement["backend"] == "cpu"
    for ours, theirs in zip(a_life(named), a_life(default), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(named.brain.efficacy, default.brain.efficacy)
    report = default.describe()["placement"]
    assert report["backend"] == "cpu" and report["pending"] is False
    assert "synapses" in report["why"]
    assert named.placement["chosen_by"] == "named"
    assert "placement" not in named.describe()


def to_the_torch_cpu(from_rows: int):
    """A machine whose device is the torch CPU device, from a number of rows."""

    def rule(rows, synapses, **options):
        here = _Candidate("cpu", None, None, "below the work")
        if rows < from_rows:
            return [here]
        return [_Candidate("torch", "cpu", "float64", "past the work"), here]

    return rule


@pytest.mark.parametrize("rows", [3, 5])
@pytest.mark.parametrize(
    "options",
    [
        {}, {"modules": (128,)}, {"modules": (16, 8), "apical": 0.25, "teaching": "deep"},
        {"recall": False}, {"recall_founder": True},
    ],
    ids=["default", "wide", "apical", "no recall", "recall founder"],
)
def test_a_brain_takes_its_backend_at_its_first_moment_and_lives_on(monkeypatch, rows, options):
    pytest.importorskip("torch")
    monkeypatch.setattr(generic, "_fastest_backends", to_the_torch_cpu(from_rows=2))
    if options.get("recall_founder") is None and options.get("recall") is None:
        options = {**options, "learning_fast_settle": False}  # the compiled CPU path's other solve
    named = small(**{"backend": "cpu", **options})
    default = small(**options)
    assert default.brain._torch is None and default.placement["pending"] is True
    host, placed = a_life(named, rows), a_life(default, rows)
    assert default.brain._torch is not None
    for ours, theirs in zip(host, placed, strict=True):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_allclose(default.brain.efficacy, named.brain.efficacy, rtol=0, atol=2e-8)
    report = default.placement
    assert report["backend"] == "torch" and report["chosen_by"] == "auto"
    assert report["device"] == "cpu" and report["precision"] == "float64"
    assert report["why"] == "past the work" and report["pending"] is False


def test_a_device_faster_at_one_row_takes_the_brain_at_compose(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(generic, "_fastest_backends", to_the_torch_cpu(from_rows=1))
    brain = small()
    assert brain.brain._torch is not None and brain.placement["pending"] is False
    recall = brain.recall
    a_life(brain)
    assert brain.recall is recall  # nothing moved at the first moment


def test_a_precision_named_with_a_device_names_torch():
    pytest.importorskip("torch")
    brain = small(device="cpu", precision="float32")
    assert brain.brain._torch is not None and brain.placement["precision"] == "float32"
    assert brain.placement["chosen_by"] == "named"


def refusing(rows, synapses, **options):
    return [
        _Candidate("torch", "no such device", None, "past the work"),
        _Candidate("cpu", None, None, "the next fastest"),
    ]


def test_a_device_that_refuses_at_compose_leaves_the_cpu_and_says_so(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(generic, "_fastest_backends", refusing)
    brain = small()
    report = brain.placement
    assert report["backend"] == "cpu" and report["pending"] is False
    assert report["why"] == "the next fastest"
    assert len(report["refused"]) == 1 and report["refused"][0].startswith("torch: ")
    a_life(brain)


def test_a_device_that_refuses_at_the_first_moment_leaves_the_cpu_and_says_so(monkeypatch):
    pytest.importorskip("torch")

    def later(rows, synapses, **options):
        return refusing(rows, synapses) if rows > 1 else [_Candidate("cpu", None, None, "small")]

    monkeypatch.setattr(generic, "_fastest_backends", later)
    brain, named = small(), small(backend="cpu")
    assert brain.placement["pending"] is True
    for ours, theirs in zip(a_life(named), a_life(brain), strict=True):
        np.testing.assert_array_equal(ours, theirs)
    report = brain.placement
    assert report["backend"] == "cpu" and len(report["refused"]) == 1
    assert report["pending"] is False


def test_invalid_observations_are_the_operations_to_refuse_and_the_next_moment_decides(monkeypatch):
    on_a_machine(monkeypatch)
    brain = small()
    with pytest.raises(ValueError, match="observations"):
        brain.act(np.zeros(12), greedy=True)
    assert brain.placement["pending"] is True
    brain.act(np.eye(12)[[1]], greedy=True)
    assert brain.placement["pending"] is False


def test_the_environment_names_a_backend_for_a_run(monkeypatch):
    monkeypatch.setenv("CADENCE_BACKEND", "cpu")
    brain = small()
    report = brain.placement
    assert report["chosen_by"] == "environment" and report["backend"] == "cpu"
    assert report["pending"] is False
    assert small(backend="cpu").placement["chosen_by"] == "named"
    monkeypatch.setenv("CADENCE_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="CADENCE_BACKEND"):
        small()


def test_the_environment_can_name_a_device_and_its_floats(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("CADENCE_BACKEND", "torch:cpu:float32")
    report = small().placement
    assert (report["backend"], report["device"], report["precision"]) == ("torch", "cpu", "float32")


def test_a_saved_brain_states_what_it_ran_on_and_a_chosen_one_chooses_again(monkeypatch, tmp_path):
    on_a_machine(monkeypatch)
    default = small()
    a_life(default)
    path = default.save(tmp_path / "chosen.npz")
    loaded = cd.Brain.load(path)
    assert loaded.placement["chosen_by"] == "auto" and loaded.placement["pending"] is True
    loaded.act(np.eye(12)[[1, 2, 3]], greedy=True)
    assert loaded.placement["pending"] is False and loaded.placement["backend"] == "cpu"
    named = small(backend="cpu")
    a_life(named)
    again = cd.Brain.load(named.save(tmp_path / "named.npz"))
    assert again.placement["chosen_by"] == "named" and again.placement["pending"] is False
    asked = cd.Brain.load(path, backend="cpu")
    assert asked.placement["pending"] is False


def test_a_brain_loaded_elsewhere_states_what_it_ran_on(tmp_path):
    pytest.importorskip("torch")
    brain = small(device="cpu", precision="float32")
    a_life(brain)
    loaded = cd.Brain.load(brain.save(tmp_path / "torch.npz"), backend="cpu")
    report = loaded.placement
    assert report["backend"] == "cpu" and report["chosen_by"] == "named"
    assert report["saved_on"]["backend"] == "torch"
    assert report["saved_on"]["precision"] == "float32"


def test_grow_and_graft_of_a_brain_the_rule_chose_keep_its_place(monkeypatch):
    on_a_machine(monkeypatch)
    brain = small()
    a_life(brain)
    grown = brain.grow(history_decays=(0.5,))
    assert grown.brain.backend == "cpu"
    grown.act(np.eye(12)[[1]], greedy=True)


def test_a_deep_copy_before_its_first_moment_takes_its_own(monkeypatch):
    import copy

    on_a_machine(monkeypatch)
    brain = small()
    twin = copy.deepcopy(brain)
    a_life(brain)
    assert twin.placement["pending"] is True
    a_life(twin)
    assert twin.placement["pending"] is False
    np.testing.assert_array_equal(brain.brain.efficacy, twin.brain.efficacy)


def test_a_first_fit_places_the_brain_at_its_batch(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(generic, "_fastest_backends", to_the_torch_cpu(from_rows=4))
    brain = small(recall=False)
    rng = np.random.default_rng(0)
    brain.fit(np.eye(12)[rng.integers(12, size=8)], rng.integers(5, size=8), epochs=1, batch=4)
    assert brain.brain._torch is not None and brain.placement["pending"] is False
    other = small(recall=False)
    other.fit(np.eye(12)[rng.integers(12, size=8)], rng.integers(5, size=8), epochs=1, batch=2)
    assert other.brain._torch is None and other.placement["pending"] is False
