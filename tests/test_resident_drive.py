"""A drive that rests on a device is refused where a host array is. ``NeuralGraph.settle_batch``
and ``equilibrate`` read a torch tensor or an MLX array they are given for finiteness on its
device and refuse it with the host's sentence, in both precisions, before a sweep runs; a drive
written in place after a settle is read again. A drive the library uploaded from an array it read
finite is marked, so a moment's settles read nothing of it for the check."""

from __future__ import annotations

import gc

import numpy as np
import pytest
from places import DEVICES, place

import cadence as cd
from cadence import brain as brain_module

REFUSAL = "drive must be finite and the brain must contain neurons"
BAD = [np.nan, np.inf, -np.inf]


def graph_and_drive(name: str):
    """A composed brain's graph on ``name`` and a finite host drive of two rows."""
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=0, **place(name))
    drive = brain.stimulus(np.eye(4)[[0, 2]], memory=False)
    return brain.brain, np.array(drive)


def resident(graph, host: np.ndarray, precision: str | None = None):
    """A copy of ``host`` resting where ``graph`` settles, in its precision or the one named."""
    if graph.backend == "mlx":
        import mlx.core as mx

        return mx.array(host.astype(np.float32))
    import torch

    kernel = graph._torch
    dtype = kernel.dtype if precision is None else getattr(torch, precision)
    if kernel.device.type == "mps":
        dtype = torch.float32
    return torch.tensor(host, dtype=dtype, device=kernel.device)


@pytest.mark.parametrize("name", DEVICES)
@pytest.mark.parametrize("bad", BAD, ids=["nan", "inf", "-inf"])
def test_a_nonfinite_resident_drive_is_refused_as_the_host_array_is(name, bad):
    graph, drive = graph_and_drive(name)
    warm = graph.settle_batch(drive, steps=4)
    spoiled = drive.copy()
    spoiled[1, graph.connectome.n - 1] = bad
    given = [spoiled, resident(graph, spoiled)]
    if graph.backend == "torch":
        import torch

        given.append(torch.as_tensor(spoiled))  # on the CPU in float64, whatever the kernel's
        given.append(resident(graph, spoiled, "float32"))
    calls = (
        lambda one: graph.settle_batch(one, steps=3),
        lambda one: graph.settle_batch(one, steps=0, state=warm),
        lambda one: graph.equilibrate(one, budget=8),
        lambda one: graph.equilibrate(one, state=warm, budget=8, damping=1),
    )
    for one in given:
        for call in calls:
            with pytest.raises(ValueError, match=REFUSAL):
                call(one)
    # the warm state is what it was, and the finite drive settles from it wherever it rests
    here = graph.settle_batch(drive, steps=3, state=warm)
    there = graph.settle_batch(resident(graph, drive), steps=3, state=warm)
    np.testing.assert_allclose(there.v, here.v, rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", DEVICES)
def test_a_resident_drive_written_in_place_is_read_again(name):
    graph, drive = graph_and_drive(name)
    held = resident(graph, drive)
    assert graph.settle_batch(held, steps=2).steps == 2
    assert graph.settle_batch(held, steps=2).steps == 2
    held[0, 1] = float("nan")  # the same array, written where it rests
    with pytest.raises(ValueError, match=REFUSAL):
        graph.settle_batch(held, steps=2)
    if graph.backend == "torch" and graph._torch.device.type == "cpu":
        import torch

        shared = drive.copy()
        tensor = torch.from_numpy(shared)  # the tensor reads the array's own memory
        assert graph.settle_batch(tensor, steps=2).steps == 2
        shared[1, 0] = np.inf
        with pytest.raises(ValueError, match=REFUSAL):
            graph.settle_batch(tensor, steps=2)


@pytest.mark.parametrize("name", [name for name in DEVICES if name.startswith("torch")])
@pytest.mark.parametrize("code", [None, 8], ids=["plain", "learned_code"])
def test_a_moments_drive_is_marked_and_no_settle_reads_it_for_the_check(name, code, monkeypatch):
    """The brain reads its observations and its drive finite on the host and uploads them once;
    the learned code's region does the same for its own phase."""
    options = {} if code is None else {"learned_code": code}
    brain = cd.Brain.compose(8, 3, modules=(12,), seed=3, **options, **place(name))
    x = np.random.default_rng(0).random((2, 8))
    brain.act(x, greedy=True)
    seen = []
    check = brain_module._drive_finite

    def recorded(drive):
        kept = brain_module._TRUSTED.get(id(drive))
        seen.append((brain_module._resident(drive), kept is not None and kept[0]() is drive))
        return check(drive)

    monkeypatch.setattr(brain_module, "_drive_finite", recorded)
    brain.act(x + 0.1, greedy=True)
    rests = [marked for there, marked in seen if there]
    assert rests and all(rests)


@pytest.mark.parametrize("name", DEVICES)
def test_a_mark_holds_for_the_drive_as_it_was_marked(name):
    graph, drive = graph_and_drive(name)
    held = resident(graph, drive)
    marks = len(brain_module._TRUSTED)
    assert brain_module._trusted(held) is held
    assert brain_module._drive_finite(held)
    if graph.backend == "torch":  # a write in place moves the tensor's version, and the mark ends
        held[0, 0] = float("inf")
        assert not brain_module._drive_finite(held)
    del held
    gc.collect()
    assert len(brain_module._TRUSTED) <= marks  # the mark went with the drive
    assert brain_module._trusted(drive) is drive  # a host array is read at every call
    spoiled = np.full_like(drive, np.nan)
    assert not brain_module._drive_finite(brain_module._trusted(spoiled))


def test_a_settle_on_mlx_reads_an_unmarked_array_with_its_first_launch(monkeypatch):
    """A launch synchronizes once and brings the drive with it, and the host reads the drive
    there: no synchronization for the drive alone. A settle of no sweeps launches nothing and
    reads the drive itself; inside a solve it leaves the drive to the solve's first launch."""
    graph, drive = graph_and_drive("mlx")
    mx = graph._mlx.mx
    seen = []
    evaluate, check = mx.eval, brain_module._drive_finite

    def evaluated(*arrays):
        seen.append("launch")
        return evaluate(*arrays)

    def read(one):
        if brain_module._resident(one) and not brain_module._marked(one):
            seen.append("read")  # the host reads the array itself
        return check(one)

    monkeypatch.setattr(mx, "eval", evaluated)
    monkeypatch.setattr(brain_module, "_drive_finite", read)
    held = resident(graph, drive)
    warm = graph.settle_batch(held, steps=3)
    assert seen == ["launch", "read"]
    del seen[:]
    assert graph.settle_batch(held, steps=0, state=warm).steps == 0
    assert seen == ["read"]
    del seen[:]
    solved = graph.equilibrate(resident(graph, drive), tolerance=1e-4)
    assert solved.qualified.all() and seen[:2] == ["launch", "read"] and seen.count("read") == 1
    del seen[:]
    marked = brain_module._trusted(resident(graph, drive))
    graph.settle_batch(marked, steps=3)
    assert seen == ["launch"]  # a marked drive is not read at all
    spoiled = drive.copy()
    spoiled[0, 0] = np.inf  # a source neuron's drive: the launch's own check does not see it
    for call in (
        lambda one: graph.settle_batch(one, steps=3),
        lambda one: graph.equilibrate(one, tolerance=1e-4),
    ):
        with pytest.raises(ValueError, match=REFUSAL):
            call(resident(graph, spoiled))
