"""The torch settle (transport plan, neuron-major sweeps, CUDA graphs) against the host path.

Every float64 device settle must reach the host's fixed point with the host's step count:
activations within 1e-10, the same sweeps under a movement tolerance, the same residuals,
apical gains and learning steps. CUDA cases run where a GPU is present; the CPU device runs
everywhere torch does.
"""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")

ATOL = 1e-10


def devices() -> list[str]:
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda:0")
    return out


@pytest.fixture(params=devices())
def device(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def composed(apical: bool, seed: int = 1, adaptation: bool = False) -> cd.Brain:
    brain = cd.Brain.compose(
        12, 6, modules=(10, 8), slots=(3, 3), seed=seed, **{"recall": False, **({"apical": 0.25} if apical else {})}
    )
    if adaptation:
        graph = brain.learner.brain
        brain.learner.brain = cd.NeuralGraph(
            graph.connectome,
            graph.neuron_model.replace(adaptation=cd.Adaptation(tau_steps=6, strength=0.3)),
            efficacy=graph.efficacy, bias=graph.bias, log_gain=graph.log_gain, apical=graph.apical,
        )
    return brain


def on(graph: cd.NeuralGraph, device: str, **changes: object) -> cd.NeuralGraph:
    return cd.NeuralGraph(
        graph.connectome, graph.neuron_model, efficacy=graph.efficacy, bias=graph.bias,
        log_gain=graph.log_gain, backend="torch", device=device, apical=graph.apical,
        **changes,  # type: ignore[arg-type]
    )


def same(host: cd.BrainState, dev: cd.BrainState) -> None:
    assert host.steps == dev.steps
    np.testing.assert_allclose(dev.activation, host.activation, rtol=0, atol=ATOL)
    np.testing.assert_allclose(dev.v, host.v, rtol=0, atol=ATOL)
    np.testing.assert_allclose(dev.adaptation, host.adaptation, rtol=0, atol=ATOL)
    np.testing.assert_allclose(dev.activity_change, host.activity_change, rtol=0, atol=1e-9)


@pytest.mark.parametrize("apical", [False, True])
@pytest.mark.parametrize("batch", [1, 7])
@pytest.mark.parametrize("adaptation", [False, True])
def test_free_and_nudged_settles_match_the_host(device, apical, batch, adaptation):
    brain = composed(apical, adaptation=adaptation)
    host = brain.learner.brain
    dev = on(host, device)
    n = host.connectome.n
    rng = np.random.default_rng(3)
    drive = rng.uniform(-0.5, 1.0, (batch, n))
    for tolerance in (None, 1e-6):
        a, b = host.settle_batch(drive, steps=70, tolerance=tolerance), dev.settle_batch(
            drive, steps=70, tolerance=tolerance)
        same(a, b)
        held = host.apical_gain(drive, a)
        if apical:
            np.testing.assert_allclose(dev.apical_gain(drive, b), held, rtol=0, atol=ATOL)
        else:
            assert held is None and dev.apical_gain(drive, b) is None
        target = np.zeros((batch, n))
        target[:, brain.learner.output_index[0]] = 1.0
        for sign in (1.0, -1.0):
            nudge = brain.learner.nudge_for(target, sign * 0.1, None, held)
            for nudged_tolerance in (None, 1e-7):
                ap = host.settle_batch(drive, steps=32, state=a, nudge=nudge,
                                       tolerance=nudged_tolerance)
                bp = dev.settle_batch(drive, steps=32, state=b, nudge=nudge,
                                      tolerance=nudged_tolerance)
                same(ap, bp)
                np.testing.assert_allclose(
                    dev.residual(drive, bp, nudge=nudge), host.residual(drive, ap, nudge=nudge),
                    rtol=0, atol=1e-12,
                )


@pytest.mark.parametrize("kind", ["quadratic", "irregular", "anchored"])
def test_nudge_kinds_and_row_weights_match_the_host(device, kind):
    brain = composed(apical=True)
    host = brain.learner.brain
    dev = on(host, device)
    n = host.connectome.n
    rng = np.random.default_rng(5)
    drive = rng.uniform(-0.4, 0.9, (4, n))
    free_host, free_dev = host.settle_batch(drive, steps=40), dev.settle_batch(drive, steps=40)
    mask = np.zeros(n)
    outputs = brain.learner.output_index
    mask[outputs] = 1.0
    mask[3] = 0.5  # a reach that is not one contiguous run
    target = rng.uniform(0, 1, (4, n))
    options: dict[str, object] = {"weight": np.array([1.0, -0.5, 0.0, 2.0])}
    if kind == "irregular":  # unequal softmax groups, one of them not contiguous
        groups = np.full(n, -1)
        groups[outputs[:2]] = 0
        groups[outputs[2:]] = 1
        groups[3] = 0
        options.update(softmax_temperature=0.3, groups=groups)
    if kind == "anchored":
        options.update(anchor=rng.uniform(0, 1, (4, n)), anchor_gain=np.where(
            np.arange(n) % 3 == 0, 0.05, 0.0))
    held = host.apical_gain(drive, free_host)
    nudge = cd.Nudge(target, mask, 0.2, apical_gain=held, **options)  # type: ignore[arg-type]
    ap = host.settle_batch(drive, steps=25, state=free_host, nudge=nudge)
    bp = dev.settle_batch(drive, steps=25, state=free_dev, nudge=nudge)
    same(ap, bp)
    np.testing.assert_allclose(
        dev.residual(drive, bp, nudge=nudge), host.residual(drive, ap, nudge=nudge),
        rtol=0, atol=1e-12,
    )


@pytest.mark.parametrize("shape", ["shared", "row", "rows"])
def test_masks_match_the_host(device, shape):
    brain = composed(apical=True)
    host = brain.learner.brain
    dev = on(host, device)
    n = host.connectome.n
    rng = np.random.default_rng(9)
    drive = rng.uniform(-0.4, 0.9, (3, n))
    keep = rng.choice([0.0, 0.5, 1.0], {"shared": (n,), "row": (1, n), "rows": (3, n)}[shape])
    a = host.settle_batch(drive, steps=30, mask=keep, tolerance=1e-6)
    b = dev.settle_batch(drive, steps=30, mask=keep, tolerance=1e-6)
    same(a, b)
    a2 = host.settle_batch(drive, steps=9, state=a, mask=keep)
    b2 = dev.settle_batch(drive, steps=9, state=b, mask=keep)
    same(a2, b2)
    np.testing.assert_allclose(dev.residual(drive, b2, mask=keep),
                               host.residual(drive, a2, mask=keep), rtol=0, atol=1e-12)


@pytest.mark.parametrize("adaptation", [False, True])
def test_a_leakless_rule_with_row_masks_matches_the_host(device, adaptation):
    graph = cd.layered(3, 24, 4, density=0.6, seed=2)
    model = cd.NeuronModel(slope=4.0, threshold=1.0, gain=0.3)  # leak 0: a hard rectifier
    if adaptation:
        model = model.replace(adaptation=cd.Adaptation(tau_steps=5, strength=0.4))
    host = cd.NeuralGraph(graph, model)
    dev = on(host, device)
    rng = np.random.default_rng(13)
    drive = rng.uniform(0.0, 3.0, (9, graph.n))
    keep = rng.choice([0.0, 0.3, 1.0], (9, graph.n))
    for tolerance in (None, 1e-8):
        same(host.settle_batch(drive, steps=80, mask=keep, tolerance=tolerance),
             dev.settle_batch(drive, steps=80, mask=keep, tolerance=tolerance))


@pytest.mark.parametrize("transport", ["grouped", "blocks", "sparse"])
def test_every_transport_matches_the_host(device, transport, monkeypatch):
    monkeypatch.setenv("CADENCE_TORCH_TRANSPORT", transport)
    brain = composed(apical=True)
    host = brain.learner.brain
    dev = on(host, device)
    assert dev._torch.plan.kind == transport
    rng = np.random.default_rng(2)
    drive = rng.uniform(-0.5, 1.0, (5, host.connectome.n))
    same(host.settle_batch(drive, steps=50, tolerance=1e-6),
         dev.settle_batch(drive, steps=50, tolerance=1e-6))


@pytest.mark.parametrize("transport", ["grouped", "blocks"])
@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 11, 12, 33, 34])
def test_still_products_match_the_host_at_every_stage(device, transport, steps, monkeypatch):
    monkeypatch.setenv("CADENCE_TORCH_TRANSPORT", transport)
    brain = composed(apical=False)
    host = brain.learner.brain
    dev = on(host, device)
    plan = dev._torch.plan
    assert plan.kind == transport and plan.still is not None
    assert dev.to_dict()["torch_transport"]["still_products"] > 0
    n = host.connectome.n
    rng = np.random.default_rng(steps)
    first, second = rng.uniform(-0.5, 1.0, (2, 4, n))
    a, b = host.settle_batch(first, steps=7), dev.settle_batch(first, steps=7)
    same(a, b)
    for _ in range(3):  # a new drive from a warm state, then the same again from where it ended
        a, b = host.settle_batch(second, steps=steps, state=a), dev.settle_batch(
            second, steps=steps, state=b)
        same(a, b)
        np.testing.assert_allclose(dev.residual(second, b), host.residual(second, a),
                                   rtol=0, atol=1e-12)
    for tolerance in (1e-3, 1e-6, 1e-30):
        a, b = host.settle_batch(first, steps=steps, state=a, tolerance=tolerance), (
            dev.settle_batch(first, steps=steps, state=b, tolerance=tolerance))
        same(a, b)
    target = np.zeros((4, n))
    target[:, brain.learner.output_index[0]] = 1.0
    nudge = brain.learner.nudge_for(target, 0.1, None, None)
    a, b = host.settle_batch(first, steps=steps, state=a, nudge=nudge), dev.settle_batch(
        first, steps=steps, state=b, nudge=nudge)
    same(a, b)
    np.testing.assert_allclose(dev.residual(first, b, nudge=nudge),
                               host.residual(first, a, nudge=nudge), rtol=0, atol=1e-12)


@pytest.mark.parametrize("steps", [2, 3, 9, 34])
def test_a_mask_or_a_nudge_on_a_source_makes_every_product(device, steps):
    brain = composed(apical=False)
    host = brain.learner.brain
    dev = on(host, device)
    kernel = dev._torch
    n = host.connectome.n
    source = np.flatnonzero(kernel.plan.structure.source)
    rng = np.random.default_rng(3)
    drive = rng.uniform(-0.5, 1.0, (3, n))
    keep = np.ones(n)
    keep[source[0]] = 0.5
    a, b = host.settle_batch(drive, steps=steps, mask=keep), dev.settle_batch(
        drive, steps=steps, mask=keep)
    same(a, b)
    assert not kernel._workspace(3).configs[("shared", None, False, None)].quiet
    mask = np.zeros(n)
    mask[source[:2]] = 1.0
    nudge = cd.Nudge(rng.uniform(0, 1, (3, n)), mask, 0.2)
    a2, b2 = host.settle_batch(drive, steps=steps, nudge=nudge), dev.settle_batch(
        drive, steps=steps, nudge=nudge)
    same(a2, b2)
    assert any(not cfg.quiet for cfg in kernel._workspace(3).configs.values())


def test_still_products_stay_out_of_a_compartment_adaptation_and_a_smaller_step(device):
    brain = composed(apical=True)
    assert on(brain.learner.brain, device)._torch.plan.still is None
    assert on(composed(apical=False, adaptation=True).learner.brain, device)._torch.plan.still is None
    graph = composed(apical=False).learner.brain
    halved = cd.NeuralGraph(graph.connectome, graph.neuron_model.replace(dt=0.5), efficacy=graph.efficacy,
                            bias=graph.bias, log_gain=graph.log_gain, backend="torch", device=device)
    assert halved._torch.plan.still is None


@pytest.mark.parametrize("transport", ["grouped", "blocks"])
def test_held_still_products_serve_only_their_own_source_neurons(device, transport, monkeypatch):
    from cadence import brain as module

    monkeypatch.setenv("CADENCE_TORCH_TRANSPORT", transport)
    monkeypatch.setattr(module, "_STILL_KEPT", 0)
    brain = composed(apical=False, seed=2)
    host = brain.learner.brain
    dev = on(host, device)
    kernel = dev._torch
    n = host.connectome.n
    rng = np.random.default_rng(14)
    first, second = rng.uniform(-0.5, 1.0, (2, 4, n))
    ws_of = lambda: kernel._workspace(4)  # noqa: E731
    a, b = host.settle_batch(first, steps=9), dev.settle_batch(first, steps=9)
    same(a, b)
    assert ws_of().reuse == 0
    # the same drive from where it ended: no product of a source neuron is made
    a, b = host.settle_batch(first, steps=9, state=a), dev.settle_batch(first, steps=9, state=b)
    held = dev.residual(first, b)  # on the device, before anything reads the state on the host
    same(a, b)
    assert ws_of().reuse == 2
    np.testing.assert_allclose(held, host.residual(first, a), rtol=0, atol=1e-12)
    # a new drive: sweep 0 reads the held products, sweep 1 makes its own
    a, b = host.settle_batch(second, steps=9, state=a), dev.settle_batch(second, steps=9, state=b)
    assert ws_of().reuse == 1
    held = dev.residual(second, b)
    same(a, b)
    np.testing.assert_allclose(held, host.residual(second, a), rtol=0, atol=1e-12)
    # a check of another state, then a settle from a state nothing held
    other = rng.uniform(-0.5, 1.0, (4, n))
    start = host.settle_batch(other, steps=9)
    fetched = cd.BrainState(np.array(start.v), np.array(start.activation),
                            np.array(start.adaptation), start.steps)
    a, b = host.settle_batch(second, steps=2, state=start), dev.settle_batch(
        second, steps=2, state=fetched)
    same(a, b)
    # a mask and a nudge on a source neuron leave nothing held
    keep = np.ones(n)
    keep[np.flatnonzero(kernel.plan.structure.source)[0]] = 0.5
    a, b = host.settle_batch(second, steps=9, state=a, mask=keep), dev.settle_batch(
        second, steps=9, state=b, mask=keep)
    same(a, b)
    assert ws_of().still_version == -1
    a, b = host.settle_batch(first, steps=1, state=a), dev.settle_batch(first, steps=1, state=b)
    same(a, b)
    assert ws_of().still_version == -1
    # new parameters: what was held is not read
    a, b = host.settle_batch(first, steps=9, state=a), dev.settle_batch(first, steps=9, state=b)
    same(a, b)
    host2 = host.with_parameters(efficacy=np.asarray(host.efficacy) * 0.9)
    dev = dev._with_device_parameters(
        torch.from_numpy(np.asarray(host2.efficacy).copy()).to(kernel.device, kernel.param_dtype),
        kernel.bias_param,
    )
    a, b = host2.settle_batch(first, steps=9, state=a), dev.settle_batch(first, steps=9, state=b)
    held = dev.residual(first, b)  # on the device, before anything reads the state on the host
    same(a, b)
    np.testing.assert_allclose(held, host2.residual(first, a), rtol=0, atol=1e-12)


def test_a_settle_of_no_sweeps_rests_where_the_state_is_unless_a_mask_moves_it(device):
    brain = composed(apical=False)
    host = brain.learner.brain
    dev = on(host, device)
    n = host.connectome.n
    rng = np.random.default_rng(23)
    drive = rng.uniform(-0.5, 1.0, (3, n))
    a, b = host.settle_batch(drive, steps=9), dev.settle_batch(drive, steps=9)
    rests = dev.settle_batch(drive, steps=0, state=b)
    assert rests.steps == 0 and rests.device["s"] is b.device["s"]
    np.testing.assert_array_equal(rests.activity_change, np.zeros(3))
    same(host.settle_batch(drive, steps=0, state=a), rests)
    # a mask republishes the activation; a settle under it does not rest for the next
    keep = np.ones(n)
    keep[3] = 0.0
    a2, b2 = host.settle_batch(drive, steps=0, state=a, mask=keep), dev.settle_batch(
        drive, steps=0, state=b, mask=keep)
    same(a2, b2)
    a3, b3 = host.settle_batch(drive, steps=5, state=a, mask=keep), dev.settle_batch(
        drive, steps=5, state=b, mask=keep)
    assert not b3.device["unmasked"]
    a4, b4 = host.settle_batch(drive, steps=0, state=a3), dev.settle_batch(drive, steps=0, state=b3)
    assert b4.device["s"] is not b3.device["s"]
    same(a4, b4)
    # a reset row returns to rest
    kept = dev._torch.keep_rows(b.device, np.array([True, False, True]))
    assert "unmasked" not in kept


def test_still_products_can_be_turned_off(device, monkeypatch):
    brain = composed(apical=False)
    host = brain.learner.brain
    with_still = on(host, device)
    monkeypatch.setenv("CADENCE_STILL_PRODUCTS", "0")
    without = on(host, device)
    assert with_still._torch.plan.still is not None and without._torch.plan.still is None
    drive = np.random.default_rng(1).uniform(-0.5, 1.0, (4, host.connectome.n))
    a, b = with_still.settle_batch(drive, steps=40), without.settle_batch(drive, steps=40)
    assert a.steps == b.steps
    np.testing.assert_allclose(a.activation, b.activation, rtol=0, atol=1e-13)


def test_learning_steps_match_the_host(device):
    brains = [composed(apical=True, seed=4) for _ in range(2)]
    brains[1].learner.brain = on(brains[1].learner.brain, device)
    rng = np.random.default_rng(11)
    n = brains[0].learner.brain.connectome.n
    for _ in range(4):
        drive = rng.uniform(-0.3, 1.0, (6, n))
        labels = rng.integers(0, 3, (6, 2))
        reports = [brain.learner.step(drive, labels)[1] for brain in brains]
        for key in ("free_steps", "nudged_steps", "opposite_steps"):
            assert reports[0][key] == reports[1][key]
    host, dev = brains[0].learner.brain, brains[1].learner.brain
    np.testing.assert_allclose(dev.efficacy, host.efficacy, rtol=0, atol=ATOL)
    np.testing.assert_allclose(dev.bias, host.bias, rtol=0, atol=ATOL)


def test_learning_steps_match_the_host_with_a_projections_sums_by_reductions(device, monkeypatch):
    from cadence import learning

    monkeypatch.setattr(learning, "_SPAN_VALUES", 0)
    brains = [composed(apical=True, seed=6) for _ in range(2)]
    brains[1].learner.brain = on(brains[1].learner.brain, device)
    rng = np.random.default_rng(12)
    n = brains[0].learner.brain.connectome.n
    assert brains[0].learner.config.normalize_relative
    for _ in range(4):
        drive = rng.uniform(-0.3, 1.0, (6, n))
        labels = rng.integers(0, 3, (6, 2))
        for brain in brains:
            brain.learner.step(drive, labels)
    kernel = brains[1].learner.brain._torch
    assert all(group[3] is not None for group in brains[1].learner._projections_on_device(kernel))
    host, dev = brains[0].learner.brain, brains[1].learner.brain
    np.testing.assert_allclose(dev.efficacy, host.efficacy, rtol=0, atol=ATOL)
    np.testing.assert_allclose(dev.bias, host.bias, rtol=0, atol=ATOL)


def test_a_float32_brain_learns_and_solves_without_a_float64_copy(device):
    brain = composed(apical=True, seed=7)
    brain.learner.brain = on(brain.learner.brain, device, precision="float32")
    rng = np.random.default_rng(13)
    n = brain.learner.brain.connectome.n
    state = None
    for _ in range(3):
        drive = rng.uniform(-0.3, 1.0, (4, n))
        solved = brain.learner.brain.equilibrate(drive, budget=64, tolerance=1e-3, state=state)
        state = solved.state
        brain.learner.step(drive, rng.integers(0, 3, (4, 2)))
        state = None
    kernel = brain.learner.brain._torch
    assert kernel._reference is None
    # asked plainly, the float64 copy answers; asked below the floor, it decides
    drive = rng.uniform(-0.3, 1.0, (4, n))
    free = brain.learner.brain.settle_batch(drive, steps=20)
    own = brain.learner.brain.residual(drive, free, own_precision=True)
    assert kernel._reference is None
    plain = brain.learner.brain.residual(drive, free)
    assert kernel._reference is not None
    np.testing.assert_allclose(own, plain, rtol=0, atol=1e-3)
    gain = brain.learner.brain.apical_gain(drive, free, own_precision=True)
    np.testing.assert_allclose(gain, brain.learner.brain.apical_gain(drive, free), rtol=0, atol=1e-3)


def test_trajectory_and_recording_paths_match_the_plain_settle(device):
    brain = composed(apical=True, adaptation=True)
    dev = on(brain.learner.brain, device)
    rng = np.random.default_rng(1)
    drive = rng.uniform(-0.5, 1.0, (2, dev.connectome.n))
    plain = dev.settle_batch(drive, steps=20, tolerance=1e-5)
    traced = dev.settle_batch(drive, steps=20, tolerance=1e-5, trajectory=True)
    assert plain.steps == traced.steps == len(traced.trajectory)
    np.testing.assert_array_equal(traced.trajectory[-1], traced.activation)
    np.testing.assert_allclose(traced.activation, plain.activation, rtol=0, atol=1e-13)


def test_a_reported_state_is_never_a_workspace_buffer(device):
    brain = composed(apical=False)
    dev = on(brain.learner.brain, device)
    drive = np.full((1, dev.connectome.n), 0.3)
    first = dev.settle_batch(drive, steps=5)
    kept = np.array(first.activation, copy=True)
    change = np.array(first.activity_change, copy=True)
    dev.settle_batch(drive * 2, steps=7)
    np.testing.assert_array_equal(first.activation, kept)
    np.testing.assert_array_equal(first.activity_change, change)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")


@cuda
def test_graph_replay_is_bitwise_the_eager_settle(monkeypatch):
    brain = composed(apical=True)
    rng = np.random.default_rng(4)
    drive = rng.uniform(-0.5, 1.0, (8, brain.learner.brain.connectome.n))
    target = np.zeros_like(drive)
    target[:, brain.learner.output_index[0]] = 1.0
    states = []
    for flag in ("0", "1"):
        monkeypatch.setenv("CADENCE_CUDA_GRAPHS", flag)
        dev = on(brain.learner.brain, "cuda:0")
        out = []
        for _ in range(3):  # the first runs eagerly, then the graphs are captured and replayed
            free = dev.settle_batch(drive, steps=60, tolerance=1e-7)
            nudge = brain.learner.nudge_for(target, 0.1, None, dev.apical_gain(drive, free))
            out.append((free, dev.settle_batch(drive, steps=32, state=free, nudge=nudge)))
        assert (dev._torch.graph_stats["replays"] > 0) == (flag == "1")
        assert dev._torch.graph_stats["failed"] is None
        states.append(out)
    for (f0, n0), (f1, n1) in zip(*states, strict=True):
        assert f0.steps == f1.steps
        np.testing.assert_array_equal(f0.activation, f1.activation)
        np.testing.assert_array_equal(n0.activation, n1.activation)
        np.testing.assert_array_equal(f0.activity_change, f1.activity_change)


@cuda
def test_graphs_are_captured_again_after_every_capture_was_dropped():
    brain = composed(apical=False)
    dev = on(brain.learner.brain, "cuda:0")
    kernel = dev._torch
    drive = np.random.default_rng(7).uniform(-0.5, 1.0, (4, dev.connectome.n))
    for _ in range(4):
        dev.settle_batch(drive, steps=40)
    ws = kernel._workspace(4)
    assert kernel.graph_stats["captured"] > 0 and ws.graphs
    expected = dev.settle_batch(drive, steps=40).activation
    ws.clear_captures()
    for _ in range(4):
        again = dev.settle_batch(drive, steps=40)
    assert kernel.graph_stats["failed"] is None and ws.graphs
    np.testing.assert_allclose(again.activation, expected, rtol=0, atol=1e-13)


@cuda
def test_the_device_settle_is_deterministic():
    brain = composed(apical=True)
    rng = np.random.default_rng(6)
    drive = rng.uniform(-0.5, 1.0, (16, brain.learner.brain.connectome.n))
    runs = [on(brain.learner.brain, "cuda:0").settle_batch(drive, steps=40) for _ in range(2)]
    np.testing.assert_array_equal(runs[0].activation, runs[1].activation)


@cuda
def test_triton_groups_match_the_batched_products(monkeypatch):
    pytest.importorskip("triton")
    from cadence import blocks

    # an anatomy of many small groups: each of 24 rows of eight hears its own window
    pre, post = [], []
    for g in range(24):
        for r in range(8):
            for c in range(12):
                pre.append((g * 3 + c) % 64)
                post.append(64 + g * 8 + r)
    connectome = cd.Connectome.from_synapses(
        64 + 192, pre=pre, post=post,
        sign=np.random.default_rng(0).uniform(-0.3, 0.3, len(pre)),
        populations={"input": range(64), "hidden": range(64, 256)},
    )
    del blocks
    model = cd.learning_neuron_model()
    drive = np.random.default_rng(1).uniform(0, 1, (5, connectome.n))
    host = cd.NeuralGraph(connectome, model)
    results = []
    for flag in ("0", "1"):
        monkeypatch.setenv("CADENCE_TRITON", flag)
        dev = cd.NeuralGraph(connectome, model, backend="torch", device="cuda:0")
        assert (dev._torch.plan.small is not None) == (flag == "1")
        results.append(dev.settle_batch(drive, steps=30, tolerance=1e-8))
    expected = host.settle_batch(drive, steps=30, tolerance=1e-8)
    for result in results:
        same(expected, result)


@cuda
def test_the_apical_launch_matches_the_products(monkeypatch):
    pytest.importorskip("triton")
    brain = composed(apical=True)
    host = brain.learner.brain
    rng = np.random.default_rng(12)
    drive = rng.uniform(-0.5, 1.0, (5, host.connectome.n))
    expected = host.settle_batch(drive, steps=40, tolerance=1e-7)
    launched = on(host, "cuda:0")
    assert launched._torch._pools is not None
    same(expected, launched.settle_batch(drive, steps=40, tolerance=1e-7))
    monkeypatch.setenv("CADENCE_TRITON", "0")
    products = on(host, "cuda:0")
    assert products._torch._pools is None
    same(expected, products.settle_batch(drive, steps=40, tolerance=1e-7))


@cuda
@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("batch", [1, 7, 70])
@pytest.mark.parametrize("weighted", [False, True])
def test_the_nudge_launch_matches_the_host(precision, batch, weighted):
    pytest.importorskip("triton")
    brain = composed(apical=False, seed=5)
    host = brain.learner.brain
    dev = on(host, "cuda:0", precision=precision)
    n = host.connectome.n
    rng = np.random.default_rng(batch)
    drive = rng.uniform(-0.4, 0.9, (batch, n))
    outputs = brain.learner.output_index
    mask = np.zeros(n)
    mask[outputs] = 1.0
    mask[outputs[1]] = 0.5
    target = np.zeros((batch, n))
    target[:, outputs[0]] = 1.0
    options: dict[str, object] = {"weight": rng.uniform(-1.0, 2.0, batch)} if weighted else {}
    groups = np.full(n, -1)
    groups[outputs[:3]], groups[outputs[3:]] = 0, 1
    nudge = cd.Nudge(target, mask, 0.2, softmax_temperature=0.3, groups=groups, **options)  # type: ignore[arg-type]
    a, b = host.settle_batch(drive, steps=20), dev.settle_batch(drive, steps=20)
    ap, bp = host.settle_batch(drive, steps=20, state=a, nudge=nudge), dev.settle_batch(
        drive, steps=20, state=b, nudge=nudge)
    launched = dev._torch._workspace(batch).nudges
    assert all(buffers.launch is not None for buffers in launched.values())
    tolerance = 1e-10 if precision == "float64" else 2e-3
    np.testing.assert_allclose(bp.activation, ap.activation, rtol=0, atol=tolerance)
    if precision == "float64":
        np.testing.assert_allclose(dev.residual(drive, bp, nudge=nudge),
                                   host.residual(drive, ap, nudge=nudge), rtol=0, atol=1e-12)


@pytest.mark.parametrize("device_name", devices())
def test_float32_settles_stay_near_the_host_and_check_in_float64(device_name):
    brain = composed(apical=True)
    host = brain.learner.brain
    dev = on(host, device_name, precision="float32")
    assert dev._torch.dtype == torch.float32
    rng = np.random.default_rng(8)
    drive = rng.uniform(-0.5, 1.0, (6, host.connectome.n))
    a, b = host.settle_batch(drive, steps=60), dev.settle_batch(drive, steps=60)
    np.testing.assert_allclose(b.activation, a.activation, rtol=0, atol=2e-3)
    # the residual and the apical gain of a float32 state are the float64 reference's, read
    # on the device by a float64 copy of the kernel
    fetched = cd.BrainState(np.array(b.v), np.array(b.activation), np.array(b.adaptation), b.steps)
    np.testing.assert_allclose(dev.residual(drive, b), host.residual(drive, fetched),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(dev.apical_gain(drive, b), host.apical_gain(drive, fetched),
                               rtol=0, atol=1e-12)
