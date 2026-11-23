"""A learned code on the torch backend. The region's own phase, the settle of its sub-network
and the tests of its step, runs in float64 on the brain's device (on the host where the device
has no float64, MPS), so a float64 brain there reads as the float64 host brain within rounding,
moment by moment, and takes the step the eigenvalues give. A phase the solver damps can end a
stalled attempt one check apart on the two, so the sweeps are not compared. Read on a torch CPU
device and on CUDA where there is one."""

from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import pytest

import cadence as cd
from cadence.brain import Nudge

torch = pytest.importorskip("torch")

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
MPS = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
TIGHT = 1e-9  # float64 on both sides: rounding, through a life's learning steps
INPUTS, ACTIONS = 12, 5
OPTIONS = {
    "modules": (32, 16), "learned_code": 24, "attention": True, "recall_heads": 2,
    "learning_fast_settle": False,  # the compiled CPU path's other solve
}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def settings(options: dict) -> dict:
    """The brain's options with ``options`` over them; a None leaves the name out."""
    return {name: value for name, value in {**OPTIONS, **options}.items() if value is not None}


def on(device: str, precision: str = "float64", seed: int = 3, **options) -> cd.Brain:
    return cd.Brain.compose(
        INPUTS, ACTIONS, seed=seed, backend="torch", device=device, precision=precision,
        **settings(options),
    )


def host(seed: int = 3, **options) -> cd.Brain:
    return cd.Brain.compose(INPUTS, ACTIONS, seed=seed, backend="cpu", **settings(options))


def tables(brain: cd.Brain) -> dict[str, np.ndarray]:
    """What the region and the lessons move, on the host."""
    region = brain.learned_code
    return {
        "dictionary": region.dictionary, "predictor": region.weights,
        "thresholds": region.thresholds, "efficacy": np.asarray(brain.brain.efficacy),
        "bias": np.asarray(brain.brain.bias),
    }


def counts(brain: cd.Brain) -> dict[str, int]:
    """The region's work without its sweeps: a stalled attempt of a damped phase ends at two
    repeated states at rounding, which the host and a device can reach one check apart."""
    return {name: value for name, value in brain.learned_code.work.items() if name != "sweeps"}


def same(a: cd.Brain, b: cd.Brain, atol: float = TIGHT) -> None:
    ours, theirs = tables(a), tables(b)
    for name, value in ours.items():
        np.testing.assert_allclose(theirs[name], value, rtol=0, atol=atol, err_msg=name)
    assert b.learned_code.halvings == a.learned_code.halvings
    assert counts(b) == counts(a)


def held_by_kernels(brain: cd.Brain) -> None:
    """The kernels hold what the tables give: the sub-network its lateral weights and
    thresholds, the whole graph the thresholds as the code units' biases in its precision and
    the region's synapses at zero."""
    region, graph = brain.learned_code, brain.brain
    np.testing.assert_array_equal(region.graph.efficacy, region.lateral_efficacy())
    np.testing.assert_array_equal(region.graph.bias, -region.thresholds)
    precision = "float32" if graph._torch.dtype == torch.float32 else "float64"
    np.testing.assert_array_equal(
        graph.bias[brain.code_index], (-region.thresholds).astype(precision)
    )
    for synapses, _, _ in brain._code_synapses.values():
        assert not graph.efficacy[synapses].any()


def observations(rng: np.random.Generator, rows: int) -> np.ndarray:
    return np.eye(INPUTS)[rng.integers(INPUTS, size=rows)] + 0.01 * rng.random((rows, INPUTS))


def taught(brains: tuple[cd.Brain, ...], rng: np.random.Generator, moments: int, rows: int = 2):
    """Lessons and greedy acts, the same for every brain; each act equal to the first brain's."""
    for _ in range(moments):
        x, labels = observations(rng, rows), rng.integers(ACTIONS, size=rows)
        for brain in brains:
            brain.lesson(x, labels)
        answer = brains[0].act(x, greedy=True)
        for brain in brains[1:]:
            np.testing.assert_array_equal(brain.act(x, greedy=True), answer)


# -- where the phase runs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_the_regions_phase_runs_in_float64_on_the_brains_device(device, precision):
    brain = on(device, precision)
    region, kernel = brain.learned_code, brain.brain._torch
    assert region.device == kernel.device
    assert region.graph.backend == "torch" and region.graph._torch.device == kernel.device
    assert region.graph._torch.dtype == torch.float64  # whatever the brain's precision
    assert kernel.dtype == (torch.float64 if precision == "float64" else torch.float32)
    assert host().learned_code.device is None
    # the tables stay on the host, and the kernel holds what they give
    assert isinstance(region.dictionary, np.ndarray)
    np.testing.assert_array_equal(region.graph.efficacy, region.lateral_efficacy())
    np.testing.assert_array_equal(region.graph.bias, -region.thresholds)


def test_mps_has_no_float64_and_is_refused_by_the_region():
    region = host().learned_code
    with pytest.raises(ValueError, match="float64, which MPS does not have"):
        region.place("mps")
    assert region.device is None and region.graph.backend == "cpu"


@pytest.mark.skipif(not MPS, reason="no MPS device")
def test_a_brain_on_mps_keeps_the_regions_phase_on_the_host(tmp_path):
    brain = cd.Brain.compose(
        INPUTS, ACTIONS, seed=3, backend="torch", device="mps", **OPTIONS
    )
    region = brain.learned_code
    assert region.device is None and region.graph.backend == "cpu"
    rng = np.random.default_rng(0)
    for _ in range(6):
        x = observations(rng, 2)
        brain.lesson(x, rng.integers(ACTIONS, size=2))
        assert brain.act(x, greedy=True).shape == (2,)
        np.testing.assert_array_equal(
            brain.brain.bias[brain.code_index], (-region.thresholds).astype("float32")
        )
    assert region.work["updates"] == 6 and region.device is None
    loaded = cd.Brain.load(brain.save(tmp_path / "brain.npz"), backend="torch", device="mps")
    assert loaded.learned_code.device is None
    np.testing.assert_array_equal(loaded.learned_code.thresholds, region.thresholds)
    assert loaded.act(observations(rng, 2), greedy=True).shape == (2,)


@pytest.mark.parametrize("device", DEVICES)
def test_the_phase_moves_to_the_host_and_back_in_a_life(device):
    reference, placed = host(), on(device)
    rng = np.random.default_rng(0)
    taught((reference, placed), rng, 4)
    region = placed.learned_code
    stepping = region.halvings
    region.place(None)
    assert region.device is None and region.graph.backend == "cpu"
    assert region.halvings == stepping
    assert region.graph.neuron_model.dt == reference.learned_code.graph.neuron_model.dt
    taught((reference, placed), rng, 4)
    region.place(device)
    assert region.device == placed.brain._torch.device and region.halvings == reference.learned_code.halvings
    taught((reference, placed), rng, 4)
    same(reference, placed)
    # a region of a brain on the CPU takes a device too: its phase alone runs there
    reference.learned_code.place(device)
    taught((placed, reference), rng, 3)
    same(placed, reference)


# -- a life against the host


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rows", [1, 3])
def test_a_life_of_lessons_and_acts_reads_as_on_the_host(device, rows):
    reference, placed = host(), on(device)
    taught((reference, placed), np.random.default_rng(0), 30, rows)
    same(reference, placed)
    assert reference.learned_code.work["updates"] == 30
    held_by_kernels(placed)


@pytest.mark.parametrize("device", DEVICES)
def test_a_life_of_outcomes_reads_as_on_the_host(device):
    reference, placed = host(), on(device)
    rng = np.random.default_rng(1)
    for moment in range(25):
        x = observations(rng, 2)
        reward = rng.normal(size=2) if moment else None
        done = rng.random(2) < 0.2 if moment else None
        answer = reference.step(x, reward=reward, done=done)
        np.testing.assert_array_equal(placed.step(x, reward=reward, done=done), answer)
    same(reference, placed)
    x = observations(rng, 2)
    np.testing.assert_allclose(placed.value(x), reference.value(x), rtol=0, atol=TIGHT)


@pytest.mark.parametrize("device", DEVICES)
def test_a_living_brain_waits_and_stays_calm_as_on_the_host(device):
    reference, placed = host(arousal=True), on(device, arousal=True)
    rng = np.random.default_rng(2)
    for moment in range(40):
        x = observations(rng, 1)
        reward = [float(rng.normal())] if moment else None
        answer = reference.live(x, reward=reward)
        np.testing.assert_array_equal(placed.live(x, reward=reward), answer)
        assert placed.last_arousal["mode"] == reference.last_arousal["mode"]
        if moment % 7 == 3:
            x = observations(rng, 1)
            reference.wait(x)
            placed.wait(x)
    same(reference, placed)


BESIDE = {
    "no recall": {"recall": False, "attention": False, "recall_heads": None},
    "the founder's recall": {"recall_founder": True, "attention": False, "recall_heads": None},
    "a compartment": {"modules": (24, 16, 16), "apical": 0.25, "teaching": "deep"},
    "normalized teaching": {"teaching": "normalized"},
    "observers": {"observers": (8,)},
    "the episodic choice": {"memory_choice": True, "attention": False, "recall_heads": None},
    "efference and two histories": {"efference_amplitude": 1.0, "history_decays": (0.0, 0.8)},
    "two slots": {"slots": (3, 2)},
    "fields": {"learned_code_fields": (8, 4)},
    "maturity": {"maturity": True},
    "a fading store": {"memory_fade": 1.2e-4},
    "qualified teaching": {"learning_qualified": True},
    "rehearsal": {"learning_rehearsal": 2},
    "tokens and the command": {"recall_tokens": range(5), "recall_gain": 1.0, "recall_action": True},
    "an observation wider than the code": {"learned_code": 8},
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(BESIDE))
def test_every_part_beside_the_region_lives_as_on_the_host(device, name):
    """Lessons with greedy acts, then steps that learn from outcomes and labels, an imagined
    moment and the memories' key: each as the host brain of the same composition."""
    options = BESIDE[name]
    reference, placed = host(**options), on(device, **options)
    assert placed.learned_code.device is not None
    rng = np.random.default_rng(0)
    slots = options.get("slots")

    def labels() -> np.ndarray:
        if slots:
            return np.stack([rng.integers(k, size=2) for k in slots], axis=1)
        return rng.integers(ACTIONS, size=2)

    for moment in range(10):
        x, y = observations(rng, 2), labels()
        if moment < 4:
            for brain in (reference, placed):
                brain.lesson(x, y)
            answer = reference.act(x, greedy=True), placed.act(x, greedy=True)
        else:
            reward = rng.normal(size=2) if reference.pending_feedback else None
            teacher = y if moment % 2 else None
            answer = (reference.step(x, reward=reward, teacher=teacher),
                      placed.step(x, reward=reward, teacher=teacher))
        np.testing.assert_array_equal(answer[1], answer[0])
    same(reference, placed)
    x = observations(rng, 2)
    np.testing.assert_allclose(
        placed.observation_code(x), reference.observation_code(x), rtol=0, atol=TIGHT
    )
    for a, b in zip(reference.imagine([x]), placed.imagine([x]), strict=True):
        np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=TIGHT)


@pytest.mark.parametrize("device", DEVICES)
def test_independent_samples_fit_and_score_as_on_the_host(device):
    reference, placed = host(), on(device)
    rng = np.random.default_rng(1)
    x, labels = observations(rng, 24), rng.integers(ACTIONS, size=24)
    assert placed.fit(x, labels, epochs=2, batch=8) == reference.fit(x, labels, epochs=2, batch=8)
    np.testing.assert_array_equal(placed.predict(x), reference.predict(x))
    assert placed.accuracy(x, labels) == reference.accuracy(x, labels)
    same(reference, placed)


def test_surprise_keeps_its_own_refusal_of_a_device():
    with pytest.raises(ValueError, match="surprise reads each neuron's residual on the host"):
        on("cpu", surprise=True)


@pytest.mark.parametrize("device", DEVICES)
def test_a_float32_brain_starts_as_the_host_and_keeps_its_kernels(device):
    """The phase is float64, and it reads the context and its warm start from the brain's
    float32 state and learns from float32 activations: the first moment reads as the host's
    within float32's rounding, and a float32 life is a run of its own, so the moments after
    it are held to what the kernels hold and compared with no other run."""
    reference, placed = host(), on(device, "float32")
    rng = np.random.default_rng(0)
    x, labels = observations(rng, 2), rng.integers(ACTIONS, size=2)
    for brain in (reference, placed):
        brain.lesson(x, labels)
        brain.act(x, greedy=True)
    ours, theirs = tables(reference), tables(placed)
    for name, value in ours.items():  # one moment of this brain, far above float32's rounding
        np.testing.assert_allclose(theirs[name], value, rtol=0, atol=1e-5, err_msg=name)
    for _ in range(12):
        x = observations(rng, 2)
        placed.lesson(x, rng.integers(ACTIONS, size=2))
        assert placed.act(x, greedy=True).shape == (2,)
        held_by_kernels(placed)
    assert placed.learned_code.graph._torch.dtype == torch.float64
    assert placed.learned_code.work["updates"] == 13


@pytest.mark.parametrize("device", DEVICES)
def test_the_two_controls_act_on_a_device_as_on_the_host(device):
    reference, placed = host(), on(device)
    for brain in (reference, placed):
        brain.learned_code.top_down = 0.0  # without the standing boundary
    taught((reference, placed), np.random.default_rng(3), 6)
    same(reference, placed)
    assert placed.learned_code.standing() is None
    for brain in (reference, placed):
        brain.learned_code.top_down = 2.0
        brain.learned_code.learning = False  # every table still
    kept = {name: value.copy() for name, value in tables(placed).items() if name != "efficacy"}
    x = observations(np.random.default_rng(4), 2)
    np.testing.assert_array_equal(placed.act(x, greedy=True), reference.act(x, greedy=True))
    for name in ("dictionary", "predictor", "thresholds"):
        np.testing.assert_array_equal(tables(placed)[name], kept[name])


# -- the step


def eigenvalue_halvings(code, v: np.ndarray) -> int:
    """The halvings the smallest coupling eigenvalue over the rows needs, read directly."""
    slopes = np.maximum(code.model.slope_at(v), 0.0)
    root = np.sqrt(slopes)
    coupling = root[:, :, None] * (code.model.gain * code.lateral())[None] * root[:, None, :]
    index = np.arange(code.units)
    coupling[:, index, index] -= code.top_down * slopes
    mu = float(np.linalg.eigvalsh(coupling)[:, 0].min())
    need = code.model.dt * (1.0 - mu) / code.STEP_MARGIN
    return 0 if need <= 1.0 else int(np.ceil(np.log2(need)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("group", [1, 3, None])
def test_the_step_on_a_device_is_the_one_the_eigenvalues_give(device, group, monkeypatch):
    """From every step held, over states from silent to steep and dictionaries from the drawn
    one to one direction shared by every unit, with the rows read one at a time, three at a
    time and at the region's own bound: the host's answer, made on the device (NumPy's
    factorization and eigenvalues are out of reach while the device reads) in groups of at
    most that many rows."""
    rng = np.random.default_rng(11)
    answers, read = set(), {"factorized": [], "solved": []}
    factorize, solve = torch.linalg.cholesky_ex, torch.linalg.eigvalsh
    monkeypatch.setattr(torch.linalg, "cholesky_ex", lambda matrix: (
        read["factorized"].append(len(matrix)), factorize(matrix))[1])
    monkeypatch.setattr(torch.linalg, "eigvalsh", lambda matrix: (
        read["solved"].append(len(matrix)), solve(matrix))[1])

    def unreachable(*args, **kwargs):
        raise AssertionError("the host's linear algebra read a step the device holds")

    for seed, overlap in enumerate((0.0, 0.3, 0.7, 1.0)):
        placed, reference = on(device, seed=seed).learned_code, host(seed=seed).learned_code
        if group is not None:
            placed.DEVICE_BYTES = 32 * placed.units * placed.units * group
        bound = max(1, placed.DEVICE_BYTES // (32 * placed.units * placed.units))
        del read["factorized"][:], read["solved"][:]
        one = rng.normal(size=(placed.inputs, 1))
        dictionary = (1.0 - overlap) * placed.dictionary + overlap * one
        dictionary = dictionary / np.linalg.norm(dictionary, axis=0)
        for code in (placed, reference):
            code.take(dictionary, code.weights, code.thresholds)
        for scale, rows in ((0.1, 1), (1.0, 1), (1.0, 4), (3.0, 4), (8.0, 16)):
            for _ in range(3):
                v = rng.normal(placed.model.threshold, scale, (rows, placed.units))
                expected = eigenvalue_halvings(reference, v)
                answers.add(expected)
                for held in {0, 1, 2, 3, 5, expected, expected + 1, max(expected - 1, 0)}:
                    placed.restep(held)
                    reference.restep(held)
                    assert reference.stability(v) == expected
                    with monkeypatch.context() as patch:
                        patch.setattr(np.linalg, "cholesky", unreachable)
                        patch.setattr(np.linalg, "eigvalsh", unreachable)
                        assert placed.stability(v) == expected
        # the factorizations by the group, an eigenvalue read by half of it
        most = max(read["factorized"]), max(read["solved"])
        assert most[0] <= bound and most[1] <= max(1, bound // 2)
        if group is None:  # the region's own bound takes these rows together
            assert most[0] > 3 and most[1] == 16
        else:
            assert most == (group, max(1, group // 2))
    assert len(answers) >= 2


@pytest.mark.parametrize("device", DEVICES)
def test_the_step_follows_a_dictionary_written_in_place(device):
    """The lateral weights the device keeps for the step's tests are made again when the
    dictionary reads otherwise, a column written in place without ``install`` included."""
    placed, reference = on(device).learned_code, host().learned_code
    rng = np.random.default_rng(4)
    v = rng.normal(placed.model.threshold, 3.0, (4, placed.units))
    assert placed.stability(v) == reference.stability(v)
    kept = placed._coupled[1]
    assert placed.stability(v) == reference.stability(v) and placed._coupled[1] is kept
    shared = rng.normal(size=placed.inputs)
    for code in (placed, reference):
        code.dictionary[:, : code.units // 2] = (shared / np.linalg.norm(shared))[:, None]
    assert placed.stability(v) == reference.stability(v) == eigenvalue_halvings(reference, v)
    assert placed._coupled[1] is not kept


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("failing", ["cholesky_ex", "eigvalsh"])
def test_a_device_that_fails_in_the_steps_tests_leaves_the_step_to_the_host(
    device, failing, monkeypatch
):
    """A solver error or an exhausted device in the step's tests does not end a life: the host
    reads the same step, and a warning names the failure."""
    reference, placed = host(), on(device)
    rng = np.random.default_rng(0)
    taught((reference, placed), rng, 3)

    def fails(*args, **kwargs):
        raise RuntimeError("cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED\nwhen calling it")

    monkeypatch.setattr(torch.linalg, failing, fails)
    region = placed.learned_code
    for held in (0, 1, 4):  # a held step asks the factorizations, another one the eigenvalues
        region.restep(held)
        reference.learned_code.restep(held)
        v = rng.normal(region.model.threshold, 3.0, (4, region.units))
        expected = reference.learned_code.stability(v)
        if failing == "eigvalsh" and expected == held:
            assert region.stability(v) == expected  # the factorizations settled it
            continue
        with pytest.warns(RuntimeWarning, match="read on the host: the device failed "
                                                r"\(cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED\)"):
            assert region.stability(v) == expected
    for brain in (reference, placed):
        brain.learned_code.restep(reference.learned_code.halvings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        taught((reference, placed), rng, 6)  # the life goes on, as the host's
    same(reference, placed)
    assert region.device is not None


@pytest.mark.parametrize("device", DEVICES)
def test_the_host_answers_alone_for_what_is_no_failure_of_the_device(device, monkeypatch):
    """What is not a batch of finite potentials is the host's to refuse, as it refuses it,
    with no warning of a device; an error that names a defect and no device passes through."""
    placed, reference = on(device).learned_code, host().learned_code
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for bad in (np.full((2, placed.units), np.nan), rng.random((2, placed.units + 1))):
            raised = []
            for code in (reference, placed):
                with pytest.raises(Exception) as refusal:
                    code.stability(bad)
                raised.append(type(refusal.value))
            assert raised[0] is raised[1] and issubclass(raised[0], ValueError)
        assert placed.stability(np.zeros((0, placed.units))) == reference.stability(
            np.zeros((0, placed.units))
        )

    def unfinished(*args, **kwargs):
        raise NotImplementedError("no such solver")

    monkeypatch.setattr(torch.linalg, "cholesky_ex", unfinished)
    placed.restep(1)
    v = rng.normal(placed.model.threshold, 0.1, (2, placed.units))
    with pytest.raises(NotImplementedError, match="no such solver"):
        placed.stability(v)


def _peak_of(device: str, call) -> int | None:
    """The most bytes ``call`` held on ``device`` beyond what was held before it: CUDA's own
    count, or on the CPU device the allocations the profiler records; None where it records
    none."""
    if device == "cuda":
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated() - before)
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as recorded:
        call()
    try:  # the profiler's own record of every allocation and release
        moves = sorted(
            (event.start_ns(), event.nbytes())
            for event in recorded.profiler.kineto_results.events() if event.name() == "[memory]"
        )
    except AttributeError:
        return None
    if not moves:
        return None
    live = peak = 0
    for _, size in moves:
        live += size
        peak = max(peak, live)
    return int(peak)


@pytest.mark.parametrize("device", DEVICES)
def test_the_steps_tests_work_within_their_bytes(device):
    """Factorized or solved for their eigenvalues, the rows of a batch are read in groups
    whose couplings, factors and workspace stay within ``DEVICE_BYTES`` (the batch's slopes
    beside them). On CUDA at a width where the solver's workspace is the matrices' and not its
    fixed part."""
    units, rows, group = (512, 64, 4) if device == "cuda" else (128, 64, 4)
    region = cd.Brain.compose(
        40, ACTIONS, seed=0, backend="torch", device=device, precision="float64",
        modules=(16,), learned_code=units,
    ).learned_code
    region.DEVICE_BYTES = 32 * units * units * group
    v = np.random.default_rng(0).normal(region.model.threshold, 1.0, (rows, units))
    needed = region.stability(v)
    whole = 8 * rows * units * units  # every row's coupling at once
    assert region.DEVICE_BYTES * 3 < whole
    for held in (needed, needed + 3, 0):  # held by factorizations, and read by eigenvalues
        region.restep(held)
        region.stability(v)  # the lateral weights rest on the device from here
        peak = _peak_of(device, lambda: region.stability(v))
        if peak is None:
            pytest.skip("this torch records no allocations on the CPU device")
        # a group's couplings at the least, and no more than the region's bytes
        assert 8 * group * units * units <= peak, (held, peak)
        assert peak <= region.DEVICE_BYTES + 4 * 8 * rows * units, (held, peak)


@pytest.mark.parametrize("device", DEVICES)
def test_a_halved_step_settles_on_the_kernel_that_holds_the_tables(device):
    """A step halved settles on a kernel of that step over the same parameters, and its states
    rest on the first; a learning step reaches every one of them."""
    placed, reference = on(device).learned_code, host().learned_code
    base = placed._base._torch
    rng = np.random.default_rng(5)
    stimulus = rng.random((3, placed.units))
    for halvings in (0, 2, 1, 0, 3):
        placed.restep(halvings)
        reference.restep(halvings)
        assert placed.graph._torch.root is base and placed._base._torch is base
        assert placed.graph.neuron_model.dt == reference.graph.neuron_model.dt
        a = reference.settle(stimulus, None, budget=512 << halvings, tolerance=1e-9)
        b = placed.settle(stimulus, None, budget=512 << halvings, tolerance=1e-9)
        assert np.all(a.qualified) and np.all(b.qualified)
        np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=1e-12)
        sensory, context = rng.random((3, placed.inputs)), rng.random((3, placed.context))
        for code, phase in ((reference, a), (placed, b)):
            assert code.learn(sensory, np.asarray(phase.state.activation), context)
        np.testing.assert_array_equal(placed.graph.efficacy, placed.lateral_efficacy())
        np.testing.assert_array_equal(placed.graph.bias, -placed.thresholds)
        np.testing.assert_allclose(placed.dictionary, reference.dictionary, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="halvings must be a nonnegative integer"):
        placed.restep(-1)


# -- the kernel takes the tables in place


@pytest.mark.parametrize("device", DEVICES)
def test_a_learning_step_writes_the_kernels_and_builds_none(device):
    brain = on(device)
    rng = np.random.default_rng(6)
    brain.act(observations(rng, 2), greedy=True)
    region = brain.learned_code
    whole, own = brain.brain._torch, region._base._torch
    workspaces = (whole._workspaces, own._workspaces)
    before = region.thresholds.copy()
    for _ in range(4):
        brain.act(observations(rng, 2), greedy=True)
    assert not np.array_equal(region.thresholds, before)
    assert brain.brain._torch is whole and region._base._torch is own
    assert (whole._workspaces, own._workspaces) == workspaces
    # the whole graph: the thresholds as the code units' biases, the region's synapses at zero
    np.testing.assert_array_equal(brain.brain.bias[brain.code_index], -region.thresholds)
    for synapses, _, _ in brain._code_synapses.values():
        assert brain.brain.efficacy[synapses].tobytes() == np.zeros(len(synapses)).tobytes()
    # the region's own: its lateral weights and thresholds
    np.testing.assert_array_equal(region.graph.efficacy, region.lateral_efficacy())
    np.testing.assert_array_equal(region.graph.bias, -region.thresholds)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_a_moved_synapse_of_the_region_goes_back_to_zero_on_the_device(device, precision):
    """As on the host: a synapse of the region off zero, or at a negative zero, is a zero again
    after the next install, and a region at zero keeps the kernel's weights as they are."""
    brain = on(device, precision)
    brain.act(observations(np.random.default_rng(1), 2), greedy=True)
    graph = brain.brain
    scale = graph._torch.scale
    assert brain._installed(graph)._torch.scale is scale
    synapses = brain._code_synapses["lateral"][0]
    for value in (0.5, -0.0):
        efficacy = np.array(graph.efficacy)
        efficacy[synapses[3]] = value
        again = brain._installed(graph.with_parameters(efficacy=efficacy))
        assert again.efficacy[synapses].tobytes() == np.zeros(len(synapses)).tobytes()
        np.testing.assert_array_equal(
            again.bias[brain.code_index], (-brain.learned_code.thresholds).astype(precision)
        )


@pytest.mark.parametrize("device", DEVICES)
def test_a_refused_learning_step_leaves_the_region_and_its_kernel(device):
    region = on(device).learned_code
    graph, base = region.graph, region._base
    kept = region.dictionary, region.weights, region.thresholds, dict(region.work)
    efficacy = np.array(graph.efficacy)
    rng = np.random.default_rng(2)
    sensory, activation = rng.random((2, region.inputs)), rng.random((2, region.units))
    with pytest.raises(ValueError, match="finite rows"):
        region.learn(sensory, activation, np.full((2, region.context), np.nan))
    region.rate = 1e308  # a proposal that overflows
    with pytest.raises(ValueError, match="must remain finite"):
        region.learn(1e200 * sensory, 1e200 * activation, None)
    assert region.graph is graph and region._base is base
    assert (region.dictionary, region.weights, region.thresholds) == kept[:3]
    assert region.work == kept[3]
    np.testing.assert_array_equal(region.graph.efficacy, efficacy)


@pytest.mark.parametrize("device", DEVICES)
def test_an_install_the_device_fails_in_returns_the_kernel_as_it_was(device):
    """The kernel is written in place before the sub-network of a halved step takes the same
    parameters; where that fails, the kernel holds the tables it held, with the region."""
    region = on(device).learned_code
    region.restep(2)
    kernel = region._base._torch
    kept = region.dictionary, region.weights, region.thresholds, region.graph, region._base
    scale, bias = kernel.host_scale(), kernel.host_bias()
    rng = np.random.default_rng(0)
    sensory, activation = rng.random((2, region.inputs)), rng.random((2, region.units))
    context = rng.random((2, region.context))
    stepped = kernel.stepped

    def fails(model):
        raise RuntimeError("CUDA out of memory")

    kernel.stepped = fails
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            region.learn(sensory, activation, context)
    finally:
        kernel.stepped = stepped
    assert (region.dictionary, region.weights, region.thresholds) == kept[:3]
    assert region.graph is kept[3] and region._base is kept[4]
    np.testing.assert_array_equal(kernel.host_scale(), scale)
    np.testing.assert_array_equal(kernel.host_bias(), bias)
    np.testing.assert_array_equal(kernel.host_scale(), region.lateral_efficacy())
    assert region.learn(sensory, activation, context)  # and the same step takes afterwards
    np.testing.assert_array_equal(region.graph.efficacy, region.lateral_efficacy())
    np.testing.assert_array_equal(region._base._torch.host_bias(), -region.thresholds)
    # tables taken whole (a checkpoint's) return with the kernel too
    kept = region.dictionary, region.weights, region.thresholds, region.graph, region._base
    scale, bias = kernel.host_scale(), kernel.host_bias()
    other = on(device, seed=8).learned_code
    kernel.stepped = fails
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            region.take(other.dictionary, other.weights, other.thresholds + 0.25)
    finally:
        kernel.stepped = stepped
    assert (region.dictionary, region.weights, region.thresholds) == kept[:3]
    assert region.graph is kept[3] and region._base is kept[4]
    np.testing.assert_array_equal(kernel.host_scale(), scale)
    np.testing.assert_array_equal(kernel.host_bias(), bias)


@pytest.mark.parametrize("device", DEVICES)
def test_the_standing_boundary_goes_to_the_device_once(device):
    """Its arrays cannot be written, so the kernel's buffers hold it while it lives; a nudge
    whose arrays can be written is filled at every settle, as before."""
    region = on(device).learned_code
    standing = region.standing()
    assert standing is region.standing()
    assert not any(a.flags.writeable for a in (standing.target, standing.mask, standing.anchor,
                                               standing.anchor_gain))
    kernel = region.graph._torch
    uploads = []
    upload = kernel._staging.upload
    kernel._staging.upload = lambda array, dtype: uploads.append(np.shape(array)) or upload(
        array, dtype)
    stimulus = np.random.default_rng(0).random((3, region.units))
    region.settle(stimulus, None, budget=512, tolerance=1e-9)
    first = len(uploads)
    del uploads[:]
    phase = region.settle(stimulus, None, budget=512, tolerance=1e-9)
    assert phase.residual_checks > 1
    assert len(uploads) == 1 < first  # the stimulus alone: a settle from rest uploads no state
    held = [buffers for buffers in kernel._workspace(3).nudges.values()]
    assert len(held) == 1 and held[0].filled() is standing
    writable = Nudge(standing.target, standing.mask, 0.0, anchor=standing.anchor,
                     anchor_gain=standing.anchor_gain)
    del uploads[:]
    region.graph.settle_batch(stimulus, steps=2, nudge=writable)
    assert held[0].filled is None and len(uploads) > 1
    host_region = host().learned_code
    assert host_region.standing() is not host_region.standing()  # the host's, made at each call


# -- copies, imagined branches, refusals


@pytest.mark.parametrize("device", DEVICES)
def test_a_copy_holds_its_own_kernels_and_learns_apart(device):
    reference, placed = host(), on(device)
    taught((reference, placed), np.random.default_rng(0), 4)
    twins = copy.deepcopy(reference), copy.deepcopy(placed)
    region, twin = placed.learned_code, twins[1].learned_code
    assert twin.device == region.device and twin.connectome is region.connectome
    assert twin.standing() is not region.standing() and twin.standing() is twin.standing()
    assert not twin.standing().anchor_gain.flags.writeable
    # the copied moment's phase rests on the copy's kernel: no third kernel is made for it
    for brain in (placed, twins[1]):
        phase = brain._coded_moment[2]
        assert phase.device["holder"] is brain.learned_code._base._torch
    assert twin._base._torch is not region._base._torch
    assert twin.graph._torch.root is twin._base._torch
    assert twins[1].brain._torch is not placed.brain._torch
    kept = {name: value.copy() for name, value in tables(placed).items()}
    taught(twins, np.random.default_rng(7), 5)  # the copies learn; the originals do not move
    same(*twins)
    for name, value in tables(placed).items():
        np.testing.assert_array_equal(value, kept[name], err_msg=name)
    np.testing.assert_array_equal(region.graph.efficacy, region.lateral_efficacy())
    taught((reference, placed), np.random.default_rng(8), 3)
    same(reference, placed)


@pytest.mark.parametrize("device", DEVICES)
def test_an_imagined_branch_leaves_the_region_as_it_found_it(device, tmp_path):
    reference, placed = host(), on(device)
    rng = np.random.default_rng(0)
    taught((reference, placed), rng, 4)
    region = placed.learned_code
    for level in (region.halvings, region.halvings + 2):
        placed._step_region(level)
        reference._step_region(level)
        held = region.graph, region.last, region.to_dict()
        before = placed.save(tmp_path / "before.npz").read_bytes()
        sequence = [observations(rng, 2) for _ in range(3)]
        ours, theirs = reference.imagine(sequence), placed.imagine(sequence)
        for a, b in zip(ours, theirs, strict=True):
            assert np.all(b.qualified)
            np.testing.assert_allclose(b.state.activation, a.state.activation, rtol=0, atol=TIGHT)
        assert region.graph is held[0] and region.last is held[1] and region.to_dict() == held[2]
        assert placed.save(tmp_path / "after.npz").read_bytes() == before
    taught((reference, placed), rng, 3)
    same(reference, placed)


@pytest.mark.parametrize("device", DEVICES)
def test_a_refused_moment_restores_the_regions_step_and_graph(device):
    reference, placed = host(), on(device)
    rng = np.random.default_rng(0)
    taught((reference, placed), rng, 3)
    region = placed.learned_code
    budget = placed.learner.config.free_steps
    held = region.halvings, region.graph, region.last
    kept = {name: value.copy() for name, value in tables(placed).items()}
    for brain in (reference, placed):
        brain.retune(learning_free_steps=1)
        with pytest.raises(RuntimeError, match="the code region did not settle"):
            brain.act(observations(np.random.default_rng(9), 2), greedy=True)
        brain.retune(learning_free_steps=budget)
    assert (region.halvings, region.graph, region.last) == held
    for name, value in tables(placed).items():
        np.testing.assert_array_equal(value, kept[name], err_msg=name)
    taught((reference, placed), rng, 3)
    same(reference, placed)


# -- saved files across backends


@pytest.mark.parametrize("device", DEVICES)
def test_a_file_saved_on_either_backend_continues_on_the_other(device, tmp_path):
    reference, placed = host(), on(device)
    rng = np.random.default_rng(0)
    taught((reference, placed), rng, 8)
    for brain in (reference, placed):  # a step the next moments do not ask for travels too
        brain._step_region(brain.learned_code.halvings + 1)
    stepping = placed.learned_code.halvings
    from_host = reference.save(tmp_path / "host.npz")
    from_device = placed.save(tmp_path / "device.npz")
    with np.load(from_host) as a, np.load(from_device) as b:
        assert sorted(a.files) == sorted(b.files)
        for name in ("learned_code/dictionary", "learned_code/weights", "learned_code/thresholds"):
            np.testing.assert_allclose(b[name], a[name], rtol=0, atol=TIGHT)
        ours, theirs = json.loads(str(a["brain"])), json.loads(str(b["brain"]))
        assert theirs["learned_code"] == ours["learned_code"]  # the device is the brain's
    brains = (
        reference, placed,
        cd.Brain.load(from_device, backend="cpu"),
        cd.Brain.load(from_host, backend="torch", device=device, precision="float64"),
        cd.Brain.load(from_device, backend="torch", device=device),
    )
    assert [brain.learned_code.device is None for brain in brains] == [
        True, False, True, False, False
    ]
    assert cd.Brain.load(from_device).learned_code.device is None  # no backend named: the host
    for brain in brains:
        assert brain.learned_code.halvings == stepping
        assert brain.learned_code.graph.neuron_model.dt == reference.learned_code.graph.neuron_model.dt
        np.testing.assert_array_equal(
            brain.learned_code.graph.efficacy, brain.learned_code.lateral_efficacy()
        )
    taught(brains, rng, 8)
    for brain in brains[1:]:
        same(reference, brain)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("last", ["act", "lesson"])
def test_a_float32_brains_file_loads_on_every_backend(device, last, tmp_path):
    """A float32 device holds the code units' biases as float32 holds the negated thresholds,
    and its file says so: the float64 host reads it, after an act and after a lesson."""
    placed = on(device, "float32")
    rng = np.random.default_rng(0)
    taught((placed,), rng, 4)
    if last == "lesson":
        placed.lesson(observations(rng, 2), rng.integers(ACTIONS, size=2))
    path = placed.save(tmp_path / "brain.npz")
    with np.load(path) as data:
        thresholds = data["learned_code/thresholds"]
        np.testing.assert_array_equal(thresholds, placed.learned_code.thresholds)  # as float64
    loaded = (
        cd.Brain.load(path, backend="cpu"),
        cd.Brain.load(path, backend="torch", device=device, precision="float32"),
        cd.Brain.load(path, backend="torch", device=device, precision="float64"),
    )
    assert [brain.learned_code.device is None for brain in loaded] == [True, False, False]
    x, labels = observations(rng, 2), rng.integers(ACTIONS, size=2)
    for brain in loaded:
        np.testing.assert_array_equal(brain.learned_code.thresholds, thresholds)
        np.testing.assert_array_equal(brain.learned_code.dictionary, placed.learned_code.dictionary)
        assert brain.learned_code.halvings == placed.learned_code.halvings
        assert brain.act(x, greedy=True).shape == (2,)
        brain.lesson(x, labels)
        assert brain._code_installed(brain.brain)


@pytest.mark.parametrize("device", DEVICES)
def test_a_file_whose_tables_disagree_with_its_synapses_is_refused_on_the_device(device, tmp_path):
    placed = on(device)
    taught((placed,), np.random.default_rng(0), 2)
    path = placed.save(tmp_path / "brain.npz")
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["learned_code/thresholds"] = arrays["learned_code/thresholds"] + 0.125
    np.savez(tmp_path / "moved.npz", **arrays)
    for backend in ({"backend": "cpu"}, {"backend": "torch", "device": device}):
        with pytest.raises(ValueError, match="disagree with the saved synapses and biases"):
            cd.Brain.load(tmp_path / "moved.npz", **backend)


# -- grow, graft, retune, lives


@pytest.mark.parametrize("device", DEVICES)
def test_a_grown_brain_keeps_the_region_on_the_device(device):
    reference, placed = host(), on(device)
    rng = np.random.default_rng(0)
    taught((reference, placed), rng, 5)
    grown = reference.grow(recall_heads=3), placed.grow(recall_heads=3)
    assert grown[1].placement["backend"] == "torch"
    assert grown[1].learned_code.device == placed.learned_code.device
    assert grown[1].learned_code._base._torch is not placed.learned_code._base._torch
    taught(grown, rng, 5)
    same(*grown)


@pytest.mark.parametrize("device", DEVICES)
def test_a_graft_and_a_retune_keep_the_region_on_the_device(device):
    def pair(seed: int) -> tuple[cd.Brain, cd.Brain]:
        options = {"modules": (16, 16), "learned_code": 16, "recall": False, "seed": seed,
                   "learning_fast_settle": False}
        return (
            cd.Brain.compose(INPUTS, ACTIONS, backend="cpu", **options),
            cd.Brain.compose(INPUTS, ACTIONS, backend="torch", device=device,
                             precision="float64", **options),
        )

    donors, hosts = pair(9), pair(3)
    rng = np.random.default_rng(0)
    taught(donors, rng, 4)
    taught(hosts, rng, 4)
    grafted = tuple(
        brain.graft(donor, ["module_0", "association"])
        for brain, donor in zip(hosts, donors, strict=True)
    )
    assert grafted[1].learned_code.device == hosts[1].learned_code.device
    with pytest.raises(ValueError, match="land on its code region"):
        hosts[1].graft(donors[1], ["prefrontal", "association"])
    for brain in grafted:
        brain.retune(actor_eta=0.02)
    with pytest.raises(ValueError, match="learned_code_fields is a construction setting"):
        grafted[1].retune(learned_code_fields=(6, 6))
    with pytest.raises(ValueError, match="backend"):
        grafted[1].retune(backend="cpu")
    taught(grafted, rng, 5)
    same(*grafted)


def test_lives_refuse_a_learned_code_by_name_on_torch_too():
    with pytest.raises(ValueError, match="Lives holds the parts its lives axis covers: no learned"):
        cd.Lives(INPUTS, ACTIONS, 2, backend="torch", device="cpu", modules=(16,), learned_code=8)
