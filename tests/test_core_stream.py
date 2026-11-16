"""The stream's parts in the native core (``cadence.core.parts`` ``code``, ``trace``,
``efference``, ``history`` and ``hearing``): their state crosses and comes back the same bytes
on both sides, and each method in the core gives what the Python gives (where the core library
is built)."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

import cadence as cd
from cadence import fused
from cadence.core import Core, _default_library, pack
from cadence.core.parts import part
from cadence.stream import PatternSeparator

NAMES = {"code": "sparse_code", "trace": "working_memory", "efference": "efference",
         "history": "history", "hearing": "hearing"}


@pytest.fixture(scope="module")
def brain():
    """A small brain with every stream part, a few moments into its life."""
    held = cd.Brain.compose(senses={"a": 24, "b": 12}, actions=3, code=32, modules=(8,),
                            recall_capacity=4, backend="cpu", seed=3, arousal=True,
                            efference_amplitude=1.0)
    rng = np.random.default_rng(1)
    for k in range(6):
        held.live(rng.random((1, 36)), reward=[0.1 * k] if k else None)
    return held


def separators() -> list[PatternSeparator]:
    """A covering code, a centred one, a dense one and a signed sparse one."""
    return [PatternSeparator(36, 64, 8, seed=1, fan_in=8, cover=2),
            PatternSeparator(36, 64, 8, seed=2, center=0.3, fan_in=8, cover=2),
            PatternSeparator(20, 32, 5, seed=3), PatternSeparator(20, 32, 5, seed=4, fan_in=6)]


def objects(brain, name):
    held = getattr(brain, NAMES[name])
    return separators() + [held] if name == "code" else [held]


def same(ours: dict, theirs: dict, tolerance: float = 0.0) -> None:
    assert list(ours) == list(theirs)
    for key, value in theirs.items():
        x, y = np.asarray(ours[key]), np.asarray(value)
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if x.dtype.kind == "f" and tolerance:
            np.testing.assert_allclose(x, y, rtol=0, atol=tolerance, err_msg=key)
        else:
            assert x.tobytes() == y.tobytes(), key


@pytest.mark.parametrize("name", list(NAMES))
def test_state_round_trips_in_python(brain, name):
    for held in objects(brain, name):
        state = part(name).state_of(held)
        fresh = copy.deepcopy(held)
        part(name).restore(fresh, {k: np.array(v) for k, v in state.items()})
        assert pack(part(name).state_of(fresh)) == pack(state)


core_built = pytest.mark.skipif(not _default_library().exists(),
                                reason="the core library is not built")


@core_built
@pytest.mark.parametrize("name", list(NAMES))
def test_state_round_trips_in_the_core(brain, name):
    core = Core()
    for held in objects(brain, name):
        state = part(name).state_of(held)
        after, out = core.call(name, "state", state, {})
        assert not out and pack(after) == pack(state)


def replayed(core, name, held, method, *args, tolerance=0.0, route=None, **kwargs):
    """``method`` of a copy of ``held`` in Python and in the core on its state: the same
    result and the same state after it (a Hearing's method of a route takes the copy's
    ``route``). The copy, moved on."""
    module = part(name)
    theirs = copy.deepcopy(held)
    if route is not None:
        args = (theirs.routes[route], *args)
    before = module.state_of(theirs)
    arguments, result = module.METHODS[method]
    value = getattr(theirs, method)(*args, **kwargs)
    after, out = core.call(name, method, before, arguments(*args, **kwargs))
    same(out, result(value), tolerance)
    same(after, module.state_of(theirs), tolerance)
    return theirs


@core_built
def test_code_in_the_core(brain):
    core = Core()
    rng = np.random.default_rng(2)
    for held in objects(brain, "code"):
        for batch, learn in ((1, False), (1, True), (3, True), (5, False), (0, True)):
            key = rng.normal(size=(batch, held.inputs)) * (rng.random((batch, held.inputs)) < 0.6)
            # the dense projection is a BLAS product natively, summed in order in the core
            tolerance = 1e-12 if not held.fan_in else 0.0
            held = replayed(core, "code", held, "code", key, learn=learn, tolerance=tolerance)


@core_built
def test_traces_in_the_core(brain):
    core = Core()
    rng = np.random.default_rng(3)
    n = brain.connectome.n
    for name in ("trace", "efference"):
        held = getattr(brain, NAMES[name])
        for batch in (1, 3, 3, 2):
            activation = rng.normal(size=(batch, n))
            drive = rng.normal(size=(batch, n))
            if name == "trace":
                held = replayed(core, name, held, "update", SimpleNamespace(activation=activation))
            else:
                command = np.eye(len(held.hidden))[rng.integers(len(held.hidden), size=batch)]
                held = replayed(core, name, held, "issue", command)
            held = replayed(core, name, held, "_source_activation",
                            SimpleNamespace(activation=activation))
            held = replayed(core, name, held, "stimulate", drive)
            held = replayed(core, name, held, "stimulus", rng.normal(size=(batch + 1, n)))
        held = replayed(core, name, held, "reset", 2, np.array([1]))
        held = replayed(core, name, held, "reset", 2, np.array([True, False]))
        held = replayed(core, name, held, "reset", 4)


@core_built
def test_history_in_the_core(brain):
    core = Core()
    rng = np.random.default_rng(4)
    held = brain.history
    n = brain.connectome.n
    for batch in (1, 1, 3, 2):
        held = replayed(core, "history", held, "hear", rng.random((batch, held.width)))
        held = replayed(core, "history", held, "stimulate", rng.normal(size=(batch, n)))
        held = replayed(core, "history", held, "stimulus", rng.normal(size=(batch + 1, n)))
    held.hearing = np.array([0.7])
    held = replayed(core, "history", held, "stimulate", rng.normal(size=(2, n)))
    held = replayed(core, "history", held, "reset", 2, np.array([0]))
    held = replayed(core, "history", held, "reset", 3)


@core_built
def test_hearing_in_the_core(brain):
    core = Core()
    rng = np.random.default_rng(5)
    held = copy.deepcopy(brain.hearing)
    graph = brain.learner.brain
    n = brain.connectome.n
    routes = len(held.routes)
    for _ in range(3):
        held = replayed(core, "hearing", held, "step", rng.normal(size=routes) * 0.1)
    held = replayed(core, "hearing", held, "set", 1.25, route=0)
    held = replayed(core, "hearing", held, "gain", route=routes - 1)
    held = replayed(core, "hearing", held, "gain", route=0)
    # the compiled moment's contrast is a compiled kernel's, where the kernels are compiled
    for compiled in (False, True) if fused.njit is not None else (False,):
        held.compiled_moment = compiled
        plus = SimpleNamespace(activation=rng.normal(size=(1, n)))
        minus = SimpleNamespace(activation=rng.normal(size=(1, n)))
        # the read-back is a BLAS product natively, summed in order in the core
        held = replayed(core, "hearing", held, "contrast_rows", graph, plus, minus, 0.1,
                        tolerance=1e-12)
        movement = rng.normal(size=(1, len(held.association)))
        held = replayed(core, "hearing", held, "_contrast", graph, movement, None, 1,
                        tolerance=1e-12)
        held = replayed(core, "hearing", held, "_contrast", graph, movement, None, 2)


@core_built
def test_the_core_refuses_what_the_python_refuses(brain):
    core = Core()
    code = separators()[0]
    state = part("code").state_of(code)
    for key in (np.full((1, 36), np.nan), np.zeros((1, 35)), np.zeros(36)):
        with pytest.raises(ValueError):
            code.code(key)
        with pytest.raises(RuntimeError):
            core.call("code", "code", state, part("code").METHODS["code"][0](key))
    efference = brain.efference
    with pytest.raises(TypeError):
        efference.update(SimpleNamespace(activation=np.zeros((1, brain.connectome.n))))
    with pytest.raises(RuntimeError):
        core.call("efference", "update", part("efference").state_of(efference),
                  {"activation": np.zeros((1, brain.connectome.n))})
    for command in (np.zeros((1, len(efference.hidden) + 1)), np.full((1, 3), np.inf)):
        with pytest.raises(ValueError):
            copy.deepcopy(efference).issue(command)
        with pytest.raises(RuntimeError):
            core.call("efference", "issue", part("efference").state_of(efference),
                      {"command": command})
    history = brain.history
    for x in (np.zeros((1, history.width + 1)), np.full((1, history.width), np.nan)):
        with pytest.raises(ValueError):
            copy.deepcopy(history).hear(x)
        with pytest.raises(RuntimeError):
            core.call("history", "hear", part("history").state_of(history),
                      {"observations": x})
    hearing = brain.hearing
    with pytest.raises(ValueError):
        copy.deepcopy(hearing).step(np.zeros(len(hearing.routes) + 1))
    with pytest.raises(RuntimeError):
        core.call("hearing", "step", part("hearing").state_of(hearing),
                  {"contrast": np.zeros(len(hearing.routes) + 1)})
