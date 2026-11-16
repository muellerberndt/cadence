"""The records in the native core (``cadence.core.parts``: ``longterm``, ``bound``,
``synaptic`` and ``memories``): their state crosses and comes back the same bytes on both
sides, and each method in the core gives what the Python gives (where the core library is
built)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from cadence.bound import BoundRecords
from cadence.core import Core, Refused, _default_library, pack
from cadence.core.parts import part
from cadence.longterm import LongTermMemory
from cadence.memories import record_action
from cadence.memory import SynapticMemory
from cadence.plasticity import Maturity

LONGTERM, BOUND, SYNAPTIC, MEMORIES = (part("longterm"), part("bound"), part("synaptic"),
                                       part("memories"))
SENSES = {"eyes": (0, 4), "heard": (4, 8), "food": {"what": (8, 10), "where": (10, 12)},
          "voice": (12, 16)}
WIDTH, INPUTS = 8, 16


def stores() -> list[LongTermMemory]:
    """A store with strength along a life: empty, filling, full (a record replaced), records
    told again (revised and blended), kept and forgotten."""
    rng = np.random.default_rng(3)
    held = LongTermMemory(6, 5, 5, cosine=True, revise=0.95, blend=True, keep=3.0, forget=6.0,
                          seed=4)
    out = [copy.deepcopy(held)]
    base = rng.normal(size=(4, 6))
    for k in range(40):
        key = base[k % 4] + (0.01 * rng.normal(size=6) if k % 3 else rng.normal(size=6))
        held.write(key, rng.normal(size=5), surprise=None if k % 2 else float(rng.random()))
        for _ in range(9 if k == 20 else 1):  # a while without writes: records forgotten
            held.moment()
        if k % 5 == 0:
            out.append(copy.deepcopy(held))
    return out


class Coder:
    """A fixed code of the records' rows (``code_of`` of ``BoundRecords.named``)."""

    def __init__(self, bound: BoundRecords, seed: int) -> None:
        self.bound = bound
        self.projection = np.random.default_rng(seed).normal(size=(INPUTS, WIDTH))

    def code(self, rows: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(rows) @ self.projection, 0.0)


def row(rng: np.random.Generator, voice: bool, sound: int | None = None) -> np.ndarray:
    out = np.zeros(INPUTS)
    out[0:4] = rng.random(4) * (rng.random(4) < 0.7)
    if rng.random() < 0.6:
        out[4:8] = np.eye(4)[rng.integers(4)] if sound is None else np.eye(4)[sound]
    out[8:10] = np.eye(2)[rng.integers(2)]
    out[10:12] = rng.random(2)
    if voice:
        out[12:16] = np.eye(4)[rng.integers(4) if sound is None else sound]
    return out


def records() -> list[tuple[BoundRecords, Coder]]:
    """Bound records with a voice along a life: moments heard, said and paid or refused."""
    rng = np.random.default_rng(8)
    held = BoundRecords(SENSES, WIDTH, INPUTS, 9, voice="voice", heard="heard", seed=11)
    coder = Coder(held, 2)
    out = [copy.deepcopy((held, coder))]
    for k in range(60):
        rows = np.stack([row(rng, voice=k % 3 == 0, sound=k % 4 if k % 2 else None)])
        held.hear(rows, coder.code(rows), surprise=rng.exponential(1.0, 1) - 0.5,
                  acted=k % 7 != 0)
        held.pay([1.0 if k % 4 == 0 else (-1.0 if k % 4 == 1 else 0.0)])
        if k % 6 == 0:
            held.say(rows, coder.code(rows))
            out.append(copy.deepcopy((held, coder)))
    return out


def memories() -> list[SynapticMemory]:
    """Place records with the surprise gate and a maturity, written by one stream and by
    several, some rows masked."""
    rng = np.random.default_rng(6)
    held = SynapticMemory(np.arange(6), np.arange(6, 10), decay=0.9, rate=0.2,
                          consolidation=0.05, rule="delta")
    held.gate_noise = np.zeros(held.consolidated.shape)
    held.maturity = Maturity(scale=3.0, recovery=200.0)
    out = [copy.deepcopy(held)]
    for k in range(24):
        batch = 1 if k < 12 else 3
        keys = rng.normal(size=(batch, 6)) * (rng.random((batch, 6)) < 0.6)
        record_action(held, keys, rng.integers(0, 4, batch), rng.normal(size=batch),
                      rng.exponential(1.0, batch), None if k % 5 else rng.random(batch) < 0.6,
                      np.zeros(1, dtype=np.int64))
        held.maturity.use += 0.5
        if k % 3 == 0:
            out.append(copy.deepcopy(held))
    return out


def round_trip(module: Any, held: Any) -> None:
    state = module.state_of(held)
    fresh = copy.deepcopy(held)
    module.restore(fresh, {k: np.array(v) for k, v in state.items()})
    assert pack(module.state_of(fresh)) == pack(state)


def test_state_round_trips_in_python():
    for held in stores():
        round_trip(LONGTERM, held)
    for held, _ in records():
        round_trip(BOUND, held)
    for held in memories():
        round_trip(SYNAPTIC, held)
        round_trip(MEMORIES, held)


def test_states_cover_the_records_laws():
    seen = [LONGTERM.state_of(h) for h in stores()]
    assert any(s["kept"].any() for s in seen) and any(s["gone"].any() for s in seen)
    assert int(seen[-1]["state"][1]) > int(seen[-1]["state"][0])  # replaced and revised
    bound = [BOUND.state_of(h) for h, _ in records()]
    assert any(s["refused"].any() for s in bound) and any((s["paid"] > 0).any() for s in bound)
    assert {int(c) for s in bound for c in s["said_from"]} >= {0, 2}


core_built = pytest.mark.skipif(not _default_library().exists(),
                                reason="the core library is not built")


@core_built
def test_state_round_trips_in_the_core():
    core = Core()
    states = [("longterm", LONGTERM.state_of(h)) for h in stores()]
    states += [("bound", BOUND.state_of(h)) for h, _ in records()]
    states += [(name, part(name).state_of(h)) for h in memories()
               for name in ("synaptic", "memories")]
    for name, state in states:
        after, out = core.call(name, "state", state, {})
        assert not out and pack(after) == pack(state)


def same(ours: dict, theirs: dict) -> None:
    assert list(ours) == list(theirs)
    for key, value in theirs.items():
        x, y = np.asarray(ours[key]), np.asarray(value)
        assert x.shape == y.shape and x.dtype == y.dtype, key
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=0, atol=1e-12, err_msg=key)
        else:
            assert np.array_equal(x, y), key


def made(core: Core, name: str, module: Any, held: Any, method: str, *args: Any,
         **kwargs: Any) -> Any:
    """The call made in the core on ``held``'s state and in the Python on ``held`` itself,
    their results and the states after compared; what the Python's call returned."""
    before = module.state_of(held)
    arguments, result = module.METHODS[method]
    value = getattr(held, method)(*args, **kwargs)
    after, out = core.call(name, method, before, arguments(*args, **kwargs))
    same(out, result(value))
    same(after, module.state_of(held))
    return value


@core_built
def test_longterm_calls_in_the_core():
    core, rng = Core(), np.random.default_rng(1)
    for held in stores():
        for _ in range(6):
            key = rng.normal(size=6)
            made(core, "longterm", LONGTERM, held, "write", key, rng.normal(size=5),
                 surprise=float(rng.random()))
            made(core, "longterm", LONGTERM, held, "write", key + 1e-3, rng.normal(size=5))
            if held.count:
                made(core, "longterm", LONGTERM, held, "write", rng.normal(size=6),
                     rng.normal(size=5), recognised=int(rng.integers(-1, held.count)))
            made(core, "longterm", LONGTERM, held, "moment")


@core_built
def test_bound_calls_in_the_core():
    core, rng = Core(), np.random.default_rng(2)
    for held, coder in records():
        for k in range(8):
            rows = np.stack([row(rng, voice=k % 2 == 0, sound=k % 4), row(rng, voice=False)])
            codes = coder.code(rows)
            made(core, "bound", BOUND, held, "complete", rows, codes)
            made(core, "bound", BOUND, held, "recall", rows[0], codes[0])
            made(core, "bound", BOUND, held, "say", rows, codes)
            template = made(core, "bound", BOUND, held, "named", rows[0], coder.code)
            things = [{"codes": {"what": rng.random(2), "where": rng.random(2)}}, {},
                      {"codes": {"what": np.eye(2)[k % 2], "eyes": rng.random(4)}}]
            made(core, "bound", BOUND, held, "salience", template or {"food/what": np.ones(2)},
                 things)
            made(core, "bound", BOUND, held, "hear", rows, codes,
                 surprise=rng.normal(size=2), acted=k % 3 != 0)
            made(core, "bound", BOUND, held, "pay", rng.normal(size=2))


@core_built
def test_synaptic_calls_in_the_core():
    core, rng = Core(), np.random.default_rng(3)
    for held in memories():
        for batch in (1, 3, 1):
            keys = rng.normal(size=(batch, 6)) * (rng.random((batch, 6)) < 0.7)
            drive = rng.normal(size=(batch, 12))
            made(core, "synaptic", SYNAPTIC, held, "recall", keys)
            made(core, "synaptic", SYNAPTIC, held, "stimulate", drive)
            made(core, "synaptic", SYNAPTIC, held, "stimulate", drive, key=keys)
            made(core, "synaptic", SYNAPTIC, held, "held_values")
            mask = rng.random((batch, 4)) < 0.5
            made(core, "synaptic", SYNAPTIC, held, "observe", keys, rng.normal(size=(batch, 4)),
                 rng.random(batch) < 0.8, salience=rng.exponential(1.0, batch),
                 value_mask=mask)
            before = MEMORIES.state_of(held)
            action = rng.integers(0, 4, batch)
            reward, importance = rng.normal(size=batch), rng.exponential(1.0, batch)
            args = (keys, action, reward, importance, None, np.zeros(1, dtype=np.int64))
            record_action(held, *args)
            after, out = core.call("memories", "record_action", before,
                                   MEMORIES.METHODS["record_action"][0](*args))
            assert not out
            same(after, MEMORIES.state_of(held))


@core_built
def test_the_core_refuses_what_the_python_refuses():
    core = Core()
    store, (bound, _), memory = stores()[3], records()[4], memories()[2]
    with pytest.raises(Refused):
        core.call("longterm", "write", LONGTERM.state_of(store),
                  {"key": np.full(6, np.nan), "value": np.zeros(5)})
    with pytest.raises(Refused):
        core.call("longterm", "write", LONGTERM.state_of(store),
                  {"key": np.ones(6), "value": np.zeros(5), "surprise": np.float64(-1.0)})
    with pytest.raises(Refused):
        core.call("bound", "hear", BOUND.state_of(bound),
                  {"rows": np.ones((1, INPUTS)), "codes": np.full((1, WIDTH), np.inf),
                   "acted": np.bool_(True)})
    with pytest.raises(Refused):
        core.call("synaptic", "observe", SYNAPTIC.state_of(memory),
                  {"key": np.ones((2, 6)), "value": np.ones((1, 4))})
    with pytest.raises(Refused):
        core.call("synaptic", "recall", SYNAPTIC.state_of(memory),
                  {"key": np.full((1, 6), np.inf)})


@core_built
def test_the_core_faults_where_the_python_fails_otherwise():
    core = Core()
    memory = memories()[2]
    # an action outside the motor neurons: the Python's IndexError
    with pytest.raises(RuntimeError):
        core.call("memories", "record_action", MEMORIES.state_of(memory),
                  {"keys": np.ones((1, 6)), "action": np.array([9]), "reward": np.ones(1),
                   "importance": np.ones(1), "offsets": np.zeros(1, dtype=np.int64)})
    # a store with a coarse address: the core does not hold it
    coarse = LongTermMemory(4, 3, 40, expansion=8, winners=2, candidates=4, seed=1)
    with pytest.raises(RuntimeError) as held:
        core.call("longterm", "state", LONGTERM.state_of(coarse), {})
    assert not isinstance(held.value, Refused)
