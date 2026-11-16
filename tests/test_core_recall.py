"""The recall memory in the native core (``cadence.core.parts.recall``): its state crosses and
comes back the same bytes on both sides, and each method a life calls gives in the core what the
Python gives, within 1e-9 (where the core library is built). Lives of small composed brains:
two heads as Wild's animals hold, one head with the address map taught by reward, three heads
without next ports over a bank longer than the position table, and a memory of two streams
driven by hand, one stream begun again and one episode ended."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cadence import Brain
from cadence.core import Core, _default_library, pack
from cadence.core.parts import part
from cadence.stream import Recall

RECALL = part("recall")
LIVES = {
    "two heads": dict(recall_heads=2, recall_capacity=6),
    "one head, address reward": dict(recall_heads=1, recall_capacity=5,
                                     recall_address_reward=0.5),
    "three heads, past the table": dict(recall_heads=3, recall_capacity=40, recall_next=False),
}
MOMENTS = {"two heads": 14, "one head, address reward": 12, "three heads, past the table": 44}
DRIVEN = "two streams, one begun again"


def composed(**options: Any) -> Any:
    return Brain.compose(senses={"eyes": 30, "body": 4}, actions=3, code=40, modules=(8,),
                         backend="cpu", seed=3, maturity=True, arousal={"youth": 4}, **options)


def life(name: str) -> tuple[Recall, list[Any]]:
    """A small brain's memory and every call of its methods (``METHODS``) in the brain's life:
    the state before, the arguments, what it returned and the state after."""
    brain = composed(**LIVES.get(name, LIVES["two heads"]))
    calls: list[Any] = []
    kept = {name: getattr(Recall, name) for name in RECALL.METHODS}

    def recorded(method: str) -> Any:
        def call(self: Recall, *args: Any, **kwargs: Any) -> Any:
            before = RECALL.state_of(self)
            value = kept[method](self, *args, **kwargs)
            arguments, result = RECALL.METHODS[method]
            calls.append((method, before, arguments(*args, **kwargs), result(value),
                          RECALL.state_of(self)))
            return value
        return call

    rng = np.random.default_rng(7)
    try:
        for method in kept:
            setattr(Recall, method, recorded(method))
        if name == DRIVEN:
            driven(brain.recall, brain.brain, rng)
        for k in range(MOMENTS.get(name, 0)):
            brain.live(rng.random((1, 34)), **({} if k == 0 else {"reward": rng.normal(size=1)}))
    finally:
        for method, value in kept.items():
            setattr(Recall, method, value)
    return brain.recall, calls


def driven(memory: Recall, graph: Any, rng: np.random.Generator) -> None:
    """Two streams by hand: each moment read, its eligibility taken and stepped, then written;
    the second stream begins again after the sixth moment and the first ends an episode after
    the tenth, so the slots a stream holds and its successors are masked."""
    n, width = graph.connectome.n, memory.key_width

    def state() -> Any:
        return SimpleNamespace(activation=rng.normal(size=(2, n)))

    for k in range(16):
        x = rng.random((2, width)) * (rng.random((2, width)) < 0.3)
        memory.stimulate(rng.normal(size=(2, n)), x)
        rows = memory.contrast_rows(graph, state(), state())
        if rows is not None:
            proposal = memory.propose(rows.mean(axis=0))
            if proposal is not None:
                memory.accept(proposal)
        memory.witness(x, state())
        if k == 5:
            memory.reset(2, np.array([1]))
        if k == 9:
            memory.end(2, np.array([0]))


@pytest.fixture(scope="module", params=[*LIVES, DRIVEN])
def calls(request: Any) -> tuple[Recall, list[Any]]:
    return life(request.param)


def test_life_covers_every_method(calls):
    memory, calls = calls
    assert {c[0] for c in calls} == set(RECALL.METHODS)
    assert any("last.event" in c[1] for c in calls)
    # the reads of every head are held with several heads
    assert any("reads.writes" in c[1] for c in calls) == (memory.heads > 1)
    if any(len(c[1]["told"]) == 2 for c in calls):  # by hand: slots a stream does not hold, an episode ended
        assert any(not c[1]["held"].all() for c in calls if c[0] == "stimulate")
        assert any(c[1]["ended"].any() for c in calls if c[0] == "witness")
        assert any(c[1]["ended"].any() and not c[1]["held"].all() for c in calls
                   if c[0] == "contrast_rows")


def test_state_round_trips_in_python(calls):
    memory, calls = calls
    for _, before, _, _, after in calls:
        for state in (before, after):
            RECALL.restore(memory, {k: np.array(v) for k, v in state.items()})
            assert pack(RECALL.state_of(memory)) == pack(state)


core_built = pytest.mark.skipif(not _default_library().exists(),
                                reason="the core library is not built")


@core_built
def test_state_round_trips_in_the_core(calls):
    core = Core()
    for _, before, _, _, after in calls[1]:
        for state in (before, after):
            back, out = core.call("recall", "state", state, {})
            assert not out and pack(back) == pack(state)


@core_built
def test_every_call_in_the_core_gives_the_python(calls):
    core = Core()
    for method, before, args, theirs_out, theirs_after in calls[1]:
        after, out = core.call("recall", method, before, args)
        for ours, theirs in ((out, theirs_out), (after, theirs_after)):
            assert set(ours) == set(theirs), method
            for key, value in theirs.items():
                x, y = np.asarray(ours[key]), np.asarray(value)
                assert x.shape == y.shape and x.dtype == y.dtype, (method, key)
                if x.dtype == np.float64:
                    np.testing.assert_allclose(x, y, rtol=0, atol=1e-9, err_msg=f"{method} {key}")
                else:
                    assert np.array_equal(x, y), (method, key)


def test_refused_options_name_themselves():
    brain = Brain.compose(senses={"eyes": 30, "body": 4}, actions=3, code=40, modules=(8,),
                          backend="cpu", seed=3, maturity=True, arousal={"youth": 4},
                          recall_strength=True)
    with pytest.raises(ValueError, match="strength"):
        RECALL.state_of(brain.recall)
