"""A free solve on the compiled CPU path runs in one compiled call: the sweeps, the checks, their
counts and every bit of the launches it stands in for, which a wrapped method still sees."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import cadence as cd
from cadence.brain import Brain as NeuralGraph

pytest.importorskip("numba")


def launches(graph: NeuralGraph, *args, **kwargs) -> cd.Equilibrium:
    """The same solve with the compiled call declined."""
    graph.__dict__["_solves_compiled"] = lambda: False
    try:
        return graph.equilibrate(*args, **kwargs)
    finally:
        del graph.__dict__["_solves_compiled"]


def assert_same(a: cd.Equilibrium, b: cd.Equilibrium) -> None:
    assert (a.steps, a.residual_checks, a.stagnation_checks, a.damping_halvings) == (
        b.steps, b.residual_checks, b.stagnation_checks, b.damping_halvings
    )
    assert a.state.steps == b.state.steps
    for name in ("residual",):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()
    for name in ("v", "activation", "adaptation", "activity_change"):
        assert getattr(a.state, name).tobytes() == getattr(b.state, name).tobytes(), name


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("budget", [0, 1, 7, 32, 33, 512])
@pytest.mark.parametrize("chunk", [1, 8, 32])
def test_a_solve_in_one_call_is_the_solve_in_launches(rows: int, budget: int, chunk: int) -> None:
    brain = cd.Brain.compose(8, 4, modules=(16,), seed=3, recall=True, recall_founder=True, episodic=False)
    graph = brain.brain
    rng = np.random.default_rng(rows + budget + chunk)
    drive = brain.stimulus(rng.random((rows, 8)))
    assert graph._solves_compiled()
    for tolerance in (3e-3, 1e-9, 0.0):
        assert_same(graph.equilibrate(drive, budget=budget, chunk=chunk, tolerance=tolerance),
                    launches(graph, drive, budget=budget, chunk=chunk, tolerance=tolerance))
    warm = graph.settle_batch(drive, steps=3)
    other = brain.stimulus(rng.random((rows, 8)))
    assert_same(graph.equilibrate(other, budget=budget, chunk=chunk, tolerance=3e-3, state=warm),
                launches(graph, other, budget=budget, chunk=chunk, tolerance=3e-3, state=warm))
    # the state it leaves reads as any: a check of it, a settle from it
    solved = graph.equilibrate(other, budget=budget, chunk=chunk, tolerance=3e-3, state=warm)
    assert "_products" in solved.state.__dict__
    assert graph.residual(other, solved.state).tobytes() == solved.residual.tobytes()
    plain = cd.BrainState(np.array(solved.state.v), np.array(solved.state.activation),
                          np.array(solved.state.adaptation), solved.state.steps)
    assert graph.settle_batch(other, steps=5, state=solved.state).v.tobytes() == (
        graph.settle_batch(other, steps=5, state=plain).v.tobytes()
    )


def test_a_state_that_diverges_stops_as_the_launches_stop() -> None:
    brain = cd.Brain.compose(8, 4, modules=(16,), seed=1, episodic=False)
    graph = brain.brain
    wild = graph.with_parameters(efficacy=graph.efficacy * 1e150)
    drive = brain.stimulus(np.random.default_rng(0).random((2, 8)))
    one, other = (
        wild.equilibrate(drive, budget=256, tolerance=3e-3),
        launches(wild, drive, budget=256, tolerance=3e-3),
    )
    assert one.steps == other.steps and one.residual_checks == other.residual_checks
    assert one.residual.tobytes() == other.residual.tobytes()


def test_a_wrapped_method_sees_every_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = cd.Brain.compose(8, 4, modules=(16,), seed=0, episodic=False)
    graph = brain.brain
    drive = brain.stimulus(np.random.default_rng(1).random((2, 8)))
    calls: list[str] = []
    settle, residual = NeuralGraph.settle_batch, NeuralGraph.residual

    def counted_settle(self: NeuralGraph, *args, **kwargs):
        calls.append("settle")
        return settle(self, *args, **kwargs)

    def counted_residual(self: NeuralGraph, *args, **kwargs):
        calls.append("check")
        return residual(self, *args, **kwargs)

    monkeypatch.setattr(NeuralGraph, "settle_batch", counted_settle)
    monkeypatch.setattr(NeuralGraph, "residual", counted_residual)
    solved = graph.equilibrate(drive, budget=64, chunk=32, tolerance=1e-12)
    assert calls.count("settle") == 1 + solved.steps // 32
    assert calls.count("check") == solved.residual_checks


def test_masks_nudges_compartments_and_adaptation_take_the_launches() -> None:
    brain = cd.Brain.compose(8, 4, modules=(12, 8), seed=2, apical=0.25, episodic=False)
    assert not brain.brain._solves_compiled()
    plain = cd.Brain.compose(8, 4, modules=(16,), seed=0, episodic=False).brain
    adapting = NeuralGraph(
        plain.connectome, plain.neuron_model.replace(adaptation=cd.Adaptation(tau_steps=5.0)),
        efficacy=plain.efficacy, bias=plain.bias,
    )
    drive = np.zeros((2, plain.connectome.n))
    drive[:, :4] = 1.0
    assert_same(adapting.equilibrate(drive, budget=64, tolerance=3e-3),
                launches(adapting, drive, budget=64, tolerance=3e-3))


def test_a_subclass_that_withholds_its_blocks_settles_its_own_way() -> None:
    brain = cd.Brain.compose(8, 4, modules=(16,), seed=0, episodic=False)
    graph = brain.brain

    class Withheld(NeuralGraph):
        def _fused_blocks(self):
            return None

    withheld = Withheld(graph.connectome, graph.neuron_model, efficacy=graph.efficacy, bias=graph.bias)
    assert not withheld._solves_compiled()
    drive = brain.stimulus(np.random.default_rng(4).random((2, 8)))
    assert withheld.equilibrate(drive, budget=64, tolerance=3e-3).qualified.all()


def test_a_brain_compiles_each_kernel_once() -> None:
    """A constant argument of a compiled call is a type of its own to Numba: a solve that passed
    constants would compile the settle and the check again beside the ones a launch takes."""
    code = """
import numpy as np
import cadence as cd
from cadence import fused
brain = cd.Brain.compose(8, 4, modules=(16,), seed=0, recall=True, episodic=False,
                         learning_surprise=True, teaching="deep")
rng = np.random.default_rng(0)
for _ in range(3):
    brain.act(rng.random((4, 8)), greedy=True)
    brain.lesson(rng.random((4, 8)), rng.integers(0, 4, 4))
print(len(fused._kernel.signatures), len(fused._residual_kernel.signatures))
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.split() == ["1", "1"]
