"""The work of a learned code whose step changes: a restep keeps the sub-network's layout, the
step's tests are asked for the neighbouring steps before any eigenvalue is solved, and on the
host the dictionary's Gram matrix is made once per dictionary. None of it changes a result: the
same lives give the same bits with these paths and with the plain ones."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd
from cadence import brain as cadence_brain
from cadence import code as cadence_code
from cadence.code import LearnedCode

INPUTS, UNITS = 48, 24


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def stiff(brain: cd.Brain, share: float = 0.55, seed: int = 3) -> LearnedCode:
    """``brain``'s code with every unit sharing a direction with the others: its lateral
    competition is strong, so the step is halved and the halvings follow the state."""
    region = brain.learned_code
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(region.inputs, 1))
    dictionary = (1.0 - share) * region.dictionary + share * shared * (region.dictionary != 0)
    region.take(dictionary / np.linalg.norm(dictionary, axis=0), region.weights, region.thresholds)
    brain.learner.brain = brain._installed(brain.brain)
    return region


def composed(**options) -> cd.Brain:
    return cd.Brain.compose(
        INPUTS, 6, seed=5,
        **{"modules": (24,), "learned_code": UNITS, "recall": False, "backend": "cpu", **options},
    )


def eigenvalue_halvings(code: LearnedCode, v: np.ndarray) -> int:
    """The step's halvings as the eigenvalues give them, the rule ``stability`` states."""
    slopes = np.maximum(code.model.slope_at(v), 0.0)
    root = np.sqrt(slopes)
    lateral = -(code.dictionary.T @ code.dictionary)
    np.fill_diagonal(lateral, 0.0)
    coupling = root[:, :, None] * (code.model.gain * lateral)[None, :, :] * root[:, None, :]
    index = np.arange(code.units)
    coupling[:, index, index] -= code.top_down * slopes
    mu = float(np.linalg.eigvalsh(coupling)[:, 0].min())
    need = code.model.dt * (1.0 - mu) / code.STEP_MARGIN
    return 0 if need <= 1.0 else int(np.ceil(np.log2(need)))


def lived(brain: cd.Brain, moments: int = 18) -> dict[str, object]:
    """A life of lessons, greedy and sampled acts and outcomes on observations of changing
    energy: everything it learned and said, and the halvings its code took."""
    region = stiff(brain)
    rng = np.random.default_rng(3)
    slots = brain.learner.slot_sizes
    acts, reports, steps = [], [], []
    for t in range(moments):
        scale = (0.2, 1.0, 3.0)[t % 3]
        x = np.abs(rng.normal(0.0, scale, size=(2, INPUTS))) * (rng.random((2, INPUTS)) < 0.5)
        labels = (np.stack([rng.integers(0, k, size=2) for k in slots], axis=1)
                  if len(slots) > 1 else rng.integers(0, slots[0], size=2))
        if t % 2:
            reports.append(brain.lesson(x, labels))
        if t % 4 == 3:  # a sampled act and its outcome
            acts.append(brain.step(x))
            steps.append(region.halvings)
            acts.append(brain.step(x[::-1].copy(), reward=np.array([1.0, -0.5])))
        else:
            acts.append(brain.act(x, greedy=True))
        steps.append(region.halvings)
        reports.append(dict(brain.last_settlement))
    return {
        "acts": np.concatenate([np.ravel(a) for a in acts]), "steps": steps, "reports": reports,
        "work": dict(region.work), "efficacy": np.array(brain.brain.efficacy),
        "bias": np.array(brain.brain.bias), "dictionary": region.dictionary.copy(),
        "weights": region.weights.copy(), "thresholds": region.thresholds.copy(),
    }


def plainly(monkeypatch) -> None:
    """The plain paths: every graph of the region laid out anew, the step's tests asked for
    the present halvings alone, the Gram matrix made at every read."""
    graph = cadence_code.NeuralGraph

    def laid_out_anew(*args, **options):
        options.pop("layout", None)
        return graph(*args, **options)

    monkeypatch.setattr(cadence_code, "NeuralGraph", laid_out_anew)
    monkeypatch.setattr(LearnedCode, "_step_candidates", staticmethod(lambda held: (held,)))
    monkeypatch.setattr(LearnedCode, "_gram", lambda self: self._gram_product())


PLACES = [{"backend": "cpu"}]
try:
    import torch  # noqa: F401

    PLACES.append({"backend": "torch", "device": "cpu", "precision": "float64"})
except ImportError:  # the host's paths alone
    pass


@pytest.mark.parametrize("place", PLACES, ids=lambda place: place["backend"])
@pytest.mark.parametrize("options", [
    {},
    {"modules": (24, 16), "slots": (3, 3), "attention": True, "recall": True, "recall_heads": 2},
    {"modules": (24, 16), "apical": 0.25, "learned_code_fields": (32, 16)},
], ids=["one module", "attention", "apical fields"])
def test_a_life_whose_step_changes_gives_the_bits_of_the_plain_paths(place, options, monkeypatch):
    ours = lived(composed(**{**options, **place}))
    with monkeypatch.context() as plain:
        plainly(plain)
        theirs = lived(composed(**{**options, **place}))
    steps = ours["steps"]
    # the step changed, by one halving and by more
    changes = [abs(a - b) for a, b in zip(steps, steps[1:], strict=False) if a != b]
    assert len(changes) >= 4 and 1 in changes
    assert steps == theirs["steps"] and ours["reports"] == theirs["reports"]
    assert ours["work"] == theirs["work"]
    for name in ("acts", "efficacy", "bias", "dictionary", "weights", "thresholds"):
        np.testing.assert_array_equal(ours[name], theirs[name], err_msg=name)


def test_a_restep_keeps_the_layout_and_lays_nothing_out_again(monkeypatch):
    code = stiff(composed())
    layout = code.graph.layout
    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(size=(2, UNITS)))
    # a first settle finds what the layout keeps: whether synapses share an entry, the
    # entries' order, the columns' forms for the residual check
    code.settle(x, None, budget=64, tolerance=1e-6)
    assert layout.parallel_synapses is False
    laid, found = [], []
    make, unique = cadence_brain.make_layout, np.unique
    monkeypatch.setattr(cadence_brain, "make_layout", lambda *a, **k: (laid.append(1), make(*a, **k))[1])
    monkeypatch.setattr(np, "unique", lambda *a, **k: (found.append(1), unique(*a, **k))[1])
    for halvings in (1, 3, 0, 2):
        code.restep(halvings)
        assert code.halvings == halvings and code.graph.layout is layout
        assert code.graph.neuron_model.dt == code.model.dt / 2**halvings
        settled = code.settle(x, None, budget=64, tolerance=1e-6)
        assert np.isfinite(settled.state.activation).all()
    code.rest()  # the step of a solve from rest, and the graph kept for it
    assert code.graph.layout is layout
    # no layout is made and none of the 552 synapses is sorted again
    assert not laid and not found
    with monkeypatch.context() as plain:  # the plain path lays every restep's graph out anew
        plainly(plain)
        code.restep(code.halvings + 1)
        code.settle(x, None, budget=64, tolerance=1e-6)
    # the layout itself, whether synapses share an entry and, on the compiled kernels, the
    # columns' forms for the residual check: a pass over every synapse each
    assert len(laid) == 1 and len(found) >= 2 and code.graph.layout is not layout


def test_a_step_one_halving_away_is_settled_without_an_eigenvalue(monkeypatch):
    code = stiff(composed())
    rng = np.random.default_rng(11)
    solved = []
    solve = np.linalg.eigvalsh
    seen = set()
    for scale, rows in ((0.5, 1), (1.0, 2), (3.0, 4), (8.0, 4)):
        for _ in range(4):
            v = rng.normal(code.model.threshold, scale, (rows, code.units))
            expected = eigenvalue_halvings(code, v)
            seen.add(expected)
            with monkeypatch.context() as counting:
                counting.setattr(np.linalg, "eigvalsh", lambda m: (solved.append(1), solve(m))[1])
                for held in {expected, expected + 1, max(expected - 1, 0)}:
                    code.restep(held)
                    del solved[:]
                    assert code.stability(v) == expected
                    assert not solved, (held, expected)
                for held in {expected + 2, max(expected - 2, 0)} - {expected, expected + 1,
                                                                   max(expected - 1, 0)}:
                    code.restep(held)
                    del solved[:]
                    assert code.stability(v) == expected
                    assert solved  # further away the eigenvalues decide, as before
    assert len(seen) >= 2 and max(seen) >= 2


def test_on_a_device_a_step_one_halving_away_is_settled_without_an_eigenvalue(monkeypatch):
    torch = pytest.importorskip("torch")
    placed = stiff(composed(backend="torch", device="cpu", precision="float64"))
    assert placed.device is not None
    reference = stiff(composed())
    rng = np.random.default_rng(11)
    solved = []
    solve = torch.linalg.eigvalsh
    monkeypatch.setattr(torch.linalg, "eigvalsh", lambda m: (solved.append(1), solve(m))[1])
    for scale, rows in ((1.0, 2), (3.0, 4), (8.0, 4)):
        v = rng.normal(placed.model.threshold, scale, (rows, placed.units))
        expected = eigenvalue_halvings(reference, v)
        for held in {expected, expected + 1, max(expected - 1, 0)}:
            placed.restep(held)
            del solved[:]
            assert placed.stability(v) == expected and not solved
        placed.restep(expected + 3)
        assert placed.stability(v) == expected and solved


def test_the_gram_is_made_once_per_dictionary_and_again_when_it_reads_otherwise(monkeypatch):
    made = []
    product = LearnedCode._gram_product
    monkeypatch.setattr(LearnedCode, "_gram_product", lambda self: (made.append(1), product(self))[1])
    code = stiff(composed())
    rng = np.random.default_rng(1)
    v = rng.normal(code.model.threshold, 3.0, (2, code.units))
    del made[:]
    held = code.stability(v)
    code.restep(held + 1)
    code.install()
    code.rest_stability()
    lateral = code.lateral()
    assert not made  # the step's tests, a restep, an install and the rest step read the one
    expected = -(code.dictionary.T @ code.dictionary)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_array_equal(lateral, expected)
    # the caller's own array: writing it changes nothing the region reads
    own = code.gram()
    own[:] = 0.0
    np.testing.assert_array_equal(code.lateral(), expected)
    assert not made
    # a dictionary written in place reads otherwise: the Gram is made again, of what it holds
    code.dictionary[:, 0] = code.dictionary[:, 1]
    written = -(code.dictionary.T @ code.dictionary)
    np.fill_diagonal(written, 0.0)
    np.testing.assert_array_equal(code.lateral(), written)
    assert len(made) == 1
    assert code.stability(v) == eigenvalue_halvings(code, v) and len(made) == 1
    # and a learning step, which replaces the dictionary, once more
    code.install()
    del made[:]
    sensory = np.abs(rng.normal(size=(2, code.inputs)))
    activation = np.abs(rng.normal(size=(2, code.units)))
    assert code.learn(sensory, activation, np.zeros((2, code.context)))
    assert len(made) == 1
    code.lateral()
    code.stability(v)
    assert len(made) == 1


def test_a_region_on_a_device_keeps_no_gram_on_the_host():
    pytest.importorskip("torch")
    placed = composed(backend="torch", device="cpu", precision="float64").learned_code
    assert placed.device is not None and placed._kept_gram is None
    placed.lateral()
    placed.install()
    assert placed._kept_gram is None  # the kernel holds the lateral weights there
    placed.place(None)
    placed.lateral()
    assert placed._kept_gram is not None
