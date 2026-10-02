"""No model queries: check exact lesions and their declared private custody."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lesion as L
import probe as P


@pytest.fixture(autouse=True)
def no_learning_or_queries(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No numerical calls in lesion fixture tests")

    for name in ("settle", "step", "observe", "observe_batch", "predict"):
        monkeypatch.setattr(P.Brain, name, forbidden)


@pytest.mark.parametrize(
    "variant,kind", [("zero_residual", "residual"), ("zero_state", "state")]
)
def test_exact_parameter_only_lesions(variant, kind):
    brain = P.make("observer", 3109)
    before = json.loads(brain.snapshot())
    original = tuple(brain.weights)
    removed = L.intervention(brain, variant)
    assert len(removed) == 2
    assert all(r["edge"][0] == kind and r["edge"][2] == 6 for r in removed)
    assert {r["edge"][1] for r in removed} == {4, 5}
    changed = {r["index"] for r in removed}
    assert all(
        w == (0.0 if i in changed else original[i]) for i, w in enumerate(brain.weights)
    )
    after = json.loads(brain.snapshot())
    assert {k: v for k, v in before.items() if k != "weights"} == {
        k: v for k, v in after.items() if k != "weights"
    }


def test_baseline_exactly_unchanged():
    brain = P.make("observer", 3109)
    before = brain.snapshot()
    assert not L.intervention(brain, "unchanged")
    assert brain.snapshot() == before


def test_cannot_lesion_missing_residual_contacts():
    brain = P.make("ordinary", 3109)
    before = brain.snapshot()
    with pytest.raises(ValueError, match="two corresponding"):
        L.intervention(brain, "zero_residual")
    assert brain.snapshot() == before
