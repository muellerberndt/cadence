"""Regression gates for the copy pathway (``recall_copy``, ``recall_tokens``)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "copy_tokens_tests", Path(__file__).with_name("copy_tokens.py")
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)

RECEIPT = Path(__file__).with_name("results") / "copy-tokens.json"


def test_the_tasks_ask_what_they_declare() -> None:
    rng = np.random.default_rng(0)
    for token in range(subject.T):
        heard, *others = subject.heldout_episode(rng, token)
        assert heard[0, token] == 1 and heard[0, subject.HEAR] == 1
        assert all(x[0, token] == 0 and x[0, subject.NOISE] == 1 for x in others)
    for _ in range(32):
        sequence, k, label = subject.succession_trial(rng, set(range(subject.TAUGHT)))
        assert label == sequence[k + 1] and label < subject.TAUGHT and k < subject.LENGTH - 1


def test_the_arms_differ_only_in_the_pathway() -> None:
    founder, recipe, control = (subject.compose(0, arm) for arm in ("founder", "identity",
                                                                       "random"))
    assert founder.recall.copy == 0.0 and recipe.recall.copy == 2.0
    assert recipe.recall.tokens == tuple(range(subject.T)) and control.recall.tokens is None
    assert founder.hippocampus is None and recipe.hippocampus is None
    # the pathway is appended: every other synapse is the founder's
    assert recipe.connectome.synapses > founder.connectome.synapses


def test_retained_receipt_is_source_bound_and_recomputed() -> None:
    ok, message = subject.verify_historical(RECEIPT)
    assert ok, message
    body = subject.cd.Receipt.read(RECEIPT).body
    assert body["passed"] and all(body["gates"].values())
    flat = body["summary"]["flat/identity"]
    assert flat["held"] >= 0.8 and body["summary"]["flat/founder"]["held"] <= 0.1


def test_the_check_refuses_a_decision_its_readings_do_not_support() -> None:
    body = copy.deepcopy(subject.cd.Receipt.read(RECEIPT).body)
    run = next(r for r in body["runs"] if r["arm"] == "identity" and r["depth"] == "flat")
    run["held"] = 0.0
    assert subject.historical_check(body) is not None
    body = copy.deepcopy(subject.cd.Receipt.read(RECEIPT).body)
    body["runs"] = body["runs"][:-1]
    assert subject.historical_check(body) is not None
