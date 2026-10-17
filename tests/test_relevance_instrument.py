"""Independent task-truth, shortcut, saved-fork and bounded-run checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence.receipts import Receipt, canonical_sha256

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/relevance/selective_recall.py"
spec = importlib.util.spec_from_file_location("cadence_relevance_instrument", SCRIPT)
assert spec is not None and spec.loader is not None
instrument = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = instrument
spec.loader.exec_module(instrument)


def test_query_supplies_identity_and_both_facts_vary_behind_identical_suffixes():
    for q in (0, 1):
        prefix, query, target = instrument.inputs(
            instrument.FACTS, np.full(8, q, dtype=int), 2
        )
        assert np.unique(query, axis=0).shape[0] == 1
        assert query[:, 5 + q].all() and query.sum(axis=1).tolist() == [1] * 8
        assert not query[:, 7:].any()
        np.testing.assert_array_equal(prefix[-1], prefix[-2])
        assert np.unique(prefix[-1], axis=0).shape[0] == 1
        # Decode the actual witnessed symbols independently of target construction.
        reconstructed = []
        for first, second in zip(prefix[0], prefix[1], strict=True):
            seen = [int(np.argmax(first)), int(np.argmax(second))]
            values = {"A" if symbol < 2 else "B": symbol % 2 for symbol in seen}
            assert set(values) == {"A", "B"}
            reconstructed.append(values["A" if q == 0 else "B"])
        np.testing.assert_array_equal(target, reconstructed)
        assert target.sum() == 4


def test_exposed_control_supplies_both_witnessed_facts_regardless_of_query():
    queries = []
    for q in (0, 1):
        _, query, _ = instrument.inputs(
            instrument.FACTS, np.full(8, q, dtype=int), 1, exposed=True
        )
        queries.append(query)
        np.testing.assert_array_equal(query[:, 7:9].argmax(axis=1), instrument.FACTS[:, 0])
        np.testing.assert_array_equal(query[:, 9:11].argmax(axis=1), instrument.FACTS[:, 1])
    np.testing.assert_array_equal(queries[0][:, 7:], queries[1][:, 7:])


def test_pair_scores_reject_the_seventy_five_percent_shortcut():
    facts = instrument.FACTS[:, :2]
    truth = instrument.scores(facts)
    assert truth["accuracy"] == truth["unequal_fact_accuracy"] == 1.0
    assert truth["relevant_flip"] == truth["irrelevant_invariance"] == 1.0
    assert truth["query_pair_correct_when_facts_differ"] == 1.0
    shortcut = np.repeat(facts.min(axis=1)[:, None], 2, axis=1)
    wrong = instrument.scores(shortcut)
    assert wrong["accuracy"] == 0.75 and wrong["unequal_fact_accuracy"] == 0.5
    assert wrong["query_flip_when_facts_differ"] == 0.0
    refused = instrument.scores(np.full((8, 2), -1))
    assert refused["accuracy"] == 0 and refused["unanswered"] == 16
    assert refused["irrelevant_invariance"] == refused["order_invariance"] == 0


def test_evaluation_forks_identical_prequery_state_and_preserves_pending_outcome(tmp_path):
    brain = cd.Brain.compose(11, 2, modules=(4,), episodic=False, seed=0)
    brain.act(np.eye(11)[[0]])  # a real pending action must survive evaluation untouched
    before = instrument.checkpoint_digest(brain.save(tmp_path / "before.npz"))
    evaluation = instrument.evaluate(brain, 1)
    assert evaluation["completed"] and evaluation["original_preserved"]
    assert instrument.checkpoint_digest(brain.save(tmp_path / "after.npz")) == before
    # Three prefix solves, then two intact and two erased queries, eight streams each.
    assert evaluation["work"]["actions"] == 7
    assert evaluation["work"]["action_rows"] == 56
    assert evaluation["work"]["lessons"] == 0
    assert evaluation["fork_origin"]
    assert brain.pending_feedback


def test_bounded_run_records_matched_exposure_frozen_parameters_and_verifiable_answers(tmp_path):
    protocol = json.loads(instrument.PROTOCOL.read_text())
    protocol.update(founders=[0], rounds=1)
    receipt = instrument.run(protocol)
    assert not receipt.body["frozen_protocol"]
    founder = receipt.body["founders"][0]
    arms = founder["arms"]
    assert len({arm["initial_parameters"] for arm in arms.values()}) == 1
    assert arms["frozen"]["initial_parameters"] == arms["frozen"]["final_parameters"]
    assert founder["schedule_sha256"] == canonical_sha256(founder["schedule"])
    for name, arm in arms.items():
        assert arm["training"]["completed"]
        work = arm["training"]["work"]
        assert work["actions"] == 4 and work["action_rows"] == 64
        assert work["presentations"] == (0 if name == "frozen" else 16)
        assert arm["resolved"]["memory"] is None
        assert arm["resolved"]["episodic"] is None
    path = receipt.write(tmp_path / "receipt.json")
    assert Receipt.verify(path, sources=instrument.sources(), check=instrument.check)[0]
    assert not founder["acquisition_control_passed"]
    assert "not localized" in founder["interpretation"]
    # Even a newly digested receipt cannot claim a fabricated arithmetic result.
    receipt.body["founders"][0]["arms"]["trace"]["evaluation"]["branches"]["intact"][
        "scores"
    ]["accuracy"] = 123.0
    corrupted = Receipt.build(receipt.kind, receipt.body, instrument.sources())
    assert not Receipt.verify(corrupted.write(tmp_path / "changed.json"),
                              check=instrument.check)[0]


def test_cli_refuses_to_overwrite_retained_results(tmp_path):
    path = tmp_path / "kept.json"
    path.write_text("retained")
    with pytest.raises(SystemExit):
        instrument.main(["--out", str(path), "--rounds", "1", "--seeds", "0"])
    assert path.read_text() == "retained"
