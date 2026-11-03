"""Offline historical reading never opens a current-source campaign prerequisite."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cadence import NeuralGraph
from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("campaign_offline_tests", HERE / "verify.py")
assert SPEC is not None and SPEC.loader is not None
offline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = offline
SPEC.loader.exec_module(offline)


@pytest.fixture(autouse=True)
def no_neural_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("historical reading attempted a neural phase")

    monkeypatch.setattr(NeuralGraph, "equilibrate", forbidden)
    monkeypatch.setattr(NeuralGraph, "_equilibrate_local", forbidden)
    monkeypatch.setattr(NeuralGraph, "settle_batch", forbidden)


@pytest.mark.parametrize("name", [
    "heads-continuation-development.json", "query-founder-capacity.json", "query-founder-h2.json",
    "minimal-binding-capacity.json", "minimal-binding-h2.json",
    "single-relevance-capacity.json", "single-relevance-h1.json",
    "neutral-variation-capacity.json", "neutral-variation-h2.json",
])
def test_actual_campaigns_verify_offline_without_a_neural_replay(name):
    valid, message = offline.verify(HERE / "results" / name)
    assert valid, message


def test_private_continuation_checker_keeps_live_receipt_class_unchanged():
    original_verify = Receipt.verify.__func__
    path = HERE / "results/heads-continuation-development.json"
    checker = offline.historical_checker("heads_continuation.py", reference=path)
    assert checker.parents().digest == checker.declared()["parent_relevance_digest"]
    valid, message = checker.verify(path)
    assert valid, message
    assert Receipt.verify.__func__ is original_verify
    current = checker.sources()
    live_valid, _ = Receipt.verify(path, sources=[
        (name, HERE.parents[1] / name) for name, _ in current])
    assert not live_valid


def test_offline_reader_refuses_actual_corrupted_source_bytes(tmp_path, monkeypatch):
    resolve = offline._resolved_sources
    altered = tmp_path / "altered.py"
    altered.write_text("changed = True\n")

    def corrupted(receipt):
        paths = resolve(receipt)
        return [(name, altered if name == "benchmarks/attention/heads.py" else path)
                for name, path in paths]

    monkeypatch.setattr(offline, "_resolved_sources", corrupted)
    valid, _ = offline.verify(HERE / "results/heads-continuation-development.json")
    assert not valid


def test_offline_reader_cannot_select_an_undeclared_checker():
    with pytest.raises(ValueError, match="declared"):
        offline.historical_checker(
            "../unexpected.py", reference=HERE / "results/query-founder-h2.json")


def single_reader_stub():
    spec = importlib.util.spec_from_file_location("single_history_stub", HERE / "historical.py")
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    session = helper.Session.__new__(helper.Session)
    session.installed = set()

    class Config:
        def to_dict(self):
            return {"eta": 0.1}

    class Oracle:
        def __init__(self):
            self.learner = SimpleNamespace(config=Config())

        def describe(self):
            return {"genes": {}}

    def factory(arm, seed):
        return Oracle()

    def panel_check(panel, causal=False, *, brain=None, arm=None):
        assert type(brain) is Oracle  # The native checker selects its oracle by exact type.

    module = SimpleNamespace(__file__="single_relevance_campaign.py", make_brain=factory,
                             panel_check=panel_check)

    def check(body):
        if body.get("nested"):
            inner = {"runs": body["runs"]}
            if body.get("inner_failure"):
                inner["raise"] = True
                with pytest.raises(ValueError, match="stub checker failure"):
                    module.check(inner)
            else:
                assert module.check(inner) is None
        brain = module.make_brain("supplied_address", 0)
        assert brain.learner.config.to_dict() == {"eta": 0.1}
        assert brain.describe()["genes"] == {}
        module.panel_check({}, brain=brain, arm="supplied_address")
        if body.get("raise"):
            raise ValueError("stub checker failure")
        return None

    module.check = check
    session.install(module)
    body = {"runs": [{"arm": "supplied_address", "seed": 0,
                       "learning_config": {"eta": 0.1}, "resolved": {"genes": {}}}]}
    return module, factory, body


def test_nested_single_reader_restores_the_enclosing_view_and_actual_oracle_type():
    module, factory, body = single_reader_stub()
    body["nested"] = True
    assert module.check(body) is None
    assert module.make_brain is factory
    body["inner_failure"] = True
    assert module.check(body) is None
    assert module.make_brain is factory
    body["raise"] = True
    with pytest.raises(ValueError, match="stub checker failure"):
        module.check(body)
    assert module.make_brain is factory


@pytest.mark.parametrize("value", [1, True, 0.0])
def test_single_reader_rejects_nonzero_or_nontyped_neutral_policy(value):
    module, factory, body = single_reader_stub()
    body["runs"][0]["learning_config"]["initial_damping"] = value
    assert module.check(body) == "historical numerical policy is not the neutral default"
    assert module.make_brain is factory
