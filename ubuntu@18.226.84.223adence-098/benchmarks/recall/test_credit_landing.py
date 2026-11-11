"""Guards of the credit-landing diagnostics: their readings, probes and receipts."""

import gzip
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("credit_landing", HERE / "credit_landing.py")
diagnostic = importlib.util.module_from_spec(spec)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    spec.loader.exec_module(diagnostic)


@pytest.fixture(scope="module")
def protocol():
    return diagnostic.load_protocol()


def test_the_historical_founders_are_the_freezes_founders():
    assert set(diagnostic.HISTORICAL) == set(range(301, 310))
    assert [seed for seed, learned in diagnostic.HISTORICAL.items() if learned] == [303, 306, 309]


def test_synapse_groups_partition_the_synapses(protocol):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        brain = diagnostic.make_brain(protocol, 0, 0.3, 0.8)
    groups = diagnostic.groups_of(brain)
    total = np.zeros(brain.connectome.synapses, dtype=int)
    for mask in groups.values():
        total += mask
    assert (total == 1).all()
    assert {"prefrontal->association", "association->motor", "sensory->association"} <= set(groups)


def test_paired_distance_reads_adjacent_rows():
    x = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [1.0, 1.0]])
    assert diagnostic.paired(x) == pytest.approx(2.5)


def test_tally_tells_a_signal_from_noise():
    tally = diagnostic.Tally(2)
    for k in range(40):
        tally.add(np.array([0.1, 0.1 if k % 2 else -0.1]))
    out = tally.by_group({"all": np.ones(2, dtype=bool)})["all"]
    assert out["moved"] == 2 and abs(out["consistency"] - 0.5) < 1e-9


def test_motor_bias_and_homeostasis_probes_move_only_motor_biases(protocol):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        brain = diagnostic.make_brain(protocol, 0, 0.3, 0.8)
    before = brain.brain.bias.copy()
    diagnostic.set_motor_bias(brain, 2.0)
    after = brain.brain.bias
    motor = brain.motor_index
    assert np.all(after[motor] == 2.0)
    others = np.setdiff1d(np.arange(len(after)), motor)
    assert np.array_equal(after[others], before[others])
    x = np.zeros((2, diagnostic.inputs.INPUTS))
    x[:, diagnostic.inputs.START] = 1
    brain.act(x, greedy=True)
    state = brain.basal_ganglia.state
    level = np.atleast_2d(np.asarray(state.activation))[:, motor].mean(axis=0)
    diagnostic.homeostasis(brain, 0.3, 0.5)
    moved = brain.brain.bias[motor] - 2.0
    assert np.allclose(moved, 0.5 * (0.3 - level))
    diagnostic.homeostasis(brain, 0.3, 0.0)
    assert np.allclose(brain.brain.bias[motor] - 2.0, moved), "rate zero changes nothing"


def test_death_needs_a_run_of_still_accepted_lessons():
    moving = {"step": {"a": 0.1}, "motor_level": 0.1, "motor_bias": 0.0, "accepted": True}
    still = {"step": {"a": 0.0}, "motor_level": 1.0, "motor_bias": 0.0, "accepted": True}
    refused = {"step": {"a": 0.0}, "motor_level": 0.1, "motor_bias": 0.0, "accepted": False}
    lessons = [moving] * 3 + [still] * diagnostic.RUN
    dead = diagnostic.death(lessons)
    assert dead["lesson"] == 3 and dead["of"] == 3 + diagnostic.RUN
    assert dead["motor_level_after"] == 1.0
    assert diagnostic.death([moving] * 3 + [still] * (diagnostic.RUN - 1)) is None
    assert diagnostic.death([moving] * 3 + [refused] * diagnostic.RUN) is None
    assert diagnostic.death([moving]) is None


def test_a_receipt_verifies_and_binds_the_current_sources(tmp_path):
    out = tmp_path / "untrained.json.gz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert (
            diagnostic.main(["--untrained", "--founders", "0", "--workers", "1", "--out", str(out)])
            == 0
        )
    assert diagnostic.main(["--verify", str(out)]) == 0
    assert diagnostic.main(["--verify", str(out), "--current"]) == 0


def test_a_short_run_of_both_modes_writes_receipts_and_reports(tmp_path, capsys):
    out = tmp_path / "untrained.json.gz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert (
            diagnostic.main(["--untrained", "--founders", "0", "--workers", "1", "--out", str(out)])
            == 0
        )
    body = json.loads(gzip.decompress(out.read_bytes()))["body"]
    assert body["mode"] == "untrained" and body["founders"][0]["seed"] == 0
    events = body["founders"][0]["episodes"]["clean-0"]
    assert {"paired_association", "paired_trace", "motor_margin"} <= set(events[0])
    out = tmp_path / "credit.json.gz"
    run = ["--credit", "--founders", "0", "--repeats", "1", "--every", "1", "--workers", "1"]
    run += ["--homeostasis", "0.3", "0.02", "--learning", "momentum=0", "--out", str(out)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert diagnostic.main(run) == 0
    body = json.loads(gzip.decompress(out.read_bytes()))["body"]
    founder = body["founders"][0]
    assert body["learning_overrides"] == {"momentum": 0} and founder["homeostasis"] == [0.3, 0.02]
    assert len(founder["lessons"]) == len(diagnostic.inputs.TRAIN_CONDITIONS)
    assert len(founder["curve"]) == 2 and "prefrontal->association" in founder["consistency"]
    assert {"motor_level", "motor_bias", "paired_association"} <= set(founder["lessons"][0])
    assert diagnostic.main(["--report", str(out)]) == 0
    assert "Where the lesson's credit lands" in capsys.readouterr().out
