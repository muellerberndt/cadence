"""Maintainer guards for the corrected key-door/3 instrument, not a new confirmation."""

import copy
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("keydoor_audit", HERE / "key_door.py")
keydoor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keydoor)


def test_recurrent_eligibilities_are_gradients_of_the_executed_policy_and_value():
    life = keydoor.Recurrent(4, hidden=3, alpha=0.2, alpha_value=0.3, lam=0, clip=1e6)
    life.wv[:] = [0.1, -0.2, 0.3]
    life.act(keydoor.CHEST, False, None, False)
    previous = life.h.copy()
    x = life._features(keydoor.LEVER, True)
    action, _ = life.act(keydoor.LEVER, True, 0.8, False)
    h, probabilities, value = life._forward(x, previous)
    np.testing.assert_allclose(life.h, h)
    assert life.last[0] == probabilities[keydoor.INTERACT]
    assert life.memo["value"] == value

    # Finite differences hold the previous hidden state fixed, as the declared
    # one-moment semi-gradient does. They include the nonterminal feedback update.
    epsilon = 1e-6
    for trace, actor in ((life.actor_trace, True), (life.value_trace, False)):
        for name, recorded in trace.items():
            parameter = getattr(life, name)
            for index in np.ndindex(recorded.shape):
                old = float(parameter if name == "bv" else parameter[index])
                values = []
                for sign in (-1, 1):
                    if name == "bv":
                        life.bv = old + sign * epsilon
                    else:
                        parameter[index] = old + sign * epsilon
                    _, policy, estimate = life._forward(x, previous)
                    values.append(np.log(policy[action]) if actor else estimate)
                if name == "bv":
                    life.bv = old
                else:
                    parameter[index] = old
                assert recorded[index] == pytest.approx(
                    (values[1] - values[0]) / (2 * epsilon), abs=2e-9
                )


def test_recurrent_guard_bounds_the_whole_actor_and_critic_gradient():
    actor = keydoor.Recurrent(0, clip=1.0, lam=0)
    actor.Wx.fill(1.0)
    actor.act(keydoor.CHEST, False, None, False)
    assert np.sqrt(sum(np.sum(v * v) for v in actor.actor_trace.values())) == pytest.approx(1.0)
    assert actor.clipped == 1
    critic = keydoor.Recurrent(0, clip=5.0, lam=0)
    critic.wv.fill(10.0)
    critic.act(keydoor.CHEST, True, None, False)
    assert np.sqrt(sum(np.sum(v * v) for v in critic.value_trace.values())) == pytest.approx(5.0)
    assert critic.value_clipped == 1


def test_recurrent_terminal_resets_hidden_state_but_truncation_carries_it():
    life = keydoor.Recurrent(4, hidden=3, alpha=0, alpha_value=0)
    life.act(keydoor.CHEST, False, None, False)
    twin = copy.deepcopy(life)
    previous = life.h.copy()
    x = life._features(keydoor.FLOOR, False)
    life.act(keydoor.FLOOR, False, 0.0, True)
    twin.act(keydoor.FLOOR, False, 0.0, False)
    np.testing.assert_allclose(life.h, life._forward(x, np.zeros_like(previous))[0])
    np.testing.assert_allclose(twin.h, twin._forward(x, previous)[0])
    assert not np.array_equal(life.h, twin.h)


def test_joint_audit_requires_same_lives_and_the_full_confirmation_census():
    protocol = json.loads(keydoor.PROTOCOL.read_text())
    rows = []
    for arm in keydoor.ARMS:
        for delay in protocol["delays"]:
            for seed in protocol["seeds"]["confirmation"]:
                rows.append(
                    {
                        "arm": arm,
                        "delay": delay,
                        "seed": seed,
                        "phases": [
                            {"fed": 1.0, "wrong": 0.0, "aroused_late": 0.0, "lag": 0}
                            for _ in range(3)
                        ],
                    }
                )
    assert keydoor.audit_gates(rows, protocol, eligible=True)["passed"]
    gated = [r for r in rows if r["arm"] == "live" and r["delay"] in (2, 5)]
    gated[0]["phases"][0]["fed"] = 0.0
    gated[1]["phases"][1]["fed"] = 0.0
    gated[2]["phases"][0]["wrong"] = 2.0
    assert keydoor.gates(rows, protocol)["passed"]  # historical marginal rule
    audit = keydoor.audit_gates(rows, protocol, eligible=True)
    assert not audit["passed"] and audit["pooled"]["jointly_qualified"] == 0.85
    assert not keydoor.audit_gates(rows[:-1], protocol, eligible=True)["admissible"]
    assert not keydoor.audit_gates(rows, protocol, eligible=False)["passed"]


def test_development_subset_is_not_a_frozen_confirmation_and_counts_controls(tmp_path, capsys):
    path = tmp_path / "audit.json"
    assert (
        keydoor.main(
            [
                "--arms",
                "recurrent",
                "tabular",
                "random",
                "--seeds",
                "0",
                "--delays",
                "2",
                "--episodes",
                "4",
                "--workers",
                "1",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    body = keydoor.read_receipt(path)
    assert body["instrument_revision"] == 2 and not body["frozen_protocol"]
    assert not body["gates"]["admissible"] and not body["gates"]["passed"]
    for row in body["rows"]:
        work = row["work"]
        moments = sum(p["moments"] for p in row["phases"])
        assert work["action_calls"] == moments
        assert work["control_updates"] == (moments - 1 if row["arm"] != "random" else 0)
        assert work["probe_calls"] == 6
        assert (
            work["control_parameters"] == {"recurrent": 419, "tabular": 20, "random": 0}[row["arm"]]
        )
    assert keydoor.verify(path, current=True)[0]
    capsys.readouterr()


def test_self_signed_impossible_probe_and_changed_trip_statistics_are_rejected(tmp_path, capsys):
    path = tmp_path / "audit.json"
    assert (
        keydoor.main(
            [
                "--arms",
                "random",
                "--seeds",
                "0",
                "--delays",
                "2",
                "--episodes",
                "30",
                "--workers",
                "1",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    original = json.loads(path.read_text())

    def changed(edit):
        stored = copy.deepcopy(original)
        edit(stored["body"])
        stored["digest"] = keydoor.canonical_sha256(
            {k: stored[k] for k in ("kind", "body", "source")}
        )
        path.write_text(keydoor.canonical_json(stored) + "\n")
        assert not keydoor.verify(path)[0]

    changed(lambda b: b["rows"][0]["phases"][0]["end_interact"][0].__setitem__(0, 999.0))
    changed(lambda b: b["rows"][0]["phases"][0].update(fed=0.1234))
    changed(lambda b: b["rows"][0]["work"].update(action_calls=1))
    changed(lambda b: b["rows"][0]["work"].update(control_parameters=999))
    changed(lambda b: b.update(frozen_protocol=True))
    changed(
        lambda b: b["rows"][0]["phases"][0].update(
            episodes=0,
            cut=30,
            moments=0,
            visits=[[0] * 5] * 2,
            fed=None,
            fed_whole=None,
            took=None,
            wrong=None,
            opened=None,
            lag=None,
        )
    )
    capsys.readouterr()


def test_failed_probe_is_counted_timed_and_checked_even_in_a_crashed_row(
    tmp_path, monkeypatch, capsys
):
    class Refusing(keydoor.Random):
        def probe(self):
            raise RuntimeError("probe failed before an action")

    monkeypatch.setattr(keydoor, "make_life", lambda *args: Refusing(0))
    path = tmp_path / "failed.json"
    assert (
        keydoor.main(
            [
                "--arms",
                "random",
                "--seeds",
                "0",
                "--delays",
                "2",
                "--episodes",
                "4",
                "--workers",
                "1",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    stored = json.loads(path.read_text())
    work = stored["body"]["rows"][0]["work"]
    assert work["probe_calls"] == work["failed_probe_calls"] == 1
    assert work["failed_call_seconds"] > 0 and work["action_calls"] == 0
    work["action_calls"] = -1
    stored["digest"] = keydoor.canonical_sha256({k: stored[k] for k in ("kind", "body", "source")})
    path.write_text(keydoor.canonical_json(stored) + "\n")
    assert not keydoor.verify(path)[0]
    capsys.readouterr()


def test_historical_third_freeze_keeps_original_bytes_and_explicit_limits():
    path = HERE / "results/confirmation-3-2026-10-08.json.gz"
    before = path.read_bytes()
    valid, reason = keydoor.verify(path)
    assert valid and "historical key-door/3" in reason
    assert path.read_bytes() == before
    body = json.loads(gzip.decompress(before))["body"]
    assert "instrument_revision" not in body
    assert body["gates"] == keydoor.gates(body["rows"], body["protocol"])
    assert not body["gates"]["passed"]


def test_early_copy_archive_is_custody_only_and_unknown_instruments_stay_rejected(tmp_path):
    path = HERE / "results/development-3-copy-amp3-2026-10-08.json.gz"
    original = json.loads(gzip.decompress(path.read_bytes()))
    valid, reason = keydoor.verify(path)
    assert valid and "custody only" in reason and "source unavailable" in reason
    for change in ("source", "arm"):
        edited = copy.deepcopy(original)
        if change == "source":
            edited["source"]["files"][0]["sha256"] = "0" * 64
            edited["source"]["manifest_sha256"] = keydoor.canonical_sha256(
                edited["source"]["files"]
            )
        else:
            edited["body"]["arms"][1] = "unknown"
        edited["digest"] = keydoor.canonical_sha256(
            {k: edited[k] for k in ("kind", "body", "source")}
        )
        forged = tmp_path / f"{change}.json"
        forged.write_text(keydoor.canonical_json(edited) + "\n")
        assert not keydoor.verify(forged)[0]
