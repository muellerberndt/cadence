"""Private stable-skill readings must not teach or disturb the continuing creature."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("key_door", HERE / "key_door.py")
kd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kd)
sys.modules["key_door"] = kd
spec = importlib.util.spec_from_file_location("retention_audit", HERE / "retention_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_stable_skill_uses_private_saved_state_and_actual_door_outcome(tmp_path):
    protocol = json.loads(audit.PROTOCOL.read_text())
    for arm in ("live", "blind"):
        life = kd.make_life(arm, protocol, 0, protocol["arousal"])
        life.act(kd.CHEST, False, None, False)
        before = life.brain.save(tmp_path / f"{arm}-before.npz")
        reading = audit.stable_door(life, tmp_path / arm, "initial")
        after = life.brain.save(tmp_path / f"{arm}-after.npz")
        assert audit.checkpoint_digests(before) == audit.checkpoint_digests(after)
        assert reading["food"] == int(reading["action"] == kd.INTERACT)
        assert reading["outcomes_delivered"] == 0
        assert reading["durable_before_sha256"] == reading["durable_after_sha256"]
        assert reading["observation"] == kd.observe(kd.DOOR, True, arm == "live").tolist()
        assert reading["work"]["checkpoints"] == 4 and reading["work"]["sweeps"] > 0
        # The real outcome is still owed once to the original creature.
        twin = kd.cd.Brain.load(before)
        np.testing.assert_array_equal(
            life.brain.live(kd.observe(kd.LEVER, True, arm == "live"), reward=[0.0]),
            twin.live(kd.observe(kd.LEVER, True, arm == "live"), reward=[0.0]),
        )


def test_development_assay_preserves_the_underlying_life_and_verifies_artifacts(tmp_path, capsys):
    protocol = json.loads(audit.PROTOCOL.read_text())
    protocol.update(episodes=2, probe_every=1)
    protocol["seeds"]["development"] = [0]
    source = tmp_path / "protocol.json"
    source.write_text(json.dumps(protocol))
    receipt = tmp_path / "run/receipt.json"
    assert audit.main(["--protocol", str(source), "--out", str(receipt)]) == 0
    assert audit.verify(receipt, current=True, artifacts=True)[0]
    original_bytes = receipt.read_bytes()
    with pytest.raises(SystemExit):
        audit.main(["--protocol", str(source), "--out", str(receipt)])
    assert receipt.read_bytes() == original_bytes
    body = json.loads(receipt.read_text())["body"]
    assert body["confirmation"] is False and not body["chamber"]["gates"]["passed"]
    for row, assays in zip(body["chamber"]["rows"], body["assays"], strict=True):
        plain = kd.run_life(row["arm"], row["seed"], row["delay"], protocol)
        assert plain["phases"] == row["phases"]
        assert plain["pending_outcome"] == row["pending_outcome"]
        assert [a["boundary"] for a in assays] == list(audit.BOUNDARIES)
        assert row["work"]["checkpoints"] - plain["work"]["checkpoints"] == 12
    original = json.loads(receipt.read_text())
    for key in ("checkpoints", "sweeps"):
        edited = copy.deepcopy(original)
        edited["body"]["assays"][0][0]["work"][key] += 1
        edited["digest"] = kd.canonical_sha256({k: edited[k] for k in ("kind", "body", "source")})
        receipt.write_text(kd.canonical_json(edited) + "\n")
        assert not audit.verify(receipt, artifacts=True)[0]
    receipt.write_text(kd.canonical_json(original) + "\n")
    artifact = receipt.parent / body["assays"][0][0]["artifacts"][0]["path"]
    artifact.write_bytes(b"damaged")
    assert not audit.verify(receipt, artifacts=True)[0]
    capsys.readouterr()


def test_eligibility_development_command_plan_is_bounded_without_running_lives(tmp_path):
    specification = importlib.util.spec_from_file_location(
        "eligibility_development", HERE / "eligibility_development.py"
    )
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    protocol = json.loads(helper.PROTOCOL.read_text())
    plan = helper.commands(tmp_path, protocol)
    assert [name for name, _ in plan] == ["control", "candidate"]
    for (_, command), value in zip(plan, (0.95, 0.98), strict=True):
        assert json.loads(command[command.index("--point") + 1]) == {"lam": value}
        assert command[command.index("--seeds") + 1 : command.index("--delays")] == ["2", "3"]
    for key, value in (
        ("seeds", [2, 3, 4]),
        ("maximum_lives", 6),
        ("base_protocol_sha256", "0" * 64),
    ):
        with pytest.raises(ValueError, match="only the declared four-life"):
            helper.commands(tmp_path, {**protocol, key: value})
