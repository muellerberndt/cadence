"""Reject plausible falsifications of the actual, published CPU/CUDA evidence."""

import copy
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import cadence as cd

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def captured():
    path = ROOT / "benchmarks/system1_cuda/results/rtx4000-ada-2026-10-04.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


@pytest.fixture
def campaign(tmp_path, captured):
    # One actual matched pair suffices to exercise the verifier, without GPU work
    # or new learning. The compact checkpoints provide the standard config metadata.
    selected = copy.deepcopy(
        [
            r
            for r in captured["records"]
            if r["kind"] == "nursery" and r["seed"] == 0 and r["backend"] == "cpu"
        ]
    )
    frozen = copy.deepcopy(captured["campaign"])
    frozen["jobs"] = [[r[k] for k in ("kind", "seed", "backend", "revision")] for r in selected]
    outcomes = []
    for record in selected:
        folder = tmp_path / record["directory"]
        folder.mkdir()
        result = record.pop("result")
        manifest = {
            "commit": frozen["revisions"][record["revision"]],
            "protocol_sha256_lf": frozen["protocol_sha256_lf"],
            "harness_sha256_lf": frozen["harness_sha256_lf"]["run.py"],
            "source_sha256_lf": captured["source_sha256_lf"][record["revision"]],
            "backend": record["backend"],
            "seed": record["seed"],
            "protocol": frozen["protocol"],
            "dirty": "",
        }
        write(folder / "manifest.json", manifest)
        write(folder / "result.json", result)
        brain = cd.Brain.compose(
            8,
            2,
            seed=0,
            learning=cd.LearnerConfig(**result["graph"]["learning"]),
            reward=cd.ActorCriticConfig(**result["graph"]["reward"]),
        )
        brain.save(folder / "acquired.npz")
        outcomes.append(record)
    write(tmp_path / "campaign.json", frozen)
    write(tmp_path / "outcomes.json", outcomes)
    write(tmp_path / "completion.json", {"harness_and_revisions_unchanged": True})
    return tmp_path, selected


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def verify(path):
    spec = importlib.util.spec_from_file_location(
        "system1_cuda_verify", ROOT / "benchmarks/system1_cuda/verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify(path)


def test_unmodified_captured_pair_passes(campaign):
    folder, _ = campaign
    assert verify(folder)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_job",
        "changed_seed",
        "wrong_device",
        "weaker_tolerance",
        "checkpoint_recipe",
        "false_admission",
        "changed_action",
        "duplicate_live_row",
        "omitted_latency_stage",
        "wrong_deadline_count",
        "changed_work",
    ],
)
def test_inconsistent_evidence_cannot_pass(campaign, change):
    folder, selected = campaign
    candidate = next(r for r in selected if r["revision"] == "candidate")
    directory = folder / candidate["directory"]
    manifest_path, result_path = directory / "manifest.json", directory / "result.json"
    manifest = json.loads(manifest_path.read_text())
    result = json.loads(result_path.read_text())
    if change == "duplicate_job":
        outcomes = json.loads((folder / "outcomes.json").read_text())
        outcomes[1] = outcomes[0]
        write(folder / "outcomes.json", outcomes)
    elif change == "changed_seed":
        manifest["seed"] = 999
    elif change == "wrong_device":
        result["execution"]["device"] = "cuda:0"
    elif change == "weaker_tolerance":
        result["graph"]["learning"]["tolerance"] = 1.0
    elif change == "checkpoint_recipe":
        path = directory / "acquired.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        meta = json.loads(str(arrays["meta"]))
        meta["config"]["tolerance"] = 1.0
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
    elif change == "false_admission":
        result["live"][0]["settlement"]["residual"][0] = 1.0
    elif change == "changed_action":
        row = result["live"][0]
        row["actions"][0] ^= 1
        # Keep its score internally consistent: the paired behavior must still fail.
        row["correct"] = sum(a == b for a, b in zip(row["actions"], row["labels"], strict=True))
    elif change == "duplicate_live_row":
        result["live"][1] = copy.deepcopy(result["live"][0])
    elif change == "omitted_latency_stage":
        result["latencies"].pop("repair")
    elif change == "wrong_deadline_count":
        result["latencies"]["stable"]["deadline_misses"] += 1
    elif change == "changed_work":
        result["counts"]["work"]["row_sweeps"] += 1
    write(manifest_path, manifest)
    write(result_path, result)
    assert not verify(folder)["passed"], change
