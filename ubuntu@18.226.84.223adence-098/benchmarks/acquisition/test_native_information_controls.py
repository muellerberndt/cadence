"""Admission tests only: no reference fitting, neural queries or teaching."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_information_controls_admission",
    Path(__file__).with_name("native_information_controls.py"),
)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


@pytest.fixture
def capsule(tmp_path):
    original, static, semantics = (tmp_path / name for name in ("original", "static", "semantics"))
    fixture = original / "source/fixture"
    fixture.mkdir(parents=True)
    (original / "selected-24").mkdir()
    static.mkdir()
    semantics.mkdir()
    verification_sources = tmp_path / "verification-sources"
    verification_sources.mkdir()
    script = verification_sources / "native_modality_static_audit.py"
    script.write_text("# static arithmetic producer\n")
    archive, provenance = {}, {"panels": {}}
    for index, (name, count) in enumerate(reference.PANELS.items()):
        x = np.zeros((count, 650))
        x[:, 0] = np.arange(count) + index / 10
        y = np.arange(count, dtype=np.int64)
        archive[name + "_inputs"], archive[name + "_labels"] = x, y
        provenance["panels"][name] = [
            {"input_sha256": reference.sha_bytes(row), "label": int(label)}
            for row, label in zip(x, y, strict=True)
        ]
    # Loading this member with allow_pickle=False raises. Admission must leave it sealed.
    archive["heldout_inputs"] = np.array([object()], dtype=object)
    np.savez_compressed(fixture / "school.npz", **archive)
    reference.write(fixture / "provenance.json", provenance)
    reference.write(
        original / "protocol.json",
        {"fixture": {"school.npz": reference.sha(fixture / "school.npz")}},
    )
    reference.write(original / "summary.json", {"scope": "synthetic admission fixture"})
    (original / "source/extract.py").write_text("# fixed body codec\n")
    for name in ("initial", "final"):
        (original / f"selected-24/{name}.npz").write_bytes(name.encode())
    for name in ("eye.py", "collect.py"):
        (semantics / name).write_text("# current semantic reference\n")
    np.savez_compressed(static / "direct-current-arrays.npz", initial_sensory_s=np.zeros((61, 650)))
    static_job = {
        "script_sha256": reference.sha(script),
        "source_data_pins": {
            str(path.relative_to(original)): reference.sha(path)
            for path in (
                original / "protocol.json",
                original / "summary.json",
                fixture / "school.npz",
                fixture / "provenance.json",
                original / "selected-24/initial.npz",
                original / "selected-24/final.npz",
            )
        },
    }
    reference.write(static / "job-protocol.json", static_job)
    reference.write(
        static / "result.json",
        {
            "verified": True,
            "heldout_decoded": False,
            "job_protocol_sha256": reference.sha(static / "job-protocol.json"),
            "raw_arrays_sha256": reference.sha(static / "direct-current-arrays.npz"),
        },
    )
    root = tmp_path / "prepared"
    reference.prepare(root, original, static, semantics)
    return root, json.loads((root / "protocol.json").read_text())


def test_admission_reads_only_declared_panels_and_charges_no_fits(capsule):
    root, protocol = capsule
    reference.guard(root, protocol, runtime=False)
    panels = reference.read_panels(root)
    assert {name: len(value["labels"]) for name, value in panels.items()} == reference.PANELS
    admission = json.loads((root / "admission.json").read_text())
    assert admission["reference_fit_calls"] == admission["neural_queries"] == 0
    assert admission["heldout_decoded"] is False
    assert not (root / "execution-started.json").exists()
    with np.load(root / "source/school.npz", allow_pickle=False) as archive:
        with pytest.raises(ValueError, match="Object arrays"):
            archive["heldout_inputs"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("seconds_cap", 61),
        ("fit_budget", 95),
        ("information_gates", {"school": 18, "independent_train": 13, "development": 15}),
        ("primary", {"arm": "expanded-train", "feature": "raw_full", "readout": "factorized"}),
    ],
)
def test_bounds_roles_and_original_gates_are_frozen(capsule, key, value):
    root, protocol = capsule
    altered = copy.deepcopy(protocol)
    altered[key] = value
    with pytest.raises(ValueError, match="scope/config/bounds"):
        reference.guard(root, altered, runtime=False)


def test_feature_producer_and_source_bytes_are_bound(capsule):
    root, protocol = capsule
    path = root / "source/static-script.py"
    path.write_text(path.read_text() + "# altered\n")
    with pytest.raises(ValueError, match="source/data changed"):
        reference.guard(root, protocol, runtime=False)


@pytest.mark.parametrize("explicit", (False, True))
def test_static_producer_source_is_portable_without_workspace_evidence(capsule, explicit):
    original_root, _protocol = capsule
    parent = original_root.parent
    historical_script = parent / "verification-sources/native_modality_static_audit.py"
    portable_script = parent / "static/static-script.py"
    portable_script.write_bytes(historical_script.read_bytes())
    historical_script.unlink()
    root = parent / "portable-prepared"
    options = {"static_script": portable_script} if explicit else {}
    reference.prepare(
        root, parent / "original", parent / "static", parent / "semantics", **options
    )
    protocol = json.loads((root / "protocol.json").read_text())
    reference.guard(root, protocol, runtime=False)
    assert reference.sha(root / "source/static-script.py") == reference.sha(portable_script)
    assert not (root / "execution-started.json").exists()


def test_changed_static_producer_is_rejected_before_creating_attempt(capsule):
    original_root, _protocol = capsule
    parent = original_root.parent
    changed = parent / "changed-static-script.py"
    changed.write_text("# different arithmetic producer\n")
    root = parent / "invalid-prepared"
    with pytest.raises(ValueError, match="producer source differs"):
        reference.prepare(
            root, parent / "original", parent / "static", parent / "semantics",
            static_script=changed,
        )
    assert not root.exists()


def test_known_body_codec_roundtrips_all_executed_actions_without_fitting():
    labels = np.arange(36, dtype=np.int64)
    for readout in ("factorized", "categorical"):
        np.testing.assert_array_equal(
            reference.labels_from_scores(reference.target_scores(labels, readout), readout), labels
        )


def test_fit_census_and_conditioning_are_fixed_without_validation_information():
    count = len(reference.ARMS) * (
        len(reference.MODES) * len(reference.LAMBDAS)
        + len(reference.FIXED["extra_categorical_controls"])
    )
    assert count == reference.FIXED["fit_budget"] == 94
    train = np.array([[1.0, 0.0], [3.0, 2.0]])
    arrays = {"school": train, "development": np.full((19, 2), 1e9)}
    z, tests, _kernel, metadata = reference.condition(train, arrays, np.ones(2), "raw_full")
    np.testing.assert_array_equal(metadata["mean"], [2.0, 1.0])
    np.testing.assert_allclose(z.mean(axis=0), 0)
    assert np.all(tests["development"] > 1e8)


def test_same_job_cannot_be_relaunched(capsule):
    root, _protocol = capsule
    (root / "execution-started.json").write_text("prior attempt retained\n")
    with pytest.raises(ValueError, match="already attempted"):
        reference.launch(root)
    assert not (root / "worker.log").exists()
