"""The links of the read-speech brain's chain read by gradient learners, on a fake cache."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ceiling", _here / "ceiling.py")
assert _spec is not None and _spec.loader is not None
ce = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("ceiling", ce)
_spec.loader.exec_module(ce)
sp = ce.sp

GENES = {"actor_eta": 0.0, "actor_eta_bias": 0.0, "actor_eta_critic": 0.0, "learned_fields": 4}


def _cache(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "meta.json").write_text(json.dumps(
        {"mels": 8, "window": 8, "mean": [0.0] * 8, "std": [1.0] * 8}))
    for name, n in (("train", 6), ("held", 2)):
        lengths = rng.integers(40, 60, size=n)
        texts = rng.integers(3, 5, size=n)
        np.savez(cache / f"{name}.npz", frames=rng.normal(size=(lengths.sum(), 8)),
                 lengths=lengths, text=rng.integers(1, 29, size=texts.sum()),
                 text_lengths=texts, identities=np.arange(n))
    return cache


def test_the_past_stands_beside_the_step() -> None:
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    (wide, _), = ce.with_past([(x, np.array([1]))], 2)
    assert wide.shape == (4, 9)
    assert wide[3].tolist() == [*x[3], *x[2], *x[1]]
    assert wide[0].tolist() == [*x[0], *x[0], *x[0]]  # the first step repeated before it


def test_a_wider_window_is_rebuilt_from_the_cached_frames(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    as_cached = sp.load(cache, "held")
    same = ce.wide_windows(cache, "held", 8)
    np.testing.assert_array_equal(same[0][0], as_cached[0][0])
    wide = ce.wide_windows(cache, "held", 12)
    assert wide[0][0].shape == (len(as_cached[0][0]), 8 * 12)


@pytest.mark.parametrize(("model", "loss"), [("linear", "ctc"), ("mlp", "path"),
                                             ("lstm", "ctc")])
def test_a_learner_reads_the_window(tmp_path: Path, model: str, loss: str) -> None:
    cache = _cache(tmp_path)
    out = tmp_path / "out"
    code = ce.main(["--cache", str(cache), "--hear", "window", "--model", model, "--width", "8",
                    "--loss", loss, "--align-every", "2", "--epochs", "2", "--past", "1",
                    "--out", str(out)])
    assert code == 0
    (path,) = out.glob("ceiling-*.json.gz")
    record = sp.lt.read_record(path)
    assert [q["epoch"] for q in record["probes"]] == [1, 2]
    assert record["inputs"] == 2 * 8 * 8 and 0 < record["parameters"]
    assert set(record["best"]) == {"epoch", "characters", "letter_from_sound",
                                   "emission_given_letter", "deletions_per_character"}


def test_a_learner_reads_a_saved_brains_fields_and_states(tmp_path: Path) -> None:
    import cadence as cd

    cache = _cache(tmp_path)
    brain = cd.Brain.compose(sp.inputs_of(cache), sp.SYMBOLS, backend="cpu", seed=0, **GENES)
    saved = brain.save(tmp_path / "brain.npz")
    held = sp.load(cache, "held")
    code = ce.fields_code(cd.Brain.load(saved, backend="cpu"), held)
    assert code[0][0].shape == (len(held[0][0]), 4 * 9)
    states = ce.settled_states(cd.Brain.load(saved, backend="cpu"), held, batch=2)
    assert states[0][0].shape == (len(held[0][0]), brain.connectome.n)
    assert states[1][0].shape == (len(held[1][0]), brain.connectome.n)
    out, keep = tmp_path / "out", tmp_path / "keep"
    for _ in range(2):  # the second run hears what the first kept
        ce.main(["--cache", str(cache), "--hear", "state", "--brain", str(saved),
                 "--populations", "sensory", "association", "--model", "linear",
                 "--epochs", "1", "--keep", str(keep), "--out", str(out)])
    assert (keep / "brain-state.npz").exists()
    (path,) = out.glob("ceiling-*.json.gz")
    record = sp.lt.read_record(path)
    assert record["inputs"] == 4 * 9 + 64 and record["populations"] == ["sensory", "association"]
