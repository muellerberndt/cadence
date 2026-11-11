"""The read-speech adapter, without the corpus; the corpus test skips where it is absent."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("speech", _here / "speech.py")
assert _spec is not None and _spec.loader is not None
sp = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("speech", sp)
_spec.loader.exec_module(sp)


def test_steps_are_forty_milliseconds_of_log_mel() -> None:
    bank = sp.mel_bank(40)
    assert bank.shape == (40, 257) and (bank >= 0).all()
    second = np.sin(np.linspace(0, 2000 * np.pi, sp.RATE))
    steps = sp.steps_of(second, bank)
    assert steps.shape == (24, 160) and np.isfinite(steps).all()


def test_text_round_trips_as_symbols() -> None:
    symbols = sp.encode("Don't  STOP, now")
    assert sp.decode(symbols.tolist()) == "don't stop now"
    assert symbols.max() < sp.SYMBOLS and (symbols > 0).all()
    path = sp.even_path(10, symbols[:3])
    assert sp.lt.collapse(path) == symbols[:3].tolist()


def test_the_reading_counts_characters_and_words() -> None:
    text = sp.encode("ab cd")
    held = [(np.zeros((8, 4)), text)]
    perfect = [np.array([1, 1, 0, 2, 28, 3, 0, 4])]
    assert sp.read(perfect, held)["characters"] == 1.0
    spaced = [np.array([1, 1, 0, 2, 0, 3, 0, 4])]
    out = sp.read(spaced, held)
    assert out["words"] < 1.0 and out["words_lost_on_spacing_alone"] == 1.0


def test_the_located_terms_read_one_utterance() -> None:
    text = sp.encode("ab")
    logp = np.log(np.full((4, sp.SYMBOLS), 0.01))
    for t, symbol in enumerate([1, 1, 2, 0]):
        logp[t, symbol] = 0.0
    said = sp.locate([np.array([1, 1, 2, 0])], [logp], [(np.zeros((4, 1)), text)])
    assert said["letter_from_sound"] == 1.0 and said["emission_given_letter"] == 1.0
    assert said["word_given_letters"] == 1.0
    silent = sp.locate([np.zeros(4, dtype=np.int64)], [logp], [(np.zeros((4, 1)), text)])
    assert silent["letter_from_sound"] == 1.0 and silent["emission_given_letter"] == 0.0


ROOT = os.environ.get("LIBRISPEECH_ROOT")


@pytest.mark.skipif(not ROOT or not (Path(ROOT) / "LibriSpeech" / "dev-clean").exists(),
                    reason="LibriSpeech is not on this machine (LIBRISPEECH_ROOT)")
def test_the_corpus_lists_utterances() -> None:
    listing = sp.utterances(Path(str(ROOT)), "dev-clean")
    assert len(listing) == 2703
    assert listing[0][1].exists()


def test_a_window_cache_hears_each_step_as_an_image(tmp_path: Path) -> None:
    import json

    frames = np.arange(40 * 30, dtype=np.float32).reshape(30, 40) % 7
    windows = sp.windows_of(frames.astype(np.float64), 8)
    assert windows.shape == (7, 8, 40)
    np.testing.assert_allclose(windows.mean(axis=(1, 2)), 0.0, atol=1e-9)
    meta = {"mels": 40, "window": 8, "mean": [0.0] * 40, "std": [1.0] * 40}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    np.savez(tmp_path / "held.npz", frames=frames, lengths=np.array([30]),
             text=np.array([1, 2]), text_lengths=np.array([2]), identities=np.array(["u"]))
    (x, text), = sp.load(tmp_path, "held")
    assert x.shape == (7, 320) and text.tolist() == [1, 2]
    assert sp.inputs_of(tmp_path) == (40, 8)
    assert np.abs(x).max() <= sp.SCALE * sp.CLIP_SIGMA + 1e-6


def test_a_batched_life_hears_utterances_side_by_side_and_resumes(tmp_path: Path) -> None:
    import json

    rng = np.random.default_rng(0)
    cache = tmp_path / "cache"
    cache.mkdir()
    meta = {"mels": 8, "window": 8, "mean": [0.0] * 8, "std": [1.0] * 8}
    (cache / "meta.json").write_text(json.dumps(meta))
    for name, n in (("train", 6), ("held", 2)):
        lengths = rng.integers(40, 60, size=n)
        texts = rng.integers(3, 5, size=n)
        np.savez(cache / f"{name}.npz", frames=rng.normal(size=(lengths.sum(), 8)),
                 lengths=lengths, text=rng.integers(1, 29, size=texts.sum()),
                 text_lengths=texts, identities=np.arange(n))
    (tmp_path / "list.txt").write_text("ab 3\nc 1\n")
    protocol = {"batch": 2, "align_every": 1, "probe_every": 2,
                "word_list": str(tmp_path / "list.txt"),
                "brain": {"genes": {"actor_eta": 0.0, "actor_eta_bias": 0.0,
                                    "actor_eta_critic": 0.0, "learned_fields": 4}}}
    out = tmp_path / "out"
    out.mkdir()
    full = sp.life_batched(cache, protocol, 1, out)
    assert [p["batches"] for p in full["probes"]] == [2, 3]
    assert "words_matched" in full["probes"][-1]["score"]
    record = sp.lt.read_record(out / "speech-brain-1.partial.json.gz")
    record["probes"] = record["probes"][:1]
    sp.lt.write_record(out / "speech-brain-1.partial.json.gz", record)
    again = sp.life_batched(cache, protocol, 1, out)
    assert again["resumed"] == [2]
    assert again["probes"][-1]["score"]["characters"] == full["probes"][-1]["score"]["characters"]


def test_two_passes_resume_between_re_alignments(tmp_path: Path) -> None:
    """An utterance heard in both passes takes a path in each; a life resumed between two
    re-alignments takes its batches' paths from the copy of the last one."""
    import json

    rng = np.random.default_rng(1)
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
    protocol = {"batch": 2, "passes": 2, "align_every": 2, "probe_every": 3,
                "brain": {"genes": {"actor_eta": 0.0, "actor_eta_bias": 0.0,
                                    "actor_eta_critic": 0.0, "learned_fields": 4}}}
    out = tmp_path / "out"
    out.mkdir()
    full = sp.life_batched(cache, protocol, 2, out)
    assert [p["batches"] for p in full["probes"]] == [3, 6]
    record = sp.lt.read_record(out / "speech-brain-2.partial.json.gz")
    record["probes"] = record["probes"][:1]
    sp.lt.write_record(out / "speech-brain-2.partial.json.gz", record)
    again = sp.life_batched(cache, protocol, 2, out)
    assert again["resumed"] == [3] and again["probes"][-1]["batches"] == 6
    assert again["resumed_alignment"][0]["batches"] == 3
