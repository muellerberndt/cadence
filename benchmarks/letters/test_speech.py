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


def _fake_cache(tmp_path: Path, seed: int, train: int = 6) -> Path:
    import json

    rng = np.random.default_rng(seed)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "meta.json").write_text(json.dumps(
        {"mels": 8, "window": 8, "mean": [0.0] * 8, "std": [1.0] * 8}))
    for name, n in (("train", train), ("held", 2)):
        lengths = rng.integers(40, 60, size=n)
        texts = rng.integers(3, 5, size=n)
        np.savez(cache / f"{name}.npz", frames=rng.normal(size=(lengths.sum(), 8)),
                 lengths=lengths, text=rng.integers(1, 29, size=texts.sum()),
                 text_lengths=texts, identities=np.arange(n))
    return cache


GENES = {"actor_eta": 0.0, "actor_eta_bias": 0.0, "actor_eta_critic": 0.0, "learned_fields": 4}


def test_a_batched_life_hears_utterances_side_by_side(tmp_path: Path) -> None:
    cache = _fake_cache(tmp_path, 0)
    (tmp_path / "list.txt").write_text("ab 3\nc 1\n")
    protocol = {"batch": 2, "align_every": 1, "probe_every": 2, "transcripts": True,
                "word_list": str(tmp_path / "list.txt"), "brain": {"genes": GENES}}
    out = tmp_path / "out"
    out.mkdir()
    full = sp.life_batched(cache, protocol, 1, out)
    assert full is not None
    assert [p["batches"] for p in full["probes"]] == [2, 3]
    last = full["probes"][-1]
    assert "words_matched" in last["score"] and len(last["transcripts"]) == 2
    assert np.asarray(last["confusions"]).shape == (28, 28)
    assert int(np.sum(last["confusions"])) == sum(len(t) for _, t in sp.load(cache, "held"))
    assert set(full["probes"][-1]["clock"]) == {"training", "reading", "aligning"}


def test_a_stopped_life_continues_from_its_last_re_alignment(tmp_path: Path) -> None:
    """Two passes: an utterance heard in both takes a path in each. A life stopped between two
    re-alignments continues from the brain saved at the last one and reads as the whole life."""
    cache = _fake_cache(tmp_path, 1)
    protocol = {"batch": 2, "passes": 2, "align_every": 2, "probe_every": 3,
                "brain": {"genes": GENES}}
    whole, cut = tmp_path / "whole", tmp_path / "cut"
    whole.mkdir()
    cut.mkdir()
    full = sp.life_batched(cache, protocol, 2, whole)
    assert full is not None and [p["batches"] for p in full["probes"]] == [3, 6]
    assert sp.life_batched(cache, protocol, 2, cut, stop_after=5) is None
    saved = sp.latest_slot(cut, 2)
    assert saved is not None and saved[1]["batches"] == 4
    again = sp.life_batched(cache, protocol, 2, cut)
    assert again is not None and again["resumed"] == [4]
    assert [p["batches"] for p in again["probes"]] == [3, 6]
    for a, b in zip(full["probes"], again["probes"], strict=True):
        assert a["score"]["characters"] == b["score"]["characters"]
        assert a["steps"] == b["steps"]
    # a save whose file does not match its recorded sha256 is passed over
    (cut / "speech-2-align-a.npz").write_bytes(b"torn")
    (cut / "speech-2-align-b.npz").write_bytes(b"torn")
    assert sp.latest_slot(cut, 2) is None


def test_a_stopped_control_continues_from_its_last_pass(tmp_path: Path) -> None:
    cache = _fake_cache(tmp_path, 2)
    protocol = {"passes": 3, "read_each_pass": True, "transcripts": True}
    whole, cut = tmp_path / "whole", tmp_path / "cut"
    whole.mkdir()
    cut.mkdir()
    full = sp.control(cache, "lstm", "online", 3, 4000, protocol, out=whole)
    assert full is not None and [p["passes"] for p in full["probes"]] == [1, 2, 3]
    assert sp.control(cache, "lstm", "online", 3, 4000, protocol, out=cut, stop_after=2) is None
    again = sp.control(cache, "lstm", "online", 3, 4000, protocol, out=cut)
    assert again is not None and again["resumed"] == [2] and again["passes"] == 3
    assert [p["score"]["characters"] for p in again["probes"]] == [
        p["score"]["characters"] for p in full["probes"]]
    offline = sp.control(cache, "transformer", "offline", 3, 4000,
                         {"offline_epochs": 2, "read_each_pass": True})
    assert offline is not None and offline["passes"] == 2
    assert len(offline["probes"]) == 2
    once = sp.control(cache, "lstm", "offline", 3, 4000, {}, epochs=2)
    assert once is not None and len(once["probes"]) == 1 and once["probes"][0]["passes"] == 2
