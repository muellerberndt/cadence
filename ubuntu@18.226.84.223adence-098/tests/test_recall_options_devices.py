"""The recall options of 0.93 (the anchor, the slope, the relative read and the when regions;
cadence#232, #234, #237) on the accelerated backends: a brain saved on the CPU and loaded on a
torch device or on MLX acts as on the CPU, and its lessons move the options' maps and the
synapses as the CPU's do, within the backend's rounding."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd

OPTIONS = dict(
    recall=True, recall_founder=True, recall_recency=0.0, recall_boundary=True, recall_anchor=True,
    recall_slope=0.1, recall_relative=True, recall_when=True, recall_position_rate=0.05,
)
TOKENS, CUE = 4, 4


def story(rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
    """Five tokens on four streams, then a cue whose answer is the first token."""
    seq = rng.integers(TOKENS, size=(4, 5))
    moments = [np.eye(TOKENS + 1)[seq[:, t]] for t in range(5)]
    return moments + [np.eye(TOKENS + 1)[np.full(4, CUE)]], seq[:, 0]


def maps(brain: cd.Brain) -> np.ndarray:
    recall = brain.recall
    parts = [recall.position_maps, recall.anchor_maps, recall.slope_maps, recall.relative_maps]
    return np.concatenate([np.ravel(a) for group in parts for head in group for a in head])


def lessons(brain: cd.Brain) -> list[np.ndarray]:
    rng, acts = np.random.default_rng(0), []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(3):
            brain.reset()
            moments, label = story(rng)
            for x in moments[:-1]:
                acts.append(np.asarray(brain.act(x, greedy=True)))
            brain.lesson(moments[-1], label)
    return acts


@pytest.mark.parametrize(
    ("backend", "device", "precision", "atol"),
    [("torch", "cpu", "float64", 1e-9), ("torch", "mps", "float32", 1e-4),
     ("mlx", None, None, 1e-4)],
    ids=["torch_cpu64", "mps32", "mlx"],
)
def test_the_options_learn_on_a_device_as_on_the_cpu(tmp_path, backend, device, precision, atol):
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    if device == "mps":
        torch = pytest.importorskip("torch")
        if not torch.backends.mps.is_available():
            pytest.skip("MPS hardware unavailable")
    cpu = cd.Brain.compose(TOKENS + 1, TOKENS, seed=5, modules=(16,), episodic=False,
                           teaching="normalized", **{"recall": False, **OPTIONS})
    cpu.save(tmp_path / "brain.npz")
    moved = cd.Brain.load(tmp_path / "brain.npz", backend=backend, device=device,
                          precision=precision)
    assert moved.brain.backend == backend
    newborn = maps(cpu)
    cpu_acts, moved_acts = lessons(cpu), lessons(moved)
    np.testing.assert_array_equal(cpu_acts[0], moved_acts[0])
    assert np.abs(maps(cpu) - newborn).max() > 1e-3  # the lessons moved the options' maps
    np.testing.assert_allclose(maps(moved), maps(cpu), rtol=0, atol=atol)
    np.testing.assert_allclose(np.asarray(moved.brain.efficacy, dtype=float),
                               np.asarray(cpu.brain.efficacy, dtype=float), rtol=0, atol=atol)
    regions = cpu.recall.when_regions(cpu.recall.heads)
    when = np.isin(cpu.connectome.pre, np.concatenate(
        [np.asarray(cpu.connectome.populations[name]) for name in regions]))
    assert np.abs(np.asarray(cpu.brain.efficacy)[when]).max() > 0  # the when synapses learned
