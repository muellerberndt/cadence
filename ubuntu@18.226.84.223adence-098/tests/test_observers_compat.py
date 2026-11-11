"""System 2 observers beside every memory, the learned code and the apical compartment: a life of
acts, lessons, `live` with an episode end and imagination, then save, load and retune."""

import importlib.util

import numpy as np
import pytest

import cadence as cd

BASE = {"inputs": 8, "actions": 3, "seed": 5, "observers": (6, 4), "arousal": True,
        "modules": (12, 8)}
CASES = {
    "plain": {},
    "apical": {"apical": 0.25, "teaching": "deep"},
    "recall and boundary": {"recall": True, "recall_founder": True, "recall_boundary": True, "apical": 0.25},
    "held heads": {"learned_code": 8, "attention": True, "recall": True, "recall_heads": 2,
                   "apical": 0.25},
    "episodic": {"memory_choice": True, "apical": 0.25},
    "efference and history": {"efference_amplitude": 1.0, "history_decays": (0.5, 0.9),
                              "apical": 0.25},
    "torch": {"apical": 0.25, "recall": True, "recall_founder": True, "backend": "torch"},
    "tokens and own actions": {"recall": True, "recall_founder": True, "recall_tokens": [0, 1, 2],
                             "recall_action": True, "apical": 0.25},
    "held heads, tokens and own actions": {
        "learned_code": 8, "attention": True, "recall": True, "recall_heads": 2,
        "recall_tokens": [0, 1, 2], "recall_action": True, "apical": 0.25,
    },
    "torch tokens and own actions": {"apical": 0.25, "recall": True, "recall_founder": True,
                                   "recall_tokens": [0, 1, 2], "recall_action": True,
                                   "backend": "torch"},
}


def life(brain: cd.Brain) -> None:
    rng = np.random.default_rng(1)

    def x() -> np.ndarray:
        return np.eye(8)[[int(rng.integers(8))]]

    for _ in range(3):
        brain.act(x(), greedy=True)
    for t in range(3):
        brain.lesson(x(), np.array([t % 3]))
        brain.act(x(), greedy=True)
    brain.live(x())
    for t in range(4):
        brain.live(x(), reward=[float(t % 2)])
    brain.reset(keep=("slots",))
    brain.live(x())
    brain.live(x(), reward=[1.0])
    brain.imagine([x(), x()])


# lessons of a tiny untaught brain may warn that a finite phase used its whole budget
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name", list(CASES))
def test_observers_live_with_every_component(name, tmp_path):
    options = CASES[name]
    if options.get("backend") == "torch" and importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    brain = cd.Brain.compose(**BASE, **{"recall": False, **options})
    assert {"observer_0", "observer_1"} <= set(brain.connectome.populations)
    if "apical" in options:
        # an observer's return lands apically: module_0, association, motor and observer_0 pool
        assert len(brain.brain.apical.members) == 4
    life(brain)
    loaded = cd.Brain.load(brain.save(tmp_path / "brain.npz"))
    described = brain.describe()
    described.pop("placement", None)  # it ran on torch and now runs on the CPU
    assert loaded.describe() == described
    x = np.eye(8)[[2]]
    np.testing.assert_array_equal(loaded.act(x, greedy=True), brain.act(x, greedy=True))
    brain.retune(working_memory_decay=0.4)
    assert brain.act(np.eye(8)[[3]], greedy=True).shape == (1,)

