"""The opt-in settings of 0.93 through a brain's life cycle: the construction settings retune
refuses, the settings it takes, which act as on a brain built with them and are saved, and a
brain grown into the recall options, or grown more heads with them, that chooses as the trained
brain did and saves and loads as it chose (cadence#202, #210, #232, #234, #236, #237, #238,
#239)."""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pytest
from checkpoints import earlier, metadata

import cadence as cd

RNG = np.random.default_rng(0)
X = [RNG.random((2, 6)) for _ in range(6)]
POSITIONS = {"recall_anchor": True, "recall_slope": 0.1, "recall_relative": True,
             "recall_when": True}
# the recall settings 0.93.0 retired: a value each took, and the brain that could take it
RETIRED = {
    "recall_address_step": ("toward", {}),
    "recall_heard": (2.0, {}),
    "recall_address_temperature": (0.05, {}),
    "recall_window_decay": (0.5, {}),
}


def lived(**options) -> cd.Brain:
    """A brain of six inputs and three actions after a few rewarded and taught steps."""
    rng = np.random.default_rng(1)
    brain = cd.Brain.compose(6, 3, modules=(16,), seed=2, **options)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain.step(X[0])
        for x in X[1:4]:
            brain.step(x, reward=rng.normal(size=2), teacher=rng.integers(0, 3, size=2))
    return brain


def greedy(brain: cd.Brain) -> list:
    brain.reset()
    return [brain.act(x, greedy=True).tolist() for x in X]


@pytest.mark.parametrize(
    ("name", "value"),
    [*POSITIONS.items(), ("memory_context_hold", "unchanged"), ("memory_extinction", 0.5)],
)
def test_retune_refuses_the_construction_settings(name, value):
    brain = cd.Brain.compose(6, 3, modules=(16,), seed=2, recall=True, recall_heads=2,
                             memory_choice=True)
    with pytest.raises(ValueError, match=f"{name} is a construction setting"):
        brain.retune(**{name: value})


@pytest.mark.parametrize("name", sorted(RETIRED))
def test_a_setting_retired_in_this_release_is_refused_naming_the_release(tmp_path, name):
    """A recall setting retired in 0.93.0 is refused at compose, retune and load with the one
    sentence of ``Recall.removed``, which names 0.93.0 and what the memory does instead; a
    founder checkpoint, and one that carries the setting at the value the memory keeps, load
    unchanged."""
    value, base = RETIRED[name]
    setting = name[len("recall_"):]
    release, kept, _ = cd.Recall.RETIRED[setting]
    sentence = cd.Recall.removed(setting)
    assert release == "0.93.0" and sentence.startswith(f"{name} was retired in 0.93.0: ")
    brain = cd.Brain.compose(6, 3, modules=(16,), seed=2, recall=True, **base)
    with pytest.raises(ValueError, match=re.escape(sentence)):
        cd.Brain.compose(6, 3, modules=(16,), seed=2, recall=True, **base, **{name: value})
    with pytest.raises(ValueError, match=re.escape(sentence)):
        brain.retune(**{name: value})
    brain.act(X[0], greedy=True)
    path = brain.save(tmp_path / "founder.npz")
    described = brain.describe()
    assert name not in described["genes"]
    with np.load(path, allow_pickle=False) as data:
        arrays = earlier({member: data[member].copy() for member in data.files})
    meta = metadata(arrays)
    meta["recall"][setting] = value  # as a file of 0.92.1 with the option set carries it
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "retired.npz", **arrays)
    with pytest.raises(ValueError, match=re.escape(sentence)):
        cd.Brain.load(tmp_path / "retired.npz")
    assert cd.Brain.load(path).describe() == described
    meta["recall"][setting] = kept
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "kept.npz", **arrays)
    assert cd.Brain.load(tmp_path / "kept.npz").describe() == described


def test_the_withdrawn_chain_setting_is_unknown():
    """A chained second head was built on the release branch and withdrawn before release:
    its setting is refused as any unknown setting is."""
    with pytest.raises(TypeError, match="unknown brain setting: 'recall_chain'"):
        cd.Brain.compose(6, 3, modules=(16,), seed=2, recall=True, recall_heads=2,
                         recall_chain=True)


def taught(brain: cd.Brain) -> np.ndarray:
    """The synapses and recall maps after a lesson whose rows are half wrong, six rewarded
    steps and a taught one."""
    rng = np.random.default_rng(3)
    rows = [rng.random((4, 6)) for _ in range(9)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        answer = np.asarray(brain.act(rows[0], greedy=True)).reshape(-1)
        labels = answer.copy()
        labels[:2] = (answer[:2] + 1) % 3
        brain.lesson(rows[0], labels)
        brain.step(rows[1])
        for x in rows[2:8]:
            brain.step(x, reward=rng.choice([-1.0, 1.0], size=4))
        brain.step(rows[8], reward=np.ones(4), teacher=labels)
    parts = [np.asarray(brain.brain.efficacy, dtype=float)]
    if brain.recall is not None:
        parts += [np.ravel(a) for head in brain.recall.position_maps for a in head]
    return np.concatenate(parts)


@pytest.mark.parametrize(
    ("name", "value", "base"),
    [("learning_surprise_share", True, {"learning_surprise": True}),
     ("learning_rehearsal", 1.0, {}),
     ("recall_position_reward", 1.0, {"recall": True})],
)
def test_a_retuned_setting_acts_as_built_and_is_saved(tmp_path, name, value, base):
    plain = cd.Brain.compose(6, 3, modules=(16,), seed=2, **base)
    retuned = cd.Brain.compose(6, 3, modules=(16,), seed=2, **base)
    retuned.retune(**{name: value})
    built = cd.Brain.compose(6, 3, modules=(16,), seed=2, **base, **{name: value})
    expected = taught(built)
    assert not np.array_equal(taught(plain), expected)  # the setting changes what is learned
    np.testing.assert_array_equal(taught(retuned), expected)
    again = cd.Brain.load(retuned.save(tmp_path / "retuned.npz"))
    assert again.describe()["genes"][name] == value


@pytest.mark.parametrize(
    ("parts", "options"),
    [({"recall": True, **POSITIONS}, {}),
     ({"recall_heads": 2}, {"recall": True, "recall_when": True, "recall_anchor": True}),
     ({"recall_heads": 2}, {"recall": True, "recall_position_reward": 1.0,
                            "recall_address_reward": 1.0, "recall_anchor": True}),
     ({"recall_heads": 3}, {"recall": True, "recall_heads": 2, "recall_anchor": True})],
    ids=["positions", "when_heads", "position_reward_heads", "three_heads"],
)
def test_a_brain_grows_into_the_recall_options_and_saves_what_it_grew(tmp_path, parts, options):
    brain = lived(**options)
    grown = brain.grow(**parts)
    genes = grown.describe()["genes"]
    assert all(genes[k] == v for k, v in {**options, **parts}.items() if k != "recall")
    assert greedy(grown) == greedy(brain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grown.lesson(X[4], np.array([1, 2]))
    again = cd.Brain.load(grown.save(tmp_path / "grown.npz"))
    assert greedy(again) == greedy(grown)
