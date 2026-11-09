"""``learning_surprise_share`` left in 0.96.0 (#281): a surprise lesson takes its step over the
rows it teaches. The name is refused at construction and under retune, and a checkpoint whose
learner was saved with the share is refused by name; cadence 0.95.2 runs it."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd


def brain() -> cd.Brain:
    return cd.Brain.compose(6, 4, seed=7, episodic=False, learning_surprise=True, recall=False)


def test_the_share_is_refused_by_name_with_what_the_lesson_does_instead():
    for value in (True, False):
        with pytest.raises(ValueError, match="learning_surprise_share was retired in 0.96.0"):
            cd.Brain.compose(6, 4, seed=7, learning_surprise=True, learning_surprise_share=value)
        with pytest.raises(ValueError, match="learning_surprise_share was retired in 0.96.0"):
            brain().retune(learning_surprise_share=value)
    assert "surprise_share" not in cd.LearnerConfig(surprise=True).to_dict()


def test_a_checkpoint_saved_with_the_share_is_refused_and_one_without_loads(tmp_path):
    path = brain().save(tmp_path / "brain.npz")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    learner = json.loads(str(arrays["meta"]))
    learner["config"]["surprise_share"] = True  # as 0.95 saved a learner with the share
    np.savez(tmp_path / "shared.npz", **{**arrays, "meta": np.array(json.dumps(learner))})
    with pytest.raises(ValueError, match="surprise_share=True.*retired in 0.96.0.*0.95.2"):
        cd.Brain.load(tmp_path / "shared.npz")
    learner["config"]["surprise_share"] = False
    np.savez(tmp_path / "plain.npz", **{**arrays, "meta": np.array(json.dumps(learner))})
    loaded = cd.Brain.load(tmp_path / "plain.npz")
    assert loaded.learner.config.surprise
