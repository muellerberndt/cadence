"""Expert scalar configurations retain their meaning through checkpoint continuation."""

import json

import numpy as np
import pytest

import cadence as cd


def numpy_learning():
    return cd.LearnerConfig(
        free_steps=np.int64(1024), nudged_steps=np.int64(12), damping=np.int64(3),
        eta=np.float32(0.02), eta_bias=np.float32(0.002),
        temperature=np.float32(0.2), momentum=np.float32(0.5),
    )


def configured_brain():
    return cd.Brain.compose(
        2, 2, modules=(4,), seed=3,
        arousal=cd.ArousalConfig(
            heat=np.float32(2.1), youth=np.int64(100), need=np.float32(0.1),
        ),
        learning=numpy_learning(),
        reward=cd.ActorCriticConfig(
            gamma=np.float32(0.9), lam=np.float32(0.8), eta=np.float32(0.03),
            eta_bias=np.float32(0.003), eta_critic=np.float32(0.3),
            eligibility_steps=np.int64(12),
        ), recall=False,
    )


def assert_learner_equal(left, right):
    assert left.config == right.config
    assert left.updates == right.updates
    for name in ("efficacy", "bias", "log_gain"):
        np.testing.assert_array_equal(getattr(left.brain, name), getattr(right.brain, name))
    for name in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def test_numpy_scalar_learner_config_resumes_the_next_teaching_update(tmp_path):
    graph = cd.layered(2, 4, 2, seed=3)
    learner = cd.Learner(
        cd.NeuralGraph(graph, cd.learning_neuron_model(dt=1)),
        graph.populations["output"], numpy_learning(),
    )
    drive = np.zeros((2, graph.n))
    drive[:, graph.populations["input"]] = np.eye(2)
    learner.step(drive, np.array([0, 1]))
    restored = cd.Learner.load(learner.save(tmp_path / "learner"))
    assert_learner_equal(learner, restored)
    for labels in (np.array([1, 0]), np.array([0, 1])):
        learner.step(drive, labels)
        restored.step(drive, labels)
        assert_learner_equal(learner, restored)


def test_configs_normalize_numpy_numbers_without_converting_booleans():
    brain = configured_brain()
    for config in (brain.learner.config, brain.basal_ganglia.config, brain.arousal.config):
        assert not any(isinstance(getattr(config, name), np.generic) for name in config.__slots__)
    with pytest.raises(ValueError, match="youth"):
        cd.ArousalConfig(youth=np.bool_(True))
    with pytest.raises(ValueError, match="eligibility_steps"):
        cd.ActorCriticConfig(eligibility_steps=np.bool_(True))


def test_numpy_scalar_brain_config_resumes_pending_feedback(tmp_path):
    brain = configured_brain()
    brain.live([[1.0, 0.0]])
    brain.live([[0.0, 1.0]], reward=[0.7])
    assert brain.pending_feedback
    restored = cd.Brain.load(brain.save(tmp_path / "brain"))
    assert restored.describe() == brain.describe()
    for observation, reward in (([[1.0, 0.0]], 0.2), ([[0.0, 1.0]], -0.1)):
        np.testing.assert_array_equal(
            brain.live(observation, reward=[reward]),
            restored.live(observation, reward=[reward]),
        )
        assert_learner_equal(brain.learner, restored.learner)
        np.testing.assert_array_equal(
            brain.basal_ganglia.w_critic, restored.basal_ganglia.w_critic
        )
        np.testing.assert_array_equal(
            brain.hippocampus.consolidated, restored.hippocampus.consolidated
        )


@pytest.mark.parametrize("kind", ["learner", "brain"])
@pytest.mark.parametrize(
    "name,value",
    [("centered", None), ("centered", 0), ("centered", 1), ("centered", False),
     ("centered", "false"), ("decay", 0.1), ("decay", None)],
)
def test_checkpoint_refuses_a_removed_teaching_setting_away_from_its_law(
    tmp_path, kind, name, value
):
    """The settings of 0.84.3 that this version removed load at the value whose law is the
    law of this version (centered True, decay 0) and are refused otherwise, naming 0.84.3."""
    brain = cd.Brain.compose(2, 2, modules=(4,), recall=False)
    owner = brain if kind == "brain" else brain.learner
    path = owner.save(tmp_path / kind)
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    metadata = json.loads(str(arrays["meta"]))
    loader = cd.Brain.load if kind == "brain" else cd.Learner.load
    metadata["config"].update(centered=True, decay=0.0)  # a founder saved by 0.84.3 loads
    arrays["meta"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)
    assert loader(path).learner.config if kind == "brain" else loader(path).config
    metadata["config"][name] = value
    arrays["meta"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match=f"{name}=.*0.84.3"):
        loader(path)
