"""A code phase is provisional until the action or outcome using it is accepted."""

from copy import deepcopy

import numpy as np
import pytest

import cadence as cd


def test_refused_cortex_restores_code_policy_and_checkpoint(tmp_path):
    # the code region settles; the cortex refuses after its budget and the rescue
    brain = cd.Brain.compose(1, 12, modules=(8,), learned_code=16, seed=1,
                             episodic=False, learning_free_steps=16, lateral=-1.0)
    region = brain.learned_code
    policy = region.halvings, region.graph, region.last
    with pytest.raises(RuntimeError, match="brain did not settle"):
        brain.act([[1.0]], greedy=True)
    assert (region.halvings, region.graph, region.last) == policy
    assert brain.basal_ganglia.state is None
    assert brain.last_settlement["qualified"] is False
    loaded = cd.Brain.load(brain.save(tmp_path / "refused.npz"))
    assert loaded.learned_code.halvings == policy[0]
    assert loaded.learned_code.graph.neuron_model.dt == region.graph.neuron_model.dt


def test_refused_prediction_restores_its_provisional_code_phase(monkeypatch):
    brain = cd.Brain.compose(1, 2, modules=(4,), learned_code=16, seed=2, episodic=False)
    region = brain.learned_code
    policy = region.halvings, region.graph, region.last

    def refused(*args, **kwargs):
        assert region.last is not policy[2]
        raise RuntimeError("cortex refused after code preparation")

    monkeypatch.setattr(brain, "_qualified", refused)
    with pytest.raises(RuntimeError, match="cortex refused"):
        brain.predict([[1.0]])
    assert (region.halvings, region.graph, region.last) == policy


@pytest.mark.parametrize("field", ["reward", "done", "bootstrap", "salience"])
def test_invalid_feedback_never_prepares_the_next_code(field):
    brain = cd.Brain.compose(1, 2, modules=(4,), learned_code=16, seed=2, episodic=False)
    brain.act([[0.0]])
    control = deepcopy(brain)
    region = brain.learned_code
    before = region.halvings, region.graph, region.last, dict(region.work)
    values = dict(reward=[0.5], done=[False], next_observations=[[1.0]])
    values[field] = [0.0, 0.0]
    with pytest.raises(ValueError):
        brain.learn(**values)
    assert region.halvings == before[0] and region.graph is before[1]
    assert region.last is before[2] and region.work == before[3]
    brain.learn([0.5], [False], [[1.0]])
    control.learn([0.5], [False], [[1.0]])
    np.testing.assert_array_equal(brain.brain.weights, control.brain.weights)
    np.testing.assert_array_equal(brain.basal_ganglia.state.v, control.basal_ganglia.state.v)
    assert brain._code_settlement == control._code_settlement
