"""Reward credit must differentiate the policy that supplied the actual action."""

from itertools import product

import numpy as np
import pytest

import cadence as cd


def actor():
    graph = cd.layered(2, 5, 5, density=1.0, seed=4)
    model = cd.NeuralGraph(graph, cd.learning_neuron_model(dt=1.0))
    config = cd.LearnerConfig(
        beta=1e-4, temperature=0.2, free_steps=160, nudged_steps=160, tolerance=1e-11
    )
    learner = cd.Learner(model, graph.populations["output"], config, slots=[2, 3])
    ac = cd.ActorCritic(learner, graph.populations["hidden"], seed=12)
    bias = model.bias.copy()
    bias[learner.output_index] = [-0.4, 0.3, 0.2, -0.3, 0.5]
    learner.brain = model.with_parameters(bias=bias)
    drive = np.zeros((1, graph.n))
    drive[:, :2] = [0.5, 2.0]
    return ac, drive


@pytest.mark.parametrize("temperature", [0.6, np.array([0.2, 0.8])])
def test_hot_score_matches_sensory_synapse_and_motor_bias_finite_differences(temperature):
    ac, drive = actor()
    choice = ac.act(drive, temperature=temperature)[0]
    _, plus, minus, _ = ac._pending
    edge, bias = ac.learner.contrast(ac.state, plus, minus)
    original = ac.learner.brain
    # Directed sensory edges need no reciprocal pair's factor of two.
    sensory = np.flatnonzero(original.connectome.pre < 2)[[0, 3, 7]]
    common = min(ac.learner.config.temperature, float(np.min(temperature)))
    for kind, indices, contrast in (
        ("efficacy", sensory, edge), ("bias", ac.learner.output_index, bias)
    ):
        for index in indices:
            values = []
            for sign in (-1, 1):
                parameter = getattr(original, kind).copy()
                parameter[index] += sign * 1e-5
                ac.learner.brain = original.with_parameters(**{kind: parameter})
                free = ac.learner.free(drive)
                p = ac.probabilities(free, temperature)[0]
                values.append(np.log(p[np.arange(2), choice]).sum())
            reference = common * (values[1] - values[0]) / 2e-5
            assert contrast[index] == pytest.approx(reference, abs=2e-7, rel=2e-4)
    ac.learner.brain = original


@pytest.mark.parametrize("temperature", [0.6, np.array([0.2, 0.8])])
def test_action_independent_reward_has_zero_expected_score_even_while_hot(temperature):
    ac, drive = actor()
    free = ac.learner.free(drive)
    probabilities = ac.probabilities(free, temperature)[0]
    corrected = np.zeros(ac.n)
    legacy = np.zeros(ac.n)
    for first, second in product(range(2), range(3)):
        target = ac.learner.targets(np.array([[first, second]]))
        probability = probabilities[0, first] * probabilities[1, second]
        for expected, actual in ((corrected, temperature), (legacy, None)):
            plus = ac._nudged_groups(drive, free, target, ac.learner.config.beta, actual)
            minus = ac._nudged_groups(drive, free, target, -ac.learner.config.beta, actual)
            expected += probability * (plus.activation[0] - minus.activation[0]) / (
                2 * ac.learner.config.beta
            )
    # A critic baseline or constant payoff supplies no evidence favoring an action.
    # The old base-policy nudge introduced a systematic update under hot sampling.
    assert np.linalg.norm(legacy[ac.learner.output_index]) > 0.01
    assert np.linalg.norm(corrected) < 2e-7
