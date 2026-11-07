"""A moment on a torch device crosses less (cadence#274): its drive is uploaded once for every
settle and check, a settle from rest uploads no zeros, the motor and critic reads bring their columns
and not the state, and a copy of a brain has a kernel of its own."""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

import cadence as cd

torch = pytest.importorskip("torch")


def brain_on_torch(seed: int = 3) -> cd.Brain:
    return cd.Brain.compose(12, 5, modules=(16,), seed=seed, backend="torch", device="cpu")


def test_a_moments_drive_reaches_the_device_once(monkeypatch):
    brain = brain_on_torch()
    kernel = brain.brain._torch
    wide = []
    upload = kernel._staging.upload

    def counted(array, dtype):
        if np.ndim(array) == 2 and np.shape(array)[1] == brain.connectome.n:
            wide.append(array)
        return upload(array, dtype)

    monkeypatch.setattr(kernel._staging, "upload", counted)
    x = np.eye(12)[[1, 7]]
    brain.act(x, greedy=True)
    assert len(wide) == 1
    wide.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain.lesson(x, np.array([0, 1]))
    assert len(wide) == 1  # the free phase starts from rest: no upload of its zeros either


def test_the_public_drive_is_a_host_array():
    brain = brain_on_torch()
    assert isinstance(brain.stimulus(np.eye(12)[[1, 7]]), np.ndarray)


def test_the_choice_and_the_critic_read_their_columns_of_a_resting_state():
    brain = brain_on_torch()
    x = np.eye(12)[[1, 7]]
    action = brain.act(x)
    state = brain.basal_ganglia.state
    assert state.__dict__.get("activation") is None and state.__dict__.get("v") is None
    motor = state.part(brain.motor_index)
    assert motor.shape == (2, 5)
    assert state.__dict__.get("activation") is None  # still no whole state on the host
    assert state.part(brain.motor_index) is motor  # read once
    np.testing.assert_array_equal(motor, state.activation[:, brain.motor_index])
    np.testing.assert_array_equal(action, np.asarray(action))


def test_a_copy_of_a_brain_on_torch_continues_alike_with_a_kernel_of_its_own():
    brain = brain_on_torch()
    x = np.eye(12)[[1, 7]]
    brain.step(x)
    twin = copy.deepcopy(brain)
    assert twin.brain._torch is not brain.brain._torch
    reward = np.array([1.0, 0.0])
    np.testing.assert_array_equal(brain.step(x, reward=reward), twin.step(x, reward=reward))
    np.testing.assert_allclose(twin.brain.efficacy, brain.brain.efficacy, rtol=0, atol=1e-12)
    before = twin.brain.efficacy.copy()
    brain.step(x, reward=reward)
    np.testing.assert_array_equal(twin.brain.efficacy, before)  # the original moved alone


@pytest.mark.parametrize("seed", range(3))
def test_a_life_on_torch_reads_as_on_the_host(seed):
    host = cd.Brain.compose(12, 5, modules=(16,), seed=seed, backend="cpu")
    device = brain_on_torch(seed)
    rng = np.random.default_rng(seed)
    answer = None
    for t in range(24):
        x = np.eye(12)[rng.integers(12, size=3)]
        if t % 3 == 2 and answer is not None:
            labels = (np.asarray(answer).reshape(-1) + 1 + t) % 5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                host.lesson(x, labels)
                device.lesson(x, labels)
        answer = host.act(x, greedy=True)
        np.testing.assert_array_equal(device.act(x, greedy=True), answer)
    np.testing.assert_allclose(device.brain.efficacy, host.brain.efficacy, rtol=0, atol=1e-9)
