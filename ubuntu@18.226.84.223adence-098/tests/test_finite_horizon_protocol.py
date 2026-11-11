"""Pure recall protocol checks: these tests construct no brain or scientific solves."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "benchmarks/recall"))
    return import_module("finite_horizon_inputs")


def test_all_conditions_have_balanced_pairs_and_no_current_answer_path(protocol):
    for condition in protocol.TEST_CONDITIONS:
        episode = protocol.make_episode(np.random.default_rng(41), condition)
        assert len(episode.observations) >= 3
        assert not episode.observations[:, :, protocol.HISTORY].any()
        assert not episode.observations[-1, :, protocol.PAYLOAD].any()
        np.testing.assert_array_equal(episode.labels[episode.permutation], 1 - episode.labels)
        for start in range(0, protocol.STREAMS, 2):
            assert set(episode.labels[start:start + 2]) == {0, 1}
            np.testing.assert_array_equal(
                episode.observations[-1, start], episode.observations[-1, start + 1],
            )
            if not condition.opposite:
                np.testing.assert_array_equal(
                    episode.token_values[:, start][np.arange(condition.load) != condition.query],
                    episode.token_values[:, start + 1][np.arange(condition.load) != condition.query],
                )
        assert not episode.observations.flags.writeable


def test_partial_and_noisy_tokens_keep_independent_recoverable_evidence(protocol):
    for condition in protocol.TRAIN_CONDITIONS:
        episode = protocol.make_episode(np.random.default_rng(41), condition)
        writes = episode.observations[episode.observations[:, 0, protocol.WRITE] == 1]
        if condition.partial:
            assert np.all((writes[:, :, protocol.PAYLOAD] != 0).sum(axis=2) == 1)
        if condition.partial or condition.noise:
            for values, obs in zip(episode.token_values, writes, strict=True):
                signal = obs[:, protocol.PAYLOAD]
                for row, value in enumerate(values):
                    own = signal[row, 2 * value:2 * value + 2]
                    other = signal[row, 2 * (1 - value):2 * (1 - value) + 2]
                    assert own.max() - other.max() >= 0.9 - 1e-15


def test_order_and_replacement_have_identical_bags_but_opposite_latest_values(protocol):
    for condition in protocol.TRAIN_CONDITIONS:
        if not condition.opposite:
            continue
        episode = protocol.make_episode(np.random.default_rng(41), condition)
        np.testing.assert_array_equal(episode.token_values[1], episode.labels)
        np.testing.assert_array_equal(episode.token_values[0], 1 - episode.labels)
        for start in range(0, protocol.STREAMS, 2):
            assert sorted(episode.token_values[:, start]) == sorted(
                episode.token_values[:, start + 1],
            )


def test_history_control_reads_the_actual_raw_payload_and_never_a_label(protocol):
    for condition in protocol.TEST_CONDITIONS:
        episode = protocol.make_episode(np.random.default_rng(41), condition)
        raw = episode.observations.copy()
        appended = protocol.append_observed_history(raw)
        writes = raw[raw[:, 0, protocol.WRITE] == 1, :, protocol.PAYLOAD]
        expected = np.zeros((protocol.STREAMS, 4, 4))
        expected[:, :condition.load] = writes.transpose((1, 0, 2))
        np.testing.assert_array_equal(appended[-1, :, protocol.HISTORY], expected.reshape((8, 16)))
        np.testing.assert_array_equal(appended[:-1], raw[:-1])
        np.testing.assert_array_equal(raw, episode.observations)
        # Changing witnessed evidence changes the appended control without consulting labels.
        write_at = np.flatnonzero(raw[:, 0, protocol.WRITE] == 1)[0]
        raw[write_at, :, protocol.PAYLOAD] *= 0.7
        assert not np.array_equal(protocol.append_observed_history(raw), appended)


def test_freeze_is_reproducible_and_preserves_all_planned_denominators(protocol, tmp_path):
    first = protocol.freeze_episodes(tmp_path / "first.npz", seed=301)
    np.random.default_rng(1).normal(size=1000)
    second = protocol.freeze_episodes(tmp_path / "second.npz", seed=301)
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert not first[name].flags.writeable
    assert np.array_equal(np.bincount(first["train/condition"]), np.full(12, 16))
    assert np.array_equal(np.bincount(first["test/condition"]), np.full(13, 24))
    assert len(first["train/condition"]) == 192
    assert all(first[f"test/{index:04d}/labels"].shape == (8,) for index in range(312))


def test_irregular_time_changes_no_stimulus_or_expected_event_count(protocol):
    episode = protocol.make_episode(np.random.default_rng(41), protocol.TRAIN_CONDITIONS[2])
    assert len(episode.timestamps) == len(episode.irregular_timestamps)
    assert np.all(np.diff(episode.irregular_timestamps) > 0)
    assert episode.timestamps[0] == episode.irregular_timestamps[0] == 0
    assert episode.timestamps[-1] == episode.irregular_timestamps[-1]
    assert not np.array_equal(episode.timestamps, episode.irregular_timestamps)


def test_independent_trace_audit_rejects_skipped_or_extra_updates(protocol):
    prior = {"trace": np.array([[0.4, 0.1]]), "last": np.zeros((1, 2)),
             "cold": np.array([True])}
    observed = np.array([[0.7, 0.5]])
    correct = {"trace": np.array([[0.46, 0.18]]), "last": observed.copy(),
               "cold": np.array([False])}
    assert protocol.check_trace_transition(prior, correct, observed) < 1e-15
    for wrong in (prior["trace"], np.array([[0.508, 0.244]])):
        with pytest.raises(ValueError, match="declared accepted-event"):
            protocol.check_trace_transition(prior, {**correct, "trace": wrong}, observed)
    with pytest.raises(ValueError, match="last/cold"):
        protocol.check_trace_transition(prior, {**correct, "cold": np.array([True])}, observed)
