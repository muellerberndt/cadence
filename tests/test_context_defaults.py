"""The composed brain's context defaults (0.82.0) and the removed readout homeostasis gene."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd


def test_the_composed_brain_holds_its_last_interpretation_clearly_and_reads_it_quietly():
    brain = cd.Brain.compose(inputs=3, actions=2, seed=1, arousal=True)
    memory = brain.describe()["working_memory"]
    assert memory["amplitude"] == 3.0 and memory["decay"] == 0.0
    assert brain.describe()["initialization"]["composition"]["working_memory_scale"] == 1.2


def test_the_default_trace_is_exactly_the_last_interpretation():
    brain = cd.Brain.compose(inputs=3, actions=2, seed=6)
    for x in ([[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]):
        brain.act(np.array(x))
        assert np.array_equal(brain.working_memory.trace, brain.working_memory.last)
    averaged = cd.Brain.compose(inputs=3, actions=2, seed=6, working_memory_decay=0.2)
    averaged.act(np.array([[1.0, 0.0, 0.0]]))
    averaged.act(np.array([[0.0, 1.0, 0.0]]))
    assert not np.allclose(averaged.working_memory.trace, averaged.working_memory.last)


def test_the_scale_changes_only_the_projection_that_reads_the_carried_interpretation():
    control = cd.Brain.compose(3, 2, modules=(9, 6), seed=11, working_memory_scale=12.0)
    quiet = cd.Brain.compose(3, 2, modules=(9, 6), seed=11)
    first, second = control.connectome, quiet.connectome
    for field in ("pre", "post", "count"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    prefrontal = np.asarray(first.populations["prefrontal"])
    reading = np.isin(first.pre, prefrontal)
    assert reading.any()
    np.testing.assert_allclose(second.sign[reading], 0.1 * first.sign[reading])
    np.testing.assert_array_equal(second.sign[~reading], first.sign[~reading])
    np.testing.assert_array_equal(control.brain.bias, quiet.brain.bias)
    with pytest.raises(ValueError, match="construction setting"):
        quiet.retune(working_memory_scale=12.0)
    with pytest.raises(ValueError):
        cd.Brain.compose(3, 2, working_memory_scale=-1.0)


def test_a_newborn_reads_its_context_as_the_earlier_quiet_operating_point_did():
    # Amplitude 3.0 into a projection of scale 1.2 starts the carried interpretation's
    # influence where amplitude 0.3 into the earlier fixed scale 12 did; the prefrontal copy
    # is ten times clearer, so learning can strengthen the influence where it pays.
    cue, go = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), np.array([[0.0, 0.0, 1.0]] * 2)
    states = []
    for options in ({}, {"working_memory_amplitude": 0.3, "working_memory_scale": 12.0}):
        brain = cd.Brain.compose(3, 2, seed=8, **options)
        brain.act(cue, greedy=True)
        _, free = brain._settled(go)
        association = np.asarray(brain.connectome.populations["association"])
        prefrontal = np.asarray(brain.connectome.populations["prefrontal"])
        states.append((free.activation[:, association], free.activation[:, prefrontal]))
    (held, held_copy), (earlier, earlier_copy) = states
    np.testing.assert_allclose(held, earlier, rtol=0.05, atol=1e-4)
    assert np.abs(held_copy).mean() > 5 * np.abs(earlier_copy).mean()


def delayed_response(brain: cd.Brain, episodes: int, seed: int) -> float:
    """tests/test_generic.py's task: a cue, then a go signal with the cue gone; reward only."""
    rng = np.random.default_rng(seed)
    batch = 32

    def episode() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cue = rng.integers(0, 2, size=batch)
        shown, go = np.zeros((batch, 3)), np.zeros((batch, 3))
        shown[np.arange(batch), cue] = 1.0
        go[:, 2] = 1.0
        return cue, shown, go

    for _ in range(episodes):
        cue, shown, go = episode()
        brain.reset()
        brain.act(shown)
        brain.learn(np.zeros(batch), np.zeros(batch, dtype=bool), go)
        action = brain.act(go)
        brain.learn((action == cue).astype(float), np.ones(batch, dtype=bool), go)
    hits = []
    for _ in range(5):
        cue, shown, go = episode()
        brain.reset()
        brain.act(shown, greedy=True)
        hits.append(float((brain.act(go, greedy=True) == cue).mean()))
    return float(np.mean(hits))


def test_the_composed_brain_learns_from_reward_to_carry_a_cue_across_a_delay():
    # Seeds 8 to 13 all solve it at the 0.82 defaults (0.96 to 1.00), as with the 0.81
    # composition; amplitude 0.3 at the earlier scale 12 solved 3 of the 6, seed 8 not.
    assert delayed_response(cd.Brain.compose(3, 2, seed=8), 1500, 8) >= 0.9


def contextual_bandit(brain: cd.Brain, seed: int, decisions: int = 600) -> float:
    """One continuing stream, history irrelevant: four one-hot contexts, the matching action
    pays one; the greedy hit rate of a frozen copy over 200 more contexts (chance 0.25)."""
    rng = np.random.default_rng(seed)
    context = int(rng.integers(0, 4))
    action = brain.step(np.eye(4)[[context]])
    for _ in range(decisions - 1):
        reward = float(action[0] == context)
        context = int(rng.integers(0, 4))
        action = brain.step(np.eye(4)[[context]], reward=[reward])
    hits = 0
    for _ in range(200):
        context = int(rng.integers(0, 4))
        hits += int(brain.act(np.eye(4)[[context]], greedy=True)[0] == context)
    return hits / 200


def test_a_continuing_brain_can_learn_to_ignore_a_history_that_does_not_matter():
    # Seeds 0 to 5 at the 0.82 defaults: 0.72, 1.00, 0.26, 0.92, 0.46, 0.23 (mean 0.60);
    # the 0.81 composition held every seed near chance (0.22 to 0.30, here 0.22).
    actor = {"actor_gamma": 0.0, "actor_lam": 0.0}
    assert contextual_bandit(cd.Brain.compose(4, 4, seed=1, **actor), 1) >= 0.9
    earlier = cd.Brain.compose(
        4, 4, seed=1, working_memory_decay=0.2, working_memory_scale=12.0, **actor
    )
    assert contextual_bandit(earlier, 1) < 0.5


def test_a_saved_brain_keeps_the_trace_it_was_built_with(tmp_path):
    earlier = cd.Brain.compose(
        inputs=3, actions=2, seed=2, working_memory_amplitude=0.3, working_memory_decay=0.2,
        working_memory_scale=12.0,
    )
    x = np.array([[1.0, 0.0, 0.5]])
    earlier.act(x)
    path = earlier.save(tmp_path / "earlier.npz")
    again = cd.Brain.load(path)
    assert again.working_memory.amplitude == 0.3 and again.working_memory.decay == 0.2
    assert again.describe()["initialization"]["composition"]["working_memory_scale"] == 12.0
    assert np.array_equal(again.working_memory.trace, earlier.working_memory.trace)


def test_provenance_saved_before_0_82_reads_as_the_fixed_scale_12(tmp_path):
    brain = cd.Brain.compose(inputs=3, actions=2, seed=7, working_memory_scale=12.0)
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(str(arrays["generic"]))
    del meta["composition"]["working_memory_scale"]
    arrays["generic"] = np.array(json.dumps(meta))
    older = tmp_path / "older.npz"
    np.savez(older, **arrays)
    loaded = cd.Brain.load(older)
    assert loaded.describe()["initialization"]["composition"]["working_memory_scale"] == 12.0
    meta["composition"]["working_memory_scale"] = -1.0
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(older, **arrays)
    with pytest.raises(ValueError):
        cd.Brain.load(older)


def test_the_carried_interpretation_reaches_the_next_settle():
    brain = cd.Brain.compose(inputs=3, actions=2, seed=3)
    x = np.array([[1.0, 0.0, 0.0]])
    brain.act(x)
    carried = brain.working_memory.trace.copy()
    drive = brain.stimulus(np.zeros((1, 3)))
    glow = brain.working_memory.glow
    assert np.allclose(drive[0, glow], 3.0 * carried[0])
    brain.working_memory.reset(1)
    assert not brain.stimulus(np.zeros((1, 3)))[0, glow].any()


def test_the_readout_homeostasis_gene_is_gone():
    with pytest.raises(TypeError):
        cd.LearnerConfig(homeostasis_rate=0.1)
    with pytest.raises((TypeError, ValueError)):
        cd.Brain.compose(inputs=3, actions=2, learning_homeostasis_rate=0.1)
    report = cd.Brain.compose(inputs=3, actions=2, seed=4)
    assert "homeostasis_rate" not in report.describe()["learning"]


def test_a_checkpoint_saved_at_the_genes_founder_still_loads(tmp_path):
    # 0.81.0 wrote the two fields into every learner configuration it saved
    brain = cd.Brain.compose(inputs=3, actions=2, seed=5)
    brain.act(np.array([[1.0, 0.0, 0.0]]))
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    written = 0
    for name in [n for n in arrays if n.endswith("meta") and arrays[n].dtype.kind == "U"]:
        meta = json.loads(str(arrays[name]))
        if meta.get("format") == "cadence-checkpoint/2":
            meta.setdefault("config", {}).update(homeostasis_rate=0.0, homeostasis_target=0.3)
            arrays[name] = np.array(json.dumps(meta))
            written += 1
    assert written
    founder = tmp_path / "founder.npz"
    np.savez(founder, **arrays)
    loaded = cd.Brain.load(founder)
    assert loaded.learner.config == brain.learner.config
    np.testing.assert_array_equal(loaded.working_memory.trace, brain.working_memory.trace)


def test_a_checkpoint_saved_with_the_gene_on_is_refused(tmp_path):
    brain = cd.Brain.compose(inputs=3, actions=2, seed=5)
    path = brain.save(tmp_path / "brain.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta_names = [name for name in arrays if name.endswith("meta") and arrays[name].dtype.kind == "U"]
    assert meta_names, "the checkpoint stores its learner metadata as a JSON string"
    for name in meta_names:
        meta = json.loads(str(arrays[name]))
        if meta.get("format") == "cadence-checkpoint/2":
            meta["format"] = "cadence-checkpoint/3"
            meta.setdefault("config", {})["homeostasis_rate"] = 0.1
            arrays[name] = np.array(json.dumps(meta))
    forged = tmp_path / "gene.npz"
    np.savez(forged, **arrays)
    with pytest.raises(ValueError, match="0.82.0 removed"):
        cd.Brain.load(forged)
