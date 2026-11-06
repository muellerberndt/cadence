"""``Brain.grow``: a trained brain with new parts.

The grown brain is the trained brain's layout composed again with the parts added. It carries
everything the trained brain learned, and every new synapse onto a trained neuron starts at zero,
so it settles and chooses as the trained brain does after ``reset()`` until learning gives the
new synapses weight. A brain composed from its own recorded construction is the brain itself.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence import generic

ACTIONS = 4


def taught(inputs=6, lessons=12, seed=1, **settings):
    """A brain composed with ``settings`` and taught a few moments of random labels."""
    brain = cd.Brain.compose(inputs, ACTIONS, seed=seed, **{"modules": (16,), **settings})
    rng = np.random.default_rng(0)
    slots = brain.learner.slot_sizes
    for _ in range(lessons):
        x = rng.random((3, int(np.prod(inputs))))
        labels = np.stack([rng.integers(0, k, size=3) for k in slots], axis=1)
        brain.step(x, teacher=labels[:, 0] if len(slots) == 1 else labels)
    return brain


def regions(brain):
    return {
        name: list(members)
        for name, members in brain.connectome.populations.items()
        if "/" not in name
    }


def check_identical(trained, grown, inputs=6, moments=8, atol=1e-12):
    """The grown brain acts as the trained brain after reset: the same choices, and every
    trained neuron settles to the same activity, within ``atol``: bit for bit where the larger
    graph sums in the same order, as on the machines this was written on, and within rounding
    where a kernel sums it in another order (a Linux CI runner differs in the last bit)."""
    graph = trained.brain
    if graph.backend == "cpu":
        twin = copy.deepcopy(trained)
    else:  # a device brain does not deep-copy: its twin comes through its checkpoint
        with tempfile.TemporaryDirectory() as folder:
            trained.save(Path(folder) / "twin.npz")
            twin = cd.Brain.load(
                Path(folder) / "twin.npz",
                backend=graph.backend,
                device="cpu",
                precision=graph.precision,
            )
    twin.reset()
    rng = np.random.default_rng(5)
    slots = len(trained.learner.slot_sizes)
    old, new = regions(twin), regions(grown)
    for _ in range(moments):
        x = rng.random((3, int(np.prod(inputs))))
        want, got = twin.act(x, greedy=True), grown.act(x, greedy=True)
        if got.ndim == 2 and got.shape[1] > slots:  # the new slots choose as well
            got = got[:, 0] if slots == 1 else got[:, :slots]
        np.testing.assert_array_equal(want, got)
        a = np.asarray(twin.basal_ganglia.state.activation)
        b = np.asarray(grown.basal_ganglia.state.activation)
        for name, members in old.items():
            moved = b[:, new[name][: len(members)]]
            np.testing.assert_allclose(a[:, members], moved, rtol=0, atol=atol, err_msg=name)


LAYOUTS = {
    "founder": dict(),
    "recall heads": dict(recall=True, recall_heads=2, recall_temperature=0.4),
    "recall action": dict(recall=True, recall_action=True, recall_copy=0.5, recall_capacity=16),
    "episodic": dict(memory_choice=True),
    "history": dict(history_decays=(0.5, 0.9), history_amplitude=2.0),
    "efference": dict(efference_amplitude=1.0, efference_decay=0.3),
    "arousal": dict(arousal=True, arousal_heat=1.5),
    "deep": dict(modules=(16, 16), apical=0.5),
    "observers": dict(observers=(5,), lateral=-0.2, slots=(2, 2)),
    "no hippocampus": dict(episodic=False),
    "normalized": dict(teaching="normalized"),
}


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_a_brain_composed_from_its_own_construction_is_the_brain(name):
    brain = cd.Brain.compose(6, ACTIONS, seed=1, **{"modules": (16,), **LAYOUTS[name]})
    inputs, actions, settings = generic._composed_again(brain)
    again = cd.Brain.compose(inputs, actions, **settings)
    assert again.connectome.digest() == brain.connectome.digest()
    np.testing.assert_array_equal(again.brain.efficacy, brain.brain.efficacy)
    assert again.describe() == brain.describe()


@pytest.mark.parametrize(
    "inputs, settings",
    [
        (40, dict(code=64)),
        (40, dict(learned_code=16)),
        (40, dict(learned_code=16, attention=True)),
        ((6, 6), dict(features=4, field=2)),
        ((5, 5, 2), dict(code=40)),
    ],
)
def test_wide_inputs_are_composed_again(inputs, settings):
    brain = cd.Brain.compose(inputs, ACTIONS, seed=2, modules=(16,), **settings)
    inputs, actions, composed = generic._composed_again(brain)
    again = cd.Brain.compose(inputs, actions, **composed)
    assert again.connectome.digest() == brain.connectome.digest()
    assert again.describe() == brain.describe()


GROWTHS = {
    "recall": ({}, dict(recall=True)),
    "recall with settings": (
        {},
        dict(
            recall=True, recall_heads=2, recall_capacity=16, recall_temperature=0.3, recall_window=2
        ),
    ),
    "a second head": (dict(recall=True), dict(recall_heads=2)),
    "a third head": (dict(recall=True, recall_heads=2), dict(recall_heads=3)),
    "a history": ({}, dict(history_decays=(0.5,), history_amplitude=2.0)),
    "more decays": (dict(history_decays=(0.5,)), dict(history_decays=(0.5, 0.9))),
    "an efference copy": ({}, dict(efference_amplitude=1.0, efference_decay=0.3)),
    "a slot": ({}, dict(slots=(4, 3))),
    "slots beside memories": (dict(recall=True, efference_amplitude=1.0), dict(slots=(4, 2, 2))),
    "everything at once": (
        {},
        dict(recall=True, history_decays=(0.5,), efference_amplitude=1.0, slots=(4, 3)),
    ),
    "recall in a deep brain": (dict(modules=(16, 16), apical=0.5), dict(recall=True)),
    "recall beside episodes": (dict(memory_choice=True), dict(recall=True)),
    "recall in an aroused brain": (dict(arousal=True), dict(recall=True)),
    "recall in a normalized brain": (dict(teaching="normalized"), dict(recall=True)),
}


@pytest.mark.parametrize("name", sorted(GROWTHS))
def test_the_grown_brain_acts_as_the_trained_brain(name):
    base, parts = GROWTHS[name]
    trained = taught(**base)
    grown = trained.grow(**parts)
    check_identical(trained, grown)


def test_new_slots_in_a_deep_brain_act_as_the_trained_brain_within_rounding():
    """The motor region lies inside the layout: a deep brain's larger graph sums in another
    order, so its trained neurons settle as before within rounding."""
    trained = taught(modules=(16, 16), apical=0.5)
    check_identical(trained, trained.grow(slots=(4, 3), recall=True), atol=1e-11)


@pytest.mark.parametrize(
    "inputs, settings, parts",
    [
        ((6, 6), dict(features=4, field=2), dict(recall=True, slots=(4, 2))),
        (40, dict(learned_code=16), dict(recall=True, slots=(4, 2))),
        (40, dict(code=64), dict(history_decays=(0.5,), efference_amplitude=1.0)),
    ],
)
def test_wide_input_brains_grow(inputs, settings, parts):
    trained = taught(inputs=inputs, seed=2, **settings)
    check_identical(trained, trained.grow(**parts), inputs=inputs)


def test_the_trained_brain_is_carried_and_the_new_synapses_start_at_zero():
    trained = taught(recall=True)
    grown = trained.grow(recall_heads=2, slots=(4, 3))
    old, new = trained.connectome, grown.connectome
    index = np.zeros(old.n, dtype=np.int64)
    for name, members in regions(trained).items():
        index[members] = new.populations[name][: len(members)]
    key = new.post * new.n + new.pre
    at = np.searchsorted(key, index[old.post] * new.n + index[old.pre])
    np.testing.assert_array_equal(grown.brain.efficacy[at], trained.brain.efficacy)
    np.testing.assert_array_equal(grown.brain.bias[index], trained.brain.bias)
    np.testing.assert_array_equal(grown.brain.log_gain[index], trained.brain.log_gain)
    np.testing.assert_array_equal(grown.learner.velocity[at], trained.learner.velocity)
    assert grown.learner.updates == trained.learner.updates
    added = np.ones(new.synapses, dtype=bool)
    added[at] = False
    known = np.zeros(new.n, dtype=bool)
    known[index] = True
    onto_trained = added & (known[new.post] | known[new.pre])
    assert onto_trained.any()
    assert not grown.brain.efficacy[onto_trained].any()
    # the stores carry their records; the new slot's actions have none
    consolidated = grown.hippocampus.consolidated
    np.testing.assert_array_equal(consolidated[:, :ACTIONS], trained.hippocampus.consolidated)
    assert not consolidated[:, ACTIONS:].any()
    assert grown.recall.heads == 2 and grown.learner.slot_sizes.tolist() == [4, 3]
    assert grown.describe()["layout"]["composition"]["slots"] == [4, 3]


def test_retuned_settings_carry_into_the_grown_brain():
    trained = taught(recall=True, efference_amplitude=1.0)
    trained.retune(
        recall_temperature=0.4,
        working_memory_amplitude=2.0,
        efference_amplitude=0.0,
        temperature=0.3,
        actor_eta=0.02,
    )
    grown = trained.grow(recall_heads=2, slots=(4, 2))
    before, after = trained.describe()["genes"], grown.describe()["genes"]
    for name in (
        "recall_temperature",
        "working_memory_amplitude",
        "efference_amplitude",
        "temperature",
        "actor_eta",
    ):
        assert after[name] == before[name], name
    check_identical(trained, grown)


def test_growing_leaves_the_trained_brain_as_it_was(tmp_path):
    trained = taught()
    assert trained.pending_feedback
    twin = copy.deepcopy(trained)
    trained.grow(recall=True, slots=(4, 2))
    assert trained.pending_feedback and trained.recall is None
    x = np.random.default_rng(3).random((3, 6))
    trained.step(x, reward=np.ones(3))
    twin.step(x, reward=np.ones(3))
    np.testing.assert_array_equal(trained.brain.efficacy, twin.brain.efficacy)
    np.testing.assert_array_equal(trained.act(x, greedy=True), twin.act(x, greedy=True))


def test_a_grown_brain_saves_and_continues(tmp_path):
    grown = taught().grow(recall=True, slots=(4, 2))
    rng = np.random.default_rng(4)
    for _ in range(3):
        grown.step(rng.random((3, 6)), teacher=rng.integers(0, 2, size=(3, 2)))
    grown.save(tmp_path / "grown.npz")
    loaded = cd.Brain.load(tmp_path / "grown.npz")
    x = rng.random((3, 6))
    labels = rng.integers(0, 2, size=(3, 2))
    np.testing.assert_array_equal(grown.step(x, teacher=labels), loaded.step(x, teacher=labels))
    np.testing.assert_array_equal(grown.brain.efficacy, loaded.brain.efficacy)


def test_a_saved_brain_grows(tmp_path):
    trained = taught(history_decays=(0.5,))
    trained.save(tmp_path / "trained.npz")
    loaded = cd.Brain.load(tmp_path / "trained.npz")
    check_identical(loaded, loaded.grow(recall=True, efference_amplitude=1.0))


def test_the_grown_brain_keeps_the_backend_and_precision(tmp_path):
    pytest.importorskip("torch")
    taught().save(tmp_path / "trained.npz")
    trained = cd.Brain.load(
        tmp_path / "trained.npz", backend="torch", device="cpu", precision="float32"
    )
    grown = trained.grow(recall=True, slots=(4, 2))
    assert (grown.brain.backend, grown.brain.precision) == ("torch", "float32")
    check_identical(trained, grown, atol=1e-5)


def twice_shown(rng, episodes, streams=16, tokens=8, length=5):
    return [
        np.stack([rng.permutation(tokens)[:length] for _ in range(streams)])
        for _ in range(episodes)
    ]


def live_through(brain, episodes, tokens=8, length=5, teach=True):
    """Each episode a fresh sequence shown twice; the target is the next token. The second
    showing is answered from what followed each token the first time."""
    eye = np.eye(tokens)
    hits = 0
    for sequence in episodes:
        brain.reset()
        for t in range(2 * length):
            x, target = eye[sequence[:, t % length]], sequence[:, (t + 1) % length]
            if teach:
                brain.step(x, teacher=target)
                continue
            choice = brain.act(x, greedy=True)  # the first showing is written as well
            if t >= length:
                hits += int(np.sum(choice == target))
    return hits / (len(episodes) * length * len(episodes[0]))


def test_a_grown_recall_memory_learns_what_the_trained_brain_could_not():
    rng = np.random.default_rng(1)
    first, then, test = twice_shown(rng, 4), twice_shown(rng, 8), twice_shown(rng, 6)
    plain = cd.Brain.compose(8, 8, seed=3, modules=(48,))
    live_through(plain, first)
    grown = plain.grow(recall=True)
    live_through(plain, then)
    live_through(grown, then)
    ports = np.isin(
        grown.connectome.pre,
        grown.connectome.populations["recall"] + grown.connectome.populations["next"],
    )
    assert np.abs(grown.brain.efficacy[ports]).mean() > 0.01
    without = live_through(copy.deepcopy(plain), test, teach=False)
    with_memory = live_through(copy.deepcopy(grown), test, teach=False)
    assert with_memory > without + 0.15


def test_a_new_slot_learns_beside_the_trained_one():
    rng = np.random.default_rng(2)
    prototypes = rng.normal(size=(12, 8))

    def views(count):
        kind = rng.integers(0, 12, size=count)
        return prototypes[kind] + 0.4 * rng.normal(size=(count, 8)), kind % 4, kind // 4

    trained = cd.Brain.compose(8, 4, seed=3, modules=(32,))
    for _ in range(150):
        x, first, _ = views(16)
        trained.lesson(x, first)
    grown = trained.grow(slots=(4, 3))
    x, first, second = views(400)
    before = trained.predict(x)
    np.testing.assert_array_equal(grown.predict(x)[:, 0], before)
    for _ in range(300):
        lesson, a, b = views(16)
        grown.lesson(lesson, np.stack([a, b], axis=1))
    after = grown.predict(x)
    assert np.mean(after[:, 1] == second) > 0.6  # chance is a third
    assert np.mean(after[:, 0] == first) >= np.mean(before == first) - 0.05


@pytest.mark.parametrize(
    "base, parts, message",
    [
        ({}, {}, "grow needs a part to add"),
        ({}, dict(recall=False), "recall=True"),
        (dict(recall=True), dict(recall=True), "adds read heads"),
        ({}, dict(recall_heads=2), "needs the recall memory"),
        (dict(recall=True, recall_heads=2), dict(recall_heads=2), "must exceed"),
        (dict(recall=True), dict(recall_heads=True), "integer"),
        (dict(recall=True), dict(recall_temperature=0.3), "keeps its construction"),
        (dict(recall=True), dict(recall_heads=2, recall_capacity=8), "keeps its construction"),
        (dict(history_decays=(0.5,)), dict(history_decays=(0.9,)), "must extend"),
        (dict(history_decays=(0.5,)), dict(history_decays=(0.5,)), "must extend"),
        (dict(history_decays=(0.5,)), dict(history_amplitude=2.0), "retune"),
        ({}, dict(history_amplitude=2.0), "grow it with history_decays"),
        ({}, dict(efference_decay=0.3), "grow it with efference_amplitude"),
        (dict(efference_amplitude=1.0), dict(efference_amplitude=2.0), "retune"),
        ({}, dict(slots=(4,)), "followed by the new ones"),
        ({}, dict(slots=(2, 2, 3)), "followed by the new ones"),
        ({}, dict(slots=2), "followed by the new ones"),
        ({}, dict(slots=(4, True)), "positive integers"),
        (dict(recall=True, recall_action=True), dict(slots=(4, 2)), "per action"),
        (dict(memory_choice=True), dict(slots=(4, 2)), "memory_choice"),
        (dict(modules=(16, 16), observers=(4,), apical=0.5), dict(slots=(4, 2)), "apical pool"),
        ({}, dict(modules=(32,)), "own loop"),
        ({}, dict(observers=(4,)), "own loop"),
        ({}, dict(learned_code=8), "own loop"),
        ({}, dict(seed=2), "own loop"),
        ({}, dict(actions=7), "own loop"),
        ({}, dict(learning_eta=0.1), "retune"),
        ({}, dict(temperature=0.5), "retune"),
    ],
)
def test_grow_refuses_what_it_cannot_add_invisibly(base, parts, message):
    trained = taught(**base, lessons=2)
    with pytest.raises(ValueError, match=message):
        trained.grow(**parts)


def test_grow_refuses_a_custom_anatomy():
    genome = cd.Genome.composed(6, ACTIONS, modules=(16,))
    brain = cd.Brain.compose(6, ACTIONS, connectome=cd.develop(genome, seed=0), seed=0)
    with pytest.raises(ValueError, match="records its construction"):
        brain.grow(recall=True)


def test_a_composed_visual_cortex_keeps_its_receptive_fields():
    """The maps project one way into the first module, so no nudge reaches them: a composed
    brain's receptive fields keep their initial draw while the rest of the brain learns (on the
    founder settings, whose phases settle; phases that stop short leave a residue)."""
    brain = cd.Brain.compose((6, 6), ACTIONS, seed=0, modules=(16,))
    connectome = brain.connectome
    fields = np.isin(connectome.pre, connectome.populations["visual/input"]) & np.isin(
        connectome.post, connectome.populations["visual/output"]
    )
    before = np.array(brain.brain.efficacy)
    rng = np.random.default_rng(0)
    for _ in range(10):
        brain.lesson(rng.random((8, 6, 6)), rng.integers(0, ACTIONS, size=8))
    after = np.asarray(brain.brain.efficacy)
    np.testing.assert_array_equal(after[fields], before[fields])
    assert not np.array_equal(after[~fields], before[~fields])


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_a_brain_saved_by_an_earlier_version_grows():
    """A brain saved by 0.94.0 with arousal, recall, a history, an efference copy and episodic
    records grows heads and decays."""
    trained = cd.Brain.load(FIXTURES / "observation_code" / "recalled.npz")
    grown = trained.grow(recall_heads=2, history_decays=(0.2, 0.9, 0.99))
    assert grown.recall.heads == 2 and tuple(grown.history.decays) == (0.2, 0.9, 0.99)
    check_identical(trained, grown, inputs=4, moments=12)


def test_a_grown_brain_grows_again():
    trained = taught()
    once = trained.grow(recall=True)
    twice = once.grow(slots=(4, 2), recall_heads=2)
    check_identical(trained, copy.deepcopy(twice))  # each check acts on the brain it reads
    check_identical(once, twice)


def test_growing_is_deterministic():
    trained = taught(recall=True)
    first, second = trained.grow(recall_heads=2), trained.grow(recall_heads=2)
    assert first.connectome.digest() == second.connectome.digest()
    np.testing.assert_array_equal(first.brain.efficacy, second.brain.efficacy)
    assert first.describe() == second.describe()


def test_the_grown_brain_describes_the_trained_brain_and_its_new_parts():
    trained = taught(recall=True)
    grown = trained.grow(recall_heads=2, history_decays=(0.5,))
    before, after = trained.describe(), grown.describe()
    assert {name: after["genes"][name] for name in before["genes"]} == before["genes"]
    assert set(after["genes"]) - set(before["genes"]) == {
        "recall_heads",
        "recall_compete",
        "history_decays",
        "history_amplitude",
    }
    assert after["memories"]["working_memory"] == before["memories"]["working_memory"]
    assert after["memories"]["hippocampus"] == before["memories"]["hippocampus"]


def test_identity_holds_while_the_recall_memory_fills_and_evicts():
    trained = taught(recall=True, recall_capacity=6)
    check_identical(trained, trained.grow(recall_heads=2, slots=(4, 3)), moments=24)


def test_a_living_brain_grows_and_lives_on():
    rng = np.random.default_rng(6)
    trained = cd.Brain.compose(6, ACTIONS, seed=1, modules=(16,), arousal=True)
    reward = None
    for _ in range(40):
        action = trained.live(rng.random((1, 6)), reward=reward)
        reward = np.array([float(action[0] == 0)])
    grown = trained.grow(recall=True)
    assert grown.arousal.to_dict() == trained.arousal.to_dict()
    ports = np.isin(grown.connectome.pre, grown.connectome.populations["recall"])
    assert not grown.brain.efficacy[ports].any()
    reward = None
    for _ in range(60):
        action = grown.live(rng.random((1, 6)), reward=reward)
        reward = np.array([float(action[0] == 0)])
    assert grown.brain.efficacy[ports].any()  # the aroused moments taught the new ports


def test_two_brains_combine_by_teaching():
    """One brain learns another's skill in a new slot: the other brain's answers are its
    teacher, its own answers at growth hold its own slot."""
    rng = np.random.default_rng(7)
    prototypes = rng.normal(size=(12, 8))

    def views(count):
        kind = rng.integers(0, 12, size=count)
        return prototypes[kind] + 0.4 * rng.normal(size=(count, 8)), kind

    first = cd.Brain.compose(8, 4, seed=3, modules=(32,))
    second = cd.Brain.compose(8, 3, seed=4, modules=(32,))
    for _ in range(200):
        x, kind = views(16)
        first.lesson(x, kind % 4)
        second.lesson(x, kind // 4)
    combined = first.grow(slots=(4, 3))
    for _ in range(300):
        x, _ = views(16)
        combined.lesson(x, np.stack([first.predict(x), second.predict(x)], axis=1))
    test, _ = views(400)
    answers = combined.predict(test)
    assert np.mean(answers[:, 1] == second.predict(test)) > 0.75
    assert np.mean(answers[:, 0] == first.predict(test)) > 0.85


def test_an_episodic_brain_on_a_fixed_code_grows():
    trained = taught(inputs=40, seed=2, code=64, memory_choice=True)
    check_identical(trained, trained.grow(recall=True), inputs=40)


def test_a_saved_separation_of_another_size_is_not_grown():
    # a checkpoint of an earlier release may hold one; the brain itself loads and lives on
    trained = taught(memory_choice=True)
    trained.hippocampus.separator = None
    with pytest.raises(ValueError, match="cadence 0.94.0 grows it"):
        trained.grow(recall=True)


def test_sampled_choices_follow_the_trained_brain_without_new_slots():
    """Without new slots the grown brain draws the trained brain's random numbers."""
    trained = taught(recall=True)
    twin = copy.deepcopy(trained)
    twin.reset()
    grown = trained.grow(recall_heads=2, history_decays=(0.5,))
    rng = np.random.default_rng(8)
    for _ in range(8):
        x = rng.random((3, 6))
        np.testing.assert_array_equal(twin.act(x), grown.act(x))


@pytest.mark.parametrize(
    "base, parts, message",
    [
        ({}, dict(efference_amplitude=0.0), "above zero"),
        ({}, dict(recall_temperature=0.3), "grow it with recall=True"),
        ({}, dict(recall=True, recall_copy=1.0, recall_tokens=[0, 1, 2, 3]), "tokens"),
        (
            dict(recall=True, recall_copy=1.0, recall_tokens=[0, 1, 2, 3]),
            dict(recall_heads=2),
            "tokens",
        ),
        (
            dict(recall=True, recall_copy=1.0, recall_tokens=[0, 1, 2, 3]),
            dict(slots=(4, 2)),
            "per action",
        ),
        ({}, dict(history_decays=0.5), "sequence"),
        ({}, dict(history_decays=np.array(0.5)), "sequence"),
        ({}, dict(arousal=True), "construction choice"),
        ({}, dict(maturity=True), "maturity is a construction choice"),
        ({}, dict(maturity_scale=5.0), "maturity is a construction choice"),
        ({}, dict(memory_choice=True), "construction choice"),
        ({}, dict(episodic=False), "construction choice"),
    ],
)
def test_grow_refuses_with_the_reason(base, parts, message):
    trained = taught(**base, lessons=2)
    with pytest.raises(ValueError, match=message):
        trained.grow(**parts)


def test_a_plain_archive_saves_and_loads_the_same_brain(tmp_path):
    brain = taught(recall=True)
    brain.save(tmp_path / "plain.npz", compressed=False)
    brain.save(tmp_path / "packed.npz")
    plain, packed = cd.Brain.load(tmp_path / "plain.npz"), cd.Brain.load(tmp_path / "packed.npz")
    assert (tmp_path / "plain.npz").stat().st_size > (tmp_path / "packed.npz").stat().st_size
    x = np.random.default_rng(9).random((3, 6))
    np.testing.assert_array_equal(plain.act(x, greedy=True), packed.act(x, greedy=True))
    np.testing.assert_array_equal(plain.brain.efficacy, packed.brain.efficacy)
