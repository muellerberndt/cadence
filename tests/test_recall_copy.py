"""The copy pathway (``recall_copy``, ``recall_tokens``): plastic synapses from every head's read
port into the motor region, learned by the settle's contrast. With declared tokens it starts as
their identity from the recall ports, at zero from the next ports and as the negative identity
from the present observation (the read's echo). The founder has none; the options are described
and saved where they are on.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd

TOKENS = 6
INPUTS = TOKENS + 3  # the tokens, a HEAR, an ASK and a NOISE flag
HEAR, ASK, NOISE = TOKENS, TOKENS + 1, TOKENS + 2


def composed(**options):
    options.setdefault("modules", (16,))
    return cd.Brain.compose(INPUTS, TOKENS, seed=3, episodic=False, recall=True, **options)


def event(*on):
    x = np.zeros((1, INPUTS))
    x[0, list(on)] = 1.0
    return x


def pathway(brain):
    """The copy synapses as ``(port, value coordinate, motor) -> efficacy`` and their groups."""
    synapses, value, motor = brain.recall.copy_synapses()
    pre = np.asarray(brain.connectome.pre)[synapses]
    ports = {}
    for h, (recall, following) in enumerate(brain.recall._heads):  # noqa: SLF001
        for name, region in (("recall", recall), ("next", following)):
            ports.update({int(i): (name, h) for i in region})
    return [
        (ports[int(p)], int(v), int(m), float(brain.brain.efficacy[s]), s)
        for p, v, m, s in zip(pre, value, motor, synapses, strict=True)
    ]


def test_the_founder_has_no_pathway_and_the_options_are_described_where_on():
    founder = composed()
    assert founder.recall.copy == 0.0 and founder.recall.tokens is None
    assert not len(founder.recall.copy_synapses()[0])
    genes = founder.describe()["genes"]
    assert "recall_copy" not in genes and "recall_tokens" not in genes
    tokened = composed(recall_copy=2.0, recall_tokens=range(TOKENS))
    genes = tokened.describe()["genes"]
    assert genes["recall_copy"] == 2.0 and genes["recall_tokens"] == list(range(TOKENS))


def test_a_random_pathway_is_the_brain_without_it_plus_its_synapses():
    founder, copied = composed(), composed(recall_copy=0.5)
    synapses = copied.recall.copy_synapses()[0]
    assert len(synapses) == 2 * INPUTS * TOKENS  # every recall and next neuron, every action
    keep = np.ones(copied.connectome.synapses, dtype=bool)
    keep[synapses] = False
    a, b = founder.connectome, copied.connectome
    pairs_a = set(zip(a.pre.tolist(), a.post.tolist(), a.sign.tolist(), strict=True))
    pairs_b = set(zip(b.pre[keep].tolist(), b.post[keep].tolist(), b.sign[keep].tolist(),
                      strict=True))
    assert pairs_a == pairs_b  # every other synapse, drawn as before
    magnitude = np.abs(copied.connectome.sign[synapses])
    assert magnitude.max() > 0 and magnitude.max() <= 0.5 * np.sqrt(6.0 / (INPUTS + TOKENS))


def test_declared_tokens_start_the_pathway_as_their_identity_with_echo_suppression():
    brain = composed(recall_copy=2.0, recall_tokens=range(TOKENS))
    for (port, _h), value, motor, weight, _ in pathway(brain):
        expected = 2.0 if port == "recall" and value == motor else 0.0
        assert weight == expected, (port, value, motor)
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    echo = np.isin(pre, brain.sensory_index) & np.isin(post, brain.motor_index)
    sensory = {int(n): i for i, n in enumerate(brain.sensory_index)}
    motor = {int(n): i for i, n in enumerate(brain.motor_index)}
    for s in np.flatnonzero(echo):
        same = sensory[int(pre[s])] == motor[int(post[s])]
        assert brain.brain.efficacy[s] == (-2.0 if same else 0.0)
    assert brain.learner.tie_groups is None  # every synapse of the pathway learns on its own


def test_a_newborn_produces_a_token_it_was_never_taught_from_its_read():
    brain = composed(recall_copy=2.0, recall_tokens=range(TOKENS))
    for token in range(TOKENS):
        brain.reset()
        brain.act(event(HEAR, token), greedy=True)
        brain.act(event(NOISE), greedy=True)  # a moment without the token
        assert brain.act(event(HEAR, ASK), greedy=True)[0] == token


def test_a_lesson_moves_the_synapses_the_read_used_and_leaves_untaught_tokens():
    brain = composed(recall_copy=2.0, recall_tokens=range(TOKENS))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        brain.act(event(HEAR, 2), greedy=True)
        brain.lesson(event(HEAR, ASK), [4])  # the read names 2, the lesson asks for 4
    weights = {(port, value, motor): weight for (port, _h), value, motor, weight, _ in
               pathway(brain)}
    assert weights[("recall", 2, 2)] < 2.0  # the read named the wrong token: it weakened
    assert weights[("recall", 2, 4)] > 0.0  # and the read's token learns the taught action
    assert all(weights[("recall", t, t)] == 2.0 for t in (0, 1, 3, 5))  # untouched, untaught


def test_the_pathway_and_its_ties_survive_a_checkpoint(tmp_path):
    brain = composed(recall_copy=2.0, recall_tokens=range(TOKENS), recall_heads=2)
    brain.act(event(HEAR, 1), greedy=True)
    brain.lesson(event(HEAR, ASK), [1])
    twin = cd.Brain.load(brain.save(tmp_path / "copy.npz"))
    assert twin.recall.copy == 2.0 and twin.recall.tokens == tuple(range(TOKENS))
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    np.testing.assert_array_equal(twin.connectome.sign, brain.connectome.sign)
    for b in (brain, twin):
        b.lesson(event(HEAR, ASK), [3])
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    assert twin.describe() == brain.describe()


def test_several_heads_each_have_their_ports_on_the_pathway():
    brain = composed(recall_copy=1.0, recall_tokens=range(TOKENS), recall_heads=2)
    ports = {port for port, *_ in pathway(brain)}
    assert ports == {("recall", 0), ("next", 0), ("recall", 1), ("next", 1)}
    for (port, _h), value, motor, weight, _ in pathway(brain):
        assert weight == (1.0 if port == "recall" and value == motor else 0.0)


def test_a_joined_learned_code_takes_tokens_after_its_units():
    brain = cd.Brain.compose(
        8, 4, modules=(16,), seed=0, learned_code=16, attention=True,
        recall_copy=1.0, recall_tokens=[0, 1, 2, 3],
    )
    for (port, _h), value, motor, weight, _ in pathway(brain):
        assert weight == (1.0 if port == "recall" and value == 16 + motor else 0.0)
    brain.act(np.eye(8)[[1]], greedy=True)
    with pytest.raises(ValueError, match="observation's own coordinates"):
        cd.Brain.compose(40, 4, modules=(16,), learned_code=16, attention=True,
                         recall_copy=1.0, recall_tokens=[0, 1, 2, 3])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(recall_tokens=range(TOKENS - 1)), "one observation coordinate per"),
        (dict(recall_copy=-1.0), "copy"),
        (dict(recall_copy=1.0, recall_tokens=range(TOKENS - 1)), "one observation coordinate per"),
        (dict(recall_copy=1.0, recall_tokens=[INPUTS] * TOKENS), "below"),
        (dict(recall_copy=1.0, recall_tokens=[0.5] * TOKENS), "integer"),
        (dict(recall_copy=1.0, recall_tokens="abc"), "sequence"),
    ],
)
def test_invalid_pathways_are_refused(options, message):
    with pytest.raises(ValueError, match=message):
        composed(**options)


def test_a_fixed_code_and_a_custom_anatomy_without_the_synapses_are_refused():
    with pytest.raises(ValueError, match="observation's own coordinates"):
        cd.Brain.compose(40, 4, code=64, recall=True, recall_copy=1.0,
                         recall_tokens=[0, 1, 2, 3])
    founder = composed()
    with pytest.raises(ValueError, match="copy needs synapses"):
        cd.Brain(founder.connectome, recall=True, recall_copy=1.0, episodic=False)
    with pytest.raises(ValueError, match="construction setting"):
        composed(recall_copy=1.0).retune(recall_copy=2.0)


def test_copy_next_starts_the_next_ports_identity_and_is_described_where_on():
    plain = composed(recall_copy=2.0, recall_tokens=range(TOKENS))
    zero = composed(recall_copy=2.0, recall_tokens=range(TOKENS), recall_copy_next=0.0)
    assert zero.connectome.digest() == plain.connectome.digest()  # at 0.0 the brain as before
    np.testing.assert_array_equal(zero.brain.efficacy, plain.brain.efficacy)
    assert zero.describe() == plain.describe()
    assert "recall_copy_next" not in plain.describe()["genes"]
    brain = composed(recall_copy=2.0, recall_tokens=range(TOKENS), recall_copy_next=1.5,
                     recall_heads=2)
    for (port, _h), value, motor, weight, _ in pathway(brain):
        expected = (2.0 if port == "recall" else 1.5) if value == motor else 0.0
        assert weight == expected, (port, value, motor)
    assert brain.describe()["genes"]["recall_copy_next"] == 1.5
    assert brain.recall.copy_next == 1.5
    assert brain.parameters() == composed(
        recall_copy=2.0, recall_tokens=range(TOKENS), recall_heads=2
    ).parameters()  # no state of its own: the synapses were there


def test_a_newborn_with_copy_next_answers_a_successor_it_was_never_taught():
    brain = composed(recall_copy=2.0, recall_tokens=range(TOKENS), recall_copy_next=2.0)
    without = composed(recall_copy=2.0, recall_tokens=range(TOKENS))
    order = [3, 0, 5, 1, 4]
    right = {"with": 0, "without": 0}
    for name, b in (("with", brain), ("without", without)):
        for k in range(len(order) - 1):
            b.reset()
            for token in order:
                b.act(event(HEAR, token), greedy=True)
            # asked for the token after ``order[k]``: the recall port holds the cue's own token,
            # which the echo removes, and the next port the successor
            right[name] += int(b.act(event(ASK, order[k]), greedy=True)[0] == order[k + 1])
    assert right["with"] == len(order) - 1
    assert right["without"] < right["with"]


def test_copy_next_survives_a_checkpoint_and_is_a_construction_setting(tmp_path):
    brain = composed(recall_copy=2.0, recall_tokens=range(TOKENS), recall_copy_next=1.0)
    brain.act(event(HEAR, 2), greedy=True)
    brain.lesson(event(ASK, 2), [5])
    twin = cd.Brain.load(brain.save(tmp_path / "next.npz"))
    assert twin.recall.copy_next == 1.0 and twin.describe() == brain.describe()
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    for b in (brain, twin):
        b.lesson(event(ASK, 2), [1])
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(recall_copy_next=2.0)
    with pytest.raises(ValueError, match="needs copy and tokens"):
        composed(recall_copy=2.0, recall_copy_next=1.0)
    with pytest.raises(ValueError, match="needs copy and tokens"):
        composed(recall_tokens=range(TOKENS), recall_copy_next=1.0)
    with pytest.raises(ValueError, match="copy_next"):
        composed(recall_copy=2.0, recall_tokens=range(TOKENS), recall_copy_next=-1.0)


def test_grow_and_graft_keep_copy_next():
    settings = dict(recall=True, recall_copy=2.0, recall_tokens=[0, 1, 2, 3],
                    recall_copy_next=2.0, episodic=False)
    brain = cd.Brain.compose(6, 4, seed=1, modules=(16,), **settings)
    rng = np.random.default_rng(0)
    for _ in range(4):
        brain.step(rng.random((3, 6)), teacher=rng.integers(0, 4, size=3))
    grown = brain.grow(history_decays=(0.5,))
    assert grown.describe()["genes"]["recall_copy_next"] == 2.0
    synapses, value, motor = grown.recall.copy_synapses()
    nexts = np.isin(np.asarray(grown.connectome.pre)[synapses],
                    grown.connectome.populations["next"])
    old = dict(zip(zip(value[nexts].tolist(), motor[nexts].tolist(), strict=True),
                   grown.brain.efficacy[synapses[nexts]].tolist(), strict=True))
    synapses0, value0, motor0 = brain.recall.copy_synapses()
    nexts0 = np.isin(np.asarray(brain.connectome.pre)[synapses0],
                     brain.connectome.populations["next"])
    for v, m, w in zip(value0[nexts0], motor0[nexts0], brain.brain.efficacy[synapses0[nexts0]],
                       strict=True):
        assert old[(int(v), int(m))] == w  # the trained next-port synapses carried
    donor = cd.Brain.compose(6, 4, seed=7, modules=(16,), recall=True, recall_copy=2.0,
                             episodic=False)
    grafted = brain.graft(donor, ["recall", "next", "motor", "association"])
    assert grafted.describe()["genes"]["recall_copy_next"] == 2.0
    np.testing.assert_array_equal(grafted.connectome.sign, brain.connectome.sign)


def test_a_lesson_whose_answer_is_no_token_leaves_the_read_as_it_was():
    # actions 0 to 3 are tokens, 4 and 5 are not (a yes and a no): the read's lessons take
    # their target from the slots' contents, which hold no yes or no
    tokens = [0, 1, 2, 3, -1, -1]
    brain = composed(recall_copy=2.0, recall_tokens=tokens)
    brain.act(event(HEAR, 1), greedy=True)
    brain.act(event(NOISE, 3), greedy=True)
    before = [np.array(m, copy=True) for m in brain.recall.head_maps[0]]
    report = brain.lesson(event(HEAR, ASK), [5])
    assert report["address_rows"] == 0.0
    for a, b in zip(before, brain.recall.head_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    report = brain.lesson(event(HEAR, ASK), [3])  # a token: the read learns
    assert report["address_rows"] == 1.0
    assert any(not np.array_equal(a, b)
               for a, b in zip(before, brain.recall.head_maps[0], strict=True))
    # every action declared no token: the pathway starts at zero and no lesson moves the read
    none = composed(recall_copy=2.0, recall_tokens=[-1] * TOKENS)
    assert not any(weight for *_, weight, _ in pathway(none))
    none.act(event(HEAR, 1), greedy=True)
    assert none.lesson(event(HEAR, ASK), [2])["address_rows"] == 0.0
    # without tokens every taught row teaches the read, as before
    plain = composed(recall_copy=2.0)
    plain.act(event(HEAR, 1), greedy=True)
    assert plain.lesson(event(HEAR, ASK), [5])["address_rows"] == 1.0


def test_under_surprise_a_lesson_whose_answer_is_no_token_teaches_every_row():
    x = np.concatenate([event(HEAR, 0), event(HEAR, 1)])
    taught = {}
    for name, tokens in (("tokens", list(range(TOKENS))), ("none", [-1] * TOKENS)):
        brain = composed(recall_tokens=tokens, learning_surprise=True)
        brain.act(x, greedy=True)
        answer = np.asarray(brain._greedy(  # noqa: SLF001  the act the lesson judges by
            brain._qualified(brain.stimulus(x), brain._activity(), operation="act"),  # noqa: SLF001
            brain._placed(x), brain._decision_key(brain._coded(x)),  # noqa: SLF001
        ))
        brain.lesson(x, answer)  # the label is the brain's own answer on both rows
        taught[name] = brain.last_learning["demonstration_accepted_presentations"]
    assert taught["tokens"] == 0.0  # answered as shown: no surprise, no lesson
    assert taught["none"] == 2.0  # no token answers: taught at every lesson, right or not


def test_tokens_without_the_pathway_declare_which_answers_are_tokens():
    # no copy pathway: the tokens only tell the read's lessons which answers a slot can hold
    brain = composed(recall_tokens=[0, 1, 2, 3, -1, -1], learning_surprise=True)
    assert not len(brain.recall.copy_synapses()[0])
    assert brain.describe()["genes"]["recall_tokens"] == [0, 1, 2, 3, -1, -1]
    plain = composed(learning_surprise=True)
    np.testing.assert_array_equal(brain.brain.efficacy, plain.brain.efficacy)
    brain.act(event(HEAR, 1), greedy=True)
    before = [np.array(m, copy=True) for m in brain.recall.head_maps[0]]
    assert brain.lesson(event(HEAR, ASK), [5])["address_rows"] == 0.0
    for a, b in zip(before, brain.recall.head_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    assert brain.last_learning["demonstration_accepted_presentations"] == 1.0
