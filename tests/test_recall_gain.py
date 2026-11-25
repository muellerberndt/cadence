"""One learned gain per read port on the declared tokens' identity (``recall_gain``, cadence#262):
at zero a newborn acts as one without it, a lesson steps each gain by the sum over the tokens of the
port's read times the token's motor movement, and what a gain learns holds for a token no lesson
used. The option is described and saved where it is on. A brain of several motor slots steps its
gains as a brain of one does (cadence#373).
"""
from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import pytest

import cadence as cd

TOKENS = 6
INPUTS = TOKENS + 3  # the tokens, a HEAR, an ASK and a NOISE flag
HEAR, ASK, NOISE = TOKENS, TOKENS + 1, TOKENS + 2
RATE = 0.05


def composed(**options):
    options.setdefault("modules", (16,))
    return cd.Brain.compose(INPUTS, TOKENS, seed=3, episodic=False, recall=True, recall_founder=True, **options)


def gained(**options):
    options.setdefault("recall_tokens", range(TOKENS))
    return composed(recall_gain=RATE, **options)


def event(*on):
    x = np.zeros((1, INPUTS))
    x[0, list(on)] = 1.0
    return x


def motor_drive(brain, x):
    """The drive on the motor neurons at ``x``, after the memories' reads."""
    return brain.stimulus(x)[:, brain.motor_index]


def test_it_is_an_option_that_needs_tokens_and_a_newborn_reads_as_without_it():
    founder = composed(recall_tokens=range(TOKENS))
    assert founder.recall.gain == 0.0 and not founder.recall.gains.size
    assert "recall_gain" not in founder.describe()["genes"]
    assert cd.Recall.options()["gain"] == 0.0
    with pytest.raises(ValueError, match="needs tokens"):
        composed(recall_gain=RATE)
    with pytest.raises(ValueError, match="gain"):
        composed(recall_gain=-0.1, recall_tokens=range(TOKENS))
    with pytest.raises(ValueError, match="observation's own coordinates"):
        cd.Brain.compose(40, 4, code=64, recall=True, recall_founder=True, recall_gain=RATE,
                         recall_tokens=[0, 1, 2, 3])
    brain = gained()
    np.testing.assert_array_equal(brain.recall.gains, np.zeros(2))
    assert brain.describe()["genes"]["recall_gain"] == RATE
    assert brain.parameters() == founder.parameters() + 2
    assert brain.connectome.synapses == founder.connectome.synapses  # no synapse is added
    np.testing.assert_array_equal(founder.brain.efficacy, brain.brain.efficacy)
    rng = np.random.default_rng(1)
    for t in range(6):
        x = event(HEAR, int(rng.integers(TOKENS))) if t % 3 else event(HEAR, ASK)
        np.testing.assert_array_equal(motor_drive(founder, x), motor_drive(brain, x))
        np.testing.assert_array_equal(founder.act(x), brain.act(x))
        np.testing.assert_array_equal(np.asarray(founder.basal_ganglia.state.activation),
                                      np.asarray(brain.basal_ganglia.state.activation))


def test_each_read_port_has_its_gain():
    assert len(gained(recall_heads=2).recall.gains) == 4
    assert len(gained(recall_next=False).recall.gains) == 1
    assert gained(recall_heads=2).parameters() == composed(
        recall_heads=2, recall_tokens=range(TOKENS)).parameters() + 4


def test_a_gain_drives_every_token_alike_a_token_no_lesson_used_included():
    brain = gained()
    brain.recall.gains = np.array([0.7, 0.0])
    for token in range(TOKENS):
        brain.reset()
        brain.act(event(HEAR, token), greedy=True)
        brain.act(event(NOISE), greedy=True)
        drive = motor_drive(brain, event(HEAR, ASK))
        expected = 0.7 * brain.recall.amplitude * brain.recall.attended_heads[0][0, 0]
        assert drive[0, token] == pytest.approx(expected)
        assert np.argmax(drive[0]) == token


def test_tokens_without_an_action_take_no_drive_and_an_action_without_a_token_none():
    tokens = [0, 1, 2, -1, -1, -1]
    brain = gained(recall_tokens=tokens)
    brain.recall.gains = np.array([1.0, 1.0])
    brain.act(event(HEAR, 4), greedy=True)  # coordinate 4 is no action's token
    brain.act(event(NOISE), greedy=True)
    assert not motor_drive(brain, event(HEAR, ASK)).any()
    brain.reset()
    brain.act(event(HEAR, 2), greedy=True)
    brain.act(event(NOISE), greedy=True)
    drive = motor_drive(brain, event(HEAR, ASK))
    assert drive[0, 2] > 0 and not drive[0, 3:].any()


def test_a_lesson_steps_each_ports_gain_by_the_sum_of_its_tokens_contrast():
    brain = gained()
    recall = brain.recall
    captured = {}
    original = recall._motor_movement  # noqa: SLF001

    def keep(plus, minus):
        captured["movement"] = original(plus, minus)
        return captured["movement"]

    recall._motor_movement = keep  # noqa: SLF001
    brain.act(event(HEAR, 2), greedy=True)
    brain.act(event(NOISE), greedy=True)
    brain.stimulus(event(HEAR, ASK))
    reads = recall._port_reads(recall._held_attention(recall.last))  # noqa: SLF001
    rate = min(RATE, cd.Recall.GAIN_BOUND / sum(float(np.sum(r * r)) for r in reads))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = brain.lesson(event(HEAR, ASK), [2])
    assert report["gain_rows"] == 1.0 and report["gain_refused"] == 0.0
    movement = captured["movement"]
    for k, r in enumerate(reads):
        assert recall.gains[k] == pytest.approx(rate * float(r[0, :TOKENS] @ movement[0]))
    assert recall.gains[0] > 0  # the read held the label: the recall port's gain rises


def test_a_rows_rate_is_bounded_by_the_energy_of_its_reads():
    """Above the bound the rate stops mattering: a row taught alone moves its own drive by at most
    ``GAIN_BOUND`` times the motor's movement, whatever the rate."""
    def stepped(rate):
        brain = composed(recall_gain=rate, recall_tokens=range(TOKENS))
        brain.act(event(HEAR, 2), greedy=True)
        brain.stimulus(event(HEAR, ASK))
        reads = brain.recall._port_reads(brain.recall._held_attention(brain.recall.last))  # noqa: SLF001
        energy = sum(float(np.sum(r * r)) for r in reads)
        return brain, brain.recall._gain_rates(reads), energy  # noqa: SLF001

    _, low, energy = stepped(1e-3)
    assert energy > 0 and low[0] == 1e-3  # below the bound: the rate
    _, high, energy = stepped(1e6)
    assert high[0] == pytest.approx(cd.Recall.GAIN_BOUND / energy)
    twice, once = stepped(1e3)[0], stepped(1e6)[0]
    for b in (twice, once):
        b.lesson(event(HEAR, ASK), [4])
    np.testing.assert_array_equal(twice.recall.gains, once.recall.gains)


VALUES = 4  # the values of each motor slot of a brain of several


def slotted(slots, rate=RATE, heads=1, streams=1, seed=3):
    """A brain of ``slots`` motor slots of ``VALUES`` values each: observation ``i`` and action
    ``i`` are one token."""
    tokens = slots * VALUES
    return cd.Brain.compose(tokens, tokens, slots=(VALUES,) * slots, modules=(16,), seed=seed,
                            episodic=False, recall=True, recall_heads=heads, recall_gain=rate,
                            recall_tokens=range(tokens))


def one_hot(state):
    """A state of one value per slot, ``(batch, slots)``, as the observation that holds it."""
    state = np.atleast_2d(state)
    x = np.zeros((len(state), state.shape[1] * VALUES))
    x[np.arange(len(state))[:, None], np.arange(state.shape[1]) * VALUES + state] = 1.0
    return x


def watched(recall):
    """The motor movement of every lesson and sampled act of ``recall`` from here on, kept."""
    kept = []
    original = recall._motor_movement  # noqa: SLF001

    def keep(plus, minus):
        kept.append(original(plus, minus))
        return kept[-1]

    recall._motor_movement = keep  # noqa: SLF001
    return kept


def test_one_motor_slot_keeps_its_rate_to_the_bit():
    """The bound of a lesson is the motor slots times ``GAIN_BOUND``: with one slot the rate is
    ``min(gain, GAIN_BOUND / E)`` as it was, the same bits."""
    brain = composed(recall_gain=1e6, recall_tokens=range(TOKENS))
    assert brain.learner.slot_count == 1
    brain.act(event(HEAR, 2), greedy=True)
    brain.stimulus(event(HEAR, ASK))
    recall = brain.recall
    reads = recall._port_reads(recall._held_attention(recall.last))  # noqa: SLF001
    energy = np.zeros(1)
    for r in reads:
        energy = energy + np.einsum("bi,bi->b", r, r)
    before = np.minimum(recall.gain, cd.Recall.GAIN_BOUND / energy)
    assert recall.motor_slots == 1
    np.testing.assert_array_equal(recall._gain_rates(reads), before)  # noqa: SLF001
    np.testing.assert_array_equal(recall._gain_rates(reads, 1), before)  # noqa: SLF001


@pytest.mark.parametrize("slots", [2, 8, 32])
def test_a_lessons_rate_is_bounded_with_the_motor_slots(slots):
    """A lesson's nudge carries ``beta / slots`` on each motor slot and the energy sums the reads
    of all of them, so the lesson's bound is ``slots * GAIN_BOUND / E``: the step of the gains is
    that rate times each port's contrast."""
    brain = slotted(slots, rate=1e6)
    recall = brain.recall
    state = np.arange(slots) % VALUES
    brain.act(one_hot(state), greedy=True)
    brain.stimulus(one_hot(state))
    reads = recall._port_reads(recall._held_attention(recall.last))  # noqa: SLF001
    energy = sum(float(np.sum(r * r)) for r in reads)
    rate = slots * cd.Recall.GAIN_BOUND / energy
    assert recall._gain_rates(reads)[0] == pytest.approx(rate)  # noqa: SLF001
    assert recall._gain_rates(reads, 1)[0] == pytest.approx(rate / slots)  # noqa: SLF001
    assert brain.learner.slot_count == recall.motor_slots == slots
    moved = watched(recall)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the lesson follows an act on the same observation
        report = brain.lesson(one_hot(state), [(state + 1) % VALUES])
    assert report["gain_rows"] == 1.0 and report["gain_refused"] == 0.0
    movement = moved[-1]
    assert movement.any()
    for k, r in enumerate(reads):
        assert recall.gains[k] == pytest.approx(rate * float(r[0] @ movement[0]))


def shared(slots, rate=1e6):
    """A brain whose ``slots`` motor slots all read the same ``VALUES`` observation coordinates
    as their tokens: one sensed value, said on every slot."""
    return cd.Brain.compose(VALUES, slots * VALUES, slots=(VALUES,) * slots, modules=(16,), seed=3,
                            episodic=False, recall=True, recall_gain=rate,
                            recall_tokens=list(range(VALUES)) * slots)


@pytest.mark.parametrize("slots", [2, 8, 32])
def test_slots_that_read_the_same_coordinates_count_them_once_for_each(slots):
    """Where every motor slot reads the same coordinates the reads' energy does not grow with the
    slots. The lesson's bound counts a coordinate once for each motor neuron that reads it, so
    its rate is the rate of one slot, ``GAIN_BOUND / E``, and the drive a row adds on the motor
    neurons stays within ``slots * GAIN_BOUND`` times their movement. A sampled act's rate takes
    the energy as it is."""
    brain = shared(slots)
    recall = brain.recall
    x = np.eye(VALUES)[[2]]
    brain.act(x, greedy=True)
    brain.stimulus(x)
    reads = recall._port_reads(recall._held_attention(recall.last))  # noqa: SLF001
    energy = sum(float(np.sum(r * r)) for r in reads)
    one = cd.Recall.GAIN_BOUND / energy
    assert recall._gain_rates(reads)[0] == pytest.approx(one)  # noqa: SLF001
    assert recall._gain_rates(reads, 1)[0] == one  # noqa: SLF001
    moved = watched(recall)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain.lesson(x, [np.full(slots, 1)])
    movement = moved[-1][0, recall._token_motors]  # noqa: SLF001
    added = sum(
        g * r[0, recall._token_values]  # noqa: SLF001
        for g, r in zip(recall.gains, reads, strict=True)
    )  # the drive the step adds on each motor neuron
    assert movement.any() and added.any()
    bound = slots * cd.Recall.GAIN_BOUND * float(np.linalg.norm(movement))
    assert float(np.linalg.norm(added)) <= bound * (1 + 1e-9)


def test_the_lessons_nudge_carries_beta_over_the_motor_slots():
    """What the lesson's bound counts on: ``Learner.nudge_for`` gives every motor slot's nudge
    ``beta / slots``, and a sampled act's nudge (``ActorCritic._nudged_groups``) the whole
    beta. The motor slots in the bound follow the first and not the second."""
    for slots in (1, 2, 8):
        brain = slotted(slots)
        learner, actor = brain.learner, brain.basal_ganglia
        assert learner.slot_count == slots
        resting = np.zeros((1, learner.brain.connectome.n))
        assert learner.nudge_for(resting, learner.config.beta).beta == learner.config.beta / slots
        kept = {}
        original = learner.brain.settle_batch

        def settle(*args, _kept=kept, _original=original, **options):
            if options.get("nudge") is not None:
                _kept["beta"] = options["nudge"].beta
            return _original(*args, **options)

        learner.brain.settle_batch = settle
        try:
            state = np.arange(slots) % VALUES
            brain.act(one_hot(state))  # sampled: its eligibility settles two nudged phases
        finally:
            learner.brain.settle_batch = original
        assert actor is not None and abs(kept["beta"]) == learner.config.beta


def held_gain(slots, lessons=40, streams=4, seed=0):
    """The recall port's gain after ``lessons`` on a state that mostly holds, sensed as a moment
    of its own and then taught (the hold example of the teaching guide)."""
    brain = slotted(slots, rate=1.0, seed=seed)
    draw = np.random.default_rng(seed + 1)
    state = draw.integers(0, VALUES, (streams, slots))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the state is sensed first: the read finds the present
        for _ in range(lessons):
            following = state.copy()
            changed = np.flatnonzero(draw.random(streams) < 0.15)
            following[changed, draw.integers(0, slots, len(changed))] = draw.integers(
                0, VALUES, len(changed)
            )
            brain.act(one_hot(state), greedy=True)
            brain.lesson(one_hot(state), following)
            state = following
    return float(brain.recall.gains[0])


def test_the_path_switches_on_as_fast_with_many_motor_slots_as_with_few():
    """cadence#373: the gains of a brain of 64 motor slots rise as those of a brain of 4 do. With
    the bound of one slot the rate fell with the slots and the larger brain's gain was a
    thirteenth of the smaller's after the same lessons."""
    few, many = held_gain(4), held_gain(64)
    assert few > 0.1
    assert many > few / 2


def test_a_loaded_grown_grafted_or_copied_brain_keeps_its_motor_slots_in_the_bound(tmp_path):
    """The memory takes the motor slots from its brain's learner wherever a brain is made: a
    loaded, a grown, a grafted and a copied brain step their gains under the bound of the brain
    they came from."""
    brain = slotted(8, rate=1.0)
    state = np.arange(8) % VALUES
    brain.act(one_hot(state), greedy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain.lesson(one_hot(state), [state])
    loaded = cd.Brain.load(brain.save(tmp_path / "slots.npz"))
    grown = brain.grow(history_decays=(*brain.history.decays, 0.5))
    donor = cd.Brain.compose(8 * VALUES, 8 * VALUES, slots=(VALUES,) * 8, modules=(16,), seed=7,
                             episodic=False, recall=True)
    grafted = brain.graft(donor, ["association", "motor"])
    copied = copy.deepcopy(brain)
    for other in (loaded, grown, grafted, copied):
        assert other.recall.motor_slots == other.learner.slot_count == 8
    assert "motor_slots" not in json.dumps(brain.describe())  # no setting: it is the learner's
    with np.load(tmp_path / "slots.npz", allow_pickle=False) as data:
        assert "motor_slots" not in str(data["brain"])  # and no file holds it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for b in (brain, loaded, copied, grown, grafted):
            before = b.recall.gains.copy()
            b.act(one_hot(state), greedy=True)
            b.lesson(one_hot(state), [(state + 1) % VALUES])
            assert (b.recall.gains != before).any()
    np.testing.assert_array_equal(loaded.recall.gains, brain.recall.gains)
    np.testing.assert_array_equal(copied.recall.gains, brain.recall.gains)
    with pytest.raises(ValueError, match="motor_slots"):
        cd.Recall(brain.connectome, motor_slots=0)


def test_reward_keeps_the_bound_of_one_slot_with_several_motor_slots():
    """A sampled act's nudge carries the whole beta on every motor slot, so its eligibility keeps
    ``GAIN_BOUND / E``: the motor slots count in a lesson's bound alone."""
    slots = 8
    brain = slotted(slots, rate=1e6)
    recall = brain.recall
    state = np.arange(slots) % VALUES
    brain.act(one_hot(state), greedy=True)
    moved = watched(recall)
    asked = []
    original = recall._gain_rates  # noqa: SLF001

    def rates(reads, motor_slots=None):
        asked.append((reads, motor_slots))
        return original(reads, motor_slots)

    recall._gain_rates = rates  # noqa: SLF001
    brain.act(one_hot(state))  # sampled: the act keeps its eligibility
    (reads, motor_slots), = asked
    assert motor_slots == 1
    energy = sum(float(np.sum(r * r)) for r in reads if r is not None)
    pending = recall.full_eligibility(brain.basal_ganglia._pending_memory)  # noqa: SLF001
    ports = len(recall.gains)
    movement = moved[-1]
    assert movement.any() and reads[0] is not None
    for k, r in enumerate(reads):
        expected = 0.0 if r is None else cd.Recall.GAIN_BOUND / energy * float(r[0] @ movement[0])
        assert pending[0, 0, pending.shape[-1] - ports + k] == pytest.approx(expected)
    assert pending[0, 0, pending.shape[-1] - ports] != 0.0


def test_a_lesson_without_a_nudged_phase_moves_nothing_and_a_brain_without_it_reports_none():
    plain = composed(recall_tokens=range(TOKENS))
    plain.act(event(HEAR, 1), greedy=True)
    assert "gain_rows" not in plain.lesson(event(HEAR, ASK), [1])
    brain = gained(learning_surprise=True)
    brain.act(event(HEAR, 1), greedy=True)
    answer = brain.act(event(HEAR, ASK), greedy=True)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a lesson after the act on the same moment
        report = brain.lesson(event(HEAR, ASK), [int(answer)])  # no surprise: no phase runs
    assert report["gain_rows"] == 0.0 and report["gain_refused"] == 0.0
    np.testing.assert_array_equal(brain.recall.gains, np.zeros(2))


def test_the_first_lesson_at_zero_moves_every_other_map_as_without_it():
    """The address lesson reads nothing back through the gains: at zero the lesson's phases are
    those of the brain without them, so its address, position and synaptic steps are too."""
    plain, brain = composed(recall_tokens=range(TOKENS)), gained()
    for b in (plain, brain):
        b.act(event(HEAR, 2), greedy=True)
        b.lesson(event(HEAR, ASK), [4])
    for a, b in zip(plain.recall.head_map(0), brain.recall.head_map(0), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(plain.recall.position_maps[0], brain.recall.position_maps[0], strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(plain.brain.efficacy, brain.brain.efficacy)
    assert brain.recall.gains.any()  # and the gains alone hold something more


def test_a_gain_step_that_is_not_finite_is_refused_and_the_lesson_stands(monkeypatch):
    brain = gained()
    plain = composed(recall_tokens=range(TOKENS))
    for b in (brain, plain):
        b.act(event(HEAR, 2), greedy=True)
    width = len(brain.recall.motor)
    monkeypatch.setattr(brain.recall, "_motor_movement",
                        lambda plus, minus: np.full((1, width), np.inf))
    report = brain.lesson(event(HEAR, ASK), [4])
    plain.lesson(event(HEAR, ASK), [4])
    assert report["gain_rows"] == 0.0 and report["gain_refused"] == 1.0
    np.testing.assert_array_equal(brain.recall.gains, np.zeros(2))
    np.testing.assert_array_equal(plain.brain.efficacy, brain.brain.efficacy)


def test_a_checkpoint_keeps_the_gains_and_continues_exactly(tmp_path):
    brain = gained(recall_heads=2)
    for token in (1, 2):
        brain.act(event(HEAR, token), greedy=True)
        brain.lesson(event(HEAR, ASK), [token])
    assert brain.recall.gains.any()
    path = brain.save(tmp_path / "gain.npz")
    with np.load(path, allow_pickle=False) as data:
        assert "memories/recall/gains" in data.files
    twin = cd.Brain.load(path)
    assert twin.recall.gain == RATE
    np.testing.assert_array_equal(twin.recall.gains, brain.recall.gains)
    for b in (brain, twin):
        b.lesson(event(HEAR, ASK), [3])
    np.testing.assert_array_equal(twin.recall.gains, brain.recall.gains)
    assert twin.describe() == brain.describe()
    plain = composed(recall_tokens=range(TOKENS))
    with np.load(plain.save(tmp_path / "plain.npz"), allow_pickle=False) as data:
        assert "memories/recall/gains" not in data.files
    assert not cd.Brain.load(tmp_path / "plain.npz").recall.gains.size


def test_a_saved_memory_without_the_gain_refuses_its_gains(tmp_path):
    brain = gained()
    brain.act(event(HEAR, 1), greedy=True)
    path = brain.save(tmp_path / "gain.npz", compressed=False)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(str(arrays["brain"]))
    del meta["memories"]["recall"]["gain"]
    arrays["brain"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="recall gains"):
        cd.Brain.load(tmp_path / "bad.npz")


def test_retune_refuses_it_as_a_construction_setting():
    with pytest.raises(ValueError, match="construction setting"):
        gained().retune(recall_gain=0.1)


def taught(**options):
    brain = gained(**options)
    for token in (1, 2, 3):
        brain.act(event(HEAR, token), greedy=True)
        brain.lesson(event(HEAR, ASK), [token])
    assert brain.recall.gains.any()
    return brain


def test_grow_carries_the_gains_and_the_grown_brain_acts_as_the_trained_one():
    trained = taught()
    grown = trained.grow(history_decays=(0.5,))
    assert grown.recall.gain == RATE
    np.testing.assert_array_equal(grown.recall.gains, trained.recall.gains)
    twin = copy.deepcopy(trained)
    twin.reset()
    rng = np.random.default_rng(8)
    for _ in range(6):
        x = rng.random((3, INPUTS))
        np.testing.assert_array_equal(twin.act(x, greedy=True), grown.act(x, greedy=True))


def test_grow_refuses_what_the_tokens_refuse():
    with pytest.raises(ValueError, match="tokens"):
        taught().grow(recall_heads=2)
    with pytest.raises(ValueError, match="per action"):
        taught().grow(slots=(TOKENS, 2))


def test_grow_adds_no_memory_with_tokens_as_it_adds_none_with_a_pathway():
    trained = cd.Brain.compose(6, 4, modules=(16,), seed=1, recall=False)
    with pytest.raises(ValueError, match="tokens"):
        trained.grow(recall=True, recall_founder=True, recall_gain=RATE, recall_tokens=[0, 1, 2, 3])


def test_graft_keeps_this_brains_gains(tmp_path):
    host = taught()
    donor = cd.Brain.compose(INPUTS, TOKENS, modules=(16,), seed=7, episodic=False, recall=True, recall_founder=True)
    grafted = host.graft(donor, ["association", "motor"])
    assert grafted.recall.gain == RATE
    np.testing.assert_array_equal(grafted.recall.gains, host.recall.gains)
    loaded = cd.Brain.load(grafted.save(tmp_path / "grafted.npz"))
    np.testing.assert_array_equal(loaded.recall.gains, host.recall.gains)


def test_reward_credits_the_gains_of_a_sampled_act(tmp_path):
    brain = cd.Brain.compose(INPUTS, TOKENS, seed=0, modules=(16,), episodic=False, recall=True, recall_founder=True,
                             recall_gain=RATE, recall_tokens=range(TOKENS))
    assert brain.recall.reward_columns == INPUTS + 2
    brain.act(event(HEAR, 2), greedy=True)
    brain.act(event(NOISE), greedy=True)
    brain.act(event(HEAR, ASK))  # sampled: the act keeps its eligibility
    pending = brain.recall.full_eligibility(brain.basal_ganglia._pending_memory)
    assert pending.shape[-1] == brain.recall.reward_columns
    assert not pending[:, 1:, INPUTS:].any()  # the gains' columns: the first row alone
    credit = pending[0, 0, INPUTS:].copy()
    assert credit[0] != 0.0  # the read holds a token: the recall port's gain has a contrast
    report = brain.learn(np.array([1.0]), np.array([True]), event(NOISE))  # a paid act
    assert report["address_reward_step"] > 0.0
    # a newborn's value is near zero, so a paid act's dopamine is positive: the gain moves with
    # its eligibility
    assert brain.recall.gains[0] != 0.0 and np.sign(brain.recall.gains[0]) == np.sign(credit[0])
    w_x, w_c, b = brain.recall.head_map(0)
    assert not (w_x.any() or w_c.any() or b.any())  # the address map stays with address_reward 0
    brain.act(event(HEAR, 3), greedy=True)
    brain.act(event(HEAR, ASK))  # a pending act: its eligibility is saved with the brain
    twin = cd.Brain.load(brain.save(tmp_path / "pending.npz"))
    for b in (brain, twin):
        b.learn(np.array([1.0]), np.array([True]), event(NOISE))
    np.testing.assert_array_equal(twin.recall.gains, brain.recall.gains)


def test_retune_keeps_the_eligibility_with_the_gain_on():
    brain = cd.Brain.compose(INPUTS, TOKENS, seed=0, modules=(16,), episodic=False, recall=True, recall_founder=True,
                             recall_gain=RATE, recall_tokens=range(TOKENS))
    brain.act(event(HEAR, 2), greedy=True)
    brain.act(event(HEAR, ASK))
    brain.learn(np.array([1.0]), np.array([True]), event(NOISE))
    assert brain.basal_ganglia.trace_memory is not None
    brain.retune(recall_amplitude=2.5)
    assert brain.basal_ganglia.trace_memory is not None


def test_a_reward_step_that_is_not_finite_is_refused_and_a_contrast_keeps_none(monkeypatch):
    brain = cd.Brain.compose(INPUTS, TOKENS, seed=0, modules=(16,), episodic=False, recall=True, recall_founder=True,
                             recall_gain=RATE, recall_tokens=range(TOKENS))
    recall = brain.recall
    shape = (recall.map_width + len(recall.association) + 1, recall.reward_columns)
    step = np.zeros(shape)
    step[0, -1] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        recall.propose(step)
    np.testing.assert_array_equal(recall.gains, np.zeros(2))
    width = len(recall.motor)
    monkeypatch.setattr(recall, "_motor_movement",
                        lambda plus, minus: np.full((1, width), np.inf))
    brain.act(event(HEAR, 2), greedy=True)
    brain.act(event(HEAR, ASK))  # sampled: this row's step is not finite
    pending = brain.basal_ganglia._pending_memory
    assert pending is not None and np.isfinite(pending).all() and not pending.any()


@pytest.mark.parametrize(
    ("backend", "device", "precision", "atol"),
    [("torch", "cpu", "float64", 1e-9), ("mlx", None, None, 1e-4)],
    ids=["torch_cpu64", "mlx"],
)
def test_the_gain_learns_on_a_device_as_on_the_cpu(tmp_path, backend, device, precision, atol):
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} not installed")
    cpu = cd.Brain.compose(INPUTS, TOKENS, seed=5, modules=(16,), episodic=False,
                           teaching="normalized", recall=True, recall_founder=True, recall_gain=RATE,
                           recall_tokens=range(TOKENS))
    cpu.save(tmp_path / "brain.npz")
    moved = cd.Brain.load(tmp_path / "brain.npz", backend=backend, device=device,
                          precision=precision)
    assert moved.brain.backend == backend
    for brain in (cpu, moved):
        rng = np.random.default_rng(0)
        for _ in range(4):
            brain.reset()
            token = int(rng.integers(TOKENS))
            brain.act(event(HEAR, token), greedy=True)
            brain.act(event(NOISE), greedy=True)
            brain.lesson(event(HEAR, ASK), [(token + 1) % TOKENS])
    assert np.abs(cpu.recall.gains).max() > 1e-3  # the lessons moved them
    np.testing.assert_allclose(moved.recall.gains, cpu.recall.gains, rtol=0, atol=atol)
    np.testing.assert_allclose(np.asarray(moved.brain.efficacy, dtype=float),
                               np.asarray(cpu.brain.efficacy, dtype=float), rtol=0, atol=atol)
