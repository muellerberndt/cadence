"""``cadence.Message``: an episode held as phased unions of a fixed sparse code (cadence#286)."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def _moments(rng: np.random.Generator, count: int, inputs: int = 35, active: int = 3) -> np.ndarray:
    x = np.zeros((count, inputs))
    for row in x:
        row[rng.choice(inputs, size=active, replace=False)] = 1.0
    return x


def test_a_code_is_ones_at_its_winners_and_silence_codes_to_nothing():
    message = cd.Message(35, units=70, winners=5, seed=1)
    x = _moments(np.random.default_rng(0), 6)
    code = message.code(x)
    assert code.shape == (6, 70) and set(np.unique(code)) == {0.0, 1.0}
    assert (code.sum(axis=1) == 5).all()
    assert not message.code(np.zeros((2, 35))).any()
    assert np.array_equal(code, cd.Message(35, units=70, winners=5, seed=1).code(x))
    assert not np.array_equal(code, cd.Message(35, units=70, winners=5, seed=2).code(x))


def test_a_code_binds_what_a_moment_holds_together():
    # two moments that share two of three active inputs share part of their code and no more
    message = cd.Message(35, seed=3)
    a, b, c = np.zeros((3, 35))
    a[[0, 4, 5]] = 1.0
    b[[0, 4, 6]] = 1.0
    c[[1, 7, 8]] = 1.0
    code = message.code(np.stack([a, b, c]))
    shared, apart = float(code[0] @ code[1]), float(code[0] @ code[2])
    assert 0 < shared < 5 and apart < shared


def test_each_moment_joins_the_union_of_its_phase():
    message = cd.Message(35, phases=3, units=40, winners=4, seed=0)
    x = _moments(np.random.default_rng(1), 5)
    for moment in x:
        message.hear(moment[None])
    assert message.state.shape == (1, 120) and message.width == 120
    held = message.state.reshape(3, 40)
    codes = message.code(x)
    assert np.array_equal(held[0], np.maximum(codes[0], codes[3]))  # moments 0 and 3
    assert np.array_equal(held[1], np.maximum(codes[1], codes[4]))
    assert np.array_equal(held[2], codes[2])
    assert message.told.tolist() == [5]
    assert np.array_equal(message.key()[0], codes.max(axis=0))


def test_the_key_is_the_same_in_any_order_and_the_state_keeps_the_order():
    rng = np.random.default_rng(2)
    x = _moments(rng, 4)
    forward, backward = cd.Message(35, seed=5), cd.Message(35, seed=5)
    for moment in x:
        forward.hear(moment[None])
    for moment in x[::-1]:
        backward.hear(moment[None])
    assert np.array_equal(forward.key(), backward.key())
    assert not np.array_equal(forward.state, backward.state)
    other = cd.Message(35, seed=5)
    for moment in _moments(rng, 4):
        other.hear(moment[None])
    assert float(forward.key()[0] @ other.key()[0]) < float(forward.key()[0] @ forward.key()[0])


def test_streams_hold_their_own_messages_and_begin_again_alone():
    message = cd.Message(35, phases=2, units=30, winners=3, seed=0)
    x = _moments(np.random.default_rng(3), 6).reshape(3, 2, 35)
    for moment in x:
        message.hear(moment)
    kept = message.state[1].copy()
    message.reset(2, rows=np.array([0]))
    assert not message.state[0].any() and message.told.tolist() == [0, 3]
    assert np.array_equal(message.state[1], kept)
    message.hear(x[0])  # stream 0 is at its first phase again, stream 1 at its fourth moment
    assert np.array_equal(message.state[0, :30], message.code(x[0])[0])
    message.keep(np.array([1]))
    assert message.state.shape == (1, 60) and message.told.tolist() == [4]
    message.hear(np.zeros((4, 35)))  # another batch begins the streams afresh
    assert message.state.shape == (4, 60) and not message.state.any()


def test_a_branch_is_private_and_the_arrays_are_what_a_checkpoint_keeps():
    message = cd.Message(35, seed=0)
    x = _moments(np.random.default_rng(4), 3)
    message.hear(x[:1])
    branch = message.branch()
    branch.hear(x[1:2])
    assert message.told.tolist() == [1] and branch.told.tolist() == [2]
    assert set(message.arrays()) == {"trace", "told", "held"}
    again = cd.Message(**message.to_dict())
    again.trace, again.told, again.held = (a.copy() for a in message.arrays().values())
    again.hear(x[1:2])
    assert np.array_equal(again.state, branch.state)


def test_a_boundary_channel_empties_the_message_or_holds_it():
    x = _moments(np.random.default_rng(6), 4)
    x[:, 33] = 0.0
    x[2, 33] = 1.0  # the third moment marks a boundary
    plain, cleared, held = (cd.Message(35, units=8, winners=2, seed=1, **k) for k in (
        {}, {"boundary": 33}, {"boundary": 33, "boundary_holds": True}))
    for m in (plain, cleared, held):
        for moment in x:
            m.hear(moment[None])
    assert plain.told.tolist() == [4] and cleared.told.tolist() == [2] and held.told.tolist() == [2]
    # cleared: the boundary moment and the one after, as a fresh message
    fresh = cd.Message(35, units=8, winners=2, seed=1)
    for moment in x[2:]:
        fresh.hear(moment[None])
    assert np.array_equal(cleared.state, fresh.state)
    # held: the two moments before the boundary, nothing after
    before = cd.Message(35, units=8, winners=2, seed=1)
    for moment in x[:2]:
        before.hear(moment[None])
    assert np.array_equal(held.state, before.state) and held.held.tolist() == [True]
    held.reset(1)
    assert held.held.tolist() == [False] and not held.state.any()
    with pytest.raises(ValueError):
        cd.Message(35, boundary=35)
    with pytest.raises(ValueError):
        cd.Message(35, boundary=1, boundary_holds=2)  # type: ignore[arg-type]


def test_the_fan_in_is_raised_until_every_channel_is_read():
    message = cd.Message(200, units=16, winners=2, fan_in=4, seed=0)
    assert message.to_dict()["fan_in"] == 50  # two readers of each of 400 channels over 16 units
    assert (message.code(np.eye(200)).sum(axis=1) > 0).all()  # a single input never codes to zero


@pytest.mark.parametrize("settings", [
    {"phases": 0}, {"units": 0}, {"winners": 0}, {"winners": 80}, {"fan_in": 0}, {"seed": -1},
    {"phases": True},
])
def test_refusals(settings):
    with pytest.raises(ValueError):
        cd.Message(35, **settings)


def test_observations_are_checked():
    message = cd.Message(35)
    with pytest.raises(ValueError):
        message.hear(np.zeros((1, 34)))
    with pytest.raises(ValueError):
        message.hear(np.full((1, 35), np.nan))


# ------------------------------------------------- the message a brain holds (compose opt-in)


def _brain(**settings):
    return cd.Brain.compose(12, 4, modules=(16,), seed=5, backend="cpu", **settings)


def _stream(rng: np.random.Generator, moments: int, rows: int = 3) -> np.ndarray:
    return _moments(rng, moments * rows, inputs=12, active=2).reshape(moments, rows, 12)


def test_a_brain_without_the_option_holds_no_message():
    brain = _brain()
    assert brain.message is None and "message" not in brain.connectome.populations
    assert "message" not in brain.memories and "message" not in brain.describe()["genes"]


def test_the_brain_with_a_message_is_the_brain_without_it_plus_the_region():
    plain, held, read = _brain(), _brain(message=8), _brain(message=8, message_read=True)
    n = plain.connectome.n
    assert held.connectome.n == n + 32 and read.connectome.n == n + 40
    assert np.array_equal(plain.brain.dense(), held.brain.dense()[:n, :n])
    assert np.array_equal(held.brain.dense(), read.brain.dense()[:n + 32, :n + 32])
    assert list(read.memories).index("message") == list(read.memories).index("recall") + 1
    assert len(read.connectome.populations["message_read"]) == 8
    genes = read.describe()["genes"]
    assert genes["message"] == 8 and genes["message_read"] is True
    assert genes["message_phases"] == 4 and genes["message_amplitude"] == 1.5
    assert {f"message_{k}" for k in range(4)} <= set(read.carried_gains)


def test_the_brain_writes_its_message_as_the_standalone_class_does():
    brain = _brain(message=8, message_winners=2)
    alone = cd.Message(12, phases=4, units=8, winners=2, seed=5)
    stream = _stream(np.random.default_rng(0), 6)
    for moment in stream:
        brain.act(moment, greedy=True)
        alone.hear(moment)
    assert np.array_equal(brain.message.state, alone.state)
    assert np.array_equal(brain.message.key(), alone.key())
    drive = brain.stimulus(stream[0])
    region = brain.connectome.populations["message"]
    assert np.allclose(drive[:, region], 1.5 * alone.state)  # the gains start at one
    brain.reset()
    assert not brain.message.state.any()


def test_a_newborn_reads_the_mean_of_its_blocks_and_lessons_move_the_read():
    brain = _brain(message=8, message_winners=2, message_read=True, message_amplitude=0.0)
    stream = _stream(np.random.default_rng(1), 5)
    for moment in stream:
        brain.act(moment, greedy=True)
    drive = brain.stimulus(stream[0])
    port = brain.connectome.populations["message_read"]
    blocks = brain.message.state.reshape(3, 4, 8)
    assert np.allclose(drive[:, port], 3.0 * blocks.mean(axis=1))
    assert not drive[:, brain.connectome.populations["message"]].any()  # the amplitude is zero
    assert not brain.message.direct and "message_0" not in brain.carried_gains
    with pytest.raises(ValueError):
        brain.retune(message_amplitude=1.0)  # composed silent: no synapse of its own
    assert not any(np.asarray(m).any() for maps in brain.message.maps for m in maps)
    moved = sum(brain.lesson(stream[0], labels)["message_rows"]
                for labels in ([1, 2, 3], [0, 0, 0], [3, 1, 2], [2, 3, 0]))
    assert moved > 0  # rows the lesson taught step the read's maps
    assert any(np.asarray(m).any() for maps in brain.message.maps for m in maps)
    x, c, p, held = brain.message.last
    assert np.allclose(p.sum(axis=1), 1.0) and p.shape == (3, 4)


def test_a_brain_holds_its_message_at_a_boundary_and_hears_a_gain_per_group():
    brain = _brain(message=8, message_winners=2, message_boundary=3, message_boundary_holds=True,
                   message_gains=2)
    stream = _stream(np.random.default_rng(7), 4)
    stream[:, :, 3] = 0.0
    stream[2, :, 3] = 1.0  # the third moment is the boundary
    alone = cd.Message(12, phases=4, units=8, winners=2, seed=5)
    for moment in stream[:2]:
        alone.hear(moment)
    for moment in stream:
        brain.act(moment, greedy=True)
    assert np.array_equal(brain.message.state, alone.state) and brain.message.held.all()
    genes = brain.describe()["genes"]
    assert genes["message_boundary"] == 3 and genes["message_boundary_holds"] is True
    assert genes["message_gains"] == 2
    routes = {name for name in brain.carried_gains if name.startswith("message_")}
    assert routes == {f"message_{k}_{g}" for k in range(4) for g in range(2)}
    route = next(r for r in brain.hearing.routes if r["name"] == "message_0_1")
    brain.hearing.set(route, 2.0)
    drive = brain.stimulus(stream[0])[:, brain.connectome.populations["message"]]
    loud = np.repeat(np.array([1.0, 2.0] + [1.0] * 6), 4)
    assert np.allclose(drive, 1.5 * loud * alone.state)
    with pytest.raises(ValueError):
        _brain(message=8, message_gains=3)  # 3 does not divide 8
    with pytest.raises(ValueError):
        _brain(message=8, message_boundary=12)  # past the input channels


def test_a_streams_message_ends_with_its_episode():
    brain = _brain(message=8)
    stream = _stream(np.random.default_rng(2), 3, rows=2)
    for moment in stream[:2]:
        brain.step(moment)
    brain.learn(np.zeros(2), np.array([True, False]), stream[2])
    assert not brain.message.state[0].any() and brain.message.state[1].any()
    assert brain.message.told.tolist() == [0, 2]


def test_a_saved_brain_keeps_its_message_and_its_read(tmp_path):
    brain = _brain(message=8, message_read=True)
    stream = _stream(np.random.default_rng(3), 4)
    for moment in stream:
        brain.lesson(moment, [0, 1, 2])
        brain.act(moment, greedy=True)
    loaded = cd.Brain.load(brain.save(tmp_path / "brain"))
    assert np.array_equal(loaded.message.state, brain.message.state)
    assert np.array_equal(loaded.message.told, brain.message.told)
    assert np.array_equal(loaded.message.previous, brain.message.previous)
    for mine, theirs in zip(brain.message.maps, loaded.message.maps, strict=True):
        assert all(np.array_equal(a, b) for a, b in zip(mine, theirs, strict=True))
    assert loaded.message.to_dict() == brain.message.to_dict()
    assert loaded.carried_gains == brain.carried_gains
    assert np.array_equal(loaded.act(stream[0], greedy=True), brain.act(stream[0], greedy=True))
    assert np.array_equal(loaded.message.state, brain.message.state)
    # a boundary and gains per group travel with the checkpoint
    marked = _brain(message=8, message_gains=2, message_boundary=3, message_boundary_holds=True)
    stream[:, :, 3] = 0.0
    stream[1, :, 3] = 1.0
    for moment in stream:
        marked.lesson(moment, [0, 1, 2])
        marked.act(moment, greedy=True)
    back = cd.Brain.load(marked.save(tmp_path / "marked"))
    assert back.message.to_dict() == marked.message.to_dict()
    assert back.carried_gains == marked.carried_gains and len(back.message.hearing) == 8
    assert np.array_equal(back.message.held, marked.message.held) and marked.message.held.all()
    assert np.array_equal(back.act(stream[0], greedy=True), marked.act(stream[0], greedy=True))


def test_an_imagined_branch_leaves_the_message_as_it_was():
    brain = _brain(message=8)
    stream = _stream(np.random.default_rng(4), 3, rows=1)
    brain.act(stream[0], greedy=True)
    before = brain.message.state.copy()
    brain.imagine([stream[1], stream[2]])
    assert np.array_equal(brain.message.state, before) and brain.message.told.tolist() == [1]


def test_the_messages_drive_is_retuned_and_its_layout_is_fixed():
    brain = _brain(message=8, message_read=True)
    brain.retune(message_amplitude=0.4, message_read_amplitude=2.0, message_read_rate=0.0)
    assert (brain.message.amplitude, brain.message.read_amplitude) == (0.4, 2.0)
    assert brain.message.read_rate == 0.0
    for change in ({"message": 16}, {"message_phases": 2}, {"message_read": False}):
        with pytest.raises(ValueError):
            brain.retune(**change)
    with pytest.raises(ValueError):
        _brain().retune(message_amplitude=1.0)
    with pytest.raises(ValueError):
        _brain(message=8).retune(message_read_rate=0.1)
    with pytest.raises(ValueError):
        brain.grow(modules=(24,))
    with pytest.raises(ValueError):
        brain.graft(_brain(), regions=["association"])


@pytest.mark.parametrize("settings, error", [
    ({"message_phases": 3}, ValueError),  # a setting without the message
    ({"message": 4, "message_winners": 9}, ValueError),
    ({"message": 4, "message_unknown": 1}, TypeError),
    ({"message": 4, "message_read": 1}, ValueError),
    ({"message": -1}, ValueError),
    ({"message": 4, "learned_code": 8}, ValueError),
])
def test_message_refusals(settings, error):
    with pytest.raises(error):
        _brain(**settings)


def test_a_moments_role_chooses_its_blocks_wherever_it_was_heard():
    roles = ((0,), (1, 2))  # two roles; a moment of neither joins the last group
    message = cd.Message(12, phases=2, units=8, winners=2, seed=3, roles=roles)
    assert message.groups == 3 and message.blocks == 6 and message.width == 48
    x = np.zeros((5, 12))
    x[0, [0, 5]] = 1.0   # role 0
    x[1, [7, 8]] = 1.0   # none
    x[2, [2, 9]] = 1.0   # role 1
    x[3, [0, 6]] = 1.0   # role 0 again: its second block
    x[4, [1, 10]] = 1.0  # role 1 again
    for moment in x:
        message.hear(moment[None])
    held = message.state.reshape(6, 8)
    code = message.code(x)
    for k, block in ((0, 0), (1, 4), (2, 2), (3, 1), (4, 3)):
        assert np.array_equal(held[block], code[k])
    assert message.counts.tolist() == [[2, 2, 1]] and message.told.tolist() == [5]
    # the same moments in another order: the agent's block holds the agent
    again = cd.Message(12, phases=2, units=8, winners=2, seed=3, roles=roles)
    for moment in x[[1, 0, 2]]:
        again.hear(moment[None])
    assert np.array_equal(again.state.reshape(6, 8)[0], code[0])
    assert set(message.arrays()) == {"trace", "told", "held", "counts"}
    rebuilt = cd.Message(**message.to_dict())
    assert rebuilt.roles == roles
    with pytest.raises(ValueError):
        cd.Message(12, roles=((12,),))
    with pytest.raises(ValueError):
        cd.Message(12, roles=((1, 1),))


def test_a_brain_keys_its_message_by_role_and_keeps_it_in_a_checkpoint(tmp_path):
    roles = ((0, 1), (2,))
    brain = _brain(message=8, message_winners=2, message_roles=roles, message_phases=2)
    alone = cd.Message(12, phases=2, units=8, winners=2, seed=5, roles=roles)
    stream = _stream(np.random.default_rng(8), 5)
    for moment in stream:
        brain.lesson(moment, [0, 1, 2])
        brain.act(moment, greedy=True)
        alone.hear(moment)
    assert np.array_equal(brain.message.state, alone.state)
    assert np.array_equal(brain.message.counts, alone.counts)
    assert len(brain.connectome.populations["message"]) == 3 * 2 * 8
    assert {f"message_{k}" for k in range(6)} <= set(brain.carried_gains)
    assert brain.describe()["genes"]["message_roles"] == [[0, 1], [2]]
    loaded = cd.Brain.load(brain.save(tmp_path / "roles"))
    assert loaded.message.roles == roles
    assert np.array_equal(loaded.message.counts, brain.message.counts)
    assert np.array_equal(loaded.act(stream[0], greedy=True), brain.act(stream[0], greedy=True))
    with pytest.raises(ValueError):
        _brain(message=8, message_roles=((12,),))
