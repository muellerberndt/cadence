"""Records of bound moments, the voice and the senses for things and sound (cadence 0.101)."""

from __future__ import annotations

import numpy as np
import pytest

from cadence import Brain
from cadence.bound import BoundRecords
from cadence.senses import Ear, Things

THINGS = Things({"hue": (0.0, 1.0), "shape": [(0.0, 1.0)] * 3, "place": [(-1.0, 1.0)]}, per=8)
EAR = Ear(frames=8, bands=10)
WORDS = {w: np.random.default_rng(k).normal(size=(20, 10)) for k, w in enumerate(
    ("cup", "apple", "red", "white", "what colour"))}
CUP = {"hue": 0.85, "shape": [0.6, 0.3, 0.2]}
APPLE = {"hue": 0.0, "shape": [1.0, 0.1, 0.1]}


def sound(word: str, rng: np.random.Generator, noise: float = 0.2) -> np.ndarray:
    return EAR.code(WORDS[word] + noise * rng.normal(size=WORDS[word].shape))


def flat(sense: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([sense[field] for field in THINGS.widths], axis=1)


def heard_as(said: np.ndarray) -> str | None:
    if not said.any():
        return None
    return max(WORDS, key=lambda w: float(said[0] @ EAR.code(WORDS[w])[0]))


def eyes(thing: dict, place: float) -> dict[str, np.ndarray]:
    return THINGS.rows({"hue": [thing["hue"]], "shape": [thing["shape"]], "place": [place]})


def parrot(**options) -> Brain:
    return Brain.compose(senses={"eyes": THINGS.widths, "ear": EAR.width}, actions=2,
                         code=512, bound=256, voice="ear", arousal=True, seed=0,
                         backend="cpu", **options)


def named(brain: Brain, rng: np.random.Generator, rounds: int = 3) -> None:
    for _ in range(rounds):
        for thing, word in ((CUP, "cup"), (APPLE, "apple")):
            brain.act({"eyes": eyes(thing, rng.uniform(-1, 1)), "ear": sound(word, rng)},
                      greedy=True)


def test_compose_lays_out_a_voice_and_holds_the_records():
    brain = parrot()
    assert list(brain.senses) == ["eyes", "ear", "voice"]
    assert brain.inputs == THINGS.width + 2 * EAR.width
    assert isinstance(brain.bound, BoundRecords) and brain.bound.store.count == 0
    genes = brain.describe()["genes"]
    assert genes["bound"] == 256 and genes["voice"] == "ear"
    plain = Brain.compose(senses={"eyes": THINGS.widths, "ear": EAR.width}, actions=2,
                          code=512, seed=0, backend="cpu")
    assert plain.bound is None and "bound" not in plain.describe()["genes"]


@pytest.mark.parametrize("options, says", [
    ({"inputs": 8, "bound": 16}, "senses"),
    ({"senses": {"ear": 8}, "bound": 16}, "code=units"),
    ({"senses": {"ear": 8}, "code": 64, "voice": "ear"}, "bound=records"),
    ({"senses": {"ear": 8}, "code": 64, "bound": 16, "voice": "nose"}, "voice names"),
    ({"senses": {"ear": 8, "voice": 8}, "code": 64, "bound": 16}, "name the declared"),
    ({"senses": {"ear": 8}, "code": 64, "bound": 0}, "positive integer"),
    ({"senses": {"eyes": 16, "ear": {"a": 8, "b": 8}}, "code": 64, "bound": 16, "voice": "ear"},
     "one field"),
    ({"senses": {"ear": 40}, "code": 64, "learned_code": 32, "bound": 16}, "learned code"),
])
def test_compose_refuses_by_name(options, says):
    with pytest.raises(ValueError, match=says):
        Brain.compose(actions=2, **options)


def test_a_sound_heard_alone_brings_back_the_thing_it_was_heard_with():
    rng = np.random.default_rng(0)
    brain = parrot()
    assert not flat(brain.complete({"ear": sound("cup", rng)})["eyes"]).any()  # nothing yet
    named(brain, rng)
    back = flat(brain.complete({"ear": sound("cup", rng)})["eyes"])
    cup_hue = eyes(CUP, 0.0)["hue"]
    assert back.shape == (1, THINGS.width)
    assert float(back[0, :8] @ cup_hue[0]) > 0.5 * float(cup_hue[0] @ cup_hue[0])
    count = brain.bound.store.count
    brain.complete({"ear": sound("apple", rng)})  # a pure read
    assert brain.bound.store.count == count


def test_the_brain_settles_on_what_its_records_bring_back():
    rng = np.random.default_rng(1)
    brain = parrot()
    named(brain, rng)
    take = sound("cup", rng)
    rows = brain._sensed_rows({"ear": take})
    assert rows[0, : THINGS.width].any()  # the eyes' columns hold the recalled cup
    held, brain.bound = brain.bound, None
    try:
        assert not brain._sensed_rows({"ear": take})[0, : THINGS.width].any()
    finally:
        brain.bound = held


def test_a_moment_writes_what_the_senses_gave_and_not_what_came_back():
    rng = np.random.default_rng(2)
    brain = parrot()
    named(brain, rng)
    brain.act({"ear": sound("apple", rng)}, greedy=True)
    slot = int(brain.bound.written[0])
    assert not brain.bound.store.values[slot, : THINGS.width].any()


def test_the_voice_says_the_name_heard_with_a_thing_and_babbles_otherwise():
    rng = np.random.default_rng(3)
    brain = parrot()
    assert not brain.say({"eyes": eyes(CUP, 0.2)}).any()  # no sound held: silence
    assert brain.bound.said_from == [None]
    for word in WORDS:
        brain.act({"ear": sound(word, rng)}, greedy=True)
    babble = brain.say({"eyes": eyes(CUP, 0.2)})
    assert babble.shape == (1, EAR.width) and brain.bound.said_from == ["babble"]
    named(brain, rng)
    assert heard_as(brain.say({"eyes": eyes(CUP, 0.1)})) == "cup"
    assert brain.bound.said_from == ["heard"]


def test_a_reward_makes_what_was_said_the_answer_and_no_reward_refuses_it():
    rng = np.random.default_rng(4)
    brain = parrot()
    for word in WORDS:
        brain.act({"ear": sound(word, rng)}, greedy=True)
    question = {"eyes": eyes(CUP, 0.3), "ear": sound("what colour", rng)}
    for _ in range(80):
        said = brain.say(question)
        right = heard_as(said) == "white"
        brain.live({**question, "voice": said})
        slot = int(brain.bound.written[0])
        brain.live({"eyes": eyes(CUP, 0.3)}, reward=[1.0 if right else 0.0])
        if right:
            break
        assert brain.bound.refused[slot]
    assert right and not brain.bound.refused[slot]
    assert brain.bound.paid[slot] == BoundRecords.PAY
    said = brain.say({"eyes": eyes(CUP, 0.3), "ear": sound("what colour", rng)})
    assert heard_as(said) == "white" and brain.bound.said_from == ["said"]


def test_a_life_without_rewards_pays_nothing():
    rng = np.random.default_rng(5)
    brain = parrot()
    for word in WORDS:
        brain.act({"ear": sound(word, rng)}, greedy=True)
    said = brain.say({"eyes": eyes(CUP, 0.0)})
    brain.live({"eyes": eyes(CUP, 0.0), "voice": said})
    slot = int(brain.bound.written[0])
    brain.live({"eyes": eyes(CUP, 0.0)})  # no reward given
    assert not brain.bound.refused[slot] and brain.bound.paid[slot] == 0.0


def test_step_pays_the_record_of_the_moment_it_rewards():
    rng = np.random.default_rng(6)
    brain = parrot()
    brain.step({"eyes": eyes(CUP, 0.0), "voice": sound("cup", rng)})
    slot = int(brain.bound.written[0])
    brain.step({"eyes": eyes(CUP, 0.0)}, reward=[1.0])
    assert brain.bound.paid[slot] == BoundRecords.PAY


def test_save_and_load_keep_the_records(tmp_path):
    rng = np.random.default_rng(7)
    brain = parrot()
    named(brain, rng)
    said = brain.say({"eyes": eyes(APPLE, 0.0)})
    path = brain.save(tmp_path / "parrot.npz")
    again = Brain.load(path)
    assert isinstance(again.bound, BoundRecords)
    assert again.bound.store.count == brain.bound.store.count
    assert np.array_equal(again.bound.refused, brain.bound.refused)
    assert again.bound.draws == brain.bound.draws
    take = sound("apple", rng)
    assert np.allclose(flat(again.complete({"ear": take})["eyes"]),
                       flat(brain.complete({"ear": take})["eyes"]))
    assert np.allclose(again.say({"eyes": eyes(APPLE, 0.0)}), said)
    import json
    with np.load(path) as data:
        assert "bound" in json.loads(str(data["brain"]))["features"]


def test_grow_and_graft_refuse_bound_records():
    brain = parrot()
    with pytest.raises(ValueError, match="bound records"):
        brain.grow(modules=(80,))
    with pytest.raises(ValueError, match="bound records"):
        brain.graft(parrot())


def test_complete_and_say_refuse_a_brain_without_records():
    plain = Brain.compose(senses={"ear": 40}, actions=2, code=64, seed=0)
    with pytest.raises(ValueError, match="bound=records"):
        plain.complete({"ear": np.ones((1, 40))})
    quiet = Brain.compose(senses={"ear": 40}, actions=2, code=64, bound=8, seed=0)
    with pytest.raises(ValueError, match="voice"):
        quiet.say({"ear": np.ones((1, 40))})
    with pytest.raises(ValueError, match="by name"):
        quiet.complete(np.ones((1, 40)))


def test_a_heard_name_draws_the_focus_to_the_thing_it_names():
    from cadence.space.map import Map

    rng = np.random.default_rng(8)
    space = Map((-1.0, 1.0, -1.0, 1.0), (12, 12), units=8,
                values={"hue": 8, "shape": 24}, surprise=True, seed=0)
    brain = parrot(trained=[space], surprise=True)
    named(brain, rng, rounds=4)
    spots = {"cup": np.array([-0.5, 0.5]), "apple": np.array([0.5, 0.5])}
    for k, (thing, _) in enumerate(((CUP, "cup"), (APPLE, "apple"))):
        space.moment(seen=[(j, spots[w], {"hue": eyes(t, 0.0)["hue"][0],
                                          "shape": eyes(t, 0.0)["shape"][0]})
                           for j, (t, w) in enumerate(((CUP, "cup"), (APPLE, "apple"))[: k + 1])])
        brain.act({"eyes": eyes(thing, 0.0)}, greedy=True)
    for word in ("cup", "apple"):
        brain.act({"ear": sound(word, rng)}, greedy=True)
        assert brain.focus is not None and brain.focus.sense == "named"
        assert np.linalg.norm(np.asarray(brain.focus.place) - spots[word]) < 0.2


def test_things_and_ear():
    rows = THINGS.rows({"hue": [0.5], "shape": [[0.1, 0.2, 0.3]]})
    assert set(rows) == {"hue", "shape"} and rows["shape"].shape == (1, 24)
    with pytest.raises(ValueError, match="no field"):
        THINGS.rows({"smell": [0.1]})
    frames = np.random.default_rng(0).normal(size=(30, 10))
    code = EAR.code(frames)
    assert code.shape == (1, 80) and np.isclose(np.linalg.norm(code), 1.0)
    assert np.allclose(EAR.code(frames + 3.0), code)  # louder: the same code
    slower = np.repeat(frames, 2, axis=0)
    assert float(EAR.code(slower)[0] @ code[0]) > 0.9  # slower: nearly the same
    assert not EAR.code(np.zeros((5, 10))).any()  # one level: silence
    with pytest.raises(ValueError, match="bands"):
        EAR.code(np.zeros((5, 3)))
    audio = np.sin(np.linspace(0, 2000, 8000))
    assert Ear().hear(audio).shape == (1, 640)


def test_what_was_said_answers_only_once_paid_however_often_it_is_told():
    rng = np.random.default_rng(9)
    for surprise in (False, True):
        brain = parrot(surprise=surprise)
        for word in WORDS:
            brain.act({"ear": sound(word, rng)}, greedy=True)
        question = {"eyes": eyes(CUP, 0.3), "ear": sound("what colour", rng)}
        said = brain.say(question)
        for _ in range(4):  # the same moment told again, no reward given
            brain.live({**question, "voice": said})
            brain.live({"eyes": eyes(CUP, 0.3)})
        brain.say(question)
        assert brain.bound.said_from == ["babble"]
        assert not brain.bound.paid[: brain.bound.store.count].any()


def test_a_reward_after_wait_pays_the_moment_acted_at():
    rng = np.random.default_rng(10)
    brain = parrot()
    for word in WORDS:
        brain.act({"ear": sound(word, rng)}, greedy=True)
    said = brain.say({"eyes": eyes(CUP, 0.0)})
    brain.live({"eyes": eyes(CUP, 0.0), "voice": said})
    acted = int(brain.bound.written[0])
    brain.wait({"eyes": eyes(APPLE, 0.5), "ear": sound("red", rng)})
    brain.live({"eyes": eyes(CUP, 0.0)}, reward=[1.0])
    assert brain.bound.paid[acted] == BoundRecords.PAY
    assert brain.bound.paid.sum() == BoundRecords.PAY


def test_records_kept_for_good_are_read_however_long_ago():
    brain = Brain.compose(senses={"eyes": THINGS.widths, "ear": EAR.width}, actions=2, code=512,
                          bound=4, seed=0, backend="cpu")
    for _ in range(15):  # told past 8 tellings, the fading of the depth taken off
        brain.act({"eyes": eyes(CUP, 0.0), "ear": EAR.code(WORDS["cup"])}, greedy=True)
    store = brain.bound.store
    assert store.kept[: store.count].any()
    for _ in range(400):
        store.moment()
    assert flat(brain.complete({"ear": EAR.code(WORDS["cup"])})["eyes"]).any()


def test_two_streams_are_paid_and_refused_each_its_own():
    rng = np.random.default_rng(12)
    brain = parrot()
    for word in WORDS:
        brain.act({"ear": np.vstack([sound(word, rng), sound(word, rng)])}, greedy=True)
    seen = {"eyes": {f: np.vstack([eyes(CUP, 0.0)[f], eyes(APPLE, 0.5)[f]])
                     for f in THINGS.widths}}
    said = brain.say(seen)
    brain.step({**seen, "voice": said})
    slots = brain.bound.written.copy()
    brain.step(seen, reward=[1.0, 0.0])
    assert brain.bound.paid[slots[0]] == BoundRecords.PAY and not brain.bound.refused[slots[0]]
    assert brain.bound.paid[slots[1]] == 0.0 and brain.bound.refused[slots[1]]


def test_imagine_and_reads_write_nothing():
    rng = np.random.default_rng(13)
    brain = parrot()
    named(brain, rng)
    before = {k: v.copy() for k, v in brain.bound.arrays().items()}
    brain.imagine([{"ear": sound("cup", rng)}, {"eyes": eyes(APPLE, 0.0)}])
    brain.complete({"ear": sound("apple", rng)})
    brain.predict({"ear": sound("apple", rng)})
    after = brain.bound.arrays()
    assert all(np.array_equal(before[k], after[k]) for k in before)
