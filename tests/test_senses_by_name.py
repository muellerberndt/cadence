"""Senses by name: ``Brain.compose(senses={...})`` and observations as a dict of sense to rows,
equal senses in the fixed sparse code, and ``trained=`` taking a module's name or a map whose
read joins the senses.

Two identities hold the rest in place. A brain composed without ``senses=`` and without a name
or a map in ``trained=`` is the brain of 0.99.1: its parameters, the acts of a short life and
the contents of the checkpoint it saves are the ones recorded on that release. A brain composed
with ``senses=`` and no code is the brain composed with ``inputs=`` at the senses' summed width
and the same seed: the same weights, the same acts and the same saved arrays, with the senses'
names recorded beside them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cadence as cd
from cadence import Brain
from cadence import brain as cadence_brain
from cadence.space.map import Map

# The acts and the array contents below were recorded on the NumPy solver; the compiled dense
# kernel (numba) orders the arithmetic otherwise and gives other bits, so those pins are read
# on the NumPy solver alone (CADENCE_FUSED=0 selects it). The connectome, the parameters and
# the description are read on every backend.
NUMPY_SOLVER = not cadence_brain._FUSED
BYTES_REASON = (
    "acts and array contents were recorded on the NumPy solver; this run uses the compiled "
    "dense kernel (CADENCE_FUSED=0 selects the NumPy loop)"
)
PLANE = (-1.0, 1.0, -0.2, 1.4)
APPLE, BREAD = 0, 1


def ahead(points: np.ndarray) -> np.ndarray:
    return np.asarray(points[:, 0] > 0.0)


def behind(points: np.ndarray) -> np.ndarray:
    return np.asarray(points[:, 0] < 0.0)


# -- digests


def parameter_digest(brain: Brain) -> str:
    connectome, graph = brain.connectome, brain.brain
    h = hashlib.sha256()
    for values in (connectome.pre, connectome.post, connectome.count, connectome.sign,
                   graph.efficacy, graph.bias, graph.log_gain):
        h.update(np.ascontiguousarray(np.asarray(values)).tobytes())
    populations = {k: [int(i) for i in v] for k, v in connectome.populations.items()}
    h.update(json.dumps(populations).encode())
    return h.hexdigest()


def description_digest(brain: Brain) -> str:
    return hashlib.sha256(json.dumps(brain.describe(), sort_keys=True).encode()).hexdigest()


def checkpoint_digest(path: Path) -> str:
    """SHA-256 over every array of a checkpoint, by name, with the metadata JSON entries read
    back without the library version that wrote them."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            value = data[name]
            h.update(name.encode())
            if name in ("meta", "brain"):
                meta = json.loads(str(value))
                meta.pop("version", None)
                h.update(json.dumps(meta, sort_keys=True).encode())
                continue
            h.update(str(value.dtype).encode() + str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


# -- short lives; ``give`` turns a row of the brain's whole width into what the caller passes


def whole(x: np.ndarray) -> Any:
    return x


def frames(width: int, count: int, rows: int = 1, seed: int = 21) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.round(rng.random((rows, width)) * (rng.random((rows, width)) < 0.5), 3)
            for _ in range(count)]


def lived(brain: Brain, moments: list[np.ndarray], give: Any = whole) -> list[int]:
    """A life under ``live``: outcomes, an end, a waited moment and an imagined branch."""
    acts = []
    for t, x in enumerate(moments):
        reward = None if t == 0 else [float(t % 3 == 0)]
        done = None if t == 0 else [t == 9]
        acts.append(int(np.asarray(brain.live(give(x), reward=reward, done=done))[0]))
        if t == 5:
            brain.wait(give(moments[(t + 1) % len(moments)]))
    phases = brain.imagine([give(x) for x in moments[:3]])
    acts.extend(
        int(np.argmax(np.asarray(phase.state.activation)[0, brain.motor_index]))
        for phase in phases
    )
    return acts


def stepped(brain: Brain, moments: list[np.ndarray], give: Any = whole) -> list[int]:
    """A life under ``step`` on two streams: outcomes, teachers, an end, a lesson, a greedy
    act, a value and free answers."""
    actions = len(brain.motor_index)
    acts: list[int] = []
    for t, x in enumerate(moments):
        if t == 0:
            action = brain.step(give(x))
        elif t % 3 == 0:
            action = brain.step(give(x), reward=[float(t % 2), 0.5],
                                teacher=[t % actions, (t + 1) % actions])
        else:
            action = brain.step(give(x), reward=[float(t % 2) - 0.5, 0.0],
                                done=[t == 7, False])
        acts.extend(int(a) for a in np.asarray(action).ravel())
    acts.append(int(round(float(brain.value(give(moments[0]))[0]) * 1e6)))
    brain.act(give(moments[1]), greedy=True)
    brain.lesson(give(moments[2]), [0, 1])
    acts.extend(int(a) for a in brain.act(give(moments[3]), greedy=True))
    acts.extend(int(a) for a in brain.predict(give(moments[4])))
    return acts


# -- the brain of 0.99.1: recorded on that release's source, read here on this one


def plain_founder(tmp_path: Path) -> tuple[Brain, Any]:
    brain = Brain.compose(inputs=4, actions=2, arousal=True, seed=3, backend="cpu")
    return brain, lambda: lived(brain, frames(4, 14))


def coded_founder(tmp_path: Path) -> tuple[Brain, Any]:
    brain = Brain.compose(inputs=40, actions=3, code=64, seed=5, memory_fade=0.02,
                          maturity=True, backend="cpu")
    return brain, lambda: stepped(brain, frames(40, 10, rows=2))


def hosted_founder(tmp_path: Path) -> tuple[Brain, Any]:
    """A brain composed around a module file, as ``trained=[module]`` builds it. The donor's
    regions are saved as drawn, so the host's parameters are the same on every solver."""
    donor = Brain.compose(inputs=6, actions=4, modules=(16,), seed=1, recall=False,
                          backend="cpu")
    path = donor.save_module(
        tmp_path / "kinds.npz", name="kinds", inputs=["sensory"], published=["motor"],
        regions=["sensory", "association", "motor"])
    brain = Brain.compose(inputs=3, actions=2, seed=5, backend="cpu",
                          trained=[cd.load_module(path)])
    return brain, lambda: stepped(brain, frames(9, 10, rows=2))


def mapped_founder(tmp_path: Path) -> tuple[Brain, Any]:
    """The long form of a brain with a map: inputs sized by hand, the map assigned, its read
    passed as the observation."""
    held = Map(PLANE, cells=(8, 6), units=4)
    brain = Brain.compose(inputs=held.count + held.units, actions=2, arousal=True, seed=2,
                          backend="cpu")
    brain.space = held

    def life() -> list[int]:
        acts = []
        brain.space.moment(seen=[(APPLE, [0.3, 0.6]), (BREAD, [0.6, 1.0])], lit=ahead)
        for t in range(10):
            brain.space.moment(lit=behind if t % 4 else ahead)
            reward = None if t == 0 else [float(t % 3 == 0)]
            acts.append(int(brain.live(brain.space.observation()[None], reward=reward)[0]))
        return acts

    return brain, life


# Recorded on the source of release 0.99.1 (tag v0.99.1): each founder's connectome and
# parameters and its description before its life, the acts of the life, and the contents of
# every array of the checkpoint it then saves.
RELEASED: dict[str, dict[str, Any]] = {
    "plain": {
        "build": plain_founder,
        "parameters": "c02f20030c8ca949fb90cea30841c7a1226a8f40d65d2dccb7fe17d1e93d2688",
        "described": "2fca9b42b6268f6aec0ad33888e16ad03e9f1d527ceff3e7b8814755c64e147d",
        "acts": [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        "checkpoint": "4b196efd1420e68be6bd3fd5adb8e05702155b5a7088ecec5980ef4c598d413c",
    },
    "coded": {
        "build": coded_founder,
        "parameters": "83feb7b83049dd26b5dd3cefc2fa5ccaed121ed5995e46d517e5dedeaffd4489",
        "described": "50a9fc4270457be04e9f686392ceca1f294db3cc80532051d923d678de594700",
        "acts": [2, 2, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0, 162682, 0, 0, 2,
                 0],
        "checkpoint": "602d8fd86efe775758ff4278565271457a12ef1267f85c267d53bef005a4d048",
    },
    "hosted": {
        "build": hosted_founder,
        "parameters": "5965eefd21f544df7104a003d9cc38ce56f94360c8a83998877c6af4ca566ceb",
        "described": "5636a4f0b7b656b467df473e8ad01cb91a3e6b2f9a2280dc5b1eb514d7280dbc",
        "acts": [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 175428, 0, 0, 0,
                 0],
        "checkpoint": "1f7d22cfaa7af09de75568fbfa5f4e5a4b075fa613106ab4e4e768931d906e93",
    },
    "mapped": {
        "build": mapped_founder,
        "parameters": "728e8341e03e907d641475129526dab575a1b4a1f261c0e0ca7ccf6c229e538e",
        "described": "0807b9754722e3264887f02d798a4ae5aad2a54fb909c689ad5126d288c829a8",
        "acts": [0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
        "checkpoint": "7974db3bbd379417e8e62f8c8cc60f7f696519d99d6fe786f9504d5852ed0075",
    },
}


def released_record(tmp_path: Path) -> dict[str, dict[str, Any]]:
    """What the two tests below read, as this source gives it: run on the source of a release
    to record it."""
    out = {}
    for name, spec in RELEASED.items():
        brain, life = spec["build"](tmp_path)
        record = {"parameters": parameter_digest(brain), "described": description_digest(brain)}
        record["acts"] = life()
        record["checkpoint"] = checkpoint_digest(brain.save(tmp_path / f"{name}.npz"))
        out[name] = record
    return out


@pytest.mark.parametrize("name", sorted(RELEASED))
def test_a_brain_without_named_senses_is_composed_as_on_the_release(tmp_path, name):
    """On every backend: the connectome, the parameters and the description; no senses are
    named, described or saved."""
    spec = RELEASED[name]
    brain, life = spec["build"](tmp_path)
    assert parameter_digest(brain) == spec["parameters"]
    assert description_digest(brain) == spec["described"]
    assert brain.senses == {} and "senses" not in brain.describe()["layout"]
    life()
    with np.load(brain.save(tmp_path / "founder.npz"), allow_pickle=False) as data:
        assert "senses" not in json.loads(str(data["brain"]))


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
@pytest.mark.parametrize("name", sorted(RELEASED))
def test_a_brain_without_named_senses_lives_and_saves_as_on_the_release(tmp_path, name):
    spec = RELEASED[name]
    brain, life = spec["build"](tmp_path)
    assert life() == spec["acts"]
    assert checkpoint_digest(brain.save(tmp_path / "founder.npz")) == spec["checkpoint"]


# -- a brain with named senses and no code is the brain of their summed width

SENSES = {"touch": 4, "eyes": {"hue": 3, "shape": 5}}          # 12 inputs in the declared order


def named(x: np.ndarray) -> Any:
    return {"touch": x[:, :4], "eyes": x[:, 4:]}


def by_field(x: np.ndarray) -> Any:
    return {"touch": x[:, :4], "eyes": {"hue": x[:, 4:7], "shape": x[:, 7:]}}


IDENTITY: dict[str, tuple[dict[str, Any], Any, int]] = {
    "default": ({"arousal": True, "seed": 7}, lived, 1),
    "deep, fading, mature": (
        {"seed": 11, "modules": (16, 16), "memory_fade": 0.02, "maturity": True}, stepped, 2),
    "episodic with rehearsal": (
        {"seed": 5, "memory_choice": True, "learning_rehearsal": 1.0}, stepped, 2),
    "learned code": ({"seed": 3, "learned_code": 16, "arousal": True}, lived, 1),
}


def members(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """A checkpoint's arrays and the brain's metadata."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name != "brain"}
        life = json.loads(str(data["brain"]))
    return arrays, life


def same_arrays(mine: dict[str, np.ndarray], theirs: dict[str, np.ndarray]) -> None:
    assert sorted(mine) == sorted(theirs)
    for name, value in mine.items():
        assert value.dtype == theirs[name].dtype and value.shape == theirs[name].shape, name
        assert np.array_equal(value, theirs[name]), name


@pytest.mark.parametrize("give", [named, by_field, whole], ids=["dict", "fields", "rows"])
@pytest.mark.parametrize("name", sorted(IDENTITY))
def test_a_brain_with_named_senses_is_the_brain_of_their_summed_width(tmp_path, name, give):
    options, life, rows = IDENTITY[name]
    plain = Brain.compose(inputs=12, actions=3, backend="cpu", **options)
    sensed = Brain.compose(senses=SENSES, actions=3, backend="cpu", **options)
    assert sensed.inputs == plain.inputs == 12
    assert parameter_digest(sensed) == parameter_digest(plain)       # the weights as drawn
    described = sensed.describe()
    assert described["layout"].pop("senses") == {
        "touch": [0, 4], "eyes": {"hue": [4, 7], "shape": [7, 12]}}
    assert described == plain.describe()
    moments = frames(12, 12, rows=rows)
    assert life(sensed, moments, give) == life(plain, moments)       # the acts of a short life
    assert parameter_digest(sensed) == parameter_digest(plain)       # the weights as taught
    mine, record = members(sensed.save(tmp_path / "sensed.npz"))
    theirs, plain_record = members(plain.save(tmp_path / "plain.npz"))
    same_arrays(mine, theirs)                                        # every saved array
    assert record.pop("senses")["named"] == {
        "touch": [0, 4], "eyes": {"hue": [4, 7], "shape": [7, 12]}}
    assert record == plain_record                                    # apart from the names


# -- the contract: senses declared once, observations by name


def same(mine: Any, theirs: Any) -> bool:
    """Whether two results hold the same values: arrays, numbers, and lists, tuples and
    dicts of them."""
    if isinstance(mine, dict):
        return mine.keys() == theirs.keys() and all(same(mine[k], theirs[k]) for k in mine)
    if isinstance(mine, (list, tuple)):
        return len(mine) == len(theirs) and all(
            same(a, b) for a, b in zip(mine, theirs, strict=True))
    return bool(np.array_equal(np.asarray(mine), np.asarray(theirs)))


def test_senses_are_declared_once_by_name_and_named_in_the_layout():
    brain = Brain.compose(senses=SENSES, actions=2, arousal=True, seed=0, backend="cpu")
    assert brain.inputs == 12
    assert brain.senses == {
        "touch": slice(0, 4), "eyes": {"hue": slice(4, 7), "shape": slice(7, 12)}}
    assert list(brain.senses) == ["touch", "eyes"]                # the declared order
    layout = brain.describe()["layout"]
    assert layout["inputs"] == 12
    assert layout["senses"] == {"touch": [0, 4], "eyes": {"hue": [4, 7], "shape": [7, 12]}}
    # the sensory region is one population: the names are held beside it
    assert len(brain.connectome.populations["sensory"]) == 12
    assert not any(name.startswith("sensory/") for name in brain.connectome.populations)
    x = frames(12, 1)[0]
    assert brain.live(named(x)).shape == (1,)
    row = x[:, brain.senses["eyes"]["shape"]]                      # a slice reads its columns
    assert np.array_equal(row, x[:, 7:12])
    assert json.loads(json.dumps(brain.describe())) == brain.describe()


def test_a_sense_left_out_is_silent():
    brain = Brain.compose(senses=SENSES, actions=2, seed=0, backend="cpu")
    x = frames(12, 1, rows=2)[0]
    touched = np.concatenate([x[:, :4], np.zeros((2, 8))], axis=1)
    assert np.array_equal(brain.observation_code({"touch": x[:, :4]}), touched)
    hue = np.concatenate([np.zeros((2, 4)), x[:, 4:7], np.zeros((2, 5))], axis=1)
    assert np.array_equal(brain.observation_code({"eyes": {"hue": x[:, 4:7]}}), hue)
    assert np.array_equal(brain.observation_code({"eyes": {}, "touch": x[:, :4]}), touched)
    assert np.array_equal(brain.observation_code({}), np.zeros((1, 12)))   # one silent row
    assert np.array_equal(brain.observation_code(named(x)), x)
    lists = {"touch": x[:, :4].tolist(), "eyes": x[:, 4:].tolist()}        # rows as lists
    assert np.array_equal(brain.observation_code(lists), x)
    assert np.array_equal(brain.observation_code(x), x)            # rows of the summed width


X1, X2 = frames(12, 2, rows=2, seed=5)
R1, R2 = frames(12, 2, rows=1, seed=6)
X8 = np.concatenate(frames(12, 4, rows=2, seed=7))
OPERATIONS: dict[str, Any] = {
    "act": lambda b, o: b.act(o(X1), greedy=True),
    "step": lambda b, o: [b.step(o(X1)), b.step(o(X2), reward=[1.0, 0.0], teacher=[0, 1])],
    "lesson": lambda b, o: [b.lesson(o(X1), [0, 1]), b.act(o(X2), greedy=True)],
    "learn": lambda b, o: [
        b.step(o(X1)), b.learn(np.array([1.0, 0.0]), np.array([False, True]), o(X2))],
    "value": lambda b, o: [b.act(o(X1)), b.value(o(X2))],
    "imagine": lambda b, o: [
        np.asarray(phase.state.activation) for phase in b.imagine([o(X1), o(X2)])],
    "predict": lambda b, o: b.predict(o(X1)),
    "accuracy": lambda b, o: b.accuracy(o(X1), [0, 1]),
    "fit": lambda b, o: b.fit(o(X8), [0, 1, 2, 0, 1, 2, 0, 1], epochs=2, batch=4),
    "observation_code": lambda b, o: b.observation_code(o(X1)),
    "stimulus": lambda b, o: [b.stimulus(o(X1)), b.stimulus(o(X2), memory=False)],
    "live": lambda b, o: [b.live(o(R1)), b.live(o(R2), reward=[1.0])],
    "wait": lambda b, o: [b.live(o(R1)), b.wait(o(R2)), b.live(o(R1), reward=[1.0])],
}


@pytest.mark.parametrize("operation", sorted(OPERATIONS))
def test_every_operation_that_takes_observations_takes_them_by_name(operation):
    def brain() -> Brain:
        return Brain.compose(senses=SENSES, actions=3, seed=4, backend="cpu", arousal=True,
                             memory_fade=0.02)

    rows, by_name, by_fields = brain(), brain(), brain()
    told = OPERATIONS[operation](rows, whole)
    assert same(OPERATIONS[operation](by_name, named), told)
    assert same(OPERATIONS[operation](by_fields, by_field), told)
    assert parameter_digest(by_name) == parameter_digest(rows) == parameter_digest(by_fields)
    assert same(by_name.hippocampus.arrays(), rows.hippocampus.arrays())


def refused(call: Any, kind: type[Exception] = ValueError) -> str:
    """The sentence the library refuses a call with."""
    with pytest.raises(kind) as caught:
        call()
    return str(caught.value)


def test_observations_by_name_are_refused_by_the_name_that_is_wrong():
    brain = Brain.compose(senses=SENSES, actions=2, seed=0, backend="cpu")
    touch, eyes = np.ones((1, 4)), np.ones((1, 8))
    before = brain.rng.bit_generator.state
    assert "no sense 'smell': its senses are 'touch', 'eyes'" in refused(
        lambda: brain.act({"smell": touch}))
    assert "sense 'touch' takes 4 values per row, got 5" in refused(
        lambda: brain.act({"touch": np.ones((1, 5))}))
    assert "different numbers of rows: 'touch' 1, 'eyes' 2" in refused(
        lambda: brain.act({"touch": touch, "eyes": np.ones((2, 8))}))
    assert "sense 'eyes' has no field 'size': its fields are 'hue', 'shape'" in refused(
        lambda: brain.act({"eyes": {"size": touch}}))
    assert "field 'hue' of sense 'eyes' takes 3 values per row, got 4" in refused(
        lambda: brain.act({"eyes": {"hue": touch}}))
    assert "the fields of sense 'eyes' give different numbers of rows" in refused(
        lambda: brain.act({"eyes": {"hue": np.ones((1, 3)), "shape": np.ones((2, 5))}}))
    assert "sense 'touch' has no fields: it takes rows of 4 values" in refused(
        lambda: brain.act({"touch": {"skin": touch}}))
    assert "sense 'touch' takes a nonempty finite batch of rows; use [row] for one" in refused(
        lambda: brain.act({"touch": np.ones(4)}))
    assert "nonempty finite batch" in refused(
        lambda: brain.act({"touch": np.full((1, 4), np.nan)}))
    assert "sense 'eyes' takes rows of 8 numbers" in refused(
        lambda: brain.act({"eyes": [["a"] * 8]}))
    assert "observations need 12 values per row, got 4" in refused(lambda: brain.act(touch))
    # a refused observation changes nothing: no stream began and no random number was drawn
    assert brain.basal_ganglia.state is None and brain.rng.bit_generator.state == before
    assert brain.act({"touch": touch, "eyes": eyes}).shape == (1,)
    plain = Brain.compose(inputs=12, actions=2, seed=0, backend="cpu")
    assert plain.senses == {}
    with pytest.raises((TypeError, ValueError)):                  # as on 0.99.1: no names
        plain.act({"touch": touch})


def test_senses_are_refused_where_they_are_declared_wrongly():
    def compose(senses: Any, **options: Any) -> Any:
        return lambda: Brain.compose(senses=senses, actions=2, backend="cpu", **options)

    assert "pass one of the two" in refused(
        lambda: Brain.compose(inputs=4, senses={"touch": 4}, actions=2))
    assert "pass one of the two" in refused(lambda: Brain.compose(4, 2, senses={"touch": 4}))
    assert "senses maps each sense's name to its width" in refused(compose({}))
    assert "senses maps each sense's name to its width" in refused(compose([("touch", 4)]))
    assert "the width of sense 'touch' must be an integer of at least 1" in refused(
        compose({"touch": 0}))
    assert "the width of sense 'touch' must be an integer of at least 1" in refused(
        compose({"touch": 4.0}))
    assert "the width of sense 'ear' must be an integer" in refused(compose({"ear": object()}))
    assert "the width of field 'hue' of sense 'eyes' must be an integer" in refused(
        compose({"eyes": {"hue": True}}))
    assert "sense 'eyes' declares no field" in refused(compose({"eyes": {}}))
    assert "a sense's name is a nonempty string" in refused(compose({"": 4}))
    assert "a sense's name is a nonempty string" in refused(compose({3: 4}))
    assert "a field's name is a nonempty string" in refused(compose({"eyes": {"": 4}}))
    assert "'space' names the sense a map's read joins as" in refused(compose({"space": 4}))
    assert "learned_fields reads an image" in refused(compose({"touch": 4}, learned_fields=8))
    assert "ports reads an image" in refused(compose({"touch": 4}, ports=2))
    assert "needs inputs (or senses=) and actions" in refused(
        lambda: Brain.compose(senses={"touch": 4}), TypeError)
    assert "needs inputs (or senses=) and actions" in refused(
        lambda: Brain.compose(actions=2), TypeError)
    assert "needs inputs (or senses=) and actions" in refused(lambda: Brain.compose(4), TypeError)
    assert "senses by name (senses=) belong to a lone brain" in refused(
        lambda: cd.Lives(4, 2, 2, senses={"touch": 4}))


def test_a_saved_brain_keeps_its_senses_and_continues_the_same_life(tmp_path):
    brain = Brain.compose(senses=SENSES, actions=3, arousal=True, seed=9, backend="cpu")
    moments = frames(12, 8)
    for t, x in enumerate(moments[:4]):
        brain.live(named(x), reward=None if t == 0 else [float(t % 2)])
    assert brain.pending_feedback                                 # an action awaits its outcome
    again = Brain.load(brain.save(tmp_path / "senses.npz"))
    assert again.senses == brain.senses and list(again.senses) == ["touch", "eyes"]
    assert list(again.senses["eyes"]) == ["hue", "shape"]
    assert again.describe()["layout"]["senses"] == brain.describe()["layout"]["senses"]
    for t, x in enumerate(moments[4:]):                           # the same next moments
        reward = [float(t % 2)]
        assert np.array_equal(again.live(by_field(x), reward=reward),
                              brain.live(named(x), reward=reward))
    mine, record = members(again.save(tmp_path / "again.npz"))
    theirs, original = members(brain.save(tmp_path / "brain.npz"))
    same_arrays(mine, theirs)
    assert record == original and record["senses"]["declared"] is True


def test_a_copy_and_a_grown_brain_keep_the_senses():
    import copy

    brain = Brain.compose(senses=SENSES, actions=3, seed=2, backend="cpu")
    x = frames(12, 1)[0]
    twin = copy.deepcopy(brain)
    assert twin.senses == brain.senses
    assert np.array_equal(twin.act(named(x), greedy=True), brain.act(x, greedy=True))
    grown = brain.grow(efference_amplitude=1.0)
    assert grown.senses == brain.senses and grown.efference is not None
    assert np.array_equal(grown.act(by_field(x), greedy=True), brain.act(x, greedy=True))
    assert brain.retune(teaching="normalized").senses == twin.senses


def test_a_checkpoint_with_senses_that_do_not_fit_is_refused(tmp_path):
    brain = Brain.compose(senses=SENSES, actions=2, seed=0, backend="cpu")
    path = brain.save(tmp_path / "senses.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    record = json.loads(str(arrays["brain"]))

    def changed(**senses: Any) -> Path:
        out = tmp_path / "changed.npz"
        meta = {**record, "senses": {**record["senses"], **senses}}
        np.savez(out, **{**arrays, "brain": np.array(json.dumps(meta))})
        return out

    assert Brain.load(changed()).senses == brain.senses
    for wrong in (
        {"named": {"touch": [0, 4], "eyes": [4, 11]}},             # short of the inputs
        {"named": {"touch": [0, 4], "eyes": [5, 12]}},             # a gap between two senses
        {"named": {"touch": [1, 4], "eyes": [4, 12]}},             # declared, and not from 0
        {"named": {"touch": [0, 4], "eyes": [4, 13]}},             # past the inputs
        {"named": {}}, {"named": {"touch": [0, 12.0]}}, {"declared": 1}, {"space": -1},
    ):
        assert "invalid saved senses" in refused(lambda wrong=wrong: Brain.load(changed(**wrong)))
    named_map = {"touch": [0, 4], "space": {"where": [4, 10], "what": [10, 12]}}
    assert "read a map of things the checkpoint does not hold" in refused(
        lambda: Brain.load(changed(named=named_map, declared=True, space=8)))


# -- equal senses: every field at unit length in the fixed sparse code


def expanded(brain: Brain, row: np.ndarray) -> np.ndarray:
    """The fixed sparse code of one row, computed from the code's own tables: each unit sums
    its channels (ON ``max(x, 0)``, then OFF ``max(-x, 0)``) by its weights, the largest
    positive sums are kept and the row is scaled to unit length."""
    code = brain.sparse_code
    channels = np.concatenate([np.maximum(row, 0.0), np.maximum(-row, 0.0)])
    drive = np.array([
        max(sum(float(w) * channels[k] for k, w in zip(code.index[unit], code.weight[unit],
                                                        strict=True)), 0.0)
        for unit in range(code.expansion)
    ])
    kept = np.zeros_like(drive)
    winners = np.argsort(-drive, kind="stable")[: code.winners]
    kept[winners] = drive[winners]
    return np.asarray(kept / np.linalg.norm(kept))


def test_every_sense_field_enters_the_fixed_code_at_unit_length():
    equal = Brain.compose(senses={"glimpse": 40, "word": 24}, actions=3, code=64, seed=0,
                          backend="cpu")
    control = Brain.compose(inputs=64, actions=3, code=64, seed=0, backend="cpu")
    assert np.array_equal(equal.sparse_code.index, control.sparse_code.index)
    assert np.array_equal(equal.sparse_code.weight, control.sparse_code.weight)
    assert parameter_digest(equal) == parameter_digest(control)  # one brain, another input
    glimpse, word = np.zeros(40), np.zeros(24)
    glimpse[[1, 5, 30]] = [3.0, 4.0, 0.0]                         # length 5
    word[[2, 9]] = [0.6, -0.8]                                    # length 1, one value negative
    row = np.concatenate([glimpse, word])
    by_hand = expanded(equal, np.concatenate([glimpse / 5.0, word / 1.0]))
    code = equal.observation_code({"glimpse": [glimpse], "word": [word]})
    assert code.shape == (1, 64) and np.isclose(np.linalg.norm(code), 1.0)
    assert np.allclose(code[0], by_hand, atol=1e-12)
    assert np.allclose(equal.observation_code(row[None])[0], by_hand, atol=1e-12)
    # a field's size does not move the code: only what it holds does
    louder = equal.observation_code({"glimpse": [7.0 * glimpse], "word": [0.01 * word]})
    assert np.allclose(louder, code, atol=1e-12)
    # a brain without declared senses keeps its code: the row as it is, to the bit
    rows = row[None]
    assert control._equal_senses(rows) is rows and equal._equal_senses(rows) is not rows
    plain = control.observation_code(rows)
    assert np.allclose(plain[0], expanded(control, row), atol=1e-12)
    assert not np.allclose(plain, code)
    assert not np.allclose(control.observation_code(np.concatenate([7.0 * glimpse, word])[None]),
                           plain)
    # a field with nothing in it stays zero
    seen = equal.observation_code({"glimpse": [glimpse]})
    assert np.allclose(seen[0], expanded(equal, np.concatenate([glimpse / 5.0, 0.0 * word])),
                       atol=1e-12)
    assert not equal.observation_code({}).any()
    # the cortex reads the same code: the sensory region is the code
    drive = equal.stimulus({"glimpse": [glimpse], "word": [word]}, memory=False)
    amplitude = equal.brain.neuron_model.stimulus_amplitude
    assert np.allclose(drive[:, equal.sensory_index], code * amplitude, atol=1e-12)


def test_a_small_sense_moves_the_code_beside_a_large_one():
    equal = Brain.compose(senses={"glimpse": 40, "word": 24}, actions=3, code=64, seed=0,
                          backend="cpu")
    control = Brain.compose(inputs=64, actions=3, code=64, seed=0, backend="cpu")
    rng = np.random.default_rng(0)
    moved = {"equal": [], "control": []}
    for _ in range(40):
        glimpse = np.zeros(40)
        glimpse[rng.choice(40, 26, replace=False)] = 1.0          # length 5.1
        first, second = rng.choice(24, 2, replace=False)
        rows = np.zeros((2, 64))
        rows[:, :40] = glimpse
        rows[0, 40 + first] = rows[1, 40 + second] = 1.0          # two words, each of length 1
        for name, brain in (("equal", equal), ("control", control)):
            a, b = brain.observation_code(rows)
            moved[name].append(int(((a > 0) != (b > 0)).sum()))
    # of 24 active units, the word changes about 17 where the senses are equal and about 5
    # where the glimpse enters at its own length
    assert np.mean(moved["equal"]) > 2 * np.mean(moved["control"])
    assert np.mean(moved["equal"]) > 12 and np.mean(moved["control"]) < 8


def test_fields_of_one_sense_are_each_at_unit_length(tmp_path):
    senses = {"eyes": {"hue": 12, "shape": 30}, "ear": 22}
    brain = Brain.compose(senses=senses, actions=2, code=64, seed=1, backend="cpu")
    rng = np.random.default_rng(1)
    hue, shape, ear = rng.random((1, 12)), 9.0 * rng.random((1, 30)), 0.1 * rng.random((1, 22))
    unit = np.concatenate([hue / np.linalg.norm(hue), shape / np.linalg.norm(shape),
                           ear / np.linalg.norm(ear)], axis=1)
    given = {"eyes": {"hue": hue, "shape": shape}, "ear": ear}
    code = brain.observation_code(given)
    assert np.allclose(code[0], expanded(brain, unit[0]), atol=1e-12)
    eyes = np.concatenate([hue, shape], axis=1)                   # the fields, however given
    assert np.array_equal(brain.observation_code({"eyes": eyes, "ear": ear}), code)
    assert np.allclose(brain.observation_code({"eyes": {"hue": 5.0 * hue, "shape": shape},
                                               "ear": 3.0 * ear}), code, atol=1e-12)
    loaded = Brain.load(brain.save(tmp_path / "coded.npz"))       # a saved brain keeps the rule
    assert np.array_equal(loaded.observation_code(given), code)


# -- trained= takes a name or a map: the map is brain.space and its read joins the senses

TOUCH = np.array([[0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0]])


def small_map() -> Map:
    return Map(PLANE, cells=(8, 6), units=4)                      # 48 cells and 4 code units


def test_a_brain_with_a_map_reads_it_itself_as_written():
    """The program of the guide: senses by name, the place module by name."""
    brain = Brain.compose(senses={"touch": 8}, actions=2, arousal=True, trained=["place"],
                          backend="cpu")
    assert isinstance(brain.space, Map) and brain.space.things() == []
    assert brain.senses == {
        "touch": slice(0, 8), "space": {"where": slice(8, 840), "what": slice(840, 856)}}
    assert brain.inputs == 8 + brain.space.count + brain.space.units == 856
    brain.space.moment(seen=[(APPLE, [0.3, 0.6])], lit=ahead)    # the eyes report; the map holds
    action = brain.live({"touch": TOUCH})                         # the brain reads its map itself
    assert action.shape == (1,)
    for _ in range(3):
        brain.space.moment(lit=behind)                            # the camera turns away
        brain.live({"touch": TOUCH}, reward=[1.0])
    assert np.allclose(brain.space.where_is(APPLE), [[0.3, 0.6]], atol=0.001)   # held unseen
    assert brain.describe()["layout"]["senses"]["space"] == {
        "where": [8, 840], "what": [840, 856]}


def test_a_thing_held_out_of_view_is_read_by_the_brains_input():
    held = small_map()
    brain = Brain.compose(senses={"touch": 8}, actions=2, arousal=True, seed=1, trained=[held],
                          backend="cpu")
    assert brain.space is held                                    # the map the caller holds
    where, what = brain.senses["space"]["where"], brain.senses["space"]["what"]
    assert (where, what) == (slice(8, 56), slice(56, 60)) and brain.inputs == 60
    # a brain without a code keys on its observation: the row it read, the map's part included
    empty = brain.observation_code({"touch": TOUCH})
    assert empty.shape == (1, 60) and np.array_equal(empty[:, :8], TOUCH)
    assert not empty[:, 8:].any()                                 # the map holds no thing
    held.moment(seen=[(APPLE, [0.3, 0.6])], lit=ahead)
    for _ in range(20):
        held.moment(lit=behind)                                   # the camera looks elsewhere
    assert not behind(np.array([[0.3, 0.6]]))[0]                  # the apple is out of view
    row = brain.observation_code({"touch": TOUCH})
    assert np.array_equal(row[0, 8:], held.observation())
    assert row[0, what][APPLE] > 0 and row[0, what][BREAD] == 0   # what is held
    cell = held.centers[np.argmax(row[0, where])]                 # and where: the apple's cell
    assert np.all(np.abs(cell - [0.3, 0.6]) <= held.size)
    drive = brain.stimulus({"touch": TOUCH}, memory=False)        # what the sensory neurons get
    amplitude = brain.brain.neuron_model.stimulus_amplitude
    assert np.allclose(drive[:, brain.sensory_index], row * amplitude)
    before = brain.live({"touch": TOUCH})
    held.moment(seen=[(BREAD, [-0.6, 1.0])], lit=behind)          # every moment reads the map
    after = brain.observation_code({"touch": TOUCH})
    assert after[0, what][BREAD] > 0 and not np.array_equal(after, row)
    assert before.shape == brain.live({"touch": TOUCH}, reward=[0.0]).shape == (1,)
    # rows of the senses' width are taken as the dict is; whole rows as they are
    assert np.array_equal(brain.observation_code(TOUCH), after)
    what_if = np.concatenate([TOUCH, np.zeros((1, 52))], axis=1)
    assert np.array_equal(brain.observation_code(what_if), what_if)
    both = brain.observation_code({"touch": np.concatenate([TOUCH, 2 * TOUCH])})
    assert both.shape == (2, 60) and np.array_equal(both[0, 8:], both[1, 8:])   # every row


def test_the_short_form_is_the_long_form_of_a_brain_with_a_map():
    """``senses=`` and a map in ``trained=`` compose the brain a caller composed by hand on
    0.99.1: inputs sized as the sum, the map assigned, its read passed with the observation."""
    mine, theirs = small_map(), small_map()
    short = Brain.compose(senses={"touch": 8}, actions=2, arousal=True, seed=2, trained=[mine],
                          backend="cpu")
    long = Brain.compose(inputs=8 + theirs.count + theirs.units, actions=2, arousal=True,
                         seed=2, backend="cpu")
    long.space = theirs
    assert parameter_digest(short) == parameter_digest(long)
    rng = np.random.default_rng(3)
    for t in range(10):
        touch = np.round(rng.random((1, 8)), 3)
        for held in (mine, theirs):
            if t == 0:
                held.moment(seen=[(APPLE, [0.3, 0.6]), (BREAD, [0.6, 1.0])], lit=ahead)
            else:
                held.moment(lit=behind if t % 4 else ahead)
        reward = None if t == 0 else [float(t % 3 == 0)]
        row = np.concatenate([touch, theirs.observation()[None]], axis=1)
        assert np.array_equal(short.live({"touch": touch}, reward=reward),
                              long.live(row, reward=reward))
    assert parameter_digest(short) == parameter_digest(long)
    assert np.array_equal(short.basal_ganglia.state.activation, long.basal_ganglia.state.activation)


def test_trained_takes_a_name_a_map_and_a_module_in_any_form():
    by_name = Brain.compose(inputs=4, actions=2, seed=0, trained="place", backend="cpu")
    assert isinstance(by_name.space, Map) and by_name.space.cells == (32, 26)
    assert by_name.senses == {"space": {"where": slice(4, 836), "what": slice(836, 852)}}
    held = small_map()
    for trained in (held, [held], (held,)):
        brain = Brain.compose(inputs=4, actions=2, seed=0, trained=trained, backend="cpu")
        assert brain.space is held and brain.inputs == 4 + 52
    # a brain composed with inputs= takes rows: its own inputs, or the whole row
    own = np.ones((1, 4))
    held.moment(seen=[(APPLE, [0.3, 0.6])])
    row = brain.observation_code(own)
    assert np.array_equal(row, np.concatenate([own, held.observation()[None]], axis=1))
    assert np.array_equal(brain.observation_code(row), row)
    assert "declares no senses by name" in refused(lambda: brain.act({"touch": own}))
    assert "observations need 4 values per row, got 5 (the brain reads the 52 of 'space'" in (
        refused(lambda: brain.act(np.ones((1, 5)))))
    # with code= such a brain codes the row as it is: equal senses is the rule of senses=
    coded = Brain.compose(inputs=40, actions=2, seed=0, code=64, trained=[small_map()],
                          backend="cpu")
    rows = np.ones((1, 92))
    assert coded._equal_senses(rows) is rows


def test_a_map_in_trained_is_refused_where_it_does_not_fit():
    held = small_map()
    assert "a brain holds one map of things (brain.space): trained= names 2" in refused(
        lambda: Brain.compose(inputs=4, actions=2, trained=[held, small_map()]))
    assert "ships no module 'kitchen'" in refused(
        lambda: Brain.compose(inputs=4, actions=2, trained=["kitchen"]))
    assert "inputs is a count (or senses=), not an image shape" in refused(
        lambda: Brain.compose(inputs=(8, 8), actions=2, trained=[held]))
    assert "inputs must be an integer of at least 1" in refused(
        lambda: Brain.compose(inputs=0, actions=2, trained=[held]))
    assert "learned_fields reads an image" in refused(
        lambda: Brain.compose(inputs=(8, 8), actions=2, trained=[held], learned_fields=4))
    assert "maps of things (cadence.space.map.Map) and names of modules" in refused(
        lambda: Brain.compose(inputs=4, actions=2, trained=[object()]))
    assert "trained module or a map" in refused(lambda: cd.Lives(4, 2, 2, trained=["place"]))
    assert "trained module or a map" in refused(lambda: cd.Lives(4, 2, 2, trained=[held]))
    brain = Brain.compose(senses={"touch": 8}, actions=2, seed=0, trained=[held], backend="cpu")
    assert "reads its sense 'space' from its map itself" in refused(
        lambda: brain.act({"touch": TOUCH, "space": np.zeros((1, 52))}))
    brain.space = None                                            # the map taken away
    assert "reads its sense 'space', 52 values, from brain.space, which holds no map" in (
        refused(lambda: brain.act({"touch": TOUCH})))
    brain.space = Map(PLANE, cells=(4, 4), units=4)               # a map of another size
    assert "from brain.space, whose map reads 20" in refused(lambda: brain.act({"touch": TOUCH}))
    brain.space = small_map()                                     # a map that fits is read
    assert brain.act({"touch": TOUCH}).shape == (1,)


def kinds_module(tmp_path: Path) -> Any:
    donor = Brain.compose(inputs=6, actions=4, modules=(16,), seed=1, recall=False,
                          backend="cpu")
    return cd.load_module(donor.save_module(
        tmp_path / "kinds.npz", name="kinds", inputs=["sensory"], published=["motor"],
        regions=["sensory", "association", "motor"]))


def test_a_trained_module_stands_beside_named_senses_and_a_map(tmp_path):
    module = kinds_module(tmp_path)
    rows = frames(9, 6, rows=2)
    # with senses= the brain is the brain of 0.99.1's trained=[module], and the module's
    # inputs are a sense under the module's name
    plain = Brain.compose(inputs=3, actions=2, seed=5, backend="cpu", trained=[module])
    sensed = Brain.compose(senses={"own": 3}, actions=2, seed=5, backend="cpu",
                           trained=[module])
    assert plain.senses == {} and parameter_digest(sensed) == parameter_digest(plain)
    assert sensed.senses == {"own": slice(0, 3), "kinds": slice(3, 9)}
    assert module.columns(sensed).tolist() == [3, 4, 5, 6, 7, 8]
    assert stepped(sensed, rows, lambda x: {"own": x[:, :3], "kinds": x[:, 3:]}) == stepped(
        plain, rows)
    # with a map as well: the brain's own senses, the map's read, then the module's inputs
    held = small_map()
    host = Brain.compose(senses={"own": 3}, actions=2, seed=5, backend="cpu",
                         trained=[module, held])
    assert host.space is held and host.inputs == 3 + 52 + 6
    assert host.senses == {
        "own": slice(0, 3), "space": {"where": slice(3, 51), "what": slice(51, 55)},
        "kinds": slice(55, 61)}
    assert module.columns(host).tolist() == [55, 56, 57, 58, 59, 60]   # of the whole row
    held.moment(seen=[(APPLE, [0.3, 0.6])])
    given = {"own": rows[0][:, :3], "kinds": rows[0][:, 3:]}
    whole_rows = np.concatenate(
        [rows[0][:, :3], np.tile(held.observation(), (2, 1)), rows[0][:, 3:]], axis=1)
    assert np.array_equal(host.observation_code(given), whole_rows)
    assert np.array_equal(host.observation_code(rows[0]), whole_rows)   # rows without the map
    assert np.array_equal(host.observation_code(whole_rows), whole_rows)
    drive = host.stimulus(given, memory=False)
    amplitude = host.brain.neuron_model.stimulus_amplitude
    populations = host.connectome.populations
    assert np.allclose(drive[:, populations["kinds/sensory"]], rows[0][:, 3:] * amplitude)
    own = np.asarray(populations["sensory"])                      # ascending: own, map, module
    assert np.allclose(drive[:, own[:3]], rows[0][:, :3] * amplitude)
    assert np.allclose(drive[:, own[3:55]], np.tile(held.observation(), (2, 1)) * amplitude)
    assert np.array_equal(host.act(given, greedy=True), host.act(rows[0], greedy=True))
    again = Brain.load(host.save(tmp_path / "host.npz"))
    assert again.senses == host.senses and list(again.senses) == ["own", "space", "kinds"]
    assert "a sense and the module 'kinds' share one name" in refused(
        lambda: Brain.compose(senses={"kinds": 3}, actions=2, trained=[module]))


def test_a_trained_module_by_name_is_the_brain_around_the_module(tmp_path, monkeypatch):
    module = kinds_module(tmp_path)
    monkeypatch.setattr(cd.modules, "load", lambda name: module)  # as a module the library ships
    by_name = Brain.compose(inputs=3, actions=2, seed=5, backend="cpu", trained=["kinds"])
    by_hand = Brain.compose(inputs=3, actions=2, seed=5, backend="cpu", trained=[module])
    assert parameter_digest(by_name) == parameter_digest(by_hand)
    assert by_name.senses == {} and by_name.describe() == by_hand.describe()
    rows = frames(9, 6, rows=2)
    assert stepped(by_name, rows) == stepped(by_hand, rows)


def test_a_saved_brain_keeps_its_map_among_its_senses_and_continues(tmp_path):
    held = small_map()
    brain = Brain.compose(senses={"touch": 8}, actions=2, arousal=True, seed=4, trained=[held],
                          backend="cpu")
    held.moment(seen=[(APPLE, [0.3, 0.6]), (BREAD, [0.6, 1.0])], lit=ahead)
    for t in range(4):
        held.moment(lit=behind)
        brain.live({"touch": TOUCH}, reward=None if t == 0 else [float(t % 2)])
    assert brain.pending_feedback
    again = Brain.load(brain.save(tmp_path / "mapped.npz"))
    assert again.senses == brain.senses and list(again.senses) == ["touch", "space"]
    assert list(again.senses["space"]) == ["where", "what"]
    assert isinstance(again.space, Map) and again.space is not held
    assert np.array_equal(again.space.potential, held.potential)
    assert np.allclose(again.observation_code({"touch": TOUCH}),
                       brain.observation_code({"touch": TOUCH}))
    for t in range(6):                                            # the same next moments
        for space in (held, again.space):
            space.moment(lit=behind if t % 3 else ahead)
        reward = [float(t % 2)]
        assert np.array_equal(again.live({"touch": TOUCH}, reward=reward),
                              brain.live({"touch": TOUCH}, reward=reward))
    assert np.allclose(again.space.where_is(APPLE), held.where_is(APPLE), atol=1e-9)
    grown = brain.grow(efference_amplitude=1.0)                   # a grown brain keeps both
    assert grown.senses == brain.senses and isinstance(grown.space, Map)
    assert grown.act({"touch": TOUCH}, greedy=True).shape == (1,)


def test_named_senses_and_a_map_under_the_compiled_moment():
    mine, theirs = small_map(), small_map()
    short = Brain.compose(senses={"touch": 8}, actions=2, seed=6, trained=[mine], backend="cpu",
                          learning_compiled_moment=True)
    long = Brain.compose(inputs=60, actions=2, seed=6, backend="cpu",
                         learning_compiled_moment=True)
    rng = np.random.default_rng(6)
    for t in range(8):
        touch = np.round(rng.random((2, 8)), 3)
        for held in (mine, theirs):
            held.moment(seen=[(APPLE, [0.3, 0.6])] if t == 0 else (), lit=ahead)
        rows = np.concatenate([touch, np.tile(theirs.observation(), (2, 1))], axis=1)
        if t % 2:
            assert short.lesson({"touch": touch}, [0, 1]) == long.lesson(rows, [0, 1])
        assert np.array_equal(short.act({"touch": touch}, greedy=True),
                              long.act(rows, greedy=True))
    assert parameter_digest(short) == parameter_digest(long)


def test_named_senses_and_a_map_on_a_torch_device():
    pytest.importorskip("torch")
    mine, theirs = small_map(), small_map()
    place = {"backend": "torch", "device": "cpu", "precision": "float64"}
    short = Brain.compose(senses={"touch": 8}, actions=2, seed=6, trained=[mine], **place)
    long = Brain.compose(inputs=60, actions=2, seed=6, **place)
    rng = np.random.default_rng(6)
    for t in range(6):
        touch = np.round(rng.random((2, 8)), 3)
        for held in (mine, theirs):
            held.moment(seen=[(APPLE, [0.3, 0.6])] if t == 0 else (), lit=ahead)
        rows = np.concatenate([touch, np.tile(theirs.observation(), (2, 1))], axis=1)
        reward = None if t == 0 else [1.0, 0.0]
        assert np.array_equal(short.step({"touch": touch}, reward=reward),
                              long.step(rows, reward=reward))
    assert short.placement["backend"] == "torch" and short.space is mine


def test_a_brain_placed_at_its_first_moment_keeps_the_map_its_caller_holds(monkeypatch):
    pytest.importorskip("torch")
    from cadence.resident import _Candidate

    held = small_map()
    brain = Brain.compose(senses={"touch": 8}, actions=2, seed=6, trained=[held], backend="cpu")
    held.moment(seen=[(APPLE, [0.3, 0.6])], lit=ahead)
    monkeypatch.setattr(Brain, "_backends_at", lambda self, rows: [
        _Candidate("torch", "cpu", "float64", "named by the test"),
        _Candidate("cpu", None, None, "the next fastest")])
    brain._unplaced = True                                        # as a brain the rule places
    brain.act({"touch": TOUCH}, greedy=True)
    assert brain.placement["backend"] == "torch"
    assert brain.space is held and brain.senses["touch"] == slice(0, 8)
    held.moment(seen=[(BREAD, [-0.6, 1.0])], lit=behind)          # the caller's map is the one
    assert brain.observation_code({"touch": TOUCH})[0, brain.senses["space"]["what"]][BREAD] > 0
