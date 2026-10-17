"""Memories key on the observation code: the observation itself, or with ``code=units`` the
brain's own sparse code of it, which ``compose(code=...)`` gives the cortex too."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence import brain as cadence_brain
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, motor_cortex

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "observation_code"
# The acts and the array contents below were recorded on the NumPy solver; the compiled dense
# kernel (numba) orders the arithmetic otherwise and gives other bits, so those pins are read
# on the NumPy solver alone (CADENCE_FUSED=0 selects it). The connectome, the parameters and
# the checkpoint's members and metadata keys are read on every backend.
NUMPY_SOLVER = not cadence_brain._FUSED
BYTES_REASON = (
    "acts and array contents were recorded on the NumPy solver; this run uses the compiled "
    "dense kernel (CADENCE_FUSED=0 selects the NumPy loop)"
)
CUE = np.array([[1.0, 0.0, 0.5, 0.0]])
GO = np.array([[0.0, 0.0, 0.0, 1.0]])
FACT = np.array([[1.0, 0.0, 0.0, 1.0]])
NAME = np.array([[1.0, 0.0, 0.0, 0.0]])
STREAM = (CUE, GO, FACT, NAME)
MB = 1024 * 1024


# -- the founders: vector brains, byte-identical to the composition before the code


def checkpoint_digest(path: Path) -> str:
    """SHA-256 over every array of a checkpoint, by name, with the metadata JSON entries
    read back without the library version that wrote them."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            value = data[name]
            h.update(name.encode())
            if name in ("meta", "generic"):
                meta = json.loads(str(value))
                meta.pop("version", None)
                h.update(json.dumps(meta, sort_keys=True).encode())
                continue
            h.update(str(value.dtype).encode() + str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def names_digest(path: Path) -> str:
    """The checkpoint's metadata keys and member names, which no backend changes."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
        names = json.dumps([sorted(meta), sorted(data.files)])
    return hashlib.sha256(names.encode()).hexdigest()


def parameter_digest(brain: cd.Brain) -> str:
    connectome, graph = brain.connectome, brain.brain
    h = hashlib.sha256()
    for values in (connectome.pre, connectome.post, connectome.count, connectome.sign,
                   graph.efficacy, graph.bias, graph.log_gain):
        h.update(np.ascontiguousarray(np.asarray(values)).tobytes())
    populations = {k: [int(i) for i in v] for k, v in connectome.populations.items()}
    h.update(json.dumps(populations).encode())
    return h.hexdigest()


def stepped(brain: cd.Brain) -> list[int]:
    width = len(brain.sensory_index)
    rng = np.random.default_rng(21)
    acts = []
    for t in range(12):
        x = STREAM[t % 4] if width == 4 else (rng.random((1, width)) < 0.2).astype(float)
        if t == 0:
            action = brain.step(x)
        elif t % 3 == 0:
            action = brain.step(x, reward=[float(t % 2)], teacher=[t % 2])
        else:
            action = brain.step(x, reward=[float(t % 2) - 0.5], done=[t == 7])
        acts.append(int(np.asarray(action).ravel()[0]))
    acts.append(int(brain.act(x, greedy=True)[0]))
    return acts


def lived(brain: cd.Brain) -> list[int]:
    width = len(brain.sensory_index)
    rng = np.random.default_rng(22)
    frames = [STREAM[t % 4] if width == 4 else (rng.random((1, width)) < 0.2).astype(float)
              for t in range(14)]
    acts = []
    for t, x in enumerate(frames):
        reward = None if t == 0 else [float(t % 3 == 0)]
        done = None if t == 0 else [t == 9]
        acts.append(int(np.asarray(brain.live(x, reward=reward, done=done))[0]))
        if t == 5:
            brain.wait(frames[(t + 1) % 14])
    phases = brain.imagine(frames[:3])
    acts.extend(
        int(np.argmax(np.asarray(phase.state.activation)[0, brain.motor_index]))
        for phase in phases
    )
    return acts


# Recorded on the composition before the observation code: each founder's connectome and
# parameters, the acts of a short life, and the format, the member and metadata names
# (``names``) and the contents of every array (``checkpoint``) of the checkpoint it saves.
# The recall memory's checkpoint metadata changed since, so a brain with it has its
# checkpoint contents compared with those of the same life lived again instead.
FOUNDERS: dict[str, dict] = {
    "stepped": {
        "build": (4, 2, {"modules": (8,), "seed": 3, "recall": False}),
        "life": stepped,
        "parameters": "b81d75ef9bab5751c83c93bf64eb78e37b8eba039698948453ddc90b8bb4bee4",
        "acts": [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        "format": "cadence-generic/2",
        "names": "5a728760433a9dea87769077dd838133cd49a8637125759627166a31d5cf16a8",
        "checkpoint": "03067e0fea6cf3e6a70d853e26355f17469035d66c097f42f1286b411a4ea5aa",
    },
    "episodic": {
        "build": (4, 2, {"modules": (8,), "seed": 7, "memory_choice": True, "arousal": True,
                         "recall": False}),
        "life": lived,
        "parameters": "aad357f3afceae3cfd5528b2a12c02ee7405eedd5ada8922cd1785cf66189164",
        "acts": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "format": "cadence-generic/6",
        "names": "2d5620ee6ddfa07ee9bdfbb174903829313bc90cb9ff24c09e9e530f7e2dd4be",
        "checkpoint": "2a2bdcd2f204f66b80c2c67d1af5d1c84dad39220a13caefc2a08bdc42986ac0",
    },
    "recalled": {
        "build": (4, 2, {"modules": (8,), "seed": 5, "arousal": True, "recall": True,
                         "history_decays": (0.2, 0.9), "efference_amplitude": 1.0,
                         "memory_choice": True}),
        "life": lived,
        "parameters": "7ccd7c7855a971743e898c29204bc3050d63d6ce0cde809cdca7fd7cadca5c4f",
        "acts": [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        "format": "cadence-generic/8",
        "names": "17b2b7a61fa822e150659b765cef59befb6219db6791a720d7ff79b32498636c",
        "checkpoint": None,
    },
    "boundary": {
        # 256 inputs: a key width at which the dense separators are the recommended setting
        # as well; the recall memory keyed on the observation itself
        "build": (256, 3, {"modules": (16,), "seed": 9, "memory_choice": True, "recall": True,
                           "arousal": True}),
        "life": lived,
        "parameters": "938905bbc2dc39c354f2f53443948e0bc4b5f06dfe132306fa4d31062ada966a",
        "acts": [2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "format": "cadence-generic/8",
        "names": "d8e0435357330cc7f0deadff49bf70738665fd3a042b9d82969d7945a3094bc0",
        "checkpoint": None,
    },
    "wide": {
        # 2,000 inputs: wider than the recommended code's 1,024 units, keyed on the
        # observation itself with dense separators, as every width is unless asked
        "build": (2_000, 3, {"modules": (8,), "seed": 13, "memory_choice": True,
                             "arousal": True, "recall": False}),
        "life": lived,
        "parameters": "63de06ff41d1403e4e706fbf3095c48d555682191afc24b659c5e7e90d28acf3",
        "acts": [2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "format": "cadence-generic/6",
        "names": "2d5620ee6ddfa07ee9bdfbb174903829313bc90cb9ff24c09e9e530f7e2dd4be",
        "checkpoint": "7b8bf9c566606464d87941f321b4aac2421a7c07b882959ae3bc160512efa90e",
    },
    "image": {
        # an image shape through Brain.build: the hippocampus of earlier releases keyed on
        # the 1,600 pixels
        "build": ((40, 40), 2, {"hidden": 8, "seed": 14}),
        "builder": "build",
        "life": stepped,
        "parameters": "aecb598d8fbf5218c210bd2ad264206e9aae0e706105a6e1255a5c44316abe54",
        "acts": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        "format": "cadence-generic/2",
        "names": "40c128b7c0aa3d9ca90623833c68cb73986f93215449e1c55a380fdbb7996665",
        "checkpoint": "67e3b55da910940341f121de5333d74efc9acf135ae4db11e1e5d1e0802b91f8",
    },
}


def founder_record(tmp_path: Path) -> dict[str, dict]:
    """What the founder test checks, as recorded on the parent composition."""
    out = {}
    for name, spec in FOUNDERS.items():
        inputs, actions, options = spec["build"]
        builder = cd.Brain.build if spec.get("builder") == "build" else cd.Brain.compose
        brain = builder(inputs, actions, **options)
        parameters = parameter_digest(brain)
        acts = spec["life"](brain)
        path = brain.save(tmp_path / f"{name}.npz")
        out[name] = {
            "parameters": parameters, "acts": acts, "checkpoint": checkpoint_digest(path),
        }
    return out


def founder(spec: dict) -> tuple[cd.Brain, int]:
    inputs, actions, options = spec["build"]
    builder = cd.Brain.build if spec.get("builder") == "build" else cd.Brain.compose
    return builder(inputs, actions, **options), int(np.prod(inputs))


@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_a_vector_or_image_brain_is_the_composition_before_the_observation_code(tmp_path, name):
    """On every backend: the connectome and parameters, what the brain keys on, and the
    checkpoint's format, members and metadata keys after the life."""
    spec = FOUNDERS[name]
    brain, inputs = founder(spec)
    options = spec["build"][2]
    assert brain.sparse_code is None and brain.code_width == brain.inputs == inputs
    assert not brain.coded_cortex
    np.testing.assert_array_equal(brain.hippocampus.pre, brain.sensory_index)
    x = (np.random.default_rng(1).random((2, inputs)) < 0.3).astype(float)
    np.testing.assert_array_equal(brain.observation_code(x), x)
    assert parameter_digest(brain) == spec["parameters"]
    spec["life"](brain)
    path = brain.save(tmp_path / "founder.npz")
    assert names_digest(path) == spec["names"]
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
        assert not any(name.startswith("code/") for name in data.files)
    assert meta["format"] == spec["format"] and "code" not in meta
    described = brain.describe()["observation_code"]
    assert described == {
        "kind": "observation", "inputs": inputs, "width": inputs, "units": 0, "fan_in": 0,
        "winners": 0, "cortex": "observation",
        "separator": "dense" if options.get("memory_choice") else None,
    }
    assert brain.hippocampus.separator is None or brain.hippocampus.separator.fan_in == 0


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_the_founders_acts_and_checkpoint_bytes_are_the_recorded_ones(tmp_path, name):
    spec = FOUNDERS[name]
    brain, _ = founder(spec)
    assert spec["life"](brain) == spec["acts"]
    if spec["checkpoint"] is not None:
        assert checkpoint_digest(brain.save(tmp_path / "founder.npz")) == spec["checkpoint"]


def test_the_saved_recall_brain_loads_continues_and_saves_the_same(tmp_path):
    """The fixture, the recall founder after its life at the final format: it loads and saves
    again with the same arrays and metadata, continues, and the continued life's checkpoint
    reloads and saves the same; on the NumPy solver the continuation is the recorded one and
    the same life lived again saves the fixture's bytes."""
    fixture = FIXTURES / "recalled.npz"
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    brain = cd.Brain.load(fixture)
    assert brain.sparse_code is None and brain.recall is not None and brain.memory_choice
    assert checkpoint_digest(brain.save(tmp_path / "again.npz")) == checkpoint_digest(fixture)
    acts = [int(np.asarray(brain.live(STREAM[(t + 1) % 4], reward=[float(t % 2)]))[0])
            for t in range(6)]
    continued = brain.save(tmp_path / "continued.npz")
    assert checkpoint_digest(cd.Brain.load(continued).save(tmp_path / "reloaded.npz")) == (
        checkpoint_digest(continued)
    )
    if NUMPY_SOLVER:
        assert acts == provenance["continuation"]["recalled"]["acts"]
        inputs, actions, options = FOUNDERS["recalled"]["build"]
        relived = cd.Brain.compose(inputs, actions, **options)
        FOUNDERS["recalled"]["life"](relived)
        assert checkpoint_digest(relived.save(tmp_path / "relived.npz")) == (
            checkpoint_digest(fixture)
        )
    with np.load(fixture, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
    assert "seed" not in meta["recall"] and "address_prior" not in meta["recall"]
    meta["recall"]["address_prior"] = 1.0  # a retired gene is refused
    with np.load(fixture, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "prior.npz", **arrays)
    with pytest.raises(ValueError, match="invalid saved recall ports"):
        cd.Brain.load(tmp_path / "prior.npz")


# -- the brain's sparse code


def frames(width: int, rows: int, seed: int, density: float = 0.05) -> np.ndarray:
    return (np.random.default_rng(seed).random((rows, width)) < density).astype(float)


def test_the_code_and_the_sparse_separation_are_opt_in():
    def keyed(inputs, **options):
        brain = cd.Brain.compose(inputs, 2, modules=(8,), memory_choice=True, recall=True,
                                 **options)
        described = brain.describe()["observation_code"]
        return (described["kind"], described["width"], described["cortex"],
                described["separator"], brain.hippocampus.separator is not None
                and brain.hippocampus.separator.fan_in, len(brain.connectome.populations["recall"]))

    # the founder at every width: the observation itself, dense separators
    assert keyed(256) == ("observation", 256, "observation", "dense", 0, 256)
    assert keyed(1025) == ("observation", 1025, "observation", "dense", 0, 1025)
    assert keyed(1025, code=0) == ("observation", 1025, "observation", "dense", 0, 1025)
    # the recommended settings above 256 key values and above 1,024 inputs
    assert keyed(1025, memory_fan_in=32) == ("observation", 1025, "observation", "sparse", 32, 1025)
    assert keyed(1025, code=1024) == ("code", 1024, "code", None, False, 1024)
    assert keyed(300, code=64) == ("code", 64, "code", None, False, 64)
    assert cd.Brain.observation_width(12_824) == 12_824
    assert cd.Brain.observation_width(300) == 300 and cd.Brain.observation_width(300, 64) == 64
    assert cd.Brain.observation_width((40, 40)) == 1600 and cd.Brain.observation_width((8, 8), 256) == 256
    # a coded brain's memories key on the code as it is: a separation of their own is refused
    with pytest.raises(ValueError, match="memory_expansion and memory_fan_in need"):
        cd.Brain.compose(300, 2, modules=(8,), memory_choice=True, code=64, memory_expansion=512)
    with pytest.raises(ValueError, match="memory_expansion and memory_fan_in need"):
        cd.Brain.compose(300, 2, modules=(8,), memory_choice=True, code=64, memory_fan_in=8)
    with pytest.raises(ValueError, match="reads observations of at least 32 values"):
        cd.Brain.compose(20, 2, modules=(8,), code=16)
    with pytest.raises(ValueError, match="at least 24 units"):
        cd.Brain.compose(300, 2, modules=(8,), code=16)
    with pytest.raises(ValueError, match="code must be an integer"):
        cd.Brain.compose(300, 2, modules=(8,), code=1.5)
    with pytest.raises(ValueError, match="construction setting"):
        cd.Brain.compose(300, 2, modules=(8,)).retune(code=64)


def test_the_code_is_a_sparse_unit_length_code_fixed_by_the_seed():
    brain = cd.Brain.compose(12_824, 3, modules=(8,), seed=4, episodic=False, recall=False,
                             code=1024)
    code = brain.sparse_code
    assert isinstance(code, cd.PatternSeparator)
    # one unit in ten active, at least 24: 102 of 1,024; each unit reads 51 channels, as many
    # as it takes 1,024 units to read the ON and the OFF channel of every input twice
    assert (code.inputs, code.expansion, code.winners, code.fan_in) == (12_824, 1024, 102, 51)
    assert code.cover == 2 and (code.weight > 0).all() and code.channels == 2 * 12_824
    assert code.index.max() < code.channels
    assert np.bincount(code.index.ravel(), minlength=code.channels).min() >= 2
    assert cd.Brain.compose(300, 2, modules=(8,), code=128).sparse_code.winners == 24
    assert cd.Brain.compose(300, 2, modules=(8,), code=240).sparse_code.winners == 24
    assert cd.Brain.compose(300, 2, modules=(8,), code=250).sparse_code.winners == 25
    assert code.index.shape == code.weight.shape == (1024, 51) and code.projection is None
    x = frames(12_824, 5, 1)
    coded = brain.observation_code(x)
    assert coded.shape == (5, 1024)
    np.testing.assert_allclose(np.linalg.norm(coded, axis=1), 1.0)
    assert ((coded > 0).sum(axis=1) <= 102).all() and (coded >= 0).all()
    np.testing.assert_array_equal(coded, brain.observation_code(x))  # fixed
    twin = cd.Brain.compose(12_824, 3, modules=(8,), seed=4, episodic=False, recall=False,
                            code=1024)
    np.testing.assert_array_equal(twin.sparse_code.index, code.index)
    other = cd.Brain.compose(12_824, 3, modules=(8,), seed=5, episodic=False, recall=False,
                             code=1024)
    assert not np.array_equal(other.sparse_code.index, code.index)
    assert not brain.observation_code(np.zeros((1, 12_824))).any()  # nothing to code
    # observations that share most of their values share most of their code
    shared = x[:1].copy()
    shared[0, :1000] = 0.0
    apart = frames(12_824, 1, 9)
    assert coded[0] @ brain.observation_code(shared)[0] > 0.5 > coded[0] @ brain.observation_code(apart)[0]


def test_every_input_reaches_the_code_in_both_signs_and_no_two_code_alike():
    """The code reads every input through an ON and an OFF channel, and every channel: each
    is dealt two code units before the units' fan-ins are filled at random, and the weights
    are positive, so a single input of either sign drives the readers of its channel and
    reaches the code. At 12,824 inputs and 1,024 units (seed 1) no one-hot input of either
    sign codes to zero and no two of the 25,648 code alike; 1,169 of them read the same units
    as another and are told apart by the weights alone. The episodic separator's independent
    signed draw over the inputs is unchanged, byte for byte."""
    brain = cd.Brain.compose(12_824, 2, modules=(8,), code=1024, seed=1, episodic=False)
    code = brain.sparse_code
    assert code.channels == 25_648
    assert np.bincount(code.index.ravel(), minlength=code.channels).min() >= 2
    zero, codes, units = {1.0: 0, -1.0: 0}, set(), {}
    for start in range(0, 12_824, 256):
        rows = np.arange(start, min(start + 256, 12_824))
        x = np.zeros((len(rows), 12_824))
        x[np.arange(len(rows)), rows] = 1.0
        for sign in (1.0, -1.0):
            for row in brain.observation_code(sign * x):
                active = np.flatnonzero(row)
                if not len(active):
                    zero[sign] += 1
                    continue
                codes.add((active.tobytes(), row[active].tobytes()))
                units[active.tobytes()] = units.get(active.tobytes(), 0) + 1
    assert zero == {1.0: 0, -1.0: 0}
    assert len(codes) == 25_648  # every one-hot input of either sign has a code of its own
    assert sum(n for n in units.values() if n > 1) <= 1_400  # 1,169 read the same units
    # a narrow code reads more channels per unit, as many as two readers per channel take
    narrow = cd.Brain.compose(12_824, 2, modules=(8,), code=128, seed=1, episodic=False)
    assert narrow.sparse_code.fan_in == 401  # ceil(2 * 2 * 12,824 / 128)
    assert np.bincount(narrow.sparse_code.index.ravel(), minlength=25_648).min() >= 2
    founder = cd.PatternSeparator(12_824, 1024, 24, seed=1, fan_in=32)
    assert founder.cover == 0 and (founder.weight < 0).any() and "cover" not in founder.to_dict()
    assert founder.channels == 12_824
    # the independent draw: each unit's inputs in order, then the signed weights
    rng = np.random.default_rng(1)
    drawn = np.stack([rng.choice(12_824, 32, replace=False) for _ in range(1024)])
    np.testing.assert_array_equal(founder.index, np.sort(drawn, axis=1))
    np.testing.assert_array_equal(founder.weight, rng.standard_normal((1024, 32)) / np.sqrt(32))
    assert code.to_dict()["cover"] == 2


def test_a_negative_input_reaches_the_code_and_a_mixed_sign_one_codes_otherwise():
    """At 256 inputs and 128 units every one-hot input of either sign reaches the code, each
    with a code of its own (12,824 inputs: the test above), and at both sizes ``e_0 - e_j``
    codes otherwise than ``e_0``: the negative half of a signed observation reaches the code."""
    brain = cd.Brain.compose(256, 2, modules=(8,), code=128, seed=1, episodic=False)
    code = brain.sparse_code
    assert (code.channels, code.fan_in) == (512, 32)
    assert np.bincount(code.index.ravel(), minlength=code.channels).min() >= 2
    one_hot = np.eye(256)
    positive, negative = brain.observation_code(one_hot), brain.observation_code(-one_hot)
    assert positive.any(axis=1).all() and negative.any(axis=1).all()
    assert len({row.tobytes() for row in np.concatenate([positive, negative])}) == 512
    for inputs, units in ((256, 128), (12_824, 1024)):
        brain = cd.Brain.compose(inputs, 2, modules=(8,), code=units, seed=1, episodic=False)
        j = np.arange(1, min(inputs, 2049))
        alone = np.zeros((1, inputs))
        alone[0, 0] = 1.0
        mixed = np.repeat(alone, len(j), axis=0)
        mixed[np.arange(len(j)), j] = -1.0
        same = (brain.observation_code(mixed) == brain.observation_code(alone)).all(axis=1)
        assert not same.any()


def test_a_nonnegative_observation_leaves_the_off_channels_silent():
    """A nonnegative observation's code is the code of its ON part beside an all-zero OFF
    part, and the readers of the OFF channels add nothing to it; a signed observation drives
    them."""
    brain = cd.Brain.compose(300, 2, modules=(8,), code=128, seed=2, episodic=False)
    code = brain.sparse_code
    rng = np.random.default_rng(5)
    x = rng.random((6, 300)) * (rng.random((6, 300)) < 0.2)  # nonnegative intensities
    # the code by hand: the ON channels and an all-zero OFF part, the winners, unit length
    channels = np.concatenate([x, np.zeros_like(x)], axis=1)
    drive = np.einsum("bek,ek->be", channels[:, code.index], code.weight)
    keep = np.argsort(-drive, axis=1)[:, : code.winners]
    expected = np.zeros_like(drive)
    np.put_along_axis(expected, keep, np.take_along_axis(drive, keep, axis=1), axis=1)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(brain.observation_code(x), expected, rtol=1e-12, atol=1e-15)
    silent = cd.PatternSeparator(**code.to_dict())
    silent.index = code.index
    silent.weight = np.where(code.index >= code.inputs, 0.0, code.weight)  # no OFF readers
    np.testing.assert_array_equal(silent.code(x), code.code(x))
    signed = x - (rng.random((6, 300)) < 0.2)
    assert (signed < 0).any(axis=1).all()
    assert not (silent.code(signed) == code.code(signed)).all(axis=1).any()


def test_a_covering_separator_needs_room_for_its_readers():
    separator = cd.PatternSeparator(20, 24, 24, seed=2, fan_in=4, cover=2)
    assert separator.channels == 40 and separator.index.max() < 40
    assert np.bincount(separator.index.ravel(), minlength=40).min() >= 2
    assert all(len(np.unique(row)) == 4 for row in separator.index)
    np.testing.assert_array_equal(
        separator.index, cd.PatternSeparator(20, 24, 24, seed=2, fan_in=4, cover=2).index
    )
    # every place taken by a reader: 2 * 2 * 24 = 24 * 4
    exact = cd.PatternSeparator(24, 24, 24, seed=5, fan_in=4, cover=2)
    np.testing.assert_array_equal(np.bincount(exact.index.ravel(), minlength=48), 2)
    # a fan-in beyond the inputs and within the channels
    assert cd.PatternSeparator(20, 24, 24, seed=2, fan_in=30, cover=2).index.max() < 40
    with pytest.raises(ValueError, match="fan_in must not exceed the inputs"):
        cd.PatternSeparator(20, 24, 24, seed=2, fan_in=30)
    with pytest.raises(ValueError, match="fan_in must not exceed the channels"):
        cd.PatternSeparator(20, 24, 24, seed=2, fan_in=41, cover=2)
    for options in ({"fan_in": 0, "cover": 2}, {"fan_in": 4, "cover": 25},
                    {"fan_in": 2, "cover": 3}, {"fan_in": 4, "cover": -1}):
        with pytest.raises(ValueError):
            cd.PatternSeparator(20, 24, 24, seed=2, **options)


def test_compose_can_make_the_sensory_region_the_code():
    raw = cd.Brain.compose(12_824, 3, modules=(64,), seed=1, episodic=False, recall=False)
    coded = cd.Brain.compose(12_824, 3, modules=(64,), seed=1, episodic=False, recall=False,
                             code=1024)
    assert raw.inputs == coded.inputs == 12_824
    assert len(raw.sensory_index) == 12_824 and len(coded.sensory_index) == 1024
    assert raw.sparse_code is None and not raw.coded_cortex  # the founder keys on the observation
    assert coded.coded_cortex and coded.describe()["observation_code"]["cortex"] == "code"

    def sensory_synapses(brain):
        pre = np.asarray(brain.connectome.pre)
        return int(np.isin(pre, brain.sensory_index).sum())

    assert sensory_synapses(raw) == 12_824 * 64 and sensory_synapses(coded) == 1024 * 64
    x = frames(12_824, 2, 2)
    drive = coded.stimulus(x, memory=False)
    np.testing.assert_array_equal(
        drive[:, coded.sensory_index],
        coded.observation_code(x) * coded.brain.neuron_model.stimulus_amplitude,
    )
    np.testing.assert_array_equal(
        raw.stimulus(x, memory=False)[:, raw.sensory_index],
        x * raw.brain.neuron_model.stimulus_amplitude,
    )
    assert coded.act(x, greedy=True).shape == (2,)
    assert coded.describe()["layout"]["inputs"] == 12_824
    assert coded.describe()["initialization"]["composition"]["code"] == 1024
    assert "code" not in raw.describe()["initialization"]["composition"]
    with pytest.raises(ValueError, match="need 12824 values"):
        coded.act(np.zeros((1, 1024)))
    heard = cd.Brain.compose(12_824, 3, modules=(8,), seed=1, code=256, history_decays=(0.5,),
                             arousal=True, recall=True)
    assert len(heard.connectome.populations["history"]) == 256  # the code as heard
    heard.live(x[:1])
    heard.wait(x[1:])
    heard.live(x[1:], reward=[1.0])
    first, second = heard.observation_code(x[:1]), heard.observation_code(x[1:])
    # heard three times at decay 0.5: the first code once, the second twice
    np.testing.assert_allclose(heard.history.trace, 0.125 * first + 0.75 * second)
    assert len(heard.imagine([x[:1], x[1:]])) == 2
    image = cd.Brain.compose((40, 40), 2, modules=(8,), features=1, seed=1, code=256)
    assert len(image.sensory_index) == 1600 and image.code_width == 256 and not image.coded_cortex
    # a custom anatomy: the memories on the code, the cortex on the observation
    custom = cd.Brain(anatomy(3_230, 16, None), code=1024)
    assert custom.sparse_code is not None and not custom.coded_cortex and custom.code_width == 1024


def test_the_memories_key_on_the_code_as_it_is():
    brain = cd.Brain.compose(2_000, 3, modules=(8,), seed=3, memory_choice=True, recall=True,
                             code=1024)
    assert brain.code_width == 1024
    np.testing.assert_array_equal(brain.hippocampus.pre, np.arange(1024))
    np.testing.assert_array_equal(brain.context_memory.pre, np.arange(1024 + 3))
    for memory in (brain.hippocampus, brain.context_memory):
        assert memory.separator is None and memory.strength.shape[1] == len(memory.pre)
    assert brain.coded_cortex and brain.recall.inputs == 0  # the sensory range is the code
    assert brain.recall.key_width == 1024 and "inputs" not in brain.describe()["recall"]
    x = frames(2_000, 1, 4, density=0.1)
    code = brain.observation_code(x)
    key = brain._episode_key(code)
    assert key.shape == (1, 1027)
    np.testing.assert_array_equal(key[0, :1024], 0.5 * code[0])
    brain.step(x, teacher=[1])  # shown once, under the code
    assert brain._context.shape == (1, 1027) and brain._paths[0][-1][2].shape == (1024,)
    assert brain.recall.keys.shape == (1, 1, 1024)
    np.testing.assert_array_equal(brain.recall.keys[0, 0], code[0])
    alike = x.copy()
    alike[0, :100] = 0.0
    votes = brain._vote(brain.observation_code(alike), brain._episode_key(brain.observation_code(alike)))
    assert votes[0, 1] > 0 and votes[0, 1] > votes[0, 0]  # found again under an observation alike
    brain.step(x, reward=[1.0])
    assert brain.hippocampus.writes >= 1
    # the hippocampus of earlier releases reads the code into the drive
    earlier = cd.Brain.compose(2_000, 3, modules=(8,), seed=3, recall=True, code=1024)
    earlier.step(x)
    earlier.step(x, reward=[1.0])
    assert earlier.hippocampus.strength.shape == (1, 1024, 3)
    read = earlier.stimulus(x)[0, earlier.motor_index]
    silent = earlier.stimulus(x, memory=False)[0, earlier.motor_index]
    assert np.abs(read - silent).max() > 0


def anatomy(sensory: int, association: int, recall_width: int | None) -> cd.Connectome:
    """A custom anatomy: a plain sensory region, an association region, three actions and,
    if sized, recall and next regions."""
    regions = [Region("sensory", sensory), Region("association", association), motor_cortex(3)]
    projections = [Projection("sensory", "association", reciprocal=False),
                   Projection("association", "motor")]
    if recall_width is not None:
        regions.extend(Region(name, recall_width) for name in ("recall", "next"))
        projections.extend(
            Projection(name, "association", reciprocal=False) for name in ("recall", "next")
        )
    return develop(Genome(tuple(regions), tuple(projections)), seed=4)


def test_a_custom_anatomy_takes_the_code_and_keeps_recall_off_unless_asked(tmp_path):
    plain = cd.Brain(anatomy(3_230, 16, None))
    assert plain.recall is None and plain.sparse_code is None and plain.code_width == 3_230
    assert not plain.coded_cortex and plain.inputs == 3_230
    # the memories' code with the cortex on the observation: the regions sized by the code
    assert cd.Brain.observation_width(3_230, 1024) == 1024
    brain = cd.Brain(anatomy(3_230, 16, 1024), code=1024, recall=True, memory_choice=True,
                     memory_context=0.5, memory_replay=8, memory_demonstrations=1.0,
                     arousal=True)
    x = frames(3_230, 1, 5, density=0.1)
    brain.live(x)
    brain.live(x, reward=[1.0])
    assert brain.recall.keys.shape == (1, 2, 1024) and brain.hippocampus.writes == 1
    path = brain.save(tmp_path / "anatomy.npz")
    twin = cd.Brain.load(path)
    assert twin.describe() == brain.describe()
    np.testing.assert_array_equal(twin.live(x, reward=[0.0]), brain.live(x, reward=[0.0]))
    # the sensory region as the code: the observation width named beside it
    coded = cd.Brain(anatomy(256, 16, 16), inputs=3_230, code=256, recall=True,
                     recall_value="cortex")
    assert coded.coded_cortex and coded.inputs == 3_230 and coded.recall.value_width == 16
    assert coded.act(x, greedy=True).shape == (1,)
    assert coded.recall.inputs == 0  # the sensory range is the code
    with pytest.raises(ValueError, match="with inputs the sensory region is the code"):
        cd.Brain(anatomy(256, 16, None), inputs=3_230, code=128)
    with pytest.raises(ValueError, match="with inputs the sensory region is the code"):
        cd.Brain(anatomy(256, 16, None), inputs=3_230)  # no code: the region is not one
    with pytest.raises(ValueError, match=r"one neuron per value coordinate \(1024\)"):
        cd.Brain(anatomy(3_230, 16, 512), code=1024, recall=True)
    cortex = cd.Brain(anatomy(3_230, 16, 16), recall=True, recall_value="cortex")
    assert cortex.recall is not None
    with pytest.raises(ValueError, match="inputs must be an integer of at least 1"):
        cd.Brain(anatomy(256, 16, None), inputs=0, code=256)


def test_storage_at_camera_width_stays_small():
    def storage(brain: cd.Brain) -> dict[str, float]:
        separators = sum(
            0 if memory.separator is None
            else sum(value.nbytes for value in memory.separator.arrays().values())
            for memory in (brain.hippocampus, brain.context_memory)
        )
        recall = brain.recall
        address = recall.w_x.nbytes + recall.w_c.nbytes + recall.b.nbytes
        tables = brain.sparse_code.index.nbytes + brain.sparse_code.weight.nbytes
        return {"separators": separators / MB, "address": address / MB, "tables": tables / MB}

    camera = cd.Brain.compose(12_824, 3, modules=(64,), seed=1, memory_choice=True, code=1024,
                              recall=True)
    assert camera.inputs == 12_824 and camera.code_width == 1024 and camera.coded_cortex
    sizes = storage(camera)
    assert sizes["separators"] == 0 and sizes["address"] < 9 and sizes["tables"] < 0.85
    assert sum(sizes.values()) < 20
    # the sensory synapses of the coded cortex against the observation's
    pre = np.asarray(camera.connectome.pre)
    assert np.isin(pre, camera.sensory_index).sum() == 1024 * 64
    # the recommended sparse separation on the observation itself: the address map stays
    # inputs by inputs
    sparse = cd.Brain.compose(12_824, 3, modules=(64,), seed=1, memory_choice=True, recall=True,
                              memory_fan_in=32)
    assert not sparse.coded_cortex and sparse.sparse_code is None
    for memory in (sparse.hippocampus, sparse.context_memory):
        assert sum(v.nbytes for v in memory.separator.arrays().values()) / MB < 0.7
    assert sparse.recall.w_x.nbytes / MB > 1000
    # the dense separators of the founder, for comparison: 12,824 x 1,024 floats
    assert 12_824 * 1024 * 8 / MB > 100
    assert np.isin(np.asarray(sparse.connectome.pre), sparse.sensory_index).sum() == 12_824 * 64


def test_a_coded_checkpoint_carries_its_code_and_a_mismatch_is_refused(tmp_path):
    brain = cd.Brain.compose(2_000, 3, modules=(8,), seed=3, memory_choice=True, arousal=True,
                             recall=True, code=1024)
    x = [frames(2_000, 1, seed, density=0.1) for seed in range(3)]
    for t in range(6):
        brain.live(x[t % 3], reward=None if t == 0 else [float(t % 2)])
    path = brain.save(tmp_path / "coded.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = json.loads(str(arrays["generic"]))
    assert meta["format"] == "cadence-generic/9"
    assert meta["code"] == {**brain.sparse_code.to_dict(), "cortex": True}
    assert arrays["code/index"].shape == arrays["code/weight"].shape == (1024, 32)
    assert meta["hippocampus"]["pre"] == 1024 and meta["episodic"]["context_memory"]["pre"] == 1027
    assert arrays["episode/path/observations"].shape[2] == 1024
    twin = cd.Brain.load(path)
    assert twin.describe() == brain.describe()
    np.testing.assert_array_equal(twin.sparse_code.index, brain.sparse_code.index)
    np.testing.assert_array_equal(twin.sparse_code.weight, brain.sparse_code.weight)
    for t in range(4):
        np.testing.assert_array_equal(
            brain.live(x[t % 3], reward=[0.5]), twin.live(x[t % 3], reward=[0.5])
        )
    coded = cd.Brain.compose(2_000, 3, modules=(8,), seed=3, code=256, arousal=True,
                             recall=True)
    coded.live(x[0])
    coded.live(x[1], reward=[1.0])
    with np.load(coded.save(tmp_path / "cortex.npz"), allow_pickle=False) as data:
        cortex_meta = json.loads(str(data["generic"]))
    assert cortex_meta["code"]["cortex"] is True and cortex_meta["composition"]["code"] == 256
    cortex_twin = cd.Brain.load(tmp_path / "cortex.npz")
    assert cortex_twin.coded_cortex and cortex_twin.inputs == 2_000
    np.testing.assert_array_equal(cortex_twin.live(x[2], reward=[0.0]), coded.live(x[2], reward=[0.0]))

    def tampered(name: str, mutate) -> Path:
        copy = {k: v.copy() for k, v in arrays.items()}
        meta = json.loads(str(copy["generic"]))
        mutate(meta, copy)
        copy["generic"] = np.array(json.dumps(meta))
        np.savez(tmp_path / f"{name}.npz", **copy)
        return tmp_path / f"{name}.npz"

    def no_code(meta, _):
        del meta["code"]

    def earlier_format(meta, _):
        meta["format"] = "cadence-generic/8"

    def before_the_cover(meta, copy):  # a code saved before the covering construction
        del meta["code"]["cover"]
        meta["code"]["fan_in"] = 32
        signed = cd.PatternSeparator(2_000, 1024, 102, seed=meta["code"]["seed"], fan_in=32)
        copy["code/index"], copy["code/weight"] = signed.index, signed.weight

    def unknown_field(meta, _):
        meta["code"]["channels"] = 4_000

    def other_coordinates(meta, copy):  # place records keyed on other coordinates of the code
        copy["episodic/pre"] = np.roll(copy["episodic/pre"], 1)

    def beyond_the_code(meta, copy):  # and on coordinates the code does not have
        copy["episodic/pre"] = copy["episodic/pre"] + 1

    def other_fan_in(meta, _):
        meta["code"]["fan_in"] = 16

    def other_winners(meta, _):
        meta["code"]["winners"] = 24

    def other_width(meta, _):
        meta["code"]["inputs"] = 2_001

    def cortex_flag(meta, _):
        meta["code"]["cortex"] = False

    def odd_table(meta, copy):  # past the 4,000 channels, an ON and an OFF per input
        copy["code/index"][0, 0] = 4_000

    def inputs_alone(meta, copy):  # positive weights on the inputs, no OFF channel read
        rng = np.random.default_rng(1)
        copy["code/index"] = np.sort(
            np.stack([rng.choice(2_000, 32, replace=False) for _ in range(1024)]), axis=1
        )

    def nonpositive_weight(meta, copy):
        copy["code/weight"][0, 0] = 0.0

    def repeated_table(meta, copy):
        copy["code/index"][0, 1] = copy["code/index"][0, 0]

    def no_table(meta, copy):
        del copy["code/weight"]

    def fewer_units(meta, copy):  # a narrower code than the records were keyed on
        meta["code"]["expansion"], meta["code"]["winners"] = 512, 51
        copy["code/index"] = copy["code/index"][:512]
        copy["code/weight"] = copy["code/weight"][:512]

    for name, mutate, message in (
        ("no_code", no_code, "belongs to checkpoint format cadence-generic/9"),
        ("earlier_format", earlier_format, "belongs to checkpoint format cadence-generic/9"),
        ("before_the_cover", before_the_cover, "lacks the covering construction's field, "
                                               "cover: it was saved before the covering code "
                                               "and cannot be loaded by this version"),
        ("unknown_field", unknown_field, "invalid saved code metadata"),
        ("other_fan_in", other_fan_in, "32 channels per unit, and keeps one unit in 10"),
        ("other_winners", other_winners, "32 channels per unit, and keeps one unit in 10"),
        # the tables read 4,000 of the 4,002 channels of 2,001 inputs
        ("other_width", other_width, "reads every channel of the observation"),
        ("inputs_alone", inputs_alone, "reads every channel of the observation"),
        ("nonpositive_weight", nonpositive_weight, "with 2 units at least, its weights positive"),
        ("cortex_flag", cortex_flag, "reads observations of 2000 values where the sensory "
                                     "region has 1024"),
        ("odd_table", odd_table, "reads distinct channels of the observation"),
        ("repeated_table", repeated_table, "reads distinct channels of the observation"),
        ("no_table", no_table, "missing saved code table"),
        ("fewer_units", fewer_units, "sensory region of 1024 neurons is not the saved code of "
                                     "512 units"),
        ("other_coordinates", other_coordinates, "keyed on other coordinates than the brain's "
                                                 "1024: a coded brain numbers them from zero"),
        ("beyond_the_code", beyond_the_code, "saved memory keys exceed their 1024 coordinates"),
    ):
        with pytest.raises(ValueError, match=message):
            cd.Brain.load(tampered(name, mutate))


def wider_than_the_graph(recall: bool = False, **options) -> cd.Brain:
    """A custom anatomy whose code is wider than its whole graph: 32 sensory, 8 association
    and 2 motor neurons and a code of 128 units, with the default episodic memory; with
    ``recall`` 8 recall and 8 next neurons for a memory of cortex values."""
    regions = [Region("sensory", 32), Region("association", 8), Region("motor", 2)]
    projections = [Projection("sensory", "association", reciprocal=False),
                   Projection("association", "motor")]
    if recall:
        regions += [Region("recall", 8), Region("next", 8)]
        projections += [Projection(name, "association", reciprocal=False)
                        for name in ("recall", "next")]
        options = {"recall": True, "recall_value": "cortex", **options}
    connectome = develop(Genome(tuple(regions), tuple(projections)), seed=3)
    return cd.Brain(connectome, code=128, seed=3, **options)


@pytest.mark.parametrize(
    "variant",
    [{}, {"recall": True}, {"memory_choice": True, "memory_context": 0.5, "arousal": True}],
    ids=["episodic", "recall", "context"],
)
def test_a_code_wider_than_the_graph_saves_and_continues_from_a_pending_action(
    tmp_path, variant
):
    """A memory keyed on the code numbers its key coordinates from zero, so a code wider than
    the graph keys on coordinates beyond the last neuron: the loader checks the keys against
    the code and the outputs against the graph. The brain saves with an action awaiting its
    outcome, loads, and continues and saves as the brain it was."""
    brain = wider_than_the_graph(**variant)
    assert brain.code_width == 128 > brain.connectome.n
    np.testing.assert_array_equal(brain.hippocampus.pre, np.arange(128))
    x = frames(32, 6, 8, density=0.3)
    brain.step(x[:1])
    for t in (1, 2, 3):
        brain.step(x[t:t + 1], reward=[float(t % 2)])
    assert brain.basal_ganglia._pending is not None  # the last action awaits its outcome
    path = brain.save(tmp_path / "wide.npz")
    twin = cd.Brain.load(path)
    assert twin.describe() == brain.describe()
    np.testing.assert_array_equal(twin.hippocampus.pre, brain.hippocampus.pre)
    np.testing.assert_array_equal(twin.hippocampus.strength, brain.hippocampus.strength)
    if brain.context_memory is not None:
        np.testing.assert_array_equal(twin.context_memory.pre, np.arange(128 + 2))
    if brain.recall is not None:
        assert brain.recall.key_width == 128 and brain.recall.value_width == 8
        np.testing.assert_array_equal(twin.recall.keys, brain.recall.keys)
    for t in (4, 5):
        np.testing.assert_array_equal(
            brain.step(x[t:t + 1], reward=[float(t % 2)]),
            twin.step(x[t:t + 1], reward=[float(t % 2)]),
        )
    assert checkpoint_digest(brain.save(tmp_path / "later.npz")) == checkpoint_digest(
        twin.save(tmp_path / "twin.npz")
    )


def test_the_loader_bounds_the_keys_by_the_code_and_the_outputs_by_the_graph(tmp_path):
    brain = wider_than_the_graph(memory_choice=True, memory_context=0.5)
    brain.step(frames(32, 1, 9, density=0.3))
    path = brain.save(tmp_path / "wide.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    n = brain.connectome.n

    def refused(name, mutate, message):
        copy = {k: v.copy() for k, v in arrays.items()}
        mutate(copy)
        np.savez(tmp_path / f"{name}.npz", **copy)
        with pytest.raises(ValueError, match=message):
            cd.Brain.load(tmp_path / f"{name}.npz")

    def beyond_the_code(copy):
        copy["episodic/pre"][-1] = 128

    def beyond_the_graph(copy):
        copy["episodic/post"][0] = n

    def context_beyond_its_code(copy):
        copy["temporal/pre"][-1] = 130

    def context_on_other_coordinates(copy):
        copy["temporal/pre"] = copy["temporal/pre"][::-1].copy()

    refused("code", beyond_the_code, "keys exceed their 128 coordinates")
    refused("graph", beyond_the_graph, "outputs exceed the connectome")
    refused("context", context_beyond_its_code, "keys exceed their 130 coordinates")
    refused("order", context_on_other_coordinates, "context records are keyed on other")


def parsed(path: Path) -> tuple[dict, dict]:
    """A checkpoint's metadata entries, parsed, and its other arrays by name."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = {name: json.loads(str(arrays.pop(name))) for name in ("generic", "meta")
            if name in arrays}
    return meta, arrays


def assert_same_saved(first: Path, second: Path) -> None:
    (meta, arrays), (again, arrays_again) = parsed(first), parsed(second)
    assert meta == again
    assert set(arrays) == set(arrays_again)
    for name, value in arrays.items():
        np.testing.assert_array_equal(arrays_again[name], value, err_msg=name)


def test_a_coded_round_trip_keeps_the_codes_seed_and_every_saved_setting(tmp_path):
    """Loading installs the saved code with its whole specification: a code drawn from a
    seed other than the default saves, loads and saves again with the same parsed metadata
    and the same arrays, and the continued lives save the same."""
    brain = cd.Brain.compose(32, 2, modules=(8,), code=24, seed=11, recall=True)
    # the code's seed: the brain's on a stream of its own, apart from the streams a brain
    # develops, acts and separates from (its seed and the next one)
    seed = int(np.random.SeedSequence([11, 7103]).generate_state(1)[0])
    assert brain.sparse_code.seed == seed and seed not in range(64)
    brain.act(np.eye(32)[:1], greedy=True)
    first = brain.save(tmp_path / "first.npz")
    twin = cd.Brain.load(first)
    assert twin.sparse_code.to_dict() == brain.sparse_code.to_dict()
    second = twin.save(tmp_path / "second.npz")
    assert parsed(second)[0]["generic"]["code"]["seed"] == seed
    assert_same_saved(first, second)
    for i in (3, 5, 2):
        x = np.eye(32)[[i]]
        np.testing.assert_array_equal(brain.act(x, greedy=True), twin.act(x, greedy=True))
    assert_same_saved(brain.save(tmp_path / "continued.npz"), twin.save(tmp_path / "twin.npz"))


def test_place_records_of_another_width_are_refused_by_the_loader(tmp_path):
    """The hippocampus of earlier releases on a coded brain: its saved records, consistent
    among themselves at another width, are refused against the brain's code."""
    brain = cd.Brain.compose(2_000, 3, modules=(8,), seed=3, code=1024, recall=True)
    x = frames(2_000, 1, 6, density=0.1)
    brain.step(x)
    brain.step(x, reward=[1.0])
    path = brain.save(tmp_path / "coded.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    meta = json.loads(str(arrays["generic"]))
    meta["hippocampus"]["pre"] = 512
    meta["hippocampus"]["persistent_parameters"] = 512 * 3
    arrays["episodic/pre"] = np.arange(512)
    arrays["episodic/strength"] = np.zeros((1, 512, 3))
    arrays["episodic/consolidated"] = np.zeros((512, 3))
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "narrow.npz", **arrays)
    with pytest.raises(ValueError, match="keyed on 512 coordinates where the brain's "
                                         "observation code has 1024"):
        cd.Brain.load(tmp_path / "narrow.npz")


def test_a_wide_brain_keyed_on_its_observation_saves_and_loads_as_before(tmp_path):
    """A wide brain without a code, the founder: its checkpoint names no code and loads keyed
    on the observation."""
    brain = cd.Brain.compose(1_500, 3, modules=(8,), seed=2, recall=True)
    x = frames(1_500, 1, 7, density=0.1)
    brain.step(x)
    brain.step(x, reward=[1.0])
    path = brain.save(tmp_path / "raw.npz")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["generic"]))
    assert meta["format"] == "cadence-generic/8" and "code" not in meta
    assert meta["hippocampus"]["pre"] == 1_500
    twin = cd.Brain.load(path)
    assert twin.sparse_code is None and twin.code_width == 1_500
    np.testing.assert_array_equal(twin.step(x, reward=[0.0]), brain.step(x, reward=[0.0]))


def test_compose_takes_an_image_shape_like_genome():
    brain = cd.Brain.compose((6, 6), 2, modules=(8,), seed=2)
    populations = brain.connectome.populations
    assert list(populations)[:3] == ["visual", "visual/input", "visual/output"]
    assert len(populations["visual/input"]) == 36 and len(populations["visual/output"]) == 128
    assert "sensory" not in populations and len(brain.sensory_index) == 36
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    forward = np.isin(post, populations["association"]) & np.isin(pre, populations["visual"])
    assert forward.sum() == 128 * 8 and np.isin(pre[forward], populations["visual/output"]).all()
    assert not (np.isin(pre, populations["visual/input"]) & ~np.isin(post, populations["visual"])).any()
    assert brain.describe()["layout"]["inputs"] == 36
    assert brain.describe()["initialization"]["composition"]["image"] == [6, 6, 1]
    assert brain.describe()["initialization"]["composition"]["inputs"] == 36
    assert brain.sparse_code is None  # 36 pixels: the memories key on the pixels themselves
    assert len(cd.Brain.compose((6, 6), 2, modules=(8,), features=2, field=2).connectome.populations["visual/output"]) == 2 * 25
    colour = cd.Brain.compose((4, 4, 3), 2, modules=(8,), seed=2)
    assert len(colour.sensory_index) == 48 and colour.code_width == 48
    heard = cd.Brain.compose((6, 6), 2, modules=(8,), history_decays=(0.5,), efference_amplitude=1.0,
                             memory_choice=True, arousal=True, recall=True)
    assert len(heard.connectome.populations["history"]) == 36  # the pixels as heard
    assert len(heard.connectome.populations["recall"]) == heard.code_width == 36
    frame = np.zeros((1, 36))
    frame[0, :18] = 1.0
    other = np.zeros((1, 36))
    other[0, 18:] = 1.0
    heard.live(frame)
    heard.live(other, reward=[1.0])
    heard.wait(frame)
    heard.live(other, reward=[0.0], done=[True])
    assert heard.recall.slots == 1 and heard.hippocampus.writes >= 1
    wide = cd.Brain.compose((40, 40), 2, modules=(8,), features=1, seed=1, memory_choice=True)
    assert wide.sparse_code is None and wide.code_width == 1600  # the pixels, as every founder
    coded = cd.Brain.compose((40, 40), 2, modules=(8,), features=1, seed=1, memory_choice=True,
                             code=1024)
    assert coded.code_width == 1024 and not coded.coded_cortex
    np.testing.assert_array_equal(coded.hippocampus.pre, np.arange(1024))
    assert cd.Brain.compose(36, 2, modules=(8,)).sparse_code is None  # a vector length, as before
    for bad in ((6,), (6, 6, 1, 1), (0, 6), "6x6"):
        with pytest.raises(ValueError):
            cd.Brain.compose(bad, 2, modules=(8,))
