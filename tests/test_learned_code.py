"""The learned code region: with ``learned_code=None`` every brain, checkpoint and life is the
composition before it; with ``learned_code=units`` a region of the brain's own neurons settles
to a sparse code of the observation, the memories key on it, and it learns every moment."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from checkpoints import earlier, metadata

import cadence as cd
from cadence import brain as cadence_brain
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, cortex, motor_cortex, prefrontal_cortex

# The acts and the array contents below were recorded on the NumPy solver; the compiled dense
# kernel (numba) orders the arithmetic otherwise and gives other bits, so those pins are read
# on the NumPy solver alone (CADENCE_FUSED=0 selects it).
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
INPUTS, ACTIONS, UNITS = 53, 4, 64


# -- the founders: byte-identical to the composition before the learned code


def checkpoint_digest(path: Path) -> str:
    """SHA-256 over every array of a checkpoint, by name, with the metadata JSON entries
    read back without the library version that wrote them."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            value = data[name]
            h.update(name.encode())
            if name in ("meta", "generic", "brain"):
                meta = json.loads(str(value))
                meta.pop("version", None)
                h.update(json.dumps(meta, sort_keys=True).encode())
                continue
            h.update(str(value.dtype).encode() + str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def names_digest(path: Path) -> tuple[str, str]:
    """The checkpoint's format, and its metadata keys and member names, which no backend
    changes."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["brain"]))
        names = json.dumps([sorted(meta), sorted(data.files)])
    return meta["format"], hashlib.sha256(names.encode()).hexdigest()


def parameter_digest(brain: cd.Brain) -> str:
    connectome, graph = brain.connectome, brain.brain
    h = hashlib.sha256()
    for values in (connectome.pre, connectome.post, connectome.count, connectome.sign,
                   graph.efficacy, graph.bias, graph.log_gain):
        h.update(np.ascontiguousarray(np.asarray(values)).tobytes())
    populations = {k: [int(i) for i in v] for k, v in connectome.populations.items()}
    h.update(json.dumps(populations).encode())
    return h.hexdigest()


def describe_digest(brain: cd.Brain) -> str:
    description = brain.describe()
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


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


def taught(brain: cd.Brain) -> list[int]:
    """Sixteen one-hot tokens, lessons judged by the act, as the binding chambers teach."""
    rng = np.random.default_rng(23)
    acts = []
    for t in range(10):
        x = np.eye(16)[rng.integers(16, size=4)]
        labels = rng.integers(8, size=4)
        if t % 2:
            brain.lesson(x, labels)
        acts.extend(int(a) for a in brain.act(x, greedy=True))
    return acts


# Recorded on the composition before the learned code (its parent commit): each founder's
# connectome and parameters, the acts of a short life, the description, and the format, the
# member and metadata names and the contents of every array of the checkpoint it saves. The
# descriptions of the founders with arousal are those of the branch the region merged into,
# whose description carries the arousal gene need_scale.
# Where the library later removed a setting that was inert at its founder, the description
# and checkpoint digests were re-recorded after every other value matched the recording.
# Re-recorded with the unset motor lateral at 0.0 (issue 220) where a founder has at most
# 8 motor neurons in a slot, and at the founder need scale 0.1 (issue 198) where it has arousal.
# The descriptions were re-recorded at the one spelling of the learner's temperature and
# without the constants of the composed brain (0.93.0, every other value matching the recording).
FOUNDERS: dict[str, dict] = {
    "stepped": {
        "build": (4, 2, {"modules": (8,), "seed": 3}),
        "life": stepped,
        "parameters": "8d42b245bdc9b36499675f913258d4ca9e532a91ad39a0bda82b666e4a441837",
        "acts": [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        "format": "cadence-brain/1",
        "names": "cf4f510fd7745ee140166219a1a6e0af36750c92b9eb41a38f9262d2272f7dda",
        "checkpoint": "995ca55ca539707fc98e5d17c3132b091dcbdc070adeb5dbff229981072998fc",
        "describe": "a80424edb7491028f605098d8dea02c63e7075f56beefeddd8e7aa5c40141490",
    },
    "episodic": {
        "build": (4, 2, {"modules": (8,), "seed": 7, "memory_choice": True, "arousal": True}),
        "life": lived,
        "parameters": "dcbc0983cda21c93c39ec2e5f75b9deebe3af815e53bdd9e9af7befbc7afac53",
        "acts": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "format": "cadence-brain/1",
        "names": "1c418d7da8207299c6e8058e90cd3f627a6c3119a3287e65090a2cbd06ed603a",
        "checkpoint": "819a806d35bf843679477ab0595e97fe8140ca880de3dc0791800e4c6befe72a",
        "describe": "61751c8ade06e677e4846ca3e097b36ce97b4728f1d5ed476c0fb9e7ebf78a9a",
    },
    # recall without the position term of 0.89.0: the acts and parameters predate it; the
    # description and the checkpoint name recall_position=0, which differs from its founder
    "recalled": {
        "build": (4, 2, {"modules": (8,), "seed": 5, "arousal": True, "recall": True,
                         "recall_position": 0, "history_decays": (0.2, 0.9),
                         "efference_amplitude": 1.0, "memory_choice": True}),
        "life": lived,
        "parameters": "bf4d075047bb0d0713d3ab20312bd17853820de818de29e228b2edaf5bcdd26c",
        "acts": [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        "format": "cadence-brain/1",
        "names": "272cd7aa87c6eb361ba04b442ba83f88b5d63f2acfdf46de8aecf36e40ac76bd",
        "checkpoint": "8508c80b74047c133e5e18f4cb11d40e36cc517fa9eed748f44cb6057b9d9bae",
        "describe": "66b1635161d1487eeff5ae1a8af0b08050a83f81414069c581834fefa55d862f",
    },
    "taught": {
        "build": (16, 8, {"modules": (64,), "lateral": 0.0, "seed": 1,
                          "learning_surprise": True, "recall": True, "recall_position": 0}),
        "life": taught,
        "parameters": "1b72292d2b15754485932c2527b342353a6934c638e8e25f3da7d05e6b1e45ce",
        "acts": [4, 7, 1, 0, 0, 4, 4, 4, 0, 4, 4, 4, 0, 0, 7, 3, 0, 0, 7, 4, 0, 0, 7, 4, 1, 0, 0, 4, 3, 0, 0, 1, 0, 6, 0, 7, 0, 0, 0, 1],
        "format": "cadence-brain/1",
        "names": "0aab0d179b44428d742d358f48b22b0571e9fd78c4cefa76ad52551d9920c16e",
        "checkpoint": "67f8a80c4c28c099ea268a7b8c0fc6d607f1c7dc46ea96e1eed2690fc730e2d3",
        "describe": "cc8ab01bf4d13f18c5a4053a6e5ad23b0affb3679b08ec32f05a96df72703f5a",
    },
    "wide": {
        "build": (2_000, 3, {"modules": (8,), "seed": 13, "memory_choice": True,
                             "arousal": True}),
        "life": lived,
        "parameters": "1eac71c04400948031c90dc1b00bd09f3cc5068037a88e69e6d3af225e0b89ae",
        "acts": [2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        "format": "cadence-brain/1",
        "names": "1c418d7da8207299c6e8058e90cd3f627a6c3119a3287e65090a2cbd06ed603a",
        "checkpoint": "d26023c492f2b68d4117f558bb347716da946a4b3aaed192b56e865b2001fd29",
        "describe": "23f961fb05db69d83cc676ed0f0db8437b40024a849698985578d9dac8961e65",
    },
}


@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_a_founder_is_the_composition_before_the_learned_code(tmp_path, name):
    """On every backend: the connectome and parameters, the description, and the checkpoint's
    format, members and metadata keys after the life; no code region, no learned-code
    metadata, the memories keyed as before."""
    spec = FOUNDERS[name]
    inputs, actions, options = spec["build"]
    brain = cd.Brain.compose(inputs, actions, **options)
    assert brain.learned_code is None and "code" not in brain.connectome.populations
    assert brain.learner.plastic_synapses.all() and brain.learner.plastic_neurons.all()
    assert parameter_digest(brain) == spec["parameters"]
    spec["life"](brain)
    assert describe_digest(brain) == spec["describe"]
    assert "learned_code" not in brain.describe()["layout"]
    assert not any(k.startswith("learned_code") for k in brain.describe()["genes"])
    path = brain.save(tmp_path / "founder.npz")
    assert names_digest(path) == (spec["format"], spec["names"])
    with np.load(path, allow_pickle=False) as data:
        assert not any("code" in member for member in data.files)


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_the_founders_acts_and_checkpoint_bytes_are_the_recorded_ones(tmp_path, name):
    spec = FOUNDERS[name]
    inputs, actions, options = spec["build"]
    brain = cd.Brain.compose(inputs, actions, **options)
    assert spec["life"](brain) == spec["acts"]
    assert checkpoint_digest(brain.save(tmp_path / "founder.npz")) == spec["checkpoint"]


def region_digest(brain: cd.Brain, acts: list[int], path: Path) -> str:
    """SHA-256 over a life with the region: its acts, every array of the checkpoint it saves
    (the metadata, which names the region's settings, left out) and the region's work."""
    h = hashlib.sha256(json.dumps(acts).encode())
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            if name in ("meta", "generic", "brain"):
                continue
            value = data[name]
            h.update(name.encode() + str(value.dtype).encode() + str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
    h.update(json.dumps(brain.learned_code.work, sort_keys=True).encode())
    return h.hexdigest()


# Two founder lives with the region, recorded on the head before the region's settings were
# folded into constants: whatever is removed or fixed, a founder lives them bit for bit.
# Where the library later removed state that was inert at these founders (the actor's
# moments, the valence statistics, the recall memory's empty context history), the digests
# were re-recorded after the acts, the region's work and every other array matched the
# recording; since an imagined branch leaves the region's work counts as they were, the
# second founder's digest was re-recorded after its acts and every array matched, its work
# short of the six settles of the three imagined moments that ends its life.
# Both were re-recorded once more under the 0.85 defaults (an unset lateral at 0.0, the
# scaled need at 0.1), after the two lives with those settings given at their earlier values
# reproduced the earlier digests exactly.
REGION_FOUNDERS: dict[str, dict] = {
    "region": {
        "build": (16, 8, {"modules": (24,), "seed": 5, "learned_code": 16, "recall": True,
                          "recall_position": 0}),  # recorded before the position term
        "life": taught,
        "digest": "8ef21e03bb9d987b50ee7ada774f3b391a75ea47e7c8b76599b3b48504b8fbae",
    },
    "region_fields": {
        "build": (16, 4, {"modules": (24,), "seed": 6, "learned_code": 16,
                          "learned_code_fields": (8, 8), "memory_choice": True,
                          "arousal": True}),
        "life": lived,
        "digest": "5c531c395ecaaec07f69c853d13a7c6477fe2a53a65bf80ca20bb84f5d97481c",
    },
}


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
@pytest.mark.parametrize("name", sorted(REGION_FOUNDERS))
def test_a_founder_with_the_region_lives_its_recorded_life(tmp_path, name):
    spec = REGION_FOUNDERS[name]
    inputs, actions, options = spec["build"]
    brain = cd.Brain.compose(inputs, actions, **options)
    acts = spec["life"](brain)
    path = brain.save(tmp_path / "region.npz")
    assert region_digest(brain, acts, path) == spec["digest"]


# the settings the readings retired, each with a value it once took
REMOVED = {"cortex": "code", "output": "unit", "floor": 0.0, "scale": "observation",
           "place": "free", "reads": "free", "held": True, "top_down": 2.0, "rate": 0.125,
           "predictor": 0.0625, "threshold": 0.1, "threshold_rate": 0.000625,
           "target": 0.1, "repair": 1.0, "silence": 10.0, "identity": 0.9, "active": 0.05}


def test_the_settings_the_readings_retired_are_refused_by_name():
    """A removed setting is refused at construction and in a retune with what the region
    does instead, whatever value it is given."""
    assert set(REMOVED) == set(cd.LearnedCode.REMOVED)
    brain = cd.Brain.compose(16, 4, modules=(8,), seed=1, learned_code=8)
    for name, value in REMOVED.items():
        with pytest.raises(ValueError, match=f"learned_code_{name} was removed"):
            cd.Brain.compose(16, 4, modules=(8,), seed=1, learned_code=8,
                             **{f"learned_code_{name}": value})
        with pytest.raises(ValueError, match=f"learned_code_{name} was removed"):
            brain.retune(**{f"learned_code_{name}": value})


def test_learned_code_none_is_the_founder_and_the_width_is_as_before():
    assert cd.Genome.observation_width(53) == 53 and cd.Genome.observation_width(53, 24) == 24
    assert cd.Genome.observation_width(53, None, 64) == 64 + 53  # the key joins the identity
    assert cd.Genome.observation_width(53, None, 16) == 16  # a wider observation: the code
    for value in (None, 0):
        brain = cd.Brain.compose(53, 4, modules=(8,), seed=2, learned_code=value)
        assert brain.learned_code is None and brain.code_width == 53
        assert "learned_code" not in brain.describe()["layout"]["composition"]


# -- the region


def composed(seed: int = 0, units: int = UNITS, **options):
    return cd.Brain.compose(
        INPUTS, ACTIONS, modules=(64,), seed=seed, learning_surprise=True, learned_code=units,
        **options,
    )


def tokens(*indices: int) -> np.ndarray:
    return np.eye(INPUTS)[list(indices)]


def test_the_region_joins_the_connectome_with_its_three_synapse_groups():
    brain = composed()
    populations = brain.connectome.populations
    assert list(populations)[:3] == ["sensory", "code", "association"]
    assert len(populations["code"]) == UNITS and brain.code_width == UNITS + INPUTS
    code = brain.learned_code
    assert isinstance(code, cd.LearnedCode)
    assert code.inputs == INPUTS and code.units == UNITS and code.context == 64
    groups = brain._code_synapses
    assert len(groups["dictionary"][0]) == INPUTS * UNITS
    assert len(groups["lateral"][0]) == UNITS * (UNITS - 1)
    assert len(groups["predictor"][0]) == 64 * UNITS
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    # the sensory region projects to the code region alone, and the code to the first module
    assert np.isin(post[np.isin(pre, brain.sensory_index)], brain.code_index).all()
    forward = np.isin(pre, brain.code_index) & np.isin(post, brain.association_index)
    assert forward.sum() == UNITS * 64
    # nothing leads back into the region, and the sensory and prefrontal neurons hear nothing
    assert not np.isin(post, np.concatenate([brain.sensory_index, brain.prefrontal_index])).any()
    into = np.isin(post, brain.code_index)
    assert np.isin(pre[into], np.concatenate(
        [brain.sensory_index, brain.code_index, brain.prefrontal_index])).all()
    # the held code: the whole graph carries the region's synapses at zero (nothing leads
    # into the code units there), the region's own sub-network the Gram as inhibition and
    # the thresholds, and its stimulus the dictionary's read of the sensory activation
    efficacy = brain.brain.efficacy
    for synapses, _, _ in groups.values():
        assert not efficacy[synapses].any()
    np.testing.assert_allclose(code.graph.efficacy, code.lateral_efficacy())
    np.testing.assert_array_equal(code.graph.bias, -code.thresholds)
    sensory = np.eye(INPUTS)[:2]
    np.testing.assert_allclose(code.stimulus(sensory, None), code.read * sensory @ code.dictionary)
    # the read gain: one input at amplitude 1 drives the unit whose column matches it by one
    model = brain.brain.neuron_model
    assert np.isclose(code.read * model.activation(np.array([model.stimulus_amplitude])), 1.0)
    gram = code.dictionary.T @ code.dictionary
    np.fill_diagonal(gram, 0.0)
    np.testing.assert_allclose(code.lateral(), -gram)
    np.testing.assert_array_equal(brain.brain.bias[brain.code_index], -code.thresholds)
    np.testing.assert_allclose(np.linalg.norm(code.dictionary, axis=0), 1.0)
    # the lesson's contrast and the dopamine leave them alone
    plastic = brain.learner.plastic_synapses
    for synapses, _, _ in groups.values():
        assert not plastic[synapses].any()
    assert plastic.sum() == brain.connectome.synapses - sum(len(g[0]) for g in groups.values())
    assert not brain.learner.plastic_neurons[brain.code_index].any()
    assert brain.learner.plastic_neurons.sum() == brain.connectome.n - UNITS
    # the memories key on the code
    np.testing.assert_array_equal(brain.hippocampus.pre, np.arange(UNITS + INPUTS))
    described = brain.describe()
    assert described["layout"]["code"] == {
        "kind": "learned", "inputs": INPUTS, "width": UNITS + INPUTS, "units": UNITS, "fan_in": 0,
        "winners": 0, "cortex": "code", "separator": None,
    }
    assert described["layout"]["learned_code"]["units"] == UNITS
    assert described["layout"]["composition"]["learned_code"] == UNITS
    assert {k: v for k, v in described["genes"].items() if k.startswith("learned_code")} == {
        f"learned_code_{name}": value
        for name, value in cd.LearnedCode.founders().items()
    }
    assert brain.parameters() >= code.parameters()


def test_the_constants_and_the_one_setting_and_bad_settings_are_refused():
    assert cd.LearnedCode.founders() == {"fields": None}
    assert cd.LearnedCode.GENES == ("fields",) and not cd.LearnedCode.RUNTIME
    code = composed().learned_code
    for name, value in (("top_down", 2.0), ("rate", 0.125), ("predictor", 0.0625),
                        ("threshold", 0.1), ("threshold_rate", 0.000625), ("target", 0.1),
                        ("repair", 1.0), ("silence", 10.0), ("active", 0.05),
                        ("identity", 0.9)):
        assert getattr(cd.LearnedCode, name.upper()) == value == getattr(code, name)
    assert code.learning is True
    with pytest.raises(ValueError, match="sum to 50, not to the 53 inputs"):
        composed(learned_code_fields=(20, 30))
    with pytest.raises(ValueError, match="positive field widths"):
        composed(learned_code_fields=(53, 0))
    with pytest.raises(ValueError, match="two units per field"):
        composed(units=4, learned_code_fields=(20, 20, 13))
    with pytest.raises(ValueError, match="at least 2 units"):
        composed(units=1)
    with pytest.raises(ValueError, match="learned_code must be an integer"):
        composed(units=2.5)
    with pytest.raises(ValueError, match="need learned_code=units"):
        cd.Brain.compose(53, 4, modules=(8,), learned_code_fields=(53,))
    with pytest.raises(TypeError, match="unknown brain setting"):
        composed(learned_code_speed=0.1)
    with pytest.raises(ValueError, match="joins the observation itself"):
        composed(code=64)
    with pytest.raises(ValueError, match="a vector brain"):
        cd.Brain.compose((8, 8), 2, modules=(8,), learned_code=32)
    episodic = composed(memory_choice=True)  # the stores separate the code's keys
    assert episodic.hippocampus.separator is not None
    assert episodic.hippocampus.separator.expansion == 1024
    assert episodic.context_memory.separator.expansion == 1024
    brain = composed()
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(learned_code=32)
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(learned_code_fields=(53,))
    # a custom anatomy needs the region and its complete synapses
    regions = [Region("sensory", 8), cortex(6, lateral=-1.0, name="code"),
               Region("association", 8), prefrontal_cortex(8), motor_cortex(2)]
    projections = [Projection("sensory", "code", reciprocal=False),
                   Projection("code", "association", reciprocal=False),
                   Projection("association", "motor"),
                   Projection("prefrontal", "association", reciprocal=False)]
    graph = develop(Genome(tuple(regions), tuple(projections)), seed=1)
    with pytest.raises(ValueError, match="complete predictor synapses"):
        cd.Brain(graph, learned_code=6)
    with pytest.raises(ValueError, match="needs a population named 'code'"):
        cd.Brain(cd.Brain.compose(8, 2, modules=(8,)).connectome, learned_code=6)
    complete = develop(Genome(tuple(regions), (*projections, Projection(
        "prefrontal", "code", scale=0.0, reciprocal=False))), seed=1)
    custom = cd.Brain(complete, learned_code=6)
    # 8 inputs, wider than the code's 6 units: the memories key on the code alone
    assert custom.learned_code is not None and custom.code_width == 6
    beside = develop(Genome(tuple(regions), (*projections, Projection(
        "prefrontal", "code", scale=0.0, reciprocal=False), Projection(
        "sensory", "association", reciprocal=False))), seed=1)
    with pytest.raises(ValueError, match="project to the code region alone"):
        cd.Brain(beside, learned_code=6)
    with pytest.raises(ValueError, match="is not the learned code of 5 units"):
        cd.Brain(complete, learned_code=5)


def settled_code(brain: cd.Brain) -> np.ndarray:
    return np.atleast_2d(brain.basal_ganglia.state.activation)[:, brain.code_index]


def test_the_code_is_a_fixed_point_of_the_whole_settle_and_the_memories_key():
    """After an act the whole graph's residual holds on the code units; the free state's code
    is the phase the memories keyed on, within the tolerance; the key is the rectified code
    at unit length; the reconstruction error fell over the phase's sweeps."""
    # the region's rates frozen: its learning after the moment would change the equations
    # the settled state is checked against
    brain = composed(recall=True)
    brain.learned_code.learning = False
    x = tokens(3, 7, 11)
    brain.act(x, greedy=True)
    report = brain.last_settlement
    assert report["qualified"] and report["code_steps"] > 0
    assert report["code_residual"] <= brain.learner.config.tolerance
    code = brain.learned_code
    s = settled_code(brain)
    sparse = code.code_of(s)
    assert (sparse >= 0).all() and np.allclose(np.linalg.norm(sparse, axis=1), 1.0)
    assert ((sparse > 0).sum(axis=1) >= 1).all() and ((sparse > 0).sum(axis=1) < UNITS // 2).all()
    key = brain._moment_key(x, sparse)  # joined to the observation's identity
    assert key.shape == (3, UNITS + INPUTS) and np.allclose(np.linalg.norm(key, axis=1), 1.0)
    np.testing.assert_allclose(brain.recall.keys[:, -1, :], key, atol=1e-3)
    np.testing.assert_allclose(brain.recall.values[:, -1, :], key, atol=1e-3)
    # the whole graph's equations hold on the code units at the settled state
    drive = brain.stimulus(x)
    state = brain.basal_ganglia.state
    graph = brain.brain
    total = graph._synaptic_input(np.atleast_2d(state.activation)) + drive + graph.bias
    residual = np.abs(total - np.atleast_2d(state.v))[:, brain.code_index].max()
    assert residual <= brain.learner.config.tolerance
    # the code phase's own fixed point: settled again under the same stimulus it stays
    phase = brain._settle_code(x)
    np.testing.assert_allclose(phase.state.activation, s, atol=1e-3)
    # the reconstruction error falls over the phase's sweeps, from rest to the fixed point
    last = code.last
    stimulus = code.stimulus(last["sensory"], last["context"])
    trajectory = code.graph.settle_batch(stimulus, steps=phase.steps or 32, trajectory=True)
    assert trajectory.trajectory is not None
    errors = [
        code.residuals(last["sensory"], a, None)["bottom_up"].mean()
        for a in (np.zeros_like(s), *trajectory.trajectory)
    ]
    assert errors[-1] < errors[0] and errors[-1] < 1.0
    # context-free: the public code reads the observation alone, in the same coordinates
    free = brain.observation_code(x)
    assert free.shape == key.shape and (free >= 0).all()


def test_the_region_learns_every_moment_on_its_residuals_and_homeostasis_moves_thresholds():
    brain = composed()
    code = brain.learned_code
    before = {name: value.copy() for name, value in code.arrays().items()}
    x = tokens(1, 2, 5, 9)
    brain.act(x, greedy=True)  # one accepted moment: one step
    assert code.work["updates"] == 1 and code.work["update_rows"] == 4
    last = code.last
    a = np.maximum(last["activation"], 0.0)
    active = (a > 0).any(axis=0)
    # the dictionary moved on the residual, in the columns of active units alone
    residual = code.read * last["sensory"] - a @ before["dictionary"].T
    expected = before["dictionary"] + code.rate * (residual.T @ a)
    expected /= np.linalg.norm(expected, axis=0, keepdims=True)
    # the step reads the sensory activation the settle holds, within the solver's tolerance
    # of its fixed point under a halved step
    np.testing.assert_allclose(code.dictionary, expected, atol=1e-6)
    np.testing.assert_array_equal(code.dictionary[:, ~active], before["dictionary"][:, ~active])
    assert not np.array_equal(code.dictionary[:, active], before["dictionary"][:, active])
    # the predictor moved by the delta rule on the top-down residual, bounded per row
    z = last["context"]
    rate = code.predictor / np.maximum(np.sum(z * z, axis=1), code.predictor**2)
    np.testing.assert_allclose(
        code.weights, before["weights"] + ((a - z @ before["weights"].T) * rate[:, None]).T @ z,
        atol=1e-12,
    )
    # the thresholds: a unit active in more rows than the target rises, a silent one falls;
    # the share reads the context-free code (the code itself at a stream's first moment,
    # within the solver's tolerance), and the repair what both codes leave unexplained
    free = last["free"]
    np.testing.assert_allclose(free, a, atol=1e-5)
    share = (free > code.active).sum(axis=0) - 4 * code.target
    x_read = code.read * last["sensory"]
    repair = code.unexplained(x_read, a) + code.unexplained(x_read, free)
    np.testing.assert_allclose(
        code.thresholds,
        np.maximum(before["thresholds"] + code.threshold_rate * (share - code.silence * repair),
                   0.0),
    )
    assert (code.thresholds[~active] < before["thresholds"][~active]).all()
    # the region's sub-network took the tables; the whole graph's region synapses stay zero
    np.testing.assert_allclose(code.graph.efficacy, code.lateral_efficacy())
    np.testing.assert_array_equal(code.graph.bias, -code.thresholds)
    for synapses, _, _ in brain._code_synapses.values():
        assert not brain.brain.efficacy[synapses].any()
    np.testing.assert_array_equal(brain.brain.bias[brain.code_index], -code.thresholds)
    # a moment the code already fits moves the dictionary little: the residual is the step
    for _ in range(40):
        brain.act(x, greedy=True)
    settled = {name: value.copy() for name, value in code.arrays().items()}
    brain.act(x, greedy=True)
    assert np.abs(code.dictionary - settled["dictionary"]).max() < 1e-2
    # the learning switch holds the tables still
    code.learning = False
    frozen = {name: value.copy() for name, value in code.arrays().items()}
    brain.act(tokens(4, 4, 4, 4), greedy=True)
    for name, value in frozen.items():
        np.testing.assert_array_equal(getattr(code, name), value)
    # the thresholds never fall below zero
    low = composed(seed=3)
    low.learned_code.thresholds[:] = 0.0
    low.learned_code.threshold_rate = 0.5
    low.learned_code.install()
    low.learner.brain = low._installed(low.brain)
    low.act(tokens(1), greedy=True)
    assert (low.learned_code.thresholds >= 0.0).all()


@pytest.mark.parametrize("batch", (1, 16, 32, 64, 128))
def test_repeated_contexts_contract_predictor_error_at_every_batch_size(batch):
    """A repeated normalized LMS context has one analytic error mode. Its coefficient
    must stay in [0, 1], including for nonzero initial weights and batches above 16 rows.
    """
    code = cd.LearnedCode(4, 8, 3, cd.learning_neuron_model(), seed=2)
    code.rate = code.threshold_rate = 0.0  # isolate the predictor from other local rules
    code.weights[:] = np.random.default_rng(7).normal(size=code.weights.shape)
    context = np.array([[0.5, -0.25, 0.75]])
    target = np.random.default_rng(8).random((1, code.units))
    rows = np.repeat(context, batch, axis=0)
    activation = np.repeat(target, batch, axis=0)
    alpha = min(batch * code.predictor, 1.0)
    for _ in range(4):
        before = code.prediction(context)
        error = np.linalg.norm(target - before)
        code.learn(np.zeros((batch, code.inputs)), activation, rows)
        after = code.prediction(context)
        np.testing.assert_allclose(after, before + alpha * (target - before), atol=1e-14)
        assert np.linalg.norm(target - after) <= error + 1e-14


def test_a_large_batch_of_continuing_brains_keeps_its_predictor_bounded():
    brain = cd.Brain.compose(4, 2, modules=(8,), learned_code=8, seed=2)
    x = np.tile([1.0, 0.0, 0.0, 0.0], (128, 1))
    for _ in range(30):
        brain.act(x, greedy=True)
        assert np.isfinite(brain.learned_code.weights).all()
        assert np.linalg.norm(brain.learned_code.weights) < 10.0


def test_routine_live_and_wait_keep_the_code_tables_still_until_aroused(tmp_path):
    brain = composed(arousal=True, arousal_youth=0, arousal_decay=0.0, arousal_need=1.0)
    code = brain.learned_code
    tables = {name: value.copy() for name, value in code.arrays().items()}
    efficacy, bias = brain.brain.efficacy.copy(), brain.brain.bias.copy()

    brain.live(tokens(1))
    assert brain.last_arousal["mode"] == "routine" and brain.pending_feedback
    brain.wait(tokens(2))
    brain.wait(tokens(3))
    resumed = cd.Brain.load(brain.save(tmp_path / "routine.npz"))
    resumed.wait(tokens(4))
    assert resumed.learned_code.work["updates"] == 0
    for name, value in tables.items():
        np.testing.assert_array_equal(getattr(resumed.learned_code, name), value)
    assert code.work["updates"] == 0
    for name, value in tables.items():
        np.testing.assert_array_equal(getattr(code, name), value)
    np.testing.assert_array_equal(brain.brain.efficacy, efficacy)
    np.testing.assert_array_equal(brain.brain.bias, bias)

    # The unmet need wakes this same acquired life; its next accepted answer and wait learn.
    brain.live(tokens(5), reward=[0.0])
    assert brain.last_arousal["mode"] == "aroused" and code.work["updates"] == 1
    assert any(not np.array_equal(getattr(code, name), value) for name, value in tables.items())
    brain.wait(tokens(6))
    assert code.work["updates"] == 2

    # Explicit controls keep their every-moment rule, including a greedy act and the wait
    # following an action sampled by step rather than a routine live decision.
    explicit = composed(arousal=True, arousal_youth=0)
    explicit.act(tokens(1), greedy=True)
    assert explicit.learned_code.work["updates"] == 1
    explicit.step(tokens(2))
    explicit.wait(tokens(3))
    assert explicit.learned_code.work["updates"] == 3


def test_the_lesson_and_the_outcome_leave_the_region_to_its_own_rule():
    brain = composed(recall=True)
    x = tokens(2, 6)
    brain.act(x, greedy=True)
    code = brain.learned_code
    tables = {name: value.copy() for name, value in code.arrays().items()}
    efficacy = brain.brain.efficacy.copy()
    report = brain.lesson(x, [1, 3])  # taught rows: the contrast moves the plastic synapses
    assert report["accepted_presentations"] > 0
    plastic = brain.learner.plastic_synapses
    assert (brain.brain.efficacy[plastic] != efficacy[plastic]).any()
    np.testing.assert_array_equal(brain.brain.efficacy[~plastic], efficacy[~plastic])
    for name, value in tables.items():
        np.testing.assert_array_equal(getattr(code, name), value)  # a lesson is no moment
    brain.step(x)  # a sampled act: one moment, then its outcome
    moved = {name: value.copy() for name, value in code.arrays().items()}
    assert not np.array_equal(moved["dictionary"], tables["dictionary"])
    efficacy = brain.brain.efficacy.copy()
    brain.learn(np.array([1.0, 0.0]), np.array([False, False]), x)
    np.testing.assert_array_equal(brain.brain.efficacy[~plastic], efficacy[~plastic])
    for name, value in moved.items():
        np.testing.assert_array_equal(getattr(code, name), value)


def test_the_key_of_a_decision_is_its_moments_code_and_the_history_hears_the_code():
    brain = composed(recall=True, history_decays=(0.5,), arousal=True, memory_choice=True)
    assert len(brain.connectome.populations["history"]) == UNITS  # it hears the code part
    x = tokens(5)
    brain.live(x)
    first = brain._moment_key(x, brain.learned_code.code_of(settled_code(brain)))
    key = brain.recall.keys[0, 0]  # the moment's key: the phase's code in the stream's context
    w = brain.learned_code.identity
    heard = key[:UNITS] / np.sqrt(1.0 - w)  # the history hears the key's code part at unit length
    np.testing.assert_allclose(brain.history.trace, 0.5 * heard[None, :], atol=1e-9)
    assert brain._lived[6] is not None
    # the keys are the phase's code; the whole settle moves the code units within tolerance
    np.testing.assert_allclose(brain._lived[6], first, atol=1e-3)
    np.testing.assert_allclose(key, first[0], atol=1e-3)
    brain.live(tokens(9), reward=[1.0])  # the outcome wakes the brain: recorded under the code
    assert brain.hippocampus.writes >= 1
    keyed = brain.hippocampus.recall(first)
    assert np.abs(keyed).max() > 0
    # a sampled act keeps the code it was keyed on until its outcome
    stepped = composed(seed=4)
    stepped.step(tokens(1, 2))
    assert stepped._moment_code is not None and stepped._moment_code.shape == (2, UNITS + INPUTS)
    stepped.step(tokens(3, 4), reward=[1.0, 0.0])
    assert stepped._moment_code is not None  # the next sampled act's


def test_a_checkpoint_carries_the_region_and_a_tampered_one_is_refused(tmp_path):
    brain = composed(recall=True, arousal=True, memory_choice=True, history_decays=(0.3,))
    x = [tokens(k) for k in (1, 4, 7)]
    for t in range(5):
        brain.live(x[t % 3], reward=None if t == 0 else [float(t % 2)])
    brain.wait(x[1])
    path = brain.save(tmp_path / "learned.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = earlier({name: data[name].copy() for name in data.files})
    meta = metadata(arrays)
    assert meta["format"] == "cadence-generic/10"
    assert meta["learned_code"]["units"] == UNITS and meta["composition"]["learned_code"] == UNITS
    assert meta["history"]["source"] == "code" and meta["recall"]["inputs"] == UNITS + INPUTS
    assert arrays["learned_code/dictionary"].shape == (INPUTS, UNITS)
    assert arrays["learned_code/weights"].shape == (UNITS, 64)
    assert arrays["learned_code/thresholds"].shape == (UNITS,)
    assert arrays["lived/code"].shape == (1, UNITS + INPUTS)  # the moment's key
    twin = cd.Brain.load(path)
    assert twin.describe() == brain.describe()
    for name, value in brain.learned_code.arrays().items():
        np.testing.assert_array_equal(getattr(twin.learned_code, name), value)
    np.testing.assert_array_equal(twin.brain.efficacy, brain.brain.efficacy)
    assert checkpoint_digest(twin.save(tmp_path / "again.npz")) == checkpoint_digest(path)
    for t in range(3):
        np.testing.assert_array_equal(
            brain.live(x[t], reward=[0.5]), twin.live(x[t], reward=[0.5])
        )
    assert checkpoint_digest(brain.save(tmp_path / "a.npz")) == checkpoint_digest(
        twin.save(tmp_path / "b.npz")
    )
    # a sampled act awaiting its outcome saves the code it was keyed on
    sampled = composed(seed=2)
    sampled.step(tokens(2, 3))
    kept = sampled.save(tmp_path / "pending.npz")
    with np.load(kept, allow_pickle=False) as data:
        assert data["moment/code"].shape == (2, UNITS + INPUTS)
    loaded = cd.Brain.load(kept)
    np.testing.assert_array_equal(loaded._moment_code, sampled._moment_code)
    np.testing.assert_array_equal(
        loaded.step(tokens(5, 6), reward=[1.0, 0.0]), sampled.step(tokens(5, 6), reward=[1.0, 0.0])
    )

    def tampered(name: str, mutate) -> Path:
        copy = {k: v.copy() for k, v in arrays.items()}
        meta = metadata(copy)
        mutate(meta, copy)
        copy["generic"] = np.array(json.dumps(meta))
        np.savez(tmp_path / f"{name}.npz", **copy)
        return tmp_path / f"{name}.npz"

    def no_region(meta, _):
        del meta["learned_code"]

    def earlier_format(meta, _):
        meta["format"] = "cadence-generic/9"

    def other_units(meta, _):
        meta["learned_code"]["units"] = 32

    def unknown_field(meta, _):
        meta["learned_code"]["fan_in"] = 8

    def bad_gene(meta, _):
        meta["learned_code"]["target"] = 2.0

    def no_table(meta, copy):
        del copy["learned_code/weights"]

    def wrong_shape(meta, copy):
        copy["learned_code/thresholds"] = copy["learned_code/thresholds"][:-1]

    def negative_threshold(meta, copy):
        copy["learned_code/thresholds"][0] = -0.1

    def long_column(meta, copy):
        copy["learned_code/dictionary"][:, 0] *= 2.0

    def held_synapse(meta, copy):  # a held code's synapses in the whole graph carry nothing
        copy["efficacy"][brain._code_synapses["dictionary"][0][0]] = 0.5

    def lone_code(meta, copy):
        del meta["lived"]
        del copy["lived/observations"]
        del copy["lived/action"]

    for name, mutate, message in (
        ("no_region", no_region, "belongs to checkpoint format cadence-generic/10"),
        ("earlier_format", earlier_format, "belongs to checkpoint format cadence-generic/10"),
        ("other_units", other_units, "is not the saved learned code of 32 units"),
        ("unknown_field", unknown_field, "invalid saved learned code metadata"),
        ("bad_gene", bad_gene, "invalid saved learned code metadata"),
        ("no_table", no_table, "missing saved learned code table: weights"),
        ("wrong_shape", wrong_shape, "invalid saved learned code table: thresholds"),
        ("negative_threshold", negative_threshold, "thresholds are nonnegative"),
        ("long_column", long_column, "unit-length columns"),
        ("held_synapse", held_synapse, "disagree with the saved synapses"),
        ("lone_code", lone_code, "saved lived code belongs to a lived action"),
    ):
        with pytest.raises(ValueError, match=message):
            cd.Brain.load(tampered(name, mutate))


def test_a_code_phase_that_cannot_settle_refuses_the_moment_before_anything_changes():
    brain = composed(seed=5, learning_free_steps=1)
    x = tokens(1, 2)
    before = brain.basal_ganglia.state
    with pytest.raises(RuntimeError, match="code region did not settle"):
        brain.act(x, greedy=True)
    assert brain.basal_ganglia.state is before and brain._coded_moment is None
    # the phase at the brain's step and once more at the step the dictionary allows, both
    # capped; the brain's step is left as it was
    assert brain.learned_code.work["capped"] == 2 and brain.learned_code.work["updates"] == 0
    assert brain.brain.neuron_model.dt == 1.0
    assert brain.last_settlement is None


def test_independent_samples_settle_the_context_free_code():
    brain = composed(seed=6, units=24, episodic=False)
    x = np.eye(INPUTS)[:6]
    labels = np.arange(6) % ACTIONS
    assert brain.predict(x).shape == (6,)
    assert 0.0 <= brain.accuracy(x, labels) <= 1.0
    history = brain.fit(x, labels, epochs=2)
    assert len(history) == 2
    assert brain.learned_code.work["updates"] == 0  # no accepted moment: no step
    np.testing.assert_array_equal(brain.observation_code(x), brain.observation_code(x))


def test_a_one_input_world_settles_at_the_step_its_dictionary_allows():
    """With one input every column of the dictionary is plus or minus the same vector, so
    every pair of units competes at full strength and the sweep at the model's step cannot
    settle the code. The first phase is settled once more at the step the dictionary's
    coupling allows over every state, and the bandit lives on without a refusal; the bound
    over every state is at least the bound at any settled state."""
    for seed in range(6):
        brain = cd.Brain.compose(1, 2, arousal=True, learned_code=64, seed=seed)
        code = brain.learned_code
        assert code.rest_stability() >= 4
        rng = np.random.default_rng(seed)
        for moment in range(100):
            reward = None if moment == 0 else [float(rng.random() < 0.7)]
            brain.live(np.ones((1, 1)), reward=reward)  # never refused
        # the code's phase steps alone; the brain keeps the model's step
        assert code.halvings >= 3 and brain.brain.neuron_model.dt == 1.0
        assert brain.brain.neuron_model.dt == 1.0
    # the bound over every state holds at states the region reaches and at random ones
    code = composed(seed=3).learned_code
    for v in np.random.default_rng(4).normal(0.0, 0.5, (20, 1, UNITS)):
        assert code.stability(v) <= code.rest_stability()


def test_a_brain_with_the_region_deep_copies_and_imagines():
    """``copy.deepcopy`` of a brain with the region: the copy shares the immutable
    connectomes and holds its own tables, state and generators, acts as the original acts,
    learns apart from it, and imagines."""
    brain = composed(seed=5, recall=True, arousal=True, memory_choice=True)
    x = tokens(3, 9)
    brain.act(x, greedy=True)
    twin = copy.deepcopy(brain)
    assert twin.connectome is brain.connectome
    assert twin.learned_code is not brain.learned_code
    assert twin.learned_code.connectome is brain.learned_code.connectome
    assert twin.learned_code.dictionary is not brain.learned_code.dictionary
    assert dict(twin.last_settlement) == dict(brain.last_settlement)
    np.testing.assert_array_equal(twin.act(x, greedy=True), brain.act(x, greedy=True))
    before = brain.learned_code.dictionary.copy()
    twin.act(tokens(7, 11), greedy=True)  # the copy learns; the original does not move
    np.testing.assert_array_equal(brain.learned_code.dictionary, before)
    phases = twin.imagine([tokens(5, 6), tokens(8, 2)])
    assert len(phases) == 2 and all(np.all(phase.qualified) for phase in phases)
    plain = cd.Brain.compose(INPUTS, ACTIONS, modules=(8,), seed=1)  # and without the region
    plain.act(x, greedy=True)
    np.testing.assert_array_equal(copy.deepcopy(plain).act(x, greedy=True), plain.act(x, greedy=True))


def test_an_imagined_branch_leaves_the_region_and_the_checkpoint_as_they_were(tmp_path):
    """An imagined branch settles the region as a real moment does, at the step its
    competition needs, and leaves the region's step, its work counts and its latest phase as
    it found them: the checkpoint saved after the branch is the one saved before it, byte for
    byte, and the brain lives on as its copy that did not imagine. Three streams of a small
    brain, at the step of its life and at a step the imagined moments do not need."""
    rewards = [1.0, 0.0, -1.0, 0.0, 1.0, 0.5, 0.0, -0.5, 0.25, 0.0, 1.0, 0.0]

    def moment(m: int) -> np.ndarray:  # a token per stream and a little noise
        x = np.zeros((3, 12))
        for row in range(3):
            x[row, (m + 3 * row) % 12] = 1.0
        return x + 0.01 * np.random.default_rng(1000 + m).random((3, 12))

    def live(brain: cd.Brain, start: int, end: int) -> list:
        acts = []
        for m in range(start, end):
            if m == 0:
                acts.append(brain.step(moment(m)).tolist())
                continue
            reward = [rewards[(m + 5 * row) % 12] for row in range(3)]
            done = [(m + 2 * row) % 7 == 6 for row in range(3)]
            acts.append(brain.step(moment(m), reward=reward, done=done).tolist())
        return acts

    brain = cd.Brain.compose(12, 4, learned_code=16, memory_choice=True, recall=True,
                             history_decays=(0.2, 0.8), efference_amplitude=1.0, seed=5)
    live(brain, 0, 13)
    region = brain.learned_code
    for level in (region.halvings, region.halvings + 3):
        brain._step_region(level)
        twin = copy.deepcopy(brain)
        before = brain.save(tmp_path / "before.npz").read_bytes()
        described, last, steps = region.to_dict(), region.last, []
        def spy(halvings: int, steps: list = steps, restep=brain._step_region) -> None:
            steps.append(halvings)  # every step the branch's code phases take
            restep(halvings)

        brain._step_region = spy
        try:
            phases = brain.imagine([moment(100 + k) for k in range(3)])
        finally:
            del brain._step_region
        assert len(phases) == 3 and all(np.all(phase.qualified) for phase in phases)
        assert region.to_dict() == described and region.last is last
        assert brain.save(tmp_path / "after.npz").read_bytes() == before
    assert steps and any(step != region.halvings for step in steps)  # the branch moved it
    assert live(brain, 13, 18) == live(twin, 13, 18)


def test_every_memory_keys_on_the_context_free_code():
    """Every memory keys on the context-free code, the observation settled alone (a second
    code phase, reported as ``context_free_steps``), so a known situation is found again
    along a new path; the region learns from the code in context; a decision keeps the key
    it was taken under."""
    brain = composed(seed=9, recall=True, memory_choice=True, arousal=True)
    code = brain.learned_code
    # a predictor with weight: at birth it is zero and the two phases coincide
    code.weights[:] = 0.2 * np.random.default_rng(3).random(code.weights.shape)
    code.install()
    brain.learner.brain = brain._installed(brain.brain)
    # the tables held still, so the moment's key and a later context-free settle agree
    code.learning = False
    context, token = tokens(40), tokens(3)
    brain.live(context)
    before = brain.memories["episodic"].context.copy()
    brain.live(token, reward=[0.0])
    report = brain.last_settlement
    assert report["context_free_steps"] >= 0 and "code_steps" in report
    coded = brain._coded_moment
    assert coded is not None and coded[0] is not None
    key, place = coded[1], coded[3]
    assert key is place  # one key: the context-free code
    # the context predicted units the context-free code leaves silent
    assert not np.allclose(code.last["activation"], code.last["free"])
    np.testing.assert_allclose(place, brain.observation_code(token), atol=5e-3)
    np.testing.assert_array_equal(brain.recall.keys[0, -1], key[0])  # the slot
    np.testing.assert_array_equal(brain._lived[6], place)  # the decision's place key
    np.testing.assert_array_equal(brain.memories["episodic"].paths[0][-1][2], place[0])  # the path
    width = brain.code_width
    np.testing.assert_allclose(  # the context key: the moment's key with the history
        brain.memories["episodic"].decision[0, :width], 0.5 * before[0, :width] + 0.5 * key[0], atol=1e-12
    )
    brain.live(tokens(44), reward=[1.0])  # the outcome wakes the brain and is recorded
    assert brain.hippocampus.writes >= 1
    assert np.abs(brain.hippocampus.recall(place)).max() > 0
    # the same token in another context: the same key
    brain.live(tokens(41), reward=[0.0])
    brain.live(token, reward=[0.0])
    np.testing.assert_allclose(brain._coded_moment[1], place, atol=5e-3)
    # without a place store the context-free phase runs all the same: the cortex reads it
    bare = composed(seed=9, episodic=False)
    bare.act(token, greedy=True)
    assert bare.last_settlement["context_free_steps"] > 0
    assert bare._coded_moment[3] is bare._coded_moment[1]
    # a step, its outcome and a lesson
    taught = composed(seed=10, memory_choice=True)
    taught.step(tokens(1, 2))
    np.testing.assert_allclose(taught._moment_code, taught.observation_code(tokens(1, 2)), atol=5e-3)
    taught.step(tokens(3, 4), reward=[1.0, 0.0])
    taught.act(tokens(5, 6), greedy=True)  # a greedy act replaces the pending one
    taught.lesson(tokens(5, 6), [1, 2])
    assert taught.hippocampus.writes >= 2


def test_a_long_stream_leaves_the_code_alive(tmp_path):
    """Six tokens heard for 2,000 moments by one stream, a lesson every tenth moment: the
    thresholds stay finite and nonnegative, every token keeps at least one active unit in
    context and context-free, and at least one unit is in use per token; a checkpoint with
    a negative threshold is refused."""
    brain = composed(seed=11, units=32, recall=True)
    code = brain.learned_code
    code.threshold_rate = 0.01
    rng = np.random.default_rng(5)
    for t in range(2000):
        token = int(rng.integers(6))
        x = tokens(token)
        if t % 10 == 9:
            brain.lesson(x, [token % ACTIONS])
        brain.act(x, greedy=True)
    assert np.isfinite(code.thresholds).all() and (code.thresholds >= 0.0).all()
    used = np.zeros(32, bool)
    for token in range(6):
        x = tokens(token)
        brain.act(x, greedy=True)
        in_context = np.maximum(code.last["activation"], 0.0)
        free = brain.observation_code(x)[:, :32]  # the key's code part
        assert (in_context > 0).sum() >= 1 and (free > 0).sum() >= 1
        used |= (free > 0)[0]
    assert used.sum() >= 6
    np.testing.assert_array_equal(brain.brain.bias[brain.code_index], -code.thresholds)
    path = brain.save(tmp_path / "alive.npz")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["learned_code/thresholds"][0] = -0.1
    np.savez(tmp_path / "below.npz", **arrays)
    with pytest.raises(ValueError, match="thresholds are nonnegative"):
        cd.Brain.load(tmp_path / "below.npz")


def test_the_repair_lowers_the_silent_units_of_rows_the_code_leaves_unexplained():
    """At a repair level the newborn's rows exceed (its random dictionary explains little),
    one moment lowers the thresholds of the units silent in those rows by the silence term
    beyond the share rule, and leaves the units that fired to the share rule; at the
    founder level 1 the same moment, whose rows are not silent, moves nothing beyond it."""
    x = tokens(1, 2, 5, 9)
    strict = composed(seed=4)
    strict.act(x, greedy=True)
    code = strict.learned_code
    free = code.last["free"]
    share = (free > code.active).sum(axis=0) - len(x) * code.target
    np.testing.assert_allclose(
        code.thresholds, np.maximum(code.threshold + code.threshold_rate * share, 0.0)
    )
    repairing = composed(seed=4)
    repairing.learned_code.repair = 0.5
    repairing.act(x, greedy=True)
    code = repairing.learned_code
    a = np.maximum(code.last["activation"], 0.0)
    left = code.residuals(code.last["sensory"], a, None)["bottom_up"]
    rows = left >= 0.5
    assert rows.any()
    free = code.last["free"]
    x_read = code.read * code.last["sensory"]
    repair = code.unexplained(x_read, a) + code.unexplained(x_read, free)
    np.testing.assert_array_equal(code.unexplained(x_read, a), (rows[:, None] & ~(a > 0)).sum(0))
    share = (free > code.active).sum(axis=0) - len(x) * code.target
    expected = code.threshold + code.threshold_rate * (share - code.silence * repair)
    np.testing.assert_allclose(code.thresholds, np.maximum(expected, 0.0))
    fired = (a > 0).any(axis=0)
    assert (code.thresholds[~fired] < code.threshold).all()
    assert code.work["repairs"] == int(repair.sum()) > 0
    assert strict.learned_code.work["repairs"] == 0  # no row was silent at the strict level


def test_the_key_joins_the_code_to_the_observations_identity():
    """The memories' key is ``unit([sqrt(1 - w) code, sqrt(w) unit(x)])`` at the region's
    identity weight ``w`` (.9): the identity part is the observation at unit length weighted
    by ``sqrt(w)``, the code part the rectified settled code."""
    brain = composed(seed=2, recall=True)
    w = cd.LearnedCode.IDENTITY
    x = tokens(4, 9, 9)
    brain.act(x, greedy=True)
    key = brain.recall.keys[:, -1, :]
    assert key.shape == (3, UNITS + INPUTS)
    np.testing.assert_allclose(key[:, UNITS:], np.sqrt(w) * x, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(key[:, :UNITS], axis=1), np.sqrt(1 - w), atol=1e-3)
    np.testing.assert_allclose(key[1], key[2], atol=1e-3)  # the same token, the same key
    assert brain.observation_code(x).shape == (3, UNITS + INPUTS)


def test_the_module_reads_the_code_at_unit_length():
    """The association neurons' input from the code units is the code at unit length
    through the forward synapses: the code's synaptic input plus the term the moment's
    drive carries."""
    brain = composed(seed=6)
    x = tokens(3, 8)
    terms = []  # the term the moment's settle carries on the association neurons
    unit_read = brain._unit_read

    def recorded(x_, drive, *, memory, index=brain.association_index):
        out = unit_read(x_, drive, memory=memory)
        terms.append((out - drive)[:, index])
        return out

    brain._unit_read = recorded
    brain.act(x, greedy=True)
    state = brain.basal_ganglia.state
    synapses, j, m = brain._code_forward
    assert len(synapses) == UNITS * 64
    weights = np.zeros((64, UNITS))
    weights[m, j] = brain.brain.efficacy[synapses]
    s = np.atleast_2d(state.activation)[:, brain.code_index]
    phase = brain.learned_code.last["activation"]  # the moment's code phase
    np.testing.assert_allclose(s, phase, atol=1e-3)  # the code units hold its fixed point
    assert terms and brain.last_settlement["qualified"]
    read = s @ weights.T + terms[0]  # the code's synaptic input plus the term
    expected = brain.learned_code.code_of(phase) @ weights.T
    np.testing.assert_allclose(read, expected, atol=1e-3)


@pytest.mark.parametrize("contacts, log_gains", [
    (False, False), (True, False), (False, True), (True, True),
], ids=["unit", "contacts", "log_gain", "combined"])
def test_the_unit_code_read_uses_effective_forward_weights(contacts, log_gains, monkeypatch):
    """Source activity plus the drive correction reads unit code through the actual
    transport weights, including contact counts and presynaptic gains."""
    brain = cd.Brain.compose(8, 2, modules=(6,), learned_code=4, seed=6)
    original = brain.brain
    synapses, _, _ = brain._code_forward
    counts = original.connectome.count.copy()
    if contacts:
        counts[synapses] = np.linspace(1.25, 2.5, len(synapses))
    connectome = replace(original.connectome, count=counts)
    np.testing.assert_array_equal(connectome.pre, original.connectome.pre)
    np.testing.assert_array_equal(connectome.post, original.connectome.post)
    efficacy, log_gain = original.efficacy.copy(), original.log_gain.copy()
    efficacy[synapses] = np.linspace(-0.4, 0.6, len(synapses))
    if log_gains:
        log_gain[brain.code_index] = np.log([1.2, 1.5, 1.8, 2.1])
    graph = cd.NeuralGraph(
        connectome, original.neuron_model, efficacy=efficacy,
        bias=original.bias, log_gain=log_gain,
    )
    brain.learner.brain = graph
    brain.learned_code.dictionary[:] = np.eye(8, 4)
    brain.learned_code.install()
    observations = np.eye(8)[[1, 4]]
    brain._prepare(observations, context=False)
    phase = brain._coded_moment[2]
    activity = np.atleast_2d(phase.activation)
    assert brain._code_settlement["code_residual"] <= brain.learner.config.tolerance
    code = brain.learned_code.code_of(activity)
    assert (activity[0] > 0).any() and (activity[0] < 0).any()
    assert (activity[1] < 0).all() and not code[1].any()
    np.testing.assert_allclose(np.linalg.norm(code[0]), 1.0, rtol=0, atol=1e-14)

    def forbidden(*args, **kwargs):
        pytest.fail("the source-read regression attempted an extra code solve")

    monkeypatch.setattr(brain, "_settle_code", forbidden)
    drive = np.random.default_rng(43).normal(size=(2, connectome.n))
    corrected = brain._unit_read(observations, drive, memory=False)
    weights = graph.neuron_model.gain * connectome.count * np.exp(log_gain)[connectome.pre]
    weights *= efficacy
    np.testing.assert_allclose(graph.weights, weights, rtol=1e-15, atol=0)
    dense = np.zeros((connectome.n, connectome.n))
    np.add.at(dense, (connectome.pre, connectome.post), weights)
    source = np.zeros_like(drive)
    source[:, brain.code_index] = activity
    actual = (source @ dense + corrected-drive)[:, brain.association_index]
    expected = code @ dense[np.ix_(brain.code_index, brain.association_index)]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_the_unit_code_read_sums_parallel_effective_weights(monkeypatch):
    """A custom raw connectome retains parallel rows; their actual transport sums
    both signed efficacies before the source's unit-code correction."""
    founder = cd.Brain.compose(8, 2, modules=(6,), learned_code=4, seed=6)
    original = founder.connectome
    source, target = founder.code_index[1], founder.association_index[0]
    edge = np.flatnonzero((original.pre == source) & (original.post == target))
    assert len(edge) == 1
    # The direct constructor retains these rows, unlike from_synapses, which merges them.
    connectome = cd.Connectome(
        original.n, np.append(original.pre, source), np.append(original.post, target),
        np.append(original.count, 2.25), np.append(original.sign, -1.0),
        original.populations, original.label,
    )
    parallel = np.flatnonzero((connectome.pre == source) & (connectome.post == target))
    assert len(parallel) == 2 and connectome.synapses == original.synapses + 1
    counts = connectome.count.copy()
    counts[parallel] = [1.75, 2.25]
    connectome = replace(connectome, count=counts)
    brain = cd.Brain.compose(8, 2, connectome=connectome, learned_code=4, seed=6)
    graph = brain.brain
    assert graph.layout.parallel_synapses
    efficacy, log_gain = graph.efficacy.copy(), graph.log_gain.copy()
    efficacy[parallel] = [0.31, -0.17]
    log_gain[brain.code_index] = np.log([1.2, 1.5, 1.8, 2.1])
    graph = graph.with_parameters(efficacy=efficacy, log_gain=log_gain)
    brain.learner.brain = graph
    brain.learned_code.dictionary[:] = np.eye(8, 4)
    brain.learned_code.install()
    observations = np.eye(8)[[1, 4]]
    brain._prepare(observations, context=False)
    activity = np.atleast_2d(brain._coded_moment[2].activation)
    assert brain._code_settlement["code_residual"] <= brain.learner.config.tolerance
    code = brain.learned_code.code_of(activity)
    assert activity[0, 1] > 0 and (activity[0] < 0).any()
    assert (activity[1] < 0).all() and not code[1].any()
    np.testing.assert_allclose(np.linalg.norm(code[0]), 1.0, rtol=0, atol=1e-14)

    def forbidden(*args, **kwargs):
        pytest.fail("the parallel-source regression attempted an extra code solve")

    monkeypatch.setattr(brain, "_settle_code", forbidden)
    drive = np.random.default_rng(44).normal(size=(2, connectome.n))
    corrected = brain._unit_read(observations, drive, memory=False)
    weights = graph.neuron_model.gain * connectome.count * np.exp(log_gain)[connectome.pre]
    weights *= efficacy
    assert weights[parallel[0]] > 0 > weights[parallel[1]]
    dense = np.zeros((connectome.n, connectome.n))
    np.add.at(dense, (connectome.pre, connectome.post), weights)
    assert dense[source, target] == np.sum(weights[parallel])
    raw_source = np.zeros_like(drive)
    raw_source[:, brain.code_index] = activity
    actual = (raw_source @ dense + corrected-drive)[:, brain.association_index]
    expected = code @ dense[np.ix_(brain.code_index, brain.association_index)]
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)


def test_the_key_takes_the_observations_norm():
    """The memories' key is the joined key times the observation's norm, the norm the
    observation would have as a key: a two-hot observation's key has norm sqrt(2), a
    one-hot one's norm 1, and the history hears the code at unit length."""
    x = tokens(3, 9)
    x[1, 20] = 1.0  # the second stream: two inputs at once
    brain = composed(seed=6, recall=True, history_decays=(0.5,))
    brain.act(x, greedy=True)
    keys = brain.recall.keys[:, -1, :]
    np.testing.assert_allclose(np.linalg.norm(keys, axis=1), [1.0, np.sqrt(2.0)], atol=1e-9)
    heard = brain._heard(x, brain.observation_code(x))
    np.testing.assert_allclose(np.linalg.norm(heard, axis=1), [1.0, 1.0], atol=1e-9)


def test_a_held_code_is_a_source_and_the_brain_keeps_the_models_step(tmp_path):
    """With the code held (the founder), nothing leads into the code units in the whole
    graph and the moment's drive holds them at the code phase's fixed point: through the free
    settle and a lesson's phases they publish the phase's code exactly, the brain integrates
    at the model's step with its budgets as configured while the region's phase steps alone,
    no standing boundary acts on the whole graph, and a checkpoint continues the life."""
    brain = composed(seed=3, recall=True)
    code = brain.learned_code
    code.top_down = 4.0  # a stepping phase
    for name in ("dictionary", "lateral", "predictor"):
        synapses = brain._code_synapses[name][0]
        assert not brain.brain.efficacy[synapses].any()
    x = tokens(3, 9)
    for _ in range(4):
        brain.act(x, greedy=True)
    assert code.halvings >= 1  # the phase stepped
    assert brain.brain.neuron_model.dt == 1.0
    assert brain.learner.config.free_steps == 1024
    held = np.atleast_2d(brain.basal_ganglia.state.activation)[:, brain.code_index]
    np.testing.assert_allclose(np.maximum(held, 0.0), code.last["free"], rtol=0, atol=1e-12)
    brain.lesson(x, [1, 2])
    brain.act(x, greedy=True)
    held = np.atleast_2d(brain.basal_ganglia.state.activation)[:, brain.code_index]
    np.testing.assert_allclose(np.maximum(held, 0.0), code.last["free"], rtol=0, atol=1e-12)
    twin = cd.Brain.load(brain.save(tmp_path / "held.npz"))
    assert twin.learned_code.halvings == code.halvings
    assert twin.describe() == brain.describe()
    np.testing.assert_array_equal(twin.act(x, greedy=True), brain.act(x, greedy=True))


def test_the_cortex_and_the_memories_read_the_context_free_code():
    """The code units are held at the context-free code's
    fixed point (the observation's code settled alone), the recall keys are the context-free
    keys, and the region learns from the code in context, the prediction's pull in it."""
    brain = composed(seed=4, recall=True)
    code = brain.learned_code
    x = tokens(2, 6)
    brain.act(x, greedy=True)  # the trace now carries a moment
    code.weights[:] = 1.0  # every unit predicted from any context: the code in context differs
    code.install()
    brain.learner.brain = brain._installed(brain.brain)
    before = {name: value.copy() for name, value in code.arrays().items()}
    brain.act(x, greedy=True)
    held = np.atleast_2d(brain.basal_ganglia.state.activation)[:, brain.code_index]
    np.testing.assert_allclose(np.maximum(held, 0.0), code.last["free"], rtol=0, atol=1e-12)
    # the moment's key is its context-free code (settled before the moment's learning step)
    free_key = brain._moment_key(x, code.code_of(code.last["free"]))
    np.testing.assert_allclose(brain.recall.keys[:, -1, :], free_key, atol=1e-9)
    # the dictionary's step reads the code in context, not the context-free one
    a = np.maximum(code.last["activation"], 0.0)
    assert not np.allclose(a, code.last["free"])
    residual = code.read * code.last["sensory"] - a @ before["dictionary"].T
    fired = (a > 0).any(axis=0)
    expected = before["dictionary"].copy()
    step = expected[:, fired] + code.rate * (residual.T @ a[:, fired])
    expected[:, fired] = step / np.linalg.norm(step, axis=0, keepdims=True)
    np.testing.assert_allclose(code.dictionary, expected, atol=1e-9)


def test_an_observation_wider_than_the_code_is_keyed_on_the_code_alone(tmp_path):
    """The key joins the observation's identity to the code for an observation no wider than
    the code; a wider one is keyed on the code alone, at the observation's norm, and a
    checkpoint keeps that width."""
    assert cd.LearnedCode.joins(53, 64) and cd.LearnedCode.joins(64, 64)
    assert not cd.LearnedCode.joins(65, 64)
    assert composed(seed=2).code_width == UNITS + INPUTS
    wide = composed(seed=2, units=32, recall=True, memory_choice=True, arousal=True)
    assert wide.code_width == 32 and wide.learned_code.identity == 0.0
    assert cd.Genome.observation_width(INPUTS, None, 32) == 32  # the width before building
    x = tokens(3, 8)
    wide.act(x, greedy=True)
    code = wide.learned_code
    key = code.code_of(code.last["free"]) * np.linalg.norm(x, axis=1)[:, None]
    assert wide.recall.keys.shape[-1] == 32
    np.testing.assert_allclose(wide.recall.keys[:, -1, :], key, atol=1e-12)
    # the episodic stores take the same width, and a checkpoint keeps it
    wide.reset()
    for t, k in enumerate((4, 5, 9, 4)):
        wide.live(tokens(k), reward=None if t == 0 else [float(t % 2)])
    assert wide.hippocampus.writes >= 1
    twin = cd.Brain.load(wide.save(tmp_path / "wide.npz"))
    assert twin.code_width == 32 and twin.learned_code.identity == 0.0
    np.testing.assert_array_equal(twin.live(tokens(6), reward=[0.0]), wide.live(tokens(6), reward=[0.0]))


def test_fields_give_each_unit_one_field_of_the_observation(tmp_path):
    """With ``learned_code_fields`` each unit reads one field of the observation's
    coordinates: the units are shared among the fields in proportion to their widths, a
    unit's dictionary column is zero outside its field through every learning step, the
    competition (the Gram's off-diagonal) is zero between fields, a checkpoint continues the
    life and refuses a column that reads another field; one field is the region as without
    fields, bit for bit."""
    brain = composed(seed=3, recall=True, learned_code_fields=(13, 40))
    code = brain.learned_code
    assert code.field_units == (16, 48) and sum(code.field_units) == UNITS
    field = np.repeat([0, 1], code.field_units)
    assert not code.dictionary[~code.mask].any()
    np.testing.assert_array_equal(code.gram()[field[:, None] != field[None, :]], 0.0)
    for t in range(12):
        brain.act(tokens(t % INPUTS, (7 * t) % INPUTS), greedy=True)
    assert code.work["updates"] == 12 and not code.dictionary[~code.mask].any()
    np.testing.assert_allclose(np.linalg.norm(code.dictionary, axis=0), 1.0)
    path = brain.save(tmp_path / "fields.npz")
    twin = cd.Brain.load(path)
    assert twin.learned_code.fields == (13, 40) and twin.learned_code.field_units == (16, 48)
    x = tokens(2, 5)
    np.testing.assert_array_equal(twin.act(x, greedy=True), brain.act(x, greedy=True))
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays["learned_code/dictionary"][0, 20] = 0.1  # unit 20 (field 1) reading input 0 (field 0)
    column = arrays["learned_code/dictionary"][:, 20]
    arrays["learned_code/dictionary"][:, 20] = column / np.linalg.norm(column)
    np.savez(tmp_path / "other.npz", **arrays)
    with pytest.raises(ValueError, match="reads its own fields only"):
        cd.Brain.load(tmp_path / "other.npz")
    one = composed(seed=3, learned_code_fields=(INPUTS,))
    np.testing.assert_array_equal(
        one.learned_code.dictionary, composed(seed=3).learned_code.dictionary
    )


def test_a_unit_below_the_active_floor_counts_as_under_active():
    """A unit that fires below the active floor 0.05 (``LearnedCode.ACTIVE``) counts as
    under-active in the thresholds' share, so a frequent observation's units are not faded
    toward zero; with the floor at 0 any firing counts."""
    code = composed(seed=5).learned_code
    x = np.zeros((2, INPUTS))
    a = np.zeros((2, UNITS))
    a[:, 0] = 0.02  # fires weakly in both rows
    a[:, 1] = 0.30  # fires strongly in both rows
    before = code.thresholds.copy()
    code.learn(x, a, None, free=a)
    step = code.threshold_rate
    assert np.isclose(code.thresholds[0], before[0] - step * 2 * code.target)  # under-active
    assert np.isclose(code.thresholds[1], before[1] + step * 2 * (1 - code.target))
    plain = composed(seed=5).learned_code
    plain.active = 0.0
    plain.learn(x, a, None, free=a)
    assert np.isclose(plain.thresholds[0], before[0] + step * 2 * (1 - plain.target))


def test_the_region_learns_from_the_moments_free_settle_alone():
    """With the working trace silenced (so that the lesson's synaptic changes reach no
    code), a stream of lessons and outcomes leaves every table of the region bit for bit as
    a stream of acts on the same observations leaves it: the nudged phases, the eligibility
    phases and the lesson's settle enter no table."""
    acting = composed(seed=3, recall=True, working_memory_amplitude=0.0)
    taught = composed(seed=3, recall=True, working_memory_amplitude=0.0)
    rng = np.random.default_rng(2)
    for t in range(12):
        x = tokens(*rng.integers(INPUTS, size=3))
        acting.act(x, greedy=True)
        taught.lesson(x, rng.integers(ACTIONS, size=3))
        taught.act(x, greedy=True)
        if t % 4 == 3:  # a sampled act and its outcome, the eligibility phases on taught
            x, following = (tokens(*rng.integers(INPUTS, size=3)) for _ in range(2))
            acting.step(x)
            taught.step(x)
            taught.learn(np.array([1.0, -1.0, 0.5]), np.array([False, False, True]), following)
    # the tables agree to the solver's tolerance (each brain's phases start warm from its
    # own stream's state and settle to the same fixed points within it); a step of any rule
    # from a lesson's phase would be of the order of the rates, a thousand times larger
    for name, value in acting.learned_code.arrays().items():
        np.testing.assert_allclose(getattr(taught.learned_code, name), value, atol=1e-6)
    assert taught.learned_code.work["updates"] == acting.learned_code.work["updates"] == 15


def test_the_thresholds_share_reads_the_context_free_code():
    """A prediction that recruits every unit in context raises no threshold: the share
    counts the units the observation drives on its own."""
    brain = composed(seed=5)
    code = brain.learned_code
    brain.act(tokens(2, 7), greedy=True)  # the trace now carries a moment
    code.weights[:] = 1.0  # every unit predicted from any context
    code.install()
    brain.learner.brain = brain._installed(brain.brain)
    before = code.thresholds.copy()
    brain.act(tokens(4, 9), greedy=True)
    a = np.maximum(code.last["activation"], 0.0)
    free = code.last["free"]
    assert (a > 0).sum() > (free > 0).sum()  # the context recruited units
    share = (free > code.active).sum(axis=0) - 2 * code.target
    repair = code.unexplained(code.read * code.last["sensory"], a) + code.unexplained(
        code.read * code.last["sensory"], free
    )
    expected = before + code.threshold_rate * (share - code.silence * repair)
    np.testing.assert_allclose(code.thresholds, np.maximum(expected, 0.0))
    recruited = (a > 0).any(axis=0) & ~(free > 0).any(axis=0)
    assert recruited.any() and (code.thresholds[recruited] <= before[recruited]).all()
    assert code.work["repairs"] == int(repair.sum())


def test_the_two_switches_act_on_the_life_and_the_checkpoint_keeps_them(tmp_path):
    """``top_down`` 0 takes the prediction off the region's drive (the control without the
    top-down boundary) and ``learning`` False holds every table still; both are the
    instance's, a deep copy and a checkpoint keep them, and a retune names neither."""
    brain = composed(seed=7)
    brain.act(tokens(3), greedy=True)
    code = brain.learned_code
    code.weights[:] = 0.1
    z = np.ones((1, code.context))
    np.testing.assert_allclose(code.stimulus(np.zeros((1, INPUTS)), z), 2.0 * code.prediction(z))
    code.top_down = 0.0
    assert not code.stimulus(np.zeros((1, INPUTS)), z).any()
    code.learning = False
    before = {name: value.copy() for name, value in code.arrays().items()}
    brain.act(tokens(5), greedy=True)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(code, name), value)
    assert brain.describe()["layout"]["learned_code"]["top_down"] == 0.0
    assert brain.describe()["layout"]["learned_code"]["learning"] is False
    twin = cd.Brain.load(brain.save(tmp_path / "switched.npz"))
    assert twin.learned_code.top_down == 0.0 and twin.learned_code.learning is False
    copied = copy.deepcopy(brain)
    assert copied.learned_code.top_down == 0.0 and copied.learned_code.learning is False
