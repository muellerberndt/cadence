"""The recall memory: the index memory of a stream's own moments, each addressed before its
settle; ``recall=False`` is its founder and control."""

import copy
import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from checkpoints import earlier, metadata

import cadence as cd
from cadence import brain as cadence_brain
from cadence.genome import Genome, Projection, develop
from cadence.regions import Region, motor_cortex

BINDING = Path(__file__).resolve().parents[1] / "benchmarks" / "binding"
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
FACT = np.array([[1.0, 0.0, 0.0, 1.0]])  # a name (coordinate 0) stated with a value (3)
NAME = np.array([[1.0, 0.0, 0.0, 0.0]])
STREAM = (CUE, GO, FACT, NAME)
ARRAYS = ("keys", "values", "nexts", "held", "previous")
MAP = ("w_x", "w_c", "b")  # the names of head 0's address map, as a checkpoint keeps them


def composed(seed=3, recall=True, **options):
    """``Brain.compose`` with the recall memory, its founder; ``recall=False`` is the control."""
    return cd.Brain.compose(4, 2, modules=(8,), seed=seed, episodic=False, recall=recall, **options)


def saved_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return earlier({name: data[name].copy() for name in data.files})


def assert_same_checkpoint(first, second):
    a, b = saved_arrays(first), saved_arrays(second)
    assert set(a) == set(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def assert_same_memory(first, second):
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
    for name, a, b in zip(MAP, first.head_map(0), second.head_map(0), strict=True):
        np.testing.assert_array_equal(a, b, err_msg=name)


def synapse_rows(connectome):
    return {
        (int(p), int(q), int(c), float(s))
        for p, q, c, s in zip(
            np.asarray(connectome.pre),
            np.asarray(connectome.post),
            np.asarray(connectome.count),
            np.asarray(connectome.sign),
            strict=True,
        )
    }


def regions(brain):
    populations = brain.connectome.populations
    return np.asarray(populations["recall"]), np.asarray(populations.get("next", ()), dtype=int)


def fork(brain):
    """An independent copy of a life, the immutable connectome shared; read-only reports are
    copied as plain mappings and restored as proxies."""
    proxies = {k: v for k, v in vars(brain).items() if isinstance(v, MappingProxyType)}
    for name, value in proxies.items():
        setattr(brain, name, dict(value))
    try:
        twin = copy.deepcopy(brain, {id(brain.connectome): brain.connectome})
    finally:
        for name, value in proxies.items():
            setattr(brain, name, value)
    for name in proxies:
        setattr(twin, name, MappingProxyType(getattr(twin, name)))
    return twin


def softmax(logits):
    p = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def instrument(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BINDING / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# -- the founder: the composition before the option


def _checkpoint_digest(path):
    """SHA-256 over every array of a checkpoint, by name: dtype, shape and bytes, the
    metadata JSON entries read back without the library version that wrote them."""
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


def _digest(brain):
    connectome, graph = brain.connectome, brain.brain
    h = hashlib.sha256()
    for values in (connectome.pre, connectome.post, connectome.count, connectome.sign,
                   graph.efficacy, graph.bias, graph.log_gain):
        h.update(np.ascontiguousarray(np.asarray(values)).tobytes())
    populations = {k: [int(i) for i in v] for k, v in connectome.populations.items()}
    h.update(json.dumps(populations).encode())
    return h.hexdigest()


def _stepped(brain):
    acts = []
    for t in range(14):
        x = STREAM[t % 4]
        if t == 0:
            action = brain.step(x)
        elif t % 3 == 0:
            action = brain.step(x, reward=[float(t % 2)], teacher=[t % 2])
        else:
            action = brain.step(x, reward=[float(t % 2) - 0.5], done=[t == 7])
        acts.append(int(np.asarray(action).ravel()[0]))
    acts.append(int(brain.act(GO, greedy=True)[0]))
    return acts


def _lived(brain):
    acts = []
    for t in range(14):
        reward = None if t == 0 else [float(t % 3 == 0)]
        done = None if t == 0 else [t == 9]
        acts.append(int(np.asarray(brain.live(STREAM[t % 4], reward=reward, done=done))[0]))
        if t == 5:
            brain.wait(GO)
    phases = brain.imagine([CUE, GO, FACT])
    acts.extend(
        int(np.argmax(np.asarray(phase.state.activation)[0, brain.motor_index]))
        for phase in phases
    )
    return acts


def _taught(brain):
    rng = np.random.default_rng(11)
    acts = []
    for _ in range(3):
        brain.reset()
        xs = [(rng.random((4, 16)) < 0.2).astype(float) for _ in range(4)]
        for x in xs[:-1]:
            acts.extend(int(a) for a in brain.act(x, greedy=True))
        acts.extend(int(a) for a in brain.step(xs[-1], teacher=rng.integers(8, size=4)))
        acts.extend(int(a) for a in brain.act(xs[-1], greedy=True))
    return acts


# Recorded with the composition before the option: each founder's connectome and parameters,
# the acts of a short life, and the format, the member and metadata names (``names``) and
# the contents of every array (``checkpoint``) of the checkpoint it saves.
# Where the library later removed a setting that was inert at its founder, the description
# and checkpoint digests were re-recorded after every other value matched the recording.
# Re-recorded with the unset motor lateral at 0.0 (issue 220) where a founder has at most
# 8 motor neurons in a slot, and at the founder need scale 0.1 (issue 198) where it has arousal.
FOUNDERS = {
    "stepped": {
        "build": (4, 2, {"modules": (8,), "seed": 3, "recall": False}),
        "life": _stepped,
        "digest": "8d42b245bdc9b36499675f913258d4ca9e532a91ad39a0bda82b666e4a441837",
        "acts": [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1],
        "format": "cadence-brain/1",
        "names": "cf4f510fd7745ee140166219a1a6e0af36750c92b9eb41a38f9262d2272f7dda",
        "checkpoint": "67c4444f374b86a738469f0fa63fde0173456ed4c3a8f82f1dba51773c6fab00",
    },
    "lived": {
        "build": (4, 2, {"modules": (8,), "seed": 5, "arousal": True,
                         "history_decays": (0.2, 0.9), "efference_amplitude": 1.0,
                         "recall": False}),
        "life": _lived,
        "digest": "30f4eae2e5b61512953080495ae1fef24fd5555d837fb4570d46e43af38d4582",
        "acts": [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        "format": "cadence-brain/1",
        "names": "73388bfd03b1d7badf6c3be759e8376562d96c9f4052822b96f76e880a7dafef",
        "checkpoint": "89f70f27208981c2916dfaf1d8bb361e89da9990eb8c9c1c201e2d15017adbb9",
    },
    "episodic": {
        "build": (4, 2, {"modules": (8,), "seed": 7, "memory_choice": True, "arousal": True,
                         "recall": False}),
        "life": _lived,
        "digest": "dcbc0983cda21c93c39ec2e5f75b9deebe3af815e53bdd9e9af7befbc7afac53",
        "acts": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "format": "cadence-brain/1",
        "names": "1c418d7da8207299c6e8058e90cd3f627a6c3119a3287e65090a2cbd06ed603a",
        "checkpoint": "cbd324c7313fb5a6af6e2a8881feb3a4a44a495a245a5a4dbed68e673784d6e5",
    },
    "taught": {
        "build": (16, 8, {"modules": (64,), "lateral": 0.0, "seed": 1,
                          "learning_surprise": True, "recall": False}),
        "life": _taught,
        "digest": "88a5654a867b175abae0582a8335aea645297027f2e082420bd046a98c7fed73",
        "acts": [4, 0, 4, 4, 0, 0, 0, 0, 7, 7, 0, 0, 4, 7, 0, 7, 1, 0, 0, 4, 4, 5, 3, 4, 0, 1,
                 0, 1, 0, 0, 0, 0, 2, 2, 6, 3, 0, 0, 4, 4, 0, 0, 0, 0, 7, 4, 0, 4, 1, 3, 0, 0,
                 4, 0, 6, 4, 4, 0, 0, 4],
        "format": "cadence-brain/1",
        "names": "22097597c1a2febccbca3fd13f4eb304480cee327bc4ff19339470e9ea42b49d",
        "checkpoint": "97738feb133fef97802d58d23778a410f18f5cec4a32293c0c6e00b5a11a7179",
    },
}


def _names(path):
    """The checkpoint's metadata keys and member names, which no backend changes."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["brain"]))
        names = json.dumps([sorted(meta), sorted(data.files)])
    return meta["format"], hashlib.sha256(names.encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_the_founder_lives_as_the_composition_before_the_option(tmp_path, name):
    """On every backend: the connectome and parameters, and the checkpoint's format, members
    and metadata keys after the life."""
    spec = FOUNDERS[name]
    inputs, actions, options = spec["build"]
    brain = cd.Brain.compose(inputs, actions, **options)
    assert brain.recall is None and "recall" not in brain.connectome.populations
    assert _digest(brain) == spec["digest"]
    spec["life"](brain)
    assert _names(brain.save(tmp_path / "founder")) == (spec["format"], spec["names"])


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_the_founders_acts_and_checkpoint_bytes_are_the_recorded_ones(tmp_path, name):
    spec = FOUNDERS[name]
    inputs, actions, options = spec["build"]
    brain = cd.Brain.compose(inputs, actions, **options)
    assert spec["life"](brain) == spec["acts"]
    assert _checkpoint_digest(brain.save(tmp_path / "founder")) == spec["checkpoint"]


@pytest.mark.parametrize(
    "options",
    [{}, {"arousal": True}, {"efference_amplitude": 1.0}, {"history_decays": (0.2, 0.8)}],
)
def test_recall_false_builds_the_brain_without_the_option_byte_identically(tmp_path, options):
    control = cd.Brain.compose(4, 2, modules=(8,), seed=3, episodic=False, **options)
    founder = composed(recall=False, **options)
    assert founder.recall is None and "recall" not in founder.connectome.populations
    for brain in (control, founder):
        brain.act(CUE, greedy=True)
    assert_same_checkpoint(control.save(tmp_path / "control"), founder.save(tmp_path / "founder"))
    with np.load(tmp_path / "founder.npz", allow_pickle=False) as data:
        meta = metadata(data)
    assert meta["format"] != "cadence-generic/8" and "recall" not in meta
    assert "recall_scale" not in meta["composition"]
    described = founder.describe()
    assert "recall" not in described["memories"]
    assert not any(name.startswith("recall_") for name in described["genes"])
    assert described == control.describe()


def test_the_founders_are_the_declared_genes():
    assert cd.Recall.founders() == {
        "amplitude": 3.0, "temperature": 0.25, "recency": 0.25, "address": 0.01,
        "capacity": 64, "next": True,
    }
    # the constants that were genes until 0.93.0 keep their one value and their saved keys
    assert (cd.Recall.heard, cd.Recall.address_temperature, cd.Recall.window_decay) == (
        1.0, 0.1, 1.0
    )
    assert cd.Recall.SAVED == ("amplitude", "temperature", "heard", "recency", "address",
                               "address_temperature", "capacity", "next")
    assert composed().describe()["layout"]["composition"]["recall_scale"] == 1.2
    assert inspect.signature(cd.Brain).parameters["recall"].default is False
    described = composed().describe()["genes"]
    assert {k: v for k, v in described.items() if k.startswith("recall_")} == {
        f"recall_{name}": value for name, value in cd.Recall.founders().items()
    }


# -- the layout


def test_the_memory_adds_its_two_regions_and_their_projections_last():
    control, candidate = composed(recall=False), composed()
    populations = dict(candidate.connectome.populations)
    recall, following = list(populations.pop("recall")), list(populations.pop("next"))
    assert {k: list(v) for k, v in populations.items()} == {
        k: list(v) for k, v in control.connectome.populations.items()
    }
    assert list(candidate.connectome.populations)[-2:] == ["recall", "next"]
    assert len(recall) == len(following) == 4 and candidate.connectome.n == control.connectome.n + 8
    # Appended last: the control's synapses are a subset, and the added ones all run from the
    # recall and next neurons into the association region.
    old, new = synapse_rows(control.connectome), synapse_rows(candidate.connectome)
    assert old <= new
    added = new - old
    assert len(added) == 8 * len(candidate.association_index)
    assert {p for p, _, _, _ in added} == set(recall + following)
    assert {q for _, q, _, _ in added} == set(candidate.association_index.tolist())
    # Sources: no synapse ends on them, and an empty memory leaves the independent answer.
    assert not np.isin(np.asarray(candidate.connectome.post), recall + following).any()
    np.testing.assert_array_equal(control.predict(CUE), candidate.predict(CUE))
    heard = composed(history_decays=(0.5,))
    assert list(heard.connectome.populations)[-3:] == ["history", "recall", "next"]
    assert "next" not in composed(recall_next=False).connectome.populations


def test_the_projections_start_at_their_constant_scale_like_the_working_trace():
    """The recall and next projections start at magnitude 1.2, a constant since 0.93.0: the
    composition records it, and ``scales["recall"]`` is refused by name."""
    assert composed().describe()["layout"]["composition"]["recall_scale"] == 1.2
    with pytest.raises(ValueError, match="recall_scale was retired in 0.93.0"):
        composed(scales={"recall": 12.0})


def test_a_custom_connectome_needs_regions_of_the_value_width():
    def graph(width, following=True):
        names = ("recall", "next") if following else ("recall",)
        layout = (Region("sensory", 2), Region("association", 4), motor_cortex(2)) + tuple(
            Region(name, width) for name in names
        )
        projections = (
            Projection("sensory", "association", reciprocal=False),
            Projection("association", "motor"),
        ) + tuple(Projection(name, "association", reciprocal=False) for name in names)
        return develop(Genome(layout, projections))

    assert cd.Brain(graph(2), episodic=False, recall=True).recall is not None
    with pytest.raises(ValueError):
        cd.Brain(graph(3), episodic=False, recall=True)
    with pytest.raises(ValueError):
        cd.Brain(graph(2, following=False), episodic=False, recall=True)
    assert cd.Brain(graph(2, following=False), episodic=False, recall=True,
                    recall_next=False).recall is not None
    # without the option, populations of those names are ordinary ones
    assert cd.Brain(graph(3), episodic=False).recall is None


# -- the instrument


def _declared(ir):
    """The instrument's genes for the library's founders: the declared index arm and code
    fan-in, with the library's slot bound (the instrument leaves the slots unbounded in a
    reset-per-trial task)."""
    return dict(ir.GENES, recency=0.25, address=0.01, capacity=64, code_fan_in=32)


@pytest.mark.parametrize(
    ("seed", "rounds", "extra", "continuing"),
    [
        (0, 12, {}, False),
        (1, 12, {}, False),
        (3, 6, {"next": False}, False),
        (4, 16, {"capacity": 12}, True),
    ],
)
def test_the_memory_is_the_instrument_index_memory_moment_for_moment(
    seed, rounds, extra, continuing
):
    """The library brain reproduces the instrument's ``IndexBrain`` exactly: the same connectome,
    the same acts, settled states, lessons, address map and slots, round after round, on the
    token-binding task. Only the genes that differ from the founders are passed to the
    library."""
    ir = instrument("index_read")
    tb = ir.tb
    read = dict(_declared(ir), **extra)
    inst = ir.IndexBrain.build(seed, read=read, learning_surprise=True)
    lib = cd.Brain.compose(  # the instrument reads no position term (``recall_position=0``)
        tb.INPUTS, tb.VALUES, modules=(tb.WIDTH,), lateral=tb.LATERAL, seed=seed,
        learning_surprise=True, recall=True, recall_position=0,
        **{f"recall_{k}": v for k, v in extra.items()},
    )
    for name in ("pre", "post", "count", "sign"):
        np.testing.assert_array_equal(
            getattr(inst.connectome, name), getattr(lib.connectome, name), err_msg=name
        )
    assert {k: list(v) for k, v in inst.connectome.populations.items()} == {
        k: list(v) for k, v in lib.connectome.populations.items()
    }
    np.testing.assert_array_equal(inst.brain.efficacy, lib.brain.efficacy)
    held = tb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([tb.SCHEDULE_SEED, seed]))
    work, moved = tb.new_work(), 0.0
    for r in range(rounds):
        trials = tb.draw(rng, tb.BATCH, held)
        seq, answer = tb.events(trials), tb.answers(trials)
        if not continuing or r == 0:
            inst.reset()
            lib.reset()
        for x in seq[:-1]:
            np.testing.assert_array_equal(inst.act(x, greedy=True), lib.act(x, greedy=True))
            np.testing.assert_array_equal(
                inst.basal_ganglia.state.activation, lib.basal_ganglia.state.activation
            )
        ir.lesson(inst, seq[-1], answer, work)
        moved += lib.lesson(seq[-1], answer)["address_rows"]
        np.testing.assert_array_equal(inst.brain.efficacy, lib.brain.efficacy)
        for name, kept in zip(MAP, lib.recall.head_map(0), strict=True):
            np.testing.assert_array_equal(getattr(inst.index, name), kept, err_msg=name)
        np.testing.assert_array_equal(inst.act(seq[-1], greedy=True), lib.act(seq[-1], greedy=True))
        for name in ("keys", "values", "nexts"):
            np.testing.assert_array_equal(
                getattr(inst.index.slots, name), getattr(lib.recall, name), err_msg=name
            )
        np.testing.assert_array_equal(inst.index.previous, lib.recall.previous)
    assert moved == work["address_rows"] > 0
    assert lib.recall.slots <= lib.recall.capacity


def test_the_bounded_address_step_is_the_instruments_bit_for_bit():
    """Beyond the bound as within it: on the same slots, moment and movement, rows of either
    side of the bound, the library's address step and the instrument's report the same and
    leave the same map."""
    ir = instrument("index_read")
    lib = composed(seed=4, recall_address=0.5).recall
    inst = ir.IndexMemory(4, 8, 4, dict(_declared(ir), address=0.5))
    rng = np.random.default_rng(14)
    for _ in range(3):
        stored = (rng.random((2, 4)) < 0.6).astype(float)
        lib._observe(stored, stored)
        inst.slots.observe(stored, stored)
    cortex = 0.3 * rng.random((2, 8))
    lib.previous, inst.previous = cortex.copy(), cortex.copy()
    events = np.array([[1.0, 1, 1, 1], [0.5, 0, 0, 0]])
    moment = lib._moment(events)
    np.testing.assert_array_equal(inst.address(events), moment.address)
    gains = 0.5 * ((events**2).sum(axis=1) + (cortex**2).sum(axis=1) + 1.0)
    assert gains[0] > 1 > gains[1]
    d_value, d_next = rng.normal(size=(2, 4)), rng.normal(size=(2, 4))
    report = lib._step(moment, np.arange(2), d_value, d_next)
    assert report == inst.learn(np.arange(2), d_value, d_next, 0.5, lib.address_temperature)
    assert report["address_rows"] == 2.0
    for name, kept in zip(MAP, lib.head_map(0), strict=True):
        np.testing.assert_array_equal(kept, getattr(inst, name), err_msg=name)


# -- addressing and reading


def test_the_address_starts_as_the_event_heard_and_reads_by_competition():
    brain = composed()
    for x in (FACT, GO):
        brain.act(x, greedy=True)
    recall = brain.recall
    assert recall.slots == 2 and recall.writes == 2
    np.testing.assert_allclose(recall.keys[:, 0], recall.heard * FACT)  # the constant, 1.0
    np.testing.assert_allclose(recall.values[:, 1], GO)
    np.testing.assert_allclose(recall.nexts[:, 0], GO)  # the slot before learns what followed
    assert not recall.nexts[:, 1].any()
    drive = brain.stimulus(NAME)
    np.testing.assert_allclose(recall.last.address, NAME)  # the map is zero: the event heard
    logits = np.einsum("bk,bmk->bm", NAME, recall.keys) / 0.25 - 0.25 * np.array([1.0, 0.0])
    p = softmax(logits)
    np.testing.assert_allclose(recall.weights(recall.last.address), p)
    first, following = regions(brain)
    np.testing.assert_allclose(drive[:, first], 3.0 * np.einsum("bm,bmc->bc", p, recall.values))
    np.testing.assert_allclose(drive[:, following], 3.0 * np.einsum("bm,bmc->bc", p, recall.nexts))
    assert drive[0, first][3] > drive[0, first][1]  # the name finds the fact stated with it
    assert not brain.stimulus(NAME, memory=False)[:, np.r_[first, following]].any()


def test_the_memory_records_the_attention_each_read_used():
    """``attended`` is the attention the latest read used, recency included, kept past the
    moment's write, read-only; a reset forgets it."""
    brain = composed()
    recall = brain.recall
    assert recall.attended_heads is None
    for x in (FACT, GO, CUE):
        brain.act(x, greedy=True)
    brain.stimulus(NAME)
    logits = np.einsum("bk,bmk->bm", NAME, recall.keys) / 0.25 - 0.25 * recall.ages()[None]
    np.testing.assert_allclose(recall.attended_heads[0], softmax(logits))
    np.testing.assert_array_equal(recall.attended_heads[0], recall.weights(recall.last.address))
    brain.act(NAME, greedy=True)  # the act's own read, then its write
    assert recall.slots == 4 and recall.attended_heads[0].shape == (1, 3)
    np.testing.assert_allclose(recall.attended_heads[0], softmax(logits))
    with pytest.raises(ValueError):
        recall.attended_heads[0][0, 0] = 0.0
    brain.reset()
    assert recall.attended_heads is None


def test_capacity_drops_the_oldest_slot_and_recency_charges_age():
    brain = composed(recall_capacity=2)
    for i in range(4):
        brain.act(np.eye(4)[i : i + 1], greedy=True)
    recall = brain.recall
    assert recall.slots == 2 and recall.writes == 4
    np.testing.assert_array_equal(recall.keys[0], np.eye(4)[2:4])
    np.testing.assert_array_equal(recall.nexts[0], [np.eye(4)[3], np.zeros(4)])
    np.testing.assert_array_equal(recall.ages(), [1.0, 0.0])
    np.testing.assert_allclose(recall.weights(np.zeros((1, 4)))[0], softmax(np.array([-0.25, 0.0])))


def test_streams_keep_their_own_slots_and_another_batch_reads_silence():
    brain = composed()
    pasts = np.eye(4)[:3]
    brain.act(pasts, greedy=True)
    recall = brain.recall
    first, following = regions(brain)
    present = np.repeat(GO, 3, axis=0)
    drive = brain.stimulus(present)
    np.testing.assert_allclose(drive[:, first], 3.0 * pasts)  # one slot each: all its weight
    before = recall.snapshot()
    brain.predict(present)
    brain.accuracy(present, [0, 1, 0])
    for name, value in before.items():
        assert getattr(recall, name) is value
    silent = recall.stimulate(np.zeros((2, brain.connectome.n)), np.zeros((2, 4)))
    assert not silent[:, np.r_[first, following]].any()
    np.testing.assert_array_equal(recall.keys, before["keys"])


def test_the_brains_code_keys_the_moments_and_its_storage_follows_its_units_whatever_the_width():
    """The memory keys on the brain's observation code (``tests/test_observation_code.py``):
    its keys, values and address map follow the code's units, whatever the width."""
    narrow = cd.Brain.compose(1_025, 2, modules=(16,), seed=1, episodic=False, code=128,
                              recall=True)
    wide = cd.Brain.compose(12_824, 2, modules=(16,), seed=1, episodic=False, code=128,
                            recall=True)
    rng = np.random.default_rng(3)
    for brain in (narrow, wide):
        for _ in range(2):
            brain.act((rng.random((2, brain.inputs)) < 0.05).astype(float), greedy=True)

    def storage(recall):
        shapes = {name: getattr(recall, name).shape for name in ARRAYS}
        return {**shapes, **{f"map_{i}": a.shape for i, a in enumerate(recall.head_map(0))}}

    assert storage(narrow.recall) == storage(wide.recall)
    assert wide.recall.keys.shape == (2, 2, 128) and wide.recall.head_map(0)[0].shape == (128, 128)
    assert len(wide.connectome.populations["recall"]) == 128
    np.testing.assert_allclose(np.linalg.norm(wide.recall.keys, axis=2), 1.0)
    assert ((wide.recall.keys > 0).sum(axis=2) <= 24).all()
    np.testing.assert_array_equal(wide.recall.values, wide.recall.keys)  # event values, W = 0
    frame = (rng.random((3, 12_824)) < 0.05).astype(float)
    np.testing.assert_array_equal(wide.observation_code(frame), wide.recall.last.event
                                  if wide.recall.last is not None else wide.observation_code(frame))


# -- the address lesson


def test_the_readback_is_the_transpose_of_the_sources_synapses():
    brain = composed(seed=4)
    graph = brain.learner.brain
    pre, post = np.asarray(brain.connectome.pre), np.asarray(brain.connectome.post)
    movement = np.random.default_rng(4).normal(size=(3, len(brain.association_index)))
    weight = graph._gain_pre * graph.efficacy
    for region, neurons in zip(("recall", "next"), regions(brain), strict=True):
        got = brain.recall.readback(graph, region, movement)
        for k, neuron in enumerate(neurons):
            out = pre == neuron
            columns = np.searchsorted(brain.association_index, post[out])
            np.testing.assert_allclose(got[:, k], movement[:, columns] @ weight[out])


def test_the_address_lesson_moves_toward_the_slot_the_movement_names():
    brain = composed(seed=5, recall_address=0.5)
    for x in (FACT, GO, CUE):
        brain.act(x, greedy=True)
    recall = brain.recall
    brain.stimulus(NAME)
    moment = recall.last
    target = recall.keys[:, 1]
    report = recall._step(moment, np.arange(1), recall.values[:, 1].copy(), None)
    assert report["address_rows"] == 1
    brain.stimulus(NAME)
    assert np.linalg.norm(recall.last.address - target) < np.linalg.norm(moment.address - target)


def test_the_answers_address_is_named_among_the_moments_attended_to():
    """A moment whose own value is the answer does not displace the moment the address found:
    the attention is the prior, so the step toward the answer's slot is a fraction of the
    step a lesson at rate 1 could take."""
    layout = (Region("sensory", 2), Region("association", 2), motor_cortex(2),
              Region("recall", 2), Region("next", 2))
    projections = (Projection("sensory", "association", reciprocal=False),
                   Projection("association", "motor"))
    connectome = develop(Genome(layout, projections))
    recall = cd.Recall(connectome, temperature=0.25, recency=0.0, address=1.0)
    recall._observe(np.array([[1.0, 0]]), np.array([[1.0, 0]]))  # the moment found
    recall._observe(np.array([[0, 1.0]]), np.array([[0, 1.0]]))  # its value: the answer
    before = recall._moment(np.array([[1.0, 0]]))
    recall._step(before, np.array([0]), np.array([[0.0, 1.0]]), None)
    after = recall._moment(np.array([[1.0, 0]]))
    toward_second = float((after.address - before.address)[0] @ recall.keys[0, 1])
    # rate 1 on |z|^2 = 2 is bounded to a gain of one: the address lands on the named moments'
    # key. The attention is the prior: at the lesson's temperature (``address_temperature``,
    # the constant 0.1) the answer's score weighs its slot e^10 against the prior's e^4 for
    # the moment found, so the step stops 0.0025 short of the answer's key, where a step
    # without the prior would land on it
    assert 0.99 < toward_second < 1.0


def test_the_address_lesson_reads_the_lessons_phases_at_the_taught_rows():
    brain = composed(seed=6, recall_address=0.2)
    x = np.eye(4)
    for past in (x, np.roll(x, 1, axis=1)):
        brain.act(past, greedy=True)
    brain.stimulus(x)
    n = brain.connectome.n
    free, plus = np.zeros((2, n)), np.zeros((2, n))
    plus[:, brain.association_index] = 0.1
    learned = SimpleNamespace(free=SimpleNamespace(activation=free),
                              nudged=SimpleNamespace(activation=plus), opposite=None,
                              taught=np.array([False, True, False, True]))
    before = brain.recall.head_map(0)[0].copy()
    report = brain.recall.learn(brain.learner.brain, learned)
    assert report["address_rows"] == 2
    assert not np.array_equal(brain.recall.head_map(0)[0], before)
    agreed = SimpleNamespace(free=learned.free, nudged=learned.free, opposite=None, taught=None)
    assert brain.recall.learn(brain.learner.brain, agreed)["address_rows"] == 0
    brain.recall.address = 0.0
    assert brain.recall.learn(brain.learner.brain, learned)["address_rows"] == 0


def _gain(recall, moment, rows):
    """Each row's gain on its own address, ``address * |z|^2`` on ``z = [x, c, 1]``."""
    x, c = moment.event[rows], moment.cortex[rows]
    return recall.address * ((x * x).sum(axis=1) + (c * c).sum(axis=1) + 1.0)


@pytest.mark.parametrize("width", [256, 512, 2_000])
def test_a_repeated_dense_query_does_not_grow_the_address_error(width):
    """One stored one-hot moment, then the same lesson on a dense observation of ones, the
    target key fixed. The founder rate on |z|^2 above two would scale the error by
    ``1 - address * |z|^2`` at every lesson; each row's rate is bounded by its own energy,
    so the first step lands the address on its target and none grows the error."""
    brain = cd.Brain.compose(width, 2, modules=(8,), episodic=False, recall=True, seed=2)
    brain.act(np.eye(width)[[0]], greedy=True)
    x = np.ones((1, width))
    brain.stimulus(x)
    assert _gain(brain.recall, brain.recall.last, [0])[0] > 2.0  # beyond the unbounded range
    errors = []
    for _ in range(6):
        report = brain.lesson(x, [1])
        assert report["address_refused"] == 0.0
        if report["address_rows"]:
            errors.append(report["address_error"])
    assert errors and errors[0] > 0.9  # the graph's own lessons can stop the movement after
    assert all(
        later <= earlier + 1e-12 for earlier, later in zip(errors, errors[1:], strict=False)
    )
    recall = brain.recall
    np.testing.assert_allclose(recall._moment(x).address, recall.keys[:, 0], atol=1e-9)


def test_two_thousand_dense_lessons_keep_the_map_finite_and_the_brain_acting():
    """The address learner alone, the graph's plasticity off: 2,000 lessons on the dense
    observation are all taken, the map and the address stay finite, and the brain acts and
    learns after them."""
    brain = cd.Brain.compose(512, 2, modules=(8,), episodic=False, recall=True, seed=2)
    brain.retune(learning=replace(brain.learner.config, eta=0.0, eta_bias=0.0))
    brain.act(np.eye(512)[[0]], greedy=True)
    x = np.ones((1, 512))
    efficacy = brain.brain.efficacy.copy()
    moved = refused = 0.0
    for _ in range(2_000):
        report = brain.lesson(x, [1])
        moved += report["address_rows"]
        refused += report["address_refused"]
    assert (moved, refused) == (2_000.0, 0.0)
    np.testing.assert_array_equal(brain.brain.efficacy, efficacy)  # the graph stood still
    recall = brain.recall
    assert all(np.isfinite(a).all() for a in recall.head_map(0))
    np.testing.assert_allclose(recall._moment(x).address, recall.keys[:, 0], atol=1e-9)
    assert brain.act(x, greedy=True).shape == (1,)
    report = brain.lesson(x, [0])
    assert report["address_rows"] == 1.0 and report["address_refused"] == 0.0


def test_finite_inputs_that_overflow_the_step_leave_the_map_as_it_was():
    """An event whose energy overflows takes a step of rate zero, and the map stays finite; an
    event whose attention overflows names no finite target, and its step is refused: the map
    is kept and the report counts the row. The memory goes on learning from ordinary
    moments."""
    brain = composed(seed=4, recall_address=0.5)
    recall = brain.recall
    for x in (FACT, GO):
        brain.act(x, greedy=True)
    brain.lesson(NAME, [1])
    kept = [a.copy() for a in recall.head_map(0)]
    loud = np.array([[1e200, 1e200, 0.0, 0.0]])  # finite; its |z|^2 is not
    report = recall._step(recall._moment(loud), np.array([0]), np.ones((1, 4)), None)
    assert report["address_rows"] == 1.0 and report["address_refused"] == 0.0
    for before, after in zip(kept, recall.head_map(0), strict=True):
        np.testing.assert_array_equal(after, before)
    louder = np.array([[1e300, 1e300, 0.0, 0.0]])
    recall._observe(louder, louder)  # a slot whose logit against the event overflows
    with np.errstate(over="ignore", invalid="ignore"):
        report = recall._step(recall._moment(louder), np.array([0]), np.ones((1, 4)), None)
    assert report == {"address_rows": 0.0, "address_error": 0.0, "address_refused": 1.0}
    for before, after in zip(kept, recall.head_map(0), strict=True):
        np.testing.assert_array_equal(after, before)
    brain.reset()
    for x in (FACT, GO):
        brain.act(x, greedy=True)
    report = brain.lesson(NAME, [0])
    assert report["address_rows"] == 1.0 and report["address_refused"] == 0.0
    assert not np.array_equal(recall.head_map(0)[0], kept[0])
    assert all(np.isfinite(a).all() for a in recall.head_map(0))


def test_a_rows_step_is_its_own_whatever_the_batch():
    """Each row's rate is bounded by its own energy and the rows' steps are averaged: a row
    taught alone moves its own address by ``min(address * |z|^2, 1)`` times its error, four
    copies of the row taught together step the map as the row alone does, and four distinct
    rows taught together step it by the mean of their steps alone."""
    rng = np.random.default_rng(13)
    events = np.array([[1.0, 1, 1, 1], [1.0, 0, 0, 0], [0.5, 0, 0, 0], [1.0, 1, 0, 0]])
    cortex = 0.1 * rng.normal(size=(4, 8))
    d_value = rng.normal(size=(4, 4))
    stored = [(rng.random((4, 4)) < 0.5).astype(float) for _ in range(2)]

    def step(order):
        """The streams ``order`` names, each with its slots, cortex, event and movement."""
        recall = composed(seed=4, recall_address=0.5).recall
        for slot in stored:
            recall._observe(slot[order], slot[order])
        recall.previous = cortex[order].copy()
        moment = recall._moment(events[order])
        before = [a.copy() for a in recall.head_map(0)]
        report = recall._step(moment, np.arange(len(order)), d_value[order], None)
        assert report["address_rows"] == len(order) and report["address_refused"] == 0.0
        moved = [a - b for a, b in zip(recall.head_map(0), before, strict=True)]
        shift = recall._moment(events[order]).address - moment.address
        return moved, report, shift, _gain(recall, moment, np.arange(len(order)))

    alone = []
    gains = []
    for row in range(4):
        moved, report, shift, gain = step([row])
        alone.append(moved)
        gains.append(gain[0])
        np.testing.assert_allclose(
            np.abs(shift[0]).mean(), min(gain[0], 1.0) * report["address_error"], rtol=1e-12
        )
        copies, _, _, _ = step([row] * 4)
        for part, single in zip(copies, moved, strict=True):
            np.testing.assert_allclose(part, single, rtol=1e-12, atol=1e-15)
    assert max(gains) > 1 > min(gains)  # rows within the bound and beyond it
    batch, _, _, _ = step([0, 1, 2, 3])
    for part, single in zip(batch, zip(*alone, strict=True), strict=True):
        np.testing.assert_allclose(part, np.mean(single, axis=0), rtol=1e-12, atol=1e-15)


def test_in_a_batch_a_row_can_pass_its_target_and_the_batch_error_does_not_grow():
    """A row taught alone lands on its target at most; the rows' steps are averaged, so in a
    batch a row can pass its target. Two streams hear the same event ``u`` (512 coordinates,
    ``|u|^2 = 100``) with one slot each, keys ``2u`` and ``6u``: alone, the first lands on
    ``2u``; together both move from ``u`` to ``4u``, the first past its target. Over random
    batches, rows within the bound and beyond it, the rows' squared errors weighted by their
    rates ``min(address, 1 / |z|^2)`` sum to no more after the step than before."""

    def stepped(keys, events, cortex=None, rate=0.01):
        """One step of a memory whose streams hold one slot each, ``keys``: the target of
        each row is its stream's key. Returns the addresses before and after the step."""
        brain = cd.Brain.compose(events.shape[1], 2, modules=(8,), episodic=False, recall=True,
                                 seed=2, recall_address=rate)
        recall = brain.recall
        recall._observe(keys, keys)
        recall.previous = np.zeros((len(events), 8)) if cortex is None else cortex
        moment = recall._moment(events)
        report = recall._step(moment, np.arange(len(events)), np.ones_like(keys), None)
        assert report["address_rows"] == len(events) and report["address_refused"] == 0.0
        return moment.address, recall._moment(events).address

    u = np.zeros(512)
    u[:100] = 1.0
    before, alone = stepped(2 * u[None], u[None])
    np.testing.assert_allclose(before, u[None])
    np.testing.assert_allclose(alone, 2 * u[None])  # rate * |z|^2 = 1.01: onto the target
    _, together = stepped(np.stack([2 * u, 6 * u]), np.stack([u, u]))
    np.testing.assert_allclose(together, np.stack([4 * u, 4 * u]))  # the first past 2u
    rng = np.random.default_rng(17)
    grew = 0
    for _ in range(60):
        n = int(rng.integers(2, 7))
        rate = float(rng.choice([0.01, 0.5, 2.0]))
        keys = rng.normal(size=(n, 16)) * rng.choice([0.1, 1.0, 5.0], size=(n, 1))
        events = rng.normal(size=(n, 16)) * rng.choice([0.05, 0.5, 3.0], size=(n, 1))
        events[1] = events[0]  # two rows of the same inputs
        cortex = 0.5 * rng.normal(size=(n, 8))
        before, after = stepped(keys, events, cortex, rate)
        weight = np.minimum(rate, 1.0 / ((events**2).sum(axis=1) + (cortex**2).sum(axis=1) + 1))
        error, later = ((keys - before) ** 2).sum(axis=1), ((keys - after) ** 2).sum(axis=1)
        assert weight @ later <= weight @ error * (1.0 + 1e-12)
        grew += int((later > error + 1e-12).any())
    assert grew > 0  # a row's own error grew in some batches; the batch error in none


def test_the_bound_leaves_the_token_task_as_it_was(monkeypatch):
    """At the widths of the declared readings no row reaches the bound, so the step there is
    the step before it, bit for bit. On the token-binding task (16 inputs, an association
    region of 64) a 256-round life of founder 0 reaches at most ``address * |z|^2`` = 0.055
    (founders 0 to 5: 0.052 to 0.059); no state of the task can reach 1, since an event lights
    at most three inputs and an activation lies in (-0.1, 1): 0.01 * (3 + 64 + 1) = 0.68."""
    ir = instrument("index_read")
    tb = ir.tb
    brain = cd.Brain.compose(tb.INPUTS, tb.VALUES, modules=(tb.WIDTH,), lateral=tb.LATERAL,
                             seed=0, learning_surprise=True, recall=True)
    model = brain.brain.neuron_model
    reach = np.abs(model.activation(np.array([-50.0, 50.0]))).max()
    assert reach <= 1.0
    width = len(brain.association_index)
    assert brain.recall.address * (3 + width * reach**2 + 1) < 0.7
    gains = []
    original = cd.Recall._step

    def recorded(self, moment, rows, d_value, d_next, **reading):
        gains.extend(_gain(self, moment, rows).tolist())
        return original(self, moment, rows, d_value, d_next, **reading)

    monkeypatch.setattr(cd.Recall, "_step", recorded)
    held = tb.holdout(0)
    rng = np.random.default_rng(np.random.SeedSequence([tb.SCHEDULE_SEED, 0]))
    for _ in range(256):
        trials = tb.draw(rng, tb.BATCH, held)
        seq = tb.events(trials)
        brain.reset()
        for x in seq[:-1]:
            brain.act(x, greedy=True)
        brain.lesson(seq[-1], tb.answers(trials))
        brain.act(seq[-1], greedy=True)
    assert len(gains) > 100
    assert max(gains) < 0.1


def test_lessons_teach_the_address_and_step_gives_the_same_lesson():
    brain = cd.Brain.compose(16, 8, modules=(64,), lateral=0.0, seed=7, recall=True,
                             recall_address=0.1, learning_surprise=True)
    rng = np.random.default_rng(7)
    rows = taught = 0.0
    for _ in range(3):
        brain.reset()
        xs = [(rng.random((8, 16)) < 0.25).astype(float) for _ in range(3)]
        for x in xs[:-1]:
            brain.act(x, greedy=True)
        report = brain.lesson(xs[-1], rng.integers(8, size=8))
        assert brain.last_learning["demonstration_address_rows"] == report["address_rows"]
        rows += report["address_rows"]
        taught += report["accepted_presentations"]
        brain.act(xs[-1], greedy=True)
    assert 0 < rows <= taught
    assert np.abs(brain.recall.head_map(0)[0]).sum() > 0
    # the lesson of ``step(..., teacher=...)``: the same parameters, then the same act
    twins = [fork(brain), fork(brain)]
    labels = rng.integers(8, size=8)
    for twin in twins:
        twin.reset()
        twin.act(xs[0], greedy=True)
    twins[0].lesson(xs[1], labels)
    shown = twins[0].act(xs[1])
    np.testing.assert_array_equal(shown, twins[1].step(xs[1], teacher=labels))
    np.testing.assert_array_equal(twins[0].brain.efficacy, twins[1].brain.efficacy)
    assert_same_memory(twins[0].recall, twins[1].recall)
    assert twins[0].last_learning == twins[1].last_learning


def test_a_lesson_is_the_founders_lesson_before_its_act_too():
    first, second = (composed(recall=False, seed=5, learning_surprise=True) for _ in range(2))
    for brain in (first, second):
        brain.act(CUE, greedy=True)
    report = first.lesson(GO, [1])
    assert "address_rows" not in report
    np.testing.assert_array_equal(first.act(GO), second.step(GO, teacher=[1]))
    np.testing.assert_array_equal(first.brain.efficacy, second.brain.efficacy)
    assert first.last_learning == second.last_learning


def test_a_lesson_needs_the_live_streams_and_no_sampled_outcome_pending():
    brain = composed()
    x = np.eye(4)[:2]
    brain.act(x)
    with pytest.raises(RuntimeError, match="awaits its outcome"):
        brain.lesson(x, [0, 1])
    brain.learn(np.zeros(2), np.zeros(2, bool), x)
    with pytest.raises(ValueError, match="live streams"):
        brain.lesson(np.eye(4)[:3], [0, 1, 0])
    with pytest.raises(ValueError, match="labels"):
        brain.lesson(x, [0])
    assert brain.lesson(x, [0, 1])["attempted_presentations"] == 2.0


# -- the life: episodes, imagination, waiting


def test_ended_streams_begin_with_empty_slots_and_a_refused_update_rolls_back(monkeypatch):
    brain = composed()
    recall = brain.recall
    x = np.eye(4)[:2]
    brain.act(x)
    brain.learn(np.zeros(2), np.array([True, False]), x)
    assert not recall.held[0].any() and recall.held[1].all()
    assert not recall.keys[0].any() and not recall.previous[0].any()
    np.testing.assert_array_equal(recall.values[1, 0], x[1])
    first, _ = regions(brain)
    drive = brain.stimulus(x)
    assert not drive[0, first].any() and drive[1, first].any()  # the ended stream reads silence
    brain.act(x)
    np.testing.assert_array_equal(recall.held, [[False, True], [True, True]])
    assert not recall.nexts[0].any()  # nothing before the ended stream's new moment learns it
    np.testing.assert_array_equal(recall.nexts[1, 0], x[1])
    before = recall.snapshot()

    def refused(*args, **kwargs):
        raise RuntimeError("the feedback update refused after the memories were touched")

    monkeypatch.setattr(brain.basal_ganglia, "learn", refused)
    with pytest.raises(RuntimeError, match="refused"):
        brain.learn(np.zeros(2), np.array([True, True]), x)
    for name in ARRAYS:  # the per-stream arrays the snapshot holds
        assert getattr(recall, name) is before[name]
    learned = recall.head_map(0)[0].copy()
    brain.reset()
    assert recall.keys.shape == (0, 0, 4) and recall.previous.shape == (0, 8)
    np.testing.assert_array_equal(recall.head_map(0)[0], learned)  # the address map is learned and stays


def test_a_routine_moment_that_ends_the_episode_starts_the_slots_afresh(tmp_path):
    brain = composed(working_memory_amplitude=0.0, arousal=cd.ArousalConfig(youth=0))
    brain.live(CUE)
    brain.live(GO, reward=[0.0])
    assert brain.last_arousal["mode"] == "routine" and brain.recall.slots == 2
    twin = cd.Brain.load(brain.save(tmp_path / "before-terminal"))
    twin.reset()
    action = brain.live(GO, reward=[0.0], done=[True])
    np.testing.assert_array_equal(action, twin.live(GO))
    assert brain.recall.slots == 1
    assert_same_memory(brain.recall, twin.recall)


def test_wait_writes_the_sensed_moment(tmp_path):
    brain = composed(arousal=True)
    brain.live(CUE)
    brain.wait(GO)
    recall = brain.recall
    assert recall.slots == 2
    np.testing.assert_array_equal(recall.values[0], np.r_[CUE, GO])
    np.testing.assert_array_equal(recall.nexts[0, 0], GO[0])
    twin = cd.Brain.load(brain.save(tmp_path / "waiting"))
    assert_same_memory(recall, twin.recall)
    np.testing.assert_array_equal(brain.live(GO, reward=[1.0]), twin.live(GO, reward=[1.0]))
    assert_same_memory(recall, twin.recall)


def test_imagination_reads_the_slots_and_writes_none(tmp_path):
    brain = composed()
    for x in (FACT, GO):
        brain.act(x, greedy=True)
    before = brain.save(tmp_path / "before")
    branch = brain.imagine([NAME, GO, NAME])
    assert len(branch) == 3 and all(np.all(phase.qualified) for phase in branch)
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))
    brain.recall.amplitude = 0.0  # a silent memory changes the first imagined solve
    silent = brain.imagine([NAME, GO, NAME])
    gaps = [
        float(np.max(np.abs(a.state.activation - b.state.activation)))
        for a, b in zip(branch, silent, strict=True)
    ]
    assert gaps[0] > 1e-6


def test_a_fork_carries_its_own_memory():
    brain = composed()
    for x in (FACT, GO):
        brain.act(x, greedy=True)
    twin = fork(brain)
    before = brain.recall.snapshot()
    twin.act(NAME, greedy=True)
    assert twin.recall.slots == brain.recall.slots + 1
    for name, value in before.items():
        assert getattr(brain.recall, name) is value


# -- continuation


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"recall_capacity": 3},
        {"memory_choice": True},
    ],
)
def test_a_saved_memory_continues_identically_and_marks_its_format(tmp_path, options):
    brain = cd.Brain.compose(
        4, 2, modules=(8,), seed=5, recall=True, recall_address=0.2, arousal=True,
        history_decays=(0.5,), efference_amplitude=1.0, learning_surprise=True, **options,
    )
    for t in range(6):
        brain.step(STREAM[t % 4], teacher=[t % 2])
    for t in range(4):
        brain.live(STREAM[t % 4], reward=[float(t % 3 == 0)])
    assert np.abs(brain.recall.head_map(0)[0]).sum() > 0 and brain.recall.slots > 1
    path = brain.save(tmp_path / "life")
    with np.load(path, allow_pickle=False) as data:
        meta = metadata(data)
        assert meta["format"] == "cadence-generic/8"
        assert meta["recall"] == brain.recall.to_dict()
        assert "history" in meta and "efference" in meta and "arousal" in meta
        assert ("episodic" in meta) == ("memory_choice" in options)
        assert meta["composition"]["recall_scale"] == 1.2
    twin = cd.Brain.load(path)
    assert isinstance(twin.recall, cd.Recall)
    assert twin.describe() == brain.describe()
    assert_same_memory(brain.recall, twin.recall)
    for t in range(8):
        x, reward = STREAM[(t + 1) % 4], [float(t % 2)]
        if t == 4:
            shown = brain.step(x, reward=reward, teacher=[1])
            np.testing.assert_array_equal(shown, twin.step(x, reward=reward, teacher=[1]))
        else:
            np.testing.assert_array_equal(brain.live(x, reward=reward), twin.live(x, reward=reward))
        assert_same_memory(brain.recall, twin.recall)
    assert_same_checkpoint(brain.save(tmp_path / "later"), twin.save(tmp_path / "twin"))


@pytest.mark.parametrize(
    "tamper",
    [
        "format_without_recall",
        "recall_without_format",
        "missing_keys",
        "wrong_key_width",
        "beyond_capacity",
        "float_held",
        "bad_temperature",
        "unexpected_array",
        "composition_without_scale",
    ],
)
def test_corrupt_saved_memories_are_refused(tmp_path, tamper):
    brain = composed()
    for x in (CUE, GO):
        brain.act(x, greedy=True)
    arrays = saved_arrays(brain.save(tmp_path / "memory"))
    meta = metadata(arrays)
    if tamper == "format_without_recall":
        del meta["recall"]
    elif tamper == "recall_without_format":
        meta["format"] = "cadence-generic/7"
    elif tamper == "missing_keys":
        del arrays["recall/keys"]
    elif tamper == "wrong_key_width":
        arrays["recall/keys"] = np.zeros((1, 2, 5))
    elif tamper == "beyond_capacity":
        meta["recall"]["capacity"] = 1
    elif tamper == "float_held":
        arrays["recall/held"] = arrays["recall/held"].astype(float)
    elif tamper == "bad_temperature":
        meta["recall"]["temperature"] = 0.0
    elif tamper == "unexpected_array":
        arrays["recall/extra"] = np.zeros(3)
    elif tamper == "composition_without_scale":
        del meta["composition"]["recall_scale"]
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError):
        cd.Brain.load(tmp_path / "bad.npz")


# -- settings


def test_the_read_and_lesson_genes_retune_and_the_layout_is_a_construction_setting():
    brain = composed(arousal=True)
    brain.live(CUE)
    brain.retune(recall_amplitude=1.5, recall_temperature=0.5, recall_address=0.0)
    assert brain.recall.amplitude == 1.5 and brain.recall.address == 0.0
    assert brain.describe()["genes"]["recall_temperature"] == 0.5
    for name, value in (("recall", False), ("recall_capacity", 8), ("recall_next", False),
                        ("scales", {"recall": 2.0})):
        with pytest.raises(ValueError, match="construction setting"):
            brain.retune(**{name: value})
    with pytest.raises(ValueError):
        brain.retune(recall_temperature=0.0)
    with pytest.raises(TypeError):
        brain.retune(recall_seed=1)
    with pytest.raises(ValueError, match="existing memory component"):
        composed(recall=False).retune(recall_amplitude=1.0)


@pytest.mark.parametrize(
    "genes",
    [
        {"recall": 1},
        {"recall": False, "recall_amplitude": 1.0},
        {"recall": True, "recall_temperature": 0.0},
        {"recall": True, "recall_capacity": 0},
        {"recall": True, "recall_next": 1},
        {"recall": True, "recall_recency": -0.1},
        {"recall": True, "recall_amplitude": float("nan")},
        {"recall": True, "scales": {"recall": -1.0}},
    ],
)
def test_invalid_genes_are_refused(genes):
    with pytest.raises(ValueError):
        cd.Brain.compose(4, 2, modules=(8,), episodic=False, **genes)


def test_an_unknown_recall_gene_is_refused_as_a_typo():
    with pytest.raises(TypeError):
        composed(recall_capcity=8)


def test_describe_reports_the_memory_and_parameters_count_its_address_and_position_maps():
    brain = composed()
    assert brain.describe()["memories"].get("recall") == brain.recall.to_dict()
    # the address map (4 + 8 + 1 rows of 4) and the position maps (4 + 8 + 1 rows of 33 buckets)
    assert brain.parameters() == brain.basal_ganglia.parameters() + 4 * 4 + 8 * 4 + 4 + 13 * 33
    plain = composed(recall_position=0)
    assert plain.parameters() == plain.basal_ganglia.parameters() + 4 * 4 + 8 * 4 + 4


# -- options beside the founder memory


def bare(width=2, association=2, **settings):
    """A recall memory on a small anatomy of its own, read and taught directly."""
    layout = (Region("sensory", width), Region("association", association), motor_cortex(2),
              Region("recall", width), Region("next", width))
    projections = (Projection("sensory", "association", reciprocal=False),
                   Projection("association", "motor"))
    return cd.Recall(develop(Genome(layout, projections)), **settings)


def _log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _founder_life(brain):
    for t in range(4):
        brain.step(STREAM[t % 4], teacher=[t % 2])
    for t in range(4):
        brain.live(STREAM[t % 4], reward=[float(t % 3 == 0)])
    brain.wait(GO)


def test_the_options_at_their_founders_build_the_founder_memory(tmp_path):
    """Each option passed at its founder builds the founder memory: the same life, the same
    checkpoint bytes and the same description, which leaves the options out."""
    assert cd.Recall.options() == {
        "null": None, "address_reward": 0.0, "position_reward": 0.0,
        "window": 1,
        "heads": 1, "position": 32, "position_rate": 0.1,
        "boundary": False, "copy": 0.0, "tokens": None, "action": False, "compete": True,
        "anchor": False, "relative": False, "slope": 0.0, "when": False,
        "address_contrast": False, "gain": 0.0,
    }
    founder = composed(seed=5, arousal=True, learning_surprise=True)
    explicit = composed(
        seed=5, arousal=True, learning_surprise=True,
        **{f"recall_{name}": value for name, value in cd.Recall.options().items()},
    )
    for brain in (founder, explicit):
        _founder_life(brain)
    assert_same_checkpoint(founder.save(tmp_path / "founder"), explicit.save(tmp_path / "explicit"))
    assert explicit.describe() == founder.describe()
    assert not set(cd.Recall.options()) & set(explicit.recall.to_dict())


def _option_lives(extra, seed=2, rounds=8):
    """The library brain with ``extra`` beside the founders and the instrument's
    ``IndexBrain`` with the same genes, compared moment for moment on the token-binding task:
    acts, settled states, the attention each read used, lessons, the address map and the
    slots. Returns the rows the address lessons moved."""
    ir = instrument("index_read")
    tb = ir.tb
    inst = ir.IndexBrain.build(seed, read=dict(_declared(ir), **extra), learning_surprise=True)
    lib = cd.Brain.compose(  # the instrument reads no position term (``recall_position=0``)
        tb.INPUTS, tb.VALUES, modules=(tb.WIDTH,), lateral=tb.LATERAL, seed=seed,
        learning_surprise=True, recall=True, recall_position=0,
        **{f"recall_{k}": v for k, v in extra.items()},
    )
    held = tb.holdout(seed)
    rng = np.random.default_rng(np.random.SeedSequence([tb.SCHEDULE_SEED, seed]))
    work, moved = tb.new_work(), 0.0
    for _ in range(rounds):
        trials = tb.draw(rng, tb.BATCH, held)
        seq, answer = tb.events(trials), tb.answers(trials)
        inst.reset()
        lib.reset()
        for x in seq[:-1]:
            np.testing.assert_array_equal(inst.act(x, greedy=True), lib.act(x, greedy=True))
            np.testing.assert_array_equal(
                inst.basal_ganglia.state.activation, lib.basal_ganglia.state.activation
            )
        ir.lesson(inst, seq[-1], answer, work)
        moved += lib.lesson(seq[-1], answer)["address_rows"]
        np.testing.assert_array_equal(inst.index.attended, lib.recall.attended_heads[0])
        np.testing.assert_array_equal(inst.brain.efficacy, lib.brain.efficacy)
        for name, kept in zip(MAP, lib.recall.head_map(0), strict=True):
            np.testing.assert_array_equal(getattr(inst.index, name), kept, err_msg=name)
        np.testing.assert_array_equal(inst.act(seq[-1], greedy=True), lib.act(seq[-1], greedy=True))
        for name in ("keys", "values", "nexts"):
            np.testing.assert_array_equal(
                getattr(inst.index.slots, name), getattr(lib.recall, name), err_msg=name
            )
    assert moved == work["address_rows"]
    return moved


# -- the address step


def test_a_movement_that_names_a_short_key_moves_the_address_onto_it():
    """Two slots with the short keys [.25, 0] and [0, .25], the heard key [1, 0], no recency
    charge: the address attends to the first slot, and a movement that names it (its value)
    gives a target near its short key. The step moves the address toward that target,
    shortening it, and the first slot's weight falls (the step on the named keys less the
    attended mixture, ``recall_address_step="toward"``, which raised it, was retired in
    0.93.0)."""
    recall = bare(recency=0.0, next=False)
    for key in 0.25 * np.eye(2):
        recall._observe(key[None], key[None])
    x = np.array([[1.0, 0.0]])
    before = recall._moment(x)
    weight = recall.weights(before.address)[0, 0]
    report = recall._step(before, np.array([0]), recall.values[:, 0].copy(), None)
    assert report["address_rows"] == 1.0 and report["address_refused"] == 0.0
    after = recall.weights(recall._moment(x).address)[0, 0]
    assert after < weight


# -- a null slot


def test_a_null_slot_takes_its_share_of_an_unrelated_read_and_reads_nothing():
    """A null slot of logit ``null / temperature`` joins the read's softmax and reads
    nothing: for a query that matches no slot it takes ``exp(null / T) / (exp(null / T) +
    sum_i exp(l_i))``, the slots' shares sum to one less it, ``attended_heads`` records them
    and the drive reads them; a full match leaves it little. Without one the shares sum to
    one."""
    brain = composed(recall_null=0.0)
    for x in np.eye(4)[:3]:
        brain.act(x[None], greedy=True)
    recall = brain.recall
    drive = brain.stimulus(np.eye(4)[[3]])  # shares nothing with a stored moment
    logits = -0.25 * recall.ages()
    null = 1.0 / (1.0 + np.exp(logits).sum())
    attended = recall.attended_heads[0]
    np.testing.assert_allclose(attended[0], np.exp(logits) * null, rtol=1e-12)
    np.testing.assert_allclose(attended.sum(), 1.0 - null, rtol=1e-12)
    np.testing.assert_array_equal(recall.weights(recall.last.address), attended)
    first, following = regions(brain)
    np.testing.assert_allclose(drive[:, first],
                               3.0 * np.einsum("bm,bmc->bc", attended, recall.values))
    np.testing.assert_allclose(drive[:, following],
                               3.0 * np.einsum("bm,bmc->bc", attended, recall.nexts))
    brain.stimulus(np.eye(4)[[0]])  # the oldest moment's own event
    assert 1.0 - recall.attended_heads[0].sum() < 0.05
    founder = composed()
    for x in np.eye(4)[:3]:
        founder.act(x[None], greedy=True)
    founder.stimulus(np.eye(4)[[3]])
    np.testing.assert_allclose(founder.recall.attended_heads[0].sum(), 1.0)
    held = composed(recall_null=0.0)  # a stream that holds no slot reads the null slot alone
    held.act(np.eye(4)[:2], greedy=True)
    held.recall.reset(2, rows=np.array([0]))
    shares = held.recall.weights(np.ones((2, 4)))
    assert not shares[0].any() and 0.0 < shares[1].sum() < 1.0


def test_the_address_lesson_reads_the_slots_alone_beside_a_null_slot():
    """The lesson's prior is the attention over the slots alone: a null slot leaves the
    address lesson's step as it is, bit for bit."""
    rng = np.random.default_rng(9)
    keys = [rng.normal(size=(2, 3)) for _ in range(3)]
    cortex, movement = rng.normal(size=(2, 2)), rng.normal(size=(2, 3))
    maps = []
    for null in (None, 2.0):
        recall = bare(width=3, null=null, address=0.5)
        for key in keys:
            recall._observe(key, key)
        recall.previous = cortex.copy()
        moment = recall._moment(np.ones((2, 3)))
        report = recall._step(moment, np.arange(2), movement, None)
        assert report["address_rows"] == 2.0
        maps.append(recall.head_map(0))
    for first, second in zip(*maps, strict=True):
        np.testing.assert_array_equal(first, second)


def test_the_null_slot_is_the_instruments_moment_for_moment():
    assert _option_lives({"null": 0.5}) > 0


def test_a_null_slot_whose_logit_is_not_finite_is_refused(tmp_path):
    """The null slot's logit ``null / temperature`` must be finite: at composition, under
    retune of either setting (nothing changes then) and in a saved memory."""
    with pytest.raises(ValueError, match="null / temperature"):
        composed(recall_null=1e300, recall_temperature=1e-10)
    brain = composed(recall_null=1e10)
    for change in ({"recall_temperature": 1e-300},
                   {"recall_null": 1e308, "recall_temperature": 0.01}):
        with pytest.raises(ValueError, match="null / temperature"):
            brain.retune(**change)
        assert (brain.recall.null, brain.recall.temperature) == (1e10, 0.25)
    brain.retune(recall_null=1e300)  # a finite logit, 4e300
    assert cd.Recall.null_logit(brain.recall.null, brain.recall.temperature) == 4e300
    brain.act(CUE, greedy=True)
    arrays = saved_arrays(brain.save(tmp_path / "null"))
    meta = metadata(arrays)
    meta["recall"]["temperature"] = 1e-10
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="null / temperature"):
        cd.Brain.load(tmp_path / "bad.npz")
    brain.retune(recall_null=None)
    assert "null" not in brain.recall.to_dict()


# -- the options in a checkpoint


def test_the_options_survive_a_checkpoint_and_a_saved_memory_without_one_takes_its_founder(
    tmp_path,
):
    options = {"recall_boundary": True, "recall_position_rate": 0.05, "recall_address": 0.2}
    brain = composed(seed=5, arousal=cd.ArousalConfig(youth=7, threshold=10.0),
                     learning_surprise=True, **options)
    for t in range(6):
        brain.step(STREAM[t % 4], teacher=[t % 2])
    for t in range(4):
        brain.live(STREAM[t % 4], reward=[float(t % 2)])
    brain.wait(GO)
    assert brain.recall.slots > 1 and np.abs(brain.recall.head_map(0)[0]).sum() > 0
    path = brain.save(tmp_path / "options")
    with np.load(path, allow_pickle=False) as data:
        meta = metadata(data)
    assert meta["format"] == "cadence-generic/8" and meta["recall"] == brain.recall.to_dict()
    assert {k: meta["recall"][k] for k in ("boundary", "position_rate", "recency")} == {
        "boundary": True, "position_rate": 0.05, "recency": 0.25,
    }
    twin = cd.Brain.load(path)
    assert twin.describe() == brain.describe()
    assert_same_memory(brain.recall, twin.recall)
    writes = brain.recall.writes
    for t in range(6):  # three aroused moments, then calm ones: every moment is written
        x, reward = STREAM[(t + 1) % 4], [float(t % 2)]
        np.testing.assert_array_equal(brain.live(x, reward=reward), twin.live(x, reward=reward))
        assert_same_memory(brain.recall, twin.recall)
    assert brain.recall.writes == twin.recall.writes == writes + 6
    assert_same_checkpoint(brain.save(tmp_path / "later"), twin.save(tmp_path / "twin"))
    # a saved memory of the present format without the options takes their founders
    arrays = saved_arrays(path)
    meta = metadata(arrays)
    for name in cd.Recall.OPTIONS:
        meta["recall"].pop(name, None)
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "founders.npz", **arrays)
    plain = cd.Brain.load(tmp_path / "founders.npz")
    assert {name: getattr(plain.recall, name) for name in cd.Recall.OPTIONS} == (
        cd.Recall.options()
    )
    for name, value in (
        ("boundary", "yes"), ("position_rate", float("nan")), ("position_rate", True),
    ):
        meta["recall"][name] = value
        arrays["generic"] = np.array(json.dumps(meta))
        np.savez(tmp_path / "bad.npz", **arrays)
        with pytest.raises(ValueError):
            cd.Brain.load(tmp_path / "bad.npz")
        meta["recall"].pop(name)
    with pytest.raises(ValueError, match="position_rate"):
        brain.retune(recall_position_rate=-1.0)
    brain.retune(recall_boundary=False)
    assert "boundary" not in brain.recall.to_dict()


# -- reward trains the address map


def _map(recall):
    return [a.copy() for a in recall.head_map(0)]


def _split(step, width):
    """``(w_x, w_c, b)`` parts of a step on the stacked map, ``(width + association + 1,
    width)``."""
    return step[:width], step[width:-1], step[-1]


def test_a_sampled_act_keeps_the_step_its_lesson_would_take_as_the_maps_eligibility():
    """At a sampled act, before the moment is written, each row's eligibility of the address
    map is the step the address lesson would take at that row were the act's action the
    lesson's label, on the rows of the event heard: the association's movement between the
    act's two nudged phases read back through the recall and next synapses, scored against
    the slots the act read. The cortex and bias rows keep none: they learn from lessons."""
    brain = composed(seed=6, recall_address_reward=0.5)
    for x in (FACT, GO, CUE):
        brain.act(x, greedy=True)
    recall, agent = brain.recall, brain.basal_ganglia
    assert agent.memory is recall and agent._pending_memory is None
    before = recall.snapshot()
    brain.act(NAME)  # sampled
    contrast = agent._pending_memory
    assert contrast.shape == (1, 4 + 8 + 1, 4) and np.abs(contrast).max() > 0
    after = recall.snapshot()
    _, plus, minus, _ = agent._pending
    movement = (plus.activation - minus.activation)[:, brain.association_index]
    recall.restore(before)  # the slots the act read, before its moment was written
    start = _map(recall)
    report = recall._step(recall._moment(NAME), np.array([0]),
                          recall.readback(brain.learner.brain, "recall", movement),
                          recall.readback(brain.learner.brain, "next", movement))
    assert report["address_rows"] == 1.0
    lesson = [a - b for a, b in zip(_map(recall), start, strict=True)]
    event, cortex, bias = _split(contrast[0], 4)
    np.testing.assert_allclose(event, lesson[0], rtol=1e-12, atol=1e-15)
    assert lesson[1].any() and lesson[2].any()  # the lesson moves the shared rows
    assert not cortex.any() and not bias.any()  # reward leaves them to the lesson
    recall._set_map(0, *start)  # noqa: SLF001
    recall.restore(after)


def test_an_outcome_steps_the_address_map_by_its_trace_and_the_dopamine():
    """A sampled act's outcome adds its contrast to the map's trace and steps the map by
    ``address_reward`` times the dopamine times the trace, as every synapse steps; a calm
    moment decays the trace as it decays the synapses', an ended stream forgets it and a reset
    clears it. The report names the step."""
    brain = composed(seed=6, arousal=cd.ArousalConfig(youth=0, threshold=10.0),
                     recall_address_reward=0.5)
    for x in (FACT, GO):
        brain.act(x, greedy=True)
    recall, agent = brain.recall, brain.basal_ganglia
    brain.act(NAME)  # sampled
    contrast = agent._pending_memory.copy()
    start = _map(recall)
    brain.live(CUE, reward=[1.0])  # the act's outcome, learned; the brain answers calm
    report = brain.last_learning
    np.testing.assert_array_equal(agent.trace_memory, contrast)  # the first trace
    step = 0.5 * report["dopamine"] * contrast[0]
    for now, then, part in zip(_map(recall), start, _split(step, 4), strict=True):
        np.testing.assert_allclose(now - then, part, rtol=1e-12, atol=1e-15)
    assert report["address_reward_step"] == pytest.approx(np.linalg.norm(step))
    assert brain.last_arousal["mode"] == "routine" and agent._pending_memory is None
    trace, moved = agent.trace_memory.copy(), _map(recall)
    brain.live(GO, reward=[0.0])  # a calm moment: nothing is learned and the traces fade
    config = agent.config
    np.testing.assert_allclose(agent.trace_memory, config.gamma * config.lam * trace)
    for now, then in zip(_map(recall), moved, strict=True):
        np.testing.assert_array_equal(now, then)
    # a sampled act and an ending outcome: the ended stream forgets the map's trace
    brain.act(FACT)
    brain.learn(np.zeros(1), np.ones(1, bool), GO)
    assert not agent.trace_memory.any()
    brain.reset()
    assert agent.trace_memory is None and agent._pending_memory is None


def test_the_maps_trace_follows_the_streams_and_a_refused_outcome_moves_nothing(monkeypatch):
    brain = composed(seed=7, recall_address_reward=0.5)
    x = np.eye(4)[:3]
    for _ in range(2):
        brain.act(x, greedy=True)
    brain.act(x)
    agent, recall = brain.basal_ganglia, brain.recall
    assert agent._pending_memory.shape == (3, 13, 4)
    brain.learn(np.array([1.0, 0.0, -1.0]), np.array([False, True, False]), x)
    assert agent.trace_memory.shape == (3, 13, 4) and not agent.trace_memory[1].any()
    assert agent.trace_memory[[0, 2]].any()
    brain.act(x)
    assert not agent._pending_memory[1].any()  # the ended stream holds no slot it read
    pending, trace, start = agent._pending_memory, agent.trace_memory.copy(), _map(recall)

    def refused(*args, **kwargs):
        raise ValueError("the graph's update refused")

    monkeypatch.setattr(brain.learner, "apply", refused)
    with pytest.raises(ValueError, match="refused"):
        brain.learn(np.ones(3), np.zeros(3, bool), x)
    np.testing.assert_array_equal(agent.trace_memory, trace)
    for now, then in zip(_map(recall), start, strict=True):
        np.testing.assert_array_equal(now, then)
    assert agent._pending_memory is pending  # the outcome can be reported again
    monkeypatch.undo()

    def overflow(step):
        raise ValueError("the address map's reward step must be finite")

    monkeypatch.setattr(recall, "propose", overflow)
    efficacy = brain.brain.efficacy.copy()
    with pytest.raises(ValueError, match="finite"):
        brain.learn(np.ones(3), np.zeros(3, bool), x)
    np.testing.assert_array_equal(brain.brain.efficacy, efficacy)  # checked before the graph's
    np.testing.assert_array_equal(agent.trace_memory, trace)
    monkeypatch.undo()
    report = brain.learn(np.ones(3), np.zeros(3, bool), x)
    assert report["address_reward_step"] > 0.0


def test_the_maps_trace_survives_a_checkpoint_and_the_life_continues_identically(tmp_path):
    brain = composed(seed=5, arousal=cd.ArousalConfig(youth=6, threshold=10.0),
                     recall_address_reward=0.5)
    for t in range(5):
        brain.live(STREAM[t % 4], reward=None if t == 0 else [float(t % 2)])
    brain.wait(GO)  # the awaited act's contrast was taken before its moment and this one
    agent = brain.basal_ganglia
    assert agent.trace_memory is not None and agent._pending_memory is not None
    assert np.abs(brain.recall.head_map(0)[0]).sum() > 0  # no lesson: the outcomes moved it
    path = brain.save(tmp_path / "trace")
    with np.load(path, allow_pickle=False) as data:
        assert {"memories/recall/trace", "memories/recall/pending"} <= set(data.files)
    twin = cd.Brain.load(path)
    np.testing.assert_array_equal(twin.basal_ganglia.trace_memory, agent.trace_memory)
    np.testing.assert_array_equal(twin.basal_ganglia._pending_memory, agent._pending_memory)
    assert twin.basal_ganglia.memory is twin.recall
    for t in range(6):
        x, reward = STREAM[(t + 1) % 4], [float(t % 3 == 0)]
        np.testing.assert_array_equal(brain.live(x, reward=reward), twin.live(x, reward=reward))
        assert_same_memory(brain.recall, twin.recall)
        np.testing.assert_array_equal(twin.basal_ganglia.trace_memory, agent.trace_memory)
    assert_same_checkpoint(brain.save(tmp_path / "later"), twin.save(tmp_path / "twin"))
    bad = saved_arrays(path)
    bad["recall/trace"] = bad["recall/trace"][:, :-1]
    np.savez(tmp_path / "shape.npz", **bad)
    with pytest.raises(ValueError, match="trace"):
        cd.Brain.load(tmp_path / "shape.npz")
    later = saved_arrays(tmp_path / "later.npz")  # a calm answer: nothing pending
    assert "recall/trace" in later and "recall/pending" not in later
    later["recall/pending"] = np.zeros_like(later["recall/trace"])
    np.savez(tmp_path / "stray.npz", **later)
    with pytest.raises(ValueError, match="pending action"):
        cd.Brain.load(tmp_path / "stray.npz")


def test_without_address_reward_nothing_is_allocated_and_no_byte_changes(tmp_path):
    """The founder ``address_reward=0`` keeps no eligibility of the map: no trace, no pending
    contrast, no report of a step and no checkpoint array, the same bytes as a brain that
    names the founder."""
    founder = composed(seed=5, arousal=True)
    named = composed(seed=5, arousal=True, recall_address_reward=0.0)
    for brain in (founder, named):
        for t in range(6):
            brain.live(STREAM[t % 4], reward=None if t == 0 else [float(t % 2)])
        brain.act(GO)
        assert brain.basal_ganglia.memory is brain.recall
        assert brain.basal_ganglia.trace_memory is None
        assert brain.basal_ganglia._pending_memory is None
        assert "address_reward_step" not in brain.last_learning
    assert_same_checkpoint(founder.save(tmp_path / "founder"), named.save(tmp_path / "named"))
    with np.load(tmp_path / "founder.npz", allow_pickle=False) as data:
        assert not {"memories/recall/trace", "memories/recall/pending"} & set(data.files)
    assert "address_reward" not in founder.recall.to_dict()
    with pytest.raises(ValueError):
        composed(recall_address_reward=-0.1)


def test_the_address_settings_cannot_change_while_an_address_contrast_is_pending():
    """A sampled act's address contrast belongs to the rates it was taken at: while its
    outcome is pending, retuning ``recall_address`` or ``recall_address_reward`` is refused,
    as ``learning_beta`` is, and nothing changes. The same values, other genes, a consumed
    outcome and a pending act that kept no contrast retune as before."""
    brain = composed(seed=6, recall_address_reward=0.5)
    for x in (FACT, GO):
        brain.act(x, greedy=True)
    brain.act(NAME)  # sampled: the contrast is pending
    assert brain.basal_ganglia._pending_memory is not None
    recall = brain.recall
    for name, value in (("recall_address", 0.02), ("recall_address_reward", 0.25)):
        with pytest.raises(ValueError, match=f"{name} cannot change while sampled feedback"):
            brain.retune(**{name: value}, recall_temperature=0.5)
    assert (recall.address, recall.address_reward, recall.temperature) == (0.01, 0.5, 0.25)
    brain.retune(recall_address=0.01, recall_temperature=0.5)  # the same rate, another gene
    assert recall.temperature == 0.5
    brain.learn(np.ones(1), np.zeros(1, bool), CUE)  # the outcome is consumed
    brain.retune(recall_address=0.02, recall_address_reward=0.25)
    assert (recall.address, recall.address_reward) == (0.02, 0.25)
    founder = composed(seed=6)
    founder.act(NAME)  # sampled, and no address contrast kept
    founder.retune(recall_address=0.02)
    assert founder.recall.address == 0.02


def test_a_brain_retuned_to_no_address_reward_reports_and_saves_as_one_composed_without(
    tmp_path,
):
    """Retuned to ``recall_address_reward=0`` the actor drops the address map's trace and its
    pending contrast and proposes nothing: from then on the brain describes, reports and
    saves as one composed at 0. A rate set to zero on the memory itself drops the trace at
    the next outcome and takes no step."""
    brain = composed(seed=5, arousal=True, recall_address_reward=0.5)
    for t in range(5):
        brain.live(STREAM[t % 4], reward=None if t == 0 else [float(t % 2)])
    agent = brain.basal_ganglia
    assert agent.trace_memory is not None and "address_reward_step" in brain.last_learning
    brain.act(GO, greedy=True)  # the pending action is replaced: no contrast is pending
    brain.retune(recall_address_reward=0.0)
    assert agent.trace_memory is None and agent._pending_memory is None
    assert "address_reward" not in brain.recall.to_dict()
    assert "recall_address_reward" not in brain.describe()["genes"]
    for t in range(5):
        brain.live(STREAM[t % 4], reward=None if t == 0 else [float(t % 2)])
        if t:  # an outcome taken
            assert "address_reward_step" not in brain.last_learning
        assert agent.trace_memory is None and agent._pending_memory is None
    with np.load(brain.save(tmp_path / "retuned"), allow_pickle=False) as data:
        assert not {"recall/trace", "recall/pending"} & set(data.files)
    # the rate set on the memory itself
    other = composed(seed=5, recall_address_reward=0.5)
    for x in (FACT, GO):
        other.act(x, greedy=True)
    other.act(NAME)
    other.learn(np.ones(1), np.zeros(1, bool), CUE)
    assert other.basal_ganglia.trace_memory is not None
    other.recall.address_reward = 0.0
    start = _map(other.recall)
    other.act(CUE)  # sampled: no contrast is kept at rate zero
    report = other.learn(np.ones(1), np.zeros(1, bool), GO)
    assert other.basal_ganglia.trace_memory is None and "address_reward_step" not in report
    for now, then in zip(_map(other.recall), start, strict=True):
        np.testing.assert_array_equal(now, then)


# -- a reset that keeps the memory


def test_a_reset_that_keeps_the_memory_keeps_the_slots_and_starts_the_rest_afresh():
    """``Brain.reset(keep=("slots",))`` keeps the recall memory's slots: each stream ends an
    episode there, the next moment of the same batch reads the slots and the latest one
    learns no successor from it. Everything else starts afresh, and a plain reset empties the
    slots."""
    brain = composed(seed=5, history_decays=(0.5,))
    for x in (FACT, GO):
        brain.act(x, greedy=True)
    keys = brain.recall.keys.copy()
    brain.reset(keep=("slots",))
    recall = brain.recall
    np.testing.assert_array_equal(recall.keys, keys)
    assert recall.ended.all() and not recall.previous.any()
    assert brain.basal_ganglia.state is None and not len(brain.history.trace)
    assert not len(brain.working_memory.trace)
    brain.stimulus(NAME)
    assert recall.attended_heads is not None and recall.attended_heads[0].shape == (1, 2)
    brain.act(NAME, greedy=True)
    assert recall.slots == 3 and not recall.nexts[0, 1].any() and not recall.ended.any()
    np.testing.assert_array_equal(recall.nexts[0, 0], GO[0])
    brain.reset()
    assert recall.slots == 0
    with pytest.raises(ValueError, match="'slots' and 'arousal'"):
        brain.reset(keep=1)
    with pytest.raises(TypeError, match="keep_memory"):  # the keyword of 0.84.3, renamed
        brain.reset(keep_memory=True)


def test_ended_marks_in_a_checkpoint_are_read_back_and_a_float_mark_is_refused(tmp_path):
    """The loader reads ``recall/ended``, the streams whose latest slot closed an episode
    (``Recall.end``, which ``reset(keep=("slots",))`` calls): the loaded twin's marked slot
    learns no successor from its stream's next moment, as the live memory's ``end`` leaves
    it, and the other stream goes on. A float-valued mark is refused."""
    brain = composed(seed=5)
    x = np.eye(4)[:2]
    for y in (x, np.roll(x, 1, axis=1)):
        brain.act(y, greedy=True)
    arrays = saved_arrays(brain.save(tmp_path / "plain"))
    assert "recall/ended" not in arrays
    arrays["recall/ended"] = np.array([True, False])
    np.savez(tmp_path / "ended.npz", **arrays)
    twin = cd.Brain.load(tmp_path / "ended.npz")
    np.testing.assert_array_equal(twin.recall.ended, [True, False])
    brain.recall.end(2, np.array([0]))  # the same end on the live memory
    y = np.eye(4)[[2, 3]]
    np.testing.assert_array_equal(brain.act(y, greedy=True), twin.act(y, greedy=True))
    assert_same_memory(brain.recall, twin.recall)
    assert not twin.recall.nexts[0, 1].any() and twin.recall.nexts[1, 1].any()
    assert not twin.recall.ended.any()
    arrays["recall/ended"] = arrays["recall/ended"].astype(float)
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="ended"):
        cd.Brain.load(tmp_path / "bad.npz")


# -- the window key


def _window(present, past, heard=1.0, decay=1.0):
    """A window key: ``heard`` times the present event, then each earlier event (zeros for
    none) at ``decay`` per position back."""
    blocks = [heard * np.asarray(present, dtype=float)]
    for j, event in enumerate(past, start=1):
        blocks.append(heard * decay**j * (np.zeros_like(blocks[0]) if event is None else event))
    return np.concatenate(blocks)


def test_the_window_key_orders_the_streams_last_events():
    """With ``recall_window=3`` a key is the moment's event, then the stream's two previous
    events at their full weight (``Recall.window_decay`` is the constant 1.0), each on
    coordinates of its own: zeros at a stream's first moments and after an episode ends, when
    the slots go, each stream of a batch its own. The address is the key while the map is
    zero; the map reads the event alone and writes the wider key; values stay the events."""
    brain = composed(seed=5, recall_window=3)
    recall = brain.recall
    assert recall.address_width == 12 and recall.head_map(0)[0].shape == (4, 12)
    a, b, c = np.eye(4)[:2], np.eye(4)[[2, 3]], np.eye(4)[[1, 0]]
    for x in (a, b, c):
        brain.act(x, greedy=True)
    for row in range(2):
        np.testing.assert_array_equal(recall.keys[row, 0], _window(a[row], [None, None]))
        np.testing.assert_array_equal(recall.keys[row, 1], _window(b[row], [a[row], None]))
        np.testing.assert_array_equal(recall.keys[row, 2], _window(c[row], [b[row], a[row]]))
    np.testing.assert_array_equal(recall.values[0], [a[0], b[0], c[0]])
    brain.stimulus(a)
    np.testing.assert_array_equal(recall.last.address[1], _window(a[1], [c[1], b[1]]))
    brain.act(a)  # sampled, so an outcome can end stream 1's episode
    brain.learn(np.zeros(2), np.array([False, True]), b)
    brain.act(b, greedy=True)
    np.testing.assert_array_equal(recall.keys[0, -1], _window(b[0], [a[0], c[0]]))
    np.testing.assert_array_equal(recall.keys[1, -1], _window(b[1], [None, None]))
    assert recall.held[1].sum() == 1
    report = brain.lesson(c, [0, 1])
    assert report["address_rows"] > 0 and recall.head_map(0)[0].shape == (4, 12)


def test_waited_moments_enter_the_window_and_imagination_keeps_its_own_copy():
    brain = composed(seed=7, arousal=True, recall_window=2)
    recall = brain.recall
    brain.live(CUE)
    brain.wait(GO)  # the waited moment is written and enters the window
    np.testing.assert_array_equal(recall.keys[0, 1], _window(GO[0], [CUE[0]]))
    np.testing.assert_array_equal(recall.past[0, 0], GO[0])
    before = recall.snapshot()
    phases = brain.imagine([NAME, FACT])
    assert all(np.all(phase.qualified) for phase in phases)
    for name, value in before.items():
        assert getattr(recall, name) is value
    brain.live(FACT, reward=[0.0])
    np.testing.assert_array_equal(recall.keys[0, 2], _window(FACT[0], [GO[0]]))


@pytest.mark.parametrize("window", [2, 4], ids=["w2", "w4"])
def test_the_window_is_the_instruments_moment_for_moment(window):
    assert _option_lives({"window": window}) > 0


def test_a_window_reads_the_successor_of_the_matching_history():
    """A memory-only contract. A stream stores the events A B C A B D, ends its episode with
    its slots kept and reads at A B: the window gives the slots whose window matches (A then
    B) and their successors more weight than the event's own key does. Read at C A B, the
    window tells apart the two B moments by the event before A, and the successor of the
    matching history, D, outweighs C, where the event's own key weighs them alike."""
    events = np.eye(4)
    A, B, C, D = events  # noqa: N806

    def stored(window, end):
        recall = bare(width=4, recency=0.0, window=window)
        n = recall.connectome.n
        for x in (A, B, C, A, B, D):
            recall.witness(x[None], SimpleNamespace(activation=np.zeros((1, n))))
        if end:
            recall.end(1)
        return recall, n

    def read(window, history, end):
        recall, n = stored(window, end)
        for x in history[:-1]:
            recall.witness(x[None], SimpleNamespace(activation=np.zeros((1, n))))
        drive = recall.stimulate(np.zeros((1, n)), history[-1][None])
        return recall.attended_heads[0][0], drive[0, recall.next_neurons]

    weight = {k: read(k, (A, B), True) for k in (1, 2)}
    matching = [1, 4]  # the B slots, each after A
    assert weight[2][0][matching].sum() > weight[1][0][matching].sum()
    assert weight[2][1][[2, 3]].sum() > weight[1][1][[2, 3]].sum()  # their successors C and D
    single, _ = read(1, (C, A, B), False)
    windowed, following = read(3, (C, A, B), False)
    assert single[1] == pytest.approx(single[4])  # the event's own key: alike
    assert windowed[4] > 2 * windowed[1]  # the window: B after C A, its successor D
    assert following[3] > following[2]


def test_the_window_survives_a_checkpoint_and_a_window_of_another_length_is_refused(tmp_path):
    brain = composed(seed=5, arousal=True, recall_window=3, recall_address=0.2,
                     learning_surprise=True)
    for t in range(5):
        brain.step(STREAM[t % 4], teacher=[t % 2])
    brain.live(GO)
    path = brain.save(tmp_path / "window")
    with np.load(path, allow_pickle=False) as data:
        meta = metadata(data)
        assert data["memories/recall/past"].shape == (1, 2, 4)
        assert data["memories/recall/keys"].shape[2] == 12
    assert meta["recall"]["window"] == 3 and "window_decay" not in meta["recall"]
    twin = cd.Brain.load(path)
    assert twin.describe() == brain.describe()
    np.testing.assert_array_equal(twin.recall.past, brain.recall.past)
    for t in range(4):
        x, reward = STREAM[(t + 1) % 4], [float(t % 2)]
        np.testing.assert_array_equal(brain.live(x, reward=reward), twin.live(x, reward=reward))
        assert_same_memory(brain.recall, twin.recall)
    assert_same_checkpoint(brain.save(tmp_path / "later"), twin.save(tmp_path / "twin"))
    with pytest.raises(ValueError, match="construction setting"):
        brain.retune(recall_window=2)
    for genes in ({"recall_window": 0}, {"recall_window": 2.0}):
        with pytest.raises(ValueError, match="window"):
            composed(**genes)
    for tamper in ("shorter", "without_past"):
        arrays = saved_arrays(path)
        meta = metadata(arrays)
        if tamper == "shorter":
            meta["recall"]["window"] = 2  # the saved window holds three events
        else:
            del arrays["recall/past"]
        arrays["generic"] = np.array(json.dumps(meta))
        np.savez(tmp_path / "bad.npz", **arrays)
        with pytest.raises(ValueError):
            cd.Brain.load(tmp_path / "bad.npz")


# -- the settings retired in 0.85.0


# each at a value 0.84.3 took, and its founder there
REMOVED = {
    "recall_write": ("aroused", "always"),
    "recall_key": ("predecessor", "event"),
    "recall_episodes": ("keep", "forget"),
    "recall_recency": (None, 0.25),
    "recall_value": ("cortex", "event"),
    "recall_context": (0.5, 0.0),
    "recall_context_decay": (0.9, 0.5),
}


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_a_removed_setting_is_refused_at_compose_and_retune_naming_the_release(name):
    """Each setting 0.84.3 took and 0.85.0 retired is refused with the one sentence of
    ``Recall.removed``, naming the release that retired it, what the memory does instead and
    0.84.3 as the release that takes it: a retired name at any value, and
    ``recall_recency=None``, the power-law charge, at that value alone (a number stays the
    linear charge). The description carries a retired name only where the 0.84.3 format
    always wrote it, at its one value."""
    value, founder = REMOVED[name]
    message = rf"{name}.*was retired in 0\.85\.0: .*cadence-net 0\.84\.3 takes it"
    brain = composed()
    with pytest.raises(ValueError, match=message):
        composed(**{name: value})
    with pytest.raises(ValueError, match=message):
        brain.retune(**{name: value})
    if name == "recall_recency":
        assert composed(recall_recency=founder).recall.recency == founder
        brain.retune(recall_recency=0.5)
        assert brain.recall.recency == 0.5 and brain.describe()["genes"]["recall_recency"] == 0.5
        with pytest.raises(ValueError, match="power-law"):
            bare(width=4, recency=None)
    else:
        with pytest.raises(ValueError, match=message):
            composed(**{name: founder})
        with pytest.raises(ValueError, match=message):
            brain.retune(**{name: founder})
        setting = name[len("recall_"):]
        assert cd.Recall.RETIRED[setting][0] == "0.85.0"
        assert cd.Recall.removed(setting) == message_of(setting)
        assert name not in brain.describe()["genes"]


def message_of(setting):
    """The sentence ``Recall.removed`` gives, built from the table's row."""
    key = "recency=None" if setting == "recency" else setting
    release, _, instead = cd.Recall.RETIRED[key]
    return f"recall_{key} was retired in {release}: {instead}"


@pytest.mark.parametrize("name", sorted(REMOVED))
def test_a_saved_brain_under_a_removed_setting_is_refused_naming_the_release(tmp_path, name):
    """A checkpoint carrying a retired setting at a value other than the one the memory keeps
    for it is refused on load with the same sentence, naming the release that retired it; one
    carrying the kept value loads (``recall_recency=None`` has none)."""
    brain = composed()
    for x in (CUE, GO):
        brain.act(x, greedy=True)
    arrays = saved_arrays(brain.save(tmp_path / "founder"))
    meta = metadata(arrays)
    setting = name[len("recall_"):]
    meta["recall"][setting] = REMOVED[name][0]
    arrays["generic"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "removed.npz", **arrays)
    with pytest.raises(ValueError, match=r"was retired in 0\.85\.0: .*cadence-net 0\.84\.3"):
        cd.Brain.load(tmp_path / "removed.npz")
    if setting != "recency":
        meta["recall"][setting] = cd.Recall.RETIRED[setting][1]
        arrays["generic"] = np.array(json.dumps(meta))
        np.savez(tmp_path / "kept.npz", **arrays)
        assert cd.Brain.load(tmp_path / "kept.npz").describe() == brain.describe()


def test_the_keys_the_last_release_wrote_load_at_their_one_value(tmp_path):
    """The keys the format of 0.84.3 always wrote for retired settings (``Recall.FORMAT``)
    and its empty context history leave the description and the checkpoint; a file of
    0.84.3 that carries them at their one value loads to the same brain."""
    brain = composed()
    for x in (CUE, GO):
        brain.act(x, greedy=True)
    path = brain.save(tmp_path / "founder")
    with np.load(path, allow_pickle=False) as data:
        saved = json.loads(str(data["brain"]))
        assert not set(cd.Recall.FORMAT) & set(saved["memories"]["recall"])
        assert "memories/recall/history" not in data.files
    described = brain.describe()
    assert not set(cd.Recall.FORMAT) & set(described["memories"]["recall"])
    assert not {f"recall_{k}" for k in cd.Recall.FORMAT} & set(described["genes"])
    arrays = saved_arrays(path)
    meta = metadata(arrays)
    meta["recall"].update(cd.Recall.FORMAT)
    arrays["generic"] = np.array(json.dumps(meta))
    arrays["recall/history"] = np.zeros((0, 4))
    np.savez(tmp_path / "released.npz", **arrays)
    assert cd.Brain.load(tmp_path / "released.npz").describe() == described
