"""A separator's actual coordinate system survives full-brain checkpoints."""

import json

import checkpoints
import numpy as np
import pytest
from checkpoints import earlier

import cadence as cd


def separated_brain(kind="consolidating", backend="cpu"):
    brain = cd.Brain.compose(
        3,
        3,
        modules=(5,),
        episodic=True,
        seed=9,
        backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    separator = cd.PatternSeparator(3, 16, 3, seed=7, center=0 if kind == "consolidating" else 0.3)
    # The seed cannot recover a customized projection or the observed running mean.
    separator.projection = np.roll(separator.projection, 4, axis=1) * 1.2
    separator.mean = np.array([0.2, -0.3, 0.4])
    memory_class = cd.SynapticMemory if kind == "consolidating" else cd.FastSynapses
    options = {"consolidation": 0.25} if kind == "consolidating" else {}
    brain.hippocampus = memory_class(
        brain.sensory_index,
        brain.motor_index,
        separator=separator,
        decay=0.85,
        rate=0.7,
        amplitude=1.3,
        rule="delta",
        **options,
    )
    brain.hippocampus.observe(np.eye(3)[:2], np.array([[1.0, -0.2, 0.5], [-0.3, 0.6, 1.2]]))
    brain.step(np.eye(3)[:2])  # The next reward must finish this saved pending action.
    return brain


def assert_memory_equal(left, right):
    assert left.to_dict() == right.to_dict()
    for name in ("pre", "post", "strength", "mass"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    if isinstance(left, cd.SynapticMemory):
        np.testing.assert_array_equal(left.consolidated, right.consolidated)
    if left.separator is not None:
        for name in ("projection", "mean"):
            a, b = getattr(left.separator, name), getattr(right.separator, name)
            np.testing.assert_array_equal(a, b)
            assert not np.shares_memory(a, b)


@pytest.mark.parametrize("kind", ["consolidating", "fast"])
@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_expanded_memory_resumes_pending_action_and_future_learning(tmp_path, kind, backend):
    if backend not in cd.available_backends():
        pytest.skip(backend)
    original = separated_brain(kind, backend)
    restored = cd.Brain.load(
        original.save(tmp_path / "separated.npz"),
        backend=backend,
        device="cpu" if backend == "torch" else None,
    )
    assert_memory_equal(original.hippocampus, restored.hippocampus)
    assert restored.hippocampus.strength.shape == (2, 16, 3)
    for i in range(5):
        x = np.eye(3)[[i % 3, (i + 1) % 3]]
        reward, done = np.array([0.7, -0.4]), np.array([i == 2, i == 3])
        np.testing.assert_array_equal(
            original.step(x, reward=reward, done=done),
            restored.step(x, reward=reward, done=done),
        )
        np.testing.assert_array_equal(original.brain.efficacy, restored.brain.efficacy)
        np.testing.assert_array_equal(
            original.basal_ganglia.w_critic, restored.basal_ganglia.w_critic
        )
        assert_memory_equal(original.hippocampus, restored.hippocampus)
    # A new stream must also reconstruct the persistent matrix in expanded coordinates.
    original.reset()
    restored.reset()
    np.testing.assert_array_equal(original.step(np.eye(3)), restored.step(np.eye(3)))
    assert_memory_equal(original.hippocampus, restored.hippocampus)


def rewrite_checkpoint(path, mutate):
    with np.load(path, allow_pickle=False) as saved:
        arrays = earlier({key: saved[key].copy() for key in saved})
    metadata = checkpoints.metadata(arrays)
    mutate(metadata, arrays)
    arrays["generic"] = np.array(json.dumps(metadata))
    np.savez(path, **arrays)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("kind", ["consolidating", "fast", "none"])
def test_legacy_unseparated_memory_keeps_its_original_rule(tmp_path, version, kind):
    original = cd.Brain.compose(3, 3, modules=(5,))
    if kind == "fast":
        original.hippocampus = cd.FastSynapses(original.sensory_index, original.motor_index)
    elif kind == "none":
        original.hippocampus = None
    original.step(np.eye(3)[:2])
    original.step(np.eye(3)[:2], reward=np.array([0.7, -0.2]))
    path = original.save(tmp_path / "legacy.npz")

    def make_legacy(metadata, arrays):
        metadata["format"] = f"cadence-generic/{version}"
        if metadata["hippocampus"] is not None:
            metadata["hippocampus"].pop("separator")

    rewrite_checkpoint(path, make_legacy)
    restored = cd.Brain.load(path)
    np.testing.assert_array_equal(
        original.step(np.eye(3)[:2], reward=np.array([0.1, 0.5])),
        restored.step(np.eye(3)[:2], reward=np.array([0.1, 0.5])),
    )
    np.testing.assert_array_equal(original.brain.efficacy, restored.brain.efficacy)
    if kind == "none":
        assert restored.hippocampus is None
    else:
        assert_memory_equal(original.hippocampus, restored.hippocampus)


@pytest.mark.parametrize(
    "defect",
    [
        "missing_projection",
        "missing_mean",
        "projection_shape",
        "projection_nan",
        "mean_shape",
        "mean_inf",
        "missing_separator",
        "missing_hippocampus",
        "separator_inputs",
        "separator_winners",
        "separator_seed",
        "separator_center",
        "separator_unknown",
        "strength_shape",
        "mass_shape",
        "mass_negative",
        "consolidated_shape",
        "consolidated_nan",
        "kind",
        "pre_bounds",
        "pre_fraction",
        "writes",
        "normalize",
        "missing_rate",
        "persistent_parameters",
    ],
)
def test_malformed_memory_rejected_before_resumed_composition_is_constructed(
    tmp_path, monkeypatch, defect
):
    original = separated_brain()
    path = original.save(tmp_path / "malformed.npz")

    def corrupt(metadata, arrays):
        memory = metadata["hippocampus"]
        if defect.startswith("missing_") and defect in ("missing_projection", "missing_mean"):
            del arrays["episodic/separator/" + defect.removeprefix("missing_")]
        elif defect == "projection_shape":
            arrays["episodic/separator/projection"] = np.zeros((3, 15))
        elif defect == "projection_nan":
            arrays["episodic/separator/projection"][0, 0] = np.nan
        elif defect == "mean_shape":
            arrays["episodic/separator/mean"] = np.zeros(4)
        elif defect == "mean_inf":
            arrays["episodic/separator/mean"][0] = np.inf
        elif defect == "missing_separator":
            del memory["separator"]
        elif defect == "missing_hippocampus":
            metadata["hippocampus"] = None
        elif defect.startswith("separator_"):
            name = defect.removeprefix("separator_")
            memory["separator"][name] = {
                "inputs": 4,
                "winners": 17,
                "seed": -1,
                "center": 0.2,
                "unknown": 0,
            }[name]
        elif defect == "strength_shape":
            arrays["episodic/strength"] = np.zeros((2, 3, 3))
        elif defect == "mass_shape":
            arrays["episodic/mass"] = np.zeros(3)
        elif defect == "mass_negative":
            arrays["episodic/mass"][0] = -1
        elif defect == "consolidated_shape":
            arrays["episodic/consolidated"] = np.zeros((3, 3))
        elif defect == "consolidated_nan":
            arrays["episodic/consolidated"][0, 0] = np.nan
        elif defect == "kind":
            memory["kind"] = "unknown"
        elif defect == "pre_bounds":
            arrays["episodic/pre"][0] = original.connectome.n
        elif defect == "pre_fraction":
            arrays["episodic/pre"] = arrays["episodic/pre"].astype(float) + 0.5
        elif defect == "writes":
            memory["writes"] = -1
        elif defect == "normalize":
            memory["normalize"] = "false"
        elif defect == "missing_rate":
            del memory["rate"]
        elif defect == "persistent_parameters":
            memory["persistent_parameters"] = 1

    rewrite_checkpoint(path, corrupt)
    before = path.read_bytes()

    def must_not_construct(*args, **kwargs):
        pytest.fail("malformed memory must be rejected before constructing the resumed brain")

    monkeypatch.setattr(cd.Brain, "__init__", must_not_construct)
    with pytest.raises(ValueError):
        cd.Brain.load(path)
    assert path.read_bytes() == before


def test_the_memories_are_one_ordered_family_under_one_contract():
    """``brain.memories`` names the present memories in the order the brain reads them; each
    member is the brain's own attribute and keeps the contract of ``cadence.memories``."""
    from cadence.memories import Memories, Moment

    brain = cd.Brain.compose(4, 2, modules=(8,), seed=3, recall=True, efference_amplitude=1.0,
                             history_decays=(0.2, 0.8))
    memories = brain.memories
    assert isinstance(memories, Memories)
    assert list(memories) == ["working_memory", "efference", "history", "recall", "hippocampus"]
    for name, memory in memories.items():
        assert memory is getattr(brain, name)
        for method in ("stimulus", "branch", "reset", "keep", "to_dict", "arrays"):
            assert callable(getattr(memory, method)), (name, method)
        for alias in ("begin", "imagined"):  # one name per memory for fresh streams, a copy
            assert not hasattr(memory, alias), (name, alias)
    assert list(cd.Brain.compose(4, 2, modules=(8,), episodic=False).memories) == [
        "working_memory"
    ]
    x = np.eye(4)[[0]]
    brain.act(x, greedy=True)
    drive = np.zeros((1, brain.connectome.n))
    moment = Moment(x, x, None, True)
    read = memories.stimulus(drive, moment)
    for memory in memories.values():
        drive = memory.stimulus(drive, moment)
    np.testing.assert_array_equal(read, drive)  # the family reads as its members in order
    branch = memories["working_memory"].branch()
    branch.trace[:] = 7.0
    assert not np.any(brain.working_memory.trace == 7.0)  # a branch's state is its own
    brain.memories["recall"].reset(1)
    assert brain.recall.slots == 0
    with pytest.raises(ValueError, match="unknown memory"):
        Memories({"cortex": brain.working_memory})


def test_the_episodic_records_are_the_last_member_and_save_as_their_group(tmp_path):
    """With ``memory_choice`` the episodic records are the member ``episodic``: they read the
    associative store and keep the context records, add nothing to the drive (their vote
    joins after the settle), give a branch its own temporal context, and save as the group
    ``memories/episodic/`` from their own settings and arrays."""
    import json

    from cadence.memories import EpisodicRecords, Moment

    brain = cd.Brain.compose(4, 2, modules=(8,), seed=3, memory_choice=True)
    records = brain.memories["episodic"]
    assert isinstance(records, EpisodicRecords) and list(brain.memories)[-1] == "episodic"
    assert records.place is brain.hippocampus and records.temporal is brain.context_memory
    assert (records.steps, records.demonstrations) == (8, 1.0)  # constants since 0.93.0
    assert "episodic" not in cd.Brain.compose(4, 2, modules=(8,), seed=3).memories
    x = np.eye(4)[[0]]
    brain.step(x)
    for reward in (0.0, 1.0):
        brain.step(np.eye(4)[[int(reward) + 1]], reward=np.array([reward]))
    drive = np.ones((1, brain.connectome.n))
    assert records.stimulus(drive, Moment(x, x, None, False)) is drive
    branch = records.branch()
    assert branch.place is records.place and branch.temporal is records.temporal
    branch.move(branch.key(x), np.array([1]))
    assert not np.array_equal(branch.context, records.context)  # the branch's own context
    saved = np.load(brain.save(tmp_path / "brain.npz"))
    meta = json.loads(str(saved["brain"]))
    assert meta["memories"]["episodic"] == records.to_dict()
    group = {k[len("memories/episodic/"):] for k in saved.files if k.startswith("memories/episodic/")}
    assert group == set(records.arrays())
    loaded = cd.Brain.load(tmp_path / "brain.npz").memories["episodic"]
    np.testing.assert_array_equal(loaded.context, records.context)
    assert len(loaded.paths[0]) == len(records.paths[0])
    brain.reset()
    assert records.context is None and records.decision is None and records.paths == []


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("keep", ["slots", ("slots", "arousal")])
def test_retained_recall_slots_resume_when_saved_immediately_after_reset(tmp_path, legacy, keep):
    """A reset can keep the old streams' recall slots before the next streams act. Saving
    at that boundary preserves the slots and the following acquired life in either format."""
    original = cd.Brain.compose(4, 2, modules=(5,), seed=9, recall=True, arousal=True)
    observations = np.eye(4)
    original.step(observations[[0, 1]])
    original.step(observations[[1, 2]], reward=[1.0, -0.5])
    original.reset(keep=keep)
    assert original.basal_ganglia.state is None and original.recall.slots == 2
    path = original.save(tmp_path / "between-streams.npz")
    if legacy:
        with np.load(path, allow_pickle=False) as saved:
            arrays = earlier({name: saved[name].copy() for name in saved.files})
        np.savez(path, **arrays)
    restored = cd.Brain.load(path)
    for name, value in original.recall.arrays().items():
        np.testing.assert_array_equal(value, restored.recall.arrays()[name], err_msg=name)
    for index in range(3):
        x = observations[[index, (index + 1) % 4]]
        feedback = {} if index == 0 else {"reward": [0.5, -0.25]}
        np.testing.assert_array_equal(original.step(x, **feedback), restored.step(x, **feedback))
        np.testing.assert_array_equal(original.brain.efficacy, restored.brain.efficacy)
        np.testing.assert_array_equal(original.brain.bias, restored.brain.bias)
        np.testing.assert_array_equal(
            original.basal_ganglia.state.activation, restored.basal_ganglia.state.activation
        )
        for name, value in original.recall.arrays().items():
            np.testing.assert_array_equal(value, restored.recall.arrays()[name], err_msg=name)
