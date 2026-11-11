"""``Brain.reset(keep=("arousal",))``: a reset that keeps what an adult brain was used to.

The arousal state carries into the new stream as it is (its level, the counts, the recent and
long-run reward, the usual errors, the reward scale, the age and the work counts), as when the
state saved before the reset is restored after it. At the default, ``keep=()``,
every brain resets and acts as in 0.84.3 and saves what 0.84.3 saved but the record channel
it no longer has: the founders, their checkpoints and the composed lives pinned below were
recorded on 0.84.3.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from checkpoints import metadata

import cadence as cd
from cadence import brain as cadence_brain

# The acts and the array contents below were recorded on the NumPy solver; the compiled dense
# kernel (numba) orders the arithmetic otherwise and gives other bits, so those pins are read
# on the NumPy solver alone (CADENCE_FUSED=0 selects it). The connectome, the parameters and
# the checkpoint's members and metadata keys are read on every backend.
NUMPY_SOLVER = not cadence_brain._FUSED
BYTES_REASON = (
    "acts and array contents were recorded on the NumPy solver; this run uses the compiled "
    "dense kernel (CADENCE_FUSED=0 selects the NumPy loop)"
)
OBS = np.array([[1.0]])  # the bandit's one constant observation
CUE = np.array([[1.0, 0.0, 0.5, 0.0]])
GO = np.array([[0.0, 0.0, 0.0, 1.0]])
FACT = np.array([[1.0, 0.0, 0.0, 1.0]])
NAME = np.array([[1.0, 0.0, 0.0, 0.0]])
STREAM = (CUE, GO, FACT, NAME)
# What the default reset forgets beside the level: the counts and the running statistics.
STATISTICS = ("outcomes", "rewards", "spreads", "square", "recent", "longrun", "usual")


def saved(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """A checkpoint's metadata without the library version that wrote it, and its arrays."""
    with np.load(path, allow_pickle=False) as data:
        meta = metadata(data)
        arrays = {name: data[name].copy() for name in data.files if name not in ("generic", "brain")}
    meta.pop("version", None)
    return meta, arrays


def assert_same_checkpoint(first: Path, second: Path, *, but: tuple[str, ...] = ()) -> None:
    """The two checkpoints hold the same arrays and the same metadata, ``but`` the named
    metadata entries."""
    (meta_a, arrays_a), (meta_b, arrays_b) = saved(first), saved(second)
    for name in but:
        meta_a.pop(name, None)
        meta_b.pop(name, None)
    assert meta_a == meta_b
    assert set(arrays_a) == set(arrays_b)
    for name in arrays_a:
        np.testing.assert_array_equal(arrays_a[name], arrays_b[name], err_msg=name)


def twin(brain: cd.Brain, path: Path) -> cd.Brain:
    return cd.Brain.load(brain.save(path))


def adult_and_action(seed: int, moments: int = 300) -> tuple[cd.Brain, int]:
    """A brain that lived ``moments`` moments of a two-armed bandit in which arm 0 pays 1.0
    and arm 1 nothing (its youth of 100 moments ends inside them), and its issued action."""
    brain = cd.Brain.compose(inputs=1, actions=2, modules=(16,), seed=seed, arousal=True, recall=False)
    action = int(brain.live(OBS)[0])
    for _ in range(moments):
        action = int(brain.live(OBS, reward=[float(action == 0)])[0])
    return brain, action


def bandit_adult(seed: int, moments: int = 300) -> cd.Brain:
    return adult_and_action(seed, moments)[0]


def reading(brain: cd.Brain) -> Mapping[str, Any]:
    """The readings of the brain's latest ``live`` moment."""
    last = brain.last_arousal
    assert last is not None
    return last


# What arm 0 and arm 1 pay in the new stream of a bandit adult.
PAYS = {"moved": (0.0, 1.0), "same": (1.0, 0.0), "more": (2.0, 0.0)}
# One moment of a stream: the action, whether it was chosen aroused, whether its outcome was
# learned from, and the settle sweeps of the call that took that outcome.
Moment = tuple[int, bool, bool, int]


def stream_life(path: Path, issued: tuple[int, bool], arm: str, stream: str) -> list[Moment]:
    """120 moments of a new stream for the saved adult: living on from the action it issued
    (``none``), after a default reset (``reset``) or after ``reset(keep=("arousal",))``
    (``keep``). A reset stream starts with ``live(OBS)``, which takes no outcome."""
    pays = PAYS[stream]
    brain = cd.Brain.load(path)
    action, aroused = issued
    if arm != "none":
        brain.reset(keep=("arousal",) if arm == "keep" else ())
        action = int(brain.live(OBS)[0])
        aroused = reading(brain)["mode"] == "aroused"
    moments: list[Moment] = []
    for _ in range(120):
        following = int(brain.live(OBS, reward=[pays[action]])[0])
        read = reading(brain)
        moments.append((action, aroused, bool(read["learned"]), int(read["sweeps"])))
        action, aroused = following, read["mode"] == "aroused"
    return moments


@pytest.fixture(scope="module")
def bandit(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, str, int], list[Moment]]:
    """Seeds 0 to 4: each bandit adult meets three new streams (its paying arm ``moved``, the
    ``same`` arm paying 1.0, and it paying 2.0, ``more``) in three ways."""
    folder = tmp_path_factory.mktemp("bandit")
    lives = {}
    for seed in range(5):
        adult, action = adult_and_action(seed)
        issued = (action, reading(adult)["mode"] == "aroused")
        path = adult.save(folder / f"adult-{seed}")
        for stream in PAYS:
            for arm in ("none", "reset", "keep"):
                lives[(stream, arm, seed)] = stream_life(path, issued, arm, stream)
    return lives


# -- Brain.reset(keep=("arousal",))


def woken_adult() -> cd.Brain:
    """A bandit adult whose arm stopped paying until it woke: aroused, with a sampled action
    awaiting its outcome."""
    brain = bandit_adult(seed=1, moments=150)
    for _ in range(30):
        brain.live(OBS, reward=[0.0])
        if reading(brain)["mode"] == "aroused":
            break
    arousal = brain.arousal
    assert arousal is not None and arousal.age > arousal.config.youth
    assert arousal.level >= arousal.config.threshold and arousal.outcomes > 0
    return brain


def test_keeping_the_arousal_keeps_the_whole_arousal_state_level_included(
    tmp_path: Path,
) -> None:
    """The arousal is the only thing ``keep=("arousal",)`` keeps, and it keeps all of it: an adult
    aroused when it is reset begins the new stream aroused. The rest of the brain resets as
    ``keep=()`` resets it."""
    brain = woken_adult()
    arousal = brain.arousal
    assert arousal is not None
    before = arousal.to_dict()
    default = twin(brain, tmp_path / "twin")
    brain.reset(keep=("arousal",))
    default.reset(keep=())
    kept = brain.arousal
    assert kept is not None and kept.to_dict() == before
    assert kept.mode == "aroused" and not brain.pending_feedback
    assert brain.basal_ganglia.state is None and brain.last_arousal is None
    left = default.arousal
    assert left is not None and left.age == kept.age and left.mode == "routine"
    assert left.level == 0.0 and all(left.to_dict()[name] == 0 for name in STATISTICS)
    assert_same_checkpoint(
        brain.save(tmp_path / "kept"), default.save(tmp_path / "forgotten"), but=("arousal",)
    )


def test_keeping_the_arousal_continues_as_the_documented_restore(tmp_path: Path) -> None:
    """A reset with ``keep=("arousal",)`` continues as a default reset after which the arousal
    state saved before it is restored: the same acts, readings and checkpoint, here for an
    adult aroused when it is reset."""
    brain = woken_adult()
    restored = twin(brain, tmp_path / "restored")
    assert restored.arousal is not None
    state = restored.arousal.to_dict()
    brain.reset(keep=("arousal",))
    restored.reset()
    restored.arousal = cd.Arousal.from_dict(state)
    lives = []
    for b in (brain, restored):
        acts = [int(b.live(OBS)[0])]
        readings = [dict(reading(b))]
        for _ in range(40):
            acts.append(int(b.live(OBS, reward=[float(acts[-1] == 1)])[0]))
            readings.append(dict(reading(b)))
        lives.append((acts, readings))
    assert lives[0] == lives[1]
    assert lives[0][1][0]["mode"] == "aroused"  # the first decision of the new stream
    assert_same_checkpoint(brain.save(tmp_path / "a"), restored.save(tmp_path / "b"))


def test_keeping_the_arousal_is_the_default_reset(tmp_path: Path) -> None:
    brain = bandit_adult(seed=2, moments=120)
    other = twin(brain, tmp_path / "other")
    brain.reset(keep=("arousal",))
    other.reset()
    for b in (brain, other):
        b.live(OBS)
        for _ in range(5):
            b.live(OBS, reward=[1.0])
    assert_same_checkpoint(brain.save(tmp_path / "a"), other.save(tmp_path / "b"))


def test_a_brain_without_arousal_resets_as_without_the_keyword(tmp_path: Path) -> None:
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=3, recall=False)
    brain.step(CUE)
    brain.step(GO, reward=[1.0])
    other = twin(brain, tmp_path / "other")
    brain.reset(keep=("arousal",))
    other.reset()
    assert brain.arousal is None
    for b in (brain, other):
        b.step(FACT)
        b.step(NAME, reward=[0.5])
    assert_same_checkpoint(brain.save(tmp_path / "a"), other.save(tmp_path / "b"))


def test_keeping_the_slots_and_the_arousal_keeps_both(tmp_path: Path) -> None:
    """Together the two names keep the recall memory's slots and the arousal state; the brain
    then continues as one reset with ``keep=("slots",)`` whose arousal state, saved before
    the reset, is restored after it."""
    brain = cd.Brain.compose(4, 2, modules=(8,), seed=5, recall=True, recall_founder=True,
                             arousal=cd.ArousalConfig(youth=6))
    brain.live(CUE)
    for t in range(1, 12):
        brain.live(STREAM[t % 4], reward=[float(t % 3 == 0)])
    assert brain.recall is not None and brain.arousal is not None
    keys = brain.recall.keys.copy()
    state = brain.arousal.to_dict()
    restored = twin(brain, tmp_path / "restored")
    forgotten = twin(brain, tmp_path / "forgotten")
    brain.reset(keep=("slots", "arousal"))
    restored.reset(keep=("slots",))
    restored.arousal = cd.Arousal.from_dict(state)
    forgotten.reset(keep=("slots",))
    np.testing.assert_array_equal(brain.recall.keys, keys)
    assert brain.recall.ended.all()
    assert brain.arousal.to_dict() == state
    assert forgotten.arousal is not None
    assert all(forgotten.arousal.to_dict()[name] == 0 for name in STATISTICS)
    # the recall memory and the rest of the stream are those of the reset that keeps the memory
    assert_same_checkpoint(brain.save(tmp_path / "kept"), forgotten.save(tmp_path / "forgot"),
                           but=("arousal",))
    acts = []
    for b in (brain, restored):
        lived = [int(b.live(GO)[0])]
        for t in range(6):
            lived.append(int(b.live(STREAM[t % 4], reward=[float(t % 2)])[0]))
        acts.append(lived)
    assert acts[0] == acts[1]
    assert_same_checkpoint(brain.save(tmp_path / "a"), restored.save(tmp_path / "b"))


def test_a_checkpoint_after_a_reset_keeping_the_arousal_continues_as_the_brain(
    tmp_path: Path,
) -> None:
    brain = bandit_adult(seed=3, moments=130)
    brain.reset(keep=("arousal",))
    copy = twin(brain, tmp_path / "reset")
    assert copy.arousal is not None and brain.arousal is not None
    assert copy.arousal.to_dict() == brain.arousal.to_dict()
    acts: list[list[int]] = [[], []]
    for b, out in zip((brain, copy), acts, strict=True):
        out.append(int(b.live(OBS)[0]))
    mid = twin(brain, tmp_path / "pending")  # saved while its first action awaits its outcome
    acts.append(list(acts[0]))
    for b, out in zip((brain, copy, mid), acts, strict=True):
        for _ in range(20):
            out.append(int(b.live(OBS, reward=[float(out[-1] == 1)])[0]))
    assert acts[0] == acts[1] == acts[2]
    end = brain.save(tmp_path / "end")
    assert_same_checkpoint(end, copy.save(tmp_path / "copy-end"))
    assert_same_checkpoint(end, mid.save(tmp_path / "mid-end"))


@pytest.mark.parametrize("value", [1, True, "yes", ("arousal", "level"), None, (None,)])
def test_keep_names_slots_and_arousal_and_a_refused_reset_changes_nothing(
    tmp_path: Path, value: Any
) -> None:
    brain = bandit_adult(seed=4, moments=110)
    before = brain.save(tmp_path / "before")
    with pytest.raises(ValueError, match="'slots' and 'arousal'"):
        brain.reset(keep=value)
    for name in ("keep_arousal", "keep_memory"):  # the keywords of 0.84.3: unknown keywords
        with pytest.raises(TypeError, match=name):
            brain.reset(**{name: True})
    assert brain.pending_feedback
    assert_same_checkpoint(before, brain.save(tmp_path / "after"))


# -- the reproduction: a two-armed bandit whose paying arm moved


@pytest.mark.parametrize("seed", range(5))
def test_a_reset_adult_learns_the_moved_arm_when_it_keeps_what_it_was_used_to(
    bandit: dict[tuple[str, str, int], list[Moment]], seed: int
) -> None:
    """An adult reset by default has no income or error references: it is never roused and
    chooses the newly paying arm (arm 1) at none of 120 moments. Reset with
    ``keep=("arousal",)`` it falls short of the income it is used to, wakes and learns the new
    arm."""
    default, kept = bandit[("moved", "reset", seed)], bandit[("moved", "keep", seed)]
    assert [sum(m[i] for m in default) for i in range(3)] == [0, 0, 0]
    assert sum(m[1] for m in kept) > 0 and sum(m[2] for m in kept) > 0
    assert sum(m[0] for m in kept[-40:]) >= 36


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
def test_the_documented_bandit_counts(bandit: dict[tuple[str, str, int], list[Moment]]) -> None:
    """The counts ``docs/start.md#reset-and-save`` gives for seeds 0 to 4."""

    def counts(stream: str, arm: str, field: int, last: int = 120) -> list[int]:
        return [sum(m[field] for m in bandit[(stream, arm, seed)][-last:]) for seed in range(5)]

    # moved: a default reset never chooses the new arm; a kept reset chooses as the brain
    # without a reset, in the same mode, at every moment, the new arm at each of the last 40
    assert counts("moved", "reset", 0) == [0] * 5
    for seed in range(5):
        assert [m[:2] for m in bandit[("moved", "keep", seed)]] == [
            m[:2] for m in bandit[("moved", "none", seed)]
        ]
    assert counts("moved", "keep", 0, last=40) == [40] * 5
    # the same arm pays as before: kept at every moment, calm at every moment
    assert all(m[:2] == (0, False) for seed in range(5) for m in bandit[("same", "keep", seed)])
    # after either reset the first 5 to 12 moments settle from rest at 32 sweeps each; the
    # brain living on needs no sweep
    for arm in ("reset", "keep"):
        for seed, lead in enumerate((5, 5, 12, 10, 5)):
            sweeps = [m[3] for m in bandit[("same", arm, seed)]]
            assert sweeps[:lead] == [32] * lead and not any(sweeps[lead:])
    assert not any(m[3] for seed in range(5) for m in bandit[("same", "none", seed)])
    # the arm pays 2.0: aroused at 9 of 600 decisions after a kept reset and at 6 of 600
    # without a reset, at none of the last 40 of any seed
    assert counts("more", "keep", 1) == [7, 0, 2, 0, 0]
    assert counts("more", "none", 1) == [6, 0, 0, 0, 0]
    assert counts("more", "keep", 1, last=40) == [0] * 5


# -- the founders' lives with the default reset, which keeps the arousal


def _checkpoint_digest(path: Path) -> str:
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


def _digest(brain: cd.Brain) -> str:
    """The founder: its connectome and parameters."""
    connectome, graph = brain.connectome, brain.brain
    h = hashlib.sha256()
    for values in (connectome.pre, connectome.post, connectome.count, connectome.sign,
                   graph.efficacy, graph.bias, graph.log_gain):
        h.update(np.ascontiguousarray(np.asarray(values)).tobytes())
    populations = {k: [int(i) for i in v] for k, v in connectome.populations.items()}
    h.update(json.dumps(populations).encode())
    return h.hexdigest()


def _names(path: Path) -> tuple[str, str]:
    """The checkpoint's format and a digest of its metadata keys and member names."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["brain"]))
        names = json.dumps([sorted(meta), sorted(data.files)])
    return str(meta["format"]), hashlib.sha256(names.encode()).hexdigest()


def _adult(brain: cd.Brain) -> tuple[str, list[dict[str, Any]]]:
    """The founder's bandit life: 150 moments of arm 0 paying, a reset, 30 moments of arm 1
    paying. The acts and what the reset left the arousal."""
    acts = [int(brain.live(OBS)[0])]
    for _ in range(150):
        acts.append(int(brain.live(OBS, reward=[float(acts[-1] == 0)])[0]))
    brain.reset()
    assert brain.arousal is not None
    left = [brain.arousal.to_dict()]
    acts.append(int(brain.live(OBS)[0]))
    for _ in range(30):
        acts.append(int(brain.live(OBS, reward=[float(acts[-1] == 1)])[0]))
    return "".join(map(str, acts)), left


def _streams(brain: cd.Brain) -> tuple[str, list[dict[str, Any]]]:
    """Three streams of a composed life: live moments with a wait and episode ends, a
    reset, more moments, a reset that keeps the memory, more moments, an imagined branch."""
    acts, left = [], []
    for start, length in ((0, 14), (14, 8), (22, 8)):
        if start == 14:
            brain.reset()
        elif start == 22:
            brain.reset(keep=("slots",))
        if start and brain.arousal is not None:
            left.append(brain.arousal.to_dict())
        for t in range(start, start + length):
            first = t == start
            reward = None if first else [float(t % 3 == 0)]
            done = None if first else [t % 7 == 6]
            acts.append(int(brain.live(STREAM[t % 4], reward=reward, done=done)[0]))
            if t == 5:
                brain.wait(GO)
    phases = brain.imagine([CUE, GO, FACT])
    acts.extend(
        int(np.argmax(np.asarray(phase.state.activation)[0, brain.motor_index]))
        for phase in phases
    )
    return "".join(map(str, acts)), left


def _stepped(brain: cd.Brain) -> tuple[str, list[dict[str, Any]]]:
    """A brain without arousal: step, teach and end episodes, with two resets."""
    acts = []
    for t in range(24):
        x = STREAM[t % 4]
        if t in (0, 8, 16):
            if t:
                brain.reset()
            action = brain.step(x)
        elif t % 3 == 0:
            action = brain.step(x, reward=[float(t % 2)], teacher=[t % 2])
        else:
            action = brain.step(x, reward=[float(t % 2) - 0.5], done=[t == 7])
        acts.append(int(np.asarray(action).ravel()[0]))
    acts.append(int(brain.act(GO, greedy=True)[0]))
    return "".join(map(str, acts)), []


Life = Callable[[cd.Brain], tuple[str, list[dict[str, Any]]]]


def _compact(state: dict[str, Any]) -> list[float]:
    """What a reset left the arousal: the level and the statistics, then the age and the work
    counts (routine and aroused moments, routine and aroused sweeps, learning sweeps)."""
    return [
        state["level"], *(state[name] for name in STATISTICS), state["age"],
        *state["moments"].values(), *state["sweeps"].values(), state["learning_sweeps"],
    ]

# Recorded on 0.84.3 (8dc92d8): each founder's connectome and parameters, the acts of its
# life and what each reset left its arousal, and the format, the member and metadata names
# (``names``) and the contents of every array (``checkpoint``) of the checkpoint it saves.
# Where the library later removed a setting that was inert at its founder, the names and
# checkpoint digests were re-recorded after every other value matched the recording.
# Re-recorded with the unset motor lateral at 0.0 (issue 220), which every founder here meets,
# and at the founder need scale 0.1 (issue 198) where a founder has arousal.
# Re-recorded with the default reset keeping the arousal (issue 213).
# Re-recorded at 0.95.2, where a lesson's free phase starts from the stream's activity the act
# settles from: the lives that teach after an act read new checkpoint digests (their efficacies
# move by at most 0.0021).
FOUNDERS: dict[str, dict[str, Any]] = {
    "adult": {
        "build": (1, 2, {"modules": (16,), "seed": 0, "arousal": True}),
        "life": _adult,
        "digest": "32f58e96ba5ed96a5b25207c07f1c01618c02af6b80c7cbf915f8c9b16e35a9f",
        "acts": (
            "10000000000000000000000000110000000000000000000000000000000000000000000000000100"
            "00000000000100110100000000000000000000000000000000000000000000000000000000000000"
            "1011010111010111101100"
        ),
        "left": [
            [0.0034539743906799613, 142, 150, 106, 0.028762852226372478, 0.9859539727985247, 0.5020542546498477, 0.29108554865201863, 151, 46, 105, 128, 3296, 544],
        ],
        "format": "cadence-brain/1",
        "names": "51719b2e743a574301384fd5a9fd16fd200f979ba8585760a6db73faaaa43233",
        "checkpoint": "efae085bd5d1c6288381463d39627b10ef57e3e65a5520304b40c65fcf6df35b",
    },
    # recall without the position term of 0.89.0: the acts predate it; the checkpoint names
    # recall_position=0, which differs from its founder
    "composed": {
        "build": (4, 2, {"modules": (8,), "seed": 5, "arousal": cd.ArousalConfig(youth=10),
                         "recall": True, "recall_founder": True, "recall_position": 0, "history_decays": (0.2, 0.9),
                         "efference_amplitude": 1.0}),
        "life": _streams,
        "digest": "bf4d075047bb0d0713d3ab20312bd17853820de818de29e228b2edaf5bcdd26c",
        "acts": "111101100100011101101111111001011",
        "left": [
            [1.0124686448349702, 9, 13, 13, 0.011089548956681777, 0.15307902426660647, 0.019458901389501283, 0.016780853750205795, 14, 0, 14, 0, 256, 140],
            [0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 22, 0, 22, 0, 448, 206],
        ],
        "format": "cadence-brain/1",
        "names": "e78b45e964572a6df43fb29ef5d436b89ce341e486e4f55a1708b355cacefd00",
        "checkpoint": "43ee564c91f2e3a077647bc3f57fa8ef105713e10b5516b0fe292f7d425a8f58",
    },
    "episodic": {
        "build": (4, 2, {"modules": (8,), "seed": 7, "arousal": True, "memory_choice": True}),
        "life": _streams,
        "digest": "dcbc0983cda21c93c39ec2e5f75b9deebe3af815e53bdd9e9af7befbc7afac53",
        "acts": "111111011111111111111111111111111",
        "left": [
            [0.2489490507255386, 10, 13, 13, 0.011089548956681777, 0.15307902426660647, 0.019458901389501283, 0.016346162123862428, 14, 0, 14, 0, 256, 196],
            [0.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 22, 0, 22, 0, 480, 294],
        ],
        "format": "cadence-brain/1",
        "names": "1c418d7da8207299c6e8058e90cd3f627a6c3119a3287e65090a2cbd06ed603a",
        "checkpoint": "905782491bb02f68bf3956d935623ba5e53e9ec7389921cfa7682c185f9dcb58",
    },
    "stepped": {
        "build": (4, 2, {"modules": (8,), "seed": 3}),
        "life": _stepped,
        "digest": "8d42b245bdc9b36499675f913258d4ca9e532a91ad39a0bda82b666e4a441837",
        "acts": "0011010000000111111101000",
        "left": [],
        "format": "cadence-brain/1",
        "names": "cf4f510fd7745ee140166219a1a6e0af36750c92b9eb41a38f9262d2272f7dda",
        "checkpoint": "be33cac231046736901e98a9ef4fe60952bcdbe4edb2e6cb14ebbde8ee61ce20",
    },
}


def _build(spec: dict[str, Any]) -> cd.Brain:
    inputs, actions, options = spec["build"]
    return cd.Brain.compose(inputs, actions, **{"recall": False, **options})


@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_the_default_reset_lives_as_recorded(tmp_path: Path, name: str) -> None:
    """On every backend: the founder's connectome and parameters, and the checkpoint's
    format, members and metadata keys after its life."""
    spec = FOUNDERS[name]
    brain = _build(spec)
    assert _digest(brain) == spec["digest"]
    life: Life = spec["life"]
    life(brain)
    assert _names(brain.save(tmp_path / "founder")) == (spec["format"], spec["names"])


@pytest.mark.skipif(not NUMPY_SOLVER, reason=BYTES_REASON)
@pytest.mark.parametrize("name", sorted(FOUNDERS))
def test_the_default_resets_acts_and_checkpoint_bytes_are_the_recorded_ones(
    tmp_path: Path, name: str
) -> None:
    spec = FOUNDERS[name]
    brain = _build(spec)
    life: Life = spec["life"]
    acts, left = life(brain)
    assert acts == spec["acts"]
    assert [_compact(state) for state in left] == spec["left"]
    assert _checkpoint_digest(brain.save(tmp_path / "founder")) == spec["checkpoint"]


# -- a reset that keeps the arousal resets every other store as the default reset does


def _filled(brain: cd.Brain) -> None:
    """Twelve moments of one stream with a ``wait`` among them and no episode end, so that
    every per-stream store of the brain holds something when it is reset."""
    brain.live(CUE)
    for t in range(1, 12):
        brain.live(STREAM[t % 4], reward=[float(t % 3 == 0)])
        if t == 5:
            brain.wait(GO)


@pytest.mark.parametrize("slots", [False, True])
@pytest.mark.parametrize("name", ["composed", "episodic"])
def test_keeping_the_arousal_resets_every_other_store_as_the_default_reset(
    tmp_path: Path, name: str, slots: bool
) -> None:
    """The recall slots (emptied, or kept with ``"slots"``), the heard history, the
    efference copy and the episodic context and path: a reset that keeps the arousal resets
    each of them as the default reset does."""
    brain = _build(FOUNDERS[name])
    _filled(brain)
    if name == "composed":
        assert brain.recall is not None and brain.recall.slots > 0
        assert brain.history is not None and brain.history.trace.any()
        assert brain.efference is not None and brain.efference.trace.any()
    else:
        held = set(saved(brain.save(tmp_path / "held"))[1])
        assert {f"memories/episodic/{name}" for name in ("context", "decision", "path/lengths")} <= held
    default = twin(brain, tmp_path / "default")
    brain.reset(keep=("slots", "arousal") if slots else ("arousal",))
    default.reset(keep=("slots",) if slots else ())
    assert_same_checkpoint(
        brain.save(tmp_path / "kept"), default.save(tmp_path / "reset"), but=("arousal",)
    )
    if brain.recall is not None:
        assert (brain.recall.slots > 0) == slots


@pytest.mark.parametrize("name", ["composed", "episodic"])
def test_keeping_the_arousal_continues_as_the_documented_restore_with_every_store(
    tmp_path: Path, name: str
) -> None:
    """With recall, heard history and efference copy, or with the episodic memory, a reset
    that keeps the arousal continues as a default reset after which the arousal state saved
    before it is restored: the same acts, readings and checkpoint."""
    brain = _build(FOUNDERS[name])
    _filled(brain)
    restored = twin(brain, tmp_path / "restored")
    assert restored.arousal is not None
    state = restored.arousal.to_dict()
    brain.reset(keep=("arousal",))
    restored.reset()
    restored.arousal = cd.Arousal.from_dict(state)
    lives = []
    for b in (brain, restored):
        acts = [int(b.live(GO)[0])]
        readings = [dict(reading(b))]
        for t in range(10):
            acts.append(int(b.live(STREAM[t % 4], reward=[float(t % 3 == 0)], done=[t == 6])[0]))
            readings.append(dict(reading(b)))
        lives.append((acts, readings))
    assert lives[0] == lives[1]
    assert_same_checkpoint(brain.save(tmp_path / "a"), restored.save(tmp_path / "b"))
