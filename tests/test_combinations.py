"""Opt-in options taken two at a time: every pair of levels of the options below meets in at least
one brain of a covering array. Each brain composes, lives a few moments (a lesson, a greedy act,
sampled moments with a reward), saves while an action awaits its outcome, loads, and the loaded
brain continues as the brain it was saved from: the same actions and, saved again, the same
arrays. ``grow`` and ``Lives`` are tried on every brain, and either take it or refuse it with a
``ValueError`` that names a part.

Where two options are known to refuse each other, the pair is left out of the array and read in
``test_a_refused_pair_is_refused_by_name``. Every other failure, of any type, is a finding.
"""

from __future__ import annotations

import itertools
import random
import warnings
from typing import Any

import numpy as np
import pytest

import cadence as cd
from cadence import Brain
from cadence import brain as kernel
from cadence.space.map import Map

# Each option and its levels; the first level is "off", the brain of 0.101.0 without it.
FACTORS: dict[str, tuple[str, ...]] = {
    "layout": ("inputs", "senses"),
    "code": ("none", "fixed", "learned"),
    "recall": ("default", "off", "heads2", "strength"),
    "surprise": ("off", "on"),
    "bound": ("off", "voice"),
    "trained": ("off", "place", "map"),
    "longterm": ("off", "records", "strength"),
    "message": ("off", "on"),
    "memory_choice": ("off", "on"),
    "maturity": ("off", "on"),
    "fade": ("off", "on"),
    "slots": ("one", "two"),
    "depth": ("one", "deep"),
    "observers": ("off", "on"),
    "history": ("default", "two"),
    "efference": ("off", "on"),
    "arousal": ("off", "on"),
}

SENSES = {"eyes": {"hue": 8, "shape": 8}, "ear": 16}
INPUTS = 32
TURN = INPUTS - 1  # the last ear channel marks a turn for the held message's records

# What an option needs to be built at all: a level that cannot be composed without another.
NEEDS = [
    ("bound", "voice", "layout", ("senses",)),  # records of named senses
    ("bound", "voice", "code", ("fixed",)),  # keyed by the fixed code where they meet
    ("longterm", "records", "message", ("on",)),  # long-term records key on the message
    ("longterm", "strength", "message", ("on",)),
    ("code", "learned", "recall", ("default", "heads2", "strength")),  # attention reads recall
]

# Pairs the library refuses with a ValueError that names the part ("What does not go
# together", docs/modules.md, and the guides of each option). Each is read below.
REFUSED = [
    ({"code": "learned", "message": "on"}, "message"),
    ({"code": "learned", "longterm": "records"}, "message"),
    ({"code": "learned", "longterm": "strength"}, "message"),
    # attention=True reads a window of two events, which a slot's strength does not hold
    ({"code": "learned", "recall": "strength"}, "strength"),
]


def _clashes(cell: dict[str, str]) -> bool:
    for factor, level, other, allowed in NEEDS:
        if cell.get(factor) == level and other in cell and cell[other] not in allowed:
            return True
    for pair, _ in REFUSED:
        if all(cell.get(f) == v for f, v in pair.items()):
            return True
    return False


def covering_array(seed: int = 0) -> list[dict[str, str]]:
    """A deterministic greedy covering array: every valid pair of levels in some row."""
    names = list(FACTORS)
    uncovered = {
        ((a, x), (b, y))
        for a, b in itertools.combinations(names, 2)
        for x in FACTORS[a]
        for y in FACTORS[b]
        if not _clashes({a: x, b: y})
    }
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    while uncovered:
        best: tuple[int, dict[str, str]] | None = None
        start = sorted(uncovered)
        for attempt in range(24):
            (a, x), (b, y) = start[0] if attempt == 0 else rng.choice(start)
            row = {a: x, b: y}
            order = [n for n in names if n not in row]
            rng.shuffle(order)
            for name in order:
                scored = []
                for level in FACTORS[name]:
                    trial = {**row, name: level}
                    if _clashes(trial):
                        continue
                    gain = sum(
                        ((n, row[n]), (name, level)) in uncovered
                        or ((name, level), (n, row[n])) in uncovered
                        for n in row
                    )
                    scored.append((gain, -FACTORS[name].index(level), level))
                row[name] = max(scored)[2]
            gain = sum(1 for p in uncovered if all(row[f] == v for f, v in p))
            if best is None or gain > best[0]:
                best = (gain, {n: row[n] for n in names})
        assert best is not None and best[0] > 0
        rows.append(best[1])
        uncovered = {p for p in uncovered if not all(best[1][f] == v for f, v in p)}
    return rows


CELLS = covering_array()

# Cells that meet a known defect of the library; each is reproduced alone at the end of this file.
F1 = (
    "#346: on the compiled kernel a held message with a turn that holds (message_boundary_holds) "
    "continues from its save in other last bits once a lesson came before (efficacy within "
    "1e-11, the same actions over 196 moments); the NumPy loop continues to the bit"
)
# Read on the compiled kernel of an Apple M4; last bits depend on the machine's compiled code,
# so the mark is not strict
F1_MARK = pytest.mark.xfail(kernel._FUSED, strict=False, reason=F1)
KNOWN = {
    "trained=map,longterm=strength,message=on,memory_choice=on,maturity=on,fade=on,observers=on,"
    "history=two,efference=on,arousal=on": F1_MARK,
}


def cells(rows: list[dict[str, str]], known: dict[str, Any] | None = None) -> list[Any]:
    known = known or {}
    return [
        pytest.param(row, id=cell_id(row), marks=known[cell_id(row)])
        if cell_id(row) in known
        else pytest.param(row, id=cell_id(row))
        for row in rows
    ]


def cell_id(cell: dict[str, str]) -> str:
    on = [f"{name}={level}" for name, level in cell.items() if level != FACTORS[name][0]]
    return ",".join(on) or "founder"


def _map(cell: dict[str, str]) -> Any:
    if cell["trained"] == "place":
        return "place"
    if cell["trained"] == "map":
        return Map((-1.0, 1.0, -1.0, 1.0), (8, 8), units=4, values={"hue": 6}, surprise=True)
    return None


def options(cell: dict[str, str]) -> dict[str, Any]:
    """The keywords of ``Brain.compose`` for a cell."""
    kw: dict[str, Any] = {"seed": 3, "backend": "cpu", "modules": (16,)}
    if cell["layout"] == "senses":
        kw["senses"] = SENSES
    else:
        kw["inputs"] = INPUTS
    kw["actions"] = 5 if cell["slots"] == "two" else 3
    if cell["slots"] == "two":
        kw["slots"] = (3, 2)
    if cell["code"] == "fixed":
        kw["code"] = 64
    elif cell["code"] == "learned":
        kw.update(learned_code=16, attention=True)
    kw.update(
        {
            "default": {},
            "off": {"recall": False},
            "heads2": {"recall_heads": 2},
            "strength": {"recall_strength": True},
        }[cell["recall"]]
    )
    if cell["surprise"] == "on":
        kw["surprise"] = True
    if cell["bound"] == "voice":
        kw.update(bound=32, voice="ear")
    held = _map(cell)
    if held is not None:
        kw["trained"] = [held]
    if cell["message"] == "on":
        kw.update(message=12, message_winners=2)
    if cell["longterm"] != "off":
        kw.update(longterm=8, message_boundary=TURN, message_boundary_holds=True)
        if cell["longterm"] == "strength":
            kw["longterm_strength"] = True
    if cell["memory_choice"] == "on":
        kw["memory_choice"] = True
    if cell["maturity"] == "on":
        kw["maturity"] = True
    if cell["fade"] == "on":
        kw["memory_fade"] = 1.2e-4
    if cell["depth"] == "deep":
        kw.update(modules=(24, 16), apical=0.25)
    if cell["observers"] == "on":
        kw["observers"] = (8,)
    if cell["history"] == "two":
        kw["history_decays"] = (0.0, 0.8)
    if cell["efference"] == "on":
        kw["efference_amplitude"] = 1.0
    if cell["arousal"] == "on":
        kw["arousal"] = True
    return kw


def observation(cell: dict[str, str], moment: int) -> Any:
    rng = np.random.default_rng((7, moment))
    row = rng.random((1, INPUTS))
    row[0, TURN] = 1.0 if moment % 4 == 2 else 0.0  # a turn every fourth moment
    if cell["layout"] == "inputs":
        return row
    return {"eyes": {"hue": row[:, :8], "shape": row[:, 8:16]}, "ear": row[:, 16:]}


def labels(cell: dict[str, str]) -> np.ndarray:
    return np.array([[2, 1]]) if cell["slots"] == "two" else np.array([1])


def report(brain: Brain, moment: int) -> None:
    """What the eyes tell a held map at a moment: one thing, moving."""
    space = getattr(brain, "space", None)
    if space is None:
        return
    place = [0.1 * (moment % 5) - 0.2, 0.5]
    if getattr(space, "values", None):
        space.moment(seen=[(0, place, {"hue": 1 + moment % 3})])
    else:
        space.moment(seen=[(0, place)])


def live(brain: Brain, cell: dict[str, str], moment: int, reward: float | None) -> np.ndarray:
    report(brain, moment)
    x = observation(cell, moment)
    paid = None if reward is None else [reward]
    if cell["arousal"] == "on":
        return np.asarray(brain.live(x, reward=paid))
    return np.asarray(brain.step(x, reward=paid))


def arrays(path: Any) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as held:
        return {name: held[name] for name in held.files}


def same_arrays(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    differ = sorted(set(a) ^ set(b))
    for name in sorted(set(a) & set(b)):
        if a[name].dtype != b[name].dtype or a[name].shape != b[name].shape:
            differ.append(name)
        elif a[name].dtype.kind in "US":
            if not np.array_equal(a[name], b[name]):
                differ.append(name)
        elif not np.array_equal(a[name], b[name], equal_nan=a[name].dtype.kind in "fc"):
            differ.append(name)
    return differ


def compose(cell: dict[str, str]) -> Brain:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Brain.compose(**options(cell))


def refusal(error: ValueError) -> str:
    message = str(error)
    assert message and len(message) > 15, f"a refusal without a sentence: {message!r}"
    return message


def test_the_array_covers_every_valid_pair():
    names = list(FACTORS)
    for a, b in itertools.combinations(names, 2):
        for x, y in itertools.product(FACTORS[a], FACTORS[b]):
            if _clashes({a: x, b: y}):
                continue
            assert any(row[a] == x and row[b] == y for row in CELLS), (a, x, b, y)
    for row in CELLS:
        assert not _clashes(row)
    assert len(CELLS) <= 40


@pytest.mark.parametrize("cell", cells(CELLS, KNOWN))
def test_a_combination_lives_saves_loads_and_continues(cell, tmp_path):
    brain = compose(cell)
    brain.lesson(observation(cell, 0), labels(cell))  # nothing awaits an outcome yet
    report(brain, 1)
    brain.act(observation(cell, 1), greedy=True)
    live(brain, cell, 2, None)  # a sampled moment at a turn
    live(brain, cell, 3, 1.0)  # the world paid it
    if brain.bound is not None:
        assert set(brain.complete({"ear": observation(cell, 3)["ear"]})) >= {"eyes"}
        assert brain.say(observation(cell, 3)).shape == (1, 16)
    saved = brain.save(tmp_path / "brain.npz")  # an action awaits its outcome
    loaded = Brain.load(saved)
    assert loaded.describe()["genes"] == brain.describe()["genes"]
    assert loaded.describe()["layout"] == brain.describe()["layout"]
    for moment, reward in ((4, 0.5), (5, 1.0), (6, 0.0)):
        assert np.array_equal(
            live(brain, cell, moment, reward), live(loaded, cell, moment, reward)
        ), moment
    report(brain, 7)
    report(loaded, 7)
    x = observation(cell, 7)
    assert np.array_equal(brain.act(x, greedy=True), loaded.act(x, greedy=True))
    if brain.bound is not None:
        assert np.array_equal(brain.say(x), loaded.say(x))
    differ = same_arrays(
        arrays(brain.save(tmp_path / "a.npz")), arrays(loaded.save(tmp_path / "b.npz"))
    )
    assert not differ, f"the continued lives differ in {differ[:6]}"


def _distinct(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if row not in out:
            out.append(row)
    return out


# grow refuses a brain with a held message, long-term records or bound records by name: the
# array's cells without them, so grow meets every other option it holds
GROW_CELLS = _distinct([{**c, "message": "off", "longterm": "off", "bound": "off"} for c in CELLS])


@pytest.mark.parametrize("cell", cells(CELLS))
def test_grow_takes_a_combination_or_refuses_it_by_name(cell, tmp_path):
    brain = compose(cell)
    brain.act(observation(cell, 0), greedy=True)
    slots = (3, 2) if cell["slots"] == "two" else (3,)
    part = {"recall": True} if cell["recall"] == "off" else {"slots": (*slots, 2)}
    try:
        grown = brain.grow(**part)
    except ValueError as error:
        refusal(error)
        return
    report(grown, 1)
    grown.act(observation(cell, 1), greedy=True)
    again = Brain.load(grown.save(tmp_path / "grown.npz"))
    report(grown, 2)
    report(again, 2)
    x = observation(cell, 2)
    assert np.array_equal(grown.act(x, greedy=True), again.act(x, greedy=True))


@pytest.mark.parametrize("cell", cells(GROW_CELLS))
def test_grow_without_the_parts_it_refuses(cell, tmp_path):
    test_grow_takes_a_combination_or_refuses_it_by_name(cell, tmp_path)


# Lives refuse a list of parts by name ("What does not go together"); the array's cells without
# them must build, and each life must live as the lone brain of its seed
LIVES_CELLS = _distinct(
    [
        {
            **c,
            "layout": "inputs",
            "trained": "off",
            "code": "none",
            "message": "off",
            "longterm": "off",
            "bound": "off",
            "memory_choice": "off",
            "efference": "off",
            "depth": "one",
            "surprise": "off",
            "recall": "off" if c["recall"] == "off" else "default",
        }
        for c in CELLS
    ]
)


@pytest.mark.parametrize("cell", cells(LIVES_CELLS))
def test_lives_live_as_their_lone_brains(cell):
    kw = options(cell)
    inputs, actions = kw.pop("inputs"), kw.pop("actions")
    kw.pop("seed")
    kw.pop("backend")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lives = cd.Lives(inputs, actions, 2, seeds=[3, 4], backend="cpu", **kw)
        lone = [Brain.compose(inputs, actions, seed=seed, backend="cpu", **kw) for seed in (3, 4)]
    rows = np.vstack([observation(cell, 0), observation(cell, 1)])
    for reward in (None, np.ones(2), np.zeros(2)):
        together = np.asarray(lives.step(rows, reward=reward))
        alone = [
            np.asarray(b.step(rows[[i]], reward=None if reward is None else reward[[i]]))
            for i, b in enumerate(lone)
        ]
        assert np.array_equal(together, np.concatenate(alone)), reward
    assert isinstance(lives.life(1), Brain)


@pytest.mark.parametrize("cell", cells(CELLS))
def test_lives_take_a_combination_or_refuse_it_by_name(cell):
    if cell["layout"] != "inputs":
        with pytest.raises(ValueError, match="senses"):
            cd.Lives(INPUTS, 3, 2, backend="cpu", senses=SENSES)
        return
    kw = options(cell)
    inputs, actions = kw.pop("inputs"), kw.pop("actions")
    kw.pop("seed")
    kw.pop("backend")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lives = cd.Lives(inputs, actions, 2, backend="cpu", **kw)
    except ValueError as error:
        refusal(error)
        return
    rows = np.vstack([observation(cell, 0), observation(cell, 1)])
    lives.step(rows)
    lives.step(rows, reward=np.ones(2))
    assert isinstance(lives.life(1), Brain)


@pytest.mark.parametrize("pair,word", REFUSED, ids=[cell_id(p) for p, _ in REFUSED])
def test_a_refused_pair_is_refused_by_name(pair, word):
    cell = {name: levels[0] for name, levels in FACTORS.items()}
    cell.update(pair)
    if cell["longterm"] != "off":
        cell["message"] = "on"
    with pytest.raises(ValueError) as error:
        compose(cell)
    assert word in refusal(error.value)


DEVICE_CELLS = [
    {**{n: v[0] for n, v in FACTORS.items()}},
    {**{n: v[0] for n, v in FACTORS.items()}, "layout": "senses", "code": "fixed"},
    {**{n: v[0] for n, v in FACTORS.items()}, "recall": "heads2", "arousal": "on"},
    {**{n: v[0] for n, v in FACTORS.items()}, "recall": "strength", "memory_choice": "on"},
    {**{n: v[0] for n, v in FACTORS.items()}, "surprise": "on"},
]


def _devices() -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    if "mlx" in cd.available_backends():
        out.append(("mlx", None))
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        out.append(("torch", "cpu"))
    return out


@pytest.mark.parametrize(
    "backend,device",
    _devices() or [pytest.param(None, None, marks=pytest.mark.skip(reason="no mlx and no torch"))],
)
@pytest.mark.parametrize("cell", DEVICE_CELLS, ids=[cell_id(c) for c in DEVICE_CELLS])
def test_a_combination_on_a_device_lives_saves_and_loads(cell, backend, device, tmp_path):
    kw = options(cell)
    kw["backend"] = backend
    if device is not None:
        kw["device"] = device
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            brain = Brain.compose(**kw)
            brain.lesson(observation(cell, 0), labels(cell))
            brain.act(observation(cell, 1), greedy=True)
            live(brain, cell, 2, None)
            live(brain, cell, 3, 1.0)
    except ValueError as error:
        refusal(error)
        return
    loaded = Brain.load(brain.save(tmp_path / "brain.npz"), backend=backend, device=device)
    for moment, reward in ((4, 0.5), (5, 1.0)):
        live(brain, cell, moment, reward)
        live(loaded, cell, moment, reward)
    assert loaded.act(observation(cell, 6), greedy=True).shape == (1,) + (
        (2,) if cell["slots"] == "two" else ()
    )


def _turns(moment: int) -> np.ndarray:
    row = np.random.default_rng((7, moment)).random((1, INPUTS))
    row[0, TURN] = 1.0 if moment % 4 == 2 else 0.0
    return row


@F1_MARK
def test_known_a_held_message_with_a_holding_turn_continues_as_it_lived(tmp_path):
    brain = Brain.compose(
        INPUTS,
        3,
        modules=(16,),
        seed=3,
        backend="cpu",
        message=12,
        message_winners=2,
        message_boundary=TURN,
        message_boundary_holds=True,
    )
    brain.lesson(_turns(0), np.array([1]))
    brain.act(_turns(1), greedy=True)
    brain.step(_turns(2))
    brain.step(_turns(3), reward=[1.0])
    loaded = Brain.load(brain.save(tmp_path / "brain.npz"))
    brain.step(_turns(4), reward=[0.5])
    loaded.step(_turns(4), reward=[0.5])
    differ = same_arrays(
        arrays(brain.save(tmp_path / "a.npz")), arrays(loaded.save(tmp_path / "b.npz"))
    )
    assert not differ, differ


def test_a_held_message_with_an_emptying_turn_lives_after_a_load(tmp_path):
    """A held message whose turn empties it (``message_boundary`` without
    ``message_boundary_holds``) lives on after a load: its counts are rebuilt."""
    brain = Brain.compose(
        INPUTS, 3, modules=(16,), seed=3, backend="cpu", message=12, message_boundary=TURN
    )
    brain.act(_turns(1), greedy=True)
    loaded = Brain.load(brain.save(tmp_path / "brain.npz"))
    loaded.act(_turns(4), greedy=True)
