"""A slot's strength on the recall memory (cadence#336): what a moment told again, a surprising
moment and a moment the reads keep fetching gain, and what the bank gives up for them.

One small world. A fact is a name stated with a value: a name is ``ACTIVE`` of ``NAME`` binary
inputs, a value one of ``VALUES`` inputs, and one more input is a touch no ordinary moment
drives. A cue is the name alone. The other moments of a life are facts of fresh names, each told
once. A read is right where the recall region's drive, read at the cue before any settle
(``brain.stimulus``), is largest on the fact's value; a slot is held where the bank holds the
fact's key.

Two brains live the same moments: ``plain``, a new brain as ``Brain.compose`` makes it, without
the associative store, and ``strength``, the same brain with ``recall_strength=True``.

Readings: the dose (a fact told n times, then other moments), the one shot (a fact told once at
a struck touch, in brains with ``surprise=True``), the use (a fact told once and then asked for
again and again), and the cost (the recent moments the bank holds, a taught answer at gaps
inside and outside the protected share, and the time of a moment). With ``--paid``, one more:
a life under ``Brain.live`` paid only, whose question comes within the bank's capacity of its
fact or two capacities after it (``paid``).

    PYTHONPATH=src python benchmarks/recall_strength/strength.py --seeds 0 1 2
    PYTHONPATH=src python benchmarks/recall_strength/strength.py --paid --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from typing import Any

import numpy as np

NAME, ACTIVE, VALUES = 32, 6, 8
WIDTH = NAME + VALUES + 1  # the last input is the touch
CAPACITY = 128  # a new brain's
DOSES = (1, 2, 4, 8, 16)
SPANS = (0.5, 2.0, 4.0, 8.0, 16.0, 24.0)  # capacities of other moments
SHORT = (0.5, 2.0, 8.0, 24.0)
STRUCK = 80.0  # a touch over nine times what its neuron tolerates
ARMS = ("plain", "strength")


def brain_of(seed: int, arm: str, capacity: int = CAPACITY, **options: Any) -> Any:
    import cadence as cd

    if arm == "strength":
        options["recall_strength"] = True
    return cd.Brain.compose(WIDTH, VALUES, modules=(32,), seed=seed, backend="cpu",
                            episodic=False, recall_capacity=capacity, **options)


class World:
    """The facts of one life, drawn from one generator: the same for both brains."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def name(self) -> np.ndarray:
        return np.sort(self.rng.choice(NAME, size=ACTIVE, replace=False))

    def fact(self, name: np.ndarray | None = None, value: int | None = None,
             touch: float = 0.0) -> tuple[np.ndarray, np.ndarray, int]:
        """A fact as a moment, with its name and its value."""
        name = self.name() if name is None else name
        value = int(self.rng.integers(VALUES)) if value is None else value
        x = np.zeros((1, WIDTH))
        x[0, name] = 1.0
        x[0, NAME + value] = 1.0
        x[0, -1] = touch
        return x, name, value


def cue(name: np.ndarray) -> np.ndarray:
    x = np.zeros((1, WIDTH))
    x[0, name] = 1.0
    return x


def held(brain: Any, fact: np.ndarray) -> bool:
    """Whether the bank holds the fact's key."""
    recall = brain.recall
    return bool(((recall.keys[0] == fact[0]).all(axis=1) & recall.held[0]).any())


def right(brain: Any, name: np.ndarray, value: int) -> bool:
    """Whether the read at the name alone is largest on the fact's value."""
    drive = brain.stimulus(cue(name))
    read = drive[0, brain.connectome.populations["recall"]][NAME:NAME + VALUES]
    return bool(read.max() > 0 and int(np.argmax(read)) == value)


def told(brain: Any, *moments: np.ndarray) -> None:
    for x in moments:
        brain.act(x, greedy=True)


def others(brain: Any, world: World, moments: int) -> None:
    """Facts of fresh names, each told once."""
    for _ in range(moments):
        told(brain, world.fact()[0])


Fact = tuple[np.ndarray, np.ndarray, int]  # the moment, its name, its value


Reads = tuple[dict[float, np.ndarray], np.ndarray]  # what ``_probe`` gives


def _probe(brain: Any, world: World, targets: list[Fact], capacity: int,
           spans: tuple[float, ...], each: Any = None) -> Reads:
    """Live ``spans`` capacities of other moments after the targets and read every target at
    each span: per span one row a target of whether it is held and whether it is read right,
    and per target the capacities of moments it stayed held (inf where the longest span ends
    with it held). ``each`` is called with every moment's index and tells a moment of its own
    in place of a fresh fact where it returns True."""
    reads: dict[float, np.ndarray] = {}
    left = np.full(len(targets), np.inf)
    moment = 0
    for span in spans:
        while moment < int(round(span * capacity)):
            if each is None or not each(moment):
                told(brain, world.fact()[0])
            moment += 1
            for k, (x, _, _) in enumerate(targets):
                if np.isinf(left[k]) and not held(brain, x):
                    left[k] = moment / capacity
        reads[span] = np.array(
            [[held(brain, x), right(brain, name, value)] for x, name, value in targets],
            dtype=float,
        )
    return reads, left


def dose(seed: int, arm: str, capacity: int = CAPACITY, doses: tuple[int, ...] = DOSES,
         spans: tuple[float, ...] = SPANS, facts: int = 4,
         ) -> dict[int, tuple[dict[float, np.ndarray], np.ndarray]]:
    """A fact told n times in a row, then other moments: per n, each fact's reads at each span
    and the capacities it stayed held (``_probe``)."""
    out = {}
    for n in doses:
        world = World(1000 * seed + n)
        brain = brain_of(seed, arm, capacity)
        others(brain, world, capacity)  # a full bank before the facts
        targets = [world.fact() for _ in range(facts)]
        for x, _, _ in targets:
            told(brain, *[x] * n)
            others(brain, world, 2)
        out[n] = _probe(brain, world, targets, capacity, spans)
    return out


def one_shot(seed: int, arm: str, capacity: int = CAPACITY,
             spans: tuple[float, ...] = SHORT, facts: int = 3) -> dict[str, Any]:
    """In brains with ``surprise=True``: one fact told once at a struck touch (one, since a
    brain is used to a touch it has felt) beside ``facts`` facts told once at none. ``struck``
    and ``plain`` hold each kind's reads at each span (``_probe``), ``tellings`` the tellings
    the brain's surprise counted at each kind's moments, and ``others kept`` how many of the
    other moments were written past the kept depth."""
    world = World(2000 * seed + 1)
    brain = brain_of(seed, arm, capacity, surprise=True)
    others(brain, world, 3 * capacity)  # the brain's usual sizes settle
    kinds: dict[str, list[Fact]] = {"struck": [], "plain": []}
    tellings: dict[str, list[float]] = {"struck": [], "plain": []}
    for kind, touch in (("struck", STRUCK), *[("plain", 0.0)] * facts):
        fact = world.fact(touch=touch)
        told(brain, fact[0])
        kinds[kind].append(fact)
        tellings[kind].append(float(np.expm1(brain.surprise.moment[0])))
        others(brain, world, 2)
    facts = 1  # the struck fact comes first below
    kept = [0]

    def count(moment: int) -> bool:
        told(brain, world.fact()[0])
        kept[0] += int(np.expm1(brain.surprise.moment[0]) >= 7.0)
        return True

    reads, left = _probe(brain, world, kinds["struck"] + kinds["plain"], capacity, spans, count)
    return {
        "struck": ({span: rows[:facts] for span, rows in reads.items()}, left[:facts]),
        "plain": ({span: rows[facts:] for span, rows in reads.items()}, left[facts:]),
        "tellings": {kind: float(np.mean(counted)) for kind, counted in tellings.items()},
        "others kept": kept[0],
    }


ASKED = ("by its name", "by a part", "never")


def use(seed: int, arm: str, capacity: int = CAPACITY, every: int = 8,
        spans: tuple[float, ...] = SHORT) -> tuple[dict[float, np.ndarray], np.ndarray]:
    """Three facts told once, in the order of ``ASKED``. One is asked for by its name every
    ``every`` moments (the same cue each time, a moment of the stream), one by a part of its
    name (four of its six inputs, another part each time), one never. Their reads at each span
    and the capacities each stayed held (``_probe``)."""
    world = World(3000 * seed + 2)
    brain = brain_of(seed, arm, capacity)
    others(brain, world, capacity)
    targets = [world.fact() for _ in ASKED]
    for x, _, _ in targets:
        told(brain, x)
        others(brain, world, 2)
    parts = [np.array(part) for part in itertools.combinations(range(ACTIVE), ACTIVE - 2)]
    asked = [0]

    def ask(moment: int) -> bool:
        if moment % every == 0:  # by its name
            told(brain, cue(targets[0][1]))
            return True
        if moment % every == every // 2:  # by a part of its name
            told(brain, cue(targets[1][1][parts[asked[0] % len(parts)]]))
            asked[0] += 1
            return True
        return False

    return _probe(brain, world, targets, capacity, spans, ask)


def recent(seed: int, arm: str, capacity: int = CAPACITY, life: float = 16.0,
           familiar: float = 0.0, known: int = 12) -> dict[str, float]:
    """A life of ``life`` capacities of facts told once, a share ``familiar`` of whose moments
    tell one of ``known`` facts again. At its end: the share of the last ``capacity`` moments
    the bank holds, the age of the oldest slot in capacities and the slots past the kept
    depth."""
    world = World(4000 * seed + 4)
    brain = brain_of(seed, arm, capacity)
    again = [world.fact()[0] for _ in range(known)]
    lived: list[bytes] = []
    for _ in range(int(life * capacity)):
        if familiar and world.rng.random() < familiar:
            x = again[int(world.rng.integers(known))]
        else:
            x = world.fact()[0]
        told(brain, x)
        lived.append(x.tobytes())
    recall = brain.recall
    keys = {key.tobytes() for key, own in zip(recall.keys[0], recall.held[0], strict=True) if own}
    last = set(lived[-capacity:])
    latest = {key: t for t, key in enumerate(lived)}  # the latest telling of each moment
    oldest = max(len(lived) - 1 - latest[key] for key in keys if key in latest)
    depth = getattr(recall, "depth", None)
    return {
        "recent held": len(last & keys) / len(last),
        "oldest slot": oldest / capacity,
        "kept slots": 0.0 if depth is None else float((depth[0] >= recall.kept).sum()),
    }


GAPS = (0.0625, 0.375, 0.625, 0.875)  # shares of the capacity: two inside the protected share


def taught(seed: int, arm: str, capacity: int = CAPACITY, lessons: int = 2400,
           gaps: tuple[float, ...] = GAPS, trials: int = 24,
           familiar: float = 0.25, known: int = 12) -> dict[float, float]:
    """A task the plain bank does: a fact, then other moments, then its name alone, and the
    answer is its value. The brain is taught at short gaps (a lesson at the cue), in a life a
    share ``familiar`` of whose other moments tell one of ``known`` facts again, then asked
    with its answers free at each of ``gaps`` (shares of the capacity). Per gap the share
    right."""
    world = World(5000 * seed + 5)
    brain = brain_of(seed, arm, capacity)
    again = [world.fact()[0] for _ in range(known)]

    def other(moments: int) -> None:
        for _ in range(moments):
            if world.rng.random() < familiar:
                told(brain, again[int(world.rng.integers(known))])
            else:
                told(brain, world.fact()[0])

    other(capacity)
    for _ in range(lessons):
        x, name, value = world.fact()
        told(brain, x)
        other(int(world.rng.integers(0, 6)))
        brain.lesson(cue(name), [value])
        told(brain, cue(name))
    out = {}
    for gap in gaps:
        hits = 0
        for _ in range(trials):
            x, name, value = world.fact()
            told(brain, x)
            other(max(int(round(gap * capacity)) - 1, 0))
            hits += int(brain.act(cue(name), greedy=True)[0] == value)
        out[gap] = hits / trials
    return out


# the paid life: the reward recall chamber's recipe (benchmarks/recall), a body that needs 0.2
# reward a moment over a youth of 300 moments and the delayed-credit actor point
PAID = dict(actor_lam=0.95, actor_gamma=0.95, actor_eta_critic=5.0, actor_eta=0.1,
            actor_eta_bias=0.01)
PAID_NAME, PAID_ACTIVE, PAID_VALUES, PAID_CAPACITY = 24, 5, 2, 16
PAID_GAPS = (4, 32)  # other moments between the fact and its question: within the bank, past it


def paid(seed: int, arm: str, gap: int, episodes: int = 3000, tellings: int = 4,
         capacity: int = PAID_CAPACITY) -> np.ndarray:
    """One continuing life under ``Brain.live``, paid only. An episode: a fact (a name with one of
    two values) lived once through ``live`` and sensed ``tellings - 1`` times more through
    ``wait``, ``gap`` other facts sensed through ``wait``, then the name alone through ``live``.
    The answer is paid +1 where it is the fact's value and -1 where not, delivered with the next
    episode's first moment and ``done``. A brain of the composed founder with ``arousal``, the
    associative store and a bank of ``capacity`` slots. Per episode, whether it answered right."""
    import cadence as cd

    options: dict[str, Any] = {"recall_strength": True} if arm == "strength" else {}
    width = PAID_NAME + PAID_VALUES
    brain = cd.Brain.compose(width, PAID_VALUES, seed=seed, backend="cpu",
                             arousal=cd.ArousalConfig(need=0.2, youth=300),
                             recall_capacity=capacity, **PAID, **options)
    rng = np.random.default_rng(6000 + seed)

    def fact() -> tuple[np.ndarray, np.ndarray, int]:
        x = np.zeros((1, width))
        name = rng.choice(PAID_NAME, size=PAID_ACTIVE, replace=False)
        value = int(rng.integers(PAID_VALUES))
        x[0, name] = 1.0
        x[0, PAID_NAME + value] = 1.0
        return x, name, value

    pay: float | None = None
    right = np.zeros(episodes)
    for episode in range(episodes):
        x, name, value = fact()
        if pay is None:
            brain.live(x)
        else:
            brain.live(x, reward=np.array([pay]), done=np.array([True]))
        for _ in range(tellings - 1):
            brain.wait(x)
        for _ in range(gap):
            brain.wait(fact()[0])
        question = np.zeros((1, width))
        question[0, name] = 1.0
        answer = int(brain.live(question)[0])
        right[episode] = float(answer == value)
        pay = 1.0 if answer == value else -1.0
    return right


def moment_time(arm: str, capacity: int = CAPACITY, moments: int = 2000) -> float:
    """Milliseconds a moment (a greedy act) once the bank is full."""
    world = World(7)
    brain = brain_of(0, arm, capacity)
    others(brain, world, capacity + 64)
    facts = [world.fact()[0] for _ in range(moments)]
    start = time.perf_counter()
    told(brain, *facts)
    return (time.perf_counter() - start) / moments * 1e3


def _row(label: str, cells: list[str]) -> str:
    return f"| {label} | " + " | ".join(cells) + " |"


def _span(values: Any) -> str:
    low, high = float(np.min(values)), float(np.max(values))
    return f"{low:.2f}" if high - low < 0.005 else f"{low:.2f} to {high:.2f}"


def _reads(runs: list[tuple[dict[float, np.ndarray], np.ndarray]], spans: tuple[float, ...],
           rows: Any = slice(None)) -> list[str]:
    """The cells of one table row: held / read right at each span, the mean over the runs'
    targets ``rows``, and the capacities they stayed held."""
    cells = []
    for span in spans:
        found = np.concatenate([run[0][span][rows].reshape(-1, 2) for run in runs]).mean(axis=0)
        cells.append(f"{found[0]:.2f} / {found[1]:.2f}")
    left = np.concatenate([np.atleast_1d(run[1][rows]) for run in runs])
    if np.isinf(left).all():
        cells.append("past the longest span")
    elif np.isinf(left).any():
        cells.append(f"{np.min(left):.1f} to past the longest span")
    else:
        cells.append(f"{np.median(left):.1f} ({np.min(left):.1f} to {np.max(left):.1f})")
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--paid", action="store_true",
                        help="the paid life under live alone (minutes a seed)")
    parser.add_argument("--episodes", type=int, default=3000, help="of the paid life")
    args = parser.parse_args(argv)
    seeds, capacity = args.seeds, args.capacity
    if args.paid:
        episodes = args.episodes
        last = episodes - episodes // 3  # the last third of the life
        print(f"## The paid life, a bank of {PAID_CAPACITY} slots, seeds {seeds}\n")
        print(_row("arm, gap", ["episodes 1 to 200", f"{last + 1} to {episodes}",
                                f"per seed, {last + 1} to {episodes}"]))
        for gap in PAID_GAPS:
            for arm in ARMS:
                lives = [paid(seed, arm, gap, episodes) for seed in seeds]
                print(_row(f"{arm}, {gap}", [
                    f"{np.mean([life[:200].mean() for life in lives]):.2f}",
                    f"{np.mean([life[last:].mean() for life in lives]):.2f}",
                    ", ".join(f"{life[last:].mean():.2f}" for life in lives),
                ]))
                sys.stdout.flush()
        return 0
    print(f"capacity {capacity}, seeds {seeds}; cells: held / read right\n")

    print("## Dose, by capacities of other moments\n")
    print(_row("arm, tellings", [f"{s:g}" for s in SPANS] + ["held, capacities"]))
    for arm in ARMS:
        # the plain bank holds no fact past its capacity: its long spans are not lived
        spans = SPANS[:2] if arm == "plain" else SPANS
        doses = [dose(seed, arm, capacity, spans=spans) for seed in seeds]
        for n in DOSES:
            cells = _reads([run[n] for run in doses], spans)
            cells[-1:-1] = [""] * (len(SPANS) - len(spans))
            print(_row(f"{arm}, n = {n}", cells))
        sys.stdout.flush()

    print("\n## One shot (surprise=True)\n")
    print(_row("arm, fact", [f"{s:g}" for s in SHORT] + ["held, capacities", "tellings"]))
    for arm in ARMS:
        shots = [one_shot(seed, arm, capacity) for seed in seeds]
        for kind in ("struck", "plain"):
            cells = _reads([run[kind] for run in shots], SHORT)
            cells.append(_span([run["tellings"][kind] for run in shots]))
            print(_row(f"{arm}, {kind}", cells))
        print(f"{arm}: other moments written past the kept depth, a life: "
              f"{[run['others kept'] for run in shots]}")
        sys.stdout.flush()

    print("\n## Use\n")
    print(_row("arm, asked", [f"{s:g}" for s in SHORT] + ["held, capacities"]))
    for arm in ARMS:
        uses = [use(seed, arm, capacity) for seed in seeds]
        for k, key in enumerate(ASKED):
            print(_row(f"{arm}, {key}", _reads(uses, SHORT, slice(k, k + 1))))
        sys.stdout.flush()

    print("\n## What it costs\n")
    names = ("recent held", "oldest slot", "kept slots")
    print(_row("arm, life", ["recent moments held", "oldest slot, capacities", "kept slots"]))
    for familiar in (0.0, 0.25):
        for arm in ARMS:
            lives = [recent(seed, arm, capacity, familiar=familiar) for seed in seeds]
            print(_row(f"{arm}, {familiar:.0%} told again",
                       [_span([run[name] for run in lives]) for name in names]))
    sys.stdout.flush()
    print()
    print(_row("arm, life", [f"gap {g:g} of the capacity" for g in GAPS]))
    for familiar in (0.0, 0.25):
        for arm in ARMS:
            answers = [taught(seed, arm, capacity, familiar=familiar) for seed in seeds]
            print(_row(f"{arm}, {familiar:.0%} told again",
                       [f"{np.mean([run[g] for run in answers]):.2f}" for g in GAPS]))
            sys.stdout.flush()
    print()
    for arm in ARMS:
        print(f"{arm}: {moment_time(arm, capacity):.2f} ms a moment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
