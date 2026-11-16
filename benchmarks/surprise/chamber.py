"""One surprise signal, two measures of it read side by side (cadence#337).

A brain composed with ``surprise=True`` reads each neuron's surprise from the settle's own
residual: the size of the potential equation's error at the state the stream carried into the
moment, under the moment's drive, before the settle repairs it. This chamber reads a second
measure beside it on the same brains and the same streams, at every moment:

* ``residual``: the library's measure (``brain.surprise``).
* ``move``: how far the moment moved each neuron's activity, the state after the moment less
  the state carried into it. It lives here and in no library module.

Both sizes pass through one law with the same genes (``cadence.surprise.Surprise``): a size
against the size the neuron is used to. A reading is the moment's largest surprise over the
neurons (``Surprise.moment``), over the whole brain, over the sensory neurons alone (what the
focus reads), and over the association and motor regions (``inner``: under greedy acts no
drive enters them).

Sections:

(a) ``deviants``: a repeating stream with rare deviants. Does the deviant read above the
    standards (the area under the curve, and the means)?
(b) ``onset``: an input held, then a change, then the change repeated, then a new change.
(c) ``learned``: a brain that lived a regular sequence, against copies of the same brain shown
    a broken one.
(d) ``cost``: milliseconds a moment with the signal off and on, and the residual's own
    evaluation.
(e) ``bits``: a brain without the option against a brain with it, to the bit, and with
    ``--main`` against another source tree of the library.

    PYTHONPATH=src python benchmarks/surprise/chamber.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

INPUTS = 24  # the first FAMILIAR carry the streams; the others are spare for a new thing
FAMILIAR = 16
ACTIVE = 4  # the inputs a pattern holds at one
ACTIONS = 6
CYCLE = 6  # the elements of a regular sequence


def brain(seed: int, **options: Any) -> Any:
    from cadence import Brain

    options.setdefault("surprise", True)
    return Brain.compose(INPUTS, ACTIONS, seed=seed, backend="cpu", **options)


STREAM = np.arange(FAMILIAR)  # the inputs a stream's patterns are drawn from
SPARE = np.arange(FAMILIAR, INPUTS)  # the inputs no stream uses


def pattern(rng: np.random.Generator, pool: np.ndarray) -> np.ndarray:
    """One pattern: ``ACTIVE`` of the inputs ``pool`` at one, ``(1, INPUTS)``."""
    x = np.zeros((1, INPUTS))
    x[0, rng.choice(pool, size=ACTIVE, replace=False)] = 1.0
    return x


def distinct(rng: np.random.Generator, count: int, pool: np.ndarray,
             taken: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """``count`` patterns over the inputs ``pool``, no two alike and none among ``taken``."""
    out: list[np.ndarray] = []
    seen = {x.tobytes() for x in taken or []}
    while len(out) < count:
        x = pattern(rng, pool)
        if x.tobytes() not in seen:
            seen.add(x.tobytes())
            out.append(x)
    return out


def used(patterns: list[np.ndarray]) -> np.ndarray:
    """The inputs that some pattern of ``patterns`` holds at one."""
    return np.flatnonzero(np.concatenate(patterns).any(axis=0))


class Reader:
    """Both measures of one brain's moments: the library's residual and the state's move."""

    def __init__(self, held: Any) -> None:
        from cadence.surprise import Surprise

        self.brain = held
        self.moved = Surprise(held.connectome.n, **held.surprise.settings())
        self.senses = np.asarray(held.sensory_index)
        # the association and the motor neurons: no sense and no trace drives them
        self.inner = np.concatenate([held.association_index, held.motor_index])

    def moment(self, x: np.ndarray, how: str = "act", **given: Any) -> dict[str, float]:
        """One moment of the stream under ``how`` (``act`` greedy, ``live`` or ``step``); gives
        each measure's surprise over the whole brain and over the senses, and its units."""
        agent = self.brain.basal_ganglia
        before = agent.state
        then = None if before is None else np.array(before.activation, dtype=np.float64)
        if how == "act":
            action = self.brain.act(x, greedy=True)
        elif how == "live":
            action = self.brain.live(x, **given)
        else:
            action = self.brain.step(x, **given)
        after = np.array(agent.state.activation, dtype=np.float64)
        if then is None or then.shape != after.shape:  # from rest: nothing was carried in
            self.moved.observe(np.zeros_like(after), carried=np.zeros(len(after), dtype=bool))
        else:
            self.moved.observe(after - then)
        residual, move = self.brain.surprise.last[0], self.moved.last[0]
        return {
            "residual": float(residual.max()), "move": float(move.max()),
            "residual_senses": float(residual[self.senses].max()),
            "move_senses": float(move[self.senses].max()),
            "residual_inner": float(residual[self.inner].max()),
            "move_inner": float(move[self.inner].max()),
            "residual_units": float((residual > 0).sum()),
            "move_units": float((move > 0).sum()),
            "action": float(np.asarray(action).reshape(-1)[0]),
        }


def area(high: list[float], low: list[float]) -> float:
    """The area under the curve: how often a reading of ``high`` lies above one of ``low``,
    a tie counting half; 0.5 where the two cannot be told apart."""
    a, b = np.asarray(high, dtype=np.float64), np.asarray(low, dtype=np.float64)
    if not len(a) or not len(b):
        return float("nan")
    above = (a[:, None] > b[None, :]).mean()
    level = (a[:, None] == b[None, :]).mean()
    return float(above + 0.5 * level)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


MEASURES = ("residual", "move", "residual_senses", "move_senses")


# -- (a) a repeating stream with rare deviants


def deviants(seed: int, stream: str = "repeat", deviant: str = "units", how: str = "act",
             moments: int = 300, every: int = 10, warm: int = 40) -> dict[str, Any]:
    """A repeating stream (``repeat``: one pattern again and again; ``cycle``: ``CYCLE``
    patterns in order) with a deviant in place of the standard at one moment in ``every`` on
    average, never twice in a row. The deviant is one pattern that comes back: on spare inputs
    (``units``), or a new combination of the inputs the stream's own patterns use
    (``combination``, a cycle's); or a new such combination each time (``fresh``). ``how`` is
    ``act`` (greedy acts: the weights stay) or
    ``live`` (reward only: the action paid is 1 at a deviant and 0 at a standard). Gives each
    measure's area under the curve of deviant against standard moments and its means at the
    deviants, at the moments after one and at the standards."""
    rng = np.random.default_rng([seed, 337, 1])
    standards = distinct(rng, 1 if stream == "repeat" else CYCLE, STREAM)
    if deviant != "units" and stream == "repeat":
        raise ValueError("one pattern's inputs hold no second combination: a cycle's do")
    own = used(standards)
    odd = pattern(rng, SPARE) if deviant == "units" else distinct(rng, 1, own, standards)[0]
    reader = Reader(brain(seed, **({"arousal": True} if how == "live" else {})))
    reads: dict[str, dict[str, list[float]]] = {
        kind: {name: [] for name in MEASURES} for kind in ("deviant", "after", "standard")
    }
    last, reward = False, None
    # the largest reading over the association and motor regions at any moment of the stream
    inner = {"residual": 0.0, "move": 0.0}
    for moment in range(warm + moments):
        is_odd = moment >= warm and not last and rng.random() < 1.0 / every
        x = standards[moment % len(standards)]
        if is_odd:
            x = distinct(rng, 1, own, standards)[0] if deviant == "fresh" else odd
        given = {"reward": reward} if how == "live" and reward is not None else {}
        read = reader.moment(x, how, **given)
        for measure in inner:
            inner[measure] = max(inner[measure], read[measure + "_inner"])
        reward = np.array([float(int(read["action"]) == int(is_odd))])
        if moment >= warm:
            kind = "deviant" if is_odd else "after" if last else "standard"
            for name in MEASURES:
                reads[kind][name].append(read[name])
        last = is_odd
    out: dict[str, Any] = {"deviants": len(reads["deviant"]["residual"]), "inner": inner}
    for name in MEASURES:
        out[name] = {
            "area": area(reads["deviant"][name], reads["standard"][name]),
            "deviant": mean(reads["deviant"][name]),
            "after": mean(reads["after"][name]),
            "standard": mean(reads["standard"][name]),
        }
    return out


# -- (b) onset and habituation


def onset(seed: int, held_moments: int = 20, repeats: int = 8, gap: int = 4) -> dict[str, Any]:
    """An input held for ``held_moments``, then a change for one moment and the held input
    again for ``gap`` moments, ``repeats`` times over, then a new change on other inputs.
    Gives each measure's reading at the last held moment, at each coming of the change, at the
    return after the first and at the new change."""
    rng = np.random.default_rng([seed, 337, 2])
    held, change = distinct(rng, 2, STREAM)
    new = pattern(rng, SPARE)
    reader = Reader(brain(seed))
    out: dict[str, Any] = {}
    for _ in range(held_moments):
        read = reader.moment(held)
    out["held"] = read
    out["changes"] = []
    for repeat in range(repeats):
        out["changes"].append(reader.moment(change))
        for moment in range(gap):
            read = reader.moment(held)
            if repeat == 0 and moment == 0:
                out["return"] = read
    out["new"] = reader.moment(new)
    return out


# -- (c) learned, not built


def learned(seed: int, cycles: int = 40) -> dict[str, Any]:
    """A regular sequence of ``CYCLE`` patterns, lived for ``cycles`` cycles by two brains of
    one seed: ``exposed`` (greedy acts: the weights stay, each neuron's usual size learns) and
    ``taught`` (``step`` with the next element as the teacher: the weights learn the order).
    Then copies of each brain are shown one more cycle: the regular one, or a broken one
    (``swap``: two neighbours change places; ``repeat``: an element comes twice and the next is
    left out; ``units``: an element is a pattern on spare inputs; ``combination``: an element
    is a new combination of the inputs the sequence's patterns use). Gives each measure's mean
    reading and
    the share of moments it reads a surprise at: over the first cycle a fresh brain lives, over
    the regular cycle, and at the moments of each break (a break's moments are those whose
    step from the moment before is none of the sequence's steps)."""
    rng = np.random.default_rng([seed, 337, 3])
    sequence = distinct(rng, CYCLE, STREAM)
    spare = pattern(rng, SPARE)
    combined = distinct(rng, 1, used(sequence), sequence)[0]
    order = list(range(CYCLE))
    broken: dict[str, list[Any]] = {
        "regular": order,
        "swap": [0, 1, 3, 2, 4, 5],
        "repeat": [0, 1, 2, 2, 4, 5],
        "units": [0, 1, 2, "spare", 4, 5],
        "combination": [0, 1, 2, "combined", 4, 5],
    }

    def shown(index: Any) -> np.ndarray:
        return spare if index == "spare" else combined if index == "combined" else sequence[index]

    def summary(reads: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        return {name: {"mean": mean([r[name] for r in reads]),
                       "share": mean([float(r[name] > 0) for r in reads])}
                for name in MEASURES}

    out: dict[str, Any] = {}
    for arm in ("exposed", "taught"):
        reader = Reader(brain(seed))
        how = "act" if arm == "exposed" else "step"
        first: list[dict[str, float]] = []
        for moment in range(cycles * CYCLE):
            at = moment % CYCLE
            given = {} if how == "act" else {"teacher": np.array([(at + 1) % CYCLE])}
            read = reader.moment(sequence[at], how, **given)
            if 0 < moment <= CYCLE:  # every step of the sequence for the first time
                first.append(read)
        rows: dict[str, Any] = {"first": summary(first)}
        steps = {(a, (a + 1) % CYCLE) for a in range(CYCLE)}
        for name, cycle in broken.items():
            twin = copy.deepcopy(reader)
            reads, before = [], CYCLE - 1
            right: list[float] = []
            for index in cycle:
                read = twin.moment(shown(index), "act")
                if name == "regular" or (before, index) not in steps:
                    reads.append(read)
                if name == "regular":  # its greedy answer: the element that comes next
                    right.append(float(int(read["action"]) == (index + 1) % CYCLE))
                before = index
            rows[name] = summary(reads)
            if name == "regular":
                rows["next_right"] = mean(right)
        out[arm] = rows
    return out


# -- (d) cost


def cost(seed: int = 0, moments: int = 200, names: tuple[str, ...] | None = None
         ) -> dict[str, Any]:
    """Milliseconds a greedy act with the signal off and on, on the composed brain and on a
    wider one, each with and without the compiled moment (every brain, or those ``names``
    lists); and the microseconds of the parts: the residual's one extra evaluation
    (``NeuralGraph.residual(per_unit=True)``), the law (``Surprise.observe``) and the state's
    move (a subtraction and the law). Medians over ``moments`` moments of a cycle of
    patterns."""
    from cadence.surprise import Surprise

    rng = np.random.default_rng([seed, 337, 4])
    stream = distinct(rng, CYCLE, STREAM)
    out: dict[str, Any] = {}
    for name, options in (
        ("composed", {}), ("wide", {"modules": (256,)}),
        ("composed, compiled moment", {"learning_compiled_moment": True}),
        ("wide, compiled moment", {"modules": (256,), "learning_compiled_moment": True}),
    ):
        if names is not None and name not in names:
            continue
        row: dict[str, Any] = {}
        for arm, surprise in (("off", False), ("on", True)):
            held = brain(seed, surprise=surprise, **options)
            for moment in range(20):  # the kernels compile and the streams begin
                held.act(stream[moment % CYCLE], greedy=True)
            times = []
            for moment in range(moments):
                x = stream[moment % CYCLE]
                start = time.perf_counter()
                held.act(x, greedy=True)
                times.append(time.perf_counter() - start)
            row[arm] = 1e3 * float(np.median(times))
        row["neurons"], row["synapses"] = int(held.connectome.n), int(held.connectome.synapses)
        graph, agent, law = held.brain, held.basal_ganglia, held.surprise
        carried, drive = agent.state, np.array(agent._drive)
        parts: dict[str, list[float]] = {"residual": [], "law": [], "move": []}
        moved = Surprise(held.connectome.n)
        other = np.array(carried.activation) * 0.5
        for _ in range(moments):
            start = time.perf_counter()
            sizes = np.abs(graph.residual(drive, carried, per_unit=True))
            parts["residual"].append(time.perf_counter() - start)
            start = time.perf_counter()
            law.observe(sizes, learn=False)
            parts["law"].append(time.perf_counter() - start)
            start = time.perf_counter()
            moved.observe(np.asarray(carried.activation) - other, learn=False)
            parts["move"].append(time.perf_counter() - start)
        row.update({f"{part}_us": 1e6 * float(np.median(v)) for part, v in parts.items()})
        out[name] = row
    return out


# -- (e) off equals main, to the bit


def life(surprise: bool | None, seed: int = 2, moments: int = 30) -> tuple[Any, list[int]]:
    """A short life of one brain: ``live`` with reward only, then ``step`` with a teacher at
    every third moment on two fresh streams. ``surprise`` None leaves the option unnamed."""
    from cadence import Brain

    options: dict[str, Any] = {} if surprise is None else {"surprise": surprise}
    held = Brain.compose(INPUTS, ACTIONS, seed=seed, backend="cpu", arousal=True, **options)
    rng = np.random.default_rng([seed, 337, 5])
    actions: list[int] = []
    reward = None
    for _ in range(moments):
        x = (rng.random((1, INPUTS)) < 0.25).astype(float)
        action = held.live(x, reward=reward)
        reward = np.array([float(action[0] == int(x[0].sum()) % ACTIONS)])
        actions.append(int(action[0]))
    held.reset()
    reward = None
    for moment in range(moments // 2):
        x = (rng.random((2, INPUTS)) < 0.25).astype(float)
        wanted = x.sum(axis=1).astype(int) % ACTIONS
        action = held.step(x, reward=reward, teacher=wanted if moment % 3 == 2 else None)
        reward = (action == wanted).astype(float)
        actions.extend(int(a) for a in action)
    return held, actions


def digest(surprise: bool | None, seed: int = 2) -> str:
    """The digest of a short life: its actions, its weights and every array of its saved file,
    without the surprise's own arrays and record and the format that names it (a brain with
    surprise saves ``cadence-brain/5`` with the feature ``surprise``)."""
    held, actions = life(surprise, seed)
    found = hashlib.sha256()
    found.update(np.asarray(actions, dtype=np.int64).tobytes())
    found.update(np.ascontiguousarray(held.brain.efficacy).tobytes())
    found.update(np.ascontiguousarray(held.brain.bias).tobytes())
    with tempfile.TemporaryDirectory(prefix="cadence-surprise-") as folder:
        path = held.save(Path(folder) / "brain.npz")
        with np.load(path) as saved:
            for name in sorted(saved.files):
                if name.startswith("surprise/"):
                    continue
                value = saved[name]
                if name == "brain":
                    record = json.loads(str(value))
                    record.pop("surprise", None)
                    if record.get("features") == ["surprise"]:  # the format that names it
                        record.pop("features")
                        record["format"] = "cadence-brain/1"
                    value = np.array(json.dumps(record, sort_keys=True))
                found.update(name.encode())
                found.update(str(value.dtype).encode() + str(value.shape).encode())
                found.update(np.ascontiguousarray(value).tobytes())
    return found.hexdigest()


def bits(main: str | None = None) -> dict[str, Any]:
    """The digests of one short life: with the option unnamed, off and on in this source, and
    with ``main`` (the ``src`` folder of another source tree of the library, such as a
    checkout of a release) the unnamed life's digest under that source, in a process of its
    own."""
    out: dict[str, Any] = {"unnamed": digest(None), "off": digest(False), "on": digest(True)}
    out["off equals unnamed"] = out["off"] == out["unnamed"]
    out["on equals unnamed"] = out["on"] == out["unnamed"]
    if main is not None:
        run = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--digest"],
            env={**os.environ, "PYTHONPATH": str(main)}, capture_output=True, text=True,
            check=True,
        )
        version, found = run.stdout.split()
        out["main"], out["main version"] = found, version
        out["unnamed equals main"] = found == out["unnamed"]
    return out


# -- the tables


def spread(values: list[float], digits: int = 2) -> str:
    """The mean of the seeds' readings with their range."""
    held = np.asarray(values, dtype=np.float64)
    if np.isnan(held).all():
        return "n/a"
    text = f"{np.nanmean(held):.{digits}f}"
    if len(held) > 1:
        text += f" ({np.nanmin(held):.{digits}f} to {np.nanmax(held):.{digits}f})"
    return text


def report(seeds: list[int], main: str | None, moments: int) -> dict[str, Any]:
    out: dict[str, Any] = {"seeds": seeds}
    print("(a) a repeating stream with rare deviants: the area under the curve of deviant\n"
          "    against standard moments, and the mean reading at deviants / after one / at\n"
          "    standards with the ratio of the first to the last; whole brain, then the\n"
          "    senses alone\n")
    print("| Stream | Deviant, one in | Lived by | residual: area | residual: means | "
          "move: area | move: means |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    out["deviants"] = {}
    for stream, deviant, how, every in (
        ("repeat", "units", "act", 10), ("repeat", "units", "act", 60),
        ("cycle", "units", "act", 10), ("cycle", "combination", "act", 10),
        ("cycle", "fresh", "act", 10),
        ("repeat", "units", "live", 10), ("cycle", "fresh", "live", 10),
    ):
        # as many deviants at any rate: a rarer deviant's stream is longer
        runs = [deviants(seed, stream, deviant, how, moments * every // 10, every)
                for seed in seeds]
        out["deviants"][f"{stream}/{deviant}/{how}/{every}"] = runs
        deviant = f"{deviant}, {every}"
        for scope in ("", "_senses"):
            cells = []
            for measure in ("residual", "move"):
                name = measure + scope
                cells.append(spread([r[name]["area"] for r in runs]))
                means = [float(np.nanmean([r[name][kind] for r in runs]))
                         for kind in ("deviant", "after", "standard")]
                ratio = f"{means[0] / means[2]:.0f}" if means[2] > 0 else "no standard reads"
                cells.append(" / ".join(f"{value:.2f}" for value in means) + f" ({ratio})")
            where = "senses" if scope else "whole brain"
            print(f"| {stream} | {deviant} | {how}, {where} | " + " | ".join(cells) + " |")
    for how in ("act", "live"):
        runs = [r for key, found in out["deviants"].items() for r in found
                if key.split("/")[2] == how]
        print(f"\nThe largest reading over the association and motor regions at any moment "
              f"of the streams lived by {how}: residual "
              f"{max(r['inner']['residual'] for r in runs):.2f}, move "
              f"{max(r['inner']['move'] for r in runs):.2f}")
    print("\n(b) onset and habituation: the reading at each moment named; whole brain "
          "(senses; association and motor)\n")
    runs = [onset(seed) for seed in seeds]
    out["onset"] = runs
    print("| Moment | residual | move |")
    print("| --- | --- | --- |")

    def both(reads: list[dict[str, float]]) -> str:
        return " | ".join(
            f"{np.mean([r[m] for r in reads]):.2f} "
            f"({np.mean([r[m + '_senses'] for r in reads]):.2f}; "
            f"{np.mean([r[m + '_inner'] for r in reads]):.2f})"
            for m in ("residual", "move"))

    print(f"| the input held, its last moment | {both([r['held'] for r in runs])} |")
    for index in (0, 1, 2, 3, 7):
        print(f"| the change, coming {index + 1} | "
              f"{both([r['changes'][index] for r in runs])} |")
    print(f"| the held input back after the first change | {both([r['return'] for r in runs])} |")
    print(f"| a new change on other inputs | {both([r['new'] for r in runs])} |")
    print("\n(c) learned, not built: the mean reading and the share of moments that read a\n"
          "    surprise, whole brain; then the same at the senses\n")
    runs = [learned(seed) for seed in seeds]
    out["learned"] = runs
    print("| Brain | Moments | residual | move |")
    print("| --- | --- | --- | --- |")
    names = {
        "first": "the sequence's first cycle, a fresh brain",
        "regular": "the regular cycle, after 40",
        "swap": "two neighbours change places",
        "repeat": "an element twice, the next left out",
        "units": "an element on spare inputs",
        "combination": "an element a new combination",
    }
    for arm in ("exposed", "taught"):
        right = np.mean([r[arm]["next_right"] for r in runs])
        for key, text in names.items():
            cells = [
                "; ".join(
                    f"{np.mean([r[arm][key][m + scope]['mean'] for r in runs]):.2f} "
                    f"({np.mean([r[arm][key][m + scope]['share'] for r in runs]):.2f})"
                    for scope in ("", "_senses"))
                for m in ("residual", "move")
            ]
            print(f"| {arm} (next element right {right:.2f}) | {text} | " + " | ".join(cells)
                  + " |")
    print("\n(d) cost: milliseconds a greedy act, and microseconds of the parts (medians)\n")
    found = cost(seeds[0])
    out["cost"] = found
    print("| Brain | neurons | synapses | off, ms | on, ms | residual, us | law, us | "
          "move, us |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, row in found.items():
        print(f"| {name} | {row['neurons']} | {row['synapses']:,} | {row['off']:.3f} | "
              f"{row['on']:.3f} | {row['residual_us']:.0f} | {row['law_us']:.0f} | "
              f"{row['move_us']:.0f} |")
    print("\n(e) to the bit: one short life's actions, weights and saved arrays\n")
    same = bits(main)
    out["bits"] = same
    for name, value in same.items():
        print(f"- {name}: {value}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--moments", type=int, default=300, help="a deviant stream's moments")
    parser.add_argument("--main", default=None,
                        help="the src folder of another source tree, for section (e)")
    parser.add_argument("--json", default=None, help="write every reading to this file")
    parser.add_argument("--digest", action="store_true",
                        help="print the library version and the unnamed life's digest")
    args = parser.parse_args(argv)
    if args.digest:
        import cadence

        print(cadence.__version__, digest(None))
        return 0
    out = report(list(args.seeds), args.main, args.moments)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
