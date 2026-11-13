"""The words chamber's declared runs: every arm's lives on the founders, and the gates read from
their files (cadence#326, PROTOCOL.md and PROTOCOL_2.md).

    python gates.py run <out> <first founder> <founders> [jobs]      # protocol 1
    python gates.py read <out>
    python gates.py run2 <out> <first founder> <founders> [jobs]     # protocol 2 (W11)
    python gates.py read2 <out>
    python gates.py run3 <out> <first founder> <founders> [jobs]     # protocol 3 (the game)
    python gates.py read3 <out>

``run`` writes one JSON file a life: ``brain_<seed>.json`` and
``<model>_<mode>_<size>_<seed>.json`` for the controls (``size`` is ``matched``, at the brain's
parameter count, or ``large``, at nine times it). ``read`` prints the table and each gate's
verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMETERS = 6656  # the brain's learned parameters: one efficacy a reciprocal pair
SIZES = {"matched": PARAMETERS, "large": 9 * PARAMETERS}
CONTROLS = [(m, mode, size) for m in ("lstm", "transformer") for mode in ("online", "offline")
            for size in ("matched",)] + [(m, "offline", "large") for m in ("lstm", "transformer")]

# the declared gates: (reading, threshold); a gate holds where the mean over the founders and at
# least ten of twelve founders reach it
GATES = {
    "W1": [("W1", 0.95), ("W1_named_only", 0.95)],
    "W2": [("W2", 0.90)],
    "W3": [("W3", 0.90)],
    "W4": [("W4_never_seen", 0.95), ("W4_never_taught", 0.95)],
    "W5": [("W5_by_colour", 0.90), ("W5_by_place", 0.90)],
}
READINGS = ["W1", "W1_practised", "W1_named_only", "W2", "W3", "W3_where", "W4_out_of_view",
            "W4_map_rate", "W4_never_seen", "W4_never_taught", "W5_by_colour", "W5_by_place",
            "W6_ablated"]


# protocol 2 (W11): practice by pistachios
PRACTICE_ROOMS = 60  # the rewarded arms' practice rooms, ten questions each
GATES_2 = {
    "W11a": [("reward", "W1", 0.95, "at least"), ("reward", "W5_by_colour", 0.90, "at least"),
             ("reward", "W5_by_place_held", 0.90, "at least")],
    "W11b": [("reward", "W1_named_only", 0.95, "at least"), ("reward", "W11_kept", 1.0,
                                                               "at least")],
    "W11c": [("withheld", "W1", 0.34, "at most"), ("withheld", "W5_by_colour", 0.34, "at most"),
             ("withheld", "W5_by_place_held", 0.34, "at most")],
}
READINGS_2 = ["W1", "W1_named_only", "W2", "W3", "W4_never_seen", "W4_never_taught",
              "W5_by_colour", "W5_by_place_held", "W11_kept", "pistachios", "W6_ablated"]


def jobs2(out: Path, first: int, count: int) -> list[tuple[Path, list[str]]]:
    python = sys.executable
    todo = []
    for seed in range(first, first + count):
        for practice, rooms in (("told", 24), ("reward", PRACTICE_ROOMS),
                                ("withheld", PRACTICE_ROOMS)):
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import words; "
                    f"print(json.dumps(words.life({seed}, practice={practice!r}, "
                    f"practice_rooms={rooms}), default=float))")
            todo.append((out / f"brain_{practice}_{seed}.json", [python, "-c", code]))
        for model in ("lstm", "transformer"):
            for mode in ("online", "offline"):
                code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import controls; "
                        f"print(json.dumps(controls.run({seed}, {model!r}, {mode!r}, "
                        f"{PARAMETERS}, practice='reward', practice_rooms={PRACTICE_ROOMS}), "
                        f"default=float))")
                todo.append((out / f"{model}_{mode}_reward_{seed}.json", [python, "-c", code]))
    return todo


# protocol 3: the training game
GATES_3 = {
    "G1": [("brain_game", r, 0.95, "at least") for r in (
        "understood_kind", "understood_colour", "understood_place", "said_kind",
        "said_colour", "said_place", "W1")] + [
        ("brain_game", "W5_by_colour", 0.90, "at least"),
        ("brain_game", "W5_by_place_held", 0.90, "at least"),
        ("brain_game", "W2", 0.90, "at least")],
    "G2": [("brain_naming", r, 0.05, "at most") for r in (
        "understood_kind", "understood_colour", "understood_place", "said_kind",
        "said_colour", "said_place")] + [("brain_naming", "W1", 0.34, "at most")],
    "G3": [("brain_withheld", "understood_colour", 0.05, "at most"),
           ("brain_withheld", "said_colour", 0.05, "at most"),
           ("brain_withheld", "W1_colour", 0.34, "at most"),
           ("brain_withheld", "understood_kind", 0.95, "at least"),
           ("brain_withheld", "understood_place", 0.95, "at least"),
           ("brain_withheld", "W1_where", 0.95, "at least")],
}
READINGS_3 = ["W1", "W1_where", "W1_colour", "W2", "W3", "W5_by_colour", "W5_by_place_held",
              "understood_kind", "understood_colour", "understood_place", "said_kind",
              "said_colour", "said_place", "turn_thinning", "point_thinning", "turn_right"]


def jobs3(out: Path, first: int, count: int) -> list[tuple[Path, list[str]]]:
    python = sys.executable
    todo = []
    for seed in range(first, first + count):
        for arm in ("game", "naming", "withheld"):
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import game; "
                    f"print(json.dumps(game.life({seed}, {arm!r}), default=float))")
            todo.append((out / f"brain_{arm}_{seed}.json", [python, "-c", code]))
        for name in ("table", "delta"):
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import game, arms; "
                    f"print(json.dumps(game.life({seed}, 'game', lexicon=arms.ARMS[{name!r}]), "
                    f"default=float))")
            todo.append((out / f"{name}_game_{seed}.json", [python, "-c", code]))
        for model in ("lstm", "transformer"):
            for mode in ("online", "offline"):
                code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import controls; "
                        f"print(json.dumps(controls.run_game({seed}, {model!r}, {mode!r}, "
                        f"{PARAMETERS}), default=float))")
                todo.append((out / f"{model}_{mode}_game_{seed}.json", [python, "-c", code]))
    return todo


def jobs(out: Path, first: int, count: int) -> list[tuple[Path, list[str]]]:
    python = sys.executable
    todo = []
    for seed in range(first, first + count):
        todo.append((out / f"brain_{seed}.json", [python, str(HERE / "words.py"), str(seed)]))
        for model, mode, size in CONTROLS:
            code = (f"import json,sys; sys.path.insert(0,{str(HERE)!r}); import controls; "
                    f"print(json.dumps(controls.run({seed}, {model!r}, {mode!r}, "
                    f"{SIZES[size]}), default=float))")
            todo.append((out / f"{model}_{mode}_{size}_{seed}.json", [python, "-c", code]))
    return todo


def run(out: Path, first: int, count: int, workers: int, protocol: int = 1) -> None:
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}

    def one(job: tuple[Path, list[str]]) -> str:
        path, command = job
        if path.exists():
            return f"kept {path.name}"
        done = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
        if done.returncode:
            (path.with_suffix(".err")).write_text(done.stderr)
            return f"failed {path.name}"
        part = path.with_suffix(".part")
        part.write_text(done.stdout)
        part.rename(path)
        return f"wrote {path.name}"

    with ThreadPoolExecutor(workers) as pool:
        for line in pool.map(one, {1: jobs, 2: jobs2, 3: jobs3}[protocol](out, first, count)):
            print(line, flush=True)


def read(out: Path) -> dict[str, Any]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out.glob("*.json")):
        arm = path.stem.rsplit("_", 1)[0]
        arms.setdefault(arm, []).append(json.loads(path.read_text()))
    head = ["arm", "founders", "parameters", "multiply-accumulates", "seconds"] + READINGS
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for arm, lives in arms.items():
        row = [arm, str(len(lives)), str(lives[0].get("parameters", "")),
               f"{np.mean([life.get('macs', 0) for life in lives]):.3g}",
               f"{np.mean([sum(life['seconds'].values()) for life in lives]):.0f}"]
        for name in READINGS:
            values = [life[name] for life in lives if name in life and life[name] == life[name]]
            row.append(f"{np.mean(values):.3f} ({min(values):.2f}-{max(values):.2f})"
                       if values else "")
        print("| " + " | ".join(row) + " |")
    verdicts: dict[str, Any] = {}
    brain = arms.get("brain", [])
    for gate, needs in GATES.items():
        held = True
        parts = []
        for name, threshold in needs:
            values = np.array([life[name] for life in brain])
            reach = int((values >= threshold).sum())
            ok = bool(len(values)) and values.mean() >= threshold and reach >= min(
                10, len(values))
            held &= ok
            parts.append(f"{name} mean {values.mean():.3f}, {reach} of {len(values)} at "
                         f"{threshold}")
        verdicts[gate] = held
        print(f"{gate}: {'holds' if held else 'fails'} ({'; '.join(parts)})")
    gaps = [abs(life["W4_out_of_view"] - life["W4_map_rate"]) for life in brain]
    verdicts["W4_out_of_view"] = bool(gaps) and max(gaps) <= 0.02
    print(f"W4 out of view: {'holds' if verdicts['W4_out_of_view'] else 'fails'} "
          f"(largest gap from the map's own rate {max(gaps) if gaps else float('nan'):.3f})")
    ablated = [life["W6_ablated"] for life in brain]
    verdicts["W6"] = bool(ablated) and max(ablated) <= 0.05
    print(f"W6 ablation: the lexicon's synapses at zero answer W1 at "
          f"{max(ablated) if ablated else float('nan'):.3f} at most")
    return verdicts


def read2(out: Path, gates: dict[str, Any] | None = None,
          readings: list[str] | None = None, prefix: str = "brain_") -> dict[str, Any]:
    """Protocol 2's (or 3's) table and verdicts: a gate holds where every reading of it, over
    the twelve founders, has its mean on the right side of its threshold and at least ten
    founders do."""
    gates = GATES_2 if gates is None else gates
    shown = READINGS_2 if readings is None else readings
    arms: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out.glob("*.json")):
        arms.setdefault(path.stem.rsplit("_", 1)[0], []).append(json.loads(path.read_text()))
    head = ["arm", "founders", "parameters", "multiply-accumulates", "seconds"] + shown
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for arm, lives in arms.items():
        row = [arm, str(len(lives)), str(lives[0].get("parameters", "")),
               f"{np.mean([life.get('macs', 0) for life in lives]):.3g}",
               f"{np.mean([sum(life['seconds'].values()) for life in lives]):.0f}"]
        for name in shown:
            values = [life[name] for life in lives if name in life and life[name] == life[name]]
            row.append(f"{np.mean(values):.3f} ({min(values):.2f}-{max(values):.2f})"
                       if values else "")
        print("| " + " | ".join(row) + " |")
    verdicts: dict[str, Any] = {}
    for gate, needs in gates.items():
        held, parts = True, []
        for arm, name, threshold, side in needs:
            key = f"{prefix}{arm}" if prefix and not arm.startswith(prefix) else arm
            values = np.array([life[name] for life in arms.get(key, [])])
            reach = values >= threshold if side == "at least" else values <= threshold
            mean_ok = (values.mean() >= threshold if side == "at least"
                       else values.mean() <= threshold) if len(values) else False
            ok = bool(len(values)) and bool(mean_ok) and int(reach.sum()) >= min(10, len(values))
            held &= ok
            parts.append(f"{arm} {name} mean {values.mean():.3f}, {int(reach.sum())} of "
                         f"{len(values)} {side} {threshold}")
        verdicts[gate] = held
        print(f"{gate}: {'holds' if held else 'fails'} ({'; '.join(parts)})")
    return verdicts


if __name__ == "__main__":
    if sys.argv[1] in ("run", "run2", "run3"):
        run(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 30,
            {"run": 1, "run2": 2, "run3": 3}[sys.argv[1]])
    elif sys.argv[1] == "read2":
        read2(Path(sys.argv[2]))
    elif sys.argv[1] == "read3":
        read2(Path(sys.argv[2]), GATES_3, READINGS_3, prefix="")
    else:
        read(Path(sys.argv[2]))
