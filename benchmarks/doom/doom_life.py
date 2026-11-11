"""One player's life in the Doom body, episode after episode (cadence#285).

    PYTHONPATH=src python benchmarks/doom/doom_life.py --player brain --scenario basic \\
        --episodes 100 --seed 0 --out results/basic-brain-0.jsonl

Players: ``brain`` (``Brain.compose`` learning through ``step``), ``brain-blind`` (its retina
dark), ``brain-frozen`` (the newborn's sampled acts), ``conv`` (the convolutional
actor-critic) and ``random``. Each line of the output is one episode; the first line declares
the run. ``--table`` prints the summary of output files.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import doom_body  # noqa: E402
import doom_players  # noqa: E402

PLAYERS = ("brain", "brain-blind", "brain-frozen", "conv", "random")


class Refused(RuntimeError):
    """The player refused a moment; the world stands at ``tic``, not advanced."""

    def __init__(self, tic: int, cause: BaseException) -> None:
        super().__init__(f"the player refused the moment at tic {tic}: {cause}")
        self.tic = tic


def play(
    body: doom_body.DoomBody,
    player: Any,
    episodes: int,
    *,
    each: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """``episodes`` episodes of one life: the player answers each observation, the body steps
    the answer. A moment the player refuses (it raises) leaves the world where it stands and
    ends the life with ``Refused``. Returns one summary per episode, with its presses of each
    button (``buttons``); ``each`` sees each one as it ends."""
    summaries: list[dict[str, Any]] = []
    observation = body.reset()
    previous: doom_body.Transition | None = None
    clock = {"player": 0.0, "world": 0.0, "moments": 0, "kills": 0, "start": body.tic}
    buttons = [0] * body.actions
    while True:
        t0 = time.perf_counter()
        try:
            action = player.act(observation, previous)
        except Exception as cause:
            raise Refused(body.tic, cause) from cause
        t1 = time.perf_counter()
        transition = body.step(action)
        t2 = time.perf_counter()
        clock["player"] += t1 - t0
        clock["world"] += t2 - t1
        clock["moments"] += 1
        clock["kills"] += transition.events.count("kill")
        buttons[action] += 1
        if not transition.done:
            assert transition.observation is not None
            observation, previous = transition.observation, transition
            continue
        summary = {
            "episode": body.episode,
            "outcome": transition.outcome,
            "tic": transition.tic,
            "tics": transition.tic - clock["start"],
            "moments": clock["moments"],
            "return": body.returned,
            "kills": clock["kills"],
            "buttons": list(buttons),
            "player_seconds": clock["player"],
            "world_seconds": clock["world"],
        }
        summaries.append(summary)
        if each is not None:
            each(summary)
        if len(summaries) == episodes:
            t0 = time.perf_counter()
            player.finish(transition)
            summary["player_seconds"] = clock["player"] + time.perf_counter() - t0
            return summaries
        observation, previous = body.reset(), transition
        clock = {"player": 0.0, "world": 0.0, "moments": 0, "kills": 0, "start": body.tic}
        buttons = [0] * body.actions


def save_life(
    directory: str | Path,
    body: doom_body.DoomBody,
    player: doom_players.BrainPlayer,
    previous: doom_body.Transition | None,
) -> doom_body.Observation:
    """Checkpoints a brain's life between two moments: the game and the body
    (``DoomBody.checkpoint``), the brain with its pending action, and the transition the brain
    has yet to learn. Returns the observation the life continues from (the checkpoint's idle
    tics move the world, in this life and in the resumed one alike)."""
    if previous is not None and previous.done:
        raise ValueError("a life is checkpointed within an episode")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    game = body.checkpoint()
    player.save(directory / "brain")
    last = None
    if previous is not None:
        last = {
            "reward": previous.reward, "outcome": previous.outcome, "tic": previous.tic,
            "tics": previous.tics, "events": list(previous.events),
        }
    (directory / "life.json").write_text(json.dumps({"game": game, "previous": last}))
    return body.observe()


def load_life(
    directory: str | Path, **load: Any
) -> tuple[doom_body.DoomBody, doom_players.BrainPlayer, doom_body.Observation,
           doom_body.Transition | None]:
    """The life ``save_life`` wrote: the body at the same tic, the brain, the observation the
    life continues from and the transition the brain has yet to learn."""
    directory = Path(directory)
    state = json.loads((directory / "life.json").read_text())
    body = doom_body.DoomBody.restore(state["game"])
    player = doom_players.BrainPlayer.load(directory / "brain", **load)
    observation = body.observe()
    last = state["previous"]
    previous = None
    if last is not None:
        if last["outcome"] is not None:
            raise ValueError("a life is checkpointed within an episode")
        previous = doom_body.Transition(
            observation, float(last["reward"]), None, int(last["tic"]), int(last["tics"]),
            tuple(last["events"]),
        )
    return body, player, observation, previous


def make_player(name: str, body: doom_body.DoomBody, seed: int, **options: Any) -> Any:
    if name == "random":
        return doom_players.RandomPlayer(body.actions, seed=seed)
    if name == "conv":
        return doom_players.ConvActorCritic(
            body.resolution, body.actions, seed=seed, device=options.pop("device", "cpu")
        )
    mode = {"brain": "learn", "brain-blind": "blind", "brain-frozen": "frozen"}[name]
    options.pop("device", None)
    return doom_players.BrainPlayer(
        body.resolution, body.actions, seed=seed, mode=mode, **options
    )


def score(scenario: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The scenario's score over episodes: the outcome counts, the mean return, the mean tics
    and, by goal, the kill goal's completions and tics to complete, or the kills and tics
    survived of a survival."""
    outcomes = {name: sum(r["outcome"] == name for r in rows) for name in doom_body.OUTCOMES}
    moments = sum(r["moments"] for r in rows)
    out: dict[str, Any] = {
        "episodes": len(rows),
        "outcomes": outcomes,
        "return": float(np.mean([r["return"] for r in rows])),
        "tics": float(np.mean([r["tics"] for r in rows])),
        "kills": float(np.mean([r["kills"] for r in rows])),
        "completion": outcomes["completion"] / max(len(rows), 1),
        "moments": moments,
        "player_seconds_per_moment": sum(r["player_seconds"] for r in rows) / max(moments, 1),
        "world_seconds_per_moment": sum(r["world_seconds"] for r in rows) / max(moments, 1),
    }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--player", choices=PLAYERS, default="brain")
    parser.add_argument("--scenario", choices=sorted(doom_body.SCENARIOS), default="basic")
    parser.add_argument("--resolution", choices=sorted(doom_body.RESOLUTIONS), default="84x84")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tics", type=int, default=4)
    parser.add_argument("--backend", default="cpu", help="the brain's backend")
    parser.add_argument("--device", default="cpu", help="the convolutional player's device")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="a named setting of the brain players' Brain.compose")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--table", nargs="*", type=Path, help="summarise these output files")
    args = parser.parse_args(argv)
    if args.table is not None:
        print_table(args.table)
        return
    body = doom_body.DoomBody(
        args.scenario, resolution=args.resolution, tics=args.tics, seed=args.seed
    )
    options: dict[str, Any] = {"device": args.device}
    settings = {k: _setting(v) for k, v in (s.split("=", 1) for s in args.set)}
    if settings and not args.player.startswith("brain"):
        parser.error("--set names a setting of the brain players")
    if args.player.startswith("brain"):
        options = {"backend": args.backend, **settings}
    player = make_player(args.player, body, args.seed, **options)
    header = {
        "run": {
            "player": args.player, "scenario": args.scenario, "resolution": args.resolution,
            "episodes": args.episodes, "seed": args.seed, "tics": args.tics,
            "backend": args.backend if args.player.startswith("brain") else None,
            "device": args.device if args.player == "conv" else None,
            "parameters": player.parameters(),
            "scenario_spec": dataclasses.asdict(body.scenario),
            "engine": doom_body.engine_identity(),
            "machine": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(),
            "threads": os.environ.get("OMP_NUM_THREADS"),
        }
    }
    if args.player.startswith("brain"):
        header["run"]["settings"] = settings
        header["run"]["inputs"] = player.width
        header["run"]["placement"] = {
            k: str(v) for k, v in dict(player.brain.placement).items() if k in
            ("backend", "device", "precision")
        }
    lines = [json.dumps(header)]

    def each(summary: dict[str, Any]) -> None:
        lines.append(json.dumps(summary))
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text("\n".join(lines) + "\n")

    rows = play(body, player, args.episodes, each=each)
    body.close()
    lines.append(json.dumps({"score": score(args.scenario, rows)}))
    if args.out is not None:
        args.out.write_text("\n".join(lines) + "\n")
    print(json.dumps({"run": header["run"]["player"], "score": score(args.scenario, rows)}))


def _setting(text: str) -> Any:
    """A ``--set`` value: a JSON number, boolean or list, else the text itself."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def print_table(paths: list[Path]) -> None:
    """One markdown row per output file."""
    print(
        "| Scenario | Player | Seed | Parameters | Episodes | Completion | Death | Deadline "
        "| Truncation | Return | Kills | Tics | Moments | Player s/moment | World s/moment |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
          "| ---: | ---: | ---: |")
    for path in paths:
        lines = [json.loads(line) for line in path.read_text().splitlines() if line]
        run = lines[0]["run"]
        rows = [line for line in lines[1:] if "episode" in line]
        s = score(run["scenario"], rows)
        o = s["outcomes"]
        print(
            f"| {run['scenario']} | {run['player']} | {run['seed']} | {run['parameters']:,} "
            f"| {s['episodes']} | {o['completion']} | {o['death']} | {o['deadline']} "
            f"| {o['truncation']} | {s['return']:.3f} | {s['kills']:.2f} | {s['tics']:.0f} "
            f"| {s['moments']:,} | {s['player_seconds_per_moment']:.4f} "
            f"| {s['world_seconds_per_moment']:.4f} |"
        )


if __name__ == "__main__":
    main()
