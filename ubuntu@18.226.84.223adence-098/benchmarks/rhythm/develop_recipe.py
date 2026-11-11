"""Development grid for the steady-rhythm recipe on the declared development seeds.

Runs teaching and the free window only, for a grid of working-trace decays, amplitudes
and teaching rates, and writes one JSON table. Every cell, including every failed cell,
is retained. The grid selects the recipe frozen in protocol.json; it is not a result
on the confirmation seeds and claims no default.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np
import rhythm_inputs as inputs
import steady_rhythm as chamber


def cell(seed: int, arm: str, protocol: dict, frozen: dict) -> dict:
    brain = chamber.make_brain(seed, protocol)
    flip = chamber.FlipFlop(protocol["controls_config"]["flipflop_rate"])
    work = chamber.Work()
    teaching = chamber.teach_life(brain, flip, frozen, arm, work)
    observations = frozen["window/observations"]
    lead_count = 1 + protocol["window"]["lead"]
    window_work = chamber.Work()
    lead = chamber.run_events(brain, observations[:lead_count], window_work)
    flip_lead = np.stack([flip.act(x) for x in observations[:lead_count]])
    anchor = chamber.last_executed(lead, frozen["window/cues"])
    actions = chamber.run_events(brain, observations[lead_count:], window_work)
    flip_actions = np.stack([flip.act(x) for x in observations[lead_count:]])
    block = protocol["window"]["block"]
    score = chamber.score_window(actions, anchor, block)
    flip_score = chamber.score_window(flip_actions, chamber.last_executed(flip_lead, anchor), block)
    return {
        "seed": seed,
        "arm": arm,
        "last_bout_agreement": teaching["last_bout_agreement"],
        "lessons": teaching["lessons_attempted"],
        "alternation_rate": score["alternation_rate"],
        "agreement": score["agreement"],
        "blocks": score["blocks"],
        "refusals": score["refusals"] + int(np.sum(lead < 0)),
        "flipflop_alternation_rate": flip_score["alternation_rate"],
        "cue_followed": int(np.sum(lead[0] == frozen["window/cues"])),
        "teacher_sweeps": work.teacher_sweeps,
        "action_sweeps": work.action_sweeps + window_work.action_sweeps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decays", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument(
        "--rates",
        nargs="+",
        default=["default", "0.05n", "0.02n", "0.005n"],
        help="'default' keeps the composed law; '<eta>n' normalizes at 0.99",
    )
    parser.add_argument("--arms", nargs="+", default=["every", "mismatch"])
    parser.add_argument(
        "--efference-amplitudes",
        type=float,
        nargs="+",
        default=[0.0],
        help="read gains of the efference copy; 0.0 is the rhythm/1 brain without one",
    )
    parser.add_argument("--efference-decays", type=float, nargs="+", default=[0.2])
    parser.add_argument("--protocol", type=Path, default=inputs.PROTOCOL_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--bouts", type=int, default=None, help="teaching bouts; default protocol")
    args = parser.parse_args(argv)
    protocol, protocol_sha = inputs.load_protocol(args.protocol)
    seeds = protocol["seeds"]["development"] if args.seeds is None else args.seeds
    if args.bouts is not None:
        protocol["teaching"]["bouts"] = args.bouts
    args.out.mkdir(parents=True, exist_ok=False)
    frozen = {
        seed: inputs.freeze_inputs(
            args.out / f"inputs-seed{seed}.npz", seed=seed, protocol=protocol
        )
        for seed in seeds
    }
    began = time.perf_counter()
    cells = []
    for decay, amplitude, rate, echo, echo_decay, arm, seed in itertools.product(
        args.decays,
        args.amplitudes,
        args.rates,
        args.efference_amplitudes,
        args.efference_decays,
        args.arms,
        seeds,
    ):
        candidate = copy.deepcopy(protocol)
        brain = candidate["recipes"]["selected"]
        brain["working_memory_decay"] = decay
        brain["working_memory_amplitude"] = amplitude
        if echo:
            brain["efference_amplitude"] = echo
            brain["efference_decay"] = echo_decay
        else:
            brain.pop("efference_amplitude", None)
            brain.pop("efference_decay", None)
        if rate != "default":
            eta = float(rate.rstrip("n"))
            brain["learning"].update(
                {"eta": eta, "eta_bias": eta / 10.0, "normalize": 0.99, "normalize_floor": 0.0001}
            )
        result = {
            "decay": decay,
            "amplitude": amplitude,
            "rate": rate,
            "efference_amplitude": echo,
            "efference_decay": echo_decay if echo else None,
            **cell(seed, arm, candidate, frozen[seed]),
        }
        cells.append(result)
        print(json.dumps(result), flush=True)
    table = {
        "schema": "steady-rhythm-development/1",
        "protocol_sha256": protocol_sha,
        "seeds": list(seeds),
        "grid": {
            "decays": args.decays,
            "amplitudes": args.amplitudes,
            "rates": args.rates,
            "efference_amplitudes": args.efference_amplitudes,
            "efference_decays": args.efference_decays,
            "arms": args.arms,
            "bouts": protocol["teaching"]["bouts"],
        },
        "cells": cells,
        "seconds": time.perf_counter() - began,
    }
    (args.out / "development.json").write_text(json.dumps(table, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
