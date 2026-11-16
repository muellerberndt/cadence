"""The compositions the browser kernel is read on: demo-sized brains, a parrot-sized one and
Wild's pocket brain, with the random observations a life takes."""
from __future__ import annotations

import numpy as np

# Wild's pocket brain (cadence-wild, wild/brain.py at its founders' gene means): senses by
# name, a code of 128 where they meet, two apical regions under the deep lesson, recall with two
# heads, records of bound moments with a voice, surprise with its focus, and the map of things
POCKET_SENSES = {"eyes": 32, "ears": 8, "heard": 16, "thing": 20,
                 "food": {"what": 8, "bearing": 8, "distance": 4}, "body": 8}
POCKET = dict(senses=POCKET_SENSES, actions=8, code=128, recall_capacity=16, maturity=True,
              modules=(32, 32), apical=0.25, teaching="deep", recall_heads=2, bound=16,
              voice="heard", surprise={"hold": 0.85, "reach": 24.0}, temperature=10 ** -0.7,
              actor_eta=1.0, actor_eta_bias=0.05)
PARROT = dict(senses={"left eye": 48, "right eye": 48, "ear": 40, "body": 8}, actions=12,
              code=128, modules=(192, 96), apical=0.25, teaching="deep", recall_heads=2,
              recall_capacity=32)
COMPOSITIONS = [
    ("plain, 8 inputs", dict(inputs=8, actions=3)),
    ("chamber, 35 inputs", dict(inputs=35, actions=5)),
    ("deep apical (128, 64, 64, 64)", dict(inputs=48, actions=6, modules=(128, 64, 64, 64),
                                          apical=0.25, teaching="deep")),
    ("recall heads 2", dict(inputs=64, actions=6, recall_heads=2, recall_capacity=32)),
    ("free solves without the fast form", dict(inputs=35, actions=5,
                                               learning_fast_settle=False)),
    ("parrot-sized", PARROT),
    ("Wild's pocket brain", POCKET),
]


def pocket_map():
    from cadence.space.map import Map

    return Map((0.0, 24.0, 0.0, 24.0), (8, 8), units=4, values={"hue": 4}, reach=3.0,
               step=1.8, stride=6.0, margin=0.3, physics={"noise": 0.3}, surprise=True)


def observation(options, rng):
    if "senses" not in options:
        return rng.random((1, options["inputs"]))
    out = {}
    for name, width in options["senses"].items():
        if isinstance(width, dict):
            out[name] = {k: rng.random((1, w)) for k, w in width.items()}
        elif name not in ("ears", "heard") or rng.random() < 0.3:  # sounds come and go
            out[name] = rng.random((1, width))
    return out


