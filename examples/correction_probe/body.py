"""Bounded physical sensor fixture; no Cadence or model-dependent targets."""

import math
import random


def sensors(z):
    x, y = z
    return (
        math.tanh(x + 0.5 * y),
        math.tanh(-0.5 * x + y),
        math.tanh(x - y + 0.25 * x * y),
        math.tanh(0.7 * x - 0.4 * y + 0.35 * x * y),
    )


def transition(z, command):
    return tuple(0.35 * v + 0.6 * a for v, a in zip(z, command, strict=True))


def record(identity, before, command, noise, visible, bias=(0.0, 0.0)):
    """The prior is issued before transition; F is revealed after prediction.

    Body-owner state is retained for verification, never included by inputs().
    The disturbance alters a coarse sensor's calibration, not the target law.
    """
    if visible not in (0, 1):
        raise ValueError("Exactly one fine coordinate is currently visible")
    after = transition(before, command)
    measurements = sensors(after)
    previous = sensors(before)
    # The alternating shutter exposed the other coordinate at the last boundary.
    # A hidden prior measurement is never supplied merely because the body knows it.
    prior = [
        previous[0] if visible == 1 else 0.0,
        previous[1] if visible == 0 else 0.0,
        previous[2],
    ]
    coarse = tuple(v + n + b for v, n, b in zip(after, noise, bias, strict=True))
    if any(abs(v) >= 1 for v in (*after, *coarse, *measurements)):
        raise ValueError("Fixture must remain strictly within the state box")
    return {
        "id": identity,
        "before": list(before),
        "command": list(command),
        "after": list(after),
        "noise": list(noise),
        "coarse_bias": list(bias),
        "coarse": list(coarse),
        "visible": visible,
        "issued_persistence": prior,
        "actual": list(measurements),
    }


def inputs(row):
    visible = row["visible"]
    return {
        "coarse": row["coarse"],
        "command": row["command"],
        "prior": row["issued_persistence"],
        "fine": [
            row["actual"][0] if visible == 0 else 0.0,
            row["actual"][1] if visible == 1 else 0.0,
            row["actual"][2],
        ],
        "present": [float(visible == 0), float(visible == 1), 1.0],
    }


def facts(row, *, reveal=False):
    result = {
        f"p{row['visible']}": [row["actual"][row["visible"]]],
        "c": [row["actual"][2]],
    }
    if reveal:
        result["f"] = [row["actual"][3]]
    return result


def independent_rows(seed, count, prefix, bias=(0.0, 0.0)):
    rng = random.Random(seed)
    return [
        record(
            f"{prefix}-{i}",
            [rng.uniform(-0.6, 0.6) for _ in range(2)],
            [rng.uniform(-0.75, 0.75) for _ in range(2)],
            [rng.uniform(-0.02, 0.02) for _ in range(2)],
            i % 2,
            bias,
        )
        for i in range(count)
    ]


def live_rows(seed, count=64):
    rng = random.Random(seed)
    z = [rng.uniform(-0.6, 0.6) for _ in range(2)]
    rows = []
    for i in range(count):
        row = record(
            f"live-{i}",
            z,
            [rng.uniform(-0.75, 0.75) for _ in range(2)],
            [rng.uniform(-0.02, 0.02) for _ in range(2)],
            i % 2,
            (0.22, -0.18) if i >= 16 else (0.0, 0.0),
        )
        rows.append(row)
        z = row["after"]
    return rows


def data():
    return {
        "train": independent_rows(20261031, 512, "train"),
        "clean": independent_rows(20261032, 64, "clean"),
        "shifted": independent_rows(20261033, 64, "shifted", (0.22, -0.18)),
        "live": live_rows(20261034),
    }
