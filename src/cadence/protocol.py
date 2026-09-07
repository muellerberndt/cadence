"""Declared tests: named sets, stimuli, rows with predicates, and a shuffled control.

A protocol is data. It says which owners a stimulus clamps, which owners a
readout reads, which few facts a model may be shown, and which facts are
held out and scored. Predicates carry their preconditions, so a dead net
cannot pass "no response after ablation" vacuously. The control is the
same protocol on a wiring whose postsynaptic endpoints were permuted with
every count and sign kept: if the wiring predicts the held-out facts and
the permuted wiring does not, the prediction came from the wiring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .settle import Settlement
from .wiring import Wiring

__all__ = ["Row", "Protocol", "evaluate_predicate", "shuffled", "select_gain", "PREDICATES"]

PREDICATES: dict[str, str] = {
    "active": "mean(readout) >= active_level",
    "inactive": "mean(readout) <= inactive_level",
    "reduced": "reference active; mean(readout) <= mean(reference) - margin",
    "retained": "reference active; mean(readout) >= active_level",
    "released": "reference inactive; mean(readout) >= mean(reference) + margin",
    "exceeds": "mean(readout) >= active_level and mean(readout) >= mean(other) + margin",
    "lateralized": (
        "max(readout, other) >= inactive_level and |mean(readout) - mean(other)| >= margin"
    ),
    "sparse": "fraction >= sparse_min; fraction(readout) <= sparse_max",
    "densified": "reference sparse; fraction(readout) >= fraction(reference) + densify_margin",
}


@dataclass(frozen=True, slots=True)
class Levels:
    active: float = 0.5
    inactive: float = 0.2
    margin: float = 0.15
    sparse_min: float = 0.005
    sparse_max: float = 0.2
    densify_margin: float = 0.05


def evaluate_predicate(
    predicate: str,
    value: Mapping[str, float],
    reference: Mapping[str, float],
    levels: Levels | None = None,
) -> bool:
    """``value`` is the row's reading, ``reference`` its comparison; both are {mean, fraction}."""
    L = levels if levels is not None else Levels()
    if predicate == "active":
        return value["mean"] >= L.active
    if predicate == "inactive":
        return value["mean"] <= L.inactive
    if predicate == "reduced":
        return reference["mean"] >= L.active and value["mean"] <= reference["mean"] - L.margin
    if predicate == "retained":
        return reference["mean"] >= L.active and value["mean"] >= L.active
    if predicate == "released":
        return reference["mean"] <= L.inactive and value["mean"] >= reference["mean"] + L.margin
    if predicate == "exceeds":
        return value["mean"] >= L.active and value["mean"] >= reference["mean"] + L.margin
    if predicate == "lateralized":
        return (
            max(value["mean"], reference["mean"]) >= L.inactive
            and abs(value["mean"] - reference["mean"]) >= L.margin
        )
    if predicate == "sparse":
        return L.sparse_min <= value["fraction"] <= L.sparse_max
    if predicate == "densified":
        return (L.sparse_min <= reference["fraction"] <= L.sparse_max) and value[
            "fraction"
        ] >= reference["fraction"] + L.densify_margin
    raise KeyError(predicate)


@dataclass(frozen=True, slots=True)
class Row:
    """One held-out fact: under ``stimulus`` less ``ablate``, ``readout`` meets ``predicate``."""

    id: str
    stimulus: str
    readout: str
    predicate: str
    reference: str = ""
    ablate: tuple[str, ...] = ()
    relative_to: str = (
        ""  # a stimulus (for reduced/released) or a readout (for exceeds/lateralized)
    )
    tier: str = "experiment"


@dataclass
class Protocol:
    stimuli: dict[str, tuple[str, ...]]  # stimulus name -> set names clamped together
    rows: Sequence[Row]
    training: Sequence[
        tuple[str, str, str]
    ] = ()  # (stimulus, readout, predicate) the model may see
    levels: Levels = field(default_factory=Levels)
    steps: int = 60

    def clamp_for(self, wiring: Wiring, stimulus: str) -> tuple[int, ...]:
        return wiring.members(*self.stimuli[stimulus])

    def score(self, engine: Settlement) -> dict[str, Any]:
        """Settle every needed (stimulus, ablation) pair once and score training facts and rows."""
        wiring = engine.wiring
        cache: dict[tuple[str, tuple[str, ...]], dict[str, dict[str, float]]] = {}
        readouts = sorted(
            {r.readout for r in self.rows}
            | {t[1] for t in self.training}
            | {r.relative_to for r in self.rows if r.relative_to in wiring.sets}
        )

        def readings(stimulus: str, ablate: tuple[str, ...] = ()) -> dict[str, dict[str, float]]:
            key = (stimulus, ablate)
            if key not in cache:
                mask = np.ones(wiring.n)
                if ablate:
                    mask[list(wiring.members(*ablate))] = 0.0
                state = engine.settle(
                    list(self.clamp_for(wiring, stimulus)), steps=self.steps, mask=mask
                )
                out = engine.readings(state, readouts)
                out["_all"] = {
                    "mean": float(state.activation.mean()),
                    "fraction": state.fraction_active(),
                }
                cache[key] = out
            return cache[key]

        training = []
        for stimulus, readout, predicate in self.training:
            value = readings(stimulus)[readout]
            training.append(
                {
                    "stimulus": stimulus,
                    "readout": readout,
                    "predicate": predicate,
                    "reading": value,
                    "passed": evaluate_predicate(predicate, value, value, self.levels),
                }
            )
        rows: list[dict[str, Any]] = []
        for row in self.rows:
            value = readings(row.stimulus, row.ablate)[row.readout]
            if row.relative_to in self.stimuli:
                reference = readings(row.relative_to)[row.readout]
            elif row.relative_to in wiring.sets:
                reference = readings(row.stimulus, row.ablate)[row.relative_to]
            else:
                reference = readings(row.stimulus)[row.readout]
            rows.append(
                {
                    "id": row.id,
                    "stimulus": row.stimulus,
                    "ablate": list(row.ablate),
                    "readout": row.readout,
                    "predicate": row.predicate,
                    "tier": row.tier,
                    "reading": value,
                    "reference": reference,
                    "passed": evaluate_predicate(row.predicate, value, reference, self.levels),
                }
            )
        return {
            "training": training,
            "training_passed": sum(1 for t in training if t["passed"]),
            "rows": rows,
            "passed": sum(1 for r in rows if r["passed"]),
            "total": len(rows),
            "passed_by_tier": {
                tier: sum(1 for r in rows if r["passed"] and r["tier"] == tier)
                for tier in sorted({str(r["tier"]) for r in rows})
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimuli": {k: list(v) for k, v in self.stimuli.items()},
            "training": [list(t) for t in self.training],
            "rows": [
                {
                    "id": r.id,
                    "stimulus": r.stimulus,
                    "readout": r.readout,
                    "predicate": r.predicate,
                    "reference": r.reference,
                    "ablate": list(r.ablate),
                    "relative_to": r.relative_to,
                    "tier": r.tier,
                }
                for r in self.rows
            ],
            "levels": asdict(self.levels),
            "steps": self.steps,
            "predicates": PREDICATES,
        }


def shuffled(wiring: Wiring, seed: int, *, keep: np.ndarray | None = None) -> Wiring:
    """Permute postsynaptic endpoints; counts, signs, out-degrees, and named sets are kept.

    ``keep`` is a boolean mask over overlaps that are left untouched, for
    example the bridges between two joined datasets.
    """
    rng = np.random.default_rng(seed)
    post = wiring.post.copy()
    movable = np.ones(wiring.edges, dtype=bool) if keep is None else ~np.asarray(keep, bool)
    rows = np.flatnonzero(movable)
    post[rows] = post[rows][rng.permutation(len(rows))]
    return Wiring(
        wiring.n,
        wiring.pre,
        post,
        wiring.count,
        wiring.sign,
        wiring.sets,
        f"{wiring.label}:shuffled:{seed}",
    )


def select_gain(
    make_engine: Callable[[float], Settlement],
    protocol: Protocol,
    grid: Sequence[float],
    *,
    sparsity_cap: float | None = 0.05,
) -> tuple[float, list[dict[str, Any]]]:
    """One global gain from the training facts alone: most facts passed, smallest gain on ties.

    A gain is admissible only while the net stays sparse under every
    training stimulus (at most ``sparsity_cap`` of owners active). Runaway
    activity lights every readout and is not a fact about the wiring.
    """
    table = []
    best: tuple[int, float] | None = None
    for gain in grid:
        engine = make_engine(gain)
        wiring = engine.wiring
        passed = 0
        fraction = 0.0
        detail = {}
        for stimulus, readout, predicate in protocol.training:
            state = engine.settle(list(protocol.clamp_for(wiring, stimulus)), steps=protocol.steps)
            value = {
                "mean": state.mean(wiring.sets[readout]),
                "fraction": state.fraction_active(wiring.sets[readout]),
            }
            passed += int(evaluate_predicate(predicate, value, value, protocol.levels))
            fraction = max(fraction, state.fraction_active())
            detail[f"{stimulus}->{readout}"] = round(value["mean"], 4)
        admissible = sparsity_cap is None or fraction <= sparsity_cap
        table.append(
            {
                "gain": gain,
                "facts_passed": passed,
                "fraction_active": fraction,
                "admissible": admissible,
                "readings": detail,
            }
        )
        if admissible and (best is None or passed > best[0]):
            best = (passed, gain)
    if best is None:
        best = (0, min(grid))
    return best[1], table
