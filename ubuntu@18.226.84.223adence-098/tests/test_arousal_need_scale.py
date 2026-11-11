"""``ArousalConfig.need_scale``: a need in units of the reward scale the brain has learned.

Above zero, at every outcome the want is at least the unmet share of ``need_scale * scale``,
with the scale as it stood before that outcome entered it; the want and heat read afterwards
use the scale as it stands; youth, and a scale no outcome has entered, leave the founder need
of zero. These are the law's tests: the gene against the reference subclass the reading used,
and the saves of 0.84.3. The composed brain's need scale is 0.1, a constant since 0.93.0
(``tests/test_public_configuration.py``); the law alone takes any value. The law's readings at
the founder are pinned as recorded on main at c7dc2f0 (the NumPy solver).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import cadence as cd
from cadence.arousal import Arousal

Outcome = tuple[float, float, bool, bool, float | None]


class ScaledNeedReference(Arousal):
    """The scaled need as a subclass of the law, the instrument the gene reproduces: the need
    set at every reading to ``need_scale`` times the reward scale the arousal has learned,
    outside the config, with the fixed need at the founder zero."""

    need_scale: float = 0.0

    def _want(self, longrun: float, recent: float, scale: float) -> float:
        want = super()._want(longrun, recent, scale)
        k = float(self.need_scale)
        if k <= 0.0 or scale <= 0.0 or self.age < self.config.youth:
            return want
        need = k * scale
        unmet = 1.0 if recent <= 0.0 else (0.0 if recent >= need else (need - recent) / need)
        return max(want, unmet)


def outcomes(count: int, seed: int) -> Iterator[Outcome]:
    """A fixed stream of outcomes: the error, the reward, whether the action was the brain's
    own, whether the outcome was learned from, and a fifth draw that keeps the stream's draws
    as they were recorded."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield (
            float(abs(rng.normal(0.0, 0.5))), float(rng.choice([0.0, 1.0, -1.0, 0.5])),
            bool(rng.random() < 0.8), bool(rng.random() < 0.5),
            None if rng.random() < 0.5 else float(abs(rng.normal(0.0, 0.5))),
        )


def unmet(need: float, recent: float) -> float:
    return 1.0 if recent <= 0.0 else (0.0 if recent >= need else (need - recent) / need)


def law_want(arousal: cd.Arousal, scale: float) -> float:
    """The law's own want at the arousal's readings, against ``scale``, at the founder need."""
    return 0.0 if scale <= 0.0 else float(
        np.clip((arousal.longrun - arousal.recent) / scale, 0.0, 1.0)
    )


def state_without_config(state: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if k not in ("format", "config")}


# -- the gene


def test_need_scale_is_a_gene_with_the_founder_0_1_outside_the_declared_space() -> None:
    """The space names the ten genes of the law beside the need scale, so a seeded mutation of
    the founder draws as it does without the gene and copies the founder need scale to the
    child; a space that adds the gene can move it."""
    config = cd.ArousalConfig()
    assert config.need_scale == 0.1 and config.to_dict()["need_scale"] == 0.1
    assert cd.ArousalConfig(need_scale=0.1).to_dict()["need_scale"] == 0.1
    assert cd.ArousalConfig(**cd.ArousalConfig(need_scale=0.1).to_dict()).need_scale == 0.1
    space = cd.ArousalConfig.space()
    assert list(space) == [
        "threshold", "decay", "tolerance", "floor", "fast", "slow", "heat", "youth",
        "value_surprise", "need",
    ]
    assert set(config.to_dict()) == set(space) | {"need_scale"}
    before = {k: v for k, v in config.to_dict().items() if k != "need_scale"}
    child = cd.genes(space)(config.to_dict(), np.random.default_rng(1))
    assert child["need_scale"] == 0.1
    assert {k: v for k, v in child.items() if k != "need_scale"} == cd.genes(space)(
        before, np.random.default_rng(1)
    )
    assert cd.ArousalConfig(**child).need_scale == 0.1
    extended = {**space, "need_scale": ("linear", 0.02, 0.0, 1.0)}
    moved = [cd.genes(extended)(config.to_dict(), np.random.default_rng(seed))["need_scale"]
             for seed in range(20)]
    assert all(value >= 0.0 for value in moved) and any(value > 0.0 for value in moved)
    assert cd.ArousalConfig(**cd.genes(extended)(config.to_dict(), np.random.default_rng(3)))


@pytest.mark.parametrize("value", [-0.1, True, np.bool_(False), np.nan, np.inf, -np.inf, "0.1"])
def test_need_scale_rejects_invalid_genes(value: Any) -> None:
    with pytest.raises(ValueError, match="need_scale"):
        cd.ArousalConfig(need_scale=value)


# -- the law


@pytest.mark.parametrize("scale", [0.1, 0.5])
def test_the_gene_is_the_reference_subclass_outcome_for_outcome(scale: float) -> None:
    """The gene and the subclass give the same surprise and want at every outcome and the
    same want, heat, mode and level after it, through a reset; the law without the scaled
    need differs."""
    gene = cd.Arousal(cd.ArousalConfig(youth=20, need_scale=scale))
    reference = ScaledNeedReference(cd.ArousalConfig(youth=20, need_scale=0.0))
    reference.need_scale = scale
    founder = cd.Arousal(cd.ArousalConfig(youth=20, need_scale=0.0))
    differed = 0
    for t, (error, reward, own, learned, _) in enumerate(outcomes(600, seed=int(10 * scale))):
        taken = [a.outcome(error, reward, own=own, learned=learned)
                 for a in (gene, reference, founder)]
        assert taken[0] == taken[1]
        assert (gene.want, gene.heat, gene.aroused, gene.level) == (
            reference.want, reference.heat, reference.aroused, reference.level
        )
        differed += taken[0] != taken[2]
        for arousal in (gene, reference, founder):
            arousal.lived(3, 1)
            if t == 300:
                arousal.reset()
    assert differed > 0
    state, other = gene.to_dict(), reference.to_dict()
    assert state["format"] == other["format"] == "cadence-arousal/4"
    assert state["config"]["need_scale"] == scale and other["config"]["need_scale"] == 0.0
    assert state_without_config(state) == state_without_config(other)


def test_an_outcome_reads_the_scale_before_it_enters_and_the_heat_the_scale_as_it_stands() -> None:
    """The want an outcome returns measures the recent reward against ``need_scale`` times
    the scale as it stood before that outcome entered it; the want and heat read afterwards
    use the scale as it stands; while no outcome has entered the scale the need is zero."""
    k = 0.5
    gene = cd.Arousal(cd.ArousalConfig(youth=0, need_scale=k, heat=2.0))
    assert gene.scale == 0.0 and gene.want == 0.0 and gene.heat == 1.0
    exceeded = 0
    for error, reward, own, learned, _ in outcomes(300, seed=3):
        before = gene.scale
        _, want = gene.outcome(error, reward, own=own, learned=learned)
        law = law_want(gene, before)
        expected = max(law, unmet(k * before, gene.recent)) if before > 0.0 else law
        assert want == expected
        after = gene.scale
        assert gene.want == (
            max(law_want(gene, after), unmet(k * after, gene.recent)) if after > 0.0
            else law_want(gene, after)
        )
        assert gene.heat == 1.0 + 2.0 * gene.want
        exceeded += before > 0.0 and want > law
    assert exceeded > 0


def test_youth_and_a_scale_no_outcome_has_entered_leave_the_founder_need() -> None:
    k = 0.5
    gene = cd.Arousal(cd.ArousalConfig(youth=30, need_scale=k))
    founder = cd.Arousal(cd.ArousalConfig(youth=30))
    stream = list(outcomes(30, seed=5))
    for error, reward, own, _, _ in stream:  # youth: every outcome enters the scale
        a = gene.outcome(error, reward, own=own, learned=True)
        b = founder.outcome(error, reward, own=own, learned=True)
        assert a == b and gene.want == founder.want and gene.scale == founder.scale
        gene.lived(1)
        founder.lived(1)
    assert gene.age == 30 == gene.config.youth and gene.scale > 0.0
    # an adult reset by default has an empty scale; calm outcomes that are not learned from
    # never enter it, so the scaled need stays at zero
    gene.reset()
    founder.reset()
    assert gene.scale == 0.0 and gene.spreads == 0
    for _, reward, _, _, _ in stream:
        assert gene.outcome(0.0, reward, learned=False) == founder.outcome(0.0, reward,
                                                                           learned=False)
        assert gene.want == founder.want == 0.0 and gene.scale == 0.0 and not gene.aroused
    # one learned outcome enters the scale; from then on the scaled need wants
    assert gene.outcome(0.0, 1.0, learned=True) == founder.outcome(0.0, 1.0, learned=True)
    assert gene.scale > 0.0
    larger = 0
    for _ in range(40):
        gene.outcome(0.0, 0.0, learned=True)
        founder.outcome(0.0, 0.0, learned=True)
        assert gene.want >= founder.want
        larger += gene.want > founder.want
    assert larger > 0 and gene.aroused


def test_a_need_and_a_need_scale_together_want_the_larger() -> None:
    both = cd.Arousal(cd.ArousalConfig(youth=0, need=0.4, need_scale=0.5))
    fixed = cd.Arousal(cd.ArousalConfig(youth=0, need=0.4))
    scaled = cd.Arousal(cd.ArousalConfig(youth=0, need_scale=0.5))
    for error, reward, own, _, _ in outcomes(200, seed=7):  # every outcome learned
        wants = [a.outcome(error, reward, own=own, learned=True)[1]
                 for a in (both, fixed, scaled)]
        assert wants[0] == max(wants[1], wants[2])
        assert both.want == max(fixed.want, scaled.want)


def test_a_save_needs_its_need_scale_and_its_readings() -> None:
    """A save of ``cadence-arousal/4`` needs every gene, the need scale among them, and every
    reading."""
    arousal = cd.Arousal(cd.ArousalConfig(youth=2, need_scale=0.0))
    for error, reward, own, learned, _ in outcomes(12, seed=9):
        arousal.outcome(error, reward, own=own, learned=learned)
        arousal.lived(5, 2)
    current = arousal.to_dict()
    assert current["format"] == "cadence-arousal/4" and current["config"]["need_scale"] == 0.0
    assert cd.Arousal.from_dict(current).to_dict() == current
    genes = {k: v for k, v in current["config"].items() if k != "need_scale"}
    with pytest.raises(ValueError, match="incomplete saved arousal config"):
        cd.Arousal.from_dict({**current, "config": genes})
    with pytest.raises(ValueError, match="incomplete saved arousal"):
        cd.Arousal.from_dict({k: v for k, v in current.items() if k != "rewards"})


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def law_readings() -> dict[str, tuple[list[list[Any]], dict[str, Any]]]:
    """The law alone, the founder and then a needy one, on one stream of outcomes with a
    reset in the middle: the surprise and want of every outcome, the want, heat and mode
    after it, and the state at the end."""
    rng = np.random.default_rng(20261021)
    out = {}
    for name, config in (("founder", cd.ArousalConfig(youth=5)),
                         ("needy", cd.ArousalConfig(youth=5, need=0.3))):
        arousal = cd.Arousal(config)
        readings: list[list[Any]] = []
        for t in range(120):
            reward = float(rng.choice([0.0, 1.0, -1.0, 0.5]))
            error = float(abs(rng.normal(0.0, 0.5)))
            own = bool(rng.random() < 0.8)
            learned = bool(rng.random() < 0.5)
            if rng.random() >= 0.5:  # the draws of the stream as it was recorded
                rng.normal(0.0, 0.5)
            surprise, want = arousal.outcome(error, reward, own=own, learned=learned)
            readings.append([surprise, want, arousal.want, arousal.heat, arousal.aroused])
            arousal.lived(int(rng.integers(1, 40)), int(rng.integers(0, 10)))
            if t == 60:
                arousal.reset()
        out[name] = (readings, arousal.to_dict())
    return out


# Recorded on main at c7dc2f0: the law's readings on the stream above; its saved state is
# the one of main in the layout of ``cadence-arousal/4``. Re-recorded at the founder need
# scale 0.1 (issue 198).
LAW: dict[str, tuple[str, str]] = {
    "founder": ("636f4e4c64d77ca7fca23131a128f277179c28f92d6a72f7d35b44a9b303a942",
                "d5aedc61ceadc9800fe5f917c86b836c1998375c6921829475a970766de82d6b"),
    "needy": ("f41e6fd01cacac4128368ce8d3cbb6b0e5d785f15edd2ca501d023913ba82066",
              "c3406214ca4f0789c3d166533ef3d98b29517be874a0da55c2ad6868412fc5e4"),
}


def test_the_law_alone_reads_and_saves_as_on_main() -> None:
    for name, (readings, state) in law_readings().items():
        assert state["format"] == "cadence-arousal/4" and state["config"]["need_scale"] == 0.1
        assert (digest(readings), digest(state)) == LAW[name]
