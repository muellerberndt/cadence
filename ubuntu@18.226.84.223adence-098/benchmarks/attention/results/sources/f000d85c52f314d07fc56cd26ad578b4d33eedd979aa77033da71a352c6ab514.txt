"""Arousal: when a continuing brain leaves routine, and how it returns.

A calm brain answers from its settled state and changes nothing. Two things rouse it:
an outcome that contradicts the forecast it made before acting (surprise), and a
reward that stays below what its life usually pays (want, the suffering of a missed
objective that it predicts correctly). An aroused brain samples its actions, keeps their
eligibility, learns from every outcome and writes memory; the more it wants, the wider
it explores. ``Brain.live`` runs this loop for a composed brain.

The law, per outcome of the stream's preceding action::

    surprise = log(|error| / (tolerance * usual + floor * scale))   when positive, else 0
    want     = max(clip((longrun - recent) / scale, 0, 1),
                   clip((need - recent) / need, 0, 1),
                   clip((need_scale * scale - recent) / (need_scale * scale), 0, 1))
    level    = decay * level + (1 - decay) * (surprise + want)

``error`` is the temporal-difference error against the forecast made before the outcome;
``usual`` is its running size; ``recent`` and ``longrun`` are the running reward at a
fast and a slow rate; ``scale`` is the spread of the outcomes the brain has learned from;
``need`` is the reward per moment the body requires, and the share of it the recent reward
leaves unmet is a want of its own, so a life that never paid, or stopped paying for good,
still wants. The need's want is measured against the need and not against the spread,
because a reward that comes rarely has a spread far above its mean: a brain fed once in
``L`` moments that loses its food falls short of its long-run reward by only ``1 / sqrt(L)``
spreads, while it is short of its whole need. A need of zero has no want of its own.
``need_scale`` is a need in units of that spread: at every outcome the want is also at
least the unmet share of ``need_scale * scale``, with the scale as it stood before the
outcome entered it, and the exploration heat reads the want with the scale as it stands.
It applies after youth and once an outcome has entered the scale; a scale of zero, as
after a default reset, leaves it at zero. A need is a level of reward and a scaled need
a share of what the life's rewards spread over, so a life that pays rarely wants once its
income falls below that share of its spread. A need scale of zero has no want of its own.
The surprise is weighted by ``value_surprise``.
Only the outcome of the brain's own best guess can surprise it and enters its usual
forecast error. Every actual outcome enters the recent and long-run reward: exploring
does not stop the brain from noticing what its life pays. The brain is aroused while
``level >= threshold`` and during its first ``youth`` moments. An aroused brain samples
its policy, with one uniformly chosen motor slot at ``1 + heat * want`` times the base
temperature; other slots keep the base temperature. The law is unchanged when
rewards, errors and the need are multiplied by one positive number, away from its absolute
``1e-12`` surprise guard. Reward shifts also preserve the law when the need and the need
scale are zero: every outcome establishes the reward reference before its spread forms. A
need is a level of reward and a scaled need a share of the spread around the reference,
and shifting the rewards changes what either leaves unmet.

Arousal gates learning and protects no acquired response, so continued learning can
interfere with earlier behaviour. Raising ``threshold`` can reduce learning after youth;
youth and a level at or above the threshold always rouse the brain. A smaller actor step
reduces interference without retaining a skill
(``docs/reward.md#traps-with-their-measurements``).

Every constant of the law is a gene of ``ArousalConfig``. The defaults are hand-set
founders and serve as the control; ``ArousalConfig.space()`` declares the space for
``cadence.genes``. The law is a broadcast signal computed from the brain's own dopamine
and reward, like the dopamine itself; it holds no settled patch state and no learned
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["Arousal", "ArousalConfig"]

MODES = ("routine", "aroused")
_TINY = 1e-12  # keeps the unit of surprise positive in a world that has paid nothing yet
_FORMAT = "cadence-arousal/4"  # every gene of the law and the state it reads
# The formats of 0.84.3, read through ``_earlier``: /2 at the founder need scale of zero, /3
# with a positive one, /1 and the unversioned save before the reward count.
_EARLIER = ("cadence-arousal/1", "cadence-arousal/2", "cadence-arousal/3")


def _log_ratio(error: float, unit: float) -> float:
    """``log(error / unit)`` for a positive error above its unit, even when the ratio
    itself is not representable."""
    ratio = error / unit
    return float(np.log(ratio) if np.isfinite(ratio) else np.log(error) - np.log(unit))


def _real(
    name: str, value: Any, *, low: float, high: float = np.inf, open_low: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    below = number <= low if open_low else number < low
    if not np.isfinite(number) or below or number > high:
        raise ValueError(f"{name} is outside its range")
    return number


@dataclass(frozen=True, slots=True)
class ArousalConfig:
    """The genes of a continuing brain's arousal; the defaults are the hand-set founders."""

    threshold: float = 0.2  # arousal at or above this leaves routine
    decay: float = 0.9  # the share of arousal one moment hands to the next
    tolerance: float = 2.0  # an error within this many usual errors is no surprise
    floor: float = 0.1  # nor is an error within this many reward scales
    fast: float = 0.05  # rate of the recent reward average
    slow: float = 0.005  # rate of the long-run reward, the usual error and the reward scale
    heat: float = 2.0  # exploration temperature gained per unit of want
    youth: int = 100  # the first moments of a life are aroused
    value_surprise: float = 1.0  # weight of the surprise at a contradicted value forecast
    need: float = 0.0  # the reward per moment the body requires; the unmet share is a want
    need_scale: float = 0.1  # a need in reward scales, need_scale * scale, after youth

    def __post_init__(self) -> None:
        # A scalar's Python/NumPy origin must not change continuation arithmetic.
        for name in self.__slots__:
            value = getattr(self, name)
            if isinstance(value, np.integer):
                object.__setattr__(self, name, int(value))
            elif isinstance(value, np.floating):
                object.__setattr__(self, name, float(value))
        _real("threshold", self.threshold, low=0.0)
        if not 0 <= _real("decay", self.decay, low=0.0) < 1:
            raise ValueError("decay must lie in [0, 1)")
        _real("tolerance", self.tolerance, low=0.0)
        _real("floor", self.floor, low=0.0)
        slow = _real("slow", self.slow, low=0.0, high=1.0, open_low=True)
        fast = _real("fast", self.fast, low=0.0, high=1.0, open_low=True)
        if fast < slow:
            raise ValueError("fast must be at least slow; equal rates remove the long-run want")
        _real("heat", self.heat, low=0.0)
        _real("value_surprise", self.value_surprise, low=0.0)
        _real("need", self.need, low=0.0)
        _real("need_scale", self.need_scale, low=0.0)
        if (
            isinstance(self.youth, (bool, np.bool_))
            or not isinstance(self.youth, (int, np.integer))
            or self.youth < 0
        ):
            raise ValueError("youth must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {name: float(getattr(self, name)) for name in self.__slots__}
        values["youth"] = int(self.youth)
        return values

    @staticmethod
    def space() -> dict[str, tuple[Any, ...]]:
        """The gene space of the law for ``cadence.genes``; mutate ``to_dict()`` over it.

        ``fast`` at ``slow`` removes the long-run want, a large ``tolerance`` or
        ``value_surprise`` at zero removes surprise, ``heat`` at zero removes the wider
        exploration and ``need`` at zero removes the need's want: the controls are inside the
        space. A mutation that leaves ``fast`` below ``slow`` is refused at
        construction. ``need_scale`` is left out, so a seeded mutation of the founder draws
        as it did before the gene and copies the founder 0.1 to the child; to select on it,
        add it to the space, as ``{**ArousalConfig.space(), "need_scale": ("linear", 0.02,
        0.0, 1.0)}``."""
        return {
            "threshold": ("log", 0.3, 0.02, 2.0),
            "decay": ("linear", 0.05, 0.0, 0.99),
            "tolerance": ("log", 0.3, 0.5, 20.0),
            "floor": ("log", 0.3, 0.01, 1.0),
            "fast": ("log", 0.3, 0.005, 0.5),
            "slow": ("log", 0.3, 0.0005, 0.05),
            "heat": ("linear", 0.5, 0.0, 4.0),
            "youth": ("int", 0, 1000),
            "value_surprise": ("linear", 0.3, 0.0, 2.0),
            "need": ("linear", 0.02, 0.0, 1.0),
        }


class Arousal:
    """The arousal of one continuing stream: its level, what it is used to, and its work.

    ``outcome`` applies the law to one outcome. ``moments`` and ``sweeps`` count, per
    mode, the moments lived and the settling sweeps of their answers and forecasts;
    eligibility and feedback sweeps of aroused moments are counted in ``learning_sweeps``.
    ``reset`` begins another stream calm, with what the brain was used to forgotten and
    its age and work kept.
    """

    def __init__(self, config: ArousalConfig | None = None) -> None:
        self.config = config or ArousalConfig()
        self.age = 0
        self.moments = {mode: 0 for mode in MODES}
        self.sweeps = {mode: 0 for mode in MODES}
        self.learning_sweeps = 0
        self.reset()

    def reset(self) -> None:
        self.level = 0.0
        self.outcomes = 0  # own outcomes that entered the usual TD error
        self.rewards = 0  # all outcomes that entered the recent and long-run reward
        self.spreads = 0  # outcomes that entered the reward scale
        self._square = 0.0
        self._recent = 0.0
        self._longrun = 0.0
        self._usual = 0.0

    # -- what the stream is used to: running averages, corrected for their short history

    def _reading(self, accumulated: float, rate: float, count: int | None = None) -> float:
        count = self.outcomes if count is None else count
        if not count:
            return 0.0
        correction = 1.0 - (1.0 - rate) ** count
        if correction == 0.0:
            # A valid positive rate can be too small for ``1 - rate`` to differ from one.
            correction = float(-np.expm1(count * np.log1p(-rate)))
        return accumulated / correction

    @property
    def scale(self) -> float:
        """The spread of the outcomes the brain has learned from: the running RMS distance
        of their reward from the long-run reward. The unit of a want and of the surprise
        floor; routine outcomes leave it alone, so a long calm does not shrink it."""
        return float(np.sqrt(max(self._reading(self._square, self.config.slow, self.spreads), 0.0)))

    @property
    def recent(self) -> float:
        return self._reading(self._recent, self.config.fast, self.rewards)

    @property
    def longrun(self) -> float:
        return self._reading(self._longrun, self.config.slow, self.rewards)

    @property
    def usual(self) -> float:
        """The running size of the temporal-difference error."""
        return self._reading(self._usual, self.config.slow)

    def _want(self, longrun: float, recent: float, scale: float) -> float:
        """The shortfall of the recent reward below the long-run reward in reward scales,
        below the body's need as a share of that need, or below the need of ``need_scale``
        reward scales as a share of that need, whichever is largest."""
        want = 0.0 if scale <= 0.0 else float(np.clip((longrun - recent) / scale, 0.0, 1.0))
        need = float(self.config.need)
        if need > 0.0:
            # Clip before arithmetic: a negative reward with a tiny need can overflow
            # the ratio, and a large need can overflow the subtraction.
            unmet = 1.0 if recent <= 0.0 else (0.0 if recent >= need else (need - recent) / need)
            want = max(want, unmet)
        share = float(self.config.need_scale)
        if share > 0.0 and scale > 0.0 and self.age >= self.config.youth:
            # The need in reward scales, measured against the scale this call reads: in
            # ``outcome`` the scale before the outcome enters it, in ``want`` the scale as it
            # stands. Youth, and a scale no outcome has entered, leave the founder need.
            need = share * scale
            unmet = 1.0 if recent <= 0.0 else (0.0 if recent >= need else (need - recent) / need)
            want = max(want, unmet)
        return want

    @property
    def want(self) -> float:
        """The shortfall of the recent reward below the long-run reward, in reward scales,
        below the body's need, as a share of the need, or below the need of ``need_scale``
        reward scales, as a share of that need; the scale as it stands."""
        return self._want(self.longrun, self.recent, self.scale)

    @property
    def aroused(self) -> bool:
        return self.age < self.config.youth or self.level >= self.config.threshold

    @property
    def mode(self) -> str:
        return MODES[int(self.aroused)]

    @property
    def heat(self) -> float:
        """The temperature factor for the one motor slot selected for extra exploration."""
        return 1.0 + self.config.heat * self.want

    # -- the law

    def outcome(
        self,
        error: float,
        reward: float,
        *,
        own: bool = True,
        learned: bool = True,
    ) -> tuple[float, float]:
        """Take one outcome: the unsigned TD error against the forecast made before it and
        the reward. Returns the surprise and the want that entered the level.

        ``own`` says the action was the brain's own best guess. Only such an outcome can
        surprise it and enters its usual forecast error. Every actual outcome enters
        recent and long-run reward, so income stays current while exploring. ``learned`` says
        the brain learns from the outcome (it was sampled while aroused); learned outcomes
        and the outcome that wakes the brain enter the reward scale. A stream's first own
        outcome has no usual error to be measured against and is no surprise."""
        error, reward = abs(float(error)), float(reward)
        if not np.isfinite(error) or not np.isfinite(reward):
            raise ValueError("error and reward must be finite")
        c = self.config
        outcomes, spreads = self.outcomes, self.spreads
        rewards = self.rewards + 1
        recent, longrun, usual = self._recent, self._longrun, self._usual
        square, scale = self._square, self.scale
        recent += c.fast * (reward - recent)
        longrun += c.slow * (reward - longrun)
        surprise = 0.0
        if own:
            first = outcomes == 0
            previous_usual = self.usual  # the error before this outcome
            outcomes += 1
            usual += c.slow * (error - usual)
            unit = c.tolerance * previous_usual + c.floor * scale + _TINY
            if error > unit and not first:
                surprise = c.value_surprise * _log_ratio(error, unit)
        expected = self._reading(longrun, c.slow, rewards)
        want = self._want(expected, self._reading(recent, c.fast, rewards), scale)
        level = c.decay * self.level + (1.0 - c.decay) * (surprise + want)
        if learned or self.age < c.youth or level >= c.threshold:
            spreads += 1
            try:
                square += c.slow * ((reward - expected) ** 2 - square)
            except OverflowError as exc:
                raise ValueError("arousal moments must remain finite") from exc
        if not np.isfinite([recent, longrun, usual, square, level]).all():
            raise ValueError("arousal moments must remain finite")
        # Admit the complete outcome together so an unrepresentable spread can be retried.
        self.outcomes, self.spreads, self.rewards = outcomes, spreads, rewards
        self._recent, self._longrun, self._usual = recent, longrun, usual
        self._square, self.level = square, level
        return surprise, want

    def lived(self, sweeps: int, learning_sweeps: int = 0) -> None:
        """Count one lived moment in the mode it was answered in."""
        mode = self.mode
        self.moments[mode] += 1
        self.sweeps[mode] += int(sweeps)
        self.learning_sweeps += int(learning_sweeps)
        self.age += 1

    # -- continuation

    def to_dict(self) -> dict[str, Any]:
        """The complete state under ``cadence-arousal/4``: every gene and every reading."""
        return {
            "format": _FORMAT,
            "config": self.config.to_dict(),
            "level": float(self.level),
            "age": int(self.age),
            "outcomes": int(self.outcomes),
            "rewards": int(self.rewards),
            "spreads": int(self.spreads),
            "square": float(self._square),
            "recent": float(self._recent),
            "longrun": float(self._longrun),
            "usual": float(self._usual),
            "moments": {mode: int(self.moments[mode]) for mode in MODES},
            "sweeps": {mode: int(self.sweeps[mode]) for mode in MODES},
            "learning_sweeps": int(self.learning_sweeps),
        }

    @classmethod
    def from_dict(cls, values: Any) -> Arousal:
        """Restore a saved arousal, rejecting incomplete or invalid state. A save of 0.84.3
        (``cadence-arousal/1`` to ``/3``, or unversioned) loads as its state in this format."""
        if not isinstance(values, dict) or not isinstance(values.get("config"), dict):
            raise ValueError("invalid saved arousal")
        try:
            if values.get("format") != _FORMAT:
                values = _earlier(values)
            config = values["config"]
            if set(config) != set(ArousalConfig.__slots__):
                raise ValueError("invalid or incomplete saved arousal config")
            arousal = cls(ArousalConfig(**config))
            for name in ("age", "outcomes", "rewards", "spreads", "learning_sweeps"):
                value = values[name]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid saved arousal count: {name}")
            if values["rewards"] < values["outcomes"]:
                raise ValueError("invalid saved arousal count: rewards")
            arousal.age, arousal.outcomes = values["age"], values["outcomes"]
            arousal.rewards, arousal.spreads = values["rewards"], values["spreads"]
            arousal.learning_sweeps = values["learning_sweeps"]
            nonnegative = ("level", "square", "usual")
            for name in ("level", "square", "recent", "longrun", "usual"):
                value = values[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"invalid saved arousal value: {name}")
                if not np.isfinite(value) or (name in nonnegative and value < 0):
                    raise ValueError(f"invalid saved arousal value: {name}")
            arousal.level = float(values["level"])
            arousal._square, arousal._recent = float(values["square"]), float(values["recent"])
            arousal._longrun, arousal._usual = float(values["longrun"]), float(values["usual"])
            for name in ("moments", "sweeps"):
                counts = values[name]
                if not isinstance(counts, dict) or set(counts) != set(MODES):
                    raise ValueError(f"invalid saved arousal counts: {name}")
                for mode in MODES:
                    value = counts[mode]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError(f"invalid saved arousal counts: {name}")
                setattr(arousal, name, {mode: counts[mode] for mode in MODES})
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid or incomplete saved arousal") from exc
        return arousal


def _earlier(values: dict[str, Any]) -> dict[str, Any]:
    """A save of 0.84.3 as its state in ``cadence-arousal/4``. Format /2 left out the founder
    need scale of zero and /3 carried a positive one; /1 and the unversioned save counted
    only own outcomes in their income, so their reward count defaults to their outcome count
    and further outcomes enter by the current law; an unversioned save without ``need`` holds
    the zero-need law. Every one of them carried the record channel, a second surprise at
    the error of the outcome forecast by the action's record, which this version removed: at
    its founder weight of zero it never entered the level, and its readings are dropped; a
    save that weighed it otherwise is refused with a message naming 0.84.3."""
    if "format" in values and values["format"] not in _EARLIER:
        raise ValueError("invalid saved arousal format")
    config = dict(values["config"])
    genes = set(ArousalConfig.__slots__) | {"record_surprise"}
    if "format" not in values and set(config) == genes - {"need", "need_scale"}:
        # the format marker distinguishes that schema from a save missing a gene
        config["need"] = 0.0
    if values.get("format") != "cadence-arousal/3":
        if config.get("need_scale", 0.0) != 0.0:
            raise ValueError("a scaled need belongs to saved arousal format cadence-arousal/3")
        if "need_scale" not in config and set(config) == genes - {"need_scale"}:
            config["need_scale"] = 0.0
    if set(config) != genes:
        raise ValueError("invalid or incomplete saved arousal config")
    weight = config.pop("record_surprise")
    if isinstance(weight, bool) or weight != 0.0:
        raise ValueError(
            "this arousal was saved with record_surprise above zero, which this version "
            "removed; load it with cadence-net 0.84.3"
        )
    records, usual = values["records"], values["usual_record"]
    if isinstance(records, bool) or not isinstance(records, int) or records < 0:
        raise ValueError("invalid saved arousal count: records")
    if (
        isinstance(usual, bool)
        or not isinstance(usual, (int, float))
        or not np.isfinite(usual)
        or usual < 0
    ):
        raise ValueError("invalid saved arousal value: usual_record")
    rewards = (
        values["rewards"]
        if values.get("format") in ("cadence-arousal/2", "cadence-arousal/3")
        else values.get("rewards", values["outcomes"])
    )
    state = {k: v for k, v in values.items() if k not in ("records", "usual_record")}
    return {**state, "format": _FORMAT, "config": config, "rewards": rewards}
