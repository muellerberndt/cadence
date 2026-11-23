"""The forager chamber's world and its teacher: bodies on a walled ground with plants and rocks.

Each body lives in a world of its own: a square ground of ``GROUND`` metres with walls, ``PLANTS``
plants and ``ROCKS`` rocks. A body walks, turns left or right, bites or rests. A bite eats a plant
within ``REACH`` that lies within ``CONE`` of its heading, and the plant grows again elsewhere.

Its senses, 104 inputs in the layout an animal of a larger simulation reads (``SENSES``):

- **eyes** (32): eight sectors of 45 degrees around the body, four channels each: how near the
  nearest thing in the sector is, and how near the nearest plant, rock and wall is, each as
  ``1 - distance / SIGHT`` (zero past ``SIGHT``).
- **ears** (8) and **heard** (16): the sounds of other animals, which the forager's task does
  not use: a sound of one of four pitches from the left or the right at a moment in
  ``1 / SOUND``, and a call of up to four notes of four pitches at a moment in ``1 / CALL``.
- **thing** (20): the nearest rock in sight, as the thing a body attends: its kind (4), its hue
  (4, none for a rock), its bearing (eight tuned cells around the body) and its distance (four
  tuned cells on a log scale).
- **food** (20): the nearest plant in sight, as three fields: ``what`` (its kind and its hue as
  tuned cells, 8), ``bearing`` (8) and ``distance`` (4), tuned as the thing's; silent when no
  plant is in sight. A rock hides the plants behind it, as far as a body walking there would
  meet it.
- **body** (8): its energy as three tuned cells and the action it took last, one of five.

The teacher (``teach``) turns toward the nearest plant in sight, walks to it and bites it; with no
plant in sight it walks on and turns away from what blocks the way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

GROUND = 12.0  # metres a side, walled
PLANTS = 10
ROCKS = 5
SIGHT = 4.0  # how far the eyes see
REACH = 0.7  # a bite reaches a plant this near (centre to centre)
CONE = 0.45  # radians off the heading within which a bite reaches
STRIDE = 0.3  # metres a walk moves
TURNING = 0.4  # radians a turn turns
BODY = 0.3  # a body's radius: a rock or a wall nearer than this stops a walk
ROCK = 0.4  # a rock's radius
TURN = 0.3  # the teacher turns where the plant lies further than this off its heading
BLOCKED = 1.0  # the teacher turns from a rock or a wall nearer than this ahead
UPKEEP = 0.004  # energy a moment
MEAL = 0.25  # energy a plant gives
SOUND = 3  # a sound reaches the ears at one moment in this many
CALL = 6  # a call is heard at one moment in this many

ACTIONS = ("rest", "walk", "left", "right", "bite")
REST, WALK, LEFT, RIGHT, BITE = range(5)
SECTORS = 8
CHANNELS = ("near", "plant", "rock", "wall")
HUES = 4
PITCHES = NOTES = 4
DISTANCES = np.array([0.5, 1.2, 2.4, 4.0])
KINDS = 4  # plant, animal, meat, rock: the kinds a larger world holds
PLANT, ROCK_KIND = 0, 3
SENSES: dict[str, Any] = {
    "eyes": len(CHANNELS) * SECTORS,
    "ears": 2 * PITCHES,
    "heard": NOTES * PITCHES,
    "thing": KINDS + HUES + SECTORS + len(DISTANCES),
    "food": {"what": KINDS + HUES, "bearing": SECTORS, "distance": len(DISTANCES)},
    "body": 3 + len(ACTIONS),
}
PLANT_HUE = 0.33  # green


def wrap(angle: Any) -> Any:
    """An angle in [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def tuned(value: np.ndarray, centres: np.ndarray, width: float, *, circular: bool = False
          ) -> np.ndarray:
    """``value`` (rows) as Gaussian tuned cells around ``centres``."""
    d = value[:, None] - centres[None, :]
    if circular:
        d = wrap(d)
    return np.asarray(np.exp(-0.5 * (d / width) ** 2))


class Worlds:
    """``bodies`` worlds of one body each, drawn from ``seed``."""

    def __init__(self, bodies: int, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.sounds = np.random.default_rng(seed + 1)  # the other animals' sounds and calls
        self.n = bodies
        self.rocks = self._free((bodies, ROCKS), margin=1.0)
        self.plants = self._free((bodies, PLANTS), margin=0.6, away=ROCK + BODY + REACH)
        self.pose = np.zeros((bodies, 3))
        self.pose[:, :2] = self._free((bodies, 1), margin=1.0, away=ROCK + BODY + 0.2)[:, 0]
        self.pose[:, 2] = self.rng.uniform(-np.pi, np.pi, bodies)
        self.energy = np.full(bodies, 0.5)
        self.did = np.full(bodies, REST)
        self.eaten = np.zeros(bodies, dtype=int)

    def _free(self, shape: tuple[int, int], margin: float, away: float = 0.0,
              bodies: np.ndarray | None = None) -> np.ndarray:
        """Places on the ground ``margin`` from its walls and, once the rocks stand, ``away`` from
        each rock's centre (for ``bodies``, every body by default)."""
        out = self.rng.uniform(margin, GROUND - margin, (*shape, 2))
        rocks = getattr(self, "rocks", None)
        if rocks is not None:
            rocks = rocks if bodies is None else rocks[bodies]
            for _ in range(50):
                near = np.linalg.norm(out[:, :, None, :] - rocks[:, None, :, :], axis=-1).min(-1)
                bad = near < away
                if not bad.any():
                    break
                out[bad] = self.rng.uniform(margin, GROUND - margin, (int(bad.sum()), 2))
        return out

    # -- what a body senses

    def _relative(self, places: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance and bearing (off the heading, left positive) of ``places`` (bodies, k, 2)."""
        d = places - self.pose[:, None, :2]
        return np.hypot(d[..., 0], d[..., 1]), wrap(np.arctan2(d[..., 1], d[..., 0])
                                                    - self.pose[:, None, 2])

    def _walls(self) -> np.ndarray:
        """How far each sector's centre ray runs to a wall, (bodies, SECTORS)."""
        angles = self.pose[:, 2:3] + np.arange(SECTORS)[None, :] * 2.0 * np.pi / SECTORS
        c, s = np.cos(angles), np.sin(angles)
        x, y = self.pose[:, 0:1], self.pose[:, 1:2]
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(c > 1e-9, (GROUND - x) / c, np.where(c < -1e-9, -x / c, np.inf))
            ty = np.where(s > 1e-9, (GROUND - y) / s, np.where(s < -1e-9, -y / s, np.inf))
        return np.asarray(np.minimum(tx, ty))

    @staticmethod
    def _sector(bearing: np.ndarray) -> np.ndarray:
        return np.asarray(np.round(bearing / (2.0 * np.pi / SECTORS)).astype(int) % SECTORS)

    def _nearest_by_sector(self, distance: np.ndarray, bearing: np.ndarray) -> np.ndarray:
        """Each sector's nearest distance among things at ``distance``/``bearing``."""
        out = np.full((self.n, SECTORS), np.inf)
        sector = self._sector(bearing)
        rows = np.repeat(np.arange(self.n)[:, None], distance.shape[1], axis=1)
        np.minimum.at(out, (rows.ravel(), sector.ravel()), distance.ravel())
        return out

    def food(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The nearest plant in sight, a rock hiding the plants behind it: whether there is one,
        its distance and its bearing."""
        return self._nearest(self.plants, 0.0, hidden=self._hidden(self.plants))

    def _hidden(self, places: np.ndarray) -> np.ndarray:
        """Whether a rock stands between each body and each of ``places`` (bodies, k): in the
        way of a body walking there."""
        start = self.pose[:, None, None, :2]
        line = places[:, :, None, :] - start  # body to place
        rock = self.rocks[:, None, :, :] - start  # body to rock
        length = np.maximum(np.linalg.norm(line, axis=-1), 1e-9)
        along = (rock * line).sum(-1) / length
        aside = np.abs(rock[..., 0] * line[..., 1] - rock[..., 1] * line[..., 0]) / length
        return np.asarray(((along > 0) & (along < length) & (aside < ROCK + BODY)).any(-1))

    def _nearest(self, places: np.ndarray, radius: float, hidden: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distance, bearing = self._relative(places)
        distance = np.maximum(distance - radius, 0.0)
        if hidden is not None:
            distance = np.where(hidden, np.inf, distance)
        k = distance.argmin(axis=1)
        rows = np.arange(self.n)
        d, b = distance[rows, k], bearing[rows, k]
        return d <= SIGHT, np.where(np.isfinite(d), d, 2.0 * SIGHT), b

    def _where(self, seen: np.ndarray, d: np.ndarray, b: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        """A bearing as eight tuned cells and a distance as four, zero where nothing is seen."""
        mask = seen[:, None].astype(float)
        bearing = tuned(b, wrap(np.arange(SECTORS) * 2.0 * np.pi / SECTORS), 0.5, circular=True)
        distance = np.exp(-0.5 * (np.log(np.maximum(d, 0.05)[:, None] / DISTANCES) / 0.6) ** 2)
        return bearing * mask, distance * mask

    def senses(self) -> dict[str, Any]:
        """What every body senses, one row a body, by sense and field."""
        near = lambda d: np.clip(1.0 - d / SIGHT, 0.0, 1.0)  # noqa: E731
        plant = near(self._nearest_by_sector(*self._relative(self.plants)))
        rock = near(np.maximum(self._nearest_by_sector(*self._relative(self.rocks)) - ROCK, 0.0))
        wall = near(self._walls())
        eyes = np.hstack([np.maximum(np.maximum(plant, rock), wall), plant, rock, wall])
        seen, d, b = self.food()
        what = np.zeros((self.n, KINDS + HUES))
        what[:, PLANT] = 1.0
        what[:, KINDS:] = _hue_cells(self.n)
        bearing, distance = self._where(seen, d, b)
        rock_seen, rock_d, rock_b = self._nearest(self.rocks, ROCK)
        thing = np.zeros((self.n, KINDS + HUES))
        thing[:, ROCK_KIND] = rock_seen  # a rock has no hue
        energy = tuned(self.energy, np.array([0.0, 0.5, 1.0]), 0.25)
        ears = np.zeros((self.n, 2 * PITCHES))
        loud = self.sounds.integers(0, SOUND, self.n) == 0
        side, pitch = self.sounds.integers(0, 2, self.n), self.sounds.integers(0, PITCHES, self.n)
        ears[loud, side[loud] * PITCHES + pitch[loud]] = self.sounds.uniform(0.3, 1.0, loud.sum())
        heard = np.zeros((self.n, NOTES * PITCHES))
        for i in np.flatnonzero(self.sounds.integers(0, CALL, self.n) == 0):
            notes = self.sounds.integers(0, PITCHES, self.sounds.integers(1, NOTES + 1))
            heard[i, np.arange(len(notes)) * PITCHES + notes] = 1.0
        return {
            "eyes": eyes,
            "ears": ears,
            "heard": heard,
            "thing": np.hstack([thing, *self._where(rock_seen, rock_d, rock_b)]),
            "food": {"what": what * seen[:, None], "bearing": bearing, "distance": distance},
            "body": np.hstack([energy, np.eye(len(ACTIONS))[self.did]]),
        }


    # -- what a body does

    def _ahead(self) -> np.ndarray:
        heading = self.pose[:, 2]
        return self.pose[:, :2] + STRIDE * np.stack([np.cos(heading), np.sin(heading)], 1)

    def clear(self) -> np.ndarray:
        """Whether a walk would move each body: no wall and no rock in its way."""
        ahead = self._ahead()
        clear = ((ahead > BODY) & (ahead < GROUND - BODY)).all(1)
        return np.asarray(
            clear & (np.linalg.norm(self.rocks - ahead[:, None], axis=-1) > ROCK + BODY).all(1))

    def step(self, actions: np.ndarray) -> np.ndarray:
        """Every body takes its action; returns the plants each ate (0 or 1)."""
        actions = np.asarray(actions, dtype=int).reshape(self.n)
        ate = np.zeros(self.n, dtype=int)
        heading = self.pose[:, 2]
        heading[actions == LEFT] += TURNING
        heading[actions == RIGHT] -= TURNING
        self.pose[:, 2] = wrap(heading)
        walk = actions == WALK
        ahead = self._ahead()
        moved = walk & self.clear()
        self.pose[moved, :2] = ahead[moved]
        distance, bearing = self._relative(self.plants)
        for i in np.flatnonzero(actions == BITE):
            reach = np.flatnonzero((distance[i] < REACH) & (np.abs(bearing[i]) < CONE))
            if len(reach):
                k = reach[distance[i, reach].argmin()]
                self.plants[i, k] = self._free((1, 1), 0.6, ROCK + BODY + REACH,
                                               bodies=np.array([i]))[0, 0]
                ate[i] = 1
        self.energy = np.clip(self.energy - UPKEEP + MEAL * ate, 0.0, 1.0)
        self.eaten += ate
        self.did = actions.copy()
        return ate


def _hue_cells(n: int) -> np.ndarray:
    """A plant's hue as four tuned cells around the colour circle, one row a body."""
    centres = np.arange(HUES) / HUES
    d = np.abs(PLANT_HUE - centres) % 1.0
    return np.tile(np.exp(-0.5 * (np.minimum(d, 1.0 - d) / 0.14) ** 2), (n, 1))


def flat(sensed: dict[str, Any]) -> np.ndarray:
    """The senses as one row a body, in the declared order (``SENSES``)."""
    food = sensed["food"]
    return np.hstack([sensed["eyes"], sensed["ears"], sensed["heard"], sensed["thing"],
                      food["what"], food["bearing"], food["distance"], sensed["body"]])


def teach(world: Worlds) -> np.ndarray:
    """The teacher's action for every body: bite a plant in reach; turn toward the nearest plant
    in sight, then walk to it; with none in sight walk on, turning from a rock or a wall ahead
    toward the freer side, and turn that way too where a walk would not move."""
    out = np.full(world.n, WALK, dtype=int)
    distance, bearing = world._relative(world.plants)
    seen, d, b = world.food()
    for i in range(world.n):
        if ((distance[i] < REACH) & (np.abs(bearing[i]) < CONE - 0.08)).any():
            out[i] = BITE
        elif seen[i]:
            out[i] = LEFT if b[i] > TURN else RIGHT if b[i] < -TURN else WALK
    rock_d, rock_b = world._relative(world.rocks)
    rocks = world._nearest_by_sector(np.maximum(rock_d - ROCK, 0.0), rock_b)
    solid = np.minimum(rocks, world._walls())  # each sector's nearest rock or wall
    blocked = (~seen & (solid[:, 0] < BLOCKED)) | ((out == WALK) & ~world.clear())
    left = solid[:, 1:3].min(1)  # ahead left and left
    right = solid[:, 6:8].min(1)  # right and ahead right
    out[blocked] = np.where(left[blocked] >= right[blocked], LEFT, RIGHT)
    return out
