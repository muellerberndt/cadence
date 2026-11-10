"""The Doom body: a VizDoom adapter, its sensors, actions, scorer and custody (cadence#285).

The body is the boundary between the game and a player. A player receives an ``Observation``:
the rendered screen, the declared own-body ports (health, armour, ammunition) and the declared
history port (the previous action of the episode), and nothing else. The engine's other
variables (kills, items, the tic) stay on this side: the scorer reads them to credit outcomes,
and custody records them per moment.

VizDoom is an optional dependency (``pip install vizdoom``; 1.3.2 was read). Its wheel carries
the free Freedoom IWADs and the scenario files, which this module reads from the installed
package; ``engine_identity`` names the version and the files' SHA-256 digests. The engine is
imported when a body is built, so this module imports without it.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# the declared button set of the body: move, turn, strafe, attack, use
BUTTONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "ATTACK",
    "USE",
)

# the declared own-body ports and the scale each is read at
PORTS = ("health", "armour", "ammunition")
PORT_SCALES = {"health": 100.0, "armour": 100.0, "ammunition": 50.0}

# the declared screens, name -> (height, width): 160 by 120 is the engine's own render, 84 by 84
# its area-weighted resampling
RESOLUTIONS = {"84x84": (84, 84), "160x120": (120, 160)}

OUTCOMES = ("completion", "death", "deadline", "truncation")

# the engine variables the body reads at every tic: the three ports, then the scorer's own,
# which never reach a player
_PORT_VARIABLES = ("HEALTH", "ARMOR", "SELECTED_WEAPON_AMMO")
_HIDDEN_VARIABLES = ("KILLCOUNT", "ITEMCOUNT")
VARIABLES = _PORT_VARIABLES + _HIDDEN_VARIABLES


@dataclass(frozen=True)
class Scenario:
    """One declared scenario: its map, goal, length, buttons and scorer.

    ``goal`` is ``"kill"``: the episode completes when the engine ends it with the player alive
    (the target is dead), and reaching ``length`` tics first is its deadline. Or ``"survive"``:
    the episode completes when the player is alive at ``length`` tics, and has no deadline.
    Rewards are per tic, per kill, and once per outcome."""

    name: str
    goal: str
    length: int
    buttons: tuple[str, ...]
    tic_reward: float = 0.0
    kill_reward: float = 0.0
    completion_reward: float = 0.0
    death_reward: float = 0.0
    deadline_reward: float = 0.0

    def outcome_reward(self, outcome: str) -> float:
        return {
            "completion": self.completion_reward,
            "death": self.death_reward,
            "deadline": self.deadline_reward,
            "truncation": 0.0,
        }[outcome]


# maps, buttons and lengths are those of VizDoom's own configuration files; the rewards are this
# body's scorer, read from engine facts (the engine's scripted rewards are not used)
SCENARIOS = {
    "basic": Scenario(
        "basic", "kill", 300, ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
        tic_reward=-0.0025, completion_reward=1.0,
    ),
    "defend_the_center": Scenario(
        "defend_the_center", "survive", 2100, ("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
        kill_reward=1.0, death_reward=-1.0,
    ),
    "health_gathering": Scenario(
        "health_gathering", "survive", 2100, ("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
        tic_reward=0.0025, death_reward=-1.0,
    ),
}


@dataclass(frozen=True)
class Observation:
    """What a player receives: the rendered screen, the own-body ports and the history port.

    ``screen`` is ``(height, width, 3)`` uint8 RGB. ``body`` holds the ports in ``PORTS`` order,
    each divided by its ``PORT_SCALES`` entry (no weapon reads 0 ammunition). ``history`` is the
    previous action of this episode, one-hot over the scenario's buttons, zeros at the
    episode's first moment."""

    screen: np.ndarray
    body: np.ndarray
    history: np.ndarray


@dataclass(frozen=True)
class Transition:
    """The world's answer to one action: the reward, how the episode ended if it did, the events
    and the tic after the moment. ``observation`` is the next observation of the episode, the
    frame at the end after a deadline, a survival's completion or a truncation, and ``None``
    when the engine ended the episode (a death, a kill goal's completion)."""

    observation: Observation | None
    reward: float
    outcome: str | None
    tic: int
    tics: int
    events: tuple[str, ...]

    @property
    def done(self) -> bool:
        return self.outcome is not None


def engine_identity() -> dict[str, Any]:
    """The engine's version and the SHA-256 digests of the IWAD and the scenario files."""
    import vizdoom as vzd

    root = Path(vzd.__file__).parent
    files = [root / "freedoom2.wad"] + [
        Path(vzd.scenarios_path) / f"{name}.{kind}"
        for name in sorted(SCENARIOS)
        for kind in ("cfg", "wad")
    ]
    return {
        "vizdoom": vzd.__version__,
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }


def resample(screen: np.ndarray, height: int, width: int) -> np.ndarray:
    """Area-weighted resampling of an ``(h, w, c)`` image to ``(height, width, c)``, uint8."""
    h, w = screen.shape[:2]
    if (h, w) == (height, width):
        return np.array(screen, dtype=np.uint8)
    rows, cols = _weights(h, height), _weights(w, width)
    image = screen.astype(np.float64)
    out = np.tensordot(cols, np.tensordot(rows, image, axes=(1, 0)), axes=(1, 1))  # (j, i, c)
    out = out.transpose(1, 0, 2)
    return np.asarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def _weights(n_in: int, n_out: int) -> np.ndarray:
    # each output cell averages the input interval it covers, partial pixels weighted
    edges = np.arange(n_out + 1) * (n_in / n_out)
    lo, hi = edges[:-1, None], edges[1:, None]
    pixels = np.arange(n_in)[None, :]
    overlap = np.clip(np.minimum(hi, pixels + 1) - np.maximum(lo, pixels), 0.0, None)
    return np.asarray(overlap / overlap.sum(axis=1, keepdims=True))


def _new_game(scenario: Scenario) -> Any:
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, f"{scenario.name}.cfg"))
    game.set_doom_game_path(os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_episode_timeout(0)  # the body keeps the length in its own tics
    game.set_living_reward(0.0)
    game.set_death_penalty(0.0)
    game.clear_available_buttons()
    for name in scenario.buttons:
        game.add_available_button(getattr(vzd.Button, name))
    game.clear_available_game_variables()
    for name in VARIABLES:
        game.add_available_game_variable(getattr(vzd.GameVariable, name))
    return game


@dataclass
class _Episode:
    index: int = -1
    moment: int = 0
    tic: int = 0
    variables: list[float] = field(default_factory=list)
    previous: int = -1
    returned: float = 0.0
    outcome: str | None = None
    screen: np.ndarray | None = None


class DoomBody:
    """A VizDoom scenario behind the body's declared ports, actions and scorer.

    ``scenario`` names an entry of ``SCENARIOS`` or is a ``Scenario`` whose ``name`` is a
    VizDoom scenario. ``resolution`` is ``"84x84"`` or ``"160x120"``. An action holds one button
    of the scenario for ``tics`` game tics. Episode ``e`` runs with the engine's seed
    ``seed + e``. ``max_moments`` ends an episode by truncation after that many moments
    (unset: never)."""

    def __init__(
        self,
        scenario: str | Scenario,
        *,
        resolution: str = "84x84",
        tics: int = 4,
        seed: int = 0,
        max_moments: int | None = None,
    ) -> None:
        if isinstance(scenario, str):
            if scenario not in SCENARIOS:
                raise ValueError(f"unknown scenario {scenario!r}; declared: {sorted(SCENARIOS)}")
            scenario = SCENARIOS[scenario]
        if scenario.goal not in ("kill", "survive"):
            raise ValueError("a scenario's goal is 'kill' or 'survive'")
        unknown = [b for b in scenario.buttons if b not in BUTTONS]
        if unknown or not scenario.buttons:
            raise ValueError(f"buttons outside the declared set: {unknown}")
        if resolution not in RESOLUTIONS:
            raise ValueError(f"unknown resolution {resolution!r}; declared: {sorted(RESOLUTIONS)}")
        for label, value in (("tics", tics), ("seed", seed)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{label} must be an integer")
        if tics < 1 or seed < 0:
            raise ValueError("tics must be positive and seed nonnegative")
        if max_moments is not None and (
            isinstance(max_moments, bool)
            or not isinstance(max_moments, (int, np.integer))
            or max_moments < 1
        ):
            raise ValueError("max_moments must be a positive integer or None")
        self.scenario = scenario
        self.resolution = resolution
        self.shape = RESOLUTIONS[resolution]
        self.tics = int(tics)
        self.seed = int(seed)
        self.max_moments = None if max_moments is None else int(max_moments)
        self.actions = len(scenario.buttons)
        self.record: list[dict[str, Any]] = []
        self._episode = _Episode()
        self._game = _new_game(scenario)
        self._game.init()

    # ------------------------------------------------------------------ sensors
    def _read(self) -> list[float]:
        import vizdoom as vzd

        return [
            float(self._game.get_game_variable(getattr(vzd.GameVariable, name)))
            for name in VARIABLES
        ]

    def _grab(self) -> np.ndarray:
        state = self._game.get_state()
        if state is None or state.screen_buffer is None:
            raise RuntimeError("the engine rendered no screen")
        return resample(np.asarray(state.screen_buffer), *self.shape)

    def _observe(self) -> Observation:
        episode = self._episode
        assert episode.screen is not None
        health, armour, ammunition = episode.variables[: len(_PORT_VARIABLES)]
        raw = (health, armour, max(ammunition, 0.0))
        body = np.array([v / PORT_SCALES[p] for p, v in zip(PORTS, raw, strict=True)])
        history = np.zeros(self.actions)
        if episode.previous >= 0:
            history[episode.previous] = 1.0
        return Observation(screen=episode.screen.copy(), body=body, history=history)

    @property
    def tic(self) -> int:
        """The game tic of the running (or last) episode, counted as the engine counts it."""
        return self._episode.tic

    @property
    def episode(self) -> int:
        """The running (or last) episode's index, -1 before the first."""
        return self._episode.index

    @property
    def running(self) -> bool:
        return self._episode.index >= 0 and self._episode.outcome is None

    @property
    def returned(self) -> float:
        """The running (or last) episode's summed reward."""
        return self._episode.returned

    def observe(self) -> Observation:
        """The running episode's present observation, as the last ``reset`` or ``step`` gave it."""
        if not self.running:
            raise RuntimeError("no episode is running")
        return self._observe()

    # ------------------------------------------------------------------ episodes
    def reset(self) -> Observation:
        """Starts the next episode and returns its first observation. Refused while an episode
        runs: ``truncate`` it first."""
        if self.running:
            raise RuntimeError("an episode is running; truncate it before a reset")
        index = self._episode.index + 1
        self._game.set_seed(self.seed + index)
        self._game.new_episode()
        self._episode = _Episode(
            index=index,
            tic=int(self._game.get_episode_time()),
            variables=self._read(),
            screen=self._grab(),
        )
        self.record.append(
            {"episode": index, "moment": 0, "kind": "start", "tic": self._episode.tic,
             "seed": self.seed + index}
        )
        return self._observe()

    def _check(self, action: Any) -> int:
        if not self.running:
            raise RuntimeError("no episode is running: reset first")
        if isinstance(action, np.ndarray) and action.shape == ():
            action = action.item()
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("an action is one integer index into the scenario's buttons")
        if not 0 <= int(action) < self.actions:
            raise ValueError(f"action {action} is outside the {self.actions} declared buttons")
        return int(action)

    def step(self, action: Any) -> Transition:
        """Holds button ``action`` for the declared tics and scores the moment. A refused action
        (anything but an integer index of a declared button, or no running episode) raises
        before the world advances."""
        action = self._check(action)
        episode, scenario, game = self._episode, self.scenario, self._game
        buttons = [0.0] * self.actions
        buttons[action] = 1.0
        before, tic_before = list(episode.variables), episode.tic
        variables = list(before)
        outcome: str | None = None
        advanced = 0
        # the moment's tics, the last of them at the scenario's length at most; the engine's
        # variables are read after every tic (they update with its state)
        tics = min(self.tics, scenario.length - episode.tic)
        game.set_action(buttons)
        for _ in range(tics):
            game.advance_action(1, True)
            advanced += 1
            if game.is_episode_finished():
                if game.is_player_dead():
                    outcome = "death"
                elif scenario.goal == "kill":
                    outcome = "completion"
                else:
                    raise RuntimeError("the engine ended a survival episode with the player alive")
                break
            variables = self._read()
        if outcome is None and episode.tic + advanced >= scenario.length:
            outcome = "completion" if scenario.goal == "survive" else "deadline"
        episode.tic += advanced
        episode.moment += 1
        if outcome is None and self.max_moments is not None and episode.moment >= self.max_moments:
            outcome = "truncation"
        kills = max(int(round(variables[3] - before[3])), 0)
        events = _events(before, variables, kills, outcome)
        reward = scenario.tic_reward * advanced + scenario.kill_reward * kills
        if outcome is not None:
            reward += scenario.outcome_reward(outcome)
        episode.variables = variables
        episode.previous = action
        episode.returned += reward
        episode.outcome = outcome
        observation = None
        if not game.is_episode_finished():
            episode.screen = self._grab()
            observation = self._observe()
        self.record.append(
            {
                "episode": episode.index, "moment": episode.moment, "kind": "moment",
                "action": action, "tic_before": tic_before, "tic": episode.tic,
                "tics": advanced, "reward": reward, "events": list(events), "outcome": outcome,
                "variables": dict(zip(VARIABLES, variables, strict=True)),
            }
        )
        return Transition(observation, float(reward), outcome, episode.tic, advanced, events)

    def truncate(self) -> Transition:
        """Ends the running episode by truncation, no tic advanced, with the frame at the cut."""
        if not self.running:
            raise RuntimeError("no episode is running")
        episode = self._episode
        episode.outcome = "truncation"
        self.record.append(
            {"episode": episode.index, "moment": episode.moment, "kind": "truncate",
             "tic": episode.tic, "reward": 0.0, "events": ["truncation"],
             "outcome": "truncation"}
        )
        return Transition(self._observe(), 0.0, "truncation", episode.tic, 0, ("truncation",))

    # ------------------------------------------------------------------ custody
    def checkpoint(self) -> dict[str, Any]:
        """The game and the body as one JSON-safe mapping, for ``DoomBody.restore``.

        The engine writes its save game within two idle tics. Both lives then continue from the
        saved world: this body loads the save it wrote, so the original and the resumed life
        continue from one state at one tic. Custody records the two tics as a ``checkpoint``
        entry, and what happens in them is scored with the next moment."""
        if not self.running:
            raise RuntimeError("a checkpoint is taken while an episode runs")
        game = self._game
        directory = tempfile.mkdtemp(prefix="doom")  # the engine ignores a long save path
        path = os.path.join(directory, "body.sav")
        try:
            game.set_action([0.0] * self.actions)
            game.save(path)
            if not os.path.exists(path):
                raise RuntimeError("the engine wrote no save game")
            saved = Path(path).read_bytes()
            game.load(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
            os.rmdir(directory)
        if game.is_episode_finished():
            raise RuntimeError("the episode ended within the checkpoint's tics")
        episode = self._episode
        tic = int(game.get_episode_time())
        self.record.append(
            {"episode": episode.index, "moment": episode.moment, "kind": "checkpoint",
             "tic_before": episode.tic, "tic": tic, "tics": tic - episode.tic}
        )
        episode.tic = tic
        episode.screen = self._grab()
        state = {
            "scenario": dataclasses.asdict(self.scenario),
            "resolution": self.resolution,
            "tics": self.tics,
            "seed": self.seed,
            "max_moments": self.max_moments,
            "episode": {
                "index": episode.index, "moment": episode.moment, "tic": episode.tic,
                "variables": episode.variables, "previous": episode.previous,
                "returned": episode.returned,
            },
            "record": self.record,
            "engine": engine_identity(),
        }
        checkpoint: dict[str, Any] = json.loads(
            json.dumps({"body": state, "game": base64.b64encode(saved).decode()})
        )
        return checkpoint

    @classmethod
    def restore(cls, checkpoint: dict[str, Any]) -> DoomBody:
        """A body continuing the checkpointed life: the same world at the same tic."""
        state = checkpoint["body"]
        if state["engine"] != engine_identity():
            raise ValueError("the checkpoint was written by another engine or other game files")
        spec = dict(state["scenario"])
        spec["buttons"] = tuple(spec["buttons"])
        body = cls(
            Scenario(**spec), resolution=state["resolution"], tics=state["tics"],
            seed=state["seed"], max_moments=state["max_moments"],
        )
        saved = state["episode"]
        body._game.set_seed(body.seed + saved["index"])
        body._game.new_episode()
        directory = tempfile.mkdtemp(prefix="doom")
        path = os.path.join(directory, "body.sav")
        try:
            Path(path).write_bytes(base64.b64decode(checkpoint["game"]))
            body._game.load(path)
        finally:
            os.remove(path)
            os.rmdir(directory)
        tic = int(body._game.get_episode_time())
        if tic != saved["tic"]:
            raise RuntimeError(f"the restored game is at tic {tic}, the save at {saved['tic']}")
        body._episode = _Episode(
            index=saved["index"], moment=saved["moment"], tic=tic,
            variables=[float(v) for v in saved["variables"]], previous=int(saved["previous"]),
            returned=float(saved["returned"]), screen=body._grab(),
        )
        body.record = json.loads(json.dumps(state["record"]))
        return body

    def close(self) -> None:
        self._game.close()

    def __enter__(self) -> DoomBody:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _events(
    before: Sequence[float], after: Sequence[float], kills: int, outcome: str | None
) -> tuple[str, ...]:
    """The moment's events from the engine variables before and after it."""
    events = ["kill"] * kills
    for down, up, i in (("damage", "heal", 0), ("armour_lost", "armour", 1),
                        ("shot", "ammunition", 2)):
        if after[i] < before[i]:
            events.append(down)
        elif after[i] > before[i]:
            events.append(up)
    if after[4] > before[4]:
        events.append("pickup")
    if outcome is not None:
        events.append(outcome)
    return tuple(events)
