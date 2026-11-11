"""The Doom body's contracts (cadence#285).

The engine's tests need VizDoom (``pip install vizdoom``) and skip without it; the declarations,
the players' reading of an observation and the life's refusal rule are read everywhere.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import doom_body  # noqa: E402
import doom_life  # noqa: E402
import doom_players  # noqa: E402

engine = pytest.mark.skipif(
    importlib.util.find_spec("vizdoom") is None, reason="VizDoom is not installed"
)
torch_present = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="PyTorch is not installed"
)


def observation(resolution: str = "84x84", actions: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    h, w = doom_body.RESOLUTIONS[resolution]
    return doom_body.Observation(
        screen=rng.integers(0, 256, (h, w, 3)).astype(np.uint8),
        body=np.array([1.0, 0.0, 0.5]),
        history=np.eye(actions)[1],
    )


# ---------------------------------------------------------------- declarations, everywhere


def test_the_declarations():
    assert doom_body.PORTS == ("health", "armour", "ammunition")
    assert len(set(doom_body.OUTCOMES)) == 4
    for name, scenario in doom_body.SCENARIOS.items():
        assert scenario.name == name and scenario.goal in ("kill", "survive")
        assert set(scenario.buttons) <= set(doom_body.BUTTONS)
        assert scenario.outcome_reward("truncation") == 0.0
    for wanted in ("MOVE_FORWARD", "TURN_LEFT", "MOVE_LEFT", "ATTACK", "USE"):
        assert wanted in doom_body.BUTTONS


def test_resampling_keeps_a_flat_screen_and_its_shape():
    flat = np.full((120, 160, 3), 77, np.uint8)
    for h, w in doom_body.RESOLUTIONS.values():
        out = doom_body.resample(flat, h, w)
        assert out.shape == (h, w, 3) and out.dtype == np.uint8 and (out == 77).all()
    weights = doom_body._weights(160, 84)
    assert np.allclose(weights.sum(axis=1), 1.0) and weights.shape == (84, 160)


def test_an_observation_holds_the_declared_ports_and_nothing_else():
    fields = [f.name for f in dataclasses.fields(doom_body.Observation)]
    assert fields == ["screen", "body", "history"]
    # the players read no part of the body but the observation and the transition
    source = (HERE / "doom_players.py").read_text()
    assert not re.search(r"\.(_game|_episode|record|tic|variables|_read)\b", source)
    for hidden in ("VARIABLES", "KILLCOUNT", "ITEMCOUNT", "get_state", "DoomBody"):
        assert hidden not in source, hidden


@pytest.mark.parametrize("resolution", sorted(doom_body.RESOLUTIONS))
def test_the_brain_reads_the_retina_and_the_ports(resolution):
    player = doom_players.BrainPlayer(resolution, 3, seed=0, backend="cpu")
    o = observation(resolution)
    x = player.encode(o)
    pool = doom_players.POOL[resolution]
    retina = doom_players.retina(o.screen, pool)
    assert x.shape == (1, doom_players.inputs(resolution, 3))
    assert np.array_equal(x[0], np.concatenate([retina, o.body, o.history]))
    blind = doom_players.BrainPlayer(resolution, 3, seed=0, mode="blind", backend="cpu")
    assert not blind.encode(o)[0, : retina.size].any()
    assert np.array_equal(blind.encode(o)[0, retina.size:], x[0, retina.size:])
    assert 0 <= player.act(o, None) < 3


class _World:
    """A world that counts tics: the life's refusal rule without the engine."""

    actions, resolution = 3, "84x84"

    def __init__(self) -> None:
        self.tic, self.episode, self.returned, self.steps = 0, 0, 0.0, 0

    def reset(self):
        return observation()

    def step(self, action):
        self.tic += 4
        self.steps += 1
        return doom_body.Transition(observation(), 0.0, None, self.tic, 4, ())


class _Refusing:
    actions = 3

    def __init__(self, after: int) -> None:
        self.after, self.moments = after, 0

    def act(self, o, previous):
        if self.moments == self.after:
            raise RuntimeError("the settle did not qualify")
        self.moments += 1
        return 0

    def finish(self, previous):
        return None


def test_a_refused_moment_leaves_the_world_where_it_stands():
    world = _World()
    with pytest.raises(doom_life.Refused) as refused:
        doom_life.play(world, _Refusing(after=3), episodes=1)
    assert world.steps == 3 and world.tic == 12 and refused.value.tic == 12


# ---------------------------------------------------------------- the engine


@engine
def test_refused_actions_do_not_advance_the_world():
    with doom_body.DoomBody("basic", seed=1) as body:
        first = body.reset()
        for _ in range(3):
            body.step(0)
        tic, moments, screen = body.tic, len(body.record), body.observe().screen
        engine_tic = body._game.get_episode_time()
        for bad in (-1, 3, 1.0, True, np.bool_(True), "0", None, np.array([0, 1])):
            with pytest.raises(ValueError):
                body.step(bad)
        assert body.tic == tic and len(body.record) == moments
        assert body._game.get_episode_time() == engine_tic
        assert np.array_equal(body.observe().screen, screen)
        with pytest.raises(RuntimeError):
            body.reset()  # an episode runs
        transition = body.truncate()
        assert transition.outcome == "truncation" and transition.tics == 0
        with pytest.raises(RuntimeError):
            body.step(0)  # no episode runs
        assert body.tic == tic and body._game.get_episode_time() == engine_tic
        assert first.screen.shape == (84, 84, 3) and first.screen.dtype == np.uint8


def _life(body, policy, limit=10_000):
    """One episode under ``policy(moment)``; the transitions."""
    body.reset()
    out = []
    for moment in range(limit):
        out.append(body.step(policy(moment)))
        if out[-1].done:
            return out
    raise AssertionError("the episode did not end")


_DRAWS = np.random.default_rng(0).integers(3, size=10_000)


@engine
@pytest.mark.parametrize(
    ("scenario", "options", "policy", "wanted"),
    [
        ("basic", {}, lambda m: 0, "deadline"),  # strafes, never shoots
        ("basic", {}, lambda m: int(_DRAWS[m]), "completion"),  # random buttons hit at last
        ("defend_the_center", {}, lambda m: 0, "death"),
        (
            doom_body.Scenario(
                "health_gathering", "survive", 60, ("TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"),
                tic_reward=0.0025, death_reward=-1.0,
            ),
            {},
            lambda m: 2,
            "completion",
        ),
        ("health_gathering", {"max_moments": 5}, lambda m: 0, "truncation"),
    ],
)
def test_outcomes_are_distinct_and_credited_once(scenario, options, policy, wanted):
    with doom_body.DoomBody(scenario, seed=1, **options) as body:
        lived = _life(body, policy)
        spec = body.scenario
        assert [t.outcome for t in lived[:-1]] == [None] * (len(lived) - 1)
        assert lived[-1].outcome == wanted
        moments = [r for r in body.record if r["kind"] == "moment"]
        assert [r["outcome"] for r in moments].count(wanted) == 1
        assert sum(r["outcome"] is not None for r in moments) == 1
        kills = sum(t.events.count("kill") for t in lived)
        tics = sum(t.tics for t in lived)
        credit = spec.tic_reward * tics + spec.kill_reward * kills + spec.outcome_reward(wanted)
        assert np.isclose(sum(t.reward for t in lived), credit)
        assert np.isclose(body.returned, credit)
        assert lived[-1].events.count(wanted) == 1
        if wanted == "deadline":
            assert body.tic == spec.length
        if wanted in ("deadline", "truncation") or spec.goal == "survive" and wanted != "death":
            assert lived[-1].observation is not None
        with pytest.raises(RuntimeError):
            body.step(0)  # an ended episode credits nothing more
        assert body.returned == pytest.approx(credit)


@engine
def test_kills_are_credited_once_each():
    with doom_body.DoomBody("defend_the_center", seed=2) as body:
        lived = _life(body, lambda m: 2 if m % 3 else 0)
        kills = sum(t.events.count("kill") for t in lived)
        final = [r for r in body.record if r["kind"] == "moment"][-1]["variables"]["KILLCOUNT"]
        assert kills >= 1 and kills == final
        assert np.isclose(body.returned, kills * 1.0 - 1.0)


@engine
def test_the_ports_are_the_hud_numbers():
    with doom_body.DoomBody("defend_the_center", seed=0, resolution="160x120") as body:
        o = body.reset()
        assert o.screen.shape == (120, 160, 3)
        health, armour, ammunition = (body._read()[i] for i in range(3))
        assert np.allclose(o.body, [health / 100, armour / 100, ammunition / 50])
        assert not o.history.any()
        t = body.step(2)
        assert t.observation is not None and t.observation.history.tolist() == [0, 0, 1]


def _digest(o):
    return hashlib.sha256(o.screen.tobytes()).hexdigest()


@engine
def test_the_same_seed_is_the_same_world():
    runs = []
    for _ in range(2):
        with doom_body.DoomBody("defend_the_center", seed=5) as body:
            lived = _life(body, lambda m: (m * 7) % 3)
            runs.append([(t.tic, t.reward, t.outcome, t.events) for t in lived])
    assert runs[0] == runs[1]


def _continue(body, player, o, previous, moments):
    out = []
    for _ in range(moments):
        action = player.act(o, previous)
        t = body.step(action)
        out.append((action, t.tic, t.reward, t.outcome, t.events, _digest(t.observation)))
        assert not t.done
        o, previous = t.observation, t
    return out


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


@engine
def test_a_checkpoint_resumes_game_body_and_brain_at_the_same_tic(tmp_path):
    body = doom_body.DoomBody("defend_the_center", seed=3)
    player = doom_players.BrainPlayer("84x84", body.actions, seed=4, backend="cpu")
    o, previous = body.reset(), None
    for _ in range(6):
        t = body.step(player.act(o, previous))
        o, previous = t.observation, t
    o = doom_life.save_life(tmp_path / "life", body, player, previous)
    tic = body.tic
    assert body.record[-1]["kind"] == "checkpoint" and body.record[-1]["tic"] == tic
    original = _continue(body, player, o, previous, 8)
    player.save(tmp_path / "original")
    body.close()

    body2, player2, o2, previous2 = doom_life.load_life(tmp_path / "life")
    assert body2.tic == tic and _digest(o2) == _digest(o)
    resumed = _continue(body2, player2, o2, previous2, 8)
    player2.save(tmp_path / "resumed")
    body2.close()

    assert resumed == original
    a, b = _arrays(tmp_path / "original.npz"), _arrays(tmp_path / "resumed.npz")
    assert a.keys() == b.keys()
    for name in a:
        assert np.array_equal(a[name], b[name]), name  # on the CPU, the brain to the bit


@engine
def test_a_checkpoint_survives_json_and_refuses_other_engines():
    with doom_body.DoomBody("basic", seed=0) as body:
        body.reset()
        body.step(1)
        saved = json.loads(json.dumps(body.checkpoint()))
    saved["body"]["engine"] = {"vizdoom": "0.0", "files": {}}
    with pytest.raises(ValueError):
        doom_body.DoomBody.restore(saved)


@engine
@torch_present
@pytest.mark.parametrize("scenario", sorted(doom_body.SCENARIOS))
def test_every_player_lives_through_every_scenario(scenario):
    for name in doom_life.PLAYERS:
        with doom_body.DoomBody(scenario, seed=0, max_moments=4) as body:
            options = {"backend": "cpu"} if name.startswith("brain") else {}
            player = doom_life.make_player(name, body, 0, **options)
            rows = doom_life.play(body, player, episodes=2)
            assert [r["outcome"] for r in rows] == ["truncation", "truncation"]
            assert all(r["moments"] == 4 for r in rows)
            if name.startswith("brain") and name != "brain-frozen":
                assert not player.brain.pending_feedback
