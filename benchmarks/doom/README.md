# The Doom body

The first stage of [the Doom player](https://github.com/muellerberndt/cadence/issues/118), its
Body stage ([#285](https://github.com/muellerberndt/cadence/issues/285)): an adapter that puts
the library into [VizDoom](https://github.com/Farama-Foundation/ViZDoom) with the contracts
every later stage relies on. It is an instrument: the library is unchanged, and the brain is
`Brain.compose` learning by its own local rule, with no backward pass.

| File | What it holds |
| --- | --- |
| [`doom_body.py`](doom_body.py) | `DoomBody`: the sensors, the actions, the scorer, custody and the checkpoint |
| [`doom_players.py`](doom_players.py) | the brain player, its blind and frozen controls, a random player and a small convolutional actor-critic |
| [`doom_life.py`](doom_life.py) | a life of episodes (`play`), a life's checkpoint (`save_life`, `load_life`), the command line and its table |
| [`test_doom.py`](test_doom.py) | the contract tests |
| [`results/`](results) | the runs read on this page |

## Install

VizDoom is an optional dependency, kept out of the package's own extras. Install it beside the
library in an environment of its own:

```sh
python -m venv doom && doom/bin/pip install vizdoom==1.3.2 numpy torch pytest
PYTHONPATH=src doom/bin/python -m pytest benchmarks/doom
```

The VizDoom wheel carries the free [Freedoom](https://freedoom.github.io) IWADs and the scenario
files, so nothing else is fetched: the body reads `freedoom2.wad` and the scenarios'
`.cfg` and `.wad` files from the installed package. `doom_body.engine_identity()` returns the
engine's version and the SHA-256 of each file, every output file records them, and a checkpoint
written by another engine or other files is refused. Read with VizDoom 1.3.2:

| File | SHA-256 |
| --- | --- |
| `freedoom2.wad` | `a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b` |
| `basic.wad` | `fd0b7e5e2422e09288c786e7fc32181bb8ee4a3faa613a75ef88a3e39b5ce94e` |
| `defend_the_center.wad` | `eb3f8d3b3f456bbef32a1d41f847189dd83cc1d4e320157ba92b75a9b0d8d5bb` |
| `health_gathering.wad` | `c77c8ba3254ae19eaff8d00418870a7718ef8a5822e5872ab68b76205cb2bacf` |

Without VizDoom the engine's tests skip, and the declarations, the players' reading of an
observation and a life's refusal rule are read. The engine runs without a window on macOS
(Apple M4, the arm64 wheel) and on Linux (Ubuntu 24.04, x86-64).

## The body

```python
import doom_body

with doom_body.DoomBody("basic", resolution="84x84", tics=4, seed=0) as body:
    observation = body.reset()                 # the first moment of episode 0
    transition = body.step(2)                  # hold ATTACK for 4 tics
    transition.reward, transition.outcome, transition.tic, transition.events
```

### What a player receives

An `Observation` holds three fields and nothing else:

| Port | Shape | Meaning |
| --- | --- | --- |
| `screen` | `(84, 84, 3)` or `(120, 160, 3)` uint8 | the rendered RGB screen: 160 by 120 as the engine renders it, 84 by 84 its area-weighted resampling; the HUD and the crosshair are not drawn |
| `body` | `(3,)` | the HUD's own-body numbers: health / 100, armour / 100, the selected weapon's ammunition / 50 (0 without a weapon) |
| `history` | `(buttons,)` | the episode's previous action, one-hot; zeros at its first moment |

The engine's other variables (the kill count, the item count, the tic) stay in the body: the
scorer reads them and custody records them. The map, enemy positions, progress and any
teacher's answer never reach a player. A test reads the players' source for any part of the body
but the observation and the transition.

### Actions

The declared button set of the body is move forward, move backward, turn left, turn right,
strafe left, strafe right, attack and use. A scenario declares the buttons it offers, and an
action is one integer index into them, held for `tics` game tics (4 here): one button per moment.
Anything else (a negative or too large index, a float, a boolean, a string, an array) raises
`ValueError`, and a step with no episode running raises `RuntimeError`, before the world
advances.

### Scorer

| Scenario | Map | Buttons | Goal | Length | Rewards |
| --- | --- | --- | --- | ---: | --- |
| `basic` | `basic.wad` | strafe left, strafe right, attack | kill: the engine ends the episode when the target dies | 300 tics | completion +1, each tic -0.0025, deadline 0 |
| `defend_the_center` | `defend_the_center.wad` | turn left, turn right, attack | survive | 2,100 tics | each kill +1, death -1 |
| `health_gathering` | `health_gathering.wad` | turn left, turn right, move forward | survive | 2,100 tics | each tic alive +0.0025, death -1 |

Maps, buttons and lengths are those of VizDoom's own scenario files. The engine's scripted
rewards and its own timeout are off: the body counts the scenario's length in its own tics and
scores from engine facts. Every episode ends in exactly one of four outcomes:

| Outcome | When |
| --- | --- |
| `completion` | a kill goal: the engine ends the episode with the player alive; a survival: the player is alive at the length |
| `death` | the player dies |
| `deadline` | a kill goal reaches its length first |
| `truncation` | the harness ends the episode: `max_moments` reached, or `truncate()` |

A moment's reward is the tic reward times the tics it advanced, plus each kill's reward, plus
the outcome's reward on the moment that ends the episode. The outcome is credited on that moment
alone, a step after it is refused, and the next episode starts with `reset()`. A
`Transition` carries the next observation, or after a deadline, a survival's completion or a
truncation the frame at the end, which a learner bootstraps a truncated episode from; it carries
none when the engine ended the episode.

### Custody

`body.record` holds one entry per episode start (its tic and the engine's seed), per moment (the
action, the tic before and after, the tics advanced, the reward, the events, the outcome and the
engine's variables) and per checkpoint. The events are `kill` (one per kill), `damage`, `heal`,
`armour_lost`, `armour`, `shot`, `ammunition`, `pickup` and the outcome. Episode `e` runs with the
engine's seed `seed + e`, and the same seed and actions give the same world.

### Checkpoint

`body.checkpoint()` returns the game (the engine's save game) and the body as one JSON-safe
mapping, and `DoomBody.restore(checkpoint)` continues it at the same tic. The engine writes its
save game within two idle tics, so the body loads the save it wrote: the original life and the
resumed one continue from one world at one tic, and custody records the two tics as a
`checkpoint` entry whose events are scored with the next moment. `doom_life.save_life` writes a
brain's whole life (the game, the body, the brain with its pending action and the transition it
has yet to learn), and `load_life` reads it back.

## The contract tests

| Contract | Test |
| --- | --- |
| A refused action does not advance the world | `test_refused_actions_do_not_advance_the_world`: eight malformed actions, a reset while an episode runs and a step after its end leave the tic, the engine's tic, custody and the screen as they were |
| A refused moment of the player leaves the world where it stands | `test_a_refused_moment_leaves_the_world_where_it_stands`: `play` ends the life with `Refused` at the tic the world stands at |
| Each outcome is credited exactly once; the four stay distinct | `test_outcomes_are_distinct_and_credited_once`: lives forced into a deadline, a kill goal's completion, a death, a survival's completion and a truncation each end in their outcome alone, credited on one moment, the return equal to the scorer's sum, and a further step refused; `test_kills_are_credited_once_each` |
| A checkpoint resumes game, body and brain at the same tic | `test_a_checkpoint_resumes_game_body_and_brain_at_the_same_tic`: a brain's life saved after six moments and resumed in a new process state takes the same eight actions at the same tics with the same rewards, events and screens as the life that went on, and on `backend="cpu"` every array of the two brains is equal to the bit |
| Only the declared ports reach the player | `test_an_observation_holds_the_declared_ports_and_nothing_else`, `test_the_ports_are_the_hud_numbers`, `test_the_brain_reads_the_retina_and_the_ports` |
| The same seed is the same world | `test_the_same_seed_is_the_same_world`, `test_a_checkpoint_survives_json_and_refuses_other_engines` |
| Every player lives through every scenario | `test_every_player_lives_through_every_scenario` |

All 20 pass on an Apple M4 (macOS, Python 3.13) and on an AWS c7i.4xlarge (Ubuntu 24.04, Python
3.12) with VizDoom 1.3.2; without VizDoom 14 skip and 6 pass.

## The players

Every player answers `act(observation, previous)` with one button index, where `previous` is the
transition that led to the observation, and `finish(previous)` takes the life's last transition.

- **`brain`**: `Brain.compose(inputs, buttons, seed=founder, backend="cpu")`, the default
  composition at its founders, learning by `step` from every outcome: the reward of the
  previous action, `done` at each episode's end, and a truncated episode's value
  (`basal_ganglia.value_of` of its last frame) as its bootstrap. It reads one vector: the
  screen's means over 4 by 4 blocks at 84 by 84 (21 by 21 by 3) or over 8 by 8 blocks at 160 by
  120 (15 by 20 by 3), in [0, 1], then the body ports and the history port: 1,329 inputs at 84
  by 84, 906 at 160 by 120.
- **`brain-blind`**, a control: the same brain with its retina dark; it reads the body and history
  ports alone.
- **`brain-frozen`**, a control: the newborn's sampled acts (`act`), nothing learned.
- **`random`**, a control: uniform buttons.
- **`conv`**: a small convolutional actor-critic at the same information (the full screen, the
  body and history ports, the same rewards and episode ends): 16 maps of 8 by 8 at stride 4 and
  32 of 4 by 4 at stride 2, a 256-unit layer that also reads the ports, a softmax policy and a
  value; advantage actor-critic with 5-step returns, RMSprop at 7e-4, entropy weight 0.01,
  gradients clipped at 0.5, in PyTorch.

```sh
PYTHONPATH=src python benchmarks/doom/doom_life.py --player brain --scenario basic \
    --episodes 300 --seed 0 --out results/basic-brain-84x84-s0.jsonl
PYTHONPATH=src python benchmarks/doom/doom_life.py --table results/*.jsonl
```

`--resolution 160x120` reads the larger screen, `--backend torch` or `--backend mlx` puts the brain
on a device, `--device mps` or `--device cuda` the convolutional player. Each output line is one
episode (outcome, tic, tics, moments, return, kills and the player's and the world's seconds),
after a first line that declares the run (the player, its parameters, the brain's placement, the
scenario's scorer, the engine's identity, the machine).

## Every scenario through the body

The runs below were declared on
[#285](https://github.com/muellerberndt/cadence/issues/285#issuecomment-6760315841) before their
first life, and ran at commit 03f8572 of this directory (the declared 246fa69 with the engine's
settings file moved to the temporary directory). One life per run, founders 0, 1 and 2, one
thread per process, 14 processes at a time on one AWS c7i.4xlarge (16 vCPUs of an Intel Xeon
Platinum 8488C, Ubuntu 24.04, Python 3.12, NumPy 2.5.4, PyTorch 2.15.0, VizDoom 1.3.2), 4 tics per
moment. Scoring well is no gate of this stage. Each row sums its founders' episodes; the returns
are the mean over founders with their standard deviation, the last quarter is each life's last
quarter of episodes. The files are in [`results/battery/`](results/battery).

At 84 by 84 (300 episodes of basic, 100 of each survival, per founder):

| Scenario | Player | Parameters | Episodes | Completion | Death | Deadline | Truncation | Return | Return, last quarter | Kills | Tics |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| basic | brain | 2,291,716 | 900 | .38 | 0 | .62 | 0 | -0.132 ± 0.050 | -0.075 ± 0.124 | 0.38 | 203 |
| basic | brain-blind | 2,291,716 | 900 | .43 | 0 | .57 | 0 | -0.054 ± 0.045 | -0.062 ± 0.068 | 0.43 | 195 |
| basic | brain-frozen | 2,291,716 | 900 | .48 | 0 | .52 | 0 | 0.014 ± 0.135 | 0.035 ± 0.123 | 0.48 | 185 |
| basic | conv | 677,684 | 900 | .57 | 0 | .43 | 0 | 0.151 ± 0.082 | 0.160 ± 0.062 | 0.57 | 169 |
| basic | random | 0 | 900 | .63 | 0 | .37 | 0 | 0.230 ± 0.065 | 0.188 ± 0.121 | 0.63 | 161 |
| defend_the_center | brain | 2,291,716 | 300 | 0 | 1.00 | 0 | 0 | 0.803 ± 1.241 | 1.107 ± 1.194 | 1.80 | 345 |
| defend_the_center | brain-blind | 2,291,716 | 300 | 0 | 1.00 | 0 | 0 | 0.427 ± 0.047 | 0.413 ± 0.061 | 1.43 | 315 |
| defend_the_center | brain-frozen | 2,291,716 | 300 | 0 | 1.00 | 0 | 0 | 1.497 ± 0.215 | 1.587 ± 0.477 | 2.50 | 393 |
| defend_the_center | conv | 677,684 | 300 | 0 | 1.00 | 0 | 0 | 0.803 ± 0.289 | 1.293 ± 1.036 | 1.80 | 350 |
| defend_the_center | random | 0 | 300 | 0 | 1.00 | 0 | 0 | 0.287 ± 0.218 | 0.307 ± 0.401 | 1.29 | 333 |
| health_gathering | brain | 2,291,716 | 300 | 0 | 1.00 | 0 | 0 | 0.053 ± 0.142 | 0.018 ± 0.100 | 0 | 421 |
| health_gathering | brain-blind | 2,291,716 | 300 | 0 | 1.00 | 0 | 0 | 0.184 ± 0.107 | 0.197 ± 0.048 | 0 | 474 |
| health_gathering | brain-frozen | 2,291,716 | 300 | .02 | .98 | 0 | 0 | 0.497 ± 0.350 | 0.441 ± 0.231 | 0 | 590 |
| health_gathering | conv | 677,684 | 300 | 0 | 1.00 | 0 | 0 | 0.347 ± 0.025 | 0.254 ± 0.072 | 0 | 539 |
| health_gathering | random | 0 | 300 | 0 | 1.00 | 0 | 0 | 0.210 ± 0.014 | 0.158 ± 0.003 | 0 | 484 |

At 160 by 120 (founder 0; the brain reads 906 inputs):

| Scenario | Player | Parameters | Episodes | Completion | Death | Deadline | Return | Return, last quarter | Kills | Tics |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| basic | brain | 1,183,033 | 300 | .44 | 0 | .56 | -0.013 | 0.139 | 0.44 | 182 |
| basic | conv | 1,931,060 | 300 | .60 | 0 | .40 | 0.186 | 0.186 | 0.60 | 165 |
| defend_the_center | brain | 1,183,033 | 100 | 0 | 1.00 | 0 | 0.390 | 0.480 | 1.39 | 307 |
| defend_the_center | conv | 1,931,060 | 100 | 0 | 1.00 | 0 | 1.290 | 1.760 | 2.29 | 391 |
| health_gathering | brain | 1,183,033 | 100 | 0 | 1.00 | 0 | 0.127 | 0.021 | 0 | 451 |
| health_gathering | conv | 1,931,060 | 100 | 0 | 1.00 | 0 | 0.314 | 0.229 | 0 | 526 |

Every episode ended in exactly one outcome. Survivals reached their 2,100 tics in 7 episodes, all
of one frozen newborn (founder 1 of health gathering), and none was truncated (the runs set no
`max_moments`). At these
exposures (about 40,000 moments of basic and 25,000 to 45,000 of each survival per life) no
learner leaves its controls: the learning brain scores below its own frozen newborn on all three
scenarios and below random buttons on basic and health gathering, and the convolutional actor-critic scores between
random and the frozen newborn.

### Where the learning brain's score goes

[`latch.py`](latch.py) reads the same lives button by button (the founder rows repeat the battery's
lives to the button, on the Mac and on Linux alike), and the same lives composed at a tenth of the
actor's step (`actor_eta=0.1`, `actor_eta_bias=0.01`), the rate [reward](../../docs/reward.md#actor-rates)
names for long reward lives. Button shares, completion and return over each life's first and last
quarter ([`results/latch/`](results/latch)):

| Scenario | `actor_eta` | Founder | Quarter | Button shares | Completion | Return | Saturation |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| basic | 1.0 | 0 | first | .02, .11, .87 | .37 | -0.120 | .69 |
| basic | 1.0 | 0 | last | .17, .22, .61 | .51 | 0.064 | .86 |
| basic | 1.0 | 1 | first | .02, .95, .02 | .19 | -0.444 | .72 |
| basic | 1.0 | 1 | last | .01, .01, .99 | .37 | -0.114 | .67 |
| basic | 1.0 | 2 | first | .67, .01, .32 | .21 | -0.380 | .67 |
| basic | 1.0 | 2 | last | .01, .01, .99 | .33 | -0.174 | .67 |
| basic | 0.1 | 0 | first | .32, .30, .38 | .57 | 0.173 | .93 |
| basic | 0.1 | 0 | last | .23, .24, .53 | .53 | 0.122 | .89 |
| basic | 0.1 | 1 | first | .24, .38, .39 | .60 | 0.197 | .88 |
| basic | 0.1 | 1 | last | .28, .33, .39 | .51 | 0.040 | .98 |
| basic | 0.1 | 2 | first | .38, .20, .42 | .56 | 0.147 | .78 |
| basic | 0.1 | 2 | last | .02, .02, .97 | .43 | -0.048 | .67 |
| health_gathering | 1.0 | 0 | last | .49, .51, .00 | 0 | -0.040 | .86 |
| health_gathering | 1.0 | 2 | last | .98, .01, .01 | 0 | -0.040 | .69 |
| health_gathering | 0.1 | 2 | last | .06, .45, .49 | .04 | 1.045 | .34 |

At the composed step the policy holds one button within the first quarter of a life: a founder
that strafes one way 95 times in 100 completes basic in .19 of its first quarter, and a founder
that never moves forward lives out its starting health in health gathering (a mean return of
-0.040, 384 tics). Saturation, the share of motor outputs
within 0.02 of a rail, reads .67 to 1.00. This is the latch [reward](../../docs/reward.md#actor-rates)
describes for long reward lives at the composed rate. At a tenth of the step the button shares
stay mixed for longer and basic's completions start near random's .63, and one founder in three
holds attack by its last quarter. These lives are a location reading outside the declared runs.

## Seconds per moment

A moment is 4 tics. The player's seconds are its `act` (with the brain's learning); the world's
are the body's `step` (the engine, the variables and the resampling).

| Machine | Player | Screen | Player ms per moment | World ms per moment |
| --- | --- | --- | ---: | ---: |
| c7i.4xlarge, one process alone (load 3.5 falling to 1.9), one core | brain, `backend="cpu"` | 84 by 84 | 9.7 to 10.0 | 2.1 |
| c7i.4xlarge, one process alone, one core | brain, `backend="cpu"` | 160 by 120 | 7.6 to 8.2 | 1.5 |
| c7i.4xlarge, one process alone, one core | conv, CPU | 84 by 84 | 1.6 to 1.7 | 2.1 |
| c7i.4xlarge, one process alone, one core | conv, CPU | 160 by 120 | 3.1 to 3.2 | 1.5 |
| c7i.4xlarge, 14 processes | brain, `backend="cpu"` | 84 by 84 | 22.2 | 2.7 |
| c7i.4xlarge, 14 processes | conv, CPU | 84 by 84 | 3.2 | 2.5 |
| Apple M4 (load about 20), one core | brain, `backend="cpu"` (float64) | 84 by 84 | 32.1 | 12.7 |
| Apple M4 (load about 20) | brain, `backend="mlx"` (float32) | 84 by 84 | 26.8 | 21.6 |
| Apple M4 (load about 20) | brain, `backend="torch"`, MPS (float32) | 84 by 84 | 80.1 | 16.9 |
| Apple M4 (load about 20), one core | conv, CPU | 84 by 84 | 5.3 | 14.5 |
| Apple M4 (load about 20) | conv, MPS | 84 by 84 | 71.2 | 25.0 |

Per tic, divide by 4: the brain's moment alone on one c7i core is 2.5 ms a tic at 84 by 84. The M4
rows are three episodes of basic each ([`results/device/`](results/device)), on a shared machine;
on all three backends the brain took the same buttons in those episodes. The one-core rows are
five episodes of every scenario per player ([`results/timing/`](results/timing)).

## What the next stage needs from the library

- **An image beside declared ports.** `Brain.compose(inputs=(h, w, c))` reads an image through a
  visual cortex, but takes no vector port beside it, so the own-body and history ports have no
  way in. The player here is a vector brain over a pooled retina. A composition with an image and
  declared vector ports, each projecting into the first module, would let the visual cortex read
  the screen while the body ports enter as themselves.
- **A visual cortex cheap enough to live in.** The default image composition at 84 by 84 by 3
  (`code=1024` for its memories; 77,139 neurons, 6.4 million parameters) settles a moment in
  4.4 s on one M4 core at a load of about 45, twenty times the 0.21 s of the vector brain's moment
at that load;
  `compose` exposes no stride for the visual cortex. Perception from pixels is #284.
- **A step that does not latch over long reward lives.** At the composed `actor_eta` the policy
  holds one button within a few thousand moments (above); the next stage's acquisition gates
  need a learning brain above its frozen newborn.
- **Credit across the seconds of a fight.** In basic a kill pays several moments after the shots
  that aimed it, and a death in defend the center closes a long sequence: the delayed credit of #283.
- **Many lives with arousal and episode ends.** `cadence.Lives` steps many brains of one
  composition on a device, and refuses arousal, image inputs and codes; the store, arousal and
  episode ends are #293. A Doom population (founders, controls, curricula) needs them.
- **A public value read.** A truncated episode bootstraps from the value of its last frame; the
  player reads it as `brain.basal_ganglia.value_of(brain.stimulus(x, memory=False))`, and a
  public read of a brain's value of an observation would serve every adapter.
- The checkpoint needs nothing more: `Brain.save` carries the pending action and the brain
  resumes to the bit.
