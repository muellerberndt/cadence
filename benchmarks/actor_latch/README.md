# A long reward life outside VizDoom

The chamber of [issue 296](https://github.com/muellerberndt/cadence/issues/296): at the composed
actor step, a brain learning from reward over tens of thousands of moments in
[the Doom body](../doom/README.md) holds one button and scores below its own frozen newborn. The
chamber keeps the Doom player's observation and drops the engine, so a life costs a brain's
moments alone.

| File | What it holds |
| --- | --- |
| [`chamber.py`](chamber.py) | the three worlds, a life, its reading by quarters, the command line |
| [`test_actor_latch.py`](test_actor_latch.py) | the worlds' declarations and a short life |

## The worlds

A player reads one vector per moment: a retina of 1,323 values (a fixed scene with the mean and
spread per pixel of a Doom screen, and independent noise each moment), three body ports and the
history port (the episode's previous button, one-hot): 1,329 inputs, as the Doom brain reads at
84 by 84. It answers one of three buttons.

| World | Shape | Rewards | Random buttons | One button held |
| --- | --- | --- | --- | --- |
| `bandit` | one paying button | -0.01 a moment; the paying button ends the episode with +1 at .04 a press; 75 moments | complete .62 (Doom basic: .63) | the paying one completes .97 |
| `aim` | Doom basic: buttons 0 and 1 move a target's offset (in -6..6) that the retina shows as a bright band; button 2 attacks | -0.01 a moment; an attack on target ends the episode with +1; 75 moments | complete .62 | attack: return -0.49 |
| `survival` | Doom health gathering: health drains .0104 a moment; the paying button finds a medkit (+.25 health) at .0625 a press | +0.01 a moment alive, death -1; 525 moments | live 171 moments | a non-paying one: 97 moments, return -0.040 |

The paying button of the bandit and of survival is the founder number modulo 3. In aim the
right button depends on what the retina shows: strafe toward the target, then attack.
`--world` changes a world's setting (`width`, `history=false` for a dark history port, `alive`,
`death`, `pay`), `--set` the brain's (`--set actor_eta=0.1`), and `--rate pre->post=x` multiplies
one projection's step.

```sh
PYTHONPATH=src python benchmarks/actor_latch/chamber.py --chamber aim --episodes 300 --seed 100 > aim-100.json
PYTHONPATH=src python benchmarks/actor_latch/chamber.py --read aim-100.json
```

`--player frozen` (the newborn's sampled acts) and `--player random` are the controls. `--read`
prints each life by quarters of its episodes: the button shares, the paying button's share,
completion, return, the mean motor activations, `saturation`, the actor's mean synapse and bias
steps, the critic's error and the trace.

## Where the hold comes from

Development founders 100 to 103 (and 104 to 107 for the survival default and its frozen
newborn), `Brain.compose(1329, 3, seed=founder, backend="cpu")` learning by `step`, each arm one
named change from the default composition; 400 episodes of the bandit, 300 of aim, 100 of
survival. Return over each life's last quarter, the mean over founders; *held*: one button at
.90 or more in the last quarter; *off*: a held button the world does not pay.

| Arm | Bandit: return, held / off | Aim: return, held | Survival: return, held / off |
| --- | --- | --- | --- |
| default | +0.57, 3/4 / 0/4 | -0.32, 1/4 | +3.43, 6/8 / 1/8 |
| frozen newborn | +0.24, 0/4 / 0/4 | -0.17, 0/4 | +0.90, 0/8 / 0/8 |
| random buttons | +0.19 | +0.15 | +0.59 |
| `actor_eta=0.1` | +0.72, 4/4 / 0/4 | +0.94, 0/4 | +2.50, 1/4 / 0/4 |
| `actor_eta=0.1, actor_eta_bias=0.01` | +0.72, 4/4 / 0/4 | +0.81, 0/4 | +3.39, 2/4 / 0/4 |
| `actor_eta=0.03, actor_eta_bias=0.003` | +0.70, 4/4 / 0/4 | +0.91, 0/4 | +2.88, 0/4 / 0/4 |
| `actor_eta_bias=0.005` | +0.73, 4/4 / 0/4 | -0.22, 1/4 | +1.48, 3/4 / 2/4 |
| `actor_eta_bias=0` | +0.72, 4/4 / 0/4 | -0.41, 1/4 | +2.05, 1/4 / 0/4 |
| `actor_eta_critic=3.0` | +0.70, 4/4 / 0/4 | -0.62, 4/4 | +1.18, 4/4 / 3/4 |
| `actor_eta_critic=0.03` | +0.73, 4/4 / 0/4 | -0.11, 0/4 | +1.64, 1/4 / 0/4 |
| `temperature=0.5` | +0.64, 0/4 / 0/4 | +0.03, 0/4 | +0.35, 0/4 / 0/4 |
| `actor_lam=0` | +0.71, 4/4 / 0/4 | -0.43, 4/4 | +2.45, 2/4 / 1/4 |
| a retina of 32 values | +0.71, 4/4 / 0/4 | -0.33, 1/4 | +0.65, 1/4 / 1/4 |
| the input projections into the association at 0.0286 of the step | +0.64, 3/4 / 0/4 | +0.44, 0/4 | +1.48, 3/4 / 2/4 |
| the association-motor pair at no step | +0.72, 4/4 / 0/4 | +0.49, 0/4 | +1.76, 0/4 / 0/4 |
| `episodic=False` | +0.70, 4/4 / 0/4 | -0.40, 0/4 | +2.35, 4/4 / 2/4 |
| `recall=False` | +0.71, 4/4 / 0/4 | +0.27, 1/4 | +0.90, 2/4 / 2/4 |
| history port dark | | | +3.57, 3/4 / 0/4 |
| working trace off (`working_memory_amplitude=0`) | | | +2.90, 1/4 / 0/4 |

- Aim reads the Doom basic order: the learning brain below its frozen newborn below random
  buttons. Survival holds a founder on a non-paying button with the latched Doom founders'
  return. A paying button alone (the bandit) is learned.
- The term is the actor's synapse step: a tenth of `actor_eta`, with the bias rate kept, takes
  aim to +0.94 on every founder (an aimer that strafes toward the target and fires reads .97).
  The bias step does not move it; a faster critic holds more; the softmax gain removes the hold
  and caps the policy (the bandit's paying share stops at .82); a trace of one moment
  (`actor_lam=0`) holds attack on every founder.
- One capped dopamine moves an association neuron's net input by .33 on average through 1,323
  retina values and by .014 through 32, while the mean step per synapse reads the same; a retina
  of 32 values holds as well, so the input width scales the step and does not make the hold.
- At birth one or two motor neurons are active and the others sit at rest at the 0 rail: `saturation`
  reads .67 to 1.00 from the first moment. An active button's contrast is .28 to .43, a resting
  one's .01 to .04.

## The declared run

Declared on [issue 296](https://github.com/muellerberndt/cadence/issues/296#issuecomment-6767730132)
before its first life, at commit 94fb1233 (0.96.1 with the Doom body). Arms: `entropy`
(`actor_entropy=0.1`, the smallest weight that removed the hold on every development founder in
aim), `rate` (`actor_eta=0.1, actor_eta_bias=0.01`) and `default`, every brain
`Brain.compose(1329, 3, seed=founder, backend="cpu")` through `step`; the frozen newborn, random
buttons and, in the Doom body, the convolutional actor-critic beside them. Fresh founders: 3 to 8
in the Doom body (84 by 84, 4 tics a moment, 300 episodes of basic and 100 of each survival),
200 to 205 in the chamber (300 episodes of aim, 100 of survival). One AWS c7i.8xlarge (Ubuntu
24.04, Python 3.13, NumPy 2.5.4, VizDoom 1.3.2), one thread per process; four Doom lives that
VizDoom refused at launch (a race over its `./_vizdoom` directory, no episode played) ran again as
the same lives. The gate: a founder passes a world when its mean return over the life's last
quarter of episodes exceeds its own frozen newborn's by two standard errors of the difference; a
world passes at 5 of 6 founders, and an arm passes when two of the three Doom scenarios do.
[`gate.py`](gate.py) prints these tables from the lives.

The Doom body:

| Arm | Scenario | Parameters | Founders above the newborn by 2 SE | Return, last quarter | Newborn | Largest button share per founder, last quarter |
| --- | --- | ---: | ---: | --- | ---: | --- |
| default | basic | 2,291,716 | 0 of 6 | -0.077 ± 0.051 | +0.101 | .98, .99, .99, .98, .96, .51 |
| default | defend the center | 2,291,716 | 0 of 6 | +0.527 ± 0.198 | +0.627 | .99, .99, .99, .99, .37, .99 |
| default | health gathering | 2,291,716 | 0 of 6 | +0.107 ± 0.106 | +0.290 | .38, .97, .98, .36, .96, .99 |
| entropy | basic | 2,291,716 | 2 of 6 | +0.250 ± 0.074 | +0.101 | .35, .35, .36, .34, .36, .34 |
| entropy | defend the center | 2,291,716 | 0 of 6 | +0.380 ± 0.208 | +0.627 | .34, .35, .36, .34, .38, .35 |
| entropy | health gathering | 2,291,716 | 0 of 6 | +0.180 ± 0.049 | +0.290 | .34, .35, .35, .35, .35, .34 |
| rate | basic | 2,291,716 | 1 of 6 | +0.117 ± 0.129 | +0.101 | .72, .98, .86, .64, .47, .84 |
| rate | defend the center | 2,291,716 | 1 of 6 | +0.713 ± 1.188 | +0.627 | .98, .98, .65, .97, .37, .98 |
| rate | health gathering | 2,291,716 | 2 of 6 | +0.294 ± 0.370 | +0.290 | .55, .79, .57, .41, .35, .86 |
| conv | basic | 677,684 | 2 of 6 | +0.211 ± 0.044 | +0.101 | .39, .47, .59, .48, .46, .47 |
| conv | defend the center | 677,684 | 2 of 6 | +0.753 ± 0.398 | +0.627 | .40, .40, .46, .45, .45, .42 |
| conv | health gathering | 677,684 | 2 of 6 | +0.313 ± 0.116 | +0.290 | .35, .42, .38, .34, .37, .38 |
| random | basic | 0 | 3 of 6 | +0.250 ± 0.095 | +0.101 | .34 each |
| random | defend the center | 0 | 1 of 6 | +0.573 ± 0.231 | +0.627 | .34 to .36 |
| random | health gathering | 0 | 0 of 6 | +0.227 ± 0.050 | +0.290 | .34 each |

Returns are the mean over founders with their standard deviation. The chamber:

| Arm | World | Parameters | Founders above the newborn by 2 SE | Return, last quarter | Newborn | Largest button share per founder, last quarter |
| --- | --- | ---: | ---: | --- | ---: | --- |
| default | aim | 2,291,716 | 2 of 6 | -0.350 ± 0.326 | -0.370 | .57, .50, .99, .99, .69, .49 |
| default | survival | 2,291,716 | 2 of 6 | +1.456 ± 1.796 | +0.810 | .99 (not paying), .95, .35, .98 (not paying), .81, .35 |
| entropy | aim | 2,291,716 | **6 of 6** | **+0.918 ± 0.021** | -0.370 | .37, .36, .34, .40, .37, .38 |
| entropy | survival | 2,291,716 | 1 of 6 | +0.644 ± 0.154 | +0.810 | .34 to .35 |
| rate | aim | 2,291,716 | **6 of 6** | +0.654 ± 0.322 | -0.370 | .83, .78, .72, .74, .44, .38 |
| rate | survival | 2,291,716 | 2 of 6 | +1.608 ± 2.249 | +0.810 | .93 (not paying), .95, .37, .87, .91 (not paying), .94 |
| random | aim | 0 | 5 of 6 | +0.149 ± 0.060 | -0.370 | .33 to .34 |
| random | survival | 0 | 2 of 6 | +0.620 ± 0.075 | +0.810 | .33 to .35 |

**No arm passes the gate.** The default holds one button at .96 to .99 in 14 of 18 Doom lives
(attack in basic and in defend the center) and passes no founder.

- **`actor_entropy=0.1` removes the hold everywhere.** No button exceeds .40 in any of its 30
  lives. In aim, whose right answer depends on what the retina shows, it reads +0.918 on every
  fresh founder against an aimer's .97. In the Doom body it holds every policy near even: basic
  reads +0.250 (random buttons +0.250), and defend the center and health gathering read below
  their newborns.
- **The failing term: the entropy reaches its even policy with every motor output at rest.** In
  every survival life under the entropy the three motor neurons sit below rest (mean activations
  -0.02 to -0.07, `saturation` 1.00 in the last quarter) and the mean synapse step falls from
  1.3e-6 to 4e-7 to 8e-7; in aim, whose kills pay +1, they rise into the sensitive band (+0.04 to
  +0.51, `saturation` .22 to .76). The entropy's step lowers an active output easily (below rest
  the leak keeps a slope) and raises a resting one hardly at all, so the even policy it settles
  in is the one at the 0 rail, where a nudge has little slope and a faint preference cannot lift
  a button: the resting button's contrast of .01 to .04 measured at birth. A longer critic horizon
  leaves it (`actor_gamma=0.99` beside the entropy: shares .32 to .35 and saturation 1.00 on
  development founders 100 and 101). Where a good policy presses one button far more (forward in
  health gathering, turning and firing in defend the center, whose newborns score +0.290 and
  +0.627 against an even policy's +0.227 and +0.573), the brain stays at the even return.
- **`actor_eta=0.1, actor_eta_bias=0.01`** lifts aim on every fresh founder (+0.654) and holds a
  non-paying survival button on 2 of 6 and attack in defend the center on 4 of 6: a tenth of the
  step slows the hold and does not remove it.
- A moment with `actor_entropy` costs 36.5 ms against the default's 27.6 on the box (its step
  forms each act's contrast on the host, where the composed actor fuses it into the trace).
