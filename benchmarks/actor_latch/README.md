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
