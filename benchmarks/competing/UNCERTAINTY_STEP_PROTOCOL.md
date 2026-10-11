# An actor step that follows the brain's own uncertainty: mechanism brief and gate (2026-10-11)

Status: branch `actor-uncertainty-step-20261011`, row 2 of the sequence order (issue
[#169](https://github.com/muellerberndt/cadence/issues/169)), the third mechanism on the plan
after the existing levers and the readout gene. Nothing here is implemented or measured yet;
the exact update law is declared on the branch before any run.

## Demonstrated limitation

A policy acquired at the nursery's actor step erodes under its own continued learning. In the
arena's sixty-fight protocol the acquired skill fell from 0.54 to 0.23 and 0.25 on two seeds at
the gene step, while a ring step ten to thirty times smaller held it at 0.49 to 0.54 with the
fight statistics moving; the nursery alone eroded licensed brains from 0.54 to 0.19 over ninety
thousand moments. In the [competing-skill ring](README.md) under the melee's signed pay the
loss reproduces at every actor step while the actor's efficacies barely move; the dopamine on
the old skill's moments is zero-mean; in a short nursery what the melee overwrites is the
episodic memory's one-shot record (`memory_rate` 1.0), and writing an average (0.2, 0.05)
halves the loss. The cause in the arena is the actor's updates on a loss-biased sample of
surprising moments followed by habituation. Users report the same shape in a two-step task:
eight of eight lives pass at 300 lesson trips, one of eight at 1,500.

Ruled out on unshipped library variants: a slow set point under every parameter (a linear
two-timescale anchor) and a sign-consistency stiffness. Neither held the skill; both were
removed.

## The mechanism

One gene of the reward learner, founder off. With the gene off the actor's step is the constant
`actor_eta`. With the gene on, the step taken on a moment is scaled by how unexpected that
moment's outcome is relative to the brain's own running scale of forecast error: the same
quantity the arousal law thresholds to decide whether a moment is routine. An outcome inside
the usual scale moves the policy little; an outcome well outside it moves the policy by the
full step. Learning slows as a skill becomes reliable and resumes under real surprise, with
nothing counted and no second set of weights. The exact law (the ratio, its floor, the scale's
rate) is declared on the branch before the first run; each constant is a gene with the founder
as the control.

The simpler controls that must be beaten or matched under the same gate: the founder's
constant step; the ring step (a fixed smaller step after the nursery, the measured remedy);
`memory_rate` 0.2 after the nursery; the frozen copy; uniform random.

## The gate

1. **The ring under melee pay** (`competing_skills.py --ring-pay`, six development seeds, then
   fresh seeds): the skill acquired in the nursery survives the ring at the nursery's own actor
   step as well as it survives at the ring step, while the ring's competing skill is acquired
   as well as at the gene step. Readings from saved copies with the trace carried and reset, the
   per-skill dopamine and the per-projection drift as in the README.
2. **The chain chamber's over-drilling reading** (`benchmarks/chain`, the `taper` arm's
   protocol from `CREDIT_PROTOCOL.md`): acquisition at the first passing probe, then the same
   reading at three and five times that budget, held.
3. **The arena's sixty-fight protocol**: twenty-fight persistence at the gene step as good as the
   ring step's, with the fight statistics moving. The arena is a private repository; its
   summary numbers are reported here as the issue reports them.

## Preservation

The gene must not buy retention with rigidity. The odour nursery (`benchmarks/reversal`) is the
witness: acquisition, reversal and return at exposures 300 and 1,000 at the founder against the
gene; a life that no longer reverses fails the track. The key-door nursery's acquisition at
delays two and five and the Patch World founder's behaviour are the other witnesses. Composed
defaults, settlement, memory and the teaching law are unchanged; a brain with the gene off is
byte-identical in every life and checkpoint.

## Deliverables

1. The gene in `src/cadence/learning.py` with tests (founder identity, the scaling under a
   declared sequence of errors, host and device agreement, save and load).
2. Receipts for the three gates on development seeds, then one confirmation on fresh seeds.
3. README sections here and in `benchmarks/chain`, the CHANGELOG entry, the issue's status.
   No default changes.
