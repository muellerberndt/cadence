# Delayed credit: reward at the end of an episode

This chamber asks whether a reward that arrives at the end of an episode reaches the decision
that caused it, 10, 50 and 200 moments earlier, under reward alone. It is the instrument of
[issue 283](https://github.com/muellerberndt/cadence/issues/283), prerequisite 2 of the
[roadmap](https://github.com/muellerberndt/cadence/issues/109) before the Doom player
([issue 118](https://github.com/muellerberndt/cadence/issues/118)). It measures settings of
the released library against the composed founder beside two conventional learners; it
establishes no default.

## The two worlds

`tasks.py`. Two actions in both worlds; nothing is paid before the episode's last act.

| World | Inputs | An episode at delay `k` | Outcome |
| --- | --- | --- | --- |
| Delayed T-maze (`tmaze`) | 9: the cues `C0`, `C1`, the corridor, six distractors `D0` to `D5` | the junction, which shows the cue; then `k` corridor moments | +1 for the cue's arm, -1 for the other |
| Delayed key-door (`keydoor`) | 11: the key on the left `KL` or right `KR`, the corridor, the six distractors, the door, the pouch | the key room; then `k - 1` corridor moments; then the door | opening with the key +1, opening without it -1, the door left shut 0 |

The decision whose credit is read is the first act: the T-maze's choice at the junction, the
key-door's reach (the reach to the key's side takes it). It is followed by `k` further acts
before the outcome arrives. Each corridor moment shows the corridor token or, on a quarter of
the moments, one of the six distractors drawn uniformly. In the T-maze the corridor is the same
whichever way the choice went, so after the junction nothing the brain senses tells a right
choice from a wrong one. In the key-door the pouch input shows the key held from the moment
after the reach, so a value of the situation can see it. Which action is right is
counterbalanced across founders: bit 0 of the founder flips the choice and the reach, bit 1
the door's opening action.

The cue is shown at the moment of the decision, so the chamber reads credit alone; carrying a
cue across moments to an answer that does not show it is read by the
[two-cue chamber](../credit/README.md).

The schedule shows both cues in a random order in every block of two episodes, each with its
own corridor. The reading panel holds both cues with four corridors drawn from the panel seed.

## The learners and the controls

| Kind | What it is |
| --- | --- |
| `brain` | `Brain.compose(inputs, 2, seed=founder, arousal=True)` with the arm's settings, every act the library's sampled act through `step`. The outcome arrives with the next episode's first observation together with `done`, so the library clears the carried context between episodes. Every outcome is learned from. |
| `lstm` | An LSTM actor-critic (`baselines.py`), its state cleared when an episode begins; one update per episode with generalized advantage estimation and exact gradients through every moment of the episode, Adam, an entropy bonus. Torch on the CPU. |
| `tdlambda` | The online TD(lambda) actor-critic of [`benchmarks/competitors.py`](../competitors.py) (an Elman cell, eligibility traces over every weight, an update every moment). |
| `random` | Uniform actions. |
| `solver` | The exact solver: every action from the present observation alone. It answers every episode right, so every outcome is reachable from what the brain senses. |

Both conventional learners live the brain's episodes in the same order with the same
observations and outcomes, and are sized to the brain's parameter count on the world
(`Brain.parameters()`, the default composition: 15,962 on the T-maze, 16,772 on the key-door).

| Control | Delivered outcome |
| --- | --- |
| `yoked` | the outcome a uniform-random partner earned on the same episode, whatever the learner did |
| `shuffled` | one of the learner's own earlier outcomes, drawn uniformly |
| `frozen` | none: the newborn is read at every budget |

## Readings

At every budget the panel is read on a copy of the life that never changes it: each panel
episode from a fresh stream, every act greedy. `choice` is the share of panel episodes whose
decision is right, `door` the share of key-door episodes whose door act is right for the pouch
it shows. The training log, per window of 128 episodes: the share of decisions right as acted
and the policy's probability of the right decision when it was taken. For the brain the credit
diagnostics: the critic's value at the decision and at the moment after it, the dopamine of
the decision's transition and of the outcome, apart for right and wrong decisions, the
decision's share of the actor's eligibility when the outcome arrives (its contrast decayed to
the outcome over the trace's norm) and their cosine.

Work: the brain's multiply-accumulates are counted analytically, each settle sweep multiplying
each synapse once (the act's free phase, the next state's settle and the two nudged phases of
the eligibility) and three more per synapse for the trace and the step. The LSTM's forward pass
per moment, its backward pass at twice the forward, Adam at ten per parameter per update. The
TD learner's own ledger. Seconds are wall-clock seconds of one process.

## Run

From the repository root, with one BLAS thread per process:

```sh
PYTHONPATH=src python benchmarks/delayed_credit/chamber.py --kinds brain \
    --arms default held --tasks tmaze keydoor --delays 10 50 --workers 4 --out DIR/lives.jsonl
PYTHONPATH=src python benchmarks/delayed_credit/chamber.py --kinds lstm tdlambda \
    --tasks tmaze keydoor --delays 10 50 200 --out DIR/baselines.jsonl
python benchmarks/delayed_credit/chamber.py --summary DIR/lives.jsonl
python -m pytest -q benchmarks/delayed_credit
```

## Where the credit is lost (development, 0.96.0)

Development founders 0 to 5, every brain through `step` from every outcome, 4,096 episodes, the
choice read greedy at the last budget.

| learner | world | delay | parameters | founders at .90 | choice | the decision's share of the actor's eligibility at the outcome | outcome dopamine, right / wrong | critic value before the outcome, right / wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default | T-maze | 10 | 15,962 | 5 of 6 | .92 | .15 | +.43 / -1.00 | +.56 / +.56 |
| default | T-maze | 50 | 15,962 | 0 of 6 | .50 | 9.5e-7 | +.99 / -.99 | -.02 / +.02 |
| default | key-door | 10 | 16,772 | 5 of 6 | .92 | .079 | +.12 / -.57 | +.85 / +.37 |
| default | key-door | 50 | 16,772 | 2 of 6 | .67 | 2.9e-6 | +.34 / -.32 | +.17 / -.01 |
| `recall=False` | T-maze | 50 | 5,022 | 0 of 6 | .42 | 6.7e-7 | +.98 / -.98 | -.01 / +.01 |
| `actor_gamma=1, actor_lam=1` | T-maze | 10 | 15,962 | 1 of 6 | .58 | .33 | -.15 / -.20 | +170 / +199 |
| `actor_gamma=1, actor_lam=1` | key-door | 10 | 16,772 | 1 of 6 | .58 | .33 | +.40 / +.20 | -728 / -872 |
| `actor_hold=1.0` | T-maze | 50 | 15,962 | 2 of 6 | .67 | .10 | +.92 / -.97 | +.06 / +.04 |
| `actor_hold=1.0` | key-door | 10 | 16,772 | 2 of 6 | .67 | .42 | +.25 / -.26 | +.44 / -.10 |
| a tenth of the step, `actor_hold=1.0` | T-maze | 50 | 15,962 | 6 of 6 | 1.00 | .059 | +.84 / -1.00 | +.16 / +.15 |
| a tenth of the step, `actor_hold=1.0` | key-door | 50 | 16,772 | 3 of 6 | .75 | .16 | +.30 / -.48 | +.36 / +.13 |
| a tenth of the step, `memory_choice=True` | T-maze | 10 | 20,040 | 6 of 6 | 1.00 | .057 | +.32 / -1.00 | +.68 / +.70 |
| a tenth of the step, `memory_choice=True`, `actor_hold=1.0` | T-maze | 50 | 20,040 | 6 of 6 | 1.00 | .13 | +.87 / -1.00 | +.13 / +.13 |
| a tenth of the step, `memory_choice=True`, `actor_hold=1.0` | key-door | 50 | 20,846 | 5 of 6 | .92 | .50 | +.75 / -.81 | +.25 / +.12 |

A tenth of the step is `actor_eta=0.1, actor_eta_bias=0.01`.

- **The eligibility's decay loses the credit.** The composed actor's eligibility decays by
  `actor_gamma * actor_lam`, .72 a moment; fifty moments after the decision its contrast is a
  millionth of the eligibility the outcome's dopamine meets. The dopamine itself has the right
  sign and its full size.
- **The critic does not carry it.** After the T-maze's junction the value is the same for a right
  and a wrong choice: nothing it reads tells them apart. The key-door's critic sees the pouch, and
  its discount leaves .006 of the door's value at the moment after the reach.
- **The discount cannot be lifted alone.** The trace's decay is the critic's discount times
  `actor_lam`: at 1 and 1 the decision's contrast stays whole and the critic's values diverge.
- **The latch is a term at the composed step** ([issue 296](https://github.com/muellerberndt/cadence/issues/296)).
  On the failing lives one action answers both cues and the eligibility's norm falls 15 to 20
  times within the first quarter of the life. A trace held through the episode at the composed
  step sums every act's contrast into each outcome's step, and the readout reaches a rail sooner.
  At a tenth of the step the hold learns the T-maze at 50; on the key-door the door's act, credited
  one moment before its outcome, latches on three founders. The episodic memory's records of the
  door's outcome keep it there.
