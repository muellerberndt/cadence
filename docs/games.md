# Learning to play a game

Two ways a patch net learns to act, both with the same free/nudged rule as
classification. The full, runnable versions with receipts and playable pages are
[04 Pong](https://github.com/muellerberndt/cadence-examples/tree/main/04_pong) in
cadence-examples and [03 Connect Four](https://github.com/muellerberndt/cadence-examples/tree/v0.5.0/03_connect_four)
at its tag v0.5.0; this page is the mechanism.

## Acting is a settlement

A game state becomes a clamp on the input owners: a board as one owner per cell per
side (1 where a disc sits), a screen as one owner per pixel (its brightness). The net
settles under that clamp; the output owners are the actions, and their rest activations
are the net's preferences. To act, take the most active legal output (a game), or draw
from `softmax(s_out / T)` (a policy that explores). There is no search inside the net
and no memory beyond what the clamp carries: two consecutive frames if the game needs
velocity, as Pong does.

## Way one: imitate a teacher (Connect Four)

If something can tell you the right move for a position, learning to play is
classification, and everything in [learning](learning.md) applies unchanged:

1. **Positions.** Play games with cheap players that make some random moves, so the
   positions are varied and plausible. Deduplicate.
2. **Labels.** For each non-terminal position, the move a deeper search prefers for the
   side to move. The teacher's depth is the ceiling of what the net can learn.
3. **Encoding from the mover's side.** One plane of the mover's discs and one of the
   opponent's, so one net plays both colours. Mirror positions left-right for free
   augmentation; mirror the labels with them.
4. **Learn.** Free phase under the position, nudged phases toward the teacher's column,
   local update. Select the hidden size on a validation split of the positions.
5. **Play.** Settle, mask illegal columns, take the most active output. Measure strength
   by matches against fixed opponents, alternating who starts, from random openings (a
   deterministic net against a deterministic search would otherwise replay one game).

What to expect: agreement with the teacher is the training signal, and play strength is
what it buys. A one-hidden-layer net that agrees with a depth-4 search on about half the
positions beats a random mover almost always and loses to a two-ply search almost always,
because it misses forced blocks often enough for an opponent that never does. An MLP of
the same size trained by Adam on the same positions lands in the same place. Imitation of
a shallow search is exactly that strong, whichever rule learns it.

## Way two: learn from reward (Pong)

When nothing says what the right action is, only how things went, the rule still applies
with one change: **the target of the nudge is the action the net took, and the strength
of the nudge is that action's advantage.**

```python
# one iteration
frames, actions, returns = rollout(policy, envs, steps)         # act by sampling softmax(s_out / T)
advantages = (returns - returns.mean()) / returns.std()
learner.step(drive(frames), actions, weight=advantages)         # free, +beta·A, -beta·A, update
```

`Nudge.weight` scales the nudge row by row. For a transition whose action paid
(`A > 0`) the output owner of that action is pulled up; for one that cost (`A < 0`) it is
pushed down; the update on every seam is then `A · d log π(a|s) / dW`, summed over the
batch, which is the REINFORCE policy gradient. The goal, a scalar reward, enters through
the nudge alone.

### Setting up the reward so credit lands on the right step

The paddle's job in Pong is to be where the ball will be. The plain reward, `+1` for a
return and `−1` for a miss, arrives only at the end of a rally, many steps after the
moves that mattered, and a policy-gradient learner with that reward and a modest budget
learns something crude: "ball high, go up; ball low, go down", in absolute rows, ignoring
where its own paddle is. It returns some balls and looks bad.

The fix is *potential-based shaping* (Ng, Harada, and Russell 1999): add to each step's
reward the change in a potential, here how much the paddle centre's distance to the
ball's row shrank during the step. This changes no optimal policy, because along any
trajectory the shaping terms telescope; it only tells the paddle at each step whether
that step helped. With a short credit horizon (`gamma = 0.5`) the advantage of a move is
then dominated by what that move did, and the net learns the relative rule, "move toward
the ball's row", which is what tracks. The Pong tutorial shows the two policies side by
side as tables of action against ball-minus-paddle offset.

### Two frames

A single frame does not say which way the ball is moving, so a paddle that reads one
frame cannot anticipate a diagonal ball and both learners return fewer than 40% of them.
Clamp the previous frame alongside the current one and the direction is visible. The
page does the same: it keeps one earlier frame, nothing more.

### Evaluating a policy

Greedy play (most active output) on fresh seeds until a fixed number of points have
ended; report balls returned over balls faced, and returns per point. Cap rally length,
or two competent paddles can keep a horizontal ball going forever. Put a same-sized
network trained by backprop REINFORCE with Adam through the same rollouts, reward, and
evaluation, and report both, with wall-clock.

## What the reward rungs found

Across Pong and cart-pole the pattern is the same: the nudged rule learns from reward,
and learns less than REINFORCE with Adam from the same rollouts. Pong from two frames
plateaus near 79% of balls returned for every variant tried (nudge strength, settle
tolerance, batch size, a local momentum, a local per-seam normalisation, potential-based
and immediate credit), against 86 to 93% for the baseline, and the same net taught the
tracker's own moves instead returns 95%. Cart-pole balances for 154 steps against 392.
The gap is not representational and not in the rule as a gradient estimator on
supervised targets, where the ladder shows parity; it is in the interaction between a
noisy, advantage-weighted target and a small-nudge estimate: a large advantage times the
nudge strength leaves the regime where the contrast is a gradient, a small one leaves
the contrast in the settle tolerance's noise, and Adam's per-parameter step sizes are
what the baseline has that the rule does not. Two consequences for practice: keep
`beta · |advantage|` below about 0.3 (clip advantages or lower `beta`), and prefer a
dense, potential-based reward with a short credit horizon, which is the only change that
moved the Pong policy from an absolute rule to a relative one.

## Putting a trained net in a page

Both game pages settle the net in JavaScript, owner by owner, with the same rule; see
[pages](pages.md).
