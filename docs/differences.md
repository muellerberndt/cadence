# How a patch net differs from a feed-forward network with backprop

Both are networks of weighted connections that learn from examples. Almost everything
else is different, and the differences are the point of the library.

## Side by side

| | feed-forward network, backprop | patch net, free/nudged rule |
|---|---|---|
| **what a unit holds** | one number, computed once per pass | a potential and an activation that keep updating until they stop moving |
| **how an answer is made** | one forward pass: layer by layer, input to output, no return | a settlement: every owner repairs its own patch from its inbox, tens of times, until the whole net is at rest |
| **direction of influence** | forward only during inference | both ways: output owners feed back on hidden owners through the same seams, so an answer is a joint rest state, not a chain of function calls |
| **where the goal enters** | at the loss, after the forward pass, as an error signal | as a *nudge*: an extra drive on the output owners during a second settlement |
| **how credit reaches a hidden unit** | a separate backward pass sends `dL/dh` down a transposed copy of the weights; this needs a global controller that has stored every layer's activations | the nudge moves the output owners, the feedback seams move the hidden owners a little, and the hidden owners' own change *is* the credit; nothing is stored and nothing is transposed |
| **what a weight update reads** | the gradient, a quantity computed elsewhere and delivered to the weight | two local numbers: the product of its own two endpoints' activations in the two phases |
| **weight symmetry** | not required; forward and "backward" weights are the same matrix used twice by the controller | required, and built in: one seam carries both directions with one scale |
| **is the update a gradient?** | exactly, by the chain rule | yes, in the limit of a small nudge and converged phases, by equilibrium propagation; the tests check it against finite differences |
| **cost of one update** | one forward pass plus one backward pass | one free settlement plus two nudged settlements, each tens of steps |
| **parameters** | weights and biases | seams and biases; a seam counts once |
| **what has to be true of the net** | differentiable | symmetric seams, responsive owners, converged phases |
| **what a trained net is** | a function | a dynamical system with a learned rest state; you can watch it settle, interrupt it, ablate an owner mid-way, or clamp any owner, and the same rule applies |

## The same thing, said with the equations

A one-hidden-layer network computes `h = f(W₁x)`, `y = g(W₂h)`, then backprop computes
`δ₂ = dL/dy`, `δ₁ = (W₂ᵀ δ₂) ⊙ f'`, and updates `ΔW₂ ∝ δ₂ hᵀ`, `ΔW₁ ∝ δ₁ xᵀ`. The
quantities `δ` exist only in the controller; no unit "has" them.

A patch net with the same shape settles `v ← v + dt (−v + W s + clamp + bias)` with `W`
symmetric (input→hidden and hidden→output seams, each with its reverse) until `s` stops
moving: that is `s⁰`. Then it settles again with `beta (target − softmax(s_out/T))` added
to the output owners: `s⁺` (and `s⁻` with the opposite sign). The update of the seam
between owners `i` and `j` is `eta · (s⁺ᵢ s⁺ⱼ − s⁻ᵢ s⁻ⱼ) / (2 beta)`. That single line does
the work of both `δ` equations, and it does it with numbers each owner already has. The
[learning](learning.md) page walks one update through a six-owner net with every value.

## What that buys, and what it costs

**Locality.** Every overlap's update is a function of its own two endpoints. The
reference engine in `cadence.reference` settles a net one owner at a time with a message
ledger and `conformance` checks any backend against it, so "no owner computed anything it
could not see" is a verified property of every result, learning included.

**One machine.** Inference and learning are the same settlement with and without a nudge.
There is no second pathway, no stored activations, no transposed weights. That is what
makes the rule a candidate for hardware where units are physical and a global backward
pass is not available.

**Sample efficiency.** On every rung of the examples the rule reaches the accuracy of a
same-sized backprop network in fewer passes over the data (digits: 20 epochs against 50;
MNIST: within 0.4 points at 10 epochs), and on Pong it learns more from the same rollouts.

**Wall-clock.** A settlement is tens of steps and an update needs three of them; a
forward-and-backward pass is two. On a CPU that is a factor of ten to a hundred in time
for the same accuracy, and no example hides it. On an accelerator with large batches the
factor shrinks; on hardware that settles natively it disappears.

**Parameters.** No advantage. A seam is a weight. The examples report parameter counts
next to the baselines' and they match.

## What is not different

Both learn from examples by moving weights along a gradient. Both need the same
ingredients to generalise: enough data, a validation split for selection, a held-out
test read once. A patch net is not a shortcut around any of that, and the examples treat
it with the same discipline as the baselines they are measured against.
