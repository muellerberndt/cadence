# Cadence documentation

Cadence is a library for *patch nets*: owners that each hold one patch of state, joined by
declared overlaps, settling to rest by owner-local repair. Read in this order the first time:

| read | to learn |
|---|---|
| [concepts](concepts.md) | what a patch net is, and why the library is shaped as it is |
| [quickstart](quickstart.md) | the seven calls from a wiring to a verified receipt |
| [learning](learning.md) | the free/nudged rule in full: every equation, a worked example with numbers, every knob |
| [differences](differences.md) | how a patch net differs from a feed-forward network trained by backprop |
| [games](games.md) | learning to play: imitating a search, and learning from reward |
| [tasks](tasks.md) | recipes: how each kind of task becomes a clamp, a target, and a readout, tabular and regression included |
| [pages](pages.md) | putting a trained net into a browser page that settles it live |
| [protocols](protocols.md) | held-out tests, predicates with preconditions, the shuffled control, gain selection |
| [backends](backends.md) | CPU and torch, precision, the dense transport |
| [receipts](receipts.md) | what a verified result is, and what goes in one |
| [embodied](embodied.md) | the sense, settle, act, learn loop in a body; several learners in one net; checkpoints |
| [api](api.md) | every public class and function, module by module |

The worked, runnable versions of everything in [learning](learning.md) and [games](games.md)
live in [cadence-examples](https://github.com/muellerberndt/cadence-examples): digits, MNIST,
Connect Four, Pong, text, a sign-writing arm, chorales, cart-pole and the *C. elegans*
connectome, each with a tutorial, a script, a receipt, and for the interactive ones a page.

## What the ladder says the rule is good at

| kind of learning | rungs | result |
|---|---|---|
| supervised, dense low-dimensional input | digits, MNIST, Connect Four, sign writer | parity with a same-sized backprop network in fewer epochs; ten to a hundred times the wall-clock |
| supervised, multi-label (a chord) | chorales | ahead of the same-shape MLP: F1 0.483 vs 0.470, 16.6 vs 24.7 bits per chord |
| supervised, wide sparse one-hot input | text | behind: 3.33 bits per character vs 3.11 for the same-window MLP and 3.08 for a one-layer transformer |
| from reward, three factors with the adaptive step ([reward](reward.md)) | cart-pole | 500 on every seed, the threshold at 40k to 60k steps against 20k to 100k for PPO with an MLP and 82k to 162k with a transformer; 1,716 parameters against 9,155 and 17,443 |
| from reward, three factors | Pong, Hopper | learns, and does not reach PPO: 0.73 of balls against 0.93; 177 at best on Hopper-v4 against about 1,000 |
| a measured, directed wiring under a protocol | *C. elegans* | the local rule cannot teach seams the nudge does not reach; a structural signal survives (5.3 vs 1.5 of 17 ablations), not a behavioural model |
| a stream that drifts, one update per chunk | electricity, synthetic drift | learns and relearns; 0.851 against 0.873 for one-step logistic regression on electricity, 0.796 against 0.878 on the synthetic drift; ahead on one-pass MNIST (0.885 against 0.844) |
| sixteen tabular tasks with the sparse-aware place code | OpenML-CC18 subset | mean 0.913 against logistic 0.906, MLP 0.921, boosting 0.901; ahead of boosting on nine, logistic on six |
