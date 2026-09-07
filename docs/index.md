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
| [pages](pages.md) | putting a trained net into a browser page that settles it live |
| [protocols](protocols.md) | held-out tests, predicates with preconditions, the shuffled control, gain selection |
| [backends](backends.md) | CPU and torch, precision, the dense transport |
| [receipts](receipts.md) | what a verified result is, and what goes in one |
| [api](api.md) | every public class and function, module by module |

The worked, runnable versions of everything in [learning](learning.md) and [games](games.md)
live in [cadence-examples](https://github.com/muellerberndt/cadence-examples): digits, MNIST,
Connect Four, Pong, each with a tutorial, a script, a receipt, and for the games a page.
