# Chained recall: does a learned read compute its next key?

Declared 15 October 2026, before the first run, for issue
[#201](https://github.com/muellerberndt/cadence/issues/201). Instrument: `chain_binding.py`;
contracts: `test_chain_binding.py`. Companion to [token binding](BINDING_PROTOCOL.md) and
[in-context succession](SEQUENCE_PROTOCOL.md).

## Question

A content memory read by the present event answers name-value pairs never seen in training,
where the existing graph answers from a table. A read keyed by the present event can only
recall what shares a token with the question. Does a learned read, whose key is the brain's
own activity coupled to the memory inside the equilibrium, answer a question whose second step
needs the first recall as its key?

## Task

Inputs: five markers (`link`, `ask1`, `ask2`, `think`, `neutral`), eight source tokens and
eight target tokens. A trial shows three link events `link + from a + to b` in random order (a
chain x -> y -> z and a distractor w -> u, five distinct tokens), one neutral event, the question
`ask1 + from x` (answer y) or `ask2 + from x` (answer z), half and half in training, and one
`think` event; the answer is the act at the `think` event, one output slot of eight choices.
Links are drawn fresh every trial. Each founder holds out one target per source (a permutation
without fixed points): held-out links never occur in training, while every token occurs as a
source and as a target. The question shares a token with the first link only. Sixteen trials
per round, one lesson per round at the last event.

## Learners, one life per founder and arm

| Arm | Memory and read | Lesson |
| --- | --- | --- |
| `graph` | the composed brain's working trace only | `act`: rows whose own act differs |
| `content` | plus the fixed read: each event written as heard, keyed by itself (Hebbian), read by the present event at amplitude 3 | `act` |
| `learned` | plus the learned read: query and recall regions coupled through the same memory inside the settle, coupling 3.0, Hebbian store, the query driven by the cortex alone (the strongest setting of the sizing smoke) | `act` |
| `gru` | supervised online GRU, 64 hidden units, cross-entropy through the trial | one step per round |

The composition is the one of the token-binding protocol on this task's ports
(`modules=(64,)`, `lateral=0.0`, founder genes).

## Readings and gates

At 256, 1024 and 4096 rounds on copies. Panels of 128 trials: `one`, `two`, `one_held`,
`two_held` (the asked path contains a held-out link), `two_more_think` (three think events).
Conditions: intact, the working trace erased, the memory erased (memory arms), each before the
last event.

Per founder at 4096 rounds: `two_held` at least .60 and `one_held` at least .75. The
development run (founders 0 to 5) passes at 5 of 6. Chance is 1/8.

## Declared expectations and decision

- `content` passes `one_held` and fails `two_held`: its key never reaches the second link.
- `graph` answers held-out links from a table, below chance.
- If `learned` passes `two_held` where `content` fails, a read whose key the brain computes is
  justified as a library candidate, with the open design question of competition between
  stored events (the sizing smoke: a read inside the equilibrium without competition is either
  saturated or too weak to compete with the working trace).
- If `gru` passes and `learned` fails, the next candidate is a read with competition between
  stored events.
- If no learner passes `two_held`, the task's budget or encoding is the limit; nothing follows
  for the library.

## Scope

Instrument level; no library change. The learned read's coupling is declared in
`learned_read.py`.

```sh
PYTHONPATH=src python benchmarks/binding/chain_binding.py --out OUT.json --workers 2 \
  --seeds 0 1 2 3 4 5 --budgets 256 1024 4096
```
