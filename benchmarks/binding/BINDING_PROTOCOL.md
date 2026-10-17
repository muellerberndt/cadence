# Token binding: does a content memory turn a table into a rule?

Declared 15 October 2026, before the development run, for issue
[#201](https://github.com/muellerberndt/cadence/issues/201) (selection that generalizes).
Instrument: `token_binding.py` in this directory; contracts: `test_token_binding.py`.

## Question

In the named-fact selection task the composed brain acquires selection through its working
trace and answers held-out fact combinations at chance: its answer is a table over trained
trace patterns. A learner that binds a value to its name when it is seen and reads it back by
the name at the query needs to learn only the rule. Does a content memory give the composed
brain that binding, so that it answers name-value pairs never shown in training, while the
existing graph answers from a table? A supervised online GRU stands beside both.

## Task

Inputs are shared tokens: three markers (`is`, `ask`, `neutral`), five name tokens and eight
value tokens. A trial shows two facts, each one event `is + name + value` with distinct names,
one neutral event, and a query `ask + name` that repeats the name token of one fact. The answer
is the value bound to that name: one output slot of eight choices. Sixteen trials per round,
one stream each, one lesson per round at the query.

Each founder holds out two values per training name, each value for exactly one name, drawn
from the founder seed: every value token and every training name token occurs in training, and
no held-out name-value pair does. The fifth name token never occurs in training.

## Learners, one life per founder and arm

| Arm | Carrier of the facts | Lesson at the query |
| --- | --- | --- |
| `graph` | the composed brain's working trace | `act`: rows whose own act differs (the learner's `surprise` rule with the act's answer); `every`: every row |
| `content` | the same composition plus a recall region read by the association region through a plastic projection (initial scale 1.2); a content memory (`FastSynapses`, sensory to recall, Hebbian outer product, no decay within a trial) writes every event as heard, keyed by itself, and is read by the present observation before each settle at amplitude 3 | `act` and `every` |
| `content_delta` | as `content` with the delta rule (unit keys, residual writes), a declared comparison of the store's rule | `act` and `every` |
| `gru` | a supervised online GRU, 64 hidden units, a softmax over the values, cross-entropy through the trial, rate 1.0, gradient norm bounded at 5 | one step per round |

The composition is `Brain.compose(16, 8, modules=(64,), lateral=0.0, seed=founder)` at its
founder genes, built region by region so that the recall region is appended last (a contract
test checks that without it the connectome equals `Brain.compose`). Nothing changes the
library.

## Readings

Each life is read at 64, 256 and 1024 rounds on copies, never changing the life. Panels of
128 trials each, from the founder's panel seed:

| Panel | What is new relative to training |
| --- | --- |
| `train` | nothing: the training distribution |
| `held_out` | the queried fact is a held-out name-value pair |
| `three_facts` | three facts in a trial (training shows two) |
| `long_delay` | four neutral events (training shows one) |
| `new_name` | the queried fact carries the name token never shown in training |
| `restated` | three facts; the first fact's name is stated again in the last with another value, and the query asks that name: the later value is the answer |

Conditions on every panel: intact; the content memory erased before the query; the working
trace erased before the query; the content memory of a donor transplanted before the query,
the donor trial having the same names, order and query and every value drawn anew (the
queried one different), read as the share of answers that follow the donor.

## Gates

Per founder at 1024 rounds, for the `content` arm under the `act` rule:

1. held-out accuracy at least .75;
2. with the content memory erased, held-out accuracy within .15 of chance (1/8);
3. the transplant: the answers follow the donor's value on at least .75 of held-out trials.

The development run passes when at least 5 of the 6 development founders (0 to 5) pass all
three. The same gates read for `graph` (its trace erased in place of the content memory) and
for `gru` (gate 1 only) are reported beside it. Confirmation founders 100 to 111 stay unopened
until the code, this protocol and the gates are frozen; confirmation passes at 10 of 12.

## Declared expectations

- `graph` answers held-out pairs below chance when it answers each name with values trained
  for that name: the signature of a table.
- `content` with the Hebbian store ties the two values of a restated name (about half
  right); `content_delta` favours the later statement except where another fact carries the
  same value token, whose residual it also subtracts.
- `new_name` tests whether the association region reads a recalled value token whatever name
  token comes with it; its initial synapses from the new name are untrained.

These are development expectations, not gates.

## What a pass licenses and what it does not

A pass licenses declaring a witnessed-content store as an opt-in library candidate under the
roadmap's contract for the learned read, with its preservation witnesses. It does not
establish a learned query (the query here is the present observation), chained binding
(an answer that requires a recalled value to become the next cue), role binding beyond one
relation, or sequence organization. A smoke run of one founder at 64 and 256 rounds measured
time and nothing else before this declaration.

## Commands

```sh
PYTHONPATH=src python benchmarks/binding/token_binding.py --out OUT.json --workers 2 \
  --seeds 0 1 2 3 4 5 --budgets 64 256 1024
python -m pytest -q benchmarks/binding
```

## Amendment 1, after the development run (15 October 2026)

The development run is retained as measured, outside this repository. The gated arm (the
Hebbian store under lessons judged by the act) does not meet the declared development gate, so
its confirmation does not follow. The code's gate read the first two criteria; it reads all
three from commit d09a1ac on, and every verdict is recomputed from the receipt.

Development selects a candidate for a separate confirmation: the **delta store**
(`content_delta`), which met the development criterion under both lesson rules and favours a
restated name's later value where the Hebbian store ties. Declared before the confirmation run:

- Founders 100 to 111, unopened; 1024 rounds; the panels, conditions and three criteria as
  above.
- Arms: `content_delta` under `act` and under `every`; `graph` under `act`; `gru`.
- The confirmation passes for a lesson rule when at least 10 of the 12 founders pass all three
  criteria at 1024 rounds with the delta store; the rules are read separately.
- A pass licenses an opt-in library candidate (a content memory written as each event is heard
  and read by the present observation into a recall region), declared with its preservation
  witnesses before any library change. A failure is retained and nothing follows for the
  library.

```sh
PYTHONPATH=src python benchmarks/binding/token_binding.py --out OUT.json --workers 2 \
  --learners content_delta graph gru --rules act every \
  --seeds 100 101 102 103 104 105 106 107 108 109 110 111 --budgets 1024
```

The command runs `graph` under both rules; its `every` arm is read as an extra control.
