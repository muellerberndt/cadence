# In-context succession: does a successor memory continue a sequence heard once?

Declared 15 October 2026, before its first run, for issue
[#201](https://github.com/muellerberndt/cadence/issues/201) and as the sequence form of
[the token-binding protocol](BINDING_PROTOCOL.md). Instrument: `sequence_binding.py`;
contracts: `test_sequence_binding.py`.

## Question

Continuing a phrase from its own beginning, a tune from its prime or a word from its first
sounds needs what followed this event earlier in the same stream, a binding written while
hearing. A table over transitions learned in the weights answers a transition it never trained
with a successor it trained. Does a memory that writes each event keyed by the event before it,
read by the present event, let the composed brain answer transitions it never saw in training?

## Task

Inputs: three markers (`seq`, `ask`, `neutral`) and nine tokens. A trial shows five distinct
tokens, one event `seq + token` each, then a cue `ask + token` naming a token of the sequence
other than the last; the answer is the token that followed it in this trial, one output slot
of nine choices. Every sequence is drawn fresh. Each founder holds out one successor per
training token (a permutation of the eight training tokens without fixed points): those
transitions never occur in training, while every training token occurs before and after other
tokens. The ninth token never occurs in training. Sixteen trials per round, one lesson per
round at the cue.

## Learners, one life per founder and arm

| Arm | Memory | Lesson at the cue |
| --- | --- | --- |
| `graph` | the composed brain's working trace only | `act` (rows whose own act differs) and `every` |
| `event` | plus a recall region read by a memory that writes each event keyed by itself (a control: the cue recalls its own event) | `act` and `every` |
| `successor` | plus a recall region read by a memory that writes each event keyed by the event before it (Hebbian outer product, no decay within a trial, amplitude 3, projection scale 1.2) | `act` and `every` |
| `gru` | supervised online GRU, 64 hidden units, cross-entropy through the trial | one step per round |

The composition is the one of the token-binding protocol with nine motor choices.

## Readings

At 64, 256 and 1024 rounds on copies of each life. Panels of 128 trials: `train`; `held_out`,
the cued transition held out (the trial's only held-out transition); `long`, eight tokens; and
`new_token`, the ninth token at the cued position. Conditions: intact, the memory erased before
the cue, the working trace erased before the cue.

## Gates

Per founder at 1024 rounds, `successor` arm, `act` rule: held-out accuracy at least .75, and
with the memory erased within .15 of chance (1/8). The development run passes at 5 of 6
founders (0 to 5). The same readings for `graph`, `event` and `gru` stand beside it.
Confirmation founders 100 to 111 stay unopened until code, protocol and gates are frozen.

## Declared expectations

`graph` answers held-out transitions with a trained successor of the cue (below chance).
`event` fails the held-out panel: its recall holds the cue's own event. `new_token` tests
whether the association region reads a recalled successor whatever the cue's token is; the
successor of the new token is a trained token, so its recall is in trained coordinates.

## Scope

Instrument level. A pass licenses a successor read as part of the witnessed-content library
candidate; it establishes neither a learned query nor chained recall, and says nothing yet
about the C64 row loop or the spoken-word stream, where the events are graded patterns rather
than one-hot tokens.

```sh
PYTHONPATH=src python benchmarks/binding/sequence_binding.py --out OUT.json --workers 2 \
  --seeds 0 1 2 3 4 5 --budgets 64 256 1024
```

## Amendment 1, after the development run (15 October 2026)

The development run meets the declared gate on the `successor` arm under lessons judged by the
act (the readings are retained outside this repository). Declared before the confirmation run:

- Founders 100 to 111, unopened; 1024 rounds; panels, conditions and the gate as above.
- Arms: `successor` under `act` and `every`; `graph` under both rules; `gru`.
- The confirmation passes for a lesson rule when at least 10 of the 12 founders pass with the
  successor memory; the rules are read separately. A pass licenses the successor read as part of
  the witnessed-content library candidate, declared with its preservation witnesses before any
  library change; it says nothing yet about graded events (the C64 row loop, the spoken-word
  stream).

```sh
PYTHONPATH=src python benchmarks/binding/sequence_binding.py --out OUT.json --workers 2 \
  --learners successor graph gru --rules act every \
  --seeds 100 101 102 103 104 105 106 107 108 109 110 111 --budgets 1024
```
