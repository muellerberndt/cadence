# Symbols chamber: the parrot that starts with no words

A brain with eyes, an ear and a voice (`Brain.compose(senses={"eyes": ..., "ear": ...},
code=2048, bound=4096, voice="ear")`) lives through what a trainer does with a young parrot:
the trainer says words, names things, asks a question and pays a reward for a right answer. The
chamber reads whether a sound comes to stand for a thing and a phrase for what it asks, through
the brain's records of its bound moments ([`cadence.bound`](../../src/cadence/bound.py), the
guide's [sounds as symbols](../../docs/memory.md#records-of-bound-moments-opt-in)), for
[#342](https://github.com/muellerberndt/cadence/issues/342) and
[#344](https://github.com/muellerberndt/cadence/issues/344).

`world.py` holds the world and the trainer: eight kinds of things in two colours each, at places
around the body, seen as tuned cells (`cadence.senses.Things`); seventeen words (the kinds, five
colours, two questions), each a pattern of log-mel energies that a speaker says faster or slower,
higher or lower, louder or softer and with the room's noise, heard as one code
(`cadence.senses.Ear`). The trainer hears what the parrot says as the nearest word. `chamber.py`
holds the life and the readings; its docstring gives each step.

```sh
PYTHONPATH=src python benchmarks/symbols/chamber.py --founders 0 1 --out dev.json
python -m pytest -q benchmarks/symbols
```

## The life

1. **Listen:** every word said three times, nothing in view.
2. **Naming:** a thing held up and its kind's name said, eight times a kind, each time at another
   place and in either colour.
3. **Reading Y1h, Y2** on that brain.
4. **Reward** (a fresh brain of the founder after Listen alone): four things, one after the other,
   each until the parrot names it right three times running or 80 exchanges, a reward of 1.0 for
   the right name and 0.0 otherwise; then the thing at places no exchange used (**Y1**); then
   "what colour is this" with four things, the same way. The same exchanges unpaid on another
   fresh brain.
5. **Controls on the same exchanges:** a lookup table keyed by (the thing, the word heard), and a
   delta-rule readout of the brain's code taught by the reward.
6. **A heard name draws the focus (N):** a brain with a map of things whose values are the eyes'
   fields and `surprise=True`, after Listen and Naming; three things appear one by one; the
   trainer says the name of one of the first two. Twelve rooms, each with the named thing in view
   and behind a screen; the same after Listen alone.

## Declared protocol

Declared on [#342](https://github.com/muellerberndt/cadence/issues/342) before the first life of
its founders. Founders **1010100 to 1010111** (twelve, fresh). Development used founders 0 to 3.
One life a founder on the CPU backend (`backend="cpu"`), one thread.

| Gate | Reading | Bar |
| --- | --- | --- |
| S1 | Y2, records: the kind read from the shape and surface a kind's name heard alone brings back (40 takes) | .90, on 10 of 12 |
| S2 | Y2, state: the kind read by a linear probe from the association's settled state after a name heard alone (the probe fitted on things seen alone, 128 views) | .90, on 10 of 12 |
| S3 | Y2 controls: the same probe with the records barred, and with the records' values dealt to other keys | each at most .25 (chance .125), on 10 of 12 |
| S4 | Y3, names: each of the four things named right three times running within 80 exchanges | all four, on 10 of 12; unpaid: the share right at most .15 (chance about .06) |
| S5 | Y3, the phrase "what colour is this": each of four things answered right three times running within 80 exchanges | all four, on 10 of 12; unpaid as S4 |
| S6 | Y1: after S4, the four things at places no exchange used (20 views), named right | .90, on 10 of 12 |
| S7 | N: the focus moves to within 0.25 of the named thing, in view and behind a screen (12 rooms each) | each .90, on 10 of 12; after Listen alone at most .10 |

Reported without a bar: Y1h (a thing named from the naming moments alone, no reward), the
exchanges each learner took (the brain, the table, the delta rule), the records held, the
parameters and the seconds a life.

**What is designed, by name.** The world, the trainer and its ear for words; the records' rule
and its constants (`BoundRecords`: the read by the nearest records that hold a field, the
heaviest group, a refused record weighing against its sound, a reward's tellings, the babble);
the name's pull on the focus (`NAMED`). **Learned:** every record, its strength, what each sound
brings back, what the parrot says, which thing a name draws the focus to.

**Expectations, from development (founders 0 to 3).** S1 1.00 on every founder; S2 .68 to 1.00
(one founder of four under .90 in the first development pass, 1.00 and .98 in the second);
S3 .03 to .28 barred and .00 to .10 dealt; S4 and S5 each thing learned in 4 to 71 exchanges
with one phrase item past the budget on one founder, unpaid .05 to .15; S6 .60 to 1.00 (one
founder of two at 1.00, one at .80 in the second pass); S7 .83 to 1.00 taught and .00 untaught.
S2, S5 and S6 may miss on more than two founders.

**The located limit (Y1h).** A thing seen alone is named from the naming moments at .82 to .90:
the read weighs the thing's place and colour as it weighs its form, and a thing's form decides
its name. Records alone do not learn which field a name follows; the reward of S4 does it for
the things it pays.

## Readings

Pending the declared run.
