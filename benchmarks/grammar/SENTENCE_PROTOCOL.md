# The whole sentence in the hidden layers: declared protocol (cadence#308)

The runs of the build for cadence#308, each declared here and on the issue before its first life.
A failed gate is reported as failed.

## Step 1: the instrument and the floors

Declared on 2026-11-11, committed before the first life. No gate: these are the floors and the
ceilings the later steps are read against.

**The instrument** (`benchmarks/grammar/meaning.py`, `plan --floors`, `lives`, `read --floors`).
- **Lessons** (`"train": "sayback"`): 4,000 batches, one half the first grammar's mix heard as
  concept moments, one half clauses of trained combinations heard as their words and said back at
  `GO` (`sayback`). Every lesson and every reading has the recall slots of the heard sentence
  emptied at `GO` (the bar of the earlier protocols), and no arm but the reference holds a
  `message`.
- **Said:** the chamber's panels heard as concepts (`clause`, `clause_held`, `roles`, `dyck`,
  `path`, `clause_long`) and the say-back panels heard as words (`back` of trained combinations,
  `back_held` of held-out ones, `back_long` of three and four distractors), whole sentences in a
  closed loop.
- **Held:** linear probes of every clause component (agent's noun and number, verb, the verb's
  form, patient's noun and number, each distractor) fitted on trained combinations and scored on
  held-out ones, after the last heard moment, at `GO` and at the last word, from the settled
  association, the carried trace, and where an arm holds a store, the store and the store read
  with the association (`joint`). Heard as concepts and heard as words.
- **Controls:** the chamber's LSTM and transformer taught the same lessons, online (one pass) and
  offline (to the stopping rule), saying the same panels, their hidden state probed at the same
  moments.

**Founders** 10628600 to 10628605 (six, no life in any lane).

| Arm | The brain | Parameters |
| --- | --- | --- |
| `default_64` | the 0.97.0 default (64 association neurons), words said by their parts, the recall read barred | 30,692 |
| `plain_256` | 256 association neurons, every row taught, the recall read barred: the working trace carries the association's last state | 152,804 |
| `slow_256` | `plain_256` with `working_memory_decay=0.8`: the carried state at a leak | 152,804 |
| `both_256` | the reference with a fixed code: `message=280`, the read of one block, also read with the message silent | 595,536 |

Controls at `plain_256`'s count (LSTM 153,138, transformer 154,770 parameters) and at `both_256`'s
(596,598 and 592,058).

**Expectations.** `default_64`, `plain_256` and `slow_256` say .00 to .10 of every clause panel
heard either way, and their settled association holds the agent's noun under .80 after the last
heard moment. `both_256` says clauses heard as concepts at .97 and over and clauses heard as words
under them; its store read with the association holds every component at .95 and over, its
association alone under .60 for the patient. The LSTM and the transformer offline say every
trained and held-out panel at .95 and over, hold every component at .95 and over, and say .00 of
the longer clauses.
