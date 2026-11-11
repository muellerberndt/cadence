# A closed-domain conversation chamber: the data and Step 0

Issue: [#305](https://github.com/muellerberndt/cadence/issues/305). Instruments: `world.py` (the
synthetic world and the dataset adapters), `chamber.py` (the brain's lives), `controls.py` (the
matched LSTM and transformer); plans in [`step0/`](step0). This file records the datasets as
downloaded and the Step 0 reading. The declared run follows in its own section, posted on the
issue before its first life.

## The datasets, read against the downloads (11 November 2026)

| | bAbI question answering | bAbI dialog |
| --- | --- | --- |
| Reference | Weston, Bordes, Chopra, Rush, van Merriënboer, Joulin and Mikolov 2016 | Bordes, Boureau and Weston 2017 |
| Archive | `babi_tasks_1-20_v1-2.tar.gz`, version 1.2 (the release's `README.txt`) | `dialog_babi.tar.gz` (folder `dialog-bAbI-tasks`) |
| Source | `https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz` | `https://dl.fbaipublicfiles.com/parlai/dialog_babi/dialog_babi.tar.gz` |
| SHA-256 | `84f5296ab9a1ad0dc9464e08c491d65cd08830fca3acae9ab86f75e0fb81573c` | `bb36155ccd41eac91f806446c5728ee90374e5596156a9f7c1b86f8342cfc383` |
| Bytes | 11,745,123 | 3,032,766 |
| Licence (the archive's `LICENSE.txt`) | Creative Commons Attribution 3.0 Unported, Facebook, Inc. 2015 | Creative Commons Attribution 3.0 Unported, Facebook, Inc. 2016 |

The release's own address for bAbI question answering answers 404 and the Facebook mirror 403;
the archive read here is the copy that Keras's dataset loader names, which holds the release's
`README.txt` and `LICENSE.txt` of version 1.2. Task 6 of bAbI dialog is converted from the Dialog
State Tracking Challenge 2 and is not read here.

**bAbI question answering** (`en/`, the 1,000-question version): 20 tasks, each 1,000 training and
1,000 test questions; one lexicon of 162 words over all tasks and answers (the same for `en-10k/`,
10,000 training questions a task).

| Task | Training stories | Statements | Longest story (lines) | Words | Answers |
| --- | --- | --- | --- | --- | --- |
| 1 single supporting fact | 200 | 2,000 | 15 | 21 | 6 |
| 2 two supporting facts | 200 | 4,338 | 61 | 35 | 6 |
| 3 three supporting facts | 200 | 14,796 | 229 | 36 | 6 |
| 6 yes and no | 200 | 2,066 | 31 | 35 | 2 |

**bAbI dialog** (tasks 1 to 5, restaurant reservation): 1,000 dialogs in each of the training,
development, test and test-OOV splits; a knowledge base of 8,400 lines over 1,200 restaurants;
4,212 candidate bot utterances. The lexicon is closed for the conversation's own words (51 to 55
user words in tasks 1 and 2) and open for the knowledge base's entities (683 to 1,066 user words
in tasks 3 to 5, most of them restaurant names, phone numbers and addresses).

| Task | Bot turns, test | said whole by a training bot turn | every word said by one | test-OOV: whole | every word |
| --- | --- | --- | --- | --- | --- |
| 1 issuing API calls | 5,936 | .832 | 1.000 | .834 | .834 |
| 2 updating API calls | 9,464 | .992 | 1.000 | .789 | .789 |
| 3 displaying options | 10,018 | .749 | .749 | .753 | .753 |
| 4 providing information | 3,498 | .572 | .572 | .570 | .570 |
| 5 full dialogs | 18,398 | .956 | .993 | .678 | .678 |

(`world.dialog_answers`.) In task 4, 43% of the test answers say a word no training bot turn
says: the entity the dialog's knowledge-base lines told. A readout that says answers from a
closed set learned in training holds at most .572 of them; the rest asks for a word said from
what the conversation told.

The archives and this inventory are kept in the lane's S3 bucket; nothing of the data is in git.
The synthetic world (`world.World`) needs no download, and every test reads it.

## Step 0: the 0.97 default with `message` and `longterm`, read before any mechanism

A development reading on founders 0 to 2, with no gate: where the answers of the library as
released fail. Plans in [`step0/`](step0).

- **The synthetic world**: 12 people, 8 places, two facts a session; a nursery of 400 lesson
  conversations of 4 sessions on 16 streams (a quarter of the person and place pairs in no
  lesson); readings of 8 conversations of 12 sessions per stream, without lessons, asked in the
  trained wordings, in held-out wordings, and in the fact's own words up to its turn (`prefix`,
  the records' exact key); and with the records emptied or the message emptied at each answer.
- **bAbI** tasks 1, 2 and 6: five passes over the 200 training stories with lessons, then the
  1,000 test questions.
- **Arms**: `previous_word`, `message`, `records`, `records_session` (`chamber.py`), and the
  LSTM and transformer matched to the `records` brain's parameters, online and offline
  (`controls.py`).

## The declared run: records keyed at every turn (declared 11 November 2026, before its first life)

**The opt-in** (`Brain.compose(longterm=records, longterm_turn=channel)`, `docs/memory.md`, "Turns
in a conversation"), in answer to Step 0's three terms:

1. **The key a question and its fact share.** A turn's key is the rare units of the message's
   codes of the moments heard since the last value: units on in at most `longterm_rare` (.03) of
   the moments the brain has heard. On development founder 0 the function words' units lie at
   .036 to .23 of the moments and the people's and places' at .010 to .025.
2. **Lessons with the region on.** The record is read at the moment after a turn, so the nursery's
   lessons at each answer see it on its region.
3. **Several facts in an episode.** Each turn writes; a session is one episode, with no reset
   between turns, so the recall memory keeps the session.

**The world.** The synthetic world with 24 people and 8 places (the Step 0 templates, frozen in
`world.py`), two facts and two questions a session, a quarter of the person and place pairs in no
lesson. Nursery: 400 lesson conversations of 4 sessions on 16 streams. Readings without lessons,
8 conversations a stream unless stated (`declared/synthetic.json`), each on a copy of the nursery's
brain; with `gap` the anchor sessions' people are told nothing more until the next anchor, which
asks about one of them exactly `gap` sessions back (`anchored`):

| Reading | Wording | Sessions | gap | Emptied at each answer |
| --- | --- | --- | --- | --- |
| R1 | trained | 6 | 1 | |
| R2 | trained | 21 | 10 | |
| R3 | held-out | 21 | 10 | |
| R4 | trained | 101 | 100 (2 conversations a stream) | |
| R5 | trained | 21 | 10 | the records |
| R6 | trained | 21 | 10 | the message |

**Arms.** `turns` (the opt-in at its founders); `turns_nearest` (the same with `longterm_match=0`,
the nearest key read, reported apart); `previous_word` (the 0.97 default, no message, no records);
the LSTM and the transformer at the `turns` brain's parameters, online and offline, on R1 to R4
(`declared/controls-synthetic.json`). Parameters, seconds and multiply-accumulates per answer are
reported for every arm: the brain's every synapse once per sweep and row of each settle and one
accumulate per key unit and record held at each records read; the controls' analytic forward pass.

**Founders** 3050 to 3055, in no run of any lane.

**Gates** (the `turns` arm, 5 of 6 founders, `summarize.gates`):

| Gate | Reading | Bar |
| --- | --- | --- |
| T1 told earlier | R1, questions about person and place pairs no lesson told | .90 |
| T2 paraphrased | R3, every question | .80 |
| T3 across sessions | R2, questions exactly ten sessions back; revised questions; stale | .90; .90; at most .02 |
| T4 held and used | R5, ten sessions back | at most `previous_word`'s R2 reading plus .05 |

Reported without a bar: one session back (R1), a hundred sessions back (R4), the message emptied
(R6), `turns_nearest`, the controls, seconds and multiply-accumulates.

**The second rung: bAbI** tasks 1 and 2 (`declared/babi.json`, `declared/controls-babi.json`): the
same arms and controls, five passes over the 200 training stories with lessons, the 1,000 test
questions, and the records emptied at each answer; reported without a bar.

**Expectations, stated before the run** (development founders 0 to 2, 150 nursery rounds):
- `turns`: every reading of the synthetic world at 1.000, the nearest-key read the same, stale
  .000, and R5 (records emptied) at .09 to .29.
- R6 (message emptied) reads as R2. In this arm the answer's path is the records'; the message
  holds nothing it reads, so emptying the message does not fall to the control.
- The controls: about Step 0's readings, at chance from one session back but for the offline
  transformer's window.
- bAbI: task 1 at .30 to .45 and task 2 near chance (development founder 0: .296 and .166 on 100
  test stories). In task 1's lexicon of four people a name is as frequent as the bar (about one
  moment in 28), so the people's units drop out of the key. Task 2 asks about an object whose
  place is told through the person who carries it, two records apart.

**Total work.** 18 synthetic lives and 36 bAbI lives of the brain; 24 synthetic and 48 bAbI lives
of the controls; one spot CPU box.

**Departure from the plan of the run.** The first launch at `74894445` stopped on the brain's side:
the counter of multiply-accumulates read `row_sweeps` from a settlement report that holds `steps`
alone, and the synthetic world's lives and part of bAbI's ended with an error before any of their
readings was written. The counter reads `steps` times the rows there, and the brain's lives were
launched again whole, on the same plans and founders; the controls' receipts of the first launch
stand. No instrument of a reading, plan or bar changed.

### Outcome of the declared run

Read on founders 3050 to 3055 and recorded in [the README](README.md#the-declared-run-records-keyed-at-every-turn),
with the receipts in `results/declared/` and `summarize.gates` recomputing each gate.

| Gate | Verdict |
| --- | --- |
| T1 | passes, 6 of 6: 1.000 |
| T2 | passes, 6 of 6: .995 to 1.000 |
| T3 | passes, 6 of 6: 1.000 ten sessions back, revised 1.000, stale .000 |
| T4 | passes, 5 of 6: .094 to .148 against `previous_word`'s .090 to .148; founder 3053 reads .148 against .090 plus .05 |

**Against the expectations.** As expected on the synthetic world and on bAbI. The records emptied
read .094 to .148 ten sessions back, inside the expected .09 to .29. **Departures from the plan of
the run.** The relaunch of the brain's lives recorded above.

## The second declaration: records read by what tells them apart (declared 11 November 2026, before its first life)

The first declaration's outcome stands. Its key kept the units on in at most .03 of the moments
heard, a bar that depends on the lexicon: in bAbI task 1 four people share the stories, a name is
on in about one moment of 28, and the people's units dropped out. The opt-in
`longterm_distinct=share` (with `longterm_turn`) reads by what tells the stream's own records
apart, with no bar on how often a unit is heard (`docs/memory.md`, "Turns in a conversation"):

1. **Entries, not units.** Each distinct heard moment is one codebook entry, so two words whose
   codes share units (in task 1's lexicon "mary" shares a unit with "the" and one with "?") stay
   two entries. A record holds its statement's entries, the window's and the value's.
2. **Weights from the records.** An entry weighs `log((n + 1) / (records holding it + 1))` among
   the stream's `n` records; a question's cue keeps its entries of at least `share` (.5) times
   its heaviest weight; the read takes the latest record holding at least `longterm_match` (.5)
   of the cue's weight. Every value is a record of its own.
3. **Chaining** (bAbI task 2, `longterm_heads=2`): of the first record's entries outside the cue,
   the one the most records hold cues the second head, which reads the latest other record
   holding it, before the answer settles.

**The arm** `distinct`: `longterm=4096, longterm_turn=TURN, longterm_distinct=0.5,
longterm_heads=2, longterm_match=0.5` (the match chosen on development founder 0, where .75 found
.872 of task 1's answers' records and .5 found 1.000). Beside it `previous_word`, and the LSTM and
the transformer at the `distinct` brain's parameters, online and offline.

**Founders** 3056 to 3061, in no run of any lane. Plans in [`declared2/`](declared2): the first
declaration's synthetic world, nursery and readings R1 to R5, and bAbI tasks 1 and 2 (five passes
over the training stories, the 1,000 test questions, and the records emptied at each answer).

**Gates** (the `distinct` arm, 5 of 6 founders, `summarize.gates(found, "distinct", SECOND)`):

| Gate | Reading | Bar |
| --- | --- | --- |
| T1 to T4 | the first declaration's, on these founders | as declared there |
| D1 | bAbI task 1, the test questions | .90 |
| D2 | bAbI task 1, the records emptied at each answer | at most `previous_word`'s reading plus .05 |

Reported without a bar: bAbI task 2 (answers, and the share of answers whose first and second
records hold the answer), the controls, seconds and multiply-accumulates per answer with the
records' read apart from the settle's sweeps.

**Expectations, stated before the run** (development founder 0): the synthetic world at 1.000 on
R1 to R3 with the records emptied at .14 ten sessions back; task 1 at .968 after one pass, its
first record holding the answer at 1.000; task 2 at .184 after two passes, its first record never
holding the answer (the first record is the object's own, whose value is the object) and its
second at .512. Task 2's second read misses where the carrier dropped the object and moved on:
the latest record of the carrier is after the drop, and the object stayed where it was dropped.
The settle's sweeps hold about nine tenths of the brain's multiply-accumulates per answer, the
records' read the rest.

**Total work.** 12 synthetic and 24 bAbI lives of the brain; 24 synthetic and 48 bAbI lives of the
controls; one spot CPU box.

### Outcome of the second declaration

Read on founders 3056 to 3061 and recorded in
[the README](README.md#the-second-declaration-records-read-by-what-tells-them-apart), with the
receipts in `results/second/` and `summarize.gates(found, "distinct", SECOND)` recomputing each
gate.

| Gate | Verdict |
| --- | --- |
| T1 | passes, 6 of 6: .997 to 1.000 |
| T2 | passes, 6 of 6: .9992 to 1.000 |
| T3 | passes, 6 of 6: .996 to 1.000 ten sessions back, revised 1.000, stale at most .001 |
| T4 | passes, 6 of 6: .094 to .137 against `previous_word`'s .082 to .144 |
| D1 | passes, 6 of 6: .993 to 1.000 |
| D2 | passes, 5 of 6: .187 to .312 against .243 to .404; founder 3061 reads .312 against .244 plus .05 |

Reported: bAbI task 2 at .315 to .436, its second record holding the answer on .506 of the
questions; the records' read 2% to 4% of the brain's multiply-accumulates per answer.

**Against the expectations.** As expected. **Departures from the plan of the run.** None.
