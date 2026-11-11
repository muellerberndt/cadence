# A brain with its own records at 100,000: the fifth declaration

Declared 11 November 2026, before the first run on its founders. Issue:
[#310](https://github.com/muellerberndt/cadence/issues/310), gate C6. It follows
[the fourth declaration](PROTOCOL_4.md), which reads the store alone, and [the third](PROTOCOL_3.md),
whose chamber it reads at ten times the records. Instrument: `brain_records.py`; plan:
[`protocol-5/brain.json`](protocol-5/brain.json); contracts: `test_memory_scale.py` and
`tests/test_longterm_records.py`.

## The brain and the chamber

The third declaration's chamber and brain (`Brain.compose(..., message=280,
message_boundary=turn, message_boundary_holds=True, longterm=records)`, a fact the act, the agent,
the turn and the patient, a question the same with a silent last moment, every episode after a
`reset()`), with three changes that the records' number asks for:

- **A lexicon of 256 noun stems in two numbers and 256 verbs** (`meanings.Lexicon(256, 256)`,
  131,072 agent-verb pairs, 1,284 inputs with the turn), so that 100,000 facts are about 100,000
  different pairs. The message's units read the chamber's share of the channels as before
  (fan-in 587).
- **Each stream's store reads through the address of pairs**: `longterm=120000,
  longterm_pairs=16, longterm_expansion=16384, longterm_winners=256, longterm_candidates=4096,
  longterm_fade=False`, the settings of the fourth declaration. Records written once do not fade.
- **One stream reads.** The nursery keeps its 16 streams and 1,500 rounds; the reading tells
  100,000 facts on one stream, then a tenth of them again with another patient, then asks 400
  questions about unrevised facts and 400 about revised ones, without lessons.

Scores as in the third declaration: **found** (the record the read held at the answer holds the
question's key and the fact's current patient), **stale** (it holds the patient a revised fact had
before), **used** (the brain's answer is the current patient; chance .125), and found over the
questions whose fact's key is its own.

## Gate

Founders 3140 to 3151, in no run of any lane.

| Gate | Reading | Bar |
| --- | --- | --- |
| C6a | used, unrevised facts, 100,000 records | .95, on 10 of 12 |
| C6b | revised facts at 100,000: found current; stale; used | .95; under .02; .95, on 10 of 12 |

Reported: found, found over facts whose key is their own, the share of facts whose key another
fact has, ms per told moment, parameters and the records' bytes.

**Not read: 1,000,000 records.** On one core a told moment costs about 11 ms with records (the
development reading below), so a million facts of four moments take about 12 hours a founder.
The gate's reading at 1,000,000 is the open remainder, with its cost located there.

## Expectations, stated before the run

From one development reading (founder 0, a lexicon of 512 stems and 256 verbs, 300 nursery
rounds, 3,000 facts, 200 questions of each kind): the nursery ends at 1.000; found, used and found
current 1.000, stale .000; 10.8 ms per told moment on a loaded box; 5,365,407 parameters. At
100,000 facts the misses are expected on facts whose key another fact has: at 280 units over a
lexicon of 256 stems and 256 verbs some agents' or verbs' codes are one code, as the third
declaration located (1.2% and 2.4% of facts at two founders of six there). C6a and C6b are
expected to hold, with found over all facts between .97 and 1.00.

## Amendment 1: the instrument's own-key count by parts (declared 11 November 2026, before the run again)

The first life of founders 3140 to 3151 started at 18:10 UTC and was stopped at 19:38 UTC. After
every founder's 100,000 facts and their revisions were told, and before any question was asked,
`own_keys` (which counts the facts whose key another fact has, a diagnostic) coded all 100,000
facts' moments at once, about 40 GB a founder; the kernel ended three workers and the run was
stopped. No score was computed.

The instrument codes those keys by parts of 2,000 facts (`brain_records.PART`); a contract test
holds the parts to the whole. The same plan, founders and gates run again, from the start.

## Outcome

Read on the declared founders under Amendment 1; tables in the
[README](README.md#the-fifth-declaration-a-brain-with-its-own-records-at-100000), receipt
`results/fifth-brain.jsonl.gz`, the gates recomputed by `summarize_million.py`.

| Gate | Result |
| --- | --- |
| C6a | used .960 to 1.000, 12 of 12: **passes** |
| C6b | current .995 to 1.000, stale .000, used .995 to 1.000, 12 of 12: **passes** |

Over the questions whose fact's key is its own, found reads 1.000 on every founder; the misses lie
among facts whose key another fact has (0.29% to 8.58% of a founder's facts). The reading at
1,000,000 records is open, its cost located: about 11 hours a founder at 9.6 ms a told moment.
