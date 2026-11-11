# The encyclopedia: facts read once, answered from the brain's own records

Issue: [#311](https://github.com/muellerberndt/cadence/issues/311). Instruments: `encyclopedia_corpus.py` (the
facts, from Wikidata), `encyclopedia_facts.py` (sentences, questions, wordings, lessons), `encyclopedia_chamber.py` (the
brain's lives), `encyclopedia_controls.py` (the matched controls); plans in [`declared/`](declared). This file
records the corpus as fetched, the development readings, and the declared run, posted on the issue
before its first life.

## The corpus, read against the endpoint (11 November 2026)

| | |
| --- | --- |
| Source | Wikidata, through the public SPARQL endpoint `https://qlever.dev/api/wikidata` |
| Dump | 2026-08-07T23:15:45Z (the endpoint's `wikibase:Dump` dates) |
| Licence | CC0 1.0 (Wikidata's structured data; the dump's `cc:license`) |
| Queries | `encyclopedia_corpus.query(relation)`, eleven, one a relation, kept with their answers |
| Table | `facts.tsv`, 127,617,289 bytes, SHA-256 `362eb279ef8eca459ba65ee61225de07d4afcc646a84a391f98027903f28c5cf` |

| Relation | Subjects | Property | Rows fetched | SHA-256 of the answer (first 16) | Facts kept |
| --- | --- | --- | --- | --- | --- |
| capital | sovereign states | P36 | 211 | `7d28fdee4a5172a9` | 67 |
| continent | sovereign states | P30 | 211 | `561ec69ef6f07ee5` | 188 |
| currency | sovereign states | P38 | 227 | `4551691600cf7eb5` | 60 |
| language | sovereign states | P37 | 387 | `bf30f9df894b1071` | 150 |
| citizenship | people, 4 or more articles | P27 (a sovereign state) | 1,001,492 | `94a5b6c25577ea42` | 887,855 |
| occupation | people | P106 | 1,704,216 | `9abb1079569f6678` | 659,076 |
| birthplace | people | P19 | 993,362 | `f2a546931bac2629` | 280,204 |
| mother tongue (native language) | people | P103 | 124,241 | `1a4612746eb20ae6` | 108,401 |
| sport | people | P641 | 390,043 | `902bc6157e6134b5` | 316,019 |
| instrument | people | P1303 | 86,108 | `f89c2feb1a2fa916` | 58,873 |
| country | cities, 4 or more articles | P17 (a sovereign state) | 6,562 | `e7fbaccfef71fbeb` | 6,250 |

Each answer is kept gzipped with its full SHA-256 in `manifest.json` in the lane's S3 bucket; none
of the data is in git. `encyclopedia_corpus.build` makes the table:

- **Names.** A subject's English label folded to lower-case ASCII words; a name two subjects share
  is dropped with both. A subject that is an answer ("italy") is said as that answer's one word.
- **The closed answer vocabulary** (an instrument's choice): each relation's most named objects
  (`encyclopedia_corpus.TOP`: 250 states, 200 birthplaces, 100 occupations, 40 native languages, 40 sports, 25
  instruments, 10 each of continents, currencies and languages), each named by at least 4 facts,
  608 answers in all. A fact whose object is outside them is dropped, so a capital is kept where it
  is a birthplace answer as well (67 capitals).
- **One object a relation.** Where Wikidata lists several answers, the one its relation names
  most, then the lowest item number.
- **Order.** Subjects by their articles, most first: 2,317,143 facts of 983,271 subjects.

**Lessons and reading.** A subject whose item number hashes under .2 (`encyclopedia_facts.lesson_subject`)
belongs to the lessons; every other subject's facts form the reading corpus in the table's order,
so the corpus of 10,000 facts is the first 10,000 of them and the corpus of 100,000 holds it. No
lesson tells or asks a fact of the reading corpus.

## The instrument

- **A word** is heard as a fixed form code: 5 of 256 input channels drawn from a hash of the word
  (an instrument's stand-in for a heard form, hand-engineered); channel 256 marks a turn.
- **A fact** is "the {relation} of {subject} is {object} .", the turn marked before the object. A
  question is its words, the turn, and a silent moment at which the brain says the answer as one
  action of the 608.
- **Wordings** (`encyclopedia_facts.py`, frozen before any life): trained ("what is the capital of italy ?",
  "the capital of italy is what ?"); held out ("which capital does italy have ?", "italy 's
  capital is which ?", "name the capital of italy ?"); synonym, the relation said by another word
  ("what is the seat of italy ?"); two facts ("what is the capital of the citizenship of ada
  lovelace ?", citizenship or birthplace then the country's capital, continent, currency,
  language or the city's country).
- **The life** (`encyclopedia_chamber.py`), one stream, every told fact and every question its own episode:
  1. the nursery, lesson sessions of 6 lesson facts drawn answer first (every answer taught about
     as often, an instrument's choice), a tenth told again with another object, a fifth of the
     questions of two facts;
  2. the reading, on the nursery's brain with its records emptied: the corpus told once each;
  3. the questions: trained, held out, synonym, two facts; then revisions (facts told again with
     another object of their relation) asked; then trained questions with the records emptied at
     each answer (`erased`).
- **Said and found.** Every answer is said greedily and scored exactly (`said`). `found` reads
  whether the record on head 0's region holds the answer: the read apart from the readout.

## The controls (`encyclopedia_controls.py`)

Each at the parameters of the declared brain, on the same founders, lessons and questions, hearing
the same moments through a linear input layer and saying one of the 608 answers:

- `closed_lstm`, `closed_transformer` (closed book): trained on every told fact of the corpus and
  of the lessons and on the lesson questions, in shuffled epochs until the loss stops falling
  (patience 3) or 40 minutes of training; then the revisions are taught the same way. Asked with
  nothing before the question; `erased` is the trained questions after the revisions.
- `context_transformer`: the facts heard as one stream, each question after the latest facts that
  fit in a window of 1,024 moments with it.
- `retrieval`: BM25 over the told sentences, the 3 best read in told order by a causal
  transformer trained on the lesson questions with the lesson facts as its index; `found` reads
  whether the answer's sentence is among them.

## Development readings (no gate)

Founders 0 and 1, the corpus of 10,000 facts unless named; said, with found in brackets.

| Arm | Parameters | Trained | Held out | Two facts | Revised (stale) | Synonym | Erased |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `records`, width 128, 3,000 sessions | 741,542 | .972, .986 (1.000) | .980, .968 (.996, 1.000) | .935, .920 (.98, .99) | .985, .985 (.000) | .415, .365 | .010, .000 |
| the same, two-fact lessons at .4 | 741,542 | .978 (1.000) | .974 (.996) | .850 (.98) | .960 (.000) | .420 | .005 |
| `records`, width 256 (the chain read from head 1) | 1,258,278 | 1.000, .998 | .974, .984 | .575, .465 | 1.000, .995 (.000) | .47, .41 | .000, .025 |
| `distinct` (#305's read), 100,000 facts, width 128 | 741,542 | .360 (.358) | .318 (.352) | .165 (.000) | .990 (.000) | .130 | .005 |
| `closed_lstm` (founder 0) | 743,712 | .778 | .092 | .190 | .855 (.030) | .020 | .800 |
| `closed_transformer` | 746,928 | .684 | .250 | .195 | .710 (.040) | .005 | .740 |
| `context_transformer` | 746,928 | .014 | .008 | .065 | .015 (.010) | .010 | .015 |
| `retrieval` | 746,928 | .668 (1.000) | .574 (1.000) | .155 (.000) | .730 (.050) | .265 | .015 |

- **The read.** `found` is 1.000 on trained questions and .996 to 1.000 on held-out ones at
  10,000 facts. The misses of two facts (.98, .99) are people whose name holds a country ("albert
  i of belgium").
- **The readout.** A brain says what its record holds where its nursery taught every answer:
  1,000 sessions said .49 (width 64) and .88 (width 128) at 2,000 facts, 3,000 sessions .84 and
  .98.
- **Two facts.** With the second record read through head 1's region, a question of two facts
  was found at .90 and said at .47 to .61, every capital chain at .000: head 1's synapses are
  taught by the two-fact lessons alone. Holding the answering record on head 0's region
  (`longterm_rest`, the chain from its end) lifts it to .92 to .935. A person's native language
  said as "native language" shared its word with a country's language; said as "mother tongue",
  the language chains read .95 to .97.
- **At 100,000 facts** cadence#305's read (the latest record above the match) finds the asked
  record on .35 of the questions: the subject's other facts and the relation's other subjects
  hold half the cue's weight. The record that holds the most (`longterm_best`) is the term.
- **The controls** (development, founder 0, the corpus before the mother-tongue wording): the
  closed-book learners hold the trained wording (.68 to .78) and lose the held-out wordings (.09
  to .25); the in-context transformer's window holds about 100 of the 10,000 facts; retrieval
  finds every asked sentence and its reader says .57 to .67.
- Seconds to take in a fact: the brain .027 (width 128, 100,000 facts, one core), the closed-book
  LSTM .051 and transformer .133 (training per fact), retrieval's index .000005.

## The declared run (declared 11 November 2026, before its first life)

Plans: [`declared/brain.json`](declared/brain.json) and [`declared/controls.json`](declared/controls.json).
Founders 3110 to 3121, twelve, fresh. Corpus `facts.tsv` as above (SHA-256 `362eb279...f28c5cf`).

**The brain** (`records`): `Brain.compose(257, 608, modules=(128,), message=280,
longterm=size + 10000, longterm_turn=256, longterm_heads=2, longterm_distinct=0.1,
longterm_match=0.5, longterm_best=True, longterm_rest=True, longterm_fade=False)`, 741,542
parameters. Nursery: 3,000 lesson sessions of 6 facts, a fifth of the questions of two
facts. Reading at 10,000 facts (gated) and 100,000 facts (E2), each life on its own: 1,000 trained and 1,000 held-out questions, 300 each of
synonym, two-fact, revised and erased.

**The controls**, at the brain's parameters, on the same founders, lessons and questions at
10,000 facts: `closed_lstm` and `closed_transformer` (closed book: every told fact of the corpus
and the lessons, and the lesson questions, in epochs until the loss stops falling or 40 minutes
of training; 2 threads), `context_transformer` (the facts in a window of 1,024 moments), `retrieval` (BM25,
3 sentences, read by a transformer trained on the lesson questions).

**Amendments, keeping the gates' meaning** (the release's schedule, Addendum 3 of the lanes'
rules):

- E1's corpus is 10,000 facts, and E2's scale is the next tenfold, 100,000 facts, with the
  brain's parameters unchanged; a million facts take about 7 hours a life to hear at this
  brain's 27 ms a fact and are the open remainder.
- Answers are said as one word of the closed answer vocabulary (step 2 of the issue's order);
  answers said whole as a sentence and names said by their parts are the open remainder.
- E4's questions of two facts are chains of two facts; comparisons are the open remainder.

| Gate | Reading | Bar, on 10 of 12 founders |
| --- | --- | --- |
| E1 read once | 10,000 facts, said: trained; held out | .90; .80 |
| E2 scale | 100,000 facts, said: trained and held out | within .05 of the founder's 10,000 reading |
| E3 revision | 10,000 facts, said: revised; stale | at least .95; under .02 |
| E4 two facts | 10,000 facts, said: two-fact chains | .80 |
| E5 parity | 10,000 facts, said: trained, held out, revised, two facts | at or above the best control's, founder by founder |

E5 also reads every learner's parameters, bytes of memory, seconds and multiply-accumulates to
take in a fact and to answer. Reported without a bar: `found` beside every said reading, the
synonym wordings, the erased records, and the controls at each reading. If E2's lives are not
read by 23:00 UTC, E2 is reported as the open remainder with what was read.

**Expectations** (the development readings above): E1 at .96 to .99 trained and .96 to .98
held out; E2 within .05 (found at 100,000 as at 10,000 with `longterm_best`); E3 at .98 with no
stale answer; E4 at .90 to .94; E5 above every control on trained, held out and two facts, and
on revised above the closed-book learners' .71 to .86. The synonym wordings at .35 to .45
(their relation word no record holds: the word-to-concept term), and the erased records at .00
to .02.
