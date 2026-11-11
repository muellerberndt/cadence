# The encyclopedia: facts read once, answered from the brain's own records

Issue: [#311](https://github.com/muellerberndt/cadence/issues/311). Instruments: `corpus.py` (the
facts, from Wikidata), `facts.py` (sentences, questions, wordings, lessons), `chamber.py` (the
brain's lives), `controls.py` (the matched controls); plans in [`declared/`](declared). This file
records the corpus as fetched, the development readings, and the declared run, posted on the issue
before its first life.

## The corpus, read against the endpoint (11 November 2026)

| | |
| --- | --- |
| Source | Wikidata, through the public SPARQL endpoint `https://qlever.dev/api/wikidata` |
| Dump | 2026-08-07T23:15:45Z (the endpoint's `wikibase:Dump` dates) |
| Licence | CC0 1.0 (Wikidata's structured data; the dump's `cc:license`) |
| Queries | `corpus.query(relation)`, eleven, one a relation, kept with their answers |
| Table | `facts.tsv`, 127,834,091 bytes, SHA-256 `057bd6f515a299343dae8731276eaa9ab6e59cb4844ad7e0f2342a67a4b9b18e` |

| Relation | Subjects | Property | Rows fetched | SHA-256 of the answer (first 16) | Facts kept |
| --- | --- | --- | --- | --- | --- |
| capital | sovereign states | P36 | 211 | `7d28fdee4a5172a9` | 67 |
| continent | sovereign states | P30 | 211 | `561ec69ef6f07ee5` | 188 |
| currency | sovereign states | P38 | 227 | `4551691600cf7eb5` | 60 |
| language | sovereign states | P37 | 387 | `bf30f9df894b1071` | 150 |
| citizenship | people, 4 or more articles | P27 (a sovereign state) | 1,001,492 | `94a5b6c25577ea42` | 887,855 |
| occupation | people | P106 | 1,704,216 | `9abb1079569f6678` | 659,076 |
| birthplace | people | P19 | 993,362 | `f2a546931bac2629` | 280,204 |
| native language | people | P103 | 124,241 | `1a4612746eb20ae6` | 108,401 |
| sport | people | P641 | 390,043 | `902bc6157e6134b5` | 316,019 |
| instrument | people | P1303 | 86,108 | `f89c2feb1a2fa916` | 58,873 |
| country | cities, 4 or more articles | P17 (a sovereign state) | 6,562 | `e7fbaccfef71fbeb` | 6,250 |

Each answer is kept gzipped with its full SHA-256 in `manifest.json` in the lane's S3 bucket; none
of the data is in git. `corpus.build` makes the table:

- **Names.** A subject's English label folded to lower-case ASCII words; a name two subjects share
  is dropped with both. A subject that is an answer ("italy") is said as that answer's one word.
- **The closed answer vocabulary** (an instrument's choice): each relation's most named objects
  (`corpus.TOP`: 250 states, 200 birthplaces, 100 occupations, 40 native languages, 40 sports, 25
  instruments, 10 each of continents, currencies and languages), each named by at least 4 facts,
  608 answers in all. A fact whose object is outside them is dropped, so a capital is kept where it
  is a birthplace answer as well (67 capitals).
- **One object a relation.** Where Wikidata lists several answers, the one its relation names
  most, then the lowest item number.
- **Order.** Subjects by their articles, most first: 2,317,143 facts of 983,271 subjects.

**Lessons and reading.** A subject whose item number hashes under .2 (`facts.lesson_subject`)
belongs to the lessons; every other subject's facts form the reading corpus in the table's order,
so the corpus of 10,000 facts is the first 10,000 of them and the corpus of 100,000 holds it. No
lesson tells or asks a fact of the reading corpus.

## The instrument

- **A word** is heard as a fixed form code: 5 of 256 input channels drawn from a hash of the word
  (an instrument's stand-in for a heard form, hand-engineered); channel 256 marks a turn.
- **A fact** is "the {relation} of {subject} is {object} .", the turn marked before the object. A
  question is its words, the turn, and a silent moment at which the brain says the answer as one
  action of the 608.
- **Wordings** (`facts.py`, frozen before any life): trained ("what is the capital of italy ?",
  "the capital of italy is what ?"); held out ("which capital does italy have ?", "italy 's
  capital is which ?", "name the capital of italy ?"); synonym, the relation said by another word
  ("what is the seat of italy ?"); two facts ("what is the capital of the citizenship of ada
  lovelace ?", citizenship or birthplace then the country's capital, continent, currency,
  language or the city's country).
- **The life** (`chamber.py`), one stream, every told fact and every question its own episode:
  1. the nursery, lesson sessions of 6 lesson facts drawn answer first (every answer taught about
     as often, an instrument's choice), a tenth told again with another object, a fifth of the
     questions of two facts;
  2. the reading, on the nursery's brain with its records emptied: the corpus told once each;
  3. the questions: trained, held out, synonym, two facts; then revisions (facts told again with
     another object of their relation) asked; then trained questions with the records emptied at
     each answer (`erased`).
- **Said and found.** Every answer is said greedily and scored exactly (`said`). `found` reads
  whether the record on head 0's region holds the answer: the read apart from the readout.

## Development readings (no gate)

Recorded in the section of the declared run.
