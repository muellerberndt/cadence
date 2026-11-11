# The encyclopedia: facts read once, answered from the brain's own records

Issue [#311](https://github.com/muellerberndt/cadence/issues/311). A brain reads a corpus of facts
from Wikidata once, one word a moment, and writes its own long-term records as it hears them. Long
after, it answers questions about those facts: in the wordings its lessons used, in wordings no
lesson used, after a fact is told again with a new object, and for questions that need two facts.
Its knowledge sits in the records; its trained synapses stay at the size of the brain. The
lessons teach reading, asking and answering on a separate set of facts, and no question asks a
fact a lesson used.

The protocol, the corpus's checksums and the declared run are in [`PROTOCOL.md`](PROTOCOL.md).

## The corpus

Wikidata (CC0 1.0), the dump of 7 August 2026, read through the public SPARQL endpoint
`https://qlever.dev/api/wikidata` with eleven queries, one a relation: a sovereign state's
capital, continent, currency and language; a person's citizenship, occupation, birthplace, native
language, sport and instrument; a city's country. The table `facts.tsv` holds 2,317,143 facts of
983,271 subjects over 608 answers (SHA-256
`057bd6f515a299343dae8731276eaa9ab6e59cb4844ad7e0f2342a67a4b9b18e`); each query's text, row count
and checksum are in `PROTOCOL.md`. The data is kept in the lane's S3 bucket, none of it in git.

## The pieces

| File | What it holds |
| --- | --- |
| `corpus.py` | The SPARQL queries, the fetch and the table of facts (`facts.tsv`, in the lane's bucket) |
| `facts.py` | Words as heard moments, the sentences, the wordings, the lessons and the revisions |
| `chamber.py` | The brain's lives: the nursery, the reading, the questions; `said` and `found` |
| `controls.py` | Closed-book LSTM and transformer, a transformer with the facts in its window, BM25 retrieval read by a transformer |
| `summarize.py` | The gates, the parity table and every learner's costs |
| `test_encyclopedia.py`, `test_controls.py` | Contracts on a tiny corpus, with no download |

## The brain

`Brain.compose(257, 608, modules=(width,), message=280, longterm=records, longterm_turn=256,
longterm_heads=2, longterm_distinct=0.1, longterm_match=0.5, longterm_best=True,
longterm_rest=True, longterm_fade=False)`:

- every turn writes the moment after it (the object) as a record holding the whole statement;
- a question's cue is its words that the records tell apart, weighted by their rarity among the
  records ("the", "is" and question words weigh nothing);
- `longterm_best` reads the record that holds the most of the cue's weight, so "what is the
  capital of italy ?" reads the record of both "capital" and "italy", and a fact told again is
  read from its latest telling;
- `longterm_rest` reads what the question asks beyond that record: "what is the currency of the
  citizenship of ada lovelace ?" reads her citizenship, then the currency of that country, and
  holds the second record on the region the brain answers from.

What is hand-engineered is named as such: the words' form codes (a hash), the closed answer
vocabulary, one action an answer, and the lessons drawn answer first.

## Run

    python benchmarks/encyclopedia/corpus.py fetch --out raw
    python benchmarks/encyclopedia/corpus.py build --raw raw --out facts.tsv
    PYTHONPATH=src python benchmarks/encyclopedia/chamber.py --plan declared/brain.json \
        --data facts.tsv --out brain.jsonl --workers 24
    PYTHONPATH=src python benchmarks/encyclopedia/controls.py --plan declared/controls.json \
        --data facts.tsv --out controls.jsonl --workers 40
    python benchmarks/encyclopedia/summarize.py --brain brain.jsonl --controls controls.jsonl
