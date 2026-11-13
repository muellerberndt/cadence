# Several records in one answer

Issue [#325](https://github.com/muellerberndt/cadence/issues/325). A transformer answers from
several places of its context at once: each attention head reads a position of its own. A brain
keeps what it read in long-term records of its own. This chamber asks questions whose one
answer needs several of those records, read by several heads in one settle and said whole by
the brain's motor slots. The protocol and the declared run are in [`PROTOCOL.md`](PROTOCOL.md).

## The world

Words are heard as form codes, 5 of 256 input channels drawn from a hash of the word, and one
more channel marks a turn. A population of subjects, each a word of its own, holds attributes
(`home`, `work`, `school`, `market`: a subject to one of 32 places) and links (`boss`, `friend`,
`mother`: a subject to another subject), each for half of the subjects. A fact is told once as
"the {relation} of {subject} is {object} .", the turn before its object, and the brain writes
it as a record of its own as it hears it.

| Question | Heard | The answer, said by eight motor slots at once |
| --- | --- | --- |
| Parts, k = 1 to 8 | "what is the home of e12 and e907 and e44 ?" | slot j says the place of the j-th subject; a slot past the parts, or of a part no record answers, says `nothing` |
| A chain of 2 or 3 hops | "what is the home of the boss of the friend of e12 ?" | slot 0 says the place at the chain's end, `nothing` where the chain breaks |

A part is absent where the named subject holds no fact of the relation, or is a word no fact
holds. The lessons use one population of subjects and the readings another: no word of a
reading subject is heard in a lesson, and the readings teach nothing.

## The pieces

| File | What it holds |
| --- | --- |
| `several_world.py` | Words as heard moments, the populations, the facts and the questions |
| `several_chamber.py` | The brain's lives: the nursery, the readings at each size, what each head delivered and what was said |
| `several_controls.py` | A transformer and an LSTM that hold the records in their context, at the brain's parameters |
| `several_summary.py` | The tables from the rows |
| `test_several.py` | Contracts on a small world |

## The brain

`Brain.compose(257, 264, slots=8, modules=(module,), message=280, longterm=records,
longterm_turn=256, longterm_heads=8, longterm_distinct=0.1, longterm_match=0.6,
longterm_fade=False, longterm_attention=1.0, longterm_motor=0.05)`:

- every told fact is one record, written at its turn; a record is found by its key, the words
  before its turn;
- at a question's answer each of the eight heads takes the question's next telling word (a
  subject) and reads the record about it that holds the most of the question's context, each
  context word at the gate of the head's learned map
  ([guide](../../docs/memory.md#several-heads-each-with-a-record-of-its-own-opt-in));
- a head whose word no record answers holds nothing; a head with no word of the question left
  follows the head before it, so a chain is read hop by hop in one settle and lies on the
  regions from its end;
- each head's region projects to the association and to the motor region, and the lessons teach
  the slots to say what the regions hold.

What is designed is named as such: the words' form codes (a hash), the closed set of places,
one motor slot a part, the order in which the heads take a question's telling words (the order
heard), and the turn the world marks. What the lessons teach: the heads' gates (which relation
word each head holds to, in a chain most of all) and every synapse that says an answer.

## Readings

A reading scores, for each kind and size of question:

- `said`: every slot says its answer; `part`: the share of the asked parts said;
- `delivered`: the heads hold the records the answer needs, each once, and nothing else;
  `ordered`: head j holds the record of part j, and a chain lies from its end;
- `silent`: of the parts no record answers, the share whose head holds nothing; `refused`: of
  the questions no record answers, the share answered `nothing` on every slot;
- an answer's milliseconds and multiply-accumulates, and the read's own.

### Step 0: where the 0.99 heads fail

A development reading on founders 0 to 2 (128 association neurons, 6,000 lessons), before any
mechanism. `rest` is the encyclopedia chamber's read with eight heads (`longterm_best`,
`longterm_rest`); `turns` reads by the cosine of the turn's key, each head the best the heads
before it left; `handed` hands every head its record at each answer, the ceiling of the
answer's path with the read given.

| Arm | Reading, 10,000 records | parts 1 | parts 2 | parts 4 | parts 8 | chain 2 | chain 3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `rest` | delivered | 1.000 | .000 to .007 | .000 | .000 | .500 to .790 | .000 |
| `rest` | said | .000 to .050 | .000 | .000 | .000 | .000 to .080 | .000 to .010 |
| `turns` | delivered | .000 | .000 | .000 | .105 to .206 | .000 | .000 |
| `turns` | ordered | .000 | .000 | .000 | .000 | .000 | .000 |
| `handed` | said | 1.000 | .527 to .948 | .060 to .258 | .000 | .950 to 1.000 | .970 to 1.000 |
| `handed`, with `longterm_motor` (1,000 records) | said | .895 to 1.000 | .992 to 1.000 | .992 to 1.000 | .106 to .184 | .970 to .990 | .980 to 1.000 |

Two terms are located:

1. **The read.** The 0.99 heads deliver one record or a chain of two. A question of several
   parts reads nothing for its parts, and a chain of three hops reads nothing; read by cosine
   the heads hold the records of eight parts on a share of the questions and never in the
   question's order, so no slot learns which region answers it.
2. **The motor's path.** With every record handed, eight parts are said at .00 and four at .06
   to .26 through the association alone. With a projection from each head's region to the motor
   region the handed records of four parts are said at .99 to 1.00, and those of eight at .11 to
   .18 after 6,000 lessons, each part at .85: the slots asked least often are taught last.

### Step 1: the mechanism, development readings

Development founders 0 to 5 (0 to 2 in the first row), marked as development: no gate reads
them. `said` per kind of question; `silent` pooled over the parts no record answers; `refused`
pooled over the questions no record answers.

| Reading | Arm | Records | parts 2 | parts 4 | parts 8 | chain 2 | chain 3 | silent | refused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10,000 lessons | `heads` | 100,000 | .969 to .992 | .991 to 1.000 | .859 to .894 | 1.000 | .980 to 1.000 | .950 to 1.000 | .927 to .991 |
| 10,000 lessons | `handed` | 100,000 | 1.000 | 1.000 | .849 to .885 | 1.000 | .990 to 1.000 | 1.000 | 1.000 |
| 20,000 lessons | `heads` | 1,000 | .965 to 1.000 | .991 to 1.000 | .917 to .975 | 1.000 | .990 to 1.000 | .970 to 1.000 | .940 to 1.000 |
| 20,000, the level of the context alone | `heads` | 1,000 | .996 to 1.000 | .992 to 1.000 | .735 to .966 | .020 to 1.000 | .000 to .995 | 1.000 | .103 to 1.000 |
| 20,000, records that hold some context | `heads` | 1,000 | 1.000 | .991 to 1.000 | .928 to .995 | 1.000 | .990 to 1.000 | 1.000 | .960 to 1.000 |
| 20,000, records that hold some context | `birth` | 1,000 | .992 to 1.000 | .992 to 1.000 | .870 to .929 | .525 to .655 | .000 to .040 | 1.000 | .419 to .747 |

Three terms were located on the way to the declared read:

1. **The lessons of the last slots.** Parts of eight are said at .86 to .89 after 10,000 lessons,
   with every record handed as well (.85 to .89): each part is said at .98, and a question of
   eight parts needs all eight. After 20,000 lessons parts of eight are said at .92 to .98.
2. **Silence crossed with chains.** In questions of two parts, a first subject that lacks the
   relation asked held one of its other records on all 39 such parts read (founder 0, 10,000
   lessons, 1,000 records); the second head stayed silent on all 34 of its own. In a chain
   the first head learns to pass over the attribute word, the fourth word of the question, and
   its gate on the subject rises: in "what is the work of e13 and e98 ?" the first head's gate
   on `work` stood at .28 and on `e13` at .56, so e13's school record reached the level .66,
   above the bar. Reading the level from the context alone made the silence whole and left
   chains of three hops unlearned on four founders of six: every first hop then starts at a
   third of the cue, under the bar, and the lesson gives a right first hop no credit while the
   later hops fall short.
3. **The declared read.** The level keeps the focus and the context at the gates, and a head
   reads only the records that hold some of the question's context: a record holding none of
   it answers another question. A lacking subject holds nothing whatever the gates, and the
   chains learn as before.

## Run

    PYTHONPATH=src python benchmarks/several/several_chamber.py --plan benchmarks/several/declared/brain.json --out brain.jsonl --workers 24
    PYTHONPATH=src python benchmarks/several/several_chamber.py --plan benchmarks/several/declared/reference.json --out brain.jsonl --workers 18
    PYTHONPATH=src python benchmarks/several/several_controls.py --plan benchmarks/several/declared/controls.json --out controls.jsonl --workers 6
    PYTHONPATH=src python benchmarks/several/several_controls.py --plan benchmarks/several/declared/offline.json --out controls.jsonl --workers 3
    python benchmarks/several/several_summary.py --rows brain.jsonl --controls controls.jsonl
