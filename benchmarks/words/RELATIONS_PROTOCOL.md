# Relations between things: declared protocol R1 (cadence#327)

Declared on 2026-11-13, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed.

## The question

A parrot holds a map of its room (`cadence.space.map.Map`), knows which way it faces, and knows
the names of the things around it from naming. Does it learn words for where things are from
each other and from itself (in front of, to the right of, behind, to the left of, near, far
from) from questions answered for it, and answer them in both directions, in rooms it has never
seen: "what is left of the ball?" (the things), "where is the ball from the cup?" and "how far is
the ball from the cup?" (the word), "is the cup near the apple?" (yes or no), from another thing
and from the parrot itself ("... you"); for pairs of things never placed from each other in
practice; and with a relation word it never heard in practice, taught through words it knows
("what is R the X?", answered with the things) in four answered questions?

## The brain

`cadence.words.Words` with the part `cadence.words.relations.Relations` at its defaults (8
direction cells of sharpness 2.5, 6 distance cells from 0.3 m to 3 m of width 0.37, `norm` 1.2,
`contrast` 1.0) and the lexicon's opt-ins `heard_only=True` and `pattern_back=True`; a map of the
chamber's plane of 4 m by 4 m, 32 by 32 cells, one unit a kind (8 kinds); a lexicon of 64 word
neurons and 37 concepts (8 kinds, 4 places around the parrot, 5 reads, 8 direction and 6
distance cells, the parrot itself, two verdicts, two anchors, the read that finds nothing);
naming shares .4 (the kind) and .3 (the place around the parrot); a concept counts as named above
.05. Learned parameters: 2,368 (one efficacy a reciprocal pair of the lexicon).

## The chamber

`benchmarks/words/relations.py` at its defaults. Every word is a nonce word drawn for the founder.
The relations: the four quarters of the turn around a reference, in the frame of where the
parrot faces; near (under 1 m) and far (1 m or more). The teacher asks only clear questions:
every thing judged lies at least 10 degrees from a quarter's edge, or 0.15 m from 1 m. One life,
one script shared with the controls (`relations.script`):

1. **Naming**: 30 rooms of 4 to 6 things of the 8 kinds, 12 namings a room (the kind's word, the
   parrot turning to a drawn heading and attending to a drawn thing); "you" named while the
   parrot attends to itself, in the first 10 rooms.
2. **Practice from the parrot**: 16 rooms, 8 questions a room: "where is the X from you?" and "how
   far is the X from you?" with their answers.
3. **Practice of every form**: 40 rooms, 10 questions a room, a form drawn evenly: things from a
   thing and from the parrot, where and how far from a thing and from the parrot, is it from a
   thing and from the parrot (yes and no balanced).
4. **The held-out word**: one of the six relations, drawn for the founder, is never asked or
   answered in naming or practice. It is taught in 4 answered questions "what is R the X?" (R
   heard for the first time at the first; each answer names at least one thing, and one of them
   lies each time in another part of what R does not name: a quarter's thing at another distance
   from 0.45 m to 3.6 m on a log scale, a distance's thing in another quarter).
5. **The test**, nothing taught: 30 fresh rooms; in each, up to 2 questions of every form about
   the practised relations, and up to 2 of every form about the held-out word.

Kind pairs: 5 of the 28 pairs of kinds, drawn for the founder, never meet in a practice room.

An answer is right where the words said are the answer's words as a set, one a thing.

## Founders and arms

Founders 32700000 to 32700011 (twelve, no life in any lane).

| Arm | What it is | Parameters |
| --- | --- | --- |
| the parrot | the brain above | 2,368 |
| LSTM, transformer, online | one pass over the script in its order, one Adam step per naming (its two sequences) or practice question (`relations_controls.py`, the words chamber's training) | at 2,368 (nearest width: 2,392 and 2,424) |
| LSTM, transformer, offline | shuffled epochs with early stopping on a tenth held aside | at 2,368 |
| LSTM, transformer, offline, large | the same at nine times the parrot's parameters (a reading, no gate) | at 21,312 |

The controls are given the same script as tokens: before every sequence, each thing of the room
as its kind and its place in the parrot's frame as it faces (two of 16 bins an axis, over the
room); each naming as two sequences (concept then word, word then concept); each question with
its answer. The held-out word's four lessons are one update each, in their place in the script.

## Readings

For every arm and founder:
- **R1** things from a relation: from a thing ("what is R the X?") and from the parrot ("what is R
  you?"), the practised relations;
- **R2** the relation of a thing: where and how far, from a thing and from the parrot;
- **R3** yes or no: from a thing and from the parrot;
- **R4** questions about things that place a held-out pair (the two things named, or the
  reference and a thing of its room);
- **R5** every question about the held-out word;
- **R6** (the parrot) the same parrot with every synapse between its words and the vector cells
  at zero, answering R1 and R2 questions whose answer is not "I do not know";
- parameters, multiply-accumulates (the parrot: every lexicon settle at its sweeps times its
  synapses, every map moment at one sweep and every map question at two over the map's
  synapses; the controls: analytic, a backward pass at twice the forward) and seconds a life.

## Gates

A gate holds where the mean over the twelve founders reaches its threshold and at least ten of
twelve founders do.

- **R1**: from a thing at .90, and from the parrot at .90.
- **R2**: from a thing at .90, and from the parrot at .90.
- **R3**: from a thing at .90, and from the parrot at .90.
- **R4**: at .90.
- **R5**: at .90.
- **R6**: every answer is the lexicon motor's letters (`Words.ask`; `tests/test_words_relations.py`),
  and with the relation synapses at zero R1 and R2 read .05 at most on every founder.
- **R7**: the table of every arm with parameters and multiply-accumulates.

## Expectations, stated before the run

From the confirmation round on development founders 900 to 915 (no gated life), at this commit:
the parrot reads R1 .999 (least founder .98) from a thing and .985 (.80) from the parrot, R2
.999 and 1.000, R3 .999 and .998, R4 .999, R5 .987 (.88), R6 .000; 2.0e9 multiply-accumulates and
40 s a life. With the held-out word taught by "where is the X from the Y?" in place of "what is
R the X?" (a reading, not this protocol) R5 reads .933 (.57). The matched controls, on an
earlier version of the chamber and two development founders, offline, read things .00 to .17,
where .45 to .63 and is it .25 to .58: below every gate.

**What a failure names.** An R1 or R3 miss with R2 right names the description's competition at
a boundary; an R2 miss names the reads or the motor's choice; an R5 miss names the new word's
group, read from four examples; an R4 gap below R1 to R3 names the pairs, which the parrot's
mechanism does not see.

## Compute

One AWS c7i.4xlarge spot box (Name=cadence-0100-words-relations), one process a life, one BLAS
thread a process, results synced to s3://cadence-0100-words-relations-20261113 (rel327/).
