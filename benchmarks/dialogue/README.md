# A closed-domain conversation chamber

Issue [#305](https://github.com/muellerberndt/cadence/issues/305): one continuing brain holds a
conversation in a closed domain, keeps what it was told as a held meaning (`message`) and as its
own long-term records (`longterm`), and says each answer whole. The data and the plans are
recorded in [`PROTOCOL.md`](PROTOCOL.md).

| File | What it holds |
| --- | --- |
| `world.py` | The synthetic conversation world, its frozen wordings, and the bAbI adapters |
| `chamber.py` | The brain's lives on four arms, its readings, and the records' key without a life |
| `controls.py` | The LSTM and the transformer matched to the brain's parameters, online and offline |
| `summarize.py` | The tables below from the receipts in [`results/`](results) |
| `test_dialogue.py` | The contracts; none needs a download |

```
PYTHONPATH=src python benchmarks/dialogue/chamber.py --plan benchmarks/dialogue/step0/synthetic.json --out rows.jsonl
PYTHONPATH=src python benchmarks/dialogue/chamber.py --plan benchmarks/dialogue/step0/babi.json --data BABI/en --out rows.jsonl
PYTHONPATH=src python benchmarks/dialogue/controls.py --plan benchmarks/dialogue/step0/controls-synthetic.json --out rows.jsonl
python benchmarks/dialogue/summarize.py --dir benchmarks/dialogue/results/step0
```

## Step 0: where the 0.97 default with `message` and `longterm` fails

A development reading on founders 0 to 2, before any mechanism and without a gate
(receipts in [`results/step0/`](results/step0); each cell is the range over the three founders).
Chance is .125 on the synthetic world, .167 on bAbI tasks 1 and 2 and .5 on task 6.

**The synthetic world** (12 people, 8 places; 400 lesson conversations of 4 sessions, then 8
reading conversations of 12 sessions on 16 streams). `gap 0` asks about a fact told earlier in
the same session; `gap 1` and up ask across sessions. `trained` and `held` are the frozen
wordings, and `prefix` asks in the fact's own words up to its turn (the records' exact key).

| Arm | Parameters | Nursery | Reading | gap 0 | gap 1 | gap 2 to 4 | held-out pairs | revised | stale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `previous_word` | 27,904 | .420 to .490 | trained | .551 to .593 | .094 to .125 | .107 to .140 | .236 to .329 | .374 to .416 | .077 to .088 |
| `message` | 100,708 | .525 to .597 | trained | .781 to .816 | .118 to .131 | .107 to .128 | .372 to .433 | .520 to .547 | .061 to .069 |
| `message` | | | held | .585 to .692 | .112 to .129 | .103 to .142 | .274 to .358 | .398 to .468 | .069 to .075 |
| `message` | | | trained, message emptied | .633 to .655 | .104 to .146 | .099 to .140 | .337 to .398 | .433 to .450 | .065 to .074 |
| `records` | 102,723 | .184 to .206 | trained | .119 to .127 | .124 to .148 | .106 to .132 | .000 | .117 to .132 | .130 to .135 |
| `records` | | | prefix | .136 to .157 | .120 to .156 | .122 to .166 | .025 to .153 | .129 to .164 | .113 to .132 |
| `records` | | | trained, records emptied | .119 to .127 | .124 to .148 | .106 to .132 | .000 | .117 to .132 | .130 to .135 |
| `records_session` | 102,723 | .407 to .499 | trained | .560 to .566 | .096 to .125 | .099 to .139 | .221 to .325 | .390 to .398 | .077 to .081 |
| LSTM, online | 103,112 | | trained | .509 to .553 | .120 to .143 | .124 to .127 | .104 to .112 | .360 to .393 | .085 to .093 |
| LSTM, offline | 103,112 | | trained | .796 to .814 | .178 to .310 | .131 to .172 | .144 to .219 | .552 to .566 | .064 to .065 |
| transformer, online | 102,188 | | trained | .283 to .300 | .281 to .325 | .191 to .204 | .211 to .255 | .180 to .202 | .148 to .155 |
| transformer, offline | 102,188 | | trained | .407 to .421 | .457 to .560 | .250 to .321 | .309 to .354 | .206 to .223 | .137 to .162 |

**The records' key without a life** (`chamber.keys_cell`, 184 questions a founder): the cosine of
a question's key (the held message at its turn) with its fact's key, the share at the records'
match (.75) or above, and the share whose nearest key among the facts told is a fact about the
same person.

| Wording | Cosine | At the match | Nearest is its own |
| --- | --- | --- | --- |
| prefix | 1.000 | 1.000 | 1.000 |
| trained | .294 to .368 | .000 | .826 to .973 |
| held | .357 to .381 | .000 | .707 to .799 |

**bAbI** (five passes over the 200 training stories with lessons, then the 1,000 test
questions):

| Arm | Task 1 | Task 2 | Task 6 (yes or no) |
| --- | --- | --- | --- |
| `previous_word` | .151 to .289 | .184 to .306 | .487 to .503 |
| `message` | .331 to .372 | .174 to .201 | .497 to .503 |
| `records` | .154 to .171 | .140 to .167 | .499 to .503 |
| `records_session` | .261 to .383 | .182 to .300 | .476 to .510 |
| LSTM, online / offline | .183 to .326 / .450 to .472 | .146 to .187 / .323 to .342 | .477 to .510 / .521 to .612 |
| transformer, online / offline | .402 to .453 / .407 to .445 | .154 to .165 / .179 to .249 | .497 to .503 / .512 to .527 |

The brain's own nursery reads .343 to .557 on the training questions of task 1 in its fifth pass.

### Where the answers fail

1. **The records' key.** A question heard as words shares one word with its fact, the person:
   its key lies at cosine .29 to .38 to the fact's, and no question of any wording reaches the
   match of .75, so the records hold nothing at any answer. The lessons never see the records'
   region on, so its synapses stay untaught, and even the exact key (`prefix`, cosine 1.000)
   reads at chance. The records emptied read the same to the digit. The nearest key is the
   question's own fact on .83 to .97 of the trained questions and .71 to .80 of the held ones.
2. **Across sessions.** From one session back every brain arm reads at chance (.09 to .15), and so
   do the online controls. A message emptied at each session's end carries nothing over, and the
   records, the one carrier across sessions, are not read (1). The offline transformer holds the
   session before in its window (.46 to .56 at gap 1).
3. **Several facts in an episode.** With each session one episode (`records_session`) the message
   is held from the first turn, so a session writes one record and the brain reads as
   `previous_word` does. With each turn one episode (`records`) every fact has its record, but
   the reset at each turn empties the recall memory as well, and within a session the brain
   falls from .55 to .59 to chance.
4. **The held message within a session.** Held over a session, the message lifts a fact told
   earlier in it from .55 to .59 to .78 to .82, above the online controls, beside the offline
   LSTM. Emptied at each answer it reads .63 to .66; the rest is the recall memory's.
   Held-out wordings fall to .59 to .69: the code of a word is no code of its concept, so a new
   wording has a new message.
5. **Held-out pairs.** A person and place pair no lesson told reads .37 to .43 with the message
   against .55 to .59 over all questions in the session: part of the answer is a learned pair.
6. **bAbI.** No brain arm learns the tasks in five passes (task 1's nursery .34 to .56); task 6 is
   at chance for every learner but the offline LSTM. A story is one session, so the records'
   terms (1) and (3) bind here as in the synthetic world.
7. **The readout of bAbI dialog** (`PROTOCOL.md`): a closed set of answers holds .572 of task 4's
   test answers; the rest says an entity the conversation told.

Time per answer, the brain's moment at a question on one core over 16 streams: .04 to .07 ms a
stream for `previous_word`, .14 to .15 with the message and .37 to .64 with the records; the controls' multiply-accumulates per answer,
the whole conversation's forward pass shared over its answers, are 1.41 million (LSTM) and 2.49
million (transformer) on the synthetic world. The brain's multiply-accumulates are not counted.
