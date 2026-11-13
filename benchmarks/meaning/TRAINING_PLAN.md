# The parrot learns what words mean: the training plan in steps (cadence#334)

The owner's approval (13 November 2026): "do the training step by step with intermediate quality
control. We really need an impressive result here." The plan runs in steps. Each step ends in a
quality check (QC): a fixed small question set on fresh founders, a pass bar, a one-minute demo
recording, and the spend so far. A step that fails its QC stops, locates the term and re-plans
before any further spend. Every step after the first starts on its own go.

The rule throughout: never engineer what can be learned. Only the world and the teacher are
designed. Each step lists the designed parts left in its path, each a gene or a control with the
learned replacement it waits for.

## 1. The final exam: the owner's target

The last step's QC is this exam, read on twelve fresh founders and in a recorded demo. Every step's
QC reads its part of it, so progress toward it is visible step by step.

| # | Target | Exam question set (examples; each set fixed before the step's first life) | Step that builds it |
| --- | --- | --- | --- |
| 1 | Maps objects in space, including out of view | Where is each thing, in view and behind; a thing starts rolling: does the parrot look at it at once? | S1 |
| 2 | Learns sounds that stand in for objects: the heard word, held as a record | A heard word said back; the name of a thing held up; the thing a heard name points to | S2 |
| 3 | Is taught relationships through the symbols | "what is near you?", "what is far?", "what is next to the apple?", "what is behind the cup?" | S4 |
| 4 | Understands a complex grammar | Word-order twins, phrases of several attributes and relations, and, or, not, clauses, tense | S6 |
| 5 | True understanding: the words are grounded | Combinations never heard; silencing a concept neuron removes exactly that meaning; a changed scene changes the answer; the plain readout and the table fail | every step from S2 |
| 6 | Position, distance, colour, material, shape, purpose, size, how many | "what is made of glass?", "is the apple round?", "can you eat it?", "what is big?", "how many round things?" | S3, S4, S5 |
| 7 | Answers arbitrary, not too complex questions | Open questions over fresh rooms, drawn from the grammar at a declared depth | S6, S7 |
| 8 | Its words sound like the learned sounds and are understood | "what are two round things?" answered "the apple and the ball", said in its own voice; an independent recogniser (Whisper, a measuring instrument only) reads the parrot's spoken answers at a declared word error rate; a short listening sample for the owner | S2 (sound), S5 (several words), S7 (English) |

## 2. The brain

One deep Cadence brain, the same through every step, grown where a step needs more.

| Part | In Cadence | Born or learned |
| --- | --- | --- |
| Eyes | Each thing held is a slot of tuned inputs: hue, lightness, six shape dimensions, size, material cues (gloss, texture, translucency), direction from where the head points, distance, salience. First steps: features given by the chamber; then the retina lane's learned features (#330) | Tuned inputs born; every category over them learned |
| The map and the focus | The map of things (#320, the physics lane #324), the parrot-brain lane's eyes and map (#332), and attention drawn by surprise (#337, the attention lane): a thing that starts moving draws the focus | The map is born (the owner's ruling); the focus learned where it can be, its surprise signal born |
| Ear | The code of each heard word: a stand-in code per form, then the hearing lane's ear (#329), then a code of the log-mel learned by the brain; the pitch contour of a word's end; the offset clock (the heard history at decays 0, .5 and .8: how long ago a sound ended) | Cochlear bands and the history's decays born; codes learned |
| Processing chain | `modules=` regions ending in the association region; the ear reaches the association region, the eyes the lowest region, so the question reaches the feature regions top-down through `apical=` (a gene; S0 reads the alternatives) | Learned |
| Recall memory | `recall=True` with `recall_heads=k`; with #336 the records strengthen by surprise and repetition | Learned addresses |
| Voice | A slot of one unit for each heard word's record and one for silence; it acts at every moment, so when to speak is learned. Several words: a planning window of slots (S5) | Learned |
| Gaze, beak | A slot turning the head (8 directions); a beak slot (bite or not) for purpose (S3) | Learned |
| Pistachio | The actor-critic: the pistachio is the reward; a praise word said at once whose value the critic learns; `actor_hold` 0.93 (a half-life near 1 s at ten moments a second) bridges the catch; an exchange ends with `done` when its pistachio is paid | Learned |
| Hearing itself | The self-hearing lesson (the inverse model of the voice, from babbling) and the self-comparison reward (the songbird's performance error) | Born rules, each a gene with the rule off as its control |

Time: ten moments a second, as on the page. A word lasts about three moments; the gap that pays a
reply is 0.3 to 2 s after the trainer stops; the pistachio is thrown and reaches the beak 0.5 s
after the act.

## 3. The steps

Founders: development 900 to 999; QC founders drawn fresh for each step from 33400000 on; gated
founders twelve fresh per step. The GA's worlds are never QC or gated founders.

### S0: Run 0, measure (authorized; running)

What a moment costs on the c7a.16xlarge, the g5.4xlarge (CUDA) and the Mac's Apple GPU (MLX), at the
four brain sizes and 8 to 256 streams; and development lives of imitation, names, attributes and
turns at the hand-set composition and its variants (where the ear enters, the self-hearing lesson,
the self-reward, depth, recall, heads, the actor's rate, the sensory scale, the hold).
**QC:** the rates read on every machine; the development lives' curves; the plan re-costed.
**Spend cap:** $10.

### S1: the map (the parrot-brain, physics and attention lanes build it; this lane reads it)

**QC question set:** where is each of 4 to 6 things, in view and out of view (behind, in the blind
sector, after the head turned away); a thing starts rolling from rest: does the parrot look at it
at once, and does it hold a steady roller less? A thing moved unseen.
**Pass bar:** the parrot-brain lane's E1 and E2 bars (#332); the look at a starting roller within
0.3 s on .90 of trials (#337's gate).
**Demo:** the page, a thing placed behind the parrot and asked for; a ball starting to roll.
**Designed parts left:** the map's identity rules (born, the owner's ruling); the tectum's salience
(born); attention's surprise signal (born, #337).
**This lane's spend:** under $2 (the readings on the lanes' brains).

### S2: sounds as symbols (turns, imitation, names by pistachio)

What the teacher does, in order: turns before words (T0 a reply in the gap pays, with the yoked
parrot as control; T1 only replies after questions pay, questions rising and statements falling,
with prosody-only and word-only probes; T2 first an echo pays, then only the answer); babbling and
imitation (M1); the trainer holds a thing up and names it, then asks "what is this?" (M2, the
model/rival game: a second trainer answers first and the parrot may say it after); a name, and
the parrot turns its head to that thing (M3). The trainer's reward delay (0.5 s, the catch) and
`actor_hold` are stated in the protocol; the delay is read at 0.5, 1, 2 and 4 s with and without
the praise word.
**The GA (lineage L1)** selects the composition on short lives of this step: the named genes
(depth, widths, apical fraction, where the ear enters, heads, `actor_hold`, rates, the self-hearing
lesson and self-reward, reply suppression), population 32, 20 generations, one world a genome,
and random search at equal evaluations; fitness = held-out tests and the pistachio curve's area,
less 0.2 times the multiply-accumulates over the hand-set's.
**QC question set (3 fresh founders, 120 trials each):** 40 turns (20 questions, 20 statements);
40 heard words to say back; 40 "what is this?" on held-out instances; 20 names to look at.
**Pass bar:** replies after .85 of questions and .15 or fewer of statements, in the gap; the yoked
parrot within .05 of the gap's share of time; words said back .90 (the parrot's own ear), a second
measure by the hearing lane's ear; names .90; looks .90. Then the gated reading on twelve founders
beside the controls.
**Demo:** a trainer asks "what is this?" holding up the duck; the parrot answers in its voice (the
replayed record of the word as it heard it, Piper voices with licences that allow it); the scene,
the question's audio and the answer, with an English gloss.
**Designed parts left:** the stand-in ear code (until #329's ear and the learned code); the voice
replaying stored frames (the owner's ruling); the praise word (the chamber's); the born rules of
self-hearing; the answer timing of #326 and #329 is replaced here by the learned turn.
**#336:** the selected brain read with and without the strength of records (the word's record kept
by repetition, the pistachio moment kept at once).

### S3: attributes: colour, shape, material, size, then purpose by biting

What the teacher does: "what is white?", "what colour is the duck?", "what is round?", "what is
made of glass?" (wood, metal, glass, cloth, fruit flesh, stone), "what is big?"; then purpose: the
parrot may bite the thing in focus (the beak slot); an apple, a grape or a pistachio pays when
bitten, a block, a cup or a key does not; then "can you eat it?" and "what can you eat?".
**QC question set (3 fresh founders):** 20 questions each of colour, shape, material and size on
held-out kind-attribute pairs; 20 of purpose on kinds never bitten in training (edible by their
material); the silencing probe for one concept of each family.
**Pass bar:** .90 on each family; purpose .85 on unbitten kinds; silencing a concept's units drops
its answers by .50 or more and every other concept's by .10 or less (M7). The plain readout fails
on held-out pairs by construction.
**Demo:** "what is made of glass?" over a room of six things; "can you eat it?" with an apple held
up, then a key.
**Designed parts left:** given features (until the retina lane's learned features); the material
cues as given channels; the concepts lane's designed parts (prototypes, the rule that decides which
sense a word names) stay out of this brain's path.

### S4: relations through the symbols

What the teacher does: "what is near you?", "what is far?", "what is next to the apple?", "what is
behind the cup?", "what is left of the ball?", taught through the words of the things (the
relations lane's part, #327, as features of pairs where the brain needs them).
**QC question set:** 60 relation questions on held-out rooms and on relation-thing pairs never
asked; 20 that change only the scene (a thing moved, the answer changes).
**Pass bar:** .90; the changed scene changes the answer on .95.
**Demo:** "what is next to the apple?", then the apple moved and the same question.

### S5: counting and conjunction: "two round things": "the apple and the ball"

What the teacher does: answers of several words through a planning window of voice slots; "how
many red things?", "what are two round things?", "is there a ball?"; number learned (no count
neurons: the learning study's untrained number region as a born gene, its words learned).
**QC question set:** 40 how-many (0 to 4), 40 several-thing answers, 20 is-there.
**Pass bar:** .90, .85, .90; answers said whole, in the voice.
**Demo:** "what are two round things?" answered "the apple and the ball".
**Designed parts left:** the untrained number region (born, a gene; the count neurons of #328 as
the control).

### S6: questions, then complex grammar

What the teacher does: every question form over the grammar of the parrot's world (the generator,
depth as a knob): word-order twins; several attributes, relations and relations of relations; and,
or, not; clauses with was ("the ball that was behind you"); was, is and will (the time lane's
records and forecast); answers as short sentences with number agreement.
**The GA (lineage L2)** selects the layout (each projection's density, sign and scale, which exist)
on short grammar lives.
**QC question set:** 100 questions: twins, depth 1 to 3 (trained to 2), held-out word pairs.
**Pass bar:** twins .90; held-out pairs .85; depth 3 .75; sentences .85, agreement .95.
**Demo:** "what is the ball left of?" against "what is left of the ball?" in one room.

### S7: open questions, a larger vocabulary, English and the listening test

What the teacher does: the vocabulary grows to 200, 500 and 2,000 words (more kinds and
attributes, numbers to ten, events, words defined by words), then the 2,000 most common English
words (#307's list) in place of nonce forms for the grounded part, beside the nonce gates.
**QC: the final exam** (section 1), with the listening test: Whisper reads the parrot's spoken
answers to English questions at a declared word error rate, and a short sample for the owner.

## 4. Controls in every gated reading

The audit's plain readout and table in the parrot's program (each reads the set of words heard,
with no order among them, so word-order twins, combinations never heard and bindings across slots
are gates they fail by construction); an LSTM and a transformer given the same experience (the
same observations, actions, pistachios and self-hearing lessons), online and offline, at the
brain's parameter count; the brain's lesions (the ear at the lowest region, no apical
compartment, no recall, one head, no self-hearing lesson, the pistachio withheld or at random, the
yoked parrot); the hand-set and random compositions of the same size; parameters and
multiply-accumulates in every table.

## 5. Compute: measured rates and the cost of each step

**Measured in S0** (one core a life with 42 lives on the c7a.16xlarge; one A10G; the Mac's Apple
GPU under a load of 130 from other sessions); about 92 sweeps a stream-moment on every machine:

| Brain | Synapses | CPU core, ms a stream-moment | A10G, 64 streams | A10G, 256 streams | Mac MLX |
| --- | --- | --- | --- | --- | --- |
| Words steps (S2 to S5) | 402,688 | 18 to 21 (2.0e9 to 2.2e9 a second) | 1.5 (2.4e10) | 1.3 (2.8e10) | 6.1 at 64 streams (6.1e9) |
| Grammar (S6) | 1,597,952 | 93 (1.6e9) | 4.4 (3.4e10) | 3.9 (3.8e10) | 17 at 16 streams (8.7e9) |
| 500 words | 4,258,560 | (S0 reading) | 15 at 16 streams | (S0 reading) | (S0 reading) |
| 2,000 words | 11,267,072 | (S0 reading) | (S0 reading) | (S0 reading) | (S0 reading) |

**Per dollar** at the highest spot prices (c7a.16xlarge $1.17 an hour, g5.4xlarge $0.89): a million
stream-moments of the words brain cost $0.10 on the CPU box and $0.37 on the GPU; of the grammar
brain $0.47 and $1.09. Brain lives therefore run on CPU boxes (one life a core) and on the Mac's
Apple GPU (one life at a time, free); GPUs run the LSTM and transformer controls.

| Step | Runs | Box-hours | $ spot (highest) |
| --- | --- | --- | --- |
| S0 | Run 0 | c7a.16xlarge 4, g5.4xlarge 2.5 | 7 |
| S1 | Readings on the lanes' brains | c7a.16xlarge 1 | 1 |
| S2 | Development (8 h of one box); GA L1: 640 lives and 640 random at about 0.8 core-hours (150,000 stream-moments); QC; gated: 12 founders, 6 arms at about 6 core-hours; controls on a g5.4xlarge (10 h) | c7a.16xlarge 8 + 16 + 1 + 7, g5.4xlarge 10 | 46 |
| S3 | Development; QC; gated (12 founders, 5 arms); controls | c7a.16xlarge 4 + 1 + 5, g5.4xlarge 6 | 17 |
| | **Phase A (S0 to S3)** | | **71** (cap $94 with reruns) |
| S4 | Development; QC; gated; controls | c7a.16xlarge 8, g5.4xlarge 6 | 15 |
| S5 | Development; QC; gated; controls | c7a.16xlarge 12, g5.4xlarge 8 | 21 |
| S6 | GA L2: 640 lives and 640 random at about 8 core-hours (grammar size, 300,000 stream-moments); QC; gated: 12 founders, 5 arms at about 40 core-hours; controls at 3.2 million parameters on GPUs (50 h) | c7a.16xlarge 160 + 38, g5.4xlarge 50 | 276 |
| S7 | 200 and 500 words (GA L3, gated); 2,000 nonce words and English, gated on CPU boxes (about 570 core-hours a life at 2,000 words); controls on GPUs (150 h); the listening test | c7a.16xlarge 250, g5.4xlarge 150 | 426 |
| | **All steps** | | **809**; with a quarter for reruns **1,010** |

S4 to S7 wait for the owner. The figures for S6 and S7 are re-read from S0's 500-word and
2,000-word rates before they are asked for.

**Stop rules:** a step whose spend reaches its row plus a quarter stops and reports; a step that
fails its QC stops, locates the term and re-plans; a box is terminated the moment its work is in
S3, and the termination checked.

## 6. Where results and brains go

The lane's bucket `cadence-0100-meaning-20261113`: `meaning334/<step>/` (each life's readings as
JSON, lineage records as `evolve` writes them, logs), brains of every gated life (`Brain.save`),
the QC demos (video and audio). Boxes sync every 60 s, terminate on shutdown and carry a 24-hour
guard. Small tables come back to the lane branch; protocols and readings go on #334.

## 7. Dependencies on other lanes

The parrot-brain lane (#332: the eyes, the map, the turn wiring and the page); the attention lane
(#337: the focus drawn by surprise, S1); the physics lane (#324: the map's motion); the hearing
lane (#329: the ear's codes, the voice's frames, the synthetic voices); the relations lane (#327,
S4); the concepts lane (learned boundaries, S3); the retina lane (#330, S3 onward); the time lane
(#326 W8 to W10, S6); the heads lane (#325, the recall rungs and S6); the memory lane (#336: the
strength of records, read with and without from S2); the paths lane (a lesson that teaches one
motor slot and leaves another open); the sentence lane (#308: the shared sentence record and the
probe of the association's settled state).
