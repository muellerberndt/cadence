# The parrot learns what words mean: the training plan (cadence#334)

For the owner's approval, with its budget, before any box starts. Every run below waits for that
approval; the phases after the first wait for the readings of the phase before. Development
readings in this plan are marked as development: founder 900, on one thread of a shared machine,
no gated life.

## 1. What the plan trains

One African Grey brain, one deep Cadence brain, learns what words mean the way a young parrot
does: it learns when to speak, then to say back what it hears, then pistachios for the right
answers to the right questions. Nothing about meaning is built in. The question's words reach the
deep layers top-down through the apical compartments; the recall memory's heads bring back the
words it heard; the pistachio teaches through the actor-critic's dopamine. The curriculum climbs
from turn-taking to names, attributes, places, combinations, grammar and a vocabulary of 2,000
words, each rung gated on twelve fresh founders beside matched controls.

The rule for every part: never engineer what can be learned. Only the world and the teacher are
designed. The composition of the brain (its regions, sizes, projections, heads, rates) is selected
by a genetic algorithm across lives, with the hand-set composition as the control; within a life
only the brain's own rules learn.

## 2. The brain

| Part | In Cadence | Born or learned |
| --- | --- | --- |
| Eyes | Each thing the parrot holds is a slot of tuned inputs: hue (a circle, weighted by saturation), lightness, six shape dimensions, size, direction from where the head points (a circle), distance, salience. Slots come in an order with no meaning. First rungs: these features given by the chamber; later rungs: the retina lane's learned features (#330) | The tuned inputs are born (colour-opponent, shape and head-direction cells are present at hatching); every category over them is learned |
| Ear | The code of each heard word: first a stand-in sparse code per form, then the hearing lane's ear (#329), then a code of the log-mel the brain learns itself (the owner lists template ears among the designed parts to replace). A pitch contour of the word's end (rise, flat, fall) for the prosody of questions | Pitch is a born auditory dimension; what a contour means is learned |
| Processing chain | `Brain.compose` regions `module_0 ... association` (hand-set: 128, 128 and the 128-neuron association) | Learned by the brain's lessons and rewards |
| Where the ear enters | The eyes reach the lowest region; the ear reaches the association region. What is heard reaches the regions below only top-down, through the apical compartment (`apical=0.25` hand-set): the question's words raise the drive of the features and places they name. (The caudolateral nidopallium is reciprocally connected with the secondary sensory areas of every modality: the wiring study, `docs/parrot-brain.md` section 2.8 on lane-0100-parrot.) The alternative, every input at the lowest region as `compose` wires it, is a gene and a control | The gain the words give each feature is learned |
| Association and working trace | The caudolateral nidopallium: holds the question while it is heard | Learned |
| Recall memory | `recall=True` with `recall_heads=2` hand-set: past moments (what was heard and seen) are fetched back into the association region; each head's address is learned | Learned addresses; the store is born |
| Voice | A motor slot of one unit for each word the parrot has heard (the unit replays that word's remembered sound, #329's voice) and one for silence; the voice can act at every moment | Which unit, and when, is learned |
| Gaze | A second motor slot: a turn of the head to one of 8 directions | Learned |
| Pistachio | The actor-critic (`ActorCritic`): the pistachio is the reward; the dopamine is the critic's prediction error; eligibility traces bridge the catch's delay (`actor_hold` a gene) | Learned |
| Hearing itself | The self-hearing lesson: after it says a word, the parrot hears it and takes a lesson that its act made that sound (the inverse model of its voice, built by babbling); and its own comparison of what it said with what it heard just before, a reward it gives itself (the songbird's performance error: the wiring study, section 2.10) | Born rules (genes `babble_lesson`, `self_reward`, each with the rule switched off as its control); what they teach is learned |

**Nothing designed in the path:** no concept unit, no table from a word to a concept, no dispatch
from a question word to a read, no question detector, no reply timer, no designed attention to a
slot. The audit's checks (#326) run on every gated brain (M8).

**Development reading of the layout** (founder 900, composed, no settle): the hand-set brain holds
402,688 synapses and 941,863 parameters at 404 inputs and a voice of 41 words; the apical synapses
are those from the association to `module_1`, from `module_1` to `module_0` and from the motor
region to the association; the ear reaches the association through one-way synapses alone.

## 3. The world and the teacher

- **The room:** things of 12 kinds (a ball, a cup, a box, a book, a duck, a key, an apple, a lamp,
  a kettle, a drum, a vase, a clock), each kind a prototype in the six shape dimensions (round,
  long, hollow, handle, flat, tall) and a width, in 6 colours (red, yellow, green and blue by hue;
  white and black by lightness), placed around the parrot. Every instance varies (no two balls
  alike). "Round" spans four kinds, so an attribute is a direction across kinds.
- **The words:** every word is a nonce word drawn for the founder (the words chamber's forms),
  question words and grammar words included. The English names serve the reader alone.
- **Time:** a moment is 250 ms of the parrot's time; a word takes one moment; the gap after the
  trainer stops (0.3 to 2 s) is moments 2 to 8 after the last word; the pistachio is thrown and
  reaches the beak 2 moments (0.5 s) after the act, as on the page.
- **Streams:** one brain lives 16 rooms side by side (rows of one batch; 64 on a GPU), each its
  own trial; the brain is one.
- **The teacher's trials**, all of them designed: babble (silence, the parrot's turn), imitate
  (a word, the parrot's turn), scaffold (a second trainer answers first and the parrot may say it
  after: the model/rival game), turn (the parrot answers alone), gaze (a name, the parrot turns
  its head), test (held-out rooms, things and combinations, greedy, on a copy that learns nothing).
- **Held out in every rung:** test rooms; instances never seen; kind-colour pairs never shown in
  training; combinations of words never heard together; word-order twins; depths above the
  trained ones; new kinds named three times; founders never used in development or selection.

## 4. The curriculum and its gates

Each rung is a chamber of its own with its protocol declared on #334 before its first gated life.
The bars below are the drafts the protocols declare; each gate reads the mean over twelve fresh
founders and at least 10 of 12 founders.

| Rung | What the teacher does | Gates (draft) | Controls that fail by construction |
| --- | --- | --- | --- |
| **T0, the turn** | Utterances of 1 to 6 words; a pistachio for a vocalization in the gap (moments 2 to 8 after the trainer stops); speaking over the trainer or later pays nothing | Of the parrot's vocalizations, .80 in the gap; a reply to .80 of utterances. The yoked parrot (the same pistachios at times unrelated to its sounds, Goldstein, King and West 2003) develops no turns: in-gap share within .05 of the gap's share of time | The yoked parrot |
| **T1, questions and statements** | Only replies after questions pay; questions end rising (.8 of them), statements falling | Replies after .85 of questions and .15 or fewer of statements, on held-out sentences (and held-out voices with the ear). Probes (readings): prosody alone, question words alone | |
| **T2, what to say** | First an echo of the last word pays; then only the answer (the focused thing's word); an echo of the question pays nothing | Echo of the question words .05 or less; the answer at the M2 bar | |
| **M1, imitation** | Babble, then imitate: the teacher says a word | A heard word said back so the parrot's own ear takes it for the heard one, .90; with the hearing lane, an independent recogniser agrees at .90 | |
| **M2, names by pistachio** | A thing held up: "what is this?" | .90 on held-out instances of trained kinds | |
| **M3, understanding by acting** | A name; the parrot turns its head to that thing among 2 to 4 | .90 | |
| **M4, attributes** | "what is white?", "what is round?", "what colour is the duck?" | .90 on held-out kind-colour pairs; "round" on held-out round kinds | The plain readout (no binding of a thing's colour to its kind across slots) |
| **M5, places** | "what is behind you?", "where is the duck?", "what is next to you?" | .90 | |
| **M6, combinations** | "what is next to you and white?" | .90 on combinations never asked in training | The plain readout and the table |
| **M7, concept neurons** | (a reading after M4 to M6) | For each concept, the units tuned to it are found on held-out scenes and drawn; silencing them drops that concept's answers by .50 or more and every other concept's by .10 or less; silencing as many random units drops each by .10 or less | |
| **G1, word order** | Twins of the same words in another order: "what is left of the ball?" against "what is the ball left of?" | Both twins right on .90 of pairs | The plain readout and the table read which words occur and answer both twins alike: at most .50 by construction |
| **G2, phrases that compose** | Several attributes ("the small red ball"), relations and relations of relations ("the cup left of the ball behind you"), and, or, not | .85 on questions holding a pair of words never heard together; trained at depth 1 and 2, .75 at depth 3 | The table (no pair it never saw) |
| **G3, answers as sentences** | "the ball is behind me", "two red balls", "one ball" (a plural marker past one), how many, is there | Whole sentence right .85; number agreement .95 | |
| **G4, tense and clauses** | was, is and will (#326 W8 to W10, the records and the forecast); "the ball that was behind you" | was/is/will told apart .90; clauses .85 | |
| **V1 to V3, vocabulary** | More kinds (procedural shape prototypes), colours (11 basic terms), sizes, materials, numbers to ten, relations, events (moved, fell, rolled, hid), words from words ("a zorp is a red ball") | At 200, 500 and 2,000 words: names and attributes .90 on trained words, .80 on a new word after three namings, the grammar gates at their bars | |
| **E, English** | The same world with the 2,000 most common English words (#307's list) in place of nonce forms, beside the nonce-word gates | The grounded words at the nonce bars; every one of the 2,000 imitated (M1) at .90; meaning beyond the grounded words read where words define words | |
| **M9, beside the controls** | Every rung | The table of every arm with parameters and multiply-accumulates (section 6) | |
| **M10, in the browser** | The selected brain in the parrot's page (#332), learning from the visitor's naming and pistachios | The page's moment cost stated | |

The grammar's generator (`benchmarks/meaning/grammar.py`, development code on the lane branch)
builds every question form over a room with depth as a knob, checks that every description picks
out exactly the things its answer names, and gives word-order twins a shared key. Development
reading over 200 random rooms at depth 1 to 3: 5,575 questions; where, colour, is-round, what,
how many, is there and word-order twins; depths 0 to 3. Its record (`Sentence`: the forms heard,
the role of each, the meaning) is offered to the whole-sentence lane (#308) through the
coordination file.

## 5. The genetic algorithm: the composition first

**The genome** has two parts, as the parrot's wiring study lays out (`docs/parrot-brain.md`
section 4; genes named the same where they overlap):

1. **Named genes** (`cadence.meaning.SPACE`, 24 genes, for `cadence.genes`): depth (0 to 4
   regions below the association), the lowest region's width (32 to 1,024), the taper, the
   association's width, the apical fraction (0 to .6), the teaching point, the recall memory on
   or off, its heads (1 to 8) and capacity, the heard history's blocks, the working trace's
   amplitude, the carried gain, the efference copy, the sensory scale, the temperature, the
   actor's rate, eligibility, discount, critic rate and hold, the recall address's reward rate,
   where the ear enters (association, lowest region, both), the self-hearing lesson, the
   self-reward. `HAND_SET` is every gene's hand-set value.
2. **The layout** (`cadence.Genome` from `anatomy()`): every projection's density, sign and scale,
   which projections exist (the ear to each region, the eyes past the lowest region, the history
   and each head's regions), mutated by `cadence.genome.mutate` with the sensory and motor regions
   fixed and the working trace tied to the association.

**Fitness**: how well a genome learns within a short life, less its compute:

```text
fitness = 0.5 * mean held-out test over the life's rungs
        + 0.5 * mean share of pistachios earned over the life's turn trials (the learning curve's area)
        - 0.2 * multiply-accumulates / the hand-set genome's on the same worlds
```

The curve's area gives selection a gradient before the tests leave chance; the compute price
keeps a genome from buying accuracy with size alone. No gene can reweight its own fitness.

**Lineages:**

| Lineage | Genome | Life | Population, keep, generations, worlds a genome | Lives |
| --- | --- | --- | --- | --- |
| L1 | named genes | T0, T1, M1, M2, M4, M5, short: 16 streams by 4,000 moments | 48, 12, 30, 2 | 2,880 |
| L2 | named genes and the layout, from L1's best | G1 and G2 short: 64 streams by 4,000 moments | 32, 8, 20, 1 | 640 |
| L3 | widths, heads, recall capacity and the ear's code width, from L2's best | V2 (500 words) short | 16, 4, 8, 1 | 128 |

Every genome of a generation lives the same worlds (`evolve` seeds them), so differences within a
generation are the genome's. Lives run side by side, one process a core, one thread each.

**Controls** (the anatomy guide's selection protocol): the hand-set genome (each lineage's first
member); random search over the same space at equal evaluations (as many lives again); random
composition of the same size (genomes drawn over the space and kept only within 10% of the
hand-set's parameters). The selected genome is read on held-out worlds and twelve fresh founders,
never on the worlds it was selected on, and the claim is only what it gains over the hand-set and
over random search. Every outcome is kept, including genomes that prune what the sources put in.

## 6. The controls of every gated reading

- **In the parrot's program:** the audit's plain trained readout (a delta-rule readout from the
  same observation and heard history to the voice, taught by the same pistachios and the same
  self-hearing lessons) and the table (each heard word set's counts of paid answers). Each reads
  the set of words heard, with no order among them, so word-order twins, combinations never
  heard and bindings across slots are gates they fail by construction.
- **An LSTM and a transformer** given the same experience: the same observations each moment, the
  same voice and gaze actions, the same pistachios (REINFORCE with a running baseline) and the
  same self-hearing lessons (cross-entropy on the own act), at the brain's parameter count;
  online (one pass, as the brain) and offline (replayed until held-aside trials stop gaining).
- **The brain's own lesions:** the ear at the lowest region (the apical compartment carrying no
  question), the apical fraction at zero, the recall memory off or with one head, no processing
  region below the association, the self-hearing lesson and the self-reward off, the pistachio
  withheld, the pistachio at random at the same rate, the yoked parrot.
- Parameters and multiply-accumulates in every table.

## 7. Compute and budget

**Work of a moment** (development counters, founder 900): an act settles in 32 sweeps, a reward
step in 13 sweeps and two eligibility phases of 12, a self-hearing lesson in 67 sweeps a taught
row; about 91 sweeps a stream-moment in all. A moment's multiply-accumulates are its sweeps times
the brain's synapses:

| Brain | Synapses | Parameters | Multiply-accumulates a stream-moment |
| --- | --- | --- | --- |
| Words rungs (T, M), hand-set | 402,688 | 941,863 | 3.7e7 |
| Grammar (256-wide regions, 4 heads, 140 words) | 1,597,952 | 3,205,177 | 1.5e8 |
| 500 words (3 regions of 384, 4 heads) | 4,258,560 | 7,852,561 | 3.9e8 |
| 2,000 words (3 regions of 512, 8 heads) | 11,267,072 | 20,753,479 | 1.0e9 |

**Rates assumed until Run 0 measures them:** 2e9 multiply-accumulates a second a CPU core (the
tune guide's chamber passes read 1.2e9 to 2.2e9 for a brain of 24,384 synapses, where Python
dominates; larger brains spend more of a moment in the kernels); on one A10G GPU (g5) with 64
streams, 1e11 at 1.6 million synapses, 2e11 at 4.3 million and 3e11 at 11 million. Run 0 reads both; a GPU phase whose
measured rate is under half the assumed one is costed again before it starts.

**Prices** (us-east-1 spot, read 13 November 2026, lowest to highest zone): c7a.16xlarge (64
cores) $0.72 to $1.17 an hour, c7i.16xlarge $1.08 to $2.01, g5.4xlarge (one A10G) $0.71 to
$0.89, g5.12xlarge (four A10G) $2.19 to $5.29. The budget uses the highest zone's price;
on-demand in us-east-2 (c7a.16xlarge $3.28, g5.4xlarge $1.62) is the ceiling where spot has no
capacity.

**The runs, in order:**

| Run | What | Boxes | Box-hours | $ spot (highest) | $ on-demand ceiling |
| --- | --- | --- | --- | --- | --- |
| R0 | Measure: lives of the four sizes at 8, 16 and 64 streams, CPU and CUDA backends; the learning speed of imitation and names at the hand-set and four variants (development) | 1 c7a.16xlarge, 1 g5.4xlarge | 4 + 4 | 8 | 20 |
| R1 | Turns T0 to T2: development, then the declared run (brain, yoked parrot, LSTM, transformer; 12 founders) with the parrot-brain lane's wiring | 1 c7a.16xlarge | 4 | 5 | 13 |
| R2 | GA lineage L1 (2,880 lives at about 0.33 core-hours) and random search at equal evaluations | 3 c7a.16xlarge in parallel | 30 (10 wall) | 35 | 98 |
| R3 | Words rungs gated (M1 to M9, 12 fresh founders): hand-set, selected, random composition, 5 lesions, the selected brain with and without the strength of records (#336); the plain readout and table on CPU; LSTM and transformer online and offline on GPU | 1 c7a.16xlarge, 1 g5.4xlarge | 3.5 + 26 | 27 | 54 |
| | **Phase A (R0 to R3)** | | | **75** | **185** |
| R4 | GA lineage L2 (layout and grammar; 640 lives and as many random, about 0.11 GPU-hours each) | 4 g5.4xlarge in parallel | 140 (35 wall) | 125 | 227 |
| R5 | Grammar rungs gated (G1 to G4, 12 founders): selected, hand-set, random composition, 2 lesions; controls at 3.2 million parameters | 4 g5.4xlarge | 92 (23 wall) | 82 | 149 |
| | **Phase B (R4, R5)** | | | **207** | **376** |
| R6 | Vocabulary 200 and 500: lineage L3 with random search (256 lives), gated readings, controls | 4 g5.4xlarge | 164 (41 wall) | 146 | 266 |
| R7 | Vocabulary 2,000 (nonce words) gated: selected, hand-set, and the selected brain with the strength of records (#336); controls at about 20 million parameters | 4 g5.4xlarge | 206 (52 wall) | 183 | 334 |
| R8 | English: the 2,000 most common English words, grounded words and imitation of all, controls | 4 g5.4xlarge | 85 (21 wall) | 76 | 138 |
| | **Phase C (R6 to R8)** | | | **405** | **738** |
| | **All runs** | | 41.5 CPU box-hours, 717 GPU-hours | **687** | **1,299** |
| | **Cap asked, with a quarter for reruns** | | | **860** | |

The parrot-brain lane's box-hour estimate for the turn wiring (R1) is folded in when it arrives.
M10 (the browser) needs no box. Storage on S3 is under $5 a month.

**Stop rules:** a phase starts on the owner's approval of its readings and its cap. A run whose
spend reaches its row's spot figure plus a quarter stops and reports. A rung whose gated reading
fails is read, its term located, and its next protocol declared; three rounds that each locate a
new term without closing the gap stop the rung and report the options.

**The strength of records (#336, opt-in on its own lane).** Records strengthen by surprise and
by repetition, cross a kept threshold, and the rest fade. The rungs that depend on it:

- M1 with the ear: the voice replays the record of each heard word, which ten noisy takes settle
  into a clean form (#336 G4);
- M2 and M3 once a name is a record fetched by the focus (#333): the name's record kept by
  repetition, a foil heard once fading;
- the pistachio's moment kept at once (one write at high surprise, #336 G2);
- G4 (was, from the records of where things were) and V1 to V3 (a new word kept after three
  namings, words never used fading).

R3 reads the selected brain with and without it (one more arm, about 20 core-hours); lineages L1
to L3 carry it as a gene (on or off, and its kept threshold) once it lands; R7 reads the
2,000-word brain with it as one more arm. The budget above holds both arms.

## 8. Where results and brains go

- The lane's bucket `cadence-0100-meaning-20261113`: `runs/<run>/` (each life's readings as JSON,
  lineage records as `evolve` writes them after every generation), `brains/<run>/<founder>/`
  (the selected and hand-set brains of every gated life, `Brain.save`), `controls/<run>/`.
- Every box syncs its results to the bucket every 60 seconds and resumes from it when a spot box
  is reclaimed; every box terminates on shutdown and carries a 24-hour safety shutdown.
- Small result tables come back to the lane branch (`benchmarks/meaning/`), protocols and
  readings on #334, the selected brain to the page (#332) for M10.

## 9. Development readings so far (founder 900, one thread of a shared machine, no gated life)

- **The whole path runs**: babble, imitate, names, attributes and places, the self-hearing lesson,
  the pistachio and the greedy test on a frozen copy.
- **Imitation is slow at the hand-set rates.** With the ear entering the lowest region (the
  layout `compose` gives), 33 words and the scene in view, 8 streams: the logged blocks of
  imitation trials 1,600 to 2,720 read .17 of the words said back on average at the deep teaching
  point (chance .03), rising slowly; the founder teaching point reads chance (.03) over its first
  1,900 trials. The located terms for R0: the heard word enters the lowest
  region beside the scene's 340 inputs; the inverse model of the voice is taught by the
  self-hearing lesson alone at the deep point's small steps. R0 reads the ear at the top (the
  hand-set of this plan), the trials imitation needs to reach .90, and the self-reward's share.
- **The turn was a located fault of the first chamber.** The parrot first answered in a silent
  moment after the last word, while the self-hearing lesson pairs the sound heard at the present
  moment with the act: the inverse model was taught on one input and asked on another. The turn
  is the voice's own act at every moment (T0 to T2), paid in the gap.

## 10. What the plan asks of other lanes

- The hearing lane (#329): the ear's codes for the ear rung, and the voice's frame format (in the
  coordination file) for M1's independent measure.
- The paths lane: a lesson that teaches the voice slot and leaves the gaze slot open (its open
  label), for the self-hearing lesson beside the gaze.
- The heads lane (#325): its heads into the records for the recall rungs and G4.
- The parrot-brain lane (#332): the turn wiring (T0 to T2) and its box-hours; the page for M10.
- The sentence lane (#308): the shared sentence record (`Sentence`, `Item`) and the probe of the
  association's settled state.
