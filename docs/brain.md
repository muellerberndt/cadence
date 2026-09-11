# What a brain has, and what it is in a patch net

Bernhard's rule for this library: every feature of an animal brain should have a simple
constituent in Cadence, and the fewer we have to build in, the better. The basic building
blocks of a brain are simple; they must be. This page is the table: each feature, the
simplest thing it is in a patch net, whether it had to be built or falls out of settlement
under detuning, where it lives in the library, and what has been measured. It ends with the
thesis on language and the status of the work.

## The primitives

Everything below is one of five things, and nothing else was added to get it:

| primitive | what it is | in the library |
|---|---|---|
| an **owner** | one patch of state, a potential and what it publishes | `Wiring`, `GradedRule` |
| a **seam** | one weight between two owners, moved by what its own two ends did; a fast copy that the learner writes and a slow copy that consolidation keeps | `Learner`, `Seams` |
| a **clamp** | a drive held on an owner from outside: an input, or the net's own previous state | `Settlement.settle_batch(drive)`, `Echo` |
| a **nudge** | a drive on some owners toward a target; the only door a goal enters by | `Nudge` |
| a **modulator** | one number broadcast to many seams or owners: a rate, a gain, a temperature, a dopamine | `LearnerConfig.eta`, `log_gain`, `ActorCritic` |

Settlement under detuning is the one mechanism: clamp something, let every owner repair its
own patch until nothing moves, read the rest state. Learning is the same settlement with a
nudge. Every row in the next table is a reading of that mechanism.

## The table

| feature of a brain | what it is in a patch net | built in, or falls out | in the library | status |
|---|---|---|---|---|
| equilibrium versus surprise | rest is the state where nothing moves; surprise is the work an input forces: the total movement of the activations between the state the net was in and the rest it reaches (`repair`) | falls out; the engine reports it | `SettledState.repair` (0.6.0) | reported on every backend; to be measured against novelty on the language stream. The step count does *not* track surprise (C4: 20 steps on clean, noisy and occluded digits alike), because a tolerance stops a settlement, not its cause |
| predicting the next moment | the free settlement's output *is* the prediction; the moment that then arrives is the nudge; the difference is what every seam learns from | falls out; this is the learning rule | `Learner.step` | measured: next character (05, 11), next chord (07), next action of a teacher (03, 06) |
| thinking longer for harder problems | a settlement runs until its owners stop moving, so an input the state disagrees with takes more steps; "sure" is when the output stops changing | falls out, weakly so far | `tolerance` and `steps` on every settlement; `repair` | measured negative on classification: 20 versus 23 steps for clean versus occluded MNIST (C4), because a stateless net has nothing to disagree with. The place it should show is the stateful net, where the carried state and the input can conflict; not yet measured |
| high parallelism (seven things at once; walking while talking) | disjoint ranges of owners settle simultaneously in one net; two learners can share one net and move only their own seams; the block transport makes the ranges explicit | falls out of the topology | `layered`, `trainable_overlaps`/`trainable_owners`, `blocks` | in use: actor and critic in one net (cart-pole), the composer's sub-nets; the number of things held apart at once is the fast-seam capacity, 128 pairs at recall 1.00 (C5) |
| muscle memory (a practised task hardcodes) | a practised decision settles warm in two steps from where the net already is; what the fast seams learned is consolidated into the slow seams during sleep | falls out (warm settlement); built in (consolidation, one line) | `settle_batch(state=)`, `Seams.sleep(consolidate=)` | measured: two warm steps per decision on cart-pole (E7); consolidation measured on permuted MNIST and it did not help at the settings tried (C1 sleep arm worse) |
| short-term memory (reverberation) | context owners clamped to a leaky trace of the hidden owners' own previous equilibria, so a state fades over ``1/(1-decay)`` inputs; and the last inputs written into fast seams | built in: one range of owners and one line, `c <- decay c + (1-decay) h` | `stateful`, `Echo` (0.6.0); fast seams in `Seams` | the carried-state test passes (a window of one names the symbol seen one input ago); associative recall 1.00 at 128 pairs with zero trained seams (C5); the language row is running |
| long-term memory (plasticity) | the seams themselves, moved by the local rule; the slow copy kept across sleeps | falls out; this is what a seam is | `Learner`, `Seams` | measured: continual permuted MNIST parity with backprop early, a three-point drift by task 30 (C1); one head on split MNIST forgets totally, like every learner tried (C2): the honest limit |
| emotions and neuromodulators steering behaviour | one broadcast number: a dopamine owner carrying the temporal-difference error to every seam's trace; a gain on a range of owners; the rate and temperature of the rule | built in as one owner and one number | `ActorCritic` (dopamine), `log_gain`, `LearnerConfig` | measured: cart-pole 5 of 5 seeds to 500 with the threshold at 40k steps (gate); the composer's moods steer what it plays (rung 10) |
| sleep | consolidate the fast seams into the slow, downscale the rest, prune what fell below a floor, sprout where owners kept firing together | built in, one function | `Seams.sleep` | measured on permuted MNIST: worse at the settings tried (C1); a bug in consolidation was found and fixed on the way (0.5.0) |
| attention | one step of a settlement in a Hebbian memory is an attention read; a gain on a range, driven by another owner, is attention as a seam | falls out (the read); the multiplicative seam is not built | `log_gain`; `c5_recall` | measured: recall 1.00 at every length with zero trained seams, where a trained transformer fails beyond its window (C5); the multiplicative seam is on the roadmap |
| habituation, adaptation, rhythm | a slow variable per owner that follows its own activation and subtracts from its drive; with mutual inhibition it turns a fixed point into a rhythm | built in, one variable per owner | `Adaptation` | measured: the half-center oscillates; the fly gates fail without it (FB3) |
| imagination and dreaming | settle with a candidate action clamped and feel it in the critic; keep what felt best; learn from the dream with the slow seams as the reference | built in as a settlement with a different clamp | `DreamActorCritic` | measured: the composer composes from silence by imagining and feeling phrases (rung 10); on Hopper the dream learner did not learn (gate) |
| curiosity | surprise as reward: the repair of a settlement fed to the dopamine owner | falls out of the two rows above; not wired | `repair` + `ActorCritic` | not measured |
| recovery from damage | the same rule keeps moving the seams that are left | falls out | any learner | measured: half the pixels dark or half the hidden owners gone, the patch net recovers to backprop's level or a point above within a hundred updates (C3) |
| homeostasis | each owner's incoming strength conserved; an activity floor | built in, one line | `Seams.conserve` | measured in the composer's life (rung 10) |
| deciding | the most active output owner at rest; outputs that inhibit each other so one wins; a softmax read of a population | falls out of lateral seams and the readout | `layered(lateral=)`, `Population`, `Bins` | in use on every classification and control rung |
| growth and pruning | seams sprout between owners that kept firing together and are pruned when they fall below a floor, within a budget per owner | built in, in sleep | `Seams.sleep(prune_below=, sprout_above=)` | measured: did not help on permuted MNIST (C1) |
| a body | the net inside a control loop, learning from the reward it gets | the rule unchanged; a trace per seam and a critic | `ActorCritic`, `embodied.md` | measured: cart-pole ahead of PPO on every row but wall-clock; Pong, Hopper, the Ant far behind: credit through time is the open gap (gates) |
| language | see the thesis below | | `stateful`, `Learner(slots=)` | the rows are running |
| specialised parts, one percept | regions of owners with their own projections settle in one net; the percept is the joint equilibrium of all of them, which is what a settlement is | the percept falls out; the parts are a mix of a constitution laid down before learning and specialisation during it | `Constitution`, `grow`, `evolve` (0.6.0); `Seams.sleep` for the specialisation | the wirings used so far were designed or read from a connectome; selection over constitutions is built and tested on a toy fitness, not yet run on a task |

Rows that fall out are the ones to keep: the fewer lines they take, the more the claim that
the mechanism is the brain's is worth. The rows that had to be built in each cost one owner,
one variable, or one line, and none of them adds a second mechanism.

## Where the wiring comes from

An animal's parts are a mix of two things: a constitution the genome lays down before any
learning (which regions exist, how large, which project to which, with what sign and
density) and specialisation during a life (seams that differentiate, sprout and prune). If
that is right, then evolution of the wiring has to be part of Cadence as a phase before
learning, or every wiring has to be designed. So far every wiring here was designed
(`layered`, `embedded`, `stateful`) or read from a connectome, which is a wiring evolution
designed. `cadence.constitution` is the simplest version of the missing phase: a
`Constitution` is a few numbers per region and per projection; `grow` develops it into a
wiring with named sets, deterministically for a seed; `mutate` perturbs sizes, densities,
signs and scales; `evolve` keeps, over generations, the constitutions whose grown nets score
best under a fitness the caller supplies (a protocol score, a learning curve, a held-out
accuracy). It adds no sixth primitive: the result is a wiring, and everything after it is
settlement under the same rule. Development during a life is `Seams.sleep`. The percept
that integrates the parts needs nothing: a settlement is one rest state of every region at
once.

## How a patch net becomes a language model

A transformer chooses the next token. It is trained to do that and only that, and it is
good at it because the corpus is static and the gradient is global. A patch net trained the
same way is a worse transformer: the text rung (05) says so, 3.33 bits per character
against 3.08, and the sizing study says more width will not close it.

Humans do not compute the next word. They think until they know what they will say, and
then they say it; the sentence is settled before the mouth reads it out. That is a
settlement, not a sequence, and it says what the patch net's language model should be:

1. **An utterance is an equilibrium, not a sequence.** The net holds a span of characters
   at once, one slot of output owners per position, and settles on the whole span jointly.
   The slots constrain each other through the seams between them and through the hidden
   owners, so a span that does not agree with itself keeps moving. `Learner(slots=K)`
   is that: one softmax choice per slot, all nudged together toward the span that was
   actually written. Reading out is serial only at the mouth.
2. **The context is a carried state, not a window.** What the net has already read or said
   lives in its own previous equilibria (`Echo`), fading over a chosen span, and in fast
   seams for what was recent. The window shrinks to a few characters; the state does the
   rest. That is the row where owned state and a window differ, so it is the row the paper
   needs.
3. **Thinking is settling until sure.** A hard span takes more steps and more repair before
   the slots stop moving; an easy one settles at once. "Sure" is a margin that stopped
   changing, which is a tolerance on the output owners. Reasoning models buy this with
   more tokens; here it is the same settlement run longer, organically, with `repair` as
   the measure of how much thinking a span took.
4. **It is trained on text alone**, because prediction of a span is still prediction of
   text, but what the state grounds in is the text's own regularities. Grounding in
   feelings and objects first, and names on top, needs a body or a picture stream feeding
   the same net; the sign writer and the paddle are where the library has one, and joining
   one of them to a language readout is the experiment that would test the sentence rather
   than assert it.

What this predicts, and what will be measured: bits per character of a span settled jointly
against one character at a time on the same corpus; the repair per span against the span's
difficulty; and whether a stateful net with a window of four beats a windowed net with a
window of thirty-two. If the joint settlement does not beat the serial one, the thesis is
wrong as stated and the table will say so.

## Status

Built in 0.6.0 for this page: `Echo` and `stateful` (short-term memory as reverberation),
`Learner(slots=)` (joint settlement of a span), `SettledState.repair` (surprise measured),
`Constitution`/`grow`/`evolve` (the wiring's origin as a phase before learning). Each is one
range of owners, one line, one number reported, or a few numbers per region; the mechanism
is unchanged and the owner-by-owner certificate still holds to rounding.

Running: the windowed language model (window 32, 2 million characters of twenty science
fiction novels), the stateful one (window 4, a carried state of 512 owners, decay 0.5) on
the same characters, both against a same-window MLP and a one-layer transformer trained on
the same text; the sizing study on MNIST and Shakespeare. Not yet run: the span row, the
repair-versus-novelty row, curiosity, attention as a seam, a language readout on a body.

Deliberately not built: anything that is not an owner, a seam, a clamp, a nudge or a
modulator. If a feature above needs a sixth thing, the table is wrong before the code is.
