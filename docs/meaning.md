# Meaning learned in one deep brain

A parrot learns what words mean the way a young bird does: it learns to say back what it hears,
then earns pistachios for right answers to the right questions. Nothing about a word's meaning is
built in. A concept (red, round, an apple, left, near, something to eat) is the settled pattern
of a deep brain's regions, and the words reach it from the ear and leave it through the voice.

The module is `cadence.meaning`, opt-in: a program that does not import it gets a library without
it.

| Part | What it is |
| --- | --- |
| `Eyes`, `Seen`, `Circle`, `Line` | The born senses of a thing: tuned inputs of its hue, lightness, shape, size, surface, direction from the head and distance, one slot a thing |
| `Forms` | A heard word's code: a sparse code drawn per word form, the same form always the same code |
| `compose`, `HAND_SET`, `SPACE`, `anatomy` | The deep brain from a genome of named genes: its regions, the apical compartment, where the ear enters, the recall memory, the voice, the actor-critic's rates |
| `host.MeaningParrot` | A trained brain on a page, stepped one moment at a time |

## A first brain

```python
import numpy as np
from cadence.meaning import Eyes, Forms, Seen, compose, observation

eyes, forms = Eyes(slots=3), Forms(seed=0)
words = ["apple", "ball", "red", "round"]
brain = compose({"recall": 0}, eyes.width + forms.width, len(words), ear=forms.width, seed=0)
apple = Seen(hue=0.0, saturation=0.9, lightness=0.5, shape=np.array([.9, .1, 0, .3, .1, .5]),
             width=0.08, direction=1.2, distance=1.5, salient=True)
x = observation(eyes, forms, [apple], "apple")   # the scene, then the code of the word heard
said = brain.act(x[None])                      # one unit a word, and one for silence
```

## The brain

- **The regions.** `modules=` regions end in the association region. The eyes reach the lowest
  region (and with `eyes_entry="both"` the association too). The ear reaches the lowest region,
  the association, or both (`ear_entry`), so a heard word and a seen thing settle in one chain.
  The apical compartment (`apical=`) carries what the upper regions hold down to the lower ones.
- **Two routes to the voice.** With `ear_motor`, what the ear hears also reaches the voice
  directly, as the song system's core carries auditory input to the vocal motor nuclei. Saying a
  heard word back then has synapses of its own, and the association's route carries meaning.
  Through one shared route, imitation and naming took each other's place.
- **The voice** is a motor slot of one unit for each word the parrot has heard (each says that
  word's remembered sound) and one for silence.
- **No unit names a concept.** A concept is the settled pattern of the regions; a linear probe
  finds its direction, and silencing it can be read (the chamber's exam).

## How it learns (the chamber `benchmarks/meaning`)

Every rule is the brain's own; only the world and the teacher are designed.

- **Hearing itself:** after the parrot says a word it hears it, and its own act is the lesson's
  label for that sound: the inverse model of its voice, built by babbling.
- **The pistachio makes the act lasting:** a lesson on the parrot's own reply, taken on the brain
  as it was when it replied, when a pistachio follows (three factors: the act, its trace, the
  reward).
- **The correction:** after a wrong reply the trainer says the right word. The voice's take of
  that word labels the state the parrot replied from: DAgger on its own states, the label the
  parrot's mirror of a heard word. The lesson is taken on the brain as it was at the reply, and
  it takes the living brain's optimizer step.
- **The word records:** a word the trainer gives as an answer is kept under the association's
  pattern of the moment the question ended. At the parrot's turn, the records its pattern recalls
  (where the three nearest agree) are replayed to its ear. A recalled record is rewritten under
  the present pattern once its reply is judged (reconsolidation).
- **Sleep:** between steps every efficacy is scaled down by half, so the brain can learn again.
- **The actor-critic:** the pistachio is the reward, its delay bridged by the eligibility
  (`actor_hold`).

The chamber's world has things of nine kinds in six colours and six materials around the parrot;
the steps teach imitation, names with the focus on a thing, colour, where (a direction and "far"),
relations (left of, next to, opposite), material and what can be eaten (by biting), and several
things at once. Its exam reads the owner's questions, probes of the concept code, neighbours, a
word heard alone, silencing, a new object and a new word by pistachios alone
([the chamber](../benchmarks/meaning/night.py), [the exam](../benchmarks/meaning/exam.py)).

## Born parts, each with its evidence

| Part | Evidence | Learned replacement |
| --- | --- | --- |
| The voice held while a trainer speaks (a question, a correction) | Feed-forward inhibition holds a call back while a songbird listens | Turn-taking learned from pistachios |
| The ear reaching the voice directly | The song system's core: auditory input to the vocal motor nuclei | None: anatomy |
| The correction's label: the voice's take of the trainer's word | HVC neurons fire alike when a bird sings a syllable and hears it | The parrot's own echo, once its inverse model holds in a question's context |
| The practice lesson kept off replies to a trainer's question | Social context gates vocal exploration: directed song suppresses it | None |
| Sleep between steps: every efficacy scaled down by half | The synaptic homeostasis of sleep | None |
| Reconsolidation of a recalled word record | A memory is labile when recalled | None |
| The glimpse: the thing in focus seen in a place of its own | The tectum's winner is looked at, and its window is read | A focus the brain moves itself |
| The scan: for a question about the whole scene, the focus visits each thing in turn | Serial search with inhibition of return | A learned gaze |
| A word's code drawn per form | The ear's record of a heard word | The hearing module's records |

## On a page

<!-- not-run: the page's files are a trained brain's -->
```python
from cadence.meaning.host import MeaningParrot

m = MeaningParrot("brain.npz", "spec.json")
m.see([{"kind": "apple", "colour": "red", "material": "fruit", "direction": 1.2,
        "distance": 1.5, "focus": True}])
for word in ("what", "is", "this"):
    m.hear(word)
    m.moment(more=True)          # the voice is held while the trainer speaks
reply = [m.moment() for _ in range(3)]   # the parrot's turn, in the gap
m.pistachio()                    # a right reply made lasting
```
