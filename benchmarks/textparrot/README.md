# The text parrot (cadence#338)

A parrot that reads a typed question about the things it sees, looks, and types its answer, one
word a moment: "what colour has the apple ?" red; "what is left of the apple ?" block; "where is
the apple ?" left; "what objects are round ?" apple cup; "what do you see ?" the blue block is
left far the red apple is left the yellow key is front the white cup is behind. Everything it
knows about the things (colour, shape, material, size, food, where they are) and the grammar to
speak about them is learned in one composed Cadence brain, pretrained to be an expert at this, and
then read from inside the brain. The declared protocol is [PROTOCOL.md](PROTOCOL.md).

| File | What it holds |
| --- | --- |
| `textworld.py` | The things, the eyes (what a glimpse reports), the places and relations, the question forms with their depth knob, the replies, and the parser that reads a reply back into the scene |
| `textchamber.py` | An exchange moment by moment (the question, the gap, the search sweep, the answer sweep, the closing) for any learner, taught by teacher forcing or DAgger or read free, and its score |
| `textparrot.py` | The brain (`Brain.compose`, its genome `HAND_SET`, the anatomy that takes the words read into the association region) and its life through the curriculum |
| `textcontrols.py` | LSTM and transformer (online, offline), the readout and the table, driven through the same moments |
| `textintrospect.py` | Concept codes: probes, neighbours, a word read alone, silencing a property, the regions and the question's switch, the owner's room, a changed scene |
| `textpistachio.py` | A new thing and its new word by pistachios alone |
| `textgates.py` | The arms, a founder's life and readings, the controls, the tables |
| `chat.py` | Talk to a trained parrot |
| `test_textparrot.py` | The chamber's contracts |

## An exchange

```text
teacher   what  objects  are  round  ?   |  (sweep: search)   |  (sweep: answer)            |
eyes      -     -        -    -      -   |  block apple cup  |  block  apple  .   cup   .   |  -
gaze                                      |  pass  pass  pass |                             |
voice                                     |                   |  _      apple  _   cup   _   |  .
```

The eyes sweep over the things in a fixed order whatever the question (born); the gaze holds the
thing a question names (learned), which then stays in view as the focus; at each thing the voice
says words, one a moment, heard back at the next, and its silence moves the eyes on. A word
reaches the brain at an input channel drawn per founder, so the forms carry no spelling.

## Run it

```text
python textgates.py life 900 d4a25 out/              # one life in an arm, every reading
python textgates.py control 900 transformer-online out/ --brain-record out/d4a25_900.json
python textgates.py read out/                        # the tables
python chat.py out/d4a25_900.npz 900                 # ask the parrot about its room
```
