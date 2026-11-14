"""The night's chamber (cadence#334, plan v3): the parrot learns what words mean from pistachios.

**The glimpse.** The thing in focus (clicked, held up, or the one the trainers attend to while
they ask about it: joint attention) is reported again in a place of its own before the scene:
what the eyes see where the parrot looks, the tectum's winner gated in (born attention).

**The scan.** A question about the scene as a whole (what is red, what objects are round, what
can you eat, what is made of wood, what is left of the apple) is answered while the focus visits
each thing in turn (born serial search, inhibition of return): one moment a thing, its features
in the glimpse; the parrot says a thing's name while it is in view, so a question with several
answers is answered word by word.

**The world.** Things of eight kinds (an apple, a banana, a ball, a block, a cup, a key, a book, a
bottle), each in a colour, of a material (fruit, wood, metal, glass, cloth, plastic), around the
parrot. The eyes report a thing's hue, lightness, shape, size, surface (gloss, roughness,
translucency, softness), its direction from where the head points and its distance; the focus
(a click, or a thing held up) is the thing's salience. Nothing reports a kind, a colour, a material
or a place as such.

**The words.** Each word reaches the brain as the code of its sound, drawn per founder (the ear's
record of the word); the brain never sees a spelling. The English forms serve the reader and the
demo alone.

**An exchange.** The trainer's question, one word a moment (about 300 ms); in the model/rival game
a second trainer says the answer after it; then the parrot's turn: three moments from the gap
after the trainer stops, in which its voice says words or nothing (the question is then in the
heard history and the working trace, and in the present sound no longer). Its reply is the
words it said, in order. While a trainer has more to say the parrot's voice is held (the born
suppression of the voice while hearing, gene ``reply_suppression``; evidence: feed-forward
inhibition in the songbird's HVC while it listens).
A right reply earns the trainer's click, heard at once, and a pistachio in the beak
``CATCH`` moments later.

**The learning rules**, every one the brain's own:

- the self-hearing lesson: hearing itself say a word, the parrot learns that its act made that
  sound (the inverse model of its voice);
- the pistachio-gated lesson: a pistachio makes the act that earned it lasting. The lesson's
  label is the parrot's own reply (what it said), on the moment it said it; nothing tells it the
  answer (gene ``gated_lesson``; off is the control);
- the actor-critic: the pistachio is the reward, the eligibility bridges the catch
  (``actor_hold``);
- biting: in a bite exchange the parrot may bite the thing in focus (a beak slot); fruit pays
  (taste), the rest does not.

**The teacher's question forms** (``FORMS``): what is this (the focus), what colour has the X,
what is <colour>, what objects are round, where is the X (a direction word, and "far" when far),
what is left of / right of / next to / opposite the X, what is made of <material>, what can you
eat, can you eat the X, can you eat this (the focus on it). A kind never shown in training (a
lemon) is the exam's new object.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cadence.meaning import HAND_SET, Eyes, Forms, Seen, compose  # noqa: E402

# kind: (shape over round, long, hollow, handle, flat, tall; width m; materials; colours)
KINDS: dict[str, tuple[tuple[float, ...], float, tuple[str, ...], tuple[str, ...]]] = {
    "apple": ((.90, .10, .00, .30, .10, .50), .08, ("fruit",), ("red", "green", "yellow")),
    "banana": ((.15, .90, .00, .00, .20, .20), .18, ("fruit",), ("yellow", "green")),
    "ball": ((.95, .10, .10, .00, .10, .50), .12, ("plastic", "cloth", "wood"), ()),
    "block": ((.05, .30, .00, .00, .30, .45), .10, ("wood", "plastic"), ()),
    "cup": ((.85, .20, .90, .90, .20, .55), .09, ("glass", "plastic", "metal"), ()),
    "key": ((.10, .90, .00, .60, .70, .10), .07, ("metal",), ()),
    "book": ((.05, .40, .00, .00, .90, .20), .18, ("cloth", "plastic"), ()),
    "bottle": ((.40, .60, .80, .00, .00, .90), .10, ("glass", "plastic"), ()),
    "pistachio": ((.70, .40, .00, .00, .20, .20), .03, ("fruit",), ("green",)),
}
KIND_NAMES = tuple(KINDS)
# a kind never shown in training: the exam's new object, a fruit (no word, never bitten)
NEW_KINDS: dict[str, tuple[tuple[float, ...], float, tuple[str, ...], tuple[str, ...]]] = {
    "lemon": ((.80, .30, .00, .00, .10, .45), .08, ("fruit",), ("yellow",)),
}
ROUND = ("apple", "ball", "cup", "lemon")
COLOURS: dict[str, tuple[float | None, float, float]] = {
    "red": (0.0, .90, .50), "yellow": (60.0, .90, .60), "green": (120.0, .80, .45),
    "blue": (230.0, .85, .45), "white": (None, .05, .95), "black": (None, .05, .08)}
# material: gloss, roughness, translucency, softness
MATERIALS: dict[str, tuple[float, float, float, float]] = {
    "fruit": (.60, .20, .10, .75), "wood": (.20, .70, .00, .20), "metal": (.90, .10, .00, .00),
    "glass": (.90, .05, .80, .00), "cloth": (.05, .80, .05, .90), "plastic": (.55, .20, .25, .25)}
EDIBLE = ("fruit",)
DIRECTIONS = ("front", "right", "behind", "left")
FAR = 1.2  # metres: "far" from here on
QUESTION_WORDS = ("what", "is", "this", "colour", "has", "the", "objects", "are", "where",
                  "left", "right", "of", "next", "to", "opposite", "made", "can", "you", "eat")
ANSWER_WORDS = ("far", "yes", "no")
WORDS = (KIND_NAMES + tuple(COLOURS) + tuple(MATERIALS) + DIRECTIONS + ("round",)
         + tuple(w for w in QUESTION_WORDS if w not in ("left", "right"))
         + ANSWER_WORDS + ("zabu",))
# the trainer's bridge signal: a click heard at once after a right reply, a sound the voice cannot
# make (an animal trainer's clicker); a praise word would be imitated (the parrot said it in the
# silent turn)
PRAISE = "<click>"
MOMENTS = ("velocity", "velocity_bias", "second_moment", "second_moment_bias")  # learner moments
NEW_WORD = "zabu"  # heard and babbled from birth, never a name in training: the new-word gate's
CATCH = 2       # moments from a right reply to the pistachio in the beak
TURN = 3        # moments of the parrot's turn
TAIL = CATCH + 2
# a step ends once a frozen greedy reading of its forms reaches this share (read every 100 blocks
# once the parrot's own sampled turns earn SAMPLED_GATE of the pistachios)
MASTERY = {"N0": .92, "N0b": .9, "N1": .92, "N2": .9, "N3": .88, "N4": .88, "N5": .9, "N6": .85}
SAMPLED_GATE = 0.5
REHEARSAL = 0.3  # the share of a step's blocks that revisit the earlier steps' forms
NIGHT: dict[str, Any] = {"actor_hold": 0.93, "actor_eta": 0.1, "ear_entry": "both",
                         "history_decays": (0.0, 0.5, 0.8), "self_reward": 0.1,
                         "reply_suppression": 1, "gated_lesson": 1.0, "recall": 0,
                         "correction_lesson": 1.0, "vocal_until": "N6", "records_agree": 3,
                         "word_records": 1, "sleep_scale": 0.5, "correction_label": "given",
                         "babble_in_turn": 0, "ear_motor": 1, "eyes_entry": "both",
                         "lesson_moments": 2}
# the steps a night lives: echo across the gap (N0b) is left out, since the correction's label
# is the voice's take of the trainer's word and no longer the parrot's own echo of it
NIGHT_STEPS = ["N0", "N1", "N2", "N3", "N4", "N5", "N6"]


@dataclass
class Thing:
    kind: str
    colour: str
    material: str
    place: np.ndarray
    shape: np.ndarray
    width: float
    hue: float
    saturation: float
    lightness: float
    surface: np.ndarray
    focus: bool = False


@dataclass
class Room:
    things: list[Thing]
    heading: float = 0.0

    def seen_of(self, t: Thing) -> Seen:
        return Seen(hue=t.hue, saturation=t.saturation, lightness=t.lightness, shape=t.shape,
                    width=t.width, direction=bearing(self.heading, t.place),
                    distance=float(np.linalg.norm(t.place)), salient=t.focus, surface=t.surface)

    def seen(self, rng: np.random.Generator) -> list[Seen]:
        out = [self.seen_of(t) for t in self.things]
        return [out[int(i)] for i in rng.permutation(len(out))]


def bearing(heading: float, point: np.ndarray) -> float:
    """The angle of a point from where the head points, positive to the left."""
    return float(np.angle(np.exp(1j * (math.atan2(point[1], point[0]) - heading))))


def direction_of(heading: float, point: np.ndarray) -> str:
    a = bearing(heading, point)
    return DIRECTIONS[int(((-a + math.pi / 4) % (2 * math.pi)) // (math.pi / 2))]


def thing(rng: np.random.Generator, kind: str, place: np.ndarray, *,
          colour: str | None = None, material: str | None = None) -> Thing:
    proto, width, materials, colours = {**KINDS, **NEW_KINDS}[kind]
    colour = colour or str(rng.choice(colours or tuple(COLOURS)))
    material = material or str(rng.choice(materials))
    hue_deg, sat, light = COLOURS[colour]
    surface = np.clip(np.asarray(MATERIALS[material]) + rng.normal(0, .06, 4), 0, 1)
    return Thing(kind, colour, material, np.asarray(place, float),
                 np.clip(np.asarray(proto) + rng.normal(0, .06, 6), 0, 1),
                 float(width * math.exp(rng.normal(0, .12))),
                 math.radians(float(rng.uniform(0, 360)) if hue_deg is None
                              else hue_deg + float(rng.normal(0, 8))),
                 float(np.clip(sat + rng.normal(0, .04), 0, 1)),
                 float(np.clip(light + rng.normal(0, .04), 0, 1)), surface)


def free_place(rng: np.random.Generator, taken: list[np.ndarray]) -> np.ndarray:
    for _ in range(1000):
        r, a = rng.uniform(0.5, 1.8), rng.uniform(-math.pi, math.pi)
        p = np.array([r * math.cos(a), r * math.sin(a)])
        if all(np.linalg.norm(p - q) >= 0.4 for q in taken):
            return p
    raise RuntimeError("no free place")


def room(rng: np.random.Generator, count: int, held: set[tuple[str, str]], *,
         test: bool = False) -> Room:
    """``count`` things of distinct kinds; a kind-colour pair in ``held`` appears in test rooms
    alone."""
    kinds = [str(k) for k in rng.choice(KIND_NAMES, size=count, replace=False)]
    things: list[Thing] = []
    for k in kinds:
        for _ in range(50):
            t = thing(rng, k, free_place(rng, [x.place for x in things]))
            if ((t.kind, t.colour) in held) == test or (not test and (t.kind, t.colour)
                                                        not in held):
                break
        things.append(t)
    return Room(things, heading=float(rng.uniform(-math.pi, math.pi)))


# -- the teacher's questions and answers


def left_of(r: Room, a: Thing, b: Thing) -> bool:
    """``a`` is left of ``b`` as the parrot sees them: its bearing 15 to 110 degrees to the
    left of ``b``'s."""
    d = math.degrees(bearing(r.heading, a.place) - bearing(r.heading, b.place))
    d = (d + 180) % 360 - 180
    return 15 <= d <= 110


def opposite(r: Room, a: Thing, b: Thing) -> bool:
    d = math.degrees(bearing(r.heading, a.place) - bearing(r.heading, b.place))
    return abs((d + 180) % 360 - 180) >= 145


def next_to(a: Thing, b: Thing) -> bool:
    return a is not b and float(np.linalg.norm(a.place - b.place)) <= 0.7


@dataclass
class Question:
    words: list[str]
    answer: list[str]
    form: str
    focus: Thing | None = None
    several: bool = False
    scan: bool = False  # the parrot's turn looks over the things one at a time


def ask(form: str, r: Room, rng: np.random.Generator) -> Question | None:
    """A question of ``form`` about room ``r`` with exactly one right answer, or None where the
    room holds none."""
    things = r.things
    x = things[int(rng.integers(len(things)))]
    if form == "name":
        x.focus = True
        return Question(["what", "is", "this"], [x.kind], form, focus=x)
    if form == "colour_of":
        x.focus = True  # the trainers' joint attention on the thing asked about
        return Question(["what", "colour", "has", "the", x.kind], [x.colour], form, focus=x)
    if form == "by_colour":
        same = [t for t in things if t.colour == x.colour]
        return Question(["what", "is", x.colour], [x.kind], form, scan=True) \
            if len(same) == 1 else None
    if form == "round":
        kinds = sorted(t.kind for t in things if t.kind in ROUND)
        return Question(["what", "objects", "are", "round"], kinds, form, several=True,
                        scan=True) if kinds else None
    if form == "where":
        x.focus = True
        words = [direction_of(r.heading, x.place)]
        if float(np.linalg.norm(x.place)) > FAR:
            words.append("far")
        return Question(["where", "is", "the", x.kind], words, form)
    if form in ("left_of", "right_of"):
        found = [t for t in things if t is not x and (
            left_of(r, t, x) if form == "left_of" else left_of(r, x, t))]
        side = "left" if form == "left_of" else "right"
        x.focus = len(found) == 1
        return Question(["what", "is", side, "of", "the", x.kind], [found[0].kind], form,
                        focus=x, scan=True) if len(found) == 1 else None
    if form == "next_to":
        found = [t for t in things if next_to(t, x)]
        x.focus = len(found) == 1
        return Question(["what", "is", "next", "to", "the", x.kind], [found[0].kind], form,
                        focus=x, scan=True) if len(found) == 1 else None
    if form == "opposite":
        found = [t for t in things if t is not x and opposite(r, t, x)]
        x.focus = len(found) == 1
        return Question(["what", "is", "opposite", "the", x.kind], [found[0].kind], form,
                        focus=x, scan=True) if len(found) == 1 else None
    if form == "material":
        same = [t for t in things if t.material == x.material]
        return Question(["what", "is", "made", "of", x.material], [x.kind], form, scan=True) \
            if len(same) == 1 else None
    if form == "eat":
        kinds = sorted(t.kind for t in things if t.material in EDIBLE)
        return Question(["what", "can", "you", "eat"], kinds, form, several=True, scan=True) \
            if kinds else None
    if form == "can_eat":
        x.focus = True
        return Question(["can", "you", "eat", "the", x.kind],
                        ["yes" if x.material in EDIBLE else "no"], form)
    if form == "can_eat_this":  # the focus on a thing, its name unsaid
        x.focus = True
        return Question(["can", "you", "eat", "this"],
                        ["yes" if x.material in EDIBLE else "no"], form, focus=x)
    raise ValueError(form)


@dataclass
class Exchange:
    room: Room
    heard: list[str | None]     # the trainers' words, one a moment
    turn: int                   # the first moment of the parrot's turn
    question: Question | None
    bite: bool = False          # a bite exchange
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def scan(self) -> list[Thing]:
        """The things the focus visits in turn during the parrot's turn (a scanning question)."""
        return list(self.extra.get("scan", []))

    @property
    def turn_length(self) -> int:
        return len(self.scan) if self.scan else TURN

    @property
    def length(self) -> int:
        return self.turn + self.turn_length + TAIL


class Parrot:
    """The deep brain with its eyes, ear, voice and beak for one founder."""

    def __init__(self, seed: int, genome: dict[str, Any] | None = None, *, streams: int = 16,
                 slots: int = 5, backend: str = "cpu"):
        self.seed, self.streams = int(seed), int(streams)
        self.genome = {**HAND_SET, **NIGHT, **(genome or {})}
        self.eyes = Eyes(slots=slots, surfaces=4)
        self.forms = Forms(seed=seed)
        order = np.random.default_rng(np.random.SeedSequence([self.seed, 334, 2])).permutation(
            len(WORDS))
        self.voice = [WORDS[int(i)] for i in order]
        self.unit = {w: i for i, w in enumerate(self.voice)}
        self.silence = len(self.voice)
        for w in WORDS:
            self.forms(w)
        self.forms(PRAISE)  # the click's code, drawn after the words'
        self.ear = self.forms.width
        # the glimpse: what the eyes report of the thing in focus, in a place of its own (the
        # tectum's winner looked at, its features gated in: born attention), then the scene
        self.inputs = self.eyes.slot_width + self.eyes.width + self.ear
        # the meeting point of question and thing as a sparse mixed code (gene ``mixed_code``,
        # the review's repair): the glimpse and the trace of the words heard in the exchange,
        # through a fixed random expansion, the most driven share of its units on (as granule
        # cells and Kenyon cells code), seen beside the scene
        self.mixed = int(self.genome.get("mixed_code", 0))
        self.heard_trace = np.zeros((streams, self.ear))
        if self.mixed:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 334, 61]))
            self.mixed_w = rng.standard_normal((self.eyes.slot_width + self.ear, self.mixed))
            self.mixed_on = max(1, int(round(float(self.genome.get("mixed_on", 0.1))
                                             * self.mixed)))
            self.inputs += self.mixed
        self.brain = compose(self.genome, self.inputs, len(self.voice), ear=self.ear, gaze=2,
                             seed=self.seed, backend=backend)
        self.synapses = int(self.brain.connectome.synapses)
        self.work = {"moments": 0, "stream_moments": 0, "act_sweeps": 0.0, "lesson_sweeps": 0.0,
                     "reward_row_sweeps": 0.0, "pistachios": 0, "gated": 0}
        # the records of heard words (gene ``word_records``): a word heard as an answer is kept
        # under the association region's settled pattern of the moment the question ended, and
        # at the parrot's turn the record its present pattern recalls is replayed to its ear
        self.vocal_period = True  # the self-hearing lesson's sensitive period (gene vocal_until)
        self.correcting = True  # the correction lesson, from the step ``corrections_from`` on
        self.records: Any = None
        self.recalled = np.zeros((streams, self.forms.width))
        if int(self.genome.get("word_records", 0)):
            from cadence.longterm import LongTermMemory

            assoc = len(self.brain.connectome.populations["association"])
            if str(self.genome.get("records_key", "association")) == "input":
                assoc = self.eyes.slot_width + self.ear  # the glimpse and the word heard
            self.records = LongTermMemory(width=assoc, value_width=self.forms.width,
                                          capacity=2048, cosine=True, revise=0.995,
                                          seed=self.seed)

    def said(self, unit: int) -> str | None:
        return None if int(unit) >= self.silence else self.voice[int(unit)]

    def mixed_code(self, glimpse: np.ndarray, trace: np.ndarray) -> np.ndarray:
        """The sparse mixed code of the glimpse and the heard trace: the ``mixed_on`` most
        driven units of the fixed expansion on (1), the rest off."""
        drive = np.concatenate([glimpse, trace]) @ self.mixed_w
        out = np.zeros(self.mixed)
        out[np.argpartition(drive, -self.mixed_on)[-self.mixed_on:]] = 1.0
        return out

    def observe(self, exs: list[Exchange], t: int, own: list[str | None],
                praise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        rows = []
        if self.mixed:
            if t == 0 or len(self.heard_trace) != len(exs):  # an exchange begins
                self.heard_trace = np.zeros((len(exs), self.ear))
            decay = float(self.genome.get("mixed_decay", 0.7))
            for i, ex in enumerate(exs):
                word = ex.heard[t] if t < len(ex.heard) else None
                self.heard_trace[i] = decay * self.heard_trace[i] + self.forms(word)
        for i, ex in enumerate(exs):
            word = ex.heard[t] if t < len(ex.heard) else None
            code = self.forms(word) + self.forms(own[i])
            if praise[i]:
                code = code + self.forms(PRAISE)
            if self.records is not None:  # a recalled word, replayed to the ear
                code = code + self.recalled[i]
            seen = ex.room.seen(rng)
            focus = next((x for x in seen if x.salient), None)
            if ex.scan and 0 <= t - ex.turn < len(ex.scan):  # the focus visits each thing
                focus = ex.room.seen_of(ex.scan[t - ex.turn])
            glimpse = self.eyes.slot(focus)
            parts = [glimpse, self.eyes.scene(seen)]
            if self.mixed:
                parts.append(self.mixed_code(glimpse, self.heard_trace[i]))
            rows.append(np.concatenate(parts + [code]))
        return np.asarray(rows)


def reply_of(said: list[str | None], *, scan: bool = False) -> list[str]:
    """The words said in the turn, in order, a word said on consecutive moments once, up to the
    first silence after a word (in a scan, every word said while the focus looked over the
    things)."""
    out: list[str] = []
    for w in said:
        if w is None:
            if out and not scan:
                break
            continue
        if not out or out[-1] != w:
            out.append(w)
    return out


def right(q: Question, reply: list[str]) -> bool:
    if q.several:
        return sorted(reply) == sorted(q.answer)
    return reply == q.answer


def live(p: Parrot, exs: list[Exchange], rng: np.random.Generator, *, learn: bool = True,
         pistachio: str = "earned") -> dict[str, Any]:
    """One block of exchanges, one a stream. ``learn=False`` reads a frozen copy greedily.
    ``pistachio``: ``earned`` (for a right reply), ``withheld`` (never), ``random`` (at the
    rate it would be earned, at random)."""
    brain = p.brain if learn else copy.deepcopy(p.brain)
    if not learn:
        brain.reset()
    g = p.genome
    n = len(exs)
    moments = max(ex.length for ex in exs) + max(  # room for a trainer's correction
        (len(ex.question.answer) for ex in exs if ex.extra.get("dagger") and ex.question),
        default=0)
    own: list[str | None] = [None] * n
    voiced = np.full(n, p.silence, dtype=np.int64)
    beak = np.zeros(n, dtype=np.int64)
    reward = np.zeros((moments + 2, n))
    praise = np.zeros((moments + 2, n), dtype=bool)
    said: list[list[str | None]] = [[] for _ in range(n)]
    moment_x: list[list[np.ndarray]] = [[] for _ in range(n)]
    moment_unit: list[list[int]] = [[] for _ in range(n)]
    bitten = [False] * n
    replies: list[list[str]] = [[] for _ in range(n)]
    judged = [False] * n
    oks = [False] * n
    turn_x: list[list[np.ndarray]] = [[] for _ in range(n)]
    turn_t: list[list[int]] = [[] for _ in range(n)]
    moment_t: list[list[int]] = [[] for _ in range(n)]
    echo_due: list[dict[int, int]] = [{} for _ in range(n)]  # moment -> answer word index
    echoes: list[tuple[int, int, int]] = []  # (stream, word index, the echo's unit)
    # the brain as it was at each moment of a turn, before it acted: a lesson about a reply is
    # taken on the state the parrot replied from, and its change added to the living brain
    snaps: dict[int, tuple[Any, np.ndarray]] = {}

    def teach_at(entries: dict[int, list[tuple[int, int, int, float]]]) -> int:
        taken = 0
        for m, rows_of in entries.items():
            if m not in snaps or not rows_of:
                continue
            snap, xm = snaps[m]
            labels = np.stack([np.full(n, p.silence), np.zeros(n, dtype=np.int64)], 1)
            w8 = np.zeros(n)
            for i, unit, bit, weight in rows_of:
                labels[i] = (unit, bit)
                w8[i] = weight
            if not hasattr(snap, "learner"):  # a control: the lesson on that moment's input
                brain.lesson(xm, labels, weights=w8)
                taken += int((w8 > 0).sum())
                continue
            graph = snap.learner.brain
            eff, bias = np.array(graph.efficacy, copy=True), np.array(graph.bias, copy=True)
            # the lesson's step is the living brain's: the snapshot starts from no momentum of
            # its own (else each lesson on a copy replays the living brain's velocity as well),
            # and what the lesson adds to the optimizer's moments is added to the living brain's
            # (else the correction's contrast never gathers momentum: a tenth of its step)
            sl, ll = snap.learner, brain.learner
            synced = int(g.get("lesson_moments", 0)) == 2
            if synced:  # the step a lesson on the living brain would take, from its moments now
                for k in MOMENTS:
                    setattr(sl, k, np.array(getattr(ll, k), copy=True))
                sl.contrast_updates = ll.contrast_updates
            elif int(g.get("lesson_moments", 0)):
                sl.velocity = np.zeros_like(sl.velocity)
                sl.velocity_bias = np.zeros_like(sl.velocity_bias)
            moments = {k: np.array(getattr(sl, k), copy=True) for k in MOMENTS}
            rep = snap.lesson(xm, labels, weights=w8)
            p.work["lesson_sweeps"] += float(rep.get("total_row_sweeps", 0.0))
            after = snap.learner.brain
            live_graph = brain.learner.brain
            brain.learner.brain = live_graph.with_parameters(
                efficacy=np.asarray(live_graph.efficacy) + (np.asarray(after.efficacy) - eff),
                bias=np.asarray(live_graph.bias) + (np.asarray(after.bias) - bias))
            if synced:
                for k in MOMENTS:
                    setattr(ll, k, np.array(getattr(sl, k), copy=True))
                ll.contrast_updates = sl.contrast_updates
            elif int(g.get("lesson_moments", 0)):
                for k in MOMENTS:
                    setattr(ll, k, np.asarray(getattr(ll, k)) + (np.asarray(getattr(sl, k))
                                                                 - moments[k]))
                ll.contrast_updates = max(ll.contrast_updates, sl.contrast_updates)
            taken += int((w8 > 0).sum())
        return taken

    assoc_at: dict[int, np.ndarray] = {}  # the association's settled pattern after each moment
    assoc_idx = (np.asarray(list(brain.connectome.populations["association"]))
                 if p.records is not None else None)
    match = float(g.get("records_match", 0.85))

    read_slots: dict[int, list[int]] = {}  # a stream's recalled records, labile until judged

    def recall(cue: np.ndarray, i: int = -1) -> np.ndarray:
        assert p.records is not None
        if not p.records.count or not np.any(cue):
            return np.zeros(p.forms.width)
        # the nearest records, one a head; a word comes back only where they all hold it
        # (gene ``records_agree``): a cue its neighbours disagree about recalls nothing
        heads = min(int(g.get("records_agree", 1)), int(p.records.count))
        got = p.records.read(cue, heads=heads, credit=learn)
        slot = got.first()
        if slot < 0:
            return np.zeros(p.forms.width)
        key = np.asarray(p.records.keys[slot], dtype=np.float64)
        cos = float(key @ cue) / (np.linalg.norm(key) * np.linalg.norm(cue) + 1e-12)
        value = p.records.value(got)
        for h in range(1, heads):
            if got.first(h) < 0 or not np.array_equal(p.records.value(got, h), value):
                return np.zeros(p.forms.width)
        if cos < match:
            return np.zeros(p.forms.width)
        if i >= 0:
            read_slots[i] = [got.first(h) for h in range(heads)]
        return value

    def reconsolidate(i: int, cue: np.ndarray, word: str | None) -> None:
        """A recalled record is labile until the reply it gave is judged: paid, it is kept under
        the present pattern; corrected, it is rewritten with the trainer's word under it."""
        assert p.records is not None
        for slot in read_slots.pop(i, []):
            value = p.forms(word) if word is not None else np.asarray(p.records.values[slot])
            p.records.write(cue, value, recognised=int(slot))
            p.work["reconsolidated"] = p.work.get("reconsolidated", 0) + 1

    def keep(i: int, at: int, word: str) -> None:
        if p.records is not None and learn and at in assoc_at:
            p.records.write(assoc_at[at][i], p.forms(word))
            p.work["records"] = p.work.get("records", 0) + 1
            q = exs[i].question
            if int(g.get("focus_records", 0)) and q is not None and q.form == "name" and \
                    0 in assoc_at and exs[i].heard and exs[i].heard[0] is None:
                # the thing's word kept under the state of looking at it, nothing heard
                p.records.write(assoc_at[0][i], p.forms(word))

    for t in range(moments):
        if p.records is not None and learn:
            p.records.moment()  # a moment of the store: unused, shallow records fade first
        if p.records is not None:
            p.recalled_kept = p.recalled if t > 0 else np.zeros((n, p.forms.width))
            p.recalled = np.zeros((n, p.forms.width))
            for i, ex in enumerate(exs):
                if not ex.bite and ex.turn <= t < ex.turn + ex.turn_length \
                        and ex.turn - 1 in assoc_at:
                    first = t == ex.turn  # the records are read once, as the turn opens
                    p.recalled[i] = recall(assoc_at[ex.turn - 1][i], i) if first or not int(
                        g.get("reconsolidate", 0)) else p.recalled_kept[i]
            for i, ex in enumerate(exs):  # the second trainer's answer, kept as a record
                q = ex.question
                if q is not None and learn and ex.extra.get("model") and \
                        ex.turn - len(q.answer) <= t < ex.turn:
                    keep(i, ex.turn - len(q.answer) - 1, q.answer[t - (ex.turn - len(q.answer))])
        x = p.observe(exs, t, own, praise[t], rng)
        if learn:
            # an exchange's end ends the episode, and the recall memory's slots with it; with
            # ``keep_slots`` the slots and the traces carry across exchanges
            done = np.full(n, t == moments - 1 and not int(g.get("keep_slots", 0)))
            if brain.pending_feedback:
                report = brain.learn(reward[t], done, x)
                p.work["reward_row_sweeps"] += n * (float(report.get("free_steps", 0.0)) + 24)
            heard_self = np.array([w is not None for w in own], float)
            if not int(g.get("babble_in_turn", 1)):
                # the voice's practice lesson in practice alone (babbling, saying back a heard
                # word): a reply to a trainer's question is directed song, and is not relabelled
                # by hearing it (its lessons are the pistachio's and the correction's)
                heard_self *= np.array([ex.question is None or ex.question.form in (
                    "imitate", "echo") for ex in exs], float)
            if int(g["babble_lesson"]) and heard_self.any() and p.vocal_period:
                labels = np.stack([voiced, np.zeros(n, dtype=np.int64)], axis=1)
                heard_x = x
                if not int(g.get("babble_eyes", 1)):  # the voice's lesson from hearing alone
                    heard_x = x.copy()
                    heard_x[:, : p.inputs - p.ear] = 0.0
                report = brain.lesson(heard_x, labels,
                                      weights=heard_self * float(g.get("babble_weight", 1.0)))
                p.work["lesson_sweeps"] += float(report.get("total_row_sweeps", 0.0))
            gated = [i for i in range(n) if praise[t, i] and moment_x[i]]
            if float(g["gated_lesson"]) > 0 and gated:
                # the acts that earned the pistachio, made lasting: each on the moment it was
                # made, the parrot's own act the label
                entries: dict[int, list[tuple[int, int, int, float]]] = {}
                for i in gated:
                    for unit, mt in zip(moment_unit[i], moment_t[i], strict=True):
                        entries.setdefault(mt, []).append(
                            (i, unit, int(exs[i].bite), float(g["gated_lesson"])))
                    moment_x[i], moment_unit[i], moment_t[i] = [], [], []
                p.work["gated"] += teach_at(entries)
            if echoes and not p.correcting:
                echoes = []  # the correction is heard (and kept as a record), not taught
            if echoes and float(g["correction_lesson"]) > 0:
                # the trainer's correction, echoed by the parrot's own imitation, learned on the
                # state the parrot was in when it replied
                entries = {}
                for i, k, unit in echoes:
                    if unit != p.silence and turn_t[i]:
                        place = min(k, len(turn_t[i]) - 1)
                        kinds = [x.kind for x in exs[i].scan]
                        q = exs[i].question
                        if kinds and q is not None and q.answer[k] in kinds:
                            place = min(kinds.index(q.answer[k]), len(turn_t[i]) - 1)
                        mt = turn_t[i][place]
                        entries.setdefault(mt, []).append(
                            (i, unit, 0, float(g["correction_lesson"])))
                        if k == 0 and kinds and q is not None and int(g.get("scan_silence", 0)):
                            # the trainer named the right things alone: at each other thing the
                            # focus visited, the right reply was to say nothing
                            for j, kind in enumerate(kinds[: len(turn_t[i])]):
                                if kind not in q.answer:
                                    entries.setdefault(turn_t[i][j], []).append(
                                        (i, p.silence, 0, float(g["correction_lesson"])))
                p.work["corrected"] = p.work.get("corrected", 0) + teach_at(entries)
                echoes = []
            if any(ex.turn <= t < ex.turn + ex.turn_length for ex in exs):
                snaps[t] = (copy.deepcopy(brain), x.copy())
            # while the voice is held to hear a trainer's correction, the motor plan is the
            # inverse model's best guess, not an exploring sample (gene listen_temperature)
            listen = g.get("listen_temperature")
            hearing_correction = any(t in echo_due[i] for i in range(n))
            action = np.asarray(brain.act(x, temperature=float(listen)) if (
                listen is not None and hearing_correction) else brain.act(x)).reshape(n, -1)
        else:
            action = np.asarray(brain.act(x, greedy=True)).reshape(n, -1)
        if assoc_idx is not None:
            if str(g.get("records_key", "association")) == "input":
                # keys that do not drift: what is seen in focus and the word heard this moment
                assoc_at[t] = np.concatenate([x[:, : p.eyes.slot_width], x[:, -p.ear:]], axis=1)
            else:
                assoc_at[t] = np.asarray(brain.basal_ganglia.state.activation[:, assoc_idx],
                                         dtype=np.float64).copy()
        p.work["moments"] += 1
        p.work["stream_moments"] += n
        p.work["act_sweeps"] += n * float((brain.last_settlement or {}).get("steps", 0))
        voiced, beak = action[:, 0].astype(np.int64), action[:, 1].astype(np.int64)
        own = []
        for i, ex in enumerate(exs):
            in_turn = ex.turn <= t < ex.turn + ex.turn_length
            # a trainer speaks (the question, or the correction after a reply): the voice is held
            hearing = t < len(ex.heard) and ex.heard[t] is not None and not in_turn
            w = p.said(voiced[i])
            if hearing and int(g["reply_suppression"]):
                w = None  # the voice is held while the parrot hears a trainer
            own.append(w)
            if ex.bite and in_turn and not bitten[i] and beak[i] == 1:
                bitten[i] = True
                if learn and ex.question is None and ex.extra["edible"]:
                    reward[t + 1, i] += 1.0  # the taste
                    praise[t + 1, i] = True
                    moment_x[i], moment_unit[i], moment_t[i] = [x[i]], [p.silence], [t]
            if t in echo_due[i]:
                k = echo_due[i][t]
                if ex.question is not None and (ex.question.form not in ("echo", "imitate")
                                                or int(g.get("echo_records", 0))):
                    keep(i, ex.turn - 1, ex.question.answer[k])  # the answer, kept as a record
                heard_unit = int(voiced[i])
                if str(g.get("correction_label", "echo")) == "given" and ex.question is not None:
                    heard_unit = p.voice.index(ex.question.answer[k])  # a diagnostic arm only
                if ex.question is not None:  # how often the echo is the word said
                    p.work["echo_labels"] = p.work.get("echo_labels", 0) + 1
                    p.work["echo_labels_right"] = p.work.get("echo_labels_right", 0) + int(
                        heard_unit == p.voice.index(ex.question.answer[k]))
                echoes.append((i, echo_due[i].pop(t), heard_unit))
            if in_turn and not ex.bite:
                turn_x[i].append(x[i])
                turn_t[i].append(t)
                said[i].append(w)
                if w is not None and learn:
                    moment_x[i].append(x[i])
                    moment_unit[i].append(int(voiced[i]))
                    moment_t[i].append(t)
            if t == ex.turn + ex.turn_length - 1 and not judged[i] and not ex.bite:
                judged[i] = True
                replies[i] = reply_of(said[i], scan=bool(ex.scan))
                assert ex.question is not None
                oks[i] = right(ex.question, replies[i])
                mode_r = int(g.get("reconsolidate", 0))
                if p.records is not None and learn and mode_r and i in read_slots and \
                        ex.turn - 1 in assoc_at:
                    if oks[i] or mode_r == 1:  # 2: a paid record is kept under the present
                        # pattern; a corrected one is left (the correction keeps its own record)
                        reconsolidate(i, assoc_at[ex.turn - 1][i],
                                      None if oks[i] else ex.question.answer[0])
                    else:
                        read_slots.pop(i, None)
                pay = oks[i] if pistachio == "earned" else (
                    False if pistachio == "withheld" else bool(rng.random() < 0.3))
                if learn and pay:
                    praise[t + 1, i] = True
                    reward[t + 1 + CATCH, i] += 1.0
                    p.work["pistachios"] += 1
                elif learn:
                    moment_x[i], moment_unit[i], moment_t[i] = [], [], []
                if learn and not oks[i] and ex.extra.get("dagger"):
                    for k, word in enumerate(ex.question.answer):  # the trainer corrects
                        while len(ex.heard) <= t + 1 + k:
                            ex.heard.append(None)
                        ex.heard[t + 1 + k] = word
                        echo_due[i][t + 1 + k] = k
    return {"replies": replies, "right": oks, "bitten": bitten}


def make_exchange(r: Room, q: Question, mode: str, rng: np.random.Generator) -> Exchange:
    """The exchange of question ``q`` in room ``r``: the trainers' words (and in ``model`` mode
    the second trainer's answer), then the parrot's turn."""
    heard: list[str | None] = list(q.words)
    if mode == "model":
        heard += list(q.answer)
    extra: dict[str, Any] = {"dagger": mode == "dagger", "model": mode == "model"}
    if q.scan:  # born serial search: the focus visits each thing once, in a drawn order
        looked = [t for t in r.things if t is not q.focus]
        extra["scan"] = [looked[int(j)] for j in rng.permutation(len(looked))]
    # the parrot's turn opens in the gap after the trainer stops (a silent moment)
    return Exchange(r, heard + [None], len(heard), q, extra=extra)


def exchange(p: Parrot, rng: np.random.Generator, form: str, mode: str,
             held: set[tuple[str, str]], *, test: bool = False) -> Exchange:
    """One exchange: ``mode`` ``model`` (a second trainer says the answer first) or ``turn``."""
    for _ in range(200):
        r = room(rng, int(rng.integers(3, 6)), held, test=test)
        if form == "bite":
            x = r.things[int(rng.integers(len(r.things)))]
            x.focus = True
            x.place = np.array([0.5 * math.cos(r.heading), 0.5 * math.sin(r.heading)])
            return Exchange(r, [], 0, None, bite=True,
                            extra={"edible": x.material in EDIBLE, "kind": x.kind})
        q = ask(form, r, rng)
        if q is None:
            continue
        return make_exchange(r, q, mode, rng)
    raise RuntimeError(f"no room for {form}")


def block(p: Parrot, rng: np.random.Generator, form: str, mode: str,
          held: set[tuple[str, str]], *, test: bool = False) -> list[Exchange]:
    exs = [exchange(p, rng, form, mode, held, test=test) for _ in range(p.streams)]
    # with focus records, every exchange opens with a moment of looking before a word is heard
    width = max(ex.turn for ex in exs) + int(p.genome.get("focus_records", 0))
    for ex in exs:  # lockstep: the trainer starts later where a question is shorter
        pad = width - ex.turn
        ex.heard = [None] * pad + ex.heard
        ex.turn += pad
    return exs


STEPS: list[tuple[str, dict[str, float], list[str]]] = [
    ("N0", {"imitate": 1.0}, ["imitate"]),
    ("N0b", {"echo:dagger": 1.0}, ["echo"]),
    ("N1", {"name:model": 0.3, "name:dagger": 0.7}, ["name"]),
    ("N2", {"colour_of:model": .15, "colour_of:dagger": .35, "by_colour:model": .15,
            "by_colour:dagger": .35}, ["colour_of", "by_colour"]),
    ("N3", {"where:model": .3, "where:dagger": .7}, ["where"]),
    ("N4", {f"{f}:{m}": w for f in ("left_of", "right_of", "next_to", "opposite")
            for m, w in (("model", .07), ("dagger", .18))}, ["left_of", "next_to", "opposite"]),
    ("N5", {"material:model": .1, "material:dagger": .3, "bite:turn": .2, "can_eat:model": .05,
            "can_eat:dagger": .15, "can_eat_this:model": .05, "can_eat_this:dagger": .15},
     ["material", "can_eat", "can_eat_this"]),
    ("N6", {"round:model": .15, "round:dagger": .35, "eat:model": .15, "eat:dagger": .35},
     ["round", "eat"]),
]


def imitate_block(p: Parrot, rng: np.random.Generator, held: set[tuple[str, str]], *,
                  gap: bool = False, dagger: bool = False) -> list[Exchange]:
    """The trainer says one word and the word said back pays: at once (the turn on the word),
    or with ``gap`` in the parrot's turn after the trainer stops; with ``dagger`` a miss is
    corrected (the trainer says the word again)."""
    exs = []
    for _ in range(p.streams):
        r = room(rng, int(rng.integers(2, 5)), held)
        w = WORDS[int(rng.integers(len(WORDS)))]
        q = Question([w], [w], "echo" if gap else "imitate")
        exs.append(Exchange(r, [w, None], 1, q, extra={"dagger": dagger}) if gap
                   else Exchange(r, [w], 0, q, extra={"dagger": dagger}))
    return exs


def blocks_of(p: Parrot, rng: np.random.Generator, form: str, mode: str,
              held: set[tuple[str, str]], *, test: bool = False) -> list[Exchange]:
    if form == "imitate":
        return imitate_block(p, rng, held, dagger=mode == "dagger")
    if form == "echo":
        return imitate_block(p, rng, held, gap=True, dagger=mode == "dagger")
    return block(p, rng, form, mode, held, test=test)


def read(p: Parrot, rng: np.random.Generator, forms: list[str], held: set[tuple[str, str]],
         n: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for form in forms:
        ok: list[bool] = []
        while len(ok) < n:
            exs = blocks_of(p, rng, form, "turn", held, test=form in ("colour_of", "by_colour"))
            ok += live(p, exs, rng, learn=False)["right"]
        out[form] = float(np.mean(ok[:n]))
    return out


def sleep(p: Parrot, scale: float) -> None:
    """A night's sleep between steps: every synapse's efficacy scaled down by ``scale`` (the
    synaptic homeostasis of sleep). The day's lessons grow the efficacies until the regions
    saturate and a new mapping can no longer be learned; scaled down, the brain keeps what it
    knows (each unit's ranking of its inputs is unchanged) and learns again."""
    graph = p.brain.learner.brain
    p.brain.learner.brain = graph.with_parameters(
        efficacy=np.asarray(graph.efficacy) * float(scale), bias=np.asarray(graph.bias))


def save_records(p: Parrot, path: str | Path) -> None:
    """The word records beside a brain: the store's arrays, as ``load_records`` takes them."""
    assert p.records is not None
    with open(path, "wb") as f:
        np.savez_compressed(f, **p.records.arrays())


def load_records(p: Parrot, path: str | Path) -> bool:
    """Take back the word records saved beside a brain, where the parrot keeps records and the
    file exists."""
    if p.records is None or not Path(path).exists():
        return False
    with np.load(path) as f:
        try:
            p.records.restore({k: f[k] for k in f.files})
        except ValueError:  # records of another key (another width): the store starts empty
            return False
    return True


def held_of(rng: np.random.Generator) -> set[tuple[str, str]]:
    """The founder's kind-colour pairs kept for the test rooms (the first draws of its life)."""
    return {(k, str(rng.choice(KINDS[k][3] or tuple(COLOURS)))) for k in KIND_NAMES}


def founder_held(seed: int) -> set[tuple[str, str]]:
    return held_of(np.random.default_rng(np.random.SeedSequence([int(seed), 334, 31])))


def life(seed: int, genome: dict[str, Any] | None = None, *, blocks: dict[str, int] | None = None,
         streams: int = 16, reads: int = 48, backend: str = "cpu", pistachio: str = "earned",
         steps: list[str] | None = None, log: Any = None, keep: bool = False,
         save: str | None = None, master: bool = True,
         control: str | None = None, ckpt: str | None = None, peek: int = 0) -> dict[str, Any]:
    """One founder's night through the steps; each step read on a frozen copy after it."""
    budget = {"N0": 1500, "N0b": 300, "N1": 1000, "N2": 1200, "N3": 800, "N4": 1500, "N5": 1200,
              "N6": 800, **(blocks or {})}
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 334, 31]))
    held = held_of(rng)
    p = Parrot(seed, genome, streams=streams, backend=backend)
    if control is not None:  # a matched control lives the same life in the brain's place
        import meaning_controls

        meaning_controls.control(control, p, parameters=int(p.brain.parameters()), seed=seed)
        p.records = None  # the records are keyed by the association; a control has its own memory
    out: dict[str, Any] = {"seed": int(seed), "parameters": int(p.brain.parameters()),
                           "synapses": p.synapses, "streams": streams, "steps": {},
                           "pistachio": pistachio, "held": sorted(held)}
    began = time.time()
    seen: list[str] = []
    start = (0, 0)  # the step and block a resumed life continues from
    paid: list[float] = []
    meta_path = Path(ckpt) / f"ckpt_{seed}.json" if ckpt and control is None else None
    if meta_path is not None and meta_path.exists():
        from cadence import Brain

        meta = json.loads(meta_path.read_text())
        p.brain = Brain.load(Path(ckpt) / f"ckpt_{seed}.npz")  # type: ignore[arg-type]
        load_records(p, Path(ckpt) / f"ckpt_{seed}.records.npz")
        p.work.update(meta["work"])
        out, seen, paid = meta["out"], meta["seen"], meta["paid"]
        start = (int(meta["step"]), int(meta["block"]))
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 334, 31, *start]))
        if log is not None:
            log(f"resumed at step {start[0]} block {start[1]}")

    def checkpoint(step: int, block: int) -> None:
        if meta_path is None:
            return
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        p.brain.save(meta_path.with_suffix(".npz"))
        if p.records is not None:
            save_records(p, meta_path.with_suffix(".records.npz"))
        part = meta_path.with_suffix(".part")
        part.write_text(json.dumps({"step": step, "block": block, "out": out, "seen": seen,
                                    "paid": paid[-60:], "work": p.work}, default=float))
        part.replace(meta_path)

    order = [n for n, _, _ in STEPS]
    until = str(p.genome.get("vocal_until", "N6"))
    for index, (name, mix, tests) in enumerate(STEPS):
        # the sensitive period of vocal learning: the self-hearing lesson through ``vocal_until``
        p.vocal_period = order.index(name) <= order.index(until) if until in order else True
        since = str(p.genome.get("corrections_from", "N0"))
        p.correcting = order.index(name) >= order.index(since) if since in order else True
        if steps is not None and name not in steps or index < start[0]:
            continue
        keys, weights = list(mix), np.array(list(mix.values()))
        first = start[1] if index == start[0] else 0
        if first == 0:
            paid = []
        earlier = [f for n_, _, ts in STEPS[:index] for f in ts if steps is None or n_ in steps]
        b = first - 1  # a life resumed past a step's budget goes on to its reading
        for b in range(first, int(budget[name])):
            if b > first and b % 250 == 0:
                checkpoint(index, b)
            share = float(p.genome.get("rehearsal", REHEARSAL))
            if "rehearsal_per_form" in p.genome:  # each earlier form its share, up to rehearsal
                share = min(share, float(p.genome["rehearsal_per_form"]) * len(set(earlier)))
            if earlier and rng.random() < share:
                # the trainer revisits earlier words
                old = str(rng.choice(earlier))
                key = f"{old}:dagger" if old != "imitate" or int(
                    p.genome.get("imitate_corrected", 0)) else old  # imitation with correction
            else:
                key = keys[int(rng.choice(len(keys), p=weights / weights.sum()))]
            form, mode = key.split(":") if ":" in key else (key, "turn")
            exs = blocks_of(p, rng, form, mode, held)
            got = live(p, exs, rng, pistachio=pistachio)
            if mode in ("turn", "dagger") and form not in ("bite",) and form in tests:
                paid.append(float(np.mean(got["right"])))
            if log is not None and b % 50 == 0:
                log(f"{name} {b} {key} right={np.mean(got['right']):.2f} "
                    f"t={time.time() - began:.0f}s")
            if peek and log is not None and b > first and b % peek == 0:  # a development read
                log(f"{name} read at block {b}: {read(p, rng, tests, held, 32)}")
            if master and b >= 100 and b % 100 == 0 and len(paid) >= 40 and float(
                    np.mean(paid[-40:])) >= SAMPLED_GATE:
                check = read(p, rng, tests, held, 32)  # the step's forms, greedy, frozen
                if log is not None:
                    log(f"{name} check at block {b}: {check}")
                if min(check.values()) >= MASTERY.get(name, 0.9):
                    if log is not None:
                        log(f"{name} mastered at block {b}")
                    break
        out.setdefault("blocks_lived", {})[name] = b + 1
        seen += tests
        reading = read(p, rng, sorted(set(seen)), held, reads)
        reading["paid_last"] = float(np.mean(paid[-50:])) if paid else float("nan")
        out["steps"][name] = reading
        if log is not None:
            log(f"READ {name} {reading}")
        if save and control is None:
            Path(save).mkdir(parents=True, exist_ok=True)
            p.brain.save(Path(save) / f"brain_{seed}_{name}.npz")
            if p.records is not None:
                save_records(p, Path(save) / f"brain_{seed}_{name}.records.npz")
        if control is None and float(p.genome.get("sleep_scale", 1.0)) != 1.0:
            sleep(p, float(p.genome["sleep_scale"]))
            if log is not None:
                log(f"slept after {name}: efficacies x {float(p.genome['sleep_scale'])}")
        checkpoint(index + 1, 0)
    out["work"] = dict(p.work)
    out["macs"] = float((p.work["act_sweeps"] + p.work["lesson_sweeps"]
                         + p.work["reward_row_sweeps"]) * p.synapses)
    out["seconds"] = time.time() - began
    if keep:
        out["parrot"] = p
    return out


if __name__ == "__main__":
    seed = int(sys.argv[1])
    given = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    target = sys.argv[3] if len(sys.argv) > 3 else None
    result = life(seed, given.get("genome"), blocks=given.get("blocks"),
                  streams=int(given.get("streams", 16)), reads=int(given.get("reads", 48)),
                  pistachio=str(given.get("pistachio", "earned")), steps=given.get("steps"),
                  save=given.get("save"), control=given.get("control"), ckpt=given.get("ckpt"),
                  master=bool(given.get("master", True)), log=lambda s: print(s, flush=True),
                  peek=int(given.get("peek", 0)))
    result["settings"] = given
    text = json.dumps(result, default=float, indent=1)
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(text)
    print(text[:3000])


def export_spec(p: Parrot, path: str | Path) -> None:
    """The spec a page needs beside the brain file (``cadence.meaning.host.MeaningParrot``):
    the eyes' layout, each word's heard code, the voice's words in unit order, the world's
    feature prototypes."""
    e = p.eyes
    spec = {
        "glimpse": True,
        "eyes": {"slots": e.slots, "hues": e.hues, "lights": e.lights, "shapes": e.shapes,
                 "per_shape": e.per_shape, "sizes": e.sizes, "directions": e.directions,
                 "distances": e.distances, "surfaces": e.surfaces, "per_surface": e.per_surface},
        "ear_width": p.forms.width,
        "codes": {w: p.forms(w).tolist() for w in (*WORDS, PRAISE)},
        "voice": list(p.voice),
        "praise": PRAISE,
        "kinds": {k: {"shape": list(v[0]), "width": v[1], "materials": list(v[2])}
                  for k, v in {**KINDS, **NEW_KINDS}.items()},
        "colours": {c: list(v) for c, v in COLOURS.items()},
        "materials": {m: list(v) for m, v in MATERIALS.items()},
        "moment_ms": 300, "turn_moments": TURN, "catch_moments": CATCH,
    }
    if p.mixed:  # the sparse mixed code's fixed expansion
        spec["mixed"] = {"w": p.mixed_w.tolist(), "on": p.mixed_on,
                         "decay": float(p.genome.get("mixed_decay", 0.7))}
    if p.records is not None:  # the word records' settings; their arrays lie beside the brain
        spec["records"] = {"settings": {k: (v.item() if hasattr(v, "item") else v)
                                        for k, v in p.records.to_dict().items()},
                           "agree": int(p.genome.get("records_agree", 1)),
                           "match": float(p.genome.get("records_match", 0.85))}
    Path(path).write_text(json.dumps(spec))
