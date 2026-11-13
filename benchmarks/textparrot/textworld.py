"""The text parrot's world: scenes of things around the parrot, its eyes, and the grammar.

**Things.** Twelve kinds; ten are shown in pretraining and two (a lemon and a pen) never are.
Each kind has a shape (round, square, long), a material (fruit, wood, metal, glass, plastic,
paper), a size (small, big) and a set of colours it comes in; a thing is food where its material
is fruit. A thing stands at one of eight places around the parrot: four directions (front, right,
behind, left) at two distances (near, far).

**The eyes.** Nothing reports a kind, a colour, a material or a place as such. A glimpse of a
thing reports its hue (six units tuned around the hue circle), saturation and lightness, its
outline (roundness, elongation, corners), its width, its surface (gloss, roughness,
translucency, softness), four details (a stem, a hollow, a handle, flatness), and where it is:
its direction from the parrot's heading (eight units tuned around the circle) and its distance
(three units), each with noise.

**Relations** between two things read their places on the floor: a thing is left of another
where it lies further to the left, in front of it where it lies further forward, next to it where
their places are neighbours, and opposite it where it lies in the opposite direction from the
parrot.

**The grammar** (``question``, ``answer``, ``parse``): the teacher's question forms, each with a
depth knob (``the apple``, ``the red apple``), and the replies: bare words ("red", "left far",
"apple cup"), yes or no, and sentences, one clause a thing ("the red apple is left far", "the
blue block is left of the red apple"). ``parse`` reads a reply back into the things and places
it says.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# kind: shape, material, size, colours (the first the most common), details (stem, hollow,
# handle, flat), outline offsets (roundness, elongation, corners)
KINDS: dict[str, tuple[str, str, str, tuple[str, ...], tuple[float, ...], tuple[float, ...]]] = {
    "apple": ("round", "fruit", "small", ("red", "green", "yellow"), (1, 0, 0, 0), (.0, .0, .0)),
    "banana": ("long", "fruit", "small", ("yellow", "green"), (1, 0, 0, 0), (.0, .1, .0)),
    "ball": ("round", "plastic", "small", ("blue", "red", "green", "yellow"), (0, 0, 0, 0),
             (.1, .0, .0)),
    "cup": ("round", "glass", "small", ("white", "blue", "red"), (0, 1, 1, 0), (-.1, .0, .0)),
    "block": ("square", "wood", "small", ("red", "blue", "yellow", "green"), (0, 0, 0, 0),
              (.0, .0, .1)),
    "box": ("square", "paper", "big", ("white", "black", "red"), (0, 1, 0, 0), (.0, .0, .0)),
    "key": ("long", "metal", "small", ("yellow", "black"), (0, 0, 1, 1), (.0, .0, .0)),
    "book": ("square", "paper", "big", ("blue", "red", "black", "green"), (0, 0, 0, 1),
             (.0, .1, .0)),
    "bottle": ("long", "glass", "big", ("green", "white", "blue"), (0, 1, 0, 0), (.0, .0, .0)),
    "plate": ("round", "metal", "big", ("white", "black"), (0, 0, 0, 1), (.0, -.1, .0)),
    # never shown in pretraining: the new objects
    "lemon": ("round", "fruit", "small", ("yellow",), (1, 0, 0, 0), (-.1, .15, .0)),
    "pen": ("long", "plastic", "small", ("black", "blue"), (0, 0, 0, 0), (.0, .15, .0)),
}
NEW_KINDS = ("lemon", "pen")
TRAIN_KINDS = tuple(k for k in KINDS if k not in NEW_KINDS)
SHAPES = {"round": (1.0, 0.0, 0.0), "square": (0.2, 0.2, 1.0), "long": (0.1, 1.0, 0.3)}
SIZES = {"small": 0.3, "big": 0.8}
# hue (degrees, None for no hue), saturation, lightness
COLOURS: dict[str, tuple[float | None, float, float]] = {
    "red": (0.0, .9, .5), "yellow": (60.0, .9, .6), "green": (120.0, .8, .45),
    "blue": (230.0, .85, .45), "white": (None, .05, .95), "black": (None, .05, .08)}
# gloss, roughness, translucency, softness
MATERIALS: dict[str, tuple[float, float, float, float]] = {
    "fruit": (.6, .2, .1, .75), "wood": (.2, .7, .0, .2), "metal": (.9, .1, .0, .0),
    "glass": (.9, .05, .8, .0), "plastic": (.55, .2, .25, .25), "paper": (.1, .6, .05, .5)}
FOOD = ("fruit",)
DIRECTIONS = ("front", "right", "behind", "left")
ANGLE = {"front": 0.0, "left": 90.0, "behind": 180.0, "right": -90.0}  # left positive
DISTANCE = {"near": 0.7, "far": 1.6}
PLACES = tuple((d, r) for d in DIRECTIONS for r in ("near", "far"))
# the floor coordinates of a place (x to the right, y forward), in steps
FLOOR = {("front", "near"): (0, 1), ("front", "far"): (0, 2), ("right", "near"): (1, 0),
         ("right", "far"): (2, 0), ("behind", "near"): (0, -1), ("behind", "far"): (0, -2),
         ("left", "near"): (-1, 0), ("left", "far"): (-2, 0)}
# the eyes' sweep over the scene (born serial scan): from the left, round the front, to the
# right, then behind
SCAN = (("left", "far"), ("left", "near"), ("front", "far"), ("front", "near"),
        ("right", "near"), ("right", "far"), ("behind", "near"), ("behind", "far"))
RELATIONS = ("left of", "right of", "in front of", "behind", "next to", "opposite")
OPPOSITE = {"front": "behind", "behind": "front", "left": "right", "right": "left"}

# colour x kind pairs never shown in pretraining scenes (the held-out combinations)
HELD_PAIRS = frozenset({("apple", "yellow"), ("ball", "green"), ("cup", "red"),
                        ("block", "yellow"), ("book", "green"), ("bottle", "white")})
# relation x referent kind pairs never asked in pretraining
HELD_RELATIONS = frozenset({("left of", "cup"), ("behind", "apple"), ("next to", "ball"),
                            ("opposite", "key"), ("right of", "book"), ("in front of", "box")})

GLIMPSE = 6 + 2 + 3 + 1 + 4 + 4 + 8 + 3  # what the eyes report of one thing: 31 values
ASKS_PRESENT = 0.8  # the share of questions about a value or relation the scene holds


@dataclass
class Thing:
    kind: str
    colour: str
    place: tuple[str, str]
    look: np.ndarray = field(repr=False)  # the glimpse's values (without the "seen" flag)

    @property
    def shape(self) -> str:
        return KINDS[self.kind][0]

    @property
    def material(self) -> str:
        return KINDS[self.kind][1]

    @property
    def size(self) -> str:
        return KINDS[self.kind][2]

    @property
    def food(self) -> bool:
        return self.material in FOOD

    @property
    def direction(self) -> str:
        return self.place[0]

    @property
    def far(self) -> bool:
        return self.place[1] == "far"


@dataclass
class Scene:
    things: list[Thing]

    def scanned(self) -> list[Thing]:
        """The things in the order the eyes sweep over them."""
        order = {p: i for i, p in enumerate(SCAN)}
        return sorted(self.things, key=lambda t: order[t.place])

    def find(self, kind: str, colour: str | None = None) -> list[Thing]:
        return [t for t in self.things if t.kind == kind and (colour is None or t.colour == colour)]


def look(kind: str, colour: str, place: tuple[str, str], rng: np.random.Generator,
         noise: float = 0.05) -> np.ndarray:
    """What the eyes report of a thing: hue (6), saturation and lightness (2), outline (3),
    width (1), surface (4), details (4), direction (8) and distance (3)."""
    shape, material, size, _, details, offsets = KINDS[kind]
    hue, sat, light = COLOURS[colour]
    hues = np.zeros(6)
    if hue is not None:
        h = hue + rng.normal(0, 8)
        centres = np.arange(6) * 60.0
        d = np.minimum(np.abs(h - centres) % 360, 360 - np.abs(h - centres) % 360)
        hues = np.exp(-(d / 40.0) ** 2)
    sl = np.array([sat, light]) + rng.normal(0, noise, 2)
    outline = np.array(SHAPES[shape]) + np.array(offsets) + rng.normal(0, noise, 3)
    width = np.array([SIZES[size]]) + rng.normal(0, noise, 1)
    surface = np.array(MATERIALS[material]) + rng.normal(0, noise, 4)
    detail = np.array(details, float) + rng.normal(0, noise, 4)
    angle = ANGLE[place[0]] + rng.normal(0, 10)
    centres = np.arange(8) * 45.0
    d = np.minimum(np.abs(angle - centres) % 360, 360 - np.abs(angle - centres) % 360)
    direction = np.exp(-(d / 35.0) ** 2)
    r = DISTANCE[place[1]] + rng.normal(0, 0.1)
    distance = np.exp(-((r - np.array([0.5, 1.1, 1.7])) / 0.35) ** 2)
    out = np.concatenate([hues, sl, outline, width, surface, detail, direction, distance])
    assert out.shape == (GLIMPSE,)
    return out


def scene(rng: np.random.Generator, count: int, *, kinds: tuple[str, ...] = TRAIN_KINDS,
          held: bool = False, include: tuple[tuple[str, str | None], ...] = ()) -> Scene:
    """``count`` things at distinct places. ``held=False`` draws no held-out colour pair;
    ``held=True`` puts at least one held-out pair in (where a held-out kind is drawn). Each pair
    of ``include`` (kind, colour or None) is placed first."""
    places = [PLACES[i] for i in rng.permutation(len(PLACES))[:count]]
    things: list[Thing] = []
    for i, place in enumerate(places):
        if i < len(include):
            kind, colour = include[i]
        else:
            kind, colour = str(rng.choice(kinds)), None
        allowed = [c for c in KINDS[kind][3]
                   if held or (kind, c) not in HELD_PAIRS or kind in NEW_KINDS]
        if colour is None:
            if held and i == 0 and any((kind, c) in HELD_PAIRS for c in KINDS[kind][3]):
                colour = next(c for c in KINDS[kind][3] if (kind, c) in HELD_PAIRS)
            else:
                # the first colour is the kind's usual one: half of its things
                weights = np.array([2.0 if j == 0 else 1.0 for j in range(len(allowed))])
                colour = str(rng.choice(allowed, p=weights / weights.sum()))
        things.append(Thing(kind, colour, place, look(kind, colour, place, rng)))
    return Scene(things)


# --- relations -------------------------------------------------------------------------

def related(a: Thing, b: Thing, relation: str) -> bool:
    """Whether ``a`` stands in ``relation`` to ``b`` (``a`` is left of ``b``, ...)."""
    if a is b:
        return False
    (xa, ya), (xb, yb) = FLOOR[a.place], FLOOR[b.place]
    if relation == "left of":
        return xa < xb
    if relation == "right of":
        return xa > xb
    if relation == "in front of":
        return ya > yb
    if relation == "behind":
        return ya < yb
    if relation == "next to":
        return math.hypot(xa - xb, ya - yb) < 1.5
    if relation == "opposite":
        return a.direction == OPPOSITE[b.direction]
    raise ValueError(relation)


# --- the grammar -----------------------------------------------------------------------

FUNCTION = ("what", "colour", "has", "the", "is", "where", "objects", "are", "made", "of",
            "can", "you", "eat", "do", "see", "a", "it", "in", "next", "to", "shape", "?", ".")
CONTENT = (tuple(KINDS) + tuple(COLOURS) + tuple(SHAPES) + tuple(SIZES) + tuple(MATERIALS)
           + ("front", "right", "behind", "left", "opposite", "far", "yes", "no"))
NONCE = ("zabu", "dax", "wug", "toma")  # heard and echoed from the start, never a name
VOCABULARY = tuple(dict.fromkeys(FUNCTION + CONTENT + NONCE))
SILENCE = "_"


@dataclass
class Question:
    """A question with what it is about: ``form`` names it, ``words`` are what the teacher says
    (the last is "?"), ``referent`` the thing the question names (None for a question over the
    whole scene), ``clauses`` the reply per thing the eyes sweep over (by the thing's index in
    the scene) and ``closing`` the words said after the sweep (before the final ".")."""
    form: str
    words: list[str]
    referent: Thing | None
    clauses: dict[int, list[str]]
    closing: list[str]
    depth: int = 1
    meta: dict = field(default_factory=dict)

    def reply(self, scene: Scene) -> list[str]:
        """The whole reply in order: the clauses in the sweep's order, the closing, and "."."""
        out: list[str] = []
        for t in scene.scanned():
            out += self.clauses.get(scene.things.index(t), [])
        return out + self.closing + ["."]


def noun_phrase(t: Thing, depth: int) -> list[str]:
    """``the K`` (depth 1) or ``the C K`` (depth 2)."""
    return ["the", t.kind] if depth == 1 else ["the", t.colour, t.kind]


def place_words(t: Thing) -> list[str]:
    return [t.direction] + (["far"] if t.far else [])


def clause(t: Thing) -> list[str]:
    """One thing as a sentence: ``the C K is D [far]``."""
    return ["the", t.colour, t.kind, "is"] + place_words(t)


def relation_clause(t: Thing, relation: str, ref: Thing) -> list[str]:
    """``the C K is R the C' K'``."""
    return ["the", t.colour, t.kind, "is"] + relation.split() + ["the", ref.colour, ref.kind]


def unique(scene: Scene, t: Thing, depth: int) -> bool:
    """Whether the noun phrase of ``t`` at ``depth`` names it alone in ``scene``."""
    same = scene.find(t.kind, None if depth == 1 else t.colour)
    return len(same) == 1


FORMS = ("it", "it_colour", "it_shape", "echo", "prompt", "which_colour", "which_shape",
         "which_size", "which_material", "which_food", "know_shape", "know_food", "colour",
         "can_eat", "where", "relation", "describe", "describe_rel")


def question(form: str, scene: Scene, rng: np.random.Generator, *, depth: int = 1,
             held_relations: bool = False, word: str | None = None) -> Question | None:
    """A question of ``form`` about ``scene``, or None where the scene does not allow one.
    ``held_relations`` draws only held-out relation x referent pairs (and none of them
    otherwise)."""
    idx = {id(t): i for i, t in enumerate(scene.things)}
    things = scene.things
    if form == "it":  # one thing in view: what is it?
        if len(things) != 1:
            return None
        return Question(form, ["what", "is", "it", "?"], None, {0: [things[0].kind]}, [])
    if form == "it_colour":
        if len(things) != 1:
            return None
        return Question(form, ["what", "colour", "is", "it", "?"], None,
                        {0: [things[0].colour]}, [])
    if form == "it_shape":
        if len(things) != 1:
            return None
        return Question(form, ["what", "shape", "is", "it", "?"], None,
                        {0: [things[0].shape]}, [])
    if form == "prompt":  # the trainer asks "what is it ?" and says a name; the parrot says it
        if len(things) != 1:
            return None
        # the thing's name two times in three, a nonce word otherwise (a prompt, never a name)
        w = word or (things[0].kind if rng.random() < 2 / 3 else str(rng.choice(NONCE)))
        return Question(form, ["what", "is", "it", "?", w], None, {0: [w]}, [], meta={"word": w})
    if form == "echo":  # the trainer says a word with one thing in view; the parrot says it back
        if len(things) != 1:
            return None
        # a third of the echoes are nonce words (heard and said back, never a name)
        w = word or str(rng.choice(NONCE) if rng.random() < 1 / 3 else rng.choice(CONTENT))
        return Question(form, [w], None, {0: [w]}, [], meta={"word": w})
    if form in ("which_colour", "which_shape", "which_size", "which_material", "which_food"):
        # the teacher asks about what is there four times in five (a value of a thing in view)
        present = things[int(rng.integers(len(things)))] if things and rng.random() < ASKS_PRESENT \
            else None
        if form == "which_colour":
            value = present.colour if present else str(rng.choice(tuple(COLOURS)))
            words, test = ["what", "is", value, "?"], (lambda t: t.colour == value)
        elif form == "which_shape":
            value = present.shape if present else str(rng.choice(tuple(SHAPES)))
            words, test = ["what", "objects", "are", value, "?"], (lambda t: t.shape == value)
        elif form == "which_size":
            value = present.size if present else str(rng.choice(tuple(SIZES)))
            words, test = ["what", "objects", "are", value, "?"], (lambda t: t.size == value)
        elif form == "which_material":
            value = present.material if present else str(rng.choice(tuple(MATERIALS)))
            words, test = ["what", "is", "made", "of", value, "?"], (lambda t: t.material == value)
        else:
            value = "food"
            words, test = ["what", "can", "you", "eat", "?"], (lambda t: t.food)
        clauses = {idx[id(t)]: [t.kind] for t in things if test(t)}
        return Question(form, words, None, clauses, [],
                        meta={"value": value})
    if form in ("know_shape", "know_food"):  # about a kind, nothing in view
        if things:
            return None
        kind = str(rng.choice(TRAIN_KINDS))
        if form == "know_shape":
            value = str(rng.choice(tuple(SHAPES)))
            yes = KINDS[kind][0] == value
            words = ["is", "a", kind, value, "?"]
        else:
            yes = KINDS[kind][1] in FOOD
            words = ["can", "you", "eat", "a", kind, "?"]
        return Question(form, words, None, {}, ["yes" if yes else "no"], meta={"kind": kind})
    # forms about a named thing
    if not things:
        return None
    choices = [t for t in things if unique(scene, t, depth)]
    if form == "relation" or form == "describe_rel":
        relation = str(rng.choice(RELATIONS))
        choices = [t for t in choices
                   if ((relation, t.kind) in HELD_RELATIONS) == held_relations]
        if rng.random() < ASKS_PRESENT:  # four times in five, about a thing with an answer
            choices = [t for t in choices if any(related(o, t, relation) for o in things)]
    if form == "describe":
        clauses = {idx[id(t)]: clause(t) for t in things}
        return Question(form, ["what", "do", "you", "see", "?"], None, clauses, [])
    if not choices:
        return None
    ref = choices[int(rng.integers(len(choices)))]
    np_ = noun_phrase(ref, depth)
    r = idx[id(ref)]
    if form == "colour":
        return Question(form, ["what", "colour", "has"] + np_ + ["?"], ref, {r: [ref.colour]},
                        [], depth)
    if form == "can_eat":
        return Question(form, ["can", "you", "eat"] + np_ + ["?"], ref,
                        {r: ["yes" if ref.food else "no"]}, [], depth)
    if form == "where":
        return Question(form, ["where", "is"] + np_ + ["?"], ref, {r: place_words(ref)}, [],
                        depth)
    if form == "relation":
        words = ["what", "is"] + relation.split() + np_ + ["?"]
        clauses = {idx[id(t)]: [t.kind] for t in things if related(t, ref, relation)}
        return Question(form, words, ref, clauses, [], depth,
                        {"relation": relation})
    if form == "describe_rel":
        words = ["what", "do", "you", "see"] + relation.split() + np_ + ["?"]
        clauses = {idx[id(t)]: relation_clause(t, relation, ref)
                   for t in things if related(t, ref, relation)}
        return Question(form, words, ref, clauses, [], depth,
                        {"relation": relation})
    raise ValueError(form)


# --- reading a reply back ------------------------------------------------------------------

def parse(words: list[str], relational: bool = False) -> dict:
    """Reads a reply of clauses back into what it says: ``things`` as (colour, kind, direction,
    far) for each ``the C K is D [far]`` clause, or with ``relational`` ``relations`` as (colour,
    kind, relation, colour, kind) for each ``the C K is R the C' K'`` clause; ``bad`` counts
    words no clause reads. A reply is its clauses, then "." (no clause: no thing)."""
    things: list[tuple[str, str, str, bool]] = []
    relations: list[tuple[str, str, str, str, str]] = []
    bad = 0
    w = [x for x in words if x != SILENCE]
    if w and w[-1] == ".":
        w = w[:-1]
    else:
        bad += 1
    multi = {("left", "of"): ("left of", 2), ("right", "of"): ("right of", 2),
             ("in", "front"): ("in front of", 3), ("next", "to"): ("next to", 2),
             ("behind",): ("behind", 1), ("opposite",): ("opposite", 1)}
    i = 0
    while i < len(w):
        if (i + 4 <= len(w) and w[i] == "the" and w[i + 1] in COLOURS and w[i + 2] in KINDS
                and w[i + 3] == "is"):
            c, k, j = w[i + 1], w[i + 2], i + 4
            if relational:
                got = multi.get(tuple(w[j:j + 2])) or multi.get(tuple(w[j:j + 1]))
                if got is not None and w[j:j + got[1]] == got[0].split():
                    j += got[1]
                    if (j + 3 <= len(w) and w[j] == "the" and w[j + 1] in COLOURS
                            and w[j + 2] in KINDS):
                        relations.append((c, k, got[0], w[j + 1], w[j + 2]))
                        i = j + 3
                        continue
            elif j < len(w) and w[j] in DIRECTIONS:
                far = j + 1 < len(w) and w[j + 1] == "far"
                things.append((c, k, w[j], far))
                i = j + (2 if far else 1)
                continue
        bad += 1
        i += 1
    return {"things": things, "relations": relations, "bad": bad}


def described(scene: Scene) -> list[tuple[str, str, str, bool]]:
    """What a description of ``scene`` says, sorted: (colour, kind, direction, far) a thing."""
    return sorted((t.colour, t.kind, t.direction, t.far) for t in scene.things)
