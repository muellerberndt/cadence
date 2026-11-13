"""Concepts learned from the senses and from naming: where red ends and what makes a ball a ball.

A brain is born with its senses: a hue, a thing's shape and size as the eyes report them, and
the direction of a place around its body, each spread over tuned inputs (a population code, as
place and head-direction cells tile a space). It is born with no category. ``Concepts`` holds one
**region** of concept neurons a sense, and every concept neuron starts silent:

- ``kind``, over the shape and size features;
- ``colour``, over the hue (a circle);
- ``place``, over the direction of a place from where the body faces (a circle).

A region's concept neurons read its sense neurons through one-way synapses and settle with the
library's graph; the most active concept neuron is the concept the sense names. A concept enters
in one of two ways, and naming moves its boundaries.

- **From the senses** (the kind region): a reading that no concept explains (no concept neuron
  settles at ``vigilance`` of ``level`` or more) takes the next silent neuron, its synapses set
  so that it settles at ``level`` on that reading. A thing of a shape never seen is a new concept
  before any word names it.
- **The word's image of each sense.** Each word neuron has synapses onto each region that hold the
  mean of what that sense gave at its namings (the hue and the direction as the sense codes them,
  the kind as the kind concepts settle on it), each reading at unit length. A word names the
  sense where its image stands apart from the image of all namings (the base): the squared
  distance between the two, less what that distance holds by chance at the word's count of
  namings. Where one sense stands apart by ``margin`` over the next and by ``sure`` at least, the
  naming teaches that sense's region; "red" stands apart in the hue, "ball" in the kinds, "behind"
  in the direction, and a word heard a few times on one thing stands apart in none.
- **From naming.** The concepts the word names in that region (the lexicon's hearing) are its
  candidates; the first that no other word names more (the lexicon's saying) is the word's
  concept. The region takes a lesson of the library's free and nudged phases: the word's concept
  is pulled to ``level`` on the reading and every other concept of the region to rest. A word
  with no concept of its own there takes the next silent neuron at the reading (on the colour
  and place regions the only way a concept enters; elsewhere a concept that two words shared
  splits).

So a word's concept grows over the readings its namings cover and gives up those another word's
namings cover: the boundary between two concepts is where the teacher's names change, for each
teacher wherever that teacher draws it, and a reading between the readings taught settles on the
concept whose readings lie nearest.

``LearnedWords`` is ``cadence.words.Words`` on learned concepts: ``see`` turns the eyes' raw
reports into the map's reports (each thing's kind and colour concepts), ``name`` takes the
naming's lesson of the concepts and then the lexicon's, and the lexicon's lesson leaves out a
concept another word names more (mutual exclusivity: a new word goes to what has no name).
``Around`` is its body: ``place(point)`` the concept of a point's direction, ``cells(place,
centers)`` the cells whose directions settle on it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from .lexicon import Lexicon
from .room import Body, Words

GROUPS = ("kind", "colour", "place")  # the regions, in the lexicon's order of its groups


class Circle:
    """A born sense of an angle: ``per`` inputs whose preferred angles tile the circle, each
    reading ``exp(-0.5 * (d / spacing) ** 2)`` with ``d`` the wrapped distance from its preferred
    angle, under ``floor`` zero. An angle lights the inputs within two spacings of it."""

    def __init__(self, per: int = 24, floor: float = 0.05):
        if isinstance(per, bool) or int(per) < 3:
            raise ValueError("a circle takes 3 tuned inputs or more")
        self.per, self.floor = int(per), float(floor)
        self.spacing = 2.0 * math.pi / self.per
        self.preferred = np.arange(self.per) * self.spacing
        self.inputs = self.per

    def __call__(self, readings: Any) -> np.ndarray:
        """``(rows,)`` or ``(rows, 1)`` angles in radians as ``(rows, per)`` rows."""
        a = np.asarray(readings, dtype=np.float64).reshape(-1, 1)
        d = np.angle(np.exp(1j * (a - self.preferred[None, :]))) / self.spacing
        out = np.exp(-0.5 * d**2)
        out[out < self.floor] = 0.0
        return np.asarray(out)


class Features:
    """A born sense of several readings together: each reading spread over ``per`` tuned inputs
    over its range (``cadence.PopulationCode``), and an input for every combination of one
    tuned input a reading, the product of their responses (``per`` to the power of the
    readings), so that two things that differ in one reading alone light different inputs."""

    def __init__(self, ranges: Any, per: int = 8, floor: float = 0.05):
        from ..tuning import PopulationCode

        self.codes = [PopulationCode([tuple(r)], per=int(per), floor=float(floor))
                      for r in ranges]
        self.readings = len(self.codes)
        self.inputs = int(per) ** self.readings

    def __call__(self, readings: Any) -> np.ndarray:
        given = np.asarray(readings, dtype=np.float64).reshape(-1, self.readings)
        out = np.ones((len(given), 1))
        for k, tuned in enumerate(self.codes):
            out = (out[:, :, None] * tuned(given[:, k : k + 1])[:, None, :]).reshape(
                len(given), -1)
        return out


class Region:
    """Concept neurons over one sense: ``code`` (a ``Circle`` or ``Features``), ``capacity``
    concept neurons. ``level`` is a concept's activity on the readings it holds, ``vigilance``
    the share of ``level`` under which a reading is unexplained (``senses`` True: such a reading
    takes a new concept), ``rate`` the share of a lesson's error the lesson closes, ``floor`` the
    activity a concept needs to count."""

    def __init__(self, code: Any, capacity: int, *, senses: bool = False, level: float = 0.5,
                 vigilance: float = 0.5, rate: float = 0.3, floor: float = 0.05,
                 backend: Any = "cpu"):
        from ..brain import Brain as Graph
        from ..connectome import Connectome
        from ..learning import learning_neuron_model

        if int(capacity) < 1 or not 0.0 < float(level) < 1.0 or not 0.0 < float(rate) <= 1.0:
            raise ValueError("a region takes a capacity of 1 or more, a level in (0, 1) and a "
                             "rate in (0, 1]")
        self.code, self.capacity, self.senses = code, int(capacity), bool(senses)
        self.level, self.vigilance = float(level), float(vigilance)
        self.rate, self.floor = float(rate), float(floor)
        self.inputs = int(code.inputs)
        self.neurons = self.inputs + self.capacity
        self.connectome = Connectome.from_synapses(
            self.neurons, label="concepts", pre=np.repeat(np.arange(self.inputs), self.capacity),
            post=np.tile(self.inputs + np.arange(self.capacity), self.inputs),
            populations={"sense": list(range(self.inputs)),
                         "concept": list(range(self.inputs, self.neurons))})
        self._model = learning_neuron_model(dt=1.0)
        self.graph = Graph(self.connectome, self._model,
                           efficacy=np.zeros(self.connectome.synapses), backend=backend)
        self._learner: Any = None
        self.used = 0  # concept neurons that have entered, in order
        self.lessons = 0
        self.recruited = {"senses": 0, "naming": 0}
        self.work = {"settles": 0, "macs": 0}

    # -- settling

    def _drive(self, readings: Any) -> np.ndarray:
        code = self.code(readings)
        drive = np.zeros((len(code), self.neurons))
        drive[:, : self.inputs] = code
        return drive

    def settle(self, readings: Any) -> np.ndarray:
        """The concept neurons settled on readings, ``(rows, capacity)``."""
        drive = self._drive(readings)
        state = self.graph.settle_batch(drive, steps=2, tolerance=None)
        self.work["settles"] += len(drive)
        self.work["macs"] += len(drive) * 2 * int(self.connectome.synapses)
        return np.asarray(np.asarray(state.activation)[:, self.inputs :], dtype=np.float64)

    def winners(self, readings: Any) -> np.ndarray:
        """The concept each reading names, ``(rows,)``: the most active concept neuron, -1
        where none is active."""
        active = self.settle(readings)
        out = np.argmax(active, axis=1)
        out[active.max(axis=1) <= 0.0] = -1
        return np.asarray(out)

    def concept(self, reading: Any) -> int | None:
        """The concept a reading names (the most active concept neuron), None where none is
        active."""
        found = int(self.winners(reading)[0])
        return None if found < 0 else found

    def explained(self, reading: Any) -> tuple[int | None, bool]:
        """The concept a reading names, and whether a concept settles at ``vigilance`` of
        ``level`` or more on it."""
        active = self.settle(reading)[0]
        top = float(active.max())
        return (int(np.argmax(active)) if top > 0.0 else None), top >= (
            self.vigilance * self.level)

    # -- learning

    def recruit(self, reading: Any, *, by: str = "senses") -> int | None:
        """The next silent concept neuron enters on ``reading``: its synapses are set so that it
        settles at ``level`` there. None where every neuron has entered."""
        if self.used >= self.capacity:
            return None
        unit = self.used
        lit = np.tanh(self.code(reading)[0] / 2.0)  # the sense neurons' activity there
        held = 2.0 * math.atanh(self.level)  # the potential that settles at level
        efficacy = np.asarray(self.graph.efficacy, dtype=np.float64).copy()
        into = np.asarray(self.connectome.post) == self.inputs + unit
        efficacy[into] = held * lit[np.asarray(self.connectome.pre)[into]] / max(
            float(lit @ lit), 1e-12)
        self.graph = self.graph.with_parameters(efficacy=efficacy)
        if self._learner is not None:
            self._learner.brain = self.graph
        self.used += 1
        self.recruited[by] += 1
        return unit

    def lesson(self, reading: Any, concept: int) -> float:
        """The naming's lesson: the library's free and nudged phases pull ``concept`` toward
        ``level`` on the reading and every other concept neuron toward rest. Gives the error
        before the lesson, as a share of the target."""
        from ..learning import Learner, LearnerConfig

        drive = self._drive(reading)
        target = np.zeros_like(drive)
        target[0, self.inputs + int(concept)] = self.level
        if self._learner is None:
            self._learner = Learner(
                self.graph, list(range(self.inputs, self.neurons)),
                LearnerConfig(nudge="quadratic", beta=0.1, eta=1.0, eta_bias=0.0, free_steps=2,
                              nudged_steps=4, tolerance=None),
                reciprocal=False, plastic_neurons=np.zeros(self.neurons, dtype=bool))
        learner = self._learner
        lit = np.tanh(drive[:, : self.inputs] / 2.0)
        shared = float((lit.sum(axis=0) ** 2).sum())
        learner.config = dataclasses.replace(
            learner.config, eta=self.rate / (0.25 * max(shared, 1e-12)))
        free = learner.free(drive)
        plus = learner.nudged(drive, free, target)
        minus = learner.nudged(drive, free, target, sign=-1.0)
        learner.update(free, plus, minus)
        self.graph = learner.brain
        self.lessons += 1
        self.work["settles"] += 3
        self.work["macs"] += 10 * int(self.connectome.synapses)  # 2 free and 4 + 4 nudged sweeps
        answer = np.asarray(free.activation)[0, self.inputs :]
        wanted = target[0, self.inputs :]
        return float(np.abs(answer - wanted).sum() / max(float(np.abs(wanted).sum()), 1e-12))


class Around(Body):
    """A body whose places around it are the concepts of a place region: ``place(point)`` the
    concept a point's direction settles on (None while the region holds no concept),
    ``cells(place, centers)`` the cells whose directions settle on that concept."""

    def __init__(self, region: Region, at: Any = (0.0, 0.0), heading: float = 0.0):
        super().__init__(at, heading)
        self.region = region

    def place(self, point: Any) -> int | None:  # type: ignore[override]
        return self.region.concept(self.angle(np.asarray(point)))

    def cells(self, place: int, centers: np.ndarray) -> np.ndarray:
        return np.asarray(self.region.winners(self.angle(centers)) == int(place))


class Concepts:
    """The regions of concepts a brain learns: ``kind`` over the ``shape`` features (their
    ranges; ``per`` tuned inputs a feature), ``colour`` over the hue (``hues`` tuned inputs)
    and ``place`` over the direction around the body (``directions``), with ``kinds``,
    ``colours`` and ``places`` concept neurons. ``words`` is the most words the images hold (the
    lexicon's capacity). ``vigilance`` is every region's, or ``{group: share}``. ``margin``,
    ``sure`` and ``least`` decide the sense a word names; ``naming`` False
    keeps every naming from the regions (the senses alone teach them: a control), and then every
    region takes new concepts from its senses."""

    def __init__(self, shape: Any, *, kinds: int = 24, colours: int = 12, places: int = 10,
                 per: int = 8, hues: int = 48, directions: int = 48, words: int = 256,
                 level: float = 0.5, vigilance: Any = 0.5, rate: float = 0.3,
                 margin: float = 1.5, sure: float = 0.1, least: int = 5, naming: bool = True,
                 backend: Any = "cpu"):
        watch = dict(vigilance) if isinstance(vigilance, dict) else dict.fromkeys(
            GROUPS, float(vigilance))
        options = {"level": level, "rate": rate, "backend": backend}
        self.regions = {
            "kind": Region(Features(shape, per=per), kinds, senses=True,
                           vigilance=watch["kind"], **options),
            "colour": Region(Circle(hues), colours, senses=not naming,
                             vigilance=watch["colour"], **options),
            "place": Region(Circle(directions), places, senses=not naming,
                            vigilance=watch["place"], **options)}
        self.margin, self.sure, self.naming = float(margin), float(sure), bool(naming)
        self.least = int(least)
        self.learning = True  # False: the regions and the images hold still (a test)
        self.settings = {"shape": [list(map(float, r)) for r in shape], "kinds": int(kinds),
                         "colours": int(colours), "places": int(places), "per": int(per),
                         "hues": int(hues), "directions": int(directions), "words": int(words),
                         "level": level, "vigilance": vigilance, "rate": rate,
                         "margin": margin, "sure": sure, "least": int(least),
                         "naming": bool(naming)}
        self.body = Around(self.regions["place"])
        # each word's image of each sense and how many namings it holds, and the base image
        width = {"kind": kinds, "colour": hues, "place": directions}
        self.images = {g: np.zeros((int(words), width[g])) for g in GROUPS}
        self.counts = {g: np.zeros(int(words), dtype=np.int64) for g in GROUPS}
        self.base = {g: np.zeros(width[g]) for g in GROUPS}
        self.based = dict.fromkeys(GROUPS, 0)
        self.sight: list[tuple[np.ndarray, np.ndarray, float]] = []  # this moment's reports
        self.lit: Any = None
        self.log = {"lessons": 0, "splits": 0, "unsure": 0, "objects": 0}

    def sizes(self) -> dict[str, int]:
        """Each group's concept neurons, as the lexicon's groups take them."""
        return {g: self.regions[g].capacity for g in GROUPS}

    # -- the senses into the map

    def percept(self, shape: Any, hue: float) -> tuple[int | None, int | None]:
        """The kind and colour concepts of a thing's raw readings; while learning, a reading no
        concept explains takes a new concept in a region that learns from its senses."""
        out: list[int | None] = []
        for group, reading in (("kind", shape), ("colour", hue)):
            region = self.regions[group]
            found, held = region.explained(reading)
            if self.learning and region.senses and not held:
                new = region.recruit(reading)
                found = new if new is not None else found
            out.append(found)
        return out[0], out[1]

    def see(self, map: Any, reports: Any, lit: Any = None) -> None:
        """One moment of sight: ``reports`` are the eyes' ``(shape, place, hue)`` of each thing
        in view; each becomes the map's report of its kind and colour concepts."""
        self.sight = [(np.asarray(s, dtype=np.float64).copy(),
                       np.asarray(p, dtype=np.float64).reshape(2).copy(), float(h))
                      for s, p, h in reports]
        self.lit = lit
        self._write(map)

    def _write(self, map: Any) -> None:
        seen = []
        for shape, place, hue in self.sight:
            kind, colour = self.percept(shape, hue)
            if kind is None:
                continue
            seen.append((kind, place.copy(), {} if colour is None else {"colour": colour}))
        map.moment(seen=seen, lit=self.lit)

    # -- a naming's lesson

    def _image(self, group: str, reading: Any) -> np.ndarray:
        """A reading as an image takes it, at unit length: the hue and the direction as their
        senses code them, a kind as its concept (``reading`` the concept's number)."""
        if group == "kind":
            x = np.zeros(self.regions["kind"].capacity)
            x[int(reading)] = 1.0
            return x
        x = np.maximum(self.regions[group].code(reading)[0], 0.0)
        return np.asarray(x / max(float(np.linalg.norm(x)), 1e-12))

    def _context(self, brain: Any) -> dict[str, np.ndarray]:
        """What each sense gives at this moment over every thing the brain holds: the kinds and
        the directions of the things on the map, the hues of the things in sight."""
        out: dict[str, list[np.ndarray]] = {g: [] for g in GROUPS}
        for entry in brain.map.things():
            if entry["place"] is None:
                continue
            out["kind"].append(self._image("kind", entry["kind"]))
            out["place"].append(self._image("place", float(self.body.angle(entry["place"])[0])))
        for _, _, hue in self.sight:
            out["colour"].append(self._image("colour", hue))
        return {g: np.mean(v, axis=0) for g, v in out.items() if v}

    def named(self, brain: Any, word: str, at: Any) -> dict[str, Any]:
        """The lesson of a naming, before the lexicon's: ``word`` heard while ``brain`` (a
        ``LearnedWords``) attends to the thing at ``at``. The readings of this moment are a kind
        and a colour where the eyes see the thing (the kind the map holds there where they do
        not), the direction of its place always. Gives what the naming did in each region."""
        if not self.learning:
            return {}
        lexicon = brain.lexicon
        unit = lexicon._enter(word)
        place = np.asarray(at, dtype=np.float64).reshape(2)
        readings: dict[str, Any] = {"place": float(self.body.angle(place)[0])}
        sighted = [r for r in self.sight if np.linalg.norm(r[1] - place) <= 1e-6]
        changed = False
        if sighted:
            readings["kind"], readings["colour"] = sighted[0][0], sighted[0][2]
            region = self.regions["kind"]
            _, held = region.explained(readings["kind"])
            if not held:
                changed = region.recruit(readings["kind"]) is not None
        for g in ("colour", "place"):  # a region that learns from its senses (a control)
            region = self.regions[g]
            if region.senses and g in readings and not region.explained(readings[g])[1]:
                changed = region.recruit(readings[g]) is not None or changed
        if self.naming:  # the word's image and the base take this moment's readings
            for g, x in self._context(brain).items():
                self.based[g] += 1
                self.base[g] += (x - self.base[g]) / self.based[g]
            kind = self.regions["kind"].concept(readings["kind"]) if sighted else (
                brain.map.what_is_at(place))
            for g, reading in (("kind", kind), ("colour", readings.get("colour")),
                               ("place", readings["place"])):
                if reading is None:
                    continue
                x = self._image(g, reading)
                self.counts[g][unit] += 1
                self.images[g][unit] += (x - self.images[g][unit]) / self.counts[g][unit]
        group = self._group(unit) if self.naming else None
        done: dict[str, Any] = {"group": group}
        if group is None and self.naming and sighted and self._unnamed(brain, unit, place):
            new = self.regions["kind"].recruit(readings["kind"], by="naming")
            done["kind"] = ("object", new)
            self.log["objects"] = self.log.get("objects", 0) + 1
            changed = changed or new is not None
        for g, reading in readings.items():
            if g != group:
                continue
            region = self.regions[g]
            own = self._own(lexicon, word, g)
            if own is None:
                new = region.recruit(reading, by="naming")
                done[g] = ("split", new)
                self.log["splits"] += 1
                changed = changed or new is not None
            else:
                done[g] = ("lesson", own, region.lesson(reading, own))
                self.log["lessons"] += 1
                changed = True
        if changed and sighted:
            self._write(brain.map)  # the thing seen again, through the concepts as they are now
        return done

    def _unnamed(self, brain: Any, unit: int, place: np.ndarray, new: int = 3) -> bool:
        """Whether a word heard ``new`` times or fewer names the thing as a new kind (mutual
        exclusivity and the whole thing): every concept of the thing there, its kind, its colour
        and its place, is named more by another word of that concept's sense."""
        if self.counts["kind"][unit] > new:
            return False
        concepts = brain.attend(place)
        return len(concepts) == len(GROUPS) and all(
            self.claimed(brain.lexicon, c, unit) for c in concepts)

    def standing(self, unit: int) -> dict[str, float]:
        """How far a word's image of each sense stands apart from the base: the squared
        distance less its expected value by chance at the word's count of namings, for each
        sense that gave ``least`` readings or more at the word's namings."""
        out = {}
        for g in GROUPS:
            n = int(self.counts[g][unit])
            if n < self.least:
                continue
            gap = self.images[g][unit] - self.base[g]
            out[g] = float(gap @ gap) - (1.0 - float(self.base[g] @ self.base[g])) / n
        return out

    def group_of(self, unit: int | None) -> str | None:
        """The sense a word names: the one whose image stands apart by ``margin`` over the next
        and by ``sure`` at least; None for a word that names no sense so (or no word)."""
        if unit is None:
            return None
        stand = self.standing(int(unit))
        ranked = sorted(stand, key=lambda g: -stand[g])
        if not ranked or stand[ranked[0]] < self.sure or (
                len(ranked) > 1 and stand[ranked[0]] < self.margin * max(stand[ranked[1]], 0.0)):
            return None
        return ranked[0]

    def _group(self, unit: int) -> str | None:
        found = self.group_of(unit)
        if found is None:
            self.log["unsure"] += 1
        return found

    def claimed(self, lexicon: Lexicon, concept: tuple[str, int], unit: int | None) -> bool:
        """Whether another word of the concept's own sense names the concept more than the word
        neuron ``unit`` does: the concept settles to the words, and the most active word, where
        one is active, is of the concept's sense and not ``unit``."""
        words = lexicon.words_for(concept)
        if not len(words) or words.max() <= 1e-3:
            return False
        top = int(np.argmax(words))
        return top != unit and self.group_of(top) == concept[0]

    def _own(self, lexicon: Lexicon, word: str, group: str, most: int = 3) -> int | None:
        """The word's concept in ``group``: of the concepts the word names most there (by the
        lexicon's hearing), the first that no other word of the group names more. None where
        none is."""
        unit = lexicon.unit(word)
        part = lexicon.hear([word])[lexicon.groups[group]]
        used = self.regions[group].used
        for c in np.argsort(-part[:used], kind="stable")[:most]:
            if part[c] <= 1e-3:
                break
            if not self.claimed(lexicon, (group, int(c)), unit):
                return int(c)
        return None

    # -- the state

    def parameters(self) -> int:
        """Learned parameters: one efficacy a synapse of every region, and the images."""
        return int(sum(r.connectome.synapses for r in self.regions.values())
                   + sum(i.size for i in self.images.values()))

    def work(self) -> dict[str, int]:
        return {"settles": sum(r.work["settles"] for r in self.regions.values()),
                "macs": sum(r.work["macs"] for r in self.regions.values())}


class LearnedWords(Words):
    """``cadence.words.Words`` on concepts it learns (``Concepts``): the map's kinds and colours
    are the concepts' neurons, the body is the concepts' ``Around``, and the lexicon's groups are
    the concepts' regions. ``exclusive`` keeps a naming from teaching the word a concept another
    word names more (mutual exclusivity). ``rate`` is the lexicon's."""

    def __init__(self, map: Any, concepts: Concepts, *, exclusive: bool = True,
                 rate: float = 0.2, capacity: int = 256, **settings: Any):
        sizes = concepts.sizes()
        held = map.values.get(settings.get("group", "colour"))
        if len(map.codes) != sizes["kind"] or held is None or (
                held.stop - held.start != sizes["colour"]):
            raise ValueError("the map holds one kind a kind concept and one value a colour "
                             "concept: Map(units=kinds, values={'colour': colours})")
        lexicon = Lexicon({"kind": sizes["kind"], "colour": sizes["colour"],
                           "place": sizes["place"], "read": 3, "none": 1}, rate=rate,
                          capacity=capacity)
        super().__init__(map, lexicon=lexicon, body=concepts.body, **settings)
        self.concepts, self.exclusive = concepts, bool(exclusive)

    def see(self, reports: Any, lit: Any = None) -> None:
        """One moment of sight from the eyes' raw reports ``(shape, place, hue)``."""
        self.concepts.see(self.map, reports, lit)

    def attend(self, place: Any) -> dict[tuple[str, int], float]:
        kind = self.map.what_is_at(place)
        if kind is None:
            return {}
        out = {("kind", int(kind)): self.shares[0]}
        value = self.map.values_at(place)[self.group]
        if value is not None:
            out[("colour", int(value))] = self.shares[1]
        around = self.body.place(place)
        if around is not None:
            out[("place", int(around))] = self.shares[2]
        return out

    def name(self, word: str, at: Any) -> float:
        """A naming: the concepts' lesson, then the lexicon's. A word whose image names a sense
        is taught that sense's concept alone (its other concepts fade); with ``exclusive``, a
        concept another word of its sense names more is left at rest."""
        self.concepts.named(self, word, at)
        concepts = self.attend(at)
        if not concepts:
            return float("nan")
        unit = self.lexicon.unit(word)
        group = self.concepts.group_of(unit)
        out = {}
        for concept, share in concepts.items():
            keep = group is None or concept[0] == group
            if keep and self.exclusive and self.concepts.claimed(self.lexicon, concept, unit):
                keep = False
            out[concept] = share if keep else 0.0
        return self.lexicon.name(word, out, groups=("kind", "colour", "place"))

    def find(self, about: tuple[str, int] | None, read: int | None) -> list[tuple[str, int]]:
        found = [c for c in super().find(about, read) if c[1] is not None]
        return found or [("none", 0)]


__all__ = ["Around", "Circle", "Concepts", "Features", "LearnedWords", "Region"]
