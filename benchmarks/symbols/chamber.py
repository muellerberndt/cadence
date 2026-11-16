"""The symbols chamber: the parrot that starts with no words (cadence 0.101, #342).

A brain with eyes, an ear and a voice (``Brain.compose(senses={"eyes": ..., "ear": ...},
code=..., bound=records, voice="ear")``) lives through what a trainer does with a young parrot,
and is read at each step. The world and the trainer are in ``world.py``.

1. **Listen.** The trainer says every word of the world three times, nothing else in view.
2. **Naming.** The trainer holds up a thing (a kind in one of its colours, at a place) and says
   its kind's name, ``ROUNDS`` times a kind.
3. **Y1, a thing named by sight.** A thing at a place no naming used, in either colour of its
   kind, nothing heard: what the parrot says (``Brain.say``) is heard by the trainer as a word.
   Right when it is the kind's name.
4. **Y2, a sound brings the thing back.** A kind's name heard alone, nothing in view:
   - what the records bring back (``Brain.complete``): the kind read from the shape and surface
     that come back (the nearest kind's);
   - in the state: the brain acts on the name heard alone and its association's settled
     activity is read by a linear probe fitted on things seen alone (no records read: probes
     fitted on a copy without them);
   - controls: the same brain with its records barred (``brain.bound = None``), and with the
     records' values dealt to other keys.
5. **Y3, meaning learned by reward** (a fresh brain of the same founder after Listen alone):
   - names: a thing in view, the parrot says something; a reward (1.0) when it says the
     kind's name. Four kinds, one after the other, each until it is said right three times
     running or ``BUDGET`` exchanges;
   - a phrase: "what colour is this" heard with a thing in view, a reward when the parrot says
     its colour. Four things (kind and colour), each until right three times running or
     ``BUDGET`` exchanges;
   - unpaid: the same exchanges with no reward.
6. **Y4, controls on the same exchanges.** A lookup table keyed by (the thing's kind and colour,
   the word heard) that keeps a paid answer and otherwise guesses among the words heard, and a
   delta-rule readout of the brain's code to the words, taught by the reward, choosing by
   softmax at temperature ``TEMPERATURE``.
7. **N, a heard name draws the focus.** A brain with a map of things whose values are the
   eyes' fields (``Map(values=..., surprise=True)`` in ``trained=``) and ``surprise=True``,
   after Listen and Naming. A room of three things appears one by one, so the focus is drawn
   to the last; the trainer says the name of another thing in the room. Right when the focus
   moves to within ``reach`` of that thing. Twice a room: the named thing in view, and behind a
   screen (the map holds it unseen). Control: the same after Listen alone.

    PYTHONPATH=src python benchmarks/symbols/chamber.py --founders 0 1 --out dev.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from world import KINDS, QUESTIONS, WORDS, Thing, Words, a_thing, things_sense  # noqa: E402

import cadence as cd  # noqa: E402
from cadence.senses import Ear  # noqa: E402

CODE = 2048
RECORDS = 4096
ROUNDS = 8
BUDGET = 80
RUNNING = 3
TEMPERATURE = 0.15
TESTS = 5  # views a kind for Y1, takes a name for Y2
PROBE = 16  # views a kind the state's probe is fitted on
PAIRS = (("cup", "white"), ("apple", "red"), ("banana", "yellow"), ("key", "blue"))
NAMED = ("cup", "apple", "banana", "key")
ROOMS = 12
SPOTS = np.array([[-0.6, 0.5], [0.0, 0.6], [0.6, 0.5]])  # where a room's three things stand
MAP_VALUES = {"hue": 8, "shape": 24, "surface": 8}


def setting() -> tuple[cd.senses.Things, Ear, Words]:
    things, ear = things_sense(), Ear()
    return things, ear, Words(ear, seed=0)


def compose(founder: int, things: Any, ear: Ear, *, space: Any = None) -> cd.Brain:
    options: dict[str, Any] = {}
    if space is not None:
        options.update(trained=[space], surprise=True)
    return cd.Brain.compose(senses={"eyes": things.widths, "ear": ear.width}, actions=2,
                            code=CODE, bound=RECORDS, voice="ear", arousal=True,
                            seed=founder, backend="cpu", **options)


def seen(things: Any, thing: Thing) -> dict[str, np.ndarray]:
    return things.rows(thing.readings())


def listen(brain: cd.Brain, words: Words, rng: np.random.Generator, reps: int = 3) -> None:
    for word in WORDS:
        for _ in range(reps):
            brain.act({"ear": words.take(word, rng)}, greedy=True)


def naming(brain: cd.Brain, things: Any, words: Words, rng: np.random.Generator,
           rounds: int = ROUNDS) -> None:
    for _ in range(rounds):
        for kind in KINDS:
            brain.act({"eyes": seen(things, a_thing(rng, kind)), "ear": words.take(kind, rng)},
                      greedy=True)


def kind_of(eyes: dict[str, np.ndarray], things: Any) -> str | None:
    """The kind whose shape and surface lie nearest a recalled thing's; None for nothing."""
    back = np.concatenate([eyes["shape"][0], eyes["surface"][0]])
    if not back.any():
        return None
    protos = {k: np.concatenate([things.rows({"shape": [list(v[0])]})["shape"][0],
                                 things.rows({"surface": [v[1]]})["surface"][0]])
              for k, v in KINDS.items()}
    return max(protos, key=lambda k: float(back @ protos[k]))


def association(brain: cd.Brain) -> np.ndarray:
    return np.asarray(list(brain.connectome.populations["association"]))


def settled(brain: cd.Brain, observation: dict[str, np.ndarray]) -> np.ndarray:
    """The association's settled activity after one greedy act from a cleared state."""
    brain.reset()
    brain.act(observation, greedy=True)
    state = np.asarray(brain.basal_ganglia.state.activation, dtype=np.float64)
    return state[0, association(brain)]


def probe(xs: np.ndarray, labels: list[str]) -> Callable[[np.ndarray], str]:
    """A ridge readout from the association's activity to the kind."""
    names = sorted(set(labels))
    y = np.eye(len(names))[[names.index(label) for label in labels]]
    x = np.hstack([xs, np.ones((len(xs), 1))])
    w = np.linalg.solve(x.T @ x + 1e-2 * np.eye(x.shape[1]), x.T @ y)
    return lambda s: names[int(np.argmax(np.append(s, 1.0) @ w))]


def read_naming(brain: cd.Brain, things: Any, words: Words, rng: np.random.Generator) -> dict:
    right, sources = [], {}
    for kind in KINDS:
        for _ in range(TESTS):
            said = brain.say({"eyes": seen(things, a_thing(rng, kind))})
            right.append(words.heard(said) == kind)
            source = str(brain.bound.said_from[0])
            sources[source] = sources.get(source, 0) + 1
    return {"named": float(np.mean(right)), "said_from": sources}


def read_completion(brain: cd.Brain, things: Any, words: Words, rng: np.random.Generator
                    ) -> dict:
    """Y2: the kind a name heard alone brings back, from the records and from the state."""
    held = brain.bound
    # the probe: things seen alone, by a copy with no records (the state a thing itself gives)
    plain = copy.deepcopy(brain)
    plain.bound = None
    xs, labels = [], []
    for _ in range(PROBE):
        for kind in KINDS:
            xs.append(settled(plain, {"eyes": seen(things, a_thing(rng, kind))}))
            labels.append(kind)
    read = probe(np.array(xs), labels)
    dealt = copy.deepcopy(brain)
    store = dealt.bound.store
    order = np.random.default_rng(1).permutation(store.count)
    store.values[: store.count] = store.values[: store.count][order]
    out: dict[str, list] = {"records": [], "state": [], "barred": [], "dealt": [],
                            "dealt_records": []}
    for kind in KINDS:
        for _ in range(TESTS):
            take = words.take(kind, rng)
            out["records"].append(kind_of(brain.complete({"ear": take})["eyes"], things) == kind)
            out["dealt_records"].append(
                kind_of(dealt.complete({"ear": take})["eyes"], things) == kind)
            out["state"].append(read(settled(brain, {"ear": take})) == kind)
            out["dealt"].append(read(settled(dealt, {"ear": take})) == kind)
            brain.bound = None
            out["barred"].append(read(settled(brain, {"ear": take})) == kind)
            brain.bound = held
    return {name: float(np.mean(v)) for name, v in out.items()}


def exchanges(brain: cd.Brain, things: Any, words: Words, rng: np.random.Generator,
              items: list[tuple[Thing, str | None, str]], *, paid: bool) -> dict:
    """Each item (a thing, a word heard with it or None, the answer the trainer pays) until it
    is answered right ``RUNNING`` times running or ``BUDGET`` exchanges: the exchanges taken
    (None where the budget ran out) and the share right over the exchanges."""
    taken, rights = [], []
    for thing0, question, answer in items:
        running, count = 0, 0
        while count < BUDGET and running < RUNNING:
            thing = a_thing(rng, thing0.kind, thing0.colour)
            observation = {"eyes": seen(things, thing)}
            if question is not None:
                observation["ear"] = words.take(question, rng)
            said = brain.say(observation)
            right = words.heard(said) == answer
            brain.live({**observation, "voice": said})
            brain.live({"eyes": seen(things, thing)},
                       reward=[1.0 if right and paid else 0.0])
            running = running + 1 if right else 0
            rights.append(right)
            count += 1
        taken.append(count if running >= RUNNING else None)
    return {"exchanges": taken, "right": float(np.mean(rights))}


def table_control(items: list[tuple[Thing, str | None, str]], rng: np.random.Generator) -> dict:
    """A lookup table keyed by (the thing's kind and colour, the word heard): a paid answer is
    kept; otherwise it guesses among the words heard."""
    table: dict[tuple, str] = {}
    taken = []
    for thing, question, answer in items:
        running, count = 0, 0
        while count < BUDGET and running < RUNNING:
            key = (thing.kind, thing.colour, question)
            said = table.get(key) or str(rng.choice(WORDS))
            right = said == answer
            if right:
                table[key] = said
            running = running + 1 if right else 0
            count += 1
        taken.append(count if running >= RUNNING else None)
    return {"exchanges": taken}


def delta_control(brain: cd.Brain, things: Any, words: Words,
                  items: list[tuple[Thing, str | None, str]], rng: np.random.Generator
                  ) -> dict:
    """A delta-rule readout from the brain's code to the words, taught by the reward."""
    w = np.zeros((brain.code_width, len(WORDS)))
    taken = []
    for thing0, question, answer in items:
        running, count = 0, 0
        while count < BUDGET and running < RUNNING:
            thing = a_thing(rng, thing0.kind, thing0.colour)
            observation = {"eyes": seen(things, thing)}
            if question is not None:
                observation["ear"] = words.take(question, rng)
            code = brain.observation_code(observation)[0]
            logits = code @ w
            p = np.exp((logits - logits.max()) / TEMPERATURE)
            choice = int(rng.choice(len(WORDS), p=p / p.sum()))
            right = WORDS[choice] == answer
            w[:, choice] += 0.5 * code * ((1.0 if right else 0.0) - logits[choice])
            running = running + 1 if right else 0
            count += 1
        taken.append(count if running >= RUNNING else None)
    return {"exchanges": taken}


def read_reward(founder: int, things: Any, ear: Ear, words: Words) -> dict:
    """Y3 and Y4 on fresh brains of the founder after Listen alone."""
    rng = np.random.default_rng((founder, 3))
    names = [(a_thing(rng, kind), None, kind) for kind in NAMED]
    colours = [(a_thing(rng, kind, colour), QUESTIONS[0], colour) for kind, colour in PAIRS]
    out: dict[str, Any] = {}
    for paid in (True, False):
        brain = compose(founder, things, ear)
        listen(brain, words, np.random.default_rng((founder, 1)))
        tag = "paid" if paid else "unpaid"
        out[f"names_{tag}"] = exchanges(brain, things, words, rng, names, paid=paid)
        # the names said on views no exchange showed: the thing at a place drawn anew
        said = [words.heard(brain.say({"eyes": seen(things, a_thing(rng, t.kind, t.colour))}))
                == t.kind for t, _, _ in names for _ in range(TESTS)]
        out[f"names_{tag}"]["held_out"] = float(np.mean(said))
        out[f"colours_{tag}"] = exchanges(brain, things, words, rng, colours, paid=paid)
    plain = compose(founder, things, ear)
    out["table"] = {"names": table_control(names, rng), "colours": table_control(colours, rng)}
    out["delta"] = {"names": delta_control(plain, things, words, names, rng),
                    "colours": delta_control(plain, things, words, colours, rng)}
    return out


def read_named(founder: int, things: Any, ear: Ear, words: Words, *, taught: bool) -> dict:
    """N: a heard name moves the focus to the thing it names, in view and behind a screen."""
    from cadence.space.map import Map

    space = Map((-1.0, 1.0, -1.0, 1.0), (16, 16), units=8, values=dict(MAP_VALUES),
                surprise=True, seed=founder)
    brain = compose(founder, things, ear, space=space)
    rng = np.random.default_rng((founder, 5))
    listen(brain, words, rng)
    if taught:
        naming(brain, things, words, rng)
    right = {"in view": [], "behind a screen": []}
    for _ in range(ROOMS):
        kinds = [str(k) for k in rng.choice(list(KINDS), size=3, replace=False)]
        placed = [Thing(kind, str(rng.choice(KINDS[kind][2])), 0.0, 0.5) for kind in kinds]
        for screened in (False, True):
            space.reset()
            brain.reset()
            brain.orienting.reset()
            for k, thing in enumerate(placed):  # they appear one by one: the last draws the focus
                space.moment(seen=[(KINDS_INDEX[t.kind], SPOTS[j], {
                    name: seen(things, t)[name][0] for name in MAP_VALUES})
                    for j, t in enumerate(placed[: k + 1])])
                brain.act({"eyes": seen(things, thing)}, greedy=True)
            target = int(rng.integers(0, 2))  # one of the two that did not appear last
            if screened:  # the named thing goes behind a screen: the map holds it unseen
                lit = np.linalg.norm(space.centers - SPOTS[target], axis=1) > 0.3
                space.moment(seen=[(KINDS_INDEX[t.kind], SPOTS[j], {
                    name: seen(things, t)[name][0] for name in MAP_VALUES})
                    for j, t in enumerate(placed) if j != target], lit=lit)
            brain.act({"ear": words.take(placed[target].kind, rng)}, greedy=True)
            focus = brain.focus
            near = focus is not None and float(
                np.linalg.norm(np.asarray(focus.place) - SPOTS[target])) <= 0.25
            right["behind a screen" if screened else "in view"].append(bool(near))
    return {name: float(np.mean(v)) for name, v in right.items()}


KINDS_INDEX = {kind: k for k, kind in enumerate(KINDS)}


def life(founder: int) -> dict:
    things, ear, words = setting()
    began = time.time()
    rng = np.random.default_rng((founder, 1))
    brain = compose(founder, things, ear)
    listen(brain, words, rng)
    naming(brain, things, words, rng)
    out: dict[str, Any] = {"founder": founder, "records": int(brain.bound.store.count),
                           "parameters": int(brain.parameters())}
    out["Y1"] = read_naming(brain, things, words, rng)
    out["Y2"] = read_completion(brain, things, words, rng)
    out["Y3"] = read_reward(founder, things, ear, words)
    out["N"] = {"taught": read_named(founder, things, ear, words, taught=True),
                "untaught": read_named(founder, things, ear, words, taught=False)}
    out["seconds"] = round(time.time() - began, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--founders", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = []
    for founder in args.founders:
        results.append(life(founder))
        print(json.dumps(results[-1]), flush=True)
        args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
