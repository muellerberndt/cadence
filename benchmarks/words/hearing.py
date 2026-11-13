"""The hearing chamber (cadence#329): the words chamber's life with words heard aloud.

The parrot lives the words chamber's script (``words.script``: rooms, naming, practice, the
test, new words), and each word it meets comes in one of two forms:

- **Naming.** The teacher names a thing aloud (a voice of the founder's training voices says the
  word alone) at the rate ``by_ear``, else in writing. The H3 kinds are named in one form only:
  ``EAR_ONLY`` aloud, ``WRITING_ONLY`` in writing. Both are among the kinds named and never
  asked about in practice, so neither form of them is met in a question before the test.
- **Practice.** A question asked aloud by a training voice at the rate ``aloud``, else typed;
  the teacher's answer is written.
- **Reading aloud.** After practice, each H3 kind's word is read aloud ``reads`` times: its
  written form shown while a training voice says it, nothing attended.
- **The test,** nothing taught: every question of the words chamber's test rooms is asked typed,
  aloud by a training voice (a new utterance), aloud by a voice never heard (``new_voices`` of
  the same model's held-out speakers) and aloud by a voice of another model (``vctk_voices``).

The audio is made by ``hearing/voices.py`` (``audio``) and cached per founder; a life reads it
by event. Every word is the founder's nonce word, spoken from its phonemes.

Readings (``life``): each question's gate is ``<form>:<words gate>`` with the form ``T``
(typed), ``H1`` (training voice), ``H2`` (new voice) or ``H2x`` (another model's voice).
Questions about an H3 kind read as ``H3_eye`` (an ear-only kind asked in writing) and
``H3_ear`` (a writing-only kind asked aloud by a training voice). ``overlap`` reads, for each
word with both forms, the concepts its written form settles on against those its sound (a
training voice, alone) settles on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "hearing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import voices  # noqa: E402
import words  # noqa: E402
import world  # noqa: E402

DEFAULTS = {"voices": 24, "new_voices": 8, "vctk_voices": 4, "by_ear": 0.5, "aloud": 0.5,
            "reads": 5, "probes": 3}
EAR_ONLY, WRITING_ONLY = (4, 5), (6, 7)
POOL = 600  # the first POOL speakers of the libritts model (in the chamber's order) train


def voice_sets(seed: int, *, count: int, new: int, other: int) -> dict[str, list[voices.Voice]]:
    """The founder's voices: ``count`` training voices and ``new`` voices of the libritts
    model's two pools, and ``other`` voices of the vctk model."""
    order = np.random.default_rng(np.random.SeedSequence([329, 0])).permutation(904)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 329, 2]))
    train = rng.choice(order[:POOL], size=count, replace=False)
    held = rng.choice(order[POOL:], size=new, replace=False)
    vctk = rng.choice(109, size=other, replace=False)
    return {"train": [voices.Voice("libritts", int(s)) for s in train],
            "new": [voices.Voice("libritts", int(s)) for s in held],
            "vctk": [voices.Voice("vctk", int(s)) for s in vctk]}


def script(seed: int, **given: Any) -> list[tuple[Any, ...]]:
    """The founder's life as events: the words chamber's, each with its form and voice.

    - ``("room", room)``, ``("turn", heading)`` as in ``words.script``;
    - ``("name", place, word, phase, form, voice)``: ``form`` "ear" or "eye";
    - ``("practice", question, form, voice)``;
    - ``("read", word, voice)``: a word read aloud;
    - ``("ask", gate, question, form, voice)``;
    - ``("probe", word, voice)``: a word said alone, its concepts read (``overlap``)."""
    settings = {**DEFAULTS, **{k: v for k, v in given.items() if k in DEFAULTS}}
    base = words.script(seed, **{k: v for k, v in given.items() if k in words.DEFAULTS})
    lex = world.lexicon(seed)
    sets = voice_sets(seed, count=settings["voices"], new=settings["new_voices"],
                      other=settings["vctk_voices"])
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 329, 3]))
    train = sets["train"]

    def voice(pool: list[voices.Voice]) -> voices.Voice:
        return pool[int(rng.integers(len(pool)))]

    ear_only = {lex.kind[k] for k in EAR_ONLY}
    eye_only = {lex.kind[k] for k in WRITING_ONLY}
    out: list[tuple[Any, ...]] = []
    read = False
    for event in base:
        kind = event[0]
        if kind == "name":
            word = event[2]
            aloud = word in ear_only or (word not in eye_only and rng.random() < settings[
                "by_ear"])
            out.append((*event, "ear" if aloud else "eye", voice(train) if aloud else None))
        elif kind == "practice":
            aloud = rng.random() < settings["aloud"]
            out.append(("practice", event[1], "ear" if aloud else "eye",
                        voice(train) if aloud else None))
        elif kind == "ask":
            if not read:
                read = True
                for k in EAR_ONLY + WRITING_ONLY:
                    out += [("read", lex.kind[k], voice(train))
                            for _ in range(settings["reads"])]
            gate, q = event[1], event[2]
            named = set(q.text.split()) | set(q.answer)
            if named & ear_only:
                out.append(("ask", "H3_eye", q, "eye", None))
                out.append(("ask", f"H1:{gate}", q, "ear", voice(train)))
            elif named & eye_only:
                out.append(("ask", f"T:{gate}", q, "eye", None))
                out.append(("ask", "H3_ear", q, "ear", voice(train)))
            else:
                out.append(("ask", f"T:{gate}", q, "eye", None))
                out.append(("ask", f"H1:{gate}", q, "ear", voice(train)))
            out.append(("ask", f"H2:{gate}", q, "ear", voice(sets["new"])))
            out.append(("ask", f"H2x:{gate}", q, "ear", voice(sets["vctk"])))
        else:
            out.append(event)
    both = [w for k, w in enumerate(lex.kind[: world.TRAINED])
            if k not in EAR_ONLY + WRITING_ONLY] + lex.colour + lex.place
    for word in both:
        out += [("probe", word, voice(train)) for _ in range(settings["probes"])]
    return out


def spoken(events: list[tuple[Any, ...]]) -> list[tuple[int, str, voices.Voice]]:
    """Every utterance of a script: ``(event number, phonemes, voice)``."""
    out = []
    for i, event in enumerate(events):
        kind = event[0]
        if kind == "name" and event[4] == "ear":
            out.append((i, voices.sentence([event[2]], question=False), event[5]))
        elif kind in ("practice", "ask") and event[-2] == "ear":
            q = event[1] if kind == "practice" else event[2]
            out.append((i, voices.sentence(q.text.split(), question=True), event[-1]))
        elif kind in ("read", "probe"):
            out.append((i, voices.sentence([event[1]], question=False), event[2]))
    return out


def _say(job: tuple[str, list[tuple[int, str, str, int]]]) -> list[tuple[int, np.ndarray]]:
    folder, items = job
    made = voices.Voices(folder)
    out = []
    for i, text, model, speaker in items:
        rng = np.random.default_rng(np.random.SeedSequence([i, speaker, 329, 4]))
        out.append((i, made.say(text, voices.Voice(model, speaker), rng)))
    return out


def audio(seed: int, folder: str, path: str, *, workers: int = 4, **given: Any) -> str:
    """Synthesize every utterance of the founder's script into ``path`` (an npz: the samples
    end to end, each utterance's event number and length)."""
    from multiprocessing import Pool

    items = [(i, text, v.model, v.speaker) for i, text, v in spoken(script(seed, **given))]
    jobs = [(folder, items[k::workers]) for k in range(workers)]
    with Pool(workers) as pool:
        made = [pair for part in pool.map(_say, jobs) for pair in part]
    made.sort(key=lambda pair: pair[0])
    np.savez(path, samples=np.concatenate([a for _, a in made]),
             events=np.array([i for i, _ in made]), lengths=np.array([len(a) for _, a in made]))
    return path


def load(path: str) -> dict[int, np.ndarray]:
    held = np.load(path)
    samples, ends = held["samples"], np.cumsum(held["lengths"])
    return {int(i): samples[e - n: e] for i, n, e in zip(held["events"], held["lengths"], ends,
                                                         strict=True)}


def make(seed: int, **settings: Any) -> Any:
    """The founder's brain with ears and a voice: the words chamber's brain and an ``Ear``
    (settings named ``ear_<setting>``)."""
    from cadence.words.hearing import Hearing

    ear = {k[4:]: v for k, v in settings.items() if k.startswith("ear_")}
    told = {k: v for k, v in settings.items() if k in ("things", "say_at")}
    return Hearing(words.make_brain(seed), **told, **ear)


def life(seed: int, sounds: dict[int, np.ndarray], *, record: Any = None,
         **given: Any) -> dict[str, Any]:
    """One founder's life with ears; the readings by gate."""
    settings = {k: v for k, v in given.items() if k in DEFAULTS or k in words.DEFAULTS}
    events = script(seed, **settings)
    hearing = make(seed, **{k: v for k, v in given.items() if k not in settings})
    brain = hearing.brain
    lex = world.lexicon(seed)
    readings: dict[str, list[bool]] = {}
    overlap: list[tuple[float, bool]] = []
    heard_as: dict[str, dict[str, int]] = {}  # each unit's namings by the word said (a reading)
    seconds: dict[str, float] = {}
    phase, began = "naming", time.time()
    r: world.Room | None = None
    # the ear's first sounds: the namings' sounds heard once, before the life, for its fields
    began = time.time()
    for i, event in enumerate(events):
        if event[0] == "name" and event[4] == "ear":
            hearing.listen(sounds[i])
    seconds["listening"] = time.time() - began
    phase, began = "naming", time.time()
    for i, event in enumerate(events):
        kind = event[0]
        now = {"name": "naming", "practice": "practice", "read": "reading", "ask": "test",
               "probe": "probe"}.get(kind)
        if now is not None and now != phase:
            seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
            phase, began = now, time.time()
        if kind == "room":
            r = event[1]
            words.look_around(brain, r)
        elif kind == "turn":
            assert r is not None
            words.turn(brain, r, event[1])
        elif kind == "name":
            if event[4] == "ear":
                hearing.name(sounds[i], event[1])
                for key in hearing.last["records"]:
                    told = heard_as.setdefault(key, {})
                    told[event[2]] = told.get(event[2], 0) + 1
            else:
                brain.name(event[2], event[1])
        elif kind == "practice":
            q = event[1]
            answer = q.answer if len(q.answer) > 1 else q.answer[0]
            if event[2] == "ear":
                hearing.practice(sounds[i], answer)
            else:
                brain.practice(q.text, answer)
        elif kind == "read":
            hearing.read(event[1], sounds[i])
        elif kind == "ask":
            gate, q = event[1], event[2]
            said = hearing.ask(sounds[i]) if event[3] == "ear" else brain.ask(q.text)
            ok = world.right(q, said)
            readings.setdefault(gate, []).append(ok)
            if record is not None:
                record.append({"gate": gate, "question": q.text, "form": q.form,
                               "about": q.about, "answer": q.answer, "said": said, "right": ok,
                               "heard": list(hearing.last.get("records", []))
                               if event[3] == "ear" else None})
        elif kind == "probe":
            sensed = np.zeros(brain.lexicon.concepts, dtype=bool)
            for g in brain.senses.groups:
                sensed[brain.lexicon.groups[g]] = True
            by_eye = np.where(sensed, brain.lexicon.hear(event[1]), 0.0)
            by_ear = np.where(sensed, hearing.heard(sounds[i], learn=False)[0], 0.0)
            cosine = float(by_eye @ by_ear / max(np.linalg.norm(by_eye) * np.linalg.norm(
                by_ear), 1e-12))
            overlap.append((cosine, int(np.argmax(by_eye)) == int(np.argmax(by_ear))))
    seconds[phase] = seconds.get(phase, 0.0) + time.time() - began
    del lex
    out = readings_of(seed, settings, readings, overlap, hearing, seconds)
    total = sum(sum(v.values()) for v in heard_as.values())
    out["ear_purity"] = sum(max(v.values()) for v in heard_as.values()) / max(total, 1)
    out["ear_named_units"] = len(heard_as)
    return out


def readings_of(seed: int, settings: dict[str, Any], readings: dict[str, list[bool]],
                overlap: list[tuple[float, bool]], hearing: Any,
                seconds: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {"seed": int(seed), **settings}
    for gate, oks in sorted(readings.items()):
        out[gate] = float(np.mean(oks)) if oks else float("nan")
        out[f"{gate}_n"] = len(oks)

    def pool(form: str, *names: str) -> float:
        oks = [ok for g, v in readings.items() for ok in v
               if g.startswith(form + ":") and (not names or g.split(":")[1].startswith(names))]
        return float(np.mean(oks)) if oks else float("nan")

    for form in ("T", "H1", "H2", "H2x"):
        out[form] = pool(form, "W1", "W4", "W5")
        for part in ("W1", "W4", "W5", "W2", "W3"):
            out[f"{form}_{part}"] = pool(form, part)
    out["overlap_cosine"] = float(np.mean([c for c, _ in overlap])) if overlap else float("nan")
    out["overlap_same"] = float(np.mean([s for _, s in overlap])) if overlap else float("nan")
    out["parameters"] = hearing.parameters()
    out["ear_records"] = hearing.ear.records
    out["words"] = len(hearing.brain.lexicon.forms)
    synapses = int(hearing.brain.map.connectome.synapses)
    work = hearing.brain.map.work
    out["macs"] = int(hearing.brain.lexicon.work["macs"] + hearing.ear.work["macs"] + synapses * (
        work["moments"] + 2 * work["questions"]))
    out["seconds"] = seconds
    return out


if __name__ == "__main__":
    import json

    command, seed = sys.argv[1], int(sys.argv[2])
    told = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    if command == "audio":
        print(audio(seed, "/data/voices", sys.argv[3], workers=int(told.pop("workers", 4)),
                    **told))
    else:
        found: list[Any] = []
        result = life(seed, load(sys.argv[3]), record=found, **told)
        print(json.dumps(result, indent=1, default=float))
        if len(sys.argv) > 5:
            Path(sys.argv[5]).write_text(json.dumps(found, default=str))
