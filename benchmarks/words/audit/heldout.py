"""What is held out, and are the nonce words the founder's own? (cadence#326 audit)

Read off the founders' scripts and lexicons alone (no brain lives):

- **The nonce words**: within a founder every form is distinct; across founders, the share of
  a founder's forms another founder also draws, and of those the share that name the same
  concept there (a fixed lexicon would read 1); no form is an English concept name; the role
  of a word (kind, colour, place, question, nothing, foil) is not readable from its length.
- **Held out**: no naming, practice or W1/W4/W5 test room holds a kind in its unseen colour;
  the new kinds appear in no room before their own namings; the named-only kinds' words appear
  in no practice question or answer; the test's foils are heard nowhere before the test; the
  unseen colour's thing (W3) is in no earlier room in that colour.
- **What W2 asks**: the share of W2 colour questions whose answer is the colour the new kind
  had at its three namings, and of W2 where questions whose answer is the place it had then
  (a question answerable from the namings alone reads 1).
- **Scenes**: test rooms whose things, kinds and colours a naming room holds within 0.1 m.

    python heldout.py <out> [first founder] [founders]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import common
import numpy as np
import world


def roles(lex: world.Lexicon) -> dict[str, str]:
    out = {w: f"kind{k}" for k, w in enumerate(lex.kind)}
    out.update({w: f"colour{c}" for c, w in enumerate(lex.colour)})
    out.update({w: f"place{p}" for p, w in enumerate(lex.place)})
    out.update({w: f"question:{n}" for n, w in lex.question.items()})
    out[lex.none] = "none"
    out.update({w: f"foil{i}" for i, w in enumerate(lex.foils)})
    return out


def lexicons(seeds: list[int]) -> dict[str, Any]:
    lexes = {s: world.lexicon(s) for s in seeds}
    shared = same = total = 0
    lengths: dict[str, list[int]] = {}
    english = set(world.KIND_NAMES + world.COLOUR_NAMES + world.PLACE_NAMES
                  + world.QUESTION_WORDS) | {"i do not know"}
    distinct, englishy = True, 0
    forms_english: set[str] = set()
    index: dict[str, list[tuple[int, str]]] = {}
    for s, lex in lexes.items():
        role = roles(lex)
        distinct &= len(role) == world.KINDS + world.COLOURS + 4 + 6 + 1 + world.FOILS
        englishy += sum(w in english for w in role)
        forms_english.update(w for w in role if w in english)
        for w, r in role.items():
            lengths.setdefault(r.split(":")[0].rstrip("0123456789"), []).append(len(w))
            index.setdefault(w, []).append((s, r))
    for held in index.values():
        for s, r in held:
            for t, other in held:
                if t != s:
                    shared += 1
                    same += int(other == r)
    total = sum(len(held) for held in index.values()) * (len(seeds) - 1)
    return {"founders": len(seeds), "distinct_within_founder": bool(distinct),
            "english_forms": englishy, "english_forms_drawn": sorted(forms_english),
            "share_of_forms_another_founder_draws": shared / max(total, 1),
            "of_those_same_concept": (same / shared) if shared else None,
            "mean_length_by_role": {r: round(float(np.mean(v)), 2) for r, v in lengths.items()}}


def held_out(seed: int, settings: dict[str, Any]) -> dict[str, Any]:
    import words

    lex = world.lexicon(seed)
    unseen = world.unseen_colours(seed)
    events = words.script(seed, **settings)
    phase = "naming"
    rooms_by_phase: dict[str, list[world.Room]] = {}
    heard_before_test: set[str] = set()
    breaks: dict[str, int] = {"unseen_combination_in_room": 0, "new_kind_before_its_naming": 0,
                              "named_only_word_in_practice": 0, "test_foil_heard_before": 0,
                              "w3_thing_in_earlier_room": 0}
    new = set(range(world.TRAINED, world.KINDS))
    new_named: dict[int, tuple[int, int]] = {}  # new kind: (colour, place) at its namings
    w2 = {"colour": [0, 0], "where": [0, 0]}
    r: world.Room | None = None
    current = "naming"
    seen_pairs: set[tuple[int, int]] = set()
    for i, event in enumerate(events):
        kind = event[0]
        if kind == "room":
            r = event[1]
            # the phase a room belongs to: the kind of the next event that is not a turn
            follow = next(e for e in events[i + 1:] if e[0] != "turn")
            current = {"name": follow[3] if follow[0] == "name" else "naming",
                       "practice": "practice", "ask": follow[1] if follow[0] == "ask" else "test",
                       "room": "naming"}.get(follow[0], "naming")
            rooms_by_phase.setdefault(current, []).append(r)
            is_w3 = current.startswith("W3")
            for t in r.things:
                if unseen.get(t.kind) == t.colour and not is_w3:
                    breaks["unseen_combination_in_room"] += 1
                if t.kind in new and current not in ("new",) and t.kind not in new_named:
                    breaks["new_kind_before_its_naming"] += 1
            if is_w3:
                odd = [t for t in r.things if unseen.get(t.kind) == t.colour]
                breaks["w3_thing_in_earlier_room"] += sum((t.kind, t.colour) in seen_pairs
                                                          for t in odd)
            if current in ("naming", "practice", "new") or current.startswith("W1"):
                seen_pairs |= {(t.kind, t.colour) for t in r.things}
        elif kind == "name":
            assert r is not None
            heard_before_test.add(event[2])
            if event[3] == "new":
                t = min(r.things, key=lambda th: float(np.linalg.norm(th.place - event[1])))
                new_named[t.kind] = (t.colour, world.place_of(r.heading, t.place))
        elif kind == "practice":
            q = event[1]
            heard_before_test |= set(q.text.split()) | set(q.answer)
            if set(lex.kind[4: world.TRAINED]) & (set(q.text.split()) | set(q.answer)):
                breaks["named_only_word_in_practice"] += 1
        elif kind == "ask":
            gate, q = event[1], event[2]
            if gate == "W4_never_taught":
                breaks["test_foil_heard_before"] += int(any(
                    w in heard_before_test for w in common.content(lex, q.text)))
            if gate in ("W2_colour", "W2_where") and q.kind in new_named:
                c, p = new_named[q.kind]
                said = lex.colour[c] if gate == "W2_colour" else lex.place[p]
                row = w2["colour" if gate == "W2_colour" else "where"]
                row[0] += 1
                row[1] += int(q.answer == [said])
        phase = current
    del phase
    # test rooms that repeat a naming room: the same kinds and colours within 0.1 m
    def key(room: world.Room) -> list[tuple[int, int, np.ndarray]]:
        return sorted(((t.kind, t.colour, t.place) for t in room.things), key=lambda x: x[0])

    repeats = 0
    for test in rooms_by_phase.get("W1_practised", []) + rooms_by_phase.get("W1_named_only", []):
        for old in rooms_by_phase.get("naming", []):
            a, b = key(test), key(old)
            if len(a) == len(b) and all(x[0] == y[0] and x[1] == y[1]
                                        and np.linalg.norm(x[2] - y[2]) < 0.1
                                        for x, y in zip(a, b, strict=True)):
                repeats += 1
    return {"seed": seed, "breaks": breaks, "test_rooms_repeating_a_naming_room": repeats,
            "w2_colour_answer_is_naming_colour": w2["colour"][1] / max(w2["colour"][0], 1),
            "w2_where_answer_is_naming_place": w2["where"][1] / max(w2["where"][0], 1),
            "w2_questions": {k: v[0] for k, v in w2.items()},
            "rooms_by_phase": {k: len(v) for k, v in rooms_by_phase.items()}}


def main(argv: list[str]) -> None:
    import json

    out = Path(argv[1]) if len(argv) > 1 else Path("audit_out")
    first = int(argv[2]) if len(argv) > 2 else common.FOUNDERS[0]
    count = int(argv[3]) if len(argv) > 3 else len(common.FOUNDERS)
    seeds = list(range(first, first + count))
    result = {"lexicons": lexicons(seeds), "lexicons_1000": lexicons(list(range(1000))),
              "founders": [held_out(s, {}) for s in seeds]}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "heldout.json"
    path.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result["lexicons"], indent=1))
    print(json.dumps({k: v for k, v in result["lexicons_1000"].items()
                      if k != "mean_length_by_role"}, indent=1))
    total: dict[str, int] = {}
    for f in result["founders"]:
        for k, v in f["breaks"].items():
            total[k] = total.get(k, 0) + v
    print("breaks over the founders:", total)
    print("test rooms repeating a naming room:",
          sum(f["test_rooms_repeating_a_naming_room"] for f in result["founders"]))
    for k in ("w2_colour_answer_is_naming_colour", "w2_where_answer_is_naming_place"):
        print(k, round(float(np.mean([f[k] for f in result["founders"]])), 3))
    print("rooms by phase (first founder):", result["founders"][0]["rooms_by_phase"])


if __name__ == "__main__":
    main(sys.argv)
