"""R6 inside the page: a parrot raised briefly (founder 33099999, 4 naming rooms), each moment of
a turn timed by its parts, and each question."""

import json
import time

import numpy as np


def run() -> str:
    import retina as chamber
    import words
    seed = 33099999  # a timing founder of this lane's range, no gated life
    events = chamber.script(seed, naming_rooms=4, practice_rooms=2, test_rooms=3, unseen_rooms=1,
                            new_rooms=1)
    eyes = chamber.make_brain(seed)
    camera = words.make_brain(seed)
    sight = chamber.Sight(seed, eyes, camera)
    retina = eyes.retina
    parts = {"render": [], "see": [], "map": [], "moment": []}
    asked = []
    r = None
    began = time.perf_counter()
    for event in events:
        if event[0] == "room":
            r = event[1]
            sight.look_around(r)
        elif event[0] == "turn":
            r.heading = event[1]
            t0 = time.perf_counter()
            images = chamber.render(seed, r, retina.eyes, eyes.body.at, r.heading, rng=sight.rng)
            t1 = time.perf_counter()
            eyes.body.heading = r.heading
            reports, lit = retina.see(images, eyes.body.at, r.heading)
            t2 = time.perf_counter()
            eyes.look(seen=reports, lit=lit)
            t3 = time.perf_counter()
            parts["render"].append(t1 - t0)
            parts["see"].append(t2 - t1)
            parts["map"].append(t3 - t2)
            parts["moment"].append(t3 - t1)
        elif event[0] == "name":
            eyes.name(event[2], event[1])
        elif event[0] == "practice":
            q = event[1]
            eyes.practice(q.text, q.answer if len(q.answer) > 1 else q.answer[0])
        elif event[0] == "ask":
            q = event[2]
            t0 = time.perf_counter()
            eyes.ask(q.text)
            asked.append(time.perf_counter() - t0)
    total = time.perf_counter() - began

    def ms(v: list[float]) -> dict[str, float]:
        v = np.array(v) * 1000.0
        return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "max": float(v.max())}

    out = {k: ms(v) for k, v in parts.items()}
    out["question"] = ms(asked)
    out["seconds_total"] = total
    out["things_seen_per_moment"] = retina.work["figures"] / max(retina.work["moments"], 1)
    import cadence
    out["cadence"] = cadence.__version__
    out["numpy"] = np.__version__
    return json.dumps(out)
