"""The grammar chamber (cadence#173): the language, its sealed splits, the panels, the scorer,
a short life of each arm and the controls' encoding."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gr = _load("grammar")


def _agrees(row) -> bool:
    words = row.target[:-1]
    verbs = [w for w in words if 8 <= w <= 11]
    return len(verbs) == 1 and (verbs[0] - 8) % 2 == row.meta["agent"][1]


def test_the_recipe_is_the_sequence_batterys():
    assert gr.ARMS["recipe"] == {
        "episodic": False, "teaching": "normalized", "recall": True, "recall_founder": True,
            "recall_recency": 0.0,
        "learning_surprise": True, "recall_boundary": True, "recall_anchor": True,
        "recall_slope": 0.1, "recall_position_rate": 0.01, "recall_relative": True}
    assert gr.ARMS["maturity"] == {**gr.ARMS["recipe"], "maturity": True}


def test_every_word_is_an_input_and_an_action():
    assert gr.ACTIONS == len(gr.WORDS) == 18
    assert gr.INPUTS == gr.ACTIONS + len(gr.CONCEPTS)
    assert all(np.argmax(gr.heard(w)) == w for w in range(gr.ACTIONS))
    assert gr.spoken(3)[gr.C["SPEAK"]] == 1.0 and gr.heard(3)[gr.C["SPEAK"]] == 0.0


def test_clauses_agree_with_the_agent_across_distractors_in_both_grammars():
    rng = np.random.default_rng(0)
    for dialect in (False, True):
        for L in range(5):
            for _ in range(20):
                row = gr.clause(rng, L, dialect=dialect)
                assert _agrees(row) and row.target[-1] == gr.END
                assert len(row.pre) == L + 3 and len(row.target) == 2 * L + 4
                m = row.meta
                k = next(i for i, w in enumerate(row.target) if 8 <= w <= 11)
                patient = gr.noun(*m["patient"])
                assert row.target[k - 1 if dialect else k + 1] == patient
                assert row.target[0] == gr.noun(*m["agent"])
    row = gr.clause(rng, 2, agent=(0, 0), patient=(1, 1), v=0, mods=[(2, 1), (3, 1)],
                    inverted=False)
    assert gr.say(row.target) == "dog near foxes near owls sees cats end"
    assert gr.say(gr.twin(row, dialect=True).target) == "dog near foxes near owls cats sees end"
    inverted = gr.twin(row, inverted=True)
    assert inverted.target == row.target and not np.array_equal(inverted.pre[1], row.pre[1])


def test_dyck_sentences_close_the_innermost_bracket_first():
    rng = np.random.default_rng(1)
    row = gr.dyck(rng, [1, 1, 0, 1], [0, 1, 0, 1])
    assert gr.say(row.target) == "( [ ] [ ] ) end"
    assert row.meta == {"prefix": 4, "depth": 2, "open": 2, "nested": False, "types": [0, 1, 0, 1]}
    for _ in range(50):
        pattern = gr.dyck_pattern(rng, int(rng.integers(1, 7)), gr.TRAIN_DEPTH)
        row = gr.dyck(rng, pattern)
        assert row.meta["depth"] <= gr.TRAIN_DEPTH and row.meta["open"] >= 1
        assert gr.score(row, row.target)["valid"]


def test_a_path_follows_the_facts_heard_and_ends_at_a_sink():
    rng = np.random.default_rng(2)
    for hops in (1, 2, 3):
        row = gr.path(rng, hops)
        heard = [int(np.argmax(x)) for x in row.pre[:-1]]
        facts = {heard[i]: heard[i + 2] for i in range(0, len(heard), 4)}
        assert len(facts) == hops + 2  # every first is told once
        chain = row.meta["chain"]
        assert all(facts[chain[h]] == chain[h + 1] for h in range(hops))
        assert chain[-1] not in facts
        assert row.target == [w for e in chain for w in (e, gr.NEAR)][:-1] + [gr.END]
        assert row.pre[-1][gr.C["FROM"]] == 1.0


def test_training_never_shows_a_held_out_combination():
    for founder in (0, 1, 2, 3):
        split = gr.Split.of(founder)
        rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
        for grammar in ("g1", "g2"):
            for _ in range(200):
                for row in gr.train_batch(rng, split, grammar):
                    if row.family == "clause":
                        m = row.meta
                        assert not split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"])
                        assert m["L"] in gr.TRAIN_L and m["dialect"] == (grammar == "g2")
                    elif row.family == "dyck":
                        assert row.meta["depth"] <= gr.TRAIN_DEPTH
                    else:
                        assert row.meta["hops"] in gr.TRAIN_HOPS


def test_the_panels_are_sealed_deterministic_and_one_shape_per_batch():
    tests = gr.panels(5)
    again = gr.panels(5)
    split = gr.Split.of(5)
    for name, batches in tests.items():
        for rows, twins in zip(batches, again[name], strict=True):
            assert [r.target for r in rows] == [r.target for r in twins]
            assert len({len(r.pre) for r in rows}) == 1 and len(rows) == gr.BATCH
            assert len({len(r.target) for r in rows}) == 1
    for rows in tests["clause_held"]:
        assert all(split.held(tuple(r.meta["agent"]), tuple(r.meta["patient"]), r.meta["verb"])
                   for r in rows)
    for rows in tests["clause_long"]:
        assert all(r.meta["L"] in gr.TEST_L for r in rows)
    for rows in tests["dyck_deep"]:
        assert all(r.meta["depth"] in gr.TEST_DEPTH for r in rows)
    for rows in tests["roles"]:
        for a, b in zip(rows[0::2], rows[1::2], strict=True):
            assert a.meta["agent"] == b.meta["patient"] and a.meta["patient"] == b.meta["agent"]
            assert {a.meta["agent"][0], a.meta["patient"][0]} == set(split.pair)
            nouns = [sorted(w for w in r.target if not 8 <= w <= 11) for r in (a, b)]
            assert nouns[0] == nouns[1] and _agrees(a) and _agrees(b)  # the same words


def test_context_twins_share_their_recent_words_and_differ_earlier():
    tests = gr.panels(3)
    for name in ("ctx_agent", "ctx_verb", "ctx_dyck"):
        for rows in tests[name]:
            for a, b in zip(rows[0::2], rows[1::2], strict=True):
                assert a.force == b.force
                assert a.target[1:a.force] == b.target[1:b.force]  # the recent words
                assert a.target[a.force:] != b.target[b.force:]  # the continuation differs
                if name == "ctx_verb":
                    assert a.target[:a.force] == b.target[:b.force]
                    assert not np.array_equal(a.pre[0], b.pre[0])  # only the meaning's verb
        for rows in tests[name + "_removed"]:
            for a, b in zip(rows[0::2], rows[1::2], strict=True):
                assert not a.pre and a.target[:a.force] == b.target[:b.force]


def test_the_scorer_reads_the_whole_sentence():
    rng = np.random.default_rng(4)
    row = gr.clause(rng, 1, agent=(0, 1), patient=(1, 0), v=1, mods=[(2, 0)], inverted=False)
    right = list(row.target)
    assert gr.say(right) == "dogs near fox chase cat end"
    s = gr.score(row, right)
    assert s["exact"] and s["agree"] and s["roles"] and s["mods"] and s["agree_own"]
    attracted = right.copy()
    attracted[3] = gr.verb(1, 0)  # "chases": agrees with the distractor
    s = gr.score(row, attracted)
    assert not s["exact"] and not s["agree"] and s["verb"] and s["roles"] and s["attracted"]
    assert gr.score(row, right)["attracted"] is False
    swapped = [gr.noun(1, 0), gr.NEAR, gr.noun(2, 0), gr.verb(1, 0), gr.noun(0, 1), gr.END]
    s = gr.score(row, swapped)
    assert not s["roles"] and s["agree_own"] and not s["agree"]
    dyck = gr.dyck(rng, [1, 1], [0, 1])
    assert gr.score(dyck, dyck.target)["valid"]
    assert not gr.score(dyck, [gr.END])["valid"]  # an empty sentence is not the one asked
    assert not gr.score(dyck, [gr.OPEN_ROUND, gr.OPEN_SQUARE, gr.CLOSE_ROUND,
                               gr.CLOSE_SQUARE, gr.END])["valid"]
    path = gr.path(rng, 2)
    s = gr.score(path, path.target)
    assert s["exact"] and s["hops"] == 2 and s["first_hop"]


def test_a_short_life_of_each_arm_and_readings_on_forks():
    small = {"phases": [["g1", 2], ["g2", 1]], "panel_batches": 1, "module": 16,
             "continuation": True,
             "forks": [{"name": "paid", "after": 0, "phases": [["g2:reward", 1]]}]}
    rows = {arm: gr.run_life({"arm": arm, "founder": 0, **small})
            for arm in ("recipe", "frozen")}
    for row in rows.values():
        assert [r["phase"] for r in row["readings"]] == [-1, 0, 1]
        panels = row["readings"][-1]["panels"]
        assert set(panels) >= set(gr.G1_PANELS) | set(gr.CONTEXT_PANELS) | {"g2", "g2_long"}
        assert all(0.0 <= p["exact"] <= 1.0 for p in panels.values())
    assert rows["recipe"]["readings"][1]["work"]["lessons"] > 0
    assert rows["recipe"]["continuation"]["identical"]
    paid = rows["recipe"]["branches"]["paid"]["readings"]
    assert [r["phase"] for r in paid] == [1] and paid[0]["work"]["steps"] > 0
    assert set(paid[0]["located"]) == set(gr.LOCATE)
    assert rows["frozen"]["readings"][1]["work"]["lessons"] == 0
    brain = gr.make_brain("recipe", 0, module=16)
    rng = np.random.default_rng(0)
    gr.speak(brain, gr.train_batch(rng, gr.Split.of(0), "g1"), teach=True)
    before = copy.deepcopy(brain)
    gr.read_panel(brain, gr.panels(0, 1)["clause"])
    for name, array in before.learner.__dict__.items():
        if isinstance(array, np.ndarray):
            assert np.array_equal(array, brain.learner.__dict__[name])
    assert np.array_equal(before.recall.head_map(0)[0], brain.recall.head_map(0)[0])


def test_maturity_is_inert_under_lessons_and_moves_under_reward():
    split = gr.Split.of(0)
    said = {}
    for arm in ("recipe", "maturity"):
        brain = gr.make_brain(arm, 0, module=16)
        rng = np.random.default_rng(0)
        for _ in range(2):
            gr.speak(brain, gr.train_batch(rng, split, "g1"), teach=True)
        said[arm] = gr.speak(brain, gr.panels(0, 1)["clause"][0], teach=False)
        if arm == "maturity":
            assert brain.basal_ganglia.maturity.use == 0.0
            gr.practise(brain, gr.train_batch(rng, split, "g1"))
            assert brain.basal_ganglia.maturity.use > 0.0
    assert said["recipe"] == said["maturity"]


def test_teaching_refuses_ragged_batches():
    rng = np.random.default_rng(0)
    rows = [gr.clause(rng, 0), gr.clause(rng, 1)]
    with pytest.raises(ValueError):
        gr.speak(gr.make_brain("recipe", 0, module=16), rows, teach=True)


def test_the_controls_hear_what_the_brain_hears():
    pytest.importorskip("torch")
    ct = _load("controls")
    rows = gr.panels(0, 1)["ctx_agent"][0]
    x, y = ct.encode(rows)
    r = rows[0]
    assert x.shape == (len(rows), len(r.pre) + len(r.target), gr.INPUTS)
    assert (y[:, : len(r.pre)] == -100).all() and list(y[0, len(r.pre):]) == r.target
    assert x[0, len(r.pre), gr.C["GO"]] == 1.0
    assert np.argmax(x[0, len(r.pre) + 1]) == r.target[0]
    for model in ct.MODELS:
        width = ct.matched_width(model, 24000)
        assert abs(ct.count(ct.build(model, width, 0)) - 24000) < 2500
    learner = ct.Learner("lstm", 8, 0, 0.003)
    learner.update(rows)
    said = learner.speak(rows)
    assert len(said) == len(rows) and all(len(s) <= len(r.target) + 4 for s in said)
    assert learner.work["updates"] == 1 and learner.work["macs"] > 0


def test_the_erasures_reach_only_the_earlier_context():
    brain = gr.make_brain("recipe", 0, module=16)
    rows = gr.panels(0, 1)["ctx_agent"][0]
    seen = {}

    def look(where):
        act = gr.erase(where)

        def wrapped(b, rs):
            before = (b.recall.values.copy(), b.working_memory.trace.copy())
            act(b, rs)
            seen[where] = before, (b.recall.values.copy(), b.working_memory.trace.copy())

        return wrapped

    for where in ("recall", "trace"):
        gr.speak(copy.deepcopy(brain), rows, teach=False, intervene=look(where))
    (values, trace), (after, after_trace) = seen["recall"]
    earlier = len(rows[0].pre) + 2  # the meaning, GO and the first given word
    slots = values.shape[1]
    recent = slots - (len(rows[0].pre) + rows[0].force)  # the boundary slot comes first
    assert recent == 1 and values[:, recent:recent + earlier].any()
    assert not after[:, : recent + earlier].any()
    assert np.array_equal(after[:, recent + earlier:], values[:, recent + earlier:])
    assert np.array_equal(after_trace, trace)
    (_, trace), (after, after_trace) = seen["trace"]
    assert trace.any() and not after_trace.any()


def test_every_sentence_is_answerable_from_what_the_brain_hears():
    for founder in (0, 7):
        for name, batches in gr.panels(founder).items():
            for rows in batches:
                for r in rows:
                    if r.meta.get("removed"):
                        continue  # the context removed: answerable by no speaker
                    assert gr.solve(r.pre) == r.target, name
        rng = np.random.default_rng(founder)
        for grammar in ("g1", "g2"):
            for _ in range(100):
                for r in gr.train_batch(rng, gr.Split.of(founder), grammar):
                    assert gr.solve(r.pre) == r.target


def test_located_reads_the_slots_as_the_read_saw_them():
    """The read at ``GO`` saw the meaning's slots and the boundary, not ``GO``'s own slot, and
    ``located`` maps its columns to moments by the ages from before the act."""
    brain = gr.make_brain("recipe", 0, module=16)
    rows = gr.panels(0, 1)["clause"][0]
    pre = len(rows[0].pre)
    assert gr.sources(rows[0])[0] == (2 if rows[0].meta["inverted"] else 1)  # the agent
    brain.reset()
    for t in range(pre):
        brain.act(np.stack([r.pre[t] for r in rows]), greedy=True)
    ages = np.asarray(brain.recall.ages())
    brain.act(np.tile(gr.concept("GO"), (len(rows), 1)), greedy=True)
    p = brain.recall.attended_heads[0]
    assert p.shape[1] == len(ages) == pre + 1  # the boundary and the meaning
    assert len(brain.recall.ages()) == pre + 2  # GO's slot written after the act
    q = pre - 1 - ages[np.argmax(p, axis=1)]
    assert ((q >= -1) & (q < pre)).all()
    located = gr.locate(brain, [rows])
    assert all(0.0 <= v["delivered"] <= 1.0 for v in located.values())


def test_a_window_of_one_is_the_chambers_life_and_wider_windows_say_k_words():
    wn = _load("window")
    assert wn.windows([0, 12, 5, 9, 2, 17], 4) == [[0, 12, 5, 9], [2, 17, 17, 17]]
    assert wn.LONGEST == 8
    split = gr.Split.of(0)
    rng = np.random.default_rng(0)
    batches = [gr.train_batch(rng, split, "g1") for _ in range(3)]
    rows = gr.panels(0, 1)["clause"][0]
    a, b = gr.make_brain("recipe", 0, module=16), wn.make_brain(1, 0, module=16)
    wa, wb = gr.new_work(), gr.new_work()
    for batch in batches:
        gr.speak(a, batch, teach=True, work=wa)
        wn.speak(b, batch, 1, teach=True, work=wb)
    assert wa == wb
    assert gr.speak(a, rows, teach=False) == wn.speak(b, rows, 1, teach=False)
    assert gr.forced(a, rows) == wn.forced(b, rows, 1)
    for k in (4, 8):
        brain = wn.make_brain(k, 0, module=16)
        assert brain.describe()["layout"]["slots"] == [gr.ACTIONS] * k
        work = gr.new_work()
        for batch in batches:
            wn.speak(brain, batch, k, teach=True, work=work)
        assert work["lessons"] == sum(len(wn.windows(batch[0].target, k)) for batch in batches)
        said = wn.speak(brain, rows, k, teach=False)
        assert all(len(s) % k == 0 or s[-1] == gr.END for s in said)
        assert all(len(s) <= len(r.target) + 4 + k for s, r in zip(said, rows, strict=True))
        located = wn.locate(brain, [rows], k)
        assert all(0.0 <= v["mass"] <= 1.0 for v in located.values())
        with pytest.raises(ValueError):
            wn.speak(brain, gr.panels(0, 1)["ctx_agent"][0], k, teach=False)
    row = wn.run_life({"k": 4, "founder": 0, "batches": 2, "panel_batches": 1, "module": 16})
    assert set(row["panels"]) == set(wn.PANELS) | {"win_agent", "win_verb"}
    assert row["work"]["sentences"] == 32 and row["advance"] == "echo"
    assert row["panels"]["roles"]["both"] >= 0.0 and "clause_L0_verb" in row["located"]["clause"]
    dyck = row["panels"]["dyck"]
    assert dyck["fits_n"] + dyck["spills_n"] == 16
    assert "mass" in row["located"]["clause"]["clause_L0_verb"]
    panels = wn.context_panels(0, 4, 1)
    for rows in panels["win_agent"]:  # the agent's number differs: the first word, then shared
        for a, b in zip(rows[0::2], rows[1::2], strict=True):
            assert a.force == b.force == 4 and a.target[1:4] == b.target[1:4]
            assert a.target[4:] != b.target[4:] and a.meta["L"] == 2
    for rows in panels["win_verb"]:  # the first window is the same; the message's verb differs
        for a, b in zip(rows[0::2], rows[1::2], strict=True):
            assert a.target[:4] == b.target[:4] and a.target[4:] != b.target[4:]
    assert wn.context_panels(0, 6, 1) == {}
    # the efference advance: the brain hears SPEAK alone, its copy holds the window it said
    brain = wn.make_brain(4, 0, module=16, advance="efference")
    assert brain.efference is not None
    x = wn.heard_window([3, 12, 7, 17], "efference")
    assert x[gr.C["SPEAK"]] == 1.0 and not x[:gr.ACTIONS].any()
    said = wn.speak(brain, rows[:4], 4, teach=False, advance="efference")
    assert len(said) == 4
    with pytest.raises(ValueError):
        wn.make_brain(4, 0, module=16, advance="hearsay")


def test_settings_name_a_lifes_backend():
    assert gr.make_brain("recipe", 0, {"backend": "cpu"}, module=16).brain.backend == "cpu"


def test_the_cost_of_a_pass_is_read_per_learner(tmp_path):
    pytest.importorskip("torch")
    _load("controls")
    cost = _load("cost")
    result = cost.run(0, 2, every=1)
    assert set(result["seconds"]) == set(cost.LEARNERS)
    assert all(seconds > 0 for seconds in result["seconds"].values())
    assert result["work"]["plain"]["acts"] == result["work"]["compiled"]["acts"] > 0
    assert [mark["batches"] for mark in result["marks"]] == [1, 2]
    assert result["against_transformer"]["transformer"] == 1.0
    assert cost.main(["--batches", "1", "--learners", "plain,compiled",
                      "--out", str(tmp_path / "cost.json")]) == 0
    assert "against_transformer" not in (tmp_path / "cost.json").read_text()
    found = cost.parts("compiled", 0, 2)
    assert found["parts"] and abs(sum(found["parts"].values()) - found["wall"]) < 1e-6
    assert set(found["parts"]) <= {part for _, _, part in cost.PARTS} | {cost.OUTSIDE}
    assert found["parts"]["kernels: a moment in one call"] > 0
    assert cost.parts("plain", 0, 2)["parts"]["kernels: the solves"] > 0
    import cadence.fused as fused

    assert not hasattr(fused.fused_fast_solve, "__wrapped__")  # restored
