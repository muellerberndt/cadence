"""Contracts of the scaling evaluation: the family's parameters, its grid and its summary."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cadence import Brain

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scaling = _load("scaling")
runs, tb = scaling.runs, scaling.tb
sb = runs._sibling("sequence_binding")


def test_the_token_vocabulary_holds_eight_facts_and_keeps_the_held_out_rule() -> None:
    names = scaling.TOKEN_NAMES
    held = tb.holdout(2, names)
    assert held.shape == (names - 1, tb.VALUES)
    assert (held.sum(axis=0) == 1).all() and (held.sum(axis=1) == 1).all()
    np.testing.assert_array_equal(tb.holdout(2), tb.holdout(2, tb.NAMES))
    trials = tb.draw(np.random.default_rng(0), 64, held, facts=8)
    assert all(sorted(row) == list(range(8)) for row in trials["names"])
    assert not held[trials["names"], trials["values"]].any()
    seq = tb.events(trials, 32, names)
    assert len(seq) == 8 + 32 + 1 and seq[0].shape[1] == tb.input_width(names) == 20
    rows = np.arange(64)
    fact = np.stack([seq[trials["query"][i]][i] for i in rows])
    shared = (seq[-1] > 0) & (fact > 0)
    assert (shared.sum(axis=1) == 1).all()
    value = tb.coordinate("value", 0, names) + tb.answers(trials)
    assert fact[rows, value].all() and tb.coordinate("value", 0, names) == 3 + names
    new = tb.draw(np.random.default_rng(1), 16, held, facts=4, new_name=True)
    assert (new["names"][np.arange(16), new["query"]] == names - 1).all()


def test_the_succession_vocabulary_holds_sixteen_tokens() -> None:
    tokens = scaling.SUCCESSION_TOKENS
    held = sb.holdout(1, tokens)
    assert held.shape == (tokens - 1, tokens - 1) and not np.diag(held).any()
    assert (held.sum(axis=0) == 1).all() and (held.sum(axis=1) == 1).all()
    np.testing.assert_array_equal(sb.holdout(1), sb.holdout(1, sb.TOKENS))
    cued = sb.draw(np.random.default_rng(2), 32, held, length=16, cue_held=True)
    s, c, rows = cued["sequences"], cued["cue"], np.arange(32)
    assert held[s[rows, c], s[rows, c + 1]].all()
    assert (held[s[:, :-1], s[:, 1:]].sum(axis=1) == 1).all()
    assert sb.events(cued, tokens)[0].shape[1] == sb.input_width(tokens) == 20


def test_the_level_panels_at_the_declared_values_are_the_instruments_first_two(
    monkeypatch,
) -> None:
    """At two facts, one neutral event and five name tokens the family reads the declared
    task's training and held-out panels, trial for trial."""
    read: list[dict[str, np.ndarray]] = []

    def capture(brain, trials, condition, neutral, donor=None, names=tb.NAMES):
        read.append(trials)
        return tb.answers(trials)

    monkeypatch.setattr(runs, "token_read", capture)
    held = tb.holdout(4)
    panels = runs.token_level_panels(SimpleNamespace(index=None), 4, held, tb.FACTS,
                                     tb.NEUTRAL, tb.NAMES)
    assert panels == {"train": {"intact": 1.0},
                      "held_out": {"intact": 1.0, "trace_erased": 1.0}}
    declared = tb.panels(4, held)
    for got, name in zip((read[0], read[1]), ("train", "held_out"), strict=True):
        for key in ("names", "values", "query"):
            np.testing.assert_array_equal(got[key], declared[name][0][key])


def test_without_the_memory_the_brain_is_the_composed_one_at_every_width() -> None:
    for width in (64, 256):
        built = runs.built({"seed": 3, "genes": None, "width": width}, inputs=20)
        composed = Brain.compose(20, tb.VALUES, modules=(width,), lateral=tb.LATERAL, seed=3,
                                 recall=False)
        assert built.index is None and "recall" not in built.connectome.populations
        np.testing.assert_array_equal(built.connectome.pre, composed.connectome.pre)
        np.testing.assert_array_equal(built.connectome.post, composed.connectome.post)
        np.testing.assert_array_equal(built.brain.efficacy, composed.brain.efficacy)


def test_a_life_without_the_memory_reads_its_trace_and_no_slots() -> None:
    spec = scaling.job("token", {"facts": 4, "neutral": 8}, "none", 64, None, 0, (1,))
    row = runs.life(spec)
    held = row["readings"][0]["panels"]["held_out"]
    assert set(held) == {"intact", "trace_erased"}
    assert set(row["readings"][0]["panels"]["train"]) == {"intact"}
    assert row["readings"][0]["read_cpu"] > 0.0


def test_the_full_grid_and_the_trimmed_grid() -> None:
    full = scaling.full_grid()
    assert len(full) == 486 and len({j["name"] for j in full}) == len(full)
    for spec in full:
        if spec["memory"] == "none":
            assert spec["genes"] is None and spec["capacity"] is None
        else:
            assert spec["genes"] == runs.genes_for(spec["task"], "horizon",
                                                   {"capacity": spec["capacity"]})
            assert spec["genes"]["recency"] is None
    long_delay = {j["capacity"] for j in full if j["family"] == "token" and j["neutral"] == 32
                  and j["memory"] == "recall"}
    assert long_delay == set(scaling.CAPACITIES)
    settings = {(j["family"], j["key"], j["memory"], j["width"], j["capacity"], j["seed"])
                for j in full}
    for spec in scaling.trimmed_grid():
        assert (spec["family"], spec["key"], spec["memory"], spec["width"], spec["capacity"],
                spec["seed"]) in settings
        assert tuple(spec["budgets"]) == scaling.BUDGETS[: len(spec["budgets"])]


def _receipt(name: str, seed: int, memory: str, scores: list[float],
             budgets: tuple[int, ...] = scaling.BUDGETS) -> dict:
    spec = scaling.job("succession", {"length": 4}, memory, 64,
                       None if memory == "none" else 64, seed, budgets)
    readings = [{"rounds": b, "panels": {"held_out": {"intact": s, "trace_erased": 0.1},
                                         "train": {"intact": 1.0}}}
                for b, s in zip(budgets, scores, strict=False)]
    return {"job": dict(spec, name=name), "readings": readings}


def test_a_life_ends_at_two_readings_at_mastery_and_the_summary_carries_it() -> None:
    early = _receipt("a", 0, "recall", [0.96, 0.97])
    assert runs.ended(early["job"], early["readings"])
    assert not runs.ended(dict(early["job"], master=None), early["readings"])
    reading, carried = scaling.at_budget(early, 16384)
    assert carried and reading["rounds"] == 1024
    late = _receipt("b", 1, "recall", [0.5, 0.9, 0.96, 0.99])
    cell = scaling.cell_summary({0: early, 1: late})
    assert cell["first_mastery"] == 4096 and cell["first_by_founder"] == {0: 256, 1: 4096}
    last = cell["budgets"][-1]
    assert last["carried"] == 1 and abs(last["held"] - 0.98) < 1e-12
    unfinished = _receipt("c", 2, "recall", [0.5, 0.6])
    assert scaling.at_budget(unfinished, 4096) == (None, False)


def test_the_supplement_reads_the_linear_charge_at_the_declared_budgets() -> None:
    """Amendment 1: the index arm's linear charge at a capacity that holds the span, on the
    lives the amendment names, at the budgets the declared grid reads, under names of their
    own."""
    supplement, trimmed = scaling.supplement_grid(), scaling.trimmed_grid()
    assert len(supplement) == 42
    assert not {j["name"] for j in supplement} & {j["name"] for j in trimmed}
    declared = {(j["family"], j["key"], j["width"], j["capacity"]): j["budgets"]
                for j in trimmed if j["memory"] == "recall"}
    for spec in supplement:
        assert spec["memory"] == "recall_linear" and spec["arm"] == "index"
        assert spec["genes"] == runs.genes_for(spec["task"], "index",
                                               {"capacity": spec["capacity"]})
        assert spec["genes"]["recency"] == 0.25 and spec["capacity"] >= spec["span"]
        assert spec["budgets"] == declared[(spec["family"], spec["key"], spec["width"],
                                            spec["capacity"])]
    families = {f: sum(j["family"] == f for j in supplement) for f in scaling.FAMILIES}
    assert families == {"token": 18, "succession": 0, "named": 0, "continuing": 6,
                        "camera": 18}
    assert {(j["key"], j["capacity"]) for j in supplement if j["family"] == "continuing"} == {
        ("k8", 64), ("k32", 256)}


def test_the_summary_sets_the_three_memories_side_by_side(tmp_path) -> None:
    lives = tmp_path / "lives"
    lives.mkdir()
    for memory, scores in (("none", [0.1, 0.1]), ("recall", [0.2, 0.3]),
                           ("recall_linear", [0.96, 0.97])):
        for seed in (0, 1):
            receipt = _receipt(f"{memory}-{seed}", seed, memory, scores)
            receipt["job"]["name"] = f"{memory}-{seed}"
            (lives / f"{memory}-{seed}.json").write_text(json.dumps(receipt))
    body = scaling.summarize(tmp_path)
    [side] = body["side_by_side"]
    assert side["cells"]["recall_linear"]["first_mastery"] == 256
    assert side["cells"]["recall"]["first_mastery"] is None
    found = {p["comparison"]: {r["rounds"]: r["difference"] for r in p["budgets"]}
             for p in body["paired"]}
    assert set(found) == {"recall less none", "recall_linear less none",
                          "recall_linear less recall"}
    assert abs(found["recall_linear less recall"][1024] - 0.67) < 1e-12
    assert found["recall_linear less recall"][16384] is None  # the recall read no further
    text = scaling.markdown(body)
    assert "| 0.10 / 0.20 / 0.96 | 0.10 / 0.30 / 0.97 | - / - / 0.97* |" in text
