"""The attention ablations: each restriction does what it says on the control transformer."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


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
ct = _load("controls")
ab = _load("ablations")


@pytest.fixture(scope="module")
def rows():
    return gr.panels(0, 1)["clause"][0]


def _moved(learner, rows, steps: int = 2) -> dict[str, bool]:
    before = {n: p.detach().clone() for n, p in learner.net.named_parameters()}
    for _ in range(steps):
        learner.update(rows)
    return {n: not torch.equal(p, before[n]) for n, p in learner.net.named_parameters()}


def test_the_control_rows_are_the_controls_themselves(rows):
    x = _tensor(rows)
    with torch.no_grad():
        assert torch.equal(ab.make("ctl", 8, 3)(x), ct.build("transformer", 8, 3)(x))
        assert torch.equal(ab.make("lstm", 8, 3)(x), ct.build("lstm", 8, 3)(x))
    assert ab.matched_width("ctl", ab.FOUNDER_PARAMETERS) == 36
    assert ab.matched_width("lstm", ab.FOUNDER_PARAMETERS) == 60


def test_the_unrestricted_row_is_the_control_to_the_parameter():
    for width in (8, 36):
        assert ab.count(ab.make("none", width, 0)) == ct.count(ct.build("transformer", width, 0))
    assert ab.ROWS["AE"] == {**ab.ROWS["A"], **ab.ROWS["E"]}


@pytest.mark.parametrize("row", list(ab.ROWS))
def test_every_row_is_rematched_to_the_brain(row):
    width = ab.matched_width(row, ab.FOUNDER_PARAMETERS)
    net = ab.make(row, width, 0)
    assert abs(ab.count(net) - ab.FOUNDER_PARAMETERS) < 1500
    assert ab.count(net, trainable=False) >= ab.count(net)


def test_frozen_and_detached_keys_and_values_never_move(rows):
    for row in ("A", "E"):
        learner = ab.Ablated(row, 8, 0, 0.003)
        moved = _moved(learner, rows)
        frozen = [n for n in moved if ".attn.k." in n or ".attn.v." in n]
        assert len(frozen) == 8 and not any(moved[n] for n in frozen)
        assert moved["inp.weight"] and all(moved[n] for n in moved if ".attn.q." in n)
    # the LSTM's analogue: the input-to-hidden weights stay, the recurrent ones learn
    learner = ab.Ablated("lstmA", 8, 0, 0.003)
    moved = _moved(learner, rows)
    assert not moved["cell.weight_ih_l0"] and moved["cell.weight_hh_l0"]


def test_the_single_read_has_no_second_attention(rows):
    for row in ("B", "Bp", "AB", "BC", "all"):
        net = ab.make(row, 8, 0)
        assert net.layers[0].attn is not None and net.layers[1].attn is None


def test_the_recurrent_read_at_a_zero_cortex_map_is_the_held_read(rows):
    x, _ = ct.encode(rows)
    x = torch.from_numpy(x)
    recurrent = ab.make("Bp", 8, 0)
    single = ab.make("B", 8, 1)
    state = {k: v for k, v in recurrent.state_dict().items() if not k.endswith("attn.c.weight")}
    single.load_state_dict(state)
    with torch.no_grad():
        recurrent.layers[0].attn.c.weight.zero_()
        assert torch.allclose(recurrent(x), single(x), atol=1e-5)
        recurrent.layers[0].attn.c.weight.normal_()
        assert not torch.allclose(recurrent(x), single(x), atol=1e-3)
    learner = ab.Ablated("Bp", 8, 0, 0.003)
    assert _moved(learner, rows)["layers.0.attn.c.weight"]


def test_raw_values_reach_the_logits_through_the_identity(rows):
    net = ab.make("C", 8, 0)
    attn = net.layers[0].attn
    assert attn.v is None and attn.o is None and attn.r is not None
    x, _ = ct.encode(rows)
    x = torch.from_numpy(x)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()  # the learned paths silent: the logits are the copied word mixture
        logits = net(x)
        read = torch.stack([p @ x[:, None] for p in net.attention]).sum(dim=(0, 2))
    assert torch.allclose(logits, read[..., : gr.ACTIONS], atol=1e-5)


def test_heads_without_a_mix_and_the_competing_heads(rows):
    x, _ = ct.encode(rows)
    x = torch.from_numpy(x)
    assert ab.make("D1", 8, 0).layers[0].attn.o is not None
    for row in ("D2", "H1b", "H2b"):
        net = ab.make(row, 8, 0)
        assert net.layers[0].attn.o is None
    shared = ab.make("H2b", 8, 0).layers[0].attn
    assert len(shared.q) == 2 and shared.k.out_features == 8 and shared.v.out_features == 8
    net = ab.make("H2b", 8, 0)
    with torch.no_grad():
        net(x)
    for p in net.attention:  # (batch, heads, positions, slots)
        assert p.shape[1] == 2
        assert (p.sum(dim=1) <= 1.0 + 1e-5).all()  # a slot is read in full at most once
        assert torch.allclose(p[:, 0].sum(-1), torch.ones_like(p[:, 0, :, 0]))
        taken = p[:, 1] <= (1.0 - p[:, 0]) + 1e-5
        assert taken.all()
        # the second head keeps its mass where slots are left; at the first position the
        # first head took the only slot whole and the second reads nothing
        assert (p[:, 1, 1:].sum(-1) > 0.5).all() and (p[:, 1, 0].sum(-1) == 0).all()


def _tensor(rows):
    return torch.from_numpy(ct.encode(rows)[0])


def _inputs(net, x) -> dict:
    """What each layer's query, key and value maps read on one forward."""
    seen, handles = {}, []
    for i, block in enumerate(net.layers):
        if block.attn is None:
            continue
        for name in ("q", "k", "v"):
            module = getattr(block.attn, name)
            if module is not None:
                handles.append(module.register_forward_hook(
                    lambda _m, args, _o, key=(i, name): seen.__setitem__(key, args[0].detach())))
    with torch.no_grad():
        net(x)
    for handle in handles:
        handle.remove()
    return seen


def test_the_rebuilt_control_is_the_controls_function_at_its_weights(rows):
    width = 8
    control = ct.build("transformer", width, 0)
    rebuilt = ab.make("none", width, 0)
    x = _tensor(rows)
    with torch.no_grad():
        assert not torch.allclose(rebuilt(x), control(x), atol=1e-3)  # another initialization
        rebuilt.inp.load_state_dict(control.inp.state_dict())
        rebuilt.out.load_state_dict(control.out.state_dict())
        for mine, theirs in zip(rebuilt.layers, control.body.layers, strict=True):
            attention = theirs.self_attn
            weights = attention.in_proj_weight.chunk(3)
            biases = attention.in_proj_bias.chunk(3)
            for name, weight, bias in zip("qkv", weights, biases, strict=True):
                getattr(mine.attn, name).weight.copy_(weight)
                getattr(mine.attn, name).bias.copy_(bias)
            mine.attn.o.load_state_dict(attention.out_proj.state_dict())
            mine.norm1.load_state_dict(theirs.norm1.state_dict())
            mine.norm2.load_state_dict(theirs.norm2.state_dict())
            mine.mlp[0].load_state_dict(theirs.linear1.state_dict())
            mine.mlp[2].load_state_dict(theirs.linear2.state_dict())
        assert torch.allclose(rebuilt(x), control(x), atol=1e-5)


# Each layer's maps: (does the input move with the embedding, with the first read).
_E, _CTX, _RAW = (True, False), (True, True), (False, False)
_SOURCES = {
    "none": {"q0": _E, "k0": _E, "q1": _CTX, "k1": _CTX},
    "Ar": {"q0": _E, "k0": _RAW, "q1": _CTX, "k1": _RAW},
    "S1": {"q0": _E, "k0": _E, "q1": _CTX, "k1": _RAW},
    "S2": {"q0": _E, "k0": _E, "q1": _E, "k1": _CTX},
    "S3": {"q0": _E, "k0": _E, "q1": _E, "k1": _E},
    "S3r": {"q0": _E, "k0": _RAW, "q1": _E, "k1": _RAW},
}


@pytest.mark.parametrize("row", list(_SOURCES))
def test_every_map_reads_what_its_row_says(rows, row):
    x = _tensor(rows)
    base = _inputs(ab.make(row, 8, 0), x)
    moved_embedding = ab.make(row, 8, 0)
    moved_read = ab.make(row, 8, 0)
    with torch.no_grad():
        moved_embedding.inp.weight.add_(1.0)
        moved_read.layers[0].attn.o.weight.add_(1.0)
    with_embedding, with_read = _inputs(moved_embedding, x), _inputs(moved_read, x)
    for key, value in base.items():
        layer, name = key
        expected = _SOURCES[row][f"{'q' if name == 'q' else 'k'}{layer}"]
        reading = (not torch.allclose(value, with_embedding[key]),
                   not torch.allclose(value, with_read[key]))
        assert reading == expected, (row, key)


@pytest.mark.parametrize("row", ["Ar", "S1", "S3r", "allr", "allrn"])
def test_the_raw_keys_are_a_fixed_code_of_the_moment_and_its_position(rows, row):
    x = _tensor(rows)
    net = ab.make(row, 8, 0)
    expected = net.code(x) + net.position[: x.shape[1]]
    for key, value in _inputs(net, x).items():
        if key[1] != "q" and key[0] in ab.ROWS[row]["raw_kv"]:
            assert torch.equal(value, expected.detach()), key
    learner = ab.Ablated(row, 8, 0, 0.003)
    moved = _moved(learner, rows)
    assert not moved["code.weight"]
    fixed = [n for n in moved if any(f"layers.{i}.attn.{m}." in n
                                     for i in ab.ROWS[row]["raw_kv"] for m in "kv")]
    assert fixed and not any(moved[n] for n in fixed)
    assert moved["inp.weight"] and moved["layers.0.attn.q.weight"]
    assert not any(p.requires_grad for p in net.code.parameters())


def test_the_code_balances_the_moment_and_its_position():
    net = ab.make("Ar", 64, 0)
    columns = net.code.weight.norm(dim=0) ** 2
    positions = net.position[:32].norm(dim=1) ** 2
    assert 0.5 < float(columns.mean() / positions.mean()) < 2.0


def test_the_learned_rest_of_a_raw_row_is_its_frozen_rows(rows):
    x = _tensor(rows)
    a, ar = ab.make("A", 8, 0).state_dict(), ab.make("Ar", 8, 0).state_dict()
    assert all(torch.equal(a[name], ar[name]) for name in a)  # the code is drawn last
    assert set(ar) - set(a) == {"code.weight"}
    assert not torch.allclose(ab.make("A", 8, 0)(x), ab.make("Ar", 8, 0)(x), atol=1e-3)


def test_the_brains_structure_in_the_transformer(rows):
    x = _tensor(rows)
    for row in ("allr", "allrn"):
        net = ab.make(row, 8, 0)
        attn = net.layers[0].attn
        assert net.layers[1].attn is None and attn.v is None and attn.o is None
        assert attn.r is not None and attn.c is not None
        assert set(_inputs(net, x)) == {(0, "q"), (0, "k")}
        with torch.no_grad():
            net.out.weight.zero_()
            net.out.bias.zero_()
            logits = net(x)  # the logits' learned path silent: only the identity is left
        assert net.attention[0][0].shape[1] == 1  # one head
        if row == "allr":
            assert logits.abs().sum() > 0
            assert (logits[:, 0] == x[:, 0, : gr.ACTIONS]).all()  # the only slot, read whole
        else:
            assert (logits == 0).all()


@pytest.mark.parametrize("row", list(ab.ROWS))
def test_no_row_hears_a_later_moment(rows, row):
    x = _tensor(rows)
    cut = x.shape[1] // 2
    later = x.clone()
    later[:, cut:] = torch.flip(x[:, cut:], dims=[0]) + 1.0
    net = ab.make(row, 8, 0)
    with torch.no_grad():
        before, after = net(x), net(later)
    assert torch.allclose(before[:, :cut], after[:, :cut], atol=1e-6)
    assert not torch.allclose(before[:, cut:], after[:, cut:], atol=1e-3)


def _earlier_changed(x):
    """The same moments with every moment but the last replaced."""
    changed = x.clone()
    changed[:, :-1] = torch.flip(x[:, :-1], dims=[1]) + 1.0
    return changed


@pytest.mark.parametrize("row", ["allr", "allrn", "all", "Bp", "B"])
def test_the_single_read_is_the_only_way_a_moment_reaches_a_later_one(rows, row):
    x = _tensor(rows)
    net = ab.make(row, 8, 0)
    seen = {}
    net.out.register_forward_hook(lambda _m, args, _o: seen.update(h=args[0].detach()))
    with torch.no_grad():
        logits = net(x)
        hears = seen["h"][:, -1].clone()
        net(_earlier_changed(x))
        assert not torch.allclose(seen["h"][:, -1], hears, atol=1e-3)  # the read carries them
        attn = net.layers[0].attn
        for parameter in (attn.r or attn.o).parameters():
            parameter.zero_()  # the read silenced: nothing of it enters the residual
        silent = net(x)
        hears = seen["h"][:, -1].clone()
        changed = net(_earlier_changed(x))
    assert torch.equal(seen["h"][:, -1], hears)  # the residual holds the present moment alone
    if row == "allr" or row == "all":
        assert not torch.equal(changed[:, -1], silent[:, -1])  # the identity still copies a read
    else:
        assert torch.equal(changed[:, -1], silent[:, -1])
    assert not torch.equal(logits, silent)


def test_the_literal_copy_row_has_no_learned_path_from_the_read(rows):
    x = _tensor(rows)
    net = ab.make("Cs", 8, 0)
    assert all(b.attn.r is None and b.attn.v is None and b.attn.o is None for b in net.layers)
    seen = {}
    handle = net.out.register_forward_hook(lambda _m, args, out: seen.update(h=args[0], out=out))
    with torch.no_grad():
        before = net(x)
        h, out = seen["h"].clone(), seen["out"].clone()
        net.layers[0].attn.q.weight.add_(1.0)
        after = net(x)
    handle.remove()
    assert torch.equal(seen["h"], h) and torch.equal(seen["out"], out)  # the residual never hears
    assert not torch.equal(before, after)  # the logits do, through the identity


def test_an_initialization_apart_from_the_founders_split_and_lessons():
    spec = {"row": "S3", "mode": "online", "founder": 0, "phases": [["g1", 2]], "width": 8,
            "panel_batches": 1, "panels": ["clause"]}
    first, second = ab.run_life(spec), ab.run_life({**spec, "init_seed": 5})
    assert first["init_seed"] == 0 and second["init_seed"] == 5
    assert first["readings"][0]["work"]["tokens"] == second["readings"][0]["work"]["tokens"]
    assert first["readings"][0]["loss_last"] != second["readings"][0]["loss_last"]


def test_the_spread_over_lives_and_the_resume(tmp_path):
    def life(row, founder, seed, exact, mode="offline"):
        return {"row": row, "mode": mode, "founder": founder, "init_seed": seed,
                "readings": [{"panels": {"clause_held": {"exact": exact}}}]}

    out = tmp_path / "s.jsonl"
    out.write_text("\n".join(json.dumps(x) for x in (
        life("none", 0, 0, 1.0), life("none", 0, 1, 0.77), life("none", 1, 1, 0.96))) + "\n")
    line = [s for s in ab.spread([out]).splitlines() if s.startswith("| none")][0]
    assert line == "| none | offline | 3 | 2 of 3 | 0.91 | 0.77 | 1.00 0.77 0.96 |"
    specs = [{"row": "none", "mode": "offline", "founder": 0},
             {"row": "none", "mode": "offline", "founder": 0, "init_seed": 1},
             {"row": "none", "mode": "offline", "founder": 0, "init_seed": 2},
             {"row": "none", "mode": "online", "founder": 0}]
    assert ab.pending(specs, out) == specs[2:]
    assert ab.pending(specs, tmp_path / "absent.jsonl") == specs


def test_a_reader_that_ignores_the_crossing_says_nine_of_sixteen():
    rng = np.random.default_rng(9)
    rows = ab.crossing(rng, 20000)
    first = np.mean([np.argmax(r.pre[3]) == r.target[0] for r in rows])
    assert abs(first - 9 / 16) < 0.015


def test_the_recipe_brain_on_the_crossing():
    pytest.importorskip("numba")
    life = ab.run_crossing_brain({"founder": 0, "batches": 2, "panel_batches": 1, "module": 16})
    panel = life["readings"][0]["panels"]["crossing"]
    assert life["row"] == "recipe" and life["mode"] == "crossing"
    assert life["readings"][0]["work"]["lessons"] == 4  # two words, two batches
    assert 0.0 <= panel["exact"] <= 1.0 and set(panel["conditions"]) <= {"c0k0", "c0k1", "c1k0",
                                                                          "c1k1"}


def test_the_crossing_asks_for_the_fact_the_context_and_question_cross(tmp_path):
    rng = np.random.default_rng(228)
    rows = ab.crossing(rng, 64)
    seen = set()
    for r in rows:
        c, k = r.meta["c"], r.meta["k"]
        seen.add((c, k))
        assert len(r.pre) == 10 and np.argmax(r.pre[0]) == gr.OPENS[c]
        assert np.argmax(r.pre[9]) == gr.MATCH[gr.OPENS[k]]
        values = [int(np.argmax(r.pre[3])), int(np.argmax(r.pre[7]))]
        assert all(v < 2 * gr.NOUNS for v in values) and r.target == [values[c ^ k], gr.END]
        assert np.argmax(r.pre[1]) != np.argmax(r.pre[5])  # two subjects
    assert len(seen) == 4
    life = ab.run_crossing({"row": "B", "founder": 0, "batches": 2, "width": 8,
                            "panel_batches": 1})
    panel = life["readings"][0]["panels"]["crossing"]
    assert life["mode"] == "crossing" and set(panel["conditions"]) <= {"c0k0", "c0k1", "c1k0",
                                                                        "c1k1"}
    assert 0.0 <= panel["exact"] <= 1.0 and "next_word" in panel
    out = tmp_path / "x.jsonl"
    out.write_text(json.dumps(life) + "\n")
    assert ab.table(out).splitlines()[0].startswith("| row | mode | width | learned (all) | "
                                                     "founders | epochs | seconds | exact |")


def test_a_short_life_and_its_table(tmp_path):
    spec = {"row": "C", "mode": "online", "founder": 0, "phases": [["g1", 2]], "width": 8,
            "panel_batches": 1, "panels": ["clause", "path"]}
    life = ab.run_life(spec)
    assert life["parameters"] == ab.count(ab.make("C", 8, 0)) and life["width"] == 8
    panels = life["readings"][0]["panels"]
    assert set(panels) == {"clause", "path"} and "L0" in panels["clause"]["conditions"]
    assert life["readings"][0]["work"]["updates"] == 2
    out = tmp_path / "a.jsonl"
    out.write_text(json.dumps(life) + "\n")
    text = ab.table(out)
    assert text.splitlines()[2].startswith("| C | online | 8 |")
    assert ab.table(out, per_founder=True) == text
    assert ab._fmt([0.5, 1.0]) == ".75 (.50-1.00)" and ab._fmt([None]) == ""
    assert np.isclose(float(ab._fmt([0.25])), 0.25)
