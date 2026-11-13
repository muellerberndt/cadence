"""The same facts held another way (cadence#310, C7): a transformer that holds the told facts
in its window, and an index of the facts' words read by the same transformer. Scored controls
beside ``brain_records.py``'s brain, never part of its answer path.

**The facts** are the brain's chamber's (``brain_records.World``): a told fact is an act, an
agent and a patient, and a question names the act and the agent. A reading tells N facts about
pairs of the reading's pool, tells a tenth of them again with another patient, then asks
``questions`` about unrevised facts and as many about revised ones. The draws are
``brain_records.read_at``'s, from its seeds, so a control reads the facts the brain read.

**The window** (``context``). A causal transformer (``layers`` layers, ``heads`` heads,
pre-norm, a feed-forward of four times the width) whose width brings its parameters nearest the
brain's (``target``). A told fact is one token: the sum of its act's, its agent's and its
patient's embeddings and a sinusoidal code of its place in the stream. A question is one more
token, a query embedding in the patient's place, and the answer is read at it as one of the
eight patients. One token a fact favours the control: the brain hears a fact as four moments.

It is taught by backpropagation (Adam, cross-entropy at the questions) on windows drawn anew
for every step from the nursery's pairs, the pairs the brain's lessons use: ``taught`` facts at
most, a tenth told again after them, then ``asked`` questions, each of which sees every fact
and no other question. The windows grow from 16 facts to ``taught`` over the first half of the
budget of ``minutes``; a window's first place is drawn anywhere in ``PLACES``.

It is read with the facts in its window: the N told facts and then the revisions enter in told
order, each token attending to every token before it, and every layer's keys and values are
kept (the window's bytes). A question is one token read against what is kept. The row holds
the seconds the window took to write, ms per written token at the window's end, ms per
question, and the multiply-accumulates of both.

**The index** (``retrieval``). Every told fact is listed under its act and under its agent. A
question fetches the ``fetched`` facts that hold both of its words, then those that hold its
agent, the latest first, and the same transformer reads them in told order before the question.
The row holds the index's bytes (two postings and three words a fact) and whether the fact's
current telling was among the fetched (``found``).

    PYTHONPATH=src python benchmarks/memory_scale/window.py --plan plan.json --out rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import brain_records as br  # noqa: E402

PLACES = 2_000_000  # a taught window's first place is drawn below this
PART = 512  # tokens that enter a read window at once
RATE, CLIP = 0.001, 1.0


def place_code(places: Any, width: int) -> Any:
    """The sinusoidal code of stream places, ``(..., width)``: wavelengths from 4 to ``8 *
    PLACES`` places."""
    import torch

    half = width // 2
    rates = torch.exp(-torch.linspace(math.log(4.0), math.log(8.0 * PLACES), half))
    angles = places.to(torch.float32).unsqueeze(-1) * rates * (2 * math.pi)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def build(acts: int, agents: int, width: int, layers: int, heads: int, seed: int) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as fn

    torch.manual_seed(int(seed))

    class Block(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.first, self.second = nn.LayerNorm(width), nn.LayerNorm(width)
            self.qkv, self.out = nn.Linear(width, 3 * width), nn.Linear(width, width)
            self.up, self.down = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

        def forward(self, x: Any, mask: Any, kept: Any = None) -> tuple[Any, Any, Any]:
            """``x`` attends to ``kept`` (a layer's keys and values so far) and to itself
            under ``mask``; the new tokens' keys and values are returned beside it."""
            rows, tokens, _ = x.shape
            q, k, v = self.qkv(self.first(x)).view(
                rows, tokens, 3, heads, width // heads).permute(2, 0, 3, 1, 4)
            keys = k if kept is None else torch.cat([kept[0], k], dim=2)
            values = v if kept is None else torch.cat([kept[1], v], dim=2)
            held = fn.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
            x = x + self.out(held.transpose(1, 2).reshape(rows, tokens, width))
            return x + self.down(fn.gelu(self.up(self.second(x)))), k, v

    class Reader(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.act, self.agent = nn.Embedding(acts, width), nn.Embedding(agents, width)
            self.patient = nn.Embedding(br.PATIENTS + 1, width)  # the last one asks
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.last = nn.LayerNorm(width)
            self.say = nn.Linear(width, br.PATIENTS)
            self.width, self.layers, self.heads = width, layers, heads

        def tokens(self, act: Any, agent: Any, patient: Any, place: Any) -> Any:
            return self.act(act) + self.agent(agent) + self.patient(patient) \
                + place_code(place, width)

        def forward(self, x: Any, mask: Any) -> Any:
            for block in self.blocks:
                x, _, _ = block(x, mask)
            return self.say(self.last(x))

    return Reader()


def parameters(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def width_for(target: int, acts: int, agents: int, layers: int, heads: int) -> int:
    """The width, a multiple of ``2 * heads``, whose reader holds the parameters nearest
    ``target``."""
    best, gap = 2 * heads, float("inf")
    for width in range(2 * heads, 2048, 2 * heads):
        blocks = layers * (12 * width * width + 13 * width)
        held = (acts + agents + br.PATIENTS + 1) * width + blocks + 2 * width \
            + br.PATIENTS * (width + 1)
        if abs(held - target) < gap:
            best, gap = width, abs(held - target)
    return best


def taught_mask(facts: int, asked: int) -> Any:
    """Who sees whom in a taught window: a fact every fact before it and itself, a question
    every fact and itself."""
    import torch

    total = facts + asked
    mask = torch.zeros(total, total, dtype=torch.bool)
    mask[:facts, :facts] = torch.tril(torch.ones(facts, facts, dtype=torch.bool))
    mask[facts:, :facts] = True
    mask[facts:, facts:] = torch.eye(asked, dtype=torch.bool)
    return mask


def lesson(rng: np.random.Generator, world: br.World, facts: int, asked: int,
           rows: int) -> tuple[np.ndarray, ...]:
    """``rows`` taught windows over the nursery's pairs: ``facts`` told facts, a tenth told
    again after them with another patient, then ``asked`` questions. The acts, agents,
    patients and places of every token, and each question's answer."""
    lex = world.lexicon
    again = max(facts // 10, 1)
    told = facts + again
    act = np.zeros((rows, told + asked), dtype=np.int64)
    agent, patient, place = act.copy(), act.copy(), act.copy()
    answer = np.zeros((rows, asked), dtype=np.int64)
    for r in range(rows):
        about = rng.choice(world.nursery, size=facts, replace=False)
        first = rng.integers(br.PATIENTS, size=facts)
        current = first.copy()
        revised = rng.choice(facts, size=again, replace=False)
        current[revised] = (first[revised] + rng.integers(1, br.PATIENTS, size=again)) \
            % br.PATIENTS
        order = np.concatenate([np.arange(facts), revised])
        said = np.concatenate([first, current[revised]])
        ask = np.concatenate([rng.choice(revised, size=min(asked // 2, again)),
                              rng.integers(facts, size=asked)])[:asked]
        act[r, :told], agent[r, :told] = about[order] % lex.verbs, about[order] // lex.verbs
        patient[r, :told] = said
        act[r, told:], agent[r, told:] = about[ask] % lex.verbs, about[ask] // lex.verbs
        patient[r, told:] = br.PATIENTS
        start = int(rng.integers(PLACES))
        place[r, :told] = start + np.arange(told)
        place[r, told:] = start + told
        answer[r] = current[ask]
    return act, agent, patient, place, answer


def teach(net: Any, world: br.World, founder: int, spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from torch.nn import functional as fn

    taught, asked = int(spec.get("taught", 256)), int(spec.get("asked", 16))
    rows, budget = int(spec.get("rows", 16)), 60.0 * float(spec.get("minutes", 20))
    optimizer = torch.optim.Adam(net.parameters(), lr=float(spec.get("rate", RATE)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, 310, 9, 1]))
    started, steps, seen, right = time.perf_counter(), 0, 0, []
    sizes = [16]
    while sizes[-1] < taught:
        sizes.append(min(4 * sizes[-1], taught))
    while True:
        spent = time.perf_counter() - started
        if spent >= budget or steps >= int(spec.get("steps", 10 ** 9)):
            break
        # the windows grow over the first half of the budget, then stay at ``taught``
        facts = sizes[min(int(2 * len(sizes) * spent / budget), len(sizes) - 1)]
        many = max(rows, min(64, rows * taught // facts))
        act, agent, patient, place, answer = (
            torch.from_numpy(a) for a in lesson(rng, world, facts, asked, many))
        told = act.shape[1] - asked
        said = net(net.tokens(act, agent, patient, place), taught_mask(told, asked))[:, told:]
        loss = fn.cross_entropy(said.reshape(-1, br.PATIENTS), answer.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        optimizer.step()
        steps += 1
        seen += many * asked
        right.append(float((said.argmax(dim=-1) == answer).float().mean()))
    return {"steps": steps, "questions_taught": seen, "taught": taught,
            "train_seconds": round(time.perf_counter() - started, 1),
            "train_right": round(float(np.mean(right[-50:])), 4) if right else None,
            "loss": round(float(loss.detach()), 5) if steps else None}


def reading(world: br.World, founder: int, n: int, questions: int) -> dict[str, np.ndarray]:
    """``brain_records.read_at``'s draws for one stream: the facts in told order, the revisions
    after them, and the questions of each kind."""
    rng = np.random.default_rng(np.random.SeedSequence([founder, 288, 3, 3, n]))
    about = rng.choice(world.pool, size=n, replace=False)
    first = rng.integers(br.PATIENTS, size=(1, n))[0]
    current = first.copy()
    revised = rng.choice(n, size=max(int(n * br.REVISED), 1), replace=False)
    for i in revised:
        current[i] = (first[i] + int(rng.integers(1, br.PATIENTS, size=1)[0])) % br.PATIENTS
    plain = np.setdiff1d(np.arange(n), revised)
    asked = {}
    for kind, pool in (("unrevised", plain), ("revised", revised)):
        asked[f"ask_{kind}"] = np.array([pool[int(rng.integers(len(pool), size=1)[0])]
                                         for _ in range(questions)])
    return {"about": about, "first": first, "current": current, "revised": revised, **asked}


def read_window(net: Any, world: br.World, drawn: dict[str, np.ndarray],
                spec: dict[str, Any]) -> dict[str, Any]:
    """The N told facts and the revisions written into the window, then every question read
    against it."""
    import torch

    lex, n = world.lexicon, len(drawn["about"])
    order = np.concatenate([np.arange(n), drawn["revised"]])
    said = np.concatenate([drawn["first"], drawn["current"][drawn["revised"]]])
    tokens = len(order)
    width, layers, heads = net.width, net.layers, net.heads
    keys = [torch.zeros(1, heads, tokens, width // heads) for _ in range(layers)]
    values = [torch.zeros(1, heads, tokens, width // heads) for _ in range(layers)]
    act = torch.from_numpy(drawn["about"][order] % lex.verbs)
    agent = torch.from_numpy(drawn["about"][order] // lex.verbs)
    patient = torch.from_numpy(said)
    started = time.perf_counter()
    last = 0.0
    with torch.no_grad():
        for start in range(0, tokens, PART):
            stop = min(start + PART, tokens)
            t0 = time.perf_counter()
            x = net.tokens(act[start:stop], agent[start:stop], patient[start:stop],
                           torch.arange(start, stop)).unsqueeze(0)
            mask = torch.ones(stop - start, stop, dtype=torch.bool)
            mask[:, start:] = torch.tril(torch.ones(stop - start, stop - start,
                                                    dtype=torch.bool))
            for layer, block in enumerate(net.blocks):
                kept = (keys[layer][:, :, :start], values[layer][:, :, :start])
                x, k, v = block(x, mask, kept if start else None)
                keys[layer][:, :, start:stop], values[layer][:, :, start:stop] = k, v
            last = (time.perf_counter() - t0) / (stop - start)
        write_seconds = time.perf_counter() - started
        out: dict[str, Any] = {}
        seconds, count = 0.0, 0
        threads = torch.get_num_threads()
        torch.set_num_threads(1)  # a question is read on one core, as the brain's is
        for kind in ("unrevised", "revised"):
            right = []
            for fact in drawn[f"ask_{kind}"]:
                t0 = time.perf_counter()
                about = int(drawn["about"][fact])
                x = net.tokens(torch.tensor([about % lex.verbs]),
                               torch.tensor([about // lex.verbs]),
                               torch.tensor([br.PATIENTS]), torch.tensor([tokens])).unsqueeze(0)
                mask = torch.ones(1, tokens + 1, dtype=torch.bool)
                for layer, block in enumerate(net.blocks):
                    x, _, _ = block(x, mask, (keys[layer], values[layer]))
                answer = int(net.say(net.last(x))[0, 0].argmax())
                seconds += time.perf_counter() - t0
                count += 1
                right.append(answer == int(drawn["current"][fact]))
            out[kind] = round(float(np.mean(right)), 4)
        torch.set_num_threads(threads)
    token_macs = layers * (2 * tokens * width + 12 * width * width)
    return {**out, "tokens": tokens, "write_seconds": round(write_seconds, 2),
            "write_ms_at_end": round(1e3 * last, 4), "ask_ms": round(1e3 * seconds / count, 4),
            "bytes_per_record": round(tokens * layers * 2 * width * 4 / n, 1),
            "macs_per_read": int(token_macs), "macs_per_write_at_end": int(token_macs),
            "write_threads": threads}


def read_index(net: Any, world: br.World, drawn: dict[str, np.ndarray],
               spec: dict[str, Any]) -> dict[str, Any]:
    """Every told fact listed under its act and its agent; a question fetches the facts that
    hold both of its words, then its agent's, the latest first, and the reader reads them in
    told order before the question."""
    import torch

    lex, n = world.lexicon, len(drawn["about"])
    fetched = int(spec.get("fetched", 8))
    order = np.concatenate([np.arange(n), drawn["revised"]])
    said = np.concatenate([drawn["first"], drawn["current"][drawn["revised"]]])
    acts, agents = drawn["about"][order] % lex.verbs, drawn["about"][order] // lex.verbs
    t0 = time.perf_counter()
    by_act = np.argsort(acts, kind="stable")
    by_agent = np.argsort(agents, kind="stable")
    act_at = np.searchsorted(acts[by_act], np.arange(lex.verbs + 1))
    agent_at = np.searchsorted(agents[by_agent], np.arange(lex.forms + 1))
    index_seconds = time.perf_counter() - t0
    out: dict[str, Any] = {}
    seconds, count, touched = 0.0, 0, 0
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.no_grad():
        for kind in ("unrevised", "revised"):
            right, found = [], []
            for fact in drawn[f"ask_{kind}"]:
                t0 = time.perf_counter()
                about = int(drawn["about"][fact])
                act, agent = about % lex.verbs, about // lex.verbs
                under_act = by_act[act_at[act] : act_at[act + 1]]
                under_agent = by_agent[agent_at[agent] : agent_at[agent + 1]]
                both = np.intersect1d(under_act, under_agent)
                rest = np.setdiff1d(under_agent, both)
                took = np.concatenate([both[::-1], rest[::-1]])[:fetched]
                took = np.sort(took)  # heard in told order
                touched += len(under_act) + len(under_agent)
                x = net.tokens(
                    torch.from_numpy(np.append(acts[took], act)),
                    torch.from_numpy(np.append(agents[took], agent)),
                    torch.from_numpy(np.append(said[took], br.PATIENTS)),
                    torch.from_numpy(np.append(took, len(order)))).unsqueeze(0)
                answer = int(net(x, taught_mask(len(took), 1))[0, -1].argmax())
                seconds += time.perf_counter() - t0
                count += 1
                right.append(answer == int(drawn["current"][fact]))
                latest = np.flatnonzero(order == fact)[-1]
                found.append(bool(np.isin(latest, took)))
            out[kind] = round(float(np.mean(right)), 4)
            out[f"found_{kind}"] = round(float(np.mean(found)), 4)
    torch.set_num_threads(threads)
    return {**out, "tokens": len(order), "fetched": fetched,
            "write_seconds": round(index_seconds, 4),
            "write_ms_at_end": round(1e3 * index_seconds / len(order), 6),
            "ask_ms": round(1e3 * seconds / count, 4),
            "bytes_per_record": round(len(order) * (2 * 8 + 3 * 4) / n, 1),
            "postings_per_read": round(touched / count, 1),
            "macs_per_read": int((fetched + 1) * net.layers * 12 * net.width * net.width)}


def run_cell(spec: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    founder = int(spec["founder"])
    torch.set_num_threads(int(spec.get("threads", 1)))
    world = br.World(founder, int(spec.get("nouns", br.NOUNS)), int(spec.get("verbs", br.VERBS)))
    lex = world.lexicon
    layers, heads = int(spec.get("layers", 3)), int(spec.get("heads", 4))
    width = int(spec["width"]) if spec.get("width") else width_for(
        int(spec["target"]), lex.verbs, lex.forms, layers, heads)
    net = build(lex.verbs, lex.forms, width, layers, heads, founder)
    base = {"founder": founder, "width": width, "layers": layers, "heads": heads,
            "parameters": parameters(net), "target": spec.get("target"),
            "nouns": lex.nouns, "verbs": lex.verbs, **teach(net, world, founder, spec)}
    net.eval()
    rows = []
    for n in spec["n"]:
        drawn = reading(world, founder, int(n), int(spec.get("questions", 400)))
        for arm, read in (("context", read_window), ("retrieval", read_index)):
            if arm not in spec.get("arms", ["context", "retrieval"]) or (
                    arm == "context" and int(n) > int(spec.get("window_most", 10 ** 9))):
                continue
            t0 = time.perf_counter()
            rows.append({**base, "arm": arm, "n": int(n), **read(net, world, drawn, spec),
                         "seconds": round(time.perf_counter() - t0, 1)})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    todo = [{**{k: v for k, v in plan.items() if k != "founders"}, "founder": f}
            for f in plan["founders"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink, get_context("spawn").Pool(args.workers,
                                                               maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_cell, todo, chunksize=1):
            for row in rows:
                sink.write(json.dumps(row) + "\n")
            sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
