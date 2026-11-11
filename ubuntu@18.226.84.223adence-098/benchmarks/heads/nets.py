"""The heads chamber's controls: the grammar chamber's matched networks on its tasks.

``lstm`` and ``transformer`` are ``controls.py``'s two networks, built by its code at a task's
inputs and actions, their width the one whose parameter count is the nearest to the ``Fd`` brain's
on that task. They hear the brain's moments and are taught its answer moments by cross-entropy,
teacher-forced; they are read on its panels by its scorer, greedy, in closed loop where the task
feeds answers back. They are independently scored controls and never part of a brain's answer path.

- ``online``: one pass in the brain's order, one Adam step per batch (rate 0.003, gradient norm
  clipped at 1): equal exposure.
- ``offline``: the same batches in shuffled epochs (rate 0.001) until the seconds spent in updates
  reach ``compute_seconds``, the brain's measured life on the same machine: equal compute.

**How a never-taught name reaches the output.** In these two networks it has no route. Its input
weights stay at their random start (it is never heard in training), and its output row is a row of
the last linear map that no lesson raised and every lesson's cross-entropy lowers, so the logit of
a never-taught name falls through a life. The brain without ``recall_gain`` has none either: its
motor neuron's synapses took no lesson with it as the label. The brain with ``recall_gain`` has one
the two networks lack: one learned scalar per read port on the token's identity. ``tied`` gives the
transformer the matching route and is reported apart: its logits are the hidden state's product
with the input weights of the same token (weight tying), so a hidden state that holds a token's
embedding names that token, never-taught ones included. It is no restriction of the brain's and no
candidate for it. ``pointer`` gives the transformer the brain's route itself: one more attention
over the moments before, its query and keys learned maps of the hidden state, whose read of the
heard tokens reaches the logits through one learned scalar from zero, beside the last linear map.
``copy`` takes that map away from the tokens: a token's logit is the log of that attention's read of
it, so the network says what it attends, a pointer network. All are reported apart, each at the
brain's parameter count.

Causality here, on any network: a moment blanked at test (``blank_answer`` the moment that told the
answer, ``blank_dup`` the duplicate's earlier mention, ``blank_dup2`` the duplicate itself), and on
the transformers the attention layers' output silenced.

    python benchmarks/heads/nets.py --task ioi --models lstm transformer tied --modes online \\
        --founders 0 1 2 --out DIR/nets.jsonl --workers 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import heads as hd
import numpy as np

ct = hd._module("controls", hd.HERE.parent / "grammar" / "controls.py")
MODELS = ("lstm", "transformer", "tied", "pointer", "copy")
MODES = ("online", "offline")
MAX_EPOCHS = 400
CAUSAL: dict[str, dict[str, Any]] = {
    "silence": {"silence": True},
    "blank_answer": {"blank": "answer"},
    "blank_dup": {"blank": "dup"},
    "blank_dup2": {"blank": "dup2"},
}


@contextmanager
def _dims(task: hd.Task) -> Any:
    """``controls.py``'s code at a task's inputs and actions."""
    saved = ct.gr
    ct.gr = SimpleNamespace(INPUTS=task.inputs, ACTIONS=task.actions)
    try:
        yield
    finally:
        ct.gr = saved


def build(model: str, width: int, seed: int, task: hd.Task) -> Any:
    import torch
    from torch import nn

    with _dims(task):
        net = ct.build("transformer" if model in ("tied", "pointer", "copy") else model, width,
                       seed)

    if model in ("lstm", "transformer"):
        return net

    class Pointer(nn.Module):  # type: ignore[misc]
        """The transformer with a read of the moments before on the tokens' identity: the read's
        attention ``p``, from learned maps of the hidden state, adds ``gain * sum_s p_s x_s`` to
        the logits of the tokens, ``gain`` one scalar from zero."""

        def __init__(self) -> None:
            super().__init__()
            self.inp, self.body, self.out = net.inp, net.body, net.out
            self.register_buffer("position", net.position)
            self.query, self.key = nn.Linear(width, width), nn.Linear(width, width)
            self.gain = nn.Parameter(torch.zeros(1))
            self.pairs = [(j, t) for j, t in enumerate(task.tokens()) if t >= 0]

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = self.body(self.inp(x) + self.position[:n], mask=mask, is_causal=True)
            logits = self.out(h)
            before = torch.triu(torch.full((n, n), float("-inf")), diagonal=0)  # s < t
            before[0, 0] = 0.0  # the first moment reads itself, weighed out below
            scores = self.query(h) @ self.key(h).transpose(1, 2) / width ** 0.5
            p = torch.softmax(scores + before, dim=-1)
            p = torch.cat([torch.zeros_like(p[:, :1]), p[:, 1:]], dim=1)  # nothing before it
            actions = [j for j, _ in self.pairs]
            tokens = [t for _, t in self.pairs]
            read = p @ x[:, :, tokens]
            added = torch.zeros_like(logits)
            if model == "copy":  # the tokens' logits are the read's alone
                logits = logits.clone()
                logits[:, :, actions] = torch.log(read + 1e-6)
                return logits
            added[:, :, actions] = self.gain * read
            return logits + added

        def macs(self, length: int) -> int:
            with _dims(task):
                return int(net.macs(length)) + 2 * length * length * width * width

    if model in ("pointer", "copy"):
        return Pointer()

    class Tied(nn.Module):  # type: ignore[misc]
        """The transformer whose logits are the hidden state's product with the same token's input
        weights; an action that is no token keeps a row of its own."""

        def __init__(self) -> None:
            super().__init__()
            self.inp, self.body = net.inp, net.body
            self.register_buffer("position", net.position)
            self.coordinates = [t for t in task.tokens() if t >= 0]
            others = len(task.tokens()) - len(self.coordinates)
            self.others = nn.Parameter(torch.randn(others, width) * 0.1) if others else None

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = self.body(self.inp(x) + self.position[:n], mask=mask, is_causal=True)
            logits = h @ self.inp.weight[:, self.coordinates]
            return logits if self.others is None else torch.cat([logits, h @ self.others.T], dim=2)

        def macs(self, length: int) -> int:
            with _dims(task):
                return int(net.macs(length))

    return Tied()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int, task: hd.Task) -> int:
    """The width whose parameter count is the nearest to ``target`` (even, as the control's)."""
    best = None
    for width in range(4, 257, 2):
        n = count(build(model, width, 0, task))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
    assert best is not None
    return best[0]


def encode(moments: list[hd.Moment]) -> tuple[np.ndarray, np.ndarray]:
    """``(batch, moments, inputs)`` as heard, and ``(batch, moments)`` with -100 untaught."""
    x = np.stack([m.x for m in moments], axis=1).astype(np.float32)
    n = len(moments[0].x)
    y = np.stack([np.full(n, -100) if m.label is None else m.label for m in moments], axis=1)
    return x, y.astype(np.int64)


class Learner:
    def __init__(self, model: str, width: int, seed: int, rate: float, task: hd.Task) -> None:
        import torch

        self.torch = torch
        self.task = task
        self.net = build(model, width, seed, task)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "tokens": 0, "macs": 0, "seconds": 0.0}

    def macs(self, length: int) -> int:
        with _dims(self.task):
            return int(self.net.macs(length))

    def update(self, moments: list[hd.Moment]) -> float:
        torch = self.torch
        started = time.monotonic()
        x, y = encode(moments)
        logits = self.net(torch.from_numpy(x))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, self.task.actions),
                                                 torch.from_numpy(y).reshape(-1),
                                                 ignore_index=-100)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), ct.CLIP)
        self.opt.step()
        self.work["updates"] += 1
        self.work["tokens"] += x.shape[0] * x.shape[1]
        self.work["macs"] += 3 * x.shape[0] * self.macs(x.shape[1])
        self.work["seconds"] += time.monotonic() - started
        return float(loss.item())

    def answers(self, moments: list[hd.Moment], plan: dict[str, Any] | None = None
                ) -> list[np.ndarray]:
        """Each answer moment's greedy answers; a moment that feeds answers back hears the
        previous answer. ``plan``: a named moment blanked, the attention silenced."""
        torch = self.torch
        plan = plan or {}
        hooks = []
        if plan.get("silence"):
            layers = getattr(getattr(self.net, "body", None), "layers", ())
            for layer in layers:
                hooks.append(layer.self_attn.register_forward_hook(
                    lambda _m, _i, out: (torch.zeros_like(out[0]), out[1])))
        x, _ = encode(moments)
        answer_at = [t for t, m in enumerate(moments) if m.label is not None]
        last = moments[answer_at[-1]]
        blank = plan.get("blank")
        if blank is not None and blank in last.slots:
            rows = np.flatnonzero(last.slots[blank] >= 0)
            x[rows, last.slots[blank][rows]] = 0.0
        out: list[np.ndarray] = []
        with torch.no_grad():
            if any(m.feedback for m in moments):
                for t in answer_at:
                    if moments[t].feedback and out:
                        x[:, t] = hd.onehot(out[-1], self.task.inputs).astype(np.float32)
                    logits = self.net(torch.from_numpy(x[:, : t + 1]))[:, -1]
                    out.append(logits.argmax(dim=1).numpy())
            else:
                logits = self.net(torch.from_numpy(x))
                out = [logits[:, t].argmax(dim=1).numpy() for t in answer_at]
        self.work["macs"] += x.shape[0] * self.macs(x.shape[1])
        for hook in hooks:
            hook.remove()
        return out


def read_panel(learner: Learner, task: hd.Task, batches: list[list[hd.Moment]],
               plan: dict[str, Any] | None = None) -> dict[str, Any]:
    tally = hd.Tally()
    exact = []
    for moments in batches:
        said = learner.answers(moments, plan)
        hits = []
        asked = [m for m in moments if m.label is not None]
        for m, action in zip(asked, said, strict=True):
            tally.add(m, action, None)
            hits.append(action == m.label)
        exact.append(np.all(np.stack(hits), axis=0))
    out = tally.summary(twins=task.twins)
    out["exact"] = round(float(np.mean(np.concatenate(exact))), 4)
    return out


def reading(learner: Learner, task: hd.Task, tests: dict[str, list[list[hd.Moment]]],
            causal: bool = True) -> dict[str, Any]:
    learner.net.eval()
    out = {}
    for name, batches in tests.items():
        panel = read_panel(learner, task, batches)
        if causal:
            panel["causal"] = {}
            for cut, plan in CAUSAL.items():
                if "blank" in plan and not any(plan["blank"] in m.slots for m in batches[0]):
                    continue
                if "silence" in plan and not hasattr(learner.net, "body"):
                    continue
                read = read_panel(learner, task, batches, plan)
                panel["causal"][cut] = {
                    "conditions": {k: {"answer": v["answer"], **({"says_avoid": v["says_avoid"]}
                                                                if "says_avoid" in v else {})}
                                   for k, v in read["conditions"].items()},
                    **{k: read[k] for k in ("both", "differ", "exact") if k in read}}
        out[name] = panel
    learner.net.train()
    return out


def brain_parameters(task: hd.Task, founder: int, arm: str = "Fd") -> int:
    import warnings

    warnings.simplefilter("ignore")
    return int(hd.make_brain(task, arm, founder).parameters())


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One control life: ``task``, ``model``, ``mode``, ``founder``, with ``lessons``,
    ``panel_batches``, ``parameters`` (the brain's by default), ``width`` and for ``offline``
    ``compute_seconds`` and ``epochs`` (the most)."""
    import torch

    torch.set_num_threads(1)
    model, mode, founder = spec["model"], spec["mode"], int(spec["founder"])
    task = hd.make_task(spec["task"], founder)
    lessons = int(spec.get("lessons", hd.LESSONS.get(task.name, hd.SINGLE_LESSONS)))
    started = time.monotonic()
    target = int(spec.get("parameters") or brain_parameters(task, founder))
    width = int(spec.get("width") or matched_width(model, target, task))
    learner = Learner(model, width, int(spec.get("init_seed", founder)), ct.RATES[mode], task)
    tests = task.panels(int(spec.get("panel_batches", 10 if task.twins else 24)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, hd.SEEDS["train"]]))
    batches = [task.train(rng) for _ in range(lessons)]
    curve: list[dict[str, Any]] = []
    epochs = 1
    if mode == "online":
        losses = [learner.update(b) for b in batches]
        last = float(np.mean(losses[-50:]))
    else:
        budget = float(spec.get("compute_seconds", 0.0))
        most = int(spec.get("epochs", MAX_EPOCHS))
        order = np.random.default_rng(np.random.SeedSequence([founder, 99, 0]))
        epochs, last = 0, float("nan")
        while epochs < most and (epochs == 0 or learner.work["seconds"] < budget):
            last = float(np.mean([learner.update(batches[int(i)])
                                  for i in order.permutation(lessons)]))
            epochs += 1
            panels = reading(learner, task, tests, causal=False)
            curve.append({"epoch": epochs, "update_seconds": round(learner.work["seconds"], 1),
                          "loss": round(last, 4),
                          "panels": {k: v["conditions"]["all"]["answer"]
                                     for k, v in panels.items()}})
    row: dict[str, Any] = {
        "task": task.name, "arm": f"{model}:{mode}", "model": model, "mode": mode,
        "founder": founder, "init_seed": int(spec.get("init_seed", founder)), "replica": 0,
        "tag": spec.get("tag"), "width": width, "parameters": count(learner.net),
        "brain_parameters": target, "lessons": lessons, "rate": ct.RATES[mode], "epochs": epochs,
        "loss_last": last, "curve": curve, "torch": torch.__version__,
        "compute_seconds": spec.get("compute_seconds"), "chance": hd.chance(task),
    }
    work = dict(learner.work)
    row["panels"] = reading(learner, task, tests)
    row["work"] = {**work, "seconds": round(work["seconds"], 1)}
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--task", choices=sorted(hd.TASKS))
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    parser.add_argument("--modes", nargs="*", default=["online"], choices=MODES)
    parser.add_argument("--founders", nargs="*", type=int, default=list(hd.FOUNDERS))
    parser.add_argument("--plan", type=Path, help="a JSON list of life specs in place of the grid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.plan is not None:
        specs = json.loads(args.plan.read_text())
    else:
        if args.task is None:
            parser.error("--task or --plan is required")
        specs = [{"task": args.task, "model": m, "mode": mode, "founder": f,
                  "tag": f"{args.task}/{m}:{mode}/{f}"}
                 for m in args.models for mode in args.modes for f in args.founders]
    done = set()
    if args.out.exists():
        done = {json.loads(line).get("tag") for line in args.out.read_text().splitlines() if line}
    specs = [s for s in specs if s.get("tag") not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        with get_context("spawn").Pool(max(args.workers, 1), maxtasksperchild=1) as pool:
            for row in pool.imap_unordered(run_life, specs, chunksize=1):
                sink.write(json.dumps(row) + "\n")
                sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
