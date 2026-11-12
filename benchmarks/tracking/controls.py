"""The tracking chamber's matched baselines: an LSTM and a causal transformer trained by gradient.

Each is given what a brain is given of a one-thing scene, step by step (the report or zeros, the
in-view flag, the commands scaled by 20 and with ``joints`` the joint readings), and answers a
position at every step. They are independently scored controls, never part of a brain's answer.

Lessons. ``truth``: the squared distance to the true position at every step of a training scene.
``camera``: the squared distance to the report at the steps in view, and nothing at any other
step; a control hides the report from itself over spans it draws (a span begins at a step in
view at rate 0.04 and lasts 20 steps on average), so its answer over a hidden span is taught by
the report it did not read. No position the camera did not give enters a ``camera`` lesson.

Modes. ``online``: one pass over the training scenes in the brain's order, one Adam step per
``rows`` scenes (rate 0.003, gradient norm clipped at 1), the brain's exposure. ``offline``:
shuffled epochs of 32 scenes a step from rate 0.002, halved whenever the loss on a tenth of the
training scenes held aside has not fallen for 8 epochs (no lower than 0.00002), until that loss
has not fallen for 25 epochs or 300 epochs are spent; the answers are those of the epoch whose
held-aside loss was least. The width is the one whose parameter count is nearest the brain's. Work
is counted as updates, analytic multiply-accumulates (a backward pass at twice the forward) and
seconds.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

RATES = {"online": 0.003, "offline": 0.002}
CLIP, PATIENCE, EPOCHS, BATCH = 1.0, 25, 300, 32
HALVE, FLOOR = 8, 2e-5  # epochs without a fall before the rate halves, and its floor
MAX_LENGTH = 1200  # steps the transformer's position table covers
COMMAND_SCALE = 20.0


def features(scene: Any, joints: bool, hidden: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """A one-thing scene as ``(steps, inputs)`` rows and its in-view flags. ``hidden`` marks
    steps whose report the learner hides from itself."""
    import tracking

    feed = tracking.given(scene)
    seen = feed["seen"].copy()
    shown = seen if hidden is None else seen & ~hidden
    parts = [feed["report"] * shown[:, None], shown[:, None].astype(np.float64),
             feed["commands"] * COMMAND_SCALE]
    if joints:
        parts.append(feed["joints"])
    return np.hstack(parts).astype(np.float32), seen


def spans(rng: np.random.Generator, seen: np.ndarray) -> np.ndarray:
    """Steps a ``camera`` learner hides from itself: spans that begin at a step in view at rate
    0.04 and last 20 steps on average. Step 0 stays."""
    hidden = np.zeros(len(seen), dtype=bool)
    t = 1
    while t < len(seen):
        if seen[t] and rng.random() < 0.04:
            length = int(rng.geometric(1.0 / 20.0))
            hidden[t : t + length] = True
            t += length
        t += 1
    return hidden


def build(model: str, width: int, inputs: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))

    class LSTM(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.LSTM(inputs, width, batch_first=True)
            self.out = nn.Linear(width, 2)

        def forward(self, x: Any) -> Any:
            h, _ = self.cell(x)
            return self.out(h)

        def macs(self, length: int) -> int:
            return length * (4 * width * (inputs + width) + width * 2)

    class Transformer(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 2

        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(inputs, width)
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(width, 2)
            position = torch.arange(MAX_LENGTH)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(MAX_LENGTH, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = self.inp(x) + self.position[:n]
            return self.out(self.body(h, mask=mask, is_causal=True))

        def macs(self, length: int) -> int:
            per_token = inputs * width + self.LAYERS * 8 * width * width + width * 2
            return length * per_token + self.LAYERS * width * length * (length + 1)

    return LSTM() if model == "lstm" else Transformer()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int, inputs: int) -> int:
    """The width whose parameter count is nearest ``target`` (even, for the transformer)."""
    step = 2 if model == "transformer" else 1

    def size(width: int) -> int:
        return count(build(model, width, inputs, 0))

    low, high = 4 // step, 1024 // step
    while low < high:
        middle = (low + high) // 2
        if size(middle * step) < target:
            low = middle + 1
        else:
            high = middle
    candidates = [width for width in (low * step - step, low * step) if width >= 4]
    return min(candidates, key=lambda width: (abs(size(width) - target), width))


class Learner:
    def __init__(self, model: str, width: int, inputs: int, seed: int, rate: float,
                 lessons: str, joints: bool) -> None:
        import torch

        self.torch = torch
        self.net = build(model, width, inputs, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.lessons, self.joints = lessons, joints
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 77]))
        self.work = {"updates": 0, "steps": 0, "macs": 0, "seconds": 0.0}

    def batch(self, scenes: list[Any], hide: bool) -> tuple[Any, Any, Any]:
        torch = self.torch
        xs, ys, ms = [], [], []
        for scene in scenes:
            if self.lessons == "camera":
                seen = scene.in_view[:, 0].copy()
                hidden = spans(self.rng, seen) if hide else None
                x, _ = features(scene, self.joints, hidden)
                ys.append(np.where(seen[:, None], scene.report[:, 0], 0.0))
                ms.append(seen)
            else:
                x, _ = features(scene, self.joints)
                ys.append(scene.truth[:, 0])
                ms.append(np.ones(scene.steps, dtype=bool))
            xs.append(x)
        return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys).astype(np.float32)),
                torch.from_numpy(np.stack(ms)))

    def loss(self, scenes: list[Any], hide: bool) -> Any:
        x, y, mask = self.batch(scenes, hide)
        squared = ((self.net(x) - y) ** 2).sum(-1)
        return (squared * mask).sum() / mask.sum().clamp(min=1)

    def update(self, scenes: list[Any]) -> float:
        torch = self.torch
        started = time.monotonic()
        loss = self.loss(scenes, hide=True)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), CLIP)
        self.opt.step()
        steps = scenes[0].steps
        self.work["updates"] += 1
        self.work["steps"] += len(scenes) * steps
        self.work["macs"] += 3 * len(scenes) * self.net.macs(steps)
        self.work["seconds"] += time.monotonic() - started
        return float(loss.item())

    def answer(self, scenes: list[Any]) -> list[np.ndarray]:
        torch = self.torch
        out = []
        with torch.no_grad():
            for scene in scenes:
                x, _ = features(scene, self.joints)
                out.append(self.net(torch.from_numpy(x[None]))[0].numpy()[:, None, :].astype(
                    np.float64))
        return out

    def milliseconds(self, scene: Any) -> float:
        """Wall time of one step's answer, in milliseconds: the LSTM's whole pass over the
        scene divided by its steps; the transformer's pass over the scene so far, at the
        scene's middle step."""
        torch = self.torch
        x, _ = features(scene, self.joints)
        with torch.no_grad():
            began = time.perf_counter()
            if hasattr(self.net, "cell"):
                self.net(torch.from_numpy(x[None]))
                return (time.perf_counter() - began) * 1000.0 / len(x)
            self.net(torch.from_numpy(x[None, : len(x) // 2]))
            return (time.perf_counter() - began) * 1000.0


def life(arm: str, training: list[Any], held_out: list[Any], founder: int, *, lessons: str,
         joints: bool, parameters: int, rows: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Train ``arm`` (``lstm-online``, ``lstm-offline``, ``transformer-online`` or
    ``transformer-offline``) on ``training`` and answer ``held_out``."""
    import torch

    torch.set_num_threads(1)
    model, mode = arm.split("-")
    inputs = features(training[0], joints)[0].shape[1]
    width = matched_width(model, parameters, inputs)
    learner = Learner(model, width, inputs, founder, RATES[mode], lessons, joints)
    epochs = 0
    if mode == "online":
        for begin in range(0, len(training), rows):
            learner.update(training[begin : begin + rows])
        epochs = 1
    else:
        aside = max(len(training) // 10, 1)
        fit, check = training[aside:], training[:aside]
        order = np.random.default_rng(np.random.SeedSequence([int(founder), 99]))
        best, since, state = float("inf"), 0, None
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            learner.opt, factor=0.5, patience=HALVE, min_lr=FLOOR)
        for _ in range(EPOCHS):
            for batch in np.array_split(order.permutation(len(fit)),
                                        max(len(fit) // BATCH, 1)):
                learner.update([fit[int(i)] for i in batch])
            epochs += 1
            with torch.no_grad():
                held = float(np.mean([learner.loss(check[i : i + BATCH], hide=True).item()
                                      for i in range(0, len(check), BATCH)]))
            schedule.step(held)
            if held < best - 1e-7:
                best, since = held, 0
                state = {k: v.clone() for k, v in learner.net.state_dict().items()}
            else:
                since += 1
                if since >= PATIENCE:
                    break
        if state is not None:
            learner.net.load_state_dict(state)
    answers = learner.answer(held_out)
    return answers, {"parameters": count(learner.net), "width": width, "epochs": epochs,
                     "macs": float(learner.work["macs"]), "work": learner.work,
                     "milliseconds": learner.milliseconds(held_out[0])}
