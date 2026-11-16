"""The launches lives rarely make, each checked against the kernels' own Python
(t/reference.py): masks, adaptation, plain, softmax and anchored nudges, weights, held apical
gains, warm starts, tolerances and the solves, on a layered graph and on a deep brain's graph.

    python kernels/wasm/browser/run.py direct.py
"""
import numpy as np  # noqa: E402
from reference import report  # noqa: E402

import cadence as cd  # noqa: E402
from cadence.brain import Nudge  # noqa: E402
from cadence.neuron import Adaptation  # noqa: E402

rng = np.random.default_rng(7)


def graphs():
    model = cd.learning_neuron_model()
    connectome = cd.layered(6, 24, 12, seed=1)
    yield "layered", cd.NeuralGraph(connectome, model)
    yield "layered, adaptation", cd.NeuralGraph(
        connectome, model.replace(adaptation=Adaptation(tau_steps=7.0, strength=0.3)))
    yield "layered, dt 1", cd.NeuralGraph(connectome, model.replace(dt=1.0))
    deep = cd.Brain.compose(10, 9, modules=(32, 16), apical=0.25, teaching="deep",
                            backend="cpu", seed=3)
    yield "deep apical", deep.learner.brain


def nudges(graph, batch):
    n = graph.connectome.n
    outputs = np.zeros(n)
    outputs[-9:] = 1.0  # a group of nine, past NumPy's eight running sums
    target = rng.random((batch, n))
    yield "no nudge", None
    yield "plain", Nudge(target=target, mask=outputs, beta=0.4)
    yield "weighted", Nudge(target=target, mask=outputs, beta=0.4,
                            weight=rng.normal(size=batch))
    yield "softmax", Nudge(target=target, mask=outputs, beta=0.6, softmax_temperature=0.3)
    groups = np.full(n, -1)
    groups[-9:-4], groups[-4:] = 0, 1
    yield "softmax groups", Nudge(target=target, mask=outputs, beta=0.6,
                                  softmax_temperature=0.5, groups=groups)
    gain = np.zeros(n)
    gain[-12:-6] = 0.7
    yield "anchored", Nudge(target=target, mask=outputs, beta=0.4, anchor=rng.random((batch, n)),
                            anchor_gain=gain)


for label, graph in graphs():
    n = graph.connectome.n
    for batch in (1, 3):
        drive = rng.normal(size=(batch, n)) * 0.8
        mask = (rng.random((batch, n)) > 0.15).astype(float)
        mask[:, :3] = 0.5
        for _name, nudge in nudges(graph, batch):
            if nudge is not None and graph.apical is not None:
                free = graph.settle_batch(drive, steps=30)
                held = graph.apical_gain(drive, free)
                nudge = Nudge(target=nudge.target, mask=nudge.mask, beta=nudge.beta,
                              softmax_temperature=nudge.softmax_temperature, weight=nudge.weight,
                              groups=nudge.groups, anchor=nudge.anchor,
                              anchor_gain=nudge.anchor_gain, apical_gain=held)
            for m in (None, mask):
                state = graph.settle_batch(drive, steps=25, mask=m, nudge=nudge)
                graph.residual(drive, state, mask=m, nudge=nudge)
                warm = graph.settle_batch(drive, steps=40, state=state, mask=m, nudge=nudge,
                                          tolerance=1e-6)
                graph.residual(drive, warm, mask=m, nudge=nudge)
                graph.equilibrate(drive, budget=200, chunk=16, tolerance=1e-7, state=warm, mask=m,
                                  nudge=nudge)
        graph.equilibrate(drive, budget=300, chunk=32, tolerance=1e-8)
        graph.equilibrate(drive, budget=300, tolerance=1e-6, fast=True)
        graph.equilibrate(drive, budget=300, chunk=8, tolerance=1e-9, damping=2)
    print(label, "done", flush=True)

# a plain learner's lessons: the optimizer's compiled passes and the relative floor
connectome = cd.layered(6, 24, 4, seed=2)
outputs = connectome.populations["output"]
for config in (cd.LearnerConfig(momentum=0.9, normalize=0.999, eta=0.02, eta_bias=0.02),
               cd.LearnerConfig(momentum=0.9, normalize=0.999, normalize_relative=0.5, eta=0.02,
                                eta_bias=0.02),
               cd.LearnerConfig(normalize=0.99, normalize_relative=0.25, eta=0.02, eta_bias=0.02),
               cd.LearnerConfig(momentum=0.8, eta=0.05)):
    learner = cd.Learner(cd.NeuralGraph(connectome, cd.learning_neuron_model()), outputs, config)
    for _ in range(6):
        drive = np.zeros((4, connectome.n))
        drive[:, :6] = rng.random((4, 6))
        learner.step(drive, rng.integers(0, 4, size=4))
print("learner done", flush=True)
report()
