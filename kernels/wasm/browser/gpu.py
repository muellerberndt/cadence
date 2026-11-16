"""A batch of lives settled on the GPU (``kernels/webgpu/lives.js``, WebGPU, float32) against the
WebAssembly kernel (float64) on the same lives: Wild's pocket brain, each life its own synapses,
a free settle of ``SWEEPS`` sweeps from rest under a drive of its own. Reads the milliseconds per
life of the WebAssembly lives kernel, of lone settles and of the GPU (upload, sweeps, read back),
and how far the GPU's activations lie from the float64 ones (passing under 1e-5 on activations,
float32 rounding over the sweeps). Without WebGPU the lives settle on the WebAssembly kernel alone,
which is the fallback. Past a hundred lives the brains repeat
(a 32-bit tab holds a hundred composed brains); every life keeps its own rows and its own drive.

    python kernels/wasm/browser/run.py gpu.py [sweeps] [lives ...]
"""
from __future__ import annotations

import json
import sys
import time

import engine

engine.choose("wasm")
SWEEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SIZES = [int(x) for x in sys.argv[2:]] or [12, 50, 200, 1000]

import numpy as np  # noqa: E402
from compositions import POCKET  # noqa: E402
from js import Object  # noqa: E402
from pyodide.code import run_js  # noqa: E402
from pyodide.ffi import to_js  # noqa: E402

import cadence as cd  # noqa: E402
from cadence import _wasm, fused  # noqa: E402

engine.fit_32_bit_numpy()
lives_call = _wasm._caller("settle_lives")
TOLERANCE = 1e-5
module = await run_js("import('/webgpu/lives.js')")  # noqa: F704
try:
    gpu = await module.GpuLives.create()  # noqa: F704
    print("adapter", gpu.adapter.info.vendor, gpu.adapter.info.architecture, flush=True)
except Exception as e:  # no WebGPU: the WebAssembly kernel settles the lives
    gpu = None
    print("no WebGPU here, WebAssembly only:", e, flush=True)


def anatomy_of(graph):
    """The GPU's edge order (every neuron's basal synapses, then every neuron's apical ones, each
    by postsynaptic neuron) and the anatomy ``GpuLives.load`` takes."""
    c = graph.connectome
    n = c.n
    post = np.asarray(c.post)
    apical = np.zeros(c.synapses, dtype=bool) if graph.apical is None else graph.apical.synapses
    order = np.lexsort((np.arange(c.synapses), post, apical))
    basal = int((~apical).sum())
    basal_start = np.searchsorted(post[order][:basal], np.arange(n + 1))
    apical_start = basal + np.searchsorted(post[order][basal:], np.arange(n + 1))
    m = graph.neuron_model
    rest = m.rest_emission
    pools_of = np.full(n, -1, dtype=np.int32)
    members, starts = [], [0]
    if graph.apical is not None:
        for k, part in enumerate(graph.apical.members):
            idx = np.arange(n)[part]
            pools_of[idx] = k
            members.extend(idx.tolist())
            starts.append(len(members))
    return order, {
        "n": n, "pre": np.asarray(c.pre)[order].astype(np.uint32),
        "basalStart": basal_start.astype(np.uint32), "apicalStart": apical_start.astype(np.uint32),
        "poolOf": pools_of, "poolStart": np.array(starts, dtype=np.uint32),
        "poolMember": np.array(members, dtype=np.uint32),
        "rule": {"dt": m.dt, "slope": m.slope, "threshold": m.threshold, "rest": rest,
                 "up": 1.0 / (1.0 - rest), "down": m.leak / rest if rest > 0 else 0.0},
        "apical": None if graph.apical is None else {
            "fraction": graph.apical.fraction, "floor2": graph.apical.floor ** 2, "tiny": 1e-24},
    }


def settler(graph, block, m: int):
    """The WebAssembly kernel's settle of ``m`` lives of ``graph``'s anatomy, each with its own
    rows (``block`` is one life's ``_fused_blocks()``, for the compartment's structure)."""
    n, lay = graph.connectome.n, graph.layout
    dt, slope, threshold, rest, leak, strength, tau = fused._rule(graph.neuron_model)
    comp = fused._compartment(block[1], lay, block[0], None, m, n)
    nud = fused._no_nudge(m, n)
    kept = np.zeros(int(_wasm._EXPORTS.kept_length(
        id(lay.starts), id(lay.pair_post), id(comp[0]), id(comp[2]), m, n)))
    fed = ~fused._fed_ranges(lay)

    def settle(standing, flat, flat_up, s):
        lives_call(np.zeros((m, n)), np.zeros((m, n)), s, standing, flat, comp[0], flat_up,
                   comp[2], lay.starts, lay.pair_pre, lay.pair_post, lay.offset, fed,
                   fused._NO_MASK, False, dt, slope, threshold, rest, leak, False, strength, tau,
                   False, *nud[:4], nud[4], *nud[5:], bool(comp[3]), *comp[4:8], *comp[8:11],
                   comp[11], False, SWEEPS, 0.0, False, np.zeros(1), False, np.zeros(m), kept)
    return settle


rows, failures = [], 0
for lives in SIZES:
    distinct = [cd.Brain.compose(**POCKET, backend="cpu", seed=s, arousal=True).learner.brain
                for s in range(min(lives, 100))]
    graphs = [distinct[k % len(distinct)] for k in range(lives)]
    g0 = graphs[0]
    n = g0.connectome.n
    same = all(np.array_equal(g.connectome.pre, g0.connectome.pre)
               and np.array_equal(g.connectome.post, g0.connectome.post) for g in graphs)
    order, anat = anatomy_of(g0)
    rng = np.random.default_rng(lives)
    drives = rng.normal(size=(lives, n)) * 0.5
    standing = drives + np.stack([g.bias for g in graphs])
    # the WebAssembly kernel on every life in one call (float64)
    blocks = [g._fused_blocks() for g in graphs]
    flat = np.stack([b[0] for b in blocks])
    flat_up = np.stack([b[1][2] for b in blocks]) if blocks[0][1] is not None else flat
    _, slope, threshold, rest, leak, _, _ = fused._rule(g0.neuron_model)
    start_s = np.empty((lives, n))
    fused._publish(np.zeros((lives, n)), start_s, np.zeros(1), False, slope, threshold, rest, leak)

    together, alone = settler(g0, blocks[0], lives), settler(g0, blocks[0], 1)
    s = start_s.copy()
    t = time.perf_counter()
    together(standing, flat, flat_up, s)
    wasm_ms = 1e3 * (time.perf_counter() - t) / lives
    lone = start_s.copy()
    t = time.perf_counter()
    for k in range(lives):
        alone(standing[k:k + 1], flat[k:k + 1], flat_up[k:k + 1], lone[k:k + 1])
    lone_ms = 1e3 * (time.perf_counter() - t) / lives
    failures += not np.array_equal(lone, s)  # a life settles alike alone and among others
    del flat, flat_up, blocks
    if gpu is None:
        print("WASM " + json.dumps({"lives": lives, "sweeps": SWEEPS,
                                    "wasm_lives_ms_per_life": round(wasm_ms, 3),
                                    "wasm_alone_ms_per_life": round(lone_ms, 3)}), flush=True)
        continue
    # the GPU: each life's synapses in the anatomy's edge order, float32
    weights = np.stack([np.asarray(g._weights)[order] for g in distinct]).astype(np.float32)
    gpu.load(to_js(anat, dict_converter=Object.fromEntries), lives,
             to_js(weights[np.arange(lives) % len(distinct)].ravel()))
    st32 = to_js(standing.astype(np.float32).ravel())
    v32 = to_js(np.zeros(lives * n, dtype=np.float32))
    s32 = to_js(start_s.astype(np.float32).ravel())
    await gpu.settle(st32, v32, s32, SWEEPS)  # noqa: F704  (compiles the pipelines)
    gpu_ms = np.inf
    for _ in range(5):  # the best of five: the GPU's round trip varies by milliseconds
        t = time.perf_counter()
        got = await gpu.settle(st32, v32, s32, SWEEPS)  # noqa: F704
        gpu_ms = min(gpu_ms, 1e3 * (time.perf_counter() - t) / lives)
    on_gpu = np.asarray(got.to_py(), dtype=np.float64).reshape(lives, n)
    if gpu.errors.length:
        raise RuntimeError("WebGPU: " + " | ".join(str(e) for e in gpu.errors.to_py()))
    gap = float(np.abs(on_gpu - s).max())
    row = {"lives": lives, "neurons": n, "synapses": int(g0.connectome.synapses), "sweeps": SWEEPS,
           "one_anatomy": bool(same), "wasm_lives_ms_per_life": round(wasm_ms, 3),
           "wasm_alone_ms_per_life": round(lone_ms, 3),
           "webgpu_ms_per_life": round(gpu_ms, 4),
           "gpu_vs_wasm": round(min(wasm_ms, lone_ms) / gpu_ms, 2),
           "largest_gap_float32_vs_float64": gap}
    rows.append(row)
    failures += not gap < TOLERANCE
    print("GPU " + json.dumps(row), flush=True)
    if SWEEPS == 1:  # one sweep in NumPy over the GPU's own edge layout, float64: the check
        pre = anat["pre"].astype(np.int64)
        basal_start, apical_start = anat["basalStart"], anat["apicalStart"]
        w = weights.astype(np.float64)
        for life in range(min(lives, 2)):
            s0 = start_s[life]
            basal = np.array([w[life, basal_start[i]:basal_start[i + 1]]
                              @ s0[pre[basal_start[i]:basal_start[i + 1]]] for i in range(n)])
            raw = np.array([w[life, apical_start[i]:apical_start[i + 1]]
                            @ s0[pre[apical_start[i]:apical_start[i + 1]]] for i in range(n)])
            basal = basal + standing[life]
            g = graphs[life]
            x = basal.copy()
            if g.apical is not None:
                for part in g.apical.members:
                    b2, r2 = np.mean(basal[part] ** 2), np.mean(raw[part] ** 2)
                    x[part] += (g.apical.fraction * np.sqrt(b2) / np.sqrt(
                        r2 + g.apical.floor ** 2 * b2 + 1e-24)) * raw[part]
            act = g.neuron_model.activation(x * g.neuron_model.dt)
            work = np.asarray((await gpu.read("work")).to_py())  # noqa: F704
            got_basal = work[: lives * n].reshape(lives, n)[life]
            print("  basal gap (float32)", float(np.abs(got_basal - basal).max()), flush=True)
            print("check", life, "numpy vs wasm", float(np.abs(act - s[life]).max()),
                  "numpy vs gpu", float(np.abs(act - on_gpu[life]).max()),
                  "worst", int(np.argmax(np.abs(act - on_gpu[life]))), flush=True)
engine.finish(not failures)
