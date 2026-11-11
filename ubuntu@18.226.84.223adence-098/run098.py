"""The declared re-read of cadence#283 on 0.97.0: every life of the protocol in one pool, the
longest first; one writer."""
import argparse, importlib.util, json, os, sys
from multiprocessing import get_context
from pathlib import Path

for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(n, "1")
spec = importlib.util.spec_from_file_location("chamber", "benchmarks/delayed_credit/chamber.py")
ch = importlib.util.module_from_spec(spec); sys.modules["chamber"] = ch; spec.loader.exec_module(ch)

F = list(range(2840, 2846)); B = [256, 1024, 2048, 4096]; T = ["tmaze", "keydoor"]
CAND = "slow_held_entropy"


def ns(**kw):
    base = dict(kinds=["brain"], arms=["default"], tasks=T, delays=[10, 50, 200], founders=F,
                control=["none"], budgets=B, settings=None, learner=None, tag=None,
                driver="step", match_arm=None)
    base.update(kw)
    return argparse.Namespace(**base)


plan = [
    ns(arms=[CAND, "slow_held", "slow_episodic_held_entropy"]),
    ns(arms=["slow_episodic_held", "default"], delays=[10, 50]),
    ns(arms=[CAND], delays=[10, 50], control=["yoked", "shuffled", "ablated", "frozen"]),
    ns(arms=[CAND], delays=[200], control=["ablated", "frozen"]),
    ns(kinds=["lstm"], tag="lstm_mc", learner='{"gamma": 1.0, "lam": 1.0}', match_arm=CAND),
    ns(kinds=["lstm"], tag="lstm_gae", match_arm=CAND),
    ns(kinds=["tdlambda"], tag="td_default", match_arm=CAND),
    ns(kinds=["tdlambda"], tag="td_mc", learner='{"gamma": 1.0, "lam": 1.0}', match_arm=CAND),
    ns(kinds=["solver", "random"], budgets=[64]),
]
specs = [s for p in plan for s in ch.specs_of(p)]
out = Path(sys.argv[1])
done = set()
if out.exists():
    done = {ch._identity(json.loads(l)) for l in out.read_text().splitlines() if l.strip()
            and "error" not in json.loads(l)}
specs = [s for s in specs if ch._identity(s) not in done]


def cost(s):
    work = {"brain": 1.0, "lstm": 0.1, "tdlambda": 0.05}.get(s["kind"], 0.0)
    if s.get("control") == "frozen":
        work = 0.0
    return -work * (s["delay"] + 1)


specs.sort(key=cost)
print(len(specs), "lives", flush=True)
if __name__ == "__main__":
    with out.open("a") as handle, get_context("spawn").Pool(int(sys.argv[2]), maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(ch._one, specs):
            handle.write(json.dumps(row) + "\n"); handle.flush()
    print("DONE", flush=True)
