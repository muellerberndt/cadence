"""Benchmark tests that once read each other's tables and modules pass in either order (#306):
one pytest process runs the pair one way and then the other."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAIRS = {
    # a knock life of the meaning chamber replaced the shared sequence battery's make_brain
    "battery": (
        "benchmarks/grammar/test_meaning.py::test_a_knock_life_leaves_the_shared_battery_as_it_is",
        "benchmarks/sequence/test_sequence.py::test_every_arm_reaches_the_brain",
    ),
    # two chambers that each hold world.py, and a lazy import of controls.py, which five hold
    "modules": (
        "benchmarks/dialogue/test_dialogue.py",
        "benchmarks/memory_scale/test_memory_scale.py::"
        "test_the_logit_terms_sum_to_the_read_and_a_question_writes_no_slot",
        "benchmarks/letters/test_speech.py::test_a_stopped_control_continues_from_its_last_pass",
    ),
}


@pytest.mark.skipif(not (ROOT / "benchmarks").is_dir(), reason="the benchmarks are not here")
@pytest.mark.parametrize("pair", sorted(PAIRS))
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_benchmark_tests_pass_in_either_order(pair, order):
    nodes = list(PAIRS[pair]) if order == "forward" else list(reversed(PAIRS[pair]))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])])}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", *nodes],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=900,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
