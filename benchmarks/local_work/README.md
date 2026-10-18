# Local work: does repair follow the disturbance?

This instrument compares the library's full settlement (`Brain.act`) with a dependency-aware
local repair prototype, written here on top of the library's settle and residual primitives,
on one composed brain that holds several independent acquired skills. One skill is disturbed
(a new cue, or one lesson); the others' inputs stay. Both lives must reach the same
qualified answer; the instrument counts what each did.

* [LOCAL_WORK_PROTOCOL.md](LOCAL_WORK_PROTOCOL.md): the hypothesis, the anatomies, the
  acquisition, the schedule, the scheduler rule, the counting conventions, the gates and the
  campaign grids, declared before any run.
* [local_work.py](local_work.py): the instrument (skills, the dense and block-sparse
  anatomies, acquisition, `LocalRepair`, the reference arms, one case and its summary).
* [run.py](run.py): the campaign runner; [tables.py](tables.py): markdown tables from the
  summaries.
* [recurrent.py](recurrent.py): conventional recurrent baselines on the same skills and the
  same lives (an online GRU, and an oracle-routed bank of one small GRU per skill as a
  privileged control), with their multiply-accumulates per decision.
* `tests/test_local_work.py` and `tests/test_local_work_baselines.py`: the block anatomy,
  the repair's contracts, determinism, the baselines' gradients and work counts, the
  per-slot lesson and the damped fallback.

Measured results stay with their runs, outside this repository. Nothing here establishes an
energy figure, a hardware claim or a learned capability; it measures the work of the current
solver and of one prototype scheduler on one acquired graph family.
