"""Close the mutation-survivor gaps in ``settle.py``.

A mutation sweep (tree-sitter byte-span mutants, oracle = this suite) left 11 survivors in
``settle.py`` — injected faults no existing test could distinguish. Each test below pins the exact
behavior one survivor broke, so the mutant now dies. The six covered here are the CPU-path survivors;
the remaining five (``available_backends`` CUDA branch and four inside ``_TorchKernel``: the device
select, the ``v.shape[0]`` batch read, the ``adapt is not None`` guard, and the ``-v + total`` update)
are unreachable without ``torch`` installed and belong with the torch conformance test.

Survivor -> test:
  L127 ``i: int = 0``        default batch row in ``fraction_active``  -> test_fraction_active_defaults_to_row_zero
  L173 ``np.zeros(0, ...)``  empty segment starts for a 0-overlap wiring -> test_zero_overlap_wiring_has_no_inbox_segments
  L202 ``log_gain is None``  ``with_parameters`` keeps an unspecified param -> test_with_parameters_preserves_unspecified_log_gain
  L240 ``array.ndim == 1``   integer index array names owners to clamp   -> test_clamp_vector_integer_indices_are_owners
  L260 ``steps: int = 60``   the documented default step count           -> test_settle_default_runs_sixty_steps
  L366 ``if masked: v *= keep`` ablated owner's POTENTIAL stays zero     -> test_mask_zeros_the_potential_of_ablated_owners
"""

from __future__ import annotations

import numpy as np

import cadence as cd


def ring(n: int = 6) -> cd.Wiring:
    return cd.Wiring.from_edges(
        n, pre=list(range(n)), post=[(i + 1) % n for i in range(n)], count=[100] * n
    )


def test_fraction_active_defaults_to_row_zero() -> None:
    # L127: the default batch row is 0. Row 0 is driven, row 1 is empty, so the two rows'
    # fractions differ -- mutating the default to row 1 would return the wrong row.
    engine = cd.Settlement(ring(), cd.GradedRule(gain=0.03))
    batch = engine.settle_batch(
        np.stack([engine.clamp_vector({0: 3.0, 1: 3.0, 2: 3.0}), engine.clamp_vector(None)]),
        steps=60,
    )
    assert batch.fraction_active(i=0) != batch.fraction_active(i=1)  # the fixture discriminates
    assert batch.fraction_active() == batch.fraction_active(i=0)  # default IS row 0, not row 1


def test_zero_overlap_wiring_has_no_inbox_segments() -> None:
    # L173: a wiring with no overlaps has zero inbox segments -- np.zeros(0), not np.zeros(1).
    wiring = cd.Wiring.from_edges(3, pre=[], post=[], count=[])
    engine = cd.Settlement(wiring, cd.GradedRule())
    assert engine._starts.shape == (0,)
    assert engine._owners_with_inbox.shape == (0,)
    st = engine.settle(clamp={0: 3.0}, steps=20)  # still settles; inbox is identically zero
    assert st.activation.shape == (3,)


def test_with_parameters_preserves_unspecified_log_gain() -> None:
    # L202: with_parameters() KEEPS an existing nonzero log_gain when it is not overridden. The
    # `is None` must stay `is None`: negating it would silently zero a parameter the caller kept.
    log_gain = np.arange(6, dtype=float) * 0.1
    engine = cd.Settlement(ring(), cd.GradedRule(), log_gain=log_gain)
    changed = engine.with_parameters(bias=np.full(6, 0.2))  # log_gain deliberately NOT passed
    assert not np.allclose(engine.log_gain, 0.0)  # guard: the kept value is genuinely nonzero
    assert np.allclose(changed.log_gain, engine.log_gain)
    assert changed.bias[0] == 0.2  # and the override that WAS passed took effect


def test_clamp_vector_integer_indices_are_owners() -> None:
    # L240: a 1-D integer array names owners to clamp at full amplitude (not a dense level vector).
    engine = cd.Settlement(ring(6), cd.GradedRule(clamp_amplitude=2.0))
    v = engine.clamp_vector(np.array([0, 1, 2, 3, 4, 5]))  # int, ndim 1, max > 1
    assert np.allclose(v, 2.0)  # every named owner at clamp_amplitude; NOT returned as levels [0..5]


def test_settle_default_runs_sixty_steps() -> None:
    # L260: the documented default is exactly 60 steps (no tolerance -> runs them all).
    engine = cd.Settlement(ring(), cd.GradedRule(gain=0.03))
    st = engine.settle(clamp={0: 3.0})
    assert st.steps == 60


def test_mask_zeros_the_potential_of_ablated_owners() -> None:
    # L366: `if masked: v *= keep` zeros an ablated owner's POTENTIAL every step. Assert on v, not
    # activation: the sibling `s *= keep` would hide the effect on activation, which is exactly why
    # this survived. Owner 0 is both ablated AND driven, so a negated guard lets v[0] accumulate.
    engine = cd.Settlement(ring(), cd.GradedRule(gain=0.03))
    mask = np.ones(6)
    mask[0] = 0.0
    st = engine.settle(clamp={0: 3.0}, steps=60, mask=mask)
    assert st.v[0] == 0.0
