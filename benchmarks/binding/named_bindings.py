"""Named bindings on shared value coordinates: the fresh-bindings trials, encoded so that a
mixture of the two facts cannot answer.

``relevance/bindings.py`` shows A's bits and B's bits on separate coordinates, so a read that
returns both facts at once still holds the asked one. Here both facts write their bits on the
same coordinates, ``2 i + v`` for bit ``i`` of value ``v``, and each fact event carries its
name (``name-A`` or ``name-B``); the query carries an ask marker (``ask-A`` or ``ask-B``)
that differs from both names. Where the two facts' bits differ, the answer needs the asked
fact's own moment: the question shares nothing with it, and an even mixture of the facts
holds both values of that bit. Trials, held-out pairs, panels, scores and the gate are those
of ``bindings``; only the encoding changes. Nothing here changes the library.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _base() -> Any:
    """A private copy of ``relevance/bindings.py``: its scores read this module's encoding."""
    name = "named_bindings_base"
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(HERE.parent / "relevance"))
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "relevance" / "bindings.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def width(bits: int) -> int:
    """Inputs: the shared value coordinates (two per bit), name-A, name-B, neutral, ask-A,
    ask-B."""
    return 2 * bits + 5


def inputs(
    rows: np.ndarray, bits: int, neutral_events: int = 1, *, exposed: bool = False
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Prefix events, the query event and the answer of trials ``(A, B, order, query)``."""
    if exposed:
        raise ValueError("the named encoding has no exposed arm")
    base = _base()
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 4 or not len(rows) or rows.dtype.kind not in "iu":
        raise ValueError("rows hold integer A, B, order and query")
    a, b, order, query = rows.T
    if ((a < 0) | (a >= 2**bits) | (b < 0) | (b >= 2**bits)).any():
        raise ValueError(f"facts are values of {bits} bits")
    n, columns, w = len(rows), np.arange(bits), width(bits)
    fa, fb = base.unpack(a, bits), base.unpack(b, bits)
    shown_a, shown_b = np.zeros((n, w)), np.zeros((n, w))
    np.put_along_axis(shown_a, 2 * columns + fa, 1.0, axis=1)
    np.put_along_axis(shown_b, 2 * columns + fb, 1.0, axis=1)
    shown_a[:, 2 * bits] = 1.0
    shown_b[:, 2 * bits + 1] = 1.0
    first = np.where(order[:, None] == 0, shown_a, shown_b)
    second = np.where(order[:, None] == 0, shown_b, shown_a)
    prefix = [first, second]
    neutral = np.zeros((n, w))
    neutral[:, 2 * bits + 2] = 1.0
    prefix.extend(neutral.copy() for _ in range(neutral_events))
    ask = np.zeros((n, w))
    ask[np.arange(n), 2 * bits + 3 + query] = 1.0
    target = np.where(query[:, None] == 0, fa, fb)
    return prefix, ask, target


def module() -> Any:
    """``bindings`` with this encoding: what ``index_runs`` reads as the fresh task."""
    base = _base()
    base.width, base.inputs = width, inputs
    return base
