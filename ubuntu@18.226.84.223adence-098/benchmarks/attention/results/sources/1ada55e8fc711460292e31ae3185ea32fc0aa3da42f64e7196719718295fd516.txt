"""Optional CSR transport without a batch-by-synapse message array.

Rows are receiving neurons and columns are publishing neurons. The connectome already
orders synapses by (post, pre), so its topology needs no sorting or dense matrix.
SciPy is part of ``cadence-net[fast]``; without it the brain keeps its NumPy sum.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from .connectome import Connectome


@lru_cache(maxsize=1)
def _csr_type() -> Any:
    try:
        from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
    except ImportError:
        return None
    return csr_matrix


def transport(connectome: Connectome, weights: np.ndarray) -> Any:
    """A sparse matrix sharing current weights and cached immutable topology, or None.

    Each brain owns its matrix wrapper: replacing parameters on a copied
    brain must not replace the old brain's data. In-place changes to a
    weights array remain visible through SciPy's shared data view.
    """
    constructor = _csr_type()
    if constructor is None:
        return None
    topology = connectome.__dict__.get("_csr_topology")
    if topology is None:
        small = max(connectome.n, connectome.synapses) <= np.iinfo(np.int32).max
        dtype = np.int32 if small else np.int64
        indices = connectome.pre.astype(dtype, copy=False)
        indptr = np.empty(connectome.n + 1, dtype=dtype)
        indptr[0] = 0
        np.cumsum(np.bincount(connectome.post, minlength=connectome.n), out=indptr[1:])
        topology = indices, indptr
        connectome.__dict__["_csr_topology"] = topology
    indices, indptr = topology
    return constructor((weights, indices, indptr), shape=(connectome.n, connectome.n), copy=False)
