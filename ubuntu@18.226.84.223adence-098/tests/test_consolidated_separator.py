from __future__ import annotations

import numpy as np
import pytest

from cadence.memory import SynapticMemory
from cadence.stream import PatternSeparator


def test_consolidation_uses_the_same_expanded_key_on_write_and_read() -> None:
    separator = PatternSeparator(2, 16, 2, seed=1)
    memory = SynapticMemory(np.arange(2), np.arange(2, 4), separator=separator, consolidation=1.0)
    key, value = np.array([[1.0, 0.0]]), np.array([[0.2, 0.8]])
    memory.observe(key, value)
    assert memory.consolidated.shape == (16, 2)
    assert np.allclose(memory.recall(key), value)
    memory.reset(1)
    assert np.allclose(memory.recall(key), value)


def test_persistent_record_rejects_drifting_separator_coordinates() -> None:
    with pytest.raises(ValueError, match="fixed separator coordinate"):
        SynapticMemory(
            np.arange(2), np.arange(2, 4), separator=PatternSeparator(2, 16, 2, center=0.9)
        )
