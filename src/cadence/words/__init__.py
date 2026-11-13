"""Words for things: a lexicon learned from naming, and a brain that answers questions about
its room from what its map holds (``docs/words.md``)."""

from .lexicon import LETTERS, Lexicon
from .room import PLACES, READS, Body, Words

__all__ = ["LETTERS", "PLACES", "READS", "Body", "Lexicon", "Words"]
