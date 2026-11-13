"""Words for things: a lexicon learned from naming, and a brain that answers questions about
its room from what its map holds (``docs/words.md``)."""

from .episodes import SOURCES, Clock, Episodes, Records
from .lexicon import LETTERS, Lexicon
from .room import PLACES, Body, Heard, Senses, Words

__all__ = ["LETTERS", "PLACES", "SOURCES", "Body", "Clock", "Episodes", "Heard", "Lexicon",
           "Records", "Senses", "Words"]
