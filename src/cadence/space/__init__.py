"""Where things are: places held while out of view. Opt-in, imported by name.

``cadence.space`` is imported only where a program names it (``from cadence.space import
Space``); ``import cadence`` loads none of it. ``place`` holds the pieces: the regions a place
brain is made of (``Interface``, ``anatomy``, ``compose``) and ``Space``, which holds a place per
thing between moments, moves it by what the brain answers and teaches the brain by what the
camera reports. The guide is docs/space.md.
"""

from .place import Interface, Space, Tuned, anatomy, compose

__all__ = ["Interface", "Space", "Tuned", "anatomy", "compose"]
