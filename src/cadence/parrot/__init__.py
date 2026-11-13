"""The parrot: an African Grey on a low perch, its eyes, its body and its brain (opt-in).

``cadence.parrot.world`` holds the scene and the body's joints, ``cadence.parrot.eyes`` renders
what each eye sees. A program that does not import ``cadence.parrot`` gets a library without it.
"""

from .eyes import Renderer, Sight, render
from .world import JOINTS, Body, Eyes, Scene, Thing

__all__ = ["JOINTS", "Body", "Eyes", "Renderer", "Scene", "Sight", "Thing", "render"]
