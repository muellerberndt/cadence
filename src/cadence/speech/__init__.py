"""Speech pieces for a recogniser built from the library: opt-in, imported by name.

``cadence.speech`` is imported only where a program names it (``from cadence.speech import
correct``); ``import cadence`` loads none of it. One module per piece: the log-mel front end
(``frontend``), the forced-alignment lesson (``align``), the decoders (``decode``), corrections
heard beside earlier utterances (``correct``) and the streaming reader (``stream``). The guide is
docs/speech.md.
"""
