# The brain in a browser page

The library ships the brains, not a page: `Connectome`, the patches, and
`record_settlements`, which captures the actual graph iterations behind an
answer so an application can draw them.

<!-- not-run: the page supplies its own brain and observation -->
```python
from cadence import record_settlements

with record_settlements() as settlements:
    action = brain.live(observation)
sweeps = [step.state for step in settlements]
```

Each entry carries the state and residual of one sweep, with the operation that
requested it. Recording adds overhead to every solve, so enable it for a display
or an audit, not for a measured run. See
[record every settling step](api.md#record-every-settling-step) for the full
contract.

[cadence-demos](https://github.com/muellerberndt/cadence-demos) holds the
application examples. Eyes, Walkers and Connectome run a brain in the browser
tab itself; Rover Lab, Patch World and Atari Arcade draw the patches and their
prediction errors while they settle.
