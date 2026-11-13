# #332, protocol 2: selection scored on more worlds, and E2 read by direction

Declared before any life of protocol 2. The world, the life, the readings and the thresholds are
protocol 1's ([PROTOCOL.md](PROTOCOL.md)) except where this file says otherwise. The selected
genome is frozen in `benchmarks/parrot/selected_2.json` (and random search's best in
`random_search_2.json`) before the first gated life.

## What changes, from protocol 1's located terms

1. **Selection scored on more worlds, its parents scored again.** Protocol 1's lineage scored a
   genome on two worlds once and took its best single score; that genome read below the
   hand-set brain on fresh founders. Protocol 2's lineage (`evolve.py --rescore`): 12
   generations of 24, four worlds a genome, the 6 best kept and scored again on each new
   generation's worlds beside their offspring; the final parents scored once more on fresh
   worlds; the selected genome is the best of them by its mean over every scoring. Seed 33292000.
   Random search at equal evaluations by the same loop (fresh draws in place of offspring).
2. **E2 read by direction.** Protocol 1's E2 asked for one held place within 0.10 m of the
   thing, which holds a distance margin, so silencing the distance neurons took E2 with it.
   Protocol 2's E2 reads whether a thing seen by the left eye alone, the right eye alone and
   both is held as exactly one thing in its direction: each held place goes to the thing whose
   bearing from the perch is nearest its own, and a thing passes where exactly one place goes to
   it, within 10 degrees of its bearing (`chamber.one_by_bearing`). E7's silenced reading asks
   this E2 to stay within .10 of the intact. Protocol 1's E2 is reported beside it, with no gate.

## Founders and arms

Founders 33200100 to 33200111 (twelve, fresh). Arms: selected, hand-set, random wiring of the
same size, random search's best, and the CNN and the ViT at matched parameters (E5). The arms at
ten times the parameters are left out. `python -m benchmarks.parrot.gated --protocol 2`.

## Gates

As protocol 1's table, with E2 (selected arm) at .85 by direction (mean, and 10 of 12
founders), and E7 silenced with the bearing within 1 degree and E2 by direction within .10 of
the intact. E6 is read on the page's composed whole brain in headless Chromium (Pyodide), the
moment's median time, 100 ms or less.

## Expectations, from protocol 1's readings (founders 33200000 to 33200011)

E1 seen .77 in every brain arm (things beyond the reach of the binocular overlap are placed by
one eye); the hand-set brain's E3 .069; E5's controls at .23 to .24.
