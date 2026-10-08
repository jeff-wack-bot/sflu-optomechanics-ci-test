# SFLU Optomechanics

Linear, frequency-domain models of optomechanical gravitational-wave
interferometers (radiation pressure, signal recycling, squeezing, filter
cavities, internal squeezing, mode mismatch), built as signal-flow graphs
and solved with SFLU from [`wield.control`](https://git.mccullerlab.com/wield/wield-control).

**Documentation:** <https://jeff-wack-bot.github.io/sflu-optomechanics-ci-test/>,
generated from the examples on every push.

```bash
./setup.sh && conda activate wield      # environment (conda: slycot)
make papers                             # paper reproductions, asserted against closed forms
make test                               # the whole example suite
make docs && make serve                 # build and preview the documentation
```

```
sflu/            the package: lib (MatrixLib, edges, elements, topologies,
                 solve, readout) -> params -> models, papers
papers/          reproductions of eight papers (KLMTV, Buonanno & Chen, ...)
optics/ models/  examples; fromgwinc/intsqz/ the internal-squeezing studies
pi/              parametric instability example
tools/           regression guard, import survey, config checks
attic/           quarantined code that does not import, with reasons
docs/            the documentation generator and hand-written guide pages
```

How the code got this shape, and why, is in [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md).
