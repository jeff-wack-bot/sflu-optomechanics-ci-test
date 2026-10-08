# How the code is organised

Everything importable lives in one package, `sflu`. Its modules form layers,
and dependencies only run downward: nothing imports from a layer below it,
and nothing in the package imports the example suite.

```
lib       sflu.lib          MatrixLib: quadrature algebra, rotations, squeezing,
                            mode mismatch, promotion to higher-order modes
          sflu.edges        edge objects: physical parameters -> SFLU edge weights
                            (MirrorEdge, LinkEdge, RPMirrorEdge, SQZEdge, BSEdge)
          sflu.elements     graph elements: which ports of an optic connect
          sflu.topologies   reusable graphs: fp_arm, signal_recycled, filter_cavity
          sflu.solve        build a graph, solve DC and AC
          sflu.readout      homodyne readout, squeezed states, noise PSDs
             |
params    sflu.params       IFO parameter sets by name; derived quantities
             |
model     sflu.models       the internal-squeezing models (reference implementation)
          sflu.papers       interferometers from the literature, with closed forms
             |
examples  */test_*.py       run a model, assert, plot -- and become these pages
```

Supporting modules: `sflu.plotting` (transfer-function plots behind the
example fixtures), `sflu.quantum_lib` (an older plane-wave matrix library,
kept for the examples that cross-check against it), `sflu.simlib`
(Optickle/Finesse comparison harness), and `sflu._vendor.gwinc`, the part of
pygwinc the models need.

## What a model is

Every model has the same three parts. The [paper reproductions](../index.md)
keep them visibly apart:

1. **Topology**: an `optics.GraphElement` that says which optics exist and
   how their ports connect. No numbers. See `sflu.topologies`.
2. **Edges**: edge objects that turn physical parameters (transmissions,
   lengths, detunings, masses) into the weights on the graph's edges. See
   `sflu.edges`.
3. **Solve and read out**: `sflu.solve` reduces the graph and returns transfer
   matrices from every input to the readout. `sflu.readout` turns those into a
   noise spectrum for a given homodyne angle and input state.

The reference internal-squeezing models in `sflu.models` predate `sflu.solve`
and spell out the same steps by hand. Their numbers are pinned bit for bit by
`tools/regression/`, so they were moved without being rewritten.

## Where things used to be

The repository grew out of examples. Its history is in
[`REFACTOR_PLAN.md`](https://github.com/jeff-wack-bot/sflu-optomechanics-ci-test/blob/refactor/structure-and-docs/REFACTOR_PLAN.md).
In short:

| old | now |
|---|---|
| models inside `fromgwinc/intsqz/test_*.py` | `sflu.models` |
| `sflu_components/{lib,edges,elements}.py` | `sflu.lib`, `sflu.edges`, `sflu.elements` (old names still import) |
| `tf_lib.py` | `sflu.plotting` (old name still imports) |
| `gwinc.load_budget(path).ifo` | `sflu.params.load_ifo(name)`, no gwinc needed |
| unimportable code mixed in with live code | `attic/`, with a README saying why |

`models/` holds an older, partly separate copy of the matrix library and is
still being worked on, so it was deliberately left untouched.
