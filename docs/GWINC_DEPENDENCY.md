# Can we drop the gwinc dependency?

Short answer: **yes for the model, no for the reference budgets.** The
dependency is not one thing, and the right move is different for each half.
Vendoring the model half is a clear win and costs about 1,100 lines. Vendoring
the other half means adopting 6,100 lines of noise physics this project does
not develop and cannot validate.

All numbers below are measured against `gwinc 0.6.2` as installed in the
`wield` environment, not estimated.

## What the repo actually uses

| gwinc surface | used by | what for |
|---|---|---|
| `Struct` | edges, simlib, common, FilterCavity, models, pi, optics | YAML load + attribute-access dict |
| `const.c`, `const.hbar` | edges, FilterCavity, the budget maths | two physical constants |
| `ifo_power(ifo)` | `intsqz/common.py::standardize_params` | arm/BS power, finesse, recycling factor |
| `dhdl(f, L)` | `intsqz/test_CCwIntSqz.py::intSqzQuantum` | strain↔length conversion |
| `arm_cavity(ifo)` | `tf_lib.py` | arm beam sizes |
| `noise.quantum_lib` | `models/`, `optics/test_simple_cavities.py` | `adjoint`, `Vnorm_sq`, `mats_planewave` |
| `load_budget(path).ifo` | every intsqz model | **loading a YAML file** |
| `load_budget(name).run()` | plots only | A+/A#/CE2 reference curves |

The last two rows are the same function doing two completely unrelated jobs.
That is the whole story.

## Finding 1 — `load_budget(...).ifo` is a pure YAML merge

Every model does this:

```python
budget = gwinc.load_budget(fpath_join('AhatTest.yaml'))
ifo = budget.ifo          # ...and never touches `budget` again
```

The `Budget` object is instantiated, drags in the entire noise machinery as an
import side effect, and is then discarded. Verified directly: re-implementing
just the `+inherit` merge with `Struct.from_file` reproduces the result
exactly.

```
ifo identical via pure yaml merge : True
```

So on the model path, gwinc is a YAML loader with a 6,000-line import.

## Finding 2 — the inherit chain leaves the repo exactly once

```
AhatTest.yaml   -> +inherit: Aplus.yaml        (local)
  Aplus.yaml    -> +inherit: 'Aplus'           (built-in gwinc budget)
    gwinc/ifo/Aplus/ifo.yaml                   336 lines  <-- the only external node
```

Every other IFO config in `fromgwinc/intsqz/` (`Ahat*`, `AhatSh*`, `AplWide*`,
`Asharp*`, `AplusTest`) chains through those two files. One 336-line parameter
file is the entire external data dependency of the model path.

## Finding 3 — every `.run()` is a comparison curve

Eleven `.run()` call sites, all of them producing reference traces to draw
underneath the model result: A+ quantum, A+ classical, A+ wideband, A♯, CE2,
`CoatingBrownian`, `SuspensionThermal`.

None feeds a model input. None is touched by the regression guard. They are
static: they depend only on a config file and a frequency grid, never on
anything the user is varying.

Running one costs:

```
gwinc submodules imported by load_budget + run : 19
their total source lines                       : 6102   (of 8171 in gwinc)
noise terms computed                           : CoatingBrownian, CoatingThermoOptic,
                                                 Newtonian, Quantum, ResidualGas, Seismic,
                                                 SubstrateBrownian, SubstrateThermoElastic,
                                                 SuspensionThermal
```

## The proposal, costed

### Vendor (recommended)

| piece | lines | notes |
|---|---|---|
| `struct.py` | 617 | verbatim; drop the `.mat`/`.m` branch and it is ~550 |
| `const.py` | 32 | trivial |
| `ifo_power`, `dhdl`, `arm_cavity` | 118 | self-contained, need only numpy + `const` + `Struct` |
| `gwinc/ifo/Aplus/ifo.yaml` | 336 | data, not code |
| **total** | **~1,100** | |

`noise/quantum_lib.py` (469 lines) is a separate question: `models/` and
`optics/test_simple_cavities.py` import `adjoint`, `Vnorm_sq` and
`mats_planewave` from it. We already hold a byte-identical copy at
`attic/noise/quantum_lib.py`. `adjoint` and `Vnorm_sq` already exist in
`sflu_components.lib`, so only `mats_planewave` genuinely needs rescuing — and
only for two `models/` files that the plan already treats as a frozen
historical corner.

### Do not vendor

The budget engine. 6,102 lines implementing nine independent noise
disciplines — coating Brownian, coating thermo-optic, Newtonian gravity
gradient, substrate Brownian and thermo-elastic, suspension thermal, residual
gas, seismic. This project neither develops nor tests any of them, and there
is no way to validate a vendored copy that has drifted. Taking permanent
ownership of that to draw a comparison line on a plot is a bad trade.

Instead, treat reference curves as **data**:

* cache them as `.npz` next to the examples, keyed by config and frequency grid;
* regenerate with `tools/refresh_reference_budgets.py`, which imports gwinc;
* make gwinc an **optional dev dependency**, needed to refresh the curves and
  nothing else.

The runtime dependency then disappears from the model path entirely, and the
one remaining use is a deliberate, occasional, human-triggered refresh.

## The real reason to do this

Not line count. **Reproducibility.**

`gwinc.load_budget('Aplus')` today silently reads whatever is in
`site-packages`. The base parameters of every published budget in this repo
therefore depend on which pygwinc happens to be installed, and nothing in the
repo records which one that was. This is not hypothetical — the earlier
`refactor/intsqz` work documented that `kuns/superQKwieldSS` and pygwinc master
disagree on:

| parameter | kuns fork | master |
|---|---|---|
| `Curvature.ITM` | 1970 | 1940 |
| `Curvature.ETM` | 2192 | 2245 |
| `ifo_power()` conditional | `if pin is not None` | `if parm is None` |

Those change results. Vendoring the 336-line base parameter file turns "which
pygwinc did you have installed?" into a line in `git log`. That is worth more
than the dependency removal itself, and it is the same class of problem as the
`PYTHONHASHSEED` finding: a published number that quietly depends on ambient
state.

Two smaller benefits come along:

* it unblocks the question left open in `REFACTOR_PLAN.md` — the repo no longer
  has to decide "which pygwinc does this target?" in order to run;
* it drops `from scipy.io.matlab.mio5_params import mat_struct`, which is
  deprecated and scheduled for removal in SciPy 2.0. That import is a live
  time-bomb in `struct.py` today and the vendored copy simply will not have it.

Licensing is a non-issue: pygwinc is released under the **Unlicense** (public
domain). No attribution requirement, no copyleft, no friction.

## Costs, stated honestly

* **Vendored parameters stop tracking upstream.** If pygwinc corrects an A+
  parameter, we will not get it for free. For a reproducibility-focused
  modelling repo this is the point rather than a regression, but it should be a
  conscious choice, and each vendored file should record the upstream commit it
  came from.
* **`ifo_power` has a known fork divergence** (the conditional above).
  Vendoring forces us to pick one. Pick deliberately, record why, and let the
  regression guard pin the consequence.
* **Do not re-implement `Struct`.** Its `update(overwrite_atoms=False,
  clear_test=...)` merge semantics are subtle and the `+inherit` chain depends
  on them exactly. Copy the file; do not paraphrase it.
* **Reference curves become slightly stale artifacts.** Cached `.npz` must be
  regenerated when a config changes, and the frequency grids currently differ
  between examples (`geomspace(10, 30e3, 1000)`, `(10, 15e3, 1000)`,
  `(30, 10e3, 1000)`), so either cache per grid or standardise the grids first.

## Recommended sequencing

This is Stage 6 in `REFACTOR_PLAN.md`, and it should come **after** Stage 3.
Stage 3 moves the models into an importable package, which is what makes the
lib/params/model seam real; vendoring along that seam afterwards is a much
smaller diff than vendoring across today's tangle.

1. vendor `struct.py`, `const.py`, the three helpers, and `Aplus/ifo.yaml`
   into `sflu/params/_gwinc/`, each with a provenance header — guard green;
2. repoint `Struct` / `const` / helper imports; delete the unused
   `from gwinc.struct import Struct` in `sflu_components/edges.py` while
   passing — guard green;
3. replace `load_budget(path).ifo` with a ~40-line `load_ifo(path)` implementing
   `+inherit` — guard green, and this is the step the guard exists for;
4. cache the reference curves, move gwinc to an optional dev dependency;
5. `models/`'s `mats_planewave` use is the last holdout — resolve it or freeze
   `models/`.

Steps 1–3 are the substance and are perhaps a day's work. Step 4 is mostly
deciding on frequency grids.

## On shims

The `fromgwinc/intsqz/{lib,optics}.py` re-export shims from Stage 2 are
transitional, not the destination: Stage 3 updates the call sites and deletes
them. Vendoring should not introduce a replacement shim layer either — the
vendored modules should be imported directly by their new names, with the
old `gwinc.*` imports removed rather than forwarded.
