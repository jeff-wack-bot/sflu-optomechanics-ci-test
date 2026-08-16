# What depends on what

A measured map of this repository, not an aspirational one. Every claim below
was produced by importing each module and by running the suite in the `wield`
conda environment on 2026-08-15, commit `68aaf0b`.

Reproduce the import survey with:

```bash
python tools/regression/import_survey.py
```

## Summary

| | files | lines |
|---|---|---|
| live Python (imports, and something uses it) | 37 | ~11,600 |
| dead Python (cannot be imported in this repo) | 15 | ~3,400 |

Roughly **a quarter of the Python in this repo cannot be imported at all.** It
is not "unused but working" — it raises `ModuleNotFoundError` on import. That
is the single biggest obstacle to seeing the dependency structure: a reader
cannot tell the vendored corpse apart from the running code by looking at the
directory tree.

## The live stack

```
                    external
  gwinc (pygwinc, site-packages)   wield.control / wield.bunch / wield.utilities
  numpy scipy matplotlib networkx  [optional: qlance+Optickle+MATLAB, pykat+Finesse]
                       |
  ---------------------+------------------------------------------------
  params               |
      fromgwinc/intsqz/*.yaml            IFO parameters (Ahat*, Aplus*, Asharp*)
          -> gwinc.load_budget(...) -> ifo Struct
          -> fromgwinc/intsqz/common.py::standardize_params(ifo) -> params Struct
                       |
  ---------------------+------------------------------------------------
  lib                  |
      sflu_components/lib.py       MatrixLib: quadrature algebra
      sflu_components/edges.py     MirrorEdge, LinkEdge, RPMirrorEdge, ...
      sflu_components/elements.py  MirrorElement, RPMirrorElement, BeamSplitterElement
      sflu_components/simlib.py    Optickle/Finesse comparison harness
      tf_lib.py                    transfer-function plotting (via conftest fixtures)
                       |
  ---------------------+------------------------------------------------
  model                |
      topology   sflu_CoupledCav()        -> SFLU graph
      plant      CoupledCavity()          -> edge objects -> resultsAC
      budget     intSqzQuantum()          -> resultsAC -> PSD + loss breakdown
                       |
  ---------------------+------------------------------------------------
  example              |
      test_*.py        run a model, assert or plot
```

That is the structure the code *means*. It is not the structure the code
*has*: every box in the `model` row lives inside a file named `test_*.py`.

## The three problems

### 1. There are two copies of the library, and one model uses both at once

`sflu_components/lib.py` and `fromgwinc/intsqz/lib.py` are the same file with
59 lines of difference; `sflu_components/edges.py` and
`fromgwinc/intsqz/optics.py` differ by 404 lines. Both define a class called
`MatrixLib` and classes called `MirrorEdge`, `LinkEdge`, `RPMirrorEdge`.

`fromgwinc/intsqz/test_CCwIntSqz.py` imports from **both**:

```python
from sflu_components.lib import MatrixLib, adjoint, Minv     # copy A
from .lib import MatsHelper, Vnorm_sq, Vnorm_sqA             # copy B
from . import optics                                         # copy B's edges
```

So two distinct `MatrixLib` classes are alive in one process, and the edge
objects default to `mlib=MatrixLib(nhom=0)` bound from copy B at import time.
It happens to work because the two classes are structurally compatible. Nobody
should have to know that.

What copy B actually adds, and all it adds:

| addition | where | what it is |
|---|---|---|
| `Vnorm_sqA(M)` | `lib.py` | `M @ adjoint(M)` instead of `adjoint(M) @ M` |
| `MatrixLib.SQZc` | `lib.py` | squeeze matrix built on `-Id` rather than `+Id` |
| `MatsHelper` | `lib.py` | accumulator for H/T/L transfer-matrix dicts |
| `MirrorEdge(loss_in_transmission=)` | `optics.py` | alternate loss convention |
| `SQZEdge` | `optics.py` | squeezing edge (the internal squeezer itself) |
| `BSEdge` | `optics.py` | beamsplitter edge (used by the FD model) |
| `edgesACSS()` on every edge | `optics.py` | state-space rather than frequency-response edges |

`models/` then carries a *third* partial copy: `models/matlib.py`,
`models/components.py`, and `models/components2.py` (which differ from each
other by 680 lines).

### 2. Models are only reachable by importing test modules

There is no importable model API. Reuse therefore happens by importing pytest
files from other pytest files:

```python
# fromgwinc/intsqz/test_CCwIntFDSqz.py
from . import test_CCwIntSqz
sfluB_noFC = test_CCwIntSqz.sflu_CoupledCav()

# fromgwinc/intsqz/test_intFDsqz_sweeps.py
from .test_CCwIntFDSqz import (...)
```

This is also why the budget calculation exists three times:

| copy | location | status |
|---|---|---|
| 1 | `test_CCwIntSqz.py:346-573` inside `test_CoupledCav` | live |
| 2 | `test_CCwIntSqz.py:576-716` in `intSqzQuantum` | live, ~130 lines identical to copy 1 |
| 3 | `test_CCwIntFDSqz.py:525-578` inline in `test_CCwIntFDSqz` | **dead** — result is only used by a commented-out plot line |

`test_CCwIntFDSqz.py` already shows the way out: it factored its own budget
into `_compute_intFDsqz_budget()`. That function is the prototype for the
shared one.

### 3. Two unrelated kinds of YAML share a directory

`fromgwinc/intsqz/` contains both:

* **IFO parameter sets** — `Ahat17.yaml`, `AhatTest.yaml`, `Asharp.yaml`, … ,
  consumed by `gwinc.load_budget`; and
* **serialized SFLU graphs** — `CoupledCavity.yaml`, `FilterCavity.yaml`,
  consumed by `SFLU.convert_yamlstr2self`.

They look alike and are not alike. `Asharp.yaml` and `Asharp_wideband.yaml`
additionally exist twice, once here and once at the repository root.

## Dead code inventory

None of the following can be imported. Each was confirmed with a real import
attempt, not by inspection.

| path | lines | why it cannot load |
|---|---|---|
| `fromgwinc/optomechanicalmodels/{common,optics,FilterCavity,CoupledCavity,DRFPMI}.py` | ~1,490 | vendored from a `gwinc` fork; uses `..struct`, `..ifo`, `..nb`, `..suspension`, which only resolve inside the `gwinc` package |
| `fromgwinc/noise/quantum_lib.py`, `fromgwinc/noise/quantum2.py` | 493 | `from ..struct import Struct` → `fromgwinc.struct` does not exist |
| `fromgwinc/intsqz/quantum_lib.py` | 469 | byte-identical to `fromgwinc/noise/quantum_lib.py`, same broken import |
| `fromgwinc/{aLIGO,Aplus,CE1,CE2silica,CE2silicon,Voyager}/__init__.py` | 355 | `from gwinc.noise.quantum2 import ...`, absent from installed pygwinc |
| `optics/test_DRFPMI.py` | 1,107 | needs `gwinc.plant` or `gwinc.noise.quantum2`; blocks collection of the whole `optics/` directory |
| `fromgwinc/intsqz/test_FP.py`, `fromgwinc/optomechanicalmodels/test/test_FP.py` | 30 | byte-identical to each other; both read `../../Aplus/ifo.yaml`, one `..` too many |
| `fromgwinc/intsqz/test_CCwIntSqz.py_` | 855 | trailing-underscore backup of an older `test_CCwIntSqz.py` |

The six `fromgwinc/<IFO>/` directories also carry `ifo.yaml` files that nothing
reads: `gwinc.load_budget('Aplus')` resolves to
`site-packages/gwinc/ifo/Aplus`, not to this repository.

## Test suite, as it stands

Measured on commit `68aaf0b` plus the uncommitted working-tree edits:

```
52 passed, 2 failed, 1 skipped, 14 errors, 1 collection error
```

* **14 errors** — `qlance` (Optickle/MATLAB) not installed. Expected; these are
  external cross-check examples.
* **1 collection error** — `optics/test_DRFPMI.py`, see above. Because pytest
  aborts collection on it, `pytest` with no arguments currently runs nothing.
* **2 failures** — `fromgwinc/intsqz/test_FP.py::test_load_IFO` (bad relative
  path, longstanding) and `models/test_simple_mirror.py::test_sflu_simple_mirror`
  (`'Struct' object has no attribute 'Thr'`, arising in uncommitted working-tree
  changes, not from `68aaf0b`).

## Reproducibility

Model outputs depend on `PYTHONHASHSEED`. See Finding 1 in `REFACTOR_PLAN.md`.
Pin it before comparing any two runs.
