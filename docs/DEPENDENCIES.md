# What depends on what

A measured map of this repository, not an aspirational one. Every claim below
was produced by importing each module and by running the suite in the `wield`
conda environment on 2026-08-15, starting from commit `68aaf0b`.

Reproduce the import survey with:

```bash
make survey        # or: python tools/regression/import_survey.py
```

## Summary

Before Stage 1 of [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md):

| | modules | lines |
|---|---|---|
| live Python | 37 | ~11,600 |
| dead Python (could not be imported at all) | 15 | ~3,400 |

Roughly **a quarter of the Python could not be imported.** It was not "unused
but working" — it raised `ModuleNotFoundError` on import, and a reader could
not tell the vendored corpse apart from the running code by looking at the
directory tree.

After Stage 1 (dead code quarantined in [`attic/`](../attic/README.md)):

| | modules | lines |
|---|---|---|
| live Python, all importable | 33 | ~13,000 |
| live, skipped for an optional dependency | 1 (`optics/test_DRFPMI.py`) | 1,107 |
| quarantined in `attic/` | 19 | ~3,370 |

`make survey` now reports **0 of 33 live modules cannot be imported**.

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

### 1. Two copies of the library, one model using both at once — fixed in Stage 2

`sflu_components/lib.py` and `fromgwinc/intsqz/lib.py` were the same file with
59 lines of difference; `sflu_components/edges.py` and
`fromgwinc/intsqz/optics.py` differed by 404 lines. Both defined a class called
`MatrixLib` and classes called `MirrorEdge`, `LinkEdge`, `RPMirrorEdge`.

`fromgwinc/intsqz/test_CCwIntSqz.py` imported from **both**:

```python
from sflu_components.lib import MatrixLib, adjoint, Minv     # copy A
from .lib import MatsHelper, Vnorm_sq, Vnorm_sqA             # copy B
from . import optics                                         # copy B's edges
```

Two distinct `MatrixLib` classes were alive in one process, and the edge objects
defaulted to `mlib=MatrixLib(nhom=0)` bound from copy B at import time. It
worked only because the two classes were structurally compatible.

What copy B carried, and all it carried:

| addition | where | what it is |
|---|---|---|
| `Vnorm_sqA(M)` | `lib.py` | `M @ adjoint(M)` instead of `adjoint(M) @ M` |
| `MatrixLib.SQZc` | `lib.py` | squeeze matrix built on `-Id` rather than `+Id` |
| `MatsHelper` | `lib.py` | accumulator for H/T/L transfer-matrix dicts |
| `MirrorEdge(loss_in_transmission=)` | `optics.py` | alternate loss convention |
| `SQZEdge` | `optics.py` | squeezing edge (the internal squeezer itself) |
| `edgesACSS()` on every edge | `optics.py` | state-space rather than frequency-response edges |

**Now:** one implementation in `sflu_components/{lib,edges}.py`, with
`fromgwinc/intsqz/{lib,optics}.py` reduced to documented re-export shims
(540 → 53 and 532 → 35 lines).

```python
sflu_components.lib.MatrixLib is fromgwinc.intsqz.lib.MatrixLib      # True
sflu_components.edges.MirrorEdge is fromgwinc.intsqz.optics.MirrorEdge  # True
```

The merge was verified before it was made: of the 80 edge-map entries both
copies produced, **all 80 were bit-identical**, so the copies differed only in
feature set, never in value. The one behavioural difference — the intsqz copy
always emitted `.fr.l` / `.bk.l`, this one emitted them only on request — is
now explicit, with the intsqz models passing `loss_ports=True` at each mirror.

`models/` still carries a *third* partial copy: `models/matlib.py`,
`models/components.py`, and `models/components2.py` (which differ from each
other by 680 lines). It is used only by `models/` and is left for later.

### 2. Models only reachable by importing test modules — fixed in Stage 3

There was no importable model API, so reuse happened by importing pytest files
from other pytest files:

```python
from . import test_CCwIntSqz              # in test_CCwIntFDSqz.py
from .test_CCwIntFDSqz import (...)       # in test_intFDsqz_sweeps.py
```

which is also why the budget calculation existed **four** times:

| copy | location | status |
|---|---|---|
| 1 | inline in `test_CoupledCav` | live |
| 2 | `intSqzQuantum` | live, 73 lines identical to copy 1 |
| 3 | inline in `test_CCwIntFDSqz` | dead — result only used by a commented-out plot |
| 4 | `test_intFDsqz_sweeps.py::_compute_d_sense_CC` | live, found during the merge |

**Now:** one implementation in `sflu/models/budget.py`, and the models are an
importable package:

```
sflu/params.py                 standardize_params, arm_gouyRT
sflu/models/budget.py          accumulate(), quantum_budget()
sflu/models/coupled_cavity.py  topology + plant + intSqzQuantum()
sflu/models/int_fd_sqz.py      topology + plant + intFDsqzQuantum()
sflu/models/filter_cavity.py   external squeezing filter cavity
sflu/models/topologies/*.yaml  serialized SFLU graphs
```

The example files are examples again: `test_CCwIntSqz.py` 891 → 235 lines,
`test_CCwIntFDSqz.py` 640 → 130.

Deduplicating turned up a bug that had been invisible in the duplication:
`intSqzQuantum` returned `LB['ASport']` aliased to `total`, so the reported
AS-port term was silently the running total. It is preserved exactly (behind
an explicit `alias_ASport` flag) rather than fixed in a structural change —
see `REFACTOR_PLAN.md`, Stage 3.

### 3. Two unrelated kinds of YAML sharing a directory — fixed in Stage 4

`fromgwinc/intsqz/` contained both:

* **IFO parameter sets** — `Ahat17.yaml`, `AhatTest.yaml`, `Asharp.yaml`, … ,
  consumed by `gwinc.load_budget`; and
* **serialized SFLU graphs** — `CoupledCavity.yaml`, `FilterCavity.yaml`,
  consumed by `SFLU.convert_yamlstr2self`.

They look alike and are not alike. `Asharp.yaml` and `Asharp_wideband.yaml`
additionally existed twice, once there and once at the repository root.

**Now:** parameter sets in `sflu/params/ifo/`, reached by name via
`sflu.params.load_ifo()` / `ifo_path()`; topologies in
`sflu/models/topologies/`, beside the model that loads them. The root-level
duplicates were byte-identical and unreferenced, and are gone.

Three sets of stale data went to `attic/` in the same pass, the notable one
being the six vendored `fromgwinc/<IFO>/ifo.yaml`: unread by anything, and
carrying **kuns-fork parameter values** (`Curvature.ITM: 1970 / ETM: 2192`)
while every actual run used installed pygwinc master (`1940 / 2245`). The
repository was storing one set of parameters and computing with another.

## Dead code inventory — resolved in Stage 1

None of the following could be imported. Each was confirmed with a real import
attempt, not by inspection. All have been `git mv`d to `attic/`; see
[`attic/README.md`](../attic/README.md) for the per-file account.

| former path | lines | why it could not load | now |
|---|---|---|---|
| `fromgwinc/optomechanicalmodels/*` | ~1,490 | vendored from a `gwinc` fork; uses `..struct`, `..ifo`, `..nb`, `..suspension`, which only resolve inside the `gwinc` package | `attic/optomechanicalmodels/` |
| `fromgwinc/noise/{quantum_lib,quantum2}.py` | 493 | `from ..struct import Struct` → `fromgwinc.struct` does not exist | `attic/noise/` |
| `fromgwinc/intsqz/quantum_lib.py` | 469 | byte-identical to the above, same broken import | `attic/intsqz_quantum_lib.py` |
| `fromgwinc/{aLIGO,Aplus,CE1,CE2silica,CE2silicon,Voyager}/__init__.py` | 355 | `from gwinc.noise.quantum2 import ...`, absent from installed pygwinc | `attic/ifo_packages/` |
| `fromgwinc/optomechanicalmodels/test/test_FP.py` | 15 | byte-identical twin of `fromgwinc/intsqz/test_FP.py` | `attic/optomechanicalmodels/test/` |
| `fromgwinc/intsqz/test_CCwIntSqz.py_` | 855 | trailing-underscore backup | `attic/test_CCwIntSqz.py_` |

Two files were repaired rather than quarantined:

* `fromgwinc/intsqz/test_FP.py` read `../../Aplus/ifo.yaml`, one `..` too many.
  Corrected to `../Aplus/ifo.yaml`; **now passes.**
* `optics/test_DRFPMI.py` needs `gwinc.plant` or `gwinc.noise.quantum2`, and an
  uncaught `ImportError` there aborted collection for the entire repository. It
  now skips at module level naming the missing dependency. 1,107 lines of
  otherwise-sound model code, so exiling it would have been the wrong call.

The `ifo.yaml` beside each of the six IFO packages **stayed put** in
`fromgwinc/<NAME>/`: it is data, it still loads, and `test_FP.py` reads it.
Note it is not what the published budgets use — `gwinc.load_budget('Aplus')`
resolves to `site-packages/gwinc/ifo/Aplus`, not to this repository.

## Test suite

On `68aaf0b` plus uncommitted working-tree edits, before Stage 1:

```
52 passed, 2 failed, 1 skipped, 14 errors, 1 collection error
```

Collection aborted on `optics/test_DRFPMI.py`, so a bare `pytest` ran **nothing
at all**; the counts above required `--ignore`.

After Stage 1, a bare `pytest` works:

```
53 passed, 1 failed, 2 skipped, 14 errors
```

* **14 errors** — `qlance` (Optickle/MATLAB) not installed. Expected; these are
  external cross-check examples, not repository faults.
* **2 skipped** — one pre-existing, plus `optics/test_DRFPMI.py`.
* **1 failure** — `models/test_simple_mirror.py::test_sflu_simple_mirror`
  (`'Struct' object has no attribute 'Thr'`). This arises in uncommitted
  working-tree changes, not from `68aaf0b`, and is untouched here.

## Reproducibility

Model outputs depend on `PYTHONHASHSEED`. See Finding 1 in `REFACTOR_PLAN.md`.
Pin it before comparing any two runs.
