# attic — quarantined code

Nothing in this directory can be imported. Every file here raised
`ModuleNotFoundError` or `ImportError` on a real import attempt against the
`wield` environment before it was moved (see `tools/regression/import_survey.py`).

It is kept, not deleted, because some of it is a useful record of where the
models came from. It is moved out of the live tree because a reader cannot
otherwise tell the vendored corpse apart from the running code — which was the
main obstacle to reading this repository's dependency structure.

Everything arrived here via `git mv`, so `git log --follow <file>` still works.

Nothing outside this directory imports anything inside it. `pytest.ini`
excludes `attic/` from collection, and `import_survey.py` excludes it from the
live-code survey.

## Contents

### `optomechanicalmodels/` — vendored `gwinc` subtree (~1,490 lines)

Copied out of the `kuns/superQKwieldSS` pygwinc fork, where it lives *inside*
the `gwinc` package. Its imports are relative to that package:

```python
from ..struct import Struct        # -> gwinc.struct, not fromgwinc.struct
from ..ifo.noises import ifo_power
from .. import nb, const
from ..suspension import precomp_suspension
```

Dropped into `fromgwinc/` those resolve to `fromgwinc.struct`, `fromgwinc.nb`,
… which do not exist, so the whole subtree has never been importable here.

`fromgwinc/intsqz/{lib,optics,common,FilterCavity}.py` are edited copies of
four of these files and *are* live; they are what the internal-squeezing models
actually use. The relationship is documented in `docs/DEPENDENCIES.md`.

Includes `test/test_FP.py`, byte-identical to `fromgwinc/intsqz/test_FP.py`.

### `noise/` — `quantum2.py`

Same problem: `from ..struct import Struct`.

`quantum_lib.py` used to sit here too. It turned out to be load-bearing: six
example modules imported it as `gwinc.noise.quantum_lib`, and that module is
**not part of any gwinc release** -- the installed distribution's RECORD does
not list it, so it had been hand-copied into `site-packages/gwinc/noise/` on
one machine. CI proved it by failing with `No module named
gwinc.noise.quantum_lib`. It is now vendored properly at
`sflu_components/quantum_lib.py`, with the one relative import repaired.

### `intsqz_quantum_lib.py` (469 lines)

Was `fromgwinc/intsqz/quantum_lib.py`. Byte-identical to `noise/quantum_lib.py`
(`diff` reports no differences), and broken the same way. Superseded by
`fromgwinc/intsqz/lib.py`, which provides the same matrix helpers via
`MatrixLib` instead of the fixed 2/4/6-dimensional `mats_*` structs.

### `ifo_packages/` — six vendored gwinc IFO definitions

The `__init__.py` of `fromgwinc/{aLIGO,Aplus,CE1,CE2silica,CE2silicon,Voyager}/`
(355 lines), each of which does `from gwinc.noise.quantum2 import ...`, absent
from the installed pygwinc.

The `ifo.yaml` beside each one joined them in Stage 4, once measurement showed
they are **stale and unused**:

* nothing reads them. `gwinc.load_budget('Aplus')` in the live models resolves
  to `site-packages/gwinc/ifo/Aplus`, not to this repository. The single
  reader was `test_FP.py`, via a relative path with one `..` too many, so it
  had been failing rather than reading them;
* they disagree with the installed pygwinc — `Aplus/ifo.yaml` by 29 lines,
  including `Curvature.ITM: 1970 / ETM: 2192` against master's `1940 / 2245`.
  Those are the **kuns-fork** values documented in `REFACTOR_PLAN.md`. So the
  repository was carrying one set of parameters while computing with another;
* `fromgwinc/Aplus/ifo.yaml` is even mislabelled, opening
  `# GWINC aLIGO interferometer parameters`.

Kept because they record what the kuns fork used, which Stage 6 will want when
it picks a base parameter set to vendor deliberately.

### `ifo_superseded/` — four retired configs

`Ahat20`, `Ahat25`, `AhatSh20`, `AhatSh25`, formerly `fromgwinc/intsqz/old/`.
Zero references from live code; the only mention of `Ahat25` is a commented-out
line. Superseded by the `Ahat17` / `Ahat22` / `Ahat30` sets in
`sflu/params/ifo/`.

### `test_CCwIntSqz.py_` (855 lines)

A trailing-underscore backup of an older `test_CCwIntSqz.py`. The trailing
underscore keeps Python and pytest from seeing it. Superseded by
`fromgwinc/intsqz/test_CCwIntSqz.py`; kept only as history.

## Not moved here

`optics/test_DRFPMI.py` also fails to import, but for a different reason: it
needs one optional upstream module (`gwinc.plant` on the superQK fork, or
`gwinc.noise.quantum2` on master). The code itself is fine and 1,107 lines
long, so instead of exiling it, it now skips at module level with a message
naming what is missing. Install a pygwinc that provides either module and it
comes back.
