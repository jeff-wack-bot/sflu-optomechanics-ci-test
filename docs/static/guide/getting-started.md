# Getting started

## Environment

The code runs in a conda environment, because `slycot` (a compiled dependency
of `wield.control`) is easiest to get from conda-forge. One script builds it:

```bash
./setup.sh                 # creates the 'wield' env, clones deps/ and installs them
conda activate wield
```

`setup.sh` is idempotent. It clones the `wield` packages into `deps/` over
anonymous HTTPS and installs them in editable mode, so `git pull` inside
`deps/<name>` is all an update takes. LaTeX (`texlive-latex-extra`,
`cm-super`) and `poppler-utils` are only needed to render the figures of the
older examples and to build these docs.

`gwinc` is **optional**. The models use only a small vendored part of it
(`sflu._vendor.gwinc`). The full package is needed only by the examples that
draw gwinc's reference budgets underneath their own curves.

## Everyday commands

Every target pins `PYTHONHASHSEED=0` (see [Conventions](conventions.md#reproducibility)).

| command | what it does |
|---|---|
| `make test` | the whole example suite |
| `make guard` | check the reference models against the committed baselines, exactly |
| `make guard-ci` | the same, with cross-machine tolerances (what CI runs) |
| `make baseline-local` then `make guard-local` | record a reference on *this* machine, then check exactly against it (use around any refactor) |
| `make survey` | list modules that cannot be imported |
| `make docs` | run the examples and regenerate this site |
| `make serve` | preview the site locally |

## Run one paper reproduction

```bash
PYTHONHASHSEED=0 pytest -s --plot papers/test_klmtv2001.py
```

The test asserts that the SFLU model agrees with the paper's closed-form
results. `--plot` keeps the figures, which land in `papers/tresults/`.

## Use a model from Python

```python
import numpy as np
from sflu.papers import klmtv2001 as kl

F_Hz = np.geomspace(10, 1e3, 200)
T = kl.arm(F_Hz, Io_over_Isql=1.0)                        # transfer matrices
noise = kl.sqrt_Sh_over_hSQL(T, kl.sflu_angle(np.pi / 2))  # phase readout, in h_SQL(gamma)
```

[Writing a model](writing-a-model.md) shows how such a model is put together.
