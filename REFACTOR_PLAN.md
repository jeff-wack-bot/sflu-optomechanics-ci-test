# Refactor plan: make the structure legible, generate the docs

Branch: `refactor/structure-and-docs`, cut from `main` at `68aaf0b`.

Two goals, from the request:

1. make the dependency structure (lib / params / model) clear;
2. produce simple documentation, auto-generated in a literate style from the
   examples already in the codebase.

The measured starting state is in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
Read that first; this document only says what to *do* about it.

## The constraint that shapes everything

The code carries very uneven review:

| layer | review | consequence for this refactor |
|---|---|---|
| `sflu_components/`, `quantum_lib`, `optics` | multiple scientists | safe to depend on, unsafe to silently alter |
| model code | one author, only simple cases tested | must not be "cleaned up" and physics-changed in the same step |
| `fromgwinc/intsqz/test_CCwIntSqz.py` | Lee McCuller's last working state | **the reference. Its numbers define correctness.** |

So the rule for every stage below is the same: **a structural change must not
change a single number.** Anything that does is a bug in the move, not an
improvement. Stage 0 exists to make that rule enforceable rather than
aspirational.

## Findings that came out of surveying the code

Three things turned up while measuring, which change what the plan has to do.

### Finding 1 — model output depends on `PYTHONHASHSEED` (up to 3e-3)

Running the identical model twice in two processes gives budgets that differ by
up to **0.3%**. Pinning `PYTHONHASHSEED=0` makes them bit-identical (verified
across four processes; unpinned runs land in one of several distinct states).

Root cause, in `wield-control`, not in this repository:

```
wield/control/SFLU/SFLU.py:76    nodes = set()
wield/control/SFLU/SFLU.py:259   self.nodes = nodes
wield/control/SFLU/SFLU.py:520   def reduce_auto(self):  ...  self.reduce(*self.nodes)
```

`reduce_auto` eliminates graph nodes in Python `set` iteration order. For a set
of strings that order depends on hash randomisation, so each run picks a
different elimination (pivot) sequence, and the reduction is ill-conditioned
enough that the choice is worth 1e-3 in the final budget — far above rounding.

Actions:

* **Now:** pin the seed wherever numbers are compared or published. Done for
  the regression harness, the docs build, and every `Makefile` target. It
  cannot be pinned from `pytest.ini` — CPython fixes the seed at interpreter
  startup — so `conftest.py` warns when a run is unpinned instead (Stage 1).
* **Separately, not in this refactor:** the fix upstream is a deterministic
  order (`self.reduce(*sorted(self.nodes))`) or a deliberate pivot rule.
  This will shift published numbers slightly, so it is a physics decision, not
  a cleanup — raise it with Lee rather than folding it into a move.
* Worth knowing regardless: **a 1e-3 sensitivity to elimination order is a
  conditioning warning.** It may be benign, or it may mean some budget
  components are less converged than they look.

### Finding 2 — the intsqz fork is the *more* capable library copy

The natural assumption is that `fromgwinc/intsqz/{lib,optics}.py` is a stale
fork of `sflu_components/{lib,edges}.py` and should be deleted in its favour.
That is backwards. `sflu_components/edges.py` has:

* no `SQZEdge` at all — the internal squeezer itself, and
* no `edgesACSS()` on any class — i.e. **no state-space path**,

and every internal-squeezing model runs with `use_SS=True`. Unification must
therefore take the union, based on the intsqz copy, and fold in
`sflu_components`' property-based accessors. Deleting the fork would delete the
feature.

Helpfully, `gwinc.const.c == scipy.constants.c` and
`gwinc.const.hbar == scipy.constants.hbar` exactly, so the two copies'
differing constant sources are not a numerical obstacle to merging.

### Finding 3 — one dead module currently disables the whole suite

`optics/test_DRFPMI.py` fails at *collection* (`gwinc.plant` / `gwinc.noise.quantum2`
are absent), and pytest aborts collection on it. A bare `pytest` in this repo
therefore runs **no tests at all**. Quarantining that one file turns the suite
back on, which is the cheapest single win available.

## Stage 0 — safety net and map (done on this branch)

Nothing in the repository's own code was touched.

| added | purpose |
|---|---|
| `tools/regression/capture_baseline.py` | calls the model entry points directly and stores every output array |
| `tools/regression/baselines/intsqz.npz` | 58 arrays: 6 intsqz configs, 1 intFDsqz config, 2 SFLU topologies |
| `tools/regression/test_regression.py` | pytest guard asserting the numbers are unchanged |
| `tools/regression/import_survey.py` | evidence for the dead-code inventory |
| `docs/generate_docs.py` | the literate documentation generator (goal 2) |
| `docs/DEPENDENCIES.md` | the measured dependency map (goal 1, descriptive half) |

The guard was tamper-tested: nudging one stored array by 1 ppm makes it fail
with `CHANGED intsqz/AhatTest/total: max rel diff 1.000e-06`. It runs in ~3 s.

```bash
python -m tools.regression.capture_baseline          # re-baseline deliberately
pytest tools/regression/test_regression.py           # guard  (run before AND after every stage)
```

## Stage 1 — quarantine the dead code (done)

**Risk: none.** Every file moved was one that raised on import, so nothing
could depend on it. This is the change that makes the tree readable.

`git mv`d into `attic/` (history preserved; nothing deleted), with
`attic/README.md` recording where each file came from and exactly why it does
not load:

```
attic/optomechanicalmodels/     vendored gwinc fork subtree, ~1,490 lines
attic/noise/                    quantum_lib.py, quantum2.py
attic/intsqz_quantum_lib.py     byte-identical to attic/noise/quantum_lib.py
attic/ifo_packages/             __init__.py of the six IFO packages
attic/test_CCwIntSqz.py_        trailing-underscore backup
```

Also done, each mechanical and individually reversible:

* fixed `fromgwinc/intsqz/test_FP.py`'s `../../Aplus/ifo.yaml` →
  `../Aplus/ifo.yaml` (one `..` too many). **It now passes**, having failed
  since before this branch;
* `pytest.ini`: added `attic docs deps` to `norecursedirs`;
* `.gitignore`: generated documentation.

### Two deviations from the plan as written

**`optics/test_DRFPMI.py` was repaired, not exiled.** The plan listed it for
`attic/`. On inspection it is 1,107 lines of sound model code blocked by a
single optional import, unlike the genuinely rotted vendored files. It now
skips at module level naming the missing dependency, which unblocks collection
without discarding the code. Install a pygwinc providing `gwinc.plant` or
`gwinc.noise.quantum2` and it returns.

**`PYTHONHASHSEED=0` cannot go in `pytest.ini`.** The plan said to put it
there; that is not implementable. CPython reads the variable at interpreter
startup, so `pytest.ini`, `conftest.py`, and `pytest-env` all run too late —
assigning `os.environ['PYTHONHASHSEED']` mid-process provably does not change
hashing. Instead:

* a `Makefile` exports `PYTHONHASHSEED=0` and wraps the common commands, so
  `make test`, `make guard`, and `make docs` are reproducible by construction;
* `conftest.py` gained a `pytest_report_header` hook that detects
  `sys.flags.hash_randomization` and prints a warning at the top of any run
  where the seed is loose. Detection is the most that layer can do.

**Verification.** `make survey` reports **0 of 33 live modules cannot be
imported**, down from 15 of 52. The regression guard stayed green throughout.
A bare `pytest`, which previously collected nothing, now runs:

```
before (with --ignore):   52 passed, 2 failed, 1 skipped, 14 errors, 1 collection error
after  (no arguments):    53 passed, 1 failed, 2 skipped, 14 errors
```

The 14 errors are the absent optional `qlance`/Optickle dependency. The
remaining failure, `models/test_simple_mirror.py::test_sflu_simple_mirror`,
comes from uncommitted working-tree changes and was not touched.

## Stage 2 — one library, not three (done)

The highest-value structural fix. Split into 2a (the matrix library) and 2b
(the edge library) so each landed with the guard green.

### Guard extended first

The existing baseline only covered the intsqz budgets, but this stage changes
`sflu_components`, whose other consumers are `optics/` and `pi/`. Checking
those revealed that **`optics/test_simple_cavities.py` and
`optics/test_radiation_pressure.py` contain zero assertions** — they only plot.
A bad merge there would have produced a silently wrong figure and a green
suite.

So the baseline was extended, before any merge, to pin `MatrixLib`'s outputs
and every edge class's edge maps, from both copies, at `nhom` 0 and 1:
**58 → 404 arrays.**

### 2a — `MatrixLib`

`fromgwinc/intsqz/lib.py` turned out to be a *strict superset* of
`sflu_components/lib.py`: `diff` reports three pure-addition hunks
(`Vnorm_sqA`, `MatrixLib.SQZc`, `MatsHelper`) and zero lines unique to
`sflu_components`. The merge was therefore the superset, and the fork became a
re-export shim. Guard afterwards: 0 changed, 3 new (the newly reachable
`SQZc` / `Vnorm_sqA` probes).

### 2b — the edge classes

Before merging, the two copies' captured edge maps were compared directly:
**80 of 80 shared entries bit-identical.** `LinkEdge`, `BSEdge`,
`RPMirrorEdge` and the `.r`/`.t` of `MirrorEdge` all agreed exactly, so the
merge could not move a number; the copies differed only in feature set and in
loss-port emission.

The union now lives in `sflu_components/edges.py`. The one real behavioural
conflict was resolved by making it explicit rather than implicit:

* the intsqz copy **always** emitted `.fr.l` / `.bk.l`;
* the `sflu_components` copy emitted them **only** when `loss_ports=True`, and
  its callers depend on that;
* so the merged class keeps `loss_ports=False` as the default, and the ten
  intsqz mirror constructions now pass `loss_ports=True` explicitly. Same
  behaviour, written down instead of implied.

`loss_in_transmission` is kept as a separate parameter: it selects a different
physical convention (`Lhr` taken out of transmission rather than reflection),
not a spelling of `loss_ports`.

The hazards flagged in the original plan resolved as follows. Nothing mutates
edge attributes after construction anywhere in the repo, so properties versus
plain attributes was immaterial; plain attributes won. `gwinc.const.c` and
`scipy.constants.c` are bit-identical, so the differing constant sources were
a non-issue.

**Result.** `fromgwinc/intsqz/lib.py` 540 → 53 lines,
`fromgwinc/intsqz/optics.py` 532 → 35 lines, both now documented shims.

```python
sflu_components.lib.MatrixLib is fromgwinc.intsqz.lib.MatrixLib          # True
sflu_components.edges.MirrorEdge is fromgwinc.intsqz.optics.MirrorEdge   # True
```

Guard exact, suite unchanged at 53 passed / 1 failed / 2 skipped / 14 errors.

`models/{matlib,components,components2}.py` is a third partial copy used only
by `models/`. Left alone deliberately — it is the least load-bearing corner,
and folding it in belongs with Stage 3 or with declaring `models/` a frozen
historical example set.

## Stage 3 — models become importable (done)

The models now live in an importable `sflu/` package and the `test_*.py` files
are examples again.

```
sflu/params.py                 standardize_params, arm_gouyRT
sflu/models/budget.py          accumulate(), quantum_budget()   <- one budget
sflu/models/coupled_cavity.py  sflu_CoupledCav(), CoupledCavity(), intSqzQuantum()
sflu/models/int_fd_sqz.py      sflu_CCwIntFDSqz(), CoupledCavityIntFC(), intFDsqzQuantum()
sflu/models/filter_cavity.py   sflu_FilterCavity(), FilterCavity()
sflu/models/topologies/*.yaml  serialized SFLU graphs, next to the code that loads them
```

`from . import test_CCwIntSqz` as a way to reuse a model is gone. So are the
`fromgwinc/intsqz/{lib,optics}.py` shims from Stage 2, which had no importers
left and were deleted.

### The budget existed four times, not three

The plan counted three copies. A fourth turned up in
`test_intFDsqz_sweeps.py::_compute_d_sense_CC`, which hand-rolled the same
injection-loss → filter-cavity → plant → readout-loss chain to get one number
out of it. All four now call `accumulate()` + `quantum_budget()`.

Example files shrank accordingly:

| file | before | after |
|---|---|---|
| `test_CCwIntSqz.py` | 891 | 235 |
| `test_CCwIntFDSqz.py` | 640 | 130 |
| `test_intFDsqz_sweeps.py` | 339 | 312 |

### Bug found while deduplicating: `LB['ASport']` was overwritten

`intSqzQuantum` did:

```python
ASport = ASquantumAll * PSDdisplacement * dhdl_sqr
total = ASport            # same array object
LB = {'ASport': ASport}
for ...:
    total += lossB        # in-place: silently overwrites LB['ASport']
```

so the returned `LB['ASport']` was not the AS-port contribution at all, it was
the running total. Confirmed against the captured baseline:
`LB['ASport']` is bit-identical to `total` for every config. The
frequency-dependent model does the same computation with `.copy()` and is
correct — there `ASport/total` is 0.84 at the low-frequency end.

Consequences are small: no current caller reads `LB['ASport']` from
`intSqzQuantum`, and the only visible symptom is the `MINRATIO` diagnostic
print, which has been reporting `total/INTSQZ` instead of `ASport/INTSQZ`.

**Not fixed here.** Correcting it changes a returned number, which is a
reporting decision rather than a refactor. It is preserved exactly, behind an
explicit `alias_ASport=True` flag with a comment, so the behaviour is now
visible instead of accidental. Flip the flag to fix it.

**Verification.** Every budget, loss-port and topology array bit-identical;
guard green; suite unchanged at 53 passed / 1 failed / 2 skipped / 14 errors;
`make survey` 0 of 36; docs rebuild clean.

## Stage 3 — original plan

Split each `test_*.py` into the three things it currently is:

```
sflu/models/coupled_cavity.py   sflu_CoupledCav()      topology
                                CoupledCavity()        plant
sflu/models/int_fd_sqz.py       sflu_CCwIntFDSqz()     topology
                                CoupledCavityIntFC()   plant
sflu/models/budget.py           quantum_budget()       resultsAC -> PSD
sflu/models/filter_cavity.py    FilterCavity()
examples/…/test_*.py            load params, call a model, plot, assert
```

`quantum_budget()` replaces the three copies of the budget calculation
(`docs/DEPENDENCIES.md`, problem 2). `_compute_intFDsqz_budget()` in
`test_CCwIntFDSqz.py` is already the right shape and should be the starting
point. Copy 3, the inline one in `test_CCwIntFDSqz`, is dead and can simply go.

Do it in this order, guard after each step:

1. extract `quantum_budget()`, have `intSqzQuantum` and `intFDsqzQuantum` both
   call it, delete the duplicated bodies;
2. `git mv` topology + plant functions into `sflu/models/`;
3. leave `from . import test_CCwIntSqz`-style shims behind until every caller
   is repointed, then remove them.

This is what finally kills `from . import test_CCwIntSqz` as a way to reuse a
model.

**Verification:** the guard. Step 1 in particular must be bit-identical — it is
pure deduplication.

## Stage 4 — params get their own place (done)

```
sflu/params/ifo/*.yaml       13 IFO parameter sets
sflu/params/standardize.py   standardize_params, arm_gouyRT
sflu/params/__init__.py      ifo_path(), load_ifo(), available()
```

Parameter sets are now reached **by name**, not by path, so no caller needs to
know where the directory is:

```python
from sflu.params import load_ifo, ifo_path
ifo = load_ifo('AhatTest')                                  # parameters
gwinc.load_budget(ifo_path('AplWide'), freq=F_Hz).run()     # reference budget
```

That replaced 20 `fpath_join('X' + '.yaml')` call sites, which had resolved
relative to whichever test file happened to contain them.

Serialized SFLU graphs went to `sflu/models/topologies/` in Stage 3 rather
than under `params/` as originally planned: a topology is part of the model,
not a parameter set. The two kinds of yaml are separated either way, which was
the point.

### Duplicates and stale data removed

* **Root-level `Asharp.yaml` / `Asharp_wideband.yaml`** — byte-identical to
  the `fromgwinc/intsqz/` copies *and* unreferenced (every read went through
  `fpath_join` into `fromgwinc/intsqz/`). Deleted.
* **`fromgwinc/intsqz/old/`** — `Ahat20`, `Ahat25`, `AhatSh20`, `AhatSh25`,
  zero live references. Moved to `attic/ifo_superseded/`.
* **`fromgwinc/<IFO>/ifo.yaml`** — six vendored gwinc parameter files, moved to
  `attic/ifo_packages/` to rejoin their already-quarantined `__init__.py`. They
  were unread, and worse, **wrong**: `Aplus/ifo.yaml` differs from the
  installed pygwinc by 29 lines, carrying `Curvature.ITM: 1970 / ETM: 2192`
  where master has `1940 / 2245`. Those are the kuns-fork values. The repo was
  storing one set of parameters and computing with another. (One of them is
  even mislabelled `# GWINC aLIGO interferometer parameters`.) Stage 6 will
  want them as a reference when it picks a base set to vendor deliberately.

`test_FP.py`, which had been failing since before this branch on a relative
path with one `..` too many, pointed at one of those stale files. It now
exercises `load_ifo()` and asserts the `+inherit` chain resolved, instead of
printing a file nothing used. **It passes.**

**Verification.** Guard exact (231 arrays, zero changes); survey 0 of 37;
suite unchanged at 53 passed / 1 failed / 2 skipped / 14 errors; docs rebuild
clean.

## Stage 4 — original plan

* `sflu/params/ifo/*.yaml` — IFO parameter sets (`Ahat*`, `Aplus*`, `Asharp*`);
* `sflu/params/topologies/*.yaml` — serialized SFLU graphs (`CoupledCavity.yaml`,
  `FilterCavity.yaml`), which are a different kind of file that merely looks
  the same;
* de-duplicate root-level `Asharp.yaml` / `Asharp_wideband.yaml` against the
  copies in `fromgwinc/intsqz/`. **Diff them before assuming they are equal.**

Paths are constructed with `fpath_join(...)` in the examples, so this stage is
a mechanical move plus path updates.

## Stage 5 — publish the docs (done)

Infrastructure only; no documentation prose was hand-written. Every page on the
site is still generated from the examples.

### CI

`.github/workflows/ci.yml` — GitHub Actions, `PYTHONHASHSEED=0` set globally:

| step | blocking | what it does |
|---|---|---|
| Guard | yes | `make guard`; a change that moves a number fails here |
| Import survey | yes | `make survey` |
| Test suite | no, for now | `make test` |
| Build documentation | yes | `make docs-site` (`--strict` + `mkdocs build`) |
| Pages deploy | default branch only | publishes `docs/_site` |

The test step is `continue-on-error` **only** because of the one known
pre-existing failure, `models/test_simple_mirror.py::test_sflu_simple_mirror`.
Drop it once that is fixed. The errors from the absent optional
`qlance`/Optickle dependency are expected in CI.

`setup.sh` is adapted from the `refactor/intsqz` branch with three corrections:
it installs **wield-pytest**, which the old script omitted even though it
supplies the `--plot` option the generator passes (docs builds would have
produced no figures without it); it takes `gwinc` from the package index rather
than a fork, per `docs/GWINC_DEPENDENCY.md`; and it clones the wield packages
over **anonymous HTTPS** rather than SSH, so neither CI nor a new contributor
needs a key. The workflow inlines the same dependency list so it can cache each
clone against the revision it pins; `tools/test_ci_config.py` fails if the two
lists drift, if either reintroduces an SSH URL, or if the guard step stops
being blocking.

### What running the CI actually found

The pipeline was exercised on a scratch GitHub repository. Six runs, four real
defects, none of which local testing could have shown:

1. **The numerical guard was machine-specific.** rtol 1e-12 with no absolute
   tolerance holds only where the baseline was recorded. On a runner, 106
   arrays differed: median 1.7e-8 from CPU/BLAS differences, worst budget
   1.16e-3 (AhatTest, the ill-conditioned one -- same magnitude and same cause
   as Finding 1), and one numerically-zero matrix element at relative
   difference 1. Topology strings matched exactly. Hence `make guard-ci`, with
   tolerances a decade above the observed noise, while `make guard` stays exact
   locally.

2. **`--plot` was not registered**, so the first docs builds produced 13 pages
   and zero figures. The `wield.pytest` plugin *was* autoloaded and still added
   no options, because it guards `--plot` behind a module-level flag that was
   already set. `conftest.py` now registers `--plot` and `--no-preclear`
   itself, catching the `ValueError` when the plugin got there first.

3. **Six examples imported a module gwinc has never shipped.**
   `gwinc.noise.quantum_lib` is absent from the installed distribution's
   RECORD, yet the file was present in `site-packages/gwinc/noise/`: it had
   been hand-copied into the installed package. No fresh environment could
   reproduce those examples. Now vendored at `sflu_components/quantum_lib.py`.
   This is precisely the failure `docs/GWINC_DEPENDENCY.md` predicts, and it
   strengthens the case for Stage 6.

4. **The CI log viewer truncates** long steps at a fixed point, which hid the
   docs failure for two runs. The docs step now writes to a file, echoes the
   tail on failure, and uploads the whole log.

Points 2 and 3 were caught *because* `--strict` refuses to publish a page with
no figures. Without that check the site would have deployed, looking complete
and containing nothing.

**The pipeline is green end to end**, including the Pages deployment:

  <https://jeff-wack-bot.github.io/sflu-optomechanics-ci-test/>

Verified live: 13 of 13 example pages and 55 of 55 figures return HTTP 200.

That scratch repository had to be made **public** to get there -- GitHub Pages
is not served from a private repository on a free plan. It holds this refactor
branch only, and it is a CI test bed, not the home of the project; the real
remote is still `git.ligo.org`. If this workflow moves back to GitLab, the
`pages` job needs rewriting for GitLab Pages, but everything else in it (the
guard tolerances, the docs `--strict` gate, the keyless dependency clone)
carries over unchanged.

### The docs build now fails loudly

`--strict` (also `make docs-strict`, and `make docs-site` for the full site)
exits nonzero when:

* a module listed in `MODULES` no longer exists — catches renames;
* a listed module produces no figures, unless it declares
  `"expect_figures": False`;
* the example run itself exits nonzero;
* an example module exists with neither a page nor an entry in the new
  `EXCLUDED` registry, which records *why* each undocumented example is
  undocumented.

All four were tamper-tested: each fails under `--strict`, stays quiet
otherwise, and the healthy tree exits 0.

`tools/test_docs_config.py` repeats the cheap half of those checks in the
normal suite (18 assertions, 0.3 s), so a rename that orphans a page is caught
by `make test` rather than at the next docs build.

### Bug found: parametrized examples lost every figure

`collect_figures` looked in `tresults/<test_name>`, but a parametrized example
writes to `tresults/<test_name>[<param>]`. Six figures across
`pi/test_pi_gain.py` and `optics/test_simple_cavities.py` were being silently
dropped — exactly the class of silent breakage this stage was meant to expose,
and it was the strict check that surfaced it. Fixed, with the parameter id kept
in the figure key so that two parameter runs writing the same filename no
longer overwrite each other.

Site is now **13 pages, 59 figures** (was 56), `mkdocs build` clean with zero
broken links.

## Stage 6 — vendor the small half of gwinc, drop the runtime dependency

Evaluated in full in [`docs/GWINC_DEPENDENCY.md`](docs/GWINC_DEPENDENCY.md).
Summary of the finding, because it changes the answer to the open question
below:

The gwinc dependency splits along the same lib/params/model seam as everything
else. `load_budget(path).ifo` — which is what every model calls — is provably a
pure YAML merge; the `Budget` object is built and discarded. The inherit chain
leaves the repo exactly once, at gwinc's 336-line `ifo/Aplus/ifo.yaml`. Every
`.run()` in the repo is a reference comparison curve for a plot, feeding no
model and no guard.

So:

* **vendor** `struct.py`, `const.py`, `ifo_power`/`dhdl`/`arm_cavity`, and the
  base `Aplus/ifo.yaml` — about 1,100 lines, and gwinc is Unlicense, so there
  is no legal friction;
* **do not vendor** the budget engine — 6,102 lines across nine noise
  disciplines this project neither develops nor can validate. Cache the
  reference curves as data and make gwinc an optional dev dependency used only
  to refresh them.

The motivation is reproducibility rather than line count: today
`gwinc.load_budget('Aplus')` silently reads whatever pygwinc is installed, and
the two candidate forks disagree on `Curvature.ITM`, `Curvature.ETM`, and an
`ifo_power` conditional — differences that move published numbers. Vendoring
the base parameter file turns that into a line in `git log`. It is the same
class of defect as Finding 1.

Sequence it **after Stage 3**: vendoring along the lib/params/model seam once
that seam exists is a far smaller diff than vendoring across today's tangle.

## What is deliberately *not* in this plan

* **No physics changes.** Not the hash-seed determinism fix, not the SEC/INTSQZ
  angle conventions, not the commented-out parameter blocks in
  `test_CoupledCav`. Each is a decision for whoever owns the model.
* **No adoption of the `refactor/intsqz` dependency approach.** That branch
  went the other way — depend on `gwinc.optomechanicalmodels` from a pygwinc
  fork instead of vendoring. It does not run in the current `wield`
  environment, because `gwinc.optomechanicalmodels` is not installed there.
  Stage 6 resolves the underlying question in the opposite direction: vendor
  the small deterministic half and stop depending on *which* pygwinc is
  present. Reviving `optics/test_DRFPMI.py` still needs a pygwinc that provides
  `gwinc.plant` or `gwinc.noise.quantum2`, and that remains a separate,
  optional decision.
* **No renaming of `fromgwinc/`.** It becomes wrong once Stage 1 removes the
  vendored gwinc code, but renaming it touches every import in the trusted
  reference model. Do it last, or never.

## Order of work, and why

Stage 1 first because it is free and makes everything else readable. Stage 2
next because two live copies of `MatrixLib` is the defect most likely to cause
a real error. Stage 3 is the biggest win for reuse but the largest diff, so it
goes after the guard has been exercised twice. Stages 4 and 5 are cosmetic and
can happen any time.

Each stage is a separate commit on this branch, guard-green before and after.
