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
  the regression harness and the docs build; add it to `pytest.ini` (Stage 1).
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

## Stage 1 — quarantine the dead code

**Risk: none.** Every file moved here is one that raises on import today, so
nothing can depend on it. This is the change that makes the tree readable.

`git mv` into `attic/` (history preserved; nothing deleted), with an
`attic/README.md` recording where each file came from and exactly why it does
not load:

```
attic/optomechanicalmodels/     5 modules, ~1,490 lines  vendored gwinc fork subtree
attic/noise/                    quantum_lib.py, quantum2.py
attic/intsqz_quantum_lib.py     byte-identical to attic/noise/quantum_lib.py
attic/ifo_packages/             aLIGO Aplus CE1 CE2silica CE2silicon Voyager
attic/test_DRFPMI.py            needs gwinc.plant; re-enable when that exists
attic/test_CCwIntSqz.py_        trailing-underscore backup
```

Also in this stage, all mechanical and individually reversible:

* fix `fromgwinc/intsqz/test_FP.py`'s `../../Aplus/ifo.yaml` → `../Aplus/ifo.yaml`
  (one `..` too many) and drop its byte-identical twin under
  `optomechanicalmodels/test/`;
* add `PYTHONHASHSEED=0` to `pytest.ini` (Finding 1);
* add `docs/_site/`, `docs/docs/`, `docs/mkdocs.yml` to `.gitignore` — they are
  generated.

**Verification:** `pytest tools/regression/test_regression.py` unchanged, and a
bare `pytest` now collects and runs the suite instead of aborting.

**Expected after Stage 1:** ~3,400 lines and 15 broken modules out of the way;
`pytest` works with no arguments.

## Stage 2 — one library, not three

The highest-value structural fix, and fully verifiable.

Build a single library from the **union**, based on the intsqz copy (Finding 2):

```
sflu_components/lib.py      += Vnorm_sqA, MatrixLib.SQZc, MatsHelper
sflu_components/edges.py    += SQZEdge, edgesACSS() on every class,
                               MirrorEdge(loss_in_transmission=)
```

Then repoint importers one at a time, running the guard after each:

1. add the missing pieces to `sflu_components` — additive, guard must stay green;
2. switch `fromgwinc/intsqz/lib.py` to re-export from `sflu_components.lib`;
3. switch `fromgwinc/intsqz/optics.py` to re-export from `sflu_components.edges`;
4. rewrite the mixed imports in `test_CCwIntSqz.py` / `test_CCwIntFDSqz.py` so
   one `MatrixLib` is alive per process instead of two;
5. only then delete the emptied shims.

Merge hazards to handle explicitly, each capable of silently changing numbers:

* `MirrorEdge.__init__` takes `loss_ports=` in one copy and
  `loss_in_transmission=` in the other. Different meanings; keep both names,
  do not unify the semantics.
* `sflu_components` exposes `r`/`t`/`l` as read-only properties, intsqz as
  plain attributes. Properties are the safer target, but the models assign
  nothing, so confirm with the guard.
* `edges.py` imports `scipy.constants`, `optics.py` imports `gwinc.const`.
  Values are identical, so pick one; the guard proves it.

`models/{matlib,components,components2}.py` is a third partial copy used only
by `models/`. Leave it alone in this stage and fold it in later, or declare
`models/` a frozen historical example set — it is the least-load-bearing corner.

**Verification:** the guard, plus the full suite at parity with the Stage 0
counts.

## Stage 3 — models become importable

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

## Stage 4 — params get their own place

* `sflu/params/ifo/*.yaml` — IFO parameter sets (`Ahat*`, `Aplus*`, `Asharp*`);
* `sflu/params/topologies/*.yaml` — serialized SFLU graphs (`CoupledCavity.yaml`,
  `FilterCavity.yaml`), which are a different kind of file that merely looks
  the same;
* de-duplicate root-level `Asharp.yaml` / `Asharp_wideband.yaml` against the
  copies in `fromgwinc/intsqz/`. **Diff them before assuming they are equal.**

Paths are constructed with `fpath_join(...)` in the examples, so this stage is
a mechanical move plus path updates.

## Stage 5 — publish the docs

`docs/generate_docs.py` already works end-to-end: 13 pages, 56 figures, clean
`mkdocs build`. Remaining work is only wiring:

* restore a CI job (the `refactor/intsqz` branch has a working
  `.gitlab-ci.yml` + `setup.sh` to adapt), running the generator with
  `PYTHONHASHSEED=0`;
* extend `MODULES` in the generator as Stage 3 renames files;
* consider failing the docs build if an example listed in `MODULES` produced no
  figures, so silent breakage is visible.

## What is deliberately *not* in this plan

* **No physics changes.** Not the hash-seed determinism fix, not the SEC/INTSQZ
  angle conventions, not the commented-out parameter blocks in
  `test_CoupledCav`. Each is a decision for whoever owns the model.
* **No dependency-pinning rework.** The `refactor/intsqz` branch's approach —
  depend on `gwinc.optomechanicalmodels` from a pygwinc fork instead of
  vendoring — is a reasonable idea, but `gwinc.optomechanicalmodels` is **not
  present in the current `wield` environment**, so that branch does not run
  here today. Vendored-and-working beats unvendored-and-unimportable until
  someone decides which pygwinc this repo targets. That decision is a
  prerequisite for reviving `optics/test_DRFPMI.py`.
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
