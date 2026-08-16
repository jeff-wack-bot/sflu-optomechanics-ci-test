#!/usr/bin/env python
"""
Capture numerical baselines for the internal-squeezing models.

This is the safety net for the structural refactor.  It calls the model entry
points directly (not through pytest) and writes every output array to a single
``.npz`` plus a JSON manifest.  ``test_regression.py`` then asserts that the
same calls still produce the same numbers, bit-for-bit-ish (rtol 1e-12).

The point is narrow and deliberate: moving a file, renaming a module, or
collapsing two copies of ``MatrixLib`` must not change any number.  If it does,
that is a bug in the move, not a modelling choice.

Usage
-----
    python -m tools.regression.capture_baseline            # write baselines
    python -m tools.regression.capture_baseline --check    # compare, don't write

Run this from the repository root, on a commit you trust, *before* touching
anything.  Commit the resulting ``.npz`` alongside the move that it protects.

Determinism
-----------
``SFLU.reduce_auto()`` eliminates nodes in Python ``set`` iteration order, so
the elimination order -- and with it the budget, at up to the 1e-3 level --
depends on ``PYTHONHASHSEED``.  This script therefore re-executes itself with
``PYTHONHASHSEED=0`` if it is not already pinned.  With the seed pinned the
results are bit-identical across processes, which is what makes an exact
regression check possible.  See ``REFACTOR_PLAN.md`` (Finding 1).
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# Must happen before numpy/gwinc/wield are imported, and before any set of
# strings is built, so re-exec rather than trying to patch it up afterwards.
HASH_SEED = "0"
if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
    os.environ["PYTHONHASHSEED"] = HASH_SEED
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"
NPZ_PATH = BASELINE_DIR / "intsqz.npz"
MANIFEST_PATH = BASELINE_DIR / "manifest.json"

# Frequency grids are pinned here rather than taken from the tests, so that
# editing a plot range in a test cannot silently invalidate the baseline.
F_BUDGET = np.geomspace(10.0, 30e3, 400)

# IFO configs driven through the non-FC internal squeezing budget.
INTSQZ_CASES = [
    "AhatTest",
    "Ahat17",
    "Ahat22",
    "Ahat30",
    "AhatSh17",
    "AhatSh22",
]

# IFO configs driven through the internal *filter cavity* budget.
INTFDSQZ_CASES = [
    "AhatTestIntFC",
]

# INTSQ_loss is set explicitly by every caller in the test suite, so pin it
# here too rather than relying on whatever the yaml happens to carry.
INTSQ_LOSS = 1000e-6


def _yaml(name):
    return str(ROOT / "fromgwinc" / "intsqz" / f"{name}.yaml")


def _load_ifo(name):
    import gwinc

    budget = gwinc.load_budget(_yaml(name))
    ifo = budget.ifo
    ifo.Optics.INTSQ_loss = INTSQ_LOSS
    return ifo


def _flatten(prefix, total, LB, extra=None):
    """Flatten a (total, loss-budget dict) result into flat npz entries."""
    out = {f"{prefix}/total": np.asarray(total)}
    for port, arr in LB.items():
        out[f"{prefix}/LB/{port}"] = np.asarray(arr)
    for key, arr in (extra or {}).items():
        out[f"{prefix}/{key}"] = np.asarray(arr)
    return out


def capture_intsqz():
    """Budgets from fromgwinc/intsqz/test_CCwIntSqz.py::intSqzQuantum."""
    from fromgwinc.intsqz import test_CCwIntSqz

    arrays = {}
    for name in INTSQZ_CASES:
        try:
            ifo = _load_ifo(name)
            total, LB = test_CCwIntSqz.intSqzQuantum(ifo, freq=F_BUDGET)
        except Exception:
            print(f"  [skip] intsqz/{name}")
            traceback.print_exc(limit=2)
            continue
        arrays.update(_flatten(f"intsqz/{name}", total, LB))
        print(f"  [ok]   intsqz/{name}  ({len(LB)} loss ports)")
    return arrays


def capture_intfdsqz():
    """Budgets from fromgwinc/intsqz/test_CCwIntFDSqz.py::intFDsqzQuantum."""
    from fromgwinc.intsqz import test_CCwIntFDSqz

    arrays = {}
    for name in INTFDSQZ_CASES:
        try:
            ifo = _load_ifo(name)
            total, LB, d_sense = test_CCwIntFDSqz.intFDsqzQuantum(ifo, freq=F_BUDGET)
        except Exception:
            print(f"  [skip] intfdsqz/{name}")
            traceback.print_exc(limit=2)
            continue
        arrays.update(
            _flatten(f"intfdsqz/{name}", total, LB, extra={"d_sense": d_sense})
        )
        print(f"  [ok]   intfdsqz/{name}  ({len(LB)} loss ports)")
    return arrays


def capture_topologies():
    """Serialized SFLU graphs.

    Cheap, physics-free, and catches accidental topology drift when the graph
    builders get moved out of the test files.
    """
    from fromgwinc.intsqz import test_CCwIntFDSqz, test_CCwIntSqz

    builders = {
        "CoupledCav": test_CCwIntSqz.sflu_CoupledCav,
        "CCwIntFDSqz": test_CCwIntFDSqz.sflu_CCwIntFDSqz,
    }
    arrays = {}
    for name, build in builders.items():
        try:
            sfluB = build()
            edges = sorted(
                f"{u} <- {v} : {lbl}"
                for (u, v), lbl in sfluB.sflu.edges.items()
            )
        except Exception:
            print(f"  [skip] topology/{name}")
            traceback.print_exc(limit=2)
            continue
        arrays[f"topology/{name}"] = np.array(edges, dtype=object)
        print(f"  [ok]   topology/{name}  ({len(edges)} edges)")
    return arrays


def collect():
    arrays = {}
    print("Capturing intsqz budgets...")
    arrays.update(capture_intsqz())
    print("Capturing intFDsqz budgets...")
    arrays.update(capture_intfdsqz())
    print("Capturing SFLU topologies...")
    arrays.update(capture_topologies())
    return arrays


def write(arrays):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_PATH, **arrays)
    manifest = {
        "F_budget": {
            "start": float(F_BUDGET[0]),
            "stop": float(F_BUDGET[-1]),
            "num": int(len(F_BUDGET)),
            "spacing": "geomspace",
        },
        "INTSQ_loss": INTSQ_LOSS,
        "PYTHONHASHSEED": HASH_SEED,
        "keys": sorted(arrays),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {NPZ_PATH.relative_to(ROOT)} ({len(arrays)} arrays)")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}")


def check(arrays, rtol=1e-12):
    if not NPZ_PATH.exists():
        print(f"No baseline at {NPZ_PATH}; run without --check first.")
        return 1
    ref = np.load(NPZ_PATH, allow_pickle=True)
    missing = sorted(set(ref.files) - set(arrays))
    added = sorted(set(arrays) - set(ref.files))
    bad = []
    for key in sorted(set(ref.files) & set(arrays)):
        a, b = ref[key], arrays[key]
        if a.dtype == object:
            if list(a) != list(b):
                bad.append((key, "topology differs"))
            continue
        if a.shape != b.shape:
            bad.append((key, f"shape {a.shape} -> {b.shape}"))
            continue
        if not np.allclose(a, b, rtol=rtol, atol=0, equal_nan=True):
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.nanmax(np.abs(b - a) / np.abs(a))
            bad.append((key, f"max rel diff {rel:.3e}"))

    for key in missing:
        print(f"  MISSING  {key}")
    for key in added:
        print(f"  NEW      {key}")
    for key, why in bad:
        print(f"  CHANGED  {key}: {why}")
    if missing or bad:
        print("\nBaseline check FAILED")
        return 1
    print(f"\nBaseline check passed ({len(ref.files)} arrays, rtol={rtol:g})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the stored baseline instead of overwriting it",
    )
    args = ap.parse_args()

    arrays = collect()
    if not arrays:
        print("Captured nothing -- is the environment set up?")
        return 1
    if args.check:
        return check(arrays)
    write(arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
