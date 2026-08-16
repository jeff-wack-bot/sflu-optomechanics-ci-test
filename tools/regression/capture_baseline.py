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


def _load_ifo(name):
    from sflu.params import load_ifo

    ifo = load_ifo(name)
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
    """Budgets from sflu.models.coupled_cavity::intSqzQuantum."""
    from sflu.models import intSqzQuantum

    arrays = {}
    for name in INTSQZ_CASES:
        try:
            ifo = _load_ifo(name)
            total, LB = intSqzQuantum(ifo, freq=F_BUDGET)
        except Exception:
            print(f"  [skip] intsqz/{name}")
            traceback.print_exc(limit=2)
            continue
        arrays.update(_flatten(f"intsqz/{name}", total, LB))
        print(f"  [ok]   intsqz/{name}  ({len(LB)} loss ports)")
    return arrays


def capture_intfdsqz():
    """Budgets from sflu.models.int_fd_sqz::intFDsqzQuantum."""
    from sflu.models import intFDsqzQuantum

    arrays = {}
    for name in INTFDSQZ_CASES:
        try:
            ifo = _load_ifo(name)
            total, LB, d_sense = intFDsqzQuantum(ifo, freq=F_BUDGET)
        except Exception:
            print(f"  [skip] intfdsqz/{name}")
            traceback.print_exc(limit=2)
            continue
        arrays.update(
            _flatten(f"intfdsqz/{name}", total, LB, extra={"d_sense": d_sense})
        )
        print(f"  [ok]   intfdsqz/{name}  ({len(LB)} loss ports)")
    return arrays


def _mlib_probe(mlib, tag, arrays):
    """Record every MatrixLib output that a model can observe.

    Each probe is recorded independently: some methods raise on some inputs
    (notably ``RPNK`` with a vector, which contradicts its own docstring but
    has no callers), and one such failure must not drop the rest.
    """
    import numpy as _np

    def probe(name, fn):
        try:
            arrays[f"{tag}/{name}"] = _np.asarray(fn())
        except Exception as exc:  # noqa: BLE001 - a raise is itself the behaviour
            arrays[f"{tag}/{name}"] = _np.array(
                [f"{type(exc).__name__}: {exc}"], dtype=object
            )

    probe("Id", lambda: mlib.Id)
    probe("Id_v", lambda: mlib.Id_v)
    probe("Id_a", lambda: mlib.Id_a)
    probe("Id_s", lambda: mlib.Id_s)
    probe("zeros", lambda: mlib.zeros)
    probe("A", lambda: mlib.A)
    probe("Ai", lambda: mlib.Ai)
    probe("Mrotation", lambda: mlib.Mrotation(0.37))
    probe("Mrotation_vec", lambda: mlib.Mrotation(_np.linspace(0.1, 1.2, 5)))
    probe("LO", lambda: mlib.LO(0.61))
    probe("RPNK_scalar", lambda: mlib.RPNK(1.7))
    probe("RPNK_vector", lambda: mlib.RPNK(_np.linspace(0.5, 3.0, 4)))
    probe("SQZ", lambda: mlib.SQZ(10 ** (-0.6), 10 ** 1.5))
    probe("diag", lambda: mlib.diag(_np.linspace(1.0, 2.0, 4)))
    probe("promote_scalar", lambda: mlib.promote(0.73))
    probe("block_diag", lambda: mlib.block_diag(_np.eye(2) * 1.3))
    if mlib.nhom > 0:
        probe("MrotationMM", lambda: mlib.MrotationMM(0.02, 0.4))
        probe("MrotationMM_inv", lambda: mlib.MrotationMM(0.02, 0.4, inv=True))
    # SQZc exists only on the intsqz copy today; Stage 2a moves it across.
    if hasattr(mlib, "SQZc"):
        probe("SQZc", lambda: mlib.SQZc(10 ** (-0.6), 10 ** 1.5))


def capture_matrixlib():
    """MatrixLib outputs.

    There used to be two copies of this library; Stage 2 collapsed them and
    Stage 3 deleted the compatibility shim, so there is one import path left.
    """
    from sflu_components import lib as sc_lib

    arrays = {}
    for tag, mod in (("mlib/sflu_components", sc_lib),):
        for nhom in (0, 1):
            try:
                _mlib_probe(mod.MatrixLib(nhom=nhom), f"{tag}/nhom{nhom}", arrays)
            except Exception:
                print(f"  [skip] {tag}/nhom{nhom}")
                traceback.print_exc(limit=2)
        # module-level helpers
        m = np.array([[2.0, 0.5], [1.0, 3.0]])
        arrays[f"{tag}/adjoint"] = mod.adjoint(m + 1j * m[::-1])
        arrays[f"{tag}/transpose"] = mod.transpose(m)
        arrays[f"{tag}/Minv"] = mod.Minv(m)
        v = np.array([[1.0 + 2j], [0.5 - 1j]])
        arrays[f"{tag}/Vnorm_sq"] = mod.Vnorm_sq(v)
        if hasattr(mod, "Vnorm_sqA"):
            arrays[f"{tag}/Vnorm_sqA"] = mod.Vnorm_sqA(mod.adjoint(v))
    print(f"  [ok]   MatrixLib probes ({len(arrays)} arrays)")
    return arrays


def _edge_probe(mod, tag, arrays, mlib):
    """Record the edge maps every edge class produces.

    Consumers of these classes (`optics/`, `pi/`) mostly plot without
    asserting, so this is the only thing standing between a bad merge and a
    silently wrong figure.
    """
    F_Hz = np.geomspace(10.0, 1e4, 7)

    def store(prefix, edge_map):
        for key, val in sorted(edge_map.items()):
            arrays[f"{tag}/{prefix}/{key}"] = np.asarray(val)

    # --- MirrorEdge, in each of its loss conventions.
    mirror_kw = dict(name="M", Thr=0.014, Lhr=30e-6, Rar=1e-4, mlib=mlib)
    variants = [
        ("default", {}),
        ("loss_ports", {"loss_ports": True}),
        ("loss_in_transmission",
         {"loss_in_transmission": True, "loss_ports": True}),
    ]
    for label, extra in variants:
        m = mod.MirrorEdge(**mirror_kw, **extra)
        store(f"MirrorEdge/{label}/DC", m.edgesDC())
        store(f"MirrorEdge/{label}/AC", m.edgesAC(F_Hz=F_Hz, resultsDC={}))

    # --- BSEdge
    bs = mod.BSEdge(name="BS", Thr=0.5, Lhr=1e-5, mlib=mlib)
    store("BSEdge/DC", bs.edgesDC())
    store("BSEdge/AC", bs.edgesAC(F_Hz=F_Hz, resultsDC={}))

    # --- LinkEdge
    gouy = None if mlib.nhom == 0 else 0.4
    link = mod.LinkEdge(name="L", L_m=4000.0, detune_rad=0.31,
                        gouy_rad=gouy, mlib=mlib)
    store("LinkEdge/DC", link.edgesDC())
    store("LinkEdge/AC", link.edgesAC(F_Hz=F_Hz))

    # --- RPMirrorEdge, with realistic DC fields at the faces
    field = np.sqrt(4e5) * mlib.LO(np.pi / 2)
    resultsDC = {
        "E.fr.i.tp": field,
        "E.fr.o.tp": -0.9 * field,
        "E.bk.i.tp": 0.1 * field,
        "E.bk.o.tp": 0.3 * field,
    }
    rp = mod.RPMirrorEdge(name="E", Thr=5e-6, Lhr=30e-6, mlib=mlib,
                          loss_ports=True,
                          suscept=lambda f: -1 / (40 * (2 * np.pi * f) ** 2))
    store("RPMirrorEdge/DC", rp.edgesDC())
    store("RPMirrorEdge/AC", rp.edgesAC(F_Hz=F_Hz, resultsDC=resultsDC))

    # --- SQZEdge exists only on the intsqz copy today
    if hasattr(mod, "SQZEdge"):
        sq = mod.SQZEdge(name="SQ", sqzDB=10.0, sqzANGdeg=-90.0, mlib=mlib)
        store("SQZEdge/DC", sq.edgesDC())
        store("SQZEdge/AC", sq.edgesAC(F_Hz=F_Hz))


def capture_edges():
    """Edge maps from the edge library."""
    from sflu_components import edges as sc_edges
    from sflu_components.lib import MatrixLib

    arrays = {}
    for tag, mod in (("edges/sflu_components", sc_edges),):
        for nhom in (0, 1):
            mlib = MatrixLib(nhom=nhom)
            try:
                _edge_probe(mod, f"{tag}/nhom{nhom}", arrays, mlib)
            except Exception:
                print(f"  [skip] {tag}/nhom{nhom}")
                traceback.print_exc(limit=2)
    print(f"  [ok]   edge probes ({len(arrays)} arrays)")
    return arrays


def capture_topologies():
    """Serialized SFLU graphs.

    Cheap, physics-free, and catches accidental topology drift when the graph
    builders get moved out of the test files.
    """
    from sflu.models import sflu_CCwIntFDSqz, sflu_CoupledCav

    builders = {
        "CoupledCav": sflu_CoupledCav,
        "CCwIntFDSqz": sflu_CCwIntFDSqz,
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
    print("Capturing MatrixLib outputs...")
    arrays.update(capture_matrixlib())
    print("Capturing edge maps...")
    arrays.update(capture_edges())
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
