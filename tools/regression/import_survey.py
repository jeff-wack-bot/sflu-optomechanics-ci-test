#!/usr/bin/env python
"""
Import every Python module in the repository and report which ones fail.

This is the evidence behind the dead-code inventory in ``docs/DEPENDENCIES.md``.
A module that cannot be imported is not "unused but working" -- it is broken,
and keeping it next to live code is what makes the dependency structure hard to
read.

    python tools/regression/import_survey.py
    python tools/regression/import_survey.py --quiet   # only the failures
"""
import argparse
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# models/ uses flat imports (`import components`), resolved by pytest's rootdir
# handling; mirror that here so those modules get a fair test.
for extra in (ROOT, ROOT / "models"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

# attic/ is quarantined code that is *known* not to import; surveying it would
# drown the signal. Its contents are catalogued in attic/README.md instead.
SKIP_DIRS = {".git", "__pycache__", "tresults", "deps", "_site", "attic"}
SKIP_NAMES = {"conftest.py"}


def modules():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if SKIP_DIRS & set(rel.parts) or rel.name in SKIP_NAMES:
            continue
        if rel.parts[0] == "tools":
            continue
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            yield ".".join(parts), rel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="list failures only")
    args = ap.parse_args()

    ok, skipped, bad = [], [], []
    for name, rel in modules():
        try:
            importlib.import_module(name)
            ok.append(rel)
        # BaseException, not Exception: pytest.skip(allow_module_level=True)
        # raises Skipped, which derives from BaseException.
        except BaseException as exc:  # noqa: BLE001 - surveying, report anything
            msg = (str(exc).splitlines() or [""])[0]
            if type(exc).__name__ == "Skipped":
                skipped.append((rel, msg))
            else:
                bad.append((rel, type(exc).__name__, msg))

    if not args.quiet:
        print(f"=== importable ({len(ok)}) ===")
        for rel in ok:
            print(f"  {rel}")
        print()

    if skipped:
        print(f"=== skipped, optional dependency absent ({len(skipped)}) ===")
        for rel, msg in skipped:
            print(f"  {rel}\n      {msg[:110]}")
        print()

    print(f"=== NOT importable ({len(bad)}) ===")
    for rel, kind, msg in bad:
        print(f"  {rel}\n      {kind}: {msg[:110]}")
    if not bad:
        print("  (none)")

    total = len(ok) + len(skipped) + len(bad)
    print(f"\n{len(bad)}/{total} live modules cannot be imported "
          f"({len(skipped)} skipped for optional dependencies).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
