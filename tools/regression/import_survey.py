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

SKIP_DIRS = {".git", "__pycache__", "tresults", "deps", "_site"}
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

    ok, bad = [], []
    for name, rel in modules():
        try:
            importlib.import_module(name)
            ok.append(rel)
        except Exception as exc:  # noqa: BLE001 - surveying, so report anything
            bad.append((rel, type(exc).__name__, str(exc).splitlines()[0]))

    if not args.quiet:
        print(f"=== importable ({len(ok)}) ===")
        for rel in ok:
            print(f"  {rel}")
        print()
    print(f"=== NOT importable ({len(bad)}) ===")
    for rel, kind, msg in bad:
        print(f"  {rel}\n      {kind}: {msg[:110]}")

    total = len(ok) + len(bad)
    print(f"\n{len(bad)}/{total} modules cannot be imported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
