"""
Coherence checks for the documentation generator's configuration.

These run in the normal suite and take no measurable time, so a rename that
orphans a documentation page is caught immediately rather than at the next
docs build. The generator itself repeats these checks under ``--strict``,
where it can also see whether an example actually produced figures.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_sflu_generate_docs", ROOT / "docs" / "generate_docs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()


def test_listed_modules_exist():
    """Every documented example still exists at the path MODULES gives."""
    missing = [m["path"] for m in gen.MODULES if not (ROOT / m["path"]).exists()]
    assert not missing, (
        f"documented examples not found: {missing}. "
        f"Update MODULES in docs/generate_docs.py."
    )


def _tracked_by_git():
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.split())


def test_excluded_modules_exist():
    """EXCLUDED should not accumulate entries for files that are long gone.

    A file that git still tracks but that is deleted in the working tree is an
    uncommitted deletion in progress, not a stale entry, so it does not count.
    """
    tracked = _tracked_by_git()
    stale = [
        p for p in gen.EXCLUDED
        if not (ROOT / p).exists() and (tracked is None or p not in tracked)
    ]
    assert not stale, (
        f"EXCLUDED names files that no longer exist: {stale}. "
        f"Drop them from docs/generate_docs.py."
    )


def test_no_module_both_listed_and_excluded():
    overlap = {m["path"] for m in gen.MODULES} & set(gen.EXCLUDED)
    assert not overlap, f"listed and excluded at once: {sorted(overlap)}"


def test_every_example_is_documented_or_excluded():
    """A new example must get a page, or an explicit reason for not having one."""
    undocumented = gen.audit_coverage()
    assert not undocumented, (
        "these example modules have no documentation page:\n  "
        + "\n  ".join(undocumented)
        + "\nAdd each to MODULES, or to EXCLUDED with a reason, in "
          "docs/generate_docs.py."
    )


def test_exclusion_reasons_are_given():
    blank = [p for p, why in gen.EXCLUDED.items() if not (why or "").strip()]
    assert not blank, f"EXCLUDED entries without a reason: {blank}"


@pytest.mark.parametrize("info", gen.MODULES, ids=lambda i: i["path"])
def test_module_entries_are_well_formed(info):
    for key in ("path", "title", "section", "trust"):
        assert info.get(key), f"{info.get('path', info)} is missing {key!r}"
    assert info["trust"] in gen.TRUST, (
        f"{info['path']}: unknown trust level {info['trust']!r}; "
        f"known: {sorted(gen.TRUST)}"
    )
