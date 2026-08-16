"""
Coherence checks between setup.sh and the CI workflow.

Both list the wield dependencies: setup.sh for a developer laptop, the workflow
inline so it can cache each clone against the revision it pins. Two lists drift,
so the drift is a test failure rather than a puzzling CI-only break.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP_SH = ROOT / "setup.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

DEP_RE = re.compile(
    r"(wield-[a-z]+)\s+(https://\S+?\.git)\s+(\S+)"
)


def _deps(text):
    return {m.group(1): (m.group(2), m.group(3)) for m in DEP_RE.finditer(text)}


@pytest.mark.skipif(not SETUP_SH.exists(), reason="no setup.sh")
def test_setup_and_workflow_list_the_same_dependencies():
    setup = _deps(SETUP_SH.read_text())
    workflow = _deps(WORKFLOW.read_text())
    assert setup, "no dependencies parsed out of setup.sh"
    assert setup == workflow, (
        "setup.sh and .github/workflows/ci.yml disagree about the wield "
        f"dependencies.\n  setup.sh: {setup}\n  workflow: {workflow}"
    )


def test_dependencies_are_cloned_anonymously():
    """No SSH URLs: CI must run without a deploy key."""
    for path in (SETUP_SH, WORKFLOW):
        if not path.exists():
            continue
        text = path.read_text()
        bad = re.findall(r"(?:ssh://|git@)[\w.@:/-]+\.git", text)
        assert not bad, f"{path.name} clones over SSH, which needs a key: {bad}"


def test_hash_seed_is_pinned_in_ci():
    """An unpinned PYTHONHASHSEED makes the regression baselines unmatchable."""
    text = WORKFLOW.read_text()
    assert re.search(r"PYTHONHASHSEED:\s*'?0'?", text), (
        "the workflow must pin PYTHONHASHSEED=0; see REFACTOR_PLAN.md Finding 1"
    )


def test_guard_is_blocking():
    """The numerical guard must not be allowed to fail quietly."""
    text = WORKFLOW.read_text()
    guard = re.search(
        r"- name: Guard[^\n]*\n(?P<body>(?:[ \t]+\S[^\n]*\n)+)", text
    )
    assert guard, "no Guard step found in the workflow"
    assert "continue-on-error" not in guard.group("body"), (
        "the Guard step must stay blocking"
    )
