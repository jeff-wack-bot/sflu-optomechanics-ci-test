"""
Paper reproductions plot with matplotlib's mathtext, never LaTeX.

Older examples elsewhere in the suite switch ``text.usetex`` on globally, and
in a full run that setting leaks into whatever module runs next. Labels
written for mathtext are not always valid LaTeX, so each paper test runs with
matplotlib's defaults restored.
"""
import matplotlib as mpl
import pytest


@pytest.fixture(autouse=True)
def mathtext_defaults():
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        yield
