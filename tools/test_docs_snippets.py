"""
The Python in the hand-written guide pages must run.

The example pages cannot drift from the code because they are generated from
it. The guide pages in ``docs/static/guide/`` are written by hand, so this
executes every ```python block in each page, in order, in one namespace per
page -- a page is one worked example. A renamed function or a changed
signature then fails here instead of on a reader.
"""
import re
from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "static" / "guide"
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
PAGES = sorted(p for p in GUIDE.glob("*.md") if BLOCK.search(p.read_text()))


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_guide_page_runs(page):
    namespace = {}
    for i, block in enumerate(BLOCK.findall(page.read_text())):
        try:
            exec(compile(block, f"{page.name}[block {i}]", "exec"), namespace)
        except Exception as exc:
            pytest.fail(f"{page.name}, python block {i}: {type(exc).__name__}: {exc}")
