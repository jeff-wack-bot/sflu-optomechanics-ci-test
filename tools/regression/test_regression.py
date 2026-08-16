"""
Numerical regression guard for the internal-squeezing models.

Asserts that the model entry points still produce the baseline numbers stored
in ``baselines/intsqz.npz``.  Run this before and after every step of the
structural refactor; a structural change that alters a number is a bug in the
change.

The work happens in a subprocess because the baseline is only reproducible
with ``PYTHONHASHSEED`` pinned (see ``capture_baseline.py``), and the hash seed
cannot be changed once the interpreter running pytest has started.

    pytest tools/regression/test_regression.py

To re-baseline deliberately (after a change you *intend* to alter numbers):

    python -m tools.regression.capture_baseline
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE / "baselines" / "intsqz.npz"


@pytest.mark.skipif(
    not BASELINE.exists(),
    reason="no baseline captured; run python -m tools.regression.capture_baseline",
)
def test_intsqz_numerics_unchanged():
    env = dict(os.environ, PYTHONHASHSEED="0")
    proc = subprocess.run(
        [sys.executable, "-m", "tools.regression.capture_baseline", "--check"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            "model outputs differ from the stored baseline:\n\n"
            + proc.stdout[-4000:]
            + "\n"
            + proc.stderr[-2000:]
        )
