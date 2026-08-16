"""
Compatibility shim: the matrix library now lives in ``sflu_components.lib``.

This module used to be a fork of ``sflu_components/lib.py`` carrying three
additions (``Vnorm_sqA``, ``MatrixLib.SQZc``, ``MatsHelper``). Because the fork
was a strict superset -- the two files differed only by those three pure
additions -- Stage 2a of ``REFACTOR_PLAN.md`` moved them into
``sflu_components.lib`` and left this re-export behind.

The point is not tidiness. Before this change two distinct ``MatrixLib``
classes were alive in the same process: models imported ``MatrixLib`` from
``sflu_components.lib`` while their edge objects defaulted to a *different*
``MatrixLib`` bound from here at import time. Now there is exactly one class,
so ``sflu_components.lib.MatrixLib is fromgwinc.intsqz.lib.MatrixLib``.

Import from ``sflu_components.lib`` in new code; this shim exists so the
existing model files keep working and will go away once they are updated.
"""
from sflu_components.lib import (  # noqa: F401 - re-exported for compatibility
    A2,
    A2i,
    MatrixLib,
    MatsHelper,
    Minv,
    Mrotation2,
    RPNK2,
    SQZ2,
    Vnorm_sq,
    Vnorm_sqA,
    adjoint,
    block_diag,
    matrix_stack,
    matrix_stack_id,
    transpose,
)

__all__ = [
    "A2",
    "A2i",
    "MatrixLib",
    "MatsHelper",
    "Minv",
    "Mrotation2",
    "RPNK2",
    "SQZ2",
    "Vnorm_sq",
    "Vnorm_sqA",
    "adjoint",
    "block_diag",
    "matrix_stack",
    "matrix_stack_id",
    "transpose",
]
