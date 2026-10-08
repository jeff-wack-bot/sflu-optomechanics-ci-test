"""
Compatibility aliases. The library now lives in the ``sflu`` package.

Stage 7 of ``REFACTOR_PLAN.md`` folded this directory into ``sflu`` so that
the whole layered stack -- lib, params, models -- is one importable package:

    sflu_components.lib          ->  sflu.lib
    sflu_components.edges        ->  sflu.edges
    sflu_components.elements     ->  sflu.elements
    sflu_components.quantum_lib  ->  sflu.quantum_lib
    sflu_components.simlib       ->  sflu.simlib

Each submodule here replaces itself in ``sys.modules`` with the real one, so
``sflu_components.lib.MatrixLib is sflu.lib.MatrixLib``: there is still exactly
one copy of every class. Kept for ``models/`` and for any uncommitted work
that still uses the old names; new code should import from ``sflu``.
"""
