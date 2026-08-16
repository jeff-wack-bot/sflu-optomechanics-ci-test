"""
SFLU optomechanics: importable models.

Everything here can be imported without pytest. That is the point of the
package: before Stage 3 of ``REFACTOR_PLAN.md`` the models lived inside
``test_*.py`` files, so the only way to reuse one was to import a test module
from another test module.

Layers, and the direction dependencies run:

    sflu.params      ifo Struct -> derived parameters (standardize_params)
        ^
    sflu.models      topology (SFLU graph) -> plant (edges -> transfer
                     functions) -> budget (transfer functions -> PSD)
        ^
    examples         test_*.py: load params, call a model, plot, assert

Models depend on ``sflu_components`` for the matrix and edge libraries, and on
nothing in the example suite.
"""
