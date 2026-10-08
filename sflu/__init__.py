"""
SFLU optomechanics: one package, in layers.

Dependencies run strictly downward in this list; nothing imports from a layer
below it, and nothing in the package imports the example suite.

``sflu.lib``, ``sflu.edges``, ``sflu.elements``      -- lib
    Quadrature matrix algebra (``MatrixLib``), the edge objects that turn
    physical parameters into SFLU edge weights, and the graph elements
    (mirrors, beamsplitters) that say how ports connect.
``sflu.solve``                                       -- lib
    The DC/AC solve every model performs: collect edge maps, invert the
    graph, scale the DC fields to a target power.
``sflu.params``                                      -- params
    IFO parameter sets by name, and derived quantities (``standardize_params``).
``sflu.models``                                      -- model
    Topology (SFLU graph) -> plant (edges -> transfer functions) -> budget
    (transfer functions -> noise PSD).
``sflu.papers``                                      -- model
    Interferometers from the literature, each with the paper's closed-form
    result alongside to check the graph against.

Supporting modules: ``sflu.plotting`` (transfer-function plots, used by the
example fixtures), ``sflu.quantum_lib`` (the older plane-wave matrix library,
kept for the examples that cross-check against it), ``sflu.simlib``
(Optickle/Finesse comparison harness), and ``sflu._vendor.gwinc`` (the small
part of pygwinc the models need; see ``docs/GWINC_DEPENDENCY.md``).

Before Stage 3 of ``REFACTOR_PLAN.md`` the models lived inside ``test_*.py``
files; before Stage 7 the library lived in a separate ``sflu_components``
package, which survives only as import aliases.
"""
