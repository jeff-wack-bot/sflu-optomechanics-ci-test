"""
Optomechanical models, split into the three things they actually are.

Each model is built from three separable pieces, which used to be tangled
together inside a single ``test_*.py``:

``topology``
    Builds the SFLU graph: which optics exist and how they are wired.
    Pure structure, no physics parameters. e.g. ``sflu_CoupledCav()``.

``plant``
    Turns parameters into edge objects, hands them to the graph, and solves
    for the transfer functions. e.g. ``CoupledCavity()``.

``budget``
    Turns those transfer functions into a noise PSD and a per-port loss
    breakdown. This step is identical across models, so there is exactly one
    implementation: ``budget.quantum_budget()``.
"""
from .budget import quantum_budget
from .coupled_cavity import CoupledCavity, intSqzQuantum, sflu_CoupledCav
from .filter_cavity import FilterCavity, sflu_FilterCavity
from .int_fd_sqz import CoupledCavityIntFC, intFDsqzQuantum, sflu_CCwIntFDSqz

__all__ = [
    "CoupledCavity",
    "CoupledCavityIntFC",
    "FilterCavity",
    "intFDsqzQuantum",
    "intSqzQuantum",
    "quantum_budget",
    "sflu_CCwIntFDSqz",
    "sflu_CoupledCav",
    "sflu_FilterCavity",
]
