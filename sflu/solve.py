"""
The DC/AC solve that every model performs, written once.

Before this module each model repeated the same ~40 lines: seed an edge map
with the ``"1"``/``"1s"`` identities, merge in every edge object's map, build a
computer, invert, and rescale the DC fields to a target power. Those lines
differed only in which nodes were asked for. Here they are four functions:

``edge_map(edge_objs, mlib, kind, ...)``  all edge objects -> one edge map
``solve_dc(sflu, edge_objs, mlib, ...)``  carrier fields at test points
``scale_dc(resultsDC, node, P_W)``        rescale those fields to a power
``solve_ac(sflu, edge_objs, mlib, ...)``  transfer matrices to a readout

None of this changes what a model computes. The reference internal-squeezing
models still spell the steps out, because their numbers are pinned bit for bit
by ``tools/regression``; new models, including everything in ``sflu.papers``,
use these.

Graphs are reduced in place by ``reduce_auto()``, so build a fresh topology
for every solve rather than reusing one.
"""
import numpy as np
from wield.control.SFLU import SFLU

from .lib import Vnorm_sq


def build(ifo):
    """Turn a ``GraphElement`` into a reduced SFLU graph, ready to solve."""
    sflu = SFLU.SFLU(edges=ifo.build_edges(), graph=True)
    ifo.update_sflu(sflu)
    sflu.reduce_auto()
    return sflu


def edge_map(edge_objs, mlib, kind="AC", F_Hz=None, resultsDC=None, extra=None):
    """Merge the edge maps of every edge object into one dict.

    Parameters
    ----------
    edge_objs : iterable or mapping of edge objects
        ``MirrorEdge``, ``LinkEdge``, ``RPMirrorEdge``, ...
    kind : ``"DC"``, ``"AC"`` or ``"ACSS"``
        Which edge map to ask each object for. ``"ACSS"`` gives state-space
        edges for ``SScomputer``.
    extra : dict, optional
        Additional fixed edges, e.g. a basis change on an input port.

    The identities ``"1"`` (field, ``mlib.Id``) and ``"1s"`` (scalar,
    ``mlib.Id_s``) that test-point and excitation edges refer to are always
    included.
    """
    if hasattr(edge_objs, "values"):
        edge_objs = edge_objs.values()
    edges = {"1": mlib.Id, "1s": mlib.Id_s}
    for obj in edge_objs:
        if kind == "DC":
            edges.update(obj.edgesDC())
        elif kind == "AC":
            edges.update(obj.edgesAC(F_Hz=F_Hz, resultsDC=resultsDC))
        elif kind == "ACSS":
            edges.update(obj.edgesACSS(F_Hz=F_Hz, resultsDC=resultsDC))
        else:
            raise ValueError(f"kind must be DC, AC or ACSS, not {kind!r}")
    if extra:
        edges.update(extra)
    return edges


def solve_dc(sflu, edge_objs, mlib, drive, test_points, extra=None):
    """Carrier fields at ``test_points`` for a carrier entering at ``drive``.

    Parameters
    ----------
    drive : dict
        ``{input node: field vector}``, e.g. ``{"ITM.bk.i.exc": mlib.LO(pi/2)}``
        for unit carrier in the amplitude quadrature.
    test_points : iterable of node names

    Returns
    -------
    dict of node -> ``(dim, 1)`` field vector, in sqrt(W) per unit drive.
    """
    comp = sflu.computer(eye=mlib.Id)
    comp.compute(edge_map=edge_map(edge_objs, mlib, "DC", extra=extra))
    return comp.inverse_col(set(test_points), dict(drive))


def scale_dc(resultsDC, node, P_W):
    """Rescale every DC field so the power at ``node`` is ``P_W``.

    The usual way to set a model's operating point: solve with unit drive,
    then fix the circulating power, rather than working out the input power
    the topology needs.
    """
    scale = np.sqrt(P_W / Vnorm_sq(resultsDC[node]))
    return {k: v * scale for k, v in resultsDC.items()}


def solve_ac(sflu, edge_objs, mlib, F_Hz, readout, inputs, resultsDC=None,
             extra=None, use_SS=False):
    """Transfer matrices from each of ``inputs`` to the ``readout`` node.

    Parameters
    ----------
    readout : str
        The output node, e.g. ``"ITM.bk.o.tp"``.
    inputs : iterable of str
        Excitation nodes: optical input ports, loss ports, ``"ETM.pos.exc"``.
    resultsDC : dict, optional
        Carrier fields, needed by any radiation-pressure edge.
    use_SS : bool
        Solve in state space (``SScomputer``) instead of frequency by
        frequency. Needs every edge to implement ``edgesACSS``.

    Returns
    -------
    dict of input -> ``(N, dim, dim)`` array for optical inputs, or
    ``(N, dim, 1)`` for a scalar (mechanical) input.
    """
    Rmap = {readout: None}
    Cset = set(inputs)
    resultsDC = resultsDC or {}
    if use_SS:
        comp = sflu.SScomputer(eye=mlib.Id)
        comp.SScompletion(edge_map(edge_objs, mlib, "ACSS", F_Hz, resultsDC, extra))
        return comp.inverse_row_fresponse(Rmap=Rmap, Cset=Cset, F_Hz=F_Hz)
    comp = sflu.computer(eye=mlib.Id)
    comp.compute(edge_map=edge_map(edge_objs, mlib, "AC", F_Hz, resultsDC, extra))
    return comp.inverse_row(Rmap=Rmap, Cset=Cset)


__all__ = ["build", "edge_map", "scale_dc", "solve_ac", "solve_dc"]
