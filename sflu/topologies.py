"""
Reusable topologies: the few graphs that most interferometers are made of.

Each builder returns a ``GraphElement`` describing which optics exist and how
their ports connect. Hand it to ``sflu.solve.build`` to get a solvable graph.
They carry no physical parameters -- those come from edge objects, so one
topology serves every parameter set. Every builder exposes the same two
external nodes:

``<IN>.bk.i.exc``   the input port (carrier, vacuum or squeezed light enters)
``<IN>.bk.o.tp``    the output port (what leaves, toward the readout)

where ``<IN>`` is the first optic the light meets. Radiation-pressure mirrors
also expose ``<name>.pos.exc``, a displacement drive -- the GW signal, for
the end mirror of an arm.

Edge names each builder expects (so the edge objects must provide them):

``fp_arm``            ``ITM``, ``ETM`` mirror edges, ``ARM.L`` link
``signal_recycled``   adds ``SRM`` mirror edge and ``SRC.L`` link; with
                      ``internal_squeezer=True`` the SRC link is split into
                      ``SRC.L``, ``SQZ.to`` / ``SQZ.fr`` (squeezer, one per
                      direction) and ``SRC.L2``
``filter_cavity``     ``<name>1`` input mirror, ``<name>2`` end mirror,
                      ``<name>.L`` link

Chaining stages that are solved separately -- an arm followed by output
filter cavities, say -- is ``cascade``.
"""
import numpy as np
import scipy.constants as scc
from wield.control.SFLU import optics

from . import elements


def _ports(ifo, name):
    """Give optic ``name`` an external input and output on its back face."""
    ifo[name].locations.update({"bk.i.exc": (7, -7), "bk.o.tp": (7, 7)})
    ifo[name].edges.update({("bk.i", "bk.i.exc"): "1", ("bk.o.tp", "bk.o"): "1"})


def fp_arm(itm_rp=False, loss_ports=False):
    """Fabry-Perot arm: ITM (input) and a radiation-pressure ETM.

    Parameters
    ----------
    itm_rp : bool
        Give the ITM radiation pressure too (``ITM.pos.exc`` appears).
    loss_ports : bool
        Expose ``ITM.frL.i`` / ``ETM.frL.i`` (and back-face) loss inputs. The
        mirror edges must then be built with ``loss_ports=True`` as well.

    The ETM's back face ``ETM.bk.i`` is also an input: vacuum through its
    transmission, if it has any.
    """
    ifo = optics.GraphElement()
    itm = elements.RPMirrorElement if itm_rp else elements.MirrorElement
    ifo.subgraph_add("ITM", itm(loss_ports=loss_ports),
                     translation_xy=(25, 0), rotation_deg=180)
    ifo.subgraph_add("ETM", elements.RPMirrorElement(loss_ports=loss_ports),
                     translation_xy=(55, 0), rotation_deg=0)
    ifo.edges.update({
        ("ETM.fr.i", "ITM.fr.o"): "ARM.L",
        ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
    })
    _ports(ifo, "ITM")
    return ifo


def signal_recycled(loss_ports=False, internal_squeezer=False):
    """Signal-recycled arm: SRM (input), ITM, radiation-pressure ETM.

    This is the differential mode of a dual-recycled Michelson with the
    beamsplitter folded away -- the reduction Buonanno & Chen use, and the
    same one as ``sflu.models.coupled_cavity``. Light enters and leaves
    through the SRM.

    With ``internal_squeezer=True`` a squeezer sits inside the signal
    recycling cavity, as in the quantum expander (Korobko et al.): the
    SRM-ITM path is ``SRC.L`` -> ``SQZ`` -> ``SRC.L2``, the squeezer being a
    pair of one-way ``SQZEdge`` edges named ``SQZ.to`` (toward the ITM) and
    ``SQZ.fr`` (toward the SRM).
    """
    ifo = optics.GraphElement()
    ifo.subgraph_add("SRM", elements.MirrorElement(loss_ports=loss_ports),
                     translation_xy=(0, 0), rotation_deg=180)
    ifo.subgraph_add("ITM", elements.MirrorElement(loss_ports=loss_ports),
                     translation_xy=(45, 0), rotation_deg=180)
    ifo.subgraph_add("ETM", elements.RPMirrorElement(loss_ports=loss_ports),
                     translation_xy=(95, 0), rotation_deg=0)
    ifo.edges.update({
        ("ETM.fr.i", "ITM.fr.o"): "ARM.L",
        ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
    })
    if internal_squeezer:
        ifo.locations.update({
            "SQZ.a.i": (20, 3), "SQZ.a.o": (20, -3),
            "SQZ.b.i": (28, -3), "SQZ.b.o": (28, 3),
        })
        ifo.edges.update({
            # SRM -> ITM
            ("SQZ.a.i", "SRM.fr.o"): "SRC.L",
            ("SQZ.b.o", "SQZ.a.i"): "SQZ.to",
            ("ITM.bk.i", "SQZ.b.o"): "SRC.L2",
            # ITM -> SRM
            ("SQZ.b.i", "ITM.bk.o"): "SRC.L2",
            ("SQZ.a.o", "SQZ.b.i"): "SQZ.fr",
            ("SRM.fr.i", "SQZ.a.o"): "SRC.L",
        })
    else:
        ifo.edges.update({
            ("ITM.bk.i", "SRM.fr.o"): "SRC.L",
            ("SRM.fr.i", "ITM.bk.o"): "SRC.L",
        })
    _ports(ifo, "SRM")
    return ifo


def filter_cavity(name="FC", loss_ports=False):
    """Two-mirror cavity used in reflection: ``<name>1`` input, ``<name>2`` end.

    Inputs are ``<name>1.bk.i.exc`` (the light to be filtered) and, through
    the end mirror, ``<name>2.bk.i``; with ``loss_ports`` also the mirrors'
    loss inputs, e.g. ``<name>2.frL.i``. The output is ``<name>1.bk.o.tp``.
    """
    ifo = optics.GraphElement()
    ifo.subgraph_add(f"{name}1", elements.MirrorElement(loss_ports=loss_ports),
                     translation_xy=(0, 0), rotation_deg=180)
    ifo.subgraph_add(f"{name}2", elements.MirrorElement(loss_ports=loss_ports),
                     translation_xy=(30, 0), rotation_deg=0)
    ifo.edges.update({
        (f"{name}2.fr.i", f"{name}1.fr.o"): f"{name}.L",
        (f"{name}1.fr.i", f"{name}2.fr.o"): f"{name}.L",
    })
    _ports(ifo, f"{name}1")
    return ifo


def detuning_rad(detune_Hz, L_m):
    """One-way link phase for a cavity detuned by ``detune_Hz``.

    ``LinkEdge.detune_rad`` is applied on each pass, so a round trip picks up
    twice this. Positive ``detune_Hz`` means the carrier sits *above* the
    cavity resonance, i.e. ``omega_0 - omega_res > 0`` -- KLMTV's ``xi > 0``
    and Buonanno & Chen's convention for the SR detuning -- checked
    numerically against KLMTV's filter-cavity phases in ``sflu.papers``.
    """
    return -2 * np.pi * detune_Hz * L_m / scc.c


def half_bandwidth_T(half_bw_Hz, L_m):
    """Input-mirror transmission giving a cavity of half-width ``half_bw_Hz``.

    For a cavity with a lossless, perfectly reflecting end mirror, so that
    the round-trip amplitude is ``sqrt(1 - T)``: the *exact* pole of
    ``1 / (1 - sqrt(1-T) exp(2 i Omega L / c))`` sits at ``gamma`` when
    ``T = 1 - exp(-4 gamma L / c)``.

    The papers write ``gamma = T c / 4L``, the first-order version. The two
    differ by ``O(T)`` -- about 1% for an aLIGO-like arm -- and that is not
    negligible: variational readout cancels back-action by subtracting two
    numbers of size ``K``, which reaches ~10^3 at low frequency, so a 1% error
    in the pole leaves residual back-action ten times the shot noise.
    """
    gamma = 2 * np.pi * half_bw_Hz
    return -np.expm1(-4 * gamma * L_m / scc.c)


def cascade(first, second, through):
    """Chain two solved stages: ``first``'s output feeds ``second``'s input.

    Parameters
    ----------
    first, second : dict
        ``{input: transfer matrix to that stage's output}``, as from
        ``solve_ac``.
    through : str
        The input of ``second`` that ``first``'s output enters.

    Returns
    -------
    dict of every input of either stage -> transfer matrix to ``second``'s
    output. Mechanical drives in ``first`` (``(N, dim, 1)``) pass through like
    any other input.
    """
    G = second[through]
    out = {k: G @ v for k, v in first.items()}
    for k, v in second.items():
        if k == through:
            continue
        if k in out:
            raise ValueError(f"input {k!r} appears in both stages")
        out[k] = v
    return out


__all__ = [
    "cascade",
    "detuning_rad",
    "filter_cavity",
    "fp_arm",
    "half_bandwidth_T",
    "signal_recycled",
]
