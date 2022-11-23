"""
Test calculation of PI gains
"""

import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
from sflu_components import elements, edges
from sflu_components.lib import MatrixLib, adjoint
import scipy.constants as scc
from gwinc.struct import Struct
from copy import deepcopy
import pytest


def sflu_FP():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "IX", optics.BasisMirror(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "EX", elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )

    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): "tau",
        ("IX.fr.i", "EX.fr.o"): "tau",
    })

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def sflu_DRFPMI():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "IX", optics.BasisMirror(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "EX", elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "IY", optics.BasisMirror(),
        translation_xy=(0, 25),
        rotation_deg=90+180,
    )
    ifo.subgraph_add(
        "EY", optics.BasisMirror(),
        translation_xy=(0, 55),
        rotation_deg=90,
    )
    ifo.subgraph_add(
        "BS", optics.BeamSplitter(),
        translation_xy=(0, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "PRM", optics.BasisMirror(),
        translation_xy=(-25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "SEM", optics.BasisMirror(),
        translation_xy=(0, -25),
        rotation_deg=90+180,
    )

    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): "XARM.tau",
        ("IX.fr.i", "EX.fr.o"): "XARM.tau",
        ("IX.bk.i", "BS.bkA.o"): "BSX.tau",
        ("BS.bkA.i", "IX.bk.o"): "BSX.tau",

        ("EY.fr.i", "IY.fr.o"): "YARM.tau",
        ("IY.fr.i", "EY.fr.o"): "YARM.tau",
        ("IY.bk.i", "BS.frB.o"): "BSY.tau",
        ("BS.frB.i", "IY.bk.o"): "BSY.tau",

        ("PRM.fr.i", "BS.frA.o"): "PRC.tau",
        ("BS.frA.i", "PRM.fr.o"): "PRC.tau",

        ("SEM.fr.i", "BS.bkB.o"): "SEC.tau",
        ("BS.bkB.i", "SEM.fr.o"): "SEC.tau",
    })

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def gouyRT_rad(Larm_m, Ri_m, Re_m):
    gi = 1 - Larm_m / Ri_m
    ge = 1 - Larm_m / Re_m
    return 2 * np.arccos(np.sign(gi) * np.sqrt(gi * ge))


par = Struct(
    Ti           = 0.0148,
    Te           = 5e-6,
    Tbs          = 0.5,
    Tp           = 0.031,
    Ts           = 0.324,
    Lhr          = 1e-4 / 2,
    Lhr_aux      = 0,
    Parm_W       = 750e3,
    Larm_m       = 3994.75,
    lambda_m     = 1064e-9,
    M_kg         = 40,
    Qm           = 37195988,
    fmech_Hz     = 6.4053e3,
    Ri_m         = 1934,
    Re_m         = 2245,
    Lprc_m       = 0,  # 55,
    Lsec_m       = 0,  # 55,
    SEC_gouy_deg = 21.3,
    PRC_gouy_deg = 29,
    overlap      = np.sqrt(1.453e-5),
    mode_order   = 2,
)


def sflu_results(sflu_func, par, *args, **kwargs):
    sflu = sflu_func(*args, **kwargs)
    sflu.reduce_auto()

    par = deepcopy(par)
    mlib = MatrixLib()
    F_Hz = par.fmech_Hz
    re = np.sqrt(1 - par.Te - par.Lhr)
    te = np.sqrt(par.Te)
    arm_gouy_rad = par.mode_order * gouyRT_rad(par.Larm_m, par.Ri_m, par.Re_m) / 2
    SEC_gouy_rad = par.mode_order * par.SEC_gouy_deg * np.pi/180
    PRC_gouy_rad = par.mode_order * par.PRC_gouy_deg * np.pi/180

    def suscept_m_N(F_Hz):
        den = par.fmech_Hz**2 - F_Hz**2 + 1j * par.fmech_Hz * F_Hz / par.Qm
        return 1 / par.M_kg / (2 * np.pi)**2 / den

    EX = edges.RPMirrorEdge(
        "EX", Thr=par.Te, Lhr=par.Lhr,
        suscept_m_N=suscept_m_N, overlap=par.overlap,
    )

    if sflu_func == sflu_FP:
        non_RP_optics = Struct(
            IX  = edges.MirrorEdge("IX", Thr=par.Ti, Lhr=par.Lhr),
        )
        links = Struct(
            L_ARM = edges.LinkEdge("tau", par.Larm_m, 0 + arm_gouy_rad),
        )
    elif sflu_func == sflu_DRFPMI:
        non_RP_optics = Struct(
            IX  = edges.MirrorEdge("IX", Thr=par.Ti, Lhr=par.Lhr),
            EY  = edges.MirrorEdge("EY", Thr=par.Te, Lhr=par.Lhr),
            IY  = edges.MirrorEdge("IY", Thr=par.Ti, Lhr=par.Lhr),
            BS  = edges.BSEdge("BS", Thr=par.Tbs, Lhr=0),
            PRM = edges.MirrorEdge("PRM", Thr=par.Tp, Lhr=0),
            SEM = edges.MirrorEdge("SEM", Thr=par.Ts, Lhr=0),
        )

        links = Struct(
            L_XARM = edges.LinkEdge("XARM.tau", par.Larm_m, 0 + arm_gouy_rad),
            L_YARM = edges.LinkEdge("YARM.tau", par.Larm_m, 0 + arm_gouy_rad),
            L_BSX  = edges.LinkEdge("BSX.tau", 0, 0 + 0),
            L_BSY  = edges.LinkEdge("BSY.tau", 0, 0 + 0),
            L_PRC  = edges.LinkEdge("PRC.tau", par.Lprc_m, 0 + PRC_gouy_rad),
            L_SEC  = edges.LinkEdge("SEC.tau", par.Lsec_m, np.pi/2 + SEC_gouy_rad),
        )

    edge_map = {
        "1": mlib.Id,
        "1s": np.eye((1)),
    }

    fieldsDC_fr_i = np.sqrt(par.Parm_W) * mlib.LO(np.pi/2)
    fieldsDC_fr_o = -re * fieldsDC_fr_i
    fieldsDC_bk_o = te * fieldsDC_fr_i
    fieldsDC_bk_i = 0 * mlib.Id_v

    resultsDC = {
        "EX.fr.i.tp": fieldsDC_fr_i,
        "EX.fr.o.tp": fieldsDC_fr_o,
        "EX.bk.i.tp": fieldsDC_bk_i,
        "EX.bk.o.tp": fieldsDC_bk_o,
    }

    for optic in non_RP_optics.values():
        edge_map.update(optic.edgesAC(F_Hz, resultsDC))
    for link in links.values():
        edge_map.update(link.edgesAC(F_Hz, resultsDC))

    edge_map.update(EX.edgesAC(F_Hz, resultsDC))

    comp = sflu.computer()
    comp.compute(edge_map=edge_map)
    results = comp.inverse_row(
        {"EX.pos.tp": None},
        {"EX.pos.exc"},
    )
    return results


# known correct gains
# compared with Sławek's code which gives
# FP:     8.823e-6
# DRFPMI: 1.034e-6 (losses implemented a bit different there)
ref_gains = Struct(
    FP = Struct(gain=8.834e-6, cl=np.array(1.00000818+0.00080928j)),
    DRFPMI = Struct(gain = 1.090e-6, cl=np.array(1.00000043+0.00081244j)),
)

@pytest.mark.parametrize(
    "sflu_func, ref_gain",
    [
        (sflu_FP, ref_gains.FP),
        (sflu_DRFPMI, ref_gains.DRFPMI),
    ],
)
def test_gain_single(sflu_func, ref_gain, pprint):
    # sflu_func = request.getfixturevalue(sflu_func)
    results = sflu_results(sflu_func, par)
    cl = results["EX.pos.exc"]
    gain = np.real(1 - 1/cl)
    pprint("closed loop:", cl)
    pprint("gain:", gain)
    assert np.isclose(cl, ref_gain.cl)
    assert np.isclose(gain, ref_gain.gain)


@pytest.mark.parametrize('sflu_func', [sflu_FP, sflu_DRFPMI])
def plot_graph(sflu_func, tpath_join):
    sflu = sflu_func()
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-8, rX=+8, Y=3, dY=-3),
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-8, rX=+8, Y=3, dY=-3),
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname=tpath_join("testG.pdf"),
        scale="10pt",
    )
