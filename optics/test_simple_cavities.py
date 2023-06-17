import numpy as np
from wield.control.SFLU import SFLU, optics, nx2tikz
from sflu_components import elements, edges
from sflu_components.lib import MatrixLib, adjoint, Vnorm_sq, Minv
from gwinc.struct import Struct
from gwinc import load_budget
import gwinc.noise.quantum_lib as qlib
import scipy.constants as scc
from copy import deepcopy
import pytest


pi2i = 2j * np.pi

def sflu_FP():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "ITM", optics.BasisMirror(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "ETM", elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )
    ifo.edges.update({
        ("ETM.fr.i", "ITM.fr.o"): "ARM.L",
        ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
    })

    ifo["ITM"].locations.update({
        "bk.i.exc": (7, -7),
        "bk.o.tp": (7, 7),
    })
    ifo["ITM"].edges.update({
        ("bk.i", "bk.i.exc"): "ITM.to",
        ("bk.o.tp", "bk.o"): "ITM.fr",
    })
    ifo["ETM"].locations["fr.o.exc"] = (-3, -10)
    ifo["ETM"].edges[("fr.o", "fr.o.exc")] = "1"

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def test_FP(tpath_join, pprint, plotTF):
    sflu = sflu_FP()
    sflu.reduce_auto()

    F_Hz = np.logspace(-2, 5, 2000)

    Larm_m = 4e3
    Ti = 0.014
    Te = 0  # consider this

    nhom = 1
    mlib = MatrixLib(nhom=nhom)
    if nhom == 0:
        qmlib = qlib.mats_planewave
        MM_ARM_L = 0
        MM_ARM_rad = 0
    elif nhom == 1:
        qmlib = qlib.mats_mode_mismatch
        MM_ARM_L = 0.3
        MM_ARM_rad = 0
    lambda_m = 1064e-9
    ARM_gouy_rad = 5 * np.pi / 180  # 155
    ARM_detune_rad = 0 * np.pi / 180
    MM_ARM = mlib.MrotationMM(MM_ARM_L, MM_ARM_rad)
    MM_ARMi = Minv(MM_ARM)
    M_kg = 40
    Parm_W = 750e3

    suscept = lambda F_Hz: -1 / (M_kg * (2 * np.pi * F_Hz)**2)
    # suscept = lambda F_Hz: np.zeros_like(F_Hz)

    ###########################################################################
    # SFLU calculation
    ###########################################################################

    # edge definitions
    edge_objs = Struct()
    edge_objs.ETM = edges.RPMirrorEdge(
        "ETM",
        Thr=Te,
        suscept=suscept,
        lambda_m=lambda_m,
        mlib=mlib,
    )
    edge_objs.ITM = edges.MirrorEdge(
        "ITM",
        Thr=Ti,
        mlib=mlib,
    )
    edge_objs.ARM = edges.LinkEdge(
        "ARM.L",
        L_m=Larm_m,
        gouy_rad=ARM_gouy_rad,
        detune_rad=-ARM_detune_rad,
        mlib=mlib,
    )
    edge_objs.ITM_to = edges.LinkEdge(
        "ITM.to",
        L_m=0,
        MM_to=MM_ARM,
        mlib=mlib,
    )
    edge_objs.ITM_fr = edges.LinkEdge(
        "ITM.fr",
        L_m=0,
        MM_fr=MM_ARMi,
        mlib=mlib,
    )

    # DC calculation
    edge_map = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        # "ITM.to": mlib.Id,
        # "ITM.fr": mlib.Id,
    }
    edgesDC = deepcopy(edge_map)

    for edge_obj in edge_objs.values():
        edgesDC.update(edge_obj.edgesDC())
    tp_dc = {
        "ETM.fr.i.tp",
        "ETM.fr.o.tp",
        "ETM.bk.o.tp",
    }

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        tp_dc,
        {
            "ITM.bk.i.exc": mlib.LO(np.pi/2),
        },
    )

    # power correction
    dc_power = Vnorm_sq(resultsDC["ETM.fr.i.tp"])
    power_correction = np.sqrt(Parm_W / dc_power)
    for k, v in resultsDC.items():
        resultsDC[k] = v * power_correction
    pprint('Arm target {:0.1f} kW'.format(Parm_W * 1e-3))
    pprint('Arm power {:0.1f} kW'.format(Vnorm_sq(resultsDC["ETM.fr.i.tp"]) * 1e-3))

    # AC calculation
    edgesAC = deepcopy(edge_map)
    for edge_obj in edge_objs.values():
        edgesAC.update(edge_obj.edgesAC(F_Hz, resultsDC))

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC = compAC.inverse_row(
        {"ITM.bk.o.tp": None},
        {
            "ITM.bk.i.exc",
            "ETM.pos.exc",
            "ETM.fr.o.exc",
        },
    )

    ###########################################################################
    # Matrix calculation
    ###########################################################################

    Re = 1 - Te
    K = 8 * np.pi * suscept(F_Hz) / (scc.c * lambda_m) * 2 * Re * np.sqrt(Parm_W)

    ti = np.sqrt(Ti)
    rITM = qmlib.diag(np.sqrt(1 - Ti))
    rETM = np.sqrt(Re) * qmlib.RPNK(K)

    delayARM = qmlib.diag(np.exp(-pi2i * F_Hz * Larm_m / scc.c))
    phaseARM = delayARM @ qmlib.Mrotation(-ARM_detune_rad, ARM_gouy_rad)

    rtARM = rETM @ phaseARM @ rITM @ phaseARM
    clARM = qmlib.Minv(qmlib.Id - rtARM)
    reflARM = rITM - ti**2 * phaseARM @ clARM @ rETM @ phaseARM
    transARM = ti * phaseARM @ clARM
    reflARM = MM_ARMi @ reflARM @ MM_ARM
    transARM = MM_ARMi @ transARM
    pprint(MM_ARMi)

    ###########################################################################
    # Results
    ###########################################################################

    LOa = adjoint(mlib.LO(0))

    def field_to_plant(ArmTrans):
        fieldsDC_fr_i = resultsDC["ETM.fr.i.tp"]
        lambda_m = edge_objs.ETM.lambda_m
        px = 4 * np.pi / lambda_m * np.sqrt(Re) * mlib.Mrotation(np.pi/2)
        d_sense = np.squeeze(LOa @ ArmTrans @ px @ fieldsDC_fr_i)
        # LOdotArmPhase = (LOa @ ArmTrans)[..., 0, 1]
        # d_sense = 4 * np.pi / lambda_m * LOdotArmPhase * np.sqrt(Parm_W)
        return d_sense

    plant_mech_sflu = np.squeeze(LOa @ resultsAC["ETM.pos.exc"])
    plant_field_sflu = field_to_plant(resultsAC["ETM.fr.o.exc"])
    plant_mat = field_to_plant(transARM)
    refl_sflu = (LOa @ resultsAC["ITM.bk.i.exc"])[..., 0, 1]
    refl_mat = (LOa @ reflARM)[..., 0, 1]
    pprint(refl_sflu.shape)

    fig = plotTF(F_Hz, plant_mech_sflu, label='SFLU mech')
    plotTF(F_Hz, plant_field_sflu, *fig.axes, ls='--', label="SFLU field")
    plotTF(F_Hz, plant_mat, *fig.axes, ls=':', c='xkcd:red', label='Matrix')
    fig.axes[0].legend()
    fig.savefig(tpath_join("plant.pdf"))

    fig = plotTF(F_Hz, refl_sflu, label='SFLU')
    plotTF(F_Hz, refl_mat, *fig.axes, ls='--', label='Matrix')
    fig.axes[0].legend()
    fig.savefig(tpath_join("refl.pdf"))

@pytest.mark.parametrize("sflu_func", [sflu_FP])
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
