import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
from sflu_components import elements, edges
from sflu_components.lib import MatrixLib, adjoint, Vnorm_sq
from gwinc.struct import Struct
from gwinc import load_budget
from gwinc.plant import plant_debug
from copy import deepcopy


def sflu_DRFPMI():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "IX", elements.RPMirrorElement(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "EX", elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "IY", elements.RPMirrorElement(),
        translation_xy=(0, 25),
        rotation_deg=90+180,
    )
    ifo.subgraph_add(
        "EY", elements.RPMirrorElement(),
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

    ifo["PRM"].locations.update({
        "bk.i.exc": (15, -10),
        # "bk.o.tp": (15, 10),
    })
    ifo["PRM"].edges.update({
        ("bk.i", "bk.i.exc"): "1",
        # ("bk.o.tp", "bk.o"): "1",
    })

    ifo["SEM"].locations.update({
        "bk.i.exc": (15, -10),
        "bk.o.tp": (15, 10),
    })
    ifo["SEM"].edges.update({
        ("bk.i", "bk.i.exc"): "SEC.to",
        ("bk.o.tp", "bk.o"): "SEC.fr",
    })

    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): "XARM.L",
        ("IX.fr.i", "EX.fr.o"): "XARM.L",
        ("IX.bk.i", "BS.bkA.o"): "BSX.L.i",
        ("BS.bkA.i", "IX.bk.o"): "BSX.L.o",

        ("EY.fr.i", "IY.fr.o"): "YARM.L",
        ("IY.fr.i", "EY.fr.o"): "YARM.L",
        ("IY.bk.i", "BS.frB.o"): "BSY.L.i",
        ("BS.frB.i", "IY.bk.o"): "BSY.L.o",

        ("PRM.fr.i", "BS.frA.o"): "PRC.L",
        ("BS.frA.i", "PRM.fr.o"): "PRC.L",

        ("SEM.fr.i", "BS.bkB.o"): "SEC.L",
        ("BS.bkB.i", "SEM.fr.o"): "SEC.L",
    })

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def sflu_FP():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "IX", elements.RPMirrorElement(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "EX", elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )

    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): "XARM.L",
        ("IX.fr.i", "EX.fr.o"): "XARM.L",
    })

    ifo["IX"].locations.update({
        "bk.i.exc": (15, -10),
        # "bk.o.tp": (15, 10),
    })
    ifo["IX"].edges.update({
        ("bk.i", "bk.i.exc"): "1",
        # ("bk.i.tp", "bk.o"): "1",
    })
    ifo["EX"].locations.update({
        "bk.i.exc": (15, -10),
        })
    ifo["EX"].edges.update({
        ("bk.i", "bk.i.exc"): "1",
    })

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def test_DRFPMI(tpath_join, plotTF, pprint):
    sflu = sflu_DRFPMI()
    sflu.reduce_auto()

    ifo = load_budget('Aplus').ifo
    F_Hz = np.logspace(0, 4, 2000)
    ifo.Optics.Loss = 0
    ifo.Optics.BSLoss = 0
    mats, access = plant_debug(F_Hz, ifo)
    Ti = ifo.Optics.ITM.Transmittance  # 0.014
    Te = ifo.Optics.ETM.Transmittance  # 10e-6
    Tp = ifo.Optics.PRM.Transmittance  # 0.03
    Ts = ifo.Optics.SRM.Transmittance  # 0.325
    Larm_m = ifo.Infrastructure.Length  # 4e3
    Lsec_m = ifo.Optics.SRM.CavityLength  # 56
    Lprc_m = 58
    Lasy_m = 0  # 0.08
    Lavg_m = 5.2
    Lbsx_m = Lavg_m + Lasy_m / 2
    Lbsy_m = Lavg_m - Lasy_m / 2
    Ls_m = Lsec_m - Lavg_m
    Lp_m = Lprc_m - Lavg_m
    M_kg = 40
    ARM_gouy_rad = None  # -23 * np.pi/180
    SEC_gouy_rad = None  # 19 * np.pi/180
    PRC_gouy_rad = None  # 25 * np.pi/180
    SEC_detune_rad = 0
    Parm_W = access.drfpmi.parm_W
    mlib = MatrixLib(nhom=0)

    SEC_detune_rad = np.pi/2 + SEC_detune_rad

    def suscept(F_Hz):
        return -1 / (M_kg * (2 * np.pi * F_Hz)**2)

    edge_objs = Struct()
    for opt_name in ['EX', 'EY']:
        edge_objs[opt_name] = edges.RPMirrorEdge(
            opt_name,
            Thr=Te,
            suscept=suscept,
            mlib=mlib,
        )
    for opt_name in ['IX', 'IY']:
        edge_objs[opt_name] = edges.RPMirrorEdge(
            opt_name,
            Thr=Ti,
            suscept=suscept,
            mlib=mlib,
        )
    edge_objs.BS = edges.BSEdge(
        'BS',
        Thr=0.5,
        mlib=mlib,
    )
    edge_objs.PRM = edges.MirrorEdge(
        'PRM',
        Thr=Tp,
        mlib=mlib,
    )
    edge_objs.SEM = edges.MirrorEdge(
        'SEM',
        Thr=Ts,
        mlib=mlib,
    )
    for link_name in ['XARM', 'YARM']:
        edge_objs[link_name] = edges.LinkEdge(
            link_name + '.L',
            L_m=Larm_m,
            gouy_rad=ARM_gouy_rad,
            mlib=mlib,
        )
    edge_objs['L_BSX_o'] = edges.LinkEdge(
        'BSX.L.o',
        L_m=Lbsx_m,
        mlib=mlib,
    )
    edge_objs['L_BSX_i'] = edges.LinkEdge(
        'BSX.L.i',
        L_m=Lbsx_m,
        mlib=mlib,
    )
    edge_objs['L_BSY_o'] = edges.LinkEdge(
        'BSY.L.o',
        L_m=Lbsy_m,
        mlib=mlib,
    )
    edge_objs['L_BSY_i'] = edges.LinkEdge(
        'BSY.L.i',
        L_m=Lbsy_m,
        mlib=mlib,
    )
    edge_objs['SEC'] = edges.LinkEdge(
        'SEC.L',
        L_m=Ls_m,
        detune_rad=SEC_detune_rad,
        gouy_rad=SEC_gouy_rad,
        mlib=mlib,
    )
    edge_objs['PRC'] = edges.LinkEdge(
        'PRC.L',
        L_m=Lp_m,
        gouy_rad=PRC_gouy_rad,
        mlib=mlib,
    )

    cSEC = mlib.Mrotation(np.pi/2)
    edge_map = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC,
        "SEC.fr": cSEC,
    }

    edgesDC = deepcopy(edge_map)
    for edge_obj in edge_objs.values():
        edgesDC.update(edge_obj.edgesDC())

    tp_dc = {
        '{:}.{:}.{:}.tp'.format(opt, side, port)
        for opt in ['IX', 'EX', 'IY', 'EY']
        for side in ['fr', 'bk']
        for port in ['o', 'i']
    }
    tp_dc.remove("EX.bk.i.tp")
    tp_dc.remove("EY.bk.i.tp")

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        tp_dc,
        {
            "PRM.bk.i.exc": mlib.LO(np.pi/2),
        },
    )
    pprint(len(resultsDC))

    # set arm power
    Px_W = Vnorm_sq(resultsDC["EX.fr.i.tp"])
    Py_W = Vnorm_sq(resultsDC["EY.fr.i.tp"])
    avg_arm_power = (Px_W + Py_W) / 2
    power_correction = np.sqrt(Parm_W / avg_arm_power)
    for k, v in resultsDC.items():
        resultsDC[k] = v * power_correction
    pprint('Xarm power: {:0.1f} kW'.format(
        Vnorm_sq(resultsDC["EX.fr.i.tp"]) * 1e-3))
    pprint('Yarm power: {:0.1f} kW'.format(
        Vnorm_sq(resultsDC["EY.fr.i.tp"]) * 1e-3))

    edgesAC = deepcopy(edge_map)
    for edge_obj in edge_objs.values():
        edgesAC.update(edge_obj.edgesAC(F_Hz, resultsDC))

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC = compAC.inverse_row(
        {"SEM.bk.o.tp": None},
        {
            "SEM.bk.i.exc",
            "EX.pos.exc",
            "EY.pos.exc",
        },
    )

    LOa = adjoint(mlib.LO(0))
    LOdotArmPhase = (LOa @ mats.H['ArmTrans'])[..., 0, 1]
    BS_factor = 1/np.sqrt(2)
    k_ = 2*np.pi/1064e-9
    d_sense = BS_factor * (2 * k_ * LOdotArmPhase * Parm_W**0.5)
    reflSEC = access.sec.reflSRC[..., 1, 1]

    plant = (resultsAC['EY.pos.exc'] - resultsAC['EX.pos.exc']) / 2
    plant = np.squeeze(LOa @ plant)
    reflIFO = resultsAC["SEM.bk.i.exc"][:, 1, 1]

    fig = plotTF(F_Hz, plant, label='SFLU')
    plotTF(F_Hz, d_sense, *fig.axes, ls='--', label='gwinc')
    fig.axes[0].legend()
    fig.savefig(tpath_join('plant.pdf'))

    fig = plotTF(F_Hz, reflIFO, label='SFLU')
    plotTF(F_Hz, reflSEC, *fig.axes, ls='--', label='gwinc')
    fig.axes[0].legend()
    fig.axes[0].set_ylim(0.5, 2)
    fig.savefig(tpath_join('reflIFO.pdf'))


def plot_graph(tpath_join):
    sflu = sflu_DRFPMI()
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


def plot_graph_FP(tpath_join):
    sflu = sflu_FP()
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
