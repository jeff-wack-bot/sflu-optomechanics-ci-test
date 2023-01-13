import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
from sflu_components import elements, edges
from sflu_components.lib import MatrixLib, adjoint, Vnorm_sq, Minv
from gwinc.struct import Struct
from gwinc import load_budget
try:
    from gwinc.plant import plant_debug, arm_gouyRT
    gwinc_type = 'superQK'
except ModuleNotFoundError:
    from gwinc.noise.quantum2 import shotrad_debug, arm_gouyRT
    gwinc_type = 'master'
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
        "PRM", optics.LossyBasisMirror(),
        translation_xy=(-25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "SEM", optics.LossyBasisMirror(),
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
        ("IX.bk.i", "BS.bkA.o"): "BSX.L.to",
        ("BS.bkA.i", "IX.bk.o"): "BSX.L.fr",

        ("EY.fr.i", "IY.fr.o"): "YARM.L",
        ("IY.fr.i", "EY.fr.o"): "YARM.L",
        ("IY.bk.i", "BS.frB.o"): "BSY.L.to",
        ("BS.frB.i", "IY.bk.o"): "BSY.L.fr",

        ("PRM.fr.i", "BS.frA.o"): "PRC.L.to",
        ("BS.frA.i", "PRM.fr.o"): "PRC.L.fr",

        ("SEM.fr.i", "BS.bkB.o"): "SEC.L.to",
        ("BS.bkB.i", "SEM.fr.o"): "SEC.L.fr",
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


def sflu_CoupledCav():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "ITM", elements.RPMirrorElement(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "ETM", elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "SEM", optics.BasisMirror(),
        translation_xy=(0, 0),
        rotation_deg=180,
    )

    ifo.edges.update({
        ("ETM.fr.i", "ITM.fr.o"): "ARM.L",
        ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
        ("ITM.bk.i", "SEM.fr.o"): "SEC.L.to",
        ("SEM.fr.i", "ITM.bk.o"): "SEC.L.fr",
    })

    ifo["SEM"].locations.update({
        "bk.i.exc": (15, -10),
        "bk.o.tp": (15, 10),
    })
    ifo["SEM"].edges.update({
        ("bk.i", "bk.i.exc"): "SEC.to",
        ("bk.o.tp", "bk.o"): "SEC.fr",
    })

    ifo["ETM"].locations.update({
        "fr.o.exc": (-3, -10),
    })
    ifo["ETM"].edges.update({
        ("fr.o", "fr.o.exc"): "1",
    })

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


def sflu_CoupledCavNoITM_RP():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "ITM", elements.optics.BasisMirror(),
        translation_xy=(25, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "ETM", elements.RPMirrorElement(),
        translation_xy=(55, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "SEM", optics.BasisMirror(),
        translation_xy=(0, 0),
        rotation_deg=180,
    )

    ifo.edges.update({
        ("ETM.fr.i", "ITM.fr.o"): "ARM.L",
        ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
        ("ITM.bk.i", "SEM.fr.o"): "SEC.L.to",
        ("SEM.fr.i", "ITM.bk.o"): "SEC.L.fr",
    })

    ifo["SEM"].locations.update({
        "bk.i.exc": (15, -10),
        "bk.o.tp": (15, 10),
    })
    ifo["SEM"].edges.update({
        ("bk.i", "bk.i.exc"): "SEC.to",
        ("bk.o.tp", "bk.o"): "SEC.fr",
    })

    ifo["ETM"].locations.update({
        "fr.o.exc": (-3, -10),
        "fr.i.exc": (-3, +10),
    })
    ifo["ETM"].edges.update({
        ("fr.o", "fr.o.exc"): "1",
        ("fr.i", "fr.i.exc"): "1",
    })
    ifo["ITM"].locations.update({
        "bk.i.exc": (7, -7),
    })
    ifo["ITM"].edges.update({
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
    ifo.Optics.ETM.Transmittance = 0
    ifo.Suspension.RPdynamics = 'FreeMass'
    if gwinc_type == 'superQK':
        _, access = plant_debug(F_Hz, ifo)
    elif gwinc_type == 'master':
        ret = shotrad_debug(F_Hz, ifo)
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

    nhom = 2
    mlib = MatrixLib(nhom=nhom)
    mode_order = np.arange(1, nhom + 1)
    # ARM_gouy_rad = -23 * np.pi/180 * mode_order
    ARM_gouy_rad = arm_gouyRT(
        ifo.Optics.Curvature.ITM,
        ifo.Infrastructure.Length,
        ifo.Optics.Curvature.ETM,
    )
    if nhom > 1:
        ARM_gouy_rad *= mode_order
    SEC_gouy_rad = 19 * np.pi/180 * mode_order
    PRC_gouy_rad = 25 * np.pi/180 * mode_order
    SEC_detune_rad = 0
    if gwinc_type == 'superQK':
        Parm_W = access.drfpmi.parm_W
    elif gwinc_type == 'master':
        Parm_W = ret.parm_W

    if nhom > 0:
        MM_SEC_XARM_L = np.linspace(0.001, 0.005, nhom)
        # MM_SEC_XARM_L = np.zeros(nhom)
        MM_SEC_XARM_rad = np.zeros_like(MM_SEC_XARM_L)  # np.linspace(0, 90, nhom) * np.pi/180
        # MM_XARM_YARM_L = MM_SEC_XARM_L
        MM_XARM_YARM_L = np.zeros_like(MM_SEC_XARM_L)
        # MM_XARM_YARM_rad = np.zeros_like(MM_SEC_XARM_rad)
        MM_XARM_YARM_rad = np.ones_like(MM_SEC_XARM_rad) * np.pi
        MM_SEC_PRC_L = np.linspace(0.001, 0.005, nhom)
        MM_SEC_PRC_rad = np.zeros_like(MM_SEC_PRC_L)
    else:
        MM_SEC_XARM_L = 0
        MM_XARM_YARM_L = 0
        MM_SEC_XARM_rad = 0
        MM_XARM_YARM_rad = 0
        MM_SEC_PRC_L = 0
        MM_SEC_PRC_rad = 0

    if nhom == 1:
        ifo.Optics.MM_ARM_SRC = MM_SEC_XARM_L
        ifo.Optics.MM_ARM_SRCphi = MM_SEC_XARM_rad
        ifo.Optics.MM_XARM_YARM = MM_XARM_YARM_L
        ifo.Optics.MM_XARM_YARMphi = MM_XARM_YARM_rad
        ifo.Optics.SRM.SRCGouy_rad = SEC_gouy_rad
        ifo.Optics.PRM.PRCGouy_rad = PRC_gouy_rad


    pprint(ARM_gouy_rad)
    # pprint(access.arm.ret.ARM_gouy_rad)
    # pprint(access.arm.ret.ARM_gouy_rad - ARM_gouy_rad)

    MM_SEC_XARM = mlib.MrotationMM(MM_SEC_XARM_L, MM_SEC_XARM_rad)
    MM_XARM_YARM = mlib.MrotationMM(MM_XARM_YARM_L, MM_XARM_YARM_rad)
    MM_SEC_YARM = MM_SEC_XARM @ MM_XARM_YARM
    pprint(np.all(MM_SEC_XARM == MM_SEC_YARM))
    MM_SEC_PRC = mlib.MrotationMM(MM_SEC_PRC_L, MM_SEC_PRC_rad)

    SEC_detune_rad = np.pi/2 + SEC_detune_rad

    def suscept(F_Hz):
        # return -1 / (M_kg * (2 * np.pi * F_Hz)**2)
        if gwinc_type == 'superQK':
            return access.drfpmi.tst_suscept
        elif gwinc_type == 'master':
            return ret.sustf.tst_suscept

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
        loss_ports=True,
    )
    edge_objs.SEM = edges.MirrorEdge(
        'SEM',
        Thr=Ts,
        mlib=mlib,
        loss_ports=True,
    )
    for link_name in ['XARM', 'YARM']:
        edge_objs[link_name] = edges.LinkEdge(
            link_name + '.L',
            L_m=Larm_m,
            gouy_rad=ARM_gouy_rad,
            mlib=mlib,
        )
    edge_objs['L_BSX_to'] = edges.LinkEdge(
        'BSX.L.to',
        L_m=Lbsx_m,
        MM_to=MM_SEC_XARM,
        mlib=mlib,
    )
    edge_objs['L_BSX_fr'] = edges.LinkEdge(
        'BSX.L.fr',
        L_m=Lbsx_m,
        MM_fr=Minv(MM_SEC_XARM),
        mlib=mlib,
    )
    edge_objs['L_BSY_to'] = edges.LinkEdge(
        'BSY.L.to',
        L_m=Lbsy_m,
        MM_to=MM_SEC_YARM,
        mlib=mlib,
    )
    edge_objs['L_BSY_fr'] = edges.LinkEdge(
        'BSY.L.fr',
        L_m=Lbsy_m,
        MM_fr=Minv(MM_SEC_YARM),
        mlib=mlib,
    )
    edge_objs['L_SEC_to'] = edges.LinkEdge(
        'SEC.L.to',
        L_m=Ls_m,
        detune_rad=SEC_detune_rad,
        gouy_rad=SEC_gouy_rad,
        mlib=mlib,
    )
    edge_objs['L_SEC_fr'] = edges.LinkEdge(
        'SEC.L.fr',
        L_m=Ls_m,
        detune_rad=SEC_detune_rad,
        gouy_rad=SEC_gouy_rad,
        mlib=mlib,
    )
    edge_objs['PRC_to'] = edges.LinkEdge(
        'PRC.L.to',
        L_m=Lp_m,
        gouy_rad=PRC_gouy_rad,
        MM_to=MM_SEC_PRC,
        mlib=mlib,
    )
    edge_objs['PRC_fr'] = edges.LinkEdge(
        'PRC.L.fr',
        L_m=Lp_m,
        gouy_rad=PRC_gouy_rad,
        MM_fr=Minv(MM_SEC_PRC),
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

    # set arm power
    Px_W = Vnorm_sq(resultsDC["EX.fr.i.tp"])
    Py_W = Vnorm_sq(resultsDC["EY.fr.i.tp"])
    avg_arm_power = (Px_W + Py_W) / 2
    power_correction = np.sqrt(Parm_W / avg_arm_power)
    for k, v in resultsDC.items():
        resultsDC[k] = v * power_correction
    approx_rp = True
    zero_hom = False
    zero_p = False
    if approx_rp:
        resultsDC["IX.bk.i.tp"] *= 0
        resultsDC["IX.bk.o.tp"] *= 0
        resultsDC["IY.bk.i.tp"] *= 0
        resultsDC["IY.bk.o.tp"] *= 0
    if zero_hom:
        tp_zero = [
            '{:}.{:}.i.tp'.format(opt, side)
            for opt in ['IX', 'EX', 'IY', 'EY']
            for side in ['fr', 'bk']
        ]
        tp_zero.remove("EX.bk.i.tp")
        tp_zero.remove("EY.bk.i.tp")
        for tp in tp_zero:
            resultsDC[tp][2:, 0] = 0
            if zero_p:
                resultsDC[tp][1, 0] = 0
    pprint('Xarm power: {:0.1f} kW'.format(
        Vnorm_sq(resultsDC["EX.fr.i.tp"]) * 1e-3))
    pprint('Yarm power: {:0.1f} kW'.format(
        Vnorm_sq(resultsDC["EY.fr.i.tp"]) * 1e-3))
    def fundamental_deg(tp):
        field = resultsDC[tp]
        ang_rad = np.arctan2(field[1, 0], field[0, 0])
        return ang_rad * 180/np.pi
    pprint(fundamental_deg("EX.fr.i.tp"))
    pprint(fundamental_deg("EY.fr.i.tp"))

    edgesAC = deepcopy(edge_map)
    for nn, edge_obj in edge_objs.items():
        try:
            edgesAC.update(edge_obj.edgesAC(F_Hz, resultsDC))
        except ValueError as err:
            pprint(nn)
            traceback.print_exc()
            raise err

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
    if nhom < 2:
        calc_gwinc = True
    else:
        calc_gwinc = False
    # calc_gwinc = False
    if calc_gwinc:
        if gwinc_type == 'superQK':
            mats, access = plant_debug(F_Hz, ifo)
            LOdotArmPhase = (LOa @ mats.H['ArmTrans'])[..., 0, 1]
            BS_factor = 1/np.sqrt(2)
            reflSEC = access.sec.reflSRC[..., 1, 1]
        elif gwinc_type == 'master':
            ret = shotrad_debug(F_Hz, ifo)
            LOdotArmPhase = (LOa @ ret.mats.H['ArmTrans'])[..., 0, 1]
            BS_factor = 1/np.sqrt(2)
            reflSEC = ret.access.sec.reflSRC[..., 1, 1]
        k_ = 2*np.pi/1064e-9
        d_sense = BS_factor * (2 * k_ * LOdotArmPhase * Parm_W**0.5)

    plant = (resultsAC['EY.pos.exc'] - resultsAC['EX.pos.exc']) / 2
    plant = np.squeeze(LOa @ plant)
    reflIFO = resultsAC["SEM.bk.i.exc"][:, 1, 1]

    fig = plotTF(F_Hz, plant, label='SFLU')
    if calc_gwinc:
        plotTF(F_Hz, d_sense, *fig.axes, ls='--', label='gwinc')
    fig.axes[0].legend()
    fig.savefig(tpath_join('plant.pdf'))

    fig = plotTF(F_Hz, reflIFO, label='SFLU')
    if calc_gwinc:
        plotTF(F_Hz, reflSEC, *fig.axes, ls='--', label='gwinc')
    fig.axes[0].legend()
    # fig.axes[0].set_ylim(0.1, 10)
    fig.savefig(tpath_join('reflIFO.pdf'))

    if calc_gwinc:
        fig1 = plotTF(F_Hz, plant / d_sense)
        fig2 = plotTF(F_Hz, reflIFO / reflSEC)
        for fig in [fig1, fig2]:
            for ax in fig.axes:
                ax.autoscale(enable=True, axis='y')
            fig.axes[0].set_yscale('linear')
        fig1.savefig(tpath_join('plant_err.pdf'))
        fig2.savefig(tpath_join('reflIFO_err.pdf'))


def test_CoupledCav(tpath_join, plotTF, pprint):
    sflu = sflu_CoupledCav()
    sflu.reduce_auto()

    ifo = load_budget('Aplus').ifo
    F_Hz = np.logspace(0, 4, 2000)
    ifo.Optics.Loss = 0
    ifo.Optics.BSLoss = 0
    ifo.Suspension.RPdynamics = 'None'
    if gwinc_type == 'superQK':
        _, access = plant_debug(F_Hz, ifo)
    elif gwinc_type == 'master':
        ret = shotrad_debug(F_Hz, ifo)
    Ti = ifo.Optics.ITM.Transmittance  # 0.014
    Te = ifo.Optics.ETM.Transmittance  # 10e-6
    Ts = ifo.Optics.SRM.Transmittance  # 0.325
    Larm_m = ifo.Infrastructure.Length  # 4e3
    Lsec_m = ifo.Optics.SRM.CavityLength  # 56

    nhom = 1
    mlib = MatrixLib(nhom=nhom)
    mode_order = np.arange(1, nhom + 1)
    ARM_gouy_rad = arm_gouyRT(
        ifo.Optics.Curvature.ITM,
        ifo.Infrastructure.Length,
        ifo.Optics.Curvature.ETM,
    )
    if nhom > 1:
        ARM_gouy_rad *= mode_order
    SEC_gouy_rad = 19 * np.pi/180 * mode_order
    SEC_detune_rad = 0
    if gwinc_type == 'superQK':
        Parm_W = access.drfpmi.parm_W
    elif gwinc_type == 'master':
        Parm_W = ret.parm_W

    if nhom > 0:
        MM_SEC_ARM_L = np.linspace(0.001, 0.005, nhom)
        MM_SEC_ARM_rad = np.zeros_like(MM_SEC_ARM_L)  # np.linspace(0, 90, nhom) * np.pi/180
    else:
        MM_SEC_ARM_L = 0
        MM_SEC_ARM_rad = 0

    if nhom == 1:
        ifo.Optics.MM_ARM_SRC = MM_SEC_ARM_L
        ifo.Optics.MM_ARM_SRCphi = MM_SEC_ARM_rad
        ifo.Optics.SRM.SRCGouy_rad = SEC_gouy_rad

    pprint(ARM_gouy_rad)
    MM_SEC_ARM = mlib.MrotationMM(MM_SEC_ARM_L, MM_SEC_ARM_rad)

    SEC_detune_rad = np.pi/2 + SEC_detune_rad

    def suscept(F_Hz):
        # return -1 / (M_kg * (2 * np.pi * F_Hz)**2)
        if gwinc_type == 'superQK':
            return access.drfpmi.tst_suscept
        elif gwinc_type == 'master':
            return ret.sustf.tst_suscept

    edge_objs = Struct()
    edge_objs.ETM = edges.RPMirrorEdge(
        'ETM',
        Thr=Te,
        suscept=suscept,
        mlib=mlib,
    )
    edge_objs.ITM = edges.RPMirrorEdge(
        'ITM',
        Thr=Ti,
        suscept=suscept,
        mlib=mlib,
    )
    edge_objs.SEM = edges.MirrorEdge(
        'SEM',
        Thr=Ts,
        mlib=mlib,
    )
    edge_objs.ARM = edges.LinkEdge(
        'ARM.L',
        L_m=Larm_m,
        gouy_rad=ARM_gouy_rad,
        mlib=mlib,
    )
    edge_objs.SEC_to = edges.LinkEdge(
        'SEC.L.to',
        L_m=Lsec_m,
        detune_rad=SEC_detune_rad,
        gouy_rad=SEC_gouy_rad,
        MM_to=Minv(MM_SEC_ARM),
        mlib=mlib,
    )
    edge_objs.SEC_fr = edges.LinkEdge(
        'SEC.L.fr',
        L_m=Lsec_m,
        detune_rad=SEC_detune_rad,
        gouy_rad=SEC_gouy_rad,
        MM_fr=MM_SEC_ARM,
        mlib=mlib,
    )

    cSEC = mlib.Mrotation(np.pi/2)
    edgesDC = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC,
        "SEC.fr": cSEC,
    }
    for edge_obj in edge_objs.values():
        edgesDC.update(edge_obj.edgesDC())

    tp_dc = {
        "{:}.{:}.{:}.tp".format(opt, side, port)
        for opt in ['ITM', 'ETM']
        for side in ['fr', 'bk']
        for port in ['o', 'i']
    }
    tp_dc.remove('ETM.bk.i.tp')

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        tp_dc,
        {
            "SEM.bk.i.exc": mlib.LO(np.pi/2),
        },
    )

    dc_power = Vnorm_sq(resultsDC["ETM.fr.i.tp"])
    power_correction = np.sqrt(Parm_W / dc_power)
    for k, v in resultsDC.items():
        resultsDC[k] = v * power_correction
    pprint('Arm target {:0.1f} kW'.format(Parm_W * 1e-3))
    pprint('Arm power {:0.1f} kW'.format(
        Vnorm_sq(resultsDC["ETM.fr.i.tp"]) * 1e-3))
    pprint('SEC power {:0.1f} kW'.format(
        Vnorm_sq(resultsDC["ITM.bk.i.tp"]) * 1e-3))

    approx_rp = True
    if approx_rp:
        resultsDC['ITM.bk.i.tp'] *= 0
        resultsDC['ITM.bk.o.tp'] *= 0
    # resultsDC['ITM.fr.i.tp'] *= 0
    # resultsDC['ITM.fr.o.tp'] *= 0

    edgesAC = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC,
        "SEC.fr": cSEC,
    }
    for edge_obj in edge_objs.values():
        edgesAC.update(edge_obj.edgesAC(F_Hz, resultsDC))

    # inds = np.array([0, 2, 3])
    # for k, v in edgesAC.items():
    #     if '.px' in k:
    #         edgesAC[k][inds] = 0

    pprint(edgesAC["ETM.fr.px"])
    pprint(edgesAC["ETM.bk.px"])
    pprint(edgesAC["ITM.fr.px"])
    pprint(edgesAC["ITM.bk.px"])

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC = compAC.inverse_row(
        {"SEM.bk.o.tp": None},
        {
            "SEM.bk.i.exc",
            "ETM.pos.exc",
            "ETM.fr.o.exc",
        },
    )

    LOa = adjoint(mlib.LO(0))


    def field_to_plant(ArmTrans, BS_factor):
        LOdotArmPhase = (LOa @ ArmTrans)[..., 0, 1]
        k_ = 2*np.pi/1064e-9
        d_sense = BS_factor * (2 * k_ * LOdotArmPhase * Parm_W**0.5)
        return d_sense

    if nhom < 2:
        calc_gwinc = True
    else:
        calc_gwinc = False
    # calc_gwinc = False
    if calc_gwinc:
        if gwinc_type == 'superQK':
            mats, access = plant_debug(F_Hz, ifo)
            # LOdotArmPhase = (LOa @ mats.H['ArmTrans'])[..., 0, 1]
            ArmTrans = mats.H['ArmTrans']
            BS_factor = 1/np.sqrt(2)
            reflSEC = access.sec.reflSRC[..., 1, 1]
        elif gwinc_type == 'master':
            ret = shotrad_debug(F_Hz, ifo)
            # LOdotArmPhase = (LOa @ ret.mats.H['ArmTrans'])[..., 0, 1]
            ArmTrans = ret.mats.H['ArmTrans']
            BS_factor = 1/np.sqrt(2)
            reflSEC = ret.access.sec.reflSRC[..., 1, 1]
            # pprint(ret.access.MM_ARM_SRC == MM_SEC_ARM)
        d_sense = field_to_plant(ArmTrans, BS_factor)

    plant = resultsAC['ETM.pos.exc'] / np.sqrt(2)
    plant = np.squeeze(LOa @ plant)
    plant_field = field_to_plant(resultsAC["ETM.fr.o.exc"], 1/np.sqrt(2))
    reflIFO = resultsAC["SEM.bk.i.exc"][:, 1, 1]

    fig = plotTF(F_Hz, plant, label='SFLU')
    plotTF(F_Hz, -plant_field, *fig.axes, ls='--', label='SFLU field')
    if calc_gwinc:
        plotTF(F_Hz, d_sense, *fig.axes, ls='-.', c='xkcd:red', label='gwinc')
    fig.axes[0].legend()
    fig.savefig(tpath_join('plant.pdf'))

    fig = plotTF(F_Hz, reflIFO, label='SFLU')
    if calc_gwinc:
        plotTF(F_Hz, reflSEC, *fig.axes, ls='--', label='gwinc')
    fig.axes[0].legend()
    # fig.axes[0].set_ylim(0.1, 10)
    fig.savefig(tpath_join('reflIFO.pdf'))

    if calc_gwinc:
        fig1 = plotTF(F_Hz, plant / d_sense)
        fig2 = plotTF(F_Hz, reflIFO / reflSEC)
        for fig in [fig1, fig2]:
            for ax in fig.axes:
                ax.autoscale(enable=True, axis='y')
            fig.axes[0].set_yscale('linear')
        fig1.savefig(tpath_join('plant_err.pdf'))
        fig2.savefig(tpath_join('reflIFO_err.pdf'))


def test_CoupledCavNoITM_RP(tpath_join, plotTF, pprint):
    sflu = sflu_CoupledCavNoITM_RP()
    sflu.reduce_auto()

    ifo = load_budget('Aplus').ifo
    F_Hz = np.logspace(-2, 4, 2000)
    ifo.Optics.Loss = 0
    ifo.Optics.BSLoss = 0
    ifo.Optics.ETM.Transmittance = 0
    ifo.Suspension.RPdynamics = 'None'
    if gwinc_type == 'superQK':
        _, access = plant_debug(F_Hz, ifo)
    elif gwinc_type == 'master':
        ret = shotrad_debug(F_Hz, ifo)
    Ti = ifo.Optics.ITM.Transmittance  # 0.014
    Te = ifo.Optics.ETM.Transmittance  # 10e-6
    Ts = ifo.Optics.SRM.Transmittance  # 0.325
    Larm_m = ifo.Infrastructure.Length  # 4e3
    Lsec_m = ifo.Optics.SRM.CavityLength  # 56

    nhom = 1
    mlib = MatrixLib(nhom=nhom)
    mode_order = np.arange(1, nhom + 1)
    ARM_gouy_rad = arm_gouyRT(
        ifo.Optics.Curvature.ITM,
        ifo.Infrastructure.Length,
        ifo.Optics.Curvature.ETM,
    )
    if nhom > 1:
        ARM_gouy_rad *= mode_order
    SEC_gouy_rad = 19 * np.pi/180 * mode_order
    SEC_detune_rad = 0
    if gwinc_type == 'superQK':
        Parm_W = access.drfpmi.parm_W
    elif gwinc_type == 'master':
        Parm_W = ret.parm_W

    if nhom > 0:
        MM_SEC_ARM_L = np.linspace(0.001, 0.005, nhom)
        MM_SEC_ARM_rad = np.zeros_like(MM_SEC_ARM_L)  # np.linspace(0, 90, nhom) * np.pi/180
    else:
        MM_SEC_ARM_L = 0
        MM_SEC_ARM_rad = 0

    if nhom == 1:
        ifo.Optics.MM_ARM_SRC = MM_SEC_ARM_L
        ifo.Optics.MM_ARM_SRCphi = MM_SEC_ARM_rad
        ifo.Optics.SRM.SRCGouy_rad = SEC_gouy_rad

    pprint(ARM_gouy_rad)
    MM_SEC_ARM = mlib.MrotationMM(MM_SEC_ARM_L, MM_SEC_ARM_rad)

    SEC_detune_rad = np.pi/2 + SEC_detune_rad

    def suscept(F_Hz):
        # return -1 / (M_kg * (2 * np.pi * F_Hz)**2)
        if gwinc_type == 'superQK':
            return access.drfpmi.tst_suscept
        elif gwinc_type == 'master':
            return 2 * ret.sustf.tst_suscept

    edge_objs = Struct()
    edge_objs.ETM = edges.RPMirrorEdge(
        'ETM',
        Thr=Te,
        suscept=suscept,
        mlib=mlib,
    )
    edge_objs.ITM = edges.MirrorEdge(
        'ITM',
        Thr=Ti,
        mlib=mlib,
    )
    edge_objs.SEM = edges.MirrorEdge(
        'SEM',
        Thr=Ts,
        mlib=mlib,
    )
    edge_objs.ARM = edges.LinkEdge(
        'ARM.L',
        L_m=Larm_m,
        gouy_rad=ARM_gouy_rad,
        mlib=mlib,
    )
    edge_objs.SEC_to = edges.LinkEdge(
        'SEC.L.to',
        L_m=Lsec_m,
        detune_rad=SEC_detune_rad,
        gouy_rad=SEC_gouy_rad,
        MM_to=Minv(MM_SEC_ARM),
        mlib=mlib,
    )
    edge_objs.SEC_fr = edges.LinkEdge(
        'SEC.L.fr',
        L_m=Lsec_m,
        detune_rad=SEC_detune_rad,
        gouy_rad=SEC_gouy_rad,
        MM_fr=MM_SEC_ARM,
        mlib=mlib,
    )

    cSEC = mlib.Mrotation(np.pi/2)
    edgesDC = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC,
        "SEC.fr": cSEC,
    }
    for edge_obj in edge_objs.values():
        edgesDC.update(edge_obj.edgesDC())

    # tp_dc = {
    #     "{:}.{:}.{:}.tp".format(opt, side, port)
    #     for opt in ['ITM', 'ETM']
    #     for side in ['fr', 'bk']
    #     for port in ['o', 'i']
    # }
    # tp_dc.remove('ETM.bk.i.tp')
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
            # "SEM.bk.i.exc": mlib.LO(np.pi/2),
            # "ETM.fr.i.exc": np.sqrt(Parm_W) * mlib.LO(np.pi/2),
            "ITM.bk.i.exc": -mlib.LO(np.pi/2),
            # "ITM.bk.i.exc": np.array([1, 0, 0, 0]).reshape(-1, 1),
        },
    )

    dc_power = Vnorm_sq(resultsDC["ETM.fr.i.tp"])
    power_correction = np.sqrt(Parm_W / dc_power)
    for k, v in resultsDC.items():
        resultsDC[k] = v * power_correction
    pprint('Arm target {:0.1f} kW'.format(Parm_W * 1e-3))
    pprint('Arm power {:0.1f} kW'.format(
        Vnorm_sq(resultsDC["ETM.fr.i.tp"]) * 1e-3))

    edgesAC = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC,
        "SEC.fr": cSEC,
    }
    for edge_obj in edge_objs.values():
        edgesAC.update(edge_obj.edgesAC(F_Hz, resultsDC))

    # inds = np.array([0, 2, 3])
    # for k, v in edgesAC.items():
    #     if '.px' in k:
    #         edgesAC[k][inds] = 0

    pprint(edgesAC["ETM.fr.px"])
    pprint(edgesAC["ETM.bk.px"])

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC = compAC.inverse_row(
        {"SEM.bk.o.tp": None},
        {
            "SEM.bk.i.exc",
            "ETM.pos.exc",
            "ETM.fr.o.exc",
        },
    )

    LOa = adjoint(mlib.LO(0))


    def field_to_plant(ArmTrans, BS_factor):
        LOdotArmPhase = (LOa @ ArmTrans)[..., 0, 1]
        k_ = 2*np.pi/1064e-9
        d_sense = BS_factor * (2 * k_ * LOdotArmPhase * Parm_W**0.5)
        return d_sense

    if nhom < 2:
        calc_gwinc = True
    else:
        calc_gwinc = False
    # calc_gwinc = False
    if calc_gwinc:
        if gwinc_type == 'superQK':
            mats, access = plant_debug(F_Hz, ifo)
            # LOdotArmPhase = (LOa @ mats.H['ArmTrans'])[..., 0, 1]
            ArmTrans = mats.H['ArmTrans']
            BS_factor = 1/np.sqrt(2)
            reflSEC = access.sec.reflSRC[..., 1, 1]
        elif gwinc_type == 'master':
            ret = shotrad_debug(F_Hz, ifo)
            # LOdotArmPhase = (LOa @ ret.mats.H['ArmTrans'])[..., 0, 1]
            ArmTrans = ret.mats.H['ArmTrans']
            BS_factor = 1/np.sqrt(2)
            reflSEC = ret.access.sec.reflSRC[..., 1, 1]
            # pprint(ret.access.MM_ARM_SRC == MM_SEC_ARM)
        d_sense = field_to_plant(ArmTrans, BS_factor)

    plant = resultsAC['ETM.pos.exc'] / np.sqrt(2)
    plant = np.squeeze(LOa @ plant)
    reflIFO = resultsAC["SEM.bk.i.exc"][:, 1, 1]
    # plant_field = field_to_plant(resultsAC["ETM.fr.o.exc"], 1/np.sqrt(2))

    ETM = edge_objs.ETM
    px = 4 * np.pi / ETM.lambda_m * ETM.r * ETM.overlap @ mlib.Mrotation(np.pi/2)
    plant_field = resultsAC["ETM.fr.o.exc"] @ px @ resultsDC['ETM.fr.i.tp']
    plant_field = np.squeeze(LOa @ plant_field)

    fig = plotTF(F_Hz, plant, label='SFLU')
    plotTF(F_Hz, -plant_field, *fig.axes, ls='--', label='SFLU field')
    if calc_gwinc:
        plotTF(F_Hz, d_sense, *fig.axes, ls='-.', c='xkcd:red', label='gwinc')
    fig.axes[0].legend()
    fig.savefig(tpath_join('plant.pdf'))

    fig = plotTF(F_Hz, reflIFO, label='SFLU')
    if calc_gwinc:
        plotTF(F_Hz, reflSEC, *fig.axes, ls='--', label='gwinc')
    fig.axes[0].legend()
    # fig.axes[0].set_ylim(0.1, 10)
    fig.savefig(tpath_join('reflIFO.pdf'))

    if calc_gwinc:
        fig1 = plotTF(F_Hz, plant / d_sense)
        fig2 = plotTF(F_Hz, reflIFO / reflSEC)
        for fig in [fig1, fig2]:
            for ax in fig.axes:
                ax.autoscale(enable=True, axis='y')
            fig.axes[0].set_yscale('linear')
        fig1.savefig(tpath_join('plant_err.pdf'))
        fig2.savefig(tpath_join('reflIFO_err.pdf'))


def plot_graph_DRFPMI(tpath_join):
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


def plot_graph_CoupledCav(tpath_join):
    sflu = sflu_CoupledCav()
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


def plot_graph_CoupledCavNoITM_RP(tpath_join):
    sflu = sflu_CoupledCavNoITM_RP()
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


def test_build_DRFPMI(tpath_join):
    sflu = sflu_DRFPMI()
    yamlstr = sflu.convert_self2yamlstr()
    with open(tpath_join('DRFPMI.yaml'), 'w') as F:
        F.write(yamlstr)
    sflu = SFLU.SFLU.convert_yamlstr2self(yamlstr)
