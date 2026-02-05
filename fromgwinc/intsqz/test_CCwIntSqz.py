"""
"""

import numpy as np
from os import path

from wield.bunch import Bunch
from wield.control.SFLU import SFLU, nx2tikz
from wield.control import SISO
from wield.utilities.mpl import mplfigB
import gwinc

from sflu_components import elements, edges
from sflu_components.lib import (
    MatrixLib,
    adjoint,
    Minv,
)
from .lib import MatsHelper, Vnorm_sq, Vnorm_sqA
from . import optics

from gwinc.struct import Struct
from gwinc import const

from . import FilterCavity
from .common import standardize_params, arm_gouyRT


def sflu_CoupledCav(
        use_ITMRP=True
):
    ifo = elements.optics.GraphElement()

    if use_ITMRP:
        ifo.subgraph_add(
            "ITM", elements.RPMirrorElement(loss_ports=True),
            translation_xy=(25, 0),
            rotation_deg=180,
        )
    else:
        ifo.subgraph_add(
            "ITM", elements.MirrorElement(loss_ports=True),
            translation_xy=(25, 0),
            rotation_deg=180,
        )

    ifo.subgraph_add(
        "ETM", elements.RPMirrorElement(loss_ports=True),
        translation_xy=(55, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "SEM", elements.MirrorElement(loss_ports=True),
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

    return Bunch(
        sflu=sflu,
        use_ITMRP=use_ITMRP,
        loss_ports=dict(
            Arm=[
                "ETM.bk.i",
                "ETM.frL.i",
                "ITM.frL.i",
            ],
            SEC=["SEM.frL.i"],
            FilterCavity=[
                "FC2.frL.i",
            ],
            LossInjection=[
                "Loss_injection",
            ],
            LossReadout=[
                "Loss_readout",
            ],
        ),
        strain_exc={"ETM.pos.exc": 1/np.sqrt(2)}
    )


def CoupledCavity(
    sflu,
    F_Hz,
    ifo,
    params,
    use_SS=True,
):
    ######################################################################
    # Extract parameters
    ######################################################################

    mlib = params.mlib
    lambda_m = ifo.Laser.Wavelength
    Ti = ifo.Optics.ITM.Transmittance
    Te = ifo.Optics.ETM.Transmittance
    Ts = ifo.Optics.SRM.Transmittance

    ARM_gouy_rad = arm_gouyRT(
        ifo.Optics.Curvature.ITM,
        params.Length_m.ARM,
        ifo.Optics.Curvature.ETM,
    ) / 2
    SEC_gouy_rad = ifo.Optics.SRM.get('SRCGouy_rad', 19 * np.pi/180)
    if mlib.nhom > 0:
        ARM_gouy_rad *= params.mode_order
        SEC_gouy_rad *= params.mode_order

    # convert to one-way and shift for RSE convention used here
    SEC_detune_rad = ifo.Optics.SRM.Tunephase / 2 + np.pi/2

    def suscept_m_N(F_Hz):
        """
        Factor of 2 because all of the susceptibility is put on the ETM here
        """
        return 2 * (1 / ((40 * 2*np.pi) * F_Hz)**2)
        # return 2 * sustf.tst_suscept

    ######################################################################
    # Build SFLU model
    ######################################################################

    edge_objs = Struct()

    ####################
    # optics
    ####################

    # TODO, currently a HACK using free space test mass susceptibility
    # should use AAA on the calculated suscept or turn that into an SS model

    # factor of 2 because all the susceptibility is on the ETM in this approximation
    suscept_ss = SISO.zpk([], [0, 0], 2 / ifo.Suspension.Stage[0].Mass).asSS
    suscept_func = lambda F_Hz: suscept_ss.fresponse(f=F_Hz).tf

    edge_objs.ETM = optics.RPMirrorEdge(
        name     = 'ETM',
        Thr      = Te,
        Lhr      = params.Loss.arm_rt / 2,
        # suscept  = suscept_m_N,
        suscept = suscept_func,
        suscept_ss = suscept_ss,
        lambda_m = lambda_m,
        mlib     = mlib,
    )
    if True:
        edge_objs.ITM = optics.RPMirrorEdge(
            name    = 'ITM',
            Thr     = Ti,
            Lhr     = params.Loss.arm_rt / 2,
            mlib    = mlib,
        )
    else:
        edge_objs.ITM = optics.MirrorEdge(
            name    = 'ITM',
            Thr     = Ti,
            Lhr     = params.Loss.arm_rt / 2,
            mlib    = mlib,
        )
    edge_objs.SEM = optics.MirrorEdge(
        name = 'SEM',
        Thr  = Ts,
        Lhr  = params.Loss.SEC_rt,
        mlib = mlib,
    )

    ####################
    # links
    ####################

    def get_MM_fr(MM):
        if params.is_OPD:
            return MM
        else:
            return mlib.Minv(MM)

    edge_objs.ARM = optics.LinkEdge(
        name     = 'ARM.L',
        L_m      = params.Length_m.ARM,
        gouy_rad = ARM_gouy_rad,
        mlib     = mlib,
    )
    edge_objs.L_SEC_to = optics.LinkEdge(
        name       = 'SEC.L.to',
        L_m        = params.Length_m.SEM,
        detune_rad = SEC_detune_rad,
        gouy_rad   = SEC_gouy_rad,
        MM_to      = params.MM.SEC_ARM,
        mlib       = mlib,
    )
    edge_objs.L_SEC_fr = optics.LinkEdge(
        name       = 'SEC.L.fr',
        L_m        = params.Length_m.SEM,
        detune_rad = SEC_detune_rad,
        gouy_rad   = SEC_gouy_rad,
        MM_fr      = get_MM_fr(params.MM.SEC_ARM),
        mlib       = mlib,
    )

    cSEC = mlib.Mrotation(np.pi/2)
    edge_map = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC @ params.MM.SQZ_SEC,
        "SEC.fr": mlib.Minv(params.MM.IFO_OMC) @ cSEC,
    }

    ######################################################################
    # AC calculation (DC not used for coupled cavity approximation)
    ######################################################################

    # set the power at the front of the ETM to be the arm power in the
    # amplitude quadrature
    Earm_rtW = np.sqrt(params.Parm_W) * mlib.LO(np.pi/2)
    resultsDC = {
        "ETM.fr.i.tp": Earm_rtW,
        "ETM.fr.o.tp": -edge_objs.ETM.r @ Earm_rtW,
        "ETM.bk.o.tp": edge_objs.ETM.t @ Earm_rtW,
    }

    inverse_row_Rmap = {
        "SEM.bk.o.tp": None,
    }
    inverse_row_Cset = {
        "SEM.bk.i.exc",
        "ETM.pos.exc",
        "ETM.bk.i",
        "ETM.frL.i",
        "ITM.frL.i",
        "SEM.frL.i",

    }

    if use_SS:
        for edge_obj in edge_objs.values():
            edge_map.update(edge_obj.edgesACSS(F_Hz=F_Hz, resultsDC=resultsDC))
        compAC = sflu.SScomputer(eye=mlib.Id)
        compAC.SScompletion(edge_map)

        resultsAC = compAC.inverse_row_fresponse(
            Rmap=inverse_row_Rmap,
            Cset=inverse_row_Cset,
            F_Hz=F_Hz,
        )

    if (not use_SS):
        for edge_obj in edge_objs.values():
            edge_map.update(edge_obj.edgesAC(F_Hz=F_Hz, resultsDC=resultsDC))

        compAC = sflu.computer(eye=mlib.Id)
        compAC.compute(edge_map=edge_map)
        resultsAC = compAC.inverse_row(
            Rmap=inverse_row_Rmap,
            Cset=inverse_row_Cset,
        )

    return dict(locals())


def test_CoupledCav(fpath_join, tpath_join, plotTF, pprint):
    use_SS = True
    F_Hz = np.geomspace(10, 10e3, 1000)

    #tpath = path.split(__file__)[0]
    #ifo = Struct.from_file(path.join(tpath, '../Aplus/ifo.yaml'))

    budgetApl = gwinc.load_budget('Aplus')
    budget = gwinc.load_budget(fpath_join('Aplus_MM_all_one_HOM' + '.yaml'))
    ifo = budgetApl.ifo
    print(ifo)

    sfluB = sflu_CoupledCav()
    sflu = sfluB.sflu
    sflu.reduce_auto()
    params = standardize_params(ifo)

    mlib = params.mlib
    mats = MatsHelper()
    mats.H['AS'] = mlib.Id

    # injection loss
    L_inj_t = (1 - params.Loss.injection)**0.5
    mats.update_scalar(L_inj_t)
    mats.T['Loss_injection'] = mlib.Id * params.Loss.injection**0.5

    # FIXME: add support for multiple filter cavities
    # FIXME: better deal with no filter cavity
    # filter cavity
    if 'Squeezer' in ifo:
        ret_FC = FilterCavity.FilterCavity(F_Hz, ifo, params, use_SS=use_SS)
        results_FC = ret_FC['resultsAC']
        mats.update_matrix(results_FC["FC1.bk.i.exc"])
        mats.T.update({k: v for k, v in results_FC.items() if k != "FC1.bk.i.exc"})

    # IFO
    ret_IFO = CoupledCavity(
        sflu=sflu,
        F_Hz=F_Hz,
        ifo=ifo,
        params=params,
        use_SS=use_SS,
    )
    results_IFO = ret_IFO['resultsAC']
    mats.update_matrix(results_IFO["SEM.bk.i.exc"])
    mats.T.update({k: v for k, v in results_IFO.items() if k != "SEM.bk.i.exc"})

    # FIXME add output filter cavities

    # readout loss
    L_read_t = (1 - params.Loss.readout)**0.5
    mats.update_scalar(L_read_t)
    mats.T['Loss_readout'] = mlib.diag(params.Loss.readout**0.5)


    ####################
    #extract parameters
    ####################

    lambda_        = ifo.Laser.Wavelength               # Laser Wavelength [m]
    k_             = 2 * np.pi / lambda_
    params         = standardize_params(ifo)
    mlib           = params.mlib
    sqz_type       = params.sqzType
    FC_use         = (params.sqzType == 'Freq Dependent')
    plant_type     = ifo.Optics.get('plant_type', 'DRFPMI')

    SQZ_DB         = params.SQZ_DB
    ASQZ_DB        = params.ANTISQZ_DB
    SQZ_angle_rad  = params.alpha
    HD_angle_rad   = params.LO_angle
    LO_angle_RMS   = params.LO_RMS
    SQZ_angle_RMS  = params.SQZ_RMS


    ############
    # Initial Setup
    ############
    sqzV = 10**(-SQZ_DB/10.)
    asqzV = 10**(ASQZ_DB/10.)

    access = Struct()

    ####################
    # Interferometer Sensing
    ####################

    def prepareLOa(loc):
        if params.follow_fringe:
            LOa = adjoint(mats.L[loc] @ mlib.LO(HD_angle_rad))
            LOa = LOa / (Vnorm_sq(LOa)**0.5).reshape(-1, 1, 1)
        else:
            #HD_angle_rad = np.pi/2
            LOa = adjoint(mlib.LO(HD_angle_rad))
        return LOa
    LOa = prepareLOa('AS')

    delta_LO_rad = 1e-6
    DpLOa = adjoint(mlib.LO(HD_angle_rad + delta_LO_rad))

    # scalar s in (E22)
    d_sense = np.sum([cc * mats.T[exc] for exc, cc in sfluB.strain_exc.items()], axis=0)
    d_sense = np.squeeze(LOa @ d_sense)

    omega = k_ * const.c
    Qnoise = const.hbar * omega / 2
    PSDdisplacement = Qnoise / abs(d_sense)**2  # \hbar\omega/2 * G * L_A^2 in (E23)

    ####################
    # Outputs
    ####################

    ASbudget = {}
    Lambda_IRO = 0

    for _k in mats.T:
        if _k in sfluB.strain_exc.keys():
            continue
        LOSSIFOOUTx = Vnorm_sqA(LOa @ mats.T[_k])
        ASbudget[_k] = LOSSIFOOUTx
        Lambda_IRO = Lambda_IRO + LOSSIFOOUTx

    LOdotAS = (LOa @ mats.H['AS'])
    ASquantumAll = Vnorm_sqA(LOdotAS @ mlib.Mrotation(SQZ_angle_rad) @ mlib.SQZ(sqzV, asqzV))

    def sum_losses(noises, loss_ports):
        loss = np.sum([noises[exc_pt] for exc_pt in loss_ports], axis=0)
        return loss

    axB = mplfigB()
    total = ASquantumAll * PSDdisplacement
    axB.ax0.loglog(F_Hz, (ASquantumAll * PSDdisplacement)**0.5/4000, label='ASport')


    for lpN, lpL  in sfluB.loss_ports.items():
        lossB = PSDdisplacement * sum_losses(ASbudget, lpL)
        total += lossB
        axB.ax0.loglog(
            F_Hz,
            (lossB)**0.5 / 4000,
            label=lpN,
        )

    axB.ax0.loglog(F_Hz, (total)**0.5/4000, label='total')
    aplB = budgetApl.run()
    aplQB = aplB.Quantum
    axB.ax0.loglog(aplQB.freq, aplQB.asd, label = 'ALIGOtotal')
    axB.ax0.legend()
    axB.save(tpath_join('cmp'))

    fig = aplB.Quantum.plot()
    fig.savefig(tpath_join('budget.pdf'))
    return

def plot_graph_CoupledCav(tpath_join):
    sfluB = sflu_CoupledCav()
    sflu = sfluB.sflu
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


def test_build_CoupledCav(tpath_join):
    sfluB = sflu_CoupledCav()
    sflu = sfluB.sflu
    yamlstr = sflu.convert_self2yamlstr()
    with open(tpath_join('CoupledCavINTSQZ.yaml'), 'w') as F:
        F.write(yamlstr)
    sflu = SFLU.SFLU.convert_yamlstr2self(yamlstr)


