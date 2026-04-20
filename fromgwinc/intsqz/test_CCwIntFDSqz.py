"""
Coupled cavity with internal frequency-dependent squeezing.

Extends the internal squeezing model by adding a detuned traveling-wave
filter cavity between the squeezing element and the IFO.
"""

#

import copy
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

from .common import standardize_params, arm_gouyRT

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

def sflu_CCwIntFDSqz(use_ITMRP=True):
    """Build SFLU graph for coupled cavity with internal FD squeezing.

    The internal filter cavity is a traveling-wave cavity
    modeled as a beamsplitter (coupling mirror) with a self-loop.
    """
    ifo = elements.optics.GraphElement()

    if use_ITMRP:
        ifo.subgraph_add(
            "ITM", elements.RPMirrorElement(loss_ports=True),
            translation_xy=(45, 0),
            rotation_deg=180,
        )
    else:
        ifo.subgraph_add(
            "ITM", elements.MirrorElement(loss_ports=True),
            translation_xy=(40, 0),
            rotation_deg=180,
        )

    ifo.subgraph_add(
        "ETM", elements.RPMirrorElement(loss_ports=True),
        translation_xy=(80, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "SEM", elements.MirrorElement(loss_ports=True),
        translation_xy=(-40, 0),
        rotation_deg=180,
    )

    # internal filter cavity — travelling-wave cavity
    # modeled as a beamsplitter (coupling mirror) with direct
    # self-loop edges on the fr ports (frA.o→frB.i, frB.o→frA.i).
    # One travelling-wave mode uses fr→bk transmission, the other
    # uses bk→fr, so the two modes do not interact.
    # Rotated 135° so A-ports face left/right (frA→right toward ITM,
    # bkA→left toward SEM) and the BS plate appears at 45°.
    # bkA face loss models internal squeezing beamsplitter losses.
    ifo.subgraph_add(
        "IFCBS", elements.BeamSplitterElement(loss_bkA_ports=True),
        translation_xy=(0, -15),
        rotation_deg=135,
    )

    ifo.edges.update({
        # arm cavity
        ("ETM.fr.i", "ITM.fr.o"): "ARM.L",
        ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
        # signal extraction cavity (ITM to IFCBS B-port)
        ("ITM.bk.i", "IFCBS.frB.o"): "SEC.L.to",
        ("IFCBS.frB.i", "ITM.bk.o"): "SEC.L.fr",
        # traveling wave: self-loop on fr ports
        # mode 1: frA.o → IFC.L → frB.i
        ("IFCBS.bkB.i", "IFCBS.bkA.o"): "IFC.L",
        # mode 2: frB.o → IFC.L → frA.i
        ("IFCBS.bkA.i", "IFCBS.bkB.o"): "IFC.L",
        # SQZ edges (IFCBS bkA to SEM)
        ("SEM.fr.i", "IFCBS.frA.o"): "INTSQZ.armfr",
        ("IFCBS.frA.i", "SEM.fr.o"): "INTSQZ.armto",
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

    # Bend the IFC.L self-loop edges so they curve outward,
    # visually suggesting the travelling-wave cavity loop below the BS.
    if sflu.G is not None:
        for u, v, d in sflu.G.edges(data=True):
            if d.get('label_default', '') == '$IFC.L$':
                # frA.o→frB.i curves one way, frB.o→frA.i the other
                if 'frA' in u:
                    d['bend'] = -60
                elif 'frB' in u:
                    d['bend'] = 60

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
            INTSQZ=[
                #"IFCBS.frAL.i",
                "IFCBS.bkAL.i",
                #"IFCBS.frAL2.i",
                "IFCBS.bkAL2.i",
            ],
            LossInjection=[
                "Loss_injection",
            ],
            LossReadout=[
                "Loss_readout",
            ],
        ),
        strain_exc={"ETM.pos.exc": 1/np.sqrt(2)},
    )

def CoupledCavityIntFC(
    sflu,
    F_Hz,
    ifo,
    params,
    use_SS=True,
):
    """Build edge objects and compute transfer functions for the coupled
    cavity with an internal travelling-wave filter cavity.

    The filter cavity is modeled as a beamsplitter (IFCBS) with direct
    self-loop edges on the fr ports.  The bk ports carry the main
    interferometer beam.
    """
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

    # internal filter cavity parameters
    ifc = ifo.intSqueezer.FilterCavity
    IFC_Ti = ifc.Ti
    IFC_L_m = ifc.L
    IFC_loss_rt = ifc.Lrt
    IFC_gouy_rad = ifc.get('Gouy_rad', 33 * np.pi/180)
    if mlib.nhom > 0:
        IFC_gouy_rad *= params.mode_order

    # For a travelling-wave cavity the link is traversed once per
    # round-trip.  Double the length, detuning, and Gouy phase relative
    # to the FP one-way values so the cavity FSR and response match.
    IFC_L_rt = IFC_L_m
    FSR_Hz = const.c / IFC_L_rt
    IFC_detune_Hz = ifc.fdetune
    IFC_detune_m = -IFC_detune_Hz / FSR_Hz * lambda_m / 2
    IFC_detune_rad = -2 * np.pi / lambda_m * IFC_detune_m
    IFC_gouy_rad_rt = 2 * IFC_gouy_rad

    ######################################################################
    # susceptibility (same as CoupledCavity)
    ######################################################################

    suscept_ss = SISO.zpk([], [0, 0], 2 / ifo.Suspension.Stage[0].Mass).asSS
    suscept_func = lambda F_Hz: suscept_ss.fresponse(f=F_Hz).tf

    ######################################################################
    # Build edge objects
    ######################################################################
    edge_objs = Struct()

    # mirrors
    edge_objs.ETM = optics.RPMirrorEdge(
        name     = 'ETM',
        Thr      = Te,
        Lhr      = params.Loss.arm_rt / 2,
        suscept  = suscept_func,
        suscept_ss = suscept_ss,
        lambda_m = lambda_m,
        mlib     = mlib,
    )
    edge_objs.ITM = optics.RPMirrorEdge(
        name     = 'ITM',
        Thr      = Ti,
        Lhr      = params.Loss.arm_rt / 2,
        mlib     = mlib,
    )
    edge_objs.SEM = optics.MirrorEdge(
        name = 'SEM',
        Thr  = Ts,
        Lhr  = params.Loss.SEC_rt,
        mlib = mlib,
    )

    # travelling-wave filter cavity coupling mirror (beamsplitter)
    # bkA face loss absorbs the former INTSQZL mirror losses
    edge_objs.IFCBS = optics.BSEdge(
        name = 'IFCBS',
        Thr  = IFC_Ti,
        Lhr_bkA  = IFC_loss_rt,
        Lhr = ifo.Optics.INTSQ_loss + params.Loss.SEC_rt/2,
        mlib = mlib,
    )

    # links
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

    # travelling-wave filter cavity self-loop link (full round-trip)
    edge_objs.L_IFC = optics.LinkEdge(
        name       = 'IFC.L',
        L_m        = IFC_L_rt,
        detune_rad = IFC_detune_rad,
        gouy_rad   = IFC_gouy_rad_rt,
        mlib       = mlib,
    )

    print(IFC_detune_rad)
    # internal squeezing edges
    MM_INTSQZ = mlib.MrotationMM(ifo.Optics.MM_INTSQZ, 0 / 180 * np.pi)
    edge_objs.INTSQZ_armto = optics.SQZEdge(
        name       = 'INTSQZ.armto',
        sqzDB      = -ifo.intSqueezer.AmplitudedB,
        sqzANGdeg  = -0 - 90,
        MM_to      = MM_INTSQZ,
        MM_fr      = mlib.Minv(MM_INTSQZ),
        mlib       = mlib,
    )
    edge_objs.INTSQZ_armfr = optics.SQZEdge(
        name       = 'INTSQZ.armfr',
        sqzDB      = ifo.intSqueezer.AmplitudedB,
        sqzANGdeg  = -0 - 90,
        MM_to      = MM_INTSQZ,
        MM_fr      = mlib.Minv(MM_INTSQZ),
        mlib       = mlib,
    )

    ######################################################################
    # Edge map
    ######################################################################

    cSEC = mlib.Mrotation(np.pi/2)
    edge_map = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "SEC.to": cSEC @ params.MM.SQZ_SEC,
        "SEC.fr": mlib.Minv(params.MM.IFO_OMC) @ cSEC,
    }

    ######################################################################
    # AC calculation
    ######################################################################

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
        #"IFCBS.frAL.i",
        #"IFCBS.frAL2.i",
        "IFCBS.bkAL.i",
        "IFCBS.bkAL2.i",
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



def _compute_intFDsqz_budget(sfluB, F_Hz, ifo, use_SS=True):
    """Compute quantum noise budget for the IntFDSqz topology.

    Returns (total, ASport, LB, d_sense).
    """
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

    # IFO (includes internal filter cavity in the SFLU graph)
    ret_IFO = CoupledCavityIntFC(
        sflu=sflu,
        F_Hz=F_Hz,
        ifo=ifo,
        params=params,
        use_SS=use_SS,
    )
    results_IFO = ret_IFO['resultsAC']
    mats.update_matrix(results_IFO["SEM.bk.i.exc"])
    mats.T.update({k: v for k, v in results_IFO.items() if k != "SEM.bk.i.exc"})

    # readout loss
    L_read_t = (1 - params.Loss.readout)**0.5
    mats.update_scalar(L_read_t)
    mats.T['Loss_readout'] = mlib.diag(params.Loss.readout**0.5)

    # extract parameters
    lambda_        = ifo.Laser.Wavelength
    k_             = 2 * np.pi / lambda_
    params         = standardize_params(ifo)
    mlib           = params.mlib

    SQZ_DB         = params.SQZ_DB
    ASQZ_DB        = params.ANTISQZ_DB
    SQZ_angle_rad  = params.alpha
    HD_angle_rad   = params.LO_angle

    sqzV = 10**(-SQZ_DB/10.)
    asqzV = 10**(ASQZ_DB/10.)

    LOa = adjoint(mlib.LO(HD_angle_rad))

    d_sense = np.sum([cc * mats.T[exc] for exc, cc in sfluB.strain_exc.items()], axis=0)
    d_sense = np.squeeze(LOa @ d_sense)

    omega = k_ * const.c
    Qnoise = const.hbar * omega / 2
    PSDdisplacement = Qnoise / abs(d_sense)**2

    ASbudget = {}
    for _k in mats.T:
        if _k in sfluB.strain_exc.keys():
            continue
        ASbudget[_k] = Vnorm_sqA(LOa @ mats.T[_k])

    LOdotAS = (LOa @ mats.H['AS'])
    ASquantumAll = Vnorm_sqA(LOdotAS @ mlib.Mrotation(SQZ_angle_rad) @ mlib.SQZ(sqzV, asqzV))

    def sum_losses(noises, loss_ports):
        return np.sum([noises[exc_pt] for exc_pt in loss_ports], axis=0)

    total = ASquantumAll * PSDdisplacement
    ASport = total.copy()

    LB = {}
    for lpN, lpL in sfluB.loss_ports.items():
        lossB = PSDdisplacement * sum_losses(ASbudget, lpL)
        LB[lpN] = lossB
        total += lossB

    return total, ASport, LB, d_sense


def intFDsqzQuantum(ifo, freq, use_SS=True):
    """Compute quantum noise budget for internal FD squeezing.

    Thin wrapper around _compute_intFDsqz_budget that builds a fresh
    SFLU graph each time (reduce_auto mutates the graph).

    Parameters
    ----------
    ifo : Struct
        Interferometer parameters (modified in place before calling).
    freq : array
        Frequency vector [Hz].
    use_SS : bool
        Use state-space computation.

    Returns
    -------
    total : array
        Total quantum noise PSD [m^2/Hz].
    LB : dict
        Loss budget breakdown by port.
    """
    sfluB = sflu_CCwIntFDSqz()
    total, ASport, LB, d_sense = _compute_intFDsqz_budget(sfluB, freq, ifo, use_SS)
    LB['ASport'] = ASport
    return total, LB, d_sense


def test_CCwIntFDSqz(fpath_join, tpath_join, plotTF, pprint):
    """Test coupled cavity with internal frequency-dependent squeezing.

    The internal filter cavity performs half the frequency-dependent
    rotation of the squeezing angle. Combined with appropriate homodyne
    angle (variational readout), this achieves broadband quantum noise
    reduction.
    """
    use_SS = True
    F_Hz = np.geomspace(10, 30e3, 1000)

    # load reference budgets
    budgetApl = gwinc.load_budget('Aplus', freq=F_Hz)


    # reference curves
    aplB = budgetApl.run()
    aplQB = aplB.Quantum
    
    # load internal FD squeezing config
    budget = gwinc.load_budget(fpath_join('AhatTestIntFC.yaml')) 
    ifo = budget.ifo

    ifo.Optics.INTSQ_loss = 1000e-6

    print(ifo.Optics)

    sfluB = sflu_CCwIntFDSqz()
    total, ASport, LB, _ = _compute_intFDsqz_budget(sfluB, F_Hz, ifo, use_SS)


    L_arm = 4000

    #
    # --- comparison with non-FC internal squeezing ---
    from . import test_CCwIntSqz
    sfluB_noFC = test_CCwIntSqz.sflu_CoupledCav()
    sflu_noFC = sfluB_noFC.sflu
    sflu_noFC.reduce_auto()
    params_noFC = standardize_params(ifo)
    mlib_noFC = params_noFC.mlib

    mats_noFC = MatsHelper()
    mats_noFC.H['AS'] = mlib_noFC.Id
    L_inj_t_noFC = (1 - params_noFC.Loss.injection)**0.5
    mats_noFC.update_scalar(L_inj_t_noFC)
    mats_noFC.T['Loss_injection'] = mlib_noFC.Id * params_noFC.Loss.injection**0.5

    ret_noFC = test_CCwIntSqz.CoupledCavity(
        sflu=sflu_noFC,
        F_Hz=F_Hz,
        ifo=ifo,
        params=params_noFC,
        use_SS=use_SS,
    )
    results_noFC = ret_noFC['resultsAC']
    mats_noFC.update_matrix(results_noFC["SEM.bk.i.exc"])
    mats_noFC.T.update({k: v for k, v in results_noFC.items() if k != "SEM.bk.i.exc"})
    L_read_t_noFC = (1 - params_noFC.Loss.readout)**0.5
    mats_noFC.update_scalar(L_read_t_noFC)
    mats_noFC.T['Loss_readout'] = mlib_noFC.diag(params_noFC.Loss.readout**0.5)

    params_noFC = standardize_params(ifo)
    mlib_noFC = params_noFC.mlib
    sqzV_noFC = 10**(-params_noFC.SQZ_DB/10.)
    asqzV_noFC = 10**(params_noFC.ANTISQZ_DB/10.)
    LOa_noFC = adjoint(mlib_noFC.LO(params_noFC.LO_angle))
    d_sense_noFC = np.sum([cc * mats_noFC.T[exc] for exc, cc in sfluB_noFC.strain_exc.items()], axis=0)
    d_sense_noFC = np.squeeze(LOa_noFC @ d_sense_noFC)
    k_noFC = 2 * np.pi / ifo.Laser.Wavelength
    PSD_noFC = (const.hbar * k_noFC * const.c / 2) / abs(d_sense_noFC)**2

    ASbudget_noFC = {}
    for _k in mats_noFC.T:
        if _k in sfluB_noFC.strain_exc.keys():
            continue
        ASbudget_noFC[_k] = Vnorm_sqA(LOa_noFC @ mats_noFC.T[_k])
    LOdotAS_noFC = LOa_noFC @ mats_noFC.H['AS']
    ASquantumAll_noFC = Vnorm_sqA(LOdotAS_noFC @ mlib_noFC.Mrotation(params_noFC.alpha) @ mlib_noFC.SQZ(sqzV_noFC, asqzV_noFC))
    total_noFC = ASquantumAll_noFC * PSD_noFC
    def sum_losses_noFC(noises, loss_ports):
        return np.sum([noises[exc_pt] for exc_pt in loss_ports], axis=0)
    for lpN, lpL in sfluB_noFC.loss_ports.items():
        if lpN == 'FilterCavity':
            continue
        total_noFC += PSD_noFC * sum_losses_noFC(ASbudget_noFC, lpL)
    
    # --- total noise comparison plot ---
    axB = mplfigB()
    axB.ax0.set_ylim(1e-25, 3e-23)
    axB.ax0.loglog(F_Hz, total**0.5 / L_arm, label='IntFDSqz (TWC)', lw=3, color='teal')
    axB.ax0.loglog(F_Hz, total_noFC**0.5 / L_arm, label='IntSqz (no FC)', lw=2, ls='--', color='orange')
    axB.ax0.loglog(aplQB.freq, aplQB.asd, label='A+ quantum', lw=2, color='black')
    axB.ax0.loglog(aplQB.freq, aplQB.asd/2, label='A+ quantum/2', lw=1, ls='--', color='black')
    axB.ax0.legend(loc='lower left', framealpha=1, fontsize=7)
    axB.ax0.set_xlabel('Frequency [Hz]')
    axB.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
    axB.save(tpath_join('intFDsqz_cmp'))

    # --- loss budget plot ---
    axL = mplfigB()
    axL.ax0.set_ylim(1e-25, 3e-23)
    axL.ax0.loglog(F_Hz, total**0.5 / L_arm, label='Total', lw=3, color='teal')
    axL.ax0.loglog(F_Hz, ASport**0.5 / L_arm, label='AS port', lw=2, color='gray')
    for lpN in LB:
        axL.ax0.loglog(F_Hz, LB[lpN]**0.5 / L_arm, label=lpN, lw=1.5, ls='--')
    axL.ax0.legend(loc='lower left', framealpha=1, fontsize=7)
    axL.ax0.set_xlabel('Frequency [Hz]')
    axL.ax0.set_ylabel('Strain ASD [1/$\\sqrt{\\mathrm{Hz}}$]')
    axL.save(tpath_join('intFDsqz_loss_budget'))

    pprint("IntFDSqz test completed")
    pprint(f"Loss port keys: {list(LB.keys())}")

    return


def test_build_CCwIntFDSqz(tpath_join):
    """Test that the SFLU graph builds and serializes correctly."""
    sfluB = sflu_CCwIntFDSqz()
    sflu = sfluB.sflu
    yamlstr = sflu.convert_self2yamlstr()
    with open(tpath_join('CoupledCavINTFDSQZ.yaml'), 'w') as F:
        F.write(yamlstr)
    sflu2 = SFLU.SFLU.convert_yamlstr2self(yamlstr)


def test_plot_graph_CCwIntFDSqz(tpath_join):
    """Plot the signal flow graph for the internal FD squeezing topology.

    Generates two graphs:
    - G1: full unreduced graph showing all nodes and edges
    - G2: reduced graph after SFLU auto-reduction
    """
    sfluB = sflu_CCwIntFDSqz()
    sflu = sfluB.sflu
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-8, rX=+8, Y=3, dY=-3)
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-8, rX=+8, Y=3, dY=-3)
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname=tpath_join("intFDsqz_graph.pdf"),
        scale="10pt",
    )
