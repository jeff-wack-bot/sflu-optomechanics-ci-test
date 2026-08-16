"""
Coupled-cavity interferometer with internal squeezing.

The reference internal-squeezing model: a squeezer placed inside the signal
extraction cavity, between the ITM and the signal recycling mirror. This is
the best-known-good state of the code and its numbers are pinned by
``tools/regression/``.

Split into the two things it is:

``sflu_CoupledCav()``   topology -- which optics exist and how they are wired
``CoupledCavity()``     plant    -- parameters to edges to transfer functions

``intSqzQuantum()`` runs both and hands the result to the shared budget.
"""
import numpy as np
from gwinc.struct import Struct
from wield.bunch import Bunch
from wield.control import SISO
from wield.control.SFLU import SFLU

from sflu_components import edges, elements
from sflu.models import filter_cavity
from sflu.models.budget import accumulate, quantum_budget
from sflu.params import arm_gouyRT, standardize_params


def sflu_CoupledCav(
        use_ITMRP=True
):
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
            translation_xy=(45, 0),
            rotation_deg=180,
        )

    ifo.subgraph_add(
        "ETM", elements.RPMirrorElement(loss_ports=True),
        translation_xy=(95, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "SEM", elements.MirrorElement(loss_ports=True),
        translation_xy=(0, 0),
        rotation_deg=180,
    )

    ##

    # add another lossy mirror for the SQZ beamsplitter model losses
    ifo.subgraph_add(
        "INTSQZL", elements.MirrorElement(loss_ports=True),
        translation_xy=(25, 0),
        rotation_deg=180,
    )

    ifo.edges.update({
        ("ETM.fr.i", "ITM.fr.o"): "ARM.L",
        ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
        ("ITM.bk.i", "INTSQZL.fr.o"): "SEC.L.to",
        ("INTSQZL.fr.i", "ITM.bk.o"): "SEC.L.fr",
        # and these two for the linkage of the internal squeezing
        ("SEM.fr.i", "INTSQZL.bk.o"): "INTSQZ.armfr",
        ("INTSQZL.bk.i", "SEM.fr.o"): "INTSQZ.armto",
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
            INTSQZ=[
                "INTSQZL.frL.i",
                "INTSQZL.bkL.i",
            ],
            # INTSQZf=[
            #     "INTSQZL.frL.i",
            # ],
            # INTSQZb=[
            #     "INTSQZL.bkL.i",
            # ],
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
        return 2 * (1 / (40 * (2*np.pi * F_Hz)**2))
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

    edge_objs.ETM = edges.RPMirrorEdge(
        name     = 'ETM',
        Thr      = Te,
        Lhr      = params.Loss.arm_rt / 2,
        # suscept  = suscept_m_N,
        suscept = suscept_func,
        suscept_ss = suscept_ss,
        lambda_m = lambda_m,
        mlib     = mlib,
        loss_ports = True,
    )
    if True:
        edge_objs.ITM = edges.RPMirrorEdge(
            name    = 'ITM',
            Thr     = Ti,
            Lhr     = params.Loss.arm_rt / 2,
            mlib    = mlib,
            loss_ports = True,
        )
    else:
        edge_objs.ITM = edges.MirrorEdge(
            name    = 'ITM',
            Thr     = Ti,
            Lhr     = params.Loss.arm_rt / 2,
            mlib    = mlib,
            loss_ports = True,
        )
    edge_objs.SEM = edges.MirrorEdge(
        name = 'SEM',
        Thr  = Ts,
        Lhr  = params.Loss.SEC_rt,
        mlib = mlib,
        loss_ports = True,
    )

    edge_objs.INTSQZL = edges.MirrorEdge(
        name = 'INTSQZL',
        Thr  = 1,
        Lhr  = ifo.Optics.INTSQ_loss + params.Loss.SEC_rt/2,
        loss_in_transmission=True,
        mlib = mlib,
        loss_ports = True,
    )

    ####################
    # links
    ####################

    def get_MM_fr(MM):
        if params.is_OPD:
            return MM
        else:
            return mlib.Minv(MM)

    edge_objs.ARM = edges.LinkEdge(
        name     = 'ARM.L',
        L_m      = params.Length_m.ARM,
        gouy_rad = ARM_gouy_rad,
        mlib     = mlib,
    )
    edge_objs.L_SEC_to = edges.LinkEdge(
        name       = 'SEC.L.to',
        L_m        = params.Length_m.SEM,
        detune_rad = SEC_detune_rad,
        gouy_rad   = SEC_gouy_rad,
        MM_to      = params.MM.SEC_ARM,
        mlib       = mlib,
    )
    edge_objs.L_SEC_fr = edges.LinkEdge(
        name       = 'SEC.L.fr',
        L_m        = params.Length_m.SEM,
        detune_rad = SEC_detune_rad - 0/180*np.pi,
        gouy_rad   = SEC_gouy_rad,
        MM_fr      = get_MM_fr(params.MM.SEC_ARM),
        mlib       = mlib,
    )

    MM_INTSQZ = mlib.MrotationMM(ifo.Optics.MM_INTSQZ, 0 / 180 * np.pi)
    shift = (SEC_gouy_rad * ifo.Optics.MM_INTSQZ) * 180/np.pi
    edge_objs.INTSQZ_armto = edges.SQZEdge(
        name       = 'INTSQZ.armto',
        sqzDB      = -ifo.intSqueezer.AmplitudedB,
        sqzANGdeg = -0 - 90,
        MM_to      = MM_INTSQZ,
        MM_fr      = mlib.Minv(MM_INTSQZ),
        mlib       = mlib,
        # dual_unphysical = True,
    )
    edge_objs.INTSQZ_armfr = edges.SQZEdge(
        name       = 'INTSQZ.armfr',
        sqzDB      = ifo.intSqueezer.AmplitudedB,
        # this micro angle detuning is for some MM loss compensation
        sqzANGdeg = -0 - 90 - 0*.005,
        MM_to      = MM_INTSQZ,
        MM_fr      = mlib.Minv(MM_INTSQZ),
        #MM_fr      = get_MM_fr(params.MM.SEC_ARM),
        mlib       = mlib,
        # dual_unphysical = True,
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
        "INTSQZL.frL.i",
        "INTSQZL.bkL.i",

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

def intSqzQuantum(ifo, freq, use_SS=True):
    """Quantum noise budget for the coupled cavity with internal squeezing.

    Builds a fresh topology each call, because ``reduce_auto()`` mutates the
    graph in place.

    Returns
    -------
    total : array
        Total quantum noise strain PSD.
    LB : dict
        Per-port loss breakdown. See the ``alias_ASport`` note in
        ``budget.quantum_budget`` for the meaning of ``LB['ASport']``.
    """
    sfluB = sflu_CoupledCav()
    sfluB.sflu.reduce_auto()
    params = standardize_params(ifo)

    mats = accumulate(
        sfluB, plant=CoupledCavity, ifo=ifo, params=params, F_Hz=freq,
        use_SS=use_SS, filter_cavity=filter_cavity.FilterCavity,
    )
    out = quantum_budget(
        sfluB, mats, ifo, params, F_Hz=freq, strain=True, alias_ASport=True,
    )
    return out.total, out.LB
