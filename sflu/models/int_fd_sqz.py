"""
Coupled cavity with internal frequency-dependent squeezing.

Extends the internal-squeezing model by putting a detuned travelling-wave
filter cavity *inside* the signal extraction cavity, so the squeeze angle
rotates with frequency without an external filter cavity.

``sflu_CCwIntFDSqz()``     topology
``CoupledCavityIntFC()``   plant
"""
import numpy as np
from gwinc import const
from gwinc.struct import Struct
from wield.bunch import Bunch
from wield.control import SISO
from wield.control.SFLU import SFLU

from sflu_components import edges, elements
from sflu.models.budget import accumulate, quantum_budget
from sflu.params import arm_gouyRT, standardize_params


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
        "IFCBS", elements.BeamSplitterElement(),
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
            INTSQZ=[],
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
    IFC_detune_m = -IFC_detune_Hz / FSR_Hz * lambda_m
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
    edge_objs.ETM = edges.RPMirrorEdge(
        name     = 'ETM',
        Thr      = Te,
        Lhr      = params.Loss.arm_rt / 2,
        suscept  = suscept_func,
        suscept_ss = suscept_ss,
        lambda_m = lambda_m,
        mlib     = mlib,
        loss_ports = True,
    )
    edge_objs.ITM = edges.RPMirrorEdge(
        name     = 'ITM',
        Thr      = Ti,
        Lhr      = params.Loss.arm_rt / 2,
        mlib     = mlib,
        loss_ports = True,
    )
    edge_objs.SEM = edges.MirrorEdge(
        name = 'SEM',
        Thr  = Ts,
        Lhr  = params.Loss.SEC_rt,
        mlib = mlib,
        loss_ports = True,
    )

    # travelling-wave filter cavity coupling mirror (beamsplitter)
    # bkA face loss absorbs the former INTSQZL mirror losses
    edge_objs.IFCBS = edges.BSEdge(
        name = 'IFCBS',
        Thr  = IFC_Ti,
        Lhr  = 0,
        mlib = mlib,
    )

    # links
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
        detune_rad = SEC_detune_rad,
        gouy_rad   = SEC_gouy_rad,
        MM_fr      = get_MM_fr(params.MM.SEC_ARM),
        mlib       = mlib,
    )

    # travelling-wave filter cavity self-loop link (full round-trip)
    edge_objs.L_IFC = edges.LinkEdge(
        name       = 'IFC.L',
        L_m        = IFC_L_rt,
        detune_rad = IFC_detune_rad,
        gouy_rad   = IFC_gouy_rad_rt,
        mlib       = mlib,
    )

    print(IFC_detune_rad)
    # internal squeezing edges
    MM_INTSQZ = mlib.MrotationMM(ifo.Optics.MM_INTSQZ, 0 / 180 * np.pi)
    edge_objs.INTSQZ_armto = edges.SQZEdge(
        name       = 'INTSQZ.armto',
        sqzDB      = -ifo.intSqueezer.AmplitudedB,
        sqzANGdeg  = 0 - 90,
        MM_to      = MM_INTSQZ,
        MM_fr      = mlib.Minv(MM_INTSQZ),
        mlib       = mlib,
    )
    edge_objs.INTSQZ_armfr = edges.SQZEdge(
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

def intFDsqzQuantum(ifo, freq, use_SS=True):
    """Quantum noise budget for internal frequency-dependent squeezing.

    No external filter cavity is chained in front of the plant: this model's
    filter cavity lives inside the interferometer graph.

    Returns
    -------
    total : array
        Total quantum noise displacement PSD.
    LB : dict
        Per-port loss breakdown, plus the AS-port term under ``'ASport'``.
    d_sense : array
        Strain drive to readout transfer function.
    """
    sfluB = sflu_CCwIntFDSqz()
    sfluB.sflu.reduce_auto()
    params = standardize_params(ifo)

    mats = accumulate(
        sfluB, plant=CoupledCavityIntFC, ifo=ifo, params=params, F_Hz=freq,
        use_SS=use_SS, filter_cavity=None,
    )
    out = quantum_budget(sfluB, mats, ifo, params, F_Hz=freq, strain=False)
    LB = dict(out.LB)
    LB['ASport'] = out.ASport
    return out.total, LB, out.d_sense
