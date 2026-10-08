"""
Strain sensitivity of a single-arm FP cavity.

Stripped-down analogue of fromgwinc/intsqz/test_CCwIntSqz.py: one arm only, no
SRM, no squeezing, no filter cavity. Parameters are pulled from the aLIGO A+
ifo.yaml, strain is injected at the end mirror with the Michelson-equivalent
1/sqrt(2) factor.
"""
import numpy as np
import matplotlib.pyplot as plt

import gwinc
from gwinc.struct import Struct
from gwinc import const

from wield.bunch import Bunch
from wield.control.SFLU import SFLU

from sflu_components import elements
from sflu_components.lib import MatrixLib, adjoint

from fromgwinc.intsqz import optics
from fromgwinc.intsqz.lib import Vnorm_sqA


F_Hz = np.geomspace(10, 10e3, 1000)


def sflu_single_arm():
    """Build a single-arm FP SFLU graph (ITM + ETM + arm link)."""
    ifo = elements.optics.GraphElement()

    ifo.subgraph_add(
        "ITM", elements.MirrorElement(loss_ports=True),
        translation_xy=(0, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "ETM", elements.RPMirrorElement(loss_ports=True),
        translation_xy=(50, 0),
        rotation_deg=0,
    )

    ifo.edges.update({
        ("ETM.fr.i", "ITM.fr.o"): "ARM.L",
        ("ITM.fr.i", "ETM.fr.o"): "ARM.L",
    })

    # ITM back face: laser/vacuum input + homodyne tap (AS readout)
    ifo["ITM"].locations.update({
        "bk.i.exc": (15, -10),
        "bk.o.tp":  (15, +10),
    })
    ifo["ITM"].edges.update({
        ("bk.i",    "bk.i.exc"): "1",
        ("bk.o.tp", "bk.o"):     "1",
    })

    # ETM back: external vacuum coupling (loss via ETM transmission)
    ifo["ETM"].locations.update({
        "bk.i.exc": (-3, -10),
    })
    ifo["ETM"].edges.update({
        ("bk.i", "bk.i.exc"): "1",
    })

    sflu = SFLU.SFLU(edges=ifo.build_edges(), graph=True)
    ifo.update_sflu(sflu)

    return Bunch(
        sflu=sflu,
        loss_ports=dict(
            Arm=[
                "ETM.bk.i.exc",   # vacuum leaking in past ETM transmission
                "ETM.frL.i",
                "ETM.bkL.i",
                "ITM.frL.i",
                "ITM.bkL.i",
            ],
        ),
        signal_readout="ITM.bk.o.tp",
        laser_input="ITM.bk.i.exc",
        strain_exc={"ETM.pos.exc": 1 / np.sqrt(2)},
    )


def _load_single_arm_params(ifo):
    """Pull single-arm-relevant fields out of the full aLIGO ifo Struct."""
    p = Struct()
    p.lambda_m   = ifo.Laser.Wavelength
    p.Pin_W      = ifo.Laser.Power
    p.Lcav_m     = ifo.Infrastructure.Length
    p.M_kg       = ifo.Suspension.Stage[0].Mass
    p.Ti         = ifo.Optics.ITM.Transmittance
    p.Te         = ifo.Optics.ETM.Transmittance
    p.Loss_rt    = 2 * ifo.Optics.Loss             # per-mirror HR loss, round-trip
    p.detune_rad = 0.0                              # on resonance
    p.mlib       = MatrixLib(nhom=0)
    return p


def test_strain_single_arm(tpath_join, pprint):
    # ---------- parameters from aLIGO A+ -----------------------------------
    budget_apl = gwinc.load_budget("Aplus", freq=F_Hz)
    ifo_apl    = budget_apl.ifo
    p          = _load_single_arm_params(ifo_apl)
    pprint({k: v for k, v in p.items() if k != "mlib"})

    mlib = p.mlib

    # ---------- build the SFLU and auto-reduce -----------------------------
    sB   = sflu_single_arm()
    sflu = sB.sflu
    sflu.reduce_auto()

    # ---------- per-edge components (copied from intsqz CoupledCavity) -----
    suscept_func = lambda F_Hz: -1 / (p.M_kg * (2 * np.pi * F_Hz) ** 2)

    ETM = optics.RPMirrorEdge(
        name="ETM",
        Thr=p.Te,
        Lhr=p.Loss_rt / 2,
        suscept=suscept_func,
        lambda_m=p.lambda_m,
        mlib=mlib,
    )
    ITM = optics.RPMirrorEdge(
        name="ITM",
        Thr=p.Ti,
        Lhr=p.Loss_rt / 2,
        mlib=mlib,
    )
    ARM = optics.LinkEdge(
        name="ARM.L",
        L_m=p.Lcav_m,
        mlib=mlib,
    )

    edge_objs = [ETM, ITM, ARM]

    edge_map = {"1": mlib.Id, "1s": mlib.Id_s}

    # ---------- AC only (coupled-cavity approximation: arm carrier is set) --
    # Arm-circulating power from gwinc's power budget
    from gwinc.ifo.noises import ifo_power
    Parm_W = ifo_power(ifo_apl).parm
    pprint("Parm_W", Parm_W)

    Earm_rtW = np.sqrt(Parm_W) * mlib.LO(np.pi / 2)
    resultsDC = {
        "ETM.fr.i.tp": Earm_rtW,
        "ETM.fr.o.tp": -ETM.r @ Earm_rtW,
        "ETM.bk.o.tp":  ETM.t @ Earm_rtW,
    }

    for eo in edge_objs:
        if isinstance(eo, optics.LinkEdge):
            edge_map.update(eo.edgesAC(F_Hz=F_Hz))
        else:
            edge_map.update(eo.edgesAC(F_Hz=F_Hz, resultsDC=resultsDC))

    inverse_row_Rmap = {sB.signal_readout: None}
    inverse_row_Cset = (
        set(sB.loss_ports["Arm"])
        | set(sB.strain_exc.keys())
        | {sB.laser_input}
    )

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edge_map)
    resultsAC = compAC.inverse_row(Rmap=inverse_row_Rmap, Cset=inverse_row_Cset)

    # ---------- quantum-noise math (mirrors test_CCwIntSqz.py) -------------
    # unit-norm LO row; carrier amplitude is already inside resultsDC
    LOa = adjoint(mlib.LO(0))

    # signal TF (scalar over F_Hz)
    d_sense = np.sum(
        [cc * resultsAC[exc] for exc, cc in sB.strain_exc.items()], axis=0
    )
    d_sense = np.squeeze(LOa @ d_sense)

    omega    = 2 * np.pi * const.c / p.lambda_m
    Qnoise   = const.hbar * omega / 2
    PSD_disp = Qnoise / np.abs(d_sense) ** 2       # m^2/Hz per unit readout

    # vacuum contributions from every loss/input port except the strain port
    vacuum_ports = [sB.laser_input] + sB.loss_ports["Arm"]
    ASbudget = {p_: Vnorm_sqA(LOa @ resultsAC[p_]) for p_ in vacuum_ports}

    PSD_displacement = sum(ASbudget.values()) * PSD_disp
    PSD_strain = PSD_displacement / p.Lcav_m ** 2
    ASD_strain = np.sqrt(PSD_strain)

    pprint("|d_sense| peak", float(np.max(np.abs(d_sense))))
    pprint("ASD @ 100 Hz", float(ASD_strain[np.argmin(np.abs(F_Hz - 100))]))

    # ---------- plot with A+ gwinc Quantum noise overlay -------------------
    apl = budget_apl.run()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(F_Hz, ASD_strain, lw=2, label="single-arm FP (SFLU)")
    ax.loglog(
        apl.Quantum.freq, apl.Quantum.asd,
        lw=1.5, color="k", alpha=0.6,
        label="A+ Quantum (gwinc, full DRFPMI)",
    )
    ax.set_xlim(F_Hz.min(), F_Hz.max())
    ax.set_ylim(1e-25, 1e-20)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Strain ASD [1/$\sqrt{\mathrm{Hz}}$]")
    ax.set_title("Single-arm FP cavity quantum noise (A+ parameters)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=1)
    fig.tight_layout()
    fig.savefig(tpath_join("strain_ASD.pdf"))
    fig.savefig(tpath_join("strain_ASD.png"), dpi=150)


def plot_graph_single_arm(tpath_join):
    from wield.control.SFLU import nx2tikz
    sB = sflu_single_arm()
    sflu = sB.sflu
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-8, rX=+8, Y=3, dY=-3)
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-8, rX=+8, Y=3, dY=-3)
    G2 = sflu.G.copy()
    nx2tikz.dump_pdf([G1, G2], fname=tpath_join("graph.pdf"), scale="10pt")
