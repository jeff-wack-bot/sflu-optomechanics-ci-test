"""
Test radiation pressure effects on a mirror

1) A single mirror illuminated by lasers from both sides with different phases
2) An optical spring
"""

import numpy as np
from wavestate.control.SFLU import SFLU, optics, nx2tikz
from sflu_components import elements, edges, simlib
from sflu_components.lib import MatrixLib, adjoint, Vnorm_sq, transpose
from gwinc.struct import Struct
from copy import deepcopy
import pytest


pi2i = 2j * np.pi
F_Hz = np.logspace(-1, 4, 3000)

################################################################################
# Single mirror
################################################################################

paramsMirr = Struct(
    Thr = 0.2,
    Pfr_W = 120,
    Pbk_W = 90,
    Plo_W = 1,
    phase_rad = 30*np.pi/180,
    lambda_m = 1064e-9,
    M_kg = 1e-6,
    f0_Hz = 1,
    Q = 100,
)


@pytest.fixture
@simlib.optickle_model("opt_mirr", paramsMirr)
def opt_mirr():
    import qlance.optickle as qopt
    eng = simlib.start_matlab_engine()

    opt = qopt.Optickle(eng, "opt", lambda0=paramsMirr.lambda_m)
    opt.addMirror("M", Thr=paramsMirr.Thr)
    opt.setMechTF("M", [], [0, 0], 1/paramsMirr.M_kg)

    opt.addSource("Laser_fr", np.sqrt(paramsMirr.Pfr_W))
    opt.addSource("Laser_bk", np.sqrt(paramsMirr.Pbk_W) * np.exp(1j*paramsMirr.phase_rad))
    opt.addLink("Laser_fr", "out", "M", "fr", 0)
    opt.addLink("Laser_bk", "out", "M", "bk", 0)

    opt.addHomodyneReadout("FR", LOpower=paramsMirr.Plo_W)
    opt.addLink("M", "fr", "FR_BS", "fr", 0)

    opt.run(F_Hz)
    return opt


@pytest.fixture
@simlib.finesse_model("kat_mirr", paramsMirr)
def kat_mirr():
    from pykat import finesse
    import qlance.finesse as qfin

    kat = finesse.kat()
    kat.lambda0 = paramsMirr.lambda_m
    qfin.addMirror(kat, "M", Thr=paramsMirr.Thr, comp=True)
    qfin.setMechTF(kat, "M", [], [0, 0], 1/paramsMirr.M_kg)

    qfin.addLaser(kat, "Laser_fr", paramsMirr.Pfr_W)
    qfin.addLaser(
        kat, "Laser_bk", paramsMirr.Pbk_W,
        phase=-paramsMirr.phase_rad*180/np.pi,
    )
    qfin.addSpace(kat, "Laser_fr_out", "M_fr", 0)
    qfin.addSpace(kat, "Laser_bk_out", "M_bk", 0)
    qfin.monitorMotion(kat, "M")

    katFR = qfin.KatFR(kat, all_drives=False)
    katFR.addDrives("M")
    katFR.run(F_Hz[0], F_Hz[-1], len(F_Hz), rtype='mech')
    return katFR


@pytest.fixture
def sflu_mirror():
    ifo = optics.GraphElement()
    ifo.subgraph_add(
        "M", elements.RPMirrorElement(),
        translation_xy=(0, 0),
        rotation_deg=0,
    )

    ifo['M'].locations.update({
        "Laser.fr": (-11, +7),
        "Laser.bk": (+11, -7),
    })
    ifo['M'].edges.update({
        ("fr.i", "Laser.fr"): "1",
        ("bk.i", "Laser.bk"): "1",
    })

    sflu = SFLU.SFLU(
        edges = ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


@pytest.fixture
def sflu_mirror_results(sflu_mirror, pprint):
    sflu = sflu_mirror
    sflu.reduce_auto()

    M = edges.RPMirrorEdge(
        "M", Thr=paramsMirr.Thr,
        suscept=lambda F_Hz: -1/(paramsMirr.M_kg * (2*np.pi*F_Hz)**2),
        lambda_m=paramsMirr.lambda_m,
    )
    mlib = M.mlib

    Id = {
        "1": mlib.Id,
        "1s": mlib.Id_s,
        "1v": mlib.Id_v,
        "1a": mlib.Id_a,
    }

    # DC calculation
    edgesDC = deepcopy(Id)
    edgesDC.update(M.edgesDC())

    compDC = sflu.computer()
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        {
            "M.fr.i.tp",
            "M.fr.o.tp",
            "M.bk.i.tp",
            "M.bk.o.tp",
        },
        {
            "M.Laser.fr": np.sqrt(paramsMirr.Pfr_W) * mlib.LO(np.pi/2),
            "M.Laser.bk": np.sqrt(paramsMirr.Pbk_W) * mlib.LO(np.pi/2 + paramsMirr.phase_rad),
        }
    )

    # AC calculation
    edgesAC = deepcopy(Id)
    edgesAC.update(M.edgesAC(F_Hz, resultsDC))

    compAC = sflu.computer()
    compAC.compute(edge_map=edgesAC)
    resultsAC = compAC.inverse_row(
        {"M.pos.tp": None},
        {"M.pos.exc"},
    )

    return resultsAC


def test_sflu_mirror(sflu_mirror_results, tpath_join, plotTF):
    resultsAC = sflu_mirror_results
    # mech_tf = resultsAC["M.fr.F.i.exc"][...,0, 0]
    chi = -1/(paramsMirr.M_kg * (2*np.pi*F_Hz)**2)
    mech_tf = resultsAC["M.pos.exc"] * chi

    fig = plotTF(F_Hz, mech_tf)
    fig.savefig(tpath_join("mech_tf.pdf"))


def test_sflu_mirror_compare_sim(
        sflu_mirror_results, opt_mirr, kat_mirr, tpath_join, plotTF):
    resultsAC = sflu_mirror_results
    # mech_tf_sflu = resultsAC["M.fr.F.i.exc"][...,0, 0]
    chi = -1/(paramsMirr.M_kg * (2*np.pi*F_Hz)**2)
    mech_tf_sflu = resultsAC["M.pos.exc"] * chi
    mech_tf_optickle = opt_mirr.getMechTF("M", "M")

    fig = plotTF(F_Hz, mech_tf_sflu, label='SFLU')
    # plotTF(F_Hz, mech_tf_optickle, *fig.axes, ls='--', label='Optickle')
    opt_mirr.plotMechTF("M", "M", *fig.axes, ls='--', label='Optickle')
    kat_mirr.plotMechTF("M", "M", *fig.axes, ls=':', c='xkcd:blood red', label='Finesse')
    fig.axes[0].legend()
    fig.savefig(tpath_join("mech_tf.pdf"))


def plot_mirror_graph(sflu_mirror, tpath_join):
    sflu = sflu_mirror
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-10, rX=+10, Y=8, dY=-8),
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-10, rX=+10, Y=8, dY=-8),
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname=tpath_join("testG.pdf"),
        scale="10pt",
    )

################################################################################
# Optical spring
################################################################################

paramsOS = Struct(
    Ti = 0.1,
    Te = 0.5,
    Larm_m = 40e3,
    M_kg = 10,
    Pfr_W = 100,
    Pbk_W = 300,
    Plo_W = 1,
    detune_rad = -10*np.pi/180,
    fr_rad = 0*np.pi/180,
    bk_rad = -90*np.pi/180,
    lambda_m = 1064e-9,
)


@pytest.fixture
def sflu_OS():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "IX", optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "EX", elements.RPMirrorElement(),
        translation_xy=(+10, 0),
        rotation_deg=0,
    )

    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): "tau",
        ("IX.fr.i", "EX.fr.o"): "tau",
    })
    ifo.locations.update({
        "IX.bk.i.exc": (-20, +5),
        "IX.bk.o.tp": (-20, -5),
        "EX.bk.i.exc": (+25, -7),
        "EX.bk.o.tp": (+25, +7),
    })
    ifo.edges.update({
        ("IX.bk.i", "IX.bk.i.exc"): "1",
        ("IX.bk.o.tp", "IX.bk.o"): "1",
        ("EX.bk.i", "EX.bk.i.exc"): "Lb",
        ("EX.bk.o.tp", "EX.bk.o"): "1",
    })
    ifo["EX"].locations["fr.o.exc"] = (-3, -10)
    ifo["EX"].edges[("fr.o", "fr.o.exc")] = "1"

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


@pytest.fixture
def sflu_OS_results(sflu_OS, pprint):
    sflu = sflu_OS
    sflu.reduce_auto()

    IX = edges.MirrorEdge("IX", Thr=paramsOS.Ti)
    EX = edges.RPMirrorEdge(
        "EX", Thr=paramsOS.Te,
        suscept=lambda F_Hz: -1/(paramsOS.M_kg * (2*np.pi*F_Hz)**2),
    )
    ArmLink = edges.LinkEdge(
        "tau", L_m=paramsOS.Larm_m,
        detune_rad=paramsOS.detune_rad,
    )
    BackLaserLink = edges.LinkEdge(
        "Lb", L_m=0,
        detune_rad=-paramsOS.detune_rad,
    )

    mlib = IX.mlib

    Id = {
        '1': mlib.Id,     # 2x2 matrix identity
        '1s': mlib.Id_s,  # 1x1 scalar identity
        '1v': mlib.Id_v,  # 2x1 vector identity
        '1a': mlib.Id_a,  # 1x2 adjoint identity
    }

    ##################################################
    # DC calculation
    ##################################################

    edgesDC = deepcopy(Id)
    edgesDC.update(IX.edgesDC())
    edgesDC.update(EX.edgesDC())
    edgesDC.update(ArmLink.edgesDC())
    edgesDC.update(BackLaserLink.edgesDC())

    vfr = mlib.LO(np.pi/2 + paramsOS.fr_rad)
    vbk = mlib.LO(np.pi/2 + paramsOS.bk_rad)

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    resultsDC = compDC.inverse_col(
        {
            "EX.fr.i.tp",
            "EX.fr.o.tp",
            "EX.bk.i.tp",
            "EX.bk.o.tp",
        },
        {
            "IX.bk.i.exc": np.sqrt(paramsOS.Pfr_W) * vfr,
            "EX.bk.i.exc": np.sqrt(paramsOS.Pbk_W) * vbk,
        }
    )

    ##################################################
    # AC calculation
    ##################################################

    edgesAC = deepcopy(Id)
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(ArmLink.edgesAC(F_Hz, resultsDC))
    edgesAC.update(BackLaserLink.edgesAC(F_Hz, resultsDC))

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC_opt = compAC.inverse_row(
        {"IX.bk.o.tp": None},
        {
            "EX.pos.exc",
            "EX.fr.o.exc",
        },
    )
    resultsAC_mech = compAC.inverse_row(
        {"EX.pos.tp": None},
        {
            "EX.pos.exc",
            "EX.fr.o.exc",
        },
    )

    return resultsDC, resultsAC_opt, resultsAC_mech


def test_sflu_OS(sflu_OS_results, tpath_join, pprint, plotTF):
    resultsDC, resultsAC_opt, resultsAC_mech = sflu_OS_results
    mlib = MatrixLib(nhom=0)
    pprint("SFLU", np.abs(resultsDC["EX.fr.o.tp"])**2)

    LOa = np.sqrt(paramsOS.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt["EX.pos.exc"]
    opt_tf = -opt_tf[..., 0, 0]

    # works well with no back laser
    fieldsDC_fr_i = mlib.Mrotation(np.pi/2) @ resultsDC["EX.fr.i.tp"]
    ArmPhase = resultsAC_opt["EX.fr.o.exc"] @ fieldsDC_fr_i
    pprint(ArmPhase.shape)
    opt_tf_field = np.squeeze(LOa @ ArmPhase) * 4 * np.pi / paramsOS.lambda_m
    opt_tf_field *= -2**(-1/2)
    # LOdotArmPhase = (LOa @ ArmPhase)[..., 0, 1]
    # opt_tf_field = 4 * np.pi / paramsOS.lambda_m * LOdotArmPhase

    chi = -1/(paramsOS.M_kg * (2*np.pi*F_Hz)**2)
    mech_tf = resultsAC_mech["EX.pos.exc"] * chi
    # mech_tf = resultsAC_mech["EX.fr.F.i.exc"][..., 0, 0]

    fig = plotTF(F_Hz, opt_tf, label="SFLU")
    plotTF(F_Hz, opt_tf_field, *fig.axes, ls='--', label="SFLU field")
    fig.axes[0].legend()
    fig.axes[0].set_title("Phase response to mirror motion")
    fig.axes[0].set_ylabel("Magnitude [W/m]")
    fig.savefig(tpath_join("optical_tf.pdf"))

    fig = plotTF(F_Hz, mech_tf, label="SFLU")
    fig.axes[0].legend()
    fig.axes[0].set_title("Radiation pressure modified mechanical susceptibility")
    fig.axes[0].set_ylabel("Magnitude [m/N]")
    fig.savefig(tpath_join("mechanical_tf.pdf"))


def test_sflu_OS_compare_sim(
        sflu_OS_results, opt_OS, kat_OS, tpath_join, plotTF, pprint):
    resultsDC, resultsAC_opt, resultsAC_mech = sflu_OS_results
    mlib = MatrixLib(nhom=0)
    opt = opt_OS
    katFR = kat_OS
    # pprint("SFLU", adjoint(resultsDC["EX.fr.i.tp"]) @ resultsDC["EX.fr.i.tp"])
    pprint("SFLU", Vnorm_sq(resultsDC["EX.fr.i.tp"]), Vnorm_sq(resultsDC["EX.fr.o.tp"]))
    pprint("Optickle", opt.getSigDC("EX_DC"))
    pprint("Finesse", katFR.getSigDC("EX_DC"))

    LOa = np.sqrt(paramsOS.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt['EX.pos.exc']
    opt_tf = -opt_tf[..., 0, 0]
    # mech_tf = resultsAC_mech['EX.fr.F.i.exc']
    chi = -1/(paramsOS.M_kg * (2*np.pi*F_Hz)**2)
    mech_tf = resultsAC_mech['EX.pos.exc'] * chi
    opt_tf_optickle = opt.getTF('REFL_DIFF', 'EX') / 2
    opt_tf_finesse = katFR.getTF('REFL_DIFF', 'EX') / 2

    fig = plotTF(F_Hz, opt_tf, label='SFLU')
    plotTF(F_Hz, opt_tf_optickle, *fig.axes, ls='--', label='Optickle')
    plotTF(katFR.ff, opt_tf_finesse, *fig.axes, ls=':', c='xkcd:blood red', label='Finesse')
    fig.axes[0].legend()
    fig.axes[0].set_title('Phase response to mirror motion')
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('optical_tf.pdf'))

    fig = plotTF(F_Hz, mech_tf, label='SFLU')
    opt.plotMechTF('EX', 'EX', *fig.axes, ls='--', label='Optickle')
    katFR.plotMechTF('EX', 'EX', *fig.axes, ls=':', c='xkcd:blood red', label='Finesse')
    fig.axes[0].set_title('Radiation pressure modified mechanical susceptibility')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [m/N]')
    fig.savefig(tpath_join('mechanical_tf.pdf'))


@pytest.fixture
@simlib.optickle_model("opt_OS", paramsOS)
def opt_OS():
    """
    Optickle model for FP cavity illuminated by a laser on both sides
    """
    import qlance.optickle as qopt
    eng = simlib.start_matlab_engine()

    opt = qopt.Optickle(eng, 'opt', lambda0=paramsOS.lambda_m)
    opt.addMirror("IX", Thr=paramsOS.Ti)
    opt.addMirror("EX", Thr=paramsOS.Te)
    opt.addLink("IX", "fr", "EX", "fr", paramsOS.Larm_m)
    opt.addLink("EX", "fr", "IX", "fr", paramsOS.Larm_m)
    opt.setMechTF("EX", [], [0, 0], 1/paramsOS.M_kg)

    detune_m = paramsOS.lambda_m * paramsOS.detune_rad / (2*np.pi)
    opt.setPosOffset("EX", detune_m)

    vfr = np.exp(1j * paramsOS.fr_rad)
    vbk = np.exp(1j * paramsOS.bk_rad)
    opt.addSource("Laser_fr", np.sqrt(paramsOS.Pfr_W) * vfr)
    opt.addSource("Laser_bk", np.sqrt(paramsOS.Pbk_W) * vbk)
    opt.addLink("Laser_fr", "out", "IX", "bk", 0)
    opt.addLink("Laser_bk", "out", "EX", "bk", 0)

    opt.addProbeIn("EX_DC", "EX", "fr", 0, 0)
    opt.addHomodyneReadout("REFL", LOpower=paramsOS.Plo_W)
    opt.addLink("IX", "bk", "REFL_BS", "fr", 0)

    opt.run(F_Hz, noise=False)
    return opt


@pytest.fixture
@simlib.finesse_model("kat_OS", paramsOS)
def kat_OS():
    """
    Optickle model for FP cavity illuminated by a laser on both sides
    """
    from pykat import finesse
    import qlance.finesse as qfin

    kat = finesse.kat()
    kat.lamba0 = paramsOS.lambda_m
    qfin.addMirror(kat, "IX", Thr=paramsOS.Ti)
    qfin.addMirror(kat, "EX", Thr=paramsOS.Te, comp=True)
    qfin.addSpace(kat, "IX_fr", "EX_fr", paramsOS.Larm_m)
    qfin.setMechTF(kat, "EX", [], [0, 0], 1/paramsOS.M_kg)
    kat.EX.phi = paramsOS.detune_rad * 180/np.pi

    qfin.addLaser(
        kat, "Laser_fr", paramsOS.Pfr_W,
        phase=-paramsOS.fr_rad*180/np.pi,
    )
    qfin.addLaser(
        kat, "Laser_bk", paramsOS.Pbk_W,
        phase=-paramsOS.bk_rad*180/np.pi + 90,
    )
    qfin.addFaradayIsolator(kat, "REFL")
    qfin.addSpace(kat, "Laser_fr_out", "REFL_fr_in", 0)
    qfin.addSpace(kat, "REFL_fr_out", "IX_bk", 0)
    qfin.addSpace(kat, "Laser_bk_out", "EX_bk", 0)

    qfin.addHomodyneReadout(kat, "REFL", LOpower=paramsOS.Plo_W)
    qfin.addSpace(kat, "REFL_bk_out", "REFL_BS_frI", 0)

    qfin.monitorMotion(kat, "EX")
    qfin.addProbe(kat, "EX_DC", "EX_fr", 0, 0, alternate_beam=True)

    katFR = qfin.KatFR(kat, all_drives=False)
    katFR.addDrives("EX")
    katFR.runDC()
    katFR.run(F_Hz[0], F_Hz[-1], len(F_Hz))
    return katFR


def plot_OS_graph(sflu_OS, tpath_join):
    sflu = sflu_OS
    G1 = sflu.G.copy()
    sflu.graph_reduce_auto_pos(lX=-10, rX=+10, Y=8, dY=-8),
    sflu.reduce_auto()
    sflu.graph_reduce_auto_pos_io(lX=-10, rX=+10, Y=8, dY=-8),
    G2 = sflu.G.copy()

    nx2tikz.dump_pdf(
        [G1, G2],
        fname=tpath_join("testG.pdf"),
        scale="10pt",
    )
