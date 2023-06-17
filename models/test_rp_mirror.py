import numpy as np
from wield.control.SFLU import SFLU, optics, nx2tikz, SFLUcompute
import components as cmp
from gwinc.struct import Struct
from gwinc.noise.quantum_lib import adjoint, Vnorm_sq
import matlib
import scipy.constants as scc
from copy import deepcopy
import matplotlib.pyplot as plt
import pytest

pi2i = 2j * np.pi

F_Hz = np.logspace(-1, 4, 3000)

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
@matlib.optickle_model("opt_mirr", paramsMirr)
def opt_mirr():
    import qlance.optickle as qopt
    eng = matlib.start_matlab_engine()

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
@matlib.finesse_model("kat_mirr", paramsMirr)
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
        "M", cmp.RPMirrorElement(),
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

    M = cmp.RPMirrorEdge(
        "M", Thr=paramsMirr.Thr, M_kg=paramsMirr.M_kg, lambda_m=paramsMirr.lambda_m
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
        {"M.fr.F.i.exc"},
    )

    Fq_fr_o = edgesAC["M.fr.Fq.o"]
    Fq_bk_o = edgesAC["M.bk.Fq.o"]
    px_fr = edgesAC["M.fr.px"]
    px_bk = edgesAC["M.bk.px"]
    chi = edgesAC["M.chi"]

    rt = chi @ (Fq_fr_o @ px_fr + Fq_bk_o @ px_bk)
    cl = mlib.Minv(mlib.Id_s - rt)
    fx = cl @ chi

    return resultsAC, cl, fx


def test_sflu_mirror(sflu_mirror_results, tpath_join, plotTF, pprint):
    resultsAC, cl, fx = sflu_mirror_results
    mech_tf = resultsAC["M.fr.F.i.exc"][...,0, 0]
    pprint(cl.shape, fx.shape)

    fig = plotTF(F_Hz, mech_tf)
    plotTF(F_Hz, fx[..., 0, 0], *fig.axes, ls='--')
    plotTF(F_Hz, cl[..., 0, 0], *fig.axes, ls='-')
    fig.savefig(tpath_join("mech_tf.pdf"))


def test_sflu_mirror_compare_optickle(
        sflu_mirror_results, opt_mirr, kat_mirr, tpath_join, plotTF):
    resultsAC, cl, fx = sflu_mirror_results
    mech_tf_sflu = resultsAC["M.fr.F.i.exc"][...,0, 0]
    mech_tf_optickle = opt_mirr.getMechTF("M", "M")

    fig = plotTF(F_Hz, mech_tf_sflu, label='SFLU')
    # plotTF(F_Hz, mech_tf_optickle, *fig.axes, ls='--', label='Optickle')
    opt_mirr.plotMechTF("M", "M", *fig.axes, ls='--', label='Optickle')
    kat_mirr.plotMechTF("M", "M", *fig.axes, ls=':', c='xkcd:blood red', label='Finesse')
    fig.axes[0].legend()
    fig.savefig(tpath_join("mech_tf.pdf"))

################################################################################

paramsOS = Struct(
    Ti = 0.5,
    Larm_m = 40e3,
    M_kg = 10,
    Pin_W = 100,
    Plo_W = 1,
    detune_rad = -3*np.pi/180,
    source_rad = 68*np.pi/180,
    lambda_m = 1064e-9,
    front = True,
)


@pytest.fixture
def sflu_OS():
    ifo = optics.GraphElement()

    ############################################################
    # basis mirror: IX
    ############################################################
    ifo.subgraph_add(
        'IX', optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        'EX', cmp.RPMirrorElement(),
        translation_xy=(+10, 0),
        rotation_deg=0,
    )

    if paramsOS.front:
        ifo.edges.update({
            ("EX.fr.i", "IX.fr.o"): 'tau',
            ("IX.fr.i", "EX.fr.o"): 'tau',
        })
    else:
        ifo.edges.update({
            ("EX.bk.i", "IX.fr.o"): 'tau',
            ("IX.fr.i", "EX.bk.o"): 'tau',
        })

    ifo.locations.update({
        'IX.bk.i.exc': (-20, +5),
        'IX.bk.o.tp': (-20, -5),
    })
    ifo.edges.update({
        ('IX.bk.i', 'IX.bk.i.exc'): '1',
        ('IX.bk.o.tp', 'IX.bk.o'): '1',
    })

    ifo.node_angle['IX.bk.o.tp'] = +45

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

    IX = cmp.MirrorEdge('IX', Thr=paramsOS.Ti)
    EX = cmp.RPMirrorEdge('EX', Thr=0, M_kg=paramsOS.M_kg)
    detune_sign = (-1)**(not paramsOS.front)
    ArmLink = cmp.LinkEdge(
        'tau', L_m=paramsOS.Larm_m,
        detune_rad=detune_sign*paramsOS.detune_rad,
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

    compDC = sflu.computer(eye=mlib.Id)
    compDC.compute(edge_map=edgesDC)
    if paramsOS.front:
        tps = {
            'EX.fr.i.tp',
            'EX.fr.o.tp',
            # 'EX.bk.i.tp',
            'EX.bk.o.tp',
        }
    else:
        tps = {
            # 'EX.fr.i.tp',
            'EX.fr.o.tp',
            'EX.bk.i.tp',
            'EX.bk.o.tp',
        }

    resultsDC = compDC.inverse_col(
        tps,
        {'IX.bk.i.exc': np.sqrt(paramsOS.Pin_W) * mlib.LO(np.pi/2 + paramsOS.source_rad)},
    )

    ##################################################
    # AC calculation
    ##################################################

    edgesAC = deepcopy(Id)
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(ArmLink.edgesAC(F_Hz))

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC_opt = compAC.inverse_row(
        {'IX.bk.o.tp': None},
        {'EX.pos.exc'},
    )
    resultsAC_mech = compAC.inverse_row(
        {'EX.pos.tp': None},
        {'EX.fr.F.i.exc'},
    )

    return resultsAC_opt, resultsAC_mech


def test_sflu_OS(sflu_OS_results, tpath_join, pprint, plotTF):
    resultsAC_opt, resultsAC_mech = sflu_OS_results
    mlib = cmp.mats_planewave

    LOa = np.sqrt(paramsOS.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt['EX.pos.exc']
    opt_tf = opt_tf[..., 0, 0]
    mech_tf = resultsAC_mech['EX.fr.F.i.exc']
    mech_tf = mech_tf[..., 0, 0]

    fig = plotTF(F_Hz, opt_tf, label='SFLU')
    fig.axes[0].legend()
    fig.axes[0].set_title('Phase response to mirror motion')
    fig.axes[0].set_ylabel('Magnitude [W/m]')
    fig.savefig(tpath_join('optical_tf.pdf'))

    fig = plotTF(F_Hz, mech_tf, label='SFLU')
    fig.axes[0].set_title('Radiation pressure modified mechanical susceptibility')
    fig.axes[0].legend()
    fig.axes[0].set_ylabel('Magnitude [m/N]')
    fig.savefig(tpath_join('mechanical_tf.pdf'))


def test_sflu_OS_compare_optickle(
        sflu_OS_results, opt_OS, kat_OS, tpath_join, plotTF):
    resultsAC_opt, resultsAC_mech = sflu_OS_results
    mlib = cmp.mats_planewave
    opt = opt_OS
    katFR = kat_OS

    LOa = np.sqrt(paramsOS.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt['EX.pos.exc']
    opt_tf = -opt_tf[..., 0, 0]
    mech_tf = resultsAC_mech['EX.fr.F.i.exc']
    mech_tf = mech_tf[..., 0, 0]
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
@matlib.optickle_model('opt_OS', paramsOS)
def opt_OS():
    """
    Optickle model for a FP cavity
    """
    import qlance.optickle as qopt
    eng = matlib.start_matlab_engine()

    opt = qopt.Optickle(eng, 'opt', lambda0=paramsOS.lambda_m)
    opt.addMirror('IX', Thr=paramsOS.Ti)
    opt.addMirror('EX', Thr=0)
    if paramsOS.front:
        opt.addLink('IX', 'fr', 'EX', 'fr', paramsOS.Larm_m)
        opt.addLink('EX', 'fr', 'IX', 'fr', paramsOS.Larm_m)
    else:
        opt.addLink('IX', 'fr', 'EX', 'bk', paramsOS.Larm_m)
        opt.addLink('EX', 'bk', 'IX', 'fr', paramsOS.Larm_m)
    opt.setMechTF('EX', [], [0, 0], 1/paramsOS.M_kg)

    detune_m = paramsOS.lambda_m*paramsOS.detune_rad / (2*np.pi)
    opt.setPosOffset('EX', detune_m)

    opt.addSource('Laser', np.sqrt(paramsOS.Pin_W) * np.exp(1j*paramsOS.source_rad))
    opt.addLink('Laser', 'out', 'IX', 'bk', 0)

    # opt.addProbeIn('EX_DC', 'EX', 'fr', 0, 0)
    opt.addHomodyneReadout('REFL', LOpower=paramsOS.Plo_W)
    opt.addLink('IX', 'bk', 'REFL_BS', 'fr', 0)

    opt.run(F_Hz)
    return opt


@pytest.fixture
@matlib.finesse_model('kat_OS', paramsOS)
def kat_OS():
    """
    Finesse model for a FP cavity
    """
    from pykat import finesse
    import qlance.finesse as qfin

    kat = finesse.kat()
    kat.lambda0 = paramsOS.lambda_m
    qfin.addMirror(kat, 'IX', Thr=paramsOS.Ti)
    qfin.addMirror(kat, 'EX', Thr=0, comp=True)
    if paramsOS.front:
        qfin.addSpace(kat, 'IX_fr', 'EX_fr', paramsOS.Larm_m)
    else:
        qfin.addSpace(kat, 'IX_fr', 'EX_bk', paramsOS.Larm_m)
    qfin.setMechTF(kat, 'EX', [], [0, 0], 1/paramsOS.M_kg)
    kat.EX.phi = paramsOS.detune_rad * 180/np.pi

    qfin.addLaser(
        kat, 'Laser', paramsOS.Pin_W,
        phase=-paramsOS.source_rad*180/np.pi,
    )
    qfin.addFaradayIsolator(kat, 'REFL')
    # qfin.addSpace(kat, 'Laser_out', 'IX_bk', 0)
    qfin.addSpace(kat, 'Laser_out', 'REFL_fr_in', 0)
    qfin.addSpace(kat, 'REFL_fr_out', 'IX_bk', 0)

    qfin.addHomodyneReadout(kat, 'REFL', LOpower=paramsOS.Plo_W)
    qfin.addSpace(kat, 'REFL_bk_out', 'REFL_BS_frI', 0)

    qfin.monitorMotion(kat, 'EX')

    katFR = qfin.KatFR(kat, all_drives=False)
    katFR.addDrives('EX')
    katFR.run(F_Hz[0], F_Hz[-1], len(F_Hz))
    return katFR

################################################################################
# optical spring with a second laser on the back of the ETM

paramsOSbk = Struct(
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
def sflu_OSbk():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "IX", optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "EX", cmp.RPMirrorElement(),
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

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


@pytest.fixture
def sflu_OSbk_results(sflu_OSbk, pprint):
    sflu = sflu_OSbk
    sflu.reduce_auto()

    IX = cmp.MirrorEdge("IX", Thr=paramsOSbk.Ti)
    EX = cmp.RPMirrorEdge("EX", Thr=paramsOSbk.Te, M_kg=paramsOSbk.M_kg)
    ArmLink = cmp.LinkEdge(
        "tau", L_m=paramsOSbk.Larm_m,
        detune_rad=paramsOSbk.detune_rad,
    )
    BackLaserLink = cmp.LinkEdge(
        "Lb", L_m=0,
        detune_rad=-paramsOSbk.detune_rad,
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

    vfr = mlib.LO(np.pi/2 + paramsOSbk.fr_rad)
    vbk = mlib.LO(np.pi/2 + paramsOSbk.bk_rad)

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
            "IX.bk.i.exc": np.sqrt(paramsOSbk.Pfr_W) * vfr,
            "EX.bk.i.exc": np.sqrt(paramsOSbk.Pbk_W) * vbk,
        }
    )

    ##################################################
    # AC calculation
    ##################################################

    edgesAC = deepcopy(Id)
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(ArmLink.edgesAC(F_Hz))
    edgesAC.update(BackLaserLink.edgesAC(F_Hz))

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC_opt = compAC.inverse_row(
        {"IX.bk.o.tp": None},
        {"EX.pos.exc"},
    )
    resultsAC_mech = compAC.inverse_row(
        {"EX.pos.tp": None},
        {"EX.fr.F.i.exc"},
    )

    return resultsDC, resultsAC_opt, resultsAC_mech


def test_sflu_OSbk(sflu_OSbk_results, tpath_join, pprint, plotTF):
    resultsDC, resultsAC_opt, resultsAC_mech = sflu_OSbk_results
    mlib = cmp.mats_planewave
    pprint("SFLU", np.abs(resultsDC["EX.fr.o.tp"])**2)

    LOa = np.sqrt(paramsOSbk.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt["EX.pos.exc"]
    opt_tf = opt_tf[..., 0, 0]
    mech_tf = resultsAC_mech["EX.fr.F.i.exc"][..., 0, 0]

    fig = plotTF(F_Hz, opt_tf, label="SFLU")
    fig.axes[0].legend()
    fig.axes[0].set_title("Phase response to mirror motion")
    fig.axes[0].set_ylabel("Magnitued [W/m]")
    fig.savefig(tpath_join("optical_tf.pdf"))

    fig = plotTF(F_Hz, mech_tf, label="SFLU")
    fig.axes[0].legend()
    fig.axes[0].set_title("Radiation pressure modified mechanical susceptibility")
    fig.axes[0].set_ylabel("Magnitude [m/N]")
    fig.savefig(tpath_join("mechanical_tf.pdf"))


def test_sflu_OSbk_compare_optickle(
        sflu_OSbk_results, opt_OSbk, kat_OSbk, tpath_join, plotTF, pprint):
    resultsDC, resultsAC_opt, resultsAC_mech = sflu_OSbk_results
    mlib = cmp.mats_planewave
    opt = opt_OSbk
    katFR = kat_OSbk
    # pprint("SFLU", adjoint(resultsDC["EX.fr.i.tp"]) @ resultsDC["EX.fr.i.tp"])
    pprint("SFLU", cmp.Vnorm_sq(resultsDC["EX.fr.i.tp"]), cmp.Vnorm_sq(resultsDC["EX.fr.o.tp"]))
    pprint("Optickle", opt.getSigDC("EX_DC"))
    pprint("Finesse", katFR.getSigDC("EX_DC"))

    LOa = np.sqrt(paramsOS.Plo_W) * adjoint(mlib.LO(0))
    opt_tf = LOa @ resultsAC_opt['EX.pos.exc']
    opt_tf = -opt_tf[..., 0, 0]
    mech_tf = resultsAC_mech['EX.fr.F.i.exc']
    mech_tf = mech_tf[..., 0, 0]
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
@matlib.optickle_model("opt_OSbk", paramsOSbk)
def opt_OSbk():
    """
    Optickle model for FP cavity illuminated by a laser on both sides
    """
    import qlance.optickle as qopt
    eng = matlib.start_matlab_engine()

    opt = qopt.Optickle(eng, 'opt', lambda0=paramsOSbk.lambda_m)
    opt.addMirror("IX", Thr=paramsOSbk.Ti)
    opt.addMirror("EX", Thr=paramsOSbk.Te)
    opt.addLink("IX", "fr", "EX", "fr", paramsOSbk.Larm_m)
    opt.addLink("EX", "fr", "IX", "fr", paramsOSbk.Larm_m)
    opt.setMechTF("EX", [], [0, 0], 1/paramsOSbk.M_kg)

    detune_m = paramsOSbk.lambda_m * paramsOSbk.detune_rad / (2*np.pi)
    opt.setPosOffset("EX", detune_m)

    vfr = np.exp(1j * paramsOSbk.fr_rad)
    vbk = np.exp(1j * paramsOSbk.bk_rad)
    opt.addSource("Laser_fr", np.sqrt(paramsOSbk.Pfr_W) * vfr)
    opt.addSource("Laser_bk", np.sqrt(paramsOSbk.Pbk_W) * vbk)
    opt.addLink("Laser_fr", "out", "IX", "bk", 0)
    opt.addLink("Laser_bk", "out", "EX", "bk", 0)

    opt.addProbeIn("EX_DC", "EX", "fr", 0, 0)
    opt.addHomodyneReadout("REFL", LOpower=paramsOSbk.Plo_W)
    opt.addLink("IX", "bk", "REFL_BS", "fr", 0)

    opt.run(F_Hz, noise=False)
    return opt


@pytest.fixture
@matlib.finesse_model("kat_OSbk", paramsOSbk)
def kat_OSbk():
    """
    Optickle model for FP cavity illuminated by a laser on both sides
    """
    from pykat import finesse
    import qlance.finesse as qfin

    kat = finesse.kat()
    kat.lamba0 = paramsOSbk.lambda_m
    qfin.addMirror(kat, "IX", Thr=paramsOSbk.Ti)
    qfin.addMirror(kat, "EX", Thr=paramsOSbk.Te, comp=True)
    qfin.addSpace(kat, "IX_fr", "EX_fr", paramsOSbk.Larm_m)
    qfin.setMechTF(kat, "EX", [], [0, 0], 1/paramsOSbk.M_kg)
    kat.EX.phi = paramsOSbk.detune_rad * 180/np.pi

    qfin.addLaser(
        kat, "Laser_fr", paramsOSbk.Pfr_W,
        phase=-paramsOSbk.fr_rad*180/np.pi,
    )
    qfin.addLaser(
        kat, "Laser_bk", paramsOSbk.Pbk_W,
        phase=-paramsOSbk.bk_rad*180/np.pi + 90,
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

################################################################################

paramsOS2 = Struct(
    Ti = 0.1,
    Te = 0.3,
    Tb = 0.2,
    Li_m = 40e3,
    Lb_m = 40e3,
    M_kg = 10,
    Pfr_W = 100,
    Pbk_W = 100,
    Plo_W = 1,
    detune_rad = 0*np.pi/180,
    fr_rad = 0*np.pi/180,
    bk_rad = 0*np.pi/180,
    lambda_m = 1064e-9,
)


@pytest.fixture
def sflu_OS2():
    ifo = optics.GraphElement()

    ifo.subgraph_add(
        "IX", optics.BasisMirror(),
        translation_xy=(-10, 0),
        rotation_deg=180,
    )
    ifo.subgraph_add(
        "EX", cmp.RPMirrorElement(),
        translation_xy=(+10, 0),
        rotation_deg=0,
    )
    ifo.subgraph_add(
        "BX", optics.BasisMirror(),
        translation_xy=(+35, 0),
        rotation_deg=0,
    )

    ifo.edges.update({
        ("EX.fr.i", "IX.fr.o"): "tau_i",
        ("IX.fr.i", "EX.fr.o"): "tau_i",
        ("EX.bk.i", "BX.fr.o"): "tau_b",
        ("BX.fr.i", "EX.fr.o"): "tau_b",
    })
    ifo.locations.update({
        "IX.bk.i.exc": (-20, +5),
        "IX.bk.o.tp": (-20, -5),
        "BX.bk.i.exc": (+40, -5),
        "BX.bk.o.tp": (+40, +5),
    })
    ifo.edges.update({
        ("IX.bk.i", "IX.bk.i.exc"): "1",
        ("IX.bk.o.tp", "IX.bk.o"): "1",
        ("BX.bk.i", "BX.bk.i.exc"): "1",
        ("BX.bk.o.tp", "BX.bk.o"): "1",
    })

    sflu = SFLU.SFLU(
        edges=ifo.build_edges(),
        graph=True,
    )
    ifo.update_sflu(sflu)
    return sflu


@pytest.fixture
def sflu_OS2_results(sflu_OS2, pprint):
    sflu = sflu_OS2
    sflu.reduce_auto()

    IX = cmp.MirrorEdge("IX", Thr=paramsOS2.Ti)
    EX = cmp.RPMirrorEdge("EX", Thr=paramsOS2.Te)
    BX = cmp.MirrorEdge("BX", Thr=paramsOS2.Tb)
    Link_i = cmp.LinkEdge(
        "tau_i", L_m=paramsOS2.Li_m,
        detune_rad=paramsOS2.detune_rad,
    )
    Link_b = cmp.LinkEdge(
        "tau_b", L_m=paramsOS2.Lb_m,
        detune_rad=-paramsOS2.detune_rad,
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
    edgesDC.update(BX.edgesDC())
    edgesDC.update(Link_i.edgesDC())
    edgesDC.update(Link_b.edgesDC())

    vfr = mlib.LO(np.pi/2 + paramsOS2.fr_rad)
    vbk = mlib.LO(np.pi/2 + paramsOS2.bk_rad)

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
            "IX.bk.i.exc": np.sqrt(paramsOS2.Pfr_W) * vfr,
            "BX.bk.i.exc": np.sqrt(paramsOS2.Pbk_W) * vbk,
        }
    )

    ##################################################
    # AC calculation
    ##################################################

    edgesAC = deepcopy(Id)
    edgesAC.update(IX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(EX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(BX.edgesAC(F_Hz, resultsDC))
    edgesAC.update(Link_i.edgesAC(F_Hz))
    edgesAC.update(Link_b.edgesAC(F_Hz))

    compAC = sflu.computer(eye=mlib.Id)
    compAC.compute(edge_map=edgesAC)
    resultsAC_opt = compAC.inverse_row(
        {"IX.bk.o.tp": None},
        {"EX.pos.exc"},
    )
    resultsAC_mech = compAC.inverse_row(
        {"EX.pos.tp": None},
        {"EX.fr.F.i.exc"},
    )

    return resultsDC, resultsAC_opt, resultsAC_mech

################################################################################

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

def plot_OSbk_graph(sflu_OSbk, tpath_join):
    sflu = sflu_OSbk
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
